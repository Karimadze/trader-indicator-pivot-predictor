"""Lock ATR stop/target rules on development data and test them after 2020.

The indicator combinations are imported from the already locked random search.
For each of the three best bullish combinations of size 3, 6 and 8, a bracket
is selected using 2010-2019 only.  That bracket is then evaluated once on the
2020+ holdout period.  Same-day stop-and-target collisions are resolved against
the strategy (stop first), which is deliberately conservative for daily bars.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from divergence_study import BENCHMARK, HOLDOUT_START, TICKERS, atr, benjamini_hochberg, download_one
from random_indicator_search import FEATURE_NAMES, MIN_GAP, ROUND_TRIP_COST, encode, indicator_states


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "random_combo_results" / "selected_locked.csv"
OUT = ROOT / "atr_bracket_results"

STOP_MULTIPLES = (1.5, 2.0, 2.5)
TARGET_MULTIPLES = (2.0, 3.0, 4.0)
MAX_HOLDS = (10, 20)
TOP_PER_SIZE = 3


def transition_positions(mask: np.ndarray, bits: np.uint32) -> list[int]:
    active = (mask & bits) == bits
    previous = np.r_[False, active[:-1]]
    transitions = np.flatnonzero(active & ~previous)
    kept: list[int] = []
    last = -10_000
    for pos in transitions:
        if int(pos) - last >= MIN_GAP:
            kept.append(int(pos))
            last = int(pos)
    return kept


def simulate_trade(
    frame: pd.DataFrame,
    signal_pos: int,
    atr_value: float,
    stop_multiple: float,
    target_multiple: float,
    max_hold: int,
) -> dict | None:
    entry_pos = signal_pos + 1
    if entry_pos >= len(frame) or not np.isfinite(atr_value) or atr_value <= 0:
        return None
    entry = float(frame["open"].iloc[entry_pos])
    if not np.isfinite(entry) or entry <= 0:
        return None
    stop_distance = stop_multiple * atr_value
    stop = entry - stop_distance
    target = entry + target_multiple * atr_value
    final_pos = min(entry_pos + max_hold - 1, len(frame) - 1)
    exit_price = float(frame["close"].iloc[final_pos])
    exit_pos = final_pos
    reason = "time"

    for pos in range(entry_pos, final_pos + 1):
        day_open = float(frame["open"].iloc[pos])
        day_low = float(frame["low"].iloc[pos])
        day_high = float(frame["high"].iloc[pos])
        if pos > entry_pos and day_open <= stop:
            exit_price, exit_pos, reason = day_open, pos, "stop_gap"
            break
        if pos > entry_pos and day_open >= target:
            exit_price, exit_pos, reason = day_open, pos, "target_gap"
            break
        hit_stop = day_low <= stop
        hit_target = day_high >= target
        if hit_stop and hit_target:
            exit_price, exit_pos, reason = stop, pos, "both_stop_first"
            break
        if hit_stop:
            exit_price, exit_pos, reason = stop, pos, "stop"
            break
        if hit_target:
            exit_price, exit_pos, reason = target, pos, "target"
            break

    net_return = exit_price / entry - 1.0 - ROUND_TRIP_COST
    risk_fraction = stop_distance / entry
    return {
        "entry_date": frame.index[entry_pos],
        "exit_date": frame.index[exit_pos],
        "entry": entry,
        "exit": exit_price,
        "net_return": net_return,
        "net_r": net_return / risk_fraction,
        "profitable": net_return > 0,
        "exit_reason": reason,
        "days_held": exit_pos - entry_pos + 1,
    }


def summarize(rows: pd.DataFrame) -> dict | None:
    if len(rows) < 15:
        return None
    monthly = rows.assign(month=rows["entry_date"].dt.to_period("M")).groupby("month")["net_return"].mean()
    if len(monthly) < 6:
        return None
    t_stat, p_value = stats.ttest_1samp(monthly.to_numpy(), 0.0)
    ticker_means = rows.groupby("ticker")["net_return"].mean()
    reasons = rows["exit_reason"].astype(str)
    return {
        "n_events": int(len(rows)),
        "n_tickers": int(rows["ticker"].nunique()),
        "mean_net_pct": float(100 * rows["net_return"].mean()),
        "median_net_pct": float(100 * rows["net_return"].median()),
        "profit_rate_pct": float(100 * rows["profitable"].mean()),
        "mean_net_r": float(rows["net_r"].mean()),
        "median_days_held": float(rows["days_held"].median()),
        "target_exit_pct": float(100 * reasons.str.startswith("target").mean()),
        "stop_exit_pct": float(100 * (reasons.str.startswith("stop") | reasons.eq("both_stop_first")).mean()),
        "positive_ticker_pct": float(100 * (ticker_means > 0).mean()),
        "t_stat_month_cluster": float(t_stat),
        "p_value": float(p_value),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SOURCE)
    selected = selected.loc[selected["direction"].eq("bull")]
    candidates = (
        selected.sort_values(["size", "score"], ascending=[True, False])
        .groupby("size", sort=True, as_index=False)
        .head(TOP_PER_SIZE)
        .reset_index(drop=True)
    )
    candidates["bracket_candidate_id"] = np.arange(len(candidates))

    qqq = download_one(BENCHMARK)
    prepared: dict[str, tuple[pd.DataFrame, np.ndarray, pd.Series]] = {}
    for ticker in TICKERS:
        frame = download_one(ticker)
        bull, _ = indicator_states(frame, qqq)
        prepared[ticker] = (frame, encode(bull), atr(frame))

    all_trades: list[dict] = []
    for candidate in candidates.itertuples(index=False):
        bits = np.uint32(int(candidate.bits))
        for ticker, (frame, bull_mask, atr_values) in prepared.items():
            positions = transition_positions(bull_mask, bits)
            for signal_pos in positions:
                for stop_multiple in STOP_MULTIPLES:
                    for target_multiple in TARGET_MULTIPLES:
                        for max_hold in MAX_HOLDS:
                            trade = simulate_trade(
                                frame,
                                signal_pos,
                                float(atr_values.iloc[signal_pos]),
                                stop_multiple,
                                target_multiple,
                                max_hold,
                            )
                            if trade is None:
                                continue
                            all_trades.append(
                                {
                                    "bracket_candidate_id": int(candidate.bracket_candidate_id),
                                    "size": int(candidate.size),
                                    "features": candidate.features,
                                    "ticker": ticker,
                                    "stop_atr": stop_multiple,
                                    "target_atr": target_multiple,
                                    "max_hold": max_hold,
                                    **trade,
                                }
                            )

    trades = pd.DataFrame(all_trades)
    development = trades.loc[trades["entry_date"] < HOLDOUT_START]
    dev_rows: list[dict] = []
    group_cols = ["bracket_candidate_id", "size", "features", "stop_atr", "target_atr", "max_hold"]
    for key, group in development.groupby(group_cols, sort=False):
        result = summarize(group)
        if result is not None:
            dev_rows.append(dict(zip(group_cols, key)) | result)
    dev = pd.DataFrame(dev_rows)

    # Lock one bracket per indicator candidate.  Prefer a >=55% development
    # profit rate, then maximize average R; fall back to average R if necessary.
    locked_rows: list[pd.Series] = []
    for _, group in dev.groupby("bracket_candidate_id", sort=True):
        eligible = group.loc[(group["profit_rate_pct"] >= 55.0) & (group["n_events"] >= 30)]
        pool = eligible if not eligible.empty else group
        locked_rows.append(pool.sort_values(["mean_net_r", "mean_net_pct"], ascending=False).iloc[0])
    locked = pd.DataFrame(locked_rows).reset_index(drop=True)
    locked.to_csv(OUT / "locked_brackets_development.csv", index=False)

    holdout = trades.loc[trades["entry_date"] >= HOLDOUT_START]
    holdout_rows: list[dict] = []
    for lock in locked.itertuples(index=False):
        group = holdout.loc[
            (holdout["bracket_candidate_id"] == lock.bracket_candidate_id)
            & (holdout["stop_atr"] == lock.stop_atr)
            & (holdout["target_atr"] == lock.target_atr)
            & (holdout["max_hold"] == lock.max_hold)
        ]
        result = summarize(group)
        if result is not None:
            holdout_rows.append(
                {
                    "bracket_candidate_id": int(lock.bracket_candidate_id),
                    "size": int(lock.size),
                    "features": lock.features,
                    "stop_atr": lock.stop_atr,
                    "target_atr": lock.target_atr,
                    "max_hold": int(lock.max_hold),
                    **result,
                }
            )
    tested = pd.DataFrame(holdout_rows)
    if not tested.empty:
        tested["q_value_bh"] = benjamini_hochberg(tested["p_value"])
        tested = tested.sort_values(["q_value_bh", "mean_net_r"], ascending=[True, False])
    tested.to_csv(OUT / "locked_brackets_holdout.csv", index=False)
    trades.to_csv(OUT / "all_bracket_trades.csv", index=False)
    candidates.to_csv(OUT / "indicator_candidates.csv", index=False)
    print(tested.to_string(index=False))


if __name__ == "__main__":
    main()
