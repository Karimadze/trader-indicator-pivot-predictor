"""Out-of-sample ATR bracket check for the VMC bull + EMA stack signal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from atr_bracket_study import simulate_trade
from divergence_study import atr
from vmc_ema_extended_study import OUT as EVENT_OUT, OUT, download_one


BRACKET_OUT = OUT / "brackets"
STOP_MULTIPLES = (1.5, 2.0, 2.5)
TARGET_MULTIPLES = (2.0, 3.0, 4.0)
MAX_HOLDS = (10, 20)


def summarize(sample: pd.DataFrame) -> dict:
    monthly = sample.assign(month=sample["entry_date"].dt.to_period("M")).groupby("month")["net_return"].mean()
    t_stat, p_value = stats.ttest_1samp(monthly.to_numpy(), 0.0) if len(monthly) >= 6 else (np.nan, np.nan)
    reasons = sample["exit_reason"].astype(str)
    return {
        "n_events": int(len(sample)),
        "n_tickers": int(sample["ticker"].nunique()),
        "mean_net_pct": 100 * float(sample["net_return"].mean()),
        "median_net_pct": 100 * float(sample["net_return"].median()),
        "profit_rate_pct": 100 * float((sample["net_return"] > 0).mean()),
        "mean_net_r": float(sample["net_r"].mean()),
        "median_days_held": float(sample["days_held"].median()),
        "target_exit_pct": 100 * float(reasons.str.startswith("target").mean()),
        "stop_exit_pct": 100 * float((reasons.str.startswith("stop") | reasons.eq("both_stop_first")).mean()),
        "t_month_cluster": float(t_stat),
        "p_value": float(p_value),
    }


def period(date: pd.Timestamp) -> str:
    if date < pd.Timestamp("2010-01-01"):
        return "development_1985_2009"
    if date < pd.Timestamp("2020-01-01"):
        return "validation_2010_2019"
    return "holdout_2020_2026"


def main() -> None:
    BRACKET_OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENT_OUT / "events.csv", parse_dates=["signal_date", "entry_date"])
    signals = events.loc[
        events["current_ndx"] & events["ema_stack"] & events["horizon"].eq(10),
        ["ticker", "signal_date"],
    ].drop_duplicates()

    rows: list[dict] = []
    for ticker, ticker_signals in signals.groupby("ticker"):
        frame = download_one(ticker)
        atr14 = atr(frame)
        positions = pd.Series(np.arange(len(frame)), index=frame.index)
        for signal_date in ticker_signals["signal_date"]:
            if signal_date not in positions.index:
                continue
            signal_pos = int(positions.loc[signal_date])
            for stop_atr in STOP_MULTIPLES:
                for target_atr in TARGET_MULTIPLES:
                    for max_hold in MAX_HOLDS:
                        trade = simulate_trade(
                            frame,
                            signal_pos,
                            float(atr14.iloc[signal_pos]),
                            stop_atr,
                            target_atr,
                            max_hold,
                        )
                        if trade is None:
                            continue
                        rows.append(
                            {
                                "ticker": ticker,
                                "signal_date": signal_date,
                                "period": period(trade["entry_date"]),
                                "stop_atr": stop_atr,
                                "target_atr": target_atr,
                                "max_hold": max_hold,
                                **trade,
                            }
                        )
    trades = pd.DataFrame(rows)
    trades.to_csv(BRACKET_OUT / "trades.csv", index=False)

    summaries: list[dict] = []
    keys = ["period", "stop_atr", "target_atr", "max_hold"]
    for values, sample in trades.groupby(keys):
        summaries.append(dict(zip(keys, values)) | summarize(sample))
    grid = pd.DataFrame(summaries)
    grid.to_csv(BRACKET_OUT / "grid_by_period.csv", index=False)

    development = grid.loc[grid["period"].eq("development_1985_2009")]
    eligible = development.loc[development["profit_rate_pct"] >= 50]
    pool = eligible if not eligible.empty else development
    locked = pool.sort_values(["mean_net_r", "mean_net_pct"], ascending=False).iloc[0]
    chosen = grid.loc[
        (grid["stop_atr"] == locked.stop_atr)
        & (grid["target_atr"] == locked.target_atr)
        & (grid["max_hold"] == locked.max_hold)
    ].sort_values("period")
    chosen.to_csv(BRACKET_OUT / "locked_rule_results.csv", index=False)
    print(chosen.to_string(index=False))


if __name__ == "__main__":
    main()
