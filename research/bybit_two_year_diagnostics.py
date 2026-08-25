"""Diagnostics and concentration checks for the Bybit two-year combo search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bybit_two_year_combo_search import (
    FEATURE_NAMES,
    HOLDOUT_START,
    HORIZONS,
    OUT,
    ROUND_TRIP_COST,
    build_dataset,
    combo_bits,
    metric,
    signal_indices,
)


def periodic_indices(data: pd.DataFrame, gap: int = 10) -> np.ndarray:
    selected: list[int] = []
    for _, group in data.groupby("ticker", sort=False):
        eligible = group.index[group["date"] + pd.offsets.BDay(1) >= HOLDOUT_START].to_numpy(int)
        selected.extend(eligible[::gap].tolist())
    return np.asarray(selected, dtype=int)


def event_frame(data: pd.DataFrame, indices: np.ndarray, direction: str, horizon: int) -> pd.DataFrame:
    rows = data.loc[indices].copy()
    rows = rows.loc[rows["date"] + pd.offsets.BDay(1) >= HOLDOUT_START]
    sign = 1 if direction == "bull" else -1
    rows["raw_net_return"] = sign * rows[f"stock_return_{horizon}"] - ROUND_TRIP_COST
    rows["excess_qqq_return"] = sign * (
        rows[f"stock_return_{horizon}"] - rows[f"qqq_return_{horizon}"]
    ) - ROUND_TRIP_COST
    return rows.dropna(subset=["raw_net_return", "excess_qqq_return"])


def main() -> None:
    data, _ = build_dataset()
    baseline_rows: list[dict] = []
    periodic = periodic_indices(data)
    for horizon in HORIZONS:
        result = metric(data, periodic, "bull", horizon, "holdout")
        baseline_rows.append({"rule": "periodic_every_10_bars", "horizon": horizon, **(result or {})})
    for feature_index, feature in enumerate(FEATURE_NAMES):
        bits = combo_bits((feature_index,))
        for direction in ("bull", "bear"):
            indices = signal_indices(data, bits, direction)
            for horizon in HORIZONS:
                result = metric(data, indices, direction, horizon, "holdout")
                if result is not None:
                    baseline_rows.append(
                        {"rule": feature, "direction": direction, "horizon": horizon, **result}
                    )
    pd.DataFrame(baseline_rows).to_csv(OUT / "single_feature_and_market_baselines.csv", index=False)

    holdout = pd.read_csv(OUT / "holdout_results.csv")
    winners = (
        holdout.loc[(holdout["direction"] == "bull") & (holdout["horizon"] == 20)]
        .sort_values(["size", "q_value_bh", "mean_raw_net_pct"], ascending=[True, True, False])
        .groupby("size", as_index=False)
        .head(1)
    )
    event_parts: list[pd.DataFrame] = []
    exchange_parts: list[dict] = []
    ticker_parts: list[dict] = []
    for winner in winners.itertuples(index=False):
        locked = pd.read_csv(OUT / "selected_locked.csv")
        bits = int(locked.loc[locked["candidate_id"] == winner.candidate_id, "bits"].iloc[0])
        indices = signal_indices(data, np.uint32(bits), "bull")
        events = event_frame(data, indices, "bull", 20)
        events["candidate_id"] = int(winner.candidate_id)
        events["size"] = int(winner.size)
        events["features"] = winner.features
        event_parts.append(events)
        for exchange, sample in events.groupby("exchange"):
            exchange_parts.append(
                {
                    "candidate_id": int(winner.candidate_id),
                    "size": int(winner.size),
                    "exchange": exchange,
                    "n_events": len(sample),
                    "n_tickers": sample["ticker"].nunique(),
                    "mean_raw_net_pct": 100 * sample["raw_net_return"].mean(),
                    "win_rate_pct": 100 * (sample["raw_net_return"] > 0).mean(),
                }
            )
        for ticker, sample in events.groupby("ticker"):
            ticker_parts.append(
                {
                    "candidate_id": int(winner.candidate_id),
                    "size": int(winner.size),
                    "ticker": ticker,
                    "contract": sample["contract"].iloc[0],
                    "exchange": sample["exchange"].iloc[0],
                    "n_events": len(sample),
                    "mean_raw_net_pct": 100 * sample["raw_net_return"].mean(),
                    "median_raw_net_pct": 100 * sample["raw_net_return"].median(),
                    "win_rate_pct": 100 * (sample["raw_net_return"] > 0).mean(),
                }
            )
    pd.concat(event_parts, ignore_index=True).to_csv(OUT / "top_candidate_events.csv", index=False)
    pd.DataFrame(exchange_parts).to_csv(OUT / "top_candidates_by_exchange.csv", index=False)
    pd.DataFrame(ticker_parts).to_csv(OUT / "top_candidates_by_ticker.csv", index=False)
    winners.to_csv(OUT / "top_one_per_size.csv", index=False)
    print(winners.to_string(index=False))
    print(pd.DataFrame(baseline_rows).loc[lambda x: x["horizon"].eq(20)].to_string(index=False))


if __name__ == "__main__":
    main()
