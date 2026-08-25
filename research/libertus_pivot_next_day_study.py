"""Test whether RSI Div - Lib PIVOT labels predict a +10% next day move.

The original Pine script confirms a pivot on bar i but draws its label on
bar i-2.  This study reports both the hindsight return beside the drawn label
and the causal return available after confirmation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bybit_two_year_combo_search import TEST_START, download_one
from divergence_study import libertus_divergences


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "libertus_pivot_next_day_results"
END_DATE = pd.Timestamp("2026-08-25")
ROUND_TRIP_COST = 0.0015
SYMBOLS = {
    "MRVL": "MRVLUSDT",
    "COHR": "COHRUSDT",
    "MU": "MUUSDT",
    "LITE": "LITEUSDT",
    "GLW": "GLWUSDT",
    "TER": "TERUSDT",
    "STX": "STXXUSDT",  # STXUSDT spot is the crypto asset Stacks.
}


def download_bybit(symbol: str) -> pd.DataFrame:
    response = requests.get(
        "https://api.bybit.com/v5/market/kline",
        params={"category": "linear", "symbol": symbol, "interval": "D", "limit": 1000},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit {symbol}: {payload.get('retMsg')}")
    rows = payload["result"]["list"]
    frame = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True).dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
    # Do not test an unfinished daily candle.
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    frame = frame.loc[frame.index + pd.Timedelta(days=1) <= now]
    return frame.dropna()


def collect_events(frame: pd.DataFrame, ticker: str, market: str) -> pd.DataFrame:
    signals = libertus_divergences(frame)
    rows: list[dict] = []
    for kind, column in (("HIGH", "libertus_pivot_high"), ("LOW", "libertus_pivot_low")):
        for confirm_i in np.flatnonzero(signals[column].to_numpy(bool)):
            pivot_i = confirm_i - 2
            tradable_i = confirm_i + 1
            if pivot_i < 0 or tradable_i >= len(frame):
                continue
            pivot_date = frame.index[pivot_i]
            confirm_date = frame.index[confirm_i]
            entry_date = frame.index[tradable_i]
            if pivot_date < TEST_START or pivot_date >= END_DATE:
                continue
            pivot_close = float(frame["close"].iloc[pivot_i])
            visual_close = float(frame["close"].iloc[pivot_i + 1]) / pivot_close - 1
            visual_high = float(frame["high"].iloc[pivot_i + 1]) / pivot_close - 1
            entry_open = float(frame["open"].iloc[tradable_i])
            causal_close_gross = float(frame["close"].iloc[tradable_i]) / entry_open - 1
            causal_high = float(frame["high"].iloc[tradable_i]) / entry_open - 1
            rows.append(
                {
                    "market": market,
                    "ticker": ticker,
                    "pivot_type": kind,
                    "pivot_date_drawn": pivot_date.date().isoformat(),
                    "confirmation_date": confirm_date.date().isoformat(),
                    "entry_date": entry_date.date().isoformat(),
                    "visual_next_close_return": visual_close,
                    "visual_next_high_return": visual_high,
                    "causal_next_close_return_gross": causal_close_gross,
                    "causal_next_close_return_net": causal_close_gross - ROUND_TRIP_COST,
                    "causal_next_high_return": causal_high,
                }
            )
    return pd.DataFrame(rows)


def summarize(events: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    if events.empty:
        return pd.DataFrame()
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, part in events.groupby(grouper, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "signals": len(part),
                "visual_mean_pct": 100 * part["visual_next_close_return"].mean(),
                "visual_median_pct": 100 * part["visual_next_close_return"].median(),
                "visual_close_ge_10pct": int((part["visual_next_close_return"] >= 0.10).sum()),
                "visual_high_ge_10pct": int((part["visual_next_high_return"] >= 0.10).sum()),
                "causal_mean_net_pct": 100 * part["causal_next_close_return_net"].mean(),
                "causal_median_net_pct": 100 * part["causal_next_close_return_net"].median(),
                "causal_win_rate_pct": 100 * (part["causal_next_close_return_net"] > 0).mean(),
                "causal_close_ge_10pct": int((part["causal_next_close_return_net"] >= 0.10).sum()),
                "causal_high_ge_10pct": int((part["causal_next_high_return"] >= 0.10).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def baseline(frame: pd.DataFrame, ticker: str, market: str) -> pd.DataFrame:
    eligible = frame.loc[(frame.index >= TEST_START) & (frame.index < END_DATE)].copy()
    eligible["net_return"] = eligible["close"] / eligible["open"] - 1 - ROUND_TRIP_COST
    eligible["high_return"] = eligible["high"] / eligible["open"] - 1
    eligible = eligible.dropna(subset=["net_return", "high_return"])
    return pd.DataFrame(
        [
            {
                "market": market,
                "ticker": ticker,
                "days": len(eligible),
                "mean_net_pct": 100 * eligible["net_return"].mean(),
                "median_net_pct": 100 * eligible["net_return"].median(),
                "win_rate_pct": 100 * (eligible["net_return"] > 0).mean(),
                "close_ge_10pct": int((eligible["net_return"] >= 0.10).sum()),
                "high_ge_10pct": int((eligible["high_return"] >= 0.10).sum()),
            }
        ]
    )


def summarize_unique_dates(events: pd.DataFrame) -> pd.DataFrame:
    """Avoid treating correlated same-day signals as independent observations."""
    rows: list[dict] = []
    lows = events.loc[events["pivot_type"] == "LOW"]
    for market, part in lows.groupby("market"):
        daily = part.groupby("entry_date")["causal_next_close_return_net"].mean()
        rows.append(
            {
                "market": market,
                "signals": len(part),
                "unique_entry_dates": len(daily),
                "mean_equal_weight_date_pct": 100 * daily.mean(),
                "median_equal_weight_date_pct": 100 * daily.median(),
                "winning_dates_pct": 100 * (daily > 0).mean(),
                "dates_ge_10pct": int((daily >= 0.10).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events: list[pd.DataFrame] = []
    baselines: list[pd.DataFrame] = []
    coverage: list[dict] = []
    for ticker, contract in SYMBOLS.items():
        underlying = download_one(ticker)
        if not underlying.empty:
            events.append(collect_events(underlying, ticker, "US underlying"))
            baselines.append(baseline(underlying, ticker, "US underlying"))
            coverage.append({"market": "US underlying", "ticker": ticker, "first": str(underlying.index.min().date()), "last": str(underlying.index.max().date()), "bars": len(underlying)})
        try:
            bybit = download_bybit(contract)
        except Exception as exc:  # Keep the underlying study usable during API outages.
            coverage.append({"market": "Bybit linear", "ticker": ticker, "error": str(exc)})
            continue
        if not bybit.empty:
            events.append(collect_events(bybit, ticker, "Bybit linear"))
            baselines.append(baseline(bybit, ticker, "Bybit linear"))
            coverage.append({"market": "Bybit linear", "ticker": ticker, "contract": contract, "first": str(bybit.index.min().date()), "last": str(bybit.index.max().date()), "bars": len(bybit)})

    all_events = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    all_baselines = pd.concat(baselines, ignore_index=True) if baselines else pd.DataFrame()
    by_ticker = summarize(all_events, ["market", "ticker", "pivot_type"])
    aggregate = summarize(all_events, ["market", "pivot_type"])
    unique_dates = summarize_unique_dates(all_events)
    all_events.to_csv(OUT / "events.csv", index=False)
    by_ticker.to_csv(OUT / "summary_by_ticker.csv", index=False)
    aggregate.to_csv(OUT / "summary_aggregate.csv", index=False)
    unique_dates.to_csv(OUT / "summary_unique_entry_dates.csv", index=False)
    all_baselines.to_csv(OUT / "baseline.csv", index=False)
    (OUT / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("AGGREGATE")
    print(aggregate.round(2).to_string(index=False))
    print("\nBY TICKER - LOW PIVOTS (long hypothesis)")
    print(by_ticker.loc[by_ticker["pivot_type"] == "LOW"].round(2).to_string(index=False))
    print("\nBASELINE")
    print(all_baselines.round(2).to_string(index=False))
    print("\nLOW PIVOTS - UNIQUE ENTRY DATES")
    print(unique_dates.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
