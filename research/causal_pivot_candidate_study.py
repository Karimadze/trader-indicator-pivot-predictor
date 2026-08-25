"""Causal comparison of early RSI-low candidates and Libertus back-plotted pivots.

Every tradable signal is timestamped on the bar where it is knowable and is
entered at the next session open.  The Libertus visual series is included only
to quantify the hindsight effect created by drawing a confirmed pivot two bars
back.  It is not treated as a tradable result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from divergence_study import download_one, libertus_divergences, rsi_wilder


OUT = Path(__file__).resolve().parent / "causal_pivot_candidate_results"
TICKERS = ["MRVL", "COHR", "MU", "LITE", "GLW", "TER", "STX"]
WATCH_LOOKBACK = 45
BUY_LOOKBACK = 60
STRONG_LOOKBACK = 90
EMA_LENGTH = 200
VOLUME_LENGTH = 20
COOLDOWN = 2
COST = 0.0015
RECENT_START = pd.Timestamp("2024-08-24")
RECENT_SPLIT = pd.Timestamp("2025-08-24")


def cooldown_mask(raw: pd.Series, cooldown: int) -> pd.Series:
    result = pd.Series(False, index=raw.index)
    last = -100_000
    for i, value in enumerate(raw.fillna(False).to_numpy(bool)):
        if value and i - last > cooldown:
            result.iloc[i] = True
            last = i
    return result


def signals(frame: pd.DataFrame) -> pd.DataFrame:
    rsi = rsi_wilder(frame["close"], 14)
    falling_rsi = rsi < rsi.shift(1)
    fresh_45 = (rsi <= rsi.rolling(WATCH_LOOKBACK, min_periods=WATCH_LOOKBACK).min() + 1e-10) & falling_rsi
    fresh_60 = (rsi <= rsi.rolling(BUY_LOOKBACK, min_periods=BUY_LOOKBACK).min() + 1e-10) & falling_rsi
    fresh_90 = (rsi <= rsi.rolling(STRONG_LOOKBACK, min_periods=STRONG_LOOKBACK).min() + 1e-10) & falling_rsi
    ema200 = frame["close"].ewm(span=EMA_LENGTH, adjust=False).mean()
    volume_average = frame["volume"].rolling(VOLUME_LENGTH).mean()
    trend_ok = frame["close"] > ema200
    volume_ok = frame["volume"] >= volume_average
    watch = cooldown_mask(fresh_45 & trend_ok, COOLDOWN)
    buy = cooldown_mask(fresh_60 & trend_ok & volume_ok, COOLDOWN)
    strong = cooldown_mask(fresh_90 & trend_ok & volume_ok, COOLDOWN)

    libertus = libertus_divergences(frame, lookback=STRONG_LOOKBACK)
    visual = pd.Series(False, index=frame.index)
    for confirmation_i in np.flatnonzero(libertus["libertus_pivot_low"].to_numpy(bool)):
        if confirmation_i >= 2:
            visual.iloc[confirmation_i - 2] = True

    return pd.DataFrame(
        {
            "watch_45": watch,
            "buy_60": buy,
            "strong_90": strong,
            "libertus_visual_backplot": visual,
            "libertus_honest_confirmation": libertus["libertus_pivot_low"],
        },
        index=frame.index,
    )


def event_rows(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    signal_frame = signals(frame)
    rows: list[dict] = []
    for signal_name in signal_frame.columns:
        for i in np.flatnonzero(signal_frame[signal_name].to_numpy(bool)):
            if i + 1 >= len(frame):
                continue
            entry_open = float(frame["open"].iloc[i + 1])
            rows.append(
                {
                    "ticker": ticker,
                    "signal": signal_name,
                    "signal_date": frame.index[i].date().isoformat(),
                    "entry_date": frame.index[i + 1].date().isoformat(),
                    "next_close_return_gross": float(frame["close"].iloc[i + 1]) / entry_open - 1,
                    "next_close_return_net": float(frame["close"].iloc[i + 1]) / entry_open - 1 - COST,
                    "next_high_return": float(frame["high"].iloc[i + 1]) / entry_open - 1,
                }
            )
    return pd.DataFrame(rows)


def summary(events: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(events["signal_date"])
    periods = {
        "2010-2019": dates < "2020-01-01",
        "2020-2023": (dates >= "2020-01-01") & (dates < "2024-01-01"),
        "2024+": dates >= "2024-01-01",
        "recent_train": (dates >= RECENT_START) & (dates < RECENT_SPLIT),
        "recent_later": dates >= RECENT_SPLIT,
        "recent_total": dates >= RECENT_START,
    }
    rows: list[dict] = []
    for period_name, period_mask in periods.items():
        for signal_name in events["signal"].unique():
            part = events.loc[period_mask & (events["signal"] == signal_name)]
            if part.empty:
                continue
            net = part["next_close_return_net"]
            rows.append(
                {
                    "period": period_name,
                    "signal": signal_name,
                    "events": len(part),
                    "net_win_rate_pct": 100 * (net > 0).mean(),
                    "mean_net_pct": 100 * net.mean(),
                    "median_net_pct": 100 * net.median(),
                    "close_ge_3pct": int((net >= 0.03).sum()),
                    "close_ge_5pct": int((net >= 0.05).sum()),
                    "close_ge_10pct": int((net >= 0.10).sum()),
                    "high_ge_10pct": int((part["next_high_return"] >= 0.10).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_events = pd.concat(
        [event_rows(ticker, download_one(ticker)) for ticker in TICKERS],
        ignore_index=True,
    )
    result = summary(all_events)
    all_events.to_csv(OUT / "events.csv", index=False)
    result.to_csv(OUT / "summary.csv", index=False)
    print(
        result.loc[result["period"].isin(["recent_train", "recent_later", "recent_total"])]
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
