"""Reproducible event study for VMC and Libertus RSI divergences on US stocks.

The study deliberately timestamps a divergence on the bar where it becomes
knowable, not on the earlier pivot bar where TradingView draws the label.
Signals are entered at the next session open.  Results are therefore causal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "divergence_data"
OUT = ROOT / "divergence_results"

# Liquid Nasdaq names used in the user's watchlist and the largest QQQ names.
# This is a fixed present-day universe, so the report explicitly discloses
# survivorship bias rather than pretending it is a point-in-time membership set.
TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "AVGO", "META", "TSLA",
    "WMT", "COST", "NFLX", "PLTR", "AMD", "MU", "CSCO", "QCOM",
    "AMAT", "LRCX", "ASML", "INTC",
]
BENCHMARK = "QQQ"
START = "2010-01-01"
END = "2026-08-22"
HORIZONS = (1, 5, 10, 20)
HOLDOUT_START = pd.Timestamp("2020-01-01")
MIN_SIGNAL_GAP = 10
RNG_SEED = 20260821


def download_one(ticker: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker}_1d.csv"
    if path.exists():
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if not frame.empty:
            return frame

    frame = yf.download(
        ticker,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=str.lower)
    wanted = [c for c in ("open", "high", "low", "close", "volume") if c in frame]
    frame = frame[wanted].dropna(subset=["open", "high", "low", "close"])
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.to_csv(path)
    return frame


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_down = down.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(avg_down != 0, 100.0)


def wavetrend(frame: pd.DataFrame) -> pd.DataFrame:
    source = (frame["high"] + frame["low"] + frame["close"]) / 3
    esa = ema(source, 9)
    dev = ema((source - esa).abs(), 9)
    ci = (source - esa) / (0.015 * dev.replace(0, np.nan))
    wt1 = ema(ci, 12)
    wt2 = wt1.rolling(3, min_periods=3).mean()
    return pd.DataFrame({"wt1": wt1, "wt2": wt2}, index=frame.index)


def vmc_divergences(frame: pd.DataFrame) -> pd.DataFrame:
    """VMC regular divergences, timestamped two bars after the pivot."""
    wt2 = wavetrend(frame)["wt2"]
    top = (
        (wt2.shift(4) < wt2.shift(2))
        & (wt2.shift(3) < wt2.shift(2))
        & (wt2.shift(2) > wt2.shift(1))
        & (wt2.shift(2) > wt2)
    )
    bottom = (
        (wt2.shift(4) > wt2.shift(2))
        & (wt2.shift(3) > wt2.shift(2))
        & (wt2.shift(2) < wt2.shift(1))
        & (wt2.shift(2) < wt2)
    )

    result = pd.DataFrame(False, index=frame.index, columns=[
        "vmc_primary_bull", "vmc_primary_bear",
        "vmc_secondary_bull", "vmc_secondary_bear",
    ])
    previous_top: tuple[float, float] | None = None
    previous_bottom: tuple[float, float] | None = None

    for i in range(len(frame)):
        pivot = i - 2
        if pivot < 0 or pd.isna(wt2.iloc[pivot]):
            continue
        if bool(top.iloc[i]):
            osc = float(wt2.iloc[pivot])
            price = float(frame["high"].iloc[pivot])
            if previous_top is not None:
                prev_osc, prev_price = previous_top
                is_bear = price > prev_price and osc < prev_osc
                result.iloc[i, result.columns.get_loc("vmc_primary_bear")] = is_bear and osc >= 45
                result.iloc[i, result.columns.get_loc("vmc_secondary_bear")] = is_bear and osc >= 15
            previous_top = (osc, price)
        if bool(bottom.iloc[i]):
            osc = float(wt2.iloc[pivot])
            price = float(frame["low"].iloc[pivot])
            if previous_bottom is not None:
                prev_osc, prev_price = previous_bottom
                is_bull = price < prev_price and osc > prev_osc
                result.iloc[i, result.columns.get_loc("vmc_primary_bull")] = is_bull and osc <= -65
                result.iloc[i, result.columns.get_loc("vmc_secondary_bull")] = is_bull and osc <= -40
            previous_bottom = (osc, price)

    result["vmc_any_bull"] = result["vmc_primary_bull"] | result["vmc_secondary_bull"]
    result["vmc_any_bear"] = result["vmc_primary_bear"] | result["vmc_secondary_bear"]
    return result


def libertus_divergences(frame: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    """Port of RSI Div - Lib's state logic, timestamped on the alert bar."""
    close = frame["close"].to_numpy(float)
    rsi = rsi_wilder(frame["close"], 14).to_numpy(float)
    n = len(frame)
    max_price = np.full(n, np.nan)
    max_rsi = np.full(n, np.nan)
    min_price = np.full(n, np.nan)
    min_rsi = np.full(n, np.nan)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)

    for i in range(n):
        if np.isnan(rsi[i]):
            continue
        start = max(0, i - lookback + 1)
        window = rsi[start : i + 1]
        current_roll_high = rsi[i] >= np.nanmax(window)
        current_roll_low = rsi[i] <= np.nanmin(window)

        prev_max_price = max_price[i - 1] if i else np.nan
        prev_max_rsi = max_rsi[i - 1] if i else np.nan
        prev_min_price = min_price[i - 1] if i else np.nan
        prev_min_rsi = min_rsi[i - 1] if i else np.nan

        max_price[i] = close[i] if current_roll_high or np.isnan(prev_max_price) else prev_max_price
        max_rsi[i] = rsi[i] if current_roll_high or np.isnan(prev_max_rsi) else prev_max_rsi
        min_price[i] = close[i] if current_roll_low or np.isnan(prev_min_price) else prev_min_price
        min_rsi[i] = rsi[i] if current_roll_low or np.isnan(prev_min_rsi) else prev_min_rsi

        if close[i] > max_price[i]:
            max_price[i] = close[i]
        if rsi[i] > max_rsi[i]:
            max_rsi[i] = rsi[i]
        if close[i] < min_price[i]:
            min_price[i] = close[i]
        if rsi[i] < min_rsi[i]:
            min_rsi[i] = rsi[i]

        if i >= 3:
            pivot_high[i] = max_rsi[i] == max_rsi[i - 2] and max_rsi[i - 2] != max_rsi[i - 3]
            pivot_low[i] = min_rsi[i] == min_rsi[i - 2] and min_rsi[i - 2] != min_rsi[i - 3]
        if i >= 2:
            bear[i] = max_price[i - 1] > max_price[i - 2] and rsi[i - 1] < max_rsi[i] and rsi[i] <= rsi[i - 1]
            bull[i] = min_price[i - 1] < min_price[i - 2] and rsi[i - 1] > min_rsi[i] and rsi[i] >= rsi[i - 1]

    return pd.DataFrame(
        {
            "libertus_bull": bull,
            "libertus_bear": bear,
            "libertus_pivot_high": pivot_high,
            "libertus_pivot_low": pivot_low,
            "rsi": rsi,
        },
        index=frame.index,
    )


def atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def align_benchmark(frame: pd.DataFrame, qqq: pd.DataFrame) -> pd.DataFrame:
    aligned = qqq.reindex(frame.index).ffill()
    out = pd.DataFrame(index=frame.index)
    out["qqq_close"] = aligned["close"]
    out["qqq_open"] = aligned["open"]
    out["qqq_ema200"] = ema(aligned["close"], 200)
    out["qqq_return126"] = aligned["close"].pct_change(126)
    return out


def add_context(frame: pd.DataFrame, qqq: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["return126"] = out["close"].pct_change(126)
    out["volume_ratio"] = out["volume"] / out["volume"].rolling(20).mean()
    out["atr_pct"] = atr(out) / out["close"]
    out["atr_pct_median"] = out["atr_pct"].rolling(252).median()
    return out.join(align_benchmark(out, qqq))


def purge_mask(mask: pd.Series, gap: int = MIN_SIGNAL_GAP) -> pd.Series:
    keep = pd.Series(False, index=mask.index)
    last = -10_000
    for i, value in enumerate(mask.fillna(False).to_numpy(bool)):
        if value and i - last >= gap:
            keep.iloc[i] = True
            last = i
    return keep


def event_rows(ticker: str, frame: pd.DataFrame, signals: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    definitions = {
        "vmc_primary": ("vmc_primary_bull", "vmc_primary_bear"),
        "vmc_any": ("vmc_any_bull", "vmc_any_bear"),
        "libertus": ("libertus_bull", "libertus_bear"),
        "confluence": ("confluence_bull", "confluence_bear"),
    }
    for signal_name, (bull_col, bear_col) in definitions.items():
        for direction, column in ((1, bull_col), (-1, bear_col)):
            mask = purge_mask(signals[column])
            for i in np.flatnonzero(mask.to_numpy()):
                entry_i = i + 1
                if entry_i >= len(frame):
                    continue
                base = {
                    "ticker": ticker,
                    "signal": signal_name,
                    "direction": "bull" if direction == 1 else "bear",
                    "signal_date": frame.index[i],
                    "signal_i": i,
                    "entry_date": frame.index[entry_i],
                    "entry_i": entry_i,
                    "trend_aligned": bool(frame["close"].iloc[i] > frame["ema200"].iloc[i]) if direction == 1 else bool(frame["close"].iloc[i] < frame["ema200"].iloc[i]),
                    "stack_aligned": bool(frame["close"].iloc[i] > frame["ema50"].iloc[i] > frame["ema200"].iloc[i]) if direction == 1 else bool(frame["close"].iloc[i] < frame["ema50"].iloc[i] < frame["ema200"].iloc[i]),
                    "market_aligned": bool(frame["qqq_close"].iloc[i] > frame["qqq_ema200"].iloc[i]) if direction == 1 else bool(frame["qqq_close"].iloc[i] < frame["qqq_ema200"].iloc[i]),
                    "rs_aligned": bool(frame["return126"].iloc[i] > frame["qqq_return126"].iloc[i]) if direction == 1 else bool(frame["return126"].iloc[i] < frame["qqq_return126"].iloc[i]),
                    "high_volume": bool(frame["volume_ratio"].iloc[i] >= 1.2),
                    "low_volume": bool(frame["volume_ratio"].iloc[i] <= 0.8),
                    "low_volatility": bool(frame["atr_pct"].iloc[i] <= frame["atr_pct_median"].iloc[i]),
                }
                for horizon in HORIZONS:
                    exit_i = entry_i + horizon - 1
                    if exit_i >= len(frame):
                        continue
                    stock_return = frame["close"].iloc[exit_i] / frame["open"].iloc[entry_i] - 1
                    qqq_return = frame["qqq_close"].iloc[exit_i] / frame["qqq_open"].iloc[entry_i] - 1
                    row = dict(base)
                    row.update(
                        {
                            "horizon": horizon,
                            "raw_return": direction * stock_return,
                            "excess_return": direction * (stock_return - qqq_return),
                        }
                    )
                    rows.append(row)
    return rows


FILTERS = {
    "none": lambda x: pd.Series(True, index=x.index),
    "trend": lambda x: x["trend_aligned"],
    "ema_stack": lambda x: x["stack_aligned"],
    "market": lambda x: x["market_aligned"],
    "relative_strength": lambda x: x["rs_aligned"],
    "high_volume": lambda x: x["high_volume"],
    "low_volume": lambda x: x["low_volume"],
    "low_volatility": lambda x: x["low_volatility"],
    "trend_market": lambda x: x["trend_aligned"] & x["market_aligned"],
    "trend_market_rs": lambda x: x["trend_aligned"] & x["market_aligned"] & x["rs_aligned"],
}


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = p_values.fillna(1.0).to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    n = len(p)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return pd.Series(out, index=p_values.index)


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict] = []
    for period_name, period_mask in {
        "development_2010_2019": events["entry_date"] < HOLDOUT_START,
        "holdout_2020_2026": events["entry_date"] >= HOLDOUT_START,
        "full": pd.Series(True, index=events.index),
    }.items():
        period = events.loc[period_mask]
        for signal in sorted(period["signal"].unique()):
            for direction in ("bull", "bear"):
                for horizon in HORIZONS:
                    base = period[
                        (period["signal"] == signal)
                        & (period["direction"] == direction)
                        & (period["horizon"] == horizon)
                    ]
                    for filter_name, filter_fn in FILTERS.items():
                        sample = base.loc[filter_fn(base)] if not base.empty else base
                        if len(sample) < 15:
                            continue
                        monthly = sample.assign(month=sample["entry_date"].dt.to_period("M")).groupby("month")["excess_return"].mean()
                        if len(monthly) < 6:
                            continue
                        t_stat, p_value = stats.ttest_1samp(monthly.to_numpy(), 0.0)
                        boot = np.array([
                            rng.choice(monthly.to_numpy(), size=len(monthly), replace=True).mean()
                            for _ in range(2000)
                        ])
                        rows.append(
                            {
                                "period": period_name,
                                "signal": signal,
                                "direction": direction,
                                "horizon": horizon,
                                "filter": filter_name,
                                "n_events": len(sample),
                                "n_months": len(monthly),
                                "mean_raw_pct": 100 * sample["raw_return"].mean(),
                                "mean_excess_pct": 100 * sample["excess_return"].mean(),
                                "median_excess_pct": 100 * sample["excess_return"].median(),
                                "raw_hit_rate_pct": 100 * (sample["raw_return"] > 0).mean(),
                                "excess_hit_rate_pct": 100 * (sample["excess_return"] > 0).mean(),
                                "t_stat_month_cluster": float(t_stat),
                                "p_value": float(p_value),
                                "ci_low_pct": 100 * np.percentile(boot, 2.5),
                                "ci_high_pct": 100 * np.percentile(boot, 97.5),
                            }
                        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["q_value_bh"] = summary.groupby(["period", "direction", "horizon"])["p_value"].transform(benjamini_hochberg)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qqq = download_one(BENCHMARK)
    all_rows: list[dict] = []
    coverage: dict[str, dict] = {}
    for ticker in TICKERS:
        frame = download_one(ticker)
        if frame.empty:
            coverage[ticker] = {"status": "missing"}
            continue
        context = add_context(frame, qqq)
        signals = vmc_divergences(context).join(libertus_divergences(context))
        recent_vmc_bull = signals["vmc_any_bull"].rolling(6, min_periods=1).max().astype(bool)
        recent_vmc_bear = signals["vmc_any_bear"].rolling(6, min_periods=1).max().astype(bool)
        recent_lib_bull = signals["libertus_bull"].rolling(6, min_periods=1).max().astype(bool)
        recent_lib_bear = signals["libertus_bear"].rolling(6, min_periods=1).max().astype(bool)
        signals["confluence_bull"] = (
            (signals["vmc_any_bull"] & recent_lib_bull)
            | (signals["libertus_bull"] & recent_vmc_bull)
        )
        signals["confluence_bear"] = (
            (signals["vmc_any_bear"] & recent_lib_bear)
            | (signals["libertus_bear"] & recent_vmc_bear)
        )
        all_rows.extend(event_rows(ticker, context, signals))
        coverage[ticker] = {
            "status": "ok",
            "first": str(frame.index.min().date()),
            "last": str(frame.index.max().date()),
            "bars": int(len(frame)),
        }

    events = pd.DataFrame(all_rows)
    if events.empty:
        raise RuntimeError("No divergence events were produced")
    events["signal_date"] = pd.to_datetime(events["signal_date"])
    events["entry_date"] = pd.to_datetime(events["entry_date"])
    summary = summarize(events)

    events.to_csv(OUT / "events.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    holdout = summary[summary["period"] == "holdout_2020_2026"].copy()
    holdout = holdout.sort_values(["q_value_bh", "mean_excess_pct"], ascending=[True, False])
    holdout.head(100).to_csv(OUT / "holdout_ranked.csv", index=False)
    metadata = {
        "tickers": TICKERS,
        "benchmark": BENCHMARK,
        "requested_start": START,
        "requested_end": END,
        "holdout_start": str(HOLDOUT_START.date()),
        "entry_rule": "next session open after causal confirmation bar",
        "signal_gap_sessions": MIN_SIGNAL_GAP,
        "horizons_sessions": HORIZONS,
        "coverage": coverage,
        "limitations": [
            "fixed present-day universe creates survivorship bias",
            "Yahoo adjusted OHLC is research-grade convenience data, not exchange audit data",
            "event study is not a complete portfolio backtest with spreads, slippage and taxes",
            "multiple testing is controlled with Benjamini-Hochberg within period/direction/horizon",
        ],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"events={len(events):,} summary_rows={len(summary):,}")
    print(holdout.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
