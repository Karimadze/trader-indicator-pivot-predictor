"""Randomized 3/6/8-indicator search with a locked 2020+ holdout period.

Twenty common indicator states are encoded into bit masks.  Combinations are
selected only on 2010-2019 data, then evaluated once on 2020-2026 data.  This
is a research screen, not a promise of future returns or a live strategy.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from divergence_study import (
    BENCHMARK,
    HOLDOUT_START,
    OUT as DIVERGENCE_OUT,
    RNG_SEED,
    TICKERS,
    atr,
    benjamini_hochberg,
    download_one,
    ema,
    libertus_divergences,
    vmc_divergences,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "random_combo_results"
ROUND_TRIP_COST = 0.0015
MIN_GAP = 10
SEARCH_HORIZON = 10
HOLDOUT_HORIZONS = (5, 10, 20)
RANDOM_PER_LARGE_SIZE = 2_000
TOP_PER_SIZE_DIRECTION = 20

FEATURE_NAMES = [
    "vmc_divergence_recent",
    "libertus_divergence_recent",
    "rsi_direction",
    "macd_direction",
    "stochastic_direction",
    "williams_r_direction",
    "mfi_direction",
    "supertrend_direction",
    "donchian_break_recent",
    "bollinger_mid_direction",
    "ema_50_200_stack",
    "price_vs_ema200",
    "adx_direction",
    "obv_direction",
    "high_volume_confirmation",
    "relative_strength_vs_qqq",
    "qqq_market_regime",
    "52_week_extreme_proximity",
    "roc_63_direction",
    "ichimoku_direction",
]


def cross_up(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=a.index)
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_down(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=a.index)
    return (a < b) & (a.shift(1) >= b.shift(1))


def recent(signal: pd.Series, bars: int = 5) -> pd.Series:
    return signal.fillna(False).rolling(bars, min_periods=1).max().astype(bool)


def dmi(frame: pd.DataFrame, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=frame.index)
    tr = atr(frame, length)
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return plus_di, minus_di, adx


def money_flow_index(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3
    raw = typical * frame["volume"]
    positive = raw.where(typical.diff() > 0, 0.0).rolling(length).sum()
    negative = raw.where(typical.diff() < 0, 0.0).rolling(length).sum()
    ratio = positive / negative.replace(0, np.nan)
    return 100 - 100 / (1 + ratio)


def supertrend_state(frame: pd.DataFrame, length: int = 10, factor: float = 3.0) -> pd.Series:
    middle = (frame["high"] + frame["low"]) / 2
    volatility = atr(frame, length)
    basic_upper = middle + factor * volatility
    basic_lower = middle - factor * volatility
    upper = basic_upper.to_numpy(float).copy()
    lower = basic_lower.to_numpy(float).copy()
    close = frame["close"].to_numpy(float)
    bullish = np.zeros(len(frame), dtype=bool)
    bullish[0] = True
    for i in range(1, len(frame)):
        if np.isnan(upper[i - 1]) or np.isnan(lower[i - 1]):
            bullish[i] = bullish[i - 1]
            continue
        if basic_upper.iloc[i] < upper[i - 1] or close[i - 1] > upper[i - 1]:
            upper[i] = basic_upper.iloc[i]
        else:
            upper[i] = upper[i - 1]
        if basic_lower.iloc[i] > lower[i - 1] or close[i - 1] < lower[i - 1]:
            lower[i] = basic_lower.iloc[i]
        else:
            lower[i] = lower[i - 1]
        if close[i] > upper[i - 1]:
            bullish[i] = True
        elif close[i] < lower[i - 1]:
            bullish[i] = False
        else:
            bullish[i] = bullish[i - 1]
    return pd.Series(bullish, index=frame.index)


def indicator_states(frame: pd.DataFrame, qqq: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = qqq.reindex(frame.index).ffill()
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]

    vmc = vmc_divergences(frame)
    lib = libertus_divergences(frame)
    rsi = lib["rsi"]

    macd = ema(close, 12) - ema(close, 26)
    macd_signal = ema(macd, 9)
    lowest14 = low.rolling(14).min()
    highest14 = high.rolling(14).max()
    stochastic_k = 100 * (close - lowest14) / (highest14 - lowest14).replace(0, np.nan)
    stochastic_d = stochastic_k.rolling(3).mean()
    williams_r = -100 * (highest14 - close) / (highest14 - lowest14).replace(0, np.nan)
    mfi = money_flow_index(frame)
    supertrend_bull = supertrend_state(frame)

    prior_high20 = high.rolling(20).max().shift(1)
    prior_low20 = low.rolling(20).min().shift(1)
    donchian_up = recent(close > prior_high20, 5)
    donchian_down = recent(close < prior_low20, 5)

    bb_mid = close.rolling(20).mean()
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    plus_di, minus_di, adx_value = dmi(frame)
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    obv_avg = ema(obv, 20)
    high_volume = volume >= 1.2 * volume.rolling(20).mean()

    stock_return126 = close.pct_change(126)
    qqq_return126 = q["close"].pct_change(126)
    qqq_ema200 = ema(q["close"], 200)
    high252 = high.rolling(252).max()
    low252 = low.rolling(252).min()
    roc63 = close.pct_change(63)

    conversion = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conversion + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    cloud_high = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_low = pd.concat([span_a, span_b], axis=1).min(axis=1)

    bull = pd.DataFrame(
        {
            "vmc_divergence_recent": recent(vmc["vmc_any_bull"], 6),
            "libertus_divergence_recent": recent(lib["libertus_bull"], 6),
            "rsi_direction": rsi > 50,
            "macd_direction": macd > macd_signal,
            "stochastic_direction": stochastic_k > stochastic_d,
            "williams_r_direction": williams_r > -50,
            "mfi_direction": mfi > 50,
            "supertrend_direction": supertrend_bull,
            "donchian_break_recent": donchian_up,
            "bollinger_mid_direction": close > bb_mid,
            "ema_50_200_stack": (close > ema50) & (ema50 > ema200),
            "price_vs_ema200": close > ema200,
            "adx_direction": (adx_value > 20) & (plus_di > minus_di),
            "obv_direction": obv > obv_avg,
            "high_volume_confirmation": high_volume,
            "relative_strength_vs_qqq": stock_return126 > qqq_return126,
            "qqq_market_regime": q["close"] > qqq_ema200,
            "52_week_extreme_proximity": close >= 0.90 * high252,
            "roc_63_direction": roc63 > 0,
            "ichimoku_direction": (conversion > base) & (close > cloud_high),
        },
        index=frame.index,
    )
    bear = pd.DataFrame(
        {
            "vmc_divergence_recent": recent(vmc["vmc_any_bear"], 6),
            "libertus_divergence_recent": recent(lib["libertus_bear"], 6),
            "rsi_direction": rsi < 50,
            "macd_direction": macd < macd_signal,
            "stochastic_direction": stochastic_k < stochastic_d,
            "williams_r_direction": williams_r < -50,
            "mfi_direction": mfi < 50,
            "supertrend_direction": ~supertrend_bull,
            "donchian_break_recent": donchian_down,
            "bollinger_mid_direction": close < bb_mid,
            "ema_50_200_stack": (close < ema50) & (ema50 < ema200),
            "price_vs_ema200": close < ema200,
            "adx_direction": (adx_value > 20) & (minus_di > plus_di),
            "obv_direction": obv < obv_avg,
            "high_volume_confirmation": high_volume,
            "relative_strength_vs_qqq": stock_return126 < qqq_return126,
            "qqq_market_regime": q["close"] < qqq_ema200,
            "52_week_extreme_proximity": close <= 1.10 * low252,
            "roc_63_direction": roc63 < 0,
            "ichimoku_direction": (conversion < base) & (close < cloud_low),
        },
        index=frame.index,
    )
    return bull.fillna(False), bear.fillna(False)


def encode(states: pd.DataFrame) -> np.ndarray:
    values = states[FEATURE_NAMES].to_numpy(dtype=bool)
    powers = (np.uint32(1) << np.arange(len(FEATURE_NAMES), dtype=np.uint32))
    return (values.astype(np.uint32) * powers).sum(axis=1, dtype=np.uint32)


def build_dataset() -> pd.DataFrame:
    qqq = download_one(BENCHMARK)
    frames: list[pd.DataFrame] = []
    for ticker in TICKERS:
        frame = download_one(ticker)
        q = qqq.reindex(frame.index).ffill()
        bull, bear = indicator_states(frame, qqq)
        data = pd.DataFrame(index=frame.index)
        data["ticker"] = ticker
        data["date"] = frame.index
        data["bull_mask"] = encode(bull)
        data["bear_mask"] = encode(bear)
        for horizon in HOLDOUT_HORIZONS:
            stock_return = frame["close"].shift(-horizon) / frame["open"].shift(-1) - 1
            qqq_return = q["close"].shift(-horizon) / q["open"].shift(-1) - 1
            data[f"stock_return_{horizon}"] = stock_return
            data[f"qqq_return_{horizon}"] = qqq_return
        frames.append(data.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def combo_bits(indices: tuple[int, ...]) -> np.uint32:
    bits = np.uint32(0)
    for i in indices:
        bits |= np.uint32(1 << i)
    return bits


def candidate_combinations(size: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    if size == 3:
        return list(combinations(range(len(FEATURE_NAMES)), size))
    choices: set[tuple[int, ...]] = set()
    while len(choices) < RANDOM_PER_LARGE_SIZE:
        choices.add(tuple(sorted(rng.choice(len(FEATURE_NAMES), size=size, replace=False).tolist())))
    return sorted(choices)


def signal_indices(data: pd.DataFrame, bits: np.uint32, direction: str) -> np.ndarray:
    masks = data[f"{direction}_mask"].to_numpy(np.uint32)
    active = (masks & bits) == bits
    ticker = data["ticker"].to_numpy()
    previous = np.roll(active, 1)
    previous[0] = False
    previous[1:] &= ticker[1:] == ticker[:-1]
    transitions = active & ~previous
    selected: list[int] = []
    for _, group in data.loc[transitions].groupby("ticker", sort=False):
        last = -10_000
        for idx in group.index.to_numpy(int):
            if idx - last >= MIN_GAP:
                selected.append(idx)
                last = idx
    return np.asarray(selected, dtype=int)


def metrics(data: pd.DataFrame, indices: np.ndarray, direction: str, horizon: int, period: str) -> dict | None:
    if len(indices) == 0:
        return None
    rows = data.loc[indices].copy()
    entry_dates = rows["date"] + pd.offsets.BDay(1)
    if period == "development":
        rows = rows.loc[entry_dates < HOLDOUT_START]
    elif period == "holdout":
        rows = rows.loc[entry_dates >= HOLDOUT_START]
    if len(rows) < 15:
        return None
    sign = 1 if direction == "bull" else -1
    raw = sign * rows[f"stock_return_{horizon}"] - ROUND_TRIP_COST
    excess = sign * (rows[f"stock_return_{horizon}"] - rows[f"qqq_return_{horizon}"]) - ROUND_TRIP_COST
    valid = raw.notna() & excess.notna()
    rows, raw, excess = rows.loc[valid], raw.loc[valid], excess.loc[valid]
    if len(rows) < 15:
        return None
    months = rows.assign(value=excess, month=rows["date"].dt.to_period("M")).groupby("month")["value"].mean()
    if len(months) < 6:
        return None
    t_stat, p_value = stats.ttest_1samp(months.to_numpy(), 0.0)
    by_ticker = rows.assign(value=excess).groupby("ticker")["value"].mean()
    return {
        "n_events": int(len(rows)),
        "n_months": int(len(months)),
        "n_tickers": int(len(by_ticker)),
        "mean_raw_net_pct": float(100 * raw.mean()),
        "mean_excess_net_pct": float(100 * excess.mean()),
        "median_excess_net_pct": float(100 * excess.median()),
        "win_rate_net_pct": float(100 * (raw > 0).mean()),
        "excess_win_rate_pct": float(100 * (excess > 0).mean()),
        "positive_ticker_pct": float(100 * (by_ticker > 0).mean()),
        "t_stat_month_cluster": float(t_stat),
        "p_value": float(p_value),
    }


def development_score(data: pd.DataFrame, indices: np.ndarray, direction: str) -> dict | None:
    base = metrics(data, indices, direction, SEARCH_HORIZON, "development")
    if base is None or base["n_events"] < 50 or base["n_tickers"] < 8:
        return None
    rows = data.loc[indices].copy()
    rows = rows.loc[rows["date"] < HOLDOUT_START]
    sign = 1 if direction == "bull" else -1
    excess = sign * (rows[f"stock_return_{SEARCH_HORIZON}"] - rows[f"qqq_return_{SEARCH_HORIZON}"]) - ROUND_TRIP_COST
    early = excess.loc[rows["date"] < pd.Timestamp("2015-01-01")].dropna()
    late = excess.loc[rows["date"] >= pd.Timestamp("2015-01-01")].dropna()
    if len(early) < 15 or len(late) < 15:
        return None
    early_mean = float(100 * early.mean())
    late_mean = float(100 * late.mean())
    # Reward repeatability across both halves, not a single lucky era.
    score = min(early_mean, late_mean) + 0.20 * base["mean_excess_net_pct"] + 0.01 * (base["win_rate_net_pct"] - 50)
    return {**base, "early_excess_pct": early_mean, "late_excess_pct": late_mean, "score": score}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    rng = np.random.default_rng(RNG_SEED)
    development_rows: list[dict] = []

    for size in (3, 6, 8):
        for indices_tuple in candidate_combinations(size, rng):
            bits = combo_bits(indices_tuple)
            names = " | ".join(FEATURE_NAMES[i] for i in indices_tuple)
            for direction in ("bull", "bear"):
                selected = signal_indices(data, bits, direction)
                result = development_score(data, selected, direction)
                if result is not None:
                    development_rows.append(
                        {
                            "size": size,
                            "direction": direction,
                            "bits": int(bits),
                            "features": names,
                            **result,
                        }
                    )

    development = pd.DataFrame(development_rows)
    development.to_csv(OUT / "development_candidates.csv", index=False)
    selected = (
        development.sort_values("score", ascending=False)
        .groupby(["size", "direction"], group_keys=False)
        .head(TOP_PER_SIZE_DIRECTION)
        .reset_index(drop=True)
    )
    selected.to_csv(OUT / "selected_locked.csv", index=False)

    holdout_rows: list[dict] = []
    for candidate_id, row in selected.iterrows():
        bits = np.uint32(row["bits"])
        signal_rows = signal_indices(data, bits, row["direction"])
        for horizon in HOLDOUT_HORIZONS:
            result = metrics(data, signal_rows, row["direction"], horizon, "holdout")
            if result is not None:
                holdout_rows.append(
                    {
                        "candidate_id": int(candidate_id),
                        "size": int(row["size"]),
                        "direction": row["direction"],
                        "horizon": horizon,
                        "features": row["features"],
                        "development_score": row["score"],
                        "development_excess_pct": row["mean_excess_net_pct"],
                        "development_win_rate_pct": row["win_rate_net_pct"],
                        **result,
                    }
                )
    holdout = pd.DataFrame(holdout_rows)
    if not holdout.empty:
        holdout["q_value_bh"] = holdout.groupby(["direction", "horizon"])["p_value"].transform(benjamini_hochberg)
        holdout = holdout.sort_values(["q_value_bh", "mean_excess_net_pct"], ascending=[True, False])
    holdout.to_csv(OUT / "holdout_results.csv", index=False)

    meta = pd.DataFrame(
        {
            "feature_number": np.arange(1, len(FEATURE_NAMES) + 1),
            "feature": FEATURE_NAMES,
        }
    )
    meta.to_csv(OUT / "features.csv", index=False)
    print(f"dataset_rows={len(data):,} development_candidates={len(development):,} locked={len(selected):,} holdout_rows={len(holdout):,}")
    print(holdout.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
