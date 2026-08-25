"""Two-year 3/6/8-indicator search for the saved Bybit TradFi stock universe.

The first year selects candidates; the second year is a locked holdout.  The
underlying adjusted spot OHLC is used because historical Bybit TradFi contract
and funding series are not available in the workspace. Results therefore test
signal direction, not exact derivative P&L.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from divergence_study import benjamini_hochberg  # noqa: E402
from random_indicator_search import FEATURE_NAMES, encode, indicator_states  # noqa: E402
from src.tools.scanner_universes import TRADFI_SYMBOLS  # noqa: E402


RESEARCH = ROOT / "research"
CACHE = RESEARCH / "bybit_two_year_data"
EXTENDED_CACHE = RESEARCH / "vmc_ema_extended_data"
OUT = RESEARCH / "bybit_two_year_results"
WARMUP_START = "2023-08-01"
TEST_START = pd.Timestamp("2024-08-24")
HOLDOUT_START = pd.Timestamp("2025-08-24")
END = "2026-08-25"
HORIZONS = (5, 10, 20)
SEARCH_HORIZON = 10
ROUND_TRIP_COST = 0.0015
MIN_GAP = 10
RANDOM_PER_LARGE_SIZE = 2_500
TOP_PER_SIZE_DIRECTION = 20
RNG_SEED = 20260824


def cache_name(ticker: str) -> str:
    return ticker.replace(".", "_").replace("/", "_") + "_1d.csv"


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(column).lower() for column in frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    frame = frame[required].dropna(subset=["open", "high", "low", "close"])
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.sort_index()


def download_one(ticker: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / cache_name(ticker)
    if path.exists():
        frame = normalize(pd.read_csv(path, index_col=0, parse_dates=True))
        if not frame.empty:
            return frame

    extended_path = EXTENDED_CACHE / f"{ticker.replace('.', '-')}_1d.csv"
    if extended_path.exists():
        frame = normalize(pd.read_csv(extended_path, index_col=0, parse_dates=True))
        if not frame.empty and frame.index.max() >= pd.Timestamp("2026-08-20"):
            frame = frame.loc[frame.index >= pd.Timestamp(WARMUP_START)]
            frame.to_csv(path)
            return frame

    frame = pd.DataFrame()
    for _ in range(2):
        frame = yf.download(
            ticker,
            start=WARMUP_START,
            end=END,
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=30,
        )
        if not frame.empty:
            break
    if frame.empty:
        return frame
    frame = normalize(frame)
    frame.to_csv(path)
    return frame


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    qqq = download_one("QQQ")
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    for number, (ticker, contract, exchange) in enumerate(TRADFI_SYMBOLS, start=1):
        frame = download_one(ticker)
        if frame.empty:
            coverage_rows.append(
                {"ticker": ticker, "contract": contract, "exchange": exchange, "status": "missing"}
            )
            print(f"[{number:03d}/{len(TRADFI_SYMBOLS):03d}] {ticker}: missing", flush=True)
            continue
        q = qqq.reindex(frame.index).ffill()
        bull, bear = indicator_states(frame, qqq)
        data = pd.DataFrame(index=frame.index)
        data["ticker"] = ticker
        data["contract"] = contract
        data["exchange"] = exchange
        data["date"] = frame.index
        data["bull_mask"] = encode(bull)
        data["bear_mask"] = encode(bear)
        for horizon in HORIZONS:
            data[f"stock_return_{horizon}"] = frame["close"].shift(-horizon) / frame["open"].shift(-1) - 1
            data[f"qqq_return_{horizon}"] = q["close"].shift(-horizon) / q["open"].shift(-1) - 1
        data = data.loc[data.index >= TEST_START]
        frames.append(data.reset_index(drop=True))
        coverage_rows.append(
            {
                "ticker": ticker,
                "contract": contract,
                "exchange": exchange,
                "status": "ok",
                "first": str(frame.index.min().date()),
                "last": str(frame.index.max().date()),
                "test_bars": int(len(data)),
            }
        )
        print(f"[{number:03d}/{len(TRADFI_SYMBOLS):03d}] {ticker}: ok", flush=True)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(coverage_rows)


def combo_bits(indices: tuple[int, ...]) -> np.uint32:
    bits = np.uint32(0)
    for index in indices:
        bits |= np.uint32(1 << index)
    return bits


def candidates(size: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
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
    previous = np.r_[False, active[:-1]]
    previous[1:] &= ticker[1:] == ticker[:-1]
    transitions = active & ~previous
    selected: list[int] = []
    for _, group in data.loc[transitions].groupby("ticker", sort=False):
        last = -10_000
        for index in group.index.to_numpy(int):
            if index - last >= MIN_GAP:
                selected.append(index)
                last = index
    return np.asarray(selected, dtype=int)


def metric(data: pd.DataFrame, indices: np.ndarray, direction: str, horizon: int, period: str) -> dict | None:
    if len(indices) == 0:
        return None
    rows = data.loc[indices].copy()
    entry_date = rows["date"] + pd.offsets.BDay(1)
    if period == "development":
        rows = rows.loc[(entry_date >= TEST_START) & (entry_date < HOLDOUT_START)]
    elif period == "holdout":
        rows = rows.loc[entry_date >= HOLDOUT_START]
    sign = 1 if direction == "bull" else -1
    raw = sign * rows[f"stock_return_{horizon}"] - ROUND_TRIP_COST
    excess = sign * (rows[f"stock_return_{horizon}"] - rows[f"qqq_return_{horizon}"]) - ROUND_TRIP_COST
    valid = raw.notna() & excess.notna()
    rows, raw, excess = rows.loc[valid], raw.loc[valid], excess.loc[valid]
    if len(rows) < 15:
        return None
    monthly = rows.assign(value=raw, month=rows["date"].dt.to_period("M")).groupby("month")["value"].mean()
    if len(monthly) < 6:
        return None
    t_stat, p_value = stats.ttest_1samp(monthly.to_numpy(), 0.0)
    by_ticker = rows.assign(value=raw).groupby("ticker")["value"].mean()
    return {
        "n_events": int(len(rows)),
        "n_months": int(len(monthly)),
        "n_tickers": int(len(by_ticker)),
        "mean_raw_net_pct": 100 * float(raw.mean()),
        "median_raw_net_pct": 100 * float(raw.median()),
        "win_rate_net_pct": 100 * float((raw > 0).mean()),
        "mean_excess_qqq_pct": 100 * float(excess.mean()),
        "excess_win_rate_pct": 100 * float((excess > 0).mean()),
        "positive_ticker_pct": 100 * float((by_ticker > 0).mean()),
        "t_month_cluster": float(t_stat),
        "p_value": float(p_value),
    }


def development_score(data: pd.DataFrame, indices: np.ndarray, direction: str) -> dict | None:
    base = metric(data, indices, direction, SEARCH_HORIZON, "development")
    if base is None or base["n_events"] < 25 or base["n_tickers"] < 12:
        return None
    rows = data.loc[indices].copy()
    sign = 1 if direction == "bull" else -1
    raw = sign * rows[f"stock_return_{SEARCH_HORIZON}"] - ROUND_TRIP_COST
    valid = raw.notna()
    rows, raw = rows.loc[valid], raw.loc[valid]
    middle = TEST_START + (HOLDOUT_START - TEST_START) / 2
    early = raw.loc[(rows["date"] >= TEST_START) & (rows["date"] < middle)]
    late = raw.loc[(rows["date"] >= middle) & (rows["date"] < HOLDOUT_START)]
    if len(early) < 8 or len(late) < 8:
        return None
    early_mean = 100 * float(early.mean())
    late_mean = 100 * float(late.mean())
    if min(early_mean, late_mean) < -0.5:
        return None
    score = (
        base["mean_raw_net_pct"]
        + 0.012 * (base["win_rate_net_pct"] - 50)
        + 0.006 * (base["positive_ticker_pct"] - 50)
        + 0.20 * min(early_mean, late_mean)
    )
    return base | {"early_raw_pct": early_mean, "late_raw_pct": late_mean, "score": score}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, coverage = build_dataset()
    coverage.to_csv(OUT / "coverage.csv", index=False)
    print(f"dataset_rows={len(data):,}", flush=True)

    rng = np.random.default_rng(RNG_SEED)
    development_rows: list[dict] = []
    for size in (3, 6, 8):
        for indices_tuple in candidates(size, rng):
            bits = combo_bits(indices_tuple)
            names = " | ".join(FEATURE_NAMES[index] for index in indices_tuple)
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
        print(f"searched_size={size}", flush=True)

    development = pd.DataFrame(development_rows)
    development.to_csv(OUT / "development_candidates.csv", index=False)
    locked = (
        development.sort_values(["size", "direction", "score"], ascending=[True, True, False])
        .groupby(["size", "direction"], as_index=False, sort=True)
        .head(TOP_PER_SIZE_DIRECTION)
        .reset_index(drop=True)
    )
    locked["candidate_id"] = np.arange(len(locked))
    locked.to_csv(OUT / "selected_locked.csv", index=False)

    holdout_rows: list[dict] = []
    full_rows: list[dict] = []
    for candidate in locked.itertuples(index=False):
        selected = signal_indices(data, np.uint32(int(candidate.bits)), candidate.direction)
        for horizon in HORIZONS:
            result = metric(data, selected, candidate.direction, horizon, "holdout")
            if result is not None:
                holdout_rows.append(
                    {
                        "candidate_id": int(candidate.candidate_id),
                        "size": int(candidate.size),
                        "direction": candidate.direction,
                        "features": candidate.features,
                        "horizon": horizon,
                        **result,
                    }
                )
            result_full = metric(data, selected, candidate.direction, horizon, "full")
            if result_full is not None:
                full_rows.append(
                    {
                        "candidate_id": int(candidate.candidate_id),
                        "size": int(candidate.size),
                        "direction": candidate.direction,
                        "features": candidate.features,
                        "horizon": horizon,
                        **result_full,
                    }
                )
    holdout = pd.DataFrame(holdout_rows)
    if not holdout.empty:
        holdout["q_value_bh"] = holdout.groupby(["direction", "horizon"])["p_value"].transform(benjamini_hochberg)
        holdout = holdout.sort_values(["direction", "horizon", "q_value_bh", "mean_raw_net_pct"], ascending=[True, True, True, False])
    holdout.to_csv(OUT / "holdout_results.csv", index=False)
    pd.DataFrame(full_rows).to_csv(OUT / "full_period_descriptive.csv", index=False)
    pd.DataFrame(
        {"feature_number": np.arange(1, len(FEATURE_NAMES) + 1), "feature": FEATURE_NAMES}
    ).to_csv(OUT / "features.csv", index=False)
    metadata = {
        "saved_contract_count": len(TRADFI_SYMBOLS),
        "available_count": int(coverage["status"].eq("ok").sum()),
        "warmup_start": WARMUP_START,
        "test_start": str(TEST_START.date()),
        "holdout_start": str(HOLDOUT_START.date()),
        "end": END,
        "development": "first year only",
        "holdout": "second year only",
        "combination_counts": {"size_3": 1140, "size_6": RANDOM_PER_LARGE_SIZE, "size_8": RANDOM_PER_LARGE_SIZE},
        "round_trip_cost": ROUND_TRIP_COST,
        "limitations": [
            "spot adjusted OHLC, not Bybit TradFi contract mark-price history",
            "funding, swap, derivative basis, liquidation and extended-hours spreads are excluded",
            "two years is a short sample and multiple-testing risk remains after correction",
            "QQQ is an imperfect benchmark for Hong Kong, Korean and Chinese listings",
        ],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    positive = holdout.loc[holdout["mean_raw_net_pct"] > 0]
    print(f"development_candidates={len(development):,} locked={len(locked)} holdout_rows={len(holdout):,}", flush=True)
    print(
        positive.sort_values(["q_value_bh", "mean_raw_net_pct"], ascending=[True, False])
        .head(30)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
