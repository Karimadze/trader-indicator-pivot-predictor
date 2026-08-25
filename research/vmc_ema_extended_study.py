"""Extended causal study of VuManChu bullish divergence plus EMA stack.

The study expands the original 20-stock/2010+ event study to the current
Nasdaq-100 security list and the longest convenient Yahoo daily history since
1985. Signals are timestamped only when the VMC pivot is confirmed, with entry
at the following session open. Current constituents introduce survivorship
bias; that limitation is explicit in the generated report metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy import stats
import yfinance as yf

from divergence_study import TICKERS as ORIGINAL_20
from divergence_study import benjamini_hochberg, ema, purge_mask, vmc_divergences


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "vmc_ema_extended_data"
OUT = ROOT / "vmc_ema_extended_results"
NASDAQ_COMPANIES_URL = "https://www.nasdaq.com/products/global-indexes/nasdaq-100/companies"
START = "1985-01-01"
END = "2026-08-25"
HORIZONS = (5, 10, 20, 40, 60)
COST = 0.0015
RNG_SEED = 20260824

# Nasdaq announced these changes after the company page's stated update date.
JUNE_2026_ADDITIONS = {"ALAB", "CRWV", "NBIS", "RKLB", "TER"}
JUNE_2026_REMOVALS = {"CHTR", "CTSH", "INSM", "VRSK", "ZS"}
JULY_2026_ADDITION = {"SPCX"}


def current_nasdaq100_symbols() -> list[str]:
    response = requests.get(
        NASDAQ_COMPANIES_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 research-script"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    symbols: set[str] = set()
    def collect_items(node: object) -> list[dict]:
        found: list[dict] = []
        if isinstance(node, dict):
            items = node.get("itemListElement")
            if isinstance(items, list):
                found.extend(item for item in items if isinstance(item, dict))
            for value in node.values():
                found.extend(collect_items(value))
        elif isinstance(node, list):
            for value in node:
                found.extend(collect_items(value))
        return found

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.string or script.get_text())
        except (TypeError, json.JSONDecodeError):
            continue
        for item in collect_items(payload):
            symbol = str(item.get("description", "")).strip().upper()
            if symbol and symbol.replace(".", "").isalnum() and len(symbol) <= 6:
                symbols.add(symbol)
    if len(symbols) < 90:
        raise RuntimeError(f"Official Nasdaq page returned only {len(symbols)} symbols")
    symbols -= JUNE_2026_REMOVALS
    symbols |= JUNE_2026_ADDITIONS | JULY_2026_ADDITION
    return sorted(symbols)


def download_one(ticker: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace(".", "-")
    path = CACHE / f"{safe}_1d.csv"
    if path.exists():
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if not frame.empty:
            return frame
    yahoo_ticker = ticker.replace(".", "-")
    frame = yf.download(
        yahoo_ticker,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=30,
    )
    if frame.empty:
        return frame
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(column).lower() for column in frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    frame = frame[required].dropna(subset=["open", "high", "low", "close"])
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.to_csv(path)
    return frame


def period_name(date: pd.Timestamp) -> str:
    if date < pd.Timestamp("2000-01-01"):
        return "1985-1999"
    if date < pd.Timestamp("2010-01-01"):
        return "2000-2009"
    if date < pd.Timestamp("2020-01-01"):
        return "2010-2019"
    return "2020-2026"


def prepare_events(ticker: str, frame: pd.DataFrame, qqq: pd.DataFrame, current_ndx: bool) -> list[dict]:
    if len(frame) < 260:
        return []
    close = frame["close"]
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    stack = (close > ema50) & (ema50 > ema200)
    above_ema200 = close > ema200
    vmc = vmc_divergences(frame)
    signal = purge_mask(vmc["vmc_any_bull"], gap=10)
    q = qqq.reindex(frame.index).ffill()
    q_ema200 = ema(q["close"], 200)

    forward: dict[int, pd.Series] = {}
    stack_control: dict[int, pd.Series] = {}
    for horizon in HORIZONS:
        ret = frame["close"].shift(-horizon) / frame["open"].shift(-1) - 1 - COST
        forward[horizon] = ret
        controls = pd.DataFrame(
            {"year": frame.index.year, "return": ret, "stack": stack},
            index=frame.index,
        )
        control_by_year = controls.loc[controls["stack"]].groupby("year")["return"].mean()
        stack_control[horizon] = pd.Series(frame.index.year, index=frame.index).map(control_by_year)

    rows: list[dict] = []
    for signal_i in np.flatnonzero(signal.to_numpy(bool)):
        entry_i = signal_i + 1
        if entry_i >= len(frame):
            continue
        signal_date = frame.index[signal_i]
        entry_date = frame.index[entry_i]
        primary = bool(vmc["vmc_primary_bull"].iloc[signal_i])
        for horizon in HORIZONS:
            exit_i = signal_i + horizon
            if exit_i >= len(frame) or not np.isfinite(forward[horizon].iloc[signal_i]):
                continue
            entry = float(frame["open"].iloc[entry_i])
            path = frame.iloc[entry_i : exit_i + 1]
            raw_return = float(forward[horizon].iloc[signal_i])
            qqq_return = np.nan
            if (
                entry_date in q.index
                and frame.index[exit_i] in q.index
                and np.isfinite(q.loc[entry_date, "open"])
                and np.isfinite(q.loc[frame.index[exit_i], "close"])
            ):
                qqq_return = float(q.loc[frame.index[exit_i], "close"] / q.loc[entry_date, "open"] - 1 - COST)
            control_return = float(stack_control[horizon].iloc[signal_i]) if stack.iloc[signal_i] else np.nan
            rows.append(
                {
                    "ticker": ticker,
                    "current_ndx": current_ndx,
                    "original20": ticker in ORIGINAL_20,
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "exit_date": frame.index[exit_i],
                    "calendar_period": period_name(entry_date),
                    "year": int(entry_date.year),
                    "horizon": horizon,
                    "primary_divergence": primary,
                    "above_ema200": bool(above_ema200.iloc[signal_i]),
                    "ema_stack": bool(stack.iloc[signal_i]),
                    "qqq_bull_regime": bool(q.loc[signal_date, "close"] > q_ema200.loc[signal_date])
                    if signal_date in q.index and np.isfinite(q_ema200.loc[signal_date])
                    else np.nan,
                    "ema50_gap_pct": 100 * float(close.iloc[signal_i] / ema50.iloc[signal_i] - 1),
                    "ema200_gap_pct": 100 * float(ema50.iloc[signal_i] / ema200.iloc[signal_i] - 1),
                    "raw_return": raw_return,
                    "qqq_return": qqq_return,
                    "excess_return": raw_return - qqq_return if np.isfinite(qqq_return) else np.nan,
                    "stack_control_return": control_return,
                    "incremental_vs_stack": raw_return - control_return if np.isfinite(control_return) else np.nan,
                    "mae": float(path["low"].min() / entry - 1),
                    "mfe": float(path["high"].max() / entry - 1),
                }
            )
    return rows


def clustered_test(values: pd.Series, dates: pd.Series) -> tuple[float, float, float, float]:
    valid = values.notna()
    if valid.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan
    monthly = pd.DataFrame(
        {"value": values.loc[valid].to_numpy(), "month": dates.loc[valid].dt.to_period("M").to_numpy()}
    ).groupby("month")["value"].mean()
    if len(monthly) < 6:
        return np.nan, np.nan, np.nan, np.nan
    t_stat, p_value = stats.ttest_1samp(monthly.to_numpy(), 0.0)
    rng = np.random.default_rng(RNG_SEED)
    boot = np.array(
        [rng.choice(monthly.to_numpy(), len(monthly), replace=True).mean() for _ in range(3000)]
    )
    return float(t_stat), float(p_value), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def metrics(sample: pd.DataFrame) -> dict | None:
    if len(sample) < 10:
        return None
    raw_t, raw_p, raw_low, raw_high = clustered_test(sample["raw_return"], sample["entry_date"])
    excess_t, excess_p, _, _ = clustered_test(sample["excess_return"], sample["entry_date"])
    inc_t, inc_p, inc_low, inc_high = clustered_test(sample["incremental_vs_stack"], sample["entry_date"])
    by_ticker = sample.groupby("ticker")["raw_return"].mean()
    return {
        "n_events": int(len(sample)),
        "n_tickers": int(sample["ticker"].nunique()),
        "mean_raw_net_pct": 100 * float(sample["raw_return"].mean()),
        "median_raw_net_pct": 100 * float(sample["raw_return"].median()),
        "win_rate_pct": 100 * float((sample["raw_return"] > 0).mean()),
        "mean_excess_qqq_pct": 100 * float(sample["excess_return"].mean()),
        "excess_win_rate_pct": 100 * float((sample["excess_return"] > 0).mean()),
        "incremental_vs_stack_pct": 100 * float(sample["incremental_vs_stack"].mean()),
        "incremental_win_rate_pct": 100 * float((sample["incremental_vs_stack"] > 0).mean()),
        "mean_mae_pct": 100 * float(sample["mae"].mean()),
        "mean_mfe_pct": 100 * float(sample["mfe"].mean()),
        "positive_ticker_pct": 100 * float((by_ticker > 0).mean()),
        "raw_t_month_cluster": raw_t,
        "raw_p_value": raw_p,
        "raw_ci_low_pct": 100 * raw_low,
        "raw_ci_high_pct": 100 * raw_high,
        "excess_t_month_cluster": excess_t,
        "excess_p_value": excess_p,
        "incremental_t_month_cluster": inc_t,
        "incremental_p_value": inc_p,
        "incremental_ci_low_pct": 100 * inc_low,
        "incremental_ci_high_pct": 100 * inc_high,
    }


def make_summary(events: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "vmc_bull_unfiltered": pd.Series(True, index=events.index),
        "vmc_bull_above_ema200": events["above_ema200"],
        "vmc_bull_ema_stack": events["ema_stack"],
    }
    periods = ["full", "1985-1999", "2000-2009", "2010-2019", "2020-2026"]
    rows: list[dict] = []
    for universe, universe_mask in {
        "current_nasdaq100": events["current_ndx"],
        "original_20": events["original20"],
    }.items():
        for period in periods:
            period_mask = pd.Series(True, index=events.index) if period == "full" else events["calendar_period"].eq(period)
            for rule, rule_mask in rules.items():
                for horizon in HORIZONS:
                    sample = events.loc[universe_mask & period_mask & rule_mask & events["horizon"].eq(horizon)]
                    result = metrics(sample)
                    if result is not None:
                        rows.append(
                            {"universe": universe, "period": period, "rule": rule, "horizon": horizon} | result
                        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["raw_q_value_bh"] = summary.groupby(["universe", "period"])["raw_p_value"].transform(benjamini_hochberg)
        summary["excess_q_value_bh"] = summary.groupby(["universe", "period"])["excess_p_value"].transform(benjamini_hochberg)
        summary["incremental_q_value_bh"] = summary.groupby(["universe", "period"])["incremental_p_value"].transform(benjamini_hochberg)
    return summary


def grouped_breakdown(events: pd.DataFrame, group: str) -> pd.DataFrame:
    focus = events.loc[events["ema_stack"] & events["horizon"].isin([10, 20])].copy()
    rows: list[dict] = []
    for keys, sample in focus.groupby([group, "horizon"], dropna=False):
        result = metrics(sample)
        if result is not None:
            value, horizon = keys
            rows.append({group: value, "horizon": horizon} | result)
    return pd.DataFrame(rows)


def descriptive_breakdown(events: pd.DataFrame, group: str) -> pd.DataFrame:
    focus = events.loc[events["ema_stack"] & events["horizon"].isin([10, 20])].copy()
    rows: list[dict] = []
    for keys, sample in focus.groupby([group, "horizon"], dropna=False):
        value, horizon = keys
        rows.append(
            {
                group: value,
                "horizon": horizon,
                "n_events": int(len(sample)),
                "mean_raw_net_pct": 100 * float(sample["raw_return"].mean()),
                "median_raw_net_pct": 100 * float(sample["raw_return"].median()),
                "win_rate_pct": 100 * float((sample["raw_return"] > 0).mean()),
                "mean_excess_qqq_pct": 100 * float(sample["excess_return"].mean()),
                "incremental_vs_stack_pct": 100 * float(sample["incremental_vs_stack"].mean()),
                "mean_mae_pct": 100 * float(sample["mae"].mean()),
                "mean_mfe_pct": 100 * float(sample["mfe"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    current_symbols = current_nasdaq100_symbols()
    symbols = sorted(set(current_symbols) | set(ORIGINAL_20))
    qqq = download_one("QQQ")
    coverage: dict[str, dict] = {}
    all_rows: list[dict] = []
    for number, ticker in enumerate(symbols, start=1):
        frame = download_one(ticker)
        if frame.empty:
            coverage[ticker] = {"status": "missing"}
        else:
            coverage[ticker] = {
                "status": "ok",
                "first": str(frame.index.min().date()),
                "last": str(frame.index.max().date()),
                "bars": int(len(frame)),
            }
            all_rows.extend(prepare_events(ticker, frame, qqq, ticker in current_symbols))
        print(f"[{number:03d}/{len(symbols):03d}] {ticker}: {coverage[ticker]['status']}", flush=True)

    events = pd.DataFrame(all_rows)
    events.to_csv(OUT / "events.csv", index=False)
    summary = make_summary(events)
    summary.to_csv(OUT / "summary.csv", index=False)
    descriptive_breakdown(events, "ticker").to_csv(OUT / "by_ticker.csv", index=False)
    grouped_breakdown(events, "calendar_period").to_csv(OUT / "by_period.csv", index=False)
    grouped_breakdown(events, "primary_divergence").to_csv(OUT / "by_signal_strength.csv", index=False)
    grouped_breakdown(events, "qqq_bull_regime").to_csv(OUT / "by_market_regime.csv", index=False)
    descriptive_breakdown(events, "year").to_csv(OUT / "by_year.csv", index=False)
    metadata = {
        "current_nasdaq100_symbols": current_symbols,
        "current_nasdaq100_symbol_count": len(current_symbols),
        "all_tested_symbols": symbols,
        "all_tested_symbol_count": len(symbols),
        "source": NASDAQ_COMPANIES_URL,
        "start": START,
        "end": END,
        "horizons": HORIZONS,
        "round_trip_cost": COST,
        "entry": "next session open after causal VMC pivot confirmation",
        "ema_stack": "signal-bar close > EMA50 > EMA200",
        "coverage": coverage,
        "limitations": [
            "current-constituent universe has survivorship and lookback membership bias",
            "GOOG and GOOGL are separate securities of the same company",
            "Yahoo adjusted daily OHLC is convenience data, not exchange audit data",
            "fixed-horizon event returns are not a portfolio backtest",
            "overlapping positions and correlated signals are possible",
        ],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    focus = summary.loc[
        summary["rule"].eq("vmc_bull_ema_stack")
        & summary["period"].isin(["full", "2010-2019", "2020-2026"])
    ]
    print(f"events={len(events):,} securities={len(symbols)}", flush=True)
    print(focus.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
