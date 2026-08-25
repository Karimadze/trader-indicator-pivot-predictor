"""Rank the saved Bybit TradFi stock universe by recent underlying volatility."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from bybit_two_year_combo_search import download_one  # noqa: E402
from src.tools.scanner_universes import TRADFI_SYMBOLS  # noqa: E402


OUT = ROOT / "research" / "bybit_volatility_snapshot.csv"


def bybit_tickers() -> dict[str, dict]:
    response = requests.get(
        "https://api.bybit.com/v5/market/tickers",
        params={"category": "linear"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit ticker request failed: {payload.get('retMsg')}")
    return {row["symbol"]: row for row in payload.get("result", {}).get("list", [])}


def annualized_volatility(close: pd.Series, window: int) -> float:
    returns = np.log(close).diff().dropna().tail(window)
    if len(returns) < max(15, window // 2):
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(252) * 100)


def maximum_drawdown(close: pd.Series, window: int) -> float:
    sample = close.dropna().tail(window)
    if sample.empty:
        return np.nan
    drawdown = sample / sample.cummax() - 1
    return float(drawdown.min() * 100)


def atr_percent(frame: pd.DataFrame, window: int = 14) -> float:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return float(atr.iloc[-1] / frame["close"].iloc[-1] * 100)


def summarize(
    ticker: str,
    contract: str,
    exchange: str,
    live_tickers: dict[str, dict],
) -> dict[str, object] | None:
    frame = download_one(ticker)
    if frame.empty or len(frame) < 30:
        return None
    returns = frame["close"].pct_change()
    dollar_volume = frame["close"] * frame["volume"]
    live = live_tickers.get(contract, {})
    bid = float(live.get("bid1Price") or np.nan)
    ask = float(live.get("ask1Price") or np.nan)
    midpoint = (bid + ask) / 2
    spread_bps = (ask - bid) / midpoint * 10_000 if midpoint > 0 else np.nan
    return {
        "ticker": ticker,
        "contract": contract,
        "exchange": exchange,
        "last_date": frame.index[-1].date().isoformat(),
        "last_close": float(frame["close"].iloc[-1]),
        "volatility_20d_pct": annualized_volatility(frame["close"], 20),
        "volatility_60d_pct": annualized_volatility(frame["close"], 60),
        "volatility_126d_pct": annualized_volatility(frame["close"], 126),
        "atr_14_pct": atr_percent(frame),
        "mean_abs_return_60d_pct": float(returns.tail(60).abs().mean() * 100),
        "max_abs_return_60d_pct": float(returns.tail(60).abs().max() * 100),
        "max_drawdown_126d_pct": maximum_drawdown(frame["close"], 126),
        "avg_dollar_volume_20d_m": float(dollar_volume.tail(20).mean() / 1_000_000),
        "bybit_turnover_24h_usdt": float(live.get("turnover24h") or np.nan),
        "bybit_open_interest_usdt": float(live.get("openInterestValue") or np.nan),
        "bybit_spread_bps": spread_bps,
        "bybit_funding_rate_pct": float(live.get("fundingRate") or np.nan) * 100,
    }


def main() -> None:
    live_tickers = bybit_tickers()
    rows = []
    for ticker, contract, exchange in TRADFI_SYMBOLS:
        row = summarize(ticker, contract, exchange, live_tickers)
        if row:
            rows.append(row)
    result = pd.DataFrame(rows).sort_values("volatility_60d_pct", ascending=False)
    result.insert(0, "rank_60d", np.arange(1, len(result) + 1))
    result.to_csv(OUT, index=False)
    print(result.head(30).to_string(index=False))
    print(f"\nSaved {len(result)} rows to {OUT}")


if __name__ == "__main__":
    main()
