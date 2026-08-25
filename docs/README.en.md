# Trader Indicator Pivot Predictor

**🌐 Languages:** [Русский](../README.md) · **English** (current) · [中文](README.zh.md) · [हिन्दी](README.hi.md) · [Deutsch](README.de.md) · [Español](README.es.md)

A set of Pine Script (TradingView) indicators plus a standalone Python scanner for Bybit that try to **predict RSI pivot points in real time** — before the bar closes, instead of after the fact like TradingView's stock "RSI Div - Lib" indicator (which only draws PIVOT labels 2 bars late).

This project is the result of honest, measured research, not marketing. There are no "90% accuracy" claims here. Below are the real numbers, measured on historical data with a time-based train/test split (train before 2021, test after) to avoid overfitting.

## ⚠️ Disclaimer

This is not financial advice and not a guarantee of profit. The indicators and scanner are research tools built on statistics of past prices. Trading on any signal, including these, carries risk of capital loss. The author is not a financial advisor. Verify everything yourself before making decisions.

## What's inside

- **`tradingview/`** — Pine Script v6 indicators for TradingView:
  - `RealtimePivotPredictor.pine` — a 0–100 reversal-probability score from a logistic-regression model with 7 features (RSI, stretch from MA, position vs EMA200, ATR, candle body, 3-bar return). Non-repainting (`barstate.isconfirmed`, no lookahead).
  - `ReversalRadar.pine` / `ReversalRadarV2.pine` — "Reversal Radar": a 7-component confluence indicator (CCI, Bollinger Bands, RSI cross, Fisher Transform, Ultimate Oscillator, TD Sequential 9, Connors RSI), symmetric bottom (LONG) and top (TAKE — not a short signal, see below) signals, an RSI panel matching RSI Div-Lib, and an optional market-breadth filter.
  - `CausalPivotCandidate.pine`, `CyclicSmoothedRSI_MTF.pine`, `NasdaqVMCSeparatePane.pine`, `BybitStockPerpRegimeBreakout.pine` — supporting/experimental indicators from the same research line.

- **`research/`** — the full path to the results: the Python scripts everything was measured with, and Markdown reports with the numbers (`REPORT_RU.md`, `PIVOT_ANATOMY_RU.md`, `BREADTH_REPORT_RU.md`, `VMC_EMA_EXTENDED_REPORT_RU.md`, `BYBIT_2Y_COMBO_REPORT_RU.md` — currently in Russian; translation help welcome), plus reproducible scripts and CSV data for the original 7-ticker dataset and a Bybit volatility snapshot.

- **`scanner/`** — a standalone Python scanner (no TradingView dependency) that scans the **entire live universe** of TradFi perpetuals and xStocks on Bybit (1h/4h), including a still-forming (unclosed) candle:
  - `bybit_radar_scanner.py` — stdlib-only, Python 3.8+.
  - Volatility filter (`--volatile`, `--min-atr`) to surface high-daily-range tickers.
  - `SCANNER_README_RU.md` — detailed usage docs (Russian; a translated summary is above).

## The honest numbers (short version)

- Single-component precision (CCI, BB, RSI-cross, Fisher, UO, TD9, Connors RSI) as reversal predictors: typically **45–63%**, never close to 90%.
- Confluence of 4+ of 7 components beats any single component, at the cost of firing less often.
- "TAKE" (top/vertex) signals reliably flag RSI vertex points (up to ~70% hit rate at conf≥4 + price below EMA200) but are **not a short signal** — average forward 5-day return after a top signal is still positive. Hence "TAKE" (take-profit/exit-long), not "SHORT".
- Candle shape (hammer, long lower wick) alone does **not** predict reversals — the speed of the preceding decline and ATR/volume expansion do ("capitulation": >12% 5-day drop + ATR ratio >1.2x + volume ratio >1.2x is the most robust filter, ~60% precision, nearly identical on train and test).
- Market breadth (share of stocks with RSI<30 in a basket) monotonically improves single-stock pivot reliability — market-wide panic makes individual reversals more reliable.

Full methodology and tables are in `research/*_REPORT_RU.md` (Russian).

## Quick start

**Pine indicators:** open a `.pine` file from `tradingview/`, paste it into the Pine Editor on TradingView → Add to chart.

**Scanner:**
```bash
cd scanner
python bybit_radar_scanner.py --volatile
```
Full flag list and examples in `scanner/SCANNER_README_RU.md`.

**Reproduce the research:**
```bash
cd research
python realtime_pivot_predictor_study.py
```

## License

[MIT](../LICENSE) — use, copy, modify freely, keep the attribution.

## Support the project

See [DONATE.md](../DONATE.md) for the EVM donation address.

---

*Every number here comes from historical backtests, reported honestly, with no inflated claims.*
