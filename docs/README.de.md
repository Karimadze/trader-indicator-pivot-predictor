# Trader Indicator Pivot Predictor

**🌐 Sprachen:** [Русский](../README.md) · [English](README.en.md) · [中文](README.zh.md) · [हिन्दी](README.hi.md) · **Deutsch** (aktuell) · [Español](README.es.md)

Eine Sammlung von Pine-Script-Indikatoren (TradingView) sowie ein eigenständiger Python-Scanner für Bybit, die versuchen, **RSI-Pivotpunkte in Echtzeit vorherzusagen** — noch bevor die Kerze schließt, statt wie der Standard-Indikator "RSI Div - Lib", der PIVOT-Marken erst 2 Kerzen im Nachhinein zeichnet.

Dieses Projekt ist das Ergebnis ehrlicher, gemessener Forschung, kein Marketing. Es gibt hier keine "90% Genauigkeit"-Versprechen. Unten stehen die tatsächlichen Zahlen, gemessen an historischen Daten mit einem zeitbasierten Train/Test-Split (Training vor 2021, Test danach), um Overfitting zu vermeiden.

## ⚠️ Haftungsausschluss

Dies ist keine Finanzberatung und keine Gewinngarantie. Die Indikatoren und der Scanner sind Forschungswerkzeuge auf Basis historischer Kursstatistiken. Handel auf Basis irgendwelcher Signale — auch dieser — birgt das Risiko von Kapitalverlust. Der Autor ist kein zugelassener Finanzberater. Bitte alles selbst überprüfen, bevor Entscheidungen getroffen werden.

## Inhalt

- **`tradingview/`** — Pine-Script-v6-Indikatoren für TradingView:
  - `RealtimePivotPredictor.pine` — ein 0–100-Reversal-Wahrscheinlichkeitsscore aus einem logistischen Regressionsmodell mit 7 Merkmalen (RSI, Abstand zum MA, Position relativ zum EMA200, ATR, Kerzenkörper, 3-Kerzen-Rendite). Nicht repaintend (`barstate.isconfirmed`, kein Lookahead).
  - `ReversalRadar.pine` / `ReversalRadarV2.pine` — "Reversal Radar": ein Confluence-Indikator aus 7 Komponenten (CCI, Bollinger Bands, RSI-Kreuzung, Fisher Transform, Ultimate Oscillator, TD Sequential 9, Connors RSI), symmetrische Boden- (LONG) und Top-Signale (TAKE — kein Short-Signal, siehe unten), ein RSI-Panel wie bei RSI Div-Lib, optionaler Marktbreite-Filter.
  - `CausalPivotCandidate.pine`, `CyclicSmoothedRSI_MTF.pine`, `NasdaqVMCSeparatePane.pine`, `BybitStockPerpRegimeBreakout.pine` — unterstützende/experimentelle Indikatoren aus derselben Forschungslinie.

- **`research/`** — der vollständige Weg zu den Ergebnissen: die Python-Skripte, mit denen alles gemessen wurde, sowie Markdown-Berichte mit den Zahlen (derzeit auf Russisch; Übersetzungshilfe willkommen), plus reproduzierbare Skripte und CSV-Daten für den ursprünglichen 7-Ticker-Datensatz sowie einen Bybit-Volatilitäts-Snapshot.

- **`scanner/`** — ein eigenständiger Python-Scanner (keine TradingView-Abhängigkeit), der das **gesamte lebende Universum** der TradFi-Perpetuals und xStocks auf Bybit scannt (1h/4h), einschließlich einer noch nicht geschlossenen Kerze:
  - `bybit_radar_scanner.py` — nur Python-Standardbibliothek, Python 3.8+.
  - Volatilitätsfilter (`--volatile`, `--min-atr`) zum Herausfiltern von Tickern mit hoher Tagesspanne.

## Die ehrlichen Zahlen (Kurzfassung)

- Genauigkeit einzelner Komponenten (CCI, BB, RSI-Cross, Fisher, UO, TD9, Connors RSI) als Reversal-Prädiktoren: typischerweise **45–63%**, nie in der Nähe von 90%.
- Confluence von 4+ der 7 Komponenten schlägt jede Einzelkomponente, löst aber seltener aus.
- "TAKE"-Signale (Top/Vertex) markieren RSI-Vertexpunkte zuverlässig (bis zu ~70% Trefferquote bei conf≥4 + Kurs unter EMA200), sind aber **kein Short-Signal** — die durchschnittliche 5-Tage-Rendite nach einem Top-Signal bleibt positiv. Daher "TAKE" (Gewinnmitnahme/Long-Ausstieg), nicht "SHORT".
- Die Kerzenform (Hammer, lange untere Docht) allein sagt Reversals **nicht** voraus — entscheidend sind Tempo des vorangegangenen Rückgangs sowie ATR-/Volumenexpansion ("Kapitulations"-Filter: >12% 5-Tage-Rückgang + ATR-Ratio >1,2x + Volumen-Ratio >1,2x ist der robusteste Filter, ~60% Genauigkeit, auf Train und Test nahezu identisch).
- Marktbreite (Anteil der Aktien mit RSI<30 in einem Korb) verbessert die Zuverlässigkeit von Einzelaktien-Pivots durchgängig — marktweite Panik macht einzelne Reversals zuverlässiger.

Vollständige Methodik und Tabellen in `research/*_REPORT_RU.md` (Russisch).

## Schnellstart

**Pine-Indikatoren:** `.pine`-Datei aus `tradingview/` öffnen, Code in den Pine Editor bei TradingView einfügen → Add to chart.

**Scanner:**
```bash
cd scanner
python bybit_radar_scanner.py --volatile
```
Vollständige Flag-Liste und Beispiele in `scanner/SCANNER_README_RU.md` (Russisch).

## Lizenz

[MIT](../LICENSE) — frei nutzen, kopieren, verändern, Urheberhinweis beibehalten.

## Projekt unterstützen

EVM-Spendenadresse siehe [DONATE.md](../DONATE.md).

---

*Jede Zahl stammt aus historischen Backtests, ehrlich berichtet, ohne übertriebene Behauptungen.*
