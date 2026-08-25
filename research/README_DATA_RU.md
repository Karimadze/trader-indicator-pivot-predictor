# О данных в этой папке

В репозиторий включены только данные, необходимые для воспроизведения ключевых отчётов:
- `divergence_data/` — дневные OHLCV для 7 исходных тикеров (MRVL, COHR, MU, LITE, GLW, TER, STX) + QQQ, использованные в `realtime_pivot_predictor_study.py`, `pivot_anatomy_study.py`, `reversal_radar_study.py`.
- `bybit_volatility_snapshot.csv`, `bybit_volatility_confirmed.csv` — снапшот волатильности инструментов Bybit из исследования сканера.

Более крупный набор данных (для широты рынка на ~90+ тикерах, `research/BREADTH_REPORT_RU.md`) не включён в репозиторий, чтобы не раздувать его сырыми OHLCV-дампами — его можно перегенерировать теми же скриптами через ваш источник данных (yfinance/аналог).
