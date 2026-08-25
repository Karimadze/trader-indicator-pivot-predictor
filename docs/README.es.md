# Trader Indicator Pivot Predictor

**🌐 Idiomas:** [Русский](../README.md) · [English](README.en.md) · [中文](README.zh.md) · [हिन्दी](README.hi.md) · [Deutsch](README.de.md) · **Español** (actual)

Un conjunto de indicadores Pine Script (TradingView) más un escáner Python independiente para Bybit que intentan **predecir los puntos de pivote del RSI en tiempo real** — antes de que cierre la vela, en lugar de después como hace el indicador estándar "RSI Div - Lib" (que dibuja las etiquetas PIVOT con 2 velas de retraso).

Este proyecto es el resultado de una investigación honesta y medida, no de marketing. Aquí no hay promesas de "90% de precisión". A continuación, los números reales, medidos sobre datos históricos con una división train/test basada en el tiempo (entrenamiento antes de 2021, prueba después) para evitar el sobreajuste.

## ⚠️ Aviso legal

Esto no es asesoramiento financiero ni garantiza beneficios. Los indicadores y el escáner son herramientas de investigación basadas en estadísticas de precios pasados. Operar con cualquier señal, incluidas estas, conlleva riesgo de pérdida de capital. El autor no es un asesor financiero autorizado. Verifica todo por tu cuenta antes de tomar decisiones.

## Contenido

- **`tradingview/`** — Indicadores Pine Script v6 para TradingView:
  - `RealtimePivotPredictor.pine` — una puntuación de probabilidad de reversión (0–100) basada en un modelo de regresión logística con 7 características (RSI, distancia a la media móvil, posición respecto a la EMA200, ATR, cuerpo de la vela, retorno de 3 velas). No repinta (`barstate.isconfirmed`, sin adelanto de información).
  - `ReversalRadar.pine` / `ReversalRadarV2.pine` — "Reversal Radar": un indicador de confluencia de 7 componentes (CCI, Bandas de Bollinger, cruce de RSI, Transformada de Fisher, Ultimate Oscillator, TD Sequential 9, Connors RSI), señales simétricas de fondo (LONG) y de techo (TAKE — no es señal de venta en corto, ver abajo), un panel de RSI similar al de RSI Div-Lib, y un filtro opcional de amplitud de mercado.
  - `CausalPivotCandidate.pine`, `CyclicSmoothedRSI_MTF.pine`, `NasdaqVMCSeparatePane.pine`, `BybitStockPerpRegimeBreakout.pine` — indicadores auxiliares/experimentales de la misma línea de investigación.

- **`research/`** — el camino completo hacia los resultados: los scripts de Python con los que se midió todo, e informes en Markdown con las cifras (actualmente en ruso; se agradece ayuda con la traducción), además de scripts reproducibles y datos CSV del conjunto original de 7 tickers, y una instantánea de volatilidad de Bybit.

- **`scanner/`** — un escáner de Python independiente (sin dependencia de TradingView) que escanea **todo el universo activo** de perpetuos TradFi y xStocks en Bybit (1h/4h), incluyendo una vela que aún se está formando (sin cerrar):
  - `bybit_radar_scanner.py` — solo librería estándar de Python, 3.8+.
  - Filtro de volatilidad (`--volatile`, `--min-atr`) para detectar tickers con alto rango diario.

## Las cifras honestas (resumen)

- La precisión de los componentes individuales (CCI, BB, cruce de RSI, Fisher, UO, TD9, Connors RSI) como predictores de reversión suele estar entre **45% y 63%**, nunca cerca del 90%.
- La confluencia de 4 o más de los 7 componentes supera a cualquier componente individual, aunque se activa con menos frecuencia.
- Las señales "TAKE" (techo/vértice) detectan de forma fiable los puntos de vértice del RSI (hasta ~70% de acierto con conf≥4 y precio bajo la EMA200), pero **no son una señal de venta en corto** — el retorno promedio a 5 días tras una señal de techo sigue siendo positivo. Por eso se etiquetan "TAKE" (tomar beneficios/salir del largo), no "SHORT".
- La forma de la vela (martillo, mecha inferior larga) por sí sola **no** predice reversiones — lo que importa es la velocidad de la caída previa y la expansión del ATR/volumen (el filtro de "capitulación": caída >12% en 5 días + ratio de ATR >1.2x + ratio de volumen >1.2x es el filtro más robusto, ~60% de precisión, casi idéntico en train y test).
- La amplitud de mercado (proporción de acciones con RSI<30 en una cesta) mejora de forma consistente la fiabilidad de los pivotes individuales — el pánico generalizado hace más fiables las reversiones puntuales.

Metodología completa y tablas en `research/*_REPORT_RU.md` (en ruso).

## Inicio rápido

**Indicadores Pine:** abre el archivo `.pine` que necesites desde `tradingview/`, copia el código en el Pine Editor de TradingView → Add to chart.

**Escáner:**
```bash
cd scanner
python bybit_radar_scanner.py --volatile
```
Lista completa de opciones y ejemplos en `scanner/SCANNER_README_RU.md` (en ruso).

## Licencia

[MIT](../LICENSE) — usa, copia y modifica libremente, manteniendo la atribución.

## Apoya el proyecto

Ver [DONATE.md](../DONATE.md) para la dirección EVM de donación.

---

*Cada cifra proviene de backtests históricos, reportados honestamente, sin afirmaciones exageradas.*
