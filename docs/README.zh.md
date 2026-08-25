# Trader Indicator Pivot Predictor

**🌐 语言:** [Русский](../README.md) · [English](README.en.md) · **中文**（当前）· [हिन्दी](README.hi.md) · [Deutsch](README.de.md) · [Español](README.es.md)

一套用于 TradingView 的 Pine Script 指标，外加一个独立的 Bybit Python 扫描器，目的是**实时预测 RSI 反转点**——在K线收盘之前就给出信号,而不是像 TradingView 自带的 "RSI Div - Lib" 指标那样,PIVOT 标签要延迟 2 根K线才画出来。

本项目基于诚实、可验证的历史数据研究,**没有"90%准确率"这种夸大宣传**。所有数字都是在按时间切分的训练/测试集上测得的(训练集截止 2021 年,测试集为之后的数据),以避免过拟合。

## ⚠️ 免责声明

本项目不构成投资建议,也不保证盈利。这些指标和扫描器只是基于历史价格统计的研究工具。使用任何信号(包括本项目提供的信号)进行交易都存在亏损风险。作者不是持牌金融顾问。请自行核实后再做决策。

## 目录结构

- **`tradingview/`** — TradingView Pine Script v6 指标:
  - `RealtimePivotPredictor.pine` — 基于 7 个特征(RSI、偏离均线幅度、相对 EMA200 位置、ATR、K线实体、3根K线收益率)的逻辑回归反转概率评分(0–100),不会重绘(`barstate.isconfirmed`,无前视偏差)。
  - `ReversalRadar.pine` / `ReversalRadarV2.pine` — "Reversal Radar":由 7 个组件(CCI、布林带、RSI 交叉、Fisher 变换、终极振荡指标、TD Sequential 9、Connors RSI)组成的共振指标,底部(LONG)和顶部(TAKE,并非做空信号,原因见下文)信号对称,RSI 面板与 RSI Div-Lib 一致,并可选市场广度过滤器。
  - `CausalPivotCandidate.pine`、`CyclicSmoothedRSI_MTF.pine`、`NasdaqVMCSeparatePane.pine`、`BybitStockPerpRegimeBreakout.pine` — 同一研究过程中的辅助/实验性指标。

- **`research/`** — 通往结果的完整路径:用于测算一切数据的 Python 脚本,以及包含具体数字的 Markdown 报告(目前为俄文,欢迎协助翻译),还有可复现的脚本和原始 7 只股票的 CSV 数据及 Bybit 波动率快照。

- **`scanner/`** — 独立的 Python 扫描器(不依赖 TradingView),扫描 Bybit 上**全部在售**的 TradFi 永续合约和 xStocks(1小时/4小时周期),包括尚在形成中(未收盘)的K线:
  - `bybit_radar_scanner.py` — 仅使用 Python 标准库,兼容 Python 3.8+。
  - 波动率过滤器(`--volatile`、`--min-atr`),用于筛选日内波动幅度较大的标的。

## 真实数据(简要版)

- 单一组件(CCI、布林带、RSI交叉、Fisher、终极振荡指标、TD9、Connors RSI)作为反转预测因子的准确率通常在 **45%–63%** 之间,从未接近 90%。
- 7 个组件中有 4 个以上共振时,准确率高于任何单一组件,但触发频率更低。
- "TAKE"(顶部/拐点)信号能可靠地标记 RSI 拐点(在 conf≥4 且价格低于 EMA200 时命中率高达约 70%),但**并非做空信号**——顶部信号出现后未来 5 天的平均收益仍为正。因此标注为 "TAKE"(止盈/平多),而非 "SHORT"。
- K线形态(锤子线、长下影线)本身**不能**预测反转——真正起作用的是此前下跌的速度以及 ATR/成交量的放大("恐慌抛售"过滤器:5日跌幅 >12% 且 ATR 比率 >1.2 倍且成交量比率 >1.2 倍,是最稳健的过滤器,准确率约 60%,训练集与测试集结果几乎一致)。
- 市场广度(一篮子股票中 RSI<30 的占比)会持续提升个股反转信号的可靠性——全市场恐慌时,个股反转信号更可靠。

完整方法论和数据表见 `research/*_REPORT_RU.md`(俄文)。

## 快速开始

**Pine 指标:** 从 `tradingview/` 目录打开所需的 `.pine` 文件,复制代码到 TradingView 的 Pine 编辑器 → Add to chart。

**扫描器:**
```bash
cd scanner
python bybit_radar_scanner.py --volatile
```
完整参数列表和示例见 `scanner/SCANNER_README_RU.md`(俄文)。

## 许可证

[MIT](../LICENSE) — 可自由使用、复制、修改,请保留版权声明。

## 支持本项目

捐赠地址见 [DONATE.md](../DONATE.md)。

---

*所有数据均来自历史回测,如实报告,不含夸大宣传。*
