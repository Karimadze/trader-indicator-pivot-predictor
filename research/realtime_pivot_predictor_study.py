import os
"""Воспроизводимая проверка Realtime Pivot Predictor.

Задача: на закрытии свечи t предсказать, окажется ли она подтверждённым
PIVOT LOW по RSI (логика Libertus: подтверждение через 2 правые свечи).

Никакого lookahead: все признаки считаются только по данным до закрытия t
включительно. Метка is_pivot использует t+1 и t+2 и служит ТОЛЬКО целью.

Запуск:  python realtime_pivot_predictor_study.py
"""
import pandas as pd, numpy as np

DATA = os.path.join(os.path.dirname(__file__), "divergence_data")
TICKERS = ["MRVL", "COHR", "MU", "LITE", "GLW", "TER", "STX"]
LOOKBACK = 45
CONFIRM = 2
B0 = -0.45
W = {"f_rsi": -0.75, "f_stretch": 0.19, "f_dn": 0.15, "f_d200": 0.21,
     "f_atr": 0.12, "f_body": -0.07, "f_ret3": 0.12}


def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up, dn = rma(d.clip(lower=0), n), rma((-d).clip(lower=0), n)
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def build(t, N=LOOKBACK):
    df = pd.read_csv(f"{DATA}/{t}_1d.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df["ticker"] = t
    c, h, l, o, v = df.close, df.high, df.low, df.open, df.volume
    r = rsi(c, 14)
    df["rsi"] = r
    df["is_new_low"] = r <= r.rolling(N, min_periods=N).min()
    fut = r.shift(-1).rolling(CONFIRM, min_periods=CONFIRM).min().shift(-(CONFIRM - 1))
    df["is_pivot"] = df.is_new_low & (fut > r)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, 14)
    rng = (h - l).replace(0, np.nan)
    ema200 = c.ewm(span=200, adjust=False).mean()
    df["f_rsi"] = (r - 35) / 15
    df["f_stretch"] = ((c - l.rolling(N).min()) / atr - 1.0) / 1.5
    df["f_dn"] = ((c.diff() < 0).rolling(3).sum() - 1.5) / 1.0
    df["f_d200"] = (c / ema200 - 1) / 0.20
    df["f_atr"] = (atr / c - 0.025) / 0.015
    df["f_body"] = ((c - o) / rng) / 0.4
    df["f_ret3"] = (c / c.shift(3) - 1) / 0.08
    lin = B0 + sum(W[f] * df[f].clip(-3, 3) for f in W)
    df["score"] = 100 / (1 + np.exp(-lin))
    df["next_open"], df["next_close"] = o.shift(-1), c.shift(-1)
    return df


d = pd.concat([build(t) for t in TICKERS], ignore_index=True)
d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["is_pivot", "score"])
c = d[d.is_new_low].copy()
c["year"] = c.Date.dt.year
print(f"кандидатов={len(c)}  базовая частота PIVOT={c.is_pivot.mean()*100:.1f}%\n")
for th in [40, 45, 48, 50, 52, 55]:
    tr = c[(c.Date < "2021-01-01") & (c.score >= th)]
    te = c[(c.Date >= "2021-01-01") & (c.score >= th)]
    if len(te) < 15:
        continue
    print(f"score>={th}: TRAIN n={len(tr):4d} prec={tr.is_pivot.mean()*100:5.1f}% | "
          f"TEST n={len(te):4d} prec={te.is_pivot.mean()*100:5.1f}%")
print("\nпо годам, score>=48")
g = c[c.score >= 48].groupby("year").agg(n=("is_pivot", "size"), prec=("is_pivot", "mean"))
print(g.assign(prec=(g.prec * 100).round(1)).to_string())
print("\nдоходность следующего дня (open t+1 -> close t+1), расходы 0.15%")
for nm, s in [("все кандидаты", c), ("score>=48", c[c.score >= 48]), ("score>=52", c[c.score >= 52])]:
    rr = ((s.next_close / s.next_open - 1) * 100 - 0.15).dropna()
    print(f"{nm:15s} n={len(rr):4d} средн={rr.mean():+.2f}% медиана={rr.median():+.2f}% положит={(rr>0).mean()*100:.1f}%")
c.to_csv("realtime_pivot_predictor_events.csv", index=False)
