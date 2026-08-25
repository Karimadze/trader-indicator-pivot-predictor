exec(open('reversal_radar_study.py').read().split("rows=[]")[0])
frames={t:(lambda d:(d,indicators(d)))(target(load(t))) for t in TICKERS}
CORE=["CCI < -200","BB нижняя лента пробита","RSI < 30 впервые","Fisher cross up (<-1.5)","Ultimate Osc cross up 30","TD Sequential buy 9","Connors RSI < 10"]
parts=[]
for t,(df,S) in frames.items():
    d=df.copy()
    cnt=None
    for nm in CORE:
        s=S[nm].fillna(False).astype(int)
        s=s.rolling(3,min_periods=1).max()   # сигнал «свежий» в окне 3 баров
        cnt=s if cnt is None else cnt+s
    d["conf"]=cnt
    d["fwd5"]=(d.close.shift(-5)/d.close-1)*100
    d=d[d.index>250]
    parts.append(d)
a=pd.concat(parts).dropna(subset=["is_pivot"])
print(f"база: PIVOT-зона ±1 = {a.pivot_zone.mean()*100:.1f}% баров; +5д средн={a.fwd5.mean():.2f}%, полож={(a.fwd5>0).mean()*100:.1f}%\n")
print("confluence = сколько из 7 сигналов активны в окне 3 баров (только первый бар каждого кластера)")
a["new"]=(a.conf>a.conf.shift()).fillna(False)
for k in range(1,7):
    s=a[(a.conf>=k)&a["new"]]
    if len(s)<30: continue
    tr=s[s.Date<"2021-01-01"]; te=s[s.Date>="2021-01-01"]
    print(f" conf>={k}: n={len(s):5d}  зона±1={s.pivot_zone.mean()*100:5.1f}% (train {tr.pivot_zone.mean()*100:.1f} / test {te.pivot_zone.mean()*100:.1f})  +5д={s.fwd5.mean():+.2f}%  полож={(s.fwd5>0).mean()*100:.1f}%  ~{len(s)/16.5/7:.1f}/год/тикер")
