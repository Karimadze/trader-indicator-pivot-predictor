import os
import pandas as pd, numpy as np, glob, os, pickle
D = os.path.join(os.path.dirname(__file__), "vmc_ema_extended_data")
def rma(s,n): return s.ewm(alpha=1/n,adjust=False).mean()
def rsi(c,n=14):
    d=c.diff(); up=rma(d.clip(lower=0),n); dn=rma((-d).clip(lower=0),n)
    return (100-100/(1+up/dn.replace(0,np.nan))).fillna(50)
N=45; CONF=2
files=sorted(glob.glob(f"{D}/*_1d.csv"))
frames={}
for f in files:
    t=os.path.basename(f).replace("_1d.csv","")
    if t=="QQQ": continue
    df=pd.read_csv(f,parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    if len(df)<400: continue
    c,h,l,o,v=df.close,df.high,df.low,df.open,df.volume
    r=rsi(c,14); df["rsi"]=r; df["ticker"]=t
    nl=r<=r.rolling(N,min_periods=N).min()
    fut=r.shift(-1).rolling(CONF,min_periods=CONF).min().shift(-(CONF-1))
    df["cand"]=nl; df["is_pivot"]=nl&(fut>r)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=rma(tr,14)
    rngv=(h-l).replace(0,np.nan)
    df["atr_r"]=atr/atr.rolling(50).mean()
    df["atr_pct"]=atr/c
    df["volr"]=v/v.rolling(20).mean()
    df["ret5"]=c.pct_change(5)*100
    df["ret"]=c.pct_change()*100
    df["dd20"]=(c/c.rolling(20).max()-1)*100
    df["stretch"]=(c-l.rolling(N).min())/atr
    df["dn3"]=(c.diff()<0).rolling(3).sum()
    df["dist200"]=(c/c.ewm(span=200,adjust=False).mean()-1)*100
    df["body"]=(c-o)/rngv
    df["below200"]=(c<c.ewm(span=200,adjust=False).mean()).astype(float)
    df["rsi_lt30"]=(r<30).astype(float)
    df["nl_flag"]=nl.astype(float)
    df["fwd5"]=(c.shift(-5)/c-1)*100
    frames[t]=df
print("тикеров:",len(frames))

# ---- breadth: агрегаты по всем тикерам на каждую дату
def agg(col):
    s=pd.concat([f.set_index("Date")[col] for f in frames.values()],axis=1)
    return s
B=pd.DataFrame()
B["b_rsi30"]=agg("rsi_lt30").mean(axis=1)*100
B["b_newlow"]=agg("nl_flag").mean(axis=1)*100
B["b_below200"]=agg("below200").mean(axis=1)*100
B["b_medret5"]=agg("ret5").median(axis=1)
B["b_medrsi"]=agg("rsi").median(axis=1)
B["n_avail"]=agg("nl_flag").notna().sum(axis=1)
B=B[B.n_avail>=30].sort_index()
B["b_rsi30_z"]=(B.b_rsi30-B.b_rsi30.rolling(250).mean())/B.b_rsi30.rolling(250).std()
B["b_newlow_z"]=(B.b_newlow-B.b_newlow.rolling(250).mean())/B.b_newlow.rolling(250).std()
B["b_rsi30_max20"]=B.b_rsi30.rolling(20).max()
B["b_newlow_5d"]=B.b_newlow.rolling(5).mean()
B=B.reset_index()
B.to_pickle("./breadth.pkl")
print(B.tail(3).to_string())
print("\nдиапазон b_rsi30:",B.b_rsi30.describe([.5,.9,.99]).round(1).to_dict())

big=pd.concat(frames.values(),ignore_index=True).merge(B,on="Date",how="inner")
big.to_pickle("./big.pkl")
c=big[big.cand].dropna(subset=["is_pivot"])
print(f"\nвсего кандидатов по {len(frames)} тикерам: {len(c)}  PIVOT={c.is_pivot.mean()*100:.1f}%")
pd.set_option("display.width",220)
big=pd.read_pickle("./big.pkl")
c=big[big.cand].dropna(subset=["is_pivot"]).copy()
HALAL=["MRVL","COHR","MU","LITE","GLW","TER","STX"]
print(f"кандидатов={len(c)}  база PIVOT={c.is_pivot.mean()*100:.1f}%\n")

print("="*90); print("BREADTH ПООДИНОЧКЕ (все 97 тикеров, вся история)"); print("="*90)
for col,bins in [("b_rsi30",[0,1,3,7,15,30,100]),
                 ("b_newlow",[0,2,5,10,20,100]),
                 ("b_newlow_z",[-5,-0.5,0.5,1.5,3,10]),
                 ("b_medret5",[-50,-6,-3,-1,1,50]),
                 ("b_below200",[0,20,40,60,80,101]),
                 ("b_medrsi",[0,35,45,55,100])]:
    c["_b"]=pd.cut(c[col],bins)
    g=c.groupby("_b",observed=True).agg(n=("is_pivot","size"),piv=("is_pivot","mean"),fwd5=("fwd5","mean"))
    g["PIVOT %"]=(g.piv*100).round(1); g["+5д %"]=g.fwd5.round(2)
    print(f"\n{col}:"); print(g[["n","PIVOT %","+5д %"]].to_string())

print("\n"+"="*90); print("КОМБО: капитуляция тикера + паника рынка"); print("="*90)
cap=(c.ret5<-12)&(c.atr_r>1.2)&(c.volr>1.2)
tests=[("только капитуляция тикера",cap),
       ("капитуляция + b_rsi30>7%",cap&(c.b_rsi30>7)),
       ("капитуляция + b_rsi30>15%",cap&(c.b_rsi30>15)),
       ("капитуляция + b_newlow_z>1.5",cap&(c.b_newlow_z>1.5)),
       ("капитуляция + b_medret5<-3%",cap&(c.b_medret5<-3)),
       ("капитуляция + СПОКОЙНЫЙ рынок b_rsi30<3%",cap&(c.b_rsi30<3)),
       ("--",None),
       ("b_rsi30>15% (без фильтра тикера)",c.b_rsi30>15),
       ("b_rsi30>30%",c.b_rsi30>30),
       ("b_newlow_z>2",c.b_newlow_z>2),
      ]
out=[]
for nm,m in tests:
    if m is None: out.append((nm,None,None,None,None,None)); continue
    s=c[m]
    if len(s)<50: continue
    tr=s[s.Date<"2021-01-01"]; te=s[s.Date>="2021-01-01"]
    out.append((nm,len(s),s.is_pivot.mean()*100,tr.is_pivot.mean()*100,te.is_pivot.mean()*100,s.fwd5.mean()))
print(pd.DataFrame(out,columns=["фильтр","n","PIVOT %","train %","test %","+5д %"]).to_string(index=False,float_format=lambda x:f"{x:.1f}"))
import pandas as pd, numpy as np, glob, os
pd.set_option("display.width",200)
D = os.path.join(os.path.dirname(__file__), "vmc_ema_extended_data")
# прокси, который есть в TradingView: доля акций выше своей MA20 (аналог INDEX:NDTW)
frames={}
for f in sorted(glob.glob(f"{D}/*_1d.csv")):
    t=os.path.basename(f).replace("_1d.csv","")
    if t=="QQQ": continue
    df=pd.read_csv(f,parse_dates=["Date"]).sort_values("Date")
    if len(df)<400: continue
    df["above20"]=(df.close>df.close.rolling(20).mean()).astype(float)
    frames[t]=df.set_index("Date")["above20"]
A=pd.concat(frames,axis=1)
ndtw=(A.mean(axis=1)*100).rename("pct_above20").to_frame().reset_index()
ndtw["n"]=A.notna().sum(axis=1).values
ndtw=ndtw[ndtw.n>=30]
print("прокси NDTW (% акций выше MA20):"); print(ndtw.pct_above20.describe([.1,.25,.5,.75,.9]).round(1).to_string())

big=pd.read_pickle("./big.pkl").merge(ndtw[["Date","pct_above20"]],on="Date",how="left")
c=big[big.cand].dropna(subset=["is_pivot","pct_above20"]).copy()
print(f"\nкорреляция с b_rsi30: {c.pct_above20.corr(c.b_rsi30):.2f}")
print("\nточность PIVOT по уровню прокси:")
c["_b"]=pd.cut(c.pct_above20,[0,15,25,40,60,80,101])
g=c.groupby("_b",observed=True).agg(n=("is_pivot","size"),piv=("is_pivot","mean"),fwd5=("fwd5","mean"))
print(g.assign(**{"PIVOT %":(g.piv*100).round(1),"+5д %":g.fwd5.round(2)})[["n","PIVOT %","+5д %"]].to_string())

cap=(c.ret5<-12)&(c.atr_r>1.2)&(c.volr>1.2)
print("\nкомбо с прокси (то, что реализуемо в Pine одной строкой):")
for nm,m in [("капитуляция",cap),
             ("капитуляция + прокси<40%",cap&(c.pct_above20<40)),
             ("капитуляция + прокси<25%",cap&(c.pct_above20<25)),
             ("капитуляция + прокси<15%",cap&(c.pct_above20<15)),
             ("капитуляция + прокси>60% (спокойно)",cap&(c.pct_above20>60))]:
    s=c[m]
    if len(s)<40: continue
    tr=s[s.Date<"2021-01-01"]; te=s[s.Date>="2021-01-01"]
    print(f"  {nm:38s} n={len(s):5d} PIVOT={s.is_pivot.mean()*100:5.1f}% (train {tr.is_pivot.mean()*100:.1f} / test {te.is_pivot.mean()*100:.1f})  +5д={s.fwd5.mean():+.2f}% полож={(s.fwd5>0).mean()*100:.0f}%")
from sklearn.linear_model import LogisticRegression
pd.set_option("display.width",220)
big=pd.read_pickle("./big.pkl")
c=big[big.cand].dropna(subset=["is_pivot"]).copy()
HALAL=["MRVL","COHR","MU","LITE","GLW","TER","STX"]

def prep(x):
    x=x.copy()
    x["f_rsi"]=(x.rsi-35)/15
    x["f_stretch"]=(x.stretch-1.0)/1.5
    x["f_dn"]=(x.dn3-1.5)/1.0
    x["f_d200"]=x.dist200/20.0
    x["f_atr"]=(x.atr_pct-0.025)/0.015
    x["f_body"]=x.body/0.4
    x["f_ret5"]=x.ret5/8.0
    x["f_atrr"]=(x.atr_r-1.0)/0.3
    x["f_vol"]=(x.volr-1.0)/0.5
    x["f_dd20"]=x.dd20/15.0
    x["b_nl"]=x.b_newlow_z.clip(-3,3)
    x["b_p30"]=(x.b_rsi30-5)/10.0
    x["b_mr5"]=x.b_medret5/4.0
    x["b_mrsi"]=(x.b_medrsi-48)/8.0
    return x
c=prep(c)
BASE=["f_rsi","f_stretch","f_dn","f_d200","f_atr","f_body","f_ret5","f_atrr","f_vol","f_dd20"]
BRD =["b_nl","b_p30","b_mr5","b_mrsi"]
c=c.replace([np.inf,-np.inf],np.nan).dropna(subset=BASE+BRD+["is_pivot"])

tr=c[c.Date<"2021-01-01"]
te7=c[(c.Date>="2021-01-01")&(c.ticker.isin(HALAL))]
teAll=c[c.Date>="2021-01-01"]
print(f"train n={len(tr)} | test 7 исходных n={len(te7)} база={te7.is_pivot.mean()*100:.1f}% | test все n={len(teAll)} база={teAll.is_pivot.mean()*100:.1f}%\n")

def run(F,label):
    m=LogisticRegression(max_iter=4000,C=0.3).fit(tr[F].clip(-3,3),tr.is_pivot)
    print(f"--- {label} ---")
    print("  b0=%.3f"%m.intercept_[0],dict(zip(F,m.coef_[0].round(3))))
    res={}
    for nm,s in [("TRAIN",tr),("TEST 7 исходных",te7),("TEST все 97",teAll)]:
        p=1/(1+np.exp(-(m.intercept_[0]+(s[F].clip(-3,3).values*m.coef_[0]).sum(1))))
        s=s.assign(p=p); res[nm]=s
    for th in [0.50,0.55,0.60,0.65]:
        line=f"  score>={th:.2f}: "
        for nm in ["TRAIN","TEST 7 исходных","TEST все 97"]:
            s=res[nm]; sel=s[s.p>=th]
            line+=f"{nm}: n={len(sel):5d} prec={sel.is_pivot.mean()*100:5.1f}%  " if len(sel)>25 else f"{nm}: мало  "
        print(line)
    return res,m

r1,m1=run(BASE,"БЕЗ breadth (10 признаков)")
print()
r2,m2=run(BASE+BRD,"С breadth (14 признаков)")

print("\n=== доходность через 5 дней и просадка, модель с breadth, тест 7 исходных ===")
s=r2["TEST 7 исходных"]
for th in [0.50,0.55,0.60]:
    sel=s[s.p>=th]
    if len(sel)<20: continue
    print(f"  score>={th}: n={len(sel):4d} PIVOT={sel.is_pivot.mean()*100:.1f}%  +5д={sel.fwd5.mean():+.2f}%  положит={(sel.fwd5>0).mean()*100:.1f}%")
print(f"  без фильтра: n={len(s)} PIVOT={s.is_pivot.mean()*100:.1f}%  +5д={s.fwd5.mean():+.2f}%  положит={(s.fwd5>0).mean()*100:.1f}%")
import pickle; pickle.dump((m2,BASE+BRD),open("./m2.pkl","wb"))
