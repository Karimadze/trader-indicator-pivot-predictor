import os
import pandas as pd, numpy as np
pd.set_option("display.width",250)
DATA=os.path.join(os.path.dirname(__file__), "divergence_data")
TICKERS=["MRVL","COHR","MU","LITE","GLW","TER","STX"]
N=45; CONF=2

def rma(s,n): return s.ewm(alpha=1/n,adjust=False).mean()
def rsi(c,n=14):
    d=c.diff(); up=rma(d.clip(lower=0),n); dn=rma((-d).clip(lower=0),n)
    return (100-100/(1+up/dn.replace(0,np.nan))).fillna(50)

qqq=pd.read_csv(f"{DATA}/QQQ_1d.csv",parse_dates=["Date"]).sort_values("Date")
qqq["q_ret"]=qqq.close.pct_change()*100
qqq["q_ret5"]=qqq.close.pct_change(5)*100
qqq["q_rsi"]=rsi(qqq.close,14)
qqq["q_above200"]=(qqq.close>qqq.close.ewm(span=200,adjust=False).mean()).astype(int)
qqq["q_dd"]=(qqq.close/qqq.close.cummax()-1)*100
Q=qqq[["Date","q_ret","q_ret5","q_rsi","q_above200","q_dd"]]

def prep(t):
    df=pd.read_csv(f"{DATA}/{t}_1d.csv",parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df["ticker"]=t
    c,h,l,o,v=df.close,df.high,df.low,df.open,df.volume
    r=rsi(c,14); df["rsi"]=r
    nl = r<=r.rolling(N,min_periods=N).min()
    fut=r.shift(-1).rolling(CONF,min_periods=CONF).min().shift(-(CONF-1))
    df["cand"]=nl; df["is_pivot"]=nl&(fut>r)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=rma(tr,14); df["atr"]=atr
    rng=(h-l).replace(0,np.nan)
    df["ret"]=c.pct_change()*100
    df["volr"]=v/v.rolling(20).mean()
    df["vol_z"]=(v-v.rolling(50).mean())/v.rolling(50).std()
    df["atr_r"]=atr/atr.rolling(50).mean()
    df["tr_atr"]=tr/atr
    df["close_pos"]=(c-l)/rng
    df["lower_wick"]=(np.minimum(o,c)-l)/rng
    df["upper_wick"]=(h-np.maximum(o,c))/rng
    df["body_abs"]=(c-o).abs()/rng
    df["body"]=(c-o)/rng
    df["gap"]=(o-c.shift())/c.shift()*100
    df["dist200"]=(c/c.ewm(span=200,adjust=False).mean()-1)*100
    df["dist50"]=(c/c.ewm(span=50,adjust=False).mean()-1)*100
    df["dd20"]=(c/c.rolling(20).max()-1)*100
    df["dn3"]=(c.diff()<0).rolling(3).sum()
    df["dn5"]=(c.diff()<0).rolling(5).sum()
    df["ret5"]=c.pct_change(5)*100
    df["ret10"]=c.pct_change(10)*100
    df["rng_atr"]=(h-l)/atr
    df["dow"]=df.Date.dt.dayofweek
    df["month"]=df.Date.dt.month
    df=df.merge(Q,on="Date",how="left")
    df["rel"]=df["ret"]-df["q_ret"]          # относительно рынка за день
    df["rel5"]=df["ret5"]-df["q_ret5"]
    # бары с прошлого подтверждённого pivot
    idx=np.where(df.is_pivot.fillna(False))[0]
    since=np.full(len(df),np.nan); last=-1
    for i in range(len(df)):
        if last>=0: since[i]=i-last
        if i in set(idx.tolist()): last=i
    df["since_pivot"]=since
    return df

d=pd.concat([prep(t) for t in TICKERS],ignore_index=True)
d=d[d.index>0]
d=d.dropna(subset=["is_pivot","cand"])
d=d[d.rsi.notna()]
c=d[d.cand].copy()
print(f"кандидатов={len(c)}  из них PIVOT={c.is_pivot.mean()*100:.1f}%\n")
d.to_pickle("anat.pkl")

print("="*90)
print("1. ЧТО ОТЛИЧАЕТ ИСТИННЫЙ PIVOT ОТ ЛОЖНОГО КАНДИДАТА (только данные до закрытия свечи)")
print("="*90)
F=["rsi","ret","volr","vol_z","atr_r","tr_atr","close_pos","lower_wick","upper_wick","body","gap",
   "dist200","dist50","dd20","dn3","dn5","ret5","ret10","rng_atr","q_ret","q_ret5","q_rsi","q_dd","rel","rel5","since_pivot"]
rows=[]
for f in F:
    a=c[c.is_pivot][f].dropna(); b=c[~c.is_pivot][f].dropna()
    if len(a)<50 or len(b)<50: continue
    pooled=np.sqrt((a.std()**2+b.std()**2)/2)
    dcoh=(a.mean()-b.mean())/pooled if pooled>0 else 0
    # прирост точности в верхнем и нижнем терциле
    x=c[f]; lo=c[x<=x.quantile(.33)].is_pivot.mean()*100; hi=c[x>=x.quantile(.67)].is_pivot.mean()*100
    rows.append((f,a.mean(),b.mean(),dcoh,lo,hi,hi-lo))
r=pd.DataFrame(rows,columns=["признак","PIVOT","ложный","эффект d","нижн.трети %","верх.трети %","разница п.п."])
r=r.reindex(r["разница п.п."].abs().sort_values(ascending=False).index)
print(r.to_string(index=False,float_format=lambda x:f"{x:.2f}"))
c=d[d.cand].copy()

print("="*95)
print("2. СОБЫТИЙНЫЙ ПРОФИЛЬ: что происходит в баре -10 ... +5 относительно PIVOT")
print("   (сравнение: истинный PIVOT против ложного кандидата)")
print("="*95)
d=d.reset_index(drop=True)
res=[]
for off in range(-10,6):
    row={"бар":off}
    for lab,mask in [("PIVOT",d.is_pivot.fillna(False)),("ложный",(d.cand&~d.is_pivot).fillna(False))]:
        idx=np.where(mask)[0]; idx=idx[(idx+off>=0)&(idx+off<len(d))]
        # не пересекать границы тикеров
        ok=d.ticker.values[idx]==d.ticker.values[idx+off]
        idx=idx[ok]
        s=d.iloc[idx+off]
        row[f"ret {lab}"]=s.ret.mean()
        row[f"объём {lab}"]=s.volr.mean()
        row[f"ATRотн {lab}"]=s.atr_r.mean()
        row[f"RSI {lab}"]=s.rsi.mean()
    res.append(row)
p=pd.DataFrame(res).set_index("бар")
print(p.to_string(float_format=lambda x:f"{x:.2f}"))

print("\n"+"="*95)
print("3. КОМБИНАЦИИ ФИЛЬТРОВ (на закрытии свечи-кандидата)")
print("="*95)
def ev(name,mask):
    s=c[mask]
    if len(s)<40: return None
    tr=s[s.Date<"2021-01-01"]; te=s[s.Date>="2021-01-01"]
    return (name,len(s),s.is_pivot.mean()*100,
            tr.is_pivot.mean()*100 if len(tr)>20 else np.nan,
            te.is_pivot.mean()*100 if len(te)>20 else np.nan)
tests=[
 ("падение за 5д < -12%", c.ret5<-12),
 ("падение за 5д < -15%", c.ret5<-15),
 ("падение за 5д < -20%", c.ret5<-20),
 ("просадка от макс.20д < -20%", c.dd20<-20),
 ("просадка от макс.20д < -25%", c.dd20<-25),
 ("день -5% и хуже", c.ret<-5),
 ("день -7% и хуже", c.ret<-7),
 ("объём > 1.5x", c.volr>1.5),
 ("объём > 2x", c.volr>2),
 ("RSI < 25", c.rsi<25),
 ("RSI < 20", c.rsi<20),
 ("рынок QQQ за 5д < -3%", c.q_ret5<-3),
 ("рынок QQQ за 5д < -5%", c.q_ret5<-5),
 ("хуже рынка за 5д на 10%+", c.rel5<-10),
 ("гэп вниз > 3%", c.gap<-3),
 ("4-5 падений из 5", c.dn5>=4),
 ("5 падений из 5", c.dn5>=5),
 ("ATR расширен >1.3x", c.atr_r>1.3),
 ("--- КОМБО ---", None),
 ("падение 5д<-12% И объём>1.3x", (c.ret5<-12)&(c.volr>1.3)),
 ("падение 5д<-12% И RSI<25", (c.ret5<-12)&(c.rsi<25)),
 ("падение 5д<-15% И объём>1.5x", (c.ret5<-15)&(c.volr>1.5)),
 ("dd20<-20% И объём>1.3x", (c.dd20<-20)&(c.volr>1.3)),
 ("dd20<-20% И рынок<-2%за5д", (c.dd20<-20)&(c.q_ret5<-2)),
 ("день<-5% И объём>1.5x", (c.ret<-5)&(c.volr>1.5)),
 ("падение5д<-12% И dn5>=4", (c.ret5<-12)&(c.dn5>=4)),
 ("падение5д<-12% И ATR>1.2x И объём>1.2x", (c.ret5<-12)&(c.atr_r>1.2)&(c.volr>1.2)),
]
out=[]
for nm,m in tests:
    if m is None: out.append((nm,None,None,None,None)); continue
    r=ev(nm,m)
    if r: out.append(r)
t=pd.DataFrame(out,columns=["фильтр","n","PIVOT %","train %","test %"])
print(t.to_string(index=False,float_format=lambda x:f"{x:.1f}"))
print(f"\nбаза без фильтра: n={len(c)}, PIVOT={c.is_pivot.mean()*100:.1f}%")

print("\n"+"="*95)
print("4. КАЛЕНДАРЬ")
print("="*95)
dw={0:"Пн",1:"Вт",2:"Ср",3:"Чт",4:"Пт"}
g=c.groupby("dow").agg(n=("is_pivot","size"),piv=("is_pivot","mean"))
g.index=[dw.get(i,i) for i in g.index]
print("день недели:"); print(g.assign(piv=(g.piv*100).round(1)).to_string())
g2=c.groupby("month").agg(n=("is_pivot","size"),piv=("is_pivot","mean"))
print("\nмесяц:"); print(g2.assign(piv=(g2.piv*100).round(1)).T.to_string())
c=d[d.cand].copy()

print("="*95); print("5. МОНОТОННОСТЬ: глубина падения за 5 дней -> вероятность PIVOT"); print("="*95)
bins=[-100,-25,-20,-15,-12,-9,-6,-3,0,100]
c["b"]=pd.cut(c.ret5,bins)
g=c.groupby("b",observed=True).agg(n=("is_pivot","size"),piv=("is_pivot","mean"))
tr=c[c.Date<"2021-01-01"].groupby("b",observed=True).is_pivot.mean()
te=c[c.Date>="2021-01-01"].groupby("b",observed=True).is_pivot.mean()
g["train %"]=(tr*100).round(1); g["test %"]=(te*100).round(1); g["PIVOT %"]=(g.piv*100).round(1)
print(g[["n","PIVOT %","train %","test %"]].to_string())

print("\n"+"="*95); print("6. ЧТО СТОИТ ОШИБКА: поведение после ложного кандидата"); print("="*95)
res=[]
for lab,m in [("истинный PIVOT",c.is_pivot),("ложный кандидат",~c.is_pivot)]:
    s=c[m]
    idx=s.index.values
    for hor in [1,3,5,10]:
        j=idx+hor; j=j[j<len(d)]
        ok=d.ticker.values[idx[:len(j)]]==d.ticker.values[j]
        fwd=(d.close.values[j[ok]]/d.close.values[idx[:len(j)][ok]]-1)*100
        # минимум внутри горизонта
        mn=[]
        for a,b in zip(idx[:len(j)][ok],j[ok]):
            mn.append((d.low.values[a+1:b+1].min()/d.close.values[a]-1)*100)
        res.append((lab,hor,np.mean(fwd),np.median(fwd),(fwd>0).mean()*100,np.mean(mn)))
r=pd.DataFrame(res,columns=["группа","дней","средн. %","медиана %","положит. %","средняя просадка %"])
print(r.to_string(index=False,float_format=lambda x:f"{x:.2f}"))

print("\n"+"="*95); print("7. МИФЫ: признаки, которые НЕ работают"); print("="*95)
myths=[("длинная нижняя тень (молот) >0.4",c.lower_wick>0.4),
       ("закрытие в верхней трети свечи",c.close_pos>0.66),
       ("бычье тело (close>open)",c.body>0),
       ("гэп вниз >2%",c.gap<-2),
       ("огромный дневной диапазон >2 ATR",c.rng_atr>2),
       ("RSI ниже 20",c.rsi<20),
       ("цена ниже EMA200",c.dist200<0),
       ("цена выше EMA200",c.dist200>0),
       ("объём-всплеск z>2",c.vol_z>2)]
o=[]
for nm,m in myths:
    s=c[m]
    if len(s)<40: continue
    o.append((nm,len(s),s.is_pivot.mean()*100,(s.is_pivot.mean()-c.is_pivot.mean())*100))
print(pd.DataFrame(o,columns=["признак","n","PIVOT %","отклонение от базы п.п."]).to_string(index=False,float_format=lambda x:f"{x:.1f}"))
print(f"база {c.is_pivot.mean()*100:.1f}%")

print("\n"+"="*95); print("8. ЛУЧШИЙ УСТОЙЧИВЫЙ ФИЛЬТР — по годам"); print("="*95)
best=(c.ret5<-12)&(c.atr_r>1.2)&(c.volr>1.2)
s=c[best].copy(); s["year"]=s.Date.dt.year
g=s.groupby("year").agg(n=("is_pivot","size"),piv=("is_pivot","mean"))
print(g.assign(piv=(g.piv*100).round(0)).to_string())
print(f"\nвсего n={len(s)}  PIVOT={s.is_pivot.mean()*100:.1f}%  (~{len(s)/16.5/7:.1f} сигн/год/тикер)")
fwd=[]
for i in s.index.values:
    if i+5<len(d) and d.ticker.values[i]==d.ticker.values[i+5]:
        fwd.append((d.close.values[i+5]/d.close.values[i]-1)*100)
print(f"цена через 5 дней: средн={np.mean(fwd):+.2f}%  положит={np.mean(np.array(fwd)>0)*100:.1f}%  (случайный бар: +0.67%, 54.8%)")
