import os
import pandas as pd, numpy as np
DATA=os.path.join(os.path.dirname(__file__), "divergence_data")
TICKERS=["MRVL","COHR","MU","LITE","GLW","TER","STX"]
N=45; CONFIRM=2

def rma(s,n): return s.ewm(alpha=1/n,adjust=False).mean()
def rsi(c,n=14):
    d=c.diff(); up=rma(d.clip(lower=0),n); dn=rma((-d).clip(lower=0),n)
    return (100-100/(1+up/dn.replace(0,np.nan))).fillna(50)

def load(t):
    df=pd.read_csv(f"{DATA}/{t}_1d.csv",parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df["ticker"]=t; return df

def target(df):
    r=rsi(df.close,14); df["rsi"]=r
    newlow = r<=r.rolling(N,min_periods=N).min()
    fut=r.shift(-1).rolling(CONFIRM,min_periods=CONFIRM).min().shift(-(CONFIRM-1))
    piv = newlow & (fut>r)
    df["is_pivot"]=piv
    # окно допуска +-1 бар
    df["pivot_zone"]= piv | piv.shift(1).fillna(False) | piv.shift(-1).fillna(False)
    return df

def indicators(df):
    c,h,l,o,v=df.close,df.high,df.low,df.open,df.volume
    hlc3=(h+l+c)/3
    S={}
    # 1. WaveTrend (LazyBear) cross up в зоне перепроданности
    esa=hlc3.ewm(span=10,adjust=False).mean()
    de=(hlc3-esa).abs().ewm(span=10,adjust=False).mean()
    ci=(hlc3-esa)/(0.015*de.replace(0,np.nan))
    wt1=ci.ewm(span=21,adjust=False).mean(); wt2=wt1.rolling(4).mean()
    S["WaveTrend cross up <-60"]=(wt1>wt2)&(wt1.shift()<=wt2.shift())&(wt2<-60)
    S["WaveTrend cross up <-53"]=(wt1>wt2)&(wt1.shift()<=wt2.shift())&(wt2<-53)
    # 2. TD Sequential buy setup 9
    cnt=np.zeros(len(c)); cond=(c<c.shift(4)).values
    for i in range(len(c)):
        cnt[i]= cnt[i-1]+1 if i>0 and cond[i] and cnt[i-1]>0 else (1 if cond[i] else 0)
    S["TD Sequential buy 9"]=pd.Series(cnt==9,index=c.index)
    S["TD Sequential buy>=8"]=pd.Series(cnt>=8,index=c.index)
    # 3. Stoch RSI cross up из <20
    r=rsi(c,14); mn=r.rolling(14).min(); mx=r.rolling(14).max()
    sr=(r-mn)/(mx-mn).replace(0,np.nan)*100
    k=sr.rolling(3).mean(); d3=k.rolling(3).mean()
    S["StochRSI cross up <20"]=(k>d3)&(k.shift()<=d3.shift())&(d3<20)
    # 4. CCI cross up из -100
    tp=hlc3; ma=tp.rolling(20).mean(); md=(tp-ma).abs().rolling(20).mean()
    cci=(tp-ma)/(0.015*md.replace(0,np.nan))
    S["CCI cross up -100"]=(cci>-100)&(cci.shift()<=-100)
    S["CCI < -200"]=(cci<-200)&(cci.shift()>=-200)
    # 5. Williams %R cross up -80
    hh=h.rolling(14).max(); ll=l.rolling(14).min()
    wr=-100*(hh-c)/(hh-ll).replace(0,np.nan)
    S["Williams %R cross up -80"]=(wr>-80)&(wr.shift()<=-80)
    # 6. Fisher Transform cross
    mid=(h+l)/2; mn2=mid.rolling(9).min(); mx2=mid.rolling(9).max()
    val=(2*((mid-mn2)/(mx2-mn2).replace(0,np.nan))-1).clip(-0.999,0.999)
    vsm=val.ewm(alpha=0.33,adjust=False).mean().clip(-0.999,0.999)
    fish=0.5*np.log((1+vsm)/(1-vsm)); fish=fish.ewm(alpha=0.5,adjust=False).mean()
    S["Fisher cross up (<-1.5)"]=(fish>fish.shift())&(fish.shift()<=fish.shift(2))&(fish<-1.5)
    # 7. Connors RSI < 10 (упрощ.)
    streak=np.zeros(len(c)); ch=(c.diff()>0).values; cd=(c.diff()<0).values
    for i in range(1,len(c)):
        streak[i]= streak[i-1]+1 if ch[i] and streak[i-1]>0 else (streak[i-1]-1 if cd[i] and streak[i-1]<0 else (1 if ch[i] else (-1 if cd[i] else 0)))
    srsi=rsi(pd.Series(streak,index=c.index),2)
    pr=c.pct_change().rolling(100).apply(lambda x: (x[:-1]<x[-1]).mean()*100,raw=True)
    crsi=(rsi(c,3)+srsi+pr)/3
    S["Connors RSI < 10"]=(crsi<10)&(crsi.shift()>=10)
    # 8. Bollinger %B < 0 и возврат
    m20=c.rolling(20).mean(); sd=c.rolling(20).std()
    pb=(c-(m20-2*sd))/(4*sd).replace(0,np.nan)
    S["BB %B: выход и возврат"]=(pb>0)&(pb.shift()<=0)
    S["BB нижняя лента пробита"]=(pb<0)&(pb.shift()>=0)
    # 9. RSI cross up 30
    S["RSI cross up 30"]=(r>30)&(r.shift()<=30)
    S["RSI < 30 впервые"]=(r<30)&(r.shift()>=30)
    # 10. Ultimate Oscillator < 30
    bp=c-pd.concat([l,c.shift()],axis=1).min(axis=1)
    tr=pd.concat([h,c.shift()],axis=1).max(axis=1)-pd.concat([l,c.shift()],axis=1).min(axis=1)
    uo=100*(4*bp.rolling(7).sum()/tr.rolling(7).sum()+2*bp.rolling(14).sum()/tr.rolling(14).sum()+bp.rolling(28).sum()/tr.rolling(28).sum())/7
    S["Ultimate Osc cross up 30"]=(uo>30)&(uo.shift()<=30)
    # 11. QQE-подобный: RSI сглаженный пересекает свой trail
    rs=r.ewm(span=5,adjust=False).mean()
    atrr=(rs.diff().abs()).ewm(alpha=1/27,adjust=False).mean()*4.238
    S["QQE-like cross up"]=(rs>rs.shift()+0)&(rs.shift()<rs.shift(2))&(rs<35)
    # 12. Свечной разворот: молот/поглощение на нисх. тренде
    rng=(h-l).replace(0,np.nan)
    hammer=((np.minimum(o,c)-l)/rng>0.5)&((c-o)/rng>-0.1)&(c<c.rolling(20).mean())
    S["Молот под MA20"]=hammer
    eng=(c>o.shift())&(o<c.shift())&(c.shift()<o.shift())&(c<c.rolling(20).mean())
    S["Бычье поглощение"]=eng
    # 13. 3 подряд падения + разворотный день
    S["3 падения + рост"]=( (c.diff()<0).rolling(3).sum().shift()==3 )&(c>o)
    # 14. Keltner/Squeeze: цена ниже нижней Keltner
    ema20=c.ewm(span=20,adjust=False).mean(); atr=rma(pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1),20)
    S["Ниже нижней Keltner"]=(c<ema20-2*atr)&(c.shift()>=ema20.shift()-2*atr.shift())
    return S

rows=[]
frames={}
for t in TICKERS:
    df=target(load(t)); frames[t]=(df,indicators(df))
allpiv=[]; alln=0
for t,(df,S) in frames.items():
    d=df.dropna(subset=["is_pivot"])
    allpiv.append(d)
base=pd.concat(allpiv)
print(f"баров: {len(base)}   доля баров в PIVOT-зоне (+-1): {base.pivot_zone.mean()*100:.1f}%   чистых PIVOT: {base.is_pivot.mean()*100:.1f}%\n")
names=list(frames[TICKERS[0]][1].keys())
out=[]
for nm in names:
    hits=[];tot=0;hit=0;hit_strict=0
    for t,(df,S) in frames.items():
        m=S[nm].fillna(False)&df.is_pivot.notna()
        m=m&(df.index>250)
        tot+=m.sum(); hit+=(m&df.pivot_zone).sum(); hit_strict+=(m&df.is_pivot).sum()
    if tot<40: continue
    out.append((nm,tot,hit/tot*100,hit_strict/tot*100, tot/ (len(base)/7) *252/7))
o=pd.DataFrame(out,columns=["Индикатор","сигналов","попал в зону ±1 бар %","точно в PIVOT %","сигн/год/тикер"]).sort_values("попал в зону ±1 бар %",ascending=False)
o["lift"]=(o["попал в зону ±1 бар %"]/(base.pivot_zone.mean()*100)).round(2)
pd.set_option("display.width",200)
print(o.to_string(index=False,float_format=lambda x:f"{x:.1f}"))

print("\n\n=== TOP-6: устойчивость train(<2021)/test(>=2021) + что было после ===")
top=["CCI < -200","BB %B: выход и возврат","RSI < 30 впервые","Fisher cross up (<-1.5)","Ultimate Osc cross up 30","TD Sequential buy 9","BB нижняя лента пробита","Connors RSI < 10"]
res=[]
for nm in top:
    parts=[]
    for t,(df,S) in frames.items():
        d=df.copy(); d["sig"]=S[nm].fillna(False); d=d[d.index>250]
        d["fwd5"]=(d.close.shift(-5)/d.close-1)*100
        d["fwd10"]=(d.close.shift(-10)/d.close-1)*100
        d["mfe5"]=(d.high.shift(-1).rolling(5).max().shift(-4)/d.close-1)*100
        parts.append(d[d.sig])
    s=pd.concat(parts)
    tr=s[s.Date<"2021-01-01"]; te=s[s.Date>="2021-01-01"]
    res.append((nm,len(tr),tr.pivot_zone.mean()*100,len(te),te.pivot_zone.mean()*100,
                s.fwd5.mean(),s.fwd10.mean(),(s.fwd5>0).mean()*100,s.mfe5.mean()))
r=pd.DataFrame(res,columns=["Индикатор","n_train","train зона%","n_test","test зона%","+5д %","+10д %","+5д полож.%","макс за 5д %"])
print(r.to_string(index=False,float_format=lambda x:f"{x:.1f}"))

print("\n=== для сравнения: случайный бар ===")
b=[]
for t,(df,S) in frames.items():
    d=df[df.index>250].copy()
    d["fwd5"]=(d.close.shift(-5)/d.close-1)*100; d["fwd10"]=(d.close.shift(-10)/d.close-1)*100
    d["mfe5"]=(d.high.shift(-1).rolling(5).max().shift(-4)/d.close-1)*100
    b.append(d)
b=pd.concat(b)
print(f"любой бар: +5д={b.fwd5.mean():.2f}% +10д={b.fwd10.mean():.2f}% полож={(b.fwd5>0).mean()*100:.1f}% макс за 5д={b.mfe5.mean():.2f}%")
