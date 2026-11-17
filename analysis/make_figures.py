import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,"axes.spines.top":False,"axes.spines.right":False})
P={"2012":("../results/physionet2012","physionet2012"),"2019":("../results/physionet2019","physionet2019")}
FE=["hr","sbp","dbp","rr","spo2","temp","glucose","lactate","vent"]; LO=np.array([30,50,20,4,50,34,40,.2,0.]); HI=np.array([220,250,150,60,100,42,600,20,1.])
LAB={"hr":"Heart rate (beats/min)","sbp":"Systolic BP (mmHg)","dbp":"Diastolic BP (mmHg)","rr":"Respiratory rate (breaths/min)","spo2":"Oxygen saturation (%)","temp":"Temperature (°C)","glucose":"Glucose (mg/dL)","lactate":"Lactate (mmol/L)","vent":"Binary variable"}
def load(ds):
    d,t=P[ds]; R=np.load(f"{d}/real_test_{t}.npz")["te"]; S=np.load(f"{d}/synthetic_{t}.npz"); return R,S
def units(U,d): return LO[d]+np.clip(U[...,d],0,1)*(HI[d]-LO[d])
COL={"Real":"#111111","TimeDiff-lite":"#1f77b4","CLiP-SSM_(ours)":"#d62728"}
# ---------- Fig 2 trajectories
fig,ax=plt.subplots(3,4,figsize=(7.2,5.0),sharex=True)
cols=[("2012","hr"),("2012","spo2"),("2019","hr"),("2019","lactate")]
rows=[("Real",None),("TimeDiff-lite","TimeDiff-lite"),("CLiP-SSM (ours)","CLiP-SSM_(ours)")]
rng=np.random.default_rng(3)
for j,(ds,v) in enumerate(cols):
    R,S=load(ds); d=FE.index(v)
    for i,(nm,key) in enumerate(rows):
        U=R if key is None else S[key]; X=units(U,d); idx=rng.choice(len(X),6,replace=False)
        a=ax[i,j]; a.plot(X[idx].T,lw=0.8,alpha=.8); a.plot(X.mean(0),color="k",lw=1.6)
        if i==0: a.set_title(f"PhysioNet {ds}: {LAB[v].split(' (')[0]}")
        if j==0: a.set_ylabel(nm)
        if i==2: a.set_xlabel("Hour")
fig.tight_layout(); fig.savefig("fig_traj.png",dpi=600,facecolor="white"); plt.close(fig)
# ---------- Fig 3 ACF curves
def acf(U,l):
    x=U-U.mean(1,keepdims=True); num=(x[:,:-l]*x[:,l:]).sum(1); den=(x**2).sum(1)
    with np.errstate(all="ignore"): return np.nanmean(num/den,axis=0)
ms=[("RecurrentVAE","RecurrentVAE","#2ca02c"),("TimeGAN","TimeGAN","#9467bd"),("TimeDiff-lite","TimeDiff-lite","#1f77b4"),("DiffusionTS-lite","DiffusionTS-lite","#8c564b"),("DG-HMM","DG-HMM_(repro)","#7f7f7f"),("CLiP-SSM","CLiP-SSM_(ours)","#d62728")]
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8),sharey=False); lags=range(1,25)
for a,ds in zip(ax,["2012","2019"]):
    R,S=load(ds); a.plot(lags,[np.nanmean(acf(R,l)) for l in lags],color="k",lw=2.2,label="Real")
    for lab,key,c in ms: a.plot(lags,[np.nanmean(acf(np.clip(S[key],0,1),l)) for l in lags],color=c,lw=1.3,label=lab)
    a.set_title(f"PhysioNet {ds}"); a.set_xlabel("Lag (hours)"); a.set_ylabel("Mean autocorrelation")
ax[0].legend(frameon=False,ncol=1,loc="upper right"); fig.tight_layout(); fig.savefig("fig_acf.png",dpi=600,facecolor="white"); plt.close(fig)
# ---------- Fig 4 per-variable KS
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8),sharey=True); ks=[f"KS_{v}" for v in FE]
sel=[("TimeGAN","#9467bd"),("TimeDiff-lite","#1f77b4"),("DiffusionTS-lite","#8c564b"),("CLiP-SSM (ours)","#d62728")]
for a,ds in zip(ax,["2012","2019"]):
    d,t=P[ds]; r=pd.read_csv(f"{d}/results_all_{t}.csv"); m=r.groupby("model").mean(numeric_only=True); w=0.2
    for i,(nm,c) in enumerate(sel): a.bar(np.arange(9)+i*w,m.loc[nm,ks].values,w,color=c,label=nm.replace(" (ours)",""))
    a.set_xticks(np.arange(9)+0.3); a.set_xticklabels(["HR","SBP","DBP","RR","SpO2","Temp","Gluc.","Lact.","Binary"],rotation=0); a.set_title(f"PhysioNet {ds}"); a.set_xlabel("Variable")
ax[0].set_ylabel("KS statistic (lower is better)"); h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",ncol=4,frameon=False,fontsize=7); fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig("fig_ks.png",dpi=600,facecolor="white"); plt.close(fig)
from PIL import Image
for n in ["fig_traj","fig_acf","fig_ks"]: print(n,Image.open(n+".png").size,Image.open(n+".png").info.get("dpi"))
