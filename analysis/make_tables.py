import pandas as pd, numpy as np, json
from scipy import stats
D={"2012":pd.read_csv('../results/physionet2012/results_all_physionet2012.csv'),"2019":pd.read_csv('../results/physionet2019/results_all_physionet2019.csv')}
ours="CLiP-SSM (ours)"
order=["HMM","MedGAN-lite","RGAN","RecurrentVAE","TimeGAN","TimeVAE","TimeDiff-lite","DiffusionTS-lite","DG-HMM (repro)",ours]
disp={"HMM":"HMM","MedGAN-lite":"MedGAN","RGAN":"RGAN","RecurrentVAE":"Recurrent VAE","TimeGAN":"TimeGAN","TimeVAE":"TimeVAE","TimeDiff-lite":"TimeDiff-style","DiffusionTS-lite":"Diffusion-TS-style","DG-HMM (repro)":"DG-HMM",ours:"CLiP-SSM"}
cols=[("KS",1,3),("CorrPres",0,3),("ACFcurve_err",1,3),("RuleGap",1,1),("Util_fore%",0,1),("Util_auc%",0,1),("MIA_cal_dev",1,3)]
def fmt(m,s,dec): return f"{m:.{dec}f}$\\pm${s:.{dec}f}"
def table_main():
    out=[]
    for ds in ["2012","2019"]:
        r=D[ds]; mean=r.groupby("model").mean(numeric_only=True); std=r.groupby("model").std(numeric_only=True)
        best={k:(mean.loc[order,k].idxmin() if low else mean.loc[order,k].idxmax()) for k,low,_ in cols}
        out.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{PhysioNet {ds}}}}} \\\\")
        for m in order:
            cells=[]
            for k,low,dec in cols:
                c=fmt(mean.loc[m,k],std.loc[m,k],dec)
                cells.append(f"\\textbf{{{c}}}" if best[k]==m else c)
            out.append(disp[m]+" & "+" & ".join(cells)+" \\\\")
        out.append("\\midrule" if ds=="2012" else "")
    return "\n".join(out)
def table_abl():
    r=D["2012"]; ab=["CLiP-SSM (ours)","CLiP -rule","CLiP -priv","CLiP -mmd","CLiP -dyn","CLiP -marg","CLiP -cor","CLiP -cw","CLiP -hold"]
    names={"CLiP-SSM (ours)":"Full model","CLiP -rule":"Without rule penalty","CLiP -priv":"Without leakage margin","CLiP -mmd":"Without MMD term","CLiP -dyn":"Without dynamics loss","CLiP -marg":"Without marginal loss","CLiP -cor":"Without correlation loss","CLiP -cw":"Without cross-window loss","CLiP -hold":"Without hold gate"}
    full=r[(r.model==ours)&(r.seed.isin([0,1]))]; rows=[]
    for m in ab:
        d=full if m==ours else r[r.model==m]; mean=d.mean(numeric_only=True)
        rows.append(names[m]+" & "+" & ".join(f"{mean[k]:.{dec}f}" for k,low,dec in cols)+" \\\\")
    return "\n".join(rows)
open("tab_main_rows.tex","w").write(table_main()); open("tab_abl_rows.tex","w").write(table_abl())
# statistics vs best other, Holm within dataset
stat={}
for ds in ["2012","2019"]:
    r=D[ds]; mean=r.groupby("model").mean(numeric_only=True); base=[m for m in order if m!=ours]; rows=[]
    for k,low,_ in cols:
        s=mean.loc[base,k].sort_values(ascending=bool(low)); bm=s.index[0]
        a=r[r.model==ours][k].values; b=r[r.model==bm][k].values; p=stats.ttest_ind(a,b,equal_var=False).pvalue
        rank=list(mean.loc[order,k].sort_values(ascending=bool(low)).index).index(ours)+1
        rows.append([k,bm,mean.loc[ours,k],mean.loc[bm,k],rank,p])
    pv=np.array([x[5] for x in rows]); o=np.argsort(pv); h=np.empty_like(pv); run=0
    for i,ix in enumerate(o): run=max(run,(len(pv)-i)*pv[ix]); h[ix]=min(1,run)
    stat[ds]=[(x[0],disp[x[1]],round(x[2],4),round(x[3],4),x[4],round(x[5],4),round(h[i],4)) for i,x in enumerate(rows)]
for ds in stat:
    print(ds); [print("  ",x) for x in stat[ds]]
# Table 4 additional measures
def add_table():
    out=[]
    spec=[("DTW",1,3,"mean"),("SkewErr",1,2,"median"),("KurtErr",1,1,"median"),("Bounds%_raw",0,1,"mean"),("MIA_worstgrp_AUC",1,3,"mean")]
    import numpy as np
    for ds,(d,t) in {"2012":("../results/physionet2012","physionet2012"),"2019":("../results/physionet2019","physionet2019")}.items():
        r=D[ds]; g=r.groupby("model")
        for k,low,dec,how in spec:
            v=(g[k].mean() if how=="mean" else g[k].median()).loc[order]; base=v.drop(ours)
            bm=base.idxmin() if low else base.idxmax()
            out.append((ds,k,v[ours],disp[bm],base[bm],list(v.sort_values(ascending=bool(low)).index).index(ours)+1,dec))
        # hold-rate error from saved samples
        R=np.load(f"{d}/real_test_{t}.npz")["te"]; S=np.load(f"{d}/synthetic_{t}.npz")
        hold=lambda U:(np.abs(np.diff(np.clip(U,0,1),axis=1))<1e-6).mean((0,1))
        keys={"HMM":"HMM","MedGAN-lite":"MedGAN-lite","RGAN":"RGAN","RecurrentVAE":"RecurrentVAE","TimeGAN":"TimeGAN","TimeVAE":"TimeVAE","TimeDiff-lite":"TimeDiff-lite","DiffusionTS-lite":"DiffusionTS-lite","DG-HMM (repro)":"DG-HMM_(repro)",ours:"CLiP-SSM_(ours)"}
        err={m:np.abs(hold(S[keys[m]])-hold(R)).mean() for m in order}; base={m:v for m,v in err.items() if m!=ours}; bm=min(base,key=base.get)
        out.append((ds,"HoldRateErr",err[ours],disp[bm],base[bm],sorted(err.values()).index(err[ours])+1,3))
    names={"DTW":"DTW distance","SkewErr":"Skewness error (median)","KurtErr":"Kurtosis error (median)","Bounds%_raw":"Raw bounds adherence (\\%)","MIA_worstgrp_AUC":"Worst-subgroup attack AUC","HoldRateErr":"Hold-rate error"}
    rows=[]
    for k in names:
        for ds in ["2012","2019"]:
            x=[o for o in out if o[0]==ds and o[1]==k][0]; dec=x[6]
            rows.append(f"PhysioNet {ds} & {names[k]} & {x[2]:.{dec}f} & {x[3]} {x[4]:.{dec}f} & {x[5]} \\\\")
    return "\n".join(rows)
open("tab_add_rows.tex","w").write(add_table()); print(open("tab_add_rows.tex").read())
print(open("tab_abl_rows.tex").read())
