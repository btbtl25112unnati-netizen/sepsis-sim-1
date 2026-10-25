import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
H=72
def gen(n,prev,seed,noise=1.0,drop=0.08):
    r=np.random.default_rng(seed); out=[]
    for i in range(n):
        sep=r.random()<prev; t0=int(r.integers(30,60)) if sep else -1
        b=dict(HR=r.normal(80,8),SpO2=r.normal(97,1),Temp=r.normal(36.8,0.3),Resp=r.normal(16,2))
        s=r.lognormal(0,0.4)*(0.7 if r.random()<0.2 else 1)  # 20% weak responders
        t=np.arange(H); ramp=np.clip((t-(t0-12))/12,0,1) if sep else np.zeros(H)
        circ=np.sin(2*np.pi*t/24)
        d=dict(HR=b['HR']+3*circ+ramp*22*s,SpO2=b['SpO2']-ramp*3*s,Temp=b['Temp']+0.15*circ+ramp*1.3*s,Resp=b['Resp']+ramp*8*s)
        sd=dict(HR=3,SpO2=1.2,Temp=0.25,Resp=1.5)  # sensor noise (+ physiologic wander)
        df=pd.DataFrame({k:d[k]+r.normal(0,sd[k]*noise,H)+np.cumsum(r.normal(0,0.15*sd[k],H)) for k in d})
        m=np.zeros(H,bool)
        for _ in range(int(drop*H/3)+1): a=r.integers(0,H); m[a:a+r.integers(1,6)]=True   # motion-artifact bursts
        df[m]=np.nan; df['pid']=i; df['t']=t; df['t0']=t0; df['sep']=sep; out.append(df)
    return pd.concat(out,ignore_index=True)
def feats(df,sensors):
    res=[]
    for pid,g in df.groupby('pid'):
        g=g.copy(); cols=[]
        cov=g[sensors[0]].notna().astype(float).rolling(6,min_periods=1).mean(); g['cov']=cov
        for s in sensors:
            x=g[s].ffill(limit=3); base=x.iloc[:24].median()
            g[s+'_c']=x; g[s+'_m6']=x.rolling(6,min_periods=3).mean(); g[s+'_sl']=(x-x.shift(5))/5
            g[s+'_sd12']=x.rolling(12,min_periods=6).std(); g[s+'_d']=g[s+'_m6']-base
        res.append(g)
    g=pd.concat(res); F=[c for c in g if any(c.endswith(k) for k in ('_c','_m6','_sl','_sd12','_d'))]
    g['y']=((g.sep)&(g.t>=g.t0-6)&(g.t<=g.t0)).astype(int); g=g[(g.t>=24)&~((g.sep)&(g.t>g.t0))]
    return g,F
def models(seed): return {'LogReg':make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,class_weight='balanced')),
 'SVM-RBF':make_pipeline(StandardScaler(),SVC(probability=False,class_weight='balanced')),
 'RandForest':RandomForestClassifier(80,min_samples_leaf=20,class_weight='balanced',random_state=seed,n_jobs=1),
 'GradBoost':HistGradientBoostingClassifier(max_depth=4,max_iter=150,random_state=seed)}
def score(m,name,X):
    if name=='SVM-RBF': return m.decision_function(X)
    return m.predict_proba(X)[:,1]
def alerts(g,s,thr,refr=6):
    g=g.assign(s=s); out={}
    for pid,q in g.groupby('pid'):
        q=q[(q.s>=thr)&(q['cov']>=0.5)]; ts=[]
        for t in q.t:
            if not ts or t-ts[-1]>=refr: ts.append(t)
        out[pid]=(ts,q.t0.iloc[0] if len(q) else None)
    return out
def evalpt(g,s,thr):
    A=alerts(g,s,thr); pat=g.groupby('pid').agg(sep=('sep','first'),t0=('t0','first'),n=('t','size'))
    ns=pat[~pat.sep]; fa=sum(len(A[p][0]) if p in A else 0 for p in ns.index)/(ns.n.sum()/24)
    det=[];lead=[]
    for p,row in pat[pat.sep].iterrows():
        ts=[t for t in (A[p][0] if p in A else []) if row.t0-24<=t<=row.t0]
        det.append(bool(ts)); 
        if ts: lead.append(row.t0-ts[0])
    return np.mean(det),fa,(np.median(lead) if lead else np.nan)
def thr_for(g,s,budget):
    lo,hi=np.percentile(s,50),np.percentile(s,100)
    for _ in range(18):
        mid=(lo+hi)/2; _,fa,_=evalpt(g,s,mid)
        lo,hi=(mid,hi) if fa>budget else (lo,mid)
    return hi
def run(seed,sensors,budget=1.0,test_kw=None,keep=False):
    tr,F=feats(gen(350,0.2,seed),sensors); va,_=feats(gen(120,0.2,seed+100),sensors)
    te,_=feats(gen(150,0.2,seed+200,**(test_kw or {})),sensors); res={}; sc={}
    ftr=tr.copy(); Xtr=ftr[F].fillna(ftr[F].median()); med=ftr[F].median()
    for n,m in models(seed).items():
        idx=np.random.default_rng(seed).choice(len(Xtr),min(len(Xtr),6000),replace=False) if n=='SVM-RBF' else slice(None)
        m.fit(Xtr.iloc[idx] if n=='SVM-RBF' else Xtr,ftr.y.iloc[idx] if n=='SVM-RBF' else ftr.y)
        sv=score(m,n,va[F].fillna(med)); st=score(m,n,te[F].fillna(med)); th=thr_for(va,sv,budget); sens,fa,ld=evalpt(te,st,th)
        res[n]=dict(AUROC=roc_auc_score(te.y,st),sens=sens,alerts_day=fa,lead_h=ld); sc[n]=(st,th)
    # HR>100 rule baseline
    r=(te['HR_m6'].fillna(0)>100).astype(float).values; sens,fa,ld=evalpt(te,r,0.5); res['HR>100 rule']=dict(AUROC=roc_auc_score(te.y,r),sens=sens,alerts_day=fa,lead_h=ld)
    return (res,te,sc,F) if keep else res
if __name__=='__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    S_A=['HR','SpO2','Temp']; S_B=S_A+['Resp']; agg={}
    for tag,S in (('A',S_A),('B',S_B)):
        runs=[run(s,S) for s in (0,1)]
        agg[tag]={m:{k:[float(np.nanmean([r[m][k] for r in runs])),float(np.nanstd([r[m][k] for r in runs]))] for k in runs[0][m]} for m in runs[0]}
        print(tag,json.dumps({m:{k:round(v[0],2) for k,v in d.items()} for m,d in agg[tag].items()}),flush=True)
    # robustness (train at noise 1x / 8% dropout; test harsher) RF on A
    rob={}
    for lab,kw in {'baseline':{},'noise x2':{'noise':2},'noise x3':{'noise':3},'dropout 20%':{'drop':.2},'dropout 35%':{'drop':.35}}.items():
        r=run(0,S_A,test_kw=kw); rob[lab]={'RandForest':[r['RandForest']['sens'],r['RandForest']['alerts_day']],'LogReg':[r['LogReg']['sens'],r['LogReg']['alerts_day']]}; print(lab,rob[lab],flush=True)
    # leakage demo: row-level split vs patient-level
    d,F=feats(gen(350,0.2,5),S_A); Xa=d[F].fillna(d[F].median()); rng=np.random.default_rng(0); msk=rng.random(len(d))<0.75
    rf=RandomForestClassifier(80,min_samples_leaf=3,random_state=0,n_jobs=1).fit(Xa[msk],d.y[msk]); leak=roc_auc_score(d.y[~msk],rf.predict_proba(Xa[~msk])[:,1])
    pm=d.pid.isin(np.arange(260)); rf2=RandomForestClassifier(80,min_samples_leaf=3,random_state=0,n_jobs=1).fit(Xa[pm],d.y[pm]); honest=roc_auc_score(d.y[~pm],rf2.predict_proba(Xa[~pm])[:,1]); print('leak',leak,'honest',honest)
    # example patient timeline
    res,te,sc,F=run(0,S_A,keep=True); st,th=sc['RandForest']; te=te.assign(s=st); sp=te[te.sep].groupby('pid').t0.first().index[2]; q=te[te.pid==sp]
    json.dump(dict(agg=agg,rob=rob,leak=[leak,honest]),open('results.json','w'),indent=1)
    fig,ax=plt.subplots(1,3,figsize=(13,3.6)); 
    ms=['HR>100 rule','LogReg','SVM-RBF','RandForest','GradBoost']; w=.38; x=np.arange(len(ms))
    ax[0].bar(x-w/2,[agg['A'][m]['sens'][0] for m in ms],w,color='#0284C7',label='HR+SpO2+Temp'); ax[0].bar(x+w/2,[agg['B'][m]['sens'][0] if m in agg['B'] else 0 for m in ms],w,color='#16A34A',label='+ Resp')
    ax[0].set_xticks(x); ax[0].set_xticklabels(ms,rotation=25,ha='right',fontsize=8); ax[0].set_ylabel('Patient-level sensitivity'); ax[0].set_title('Sensitivity at ~1 alert/patient-day',fontsize=9); ax[0].legend(fontsize=7); ax[0].set_ylim(0,1)
    labs=list(rob); ax[1].bar(range(5),[rob[l]['RandForest'][0] for l in labs],color=['#0284C7']+['#F59E0B']*4); ax[1].set_xticks(range(5)); ax[1].set_xticklabels(labs,rotation=25,ha='right',fontsize=8); ax[1].set_title('Robustness: RF trained on clean data',fontsize=9); ax[1].set_ylim(0,1)
    ax[2].plot(q.t,q.s,color='#1E3A8A'); ax[2].axhline(th,ls='--',color='#DC2626',lw=1); ax[2].axvline(q.t0.iloc[0],color='k',ls=':',lw=1); ax[2].set_title('Example septic patient: risk score vs time',fontsize=9); ax[2].set_xlabel('hour'); ax[2].text(q.t0.iloc[0]+.5,th*0.9,'onset',fontsize=7); ax[2].text(25,th*1.02,'alert threshold',fontsize=7,color='#DC2626')
    plt.tight_layout(); plt.savefig('sim_fig.png',dpi=170)
