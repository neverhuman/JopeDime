"""Fit competitor methods on train, score on test. Usage: run_methods.py DATASET_DIR METHOD[,..] OUT_JSONL [task]"""
import sys,json,time,os,numpy as np,pandas as pd
os.environ.setdefault("OMP_NUM_THREADS","8")
def load_pair(d,tr,te,task):
    h=json.load(open(f"{d}/meta.json")).get("has_header",False) if os.path.exists(f"{d}/meta.json") else False
    a=pd.read_csv(f"{d}/{tr}",header=0 if h else None,low_memory=False); b=pd.read_csv(f"{d}/{te}",header=0 if h else None,low_memory=False)
    b.columns=a.columns; n=len(a); df=pd.concat([a,b],ignore_index=True)
    X=pd.DataFrame(index=df.index)
    for c in df.columns[:-1]:
        v=pd.to_numeric(df[c],errors="coerce")
        if v.notna().mean()<0.9*df[c].notna().mean(): v=pd.Series(pd.factorize(df[c].astype(str))[0],dtype=float)
        X[c]=v
    X=X.to_numpy(np.float32); t=df.iloc[:,-1]
    if task=="regression": y=pd.to_numeric(t,errors="coerce").to_numpy(np.float64); ok=np.isfinite(y)
    else:
        ok=t.notna().to_numpy(); cl=sorted(t[:n][ok[:n]].astype(str).unique()); y=(t.astype(str)==cl[-1]).to_numpy(int)
    i=np.arange(len(df)); tr_m=(i<n)&ok; te_m=(i>=n)&ok
    return X[tr_m],y[tr_m],X[te_m],y[te_m]
def make(name,task,seed=0):
    reg=task=="regression"; T=8
    if name=="random_forest":
        from sklearn.ensemble import RandomForestRegressor as R,RandomForestClassifier as C
        return (R if reg else C)(n_estimators=300,n_jobs=T,random_state=seed)
    if name=="xgboost":
        import xgboost as x; k=dict(n_estimators=500,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,n_jobs=T,random_state=seed,tree_method="hist")
        return x.XGBRegressor(**k) if reg else x.XGBClassifier(**k)
    if name=="lightgbm":
        import lightgbm as l; k=dict(n_estimators=500,learning_rate=0.05,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,n_jobs=T,random_state=seed,verbose=-1)
        return l.LGBMRegressor(**k) if reg else l.LGBMClassifier(**k)
    if name=="catboost":
        import catboost as c; k=dict(iterations=1000,learning_rate=0.05,depth=6,thread_count=T,random_seed=seed,verbose=0)
        return c.CatBoostRegressor(**k) if reg else c.CatBoostClassifier(**k)
    if name=="tabpfn":
        import tabpfn; dev=os.environ.get("TABPFN_DEVICE","cpu")
        if reg: return tabpfn.TabPFNRegressor(random_state=seed,device=dev,model_path=os.environ.get("TABPFN_REG_CKPT","auto"))
        return tabpfn.TabPFNClassifier(random_state=seed,device=dev,model_path=os.environ.get("TABPFN_CLF_CKPT","auto"))
    if name=="ridge":
        from sklearn.pipeline import make_pipeline; from sklearn.impute import SimpleImputer; from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV,LogisticRegression
        return make_pipeline(SimpleImputer(),StandardScaler(),RidgeCV() if reg else LogisticRegression(max_iter=2000))
    raise ValueError(name)
def score(task,y,p):
    from sklearn.metrics import roc_auc_score
    if task=="regression": return "rmse",float(np.sqrt(np.mean((y-p)**2)))
    return "auc",float(roc_auc_score(y,p))
def run(d,methods,out,task="regression",tr="train.csv",te="test.csv"):
    Xtr,ytr,Xte,yte=load_pair(d,tr,te,task)
    if task=="binary" and len(np.unique(yte))<2: raise SystemExit("single-class test")
    for mname in methods:
        rec=dict(dataset=os.path.basename(d.rstrip("/")),method=mname,task=task,n_train=len(ytr),n_test=len(yte))
        t=time.time()
        try:
            if mname=="autogluon":
                from autogluon.tabular import TabularPredictor
                tr_df=pd.DataFrame(Xtr);tr_df["y"]=ytr; te_df=pd.DataFrame(Xte)
                pr=TabularPredictor("y",problem_type="regression" if task=="regression" else "binary",eval_metric="rmse" if task=="regression" else "roc_auc",path=f"/tmp/ag_{os.getpid()}",verbosity=0).fit(tr_df,time_limit=300,num_cpus=8,num_gpus=0)
                p=pr.predict(te_df).to_numpy() if task=="regression" else pr.predict_proba(te_df).iloc[:,-1].to_numpy()
            else:
                m=make(mname,task); Xa,Xb=Xtr,Xte
                if mname=="tabpfn" and len(ytr)>10000:
                    idx=np.random.default_rng(0).choice(len(ytr),10000,replace=False); m.fit(Xa[idx],ytr[idx])
                else: m.fit(Xa,ytr)
                p=m.predict(Xb) if task=="regression" else m.predict_proba(Xb)[:,1]
            rec["metric"],rec["value"]=score(task,yte,np.asarray(p,dtype=np.float64)); rec["status"]="ok"
        except Exception as e: rec["status"]="error"; rec["error"]=repr(e)[:300]
        rec["seconds"]=round(time.time()-t,2)
        with open(out,"a") as f: f.write(json.dumps(rec)+"\n")
        print(json.dumps(rec),flush=True)
if __name__=="__main__":
    run(sys.argv[1],sys.argv[2].split(","),sys.argv[3],*(sys.argv[4:5] or ["regression"]))
