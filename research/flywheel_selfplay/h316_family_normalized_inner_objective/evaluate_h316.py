"""Raw utility evaluation and fixed paired whole-family H316 decision."""
import hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
from run_h316 import durable,PARTS,SEEDS
ELO_LABEL='reference = r882 slot0 (DL-Train outer)'
ARCHIVED={
 'P0':('H303','3CUeOo9dmsTM1rZyuAeS9pF7H39Vv80N','0f8435010390c5bff8d6c95a7650163a2ab4c5952e56264fbee96204784d5ea4','ENS_VOTE2_NESTED'),
 'P1':('H308','uXsQkA27JSci7XVbEFvbyJvfvzmO3nP0','ac16ccdfd715d000a249dfea87d707159329f189a43475452af038a03517290f','ENS_VOTE2_NESTED_P1'),
 'P2':('H308','uXsQkA27JSci7XVbEFvbyJvfvzmO3nP0','ac16ccdfd715d000a249dfea87d707159329f189a43475452af038a03517290f','ENS_VOTE2_NESTED_P2')}
# Exact recorded S3 version, not an inferred latest-object read.
PUBLISHED={'P0':0.0643163334285782,'P1':0.04821480030677507,'P2':0.04360142642936269}


def family_d(u,families):
    f=np.asarray(families);names=sorted(set(families))
    return np.stack([np.array([u[s,f==x,1].mean()-u[s,f==x,0].mean() for x in names]) for s in range(3)])


def fetch(part,data):
    wave,version,digest,name=ARCHIVED[part]
    dest=data/'archived'/f'{wave}_picks.npz';dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        subprocess.run(['aws','s3api','get-object','--bucket','veox-jopedime-learning-use2','--key',
                        f'experiments/rd_breakthrough_20261030/{wave}/picks.npz','--version-id',version,str(dest)],
                       check=True,stdout=subprocess.DEVNULL,env=dict(os.environ,AWS_PROFILE='veox-jepson-prod'))
    assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
    with np.load(dest) as z:return np.asarray(z[name],dtype=np.int32)


def regress(r303,base,data):
    import evaluate_h303 as ev
    G=r303.G;families=[r['family_id'] for r in G['train_rows']];reg={}
    for part in PARTS:
        want=fetch(part,data);got=base[part]
        folds={f:__import__('run_h316').outer_fold(part,f,G['a302']) for f in set(families)}
        per_fold={}
        for k in range(4):
            ix=[i for i,f in enumerate(families) if folds[f]==k]
            for s,seed in enumerate(SEEDS):
                a=np.asarray(got[s,ix],dtype=np.int32);b=np.asarray(want[s,ix],dtype=np.int32)
                per_fold[f'{k}:{seed}']=dict(equal=bool(np.array_equal(a,b)),got=hashlib.sha256(a.tobytes()).hexdigest(),want=hashlib.sha256(b.tobytes()).hexdigest())
        u,_=ev.panel(G,got);gains=family_d(u,families).mean(axis=1)
        median=float(np.median(gains))
        reg[part]=dict(per_fold=per_fold,per_seed_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in got],
                       per_seed_equal=[bool(np.array_equal(a,b)) for a,b in zip(got,want)],
                       per_seed_gain=gains.tolist(),median=median,published_median=PUBLISHED[part],median_exact=median==PUBLISHED[part])
    ok=all(x['median_exact'] and all(x['per_seed_equal']) and all(f['equal'] for f in x['per_fold'].values()) for x in reg.values())
    durable(data/'regression.json',dict(status='PASS' if ok else 'REGRESSION_FAIL',partitions=reg))
    print(json.dumps(dict(regression='PASS' if ok else 'REGRESSION_FAIL',median={p:x['median'] for p,x in reg.items()})),flush=True)
    if not ok:raise RuntimeError('REGRESSION_FAIL: stop before NORM')


def drop_top(d,n=10):
    return float(np.sort(d)[::-1][n:].mean())


def paired_drop_top(norm,base,seed=316160,draws=10000,n=10):
    assert norm.shape==base.shape and norm.ndim==2
    idx=np.random.default_rng(seed).integers(0,norm.shape[1],size=(draws,norm.shape[1]))
    obs=np.array([drop_top(m,n)-drop_top(b,n) for m,b in zip(norm,base)])
    diffs=np.empty((draws,len(norm)))
    for c in range(len(norm)):
        m=np.sort(norm[c][idx],axis=1)[:,:-n];b=np.sort(base[c][idx],axis=1)[:,:-n]
        diffs[:,c]=m.mean(axis=1)-b.mean(axis=1)
    med=np.median(diffs,axis=1)
    return dict(per_cell=obs.tolist(),median=float(np.median(obs)),ci95=np.quantile(med,[.025,.975]).tolist(),lcb95=float(np.quantile(med,.05)),rng=seed,draws=draws)


def decide(paired,gain_norm,gain_base,harm_norm,harm_base):
    gates=dict(paired_drop_top10_positive=bool(paired['median']>0 and paired['lcb95']>0),
               gain_retention=bool(np.median(gain_norm)>=.9*np.median(gain_base)),
               harmed_nonincrease=bool(np.median(harm_norm)<=np.median(harm_base)))
    return gates,'NORM_ADVANCE' if all(gates.values()) else 'NORM_NOT_ADVANCED'


def evaluate(r303,base,norm,data):
    import evaluate_h303 as ev
    G=r303.G;families=[r['family_id'] for r in G['train_rows']]
    Ds={'BASE':[],'NORM':[]};cells=[]
    for part in PARTS:
        for arm,pp in [('BASE',base),('NORM',norm)]:
            u,inv=ev.panel(G,pp[part]);D=family_d(u,families)
            stats=G['a302'].summarize_three_seeds(u,inv,families,pp[part])
            Ds[arm].extend(D)
            for s,seed in enumerate(SEEDS):
                d=D[s]
                cells.append(dict(partition=part,root_seed=seed,arm=arm,
                                  family_mean_gain=float(d.mean()),family_median_gain=float(np.median(d)),
                                  gain_ci95=stats['per_seed_ci95'][s],
                                  drop_top={str(n):drop_top(d,n) for n in (1,5,10)},
                                  family_wtl=[int((d>1e-9).sum()),int((np.abs(d)<=1e-9).sum()),int((d< -1e-9).sum())],
                                  harmed_share=float(np.mean(d<-.01)),invalid=int(inv[s,:,1].sum()),slot0_invalid=int(inv[s,:,0].sum()),
                                  switch_rate=float(np.mean(pp[part][s]!=0)),elo=stats['elo_by_seed'][s],
                                  picks_sha256=hashlib.sha256(pp[part][s].tobytes()).hexdigest()))
    Dn,Db=(np.asarray(Ds[a]) for a in ('NORM','BASE'))
    paired=paired_drop_top(Dn,Db)
    gains={a:[x['family_mean_gain'] for x in cells if x['arm']==a] for a in Ds}
    harms={a:[x['harmed_share'] for x in cells if x['arm']==a] for a in Ds}
    criteria,verdict=decide(paired,gains['NORM'],gains['BASE'],harms['NORM'],harms['BASE'])
    # Same whole-family draws preserve dependence across partitions and seeds.
    ix=np.random.default_rng(316160).integers(0,Dn.shape[1],(10000,Dn.shape[1]))
    agg={}
    for arm,D in [('BASE',Db),('NORM',Dn)]:
        draws=np.median(D[:,ix].mean(axis=2),axis=0)
        agg[arm]=dict(median_gain=float(np.median(gains[arm])),gain_ci95=np.quantile(draws,[.025,.975]).tolist(),
                      median_harmed=float(np.median(harms[arm])),median_elo=float(np.median([c['elo'] for c in cells if c['arm']==arm])))
    locks={}
    for p in PARTS:
        for arm in ('BASE','NORM'):locks[f'{arm}_{p}']=json.loads((data/f'{arm}_{p}_locks.json').read_text())
    full=dict(wave='H316',status='COMPLETE',verdict=verdict,criteria=criteria,elo_label=ELO_LABEL,r3175_train='NOT_MEASURED',
              regression=json.loads((data/'regression.json').read_text()),paired_drop_top10=paired,aggregate=agg,cells=cells,fit_lock_sha256=locks,
              loader=__import__('run_h316').R['loader_counters'],normalization_scope='inner-TRAIN in-sample; preregistered',
              outer_test_accesses_before_lock=0,outer_packets_prebuilt=True)
    durable(data/'metrics.json',full)
    np.savez_compressed(data/'picks.npz',**{f'{a}_{p}':d[p] for a,d in [('BASE',base),('NORM',norm)] for p in PARTS})
    np.savez_compressed(data/'raw_family_gains.npz',BASE=Db,NORM=Dn)
    print(json.dumps(dict(verdict=verdict,criteria=criteria,paired=paired,aggregate=agg)),flush=True)
