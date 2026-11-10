"""Frozen BASE regression and raw-utility H378 metrics; no new BASE model scoring."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
from run_h378 import PARTS,SEEDS,durable,outer_fold,R,P
ELO_LABEL='reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN'
PUBLISHED={'P0':.0643163334285782,'P1':.04821480030677507,'P2':.04360142642936269}

def family_d(u,families):
 f=np.asarray(families);names=sorted(set(families))
 return np.stack([np.array([u[s,f==x,1].mean()-u[s,f==x,0].mean() for x in names]) for s in range(3)])

def regress(r303,data):
 import evaluate_h303 as ev
 assert json.loads((P/'base-fetch-receipt.json').read_text())['status']=='EXACT_VERSION_GET_AND_SHA_PASS'
 G=r303.G;families=[r['family_id'] for r in G['train_rows']];old=json.loads((data/'base/regression.json').read_text());base={};report={}
 with np.load(data/'base/h315_picks.npz') as z:
  for part in PARTS:
   got=np.load(data/'base'/f'BASE_{part}.npy');want=np.asarray(z['BASE_'+part],dtype=np.int32)
   assert got.shape==(3,1219);got.setflags(write=False);base[part]=got
   seed_equal=[bool(np.array_equal(a,b)) for a,b in zip(got,want)]
   seed_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in got]
   expected=old['partitions'][part];hash_equal=seed_hashes==expected['per_seed_sha256']
   folds={};lock_hashes=json.loads((data/'base'/f'BASE_{part}_locks.json').read_text())
   for k in range(4):
    ix=[i for i,f in enumerate(families) if outer_fold(part,f,G['a302'])==k]
    for s,seed in enumerate(SEEDS):
     key=f'{k}:{seed}';digest=hashlib.sha256(got[s,ix].astype(np.int32).tobytes()).hexdigest()
     lp=data/'base'/part/f'fold_{k}'/f'root_{seed}'/'fit_lock.json'
     lock_sha=hashlib.sha256(lp.read_bytes()).hexdigest();assert lock_sha==lock_hashes[key]
     folds[key]=dict(equal=digest==expected['per_fold'][key]['want'],picks_sha256=digest,base_lock_sha256=lock_sha)
   u,_=ev.panel(G,got);gains=family_d(u,families).mean(axis=1);median=float(np.median(gains))
   report[part]=dict(per_seed_equal=seed_equal,per_seed_hashes_exact=hash_equal,per_seed_sha256=seed_hashes,
       per_fold=folds,per_seed_gain=gains.tolist(),median=median,median_exact=median==PUBLISHED[part])
 ok=all(all(r['per_seed_equal']) and r['per_seed_hashes_exact'] and r['median_exact'] and all(x['equal'] for x in r['per_fold'].values()) for r in report.values())
 durable(data/'regression.json',dict(status='PASS' if ok else 'REGRESSION_FAIL',partitions=report,BASE_refitting=0,BASE_new_model_scores=0,loader=R['loader_counters']))
 print(json.dumps(dict(regression='PASS' if ok else 'REGRESSION_FAIL',medians={p:r['median'] for p,r in report.items()})),flush=True)
 if not ok:raise RuntimeError('REGRESSION_FAIL; no BBQ fitting')
 return base

def drop_top(d,n=10):return float(np.sort(d)[::-1][n:].mean())
def paired_drop_top(dart,base,seed=378191,draws=10000,n=10):
 assert dart.shape==base.shape
 ix=np.random.default_rng(seed).integers(0,dart.shape[1],(draws,dart.shape[1]));obs=[];diffs=np.empty((draws,len(base)))
 for c in range(len(base)):
  obs.append(drop_top(dart[c],n)-drop_top(base[c],n))
  a=np.sort(dart[c][ix],axis=1)[:,:-n];b=np.sort(base[c][ix],axis=1)[:,:-n];diffs[:,c]=a.mean(axis=1)-b.mean(axis=1)
 boot=np.median(diffs,axis=1)
 per_root=[]
 for s,root in enumerate(SEEDS):
  selected=list(range(s,len(base),3));root_boot=np.median(diffs[:,selected],axis=1)
  per_root.append(dict(root_seed=root,median=float(np.median(np.asarray(obs)[selected])),ci95=np.quantile(root_boot,[.025,.975]).tolist(),lcb95=float(np.quantile(root_boot,.05))))
 return dict(per_cell=obs,median=float(np.median(obs)),ci95=np.quantile(boot,[.025,.975]).tolist(),lcb95=float(np.quantile(boot,.05)),per_root=per_root,rng=seed,draws=draws)
def decide(paired,gain_c,gain_b,harm_c,harm_b):
 per_seed=np.median(np.asarray(paired['per_cell']).reshape(3,3),axis=0)
 gates=dict(paired_drop_top10_at_least_002=bool(paired['median']>=.002 and paired['lcb95']>0),gain_retention=bool(np.median(gain_c)>=.9*np.median(gain_b)),harmed_nonincrease=bool(np.median(harm_c)<=np.median(harm_b)),positive_on_two_seeds=bool(np.sum(per_seed>0)>=2))
 return gates,'ADVANCE' if all(gates.values()) else 'MDS1_NOT_ADVANCED'

def jaccard(a,b):
 a=set(np.flatnonzero(np.asarray(a)!=0));b=set(np.flatnonzero(np.asarray(b)!=0))
 return dict(intersection=len(a&b),union=len(a|b),jaccard=len(a&b)/len(a|b) if a|b else 1.)

def evaluate(r303,base,dart,data,incumbent):
 import evaluate_h303 as ev
 G=r303.G;families=[r['family_id'] for r in G['train_rows']];D={'BASE':[],'NL15':[],'MDS1':[]};cells=[];choices=[];locks={};jac={}
 for part in PARTS:
  for k in range(4):
   for seed in SEEDS:
    b=json.loads((data/'base'/part/f'fold_{k}'/f'root_{seed}'/'fit_lock.json').read_text())['chosen']
    c=json.loads((data/part/f'fold_{k}'/f'root_{seed}'/'fit_lock.json').read_text())['chosen']
    n=json.loads((data/'nl15'/part/f'fold_{k}'/f'root_{seed}'/'fit_lock.json').read_text())['chosen']
    choices.append(dict(partition=part,fold=k,root_seed=seed,BASE={x:b[x] for x in ('tau','k','a')},NL15={x:n[x] for x in ('tau','k','a')},MDS1={x:c[x] for x in ('tau','k','a')}))
  for arm,pp in [('BASE',base),('NL15',incumbent),('MDS1',dart)]:
   u,invalid=ev.panel(G,pp[part]);d=family_d(u,families);stats=G['a302'].summarize_three_seeds(u,invalid,families,pp[part],boot_seed=378191);D[arm].extend(d)
   ix=np.random.default_rng(378191).integers(0,d.shape[1],(10000,d.shape[1]))
   jac[arm+'_'+part]=[{**jaccard(pp[part][a],pp[part][b]),'seed_pair':[SEEDS[a],SEEDS[b]]} for a,b in itertools.combinations(range(3),2)]
   for s,seed in enumerate(SEEDS):
    v=d[s];medboot=np.median(v[ix],axis=1)
    agreement=sum(x['NL15']==x['MDS1'] for x in choices if x['partition']==part and x['root_seed']==seed)
    cells.append(dict(partition=part,root_seed=seed,arm=arm,family_mean_gain=float(v.mean()),family_median_gain=float(np.median(v)),
       mean_gain_ci95=stats['per_seed_ci95'][s],family_median_ci95=np.quantile(medboot,[.025,.975]).tolist(),drop_top={str(n):drop_top(v,n) for n in (1,5,10)},
       lofo_min_mean_gain=float(np.min((v.sum()-v)/(len(v)-1))),family_wtl=[int((v>1e-4).sum()),int((abs(v)<=1e-4).sum()),int((v< -1e-4).sum())],
       harmed_share=float(np.mean(v<-.01)),invalid=int(invalid[s,:,1].sum()),slot0_invalid=int(invalid[s,:,0].sum()),
       switches=int(np.sum(pp[part][s]!=0)),switch_rate=float(np.mean(pp[part][s]!=0)),elo=stats['elo_by_seed'][s],elo_ci95=stats['elo_ci95_by_seed'][s],
       config_agreement_folds=agreement,picks_sha256=hashlib.sha256(pp[part][s].tobytes()).hexdigest()))
  locks['MDS1_'+part]=json.loads((data/f'MDS1_{part}_locks.json').read_text());locks['BASE_'+part]=json.loads((data/'base'/f'BASE_{part}_locks.json').read_text());locks['NL15_'+part]=json.loads((data/'nl15'/f'NL15_{part}_locks.json').read_text())
 dn,db,di=np.asarray(D['MDS1']),np.asarray(D['BASE']),np.asarray(D['NL15']);paired=paired_drop_top(dn,di);secondary=paired_drop_top(dn,db);paired['per_seed_partition_median']=np.median(np.asarray(paired['per_cell']).reshape(3,3),axis=0).tolist();agg={}
 gain={a:[x['family_mean_gain'] for x in cells if x['arm']==a] for a in D};harm={a:[x['harmed_share'] for x in cells if x['arm']==a] for a in D}
 gates,verdict=decide(paired,gain['MDS1'],gain['NL15'],harm['MDS1'],harm['NL15'])
 ix=np.random.default_rng(378191).integers(0,dn.shape[1],(10000,dn.shape[1]))
 for arm,v in [('BASE',db),('NL15',di),('MDS1',dn)]:
  draws=np.median(v[:,ix].mean(axis=2),axis=0);meddraws=np.median(np.median(v[:,ix],axis=2),axis=0)
  agg[arm]=dict(median_gain=float(np.median(gain[arm])),gain_ci95=np.quantile(draws,[.025,.975]).tolist(),literal_family_median_gain=float(np.median(np.median(v,axis=1))),
   literal_family_median_ci95=np.quantile(meddraws,[.025,.975]).tolist(),median_harmed=float(np.median(harm[arm])),median_elo=float(np.median([x['elo'] for x in cells if x['arm']==arm])),
   median_seed_jaccard=float(np.median([x['jaccard'] for key,vals in jac.items() if key.startswith(arm+'_') for x in vals])))
 durable(data/'lock_inventories.json',locks)
 full=dict(wave='H378',status='COMPLETE',verdict=verdict,criteria=gates,elo_label=ELO_LABEL,r3175_train='NOT_MEASURED',regression=json.loads((data/'regression.json').read_text()),
    primary_comparator='H352 NL15',paired_drop_top10=paired,secondary_paired_vs_BASE=secondary,nl15_regression=dict(status='PASS',panels=9,locks_scores=36,receipt_sha256=hashlib.sha256((data/'nl15-regression.json').read_bytes()).hexdigest()),aggregate=agg,cells=cells,chosen_configs=choices,config_equal_count=sum(x['NL15']==x['MDS1'] for x in choices),seed_jaccard=jac,
    fit_lock_inventory_sha256=hashlib.sha256((data/'lock_inventories.json').read_bytes()).hexdigest(),loader=R['loader_counters'],intervention='Rankers-only max_delta_step0->1.0 vs H352 NL15 leaves15/max_depth3; BASEreg_lambda1/implicitreg_alpha0/minchild20/LambdaRank/sigmoid1/labels/defaultgain/groups/GBDT/all other args/veto unchanged',dope=dict(composite_sha256_prefix='c51a56d0',loaded=False,scope='CPU selection layer only'),
    tau_units='unchanged invalid probability',margin_units='original LambdaRank score units; original numeric grid unchanged',outer_fit_consumer_accesses_before_lock=0,outer_packets_prebuilt=False,physical_outer_rows_read_before_lock=0,config_sensitivity='CONFIG_INSENSITIVE' if sum(x['NL15']==x['MDS1'] for x in choices)>=30 else 'CONFIG_SENSITIVE')
 durable(data/'metrics.json',full);np.savez_compressed(data/'picks.npz',**{a+'_'+p:d[p] for a,d in [('BASE',base),('NL15',incumbent),('MDS1',dart)] for p in PARTS});np.savez_compressed(data/'raw_family_gains.npz',BASE=db,NL15=di,MDS1=dn)
 print(json.dumps(dict(verdict=verdict,criteria=gates,paired=paired,aggregate=agg)),flush=True)
