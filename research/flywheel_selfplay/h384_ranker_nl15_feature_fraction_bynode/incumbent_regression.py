"""Archived H352 NL15 locks, once-only picks and raw utilities; zero model scoring."""
from pathlib import Path
import hashlib,json
import numpy as np
from run_h384 import PARTS,SEEDS,P,R,durable,outer_fold

def regress_nl15(r303,data):
 import evaluate_h303 as ev
 from evaluate_h384 import family_d
 assert json.loads((P/'nl15-fetch-receipt.json').read_text())['status']=='PASS'
 d=data/'nl15';old=json.loads((d/'metrics.json').read_text());G=r303.G;families=[r['family_id'] for r in G['train_rows']];folds={};panels=[];raw=[];configs=[]
 assert json.loads((d/'fit_done.json').read_text())==dict(status='PASS',locks=36)
 with np.load(d/'picks.npz') as picks,np.load(d/'raw_family_gains.npz') as gains:
  for part in PARTS:
   got=np.load(d/f'NL15_{part}.npy');assert got.shape==(3,1219) and np.array_equal(got,picks['NL15_'+part]);hashes=json.loads((d/f'NL15_{part}_locks.json').read_text())
   for k in range(4):
    ix=[i for i,f in enumerate(families) if outer_fold(part,f,G['a302'])==k]
    for s,seed in enumerate(SEEDS):
     t=d/part/f'fold_{k}'/f'root_{seed}';lock=json.loads((t/'fit_lock.json').read_text());marker=json.loads((t/'score_once.json').read_text());sha=hashlib.sha256((t/'fit_lock.json').read_bytes()).hexdigest()
     assert sha==hashes[f'{k}:{seed}']==old['fit_lock_sha256']['NL15_'+part][f'{k}:{seed}']==marker['fit_lock_sha256']
     assert marker['state']=='COMPLETE' and marker['contexts']==len(ix) and marker['picks_sha256']==hashlib.sha256(got[s,ix].astype(np.int32).tobytes()).hexdigest()
     folds[f'{part}/{k}/{seed}']=dict(lock_sha256=sha,score_sha256=hashlib.sha256((t/'score_once.json').read_bytes()).hexdigest());configs.append(lock['chosen'])
   u,_=ev.panel(G,got);dg=family_d(u,families);raw.extend(dg)
   for s,seed in enumerate(SEEDS):
    expected=next(x for x in old['cells'] if x['arm']=='NL15' and x['partition']==part and x['root_seed']==seed)
    assert hashlib.sha256(got[s].tobytes()).hexdigest()==expected['picks_sha256'] and float(dg[s].mean())==expected['family_mean_gain'];panels.append(dict(partition=part,root_seed=seed,picks_sha256=expected['picks_sha256'],family_mean_gain=float(dg[s].mean())))
  assert np.array_equal(np.asarray(raw),gains['NL15'])
 result=dict(status='PASS',comparator='archived H352 NL15',panels=panels,fold_roots=folds,panels_verified=9,locks_and_once_only_scores_verified=36,raw_family_gains_exact=True,NL15_refits=0,new_model_predictions=0)
 durable(P/'nl15-regression.json',result);durable(data/'nl15-regression.json',result);print(json.dumps(dict(NL15_regression='PASS',panels=9,locks_scores=36,raw_family_gains='EXACT')))
