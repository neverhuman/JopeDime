"""Exact recovered H352 picks, raw utilities, fold locks and score marker regression."""
from pathlib import Path
import json,hashlib
import numpy as np
def regress(r303,data,parts,seeds,outer_fold,durable):
 import evaluate_h303 as ev
 G=r303.G;families=[r['family_id'] for r in G['train_rows']];root=data/'nl15';pinned=json.loads((Path(__file__).parent/'h352_metrics_reference.json').read_text());report={}
 with np.load(root/'picks.npz') as z,np.load(root/'raw_family_gains.npz') as raw:
  for pindex,part in enumerate(parts):
   picks=np.load(root/f'NL15_{part}.npy');assert picks.shape==(3,1219) and np.array_equal(picks,z['NL15_'+part])
   u,_=ev.panel(G,picks);f=np.asarray(families);names=sorted(set(families));gains=np.stack([np.array([u[s,f==n,1].mean()-u[s,f==n,0].mean() for n in names]) for s in range(3)])
   assert np.array_equal(gains,raw['NL15'][pindex*3:pindex*3+3])
   oldcells=[next(c for c in pinned['cells'] if c['arm']=='NL15' and c['partition']==part and c['root_seed']==seed) for seed in seeds]
   assert all(hashlib.sha256(picks[s].tobytes()).hexdigest()==oldcells[s]['picks_sha256'] and float(gains[s].mean())==oldcells[s]['family_mean_gain'] for s in range(3))
   hashes=json.loads((root/f'NL15_{part}_locks.json').read_text());assert hashes==pinned['fit_lock_sha256']['NL15_'+part];folds={}
   for k in range(4):
    ix=[i for i,fam in enumerate(families) if outer_fold(part,fam,G['a302'])==k]
    for s,seed in enumerate(seeds):
     td=root/part/f'fold_{k}'/f'root_{seed}';dig=hashlib.sha256((td/'fit_lock.json').read_bytes()).hexdigest();marker=json.loads((td/'score_once.json').read_text());picksha=hashlib.sha256(picks[s,ix].astype(np.int32).tobytes()).hexdigest()
     assert dig==hashes[f'{k}:{seed}']==marker['fit_lock_sha256'] and marker['state']=='COMPLETE' and marker['contexts']==len(ix) and marker['picks_sha256']==picksha
     folds[f'{k}:{seed}']=dict(lock_sha256=dig,picks_sha256=picksha,exact=True)
   report[part]=dict(panels=3,seed_picks_sha256=[c['picks_sha256'] for c in oldcells],raw_family_arrays_exact=True,median_gain=float(np.median(gains.mean(axis=1))),folds=folds)
 result=dict(status='PASS',partitions=report,panels=9,fold_locks_and_scores=36,NL15_refits=0,NL15_new_model_predictions=0,raw_family_arrays_exact=True,committed_metrics_sha256=hashlib.sha256((Path(__file__).parent/'h352_metrics_reference.json').read_bytes()).hexdigest())
 durable(data/'nl15_regression.json',result);durable(Path(__file__).parent.parent/'nl15-regression.json',result);return result
