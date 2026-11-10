"""H321 paired TRAIN-only target-and-loss adapter; frozen formula in PREREG.md."""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os,sys,json,hashlib,time
import numpy as np
HERE=Path(__file__).resolve().parent
P=Path(os.environ.get('H321_CUSTODY',str(HERE.parent)))
FS=Path(os.environ.get('H321_FS',str(P/'vendor')))
DATA=Path(os.environ.get('H321_DATA','/home/ubuntu/jope-data/h321-solhwv'))
PARTS=('P0','P1','P2');SEEDS=(30001,30002,30003);R={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def durable(path,obj):
 path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.new')
 with tmp.open('w') as f:json.dump(obj,f,indent=1,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def outer_fold(part,family,a302):
 if part=='P0':return a302.outer_fold_of_family(family)
 return hashlib.sha256({'P1':b'H308-outer-v1\x00','P2':b'H308-outer-v2\x00'}[part]+family.encode()).digest()[-1]%4

def normalized_target(utility):
 u=np.asarray(utility,dtype=np.float64);mad=float(np.median(np.abs(u-np.median(u))))
 return np.clip((u-u[0])/max(mad,1e-3),-3,3).astype(np.float32),mad

def fit_normalized(X,y,n_groups,k,lr,n_est,random_state):
 import lightgbm as lgb
 assert len(y)==n_groups*k and np.isfinite(y).all() and np.min(y)>=-3 and np.max(y)<=3
 model=lgb.LGBMRegressor(objective='regression_l2',metric='l2',boosting_type='gbdt',max_depth=3,num_leaves=7,
  min_child_samples=20,learning_rate=float(lr),n_estimators=int(n_est),subsample=.8,subsample_freq=1,
  colsample_bytree=.8,reg_lambda=1.,n_jobs=1,random_state=int(random_state),verbosity=-1,deterministic=True,force_row_wise=True)
 model.fit(X,y)
 return model

def fit_normalized_200(X,y,n_groups,k,lr,random_state):return fit_normalized(X,y,n_groups,k,lr,200,random_state)

def fit_members(*args):
 fold_dir,k,seed,rows,packets,fam_by_cid,tau=args;path=Path(fold_dir)/'members.json'
 if path.exists():return json.loads(path.read_text())
 local={};records={}
 for row in rows:
  c=row['context_sha256'];original=packets[c];target,mad=normalized_target(original['utility_k'])
  local[c]=dict(original,rel_k=target);records[c]=dict(mad=mad,target_sha256=hashlib.sha256(target.tobytes()).hexdigest())
 members=R['fit_original'](fold_dir,k,seed,rows,local,fam_by_cid,tau)
 durable(path,members);durable(Path(fold_dir)/'target_receipt.json',dict(formula='clip((u-u0)/max(MAD(u),.001),-3,3)',
   objective='regression_l2',contexts=len(local),context_targets=records,raw_utility_and_gate_unchanged=True))
 return members

def prepare():
 assert json.loads((P/'prereg-published.json').read_text())['published']
 manifest=json.loads((HERE/'manifest.json').read_text());assert sha(HERE/'PREREG.md')==manifest['prereg_sha256']
 for rel,digest in manifest['sources'].items():assert sha(FS/rel)==digest,rel
 sys.path.insert(0,str(FS/'h303_ensvote2_fully_nested_hparams'));sys.path.insert(1,str(FS/'h308_ensvote2_trainonly_repartition'))
 import run_h303 as r303
 import trainonly_loader as tl
 selected=json.loads(Path('/home/ubuntu/jope-data/h50/input/selected.json').read_text())
 r303.np=tl.TrainOnlyNumpy({r['context_sha256'] for r in selected['train']});r303.SCRATCH=DATA/'setup';r303.setup();r303.np=np
 assert tl.COUNTERS['dlval_feature_rows_read']==0
 assert all(r.get('has_meta_test') is not True and r['task']=='regression' for r in r303.G['train_rows'])
 R.update(r303=r303,loader=tl,loader_counters=dict(tl.COUNTERS),fit_original=r303.fit_members)
 return r303

def install_adapter():
 r303=R['r303'];R['original_gate_fit']=r303.G['veto'].fit_gate
 sys.modules['topk'].fit_ranker=fit_normalized_200;r303.G['bagged'].fit_ranker_n=fit_normalized
 r303.fit_members=fit_members
 assert r303.G['veto'].fit_gate is R['original_gate_fit']

def task(args):
 part,k,seed=args;r303=R['r303'];G=r303.G
 G['fam_to_fold']={f:outer_fold(part,f,G['a302']) for f in set(G['fam_by_cid'].values())}
 tdir=DATA/part/f'fold_{k}'/f'root_{seed}';r303.SCRATCH=DATA/part;start=time.time()
 full=G['P8'];test={r['context_sha256'] for r in G['train_rows'] if G['fam_to_fold'][r['family_id']]==k}
 guarded=R['loader'].GuardedPackets(full,test);G['P8']=guarded
 try:
  lock=r303.task((k,seed));assert guarded.denied==0
  durable(tdir/'guard_receipt.json',dict(outer_test_packet_accesses_before_lock=0,guard_denials=0,
    outer_test_packets_prebuilt=len(test),fit_lock_sha256=sha(tdir/'fit_lock.json'),target_and_loss='CNORM regression_l2'))
  from archive_h321 import compress_inner
  compress_inner(tdir,durable)
 finally:G['P8']=full
 print(json.dumps(dict(task=args,seconds=time.time()-start,chosen=lock['chosen'])),flush=True)
 return lock['chosen']

def score(part):
 r303=R['r303'];G=r303.G;G['fam_to_fold']={f:outer_fold(part,f,G['a302']) for f in set(G['fam_by_cid'].values())}
 cids=[r['context_sha256'] for r in G['train_rows']];picks=np.full((3,len(cids)),-1,dtype=np.int32);hashes={}
 for k in range(4):
  test=[i for i,c in enumerate(cids) if G['fam_to_fold'][G['fam_by_cid'][c]]==k]
  for s,seed in enumerate(SEEDS):
   tdir=DATA/part/f'fold_{k}'/f'root_{seed}';lp=tdir/'fit_lock.json';lock=json.loads(lp.read_text())
   hashes[f'{k}:{seed}']=sha(lp);marker=tdir/'score_once.json';assert not marker.exists(),'outer already scored'
   durable(marker,dict(state='STARTED',fit_lock_sha256=sha(lp)))
   X=np.concatenate([G['P8'][cids[i]]['X'] for i in test])
   outs=r303.score_members(tdir/'refit',lock['members'],X,len(test),G['ens']);ch=lock['chosen']
   ml=[r303.member_picks(o,ch['tau'],ch['k'],G['veto'],G['core'],G['bagged']) for o in outs]
   p=r303.ens_picks(ml,ch['a'],G['ens']);picks[s,test]=p
   durable(marker,dict(state='COMPLETE',fit_lock_sha256=sha(lp),contexts=len(test),picks_sha256=hashlib.sha256(np.asarray(p,dtype=np.int32).tobytes()).hexdigest()))
 assert (picks>=0).all();np.save(DATA/f'CNORM_{part}.npy',picks);durable(DATA/f'CNORM_{part}_locks.json',hashes)
 return picks

def main():
 assert os.uname().nodename.split('.')[0]=='xbabe2' and os.getpriority(os.PRIO_PROCESS,0)==19
 import evaluate_h321 as ev
 r303=prepare();base=ev.regress(r303,DATA)
 install_adapter();tasks=[(p,k,s) for p in PARTS for k in range(4) for s in SEEDS]
 with ProcessPoolExecutor(max_workers=int(os.environ.get('H321_WORKERS','6'))) as pool:list(pool.map(task,tasks))
 cnorm={p:score(p) for p in PARTS};ev.evaluate(r303,base,cnorm,DATA)

if __name__=='__main__':main()
