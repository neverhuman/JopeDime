"""H329: original nested LambdaRank with fixed ranker-only DART."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import functools,os,sys,json,hashlib,time,gc
import numpy as np
HERE=Path(__file__).resolve().parent;P=HERE.parent;FS=P/'vendor';DATA=Path('/home/ubuntu/jope-data/h329-gov03')
PARTS=('P0','P1','P2');SEEDS=(30001,30002,30003);R={};CURRENT_SEED=None;AUDIT=None

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

def ctor_audit(original,kind):
 @functools.wraps(original)
 def wrapped(self,*args,**kwargs):
  assert CURRENT_SEED is not None,'all campaign fits must have a member seed'
  original(self,*args,**kwargs)
  params=self.get_params()
  if kind=='LGBMRanker':
   assert params['boosting_type']=='dart' and params['objective']=='lambdarank'
   from dart_patch import EXTRA
   assert all(params[k]==v for k,v in EXTRA.items())
   assert params['drop_seed']==params['random_state']
   assert 'extra_trees' not in kwargs and 'extra_seed' not in kwargs and 'label_gain' not in kwargs
  else:
   assert params['boosting_type']=='gbdt' and params['objective']=='binary'
   assert not any(k in kwargs for k in ('drop_rate','skip_drop','max_drop','uniform_drop','xgboost_dart_mode','drop_seed'))
  if AUDIT is not None:
   with Path(AUDIT).open('a') as f:f.write(json.dumps(dict(kind=kind,member_seed=CURRENT_SEED,params=params),default=str)+'\n')
 return wrapped


def member_scope(original):
 @functools.wraps(original)
 def wrapped(*args,**kwargs):
  global CURRENT_SEED
  assert CURRENT_SEED is None
  CURRENT_SEED=int(kwargs.get('member_seed',args[4] if len(args)>4 else -1));assert CURRENT_SEED>=0
  try:return original(*args,**kwargs)
  finally:CURRENT_SEED=None
 return wrapped

def initialize():
 import runtime_h329 as rt
 G=rt.initialize();R.update(r303=rt.r303,loader=rt.tl,loader_counters=dict(dlval_feature_rows_read=0,physical_outer_rows_read_before_fit_lock=0),rt=rt)
 manifest=json.loads((HERE/'manifest.json').read_text());assert sha(HERE/'PREREG.md')==manifest['prereg_sha256']
 for name,digest in manifest['source_sha256'].items():assert sha(HERE/name)==digest,name
 return rt.r303

def full_packets():
 rt=R['rt'];G=R['r303'].G;rows=G['train_rows'];cids={r['context_sha256'] for r in rows}
 scope=rt.Scope(cids,cids);G['P8']=rt.packets(rows,scope);assert scope.denied==0
 return scope

def install_dart():
 import lightgbm as lgb
 from dart_patch import install
 record=install(R['r303'].G);durable(DATA/'dart_patch_receipt.json',record)
 for cls in [lgb.LGBMRanker,lgb.LGBMClassifier]:cls.__init__=ctor_audit(cls.__init__,cls.__name__)
 r302=R['r303'].G['r302'];r302.fit_and_freeze_member=member_scope(r302.fit_and_freeze_member)


def audit_summary(path):
 from collections import Counter
 from dart_patch import EXTRA
 counts=Counter();seeds=set();n=0
 for line in Path(path).read_text().splitlines():
  x=json.loads(line);p=x['params'];assert p['n_jobs']==1
  if x['kind']=='LGBMRanker':
   assert p['objective']=='lambdarank' and p['boosting_type']=='dart'
   assert all(p[k]==v for k,v in EXTRA.items()) and p['drop_seed']==p['random_state']
  else:assert x['kind']=='LGBMClassifier' and p['objective']=='binary' and p['boosting_type']=='gbdt'
  n+=1;counts[x['kind']]+=1;seeds.add(x['member_seed'])
 assert counts['LGBMRanker']==819 and counts['LGBMClassifier']==195
 return dict(status='PASS',calls=n,counts=dict(counts),member_seeds=sorted(seeds),audit_sha256=sha(path))


def task(args):
 global AUDIT
 part,k,seed=args;r303=R['r303'];G=r303.G;rt=R['rt'];start=time.time();rows=G['train_rows'];allc={r['context_sha256'] for r in rows}
 fitrows=[r for r in rows if outer_fold(part,r['family_id'],G['a302'])!=k];allowed={r['context_sha256'] for r in fitrows}
 scope=rt.Scope(allowed,allc,fit=True);sys.addaudithook(scope.hook)
 tdir=DATA/part/f'fold_{k}'/f'root_{seed}';tdir.mkdir(parents=True,exist_ok=True);assert not (tdir/'fit_lock.json').exists()
 AUDIT=tdir/'dart_constructor_audit.jsonl';assert not AUDIT.exists()
 try:
  assert 'P8' not in G,'fit parent must not preload outer packets'
  G['P8']=rt.packets(fitrows,scope);assert set(G['P8'])==allowed
  G['fam_to_fold']={f:outer_fold(part,f,G['a302']) for f in set(G['fam_by_cid'].values())};r303.SCRATCH=DATA/part
  lock=r303.task((k,seed));assert scope.denied==0 and scope.contexts==allowed
  # H303 fsyncs the file. Add parent fsync before any scoring can begin.
  durable(tdir/'fit_lock.json',lock)
  expected=set(G['a302'].EXPECTED_MEMBER_SEEDS[seed]);audit=audit_summary(AUDIT);assert set(audit['member_seeds'])==expected
  durable(tdir/'input_identity.json',dict(allowed_contexts=sorted(allowed),forbidden_contexts=sorted(allc-allowed),ranges=scope.ranges,packet_hashes=scope.packet_hashes))
  durable(tdir/'guard_receipt.json',dict(outer_test_packet_accesses_before_lock=0,guard_denials=0,outer_test_packets_prebuilt=0,physical_outer_rows_read_before_lock=0,
   fit_lock_sha256=sha(tdir/'fit_lock.json'),feature_bytes_read=scope.bytes,input_identity_sha256=sha(tdir/'input_identity.json'),constructor_audit=audit))
  from archive_h329 import compress_inner
  compress_inner(tdir,durable)
 finally:
  AUDIT=None;G.pop('P8',None);scope.active=False;gc.collect()
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
 assert (picks>=0).all();np.save(DATA/f'DART_{part}.npy',picks);durable(DATA/f'DART_{part}_locks.json',hashes)
 return picks

def verify_locks():
 records={}
 for part in PARTS:
  for k in range(4):
   for seed in SEEDS:
    td=DATA/part/f'fold_{k}'/f'root_{seed}';lp=td/'fit_lock.json';lock=json.loads(lp.read_text());guard=json.loads((td/'guard_receipt.json').read_text())
    assert guard['fit_lock_sha256']==sha(lp) and guard['physical_outer_rows_read_before_lock']==guard['guard_denials']==guard['outer_test_packets_prebuilt']==0
    assert sha(td/'input_identity.json')==guard['input_identity_sha256']
    assert audit_summary(td/'dart_constructor_audit.jsonl')==guard['constructor_audit']
    assert len(lock['inner_table'])==18 and len(lock['members'])==3
    for member in lock['members']:
     assert len(member['bags'])==8
     for model in [member['ranker'],member['veto'],*member['bags']]:assert sha(td/'refit'/model['file'])==model['sha256']
    arc=json.loads((td/'inner_archive.json').read_text());assert arc['exact_version_get_sha256_verified']
    records[f'{part}/{k}/{seed}']=sha(lp)
 durable(DATA/'lock_review.json',dict(status='PASS',locks=records,count=len(records)))

def main(phase):
 assert os.uname().nodename.split('.')[0]=='xbabe2' and os.getpriority(os.PRIO_PROCESS,0)==19
 import evaluate_h329 as ev
 r303=initialize()
 if phase=='regress':
  full_packets();ev.regress(r303,DATA);return
 assert json.loads((DATA/'regression.json').read_text())['status']=='PASS'
 assert json.loads((P/'input_identity_validation.json').read_text())['status']=='PASS'
 if phase=='fit':assert json.loads((DATA/'runtime_forecast.json').read_text())['launch_full_36']
 if phase=='fit':
  install_dart();tasks=[(p,k,s) for p in PARTS for k in range(4) for s in SEEDS]
  with ProcessPoolExecutor(max_workers=int(os.environ.get('H329_WORKERS','5'))) as pool:list(pool.map(task,tasks))
  verify_locks();durable(DATA/'fit_done.json',dict(status='PASS',locks=36));return
 verify_locks();assert json.loads((DATA/'fit_done.json').read_text())['status']=='PASS'
 full_packets();base={p:np.load(DATA/'base'/f'BASE_{p}.npy') for p in PARTS};dart={p:score(p) for p in PARTS};ev.evaluate(r303,base,dart,DATA)

if __name__=='__main__':
 sys.modules['run_h329']=sys.modules[__name__]
 main(sys.argv[1])
