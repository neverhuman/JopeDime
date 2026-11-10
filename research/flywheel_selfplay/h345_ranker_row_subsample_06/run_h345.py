"""H345: original nested LambdaRank with only the original ranker subsample value changed to .6."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import functools,os,sys,json,hashlib,time,gc,resource,datetime
import numpy as np
HERE=Path(__file__).resolve().parent;P=HERE.parent;FS=P/'vendor';DATA=Path('/home/ubuntu/jope-data/h345-solhwv')
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

def observe_init(original,kind):
 @functools.wraps(original)
 def wrapped(self,*args,**kwargs):
  original(self,*args,**kwargs)
  if AUDIT is not None:
   assert CURRENT_SEED is not None
   params=self.get_params();validate_parameters(kind,params)
   with Path(AUDIT).open('a') as f:f.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind=kind,member_seed=CURRENT_SEED,params=params),default=str)+'\n')
 return wrapped

def validate_parameters(kind,params):
 assert params['n_jobs']==1 and all(x not in params for x in ['extra_trees','extra_seed','label_gain','max_delta_step','feature_fraction_bynode','feature_fraction_seed','bagging_seed'])
 assert params['subsample_freq']==1 and params['colsample_bytree']==.8
 if kind=='LGBMRanker':assert params['objective']=='lambdarank' and params['subsample']==.6
 else:assert kind=='LGBMClassifier' and params['objective']=='binary' and params['subsample']==.8

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
 import runtime_h345 as rt
 G=rt.initialize();R.update(r303=rt.r303,loader=rt.tl,loader_counters=dict(dlval_feature_rows_read=0,physical_outer_rows_read_before_fit_lock=0),rt=rt)
 spec=json.loads((P/'base-native-spec.json').read_text());assert spec['status']=='PASS' and spec['ranker_models']==12636
 manifest=json.loads((HERE/'manifest.json').read_text());assert sha(HERE/'PREREG.md')==manifest['prereg_sha256']
 for name,digest in manifest['source_sha256'].items():assert sha(HERE/name)==digest,name
 return rt.r303

def full_packets():
 rt=R['rt'];G=R['r303'].G;rows=G['train_rows'];cids={r['context_sha256'] for r in rows}
 scope=rt.Scope(cids,cids);G['P8']=rt.packets(rows,scope);assert scope.denied==0
 return scope

def install_rowsub():
 import lightgbm as lgb
 from row_subsampling import install_function
 G=R['r303'].G
 receipts=[install_function(G['topk'],'fit_ranker'),install_function(G['bagged'],'fit_ranker_n')]
 durable(DATA/'setup/rowsub_override.json',dict(call_sites=receipts,veto_unmodified=True,only_changed_keyword='subsample',BASE_value=.8,ROWSUB_value=.6,subsample_freq=1,no_new_seed_keyword=True))
 for cls in [lgb.LGBMRanker,lgb.LGBMClassifier]:cls.__init__=observe_init(cls.__init__,cls.__name__)
 G['r302'].fit_and_freeze_member=member_scope(G['r302'].fit_and_freeze_member)

def audit_summary(path):
 from collections import Counter
 counts=Counter();seeds=set();n=0
 for line in Path(path).read_text().splitlines():
  x=json.loads(line);p=x['params'];validate_parameters(x['kind'],p)
  assert (x['kind']=='LGBMRanker' and p['objective']=='lambdarank') or (x['kind']=='LGBMClassifier' and p['objective']=='binary')
  n+=1;counts[x['kind']]+=1;seeds.add(x['member_seed'])
 assert n>0 and counts['LGBMRanker']>0 and counts['LGBMClassifier']>0
 return dict(status='PASS',calls=n,counts=dict(counts),member_seeds=sorted(seeds),audit_sha256=sha(path))

def task(args):
 global AUDIT
 part,k,seed=args;r303=R['r303'];G=r303.G;rt=R['rt'];start=time.time();before=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN);cpu_start=before.ru_utime+before.ru_stime+children.ru_utime+children.ru_stime;rows=G['train_rows'];allc={r['context_sha256'] for r in rows}
 fitrows=[r for r in rows if outer_fold(part,r['family_id'],G['a302'])!=k];allowed={r['context_sha256'] for r in fitrows}
 scope=rt.Scope(allowed,allc,fit=True);sys.addaudithook(scope.hook)
 tdir=DATA/part/f'fold_{k}'/f'root_{seed}';tdir.mkdir(parents=True,exist_ok=True);assert not (tdir/'fit_lock.json').exists()
 durable(tdir/'fit_started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task=list(args),prereg_commit=json.loads((P/'prereg-published.json').read_text())['commit']))
 AUDIT=tdir/'rowsub_constructor_audit.jsonl';assert not AUDIT.exists()
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
  from archive_h345 import compress_inner
  compress_inner(tdir,durable)
 finally:
  AUDIT=None;G.pop('P8',None);scope.active=False;gc.collect()
 print(json.dumps(dict(task=args,seconds=time.time()-start,chosen=lock['chosen'])),flush=True)
 after=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN);cpu_end=after.ru_utime+after.ru_stime+children.ru_utime+children.ru_stime
 timing=dict(task=list(args),chosen=lock['chosen'],wall_seconds=time.time()-start,cpu_seconds=cpu_end-cpu_start,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 durable(tdir/'timing.json',timing);return timing

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
 assert (picks>=0).all();np.save(DATA/f'ROWSUB_{part}.npy',picks);durable(DATA/f'ROWSUB_{part}_locks.json',hashes)
 return picks

def verify_locks():
 records={}
 for part in PARTS:
  for k in range(4):
   for seed in SEEDS:
    td=DATA/part/f'fold_{k}'/f'root_{seed}';lp=td/'fit_lock.json';lock=json.loads(lp.read_text());guard=json.loads((td/'guard_receipt.json').read_text())
    assert guard['fit_lock_sha256']==sha(lp) and guard['physical_outer_rows_read_before_lock']==guard['guard_denials']==guard['outer_test_packets_prebuilt']==0
    assert sha(td/'input_identity.json')==guard['input_identity_sha256']
    assert audit_summary(td/'rowsub_constructor_audit.jsonl')==guard['constructor_audit']
    assert len(lock['inner_table'])==18 and len(lock['members'])==3
    for member in lock['members']:
     assert len(member['bags'])==8
     for model in [member['ranker'],member['veto'],*member['bags']]:assert sha(td/'refit'/model['file'])==model['sha256']
    arc=json.loads((td/'inner_archive.json').read_text());assert arc['exact_version_get_sha256_verified']
    records[f'{part}/{k}/{seed}']=sha(lp)
 unchanged=0
 for part in PARTS:
  for k in range(4):
   for seed in SEEDS:
    td=DATA/part/f'fold_{k}'/f'root_{seed}';rowsub=json.loads((td/'fit_lock.json').read_text());base=json.loads((DATA/'base'/part/f'fold_{k}'/f'root_{seed}'/'fit_lock.json').read_text())
    for a,b in zip(rowsub['members'],base['members']):assert a['member_index']==b['member_index'] and a['seed']==b['seed'] and a['veto']['sha256']==b['veto']['sha256'];unchanged+=1
 assert unchanged==108
 durable(DATA/'lock_review.json',dict(status='PASS',locks=records,count=len(records),unchanged_final_veto_model_hashes=unchanged))

def main(phase):
 assert os.uname().nodename.split('.')[0]=='xbabe2' and os.getpriority(os.PRIO_PROCESS,0)==19
 import evaluate_h345 as ev
 r303=initialize()
 if phase=='regress':
  full_packets();ev.regress(r303,DATA);return
 assert json.loads((DATA/'regression.json').read_text())['status']=='PASS'
 assert json.loads((P/'input_identity_validation.json').read_text())['status']=='PASS'
 if phase=='fit':
  install_rowsub();tasks=[(p,k,s) for p in PARTS for k in range(4) for s in SEEDS]
  workers=int(os.environ['H345_WORKERS']);admission=json.loads((DATA/'cpu_admission.json').read_text());assert workers==len(admission['admitted']) and set(os.sched_getaffinity(0))==set(admission['admitted'])
  from admission_h345 import forecast
  with ProcessPoolExecutor(max_workers=workers) as pool:
   futures=[pool.submit(task,args) for args in tasks]
   for count,future in enumerate(as_completed(futures),1):
    result=future.result()
    if count==1:
     work=max(result['cpu_seconds'],result['wall_seconds'])*35;now=datetime.datetime.now(datetime.timezone.utc);projection=forecast(work,workers,now)
     durable(DATA/'first_cell_forecast.json',dict(status='PASS' if projection['fits'] else 'INCOMPLETE_RUNTIME_FORECAST',first_completed=result,completed=1,remaining=35,rule='max(first cell own CPU delta, wall)*35; ignore all in-flight progress',**projection))
     if not projection['fits']:
      durable(DATA/'STOP.json',dict(reason=['INCOMPLETE_RUNTIME_FORECAST'],projection=projection))
      import subprocess
      subprocess.run(['systemctl','--user','stop','h345r-run.service'],check=False)
      raise RuntimeError('INCOMPLETE_RUNTIME_FORECAST; no partial scoring/restart')
  verify_locks();durable(DATA/'fit_done.json',dict(status='PASS',locks=36));return
 verify_locks();assert json.loads((DATA/'fit_done.json').read_text())['status']=='PASS'
 full_packets();base={p:np.load(DATA/'base'/f'BASE_{p}.npy') for p in PARTS};rowsub={p:score(p) for p in PARTS};ev.evaluate(r303,base,rowsub,DATA)

if __name__=='__main__':
 sys.modules['run_h345']=sys.modules[__name__]
 main(sys.argv[1])
