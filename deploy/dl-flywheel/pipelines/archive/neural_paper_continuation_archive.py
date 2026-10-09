"""Versioned paper custody of a closed neural run; no scientific-result mutation."""
import fcntl,hashlib,importlib.util,json,os,sqlite3,subprocess,sys,time
from pathlib import Path

ROOT=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final')
CAP=384*1024**2
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def verify_prior_reference(raw,ref):
 need(sha(raw)==ref['sha256'],'exact prior paper custody hash')
 if 'bytes' in ref:
  need(type(ref['bytes']) is int and len(raw)==ref['bytes'],'exact prior paper custody declared size')
 return json.loads(raw)
def closed_unit(unit):
 need(unit=='jopedime-flywheel-neural-continuation1-recovery1-20261009','owned unit only')
 r=subprocess.run(['systemctl','--user','show',unit,'--property=ActiveState,MainPID'],text=True,capture_output=True,check=True,timeout=5)
 fields=dict(s.split('=',1)for s in r.stdout.splitlines()if '='in s)
 need(fields.get('ActiveState')in('inactive','failed')and fields.get('MainPID')=='0','neural run must be closed')
 return fields
def selected(root):
 """Closed-run exact allowlist; GP binaries remain in existing result custody."""
 names=['config.json','registry.json','window.json','status.json','anchor.json','archive-config.json','research-admission-verification.json','loaded-receipt.json','role-proof.json','put-attempts.jsonl','prior-session-gate.json','snapshot-cache-proof.json','local-binary-inventory.json','prior-planner.sqlite']
 result=[]
 for name in names:
  p=root/name
  need(p.is_file()and not p.is_symlink(),'required evidence missing: '+name);result.append((name,p))
 for folder in ('runtime','training','attempt-custody'):
  d=root/folder;need(d.is_dir()and not d.is_symlink(),'evidence directory missing')
  for p in sorted(d.rglob('*')):
   need(not p.is_symlink(),'symlink in evidence')
   if p.is_file()and p.suffix in ('.py','.json','.jsonl','.safetensors','.sqlite'):
    result.append((str(p.relative_to(root)),p))
 for name in ('neural_core.py','neural_session.py','neural_resume.py','resume_gate.py','snapshot_cache.py','neural_cli.py','neural_admission.py','campaign_custody.py','flywheel_kumos_snapshot.py','neural-failure.json'):
  p=root/name
  if p.exists():result.append((name,p))
 need(0<len(result)<=512 and len({n for n,p in result})==len(result),'bounded unique evidence files')
 need(sum(p.stat().st_size for n,p in result)<=CAP,'total evidence cap384MiB')
 return result
def session_steps(status):
 start=status['session_start_optimizer_step'];end=status['optimizer_steps'];count=status['session_optimizer_steps']
 need(type(start)is int and start==8 and type(end)is int and type(count)is int and 0<=count<=32 and end-start==count,'exact bounded continuation step interval')
 return range(start+1,end+1)

def planner_copy_source(root,config,result,b,deadline):
 need(type(result['optimizer_steps'])is int and result['optimizer_steps']>0 and len(result['proofs'])==result['optimizer_steps'],'cumulative prior and new optimizer proofs required')
 path=Path(config['resume']['planner_path'])
 expected=ROOT/'flywheel_neural_20261009_v1/training/batches.sqlite'
 need(path==expected,'exact shared original planner')
 for suffix in('','-wal','-shm'):
  side=Path(str(path)+suffix)
  if side.exists():
   f,_=b.safe_open(side,64*1024**2);f.close()
 sys.path.insert(0,str(root/'runtime'))
 from jope_flywheel.batch_stream import BatchStream,validate_manifest
 raw=b.read_small(root/'training/batch-manifest.json',32*1024**2)
 db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.5,isolation_level=None);db.row_factory=sqlite3.Row
 try:
  db.execute('BEGIN');p=BatchStream.__new__(BatchStream);p.db=db
  p.config=json.loads(db.execute("SELECT value FROM state WHERE key='config'").fetchone()[0])
  p.manifest,p.rows=validate_manifest(raw,sha(raw));p.manifest_sha256=sha(raw)
  need(p._verify()==result['planner'],'current shared planner does not match terminal session')
  need(p.pending()is None,'unresolved optimizer cannot be archived as closed continuation')
  summary=p.summary()
  need(summary['optimizer_steps_reported']==result['optimizer_steps']and summary['checkpoint_sha256']==result['proofs'][-1]['checkpoint_sha256'],'shared cursor/physical last checkpoint')
  db.execute('ROLLBACK')
 finally:db.close()
 b.time_check(deadline)
 return path

def main():
 here=Path(__file__).resolve().parent
 config_raw=(here/'config.json').read_bytes();c=json.loads(config_raw)
 need(c['helper_sha256']==sha(Path(__file__).read_bytes()),'helper pin')
 need(c['schema']=='jope.neural-paper-continuation-custody-config/v1','config schema')
 root=Path(c['source_root']);need(root==ROOT/'flywheel_neural_continuation1_recovery1_20261009','source confinement')
 need(all(not p.is_symlink()for p in(root,*root.parents)),'source symlink')
 need(sha((root/'config.json').read_bytes())==c['source_config_sha256'],'neural config pin')
 deadline=c['deadline_unix'];need(deadline-c['created_unix']==180 and c['created_unix']<=time.time()<deadline,'fixed180sec new custody window')
 source=Path(c['base_helper']['path']);need(sha(source.read_bytes())==c['base_helper']['sha256'],'pinned archive primitives')
 spec=importlib.util.spec_from_file_location('paper_base',source);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
 limits=b.cgroup_check()
 neural_config=json.loads((root/'config.json').read_bytes())
 need(neural_config['schema']=='jope.flywheel-neural-continuation-campaign/v1','continuation source config')
 for name,pin in neural_config['owned_sources'].items():need(sha(b.read_small(root/name,32*1024**2))==pin,'owned continuation source pin')
 for ref in neural_config['source_pins']:
  raw=b.read_small(Path(ref['path']),32*1024**2);need(sha(raw)==ref['sha256']and len(raw)==ref['bytes'],'frozen source pin')
 lock=Path(neural_config['resume']['lock_path'])
 need(lock==ROOT/'flywheel_neural_persistent_v1'/(neural_config['neural']['initial_parameter_sha256']+'.lock'),'exact persistent model lock')
 need(all(not p.is_symlink()for p in(lock,*lock.parents))and lock.is_file(),'existing nonsymlink model lock')
 lockfd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
 fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 service=closed_unit(c['source_unit'])
 out=here/'result';out.mkdir(mode=0o700)
 status=json.loads(b.read_small(root/'status.json'))
 need(status['config_sha256']==c['source_config_sha256']and status['attempt_custody']['status']=='PASS','closed native attempt custody')
 selected_files=selected(root);entries=[]
 result_ref=status['result'];result_raw=b.read_small(Path(result_ref['path']),32*1024**2)
 need(sha(result_raw)==result_ref['sha256']and len(result_raw)==result_ref['bytes'],'terminal result pin')
 result=json.loads(result_raw);need(result['optimizer_steps']==status['optimizer_steps'],'same global step')
 shared_planner=planner_copy_source(root,neural_config,result,b,deadline)
 selected_files.append(('shared/current-batches.sqlite',shared_planner))
 prior_ref=neural_config['resume']['archive_status'];prior_raw=b.read_small(Path(prior_ref['path']))
 verify_prior_reference(prior_raw,prior_ref)
 prior=json.loads(prior_raw)
 need(prior['state']=='verified'and prior['source_config_sha256']==neural_config['resume']['prior_config']['sha256']and prior['optimizer_steps']==status['session_start_optimizer_step'],'prior paper chain coverage')
 selected_files.append(('external/prior-paper-status.json',Path(prior_ref['path'])))
 failed_root=ROOT/'flywheel_neural_continuation1_20261009'
 need(set(c['failed_attempt_refs'])=={'config.json','window.json','fatal-status.json','resume_gate.py','neural_cli.py'},'exact preserved pre-load failure closure')
 for name,ref in c['failed_attempt_refs'].items():
  path=Path(ref['path']);need(path==failed_root/name,'failed-attempt source confinement')
  body=b.read_small(path,32*1024**2);need(sha(body)==ref['sha256']and len(body)==ref['bytes'],'failed-attempt source pin')
  selected_files.append(('external/failed-preload/'+name,path))
 for step in session_steps(status):
  checkpoint=root/'training'/('checkpoint-%04d'%step)
  metadata=json.loads(b.read_small(checkpoint/'manifest.json'))
  tensor=checkpoint/'state.safetensors'
  stream,info=b.safe_open(tensor,64*1024**2);stream.close()
  need(metadata['step']==step and metadata['config_sha256']==c['source_config_sha256']and
       metadata['state_bytes']==info.st_size and metadata['state_sha256']==b.sha_file(tensor),'physical checkpoint coverage/hash')
 neural_config=json.loads((root/'config.json').read_bytes())
 for key in ('snapshot_acceptance','snapshot_status','snapshot_coverage','gpu_preflight'):
  ref=neural_config[key];path=Path(ref['path']);body=b.read_small(path,32*1024**2)
  need(sha(body)==ref['sha256']and len(body)==ref['bytes'],'exact referenced paper evidence: '+key)
  selected_files.append(('external/'+key+'.json',path))
 snapshot_status=json.loads(b.read_small(Path(neural_config['snapshot_status']['path'])))
 for key in ('snapshot','loaded_receipt'):
  ref=snapshot_status[key];path=Path(ref['path']);body=b.read_small(path,32*1024**2)
  need(sha(body)==ref['sha256']and len(body)==ref['bytes'],'exact referenced snapshot evidence: '+key)
  selected_files.append(('external/'+key+'.json',path))
 for name,path in selected_files:
  b.time_check(deadline)
  if path.suffix=='.sqlite':
   target=out/('db-'+sha(name.encode())[:16]+'.sqlite')
   for suffix in ('','-wal','-shm'):
    sidecar=Path(str(path)+suffix)
    if sidecar.exists():
     stream,_=b.safe_open(sidecar,64*1024**2);stream.close()
   connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.5)
   copied=sqlite3.connect(str(target))
   try:
    connection.execute('PRAGMA query_only=ON')
    connection.backup(copied,pages=128,progress=lambda status,remaining,total:b.time_check(deadline),sleep=.01)
    need(copied.execute('PRAGMA integrity_check').fetchone()[0]=='ok','SQLite evidence integrity')
   finally:connection.close();copied.close()
   path=target
  stream,info=b.safe_open(path,64*1024**2);stream.close()
  entries.append((name,path,b.sha_file(path),info.st_size))
 manifest={'schema':'jope.neural-paper-manifest/v1','source_config_sha256':c['source_config_sha256'],
   'source_root':str(root),'closed_service':service,'at':time.time(),'optimizer_steps':status.get('optimizer_steps',0),
   'terminal_state':status['state'],'session_start_optimizer_step':status['session_start_optimizer_step'],'session_optimizer_steps':status['session_optimizer_steps'],'prior_paper_custody':prior,'artifacts':[{'path':n,'sha256':h,'bytes':size}for n,p,h,size in entries],
   'scope':'Physical checkpoints including optimizer/RNG, source, preprocessing, exact batch order/proofs, full loss and measured games. GP executable/data artifacts retain their separate exact-version S3 result references. Custody verifies evidence bytes; it does not establish generalization.'}
 raw=canonical(manifest);b.write_new(out/'manifest.json',raw,out)
 entries.append(('PAPER_MANIFEST.json',out/'manifest.json',sha(raw),len(raw)))
 need(len(entries)<=512 and sum(size for name,path,digest,size in entries)<=CAP,'full archive evidence cap384MiB including external references and SQLite copies')
 package=out/'archive.tar';pins=b.package_tar(package,entries,out,deadline);b.verify_tar(package,pins,deadline)
 closed_unit(c['source_unit'])
 from botocore.config import Config
 import boto3
 client=boto3.Session(profile_name='veox-jepson-prod').client('s3',config=Config(connect_timeout=5,read_timeout=10,retries={'max_attempts':0},max_pool_connections=1))
 package_sha=b.sha_file(package);prefix=b.PREFIX+'neural/'+c['source_config_sha256']+'/'+package_sha+'/'
 uploaded=b.put_verified(client,prefix+'archive.tar',package,package_sha,package.stat().st_size,deadline)
 b.time_check(deadline)
 proof={'schema':'jope.neural-paper-custody/v1','state':'verified','at':time.time(),'source_config_sha256':c['source_config_sha256'],
   'optimizer_steps':status.get('optimizer_steps',0),'session_start_optimizer_step':status['session_start_optimizer_step'],'session_optimizer_steps':status['session_optimizer_steps'],'prior_paper_custody':prior,'terminal_state':status['state'],'package':uploaded,'manifest_sha256':sha(raw),
   'files':len(pins),'checkpoint_files':sum(x['path'].endswith('.safetensors')for x in pins),'resource_limits':limits}
 proof_raw=canonical(proof)
 proof['receipt']=b.put_verified(client,prefix+'verification.json',proof_raw,sha(proof_raw),len(proof_raw),deadline)
 b.time_check(deadline)
 b.write_new(out/'status.json',canonical(proof),out);fcntl.flock(lockfd,fcntl.LOCK_UN);os.close(lockfd);print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
