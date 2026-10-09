"""Stopped-run/archive/exposure gate and one persistent single-writer lock."""
import fcntl,json,os,sqlite3,subprocess
from pathlib import Path
from jope_flywheel.program_io import bound,need,read,tick,create,sha
from jope_flywheel.protocol import canonical
from jope_flywheel.batch_stream import BatchStream,validate_manifest

class RunLock:
 def __init__(self,path,origin):
  path=Path(path);need(path.name==origin+'.lock'and path.is_absolute()and'..'not in path.parts and
   all(not p.is_symlink()for p in(path,*path.parents)),'fixed original model ownership lock')
  self.fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
  try:fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BaseException:os.close(self.fd);raise
 def close(self):
  if self.fd is not None:fcntl.flock(self.fd,fcntl.LOCK_UN);os.close(self.fd);self.fd=None
 def __enter__(self):return self
 def __exit__(self,*exc):self.close()

def prerequisite_documents(refs,loader=bound):
 docs={key:json.loads(loader(refs[key]))for key in('prior_config','prior_status','prior_result','preprocessing','batch_manifest','checkpoint_manifest','archive_status','admission_status')}
 original=refs['prior_config']['sha256'];status,result=docs['prior_status'],docs['prior_result'];step=refs['optimizer_step']
 need(status['state']=='complete'and status['config_sha256']==original and status['optimizer_steps']==step and
      status['result']==refs['prior_result']and status['attempt_custody']['status']=='PASS','closed complete prior optimizer/custody identity')
 need(result['state']=='completed_bounded_qualification'and result['config_sha256']==original and result['optimizer_steps']==step and
      len(result['proofs'])==step and result['planner']==refs['planner_anchor']and result['planner']['unresolved_batches']==0,'exact reconciled committed prior result')
 archive=docs['archive_status'];need(archive['schema']=='jope.neural-paper-custody/v1'and archive['state']=='verified'and
     archive['source_config_sha256']==original and archive['optimizer_steps']==step,'physical paper archive required')
 admission=docs['admission_status'];need(admission['schema']=='jope.flywheel-training-admission-controller/v1'and
     admission['state']=='committed'and admission['database_modified']is True,'actual exposure CAS commit required, not preview')
 plan=json.loads(loader(admission['plan']));commit=json.loads(loader(admission['commit']))
 need(plan['neural_config_sha256']==original and plan['planner']==result['planner'],'exposure CAS exact original planner head')
 need(commit['schema']=='jope.flywheel-training-admission-commit/v1'and commit['plan_sha256']==admission['plan']['sha256'],
      'admission commit exact plan')
 ids=[r['row']['identity']for r in plan['accepted']]
 need(len(ids)==len(set(ids))==admission['accepted']and set(ids)==set(commit['newly_admitted']+commit['already_identical'])and
      commit['released_capacity']==len(commit['newly_admitted']),'actual CAS closure')
 cp=docs['checkpoint_manifest'];need(cp['step']==step and cp['config_sha256']==original and cp['parameter_sha256']==result['proofs'][-1]['checkpoint_sha256'],
     'last physical checkpoint matches committed chain')
 return docs

def failed_attempt(refs,loader=bound,service_reader=None):
 attempt=refs['failed_attempt'];config=json.loads(loader(attempt['config']));fatal=json.loads(loader(attempt['fatal']));window=json.loads(loader(attempt['window']))
 need(config['owned_root']==attempt['root']and config['resume']['planner_anchor']==refs['planner_anchor']and
      window['config_sha256']==attempt['config']['sha256'],'failed continuation config/window/planner binding')
 need(fatal['state']=='failed'and fatal['automatic_retry']is False and
      fatal['error']=='ValueError: verification requires an independent read transaction','exact pre-optimizer failure only')
 if service_reader is None:service_reader=lambda unit:subprocess.check_output(['systemctl','--user','show',unit,'--property=ActiveState,MainPID,ExecMainStatus,NRestarts'],text=True,timeout=5)
 state=dict(x.split('=',1)for x in service_reader(attempt['unit']).splitlines()if'='in x)
 need(state.get('ActiveState')=='failed'and state.get('MainPID')=='0'and state.get('ExecMainStatus')=='1'and state.get('NRestarts')=='0','failed attempt must remain stopped without retry')
 return {'references':attempt,'unit_readback':state,'preserved_prior_window':window,'fatal':fatal}

def prepare(refs,root,deadline):
 """Read old SQL under a consistent transaction; copy only into new owned root."""
 tick(deadline);need(refs['root_shared_planner_released']is True,'root archive/CAS release required')
 service=subprocess.check_output(['systemctl','--user','show',refs['prior_unit'],'--property=ActiveState,MainPID'],text=True,timeout=5)
 state=dict(x.split('=',1)for x in service.splitlines()if'='in x)
 need(state.get('ActiveState')in('inactive','failed')and state.get('MainPID')=='0','prior neural unit must be stopped')
 docs=prerequisite_documents(refs);failed=failed_attempt(refs);path=Path(refs['planner_path'])
 need(path==Path(refs['prior_root'])/'training/batches.sqlite'and path.is_absolute()and
  all(not p.is_symlink()for p in(path,*path.parents)),'exact original planner location')
 for suffix in('-wal','-shm'):need(not Path(str(path)+suffix).is_symlink(),'planner sidecar symlink')
 need(path.stat().st_size<=32*1024**2,'bounded original planner')
 db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True,timeout=1,isolation_level=None);db.row_factory=sqlite3.Row
 try:
  db.execute('BEGIN');p=BatchStream.__new__(BatchStream);p.db=db;p.config=json.loads(db.execute("SELECT value FROM state WHERE key='config'").fetchone()[0])
  manifest_body=bound(refs['batch_manifest']);p.manifest,p.rows=validate_manifest(manifest_body,refs['batch_manifest']['sha256']);p.manifest_sha256=refs['batch_manifest']['sha256']
  need(p.config['initial_checkpoint_sha256']==refs['origin_parameter_sha256'],'same initialization origin')
  # Already inside the read-only snapshot also used by summary and backup.
  need(p._verify()==refs['planner_anchor']and p.pending()is None,'original planner changed or unknown batch')
  summary=p.summary();need(summary['optimizer_steps_reported']==refs['optimizer_step']and summary['checkpoint_sha256']==docs['checkpoint_manifest']['parameter_sha256'],'physical cursor equality')
  backup=Path(root)/'prior-planner.sqlite';fd=os.open(backup,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
  copied=sqlite3.connect(backup,timeout=1)
  try:db.backup(copied,pages=128,progress=lambda *args:tick(deadline),sleep=.01)
  finally:copied.close()
  db.execute('ROLLBACK')
 finally:db.close()
 body=read(backup,32*1024**2);proof={'schema':'jope.flywheel-neural-prior-session-gate/v1','planner_anchor':refs['planner_anchor'],
  'source_config_sha256':refs['prior_config']['sha256'],'paper_archive_verified':True,'actual_exposure_commit_verified':True,
  'prior_planner_backup':{'path':str(backup),'sha256':sha(body),'bytes':len(body)},'no_old_sql_writes':True,'unit':state,'preserved_failed_attempt':failed}
 create(Path(root)/'prior-session-gate.json',canonical(proof));tick(deadline)
 return {'prior_result':docs['prior_result'],'optimizer_step':refs['optimizer_step'],'preprocessing':docs['preprocessing'],
  'checkpoint_path':str(Path(refs['checkpoint_manifest']['path']).parent),'checkpoint_manifest_sha256':refs['checkpoint_manifest']['sha256'],
  'checkpoint_config_sha256':refs['prior_config']['sha256'],'parameter_sha256':docs['checkpoint_manifest']['parameter_sha256'],
  'batch_stream_path':str(path),'origin_parameter_sha256':refs['origin_parameter_sha256']},docs['batch_manifest']
