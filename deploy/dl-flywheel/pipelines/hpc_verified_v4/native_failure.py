"""Nonnumeric native failure custody. Never publishes canonical score rows.

Old unknown-attempt states remain untouched. New live leases may terminate with
the already-supported failure_custody_verified outcome after exact S3 readback.
"""
import json,math,os,sqlite3,struct,time
from pathlib import Path
from neural_outbox import Outbox,canonical,sha,digest,need,hash64
from jope_flywheel.program_io import tick,read,create
from jope_flywheel.s3_library import BUCKET,code

PREFIX='experiments/gp-policy-dl/v2-trainonly/program-attempts/flywheel/v1/native/'
MAX_BYTES=8*1024**2
TOLERANCE=1e-9

class ParityRejected(ValueError):
 def __init__(self,diagnostic):
  self.diagnostic=diagnostic
  super().__init__('emitted/native prediction parity failed; tolerance is fixed')

def parity_diagnostic(actual,reference):
 need(len(actual)==len(reference)and 25<=len(actual)<=2000 and all(math.isfinite(x)for x in actual+reference),'finite complete parity diagnostics')
 errors=[abs(a-b)for a,b in zip(actual,reference)]
 normalized=[e/max(1.,abs(b))for e,b in zip(errors,reference)]
 failed=[i for i,e in enumerate(normalized)if e>TOLERANCE]
 need(bool(failed),'diagnostic requires actual fixed-tolerance rejection')
 return {'schema':'jope.native-rust-parity-rejection/v1','kind':'fixed_prediction_parity_rejection',
  'passed':False,'rows':len(actual),'bit_exact_rows':sum(struct.pack('<d',a)==struct.pack('<d',b)for a,b in zip(actual,reference)),
  'tolerance':TOLERANCE,'criterion':'abs(actual-reference) <= tolerance * max(1,abs(reference))',
  'max_absolute_error':max(errors),'max_normalized_error':max(normalized),'failed_rows':failed,
  'native_predictions':reference,'rust_predictions':actual,
  'native_predictions_f64_sha256':sha(struct.pack('<'+'d'*len(reference),*reference)),
  'rust_predictions_f64_sha256':sha(struct.pack('<'+'d'*len(actual),*actual)),
  'numeric_score':None,'teaching_eligible':False,'optimizer_reward_allowed':False}

class VerifiedReadSnapshot:
 """Let the frozen verifier own logical BEGIN/COMMIT inside one physical RO txn.

 Only its SELECT statements reach SQLite; transaction-control statements are
 acknowledged without ending the caller's snapshot. No SQL mutations are legal.
 """
 def __init__(self,db):self.db=db
 @property
 def in_transaction(self):return False
 def execute(self,sql,*args):
  if sql in ('BEGIN','COMMIT'):return self.db.execute('SELECT 1')
  need(sql.lstrip().upper().startswith('SELECT '),'readonly verifier SQL only')
  return self.db.execute(sql,*args)

def anchored_emission(path,identity,expected_config,expected_campaign):
 p=Path(path);need(p.is_absolute()and all(not x.is_symlink()for x in(p,*p.parents)),'outbox nonsymlink')
 db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,isolation_level=None,timeout=2);db.row_factory=sqlite3.Row
 try:
  db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
  checker=Outbox.__new__(Outbox);checker.db=VerifiedReadSnapshot(db);head=checker.verify()
  conf=json.loads(db.execute('SELECT body FROM config').fetchone()[0]);need(conf==expected_config,'outbox config mismatch')
  row=db.execute('SELECT *FROM emissions WHERE id=?',(identity,)).fetchone();need(row is not None,'unknown emission')
  body=json.loads(row['body']);need(body['batch']['campaign_sha256']==expected_campaign,'emission campaign mismatch')
  e=body['emission'];need(sha(e['raw_program_utf8'].encode())==e['raw_program_sha256'],'emission program digest')
  result={'source_outbox':str(p),'head':head,'config':conf,'emission_id':identity,'body':body,
   'state':row['state'],'lease':None if row['lease']is None else json.loads(row['lease']),
   'outcome':None if row['outcome']is None else json.loads(row['outcome'])}
  db.execute('COMMIT');return result
 finally:
  if db.in_transaction:db.execute('ROLLBACK')
  db.close()

class CustodyStore:
 def __init__(self,client,deadline):self.client,self.deadline=client,deadline
 def get(self,ref):
  tick(self.deadline);need(ref['bucket']==BUCKET and ref['key'].startswith(PREFIX),'custody namespace')
  need(type(ref['bytes'])is int and 0<ref['bytes']<=MAX_BYTES and hash64(ref['sha256']),'bounded custody SHA')
  need(type(ref['version_id'])is str and ref['version_id']not in('','null'),'exact version required')
  r=self.client.get_object(Bucket=BUCKET,Key=ref['key'],VersionId=ref['version_id'])
  need(r.get('VersionId')==ref['version_id']and r.get('ContentLength')==ref['bytes'],'exact version/size')
  out=bytearray()
  with r['Body']as stream:
   while True:
    tick(self.deadline);part=stream.read(min(1024**2,ref['bytes']-len(out)+1))
    if not part:break
    out.extend(part);need(len(out)<=ref['bytes'],'custody body bound')
  tick(self.deadline);need(sha(out)==ref['sha256']and len(out)==ref['bytes'],'custody exact SHA');return bytes(out)
 def put(self,key,body):
  tick(self.deadline);need(key.startswith(PREFIX)and 0<len(body)<=MAX_BYTES,'bounded own custody')
  try:r=self.client.put_object(Bucket=BUCKET,Key=key,Body=body,ContentLength=len(body),IfNoneMatch='*',Metadata={'sha256':sha(body)})
  except Exception as exc:
   tick(self.deadline)
   if code(exc)not in('412','PreconditionFailed'):raise
   r=self.client.head_object(Bucket=BUCKET,Key=key)
  ref={'bucket':BUCKET,'key':key,'version_id':r.get('VersionId'),'sha256':sha(body),'bytes':len(body)}
  need(self.get(ref)==body,'custody readback');return ref

def publication(snapshot,diagnostic,error,source,store,out):
 need(snapshot['state']in('leased','unknown_attempt','pending_unsupported'),'only unresolved attempts custody')
 need(type(error)is str and 0<len(error)<=4000,'explicit failure')
 if diagnostic is not None:
  need(diagnostic['schema']=='jope.native-rust-parity-rejection/v1','typed parity failure')
  recomputed=parity_diagnostic(diagnostic['rust_predictions'],diagnostic['native_predictions'])
  need(all(diagnostic[k]==v for k,v in recomputed.items()),'exact unchanged parity diagnostics')
  need(diagnostic['program_sha256']==snapshot['body']['emission']['raw_program_sha256'] and diagnostic['context']['identity']==snapshot['body']['emission']['context_id'],'diagnostic exact emission')
 document={'schema':'jope.native-program-failure-custody/v1','snapshot':snapshot,'diagnostic':diagnostic,
  'error':error,'source':source,'archived_utc_time_ns':time.time_ns(),'archived_monotonic_ns':time.monotonic_ns(),
  'numeric_score':None,'canonical_score_rows':[],'teaching_eligible':False,'optimizer_reward_allowed':False,
  'source_state_changed':False,'unknown_attempt_retried':False}
 body=canonical(document);need(len(body)<=MAX_BYTES,'bounded failure document');out=Path(out);out.mkdir(exist_ok=False)
 create(out/'snapshot.json',body);snapshot_ref=store.put(PREFIX+'snapshots/'+sha(body)+'.json',body)
 receipt={'schema':'jope.native-program-failure-publication/v1','snapshot':snapshot_ref,'emission_id':snapshot['emission_id'],
  'emission_body_sha256':digest(snapshot['body']),'source_outbox':snapshot['source_outbox'],'source_head':snapshot['head'],
  'kind':'fixed_prediction_parity_rejection'if diagnostic else'unknown_or_infrastructure_failure',
  'numeric_score':None,'canonical_score_rows':[],'teaching_eligible':False,'optimizer_reward_allowed':False}
 receipt_body=canonical(receipt);create(out/'receipt.json',receipt_body)
 ref=store.put(PREFIX+'receipts/'+sha(receipt_body)+'.json',receipt_body)
 need(verify_publication(store,ref,snapshot['body'],snapshot['emission_id'])==receipt,'custody closure')
 create(out/'publication.json',canonical({'receipt':ref,'snapshot':snapshot_ref}));return ref

def verify_publication(store,ref,emission,identity):
 receipt=json.loads(store.get(ref));need(receipt['schema']=='jope.native-program-failure-publication/v1'and receipt['emission_id']==identity and receipt['emission_body_sha256']==digest(emission),'failure receipt exact emission')
 doc=json.loads(store.get(receipt['snapshot']));need(doc['snapshot']['body']==emission and doc['snapshot']['emission_id']==identity,'failure snapshot exact emission')
 for value in (doc,receipt):need(value['numeric_score']is None and value['canonical_score_rows']==[]and value['teaching_eligible']is False and value['optimizer_reward_allowed']is False,'nonnumeric ineligible failure')
 return receipt

def finish_failure(outbox,lease,body,exc,config,config_sha,store,out):
 """Return (row, continue_other_jobs). Only typed fixed parity may continue.

 Publication/lease errors pause this worker and never turn a missing artifact
 into a completed score. Old verifier event grammar remains unchanged.
 """
 error=type(exc).__name__+': '+str(exc)[:3000];known=isinstance(exc,ParityRejected)
 row={'state':'unknown_attempt','error':error,'observed_utc_time_ns':time.time_ns(),'observed_monotonic_ns':time.monotonic_ns()};ref=None
 try:
  snapshot=anchored_emission(config['outbox_path'],lease['emission_id'],config['outbox_config'],config['training_config']['sha256'])
  need(snapshot['state']=='leased'and snapshot['lease']==lease and snapshot['body']==body,'exact live attempt snapshot')
  source={'config_sha256':config_sha,'source_pins':config['source_pins'],'native_build_receipt':config['native_build_receipt'],
   'worker':config['worker'],'emitter':config['emitter'],'rustc':config['rustc'],'host':os.uname().nodename,'pid':os.getpid(),
   'observed_utc_time_ns':row['observed_utc_time_ns'],'observed_monotonic_ns':row['observed_monotonic_ns'],
   'classification':'Native/Rust implementation disagreement, not GP infeasibility.'if known else'Unknown or infrastructure error; worker paused.'}
  ref=publication(snapshot,exc.diagnostic if known else None,error,source,store,out);row['attempt_custody_ref']=ref
  outcome={'state':'failure_custody_verified'if known else'unknown_attempt','evaluation_key':None,'record_ref':None,'attempt_custody_ref':ref,'error':error}
  def verify_failure(b,o):
   need(b==body,'unchanged emission');verify_publication(store,o['attempt_custody_ref'],b,lease['emission_id']);return True
  outbox.finish(lease,outcome,verify_outcome=verify_failure);row['state']=outcome['state']
 except BaseException as custody_error:
  row['custody_error']=type(custody_error).__name__+': '+str(custody_error)[:1000];known=False
  try:outbox.finish(lease,{'state':'unknown_attempt','evaluation_key':None,'record_ref':None,'attempt_custody_ref':ref,'error':error},verify_outcome=lambda b,o:b==body and o['state']=='unknown_attempt')
  except BaseException as finish_error:row['finish_error']=str(finish_error)[:1000]
 return row,known
