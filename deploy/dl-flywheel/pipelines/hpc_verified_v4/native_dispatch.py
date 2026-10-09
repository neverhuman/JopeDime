"""Drain mandatory original-context emissions with native measured S3 closure."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,sqlite3,sys,threading,time
from dataclasses import asdict
from native_inputs import NativeInputs,OriginalScoreContext,bound,sha,need,canonical,digest
from native_evaluate import NativeEvaluator,NativeLibrary,request_identity,local_entry,verified_measurement
from native_failure import ParityRejected,CustodyStore,anchored_emission,publication,verify_publication,finish_failure,PREFIX as FAILURE_PREFIX
from neural_outbox import Outbox
from jope_flywheel.program_io import NativeTools,bind_campaign_deadline,create,tick,remaining
from jope_flywheel.s3_library import client,BUCKET,PREFIX
import native_corpus_authority
from jope_flywheel import timing
from verified_cache import CachedS3Library,CachedNativeTools
class CachedNativeLibrary(NativeLibrary,CachedS3Library):pass

def progress(root,value):
 tmp=root/'progress.tmp'
 with tmp.open('xb')as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())
 os.replace(tmp,root/'progress.json')

def read_outbox(c):
 con=sqlite3.connect(Path(c['outbox_path']).as_uri()+'?mode=ro',uri=True,isolation_level=None);con.row_factory=sqlite3.Row
 try:
  checker=Outbox.__new__(Outbox);checker.db=con;checker.verify()
  conf=json.loads(con.execute('SELECT body FROM config').fetchone()[0]);need(conf==c['outbox_config'],'exact outbox authority')
  rows=[json.loads(x[0])for x in con.execute('SELECT body FROM emissions')];contexts={}
  for row in rows:
   need(row['batch']['campaign_sha256']==c['training_config']['sha256'],'training origin');e=row['emission'];cid=e['context_id'];d=e['input_descriptor']
   need(cid==d['context_id']and e['parent_lineage_sha256']==d['parent_lineage_sha256'],'emission source')
   if cid in contexts:need(contexts[cid]==d,'conflicting native descriptor')
   if d.get('source_kind')=='native_s1_context':contexts[cid]=d
  return contexts
 finally:con.close()

def unsupported_count(path):
 con=sqlite3.connect(Path(path).as_uri()+'?mode=ro',uri=True)
 try:return con.execute("SELECT count(*)FROM emissions WHERE state='recorded'AND json_extract(body,'$.emission.input_descriptor.source_kind')!='native_s1_context'").fetchone()[0]
 finally:con.close()

def claim_native(outbox,owner,*,lease_seconds,deadline):
 # Same frozen event grammar. Nonnative rows stay recorded for their proper
 # adapter; they cannot be silently consumed by this original-context worker.
 now=outbox.now();need(type(owner)is str and 0<len(owner)<=128 and type(lease_seconds)is int and 1<=lease_seconds<=600 and now+lease_seconds<=deadline,'bounded native lease')
 db=outbox.db;db.execute('BEGIN IMMEDIATE')
 try:
  row=db.execute("SELECT *FROM emissions WHERE state='recorded'AND json_extract(body,'$.emission.input_descriptor.source_kind')='native_s1_context'ORDER BY rowid LIMIT 1").fetchone()
  if row is None:db.execute('COMMIT');return None
  lease={'emission_id':row['id'],'owner':owner,'started_at':now,'expires_at':now+lease_seconds,'token':digest({'id':row['id'],'owner':owner,'at':now})}
  db.execute("UPDATE emissions SET state='leased',lease=?WHERE id=?",(canonical(lease),row['id']));outbox._event('emission.leased',lease);db.execute('COMMIT');return {'lease':lease,'body':json.loads(row['body'])}
 except BaseException:
  if db.in_transaction:db.execute('ROLLBACK')
  raise

def preflight(c):
 for r in c['source_pins']+[c['training_config'],c['snapshot'],c['native_build_receipt']]+[c[k]for k in ('worker','emitter','rustc','native_canonicalizer')]:bound(r,64*1024**2)
 build=json.loads(bound(c['native_build_receipt']));need(build['source']['pin']==c['source_commit'] and build['binaries']['worker_honest']['sha256']==c['worker']['sha256'],'native binary/source receipt')
 spec=json.loads(bound(c['snapshot']));proof=native_corpus_authority.check(spec);contexts=read_outbox(c)
 inputs=NativeInputs(c['input_lists'],contexts,authority=lambda:native_corpus_authority.check(spec),authority_pins=spec['authority_pins'])
 need(set(contexts)<=set(inputs.rows),'every emitted native context has registered source')
 con=sqlite3.connect(Path(c['outbox_path']).as_uri()+'?mode=ro',uri=True)
 rows=[json.loads(x[0])for x in con.execute("SELECT body FROM emissions WHERE state='recorded'ORDER BY rowid LIMIT 2")];con.close()
 rows=[r for r in rows if r['emission']['input_descriptor'].get('source_kind')=='native_s1_context']
 examples=[]
 for row in rows:
  e=row['emission'];context,raw,x,y,f,info=inputs.describe(e['input_descriptor'],e['phase']);examples.append({'context':asdict(context),'bytes':len(raw),'shape':list(x.shape),'phase':e['phase']})
 return inputs,{'state':'preflight_pass','registered_contexts':len(inputs.rows),'emitted_contexts':len(contexts),'examples':examples,'authority_sha256':digest(proof)}

def main():
 a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--config-sha',required=True);a.add_argument('--preflight-only',action='store_true');args=a.parse_args()
 raw=Path(args.config).read_bytes();need(sha(raw)==args.config_sha,'config');c=json.loads(raw);root=Path(c['out']);need(root==Path(__file__).resolve().parent and os.uname().nodename=='xbabe2','owned host/root')
 need(time.time()<c['deadline_unix']and c['threads']==1 and 1<=c['maximum_emissions']<=128 and c['deadline_unix']<=c['global_deadline_unix'],'fixed bounded work')
 bind_campaign_deadline(c['deadline_unix'])
 inputs,proof=preflight(c)
 if args.preflight_only:print(json.dumps(proof));return
 cg=next(x.split(':',2)[2]for x in Path('/proc/self/cgroup').read_text().splitlines()if x.startswith('0::'));group=Path('/sys/fs/cgroup')/cg.lstrip('/');quota,period=(group/'cpu.max').read_text().split()
 need(quota!='max'and int(quota)<=int(period)and int((group/'memory.max').read_text())<=1024**3 and int((group/'memory.swap.max').read_text())==0,'CPU/RAM/swap limits')
 need(os.environ.get('CUDA_VISIBLE_DEVICES')==''and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),'CPU only single BLAS')
 deadline=c['deadline_unix'];bind_campaign_deadline(deadline);timing.ACTIVE=timing.Timing(root/'stage-timing.jsonl',group,config_sha256=args.config_sha);timing.ACTIVE.event('observation','packet.start',outbox=c['outbox_path']);create(root/'preflight.json',canonical(proof));create(root/'put-attempts.jsonl',b'')
 shared=threading.Lock();outcomes=[];counts={'claimed':0,'puts':0,'executions':0,'cached':0,'published':0,'failed':0,'known_rejections':0,'failure_custody_verified':0,'unsupported':0,'unknown':0,'gets':0,'heads':0};stop=threading.Event()
 def report(state):
  progress(root,{'schema':'jope.native-original-drain-progress/v2','config_sha256':args.config_sha,'at':time.time(),'state':state,'deadline_unix':deadline,'counts':dict(counts),'outcomes':outcomes[-12:],'timing_events':timing.ACTIVE.count,'outbox_path':c['outbox_path'],'numeric_failure_scores_created':0,'unsupported_recorded':unsupported_count(c['outbox_path'])})
 def worker(index):
  outbox=Outbox(c['outbox_path'],**{k:v for k,v in c['outbox_config'].items()if k!='schema'})
  class TimedNative(CachedNativeTools):
   def call(self,mode,value):
    with timing.span('gp.canonicalizer',mode=mode):return super().call(mode,value)
  native=TimedNative(c['native_canonicalizer'],deadline);rawclient=client();wroot=root/('worker-'+str(index));wroot.mkdir(exist_ok=False)
  class Budget:
   def __getattr__(self,name):return getattr(rawclient,name)
   def head_object(self,**kw):
    tick(deadline);need(counts['heads']<c['maximum_emissions']*16,'HEAD attempt cap');counts['heads']+=1
    with timing.span('s3.HEAD',key=kw['Key']):return rawclient.head_object(**kw)
   def get_object(self,**kw):
    tick(deadline);need(counts['gets']<c['maximum_emissions']*256,'GET attempt cap');counts['gets']+=1
    with timing.span('s3.GET_headers',key=kw['Key'],version_id=kw.get('VersionId')):r=rawclient.get_object(**kw)
    body=r['Body'];key=kw['Key']
    class TimedBody:
     def __enter__(self):return self
     def __exit__(self,*args):body.close()
     def read(self,size=-1):
      with timing.span('s3.GET_body',key=key,requested_bytes=size):return body.read(size)
    r['Body']=TimedBody();return r
   def put_object(self,**kw):
    with shared:
     need(counts['puts']<c['maximum_emissions']*10 and kw.get('IfNoneMatch')=='*'and kw['Bucket']==BUCKET and (kw['Key'].startswith(PREFIX)or kw['Key'].startswith(FAILURE_PREFIX)),'create only bounded GP namespace')
     counts['puts']+=1;row={'seq':counts['puts'],'at':time.time(),'key':kw['Key'],'sha256':kw['Metadata']['sha256'],'bytes':kw['ContentLength']}
     with(root/'put-attempts.jsonl').open('ab')as f:f.write(canonical(row)+b'\n');f.flush();os.fsync(f.fileno())
    with timing.span('s3.PUT',key=kw['Key'],bytes=kw['ContentLength']):return rawclient.put_object(**kw)
  budget=Budget();store=CachedNativeLibrary(budget,deadline=deadline,nativecanonical_json=native.canonical_json);custody=CustodyStore(budget,deadline);ev=NativeEvaluator(inputs,store,native,c,deadline,args.config_sha)
  try:
   while not stop.is_set()and remaining(deadline)>610:
    with shared:
     if counts['claimed']>=c['maximum_emissions']:break
     used=sum(p.stat().st_size for p in root.rglob('*')if p.is_file())
     need(used<8*1024**3-128*1024**2,'owned output cap before new claim')
     claim=claim_native(outbox,c['worker_id'],lease_seconds=600,deadline=deadline)
     if claim is None:break
     counts['claimed']+=1;report('running')
    lease=claim['lease'];body=claim['body'];e=body['emission'];row={'emission_id':lease['emission_id'],'context_id':e['context_id'],'phase':e['phase'],'raw_program_sha256':e['raw_program_sha256']}
    try:
     timing.ACTIVE.context={'emission_id':lease['emission_id'],'context_id':e['context_id'],'phase':e['phase']}
     timing.ACTIVE.event('start','gp.total')
     descriptor=e['input_descriptor']
     need(body['batch']['campaign_sha256']==c['training_config']['sha256']and descriptor['context_id']==e['context_id']and descriptor['parent_lineage_sha256']==e['parent_lineage_sha256'],'exact growing outbox origin')
     need(descriptor.get('source_kind')=='native_s1_context','native-only claim invariant')
     if e['context_id']in inputs.contexts:need(inputs.contexts[e['context_id']]==descriptor,'changed native descriptor')
     else:inputs.contexts[e['context_id']]=descriptor
     context,raw,x,y,f,desc=inputs.describe(descriptor,e['phase']);program=native.canonical_program(e['raw_program_utf8']);key=request_identity(context,program)
     entry=store.lookup(key);cached=entry is not None
     if not cached:
      receipt=ev.execute(e['input_descriptor'],e['phase'],program,wroot/key);record=json.loads(bound(receipt));need(record['state']=='executed','actual native execution')
      with shared:counts['executions']+=1
      bodies={k:bound(r,64*1024**2)for k,r in record['artifacts'].items()};entry=local_entry(context,program,bodies)
      entry['artifacts']={k:store.put_artifact(k,v)for k,v in bodies.items()};published=store.commit(context,program,entry);recordref=published['record']
     else:
      r=store.client.head_object(Bucket=BUCKET,Key=PREFIX+key+'/record.json');recordref={'bucket':BUCKET,'key':PREFIX+key+'/record.json','version_id':r.get('VersionId'),'sha256':r.get('Metadata',{}).get('sha256'),'bytes':r['ContentLength']}
     measured=verified_measurement(store,context,program,entry);actual,version=store.get_version(recordref);need(json.loads(actual)==entry,'exact result wrapper readback')
     inputs.check_authority(e['input_descriptor'],e['phase'])
     outcome={'state':'cached_verified'if cached else'published_verified','evaluation_key':key,'record_ref':recordref,'attempt_custody_ref':None,'error':None}
     def verify(emission,outcome):
      need(emission==body and outcome['evaluation_key']==request_identity(context,program),'emission/result join')
      raw,_=store.get_version(outcome['record_ref']);verified_measurement(store,context,program,json.loads(raw));return True
     outbox.finish(lease,outcome,verify_outcome=verify);row.update(state=outcome['state'],evaluation_key=key,program_sha256=sha(program.encode()),measurement=measured,record_ref=recordref)
     with shared:counts['cached'if cached else'published']+=1
    except BaseException as exc:
     result,known=finish_failure(outbox,lease,body,exc,c,args.config_sha,custody,root/('failure-'+lease['emission_id']))
     row.update(result)
     if known:counts['known_rejections']+=1;counts['failure_custody_verified']+=1
     else:stop.set();counts['unknown']+=1
     counts['failed']+=1
    with shared:
     timing.ACTIVE.event('end','gp.total',outcome=row)
     create(root/(lease['emission_id']+'.json'),canonical(row));outcomes.append(row);report('running'if not stop.is_set()else'failed')
  finally:
   timing.ACTIVE.event('observation','cache.final',s3=store.cache_stats(),native=native.cache_stats());outbox.close()
 with concurrent.futures.ThreadPoolExecutor(max_workers=c['threads'])as pool:list(pool.map(worker,range(c['threads'])))
 with shared:report('paused_on_infrastructure_or_unknown'if counts['unknown']else'completed_bounded_packet')
 print(json.dumps(counts))
if __name__=='__main__':main()
