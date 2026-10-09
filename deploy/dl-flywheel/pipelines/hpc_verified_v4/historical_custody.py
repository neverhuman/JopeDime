"""Read-only old unknown failures -> nonnumeric S3 custody; no GP reexecution."""
import argparse,json,os,time
from pathlib import Path
from native_inputs import bound,need,canonical,sha
from native_dispatch import preflight
from native_failure import anchored_emission,parity_diagnostic,CustodyStore,publication,PREFIX
from jope_flywheel.program_worker import split_scale
from jope_flywheel.program_io import bind_campaign_deadline,tick,create
from jope_flywheel.s3_library import client,BUCKET

def reconstruct(item,body,inputs):
 m=item['matches'][0];source=json.loads(bound(m['source_config']))
 for r in source['source_pins']:bound(r,64*1024**2)
 for k in('worker','emitter','rustc','native_canonicalizer'):bound(source[k],64*1024**2)
 outcome=json.loads(bound(m['outcome']));need(outcome['emission_id']==item['emission_id']and outcome['error']=='ValueError: emitted/native prediction parity failed; tolerance is fixed','exact prior parity failure')
 e=body['emission'];context,raw,x,y,folds,info=inputs.describe(e['input_descriptor'],e['phase'])
 need(raw==bound(m['input']),'exact original masked input')
 candidate=json.loads(bound(m['candidate']));need(candidate['dsl']==e['raw_program_utf8']and candidate['program_sha256']==e['raw_program_sha256'],'original failed program')
 evaluation=json.loads(bound(m['evaluation']));need(evaluation['program']==candidate and evaluation['fold_protocol']=='jope.s1-two-fold/v1','original evaluation identity')
 for k in('rust_source','rust_executable'):bound(m[k],64*1024**2)
 diagnostics=[]
 for r in m['prediction_files']:
  i=int(Path(r['path']).name[4]);need(i in(0,1),'fold index')
  split=split_scale(x,y,folds,i);f=evaluation['inner_cv'][i]
  need(f['index']==i and f['targets']==split['targets']and f['scaler_sha256']==split['scaler_sha256']and f['serialized_prediction_replay_exact']is True,'original native fit/replay closure')
  values=[float(z)for z in bound(r).decode().splitlines()];actual=[v*split['yspan']+split['ylo']for v in values]
  try:d=parity_diagnostic(actual,f['predictions'])
  except ValueError as error:
   if str(error)=='diagnostic requires actual fixed-tolerance rejection':continue
   raise
  d.update({'fold':i,'scaler_sha256':split['scaler_sha256'],'scaled_predictions_sha256':r['sha256'],
   'evaluation_sha256':m['evaluation']['sha256'],'rust_source_sha256':m['rust_source']['sha256'],
   'rust_executable_sha256':m['rust_executable']['sha256'],'input_sha256':sha(raw),
   'program_sha256':candidate['program_sha256'],'context':context.__dict__,
   'source_commit':source['source_commit'],'worker':source['worker'],'emitter':source['emitter'],'rustc':source['rustc']})
  diagnostics.append(d)
 need(len(diagnostics)==1,'one captured rejecting fold; no invented later fold')
 events=[]
 for line in bound(m['timing'],64*1024**2).splitlines():
  event=json.loads(line)
  if event.get('emission_id')==item['emission_id']or event.get('context',{}).get('emission_id')==item['emission_id']:events.append(event)
 need(events,'original timestamped attempt events')
 source_proof={'original_source_config':m['source_config'],'original_outcome':m['outcome'],'original_timing':m['timing'],
  'original_timing_events':events,'original_artifact_refs':m,'reconstructed_at_utc_time_ns':time.time_ns(),
  'reconstructed_at_monotonic_ns':time.monotonic_ns(),'evaluation_reexecuted':False,
  'classification':'Native/Rust implementation disagreement; no GP feasibility label inferred.'}
 return diagnostics[0],source_proof

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--config-sha',required=True);ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args()
 raw=Path(a.config).read_bytes();need(sha(raw)==a.config_sha,'custody config');c=json.loads(raw)
 bind_campaign_deadline(c['deadline_unix']);tick(c['deadline_unix'])
 need(os.uname().nodename=='xbabe2'and Path(c['out'])==Path(__file__).resolve().parent and c['deadline_unix']<=c['global_deadline_unix'],'owned custody host/root/deadline')
 if not a.preflight_only:
  cg=next(x.split(':',2)[2]for x in Path('/proc/self/cgroup').read_text().splitlines()if x.startswith('0::'));g=Path('/sys/fs/cgroup')/cg.lstrip('/');q,p=(g/'cpu.max').read_text().split()
  need(q!='max'and int(q)<=int(p)and int((g/'memory.max').read_text())<=512*1024**2 and int((g/'memory.swap.max').read_text())==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','historical CPU1/512MiB/swap0/noGPU')
 for r in c['source_pins']:bound(r,64*1024**2)
 original=json.loads(bound(c['original_config']));inputs,proof=preflight(original)
 inventory=json.loads(bound(c['inventory']));need(len(inventory['rows'])==6 and all(len(x['matches'])==1 for x in inventory['rows']),'exact six historical failures')
 if a.preflight_only:
  for item in inventory['rows']:
   snap=anchored_emission(original['outbox_path'],item['emission_id'],original['outbox_config'],original['training_config']['sha256'])
   need(snap['state']=='unknown_attempt','original unknown unchanged');reconstruct(item,snap['body'],inputs)
  print(json.dumps({'state':'preflight_pass','historical_failures':6,'native_executions':0,'SQL_writes':0,'S3_requests':0}));return
 root=Path(c['out']);create(root/'put-attempts.jsonl',b'');rawclient=client();counts={'puts':0,'gets':0,'heads':0}
 class Budget:
  def get_object(self,**kw):
   tick(c['deadline_unix']);need(counts['gets']<36,'GET cap');counts['gets']+=1;return rawclient.get_object(**kw)
  def head_object(self,**kw):
   tick(c['deadline_unix']);need(counts['heads']<12,'HEAD cap');counts['heads']+=1;return rawclient.head_object(**kw)
  def put_object(self,**kw):
   tick(c['deadline_unix']);need(counts['puts']<12 and kw['Bucket']==BUCKET and kw['Key'].startswith(PREFIX)and kw.get('IfNoneMatch')=='*','12 create-only attempts')
   counts['puts']+=1
   with(root/'put-attempts.jsonl').open('ab')as f:f.write(canonical({'at':time.time(),'key':kw['Key'],'sha256':kw['Metadata']['sha256'],'bytes':kw['ContentLength']})+b'\n');f.flush();os.fsync(f.fileno())
   return rawclient.put_object(**kw)
 store=CustodyStore(Budget(),c['deadline_unix']);results=[]
 for item in inventory['rows']:
  tick(c['deadline_unix']);snap=anchored_emission(original['outbox_path'],item['emission_id'],original['outbox_config'],original['training_config']['sha256'])
  need(snap['state']=='unknown_attempt'and sha(canonical(snap['body']))==item['body_sha256'],'old unresolved row unchanged')
  diagnostic,source=reconstruct(item,snap['body'],inputs)
  ref=publication(snap,diagnostic,snap['outcome']['error'],source,store,root/item['emission_id'])
  after=anchored_emission(original['outbox_path'],item['emission_id'],original['outbox_config'],original['training_config']['sha256'])
  need(all(after[k]==snap[k]for k in('body','state','lease','outcome')),'source failed row unchanged')
  results.append({'emission_id':item['emission_id'],'receipt':ref,'source_state':'unknown_attempt','source_mutated':False})
 tick(c['deadline_unix']);create(root/'result.json',canonical({'schema':'jope.native-historical-failure-custody/v1','state':'completed','config_sha256':a.config_sha,'rows':results,'counts':counts,'executions':0,'numeric_failure_scores':0,'source_states_changed':0}));print(json.dumps(counts))
if __name__=='__main__':main()
