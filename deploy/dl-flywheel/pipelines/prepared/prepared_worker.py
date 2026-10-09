"""Bounded CPU-only acceptance; no model, optimizer, planner, queue or S3 writes."""
import hashlib,json,os,socket,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def read(p,cap=16*1024**2):
 p=Path(p);need(p.is_absolute()and all(not q.is_symlink()for q in(p,*p.parents))and p.is_file()and p.stat().st_size<=cap,'bounded safe source');b=p.read_bytes();need(len(b)<=cap,'source grew');return b
def create(p,b):
 with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
def main():
 started=time.monotonic();raw=read(ROOT/'config.json');need(len(sys.argv)==2 and sha(raw)==sys.argv[1],'explicit config SHA');c=json.loads(raw)
 need(c['owned_root']==str(ROOT)and ROOT.name=='runtime' and ROOT.parent.name=='dl-flywheel-prepared128-20261009'and socket.gethostname().split('.')[0]=='xbabe2'and os.environ.get('CUDA_VISIBLE_DEVICES')=='','owned CPU-only root')
 w=json.loads(read(ROOT/'window.json'));deadline=w['deadline_unix']-30
 need(w['config_sha256']==sha(raw)and 0<w['deadline_unix']-w['created_unix']<=360 and time.time()<deadline<=1791592146.380285,'fixed bounded acceptance window')
 mono=time.monotonic()+deadline-time.time()
 cg=Path('/sys/fs/cgroup')/next(x[3:]for x in Path('/proc/self/cgroup').read_text().splitlines()if x.startswith('0::')).lstrip('/');quota,period=(cg/'cpu.max').read_text().split()
 need(quota!='max'and int(quota)<=2*int(period)and int((cg/'memory.max').read_text())<=2*1024**3 and int((cg/'memory.swap.max').read_text())==0 and int((cg/'pids.max').read_text())<=32,'CPU2 RAM2GiB swap0 tasks32')
 for name,pin in c['owned_sources'].items():need('/'not in name and sha(read(ROOT/name))==pin,'own exact source:'+name)
 origin=Path(c['origin']['path']);ob=read(origin/'config.json');need(sha(ob)==c['origin']['config_sha256'],'immutable origin config');old=json.loads(ob)
 for rel,pin in old['owned_sources'].items():
  p=Path(rel);need(not p.is_absolute()and'..'not in p.parts and sha(read(origin/p))==pin,'frozen qualified source:'+rel)
 sys.path[:0]=[str(ROOT),str(origin/'consumer'),str(origin/'runtime'),str(origin)]
 from exact_artifact_cache import canonical,identity
 import prepared_snapshot as P
 import resource
 from verified_cache import CachedNativeTools
 from jope_flywheel.program_io import bound,tick,bind_campaign_deadline
 from jope_flywheel import training_snapshot_native as N,native_variant_snapshot as V
 import native_corpus_authority as authority
 bind_campaign_deadline(deadline);bind_campaign_deadline(time.time()+mono-time.monotonic())
 need(sha(read(Path(V.__file__)))=='68c20ee69226e7be9bf92ad40e3889c91fddd4af337c65c9d838d8f51d875258','unchanged current-compatible adapter')
 ext=read(ROOT/'extension.json');need(sha(ext)==c['extension_sha256']and len(json.loads(ext)['tasks'])==128,'exact measured128 extension')
 rb=read(ROOT/'artifact-refs.json');need(sha(rb)==c['artifact_refs_sha256'],'exact object inventory');refs=json.loads(rb);need(len(refs)==6053 and sum(r['bytes']for r in refs)==4714394119,'measured artifact inventory')
 specraw=bound(old['snapshot']);spec=json.loads(specraw)
 def fresh():
  tick(deadline);body=canonical(authority.check(spec));p=json.loads(body);need(p.get('passed')is True and 0<=time.time()-p['checked_at_unix']<60,'actual current-role proof');return body
 native=N.load_snapshot(specraw,old['snapshot']['sha256'],N.LocalStore(old['read_roots']),authority_bytes=fresh())
 tools=CachedNativeTools(old['native_canonicalizer'],deadline)
 accepted=Path(c['accepted_root']);accepted_status=json.loads(bound(c['accepted_status']))
 need(accepted_status['state']=='passed'and accepted_status['extension_sha256']==c['extension_sha256']and accepted_status['source_adapter_sha256']==sha(read(Path(V.__file__)))and accepted_status['optimizer_updates']==accepted_status['planner_mutations']==accepted_status['s3_writes']==0,'qualified exact prior artifact acceptance')
 ac=accepted/'artifacts';manifestraw=read(ac/'manifest.json');complete=json.loads(read(ac/'complete.json'));manifest=json.loads(manifestraw)
 need(sha(manifestraw)==accepted_status['artifact_network_proof']['manifest_sha256']==complete['manifest_sha256']and complete['state']=='complete'and complete['objects']==6053 and complete['bytes']==4714394119,'prior closed exact-version cache')
 declared={identity(r)[1]:identity(r)[0]for r in refs};need({identity(r)[1]:identity(r)[0]for r in manifest['refs']}==declared,'exact accepted cache roster')
 class ClosedCache:
  nativecanonical_json=tools.canonical_json
  counters={'verified_local_reads':0,'verified_local_bytes':0,'verified_local_seconds':0.,'network_gets':0,'s3_writes':0}
  def get_version(self,source):
   tick(deadline);r,k=identity(source);need(declared.get(k)==r,'exact cached ref only');start=time.monotonic();body=read(ac/(k+'.bin'),32*1024**2);need(len(body)==r['bytes']and sha(body)==r['sha256'],'exact accepted artifact bytes unchanged');tick(deadline)
   self.counters['verified_local_reads']+=1;self.counters['verified_local_bytes']+=len(body);self.counters['verified_local_seconds']+=time.monotonic()-start;return body,r['version_id']
 cache=ClosedCache();prefetch={'schema':'jope.prepared-prior-cache/v1','accepted_status':c['accepted_status'],'manifest_sha256':sha(manifestraw),'objects':6053,'bytes':4714394119,'network_gets':0,'s3_writes':0}
 create(ROOT/'prefetch.json',canonical(prefetch));fresh()
 loaded=V.compose(native,ext,c['extension_sha256'],cache,native_spec_bytes=specraw,registry_bytes=bound(old['parent_registry']),preprocessing_bytes=bound(old['preprocessing']),authority_callback=fresh,deadline=deadline)
 need(loaded.receipt['counts']['variants_loaded']==128 and loaded.receipt['counts']['contexts']==7248 and loaded.receipt['counts']['partitions']=={'train':6982,'validation':266},'all128 and exact inherited roles')
 rows={cid:[{k:getattr(r,k)for k in('program_sha256','outcome','q','q_opt','q_grid','feasibility','runtime_seconds','runtime_censored','ce_eligible','quality_eligible','pairwise_eligible','runtime_eligible','source_kind')}for r in rs]for cid,rs in loaded.variant_records.items()}
 need(sum(map(len,rows.values()))==1027 and all(r['ce_eligible']is False for rs in rows.values()for r in rs),'full measured support without imitation teachers')
 for name,body in [('loaded-receipt.json',loaded.receipt),('batch-manifest.json',loaded.batch_manifest),('records.json',rows),('final-authority.json',json.loads(fresh()))]:create(ROOT/name,canonical(body))
 prepared_started=time.monotonic_ns();cpu0=resource.getrusage(resource.RUSAGE_SELF);body=P.prepare(loaded,ext);create(ROOT/'prepared.json',body)
 prepared_ref={'path':str(ROOT/'prepared.json'),'sha256':sha(body),'bytes':len(body)}
 probe={'schema':'jope.prepared-snapshot-receipt/v1','state':'passed','prepared':prepared_ref,'optimizer_updates':0,'planner_mutations':0,'s3_writes':0}
 hydrate_started=time.monotonic_ns()
 def current(extension):
  need(extension==ext,'same exact extension');V.check_authority(json.loads(extension),spec,fresh(),{p['parent_lineage_sha256']:p for p in map(N.document,bound(old['parent_registry']).splitlines())},time.time())
 hydrated=P.load(body,native,receipt=probe,expected_extension_sha256=c['extension_sha256'],expected_preprocessing_sha256=old['preprocessing']['sha256'],expected_parent_registry_sha256=old['parent_registry']['sha256'],authority_check=current)
 hydration_ns=time.monotonic_ns()-hydrate_started
 need(hydrated.contexts==loaded.contexts and hydrated.variant_records==loaded.variant_records and hydrated.best_observed==loaded.best_observed and hydrated.proposal_programs==loaded.proposal_programs and hydrated.batch_manifest==loaded.batch_manifest and hydrated.receipt==loaded.receipt,'complete exact materialization roundtrip')
 for cid,z in loaded.variant_vectors.items():need(__import__('numpy').array_equal(hydrated.variant_vectors[cid],z),'bit exact encoder roundtrip')
 cpu1=resource.getrusage(resource.RUSAGE_SELF)
 tick(deadline);status={'schema':'jope.prepared-snapshot-receipt/v1','state':'passed','at':time.time(),'observed_utc_time_ns':time.time_ns(),'observed_monotonic_ns':time.monotonic_ns(),'seconds':time.monotonic()-started,'config_sha256':sha(raw),'extension_sha256':c['extension_sha256'],'origin_config_sha256':c['origin']['config_sha256'],'source_adapter_sha256':sha(read(Path(V.__file__))),'prepared_source_sha256':sha(read(ROOT/'prepared_snapshot.py')),'counts':loaded.receipt['counts'],'program_records':1027,'variant_ce_teachers':0,'prepared':prepared_ref,'hydration_monotonic_ns':hydration_ns,'prepare_and_roundtrip_monotonic_ns':time.monotonic_ns()-prepared_started,'prepare_and_roundtrip_cpu_user_seconds':cpu1.ru_utime-cpu0.ru_utime,'prepare_and_roundtrip_cpu_system_seconds':cpu1.ru_stime-cpu0.ru_stime,'verified_local_cache':cache.counters,'native_canonical_cache':tools.cache_stats(),'optimizer_updates':0,'planner_mutations':0,'admission_committed':False,'s3_writes':0,'network_gets':0,'memory_peak_bytes':int((cg/'memory.peak').read_text()),'scope':'CPU preparation only. Active trainer unchanged; no inbox/ready pointer. Requires reviewed future consumer and reconciled checkpoint/control/planner/outbox ancestry before epoch-boundary adoption.'}
 tick(deadline);create(ROOT/'status.json',canonical(status));print(json.dumps(status))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  try:create(ROOT/'failure.json',json.dumps({'schema':'jope.prepared-snapshot-receipt/v1','state':'failed','at':time.time(),'error':type(error).__name__+': '+str(error)[:3000],'automatic_retry':False,'optimizer_updates':0,'planner_mutations':0,'admission_committed':False},sort_keys=True).encode())
  except BaseException:pass
  raise
