"""Qualified fresh-context CPU worker; unchanged native fit/parity/score schema."""
import json,os,resource,time
from pathlib import Path
from common import *
from epoch import execution_config
from history_collect import CapturedHistory
from global_attempts import Attempts,ownership
from execution_boundary import ExecutionBoundary
from staged_inputs import StagedInputs
from jope_flywheel.program_io import NativeTools,bind_campaign_deadline,tick
from jope_flywheel.program_observer import ProgramObserver
from jope_flywheel.program_worker import ProgramWorker
from jope_flywheel.program_gate import verified_measurement
from jope_flywheel.queue import Queue
from jope_flywheel.s3_library import S3Library,client as s3client,BUCKET as RESULT_BUCKET,PREFIX as RESULT_PREFIX
import label_campaign as campaign
import campaign_custody as custody

def proof(epoch):
    r=json.loads(read(Path(epoch['relay_root'])/'ownership/current.json',4096));p=Path(r['path'])
    need(p.parent==Path(epoch['relay_root'])/'ownership/proofs'and p.name==r['sha256']+'.json','owned exact ownership proof')
    return obj(r,2*MIB)

def envelope(cgroup_root=Path('/sys/fs/cgroup')):
    rel=next(x.split(':',2)[2]for x in Path('/proc/self/cgroup').read_text().splitlines()if x.startswith('0::'))
    p=cgroup_root/rel.lstrip('/');cpu=(p/'cpu.max').read_text().split()
    need(cpu[0]!='max'and int(cpu[0])/int(cpu[1])<=1 and int((p/'memory.max').read_text())<=4*1024**3 and
         int((p/'memory.swap.max').read_text())==0 and int((p/'pids.max').read_text())<=64,'CPU1/4GiB/no-swap/tasks64 envelope')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==''and os.getpriority(os.PRIO_PROCESS,0)>=10 and
         all(os.environ.get(x)=='1'for x in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')),
         'CPU only and single-thread numerical runtime')
    return p

def cfg(epoch_ref,shard):
    e=obj(epoch_ref);need(e['schema']=='jope.fresh128-epoch/v2'and e['global_deadline_unix']==GLOBAL and
         e['deadline_unix']-e['created_unix']==1800 and e['work_deadline_unix']==e['deadline_unix']-300,'exact original-bounded epoch')
    need(e['created_unix']<=time.time()<e['work_deadline_unix'],'fixed live work window')
    source_closure(e['source_manifest']);c=execution_config(epoch_ref,shard);p=obj(e['prepared']);s=next(s for s in e['shards']if s['id']==shard)
    need(len(s['variant_ids'])==8 and s['planned_slots']==64 and s['max_put_attempts']==514,'exact8-context shard')
    registry=obj(p['registry']);need(len(registry['inputs'])==128,'full128 prepared registry')
    entries={r['variant_identity']:r for r in obj(p['contexts'])}
    c.update(variants=[{'identity':v,'parent_lineage_sha256':entries[v]['descriptor']['parent_lineage_sha256'],
        'partition':entries[v]['partition'],'rows':entries[v]['descriptor']['rows'],'features':entries[v]['descriptor']['features'],
        'target_sha256':entries[v]['context']['target_sha256'],'content_sha256':entries[v]['context']['dataset_sha256']}for v in s['variant_ids']],
        campaign_deadline_unix=e['deadline_unix'],proposal_provenance_sha256=digest({'epoch':epoch_ref,'fixed_programs':p['programs']}),
        source_commit=p['config_base']['source_commit'],source_build_plan_sha256=p['config_base']['source_build_plan_sha256'])
    for k in('worker','emitter','rustc','native_canonicalizer'):bound(c[k],64*MIB)
    c['maximum_actual_evaluations']=64
    return e,c,p,s,registry

def check_inputs(e,c,p,s,registry,client,native,root):
    # Only selected8 raw tasks replay per worker; full128 identities remain in
    # immutable registry/current-role proof. No borrowed source DB is writable.
    c=dict(c,owned_root=str(root));inputs=StagedInputs(c,registry,native,client,e['work_deadline_unix'])
    history=CapturedHistory(e['history_sources'],e['required_outboxes']);aliases=history.check()
    expected={v['variant_identity']:v for v in obj(p['contexts'])}
    ownership(proof(e),c,e['context_descriptors'],time.time())
    for identity in s['variant_ids']:
        start=time.monotonic_ns();ctx,raw,desc=inputs.describe(identity);want=expected[identity]
        need(ctx.identity==want['context_id']and canonical(desc)==canonical(want['descriptor']),'replayed prepared native descriptor differs')
        for program in p['programs']:
            need((ctx.identity,None)not in aliases and(ctx.identity,program['program_sha256'])not in aliases,'historical canonical/context hold')
        timing(root,'input_replayed',variant=identity,context_id=ctx.identity,bytes=len(raw),duration_ns=time.monotonic_ns()-start)
    for program in p['programs']:need(native.canonical_program(program['dsl'])==program['dsl'],'fixed native panel parser')
    return inputs,history

def preflight(epoch_ref,shard,out,client=None,*,host_check=True,envelope_check=True,native_factory=NativeTools):
    if host_check:host('xbabe3')
    if envelope_check:envelope()
    e,c,p,s,registry=cfg(epoch_ref,shard)
    if envelope_check:
        from resources import observe
        resource_proof=observe(e)
    else:resource_proof={'fixture_only':True}
    root=fresh(out);(root/'data').mkdir()
    native=native_factory(c['native_canonicalizer'],e['work_deadline_unix'])
    import numpy as np
    need(str(np.__version__)==p['runtime_versions']['numpy'],'same prepared NumPy replay runtime')
    from native_runtime import verify
    native_proof=verify(p['native_closure'],p['execution_native_root'],e['work_deadline_unix'])
    inputs,history=check_inputs(e,c,p,s,registry,client or s3client('veox-jepson-prod'),native,root)
    # Instantiate the actual worker/queue budget contract in a separate owned
    # empty preflight DB; no claim, fitting, publishing or future-root mutation.
    from jope_flywheel.program_worker import evaluation_limit
    need(evaluation_limit(c)==64 and len(campaign.schedule(c['variants'],c['programs']))==64,'actual worker scheduling budget')
    with Queue(root/'invocation.sqlite',max_jobs=128)as empty:
        ProgramWorker(empty,inputs,None,native,c,root,e['work_deadline_unix'],epoch_ref['sha256'],execution_guard=object())
        need(empty.snapshot()['total_jobs']==0,'preflight queue must remain empty')
    # Refuse namespace reuse without creating the future worker directory/queue.
    need(not Path(s['owned_root']).exists(),'one-use worker output already exists')
    r={'schema':'jope.fresh128-worker-preflight/v2','epoch_sha256':epoch_ref['sha256'],'shard':shard,
       'contexts':8,'slots':64,'native_executions':0,'queue_claims':0,'S3_PUTs':0,'checked_at_unix':time.time(),
       'source_manifest_sha256':e['source_manifest']['sha256'],'native_runtime':native_proof,'resources':resource_proof,'state':'PASS'}
    return put(root,'preflight.json',r)

def initialize_registry(epoch_ref):
    host('xbabe3');e=obj(epoch_ref);source_closure(e['source_manifest']);c=execution_config(epoch_ref)
    ownership(proof(e),c,e['context_descriptors'],time.time())
    path=Path(e['epoch_root'])/'whole-context-attempts.sqlite'
    a=Attempts(path,c,create=True)
    try:return {'path':str(path),'proof':a.verify(),'created_once':True}
    finally:a.close()

class TimedWorker(ProgramWorker):
    def evaluate(self,job):
        start=time.monotonic_ns();timing(self.root.parent,'evaluate_start',job=job['id'])
        try:return super().evaluate(job)
        finally:timing(self.root.parent,'evaluate_end',job=job['id'],duration_ns=time.monotonic_ns()-start,
                       process_maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    def publish(self,job):
        start=time.monotonic_ns();timing(self.root.parent,'publication_start',job=job['id'])
        try:return super().publish(job)
        finally:timing(self.root.parent,'publication_end',job=job['id'],duration_ns=time.monotonic_ns()-start)

def _run(epoch_ref,shard,preflight_ref,client=None):
    host('xbabe3');group=envelope();e,c,p,s,registry=cfg(epoch_ref,shard)
    from resources import observe
    observe(e)
    pre=obj(preflight_ref);need(pre['schema']=='jope.fresh128-worker-preflight/v2'and pre['state']=='PASS'and
        pre['epoch_sha256']==epoch_ref['sha256']and pre['shard']==shard and 0<=time.time()-pre['checked_at_unix']<120,
        'fresh exact CPU-only preflight required')
    root=fresh(s['owned_root']);(root/'data').mkdir();(root/'tmp').mkdir()
    need(os.environ.get('TMPDIR')==str(root/'tmp'),'owned native temporary directory')
    config_ref=put(root,'config.json',c);create(root/'data/identity.json',canonical({'config_sha256':config_ref['sha256']}))
    clock=Clock(e['deadline_unix'],1800);bind_campaign_deadline(e['work_deadline_unix']);native=NativeTools(c['native_canonicalizer'],e['work_deadline_unix'])
    budget=campaign.PutBudget(client or s3client('veox-jepson-prod'),root,clock,maximum=514)
    inputs,history=check_inputs(e,c,p,s,registry,budget,native,root);budget.authority=inputs.lease
    global_config=execution_config(epoch_ref)
    attempts=Attempts(Path(e['epoch_root'])/'whole-context-attempts.sqlite',global_config)
    attempts.reserve_contexts(e['context_descriptors'],proof(e),shard)
    class Store(S3Library):
        def __init__(self):super().__init__(budget,deadline=e['work_deadline_unix'],nativecanonical_json=native.canonical_json);self.record_refs={}
        def lookup(self,key):
            entry=super().lookup(key)
            if entry is not None:
                response=budget.head_object(Bucket=RESULT_BUCKET,Key=RESULT_PREFIX+key+'/record.json')
                r={'bucket':RESULT_BUCKET,'key':RESULT_PREFIX+key+'/record.json','version_id':response.get('VersionId'),
                   'bytes':response['ContentLength'],'sha256':response.get('Metadata',{}).get('sha256')}
                need(r['bytes']<=2*MIB and json.loads(self.get_version(r)[0])==entry,'exact versioned record custody')
                self.record_refs[key]=r
            return entry
    store=Store();final=None;fatal=None
    with Queue(root/'data/queue.sqlite',max_jobs=128)as queue:
        guard=ExecutionBoundary(global_config,queue,inputs,attempts,history,shard,lambda:proof(e))
        observer=ProgramObserver(queue,inputs,store,plan_sha256=config_ref['sha256'],canonicalize=native.canonical_program)
        worker=TimedWorker(queue,inputs,store,native,c,root/'data',e['work_deadline_unix'],config_ref['sha256'],execution_guard=guard)
        try:
            final=campaign.run_schedule(c,queue,observer,worker,inputs,root,e['work_deadline_unix'],config_ref['sha256'],
                progress_extra=lambda:{'epoch_sha256':epoch_ref['sha256'],'shard':shard,'put_attempts':budget.attempts,
                    'memory_current_bytes':int((group/'memory.current').read_text()),'whole_context_registry':attempts.verify()})
            # Successful native results become completed only after the exact
            # same store verifier and VersionId closure used by the worker.
            for row in final['outcomes']:
                if row.get('state')!='ready':continue
                ctx,_,_=inputs.describe(row['variant_identity']);program=next(v['dsl']for v in c['programs']if v['program_sha256']==row['program_sha256'])
                entry=store.lookup(row['evaluation_key'])
                need(entry is not None,'finished slot missing verified S3 record')
                measured=verified_measurement(store,ctx,program,entry)
                found=attempts.db.execute('SELECT state FROM attempts WHERE evaluation_key=?',(row['evaluation_key'],)).fetchone()
                if found is not None and found['state']=='started':
                    attempts.complete(ctx,program,shard,store.record_refs[row['evaluation_key']],
                        lambda context,prog,receipt:verified_measurement(store,context,prog,store.lookup(row['evaluation_key']))==measured)
        except BaseException as exc:
            fatal=type(exc).__name__+': '+str(exc)[:1500];put(root,'campaign-failure.json',{'error':fatal,'at':time.time(),'automatic_retry':False})
        anchor=put(root,'anchor.json',{'schema':'jope.flywheel-program-label-anchor/v1','config_sha256':config_ref['sha256'],
                                     'queue':queue.verify(),'observations':observer.journal.verify(),'at':time.time()})
        desc={r['context_id']:r['descriptor_sha']for r in queue.db.execute('SELECT *FROM program_inputs')}
    pool=put(root,'scored-pool.json',{'schema':'jope.flywheel-scored-program-pool/v1','epoch_sha256':epoch_ref['sha256'],
       'config_sha256':config_ref['sha256'],'all_planned_slots':64,'entries':[
        {**{k:r[k]for k in('evaluation_key','variant_identity','context_id','partition','phase','program_sha256','program_name','measurement')},
         'record':store.record_refs[r['evaluation_key']]}for r in(final['outcomes']if final else[])if r.get('state')=='ready'],
       'teaching_eligible':False,'selection_uses_validation':False,'model_generated':False,'dl_win_rate':None})
    registry_proof=attempts.verify();attempts.close()
    archive_cfg={'source_config':config_ref,'registry':p['registry'],'anchor':anchor,'descriptor_pins':desc}
    archive_ref=put(root,'archive-config.json',archive_cfg);out=fresh(root/'attempt-custody')
    try:
        body,snap=custody.freeze(archive_cfg,clock);archive=custody.publish(body,snap,archive_ref['sha256'],
             hashlib.sha256(Path(custody.__file__).read_bytes()).hexdigest(),out,custody.Store(budget,clock))
        result={'schema':'jope.fresh128-shard-status/v2','state':'complete'if final and not final['not_started']and not final['counts']['failed_slots']else'partial',
                'epoch_sha256':epoch_ref['sha256'],'shard':shard,'config_sha256':config_ref['sha256'],'counts':final['counts']if final else snap['counts'],
                'fatal_error':fatal,'pool':pool,'attempt_custody':archive,'registry':registry_proof,'put_attempts':budget.attempts,
                'memory_peak_bytes':int((group/'memory.peak').read_text()),'teaching_eligible':False,'optimizer_reward_allowed':False}
        return put(root,'status.json',result)
    except BaseException as exc:
        put(root,'status.json',{'schema':'jope.fresh128-shard-status/v2','state':'custody_failed','epoch_sha256':epoch_ref['sha256'],
            'shard':shard,'error':type(exc).__name__+': '+str(exc),'source_attempts_retained':True,'automatic_retry':False});raise


def run(epoch_ref,shard,preflight_ref,client=None):
    try:return _run(epoch_ref,shard,preflight_ref,client)
    except BaseException as exc:
        # Early input/namespace/registry failures are retained even before the
        # worker has created a queue. Never overwrite a later custody status.
        e=obj(epoch_ref);s=next(x for x in e['shards']if x['id']==shard);root=Path(s['owned_root'])
        if root.is_dir()and not root.is_symlink()and not(root/'early-failure.json').exists():
            put(root,'early-failure.json',{'schema':'jope.fresh128-early-failure/v2','epoch_sha256':epoch_ref['sha256'],
                'shard':shard,'error':type(exc).__name__+': '+str(exc)[:2000],'automatic_retry':False,'at':time.time()})
        raise
