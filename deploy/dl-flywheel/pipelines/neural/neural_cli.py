"""Owned finite neural qualification with mandatory measured-program custody."""
import hashlib,json,os,socket,sys,time
from pathlib import Path
sys.dont_write_bytecode=True

def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(x):return hashlib.sha256(x).hexdigest()
def read(path,cap=32*1024**2):
    path=Path(path);need(path.is_absolute()and '..'not in path.parts and all(not x.is_symlink()for x in(path,*path.parents)),'absolute non-symlink input')
    need(0<path.stat().st_size<=cap,'file size cap')
    with path.open('rb')as f:body=f.read(cap+1)
    need(len(body)<=cap,'bounded source read');return body

def main():
    root=Path(__file__).resolve().parent;raw=read(root/'config.json');c=json.loads(raw);config_sha=sha(raw)
    window=json.loads(read(root/'window.json'));global_deadline=window['deadline_unix'];monotonic=time.monotonic()+max(0,global_deadline-time.time())
    need(c['neural']['max_batches']==32 and c['neural']['batch_size']==2 and c['maximum_actual_evaluations']==96, 'frozen count caps')
    need(c['schema']=='jope.flywheel-neural-continuation-campaign/v1'and c['host']=='xbabe2'and socket.gethostname().split('.')[0]=='xbabe2'
         and c['owned_root']==str(root),'owned reviewed neural runtime')
    need(window['config_sha256']==config_sha and global_deadline-window['created_unix']==1200 and window['created_unix']<=time.time()<global_deadline,'fixed1200s window')
    for ref in c['source_pins']:
        need(sha(read(ref['path']))==ref['sha256'],'source pin mismatch')
    for name,expected in c['owned_sources'].items():need(sha(read(root/name))==expected,'owned source identity')
    cg=next(x for x in Path('/proc/self/cgroup').read_text().splitlines()if x.startswith('0::'))
    group=Path('/sys/fs/cgroup')/cg[3:].lstrip('/');q,p=(group/'cpu.max').read_text().split()
    need(q!='max'and int(q)<=8*int(p)and int((group/'memory.max').read_text())<=8*1024**3 and
         int((group/'memory.swap.max').read_text())==0 and int((group/'pids.max').read_text())<=64,'CPU8/8GiB/swap0/tasks64')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'and
         os.environ.get('TMPDIR')==str(root/'tmp')and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')),'one GPU/owned temporary/single-thread BLAS')
    sys.path.insert(0,str(root/'runtime'))
    import torch,numpy as np,safetensors,boto3
    actual={'python':sys.version.split()[0],'numpy':str(np.__version__),'torch':str(torch.__version__),
        'safetensors':str(safetensors.__version__),'boto3':str(boto3.__version__)}
    need(actual==c['runtime_versions'],'exact scientific runtime versions')
    need(torch.cuda.is_available()and torch.cuda.device_count()==1,'one visible available GPU')
    from jope_flywheel.program_io import NativeTools,bound,create,tick,bind_campaign_deadline
    from jope_flywheel.protocol import canonical,digest
    from jope_flywheel import training_snapshot_kumos as S
    from jope_flywheel.program_inputs import RegisteredInputs
    from jope_flywheel.program_observer import ProgramObserver
    from jope_flywheel.program_worker import ProgramWorker
    from jope_flywheel.queue import Queue
    from jope_flywheel.s3_library import S3Library,client
    import flywheel_kumos_snapshot as snapshot
    import campaign_custody as custody
    import neural_session,neural_admission,resume_gate,snapshot_cache,atexit
    deadline=global_deadline-180;bind_campaign_deadline(deadline)
    bind_campaign_deadline(time.time()+max(0,monotonic-time.monotonic()-180))
    clock=custody.Clock(global_deadline);clock.monotonic=min(clock.monotonic,monotonic)
    preflight=json.loads(bound(c['gpu_preflight']))
    need(preflight['state']=='passed'and preflight['optimizer_updates']==0 and preflight['parameters']==828421 and
         preflight['parameter_sha256']==c['neural']['initial_parameter_sha256']and preflight['torch']==c['runtime_versions']['torch'],
         'actual GPU preflight and unchanged initial model origin')
    native=NativeTools(c['native_canonicalizer'],deadline)
    lock=resume_gate.RunLock(c['resume']['lock_path'],c['neural']['initial_parameter_sha256']);atexit.register(lock.close)
    resume,manifest=resume_gate.prepare(c['resume'],root,deadline)
    for role in('worker','emitter','rustc'):bound(c[role],64*1024**2)
    for program in c['neural']['proposal_programs']:
        need(native.canonical_program(program['dsl'])==program['dsl']and sha(program['dsl'].encode())==program['program_sha256'],'preregistered canonical native trie')
    raw_client=client(c['aws_profile'])
    class Budget:
        def __init__(self):self.attempts=0;self.previous='0'*64;create(root/'put-attempts.jsonl',b'')
        def __getattr__(self,name):return getattr(raw_client,name)
        def put_object(self,**kw):
            clock.tick();key=kw['Key'];label='/program-results/flywheel/v1/'in key
            need(kw['IfNoneMatch']=='*'and (label or '/program-attempts/flywheel/v1/'in key),'create-only owned namespace')
            need(self.attempts<770 and(not label or self.attempts<768),'768 results plus2 custody PUT cap')
            self.attempts+=1;row={'seq':self.attempts,'previous':self.previous,'key':key,'sha256':kw['Metadata']['sha256'],'bytes':kw['ContentLength'],'at':time.time()}
            self.previous=digest(row);row['event_sha256']=self.previous
            with (root/'put-attempts.jsonl').open('ab')as stream:stream.write(canonical(row)+b'\n');stream.flush();os.fsync(stream.fileno())
            return raw_client.put_object(**kw)
    budget=Budget()
    # Reload the exact admitted snapshot, including its artifact bytes, under
    # fresh source-role proof. Only the allowed KumoS vector is ever fetched.
    acceptance=S.document(bound(c['snapshot_acceptance']));status=S.document(bound(c['snapshot_status']))
    need(status['state']=='loaded'and status['config_sha256']==c['snapshot_acceptance']['sha256'],'accepted snapshot identity')
    spec_bytes=bound(status['snapshot']);coverage=S.document(bound(c['snapshot_coverage']))
    inventory=snapshot.artifact_budget(coverage['artifact_inventory']['objects'])
    local_binary_refs=json.loads(bound(c['local_binary_inventory']))['binaries']
    snap_store=snapshot_cache.VerifiedLocalStore(budget,native,deadline,accepted_inventory=inventory['objects'],local_binary_refs=local_binary_refs)
    need(inventory==coverage['artifact_inventory'],'exact snapshot artifact inventory')
    for ref in inventory['objects']:snap_store.get_version(ref)
    authority=snapshot.fresh_authority(acceptance,deadline)
    loaded=S.load_snapshot(spec_bytes,sha(spec_bytes),snap_store,registry_bytes=bound(acceptance['parent_registry']),
        authority_bytes=authority,authority_sha256=sha(authority),grammar_bytes=bound(acceptance['grammar']))
    expected=S.document(bound(status['loaded_receipt']))
    need(loaded.receipt['train_data_content_sha256']==expected['train_data_content_sha256']and
         loaded.receipt['all_membership_sha256']==expected['all_membership_sha256'],'actual reload equals admitted numeric/label corpus')
    need(loaded.receipt['counts']['fit_group_contexts']>0 and any(loaded.receipt['contexts'][cid]['split_role']=='dev'and
         p['stream']=='group'for cid,p in loaded.receipt['plans'].items()),
         'waiting for CE-eligible training AND development groups; feasibility-only is not policy learning qualification')
    permit=neural_admission.validate(c['research_optimizer_admission'],loaded,json.loads(authority))
    create(root/'research-admission-verification.json',canonical(permit))
    pending=bound(acceptance['pending_inventory']);current_manifest=S.batch_manifest(loaded,pending_inventory_bytes=pending,
        pending_inventory_sha256=acceptance['pending_inventory']['sha256'])
    need(manifest['rows']==current_manifest['rows']and manifest['snapshot_sha256']==current_manifest['snapshot_sha256'],
         'same frozen125 membership; no mid-epoch snapshot adoption')
    create(root/'snapshot-cache-proof.json',canonical({'local_hits':snap_store.local_hits,'local_bytes_read':snap_store.local_bytes_read,
        'remote_bytes_read':snap_store.remote_bytes_read,'total_verified_bytes':snap_store.bytes_read,'accepted_inventory_sha256':digest(inventory)}))
    create(root/'loaded-receipt.json',canonical(loaded.receipt));create(root/'role-proof.json',authority)
    del snap_store
    registry=json.loads(bound(c['registry']));inputs=RegisteredInputs(c,registry,native,budget,deadline)
    bindings={}
    for cid,details in loaded.receipt['contexts'].items():
        context,_,descriptor=inputs.describe(details['variant_identity'])
        need(context.identity==cid and context.__dict__==details['context'],'admitted snapshot/context binding')
        bindings[cid]={'context':context.__dict__,'descriptor_sha256':digest(descriptor)}
    pool={cid:[{'program_sha256':p['program_sha256'],'measurement':p,'record':p['wrapper']}for p in details['programs']]
          for cid,details in loaded.receipt['contexts'].items()}
    data=root/'data';data.mkdir();create(data/'identity.json',canonical({'config_sha256':config_sha}))
    out=root/'training';out.mkdir();final=None;failure=None
    class Store(S3Library):
        def tick(self):tick(self.deadline)
    store=Store(budget,deadline=deadline,nativecanonical_json=native.canonical_json)
    with Queue(data/'queue.sqlite',max_jobs=192)as queue:
        observer=ProgramObserver(queue,inputs,store,plan_sha256=config_sha,canonicalize=native.canonical_program)
        worker=ProgramWorker(queue,inputs,store,native,c,data,deadline,config_sha)
        resolver=neural_session.MandatoryResolver(queue,observer,worker,inputs,store,deadline,bindings)
        try:
            final=neural_session.continue_training(c['neural'],loaded,manifest,pool,resolver,out,deadline,config_sha,resume=resume,
                progress_extra=lambda:{'put_attempts':budget.attempts,'memory_current_bytes':int((group/'memory.current').read_text()),
                    'deadline_unix':global_deadline,'gpu':'cuda:0','phase':'bounded pipeline qualification'},
                fresh_admission=lambda:neural_admission.recheck_roles(inputs,c['research_optimizer_admission']))
        except BaseException as error:
            failure=type(error).__name__+': '+str(error)[:2000]
            if (out/'result.json').exists():final=json.loads(read(out/'result.json'))
            create(root/'neural-failure.json',canonical({'state':'failed','error':failure,'automatic_retry':False,'at':time.time()}))
        anchor={'config_sha256':config_sha,'queue':queue.verify(),'observations':observer.journal.verify(),'at':time.time()}
        create(root/'anchor.json',canonical(anchor))
        descriptors={r['context_id']:r['descriptor_sha']for r in queue.db.execute('SELECT * FROM program_inputs')}
    def ref(path):body=read(path);return {'path':str(path),'sha256':sha(body),'bytes':len(body)}
    archive_config={'source_config':ref(root/'config.json'),'registry':c['registry'],'anchor':ref(root/'anchor.json'),'descriptor_pins':descriptors}
    create(root/'archive-config.json',canonical(archive_config));archive_out=root/'attempt-custody';archive_out.mkdir()
    try:
        body,snapshot_doc=custody.freeze(archive_config,clock)
        archive=custody.publish(body,snapshot_doc,digest(archive_config),c['owned_sources']['campaign_custody.py'],archive_out,custody.Store(budget,clock))
        state='complete'if final and final['state']=='completed_bounded_qualification'else'partial_or_failed'
        status={'schema':'jope.flywheel-neural-status/v1','state':state,'config_sha256':config_sha,'at':time.time(),'window':window,
            'optimizer_steps':c['resume']['optimizer_step']if final is None else final['optimizer_steps'],'session_start_optimizer_step':c['resume']['optimizer_step'],
            'session_optimizer_steps':0 if final is None else final['optimizer_steps']-c['resume']['optimizer_step'],'result':ref(out/'result.json')if(out/'result.json').exists()else None,
            'attempt_custody':archive,'put_attempts':budget.attempts,'memory_peak_bytes':int((group/'memory.peak').read_text()),'error':failure}
        create(root/'status.json',canonical(status));print(json.dumps(status),flush=True)
    except BaseException as error:
        create(root/'status.json',canonical({'schema':'jope.flywheel-neural-status/v1','state':'custody_failed','config_sha256':config_sha,
            'at':time.time(),'error':type(error).__name__+': '+str(error)[:2000],'automatic_retry':False}));raise

if __name__=='__main__':
    try:main()
    except BaseException as error:
        root=Path(__file__).resolve().parent
        if not(root/'status.json').exists():
            with (root/'fatal-status.json').open('x')as f:json.dump({'state':'failed','error':type(error).__name__+': '+str(error)[:2000],'at':time.time(),'automatic_retry':False},f);f.flush();os.fsync(f.fileno())
        raise
