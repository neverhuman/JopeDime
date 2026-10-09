"""One shard of a fixed120minute native panel, including reserved final failure custody."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
sys.dont_write_bytecode=True

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(body):return hashlib.sha256(body).hexdigest()
def read(path,cap=32*1024**2):
    path=Path(path);need(path.is_absolute() and '..'not in path.parts and all(not p.is_symlink()for p in (path,*path.parents)),'nonsymlink absolute source')
    need(path.is_file() and 0<path.stat().st_size<=cap,'bounded file')
    with path.open('rb')as f:body=f.read(cap+1)
    need(len(body)<=cap,'file cap');return body
def localref(path):
    body=read(path);return {'path':str(path),'bytes':len(body),'sha256':sha(body)}

def main():
    root=Path(__file__).resolve().parent;config_bytes=read(root/'config.json');config_sha=sha(config_bytes);config=json.loads(config_bytes)
    window=json.loads(read(root/'window.json'));global_deadline=window['deadline_unix']
    global_monotonic=time.monotonic()+max(0,global_deadline-time.time())
    need(config['schema']=='jope.flywheel-program-label112-shard/recovery1' and str(root)==config['owned_root'] and
         socket.gethostname().split('.')[0]==config['host']=='xbabe3','campaign source identity')
    need(window['config_sha256']==config_sha and global_deadline-window['created_unix']==7200 and
         window['created_unix']<=time.time()<global_deadline,'fixed7200s original window')
    need(config['resources']['MemoryMax']=='4G' and config['maximum_actual_evaluations']==448,'reviewed resource/count caps')
    for name,key in [('label_campaign.py','helper_sha256'),('label_campaign_cli.py','cli_sha256'),('campaign_custody.py','custody_sha256')]:
        need(sha(read(root/name))==config[key],'owned executable source pin')
    need(sha(read(root/'staged_inputs.py'))==config['staged_inputs_sha256'],'staged reader pin')
    for ref in config['runtime_pins']:
        need(Path(ref['path']).parent==root.parent/'runtime/jope_flywheel' and sha(read(ref['path']))==ref['sha256'],'immutable package closure')
    original=json.loads(read(config['original_zero_attempt_proof']['path']))
    need(sha(read(config['original_zero_attempt_proof']['path']))==config['original_zero_attempt_proof']['sha256'] and
         original['all_native_attempts']==0 and original['all_result_puts']==0 and original['closed'] is True,
         'closed original campaign zero-attempt proof')
    group=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
    cpu=(group/'cpu.max').read_text().split()
    need(cpu[0]!='max' and int(cpu[0])/int(cpu[1])<=4 and int((group/'memory.max').read_text())<=4*1024**3 and
         int((group/'memory.swap.max').read_text())==0 and int((group/'pids.max').read_text())<=64,'actual CPU4/4GiB/no swap/task64')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
         os.environ.get('TMPDIR')==str(root/'tmp') and all(os.environ.get(k)=='1'for k in
         ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')),'owned CPU-only single-thread runtime')
    sys.path.insert(0,str(root.parent/'runtime'))
    import campaign_custody as custody
    import label_campaign as campaign
    from jope_flywheel.program_io import NativeTools,bind_campaign_deadline,bound,create,tick
    from jope_flywheel.protocol import canonical,digest
    from staged_inputs import StagedInputs
    from jope_flywheel.program_inputs import dataset_object
    from jope_flywheel.program_observer import ProgramObserver
    from jope_flywheel.program_worker import ProgramWorker
    from jope_flywheel.queue import Queue
    from jope_flywheel.s3_library import S3Library,client,BUCKET,PREFIX
    import numpy as np
    import boto3
    need({'python':sys.version.split()[0],'numpy':str(np.__version__),'boto3':str(boto3.__version__)}==config['runtime_versions'],'runtime version pins')
    clock=custody.Clock(global_deadline);clock.monotonic=min(clock.monotonic,global_monotonic)
    deadline=global_deadline-300;bind_campaign_deadline(deadline);clock.tick()
    native=NativeTools(config['native_canonicalizer'],deadline)
    for role in ('worker','emitter','rustc'):bound(config[role],64*1024**2)
    registry=json.loads(bound(config['registry']));campaign.schedule(config['variants'],config['programs'])
    published_encoders=json.loads(bound(config['encoder_publication']))
    need(published_encoders['completed_variants']==112 and published_encoders['verified_vectors']==112 and
         published_encoders['state']=='complete','complete112xKumoS publication')
    byvariant={v['variant_identity']:v for v in published_encoders['variants']}
    need(len(byvariant)==112 and set(byvariant)==set(config['all_variant_ids']),'exact112 complement')
    for variant in config['variants']:
        source=byvariant[variant['identity']]
        need(all(variant[k]==source[k]for k in ('rows','features','partition','parent_lineage_sha256','target_sha256','content_sha256')) and
             variant['encoding_receipt']==source['receipt'] and source['completed_encoders']==['kumo_tabular_s'],
             'exact KumoS-qualified panel')
    for program in config['programs']:
        need(native.canonical_program(program['dsl'])==program['dsl'] and sha(program['dsl'].encode())==program['program_sha256'],'fixed native parser qualification')
    data=root/'data';data.mkdir(mode=0o700);create(data/'identity.json',canonical({'config_sha256':config_sha}))
    budget=campaign.PutBudget(client(config['aws_profile']),root,clock)
    class Store(S3Library):
        def __init__(self):
            super().__init__(budget,deadline=deadline,nativecanonical_json=native.canonical_json);self.record_refs={}
        def tick(self):tick(self.deadline)
        def lookup(self,key):
            entry=super().lookup(key)
            if entry is not None:
                self.tick();location=PREFIX+key+'/record.json';response=budget.head_object(Bucket=BUCKET,Key=location)
                ref={'bucket':BUCKET,'key':location,'version_id':response.get('VersionId'),
                    'bytes':response['ContentLength'],'sha256':response.get('Metadata',{}).get('sha256')}
                need(ref['bytes']<=2*1024**2 and json.loads(self.get_version(ref)[0])==entry,'exact scored-pool record reference')
                self.record_refs[key]=ref
            return entry
    store=Store();inputs=StagedInputs(config,registry,native,budget,deadline);budget.authority=inputs.lease
    inputs.lease.check()
    # The vector bytes are not needed for native fits. Their exact published
    # qualification receipt closes the selected fully encoded context panel.
    for variant in config['variants']:
        raw=dataset_object(budget,variant['encoding_receipt'],variant=variant['identity'],name='receipt',deadline=deadline)
        doc=json.loads(raw)
        need(doc['schema']=='jope.flywheel-evergreen-kumos-publication/v1' and doc['variant_identity']==variant['identity'] and
             doc['completed_encoders']==['kumo_tabular_s'] and doc['parent_lineage_sha256']==variant['parent_lineage_sha256'] and
             doc['content_sha256']==variant['content_sha256'] and doc['target_sha256']==variant['target_sha256'] and
             doc['partition']==variant['partition'] and doc['teaching_eligible'] is False,'encoder receipt variant identity')
        create(data/('encoder-receipt-'+variant['identity']+'.json'),raw)
    run_config={**config,'campaign_deadline_unix':global_deadline}
    final=None;fatal=None
    with Queue(data/'queue.sqlite',max_jobs=896)as queue:
        observer=ProgramObserver(queue,inputs,store,plan_sha256=config_sha,canonicalize=native.canonical_program)
        worker=ProgramWorker(queue,inputs,store,native,config,data,deadline,config_sha)
        try:
            final=campaign.run_schedule(run_config,queue,observer,worker,inputs,root,deadline,config_sha,
                progress_extra=lambda:{'put_attempts':budget.attempts,'memory_current_bytes':int((group/'memory.current').read_text())})
        except BaseException as error:
            fatal=type(error).__name__+': '+str(error)[:1500]
            create(root/'campaign-failure.json',canonical({'schema':'jope.flywheel-program-label-failure/v1','error':fatal,
                'at':time.time(),'automatic_retry':False,'teaching_eligible':False}))
        anchor={'schema':'jope.flywheel-program-label-anchor/v1','config_sha256':config_sha,
            'queue':queue.verify(),'observations':observer.journal.verify(),'at':time.time()}
        descriptors={r['context_id']:r['descriptor_sha']for r in queue.db.execute('SELECT * FROM program_inputs')}
        create(root/'anchor.json',canonical(anchor))
    pool={'schema':'jope.flywheel-scored-program-pool/v1','config_sha256':config_sha,
        'entries':[{'evaluation_key':r['evaluation_key'],'variant_identity':r['variant_identity'],
                    'context_id':r['context_id'],'partition':r['partition'],'phase':r['phase'],
                    'program_sha256':r['program_sha256'],'program_name':r['program_name'],
                    'record':store.record_refs[r['evaluation_key']],'measurement':r['measurement']}
                   for r in (final['outcomes']if final else []) if r.get('state')=='ready'],
        'all_planned_slots':448,'label_results':localref(root/'label-results.json')if final else None,
        'selection_uses_validation':False,'teaching_eligible':False,'model_generated':False,'dl_win_rate':None}
    create(root/'scored-pool.json',canonical(pool))
    archive_config={'source_config':localref(root/'config.json'),'registry':config['registry'],
        'anchor':localref(root/'anchor.json'),'descriptor_pins':descriptors}
    create(root/'archive-config.json',canonical(archive_config));archive_sha=sha(canonical(archive_config))
    archive_out=root/'attempt-custody';archive_out.mkdir(mode=0o700)
    try:
        body,snapshot=custody.freeze(archive_config,clock)
        archive=custody.publish(body,snapshot,archive_sha,config['custody_sha256'],archive_out,custody.Store(budget,clock))
        status={'schema':'jope.flywheel-program-label-status/v1','state':'complete'if final and not final['not_started'] and
             final['counts']['failed_slots']==0 else'partial','config_sha256':config_sha,'window':window,
            'at':time.time(),'counts':final['counts']if final else snapshot['counts'],'fatal_error':fatal,
            'pool':localref(root/'scored-pool.json'),'attempt_custody':archive,'put_attempts':budget.attempts,
            'memory_peak_bytes':int((group/'memory.peak').read_text()),'teaching_eligible':False,'optimizer_reward_allowed':False,
            'model_generated':False,'dl_win_rate':None}
        create(root/'status.json',canonical(status));print(json.dumps(status))
    except BaseException as error:
        status={'schema':'jope.flywheel-program-label-status/v1','state':'custody_failed','config_sha256':config_sha,
            'window':window,'at':time.time(),'error':type(error).__name__+': '+str(error)[:1500],
            'label_results_preserved':final is not None,'partial_s3_possible':True,'automatic_retry':False}
        create(root/'status.json',canonical(status));print(json.dumps(status));raise

if __name__=='__main__':
    try:main()
    except BaseException as error:
        root=Path(__file__).resolve().parent
        if not (root/'status.json').exists():
            with (root/'fatal-status.json').open('x')as f:
                json.dump({'state':'failed','error':type(error).__name__+': '+str(error)[:1500],
                           'at':time.time(),'automatic_retry':False},f);f.flush();os.fsync(f.fileno())
        raise
