"""Fixed native candidate measurement campaign; no neural generation or reward."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from jope_flywheel.program_io import create, bound, need, tick, remaining, sha
from jope_flywheel.protocol import canonical, digest

def schedule(variants, programs):
    need(1<=len(variants)<=16 and len({v['identity']for v in variants})==len(variants),'bounded distinct contexts')
    need(len({v['parent_lineage_sha256']for v in variants})==len(variants),'distinct parent lineages')
    need(all(v['partition']in('train','validation')for v in variants),'inherited development partitions')
    need(len(programs)==8 and len({p['program_sha256']for p in programs})==8,'eight fixed canonical programs')
    return [{'slot':r*len(variants)+i,'round':r,'variant':variant,'program':program,
             'phase':'training' if variant['partition']=='train' else 'validation'}
            for r,program in enumerate(programs) for i,variant in enumerate(variants)]

class PutBudget:
    """Counts durable attempted writes before network I/O; no retries or reset."""
    def __init__(self, client, root, clock, maximum=1026):
        self.client,self.root,self.clock=client,Path(root),clock
        self.maximum=maximum;self.attempts=0;self.previous='0'*64;self.authority=None
        create(self.root/'put-attempts.jsonl',b'')
    def __getattr__(self,name):return getattr(self.client,name)
    def put_object(self,**kwargs):
        self.clock.tick();
        need(self.attempts<self.maximum,'campaign total PUT attempt cap')
        key=kwargs['Key'];is_label='/program-results/flywheel/v1/'in key
        if is_label and self.authority is not None:self.authority.check()
        need(is_label or '/program-attempts/flywheel/v1/'in key,'owned GP/custody namespace only')
        need(not is_label or self.attempts<self.maximum-2,'reserve two final custody PUT attempts')
        need(kwargs.get('IfNoneMatch')=='*','create-only PUT required')
        self.attempts+=1
        row={'seq':self.attempts,'key':key,'bucket':kwargs['Bucket'],'bytes':kwargs['ContentLength'],
             'sha256':kwargs['Metadata']['sha256'],'at':time.time(),'previous':self.previous}
        self.previous=digest(row);row['event_sha256']=self.previous
        with (self.root/'put-attempts.jsonl').open('ab')as f:
            f.write(canonical(row)+b'\n');f.flush();os.fsync(f.fileno())
        return self.client.put_object(**kwargs)

def update(root, value):
    tmp=Path(root)/'progress.pending';create(tmp,canonical(value));os.replace(tmp,Path(root)/'progress.json')
    fd=os.open(root,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def counts(rows, queue, planned_slots):
    snapshot=queue.snapshot()
    return {'planned_slots':planned_slots,'observed_slots':sum('observation'in r or r.get('state')=='observation_error'for r in rows),
        'completed_actual_executions':sum(r.get('actual_execution',False)for r in rows),
        'completed_publications':sum(r.get('published',False)for r in rows),
        'initial_cache_hits':sum(r.get('initial_cache_hit',False)for r in rows),
        'post_publication_cache_hits':sum(r.get('post_publication_cache_hit',False)for r in rows),
        'failed_slots':sum(r.get('state')in('observation_error','worker_error','readback_error','input_error')for r in rows),
        'finished_slots':sum(r.get('state')in('ready','worker_error','observation_error','readback_error','input_error')for r in rows),
        'queue':snapshot,'numeric_failure_scores':0}

def run_schedule(config, queue, observer, worker, inputs, root, deadline, config_sha, progress_extra=None):
    rows=[];descriptors={};root=Path(root);planned=schedule(config['variants'],config['programs'])
    def progress(state,current=None):
        update(root,{'schema':'jope.flywheel-program-label-progress/v1','state':state,'at':time.time(),
            'config_sha256':config_sha,'deadline_unix':config['campaign_deadline_unix'],
            'work_deadline_unix':deadline,'current':current,'counts':counts(rows,queue,len(planned)),
            'outcomes':rows,'observation_chain':observer.journal.verify(),
            'research_only':True,'teaching_eligible':False,'optimizer_reward_allowed':False,
            'model_generated':False,'dl_win_rate':None,**(progress_extra()if progress_extra else {})})
    progress('starting')
    for item in planned:
        # Leave the entire declared final custody reserve untouched. No window extension.
        if remaining(deadline)<20:break
        tick(deadline)
        variant=item['variant'];program=item['program']
        row={'slot':item['slot'],'round':item['round'],'variant_identity':variant['identity'],
             'parent_lineage_sha256':variant['parent_lineage_sha256'],'partition':variant['partition'],
             'phase':item['phase'],'program_name':program['name'],'program_sha256':program['program_sha256'],
             'state':'preflight'}
        rows.append(row);progress('observing',{'round':item['round'],'slot':item['slot'],'variant_identity':variant['identity']})
        try:
            context,_,descriptor=inputs.describe(variant['identity'])
            need(descriptor['partition']==variant['partition'] and descriptor['parent_lineage_sha256']==variant['parent_lineage_sha256'],
                 'inherited phase and lineage remain fixed')
            if 'rows' in variant:
                need(descriptor['rows']==variant['rows'] and descriptor['features']==variant['features'] and
                     context.target_sha256==variant['target_sha256'] and context.dataset_sha256==variant['content_sha256'],
                     'exact encoded context shape and new target identity')
            descriptors[context.identity]=digest(descriptor)
        except BaseException as error:
            row.update(state='input_error',error=type(error).__name__+': '+str(error)[:1000]);progress('stopped_source_error');break
        request={'variant_identity':variant['identity'],'context':asdict(context),'descriptor_sha256':digest(descriptor),
            'dsl':program['dsl'],'phase':item['phase'],'checkpoint_sha256':config['proposal_provenance_sha256'],
            'step':item['slot'],'producer':'fixed-native-program-panel-v1-not-neural'}
        try:
            row['observation']=observer.observe(request)
        except BaseException as error:
            row.update(state='observation_error',error=type(error).__name__+': '+str(error)[:1000]);progress('stopped_observation_error');break
        row['context_id']=context.identity;row['evaluation_key']=row['observation']['evaluation_key']
        if row['observation']['state']=='ready':
            row.update(state='ready',initial_cache_hit=True,measurement=row['observation']['measurement']);progress('running');continue
        try:
            work=worker.once();row['evaluation_work']=work
            need(work['state']=='completed' and work['stage']=='evaluate' and work['job']=='gp-evaluate-'+row['evaluation_key'],
                 'expected exact candidate evaluation')
            receipt=json.loads(bound(work['receipt']));row['actual_execution']=receipt['state']=='executed'
            progress('publishing',{'round':item['round'],'slot':item['slot'],'variant_identity':variant['identity']})
            publication=worker.once();row['publication_work']=publication
            need(publication['state']=='completed' and publication['stage']=='publish' and publication['job']=='gp-publish-'+row['evaluation_key'],
                 'expected exact candidate publication')
            published=json.loads(bound(publication['receipt']));row['published']=published['state']=='published'
            row['publication']=published
        except BaseException as error:
            row.update(state='worker_error',error=type(error).__name__+': '+str(error)[:1000]);progress('running_partial')
            break # retain the first failed attempt; no generic continuation/retry on invocation-contract errors
        try:
            after=observer.observe(request);need(after['state']=='ready','authoritative exact S3 cache readback')
            row.update(state='ready',post_publication_cache_hit=True,readback=after,measurement=after['measurement'])
        except BaseException as error:
            row.update(state='readback_error',error=type(error).__name__+': '+str(error)[:1000]);progress('stopped_readback_error');break
        progress('running')
    planned_not_started=[{'slot':x['slot'],'round':x['round'],'variant_identity':x['variant']['identity'],
                          'program_sha256':x['program']['program_sha256'],'phase':x['phase'],'state':'not_started'}
                         for x in planned[len(rows):]]
    final={'schema':'jope.flywheel-program-label-results/v1','config_sha256':config_sha,'outcomes':rows,
        'not_started':planned_not_started,'counts':counts(rows,queue,len(planned)),'descriptor_pins':descriptors,
        'queue':queue.verify(),'observations':observer.journal.verify(),
        'teaching_eligible':False,'optimizer_reward_allowed':False,'model_generated':False,
        'frozen_program_panel':config['programs'],'selection_uses_validation':False,
        'comparisons':'All fixed-panel outcomes retained, including losers and failed attempts. This is a scored candidate pool for later exact-context comparisons, not a learned-policy win rate.'}
    create(root/'label-results.json',canonical(final));progress('measurement_finished')
    return final
