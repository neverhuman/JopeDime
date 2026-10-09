"""Owned queue consumer: real local S1 fit, emitted Rust, parity, versioned S3."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import struct
import time
from . import evergreen
from .program_gate import ARTIFACTS, SCHEMA, request_identity, verified_measurement
from .program_io import bound, create, need, read, run, sha, tick
from .protocol import Context, canonical, digest
from .score_join import build_records

TOLERANCE = 1e-9


def split_scale(x,y,folds,fold):
    train = [i for i,v in enumerate(folds) if v != fold]
    valid = [i for i,v in enumerate(folds) if v == fold]
    need(min(len(train),len(valid)) >= 25, 'inner fold minimum')
    lo = [min(float(x[i,j]) for i in train) for j in range(x.shape[1])]
    span = [max(float(x[i,j]) for i in train)-v for j,v in enumerate(lo)]
    span = [v if v > 0 else 1. for v in span]
    ylo = min(float(y[i]) for i in train); hi = max(float(y[i]) for i in train)
    yspan = hi-ylo if hi-ylo > 0 else 1.
    numbers = lo+span+[ylo,yspan]
    return {'train': [[(float(x[i,j])-lo[j])/span[j] for j in range(len(lo))]+[(float(y[i])-ylo)/yspan] for i in train],
            'validation': [[(float(x[i,j])-lo[j])/span[j] for j in range(len(lo))] for i in valid],
            'targets':[float(y[i]) for i in valid], 'scaler_sha256':sha(struct.pack('<'+'d'*len(numbers),*numbers)),
            'ylo':ylo,'yspan':yspan,'train_rows':len(train),'score_rows':len(valid)}


def csv_bytes(rows):
    need(rows and len({len(r) for r in rows}) == 1 and all(math.isfinite(v) for row in rows for v in row), 'finite rectangular CSV')
    return ('\n'.join(','.join(repr(v) for v in row) for row in rows)+'\n').encode()


def compare_predictions(actual,reference):
    need(len(actual) == len(reference) and len(actual) >= 25 and all(math.isfinite(v) for v in actual+reference), 'finite complete predictions')
    errors = [abs(a-b) for a,b in zip(actual,reference)]
    result = {'rows':len(actual),'bit_exact_rows':sum(struct.pack('<d',a)==struct.pack('<d',b) for a,b in zip(actual,reference)),
        'max_absolute_error':max(errors),'tolerance':TOLERANCE,
        'criterion':'abs(actual-reference) <= tolerance * max(1,abs(reference))',
        'passed':all(e <= TOLERANCE*max(1.,abs(b)) for e,b in zip(errors,reference))}
    need(result['passed'], 'emitted/native prediction parity failed; tolerance is fixed')
    return result


class LocalStore:
    def __init__(self,bodies,native): self.bodies,self.nativecanonical_json = bodies,native
    def get_version(self,ref): return self.bodies[ref['key']],ref['version_id']


def local_entry(context,program,bodies):
    refs = {name:{'bucket':'local-semantic-preflight-only','key':name,'version_id':'local-semantic-preflight-only',
                  'bytes':len(body),'sha256':sha(body)} for name,body in bodies.items()}
    return {'schema':SCHEMA,'context':asdict(context),'evaluation_key':request_identity(context,program),
            'program_sha256':sha(program.encode()),'status':'completed','measurement':'executed',
            'artifacts':refs,'teaching_eligible':False}


def evaluation_limit(config):
    """The legacy ceiling remains128. Only this reviewed fixed112 recovery gets448."""
    maximum=config.get('maximum_actual_evaluations',1)
    if config.get('schema')!='jope.flywheel-program-label112-shard/recovery1':
        need(type(maximum)is int and 1<=maximum<=128,'bounded authorized evaluation count')
        return maximum
    need(type(maximum)is int and maximum==448,'exact recovery448 limit')
    need(config.get('recovery_of_campaign_sha256')=='433eb4cb2ce44b75466ef2be1b0334803e5879ec280fac553749a1d23db3b35c',
         'one exact original refused campaign')
    variants=config['variants'];programs=config['programs'];resources=config['resources']
    need(len(variants)==len({v['identity']for v in variants})==56 and
         len(programs)==len({p['program_sha256']for p in programs})==8 and
         all(v['partition']=='train'for v in variants),'exact56train x8program slots')
    need(config['host']=='xbabe3' and resources['CPUQuota']=='400%' and resources['MemoryMax']=='4G' and
         resources['MemorySwapMax']=='0' and resources['TasksMax']==64 and resources['CUDA_VISIBLE_DEVICES']=='' and
         resources['max_actual_evaluations']==maximum and resources['max_total_put_attempts']==maximum*8+2 and
         resources['automatic_retry']is False and resources['deadline_extension']is False,'reviewed recovery budget envelope')
    return maximum


class ProgramWorker:
    def __init__(self,queue,inputs,store,native,config,root,deadline,plan_sha256):
        self.queue,self.inputs,self.store,self.native = queue,inputs,store,native
        self.config,self.root,self.deadline,self.plan = config,Path(root),deadline,plan_sha256

    def resolve(self,context):
        row = self.queue.db.execute('SELECT * FROM program_inputs WHERE context_id=?',(context.identity,)).fetchone()
        need(row is not None, 'worker requires observed registered input')
        actual, raw, descriptor = self.inputs.describe(row['variant_identity'])
        need(actual == context and digest(descriptor) == row['descriptor_sha'] and canonical(descriptor).decode() == row['descriptor'], 'registered context changed')
        return raw,descriptor

    def evaluate(self,job):
        payload = job['payload']; context = Context(**payload['context']); program = payload['program']
        key = request_identity(context,program)
        need(key == payload['evaluation_key'] and self.native.canonical_program(program) == program, 'canonical evaluation identity')
        publication = self.queue.db.execute('SELECT * FROM jobs WHERE id=?',('gp-publish-'+key,)).fetchone()
        dependencies = [r[0] for r in self.queue.db.execute('SELECT dependency FROM dependencies WHERE job=?',('gp-publish-'+key,))]
        need(publication is not None and publication['stage'] == 'publish' and publication['state'] == 'pending' and
             publication['plan_sha'] == self.plan and json.loads(publication['payload']).get('evaluation_key') == key and
             dependencies == [job['id']], 'evaluation requires a matching dependent publication job')
        raw,descriptor = self.resolve(context)
        previous = self.store.lookup(key)
        dest = self.root/('gp-evaluate-'+key); dest.mkdir(exist_ok=False)
        if previous is not None:
            measured = verified_measurement(self.store,context,program,previous)
            return create(dest/'result.json',canonical({'schema':'jope.flywheel-worker-result/v1','state':'cached',
                'context':asdict(context),'program':program,'measurement':measured,'teaching_eligible':False}))
        create(dest/'input.jtt1',raw); create(dest/'input-descriptor.json',canonical(descriptor))
        candidate = {'dsl':program,'program_sha256':sha(program.encode()),'birth_seq':0,'parents':[],
                     'operator':'observed_external_program'}
        create(dest/'candidate.json',canonical(candidate))
        for role in ('worker','emitter','rustc'): bound(self.config[role],64*1024**2)
        commands = {}
        commands['lambda'] = run([self.config['worker']['path'],'--fit-s1',str(dest/'input.jtt1'),
            str(dest/'candidate.json'),str(dest/'evaluation.json')],dest,'lambda',self.deadline,90)
        evaluation_bytes = read(dest/'evaluation.json',4*1024**2); evaluation = json.loads(evaluation_bytes)
        need(evaluation['schema'] == 'jope.s1-evaluation/v1' and evaluation['program'] == candidate and
             evaluation['status'] == 'completed' and evaluation['fold_protocol'] == 'jope.s1-two-fold/v1' and
             evaluation['inner_cv_folds'] == 2 and evaluation['input_semantics'] == 'unscaled_fit_rows_v1' and
             'official_test' not in evaluation and len(evaluation['inner_cv']) == 2, 'complete native two-fold evaluation')
        x,y,folds = evergreen.decode(raw); scaled = [split_scale(x,y,folds,i) for i in range(2)]
        for i,split in enumerate(scaled):
            fold = evaluation['inner_cv'][i]
            need(fold['index'] == i and fold['kind'] == 'inner_cv' and fold['scaler_sha256'] == split['scaler_sha256'] and
                 fold['train_rows'] == split['train_rows'] and fold['score_rows'] == split['score_rows'] and
                 fold['targets'] == split['targets'] and fold['serialized_prediction_replay_exact'] is True,
                 'native fit-only scaler/membership/model replay')
            create(dest/('fold'+str(i)+'-train.csv'),csv_bytes(split['train']))
            create(dest/('fold'+str(i)+'-validation.csv'),csv_bytes(split['validation']))
        commands['emit'] = run([self.config['emitter']['path'],str(dest/'candidate.json'),str(dest/'emitted')],
                               dest,'emit',self.deadline,30)
        manifest_bytes = read(dest/'emitted/manifest.json'); manifest = json.loads(manifest_bytes)
        code = read(dest/'emitted/model.rs',4*1024**2)
        need(manifest['schema'] == 'battle-artifact-v1' and manifest['program_sha256'] == candidate['program_sha256'] and
             manifest['rs_sha256'] == sha(code) and manifest['program'] == program and
             read(dest/'emitted/program.dsl') == program.encode(), 'native emitted source/program identity')
        binary = dest/'emitted/model'
        commands['compile'] = run([self.config['rustc']['path'],'--edition=2021','-O',str(dest/'emitted/model.rs'),'-o',str(binary)],
                                   dest,'compile',self.deadline,120)
        checks,predictions = [],[]
        for i,split in enumerate(scaled):
            predicted = dest/('fold'+str(i)+'-predictions.csv')
            commands['execute'+str(i)] = run([str(binary),str(dest/('fold'+str(i)+'-train.csv')),
                str(dest/('fold'+str(i)+'-validation.csv')),str(predicted)],dest,'execute'+str(i),self.deadline,90)
            body = read(predicted,2*1024**2); predictions.append(body.decode())
            values = [float(line) for line in body.decode().splitlines()]
            actual = [v*split['yspan']+split['ylo'] for v in values]
            result = compare_predictions(actual,evaluation['inner_cv'][i]['predictions'])
            result.update({'fold':i,'scaler_sha256':split['scaler_sha256'],'scaled_predictions_sha256':sha(body)})
            checks.append(result)
        for role in ('worker','emitter','rustc'): bound(self.config[role],64*1024**2)
        self.resolve(context)  # Role freshness and exact generated bytes again before admission to publication.
        bodies = {'input':raw,'program':program.encode(),'evaluation':evaluation_bytes,
                  'rust_source':code,'rust_executable':read(binary,16*1024**2)}
        provenance = {'schema':'jope.flywheel-native-execution-provenance/v1','evaluator_sha256':context.evaluator_sha256,
            'input_sha256':sha(raw),'program_sha256':candidate['program_sha256'],'evaluation_sha256':sha(evaluation_bytes),
            'source_commit':self.config['source_commit'],'source_build_plan_sha256':self.config['source_build_plan_sha256'],
            'execution_timestamp':int(time.time()),'worker':self.config['worker'],'emitter':self.config['emitter'],
            'rustc':self.config['rustc'],'canonicalizer':self.config['native_canonicalizer'],'commands':commands,
            'input_descriptor':descriptor,'emitter_manifest_json_utf8':manifest_bytes.decode(),
            'emitted_predictions_csv_utf8':predictions,'fold_checks':checks,'plan_sha256':self.plan,
            'scope':'Owned local execution custody; prepared synthetic target units; no outer holdout; research only.'}
        parity = {'schema':'jope.flywheel-rust-parity/v1','passed':True,'executed':True,'folds_verified':2,
                  'evaluation_key':key,'program_sha256':candidate['program_sha256'],'evaluator_sha256':context.evaluator_sha256,
                  'native_provenance':provenance}
        parity.update({role+'_sha256':sha(bodies[role]) for role in ('input','evaluation','rust_source','rust_executable')})
        bodies['rust_parity'] = canonical(parity)
        bodies['score_records'] = build_records(context,candidate['program_sha256'],key,
            [f['metrics']['rmse'] for f in evaluation['inner_cv']],sha(bodies['rust_parity']),self.config['source_commit'],
            provenance['execution_timestamp'],self.native.canonical_json)
        entry = local_entry(context,program,bodies)
        verified_measurement(LocalStore(bodies,self.native.canonical_json),context,program,entry)
        artifacts = {}
        for role,body in bodies.items(): artifacts[role] = create(dest/('artifact-'+role),body)
        tick(self.deadline)
        return create(dest/'result.json',canonical({'schema':'jope.flywheel-worker-result/v1','state':'executed',
            'context':asdict(context),'program':program,'artifacts':artifacts,'teaching_eligible':False}))

    def publish(self,job):
        key = job['payload']['evaluation_key']; prerequisite = 'gp-evaluate-'+key
        row = self.queue.db.execute('SELECT * FROM jobs WHERE id=?',(prerequisite,)).fetchone()
        need(row is not None and row['state'] == 'completed', 'qualified evaluation prerequisite')
        receipt = json.loads(bound({'path':str(self.root/prerequisite/'result.json'),'sha256':row['receipt_sha']}))
        context = Context(**receipt['context']); program = receipt['program']
        need(request_identity(context,program) == key and self.native.canonical_program(program) == program, 'publication request identity')
        self.resolve(context)
        old = self.store.lookup(key)
        if old is not None:
            result = {'state':'cached','measurement':verified_measurement(self.store,context,program,old)}
        else:
            need(receipt['state'] == 'executed' and set(receipt['artifacts']) == set(ARTIFACTS), 'full local executed closure')
            bodies = {role:bound(ref) for role,ref in receipt['artifacts'].items()}
            entry = local_entry(context,program,bodies)
            verified_measurement(LocalStore(bodies,self.native.canonical_json),context,program,entry)
            entry['artifacts'] = {role:self.store.put_artifact(role,body) for role,body in bodies.items()}
            result = self.store.commit(context,program,entry)
            looked_up = self.store.lookup(key); need(looked_up == entry,'authoritative published record mismatch')
            need(verified_measurement(self.store,context,program,looked_up) == result['measurement'], 'published readback measurement')
            result.update({'state':'published','artifacts':entry['artifacts']})
        result.update({'schema':'jope.flywheel-worker-publication/v1','teaching_eligible':False,'optimizer_reward_allowed':False})
        return create(self.root/('gp-publish-'+key+'.json'),canonical(result))

    def once(self):
        tick(self.deadline)
        remaining = int(self.deadline-time.time()); need(remaining >= 2,'insufficient remaining fixed window')
        maximum = evaluation_limit(self.config)
        consumed = self.queue.db.execute("SELECT coalesce(sum(attempts),0) FROM jobs WHERE stage='evaluate'").fetchone()[0]
        need(consumed<=maximum,'consumed attempts exceed fixed plan')
        stages = ['evaluate','publish'] if consumed < maximum else ['publish']
        job = self.queue.claim('owned-local-lambda-worker',stages,lease_seconds=min(remaining,600),max_running=1)
        if job is None: return {'state':'idle'}
        try:
            need(job['plan_sha256'] == self.plan,'worker plan identity')
            receipt = self.evaluate(job) if job['stage'] == 'evaluate' else self.publish(job)
            self.queue.finish(job['id'],job['lease_token'],receipt['sha256'],success=True)
            return {'state':'completed','stage':job['stage'],'job':job['id'],'receipt':receipt}
        except BaseException as error:
            failure = create(self.root/(job['id']+'-failure.json'),canonical({'schema':'jope.flywheel-worker-failure/v1',
                'job':job['id'],'stage':job['stage'],'error':type(error).__name__+': '+str(error)[:1000],
                'at':time.time(),'external_outcome_may_be_partial':True,'automatic_retry':False}))
            # An expired lease remains needs_reconcile. A failure never frees work for retry.
            try: self.queue.finish(job['id'],job['lease_token'],failure['sha256'],success=False,reason='owned_worker_failed_no_retry')
            except ValueError: pass
            raise
