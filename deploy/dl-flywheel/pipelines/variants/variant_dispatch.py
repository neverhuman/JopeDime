"""Bounded serial model-outbox variant execution; never optimizer admission."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time
from neural_outbox import Outbox
from jope_flywheel import timing
from jope_flywheel.program_gate import request_identity
from jope_flywheel.program_inputs import RegisteredInputs
from jope_flywheel.program_io import NativeTools, bind_campaign_deadline, bound, create, need, remaining, sha
from jope_flywheel.program_observer import ProgramObserver
from jope_flywheel.program_worker import ProgramWorker
from jope_flywheel.protocol import canonical, digest
from jope_flywheel.queue import Queue
from jope_flywheel.s3_library import S3Library, BUCKET, PREFIX, client, code
from jope_flywheel.variant_outbox import VariantConsumer

SCHEMA = 'jope.variant-model-outbox-packet/v1'
FIELDS = {'schema','out','host','created_unix','deadline_unix','global_deadline_unix',
          'maximum_emissions','source_pins','training_config','execution_config',
          'adoption_receipt','prepared','outbox_path','outbox_config','worker_config',
          'historical_census','worker_id','lock_path'}


def progress(root, value):
    temp = root / 'progress.tmp'
    with temp.open('xb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, root / 'progress.json')


class Library(S3Library):
    """Capture the same exact-version wrapper used by the authoritative lookup."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_refs = {}

    def lookup(self, key):
        from jope_flywheel.protocol import sha256
        sha256(key); self.tick()
        path = PREFIX + key + '/record.json'
        try:
            header = self.client.head_object(Bucket=BUCKET, Key=path)
        except Exception as error:
            self.tick()
            if code(error) in ('404','NoSuchKey','NotFound'):
                return None
            raise
        ref = {'bucket':BUCKET,'key':path,'version_id':header.get('VersionId'),
               'sha256':header.get('Metadata',{}).get('sha256'),'bytes':header['ContentLength']}
        need(0 < ref['bytes'] <= 2 * 1024**2, 'bounded result wrapper')
        raw, _ = self.get_version(ref)
        entry = json.loads(raw)
        self.record_refs[key] = ref
        return entry


def outbox_snapshot(path):
    """Private indexed memory copy from one physical read-only WAL snapshot.

    The frozen verifier is unchanged. No DDL or repair reaches the live database.
    A source SELECT fixes the transaction before backup; proof uses copied events,
    batch receipts and states, and each subsequent claim uses the live outbox.
    """
    p = Path(path)
    need(p.is_absolute() and all(not x.is_symlink() for x in (p,*p.parents)), 'nonsymlink outbox authority')
    source = sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,isolation_level=None,timeout=2)
    snapshot = sqlite3.connect(':memory:',isolation_level=None)
    snapshot.row_factory = sqlite3.Row
    try:
        source.execute('PRAGMA query_only=ON'); source.execute('BEGIN')
        source.execute('SELECT body FROM config').fetchone()
        source.backup(snapshot)
        snapshot.execute('CREATE INDEX variant_private_batch_lookup ON emissions(batch_id)')
        return snapshot
    except BaseException:
        snapshot.close(); raise
    finally:
        source.close()


def preflight(c):
    need(set(c) == FIELDS and c['schema'] == SCHEMA, 'closed packet config')
    root = Path(c['out'])
    need(root.is_absolute() and all(not p.is_symlink() for p in (root,*root.parents)) and
         str(root).startswith('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-variant-outbox-'),
         'owned isolated proposal root')
    need(os.uname().nodename == c['host'] == 'xbabe2', 'qualified registry host')
    need(1 <= c['maximum_emissions'] <= 4 and
         0 < c['deadline_unix'] - c['created_unix'] <= 1800 and
         c['deadline_unix'] <= c['global_deadline_unix'] == 1791592146.380285 and
         time.time() < c['deadline_unix'], 'fixed bounded packet; original global clock')
    for ref in c['source_pins']:
        bound(ref)
    logical = json.loads(bound(c['training_config']))
    executor = json.loads(bound(c['execution_config']))
    prepared = json.loads(bound(c['prepared']))
    adoption = json.loads(bound(c['adoption_receipt']))
    need(executor['campaign_config'] == c['training_config'] and
         executor['prepared_candidate']['prepared'] == c['prepared'], 'same reviewed executor')
    need(adoption['schema'] == 'jope.prepared-epoch-adoption/v1' and
         adoption['snapshot_sha256'] == prepared['extension_sha256'] and
         adoption['manifest_sha256'] == digest(prepared['batch_manifest']), 'physically admitted prepared roster')
    descriptors = prepared['variant_contexts']
    need(len(descriptors) == 128 and len({d['variant_identity'] for d in descriptors.values()}) == 128,
         'complete unchanged admitted variant roster')
    need(c['outbox_path'] == str(Path(logical['owned_root'])/'training/outbox.sqlite') and
         c['lock_path'] == c['outbox_path'] + '.variant-consumer.lock', 'exact outbox/lock authority')
    db = outbox_snapshot(c['outbox_path'])
    try:
        checker = Outbox.__new__(Outbox); checker.db = db
        proof = checker.verify()
        need(json.loads(db.execute('SELECT body FROM config').fetchone()[0]) == c['outbox_config'], 'frozen outbox control')
        rows = db.execute("SELECT body FROM emissions WHERE json_extract(body,'$.emission.input_descriptor.source_kind')='flywheel_variant'").fetchall()
        for row in rows:
            body = json.loads(row[0]); e = body['emission']
            need(body['batch']['campaign_sha256'] == c['training_config']['sha256'] and
                 e['context_id'] in descriptors and e['input_descriptor'] == descriptors[e['context_id']],
                 'emitted descriptor in admitted immutable roster')
    finally:
        db.close()
    census = json.loads(bound(c['historical_census']))
    need(census['schema'] == 'jope.variant-historical-census/v1' and census['source_queues'] and
         census['unknown_outcomes_preserved'] is True and
         census['blocked_evaluation_keys'] == sorted(set(census['blocked_evaluation_keys'])) and
         census['completed_evaluation_keys'] == sorted(set(census['completed_evaluation_keys'])), 'closed historical custody')
    worker = c['worker_config']
    need(worker['maximum_actual_evaluations'] == c['maximum_emissions'] and worker['research_only'] is True,
         'bounded existing measured worker')
    registry = json.loads(bound(worker['registry']))
    for role in ('worker','emitter','rustc','native_canonicalizer'):
        bound(worker[role],64*1024**2)
    native = NativeTools(worker['native_canonicalizer'],c['deadline_unix'])
    inputs = RegisteredInputs(worker, registry, native, client(), c['deadline_unix'])
    example = None
    if rows:
        e = json.loads(rows[0][0])['emission']
        actual, raw, descriptor = inputs.describe(e['input_descriptor']['variant_identity'])
        need(asdict(actual) == e['input_descriptor']['context'], 'qualified executable example')
        example = {'context_id':actual.identity,'bytes':len(raw),'partition':descriptor['partition']}
    return inputs, native, descriptors, census, {'schema':'jope.variant-outbox-preflight/v1',
        'state':'passed','outbox_proof':proof,'recorded_and_historical_variant_rows':len(rows),
        'admitted_variant_contexts':128,'example':example,'claims':0,'gp_execution':0,'s3_writes':0,'optimizer_updates':0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True); parser.add_argument('--config-sha',required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    raw = bound({'path':str(Path(args.config).resolve()),'sha256':args.config_sha})
    c = json.loads(raw); root = Path(c['out'])
    bind_campaign_deadline(c['deadline_unix'])
    inputs,native,descriptors,census,proof = preflight(c)
    if args.preflight_only:
        print(json.dumps(proof)); return
    cg = next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    group = Path('/sys/fs/cgroup')/cg.lstrip('/')
    quota, period = (group/'cpu.max').read_text().split()
    need(quota != 'max' and int(quota) <= 2*int(period) and
         int((group/'memory.max').read_text()) <= 2*1024**3 and
         int((group/'memory.swap.max').read_text()) == 0, 'CPU2/RAM2GiB/swap0 ceiling')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '1'
         for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')), 'CPU-only single BLAS')
    # Reject symlinks in every path component; never truncate another owner's lock.
    lockpath = Path(c['lock_path'])
    need(all(not p.is_symlink() for p in (lockpath,*lockpath.parents)), 'nonsymlink lock')
    lock = os.fdopen(os.open(lockpath,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600),'a+b')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # A packet is single-use. Every service exit closes the kernel lock, not its custody files.
    create(root/'execution-started.json',canonical({'config_sha256':args.config_sha,'at_utc_ns':time.time_ns(),
           'deadline_unix':c['deadline_unix'],'automatic_retry':False}))
    create(root/'preflight.json',canonical(proof))
    timing.ACTIVE = timing.Timing(root/'stage-timing.jsonl',group,config_sha256=args.config_sha,
                                 cuda_execution_time_ns=None,gpu_used=False)
    counts = {'claimed':0,'cached_verified':0,'published_verified':0,'unknown_attempt':0,'heads':0,'gets':0,'puts':0}
    results = []
    def report(state):
        progress(root,{'schema':'jope.variant-outbox-progress/v1','at_utc_ns':time.time_ns(),
                 'state':state,'counts':dict(counts),'results':results,'deadline_unix':c['deadline_unix'],
                 'optimizer_updates':0,'optimizer_reward_allowed':False,'timing_events':timing.ACTIVE.count})
    rawclient = client()
    class Budget:
        def __getattr__(self,name): return getattr(rawclient,name)
        def head_object(self,**kw):
            need(counts['heads'] < 128,'HEAD attempt cap'); counts['heads'] += 1
            with timing.span('s3.HEAD',key=kw['Key']): return rawclient.head_object(**kw)
        def get_object(self,**kw):
            need(counts['gets'] < 1024,'GET attempt cap'); counts['gets'] += 1
            with timing.span('s3.GET_headers',key=kw['Key'],version_id=kw.get('VersionId')):
                result = rawclient.get_object(**kw)
            body = result['Body']
            class Body:
                def __enter__(self): return self
                def __exit__(self,*args): body.close()
                def read(self,size=-1):
                    with timing.span('s3.GET_body',key=kw['Key'],requested_bytes=size): return body.read(size)
            result['Body'] = Body(); return result
        def put_object(self,**kw):
            need(counts['puts'] < 32 and kw['Bucket'] == BUCKET and kw['Key'].startswith(PREFIX)
                 and kw.get('IfNoneMatch') == '*','create-only bounded program-library PUT')
            counts['puts'] += 1
            create(root/('put-attempt-%03d.json'%counts['puts']),canonical({'utc_ns':time.time_ns(),
                   'key':kw['Key'],'bytes':kw['ContentLength'],'sha256':kw['Metadata']['sha256']}))
            with timing.span('s3.PUT',key=kw['Key'],bytes=kw['ContentLength']): return rawclient.put_object(**kw)
    budget = Budget(); inputs.client = budget
    store = Library(budget,deadline=c['deadline_unix'],nativecanonical_json=native.canonical_json)
    q = Queue(root/'queue.sqlite',max_jobs=8)
    outbox = Outbox(c['outbox_path'],**{k:v for k,v in c['outbox_config'].items() if k != 'schema'})
    observer = ProgramObserver(q,inputs,store,plan_sha256=args.config_sha,canonicalize=native.canonical_program)
    workerroot = root/'worker'; workerroot.mkdir(exist_ok=False)
    class Worker(ProgramWorker):
        def evaluate(self,job):
            with timing.span('gp.native_emit_rust_parity.total',job=job['id']): return super().evaluate(job)
        def publish(self,job):
            with timing.span('gp.seven_artifact_publication',job=job['id']): return super().publish(job)
    worker = Worker(q,inputs,store,native,c['worker_config'],workerroot,c['deadline_unix'],args.config_sha)
    def failure(claim,message):
        create(root/('unknown-'+claim['lease']['emission_id']+'.json'),canonical({
            'schema':'jope.variant-unknown-attempt/v1','claim':claim,'error':message,
            'config_sha256':args.config_sha,'utc_ns':time.time_ns(),'queue_proof':q.verify(),
            'automatic_retry':False,'numeric_score':None,'s3_failure_custody_verified':False}))
        return None
    consumer = VariantConsumer(outbox,inputs,observer,worker,store,descriptors=descriptors,
        canonicalize=native.canonical_program,blocked_keys=census['blocked_evaluation_keys'],
        failure_custody=failure,owner=c['worker_id'],deadline=c['deadline_unix'],
        completed_keys=census['completed_evaluation_keys'])
    state = 'running'
    try:
        report(state)
        while counts['claimed'] < c['maximum_emissions'] and remaining(c['deadline_unix']) > 610:
            used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            need(used < 1024**3-128*1024**2, 'owned output ceiling before next claim')
            with timing.span('gp.model_outbox.total'):
                result = consumer.once()
            if result['state'] == 'empty': break
            counts['claimed'] += 1; counts[result['state']] += 1
            create(root/(result['emission_id']+'.json'),canonical(result)); results.append(result)
            if result['state'] == 'unknown_attempt':
                state = 'paused_unknown_no_retry'; report(state); break
            report(state)
        if state == 'running': state = 'completed_bounded_packet'
    except BaseException as error:
        state = 'paused_unknown_or_integrity_stop'
        create(root/'fatal.json',canonical({'utc_ns':time.time_ns(),'error':repr(error),
               'automatic_retry':False,'outbox_leases_preserved':True,'numeric_score':None}))
        raise
    finally:
        timing.ACTIVE.event('observation','packet.closed',state=state,queue=q.verify())
        report(state); q.close(); outbox.close(); lock.close()
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
