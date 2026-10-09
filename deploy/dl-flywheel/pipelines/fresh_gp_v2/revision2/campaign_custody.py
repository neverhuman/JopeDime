"""Lossless, bounded custody of program observations; never a score producer.

Source is one SQLite read transaction. No source Queue/Observations constructor
is called. Normal SQLite WAL/SHM read coordination is possible. A pinned external
head prevents silently exporting a truncated or advanced journal. All events are
retained or the operation fails; no event sampling and no numeric failure labels.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import time
from jope_flywheel.protocol import Context, canonical, digest
from jope_flywheel.queue import Queue
from jope_flywheel.program_observer import Observations

BUCKET = 'veox-jopedime-learning-use2'
PREFIX = 'experiments/gp-policy-dl/v2-trainonly/program-attempts/flywheel/v1/'
MAX_BYTES = 64 * 1024**2
MAX_ROWS = 8192
TABLES = {'program_emissions':('seq','body'), 'transitions':('seq','detail'),
          'jobs':('id','payload'), 'dependencies':('job,dependency',None),
          'program_inputs':('context_id','descriptor'), 'gp_observations':('id',None)}

def need(value, reason):
    if not value: raise ValueError(reason)

def sha(body): return hashlib.sha256(body).hexdigest()

class Clock:
    def __init__(self, deadline):
        self.deadline = deadline
        self.monotonic = time.monotonic() + max(0, deadline-time.time())
    def tick(self):
        if time.time() >= self.deadline or time.monotonic() >= self.monotonic:
            raise TimeoutError('attempt custody shared deadline')

def safe_path(path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'absolute owned source path')
    for item in (*reversed(path.parents), path):
        need(not item.is_symlink(), 'source symlink refused')
    return path

def read(path, cap=MAX_BYTES):
    path = safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular source')
        with os.fdopen(fd, 'rb', closefd=False) as stream: body = stream.read(cap+1)
        after = os.fstat(fd)
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) ==
             (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and len(body)==before.st_size,
             'source changed during read')
        return body
    finally: os.close(fd)

def bound(ref, cap=MAX_BYTES):
    body = read(ref['path'],cap)
    need(sha(body)==ref['sha256'] and len(body)==ref['bytes'], 'source reference hash/size')
    return body

def create(path, body):
    path = safe_path(path)
    with path.open('xb') as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def source_receipts(root, jobs, clock):
    receipts=[]; total=0
    for job in jobs:
        clock.tick()
        need(job['stage'] in ('evaluate','publish'), 'only program pipeline queue supported')
        key=json.loads(job['payload'])['evaluation_key']
        need(len(key)==64 and all(c in '0123456789abcdef' for c in key), 'evaluation key')
        need(job['id']=='gp-'+job['stage']+'-'+key, 'fixed job identity')
        if job['state'] not in ('completed','failed','needs_reconcile'): continue
        if job['state']=='needs_reconcile' and not (root/(job['id']+'-failure.json')).exists(): continue
        path = root/(job['id']+'-failure.json') if job['state'] in ('failed','needs_reconcile') else (
            root/job['id']/'result.json' if job['stage']=='evaluate' else root/(job['id']+'.json'))
        body=read(path,128*1024); total+=len(body); need(total<=16*1024**2,'receipt total cap')
        if job['state']!='needs_reconcile': need(sha(body)==job['receipt_sha'], 'terminal receipt chain binding')
        doc=json.loads(body)
        if job['state'] in ('failed','needs_reconcile'):
            need(doc['schema']=='jope.flywheel-worker-failure/v1' and doc['job']==job['id'] and
                 doc['stage']==job['stage'] and type(doc['error']) is str, 'failure receipt identity')
            need(not ({'score','reward','metrics','measurement'} & set(doc)), 'failure cannot supply numeric score')
        elif job['stage']=='evaluate':
            need(doc['schema']=='jope.flywheel-worker-result/v1' and doc['state'] in ('executed','cached'),
                 'actual evaluation result receipt')
        else:
            need(doc['schema']=='jope.flywheel-worker-publication/v1' and doc['state'] in ('published','cached'),
                 'publication receipt')
        receipts.append({'job':job['id'],'sha256':sha(body),'bytes':len(body),'body_utf8':body.decode(),'queue_receipt_committed':job['state']!='needs_reconcile'})
    return receipts

def verify_contexts(rows, registry, descriptor_pins):
    catalog={r['identity']:r for r in registry['inputs']}
    need(len(catalog)==len(registry['inputs']), 'registry duplicates')
    need(set(descriptor_pins)=={r['context_id'] for r in rows}, 'complete registered context pin closure')
    contexts={}
    for row in rows:
        obj=json.loads(row['descriptor']); context=Context(**obj['context']); item=catalog[obj['variant_identity']]
        need(canonical(obj).decode()==row['descriptor'] and digest(obj)==row['descriptor_sha']==descriptor_pins[row['context_id']],
             'registered descriptor hash')
        need(context.identity==row['context_id'] and row['variant_identity']==obj['variant_identity'], 'registered context identity')
        need(obj['schema']=='jope.flywheel-program-input/v1' and obj['research_only'] is True and
             obj['teaching_eligible'] is False and obj['outer_test_rows']==0, 'research context scope')
        need(context.dataset_sha256==item['content_sha'] and obj['metadata_sha256']==item['metadata_sha'] and
             obj['parent_lineage_sha256']==item['parent'] and obj['dataset_s3']==item['s3'], 'registry/source/context closure')
        contexts[row['context_id']]=obj
    return contexts

def summarize(tables, contexts, receipts):
    observed={}; outcomes={}; phases={}; keys=set()
    for row in tables['program_emissions']:
        body=json.loads(row['body'])
        if row['kind']=='observed':
            need(row['id'] not in observed,'observation identity reused after outcome')
            observed[row['id']]=body
            phase=body.get('phase'); phase=phase if phase in ('training','validation') else 'invalid_or_missing'
            phases[phase]=phases.get(phase,0)+1
        else:
            need(row['id'] in observed and row['id'] not in outcomes, 'unique outcome closure')
            need(body.get('state') in ('error','pending','ready'), 'known observation outcome')
            need(body.get('teaching_eligible') is False and body.get('optimizer_reward_allowed') is False, 'research-only outcome')
            if body['state']=='error':
                need(not ({'score','reward','metrics','measurement'} & set(body)), 'error cannot supply numeric score')
            else:
                req=observed[row['id']]; ctx=Context(**req['context']); desc=contexts[ctx.identity]
                need(req['variant_identity']==desc['variant_identity'] and req['descriptor_sha256']==digest(desc),
                     'successful observation registered closure')
                need(body['observation_id']==row['id'],'outcome request identity')
                key=body['evaluation_key'];keys.add(key)
                if body['state']=='pending': need(body.get('measurement') is None,'pending has no measurement')
                else:
                    measurement=body['measurement']
                    need(measurement['context_id']==ctx.identity and measurement['evaluation_key']==key and
                         measurement['status']=='completed' and measurement['measurement']=='executed', 'ready measurement identity')
            outcomes[row['id']]=body
    for row in tables['gp_observations']:
        need(row['id']==digest({'key':row['evaluation_key'],'phase':row['phase'],
             'checkpoint':row['checkpoint_sha'],'step':row['step']}), 'gate observation identity')
        need(any(r.get('phase')==row['phase'] and r.get('checkpoint_sha256')==row['checkpoint_sha'] and
                 r.get('step')==row['step'] for r in observed.values()), 'gate observation without durable emission')
    jobs=tables['jobs']; byid={r['id']:r for r in jobs}
    deps={(r['job'],r['dependency']) for r in tables['dependencies']}
    for job in jobs:
        payload=json.loads(job['payload']);key=payload['evaluation_key'];keys.add(key)
        if job['stage']=='evaluate':
            context=Context(**payload['context']); need(context.identity in contexts,'job registered context')
            program=payload['program'];ph=sha(program.encode())
            need(ph==payload['program_sha256'] and key==digest({'schema':'jope.program-evaluation-key/v1',
                 'context':payload['context'],'program_sha256':ph}), 'job exact program/context identity')
            need(('gp-publish-'+key,job['id']) in deps and 'gp-publish-'+key in byid,'atomic publication dependency')
        else: need(('gp-publish-'+key,'gp-evaluate-'+key) in deps,'publication dependency closure')
    docs={r['job']:json.loads(r['body_utf8']) for r in receipts}
    executions=sum(job['stage']=='evaluate' and job['state']=='completed' and docs[job['id']]['state']=='executed' for job in jobs)
    failures=[{'job':job['id'],'stage':job['stage'],'state':job['state'],'reason':job['reason'],
               'receipt_sha256':job['receipt_sha']} for job in jobs if job['state'] in ('failed','needs_reconcile')]
    return {'observations':len(observed),'phases':phases,'emission_events':len(tables['program_emissions']),
        'observation_errors':sum(o['state']=='error' for o in outcomes.values()),
        'cache_hit_observations':sum(o['state']=='ready' for o in outcomes.values()),
        'pending_observations':sum(o['state']=='pending' for o in outcomes.values()),
        'unresolved_observations':sorted(set(observed)-set(outcomes)),
        'distinct_evaluation_keys':len(keys),'completed_unique_executions':executions,
        'completed_publications':sum(j['stage']=='publish' and j['state']=='completed' for j in jobs),
        'failed_jobs':sum(j['state']=='failed' for j in jobs),
        'unknown_external_outcomes':sum(j['state']=='needs_reconcile' for j in jobs),
        'evaluation_job_attempts':sum(j['attempts'] for j in jobs if j['stage']=='evaluate'),
        'failures_and_unknowns':failures,
        'execution_count_semantics':'Completed executed receipts only; failed/unknown attempts are separate, not measured GP results.'}

def freeze(config, clock):
    clock.tick()
    source_body=bound(config['source_config']);source=json.loads(source_body)
    registry_body=bound(config['registry']);registry=json.loads(registry_body)
    anchor_body=bound(config['anchor']);anchor=json.loads(anchor_body)
    need(anchor['config_sha256']==config['source_config']['sha256'],'anchor source config')
    need(source['registry']==config['registry'],'source registry pin')
    root=safe_path(source['owned_root']); dbpath=safe_path(root/'data/queue.sqlite')
    need(json.loads(read(root/'data/identity.json'))=={'config_sha256':config['source_config']['sha256']},'source runtime identity')
    db=sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True,isolation_level=None,timeout=2)
    db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
    db.set_progress_handler(lambda: int(time.time()>=clock.deadline or time.monotonic()>=clock.monotonic),1000)
    try:
        db.execute('BEGIN');tables={};bytes_total=0
        for table,(order,text_column) in TABLES.items():
            clock.tick(); count=db.execute('SELECT count(*) FROM '+table).fetchone()[0]
            need(count<=MAX_ROWS,'table row cap; no partial export')
            if text_column:
                size=db.execute('SELECT coalesce(sum(length(CAST('+text_column+' AS BLOB))),0) FROM '+table).fetchone()[0]
                bytes_total+=size;need(bytes_total<=MAX_BYTES//2,'source text cap; no partial export')
            tables[table]=[dict(r) for r in db.execute('SELECT * FROM '+table+' ORDER BY '+order)]
        q=object.__new__(Queue);q.db=db;j=object.__new__(Observations);j.db=db
        queue_head=q.verify();emission_head=j.verify()
        need(queue_head==anchor['queue'] and emission_head==anchor['observations'],'external anchored heads mismatch')
        contexts=verify_contexts(tables['program_inputs'],registry,config['descriptor_pins'])
        receipts=source_receipts(root/'data',tables['jobs'],clock)
        counts=summarize(tables,contexts,receipts)
        db.execute('COMMIT')
    finally:
        if db.in_transaction: db.execute('ROLLBACK')
        db.close()
    snapshot={'schema':'jope.flywheel-program-attempt-snapshot/v1','source_config':config['source_config'],
        'source_registry':config['registry'],'source_anchor':config['anchor'],'source_anchor_json':anchor,
        'source_registry_json':registry,'queue_head':queue_head,'emission_head':emission_head,
        'tables':tables,'terminal_receipts':receipts,'counts':counts,
        'teaching_eligible':False,'optimizer_reward_allowed':False,
        'scope':'Source observations and outcomes preserved verbatim. Numeric fields in existing ready outcomes are source measurements; errors/failures receive no numeric score. This archive is custody, not score admission.',
        'source_access':'SQLite mode=ro, query_only, single read transaction; normal WAL/SHM read coordination may occur. No source SQL, event or CAS mutation.'}
    body=canonical(snapshot);need(len(body)<=MAX_BYTES,'snapshot byte cap');clock.tick()
    return body,snapshot

class Store:
    """Same create-only + exact-VersionId GET protocol as measured S3 library."""
    def __init__(self, client, clock): self.client=client;self.clock=clock;self.put_attempts=0
    def get(self, ref):
        self.clock.tick();need(ref['bucket']==BUCKET and ref['key'].startswith(PREFIX),'attempt namespace')
        need(type(ref['bytes']) is int and 0<ref['bytes']<=MAX_BYTES,'object cap')
        need(type(ref['version_id']) is str and ref['version_id'] not in ('','null'),'exact version required')
        result=self.client.get_object(Bucket=BUCKET,Key=ref['key'],VersionId=ref['version_id'])
        need(result.get('VersionId')==ref['version_id'] and result.get('ContentLength')==ref['bytes'],'exact version/size')
        data=bytearray()
        with result['Body'] as stream:
            while True:
                self.clock.tick();part=stream.read(min(1024**2,ref['bytes']-len(data)+1))
                if not part: break
                data.extend(part);need(len(data)<=ref['bytes'],'GET exceeded cap')
        self.clock.tick();need(len(data)==ref['bytes'] and sha(data)==ref['sha256'],'exact version SHA')
        return bytes(data)
    def put(self,key,body):
        self.clock.tick();need(key.startswith(PREFIX) and type(body) is bytes and 0<len(body)<=MAX_BYTES,'own bounded object')
        need(self.put_attempts<2,'two PUT attempt maximum');self.put_attempts+=1
        try:
            response=self.client.put_object(Bucket=BUCKET,Key=key,Body=body,ContentLength=len(body),
                IfNoneMatch='*',Metadata={'sha256':sha(body)})
        except Exception as error:
            self.clock.tick()
            if getattr(error,'response',{}).get('Error',{}).get('Code') not in ('PreconditionFailed','412'): raise
            response=self.client.head_object(Bucket=BUCKET,Key=key)
        ref={'bucket':BUCKET,'key':key,'version_id':response.get('VersionId'),'sha256':sha(body),'bytes':len(body)}
        self.get(ref);return ref

def publish(snapshot_body, snapshot, config_sha, helper_sha, out, store):
    out=safe_path(out); need(out.is_dir(),'new owned output directory required')
    create(out/'snapshot.json',snapshot_body)
    snapshot_ref=store.put(PREFIX+'snapshots/'+sha(snapshot_body)+'.json',snapshot_body)
    create(out/'snapshot-s3.json',canonical(snapshot_ref))
    receipt={'schema':'jope.flywheel-program-attempt-publication/v1','snapshot':snapshot_ref,
        'config_sha256':config_sha,'helper_sha256':helper_sha,'queue_head':snapshot['queue_head'],
        'emission_head':snapshot['emission_head'],'counts':snapshot['counts'],
        'teaching_eligible':False,'optimizer_reward_allowed':False,'numeric_failure_scores_created':0}
    body=canonical(receipt);create(out/'receipt.json',body)
    receipt_ref=store.put(PREFIX+'receipts/'+sha(body)+'.json',body)
    result={'schema':'jope.flywheel-program-attempt-publication-result/v1','status':'PASS',
        'snapshot':snapshot_ref,'receipt':receipt_ref,'counts':snapshot['counts'],
        'put_attempts':store.put_attempts,'config_sha256':config_sha,'helper_sha256':helper_sha}
    create(out/'result.json',canonical(result));return result
