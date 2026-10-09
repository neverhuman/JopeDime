"""Proof-gated, exact-row training admission; never launches or fabricates training.

prepare(spec_bytes, sha, store, authority_bytes, authority_sha256) verifies a
completed native neural run, physical safetensors/checkpoint chain, read-only
planner journal, measured generated-program S3 closures and current roles. It
returns an immutable JSON-shaped proposal with explicit per-variant refusals.
commit(plan_bytes, plan_sha, spec_bytes, store, authority_bytes, authority_sha256) performs one
atomic exact-row CAS. The caller must obtain authority bytes by executing the
pinned current-role checker immediately before each call. No saved historical
proof is a substitute. Both phases reject stale proofs. No DB constructor from
the generator is invoked, and no generator policy/configuration is changed.
"""
from __future__ import annotations
import hashlib,json,math,sqlite3,struct,time
from pathlib import Path
from urllib.parse import quote
import numpy as np
from .protocol import canonical,digest,sha256,Context
from .program_io import bound
from .training_snapshot_kumos import document,_Reader,mean_baselines
from .program_gate import verified_measurement
from .batch_stream import BatchStream,validate_manifest,decode

SCHEMA='jope.flywheel-training-admission/v1'
PLAN_SCHEMA='jope.flywheel-training-admission-plan/v1'
MAX_TOTAL=256*1024**2
MAX_ROWS=128

def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def path_safe(path):
    p=Path(path);need(p.is_absolute()and '..'not in p.parts and all(not q.is_symlink()for q in(p,*p.parents)),'absolute non-symlink path required');return p

def model_hash(state):
    """Recompute native PolicyLM parameter SHA directly from bounded safetensors."""
    need(8<len(state)<=64*1024**2,'checkpoint cap')
    n=struct.unpack('<Q',state[:8])[0];need(0<n<=1024**2 and 8+n<=len(state),'safetensors header cap')
    header=document(state[8:8+n]);payload=memoryview(state)[8+n:];h=hashlib.sha256();seen=[];count=0
    for key in sorted(header):
        if key=='__metadata__':continue
        item=header[key];offset=item['data_offsets'];shape=item['shape']
        need(type(offset)is list and len(offset)==2 and all(type(x)is int for x in offset)
             and 0<=offset[0]<=offset[1]<=len(payload),'tensor offsets')
        need(type(shape)is list and all(type(x)is int and 0<=x<=10000000 for x in shape),'tensor shape')
        seen.append(tuple(offset))
        if key.startswith('model.'):
            need(item['dtype']=='F32'and math.prod(shape)*4==offset[1]-offset[0],'float32 model shape')
            data=payload[offset[0]:offset[1]];need(np.isfinite(np.frombuffer(data,dtype='<f4')).all(),'finite model parameters')
            name=key[len('model.'):];need(name,'model parameter name')
            h.update((name+'\0F32\0'+','.join(map(str,shape))+'\n').encode());h.update(data);count+=1
    need(count>0,'model parameters required')
    positive=sorted(x for x in seen if x[0]!=x[1]);need(all(a[1]<=b[0]for a,b in zip(positive,positive[1:])),'overlapping tensor ranges')
    return h.hexdigest()

class Local:
    def __init__(self):self.pins={};self.cache={};self.total=0
    def read(self,ref):
        p=str(path_safe(ref['path']));binding=(sha256(ref['sha256']),ref['bytes'])
        need(type(ref['bytes'])is int and 0<ref['bytes']<=64*1024**2,'local artifact cap')
        need(self.pins.setdefault(p,binding)==binding,'same path different artifact')
        if p not in self.cache:
            need(self.total+ref['bytes']<=MAX_TOTAL,'local read budget')
            self.cache[p]=bound(ref,64*1024**2);self.total+=len(self.cache[p])
        return self.cache[p]
    def doc(self,ref):return document(self.read(ref))
    def descriptors(self):return [{'path':p,'sha256':v[0],'bytes':v[1]}for p,v in sorted(self.pins.items())]

def role_identity(current,reference):
    need(current.get('state')=='passed'and current.get('host')=='xbabe2'and
         current.get('schema')=='jope.flywheel-evergreen-encoding-authority/v1','current authority schema/state')
    for key in ('authority_helper_sha256','generator_config_sha256','parent_registry_sha256','support','authority_pins'):
        need(current.get(key)==reference.get(key),'current authority changed: '+key)
    return current

def role_check(raw,expected_sha,reference,now):
    need(sha(raw)==sha256(expected_sha),'authority hash')
    current=document(raw);at=current.get('checked_at_unix')
    need(type(at)in(int,float)and math.isfinite(at)and 0<=now-at<120,'fresh current authority required')
    return role_identity(current,reference)

def planner_read(path,manifest_sha):
    path=path_safe(path)
    for suffix in ('-wal','-shm'):need(not Path(str(path)+suffix).is_symlink(),'planner sidecar symlink')
    db=sqlite3.connect('file:'+quote(str(path))+'?mode=ro',uri=True,isolation_level=None);db.row_factory=sqlite3.Row
    try:
        db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        obj=BatchStream.__new__(BatchStream);obj.db=db;obj.config=obj._get('config')
        need(obj._get('manifest')==manifest_sha,'planner active manifest mismatch')
        anchor=obj._verify();need(anchor['unresolved_batches']==0,'unknown optimizer reservation')
        rows=[{'body':decode(r['body']),'commit':decode(r['commit_body'])}for r in db.execute("SELECT * FROM batches WHERE state='committed' ORDER BY epoch,batch_index")]
        db.execute('COMMIT');return anchor,rows
    finally:db.close()


def custody_bindings(reader,status,result,cfg,config_sha):
    """Read exact S3 custody and reverify observed/outcome chain before row release."""
    closure=status['attempt_custody'];need(closure['schema']=='jope.flywheel-program-attempt-publication-result/v1'
        and closure['status']=='PASS'and closure['helper_sha256']==cfg['owned_sources']['campaign_custody.py'],'completed pinned attempt custody')
    for ref in (closure['snapshot'],closure['receipt']):
        need(ref['bucket']=='veox-jopedime-learning-use2'and ref['key'].startswith('experiments/gp-policy-dl/v2-trainonly/program-attempts/flywheel/v1/'),'attempt custody namespace')
    snapshot=document(reader.read(closure['snapshot']));receipt=document(reader.read(closure['receipt']))
    need(snapshot['schema']=='jope.flywheel-program-attempt-snapshot/v1'and snapshot['source_config']['sha256']==config_sha
         and snapshot['queue_head']==result['queue']and snapshot['emission_head']==result['observations'],'attempt snapshot source/run heads')
    need(receipt['schema']=='jope.flywheel-program-attempt-publication/v1'and receipt['snapshot']==closure['snapshot']
         and receipt['helper_sha256']==closure['helper_sha256']and receipt['config_sha256']==closure['config_sha256']
         and receipt['queue_head']==snapshot['queue_head']and receipt['emission_head']==snapshot['emission_head'],'attempt publication receipt binding')
    previous='0'*64;observed={};outcomes={};pending=set();sequence=0
    for row in snapshot['tables']['program_emissions']:
        sequence+=1;event={k:row[k]for k in('seq','id','at','kind','previous')};event['body']=document(row['body'].encode())
        need(row['seq']==sequence and row['previous']==previous and digest(event)==row['sha'],'archived emission chain')
        previous=row['sha'];identity=row['id']
        if row['kind']=='observed':
            need(identity not in observed,'duplicate archived emission');observed[identity]=event['body'];pending.add(identity)
        elif row['kind']=='outcome':
            need(identity in pending,'outcome without observed emission');pending.remove(identity);outcomes[identity]=event['body']
        else:raise ValueError('unknown archived emission event')
    expected={'events':sequence,'head_sha256':previous,'unresolved_observations':sorted(pending)}
    need(expected==result['observations']and not pending,'unresolved or mismatched emission custody')
    return observed,outcomes


def prepare(spec_bytes,expected_sha,store,*,authority_bytes,authority_sha256,now=time.time):
    need(sha(spec_bytes)==sha256(expected_sha),'admission spec hash');spec=document(spec_bytes)
    need(set(spec)=={'schema','source_pins','database','generator_config','parent_registry','snapshot_status','accepted_loaded_receipt','accepted_batch_manifest','loaded_receipt','batch_manifest',
                    'neural_config','neural_status','neural_result','neural_role_proof','neural_window','planner_path','batch_evidence','checkpoints','authority_reference','rows'}
         and spec['schema']==SCHEMA,'closed admission spec')
    need(type(spec['rows'])is list and 0<len(spec['rows'])<=MAX_ROWS,'bounded requested rows')
    local=Local()
    for ref in spec['source_pins']:local.read(ref)
    generator=local.doc(spec['generator_config']);parents=[document(x)for x in local.read(spec['parent_registry']).splitlines()]
    parent_ref=generator['parent_registry']
    need(parent_ref['sha256']==spec['parent_registry']['sha256']
         and ('bytes'not in parent_ref or parent_ref['bytes']==spec['parent_registry']['bytes']),'generator parent registry binding')
    expected_db=digest({'config_sha':spec['generator_config']['sha256'],'parents':parents,'target':generator['target'],'backlog':generator['backlog_limit']})
    need(spec['database']['path']==str(Path(generator['out'])/'expansion.sqlite')and spec['database']['config_sha256']==expected_db,'generator DB/config binding')
    path_safe(spec['database']['path'])
    reference=local.doc(spec['authority_reference']);role_check(authority_bytes,authority_sha256,reference,now())
    snapshot=local.doc(spec['snapshot_status']);loaded=local.doc(spec['loaded_receipt']);manifest_raw=local.read(spec['batch_manifest'])
    manifest,rows=validate_manifest(manifest_raw,spec['batch_manifest']['sha256'])
    need(snapshot['state']=='loaded'and snapshot['loaded_receipt']==spec['accepted_loaded_receipt']and snapshot['batch_manifest']==spec['accepted_batch_manifest'],'successful actual snapshot closure')
    accepted=local.doc(spec['accepted_loaded_receipt']);accepted_manifest=local.doc(spec['accepted_batch_manifest'])
    # The native reload has a new timestamped role proof. Every other loader
    # receipt field and every membership row must remain exactly the accepted
    # snapshot, including data, labels, eligibility and source identities.
    need({k:v for k,v in accepted.items()if k!='authority_sha256'}=={k:v for k,v in loaded.items()if k!='authority_sha256'},'neural loaded corpus differs from accepted snapshot')
    need({k:v for k,v in accepted_manifest.items()if k!='role_authority_sha256'}=={k:v for k,v in manifest.items()if k!='role_authority_sha256'},'neural membership differs from accepted snapshot')
    need(loaded['state']=='loaded'and loaded['components']=={'kumo_tabular_s':2051}and loaded['snapshot_sha256']==manifest['snapshot_sha256'],'KumoS snapshot binding')
    cfg=local.doc(spec['neural_config']);status=local.doc(spec['neural_status']);result=local.doc(spec['neural_result'])
    config_sha=spec['neural_config']['sha256']
    need(cfg['snapshot_status']==spec['snapshot_status']and cfg['snapshot_acceptance']['sha256']==snapshot['config_sha256'],'neural consumed accepted snapshot')
    native_role=local.doc(spec['neural_role_proof']);role_identity(native_role,reference)
    window=local.doc(spec['neural_window']);at=native_role.get('checked_at_unix')
    need(window['config_sha256']==config_sha and type(at)in(int,float)and math.isfinite(at)
         and window['created_unix']<=at<window['deadline_unix'],'historical native role proof/window binding')
    need(loaded['authority_sha256']==spec['neural_role_proof']['sha256']==manifest['role_authority_sha256'],'native role proof/loaded manifest binding')
    root=path_safe(cfg['owned_root'])
    for key,suffix in (('loaded_receipt','loaded-receipt.json'),('neural_role_proof','role-proof.json'),('neural_window','window.json'),
                       ('batch_manifest','training/batch-manifest.json'),('neural_result','training/result.json'),('neural_status','status.json')):
        need(spec[key]['path']==str(root/suffix),'native owned artifact path: '+key)
    need(spec['planner_path']==str(root/'training/batches.sqlite'),'native planner owned path')
    need(status['schema']=='jope.flywheel-neural-status/v1'and status['state']=='complete'and status['config_sha256']==config_sha
         and status['result']==spec['neural_result']and status.get('error')is None,'successful neural custody/status required')
    need(result['schema']=='jope.flywheel-neural-result/v1'and result['state']=='completed_bounded_qualification'
         and result['config_sha256']==config_sha and result.get('error')is None and result['optimizer_steps']>0,'real completed optimizer run required')
    anchor,committed=planner_read(spec['planner_path'],spec['batch_manifest']['sha256'])
    need(anchor==result['planner']and anchor['committed_batches']==result['optimizer_steps'],'native planner closure')
    proofrefs={r['sha256']:r for r in spec['batch_evidence']};proofs={};train=set();measured=set();dev=set();reader=_Reader(store)
    emissions,outcomes=custody_bindings(reader,status,result,cfg,config_sha)
    checkpoints={}
    for item in spec['checkpoints']:
        cm=local.doc(item['manifest']);state=local.read(item['state'])
        need(cm['schema']=='jope.flywheel-neural-checkpoint/v1'and cm['config_sha256']==config_sha
             and cm['state_sha256']==sha(state)and cm['state_bytes']==len(state)and cm['parameter_sha256']==model_hash(state),'physical checkpoint identity')
        need(cm['model_config']==cfg['neural']['model']and cm['optimizer']=='AdamW'
             and cm['learning_rate']==cfg['neural']['learning_rate']and cm['weight_decay']==.01,'physical checkpoint approved model/optimizer')
        need(cm['step']not in checkpoints,'duplicate checkpoint step');checkpoints[cm['step']]=cm
    need(set(checkpoints)==set(range(len(committed)+1))and checkpoints[0]['parameter_sha256']==committed[0]['commit']['checkpoint_before'],'complete physical checkpoint chain')
    declared={p['batch_id']:p for p in result['proofs']};need(len(declared)==len(result['proofs'])==len(committed),'all committed batch proofs')
    for pair in committed:
        intent,commit=pair['body'],pair['commit']
        # The durable planner stores ordered IDs; the runner's Batch includes
        # those exact manifest rows as well. Reconstruct instead of trusting a
        # caller-provided expanded membership list.
        batch={**intent,'rows':[rows[rid]for rid in intent['ordered_data_run_ids']]}
        bid=batch['batch_id'];decl=declared[bid]
        need(commit['evidence']['origin']=='native_optimizer'and commit['evidence']['event_sha256']==decl['evidence_sha256'],'optimizer proof hash')
        need(decl['evidence_sha256']in proofrefs,'missing optimizer evidence')
        proof=local.doc(proofrefs[decl['evidence_sha256']]);update=proof['update']
        need(proof['schema']=='jope.flywheel-neural-batch/v1'and proof['batch']==batch and proof['source_snapshot_sha256']==loaded['snapshot_sha256'],'batch snapshot identity')
        step=commit['optimizer_step'];need(step in checkpoints and proof['physical_checkpoint']==checkpoints[step],'batch physical checkpoint')
        need(update['checkpoint_before']==commit['checkpoint_before']and update['checkpoint_after']==commit['checkpoint_after']==checkpoints[step]['parameter_sha256']
             and update['checkpoint_before']!=update['checkpoint_after']and math.isfinite(update['loss'])and math.isfinite(update['gradient_norm_before_clip']),'actual changed optimizer tensors')
        slots=proof['draw']['slots'];need([s['context_sha256']for s in slots]==[r['payload']['context_sha256']for r in batch['rows']],'native draw actual context membership')
        for row,slot in zip(batch['rows'],slots):
            rid=row['data_run_id'];need(rows[rid]==row and row['partition']=='train'and row['status']=='eligible','optimizer membership/role')
            cid=row['payload']['context_sha256'];plan=loaded['plans'][cid]
            if slot['stream']=='group':
                need(plan['stream']=='group'and len(slot['members'])==8 and len(set(slot['members']))==8
                     and [p for p,role in zip(slot['members'],slot['roles'])if role=='teacher']==plan['teachers']
                     and len(slot['roles'])==8 and set(slot['roles'])<=set(('teacher','comparator','failure'))
                     and slot['family_id']==row['parent_lineage_sha256']and slot['dataset_hash']==row['data_run']['variant_identity']
                     and len(plan['teachers'])==4 and set(slot['members'])<=set(plan['labelled']),'actual eligible4teacher/group8 support')
                train.add(rid)
        for measurement in proof['measurements']:
            need(measurement['state']=='ready','generated program unresolved')
            rid=measurement['data_run_id'];need(rid in rows,'generated unknown data_run');row=rows[rid];cid=row['payload']['context_sha256']
            need(measurement['context_id']==cid and measurement['phase']==('training'if row['partition']=='train'else'validation'),'generated phase/context')
            wrapper=document(reader.read(measurement['record']));dsl=reader.read(wrapper['artifacts']['program']).decode()
            observation=measurement.get('readback',measurement.get('observation'));need(observation is not None,'actual observer event required')
            oid=observation['observation_id'];need(oid in emissions and outcomes.get(oid)==observation and observation['state']=='ready','measured observer outcome custody')
            emission=emissions[oid];need(emission['dsl']==dsl and emission['variant_identity']==row['data_run']['variant_identity']
                 and emission['phase']==measurement['phase']and emission['checkpoint_sha256']==commit['checkpoint_before']
                 and emission['step']==step and emission['producer']=='PolicyLM-autoregressive-finite-trie-v1','actual model-emission binding')
            ctx=Context(**loaded['contexts'][cid]['context']);actual=verified_measurement(reader,ctx,dsl,wrapper)
            need(actual==measurement['measurement'],'actual generated S3measurement differs')
            mean_baselines(reader.read(wrapper['artifacts']['input']),document(reader.read(wrapper['artifacts']['evaluation'])))
            measured.add(rid)
        proofs[bid]=decl
    for point in result['loss_history']:
        if point['batch']<=0:continue
        for role in ('train','validation'):
            report=point[role];need(report['dropout']is False and report['teacher_members_frozen']is True and report['objective']=='successor-v1'and math.isfinite(report['loss']),'comparable real readout')
        # Explicit IDs were added to the native runner; do not infer a whole panel from aggregate losses.
        dev.update(point['validation']['evaluated_context_ids'])
    support={r['variant_identity']:r for r in reference['support']};accepted=[];refused=[];seen=set()
    for proposed in spec['rows']:
        need(set(proposed)=={'sequence','parent','ordinal','identity','content_sha','metadata_sha','data_run_id'}and proposed['identity']not in seen,'closed unique CAS row')
        need(type(proposed['sequence'])is int and proposed['sequence']>=0 and type(proposed['ordinal'])is int and proposed['ordinal']>=0,'integer sequence/ordinal')
        seen.add(proposed['identity']);rid=proposed['data_run_id'];need(rid in rows,'requested row absent from full manifest');row=rows[rid];cid=row['payload'].get('context_sha256')
        need(row['data_run']['variant_identity']==proposed['identity']and row['parent_lineage_sha256']==proposed['parent']and row['data_run']['dataset_sha256']==proposed['content_sha'],'CAS row/data_run binding')
        reason=None
        if row['status']!='eligible':reason='NOT_ELIGIBLE'
        elif cid not in loaded['plans']or loaded['plans'][cid]['stream']!='group':reason='NOT_GROUP_ELIGIBLE'
        elif rid not in measured:reason='NO_ACTUAL_GENERATED_PROGRAM_MEASUREMENT'
        elif row['partition']=='train'and rid not in train:reason='NO_ACTUAL_GROUP_OPTIMIZER_EXPOSURE'
        elif row['partition']=='validation'and cid not in dev:reason='NO_ACTUAL_FIXED_DEV_EVALUATION'
        elif proposed['identity']not in support:reason='NO_CURRENT_ROLE_SUPPORT'
        if reason:refused.append({'row':proposed,'reason':reason})
        else:
            dataset_pub=document(reader.read(loaded['contexts'][cid]['source_publications']['dataset_publication']))
            need(dataset_pub['variant_identity']==proposed['identity']and dataset_pub['content_sha256']==proposed['content_sha']
                 and dataset_pub['metadata_sha256']==proposed['metadata_sha'],'CAS source metadata/content binding')
            accepted.append({'row':proposed,'partition':row['partition'],'context_id':cid})
    need(accepted,'no proven rows to admit')
    role_check(authority_bytes,authority_sha256,reference,now())
    return {'schema':PLAN_SCHEMA,'spec_sha256':expected_sha,'database':spec['database'],'source_pins':local.descriptors(),
            'authority_reference':reference,'snapshot_sha256':loaded['snapshot_sha256'],'neural_config_sha256':config_sha,
            'planner':anchor,'accepted':accepted,'refused':refused,'source_manifest_rows':len(rows),'observed_measurements':len(measured),
            'proofs':list(proofs.values()),'local_bytes_read':local.total,'s3_bytes_read':reader.total,'database_modified':False}


def commit(plan_bytes,expected_sha,*,spec_bytes,store,authority_bytes,authority_sha256,now=time.time):
    """Single all-or-nothing CAS; exact retries return existing admission evidence."""
    need(sha(plan_bytes)==sha256(expected_sha),'plan hash');plan=document(plan_bytes)
    need(plan['schema']==PLAN_SCHEMA and plan['database_modified']is False and 0<len(plan['accepted'])<=MAX_ROWS,'prepared plan required')
    # A caller-supplied plan is not itself execution evidence. Reverify the exact
    # closed source specification and all physical/journal/S3 proof before CAS.
    fresh=prepare(spec_bytes,plan['spec_sha256'],store,authority_bytes=authority_bytes,authority_sha256=authority_sha256,now=now)
    need(canonical(fresh)==canonical(plan),'admission proof changed after preparation')
    for ref in plan['source_pins']:bound(ref,64*1024**2)
    role_check(authority_bytes,authority_sha256,plan['authority_reference'],now())
    path=path_safe(plan['database']['path']);need(path.is_file(),'generator DB must already exist')
    for suffix in ('-wal','-shm'):need(not Path(str(path)+suffix).is_symlink(),'generator sidecar symlink')
    db=sqlite3.connect('file:'+quote(str(path))+'?mode=rw',uri=True,timeout=10,isolation_level=None);db.row_factory=sqlite3.Row
    evidence=canonical({'schema':SCHEMA,'plan_sha256':expected_sha,'plan':plan});proofsha=sha(evidence)
    try:
        db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE')
        need(db.execute('SELECT sha FROM config WHERE id=1').fetchone()['sha']==plan['database']['config_sha256'],'DB config changed')
        role_check(authority_bytes,authority_sha256,plan['authority_reference'],now())
        db.execute("CREATE TABLE IF NOT EXISTS dl_training_admissions_v1(identity TEXT PRIMARY KEY,sequence INTEGER UNIQUE NOT NULL,proof_sha TEXT NOT NULL,evidence BLOB NOT NULL)")
        changed=[];identical=[]
        for item in plan['accepted']:
            want=item['row'];actual=db.execute('SELECT *FROM variants WHERE sequence=?',(want['sequence'],)).fetchone()
            need(actual is not None and all(actual[k]==want[k]for k in('sequence','parent','ordinal','identity','content_sha','metadata_sha')),'CAS row identity changed')
            old=db.execute('SELECT *FROM dl_training_admissions_v1 WHERE identity=?',(want['identity'],)).fetchone()
            if actual['state']=='admitted':
                need(old is not None and old['proof_sha']==proofsha and bytes(old['evidence'])==evidence,'conflicting existing admission');identical.append(want['identity']);continue
            need(actual['state']=='published'and old is None,'CAS row is not published/unadmitted')
            db.execute('INSERT INTO dl_training_admissions_v1 VALUES(?,?,?,?)',(want['identity'],want['sequence'],proofsha,evidence))
            keys=('sequence','parent','ordinal','identity','content_sha','metadata_sha')
            sql="UPDATE variants SET state='admitted' WHERE "+' AND '.join(k+'=?'for k in keys)+" AND state='published'"
            need(db.execute(sql,tuple(want[k]for k in keys)).rowcount==1,'CAS lost race');changed.append(want['identity'])
        db.execute('COMMIT')
        return {'schema':'jope.flywheel-training-admission-commit/v1','plan_sha256':expected_sha,'proof_sha256':proofsha,
                'newly_admitted':changed,'already_identical':identical,'released_capacity':len(changed),'refused':plan['refused']}
    except BaseException:
        if db.in_transaction:db.execute('ROLLBACK')
        raise
    finally:db.close()
