"""Exact-context novelty gate shared by training and validation producers.

The supplied store must do an authoritative lookup: None means a confirmed absent
entry, never a network/authentication error. A result wrapper is accepted only
after every required object is fetched at its recorded S3 VersionId and checked.
This module does not claim to be wired into a trainer merely by being imported.
"""
from dataclasses import asdict,dataclass
import hashlib,json,math,struct,time
import numpy as np
from .augmentation import array_sha256
from .protocol import Context,canonical,digest,sha256
from .score_join import validate_records

SCHEMA='jope.flywheel-program-result/v1'
MAX_BYTES=64*1024**2
ARTIFACTS=('input','program','evaluation','rust_source','rust_executable','rust_parity','score_records')
ARTIFACT_LIMITS={'input':8*1024**2,'program':65536,'evaluation':4*1024**2,'rust_source':4*1024**2,
                'rust_executable':16*1024**2,'rust_parity':4*1024**2,'score_records':2*1024**2}

def need(ok,msg):
    if not ok:raise ValueError(msg)
def bytes_sha(b):return hashlib.sha256(b).hexdigest()

def request_identity(context,program):
    need(type(context)is Context,'exact typed context required')
    need(context.task=='regression' and context.metric=='rmse' and context.direction=='minimize','S1 gate only supports regression/RMSE/minimize')
    need(type(program)is str and 0<len(program.encode())<=65536,'bounded canonical DSL required')
    return digest({'schema':'jope.program-evaluation-key/v1','context':asdict(context),'program_sha256':bytes_sha(program.encode())})

def object_bytes(store,ref):
    need(type(ref)is dict and set(ref)=={'bucket','key','version_id','sha256','bytes'},'exact versioned object reference required')
    sha256(ref['sha256'])
    need(type(ref['bytes'])is int and 0<ref['bytes']<=MAX_BYTES,'artifact byte bound')
    need(type(ref['version_id'])is str and ref['version_id'] not in ('','null'),'versioned object required')
    need(type(ref['bucket'])is str and bool(ref['bucket']) and type(ref['key'])is str and bool(ref['key']),'object location required')
    # Store enforces its fixed bucket/prefix and reports the returned VersionId.
    body,returned_version=store.get_version(ref)
    need(type(body)is bytes and returned_version==ref['version_id'],'exact destination version readback required')
    need(len(body)==ref['bytes'] and bytes_sha(body)==ref['sha256'],'artifact size/hash mismatch')
    return body

def verified_measurement(store,context,program,entry):
    need(entry['schema']==SCHEMA and entry['context']==asdict(context),'score context/evaluator/target/split mismatch')
    key=request_identity(context,program);ph=bytes_sha(program.encode())
    need(entry['evaluation_key']==key and entry['program_sha256']==ph,'program identity mismatch')
    need(entry['status']=='completed' and entry['measurement']=='executed','completed measured result required')
    need(set(entry['artifacts'])==set(ARTIFACTS),'program, score, Rust code/executable/parity closure required')
    need(all(type(entry['artifacts'][role].get('bytes')) is int and 0<entry['artifacts'][role]['bytes']<=ARTIFACT_LIMITS[role]
         for role in ARTIFACTS),'role-specific artifact memory bound')
    bodies={k:object_bytes(store,entry['artifacts'][k])for k in ARTIFACTS}
    need(bodies['program']==program.encode(),'stored DSL differs')
    raw=bodies['input'];need(len(raw)>=17 and raw[:4]==b'JTT1','exact JTT1 input required')
    n,nt,p,k=struct.unpack('<IIIB',raw[4:17])
    need(50<=n<2000 and nt==0 and 1<=p<=512 and k==2 and len(raw)==17+n+8*n*(p+1),'stripped bounded input required')
    labels=np.frombuffer(raw[17:17+n],dtype=np.uint8)
    data=np.frombuffer(raw[17+n:],dtype='<f8').reshape(n,p+1)
    need(np.isfinite(data).all() and set(labels.tolist())=={0,1} and min(np.bincount(labels))>=2,'finite two-fold input required')
    need(bytes_sha(raw)==context.dataset_sha256 and array_sha256(data[:,-1])==context.target_sha256
         and bytes_sha(labels.tobytes())==context.split_sha256,'input/target/fold identity mismatch')
    evaluation=json.loads(bodies['evaluation']);parity=json.loads(bodies['rust_parity'])
    need(evaluation['schema']=='jope.s1-evaluation/v1' and evaluation['status']=='completed','native S1 result required')
    need(evaluation['fold_protocol']=='jope.s1-two-fold/v1' and evaluation['inner_cv_folds']==2 and evaluation.get('input_semantics')=='unscaled_fit_rows_v1' and 'official_test' not in evaluation,'two-fold training-only protocol required')
    need(evaluation['program']['dsl']==program and evaluation['program']['program_sha256']==ph,'native result program mismatch')
    folds=evaluation['inner_cv'];need(len(folds)==2,'two observed folds required')
    scores=[]
    for index,fold in enumerate(folds):
        expected=data[labels==index,-1]
        need(fold['schema']=='jope.teacher-fit/v2' and fold['kind']=='inner_cv' and fold['index']==index,'ordered unique native folds required')
        need(fold['score_rows']==len(expected) and fold['train_rows']==n-len(expected),'native fold support mismatch')
        value=fold['metrics']['rmse'];pred=fold['predictions'];target=fold['targets']
        need(type(value)in(int,float) and math.isfinite(value) and value>=0,'finite actual RMSE required')
        need(len(pred)==len(target)>=2 and all(type(x)in(int,float) and math.isfinite(x)for x in pred+target),'finite prediction/target closure required')
        need(np.array_equal(np.asarray(target,dtype='<f8'),expected),'native evaluation targets differ from exact input membership')
        observed=math.sqrt(math.fsum((a-b)**2 for a,b in zip(pred,target))/len(pred))
        need(math.isclose(value,observed,rel_tol=1e-10,abs_tol=1e-12),'RMSE does not reconstruct from predictions')
        need(fold['serialized_prediction_replay_exact'] is True,'native model replay qualification required')
        scores.append(value)
    need(parity['schema']=='jope.flywheel-rust-parity/v1' and parity['passed'] is True,'executed Rust parity required')
    for name in ('input','evaluation','rust_source','rust_executable'):
        need(parity[name+'_sha256']==entry['artifacts'][name]['sha256'],'Rust parity artifact binding mismatch')
    need(parity['evaluation_key']==key and parity['program_sha256']==ph and parity['evaluator_sha256']==context.evaluator_sha256,'Rust parity context mismatch')
    need(parity['folds_verified']==2 and parity['executed'] is True,'both emitted-code folds must execute')
    provenance=parity['native_provenance']
    need(provenance['schema']=='jope.flywheel-native-execution-provenance/v1','native execution provenance required')
    need(provenance['evaluator_sha256']==context.evaluator_sha256 and provenance['input_sha256']==context.dataset_sha256
         and provenance['program_sha256']==ph and provenance['evaluation_sha256']==entry['artifacts']['evaluation']['sha256'],
         'native producer provenance identity mismatch')
    score_rows=validate_records(bodies['score_records'],context,ph,key,scores,entry['artifacts']['rust_parity']['sha256'],
        provenance['source_commit'],provenance['execution_timestamp'],getattr(store,'nativecanonical_json',None))
    # No normalization or pooled metric is silently substituted for these rows.
    return {'evaluation_key':key,'program_sha256':ph,'context_id':context.identity,'measurement':'executed',
            'status':'completed','inner_fold_rmse':scores,'evaluation_sha256':entry['artifacts']['evaluation']['sha256'],
            'artifact_versions':{k:v['version_id']for k,v in entry['artifacts'].items()},
            'canonical_score_row_sha256':[r['row_sha256']for r in score_rows],'teaching_eligible':False}

@dataclass(frozen=True)
class GateResult:
    state:str
    evaluation_key:str
    measurement:dict|None=None
    def require_measurement(self):
        if self.state!='ready' or self.measurement is None:raise RuntimeError('GP evaluation/publication pending; do not train on a predicted reward')
        return self.measurement

class ProgramGate:
    def __init__(self,queue,store,*,plan_sha256,canonicalize):
        sha256(plan_sha256);self.queue=queue;self.store=store;self.plan=plan_sha256;self.canonicalize=canonicalize
        queue.db.execute('CREATE TABLE IF NOT EXISTS gp_observations(id TEXT PRIMARY KEY,evaluation_key TEXT NOT NULL,phase TEXT NOT NULL,checkpoint_sha TEXT NOT NULL,step INTEGER NOT NULL,observed REAL NOT NULL)')
    def ensure(self,context,dsl,*,phase,checkpoint_sha256,step):
        need(phase in ('training','validation'),'producer phase required');sha256(checkpoint_sha256)
        need(type(step)is int and step>=0,'nonnegative producer step required')
        program=self.canonicalize(dsl);key=request_identity(context,program)
        event=digest({'key':key,'phase':phase,'checkpoint':checkpoint_sha256,'step':step})
        self.queue.db.execute('INSERT OR IGNORE INTO gp_observations VALUES(?,?,?,?,?,?)',(event,key,phase,checkpoint_sha256,step,time.time()))
        # Lookup errors propagate: failure to read S3 is not evidence of novelty.
        entry=self.store.lookup(key)
        if entry is not None:return GateResult('ready',key,verified_measurement(self.store,context,program,entry))
        payload={'schema':'jope.flywheel-evaluate-request/v1','evaluation_key':key,'context':asdict(context),
                 'program':program,'program_sha256':bytes_sha(program.encode()),'backend':'local_lambda',
                 'require_rust_execution_parity':True,'require_versioned_s3_publication':True}
        self.queue.enqueue_many([
            {'job_id':'gp-evaluate-'+key,'stage':'evaluate','plan_sha256':self.plan,'payload':payload},
            {'job_id':'gp-publish-'+key,'stage':'publish','plan_sha256':self.plan,
             'payload':{'schema':'jope.flywheel-program-publication-request/v1','evaluation_key':key},
             'dependencies':['gp-evaluate-'+key]}])
        return GateResult('pending',key)
