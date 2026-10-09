from dataclasses import asdict,replace
import json,tempfile,unittest,struct
import numpy as np
from jope_flywheel.augmentation import array_sha256
from jope_flywheel.program_gate import ProgramGate,verified_measurement,request_identity,bytes_sha,SCHEMA,ARTIFACTS
from jope_flywheel.protocol import Context,canonical
from jope_flywheel.queue import Queue
from jope_flywheel.score_join import build_records

class Store:
    def __init__(self):self.entry=None;self.blobs={};self.fail=False;self.nativecanonical_json=canonical
    def lookup(self,key):
        if self.fail:raise OSError('S3 unavailable')
        return self.entry
    def get_version(self,ref):return self.blobs[ref['key']],ref['version_id']

class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.q=Queue(self.tmp.name+'/q.sqlite');self.store=Store()
        self.ctx=Context('regression','a'*64,'b'*64,'cut','c'*64,'d'*64,'e'*64,'f'*64,'rmse','target','minimize')
        self.dsl='(pipe X (ridge))';self.g=ProgramGate(self.q,self.store,plan_sha256='1'*64,canonicalize=lambda x:x.strip())
    def tearDown(self):self.q.close();self.tmp.cleanup()
    def ensure(self,**kw):
        args=dict(phase='training',checkpoint_sha256='2'*64,step=1);args.update(kw);return self.g.ensure(self.ctx,self.dsl,**args)
    def fixture(self):
        labels=np.array([0]*25+[1]*25,dtype=np.uint8);y=np.arange(50,dtype='<f8')
        raw=b'JTT1'+struct.pack('<IIIB',50,0,1,2)+labels.tobytes()+np.column_stack([y,y]).astype('<f8').tobytes()
        self.ctx=replace(self.ctx,dataset_sha256=bytes_sha(raw),target_sha256=array_sha256(y),split_sha256=bytes_sha(labels.tobytes()))
        ph=bytes_sha(self.dsl.encode());key=request_identity(self.ctx,self.dsl)
        ev={'schema':'jope.s1-evaluation/v1','status':'completed','fold_protocol':'jope.s1-two-fold/v1','inner_cv_folds':2,'input_semantics':'unscaled_fit_rows_v1',
            'program':{'dsl':self.dsl,'program_sha256':ph},'inner_cv':[{'schema':'jope.teacher-fit/v2','kind':'inner_cv','index':i,'score_rows':25,'train_rows':25,
                'metrics':{'rmse':1.},'predictions':(y[labels==i]-1).tolist(),'targets':y[labels==i].tolist(),'serialized_prediction_replay_exact':True}for i in range(2)]}
        bodies={'input':raw,'program':self.dsl.encode(),'evaluation':canonical(ev),'rust_source':b'fn main() {}','rust_executable':b'compiled fixture'}
        parity={'schema':'jope.flywheel-rust-parity/v1','passed':True,'executed':True,'folds_verified':2,'evaluation_key':key,'program_sha256':ph,'evaluator_sha256':self.ctx.evaluator_sha256}
        parity['native_provenance']={'schema':'jope.flywheel-native-execution-provenance/v1','evaluator_sha256':self.ctx.evaluator_sha256,
            'input_sha256':self.ctx.dataset_sha256,'program_sha256':ph,'evaluation_sha256':bytes_sha(bodies['evaluation']),
            'source_commit':'a'*40,'execution_timestamp':123}
        parity.update({n+'_sha256':bytes_sha(bodies[n])for n in ['input','evaluation','rust_source','rust_executable']});bodies['rust_parity']=canonical(parity)
        bodies['score_records']=build_records(self.ctx,ph,key,[1.,1.],bytes_sha(bodies['rust_parity']),'a'*40,123,self.store.nativecanonical_json)
        refs={k:{'bucket':'fixture','key':k,'version_id':'v1','sha256':bytes_sha(v),'bytes':len(v)}for k,v in bodies.items()}
        self.store.blobs=bodies;self.store.entry={'schema':SCHEMA,'context':asdict(self.ctx),'program_sha256':ph,'evaluation_key':key,'status':'completed','measurement':'executed','artifacts':refs}
    def test_novel_program_both_phases_is_single_durable_work(self):
        a=self.ensure();b=self.ensure(phase='validation',step=2)
        self.assertEqual(a.evaluation_key,b.evaluation_key);self.assertEqual(self.q.snapshot()['total_jobs'],2)
        self.assertEqual(self.q.db.execute('SELECT count(*) FROM gp_observations').fetchone()[0],2)
        with self.assertRaises(RuntimeError):a.require_measurement()
    def test_changed_target_and_evaluator_are_new_contexts(self):
        a=self.ensure();self.ctx=replace(self.ctx,target_sha256='3'*64);b=self.ensure()
        self.assertNotEqual(a.evaluation_key,b.evaluation_key);self.assertEqual(self.q.snapshot()['total_jobs'],4)
    def test_lookup_failure_does_not_dispatch(self):
        self.store.fail=True
        with self.assertRaises(OSError):self.ensure()
        self.assertEqual(self.q.snapshot()['total_jobs'],0)
    def test_complete_receipt_closure_required(self):
        self.fixture();r=self.ensure();self.assertEqual(r.state,'ready');self.assertEqual(r.require_measurement()['inner_fold_rmse'],[1.,1.]);self.assertEqual(self.q.snapshot()['total_jobs'],0)
        self.store.entry['artifacts'].pop('rust_source')
        with self.assertRaises(ValueError):self.ensure()
    def test_changed_object_or_wrong_context_refused(self):
        self.fixture();self.store.blobs['rust_source']+=b'changed'
        with self.assertRaises(ValueError):self.ensure()
        self.fixture();self.store.entry['context']['target_sha256']='0'*64
        with self.assertRaises(ValueError):self.ensure()
    def test_prediction_cannot_be_reward(self):
        self.fixture();self.store.entry['measurement']='predicted'
        with self.assertRaises(ValueError):self.ensure()
    def test_duplicate_fold_and_changed_support_refused(self):
        self.fixture();ev=json.loads(self.store.blobs['evaluation']);ev['inner_cv'][1]=ev['inner_cv'][0]
        body=canonical(ev);self.store.blobs['evaluation']=body;self.store.entry['artifacts']['evaluation'].update(sha256=bytes_sha(body),bytes=len(body))
        with self.assertRaises(ValueError):self.ensure()
    def test_unsupported_task_metric_direction_refused_before_dispatch(self):
        for field,value in [('task','classification'),('metric','accuracy'),('direction','maximize')]:
            original=self.ctx;self.ctx=replace(self.ctx,**{field:value})
            with self.assertRaises(ValueError):self.ensure()
            self.ctx=original
        self.assertEqual(self.q.snapshot()['total_jobs'],0)
    def test_unknown_input_semantics_refused(self):
        self.fixture();ev=json.loads(self.store.blobs['evaluation']);ev.pop('input_semantics')
        body=canonical(ev);self.store.blobs['evaluation']=body;self.store.entry['artifacts']['evaluation'].update(sha256=bytes_sha(body),bytes=len(body))
        with self.assertRaises(ValueError):self.ensure()
    def test_native_callback_and_score_records_required(self):
        self.fixture();self.store.nativecanonical_json=None
        with self.assertRaisesRegex(ValueError,'native canonical'):self.ensure()
        self.store.nativecanonical_json=canonical;self.fixture();self.store.entry['artifacts'].pop('score_records')
        with self.assertRaises(ValueError):self.ensure()
    def test_canonical_score_value_and_context_cannot_be_forged(self):
        for field,value in [('value',2.),('evaluator_sha256','0'*64),('benchmark_valid',True),('cut_id','0'*64)]:
            self.fixture();rows=json.loads(self.store.blobs['score_records']);rows[0][field]=value
            body=canonical(rows);self.store.blobs['score_records']=body;self.store.entry['artifacts']['score_records'].update(sha256=bytes_sha(body),bytes=len(body))
            with self.assertRaises(ValueError):self.ensure()
    def test_oversized_role_refused_before_reading_objects(self):
        self.fixture();self.store.entry['artifacts']['score_records']['bytes']=3*1024**2
        self.store.get_version=lambda ref:(_ for _ in ()).throw(AssertionError('must not fetch'))
        with self.assertRaisesRegex(ValueError,'memory bound'):self.ensure()
if __name__=='__main__':unittest.main()
