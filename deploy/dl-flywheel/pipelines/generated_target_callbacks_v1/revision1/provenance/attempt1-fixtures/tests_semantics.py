"""NEW offline fixtures; the real Linux canonicalizer remains a live gate.

Synthetic data exercise unchanged program_gate math. The Python fixture native
process tests pipe/argv bounds only; it is not native serde_json parity evidence.
"""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from . import callbacks as C,bindings as B
from .readonly_io import Failure,NativeCanonical,Window,canonical,sha,read_bound
from .vendor.protocol import Context
from .vendor.augmentation import array_sha256
from .vendor.program_gate import verified_measurement,request_identity
from .vendor.score_join import build_records

class FakeNativeBoundary:
    # Tests only: canonical JSON stand-in explicitly does NOT qualify serde_json.
    canonical_json=staticmethod(canonical)


def artifacts():
    folds=np.array([0]*25+[1]*25,dtype=np.uint8);y=np.arange(50,dtype='<f8')
    raw=b'JTT1'+struct.pack('<IIIB',50,0,1,2)+folds.tobytes()+np.column_stack([y,y]).astype('<f8').tobytes()
    ctx=Context('regression',sha(raw),array_sha256(y),'1'*64,B.EVALUATOR,'2'*64,'3'*64,sha(folds.tobytes()),'rmse','prepared_synthetic_target_units','minimize')
    dsl='(pipe X (ridge))';ph=sha(dsl.encode());key=request_identity(ctx,dsl)
    ev={'schema':'jope.s1-evaluation/v1','status':'completed','fold_protocol':'jope.s1-two-fold/v1','inner_cv_folds':2,
        'input_semantics':'unscaled_fit_rows_v1','program':{'dsl':dsl,'program_sha256':ph},
        'inner_cv':[{'schema':'jope.teacher-fit/v2','kind':'inner_cv','index':i,'score_rows':25,'train_rows':25,
         'metrics':{'rmse':1.},'predictions':(y[folds==i]-1).tolist(),'targets':y[folds==i].tolist(),'serialized_prediction_replay_exact':True}for i in range(2)]}
    bodies={'input':raw,'program':dsl.encode(),'evaluation':canonical(ev),'rust_source':b'fixture source','rust_executable':b'fixture compiled proof'}
    parity={'schema':'jope.flywheel-rust-parity/v1','passed':True,'executed':True,'folds_verified':2,'evaluation_key':key,
            'program_sha256':ph,'evaluator_sha256':ctx.evaluator_sha256,
            'native_provenance':{'schema':'jope.flywheel-native-execution-provenance/v1','evaluator_sha256':ctx.evaluator_sha256,
                'input_sha256':ctx.dataset_sha256,'program_sha256':ph,'evaluation_sha256':sha(bodies['evaluation']),'source_commit':'a'*40,'execution_timestamp':123}}
    parity.update({n+'_sha256':sha(bodies[n])for n in('input','evaluation','rust_source','rust_executable')})
    bodies['rust_parity']=canonical(parity)
    bodies['score_records']=build_records(ctx,ph,key,[1.,1.],sha(bodies['rust_parity']),'a'*40,123,canonical)
    refs={k:{'bucket':'veox-jopedime-learning-use2','key':'experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/'+key+'/'+k,
             'version_id':'v1','sha256':sha(v),'bytes':len(v)}for k,v in bodies.items()}
    entry={'schema':'jope.flywheel-program-result/v1','context':asdict(ctx),'program_sha256':ph,'evaluation_key':key,
           'status':'completed','measurement':'executed','artifacts':refs}
    return ctx,dsl,entry,bodies


class SemanticTests(unittest.TestCase):
    def test_real_gate_recomputes_two_folds_through_seven_body_store(self):
        ctx,dsl,entry,bodies=artifacts();store=C.SevenBodies(entry,bodies,FakeNativeBoundary())
        got=verified_measurement(store,ctx,dsl,entry)
        self.assertEqual(got['inner_fold_rmse'],[1.,1.]);self.assertFalse(got['teaching_eligible'])
        self.assertFalse(hasattr(store,'put_object'));self.assertFalse(hasattr(store,'evaluate'))
    def test_changed_context_or_program_cannot_be_verified(self):
        for field in('context','program_sha256'):
            ctx,dsl,e,b=artifacts()
            if field=='context':e[field]['target_sha256']='0'*64
            else:e[field]='0'*64
            with self.assertRaises(ValueError):verified_measurement(C.SevenBodies(e,b,FakeNativeBoundary()),ctx,dsl,e)
    def test_unknown_or_unpassed_parity_is_nonnumeric(self):
        for mutation in('unknown','parity'):
            ctx,dsl,e,b=artifacts()
            if mutation=='unknown':e['status']='unknown_attempt'
            else:
                doc=json.loads(b['rust_parity']);doc['passed']=False;b['rust_parity']=canonical(doc)
                e['artifacts']['rust_parity'].update(sha256=sha(b['rust_parity']),bytes=len(b['rust_parity']))
            with self.assertRaises(ValueError):verified_measurement(C.SevenBodies(e,b,FakeNativeBoundary()),ctx,dsl,e)
    def test_missing_wrong_version_and_poisoned_bodies_refused(self):
        ctx,dsl,e,b=artifacts();store=C.SevenBodies(e,b,FakeNativeBoundary())
        wrong=dict(e['artifacts']['input'],version_id='v2')
        with self.assertRaises(ValueError):store.get_version(wrong)
        for mutation in('missing','poison'):
            bad=dict(b)
            if mutation=='missing':bad.pop('evaluation')
            else:bad['evaluation']+=b' '
            with self.assertRaises(ValueError):C.SevenBodies(e,bad,FakeNativeBoundary())
    def test_score_row_value_forgery_refused(self):
        ctx,dsl,e,b=artifacts();rows=json.loads(b['score_records']);rows[0]['value']=2
        b['score_records']=canonical(rows);e['artifacts']['score_records'].update(bytes=len(b['score_records']),sha256=sha(b['score_records']))
        with self.assertRaises(ValueError):verified_measurement(C.SevenBodies(e,b,FakeNativeBoundary()),ctx,dsl,e)
    def test_supplied_extra_artifact_ref_cannot_be_read(self):
        ctx,dsl,e,b=artifacts();store=C.SevenBodies(e,b,FakeNativeBoundary())
        with self.assertRaises(ValueError):store.get_version(dict(e['artifacts']['program'],key=e['artifacts']['program']['key']+'other'))
    def test_replay_descriptor_rejects_parent_role_context_and_receipt_drift(self):
        ctx,dsl,e,b=artifacts();d={'variant_identity':'4'*64,'context_id':ctx.identity,'context':asdict(ctx),
            'parent_lineage_sha256':'5'*64,'partition':'train','dataset_publication':{'v':1}}
        actual={'parent_lineage_sha256':'5'*64,'partition':'train','dataset_s3':{'receipt':{'v':1}}}
        class Inputs:
            def describe(self,identity):return ctx,b['input'],actual
        B.replay_descriptor(Inputs(),d)
        for field,value in [('partition','validation'),('parent_lineage_sha256','6'*64),('dataset_publication',{'v':2}),('context_id','7'*64)]:
            bad=copy.deepcopy(d);bad[field]=value
            with self.assertRaises(ValueError):B.replay_descriptor(Inputs(),bad)
    def test_only_top_level_no_closure_or_default_callbacks(self):
        for name in('read_version','canonicalize','verify','authority'):
            fn=getattr(C,name);self.assertIsNone(fn.__closure__);self.assertIsNone(fn.__defaults__);self.assertIsNone(fn.__kwdefaults__)
    def test_authority_never_accepts_an_unconfigured_echo(self):
        with self.assertRaisesRegex(ValueError,'configured production'):C.authority({'descriptors':{'forged':True}})
    def test_valid_hex_wrong_weight_license_is_refused_before_file_read(self):
        for key in('audit','license','code'):
            refs={'audit':{'sha256':B.KUMO_AUDIT},'license':{'sha256':B.LICENSE},'code':{'sha256':B.CODE_LICENSE}}
            refs[key]['sha256']='a'*64
            with self.assertRaisesRegex(ValueError,'license pins'):B.license_contract(refs['audit'],refs['license'],refs['code'])
    def test_exact_local_file_refuses_symlink_and_changed_hash(self):
        with tempfile.TemporaryDirectory()as root:
            p=Path(root)/'x';p.write_bytes(b'abc');r={'path':str(p),'sha256':sha(b'abc')};self.assertEqual(read_bound(r),b'abc')
            p.write_bytes(b'abd')
            with self.assertRaises(ValueError):read_bound(r)
            q=Path(root)/'link';q.symlink_to(p)
            with self.assertRaises(ValueError):read_bound({'path':str(q),'sha256':sha(b'abd')})
    def test_production_config_wrong_canonicalizer_refused_without_execution(self):
        # No credentials/client/native creation precedes configuration validation.
        c={'schema':'jope.generated-target-production-callbacks/v1','mode':'selected-qualification','created_unix':100.,'deadline_unix':200.,
           'global_deadline_unix':C.GLOBAL_DEADLINE,'max_gets':64,'max_read_bytes':1024,'runtime':{},'package_files':{},'authority_source_pins':[],
           'selected':['x'],'refs':{k:{'sha256':v}for k,v in C.PINNED_INPUTS.items()},'worker_config':{'registry':{'sha256':B.REGISTRY},
           'generator_config':{'sha256':B.GENERATOR},'native_canonicalizer':{'sha256':'a'*64},'worker':{'sha256':B.EVALUATOR}}}
        for k in('execution_config','kumo_audit','weight_license','code_license','candidate_observation'):c['refs'][k]={'sha256':'a'*64}
        c['refs']['labels_source']={'sha256':'c2554a33554e69ee36f7c9eeab094989609c212b9e5c954ccf6f712c35b0befd'}
        c['refs']['compose_source']={'sha256':'68c20ee69226e7be9bf92ad40e3889c91fddd4af337c65c9d838d8f51d875258'}
        with patch.object(C.time,'time',return_value=150):
            with self.assertRaisesRegex(ValueError,'producer pins'):C._config(canonical(c),sha(canonical(c)))
    def test_encoder_weight_is_exact_not_any_hex(self):
        desc={'variant_identity':'a'*64,'parent_lineage_sha256':'b'*64,'partition':'train','context':{'dataset_sha256':'c'*64,'target_sha256':'d'*64},'encoder_publication':{}}
        r={'component':'kumo_tabular_s','dimension':2051,'variant_identity':'a'*64,'input_raw_sha256':'c'*64,'target_sha256':'d'*64,'partition':'train',
           'outer_test_rows':0,'teaching_eligible':False,'producer':{'weight':{'sha256':'f'*64,'repo':'nvidia/Kumo-Tabular','file':'small/regressor.pt','revision':'3c3e10bbdb590ace29e7026847f92db3c603096d'}}}
        doc={'schema':'jope.flywheel-evergreen-kumos-publication/v1','variant_identity':'a'*64,'parent_lineage_sha256':'b'*64,'partition':'train',
             'encoder_records':{'kumo_tabular_s':{'record':r,'qualification':{'body':{}}}}}
        class Reader:
            def get_version(self,ref):return canonical(doc),'v1'
        with self.assertRaisesRegex(ValueError,'Kumo weight'):C._encoder_binding(Reader(),desc)


class CanonicalPipeTests(unittest.TestCase):
    def fixture(self,body):
        root=tempfile.TemporaryDirectory();self.addCleanup(root.cleanup);p=Path(root.name)/'native'
        p.write_text('#!'+sys.executable+'\n'+body);p.chmod(0o700)
        return NativeCanonical({'path':str(p),'sha256':sha(p.read_bytes())},Window(time.time()+15,2,1024))
    def test_exact_canonical_only_argv_and_json_values(self):
        native=self.fixture("import sys,json,hashlib\nv=json.load(sys.stdin)\nassert sys.argv[1] in ['--canonical-json','--canonical-program']\nif sys.argv[1]=='--canonical-program':v={'dsl':v['dsl'].strip(),'program_sha256':hashlib.sha256(v['dsl'].strip().encode()).hexdigest()}\nprint(json.dumps(v,sort_keys=True,separators=(',',':')))\n")
        self.assertEqual(native.canonical_json({'a':1}),b'{"a":1}')
        self.assertEqual(native.canonical_program(' (pipe X (ridge)) '),'(pipe X (ridge))')
        with self.assertRaises(ValueError):native._call('--evaluate',{})
    def test_stdout_stderr_caps_and_nonzero_exit_fail(self):
        for body in ["import sys;sys.stdout.write('x'*(2*1024**2+2))", "import sys;sys.stderr.write('x'*65537)", "raise SystemExit(3)"]:
            native=self.fixture(body)
            with self.assertRaises(ValueError):native.canonical_json({})
    def test_changed_native_bytes_refused_before_subprocess(self):
        native=self.fixture("print('{}')");Path(native.ref['path']).write_text('changed')
        with self.assertRaises(ValueError):native.canonical_json({})
