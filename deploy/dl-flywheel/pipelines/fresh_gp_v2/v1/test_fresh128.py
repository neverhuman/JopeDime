"""New compatibility/guard fixtures only; no native fitting or network access."""
import ast
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

H=Path(__file__).resolve().parent
sys.path[:0]=[str(H),str(H/'runtime')]
import numpy as np
from jope_flywheel.protocol import Context,canonical,digest
from jope_flywheel.program_gate import request_identity
from jope_flywheel.program_io import sha
from jope_flywheel.queue import Queue
from jope_flywheel.program_worker import ProgramWorker
from global_attempts import Attempts,History,alias,ownership
from execution_boundary import ExecutionBoundary
from authority_lease import AuthorityLease
from staged_inputs import VersionCache
from receipt_gate import validate_current_receipts,PIPELINE,GENERATOR_PUBLISHER,WORKER,WEIGHT,SCHEMA,PREFIX,BUCKET
from cold_config import resource_gate,validate


class Native:
    def canonical_program(self,s):return ' '.join(s.split())


def context(n=1):
    return Context('regression',f'{n:064x}','2'*64,'cut'+str(n),'3'*64,'4'*64,'5'*64,'6'*64,
                   'rmse','prepared_synthetic_target_units','minimize')


class Cases(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.ctx=context();self.program='(ridge alpha=1)';self.owner='shard0'
        self.descriptor={'variant_identity':'a'*64,'partition':'validation','parent_lineage_sha256':'b'*64,
                         'teaching_eligible':False,'raw_units':False}
        self.descriptors=[{'context_id':context(i+1).identity,'variant_identity':f'{i+10:064x}',
                           'descriptor_sha256':'d'*64,'partition':'train'}for i in range(128)]
        self.descriptors[0]={'context_id':self.ctx.identity,'variant_identity':'a'*64,
                            'descriptor_sha256':digest(self.descriptor),'partition':'validation'}
        self.config={'campaign_sha256':'c'*64,'selection_sha256':'f'*64,'ownership_producer_sha256':'e'*64,
                     'history_sources':[],'work_deadline_unix':time.time()+600,'global_deadline_unix':time.time()+900,
                     'shard_ids':[self.owner],'context_owners':{r['context_id']:self.owner for r in self.descriptors},
                     'context_descriptors':self.descriptors,'source_pins':[]}
        self.proof={'schema':'jope.fresh-gp-context-ownership/v1','passed':True,'host':'xbabe3',
                    'campaign_sha256':'c'*64,'producer_sha256':'e'*64,'history_census_sha256':digest([]),
                    'selection_sha256':'f'*64,'checked_at_unix':time.time(),'lease_expires_unix':time.time()+300,
                    'global_deadline_unix':self.config['global_deadline_unix'],
                    'legacy_variant_workers_closed':True,'legacy_launch_controllers_closed':True,
                    'active_xbabe2_source_kind':'native_s1_context','active_native_context_overlap':[],
                    'other_variant_context_overlap':[],'all_legacy_units_physically_closed':True,
                    'context_descriptors':self.descriptors}
    def tearDown(self):self.temp.cleanup()
    def attempts(self):
        a=Attempts(self.root/'attempts.sqlite',self.config,create=True)
        self.addCleanup(a.close);a.reserve_contexts(self.descriptors,self.proof,self.owner);return a
    def queue(self):
        q=Queue(self.root/'jobs.sqlite');self.addCleanup(q.close)
        key=request_identity(self.ctx,self.program)
        payload={'context':asdict(self.ctx),'program':self.program,'evaluation_key':key}
        q.enqueue('gp-evaluate-'+key,'evaluate','9'*64,payload)
        q.enqueue('gp-publish-'+key,'publish','9'*64,{'evaluation_key':key},['gp-evaluate-'+key])
        return q,q.claim('fixture',['evaluate'],lease_seconds=300,max_running=1)
    def boundary(self,attempts,q,history=None):
        outer=self
        class Inputs:
            parents={'b'*64:{}}
            def describe(self,identity):return outer.ctx,b'raw',deepcopy(outer.descriptor)
            class Lease:
                def check(self,*args):return True
            lease=Lease()
        class Hist:
            pin='h'*64
            def check(self):return history or {}
        return ExecutionBoundary(self.config,q,Inputs(),attempts,Hist(),self.owner,lambda:deepcopy(self.proof))
    def test_whole_context_duplicate_launch_and_alias_refused(self):
        a=self.attempts()
        with self.assertRaises(sqlite3.IntegrityError):a.reserve_contexts(self.descriptors,self.proof,self.owner)
        with self.assertRaises(FileExistsError):Attempts(self.root/'attempts.sqlite',self.config,create=True)
        before=a.verify();bad=deepcopy(self.descriptors);bad[0]['variant_identity']='f'*64
        with self.assertRaises(ValueError):a.reserve_contexts(bad,self.proof,self.owner)
        self.assertEqual(a.verify(),before)
    def test_context_role_not_reassigned_or_optimizer_admitted(self):
        a=self.attempts();a.assert_owner(self.ctx,self.descriptor,self.owner)
        wrong=dict(self.descriptor,partition='train')
        with self.assertRaises(ValueError):a.assert_owner(self.ctx,wrong,self.owner)
        self.assertEqual(a.db.execute('SELECT partition FROM contexts WHERE context_id=?',(self.ctx.identity,)).fetchone()[0],'validation')
        self.assertIs(self.descriptor['teaching_eligible'],False)
    def test_single_key_cannot_invoke_twice_even_other_owner_or_reopen(self):
        a=self.attempts();q,j=self.queue();guard=self.boundary(a,q)
        limit=guard.before_fit(j,self.ctx,self.program,b'raw',self.descriptor)
        self.assertLessEqual(limit,j['lease_deadline']+.001)
        with self.assertRaises(ValueError):guard.before_fit(j,self.ctx,self.program,b'raw',self.descriptor)
        b=Attempts(self.root/'attempts.sqlite',self.config);self.addCleanup(b.close)
        with self.assertRaises(ValueError):b.begin_attempt(self.ctx,self.program,'other',j,{},'h'*64)
        self.assertEqual(b.verify()['held_attempts'],1)
    def test_parallel_duplicate_key_native_callback_only_once(self):
        a=self.attempts();q,j=self.queue()
        def one(_):
            registry=Attempts(self.root/'attempts.sqlite',self.config)
            try:
                registry.begin_attempt(self.ctx,self.program,self.owner,j,{},'h'*64)
                return 'native_callback_permitted'
            except ValueError:return 'permanent_hold'
            finally:registry.close()
        with ThreadPoolExecutor(max_workers=2)as pool:result=list(pool.map(one,range(2)))
        self.assertEqual(sorted(result),['native_callback_permitted','permanent_hold'])
        self.assertEqual(a.verify()['held_attempts'],1)
    def test_null_key_canonical_historical_alias_is_permanent_hold(self):
        row={'id':'unknown-alias','state':'needs_reconcile','payload':{'context':asdict(self.ctx),
             'program':'  (ridge   alpha=1) ','evaluation_key':None}}
        r=alias(row,Native(),{})
        self.assertEqual(r['evaluation_key'],request_identity(self.ctx,self.program))
        a=self.attempts();q,j=self.queue();guard=self.boundary(a,q,{r['evaluation_key']:[r]})
        with self.assertRaises(ValueError):guard.before_fit(j,self.ctx,self.program,b'raw',self.descriptor)
        self.assertEqual(a.db.execute('SELECT state FROM attempts').fetchone()[0],'prior_hold')
        with self.assertRaises(ValueError):a.begin_attempt(self.ctx,self.program,self.owner,j,{},'new-source')
    def test_all_historical_states_are_holds_and_unresolved_alias_blocks(self):
        for state in ('completed','failed','needs_reconcile','running','pending'):
            r=alias({'id':state,'state':state,'payload':{'context':asdict(self.ctx),'program':self.program}},Native(),{})
            self.assertEqual(r['evaluation_key'],request_identity(self.ctx,self.program))
        with self.assertRaises(ValueError):alias({'id':'x','state':'unknown','payload':{'program':self.program}},Native(),{})
    def test_context_target_difference_yields_distinct_native_identity(self):
        c=Context(**{**asdict(self.ctx),'target_sha256':'7'*64})
        self.assertNotEqual(c.identity,self.ctx.identity)
        self.assertNotEqual(request_identity(c,self.program),request_identity(self.ctx,self.program))
    def test_history_rehash_and_full_unfiltered_census(self):
        refs=[]
        for i in range(20):
            obj={'schema':'jope.fresh-gp-readonly-history/v1','physical_readonly':True,'source_verified':True,
                 'capture_complete':True,'kind':'variant_queue'if i<18 else('original_outbox'if i==18 else'continuation_outbox'),
                 'source_row_count':0,'records':[]}
            p=self.root/f'history{i}.json';p.write_bytes(canonical(obj));refs.append({'path':str(p),'sha256':sha(p.read_bytes())})
        h=History(refs,Native(),{});self.assertEqual(h.check(),{})
        Path(refs[0]['path']).write_bytes(b'{}')
        with self.assertRaises(ValueError):h.check()
        with self.assertRaises(ValueError):History(refs[:-1],Native(),{})
    def test_registry_manual_state_change_rejected(self):
        a=self.attempts();a.db.execute("UPDATE contexts SET partition='train'")
        with self.assertRaises(ValueError):a.verify()
    def test_lease_expiry_during_costly_input_preparation_zero_intent(self):
        a=self.attempts();q,j=self.queue();guard=self.boundary(a,q)
        def expire(identity):
            q.db.execute('UPDATE jobs SET lease_deadline=? WHERE id=?',(time.time()-1,j['id']))
            return self.ctx,b'raw',self.descriptor
        guard.inputs.describe=expire
        with self.assertRaises(ValueError):guard.before_fit(j,self.ctx,self.program,b'raw',self.descriptor)
        self.assertEqual(a.verify()['held_attempts'],0)
    def test_job_body_token_or_source_change_refuses_before_fit(self):
        a=self.attempts();q,j=self.queue();guard=self.boundary(a,q)
        for field,value in [('lease_token','changed'),('payload','{}')]:
            original=q.db.execute('SELECT '+field+' FROM jobs WHERE id=?',(j['id'],)).fetchone()[0]
            q.db.execute('UPDATE jobs SET '+field+'=? WHERE id=?',(value,j['id']))
            with self.assertRaises(ValueError):guard.before_fit(j,self.ctx,self.program,b'raw',self.descriptor)
            q.db.execute('UPDATE jobs SET '+field+'=? WHERE id=?',(original,j['id']))
        path=self.root/'source.py';path.write_bytes(b'original')
        self.config['source_pins']=[{'path':str(path),'sha256':sha(b'original')}];path.write_bytes(b'changed')
        with self.assertRaises(ValueError):guard.before_fit(j,self.ctx,self.program,b'raw',self.descriptor)
        self.assertEqual(a.verify()['held_attempts'],0)
    def test_authority_expired_between_preparation_and_fit_is_held(self):
        a=self.attempts();q,j=self.queue();guard=self.boundary(a,q);calls=0
        def lease(*args):
            nonlocal calls
            calls+=1
            if calls==2:raise ValueError('authority became protected')
        guard.inputs.lease.check=lease
        with self.assertRaises(ValueError):guard.before_fit(j,self.ctx,self.program,b'raw',self.descriptor)
        self.assertEqual(a.verify()['held_attempts'],1)
    def test_ownership_excludes_active_other_variant_and_expiry(self):
        for key,value in [('other_variant_context_overlap',[self.ctx.identity]),('legacy_launch_controllers_closed',False),
                          ('active_xbabe2_source_kind','variant'),('lease_expires_unix',time.time()-1)]:
            p=deepcopy(self.proof);p[key]=value
            with self.assertRaises(ValueError):ownership(p,self.config,self.descriptors,time.time())
    def test_global_work_and_monotonic_minimum(self):
        a=self.attempts();q,j=self.queue();guard=self.boundary(a,q)
        guard.mono_end=time.monotonic()-1
        with self.assertRaises(ValueError):guard.before_fit(j,self.ctx,self.program,b'raw',self.descriptor)
        self.assertEqual(a.verify()['held_attempts'],0)
    def test_actual_worker_hook_prevents_native_command_after_replay(self):
        a=self.attempts();q,j=self.queue();guard=self.boundary(a,q)
        worker=ProgramWorker(q,guard.inputs,type('Store',(),{'lookup':lambda s,k:None})(),Native(),
                             {'schema':'jope.fresh-variant-gp-campaign/v1',**{k:{'path':'/unused','sha256':'0'*64}for k in ('worker','emitter','rustc')}},
                             self.root,time.time()+600,'9'*64,execution_guard=guard)
        worker.resolve=lambda c:(b'raw',self.descriptor)
        guard.inputs.describe=lambda i:(_ for _ in ()).throw(ValueError('late role failure'))
        with patch('jope_flywheel.program_worker.bound',return_value=b'x'),patch('jope_flywheel.program_worker.run')as run:
            with self.assertRaises(ValueError):worker.evaluate(j)
            run.assert_not_called()
    def test_cached_bad_measurement_never_falls_through_to_fit(self):
        a=self.attempts();q,j=self.queue();guard=self.boundary(a,q);key=request_identity(self.ctx,self.program)
        worker=ProgramWorker(q,guard.inputs,type('Store',(),{'lookup':lambda s,k:{'evaluation_key':key}})(),Native(),
                             {'schema':'jope.fresh-variant-gp-campaign/v1'},self.root,time.time()+600,'9'*64,execution_guard=guard)
        worker.resolve=lambda c:(b'raw',self.descriptor)
        with patch('jope_flywheel.program_worker.verified_measurement',side_effect=ValueError('bad cached closure')),patch('jope_flywheel.program_worker.run')as run:
            with self.assertRaises(ValueError):worker.evaluate(j)
            run.assert_not_called()
        self.assertEqual(a.verify()['held_attempts'],0)
    def test_worker_requires_boundary_for_new_schema(self):
        with self.assertRaises(ValueError):ProgramWorker(None,None,None,None,{'schema':'jope.fresh-variant-gp-campaign/v1'},self.root,time.time()+1,'0'*64)
    def test_numerical_and_parity_functions_identical(self):
        old=ast.parse((H/'provenance/program_worker.py').read_text());new=ast.parse((H/'runtime/jope_flywheel/program_worker.py').read_text())
        old_functions={n.name:ast.dump(n,include_attributes=False)for n in old.body if isinstance(n,ast.FunctionDef)}
        new_functions={n.name:ast.dump(n,include_attributes=False)for n in new.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(old_functions,new_functions)
        old_methods={n.name:ast.dump(n,include_attributes=False)for n in next(n for n in old.body if isinstance(n,ast.ClassDef)and n.name=='ProgramWorker').body if isinstance(n,ast.FunctionDef)}
        new_methods={n.name:ast.dump(n,include_attributes=False)for n in next(n for n in new.body if isinstance(n,ast.ClassDef)and n.name=='ProgramWorker').body if isinstance(n,ast.FunctionDef)}
        for name in ('resolve','once'):self.assertEqual(old_methods[name],new_methods[name])
        self.assertIn('TOLERANCE = 1e-9',(H/'runtime/jope_flywheel/program_worker.py').read_text())
    def test_full_evaluate_body_diff_only_execution_boundary_and_deadline(self):
        old=ast.parse((H/'provenance/program_worker.py').read_text());new=ast.parse((H/'runtime/jope_flywheel/program_worker.py').read_text())
        def method(tree):return next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='evaluate')
        class Normalize(ast.NodeTransformer):
            def visit_If(self,node):
                if ast.unparse(node.test)=='self.execution_guard is not None':return None
                return self.generic_visit(node)
            def visit_Assign(self,node):
                if any(isinstance(t,ast.Name)and t.id=='fit_deadline'for t in node.targets):return None
                return self.generic_visit(node)
            def visit_Name(self,node):
                return ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='deadline',ctx=node.ctx)if node.id=='fit_deadline'else node
        self.assertEqual(ast.dump(method(old),include_attributes=False),ast.dump(Normalize().visit(method(new)),include_attributes=False))


class AuthorityTests(unittest.TestCase):
    def test_exact_roster_validation_and_stale_support(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td).resolve();(root/'proofs').mkdir()
            selected=[{'identity':f'{i:064x}','parent_lineage_sha256':f'{i+128:064x}','partition':'validation'if i<10 else'train'}for i in range(128)]
            c={'selection':selected,'selected_count':128,'selection_sha256':digest(selected),'campaign_sha256':'a'*64,
               'authority_root':str(root),'authority_pointer':str(root/'current.json'),'authority_deadline_unix':time.time()+600,
               'generator_config':{'sha256':'b'*64},'parent_registry_sha256':'c'*64,'authority_pins':[]}
            p={'schema':'jope.fresh-variant-gp-authority/v1','passed':True,'host':'xbabe2','campaign_sha256':'a'*64,
               'selection_sha256':digest(selected),'generator_config_sha256':'b'*64,'parent_registry_sha256':'c'*64,
               'authority_pins':[],'checked_at_unix':time.time(),'deadline_unix':c['authority_deadline_unix'],
               'support':[{**x,'protected_overlap':0,'parent_record_sha256':'d'*64,'view_id':str(i)}for i,x in enumerate(selected)]}
            def save(p):
                b=canonical(p);path=root/'proofs'/(sha(b)+'.json');path.write_bytes(b)
                (root/'current.json').write_bytes(canonical({'path':str(path),'sha256':sha(b),'bytes':len(b)}))
            save(p);self.assertEqual(sum(x['partition']=='validation'for x in AuthorityLease(c).check()['support']),10)
            for mutate in ('relabel','missing','duplicate','expired','future','protected'):
                q=deepcopy(p)
                if mutate=='relabel':q['support'][0]['partition']='train'
                elif mutate=='missing':q['support'].pop()
                elif mutate=='duplicate':q['support'][0]=q['support'][1]
                elif mutate=='expired':q['checked_at_unix']-=121
                elif mutate=='future':q['checked_at_unix']+=5
                else:q['support'][0]['protected_overlap']=1
                save(q)
                with self.assertRaises(ValueError,msg=mutate):AuthorityLease(c).check()


class ColdTests(unittest.TestCase):
    def test_unqualified_actual_template_cannot_be_launched(self):
        c=json.loads((H/'config.template.json').read_bytes())
        with self.assertRaisesRegex(ValueError,'not approved'):validate(c,time.time())
        self.assertEqual(len(c['selection']),128)
        self.assertEqual(sum(r['partition']=='validation'for r in c['selection']),10)
        self.assertIsNone(c['created_unix'])
    def test_residual_encoder_and_gp_reservations_are_charged(self):
        g=1024**3
        own=[{'memory_max':12*g,'memory_current':2*g,'observed':True},
             {'memory_max':4*g,'memory_current':g,'observed':True}]
        with self.assertRaises(ValueError):resource_gate(40*g,own,1)
        self.assertEqual(resource_gate(41*g,own,1)['required_available_bytes'],41*g)
        with self.assertRaises(ValueError):resource_gate(44*g,own,2)
        with self.assertRaises(ValueError):resource_gate(100*g,[{'memory_max':g,'memory_current':0,'observed':False}],1)


class ReceiptTests(unittest.TestCase):
    def fixture(self):
        ident='a'*64;part='validation';rawsha='b'*64;metasha='c'*64
        def ref(name,body):return {'bucket':BUCKET,'key':PREFIX+'datasets/flywheel/v1/'+ident+'/'+name,
            'version_id':'exact1','sha256':sha(body),'bytes':len(body)}
        files={n:ref(n,n.encode())for n in ('encoder.jtt1.gz','metadata.json','generation.json')}
        s3=dict(files);s3['receipt']=ref('prior.json',b'prior')
        meta={'content_sha256':rawsha,'target_sha256':'d'*64,'source_target_sha256':'e'*64,'parent_lineage_sha256':'f'*64,
              'partition':part,'rows':80,'features':3,'inner_counts':[40,40],'outer_test_rows':0}
        proof={'source_sha256':WORKER,'producer_phase_sealed':True,'original_coordinator_state':'complete',
               'finalization_custody':None,'manifest':{'sha256':'1'*64},'coordinator_ledger':{'event_count':9},
               'phase_ledgers':{'ofm':{'event_count':67}}}
        arr=io.BytesIO();np.save(arr,np.ones(2051,dtype='<f4'),allow_pickle=False);vector=arr.getvalue();vr=ref('encodings/kumo_tabular_s.npy',vector)
        qb={'passed':True,'tests':{str(i):True for i in range(6)}};qref={'path':'q.json','sha256':sha(canonical(qb)),'bytes':len(canonical(qb))}
        runtime={'sha256':'2'*64,'path':'/qualified/runtime.json'}
        record={'variant_identity':ident,'component':'kumo_tabular_s','dimension':2051,'phase':'ofm','manifest_sha256':'1'*64,
                'host':'xbabe3','input_raw_sha256':rawsha,'target_sha256':meta['target_sha256'],'partition':part,
                'parent_lineage_sha256':meta['parent_lineage_sha256'],'outer_test_rows':0,'teaching_eligible':False,
                'source_input':files['encoder.jtt1.gz'],'source_metadata':files['metadata.json'],
                'producer':{'weight':{'repo':'nvidia/Kumo-Tabular','revision':'3c3e10bbdb590ace29e7026847f92db3c603096d',
                                     'file':'small/regressor.pt','sha256':WEIGHT},'runtime_inventory':runtime},
                'qualification':qref,'output':vr}
        rr={'sha256':sha(canonical(record)),'bytes':len(canonical(record))}
        receipt={'schema':SCHEMA,**meta,'variant_identity':ident,'sequence':938,'metadata_sha256':metasha,'host':'xbabe3',
                 'required_encoders':['kumo_tabular_s'],'completed_encoders':['kumo_tabular_s'],'missing_encoders':[],
                 'roster_version':'kumo-s-only-v1','teaching_eligible':False,'lift_eligible':False,'new_gp_labels':0,
                 'native_context_admission':'unqualified','prepared':True,'raw_units':False,'rights':'research',
                 'publisher_sha256':'3'*64,'source_proof':proof,'prior_dataset_publication':{'receipt':s3['receipt'],'objects':files},
                 'objects':[vr],'encoder_records':{'kumo_tabular_s':{'record':record,'record_ref':rr,
                        'vector_ref':vr,'qualification':{'ref':qref,'body':qb}}}}
        row={'identity':ident,'sequence':938,'content_sha':rawsha,'metadata_sha':metasha,'parent':meta['parent_lineage_sha256'],
             'files':files,'s3':s3,'kumo':{'receipt':ref('kumo.json',canonical(receipt)+b'\n'),'publisher_sha256':'3'*64,
                'source_proof_sha256':digest(proof),'manifest_sha256':'1'*64,'runtime_inventory_sha256':digest(runtime),'vector':vr}}
        pub={'schema':'jope.flywheel-evergreen-dataset-publication/v1','variant_identity':ident,'content_sha256':rawsha,
             'metadata_sha256':metasha,'parent_lineage_sha256':meta['parent_lineage_sha256'],'teaching_eligible':False,'new_gp_labels':0,
             'publication_helper_sha256':GENERATOR_PUBLISHER,'pipeline_config_sha256':PIPELINE,'required_encoders':['kumo_tabular_s'],
             'qualification':{'partition':part,'deterministic_replay_exact':True,'target_sha256':meta['target_sha256'],'outer_test_rows':0}}
        config={'required_encoders':['kumo_tabular_s'],'generator_config':{'sha256':'4'*64}}
        class Cache:
            def get(self,ref,identity,name):return vector if name.endswith('.npy') else canonical(receipt)+b'\n'
        return config,row,meta,{'config_sha256':'4'*64},pub,Cache(),time.time()+60,receipt
    def test_current_receipt_role_is_validation_not_admission(self):
        f=self.fixture();r=validate_current_receipts(*f[:-1]);self.assertEqual(r['partition'],'validation')
        self.assertIs(r['training_admission'],False)
    def test_wrong_versions_targets_weights_qualification_and_roster_fail(self):
        for case in ('version','target','weight','qualification','roster','publisher','phase','runtime'):
            f=self.fixture();r=f[-1]
            if case=='version':r['prior_dataset_publication']['receipt']=dict(r['prior_dataset_publication']['receipt'],version_id='other')
            elif case=='target':r['target_sha256']='0'*64
            elif case=='weight':r['encoder_records']['kumo_tabular_s']['record']['producer']['weight']['sha256']='0'*64
            elif case=='qualification':r['encoder_records']['kumo_tabular_s']['qualification']['body']['passed']=False
            elif case=='roster':r['required_encoders'].append('disallowed')
            elif case=='publisher':r['publisher_sha256']='0'*64
            elif case=='phase':r['source_proof']['producer_phase_sealed']=False
            else:r['encoder_records']['kumo_tabular_s']['record']['producer']['runtime_inventory']['sha256']='0'*64
            with self.assertRaises(ValueError,msg=case):validate_current_receipts(*f[:-1])
    def test_exact_version_read_wrong_version_and_tampered_cache_refused(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td).resolve();data=b'abc';ident='a'*64
            ref={'bucket':BUCKET,'key':PREFIX+'datasets/flywheel/v1/'+ident+'/metadata.json','version_id':'v1','sha256':sha(data),'bytes':3}
            class Client:
                version='bad'
                def get_object(self,**kw):return {'VersionId':self.version,'ContentLength':3,'Body':io.BytesIO(data)}
            client=Client();cache=VersionCache(root/'cache',client,time.time()+60)
            with self.assertRaises(ValueError):cache.get(ref,ident,'metadata.json')
            client.version='v1';self.assertEqual(cache.get(ref,ident,'metadata.json'),data)
            next(cache.root.glob('*.bytes')).write_bytes(b'bad')
            with self.assertRaises(ValueError):cache.get(ref,ident,'metadata.json')


if __name__=='__main__':unittest.main()
