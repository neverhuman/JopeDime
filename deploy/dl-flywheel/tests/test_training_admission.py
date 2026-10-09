"""CAS unit tests use a mocked prior proof verifier; no live admission claim.

Physical tensor hashing and readonly real BatchStream validation are tested
separately. Actual native runner acceptance remains a required integration gate.
"""
import copy,hashlib,json,sqlite3,struct,tempfile,time,unittest
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import numpy as np
from jope_flywheel import training_admission as A
from jope_flywheel.protocol import canonical,digest
from jope_flywheel.batch_stream import BatchStream
class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.db=self.root/'db.sqlite';self.now=1000.
        db=sqlite3.connect(self.db);db.executescript('CREATE TABLE config(id INTEGER PRIMARY KEY,sha TEXT);CREATE TABLE variants(sequence INTEGER PRIMARY KEY,parent TEXT,ordinal INTEGER,identity TEXT,state TEXT,content_sha TEXT,metadata_sha TEXT);')
        db.execute('INSERT INTO config VALUES(1,?)',('c'*64,));self.rows=[]
        for i in range(2):
            row={'sequence':i,'parent':digest({'parent':i}),'ordinal':0,'identity':digest({'variant':i}),'content_sha':digest({'content':i}),'metadata_sha':digest({'meta':i}),'data_run_id':digest({'run':i})};self.rows.append(row)
            db.execute('INSERT INTO variants VALUES(?,?,?,?,?,?,?)',(i,row['parent'],0,row['identity'],'published',row['content_sha'],row['metadata_sha']))
        db.commit();db.close()
        self.authority={'schema':'jope.flywheel-evergreen-encoding-authority/v1','state':'passed','host':'xbabe2','checked_at_unix':self.now,'authority_helper_sha256':'a'*64,'generator_config_sha256':'b'*64,'parent_registry_sha256':'c'*64,'support':[],'authority_pins':[]}
        self.plan={'schema':A.PLAN_SCHEMA,'spec_sha256':'d'*64,'database':{'path':str(self.db),'config_sha256':'c'*64},'source_pins':[],'authority_reference':self.authority,'accepted':[{'row':r,'partition':'train','context_id':'e'*64}for r in self.rows],'refused':[],'database_modified':False}
    def tearDown(self):self.tmp.cleanup()
    def commit(self,plan=None):
        plan=plan or self.plan;b=canonical(plan);auth=canonical(self.authority)
        with patch.object(A,'prepare',return_value=plan):return A.commit(b,A.sha(b),spec_bytes=b'fixture',store=None,authority_bytes=auth,authority_sha256=A.sha(auth),now=lambda:self.now)
    def states(self):
        db=sqlite3.connect(self.db)
        try:return db.execute('SELECT state FROM variants ORDER BY sequence').fetchall()
        finally:db.close()
    def test_success_atomic_and_exact_retry(self):
        first=self.commit();self.assertEqual(first['released_capacity'],2);self.assertEqual(self.states(),[('admitted',),('admitted',)])
        second=self.commit();self.assertEqual(second['released_capacity'],0);self.assertEqual(len(second['already_identical']),2)
    def test_changed_proof_collision(self):
        self.commit();p=copy.deepcopy(self.plan);p['refused']=[{'reason':'new'}]
        with self.assertRaisesRegex(ValueError,'conflicting'):self.commit(p)
    def test_second_row_conflict_rolls_back_first_and_table(self):
        db=sqlite3.connect(self.db);db.execute("UPDATE variants SET metadata_sha='changed'WHERE sequence=1");db.commit();db.close()
        with self.assertRaisesRegex(ValueError,'identity changed'):self.commit()
        self.assertEqual(self.states(),[('published',),('published',)])
        db=sqlite3.connect(self.db);self.assertEqual(db.execute("SELECT count(*)FROM sqlite_master WHERE name='dl_training_admissions_v1'").fetchone()[0],0);db.close()
    def test_wrong_db_config_refused(self):
        self.plan['database']['config_sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'config changed'):self.commit()
        self.assertEqual(self.states(),[('published',),('published',)])
    def test_no_generated_or_reserved_release(self):
        db=sqlite3.connect(self.db);db.execute("UPDATE variants SET state='generated'WHERE sequence=0");db.commit();db.close()
        with self.assertRaisesRegex(ValueError,'not published'):self.commit()
    def test_stale_authority_and_role_drift(self):
        self.now+=121
        with self.assertRaisesRegex(ValueError,'fresh'):self.commit()
        self.now=1000.;bad=copy.deepcopy(self.authority);bad['support']=[{'changed':True}]
        with self.assertRaisesRegex(ValueError,'changed'):A.role_check(canonical(bad),A.sha(canonical(bad)),self.authority,self.now)
    def test_source_drift_refused(self):
        p=self.root/'source.py';p.write_bytes(b'changed');self.plan['source_pins']=[{'path':str(p),'sha256':'a'*64,'bytes':7}]
        with self.assertRaisesRegex(ValueError,'SHA'):self.commit()
    def test_missing_db_is_not_created(self):
        missing=self.root/'absent.sqlite';self.plan['database']['path']=str(missing)
        with self.assertRaisesRegex(ValueError,'already exist'):self.commit()
        self.assertFalse(missing.exists())
    def test_unverified_arbitrary_plan_cannot_commit(self):
        b=canonical(self.plan);auth=canonical(self.authority)
        with self.assertRaisesRegex(ValueError,'spec hash'):A.commit(b,A.sha(b),spec_bytes=b'{}',store=None,authority_bytes=auth,authority_sha256=A.sha(auth),now=lambda:self.now)
        self.assertEqual(self.states(),[('published',),('published',)])
    def test_reprepare_changed_plan_cannot_commit(self):
        b=canonical(self.plan);auth=canonical(self.authority);different=copy.deepcopy(self.plan);different['refused']=[{}]
        with patch.object(A,'prepare',return_value=different):
            with self.assertRaisesRegex(ValueError,'proof changed'):A.commit(b,A.sha(b),spec_bytes=b'x',store=None,authority_bytes=auth,authority_sha256=A.sha(auth),now=lambda:self.now)
    def test_concurrent_identical_commits_release_once(self):
        with patch.object(A,'prepare',return_value=self.plan):
            def run():
                b=canonical(self.plan);a=canonical(self.authority)
                return A.commit(b,A.sha(b),spec_bytes=b'x',store=None,authority_bytes=a,authority_sha256=A.sha(a),now=lambda:self.now)
            with ThreadPoolExecutor(max_workers=2)as pool:r=list(pool.map(lambda _:run(),range(2)))
        self.assertEqual(sum(x['released_capacity']for x in r),2)
    def test_physical_safetensor_hash_and_tamper(self):
        values=np.array([1.,2.],dtype='<f4');header=canonical({'model.w':{'dtype':'F32','shape':[2],'data_offsets':[0,8]},'rng.torch_cpu':{'dtype':'U8','shape':[2],'data_offsets':[8,10]}})
        raw=struct.pack('<Q',len(header))+header+values.tobytes()+b'ab';expected=hashlib.sha256(b'w'+bytes([0])+b'F32'+bytes([0])+b'2\n'+values.tobytes()).hexdigest()
        self.assertEqual(A.model_hash(raw),expected);self.assertNotEqual(A.model_hash(raw[:-10]+np.array([1.,3.],dtype='<f4').tobytes()+b'ab'),expected)
        bad=struct.pack('<Q',len(header))+header+np.array([np.nan,3.],dtype='<f4').tobytes()+b'ab'
        with self.assertRaisesRegex(ValueError,'finite'):A.model_hash(bad)
    def test_readonly_native_planner_chain_and_unresolved_refusal(self):
        run={'schema':'jope.flywheel-data-run/v1','variant_identity':'a'*64,'context_id':'b'*64,'dataset_sha256':'c'*64,'target_sha256':'d'*64,'split_sha256':'e'*64,'protocol_sha256':'f'*64,'cut_id':'cut'}
        row={'data_run':run,'data_run_id':digest(run),'parent_lineage_sha256':'a'*64,'partition':'train','status':'eligible','reason':None,'payload':{}}
        m={'schema':'jope.flywheel-batch-manifest/v1','snapshot_sha256':'1'*64,'role_authority_sha256':'2'*64,'metadata':{},'rows':[row]};b=canonical(m);p=self.root/'batch.sqlite';stream=BatchStream(p,b,A.sha(b),seed=1,batch_size=1,initial_checkpoint_sha256='3'*64)
        batch=stream.reserve('3'*64)
        with self.assertRaisesRegex(ValueError,'unknown optimizer'):A.planner_read(p,A.sha(b))
        stream.commit(batch.batch_id,checkpoint_before='3'*64,checkpoint_after='4'*64,optimizer_step=1,ordered_data_run_ids=batch.ordered_data_run_ids,evidence={'origin':'native_optimizer','event_sha256':'5'*64})
        anchor,rows=A.planner_read(p,A.sha(b));self.assertEqual(anchor['committed_batches'],1);self.assertEqual(len(rows),1);stream.close()

    def custody_fixture(self):
        rows=[];previous='0'*64
        request={'dsl':'actual fixture DSL','checkpoint_sha256':'1'*64,'step':1}
        outcome={'observation_id':'o1','state':'ready','measurement':{'fixture':True}}
        for i,(kind,body)in enumerate((('observed',request),('outcome',outcome)),1):
            e={'seq':i,'id':'o1','at':1000.,'kind':kind,'body':body,'previous':previous};previous=digest(e)
            rows.append({**e,'body':canonical(body).decode(),'sha':previous})
        result={'queue':{'head_sha256':'q'},'observations':{'events':2,'head_sha256':previous,'unresolved_observations':[]}}
        cfg={'owned_sources':{'campaign_custody.py':'a'*64}}
        prefix='experiments/gp-policy-dl/v2-trainonly/program-attempts/flywheel/v1/'
        snap={'bucket':'veox-jopedime-learning-use2','key':prefix+'snapshot','version_id':'1'}
        receipt={'bucket':snap['bucket'],'key':prefix+'receipt','version_id':'2'}
        closure={'schema':'jope.flywheel-program-attempt-publication-result/v1','status':'PASS','helper_sha256':'a'*64,'config_sha256':'b'*64,'snapshot':snap,'receipt':receipt}
        docs={snap['key']:{'schema':'jope.flywheel-program-attempt-snapshot/v1','source_config':{'sha256':'c'*64},'queue_head':result['queue'],'emission_head':result['observations'],'tables':{'program_emissions':rows}},receipt['key']:{'schema':'jope.flywheel-program-attempt-publication/v1','snapshot':snap,'helper_sha256':'a'*64,'config_sha256':'b'*64,'queue_head':result['queue'],'emission_head':result['observations']}}
        class Reader:
            def read(self,ref):return canonical(docs[ref['key']])
        return Reader(),{'attempt_custody':closure},result,cfg,docs,request,outcome

    def test_exact_archived_emission_and_outcome_chain(self):
        reader,status,result,cfg,docs,request,outcome=self.custody_fixture()
        observed,outcomes=A.custody_bindings(reader,status,result,cfg,'c'*64)
        self.assertEqual(observed,{'o1':request});self.assertEqual(outcomes,{'o1':outcome})

    def test_archived_emission_tamper_or_wrong_source_rejected(self):
        reader,status,result,cfg,docs,request,outcome=self.custody_fixture()
        snapshot=docs[status['attempt_custody']['snapshot']['key']]
        snapshot['tables']['program_emissions'][0]['body']='{}'
        with self.assertRaisesRegex(ValueError,'chain'):A.custody_bindings(reader,status,result,cfg,'c'*64)
        reader,status,result,cfg,docs,request,outcome=self.custody_fixture()
        with self.assertRaisesRegex(ValueError,'source/run'):A.custody_bindings(reader,status,result,cfg,'d'*64)

    def test_incomplete_custody_and_wrong_helper_rejected(self):
        reader,status,result,cfg,docs,request,outcome=self.custody_fixture()
        snapshot=docs[status['attempt_custody']['snapshot']['key']];snapshot['tables']['program_emissions'].pop()
        with self.assertRaisesRegex(ValueError,'unresolved'):A.custody_bindings(reader,status,result,cfg,'c'*64)
        reader,status,result,cfg,docs,request,outcome=self.custody_fixture();cfg['owned_sources']['campaign_custody.py']='d'*64
        with self.assertRaisesRegex(ValueError,'pinned'):A.custody_bindings(reader,status,result,cfg,'c'*64)

    def test_full_prepare_fixture_with_native_snapshot_and_physical_journal(self):
        # Fixture only: real consumer/planner/parsers, synthetic artifact contents;
        # this does not assert a live optimizer run or real S3 custody.
        from test_training_snapshot_kumos import SnapshotTests
        from jope_flywheel import training_snapshot_kumos as S
        f=SnapshotTests();f.setUpClass();f.setUp();f.task(programs=8);f.task('validation',programs=8)
        loaded=f.load();registry,authority=f.freeze();manifest=S.batch_manifest(loaded)
        def save(name,obj,raw=False):
            b=obj if raw else canonical(obj);p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
            return {'path':str(p),'sha256':A.sha(b),'bytes':len(b)}
        reg=save('parents.jsonl',registry,True)
        # Actual immutable generator config has only path+SHA for its registry.
        gen=save('generator.json',{'out':str(self.root),'parent_registry':{k:reg[k]for k in ('path','sha256')},'target':100,'backlog_limit':16})
        accepted_loaded=save('accepted-loaded.json',loaded.receipt);accepted_manifest=save('accepted-manifest.json',manifest)
        snapshot=save('snapshot-status.json',{'state':'loaded','config_sha256':'a'*64,'loaded_receipt':accepted_loaded,'batch_manifest':accepted_manifest})
        native_role=json.loads(authority);native_role['checked_at_unix']+=.5
        roleref=save('role-proof.json',native_role);native_loaded=copy.deepcopy(loaded.receipt);native_loaded['authority_sha256']=roleref['sha256']
        loadedref=save('loaded-receipt.json',native_loaded);manifest['role_authority_sha256']=roleref['sha256'];mref=save('training/batch-manifest.json',manifest)
        cfg={'owned_root':str(self.root),'neural':{'model':{'ctx_dim':2051},'learning_rate':.0003},'snapshot_status':snapshot,'snapshot_acceptance':{'sha256':'a'*64},'owned_sources':{'campaign_custody.py':'b'*64}}
        cfgref=save('neural-config.json',cfg);window=save('window.json',{'config_sha256':cfgref['sha256'],'created_unix':900.,'deadline_unix':1200.});checkpoints=[];bodies=[]
        for step in (0,1):
            h=canonical({'model.w':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}})
            raw=struct.pack('<Q',len(h))+h+np.array([1.+step],dtype='<f4').tobytes()
            state=save('state%d'%step,raw,True);cm={'schema':'jope.flywheel-neural-checkpoint/v1','config_sha256':cfgref['sha256'],'state_sha256':state['sha256'],'state_bytes':state['bytes'],'parameter_sha256':A.model_hash(raw),'step':step,'model_config':cfg['neural']['model'],'optimizer':'AdamW','learning_rate':.0003,'weight_decay':.01}
            bodies.append(cm);checkpoints.append({'manifest':save('checkpoint%d.json'%step,cm),'state':state})
        planner=BatchStream(self.root/'training/batches.sqlite',canonical(manifest),mref['sha256'],seed=1,batch_size=1,initial_checkpoint_sha256=bodies[0]['parameter_sha256'])
        batch=planner.reserve(bodies[0]['parameter_sha256']);rows=manifest['rows'];train=batch.rows[0];cid=train['payload']['context_sha256'];plan=loaded.receipt['plans'][cid]
        members=plan['teachers']+[x for x in plan['labelled']if x not in plan['teachers']]
        slot={'context_sha256':cid,'stream':'group','members':members,'roles':['teacher']*4+['comparator']*4,'family_id':train['parent_lineage_sha256'],'dataset_hash':train['data_run']['variant_identity']}
        events=[];previous='0'*64;measurements=[]
        tasks={t['variant_identity']:t for t in f.spec['tasks']}
        for i,row in enumerate(rows):
            ref=tasks[row['data_run']['variant_identity']]['program_results'][0];wrapper=f.store.doc(ref);dsl=f.store.blobs[(wrapper['artifacts']['program']['key'],wrapper['artifacts']['program']['version_id'])].decode()
            measured=A.verified_measurement(f.store,A.Context(**tasks[row['data_run']['variant_identity']]['context']),dsl,wrapper)
            phase='training'if row['partition']=='train'else'validation';oid='o'+str(i)
            request={'dsl':dsl,'variant_identity':row['data_run']['variant_identity'],'phase':phase,'checkpoint_sha256':bodies[0]['parameter_sha256'],'step':1,'producer':'PolicyLM-autoregressive-finite-trie-v1'}
            outcome={'observation_id':oid,'state':'ready','measurement':measured}
            for kind,body in (('observed',request),('outcome',outcome)):
                e={'seq':len(events)+1,'id':oid,'at':self.now,'kind':kind,'body':body,'previous':previous};previous=digest(e);events.append({**e,'body':canonical(body).decode(),'sha':previous})
            measurements.append({'state':'ready','data_run_id':row['data_run_id'],'context_id':row['payload']['context_sha256'],'phase':phase,'record':ref,'readback':outcome,'measurement':measured})
        proof={'schema':'jope.flywheel-neural-batch/v1','batch':asdict(batch),'source_snapshot_sha256':loaded.receipt['snapshot_sha256'],'draw':{'slots':[slot]},'update':{'checkpoint_before':bodies[0]['parameter_sha256'],'checkpoint_after':bodies[1]['parameter_sha256'],'loss':1.,'gradient_norm_before_clip':.5},'physical_checkpoint':bodies[1],'measurements':measurements}
        pref=save('batch-evidence.json',proof)
        planner.commit(batch.batch_id,checkpoint_before=bodies[0]['parameter_sha256'],checkpoint_after=bodies[1]['parameter_sha256'],optimizer_step=1,ordered_data_run_ids=batch.ordered_data_run_ids,evidence={'origin':'native_optimizer','event_sha256':pref['sha256']})
        report={'dropout':False,'teacher_members_frozen':True,'objective':'successor-v1','loss':1.,'evaluated_context_ids':[r['payload']['context_sha256']for r in rows if r['partition']=='validation']}
        result={'schema':'jope.flywheel-neural-result/v1','state':'completed_bounded_qualification','config_sha256':cfgref['sha256'],'error':None,'optimizer_steps':1,'planner':planner.verify(),'queue':{'fixture':True},'observations':{'events':len(events),'head_sha256':previous,'unresolved_observations':[]},'proofs':[{'batch_id':batch.batch_id,'evidence_sha256':pref['sha256']}],'loss_history':[{'batch':1,'train':report,'validation':report}]};planner.close()
        resultref=save('training/result.json',result);prefix='experiments/gp-policy-dl/v2-trainonly/program-attempts/flywheel/v1/'
        snap=f.store.put(prefix+'snapshot',canonical({'schema':'jope.flywheel-program-attempt-snapshot/v1','source_config':cfgref,'queue_head':result['queue'],'emission_head':result['observations'],'tables':{'program_emissions':events}}));snap['bucket']='veox-jopedime-learning-use2'
        receipt=f.store.put(prefix+'receipt',canonical({'schema':'jope.flywheel-program-attempt-publication/v1','snapshot':snap,'helper_sha256':'b'*64,'config_sha256':'c'*64,'queue_head':result['queue'],'emission_head':result['observations']}));receipt['bucket']=snap['bucket']
        status=save('status.json',{'schema':'jope.flywheel-neural-status/v1','state':'complete','config_sha256':cfgref['sha256'],'result':resultref,'error':None,'attempt_custody':{'schema':'jope.flywheel-program-attempt-publication-result/v1','status':'PASS','helper_sha256':'b'*64,'config_sha256':'c'*64,'snapshot':snap,'receipt':receipt}})
        proposed=[]
        for i,row in enumerate(rows):
            pub=f.store.doc(tasks[row['data_run']['variant_identity']]['dataset_publication'])
            proposed.append({'sequence':i,'parent':row['parent_lineage_sha256'],'ordinal':0,'identity':row['data_run']['variant_identity'],'content_sha':row['data_run']['dataset_sha256'],'metadata_sha':pub['metadata_sha256'],'data_run_id':row['data_run_id']})
        spec={'schema':A.SCHEMA,'source_pins':[],'database':{'path':str(self.root/'expansion.sqlite'),'config_sha256':digest({'config_sha':gen['sha256'],'parents':f.parents,'target':100,'backlog':16})},'generator_config':gen,'parent_registry':reg,'snapshot_status':snapshot,'accepted_loaded_receipt':accepted_loaded,'accepted_batch_manifest':accepted_manifest,'loaded_receipt':loadedref,'batch_manifest':mref,'neural_role_proof':roleref,'neural_window':window,'neural_config':cfgref,'neural_status':status,'neural_result':resultref,'planner_path':str(self.root/'training/batches.sqlite'),'batch_evidence':[pref],'checkpoints':checkpoints,'authority_reference':save('authority.json',authority,True),'rows':proposed}
        raw=canonical(spec);plan=A.prepare(raw,A.sha(raw),f.store,authority_bytes=authority,authority_sha256=A.sha(authority),now=lambda:self.now+1)
        self.assertEqual(len(plan['accepted']),2);self.assertEqual(plan['refused'],[]);self.assertFalse(plan['database_modified'])
        self.assertNotEqual(accepted_manifest['sha256'],mref['sha256'])
        self.assertNotEqual(accepted_loaded['sha256'],loadedref['sha256'])
        # A valid physical chain plus aggregate validation loss is insufficient:
        # explicit evaluated dev membership is required for that row's release.
        result['loss_history'][0]['validation']['evaluated_context_ids']=[]
        newresult=save('training/result.json',result);newstatus=json.loads(Path(status['path']).read_text());newstatus['result']=newresult
        spec['neural_result']=newresult;spec['neural_status']=save('status.json',newstatus);raw=canonical(spec)
        plan=A.prepare(raw,A.sha(raw),f.store,authority_bytes=authority,authority_sha256=A.sha(authority),now=lambda:self.now+1)
        self.assertEqual(len(plan['accepted']),1);self.assertEqual(plan['refused'][0]['reason'],'NO_ACTUAL_FIXED_DEV_EVALUATION')
        native_loaded['all_membership_sha256']='f'*64
        spec['loaded_receipt']=save('loaded-receipt.json',native_loaded);raw=canonical(spec)
        with self.assertRaisesRegex(ValueError,'loaded corpus differs'):
            A.prepare(raw,A.sha(raw),f.store,authority_bytes=authority,authority_sha256=A.sha(authority),now=lambda:self.now+1)

if __name__=='__main__':unittest.main()
