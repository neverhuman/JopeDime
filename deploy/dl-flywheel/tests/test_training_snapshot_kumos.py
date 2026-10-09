"""CPU integration with the real pinned dltrain modules; no native execution.

Run with JOPE_POLICYLM_SOURCE=/path/to/pinned/package-parent on PYTHONPATH.
The in-memory VersionId store and score canonicalizer are TEST FIXTURES, not S3
custody or native-serde qualification. Separate captured-artifact proof is saved.
"""
import copy,gzip,hashlib,importlib,io,json,math,os,struct,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from jope_flywheel import training_snapshot_kumos as S
from jope_flywheel.augmentation import array_sha256
from jope_flywheel.protocol import Context,canonical,digest,partition
from jope_flywheel.program_gate import request_identity
from jope_flywheel.score_join import build_records

if os.environ.get('JOPE_POLICYLM_SOURCE'):
    sys.path.insert(0,os.environ['JOPE_POLICYLM_SOURCE'])

class Store:
    nativecanonical_json=staticmethod(canonical)  # fixture ONLY
    def __init__(self):self.blobs={};self.calls=0;self.wrong_version=False
    def put(self,key,body):
        ref={'bucket':'fixture','key':key,'version_id':'v-'+S.byte_sha(body),'sha256':S.byte_sha(body),'bytes':len(body)}
        self.blobs[(key,ref['version_id'])]=body
        return ref
    def get_version(self,ref):
        self.calls+=1
        return self.blobs[(ref['key'],ref['version_id'])],('wrong' if self.wrong_version else ref['version_id'])
    def replace(self,ref,obj):return self.put(ref['key'],canonical(obj))
    def doc(self,ref):return json.loads(self.blobs[(ref['key'],ref['version_id'])])

class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from dltrain import dsl_tokenizer as T,automaton as A
        except ImportError as error:raise unittest.SkipTest('real pinned dltrain package required: '+str(error))
        cls.T,cls.A=T,A;cls.vocab=T.Vocab.from_names({});sig=A.build_signatures([])
        cls.grammar=canonical({'schema':'jope.flywheel-policy-grammar/v1','mode':'SYNTAX','vocab_sha256':cls.vocab.sha256,
            'max_seq':128,'signatures':sig,'grammar':A.build_grammar(sig,cls.vocab,128)})
        cls.pins={name:S.byte_sha(Path(importlib.import_module(name).__file__).read_bytes()) for name in S.CONSUMER_MODULES}
    def setUp(self):
        self.store=Store();self.parents=[];self.support=[];self.raws={};self.now=1000.
        self.spec={'schema':S.SCHEMA,'roster_audit_sha256':S.ROSTER_AUDIT_SHA256,'tasks':[],'components':S.COMPONENTS,'consumer_source_sha256':self.pins.copy(),
            'adapter_dependency_sha256':{n:S.byte_sha(Path(importlib.import_module(n).__file__).read_bytes())for n in S.ADAPTER_MODULES},
            'vocab_sha256':self.vocab.sha256,'grammar_sha256':S.byte_sha(self.grammar),'max_seq':128,
            'parent_registry_sha256':'a'*64,'generator_config_sha256':'b'*64,
            'authority_helper_sha256':'d'*64,'dataset_publisher_sha256':'e'*64,'encoder_publisher_sha256':'f'*64,
            'encoder_releases':[{'source_sha256':'1'*64,'manifest_sha256':'2'*64}], 'shared_seed_program_sha256':[],
            'evaluation_contract':{'evaluator_sha256':'3'*64,'vocabulary_sha256':'4'*64,'protocol_sha256':'5'*64,'units':'prepared_synthetic_target_units'}}
    def task(self,role='train',programs=2,ordinal=0,copy_family=None,program_texts=None):
        namespace='6'*64
        lineage=copy_family
        if lineage is None:
            i=len(self.parents)*100
            while True:
                lineage=digest({'lineage':i});i+=1
                if partition(lineage,namespace)==role:break
        folds=np.array([0,1]*25,dtype=np.uint8);y=np.linspace(0,1,50,dtype='<f8')
        x=np.cos(np.arange(50,dtype='<f8')+ordinal)
        raw=b'JTT1'+struct.pack('<IIIB',50,0,1,2)+folds.tobytes()+np.column_stack([x,y]).astype('<f8').tobytes()
        identity=digest({'variant':lineage,'ordinal':ordinal});self.raws[identity]=raw
        parent={'parent_lineage_sha256':lineage,'partition':role,'split_namespace_sha256':namespace,'target_sha256':array_sha256(y),
            'parent_context_sha256':'7'*64,'view_id':'fixture-'+lineage,'retained_lineage_context_sha256':['7'*64]}
        if copy_family is None:self.parents.append(parent)
        meta={'schema':'jope.flywheel-dataset/v1','variant_identity':identity,'parent':parent,'parent_record_sha256':digest(parent),
            'parent_lineage_sha256':lineage,'partition':role,'content_sha256':S.byte_sha(raw),'target_sha256':array_sha256(y),
            'split_sha256':S.byte_sha(folds.tobytes()),'source_target_sha256':parent['target_sha256'],'rows':50,'features':1,'inner_counts':[25,25],
            'outer_test_rows':0,'prepared':True,'raw_units':False,'inherited_gp_rewards':False,'teaching_eligible':False}
        prefix='fixture/datasets/flywheel/v1/'+identity+'/'
        packed=gzip.compress(raw,mtime=0);refs={'encoder.jtt1.gz':self.store.put(prefix+'encoder.jtt1.gz',packed),
            'metadata.json':self.store.put(prefix+'metadata.json',canonical(meta))}
        gen={'schema':'jope.flywheel-evergreen/v1','raw_sha256':S.byte_sha(raw),'variant_identity':identity,
            'encoder_sha256':S.byte_sha(packed),'metadata_sha256':refs['metadata.json']['sha256'],'config_sha256':self.spec['generator_config_sha256']}
        refs['generation.json']=self.store.put(prefix+'generation.json',canonical(gen))
        pub={'schema':'jope.flywheel-evergreen-dataset-publication/v1','variant_identity':identity,
            'generator_config_sha256':self.spec['generator_config_sha256'],'publication_helper_sha256':self.spec['dataset_publisher_sha256'],
            'parent_registry':{'sha256':'REPLACE'},'objects':list(refs.values()),'parent_lineage_sha256':lineage,'content_sha256':S.byte_sha(raw),
            'metadata_sha256':refs['metadata.json']['sha256'],'qualification':{'deterministic_replay_exact':True}}
        datasetref=self.store.put('fixture/receipt/'+identity,canonical(pub))
        anchor={'schema':'jope.flywheel-evidence/v1','event_count':1,'head_sha256':'8'*64,'runs':{'fixture':{'sealed':True}}}
        records={};vectors=[]
        for name,dim in S.COMPONENTS.items():
            stream=io.BytesIO();np.save(stream,np.arange(dim,dtype='<f4'),allow_pickle=False)
            ref=self.store.put(prefix+'encodings/'+name+'.npy',stream.getvalue());vectors.append(ref)
            local={'path':'variants/'+identity+'/'+name+'.npy','sha256':ref['sha256'],'bytes':ref['bytes']};q={'passed':True,'fixture':True};qb=canonical(q)
            qr={'path':name+'.qualification.json','sha256':S.byte_sha(qb),'bytes':len(qb)}
            rec={'phase':'ofm','manifest_sha256':'2'*64,'variant_identity':identity,'component':name,'dimension':dim,'input_raw_sha256':S.byte_sha(raw),
                'target_sha256':meta['target_sha256'],'parent_lineage_sha256':lineage,'partition':role,'source_metadata':refs['metadata.json'],
                'source_input':refs['encoder.jtt1.gz'],'output':local,'qualification':qr,
                'producer':{'weight':{'repo':'nvidia/Kumo-Tabular','revision':S.MODEL_REVISION,'file':'small/regressor.pt','sha256':S.WEIGHT_SHA256}}}
            records[name]={'record':rec,'record_ref':{'path':name+'.record.json','sha256':S.byte_sha(canonical(rec)),'bytes':len(canonical(rec))},'vector_ref':{**local,'path':'/fixture/run/ofm/'+local['path']},'qualification':{'ref':qr,'body':q}}
        ep={'schema':'jope.flywheel-evergreen-encoder-publication/v1','variant_identity':identity,
            'publisher_sha256':self.spec['encoder_publisher_sha256'],'completed_encoders':list(S.COMPONENTS),'missing_encoders':[],
            'teaching_eligible':False,'lift_eligible':False,'prior_dataset_publication':{'receipt':datasetref,'objects':refs},
            'source_proof':{'source_release':{'out':'/fixture/run','source_sha256':'1'*64},'source_sha256':'1'*64,'manifest':{'sha256':'2'*64},'coordinator_ledger':anchor,
                'phase_ledgers':{n:copy.deepcopy(anchor)for n in ('v6','ofm','dope')}},'objects':vectors,'encoder_records':records}
        ep.update({k:meta[k]for k in ('content_sha256','target_sha256','source_target_sha256','parent_lineage_sha256','partition','rows','features','inner_counts','outer_test_rows')})
        er=self.store.put('fixture/encoding-receipt/'+identity,canonical(ep))
        cut={'schema':'jope.flywheel-prepared-two-fold-cut/v1','variant_identity':identity,'metadata_sha256':refs['metadata.json']['sha256'],
            'dataset_sha256':S.byte_sha(raw),'target_sha256':meta['target_sha256'],'split_sha256':meta['split_sha256'],'protocol_sha256':'5'*64}
        ctx=Context('regression',S.byte_sha(raw),meta['target_sha256'],S.byte_sha(canonical(cut)),'3'*64,'4'*64,'5'*64,meta['split_sha256'],'rmse','prepared_synthetic_target_units','minimize')
        task={'variant_identity':identity,'context':ctx.__dict__,'dataset_publication':datasetref,'encoder_publication':er,'program_results':[]}
        baselines=[math.sqrt(math.fsum((float(v)-float(np.mean(y[folds!=i])))**2 for v in y[folds==i])/25) for i in (0,1)]
        for j in range(programs):
            dsl=f'(pipe X (op{j}))' if program_texts is None else program_texts[j];ph=S.byte_sha(dsl.encode());key=request_identity(ctx,dsl);rmse=.01*(j+1)
            ev={'schema':'jope.s1-evaluation/v1','status':'completed','fold_protocol':'jope.s1-two-fold/v1','inner_cv_folds':2,
                'input_semantics':'unscaled_fit_rows_v1','program':{'dsl':dsl,'program_sha256':ph},'inner_cv':[]}
            for i in (0,1):
                target=y[folds==i];pred=target-rmse
                ev['inner_cv'].append({'schema':'jope.teacher-fit/v2','kind':'inner_cv','index':i,'score_rows':25,'train_rows':25,
                    'metrics':{'rmse':rmse},'baseline_metrics':{'rmse':baselines[i]},'predictions':pred.tolist(),'targets':target.tolist(),
                    'serialized_prediction_replay_exact':True,'wall_us':100})
            bodies={'input':raw,'program':dsl.encode(),'evaluation':canonical(ev),'rust_source':b'fixture code','rust_executable':b'fixture binary'}
            parity={'schema':'jope.flywheel-rust-parity/v1','passed':True,'executed':True,'folds_verified':2,'evaluation_key':key,
                'program_sha256':ph,'evaluator_sha256':'3'*64,'native_provenance':{'schema':'jope.flywheel-native-execution-provenance/v1',
                'evaluator_sha256':'3'*64,'input_sha256':S.byte_sha(raw),'program_sha256':ph,'evaluation_sha256':S.byte_sha(bodies['evaluation']),
                'source_commit':'a'*40,'execution_timestamp':123}}
            parity.update({k+'_sha256':S.byte_sha(bodies[k])for k in ('input','evaluation','rust_source','rust_executable')});bodies['rust_parity']=canonical(parity)
            bodies['score_records']=build_records(ctx,ph,key,[rmse,rmse],S.byte_sha(bodies['rust_parity']),'a'*40,123,canonical)
            ar={n:self.store.put('fixture/program/'+key+'/'+n,b)for n,b in bodies.items()}
            entry={'schema':'jope.flywheel-program-result/v1','context':ctx.__dict__,'program_sha256':ph,'evaluation_key':key,
                'status':'completed','measurement':'executed','artifacts':ar}
            task['program_results'].append(self.store.put('fixture/program/'+key+'/wrapper',canonical(entry)))
        self.spec['tasks'].append(task);self.support.append({'variant_identity':identity,'parent_lineage_sha256':lineage,
            'parent_context_sha256':parent['parent_context_sha256'],'view_id':parent['view_id'],'partition':role,'parent_record_sha256':digest(parent),
            'retained_contexts_checked':1,'eligible_view':True,'excluded_contexts_or_views':False})
        return task
    def freeze(self):
        registry=b'\n'.join(canonical(p)for p in self.parents)+b'\n';self.spec['parent_registry_sha256']=S.byte_sha(registry)
        for t in self.spec['tasks']:
            pub=self.store.doc(t['dataset_publication']);pub['parent_registry']['sha256']=S.byte_sha(registry)
            t['dataset_publication']=self.store.replace(t['dataset_publication'],pub)
            ep=self.store.doc(t['encoder_publication']);ep['prior_dataset_publication']['receipt']=t['dataset_publication'];t['encoder_publication']=self.store.replace(t['encoder_publication'],ep)
        authority={'schema':'jope.flywheel-evergreen-encoding-authority/v1','state':'passed','host':'xbabe2','checked_at_unix':self.now,
            'authority_helper_sha256':self.spec['authority_helper_sha256'],'parent_registry_sha256':S.byte_sha(registry),
            'generator_config_sha256':self.spec['generator_config_sha256'],'support':self.support}
        ar=canonical(authority);self.authority_sha256=S.byte_sha(ar)
        return registry,ar
    def load(self,mutate=None,now=None):
        registry,authority=self.freeze()
        if mutate:mutate()
        spec=canonical(self.spec)
        return S.load_snapshot(spec,S.byte_sha(spec),self.store,registry_bytes=registry,authority_bytes=authority,authority_sha256=self.authority_sha256,grammar_bytes=self.grammar,now=now or(lambda:self.now+1))
    def test_captured_real_relative_output_resolves_to_absolute_publication(self):
        path=Path(__file__).resolve().parent/'fixtures/kumos-component-publication.json'
        captured=json.loads(path.read_bytes());pub=captured['publication'];entry=pub['encoder_records']['kumo_tabular_s'];record=entry['record']
        self.assertNotEqual(record['output']['path'],entry['vector_ref']['path'])
        self.assertEqual(S.byte_sha(canonical(record)),entry['record_ref']['sha256'])
        self.assertTrue(S.component_output_binding(record,entry,pub['source_proof'],captured['task']['variant_identity'],'kumo_tabular_s'))
        for change in ('absolute','traversal','wrong_variant','wrong_component','wrong_root','wrong_sha','wrong_bytes','wrong_phase'):
            r=copy.deepcopy(record);e=copy.deepcopy(entry);proof=copy.deepcopy(pub['source_proof'])
            if change=='absolute':r['output']['path']=e['vector_ref']['path']
            elif change=='traversal':r['output']['path']='../'+r['output']['path']
            elif change=='wrong_variant':r['output']['path']=r['output']['path'].replace(captured['task']['variant_identity'],'a'*64)
            elif change=='wrong_component':r['output']['path']=r['output']['path'].replace('kumo_tabular_s','other')
            elif change=='wrong_root':proof['source_release']['out']+='/other'
            elif change=='wrong_sha':e['vector_ref']['sha256']='a'*64
            elif change=='wrong_bytes':e['vector_ref']['bytes']+=1
            elif change=='wrong_phase':r['phase']='dope'
            with self.subTest(change=change),self.assertRaises(ValueError):S.component_output_binding(r,e,proof,captured['task']['variant_identity'],'kumo_tabular_s')

    def test_only_kumos_read_even_if_other_components_are_unavailable(self):
        task=self.task();pub=self.store.doc(task['encoder_publication'])
        pub['completed_encoders']+=['tabpfn35','a0']
        # No blob exists for these references: any read would fail the test.
        for name in ('tabpfn35','a0'):
            pub['objects'].append({'bucket':'fixture','key':'fixture/datasets/flywheel/v1/'+task['variant_identity']+'/encodings/'+name+'.npy',
                                  'version_id':'missing','sha256':'a'*64,'bytes':123})
        pub['source_proof']['phase_ledgers'].pop('v6');pub['source_proof']['phase_ledgers'].pop('dope')
        task['encoder_publication']=self.store.replace(task['encoder_publication'],pub)
        loaded=self.load();self.assertEqual(loaded.data.ctx_dim,2051)
        self.assertEqual(set(next(iter(loaded.receipt['contexts'].values()))['vectors']),{'kumo_tabular_s'})
        self.assertEqual(set(loaded.receipt['components']),{'kumo_tabular_s'})

    def test_wrong_weight_revision_and_roster_fail_closed(self):
        task=self.task();pub=self.store.doc(task['encoder_publication']);entry=pub['encoder_records']['kumo_tabular_s']
        entry['record']['producer']['weight']['revision']='not-reviewed'
        entry['record_ref'].update(sha256=S.byte_sha(canonical(entry['record'])),bytes=len(canonical(entry['record'])))
        task['encoder_publication']=self.store.replace(task['encoder_publication'],pub)
        with self.assertRaisesRegex(ValueError,'weight/revision'):self.load()
        self.spec['roster_audit_sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'roster audit'):self.load()

    def test_native_eight_dsl_teacher_compatibility(self):
        # Actual native canonical parser outputs (69975947), numeric scores are fixture-only.
        texts=['(pipe X (ridge))','(pipe X (standardize) (ridge lambda=1))',
          '(pipe X (robust_scale) (ridge lambda=0.1))','(pipe X (rank) (ridge lambda=0.1))',
          '(pipe X (rff_cos append=false gamma=0.7 n=16 seed=5) (ridge lambda=0.1))',
          '(pipe X (rand_proj append=false k=8 seed=7) (ridge lambda=0.1))',
          '(pipe X (row_stats append=false) (ridge lambda=0.1))',
          '(pipe X (elasticnet alpha=0.01 l1_ratio=0.3 sweeps=20))']
        task=self.task(programs=8,program_texts=texts);loaded=self.load()
        self.assertEqual(loaded.receipt['counts']['records_kept'],8)
        self.assertEqual(loaded.receipt['counts']['fit_group_contexts'],1)
        self.assertEqual(loaded.receipt['refusals'],[])
        for dsl in texts:self.assertEqual(self.T.detokenize(self.T.tokenize(dsl,self.vocab),self.vocab),dsl)

    def test_batch_manifest_retains_every_status_and_excludes_dev_from_draw(self):
        from jope_flywheel.batch_stream import BatchStream
        fit=self.task(programs=8);dev=self.task('validation',programs=8)
        pending=self.task(programs=0,ordinal=2)
        refused=self.task(programs=1,ordinal=3,program_texts=['(pipe X'+(' (ridge)'*40)+')'])
        feas=self.task(programs=2,ordinal=4)
        loaded=self.load();manifest=S.batch_manifest(loaded);rows=manifest['rows']
        self.assertEqual(len(rows),5)
        self.assertEqual(sorted(r['status']for r in rows),['eligible','eligible','eligible','pending','refused'])
        self.assertTrue(any(r['payload']['stream']=='feasibility' and r['status']=='eligible'for r in rows))
        with tempfile.TemporaryDirectory()as folder:
            body=canonical(manifest);stream=BatchStream(Path(folder).resolve()/'batch.sqlite',body,S.byte_sha(body),seed=7,batch_size=3,initial_checkpoint_sha256='1'*64)
            batch=stream.reserve('1'*64)
            self.assertEqual({r['payload']['context_sha256']for r in batch.rows},{Context(**fit['context']).identity,Context(**feas['context']).identity})
            self.assertEqual(len({r['parent_lineage_sha256']for r in batch.rows}),len(batch.rows))
            self.assertTrue(all(r['partition']=='train'for r in batch.rows));stream.close()

    def test_all_pending_batch_has_no_fictional_epoch(self):
        from jope_flywheel.batch_stream import BatchStream,Waiting
        self.task(programs=0);loaded=self.load();manifest=S.batch_manifest(loaded)
        with tempfile.TemporaryDirectory()as folder:
            body=canonical(manifest);stream=BatchStream(Path(folder).resolve()/'batch.sqlite',body,S.byte_sha(body),seed=7,batch_size=3,initial_checkpoint_sha256='1'*64)
            self.assertIsInstance(stream.reserve('1'*64),Waiting);self.assertEqual(stream.summary()['epoch'],0);stream.close()

    def test_pinned_inventory_retains_pending_without_fake_admission(self):
        self.task(programs=8);loaded=self.load();first=S.batch_manifest(loaded)
        row=copy.deepcopy(first['rows'][0]);row.update(status='pending',reason='AWAITING_KUMOS_AND_VERIFIED_GP_CLOSURE')
        pending=copy.deepcopy(row);pending['data_run'].update(variant_identity='7'*64,context_id='8'*64)
        pending['data_run_id']=digest(pending['data_run']);pending['parent_lineage_sha256']='9'*64
        inventory={'schema':'jope.flywheel-pending-data-runs/v1','source_registry_sha256':'a'*64,'rows':[row,pending]}
        raw=canonical(inventory);m=S.batch_manifest(loaded,pending_inventory_bytes=raw,pending_inventory_sha256=S.byte_sha(raw))
        self.assertEqual(sorted(r['status']for r in m['rows']),['eligible','pending'])
        self.assertEqual(m['metadata']['pending_inventory_sha256'],S.byte_sha(raw))
        pending['status']='eligible';raw=canonical(inventory)
        with self.assertRaisesRegex(ValueError,'unadmitted pending'):S.batch_manifest(loaded,pending_inventory_bytes=raw,pending_inventory_sha256=S.byte_sha(raw))
        with self.assertRaisesRegex(ValueError,'inventory hash'):S.batch_manifest(loaded,pending_inventory_bytes=raw,pending_inventory_sha256='b'*64)

    def test_real_loader_two_programs_is_feasibility_only_full64(self):
        task=self.task();loaded=self.load();r=loaded.receipt
        self.assertEqual(r['state'],'loaded');self.assertEqual(r['ctx_dim'],2051);self.assertEqual(r['counts']['fit_group_contexts'],0)
        self.assertEqual(r['counts']['fit_feasibility_contexts'],1);self.assertEqual(r['counts']['records_kept'],2)
        self.assertEqual(next(iter(loaded.data.corpus.contexts.values()))['dataset_hash'],task['variant_identity'])
        self.assertEqual(len(loaded.data.content_sha256),64);self.assertFalse(r['database_admission_performed'])
        self.assertEqual(next(iter(r['plans'].values()))['reason'],'FEW_SKELETONS')
    def test_eight_distinct_programs_real_group_and_dev_exclusion(self):
        a=self.task(programs=8);b=self.task('validation',programs=8);loaded=self.load()
        self.assertEqual(loaded.receipt['counts']['fit_group_contexts'],1);self.assertEqual(loaded.receipt['counts']['dev_contexts'],1)
        self.assertEqual(set(loaded.data.fit.pi),{Context(**a['context']).identity});self.assertNotIn(Context(**b['context']).identity,loaded.data.fit.pi)
    def test_duplicate_identity_refused(self):
        a=self.task(programs=2);fam=self.parents[0]['parent_lineage_sha256'];self.task(programs=2,copy_family=fam);self.spec['tasks'].pop();self.support.pop()
        # Duplicate identity is refused before it can alter mass.
        self.spec['tasks'].append(copy.deepcopy(a))
        with self.assertRaisesRegex(ValueError,'unique task'):self.load()
    def test_hierarchical_family_weights_not_exhaustive_epoch(self):
        self.task();fam=self.parents[0]['parent_lineage_sha256'];self.task(copy_family=fam,ordinal=1);self.task(ordinal=2)
        loaded=self.load();mass={}
        for ctx,pi in loaded.data.fit.pi.items():
            family=loaded.data.corpus.contexts[ctx]['family_id'];mass[family]=mass.get(family,0)+pi
        self.assertEqual(sorted(mass.values()),[.5,.5]);self.assertEqual(loaded.receipt['counts']['contexts_loaded'],3)
    def test_fresh_recheck_preserves_snapshot_identity(self):
        self.task();first=self.load();self.now+=10;second=self.load()
        self.assertEqual(first.data.content_sha256,second.data.content_sha256)
        self.assertNotEqual(first.receipt['authority_sha256'],second.receipt['authority_sha256'])
    def test_nonfinite_full_vector_refused_after_valid_hash_read(self):
        task=self.task();pub=self.store.doc(task['encoder_publication']);name='kumo_tabular_s'
        vec=np.arange(S.COMPONENTS[name],dtype='<f4');vec[3]=np.nan;stream=io.BytesIO();np.save(stream,vec,allow_pickle=False)
        old=next(r for r in pub['objects']if r['key'].endswith('/kumo_tabular_s.npy'));ref=self.store.put(old['key'],stream.getvalue())
        pub['objects'][pub['objects'].index(old)]=ref
        for obj in [pub['encoder_records'][name]['record']['output'],pub['encoder_records'][name]['vector_ref']]:obj.update(sha256=ref['sha256'],bytes=ref['bytes'])
        r=pub['encoder_records'][name];r['record_ref'].update(sha256=S.byte_sha(canonical(r['record'])),bytes=len(canonical(r['record'])))
        task['encoder_publication']=self.store.replace(task['encoder_publication'],pub)
        with self.assertRaisesRegex(ValueError,'finite float32'):self.load()
    def test_missing_component_fail_closed(self):
        t=self.task();ep=self.store.doc(t['encoder_publication']);ep['objects'].pop();t['encoder_publication']=self.store.replace(t['encoder_publication'],ep)
        with self.assertRaisesRegex(ValueError,'object closure'):self.load()
    def test_qualification_body_tamper(self):
        t=self.task();ep=self.store.doc(t['encoder_publication']);ep['encoder_records']['kumo_tabular_s']['qualification']['body']['passed']=False
        t['encoder_publication']=self.store.replace(t['encoder_publication'],ep)
        with self.assertRaisesRegex(ValueError,'qualification'):self.load()
    def test_stale_and_midload_expired_authority(self):
        self.task()
        with self.assertRaisesRegex(ValueError,'stale'):self.load(now=lambda:self.now+121)
        ticks=iter([self.now+1,self.now+121])
        with self.assertRaisesRegex(ValueError,'stale'):self.load(now=lambda:next(ticks))
    def test_protected_role_refused(self):
        self.task();self.support[0]['excluded_contexts_or_views']=True
        with self.assertRaisesRegex(ValueError,'role/exclusion'):self.load()
    def test_wrong_target_or_cut_refused(self):
        t=self.task();t['context']['target_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'context input'):self.load()
    def test_missing_score_artifact_refused(self):
        t=self.task();p=self.store.doc(t['program_results'][0]);p['artifacts'].pop('score_records');t['program_results'][0]=self.store.replace(t['program_results'][0],p)
        with self.assertRaisesRegex(ValueError,'seven'):self.load()
    def test_wrong_version_and_consumer_drift(self):
        self.task();self.store.wrong_version=True
        with self.assertRaisesRegex(ValueError,'version/bytes'):self.load()
        self.store.wrong_version=False;self.spec['consumer_source_sha256']['dltrain.labels']='0'*64
        with self.assertRaisesRegex(ValueError,'source drift'):self.load()
    def test_all_validation_no_fake_optimizer(self):
        self.task('validation');loaded=self.load();self.assertIsNone(loaded.data)
        self.assertEqual(loaded.receipt['state'],'validated_no_optimizer_universe');self.assertEqual(loaded.receipt['counts']['fit_contexts'],0)
    def test_numeric_variants_are_not_four_distinct_skeletons(self):
        self.task(programs=8,program_texts=[f'(pipe X (ridge alpha={i+1}))' for i in range(8)])
        loaded=self.load();plan=next(iter(loaded.receipt['plans'].values()))
        self.assertEqual(plan['stream'],'feasibility');self.assertEqual(plan['reason'],'FEW_SKELETONS');self.assertEqual(len(plan['teachers']),1)
    def test_all_tokenizer_refusals_preserved(self):
        texts=['(pipe X (ridge))','(pipe X'+(' (ridge)'*40)+')']
        self.task(programs=2,program_texts=texts);loaded=self.load();r=loaded.receipt
        self.assertEqual(r['counts']['programs_verified'],2);self.assertEqual(r['counts']['records_kept'],1)
        self.assertEqual(r['counts']['records_refused'],1);self.assertEqual(r['refusals'][0]['reason'],'over_max_seq')
        self.assertEqual(len(next(iter(r['contexts'].values()))['programs']),2)
    def test_sampler_collate_members_are_frozen_and_validation_absent(self):
        from dltrain import sampler
        a=self.task(programs=8);b=self.task('validation',programs=8);loaded=self.load()
        draw=sampler.MacroSampler(loaded.data.fit,bytes(32),loaded.data.config).draw(0)
        batch=sampler.collate(draw,loaded.data)
        self.assertTrue(batch.member_ids);self.assertEqual(set(batch.ctx_ids),{Context(**a['context']).identity})
        for ctx,program in batch.member_ids:self.assertIn(program,{r['program_sha256']for r in loaded.data.corpus.records[ctx]})
    def test_empty_programs_accounted_not_fabricated_failure(self):
        self.task(programs=0);loaded=self.load();self.assertIsNone(loaded.data);self.assertEqual(loaded.receipt['counts']['records_kept'],0)
    def test_baseline_reconstruction_not_same_fold_mean(self):
        folds=np.array([0]*25+[1]*25,dtype=np.uint8);y=np.r_[np.arange(25),np.arange(25)+100.].astype('<f8')
        raw=b'JTT1'+struct.pack('<IIIB',50,0,1,2)+folds.tobytes()+np.column_stack([y*.2,y]).astype('<f8').tobytes()
        ev={'inner_cv':[{'index':i,'kind':'inner_cv','baseline_metrics':{'rmse':math.sqrt(np.mean((y[folds==i]-np.mean(y[folds!=i]))**2))}}for i in (0,1)]}
        self.assertEqual(len(S.mean_baselines(raw,ev)),2)
        ev['inner_cv'][0]['baseline_metrics']['rmse']=float(np.std(y[folds==0]))
        with self.assertRaisesRegex(ValueError,'opposite-fold'):S.mean_baselines(raw,ev)
    def test_no_outer_rows_and_read_budget(self):
        t=self.task();raw=self.raws[t['variant_identity']];bad=raw[:8]+struct.pack('<I',1)+raw[12:]
        with self.assertRaisesRegex(ValueError,'stripped'):S.decode_input(bad)
        reader=S._Reader(self.store);ref={'bucket':'x','key':'x','version_id':'v','sha256':'a'*64,'bytes':S.MAX_OBJECT+1}
        with self.assertRaisesRegex(ValueError,'budget'):reader.read(ref)
        self.assertEqual(self.store.calls,0)
    def test_receipt_create_only_and_duplicate_json(self):
        self.task();loaded=self.load()
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp).resolve()/'receipt.json';S.save_receipt(loaded,path)
            with self.assertRaises(FileExistsError):S.save_receipt(loaded,path)
        with self.assertRaisesRegex(ValueError,'duplicate'):S.document(b'{"x":1,"x":2}')
        with self.assertRaises(ValueError):S.document(b'{"x":1e999}')

if __name__=='__main__':unittest.main()
