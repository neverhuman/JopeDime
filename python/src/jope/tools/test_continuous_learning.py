import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import continuous_learning as c
import experiment_runtime as runtime


class ContinuousLearningTests(unittest.TestCase):
    def test_child_evaluator_uses_verified_manifest_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'cpu').write_bytes(b'bound executable fixture')
            manifest = dict(cpu_executable='cpu', bindings={'cpu': c.sha(root / 'cpu')})
            with patch.dict(c.os.environ, {'JOPE_CPU_EVALUATOR': '/unbound/neural'}):
                env = c.child_environment(root, manifest)
            self.assertEqual(env['JOPE_CPU_EVALUATOR'], str(root / 'cpu'))
            self.assertEqual(env['JOPE_GP_EXECUTABLE'], env['JOPE_CPU_EVALUATOR'])
            (root / 'cpu').write_bytes(b'replaced executable')
            with self.assertRaisesRegex(RuntimeError, 'CPU evaluator is not bound'):
                c.child_environment(root, manifest)
            manifest['bindings'] = {}
            with self.assertRaisesRegex(RuntimeError, 'CPU evaluator is not bound'):
                c.child_environment(root, manifest)

    def test_pending_proposals_do_not_reopen_historical_queue_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);supervisor=self.scheduler(root)
            supervisor.s['generations']=[dict(number=0,path='must-not-open',stage=4),
                dict(number=1,path='pending',stage=2,corpus='corpus')]
            supervisor.generation_work()
            self.assertEqual(supervisor.s['tasks']['proposal']['generation'],1)

    def test_queue_count_stops_at_reserve_before_older_generation_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);supervisor=self.scheduler(root)
            newest=self.admitted_generation(root,supervisor,1000,name='newest',number=1)
            supervisor.s['generations']=[dict(number=0,path='must-not-open',stage=4),
                newest]
            supervisor.s['tasks']['pack']={}
            supervisor.generation_work()
            inventory=supervisor.s['proposal_input_inventory']
            self.assertEqual(inventory['scanned_jobs'],312)
            self.assertEqual(inventory['validated_proposal_metadata'],312)
            self.assertIsNone(inventory['verified_proposal_inputs'])
            self.assertEqual(inventory['unverified_manifests'],0)
            self.assertFalse(inventory['scan_complete'])
            self.assertNotIn('proposal',supervisor.s['tasks'])

    def test_monitor_upgrade_records_new_identity_without_overwriting_old_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);supervisor=self.scheduler(root)
            (root/'runs').mkdir()
            (root/'checkpoint').mkdir();(root/'corpus').mkdir()
            c.atomic(root/'checkpoint/state.json',{})
            c.atomic(root/'corpus/index.json',{})
            c.atomic(root/'panel.json',{})
            supervisor.m.update(monitor_executable='monitor',monitor_panel='panel.json',monitor_corpus='corpus',
                                bindings={'monitor':'old-exe'})
            supervisor.s.update(serving_checkpoint='checkpoint',candidates=[],monitored=[c.sha(root/'checkpoint/state.json')])
            supervisor.monitoring_work()
            task=supervisor.s['tasks']['monitor']
            c.atomic(root/task['receipt'],{'binding':{'checkpoint_sha256':task['checkpoint_sha256']}})
            supervisor.finish_task('monitor')
            supervisor.monitoring_work();self.assertNotIn('monitor',supervisor.s['tasks'])
            supervisor.m['bindings']['monitor']='new-exe'
            supervisor.monitoring_work()
            self.assertNotEqual(task['receipt'],supervisor.s['tasks']['monitor']['receipt'])
            self.assertTrue((root/task['receipt']).is_file())

    def test_unqualified_training_waits_without_consuming_discoveries(self):
        with tempfile.TemporaryDirectory() as folder:
            supervisor=self.scheduler(Path(folder))
            supervisor.m['automatic_training']=False
            supervisor.s.update(snapshots=[{'corpus':'fresh'}],consumed_training_snapshots=[])
            before=json.loads(json.dumps(supervisor.s))
            supervisor.training_work()
            self.assertEqual(supervisor.s,before)

    def test_progress_hold_latches_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            supervisor=self.scheduler(root)
            supervisor.journal=root/'journal.jsonl'
            (root/'runs').mkdir()
            c.atomic(root/'runs/continuous-progress.json',{
                'automatic_training_hold':{'status':'held','block_new_training':True}})
            supervisor.s.update(snapshots=[{'corpus':'fresh'}],consumed_training_snapshots=[])
            supervisor.training_work()
            self.assertTrue(supervisor.s.get('automatic_training_hold'))
            self.assertNotIn('train', supervisor.s['tasks'])

    def test_three_diagnostic_rejects_hold_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            supervisor=self.scheduler(root)
            supervisor.journal=root/'journal.jsonl'
            for i in range(3):
                self.diagnostic(root, supervisor.journal, str(i)*64)
            supervisor.s.update(snapshots=[{'corpus':'fresh'}],consumed_training_snapshots=[])
            supervisor.training_work()
            self.assertTrue(supervisor.s.get('automatic_training_hold'))
            self.assertNotIn('train', supervisor.s['tasks'])

    def test_packing_preserves_three_worker_wave_proposal_reserve(self):
        for remaining,refill in ((103,True),(104,True),(208,True),(311,True),(312,False)):
            with self.subTest(remaining=remaining),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);supervisor=self.scheduler(root)
                supervisor.s['generations']=[self.admitted_generation(root,supervisor,remaining)]
                supervisor.s['tasks']['pack']={'kind':'pack'}
                with patch.object(c,'verify_checkpoint'):supervisor.generation_work()
                self.assertEqual('proposal' in supervisor.s['tasks'],refill)

    def test_reference_format_or_validation_growth_cannot_schedule_repeated_training(self):
        before=dict(schema='parallel.teaching_corpus.v1',groups=[
            dict(context=0,partition='meta_train',records=list(range(8))),
            dict(context=1,partition='meta_validation',records=list(range(8,16)))])
        after=dict(schema='parallel.teaching_shards.v1',label_policy='reference_v1',
            contexts=[dict(partition='meta_train'),dict(partition='meta_validation')],
            shards=[dict(context=0,records=8),dict(context=1,records=100)])
        self.assertEqual(c.reference_training_growth(before,after),0)
        after['shards'][0]['records']=9
        self.assertEqual(c.reference_training_growth(before,after),1)
        after['label_policy']='strict_midrank_v4'
        self.assertIsNone(c.reference_training_growth(before,after))

    def diagnostic(self, root, journal, checkpoint, outcome='rejected'):
        def artifact(name,value):
            path=root/name;c.atomic(path,value)
            return dict(path=name,sha256=c.sha(path))
        binding=dict(checkpoint_sha256=checkpoint,executable_sha256='e'*64,
                     corpus_sha256='f'*64,panel_sha256='d'*64)
        reference=artifact('reference.json',dict(schema='parallel.frozen_monitoring.v1',binding=dict(binding,checkpoint_sha256='0'*64),
            sequence_nll=1.,ranking_bt=1.,ranking_listwise=1.))
        monitor=artifact(checkpoint+'-monitor.json',dict(schema='parallel.frozen_monitoring.v1',binding=binding,
            sequence_nll=1.1 if outcome=='rejected' else 1.,ranking_bt=1.,ranking_listwise=1.))
        nomination=artifact(checkpoint+'-nomination.json',dict(schema='parallel.training_nomination_result.v1',
            status='nominated',training_only=True,checkpoint_sha256=checkpoint))
        completion=dict(exit_code=0,expired=False,receipt_sha256=monitor['sha256'])
        runtime.append(journal,dict(event='original_process_completion',kind='monitor',completion=completion))
        evidence=artifact(checkpoint+'.json',dict(schema='parallel.training_diagnostic.v1',
            checkpoint_sha256=checkpoint,outcome=outcome,nomination=nomination,reference=reference,monitor=monitor,
            original_completion=completion,policy=dict(sequence_nll=.005,ranking_bt=.01,ranking_listwise=.01),
            reasons=['sequence_nll'] if outcome=='rejected' else []))
        runtime.append(journal,dict(event='diagnostic_'+outcome,checkpoint_sha256=checkpoint,evidence=evidence))

    def test_diagnostic_replay_uses_distinct_models_and_full_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); supervisor=self.scheduler(root)
            self.diagnostic(root, supervisor.journal, 'a'*64)
            for _ in range(300):
                runtime.append(supervisor.journal, {'event':'heartbeat'})
            self.diagnostic(root, supervisor.journal, 'a'*64)
            self.diagnostic(root, supervisor.journal, 'b'*64)
            self.assertFalse(supervisor.automatic_training_hold())
            self.diagnostic(root, supervisor.journal, 'c'*64)
            self.assertTrue(supervisor.automatic_training_hold())
            restarted=self.scheduler(root)
            self.assertTrue(restarted.automatic_training_hold())
            self.assertEqual(len(restarted.s['diagnostic_journal_index']['seen']), 3)

    def test_monitor_produces_original_diagnostic_for_intake_nomination(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); supervisor=self.scheduler(root)
            checkpoint='a'*64
            self.diagnostic(root,supervisor.journal,checkpoint)
            runtime.journal(supervisor.journal)
            original=c.read(root/(checkpoint+'.json'))
            supervisor.s['candidates']=[dict(checkpoint_sha256=checkpoint,nomination=original['nomination'])]
            supervisor.m['diagnostic_reference']=original['reference']
            task=dict(checkpoint_sha256=checkpoint,original_completion=original['original_completion'])
            supervisor.record_diagnostic(task,root/original['monitor']['path'])
            receipt=root/'diagnostics'/(checkpoint+'.json')
            self.assertEqual(c.read(receipt),original)
            self.assertEqual(runtime.journal(supervisor.journal)[-1]['event'],'diagnostic_rejected')

    def test_diagnostic_replay_refuses_orphaned_completion_and_rewritten_measurement(self):
        for mutation in ('orphan','measurement'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                root=Path(folder); supervisor=self.scheduler(root)
                self.diagnostic(root,supervisor.journal,'a'*64)
                if mutation=='orphan':
                    supervisor.journal.write_bytes(supervisor.journal.read_bytes().split(b'\n',1)[1])
                else:
                    c.atomic(root/('a'*64+'-monitor.json'),{'changed':True})
                with self.assertRaises(RuntimeError):supervisor.automatic_training_hold()

    def test_corrupt_and_anonymous_diagnostics_are_not_silently_skipped(self):
        for data in ('{bad}\n', '{}', '{"event":"diagnostic_rejected"}\n'):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as folder:
                root=Path(folder); supervisor=self.scheduler(root)
                supervisor.journal.write_text(data)
                with self.assertRaises(RuntimeError):
                    supervisor.automatic_training_hold()

    def test_missing_or_boolean_funding_never_schedules_training(self):
        for value in (None, False, True):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                supervisor=self.scheduler(Path(folder))
                supervisor.m['automatic_training']=True
                if value is not None:
                    supervisor.m['funded_task']=value
                supervisor.s.update(snapshots=[{'corpus':'fresh'}],consumed_training_snapshots=[])
                supervisor.training_work()
                self.assertNotIn('train', supervisor.s['tasks'])
                self.assertEqual(supervisor.s['consumed_training_snapshots'], [])

    def test_new_training_requires_exact_reservation_and_launch_consumes_once(self):
        import time

        from qualification_budget import Budget
        from test_qualification_budget import synthetic_domain
        with tempfile.TemporaryDirectory() as folder, synthetic_domain(folder):
            root=Path(folder); supervisor=self.scheduler(root)
            (root/'checkpoint').mkdir(); (root/'corpus').mkdir()
            c.atomic(root/'corpus/index.json', {})
            # Metadata-only checkpoint fixture; bind_task requires the complete
            # parent contract even though this test never loads or trains it.
            for name in ('model.ot', 'optimizer.ot'):
                (root/'checkpoint'/name).write_bytes(b'synthetic checkpoint fixture')
            c.atomic(root/'checkpoint/optimizer.json', dict(step=32))
            c.atomic(root/'checkpoint/config.json', {})
            c.atomic(root/'checkpoint/state.json', dict(step=32, seed=42, global_batch=2048,
                model_sha256=c.sha(root/'checkpoint/model.ot'),
                optimizer_sha256=c.sha(root/'checkpoint/optimizer.ot'),
                optimizer_metadata_sha256=c.sha(root/'checkpoint/optimizer.json'),
                config_sha256=c.sha(root/'checkpoint/config.json'),
                corpus_sha256=c.sha(root/'corpus/index.json'), vocabulary_sha256='synthetic vocabulary',
                sampler_id='synthetic sampler', ranking_backward='synthetic objective',
                runtime_objective='synthetic runtime objective', rng='synthetic rng'))
            supervisor.m['steps']=1000
            task=dict(kind='train', command=['verified-training'], receipt='train/complete.json',
                output='train', corpus='corpus', checkpoint='checkpoint')
            identity=supervisor.training_task_identity(task)
            Budget(root).reserve('one', node_seconds=300, capability='regression_training',
                task_sha256=identity, deadline=time.time()+250)
            supervisor.m['funded_task']=dict(domain_id='synthetic-tests', identity='one', task_sha256=identity)
            supervisor.new_task(**task)
            scheduled=supervisor.s['tasks']['train']
            self.assertEqual(supervisor.training_task_identity(scheduled), identity)
            Budget(root).verify(scheduled['reservation'], identity, consume=True)
            with self.assertRaisesRegex(RuntimeError, 'already consumed'):
                Budget(root).verify(scheduled['reservation'], identity, consume=True)
            del supervisor.s['tasks']['train']
            task['command']=['different-training']
            with self.assertRaises(RuntimeError):
                supervisor.new_task(**task)
            self.assertNotIn('train', supervisor.s['tasks'])

    def scheduler(self, root):
        supervisor=c.Supervisor.__new__(c.Supervisor)
        supervisor.root=supervisor.output=root
        supervisor.manifest_sha='test'
        supervisor.state_path=root/'state.json'
        supervisor.journal=root/'journal.jsonl'
        supervisor.m=dict(gp_workers=104, cpu_executable='bin/gp', neural_executable='bin/neural',
            contexts='contexts.json', contexts_sha256='contexts', rng_seed=42, cuts=2, cut_start=0)
        supervisor.s=dict(tasks={},generations=[],checkpoint='checkpoint',serving_corpus='corpus')
        return supervisor

    def admitted_generation(self, root, supervisor, count, name='old', number=0, partition='meta_train'):
        (root/'checkpoint').mkdir(exist_ok=True)
        c.atomic(root/'checkpoint/state.json',{})
        (root/'prepared').mkdir(exist_ok=True)
        cut=dict(index=0,train_indices=[0,1],validation_indices=[2],training_sha256='a'*64,
            validation_sha256='b'*64,preprocessing_sha256='c'*64)
        c.atomic(root/'prepared/prepared.json',dict(schema='parallel.prepared.v1',dataset='fixture',
            family='fixture',meta_partition=partition,cuts=[cut],task='regression',
            family_partition_frozen=True,clean_evaluation=True,qualification_only=False,clean_status='clean'))
        (root/'context.bin').write_bytes(b'\0'*8192)
        row=dict(prepared='prepared',prepared_sha256=c.sha(root/'prepared/prepared.json'),
            family='fixture',dataset='fixture',partition=partition,cut=0,path='context.bin',sha256=c.sha(root/'context.bin'))
        c.atomic(root/'context.json',dict(schema='parallel.encoding.v1',output_sha256=row['sha256'],
            prepared_sha256=row['prepared_sha256'],dataset='fixture',family='fixture',partition=partition,cut=0,
            encoder_spec_sha256='d'*64,training_sha256=cut['training_sha256'],
            preprocessing_sha256=cut['preprocessing_sha256'],
            selected_training_rows_sha256=c.hashlib.sha256(b'[0,1]').hexdigest(),
            dataset_dimensions=1536,target_dimensions=512))
        c.atomic(root/'seeds.json',dict(schema='parallel.seed_population.v1',prepared_sha256=row['prepared_sha256'],
            cut=0,training_sha256=cut['training_sha256'],context_sha256=row['sha256'],checkpoint_sha256='e'*64,
            candidates=[dict(genome=f'(metadata-fixture-{i})',origin='generic') for i in range(80)]))
        row['receipt_sha256']=c.sha(root/'context.json')
        c.atomic(root/'contexts.json',dict(schema='parallel.context_index.v1',dimensions=2048,
            qualification=False,assembly_sha256='d'*64,encodings=[row]))
        supervisor.m['contexts_sha256']=c.sha(root/'contexts.json')
        generation=dict(number=number,path=name,stage=4,checkpoint='checkpoint',
            checkpoint_sha256=c.sha(root/'checkpoint/state.json'))
        jobs=dict(schema='parallel.gp_jobs.v1',qualification=False,jobs=[dict(output=f'{name}/search-{i}',
            generation=number,rng_seed=i,parent_checkpoint='checkpoint',
            parent_checkpoint_sha256=generation['checkpoint_sha256'],prepared='prepared',
            prepared_sha256=row['prepared_sha256'],cut=0,seeds='seeds.json',
            seeds_sha256=c.sha(root/'seeds.json')) for i in range(count)])
        self.admit_jobs(root,supervisor,generation,jobs)
        return generation

    def admit_jobs(self, root, supervisor, generation, jobs):
        path=root/generation['path']/'jobs.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        c.atomic(path,jobs)
        digest=c.sha(path)
        (root/'runs/queue-receipts').mkdir(parents=True,exist_ok=True)
        c.atomic(root/'runs/queue-receipts'/(digest+'.json'),dict(
            schema='parallel.encoded_gp_admission.v1',jobs_sha256=digest,
            contexts_sha256=supervisor.m['contexts_sha256'],
            campaign_contexts_sha256=supervisor.m['contexts_sha256'],campaign_bound=True,
            qualification=jobs['qualification'],inputs_rechecked_before_merge=True,verified_contexts=len(jobs['jobs'])))

    def rebind_fixture(self, root, supervisor, generation):
        """Keep hashes consistent so rejection must inspect the invalid metadata."""
        prepared_sha=c.sha(root/'prepared/prepared.json')
        encoding=c.read(root/'context.json'); encoding['prepared_sha256']=prepared_sha
        c.atomic(root/'context.json',encoding)
        seeds=c.read(root/'seeds.json')
        if isinstance(seeds,dict):
            seeds['prepared_sha256']=prepared_sha
            c.atomic(root/'seeds.json',seeds)
        contexts=c.read(root/'contexts.json')
        contexts['encodings'][0].update(prepared_sha256=prepared_sha,receipt_sha256=c.sha(root/'context.json'))
        c.atomic(root/'contexts.json',contexts)
        supervisor.m['contexts_sha256']=c.sha(root/'contexts.json')
        jobs=c.read(root/generation['path']/'jobs.json')
        for job in jobs['jobs']:
            job.update(prepared_sha256=prepared_sha,seeds_sha256=c.sha(root/'seeds.json'))
        self.admit_jobs(root,supervisor,generation,jobs)

    def test_queue_metadata_rejects_native_invalid_but_consistently_rebound_values(self):
        cases=[('prepared/prepared.json','task','binary'),
               ('prepared/prepared.json','family_partition_frozen',False),
               ('prepared/prepared.json','clean_evaluation',False),
               ('prepared/prepared.json','qualification_only',0),
               ('context.json','cut',False),('context.json','dataset_dimensions',True),
               ('context.json','encoder_spec_sha256','f'*64),
               ('context.json','training_sha256','f'*64),
               ('context.json','preprocessing_sha256','f'*64),
               ('context.json','selected_training_rows_sha256','f'*64),
               ('seeds.json','cut',False),('seeds.json','candidates',[])]
        for name,key,value in cases:
            with self.subTest(name=name,key=key),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);supervisor=self.scheduler(root)
                generation=self.admitted_generation(root,supervisor,1)
                document=c.read(root/name);document[key]=value;c.atomic(root/name,document)
                self.rebind_fixture(root,supervisor,generation)
                result=c.proposal_input_inventory(root,[generation],supervisor.m)
                self.assertEqual(result['validated_proposal_metadata'],0)
                self.assertIsNone(result['verified_proposal_inputs'])
                self.assertIsNone(result['runnable_jobs'])

    def test_queue_metadata_rejects_boolean_unsigned_identity_fields(self):
        for target in ('generation_number','generation_stage','job_generation','rng_seed','job_cut',
                       'context_cut','prepared_cut','verified_contexts','qualification'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);supervisor=self.scheduler(root)
                generation=self.admitted_generation(root,supervisor,1)
                jobs=c.read(root/'old/jobs.json')
                if target=='generation_number':generation['number']=False
                elif target=='generation_stage':generation['stage']=True
                elif target in ('job_generation','rng_seed','job_cut'):
                    key={'job_generation':'generation','rng_seed':'rng_seed','job_cut':'cut'}[target]
                    jobs['jobs'][0][key]=False
                    self.admit_jobs(root,supervisor,generation,jobs)
                elif target=='qualification':
                    jobs['qualification']=0;self.admit_jobs(root,supervisor,generation,jobs)
                elif target=='verified_contexts':
                    path=root/'runs/queue-receipts'/(c.sha(root/'old/jobs.json')+'.json')
                    document=c.read(path);document['verified_contexts']=True;c.atomic(path,document)
                else:
                    path=root/('contexts.json' if target=='context_cut' else 'prepared/prepared.json')
                    document=c.read(path)
                    document['encodings' if target=='context_cut' else 'cuts'][0][
                        'cut' if target=='context_cut' else 'index']=False
                    c.atomic(path,document);self.rebind_fixture(root,supervisor,generation)
                result=c.proposal_input_inventory(root,[generation],supervisor.m)
                self.assertEqual(result['validated_proposal_metadata'],0)

    def test_queue_checks_metadata_before_context_payload_and_denies_raw_paths(self):
        for failure in ('receipt','data','sealed'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);supervisor=self.scheduler(root)
                generation=self.admitted_generation(root,supervisor,1)
                if failure=='receipt':
                    encoding=c.read(root/'context.json');encoding['partition']='meta_test'
                    c.atomic(root/'context.json',encoding)
                else:
                    contexts=c.read(root/'contexts.json');contexts['encodings'][0]['path']=failure+'/context.bin'
                    c.atomic(root/'contexts.json',contexts)
                self.rebind_fixture(root,supervisor,generation)
                original=c.os.open;opened=[]
                def guarded(path,*args,**kwargs):
                    opened.append(Path(path))
                    if Path(path).name=='context.bin':
                        raise AssertionError('payload opened before metadata exclusion')
                    return original(path,*args,**kwargs)
                with patch.object(c.os,'open',side_effect=guarded):
                    result=c.proposal_input_inventory(root,[generation],supervisor.m)
                self.assertEqual(result['validated_proposal_metadata'],0)
                self.assertFalse(any('data' in path.parts or 'sealed' in path.parts for path in opened))

    def test_208_metadata_reserve_yields_to_a_stage_that_actually_schedules(self):
        for count in (207,208,311):
            with self.subTest(count=count),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);supervisor=self.scheduler(root)
                supervisor.s['generations']=[self.admitted_generation(root,supervisor,count)]
                def stage():
                    if not supervisor.s['tasks']:
                        supervisor.s['tasks']['monitor']={'kind':'monitor','admission':'still required'}
                with patch.object(supervisor,'monitoring_work',side_effect=stage),\
                        patch.object(supervisor,'training_work'),patch.object(c,'verify_checkpoint'):
                    supervisor.gpu_work()
                self.assertEqual(set(supervisor.s['tasks']),{'proposal' if count<208 else 'monitor'})

    def test_208_metadata_reserve_tops_up_when_other_stages_do_not_schedule(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);supervisor=self.scheduler(root)
            supervisor.s['generations']=[self.admitted_generation(root,supervisor,208)]
            with patch.object(supervisor,'monitoring_work'),patch.object(supervisor,'training_work'),\
                    patch.object(c,'verify_checkpoint'):
                supervisor.gpu_work()
            self.assertEqual(set(supervisor.s['tasks']),{'proposal'})

    def test_missing_search_without_admission_is_not_a_reserve(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); supervisor=self.scheduler(root)
            generation=self.admitted_generation(root,supervisor,312)
            jobs_path=root/'old/jobs.json'
            (root/'runs/queue-receipts'/(c.sha(jobs_path)+'.json')).unlink()
            supervisor.s['generations']=[generation]
            with patch.object(c,'verify_checkpoint'):supervisor.generation_work()
            inventory=supervisor.s['proposal_input_inventory']
            self.assertEqual(inventory['validated_proposal_metadata'],0)
            self.assertIsNone(inventory['verified_proposal_inputs'])
            self.assertEqual(inventory['unverified_manifests'],1)
            self.assertIn('proposal',supervisor.s['tasks'])
            self.assertIsNone(inventory['eligible_jobs'])
            self.assertIsNone(inventory['runnable_jobs'])

    def test_queue_inventory_distinguishes_verified_ambiguous_invalid_and_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); supervisor=self.scheduler(root)
            generation=self.admitted_generation(root,supervisor,5)
            jobs=c.read(root/'old/jobs.json')
            # A receipt with a missing committed result cannot become an unstarted job.
            (root/'old/search-1').mkdir();(root/'old/search-2').mkdir()
            c.atomic(root/'old/search-1/progress.json',dict(phase='running'))
            c.atomic(root/'old/search-2/search.json',dict(unverified=True))
            jobs['jobs'][3]['seeds']='missing-population.json'
            jobs['jobs'][4]['rng_seed']=0
            self.admit_jobs(root,supervisor,generation,jobs)
            inventory=c.proposal_input_inventory(root,[generation],supervisor.m)
            for field in ('validated_proposal_metadata','ambiguous_outputs','completion_present',
                          'invalid_inputs','duplicate_identities'):
                self.assertEqual(inventory[field],1,field)
            self.assertEqual(inventory['planned_unfinished'],4)
            self.assertTrue(inventory['scan_complete'])
            self.assertIsNone(inventory['eligible_jobs'])
            self.assertIsNone(inventory['runnable_jobs'])

    def test_queue_inventory_excludes_sealed_or_withdrawn_inputs(self):
        for failure in ('sealed','withdrawn'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as folder:
                root=Path(folder); supervisor=self.scheduler(root)
                generation=self.admitted_generation(root,supervisor,1,
                    partition='meta_test' if failure=='sealed' else 'meta_train')
                if failure=='withdrawn':c.atomic(root/'old/invalidated.json',{})
                inventory=c.proposal_input_inventory(root,[generation],supervisor.m)
                self.assertEqual(inventory['ineligible_inputs'],1)
                self.assertEqual(inventory['validated_proposal_metadata'],0)

    def test_queue_inventory_rejects_changed_bound_inputs(self):
        for name in ('contexts.json','checkpoint/state.json','prepared/prepared.json',
                     'context.bin','context.json','seeds.json','old/jobs.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as folder:
                root=Path(folder); supervisor=self.scheduler(root)
                generation=self.admitted_generation(root,supervisor,1)
                (root/name).write_bytes(b'{}')
                inventory=c.proposal_input_inventory(root,[generation],supervisor.m)
                self.assertEqual(inventory['validated_proposal_metadata'],0)
                self.assertIsNone(inventory['runnable_jobs'])

    def test_queue_inventory_bounds_invalid_history_without_inventing_a_shortage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); supervisor=self.scheduler(root)
            generation=self.admitted_generation(root,supervisor,1)
            supervisor.s['generations']=[generation]+[
                dict(number=i+1,path=f'absent-{i}',stage=4) for i in range(257)]
            supervisor.generation_work()
            inventory=supervisor.s['proposal_input_inventory']
            self.assertTrue(inventory['scan_limit_reached'])
            self.assertEqual(inventory['scanned_generations'],256)
            self.assertFalse(inventory['scan_complete'])
            self.assertNotIn('proposal',supervisor.s['tasks'])
            position=inventory['cursor']['generation']
            # Persist/reload just as the supervisor does across process restarts.
            supervisor.s=json.loads(json.dumps(supervisor.s))
            with patch.object(c,'verify_checkpoint'):supervisor.generation_work()
            inventory=supervisor.s['proposal_input_inventory']
            self.assertTrue(inventory['census_finished'])
            self.assertIsNone(inventory['cursor'])
            self.assertEqual(position,1)
            self.assertEqual(inventory['scanned_generations'],258)
            self.assertEqual(inventory['validated_proposal_metadata'],1)
            self.assertIn('proposal',supervisor.s['tasks'])

    def test_queue_inventory_resumes_inside_a_large_unchanged_job_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);supervisor=self.scheduler(root)
            generation=self.admitted_generation(root,supervisor,4097)
            jobs=c.read(root/'old/jobs.json')
            for job in jobs['jobs'][:-1]:job['seeds']='absent-population.json'
            self.admit_jobs(root,supervisor,generation,jobs)
            first=c.proposal_input_inventory(root,[generation],supervisor.m)
            self.assertTrue(first['scan_limit_reached'])
            self.assertEqual(first['cursor']['job'],4096)
            second=c.proposal_input_inventory(root,[generation],supervisor.m,json.loads(json.dumps(first)))
            self.assertTrue(second['census_finished'])
            self.assertEqual(second['scanned_jobs'],4097)
            self.assertEqual(second['invalid_inputs'],4096)
            self.assertEqual(second['validated_proposal_metadata'],1)
            self.assertEqual(second['scan_started_at'],first['scan_started_at'])

    def test_progress_retains_observation_time_and_unavailable_queue_authority(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); supervisor=self.scheduler(root)
            generation=self.admitted_generation(root,supervisor,1)
            inventory=c.proposal_input_inventory(root,[generation],supervisor.m)
            supervisor.s.update(proposal_input_inventory=inventory,snapshots=[],paid_gpu_idle_seconds=0,
                completed_blocks=0,serving_checkpoint='checkpoint',serving_status='unqualified',
                candidate_checkpoint=None,promotion_decisions=[])
            supervisor.children={};supervisor.stopping=False
            supervisor.idle_since=None;supervisor.last_tick=c.time.time()
            with patch.object(c.subprocess,'check_output',side_effect=OSError('metadata fixture')):
                supervisor.progress()
            status=c.read(root/'runs/continuous-progress.json')['proposal_input_inventory']
            self.assertEqual(status,inventory)
            self.assertEqual(status['validated_proposal_metadata'],1)
            self.assertIsNone(status['verified_proposal_inputs'])
            self.assertIsNone(status['eligible_jobs'])
            self.assertIsNone(status['runnable_jobs'])
            self.assertEqual(status['dispatch_authority'],'unavailable')

    def test_queue_refill_finishes_before_ready_training_or_monitoring(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); supervisor=self.scheduler(root)
            supervisor.s['generations']=[dict(number=0,path='generation',stage=2,corpus='corpus')]
            # No monitoring/training inputs: accessing them instead of finishing
            # the pending generation would fail as well as leave workers idle.
            supervisor.gpu_work()
            self.assertEqual(set(supervisor.s['tasks']), {'proposal'})
            self.assertIn('comparison-baseline',supervisor.s['tasks']['proposal']['command'])

    def test_queue_reserve_refills_at_two_waves_and_waits_at_three(self):
        for remaining, refill in ((208,True),(312,False)):
            with self.subTest(remaining=remaining), tempfile.TemporaryDirectory() as folder:
                root=Path(folder); supervisor=self.scheduler(root)
                supervisor.s['generations']=[self.admitted_generation(root,supervisor,remaining)]
                with patch.object(c,'verify_checkpoint'):
                    supervisor.generation_work()
                self.assertEqual('proposal' in supervisor.s['tasks'],refill)
                if refill:
                    self.assertEqual(supervisor.s['generations'][-1]['checkpoint'],'checkpoint')
                    self.assertEqual(supervisor.s['generations'][-1]['corpus'],'corpus')

    def test_queue_refill_never_interrupts_an_active_gpu_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            supervisor=self.scheduler(Path(folder))
            task=dict(kind='train',receipt='complete.json')
            supervisor.s['tasks']['train']=task
            supervisor.gpu_work()
            self.assertEqual(supervisor.s['tasks'],{'train':task})
            self.assertEqual(supervisor.s['generations'],[])

    def test_pending_report_survives_scheduler_ticks_and_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            supervisor = self.scheduler(root)
            (root / 'generation').mkdir()
            c.atomic(root / 'generation/jobs.json', dict(jobs=[]))
            supervisor.s['generations'] = [dict(number=0, path='generation', stage=4)]
            supervisor.reporting_work()
            task = dict(supervisor.s['tasks']['report'])
            for _ in range(3):
                supervisor.reporting_work()
                # GPU ownership cannot schedule a second report either.
                supervisor.s['tasks']['train'] = {}
                supervisor.gpu_work()
                del supervisor.s['tasks']['train']
            restarted = self.scheduler(root)
            restarted.s = c.read(root / 'state.json')
            restarted.reporting_work()
            self.assertEqual(restarted.s['tasks']['report'], task)

    def test_invalid_handoff_is_reported_without_crashing_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'runs').mkdir()
            supervisor=c.Supervisor.__new__(c.Supervisor); supervisor.root=root
            c.atomic(root/'runs/qualification-control.json',dict(schema='parallel.qualification_control.v1',
                active=True,requested_at=1,deadline=7202))
            self.assertFalse(supervisor.qualification_handoff())
            self.assertFalse(c.read(root/'runs/qualification-handoff.json')['ready'])
            c.atomic(root/'runs/qualification-control.json',dict(active=False))
            self.assertFalse(supervisor.qualification_handoff())

    def test_healthy_gp_wait_is_explained_but_ready_training_and_stale_gp_are_not(self):
        gp = dict(phase='running', timestamp=100, active=104)
        reason = c.idle_explanation({}, {}, gp, 101)
        self.assertIn('fresh GP results', reason)
        self.assertIsNone(c.idle_status(301, True, reason is not None))
        self.assertIsNone(c.idle_explanation({'train': {}}, {}, gp, 101))
        self.assertIsNone(c.idle_explanation({}, {}, gp, 131))
        self.assertIsNone(c.idle_explanation({}, {}, dict(gp, phase='failed'), 101))
        self.assertIsNone(c.idle_explanation({'proposal': {}}, {'pack': object()}, gp, 101))
        self.assertEqual(c.idle_explanation({'pack': {}}, {'pack': object()}, gp, 101),
                         'packing fresh targets')

    def test_snapshot_and_inactivity_thresholds(self):
        self.assertFalse(c.snapshot_due(103, True, 100, 699))
        self.assertTrue(c.snapshot_due(104, True, 100, 101))
        self.assertTrue(c.snapshot_due(1, True, 100, 700))
        self.assertFalse(c.snapshot_due(1000, False, 100, 900))
        self.assertIsNone(c.idle_status(29, True, False))
        self.assertEqual(c.idle_status(30, True, False), 'warning')
        self.assertEqual(c.idle_status(60, True, False), 'incident')
        self.assertIsNone(c.idle_status(900, True, True))
        self.assertIsNone(c.idle_status(900, False, False))

    def test_heartbeat_alone_is_not_task_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); progress=root/'packing.json'
            c.atomic(progress,dict(timestamp=1, heartbeat_sequence=1, work=dict(completed=3)))
            first=c.meaningful_progress(root,dict(kind='pack'),progress)
            c.atomic(progress,dict(timestamp=100, heartbeat_sequence=80, work=dict(completed=3)))
            self.assertEqual(first,c.meaningful_progress(root,dict(kind='pack'),progress))
            c.atomic(progress,dict(timestamp=100, heartbeat_sequence=80, work=dict(completed=4)))
            self.assertNotEqual(first,c.meaningful_progress(root,dict(kind='pack'),progress))

    def test_process_reconciliation_rejects_reused_pid(self):
        identity=dict(pid=123, start_ticks='1', boot='boot')
        child=c.SurvivingProcess(identity)
        with patch.object(c,'process_identity',return_value=identity):
            self.assertIsNone(child.poll())
        with patch.object(c,'process_identity',return_value=dict(identity,start_ticks='2')):
            self.assertEqual(child.poll(),-1)

    def fixture(self, root):
        contexts = root / 'contexts.json'
        c.atomic(contexts, dict(schema='parallel.context_index.v1', encodings=[]))
        manifest = dict(schema='parallel.continuous_learning.v1', run_until_paused=True,
            world=8, microbatch=256, gp_workers=104, steps=3000,
            contexts='contexts.json', contexts_sha256=c.sha(contexts),
            bindings={'contexts.json': c.sha(contexts)}, withdrawn_families=['withdrawn'],
            bootstrap_cycle='runs/recovery', output='runs/continuous')
        path = root / 'manifest.json'
        c.atomic(path, manifest)
        return path

    def test_check_is_read_only_and_never_launches_fresh_campaign(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.fixture(root)
            before = {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch('sys.argv', ['continuous_learning.py', '--root', str(root), '--manifest', str(path), '--check']), contextlib.redirect_stdout(io.StringIO()), patch.object(c.runtime, 'popen', side_effect=AssertionError('launched')):
                c.main()
            self.assertEqual(before, {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            with self.assertRaises(FileNotFoundError):
                c.Supervisor(root, path)

    def test_manifest_hash_and_withdrawal_are_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = self.fixture(root)
            c.atomic(root / 'contexts.json', dict(schema='parallel.context_index.v1', encodings=[dict(family='withdrawn')]))
            with self.assertRaisesRegex(RuntimeError, 'input changed'):
                c.validate_manifest(root, path)
            manifest = c.read(path)
            manifest['contexts_sha256'] = manifest['bindings']['contexts.json'] = c.sha(root / 'contexts.json')
            c.atomic(path, manifest)
            with self.assertRaisesRegex(RuntimeError, 'withdrawn cohort'):
                c.validate_manifest(root, path)

    def test_pause_resume_are_durable_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = self.fixture(root)
            for flag, expected in [('--pause', True), ('--resume', False)]:
                with patch('sys.argv', ['continuous_learning.py','--root',str(root),'--manifest',str(path),flag]):
                    c.main()
                self.assertEqual(c.read(root / 'runs/continuous-control.json')['paused'], expected)

    def test_committed_stage_cannot_be_applied_twice_after_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c.atomic(root / 'receipt.json', {'saved':True})
            supervisor = c.Supervisor.__new__(c.Supervisor)
            supervisor.root = supervisor.output = root
            supervisor.manifest_sha = 'test'
            supervisor.state_path = root / 'state.json'
            supervisor.journal = root / 'journal.jsonl'
            supervisor.s = {'generations':[{'stage':2}], 'tasks':{'proposal':{'receipt':'receipt.json','generation':0}}}
            supervisor.finish_task('proposal')
            saved = c.read(supervisor.state_path)
            self.assertEqual(saved['generations'][0]['stage'], 3)
            self.assertEqual(saved['tasks'], {})
            self.assertEqual(runtime.journal(supervisor.journal)[0]['event'], 'stage_committed')

    def test_legacy_pack_finishing_after_rollout_keeps_qualified_store_source_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);supervisor=self.scheduler(root)
            supervisor.m.update(teaching_store='store',vocabulary_sha256='vocabulary')
            supervisor.s.update(snapshots=[],teaching_sources=['first','second'])
            for incremental in (False,True):
                corpus='incremental' if incremental else 'reference'
                (root/corpus).mkdir()
                command=['packer','build-corpus']
                if incremental:command.extend(['--teaching-store',str(root/'store')])
                task=dict(receipt=corpus+'/provenance.json',output=corpus,command=command,
                          covered_searches=['search'],sources=['first','second','third'] if incremental else ['second','first'])
                c.atomic(root/task['receipt'],{})
                c.atomic(root/corpus/'index.json',dict(records=8,records_sha256='records'))
                supervisor.s['tasks']['pack']=task
                with patch.object(c,'verify_corpus',return_value=c.read(root/corpus/'index.json')):
                    supervisor.finish_task('pack')
                self.assertEqual(supervisor.s['teaching_sources'],
                                 ['first','second','third'] if incremental else ['first','second'])

    def test_torn_journal_suffix_is_preserved_before_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'journal.jsonl'
            path.write_bytes(b'{"event":"committed"}\n{"event":')
            self.assertEqual(runtime.journal(path, repair=True), [{'event':'committed'}])
            self.assertEqual(path.read_bytes(), b'{"event":"committed"}\n')
            self.assertEqual(next(path.parent.glob('*.torn-*')).read_bytes(), b'{"event":')

    def test_legacy_rng_receipt_and_invalidated_generation_cannot_teach(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = dict(generation=0, rng_seed=123, parent_checkpoint='checkpoint', parent_checkpoint_sha256='parent',
                       prepared='prepared', prepared_sha256='prepared-hash', cut=0, seeds='seeds',
                       seeds_sha256='population', output='generation/search')
            output = root / job['output']
            output.mkdir(parents=True)
            c.atomic(output / 'job-binding.json', job)
            c.atomic(output / 'observations.json', [])
            self.assertFalse(c.verified_search(root, job))
            c.atomic(output / 'search.json', dict(seeds_sha256='population'))
            with self.assertRaisesRegex(RuntimeError, 'RNG identity'):
                c.verified_search(root, job)
            c.atomic(output / 'search.json', dict(seeds_sha256='population', rng_seed=123))
            self.assertTrue(c.verified_search(root, job))
            c.atomic(output.parent / 'invalidated.json', dict(reason='mixed executable and RNG'))
            with self.assertRaisesRegex(RuntimeError, 'invalidated GP'):
                c.verified_search(root, job)

    def test_hold_blocks_new_generation_but_finishes_inflight(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);supervisor=self.scheduler(root)
            supervisor.s['automatic_training_hold']=True
            supervisor.s['generations']=[dict(number=0,path='old',stage=4)]
            supervisor.generation_work()
            self.assertNotIn('proposal', supervisor.s['tasks'])
            self.assertEqual(len(supervisor.s['generations']), 1)
            supervisor.s['generations']=[dict(number=0,path='generation',stage=2,corpus='corpus')]
            supervisor.generation_work()
            self.assertIn('proposal', supervisor.s['tasks'])
            self.assertIn('comparison-baseline', supervisor.s['tasks']['proposal']['command'])

    def test_new_generation_omits_shuffled_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);supervisor=self.scheduler(root)
            (root/'checkpoint').mkdir(); (root/'old').mkdir()
            c.atomic(root/'checkpoint/state.json',{})
            c.atomic(root/'old/jobs.json',dict(jobs=[dict(output=f'search-{i}') for i in range(100)]))
            supervisor.s['generations']=[dict(number=0,path='old',stage=4)]
            with patch.object(c,'verify_checkpoint'):
                supervisor.generation_work()
            pending=supervisor.s['generations'][-1]
            self.assertTrue(pending['omit_shuffled'])
            self.assertEqual(pending['coverage_selection_protocol'], 'family_dataset_cut_strict_v1')
            self.assertIn('propose', supervisor.s['tasks']['proposal']['command'])
            self.assertIn('--generation-manifest', supervisor.s['tasks']['proposal']['command'])
            self.assertNotIn('--shuffled', supervisor.s['tasks']['proposal']['command'])

    def test_omit_shuffled_skips_stage_one(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);supervisor=self.scheduler(root)
            pending=dict(number=1,path='generation',stage=1,omit_shuffled=True,corpus='corpus',
                         contexts='contexts.json',rng_seed=1,checkpoint='checkpoint',checkpoint_sha256='p'*64)
            supervisor.s['generations']=[pending]
            supervisor.generation_work()
            self.assertEqual(supervisor.s['generations'][0]['stage'], 2)
            self.assertIn('comparison-baseline', supervisor.s['tasks']['proposal']['command'])

    def test_assemble_jobs_without_shuffled(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            c.atomic(root/'contexts.json', {'encodings':[dict(prepared='p',cut=0,partition='meta_train',prepared_sha256='h')]})
            arms={}
            for arm in ('neural','generic'):
                path=root/arm; path.mkdir(); (path/'seeds.json').write_text('{}')
                arms[arm]=arm
                c.atomic(path/'jobs.json', dict(vocabulary='v', jobs=[dict(prepared='p',cut=0,prepared_sha256='h',
                    output=arm+'/search',seeds=arm+'/seeds.json',seeds_sha256=c.sha(path/'seeds.json'))]))
            spec=dict(contexts='contexts.json',rng_seed=123,checkpoint='checkpoint',checkpoint_sha256='parent')
            jobs=c.assemble_jobs(root,0,spec,arms)
            self.assertEqual(len(jobs['jobs']), 2)
            self.assertEqual({j['output'].split('/')[0] for j in jobs['jobs']}, {'neural','generic'})

    def test_generation_seeds_are_matched_new_and_cohort_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c.atomic(root / 'contexts.json', {'encodings':[dict(prepared='p',cut=0,partition='meta_train',prepared_sha256='h')]})
            arms = {}
            for arm in ('neural','generic','shuffled'):
                path = root / arm; path.mkdir(); (path / 'seeds.json').write_text('{}')
                arms[arm] = arm
                c.atomic(path / 'jobs.json', dict(vocabulary='v', jobs=[dict(prepared='p',cut=0,prepared_sha256='h',
                    output=arm+'/search',seeds=arm+'/seeds.json',seeds_sha256=c.sha(path/'seeds.json'))]))
            spec = dict(contexts='contexts.json',rng_seed=123,checkpoint='checkpoint',checkpoint_sha256='parent')
            first = c.assemble_jobs(root,0,spec,arms)
            second = c.assemble_jobs(root,1,spec,arms)
            self.assertEqual(len({j['rng_seed'] for j in first['jobs']}),1)
            self.assertNotEqual(first['jobs'][0]['rng_seed'],second['jobs'][0]['rng_seed'])
            c.atomic(root / 'contexts.json', {'encodings':[dict(prepared='p',cut=0,partition='meta_test',prepared_sha256='h')]})
            with self.assertRaisesRegex(RuntimeError,'outside frozen eligible'):
                c.assemble_jobs(root,2,spec,arms)


if __name__ == '__main__':
    unittest.main()
