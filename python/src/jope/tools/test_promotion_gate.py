import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import continuous_learning as controller
import promotion_gate as gate


class PromotionRequestTests(unittest.TestCase):
    def test_request_names_blocked_by_and_keeps_protocol_raising(self):
        body = gate.promotion_request('a' * 64, gate.default_blocked_by(gate.RECOVERY_PARENT))
        self.assertEqual(body['schema'], 'parallel.promotion_request.v1')
        self.assertIsNone(body['manifest'])
        self.assertIsNone(body['evidence'])
        self.assertFalse(body['promotion_producer'])
        self.assertIn('same_parent_c2f229c9', body['blocked_by'])
        self.assertIn('verify_execution_protocol_raises', body['blocked_by'])
        with self.assertRaises(RuntimeError):
            gate.verify_execution_protocol(None, {}, {})

    def test_request_refuses_empty_block_list(self):
        with self.assertRaises(RuntimeError):
            gate.promotion_request('a' * 64, [])


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='/dev/shm')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # These synthetic JSON fixtures test decision mathematics and pointer
        # recovery only. They have never been qualified execution evidence.
        self.execution_verifier = patch.object(gate, 'verify_execution_protocol')
        self.execution_verifier.start()
        self.addCleanup(self.execution_verifier.stop)

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        controller.atomic(path, value)
        return dict(path=name, sha256=gate.sha(path))

    def checkpoint(self, name, corpus, vocab):
        state = dict(step=3000, corpus_sha256=corpus, vocabulary_sha256=vocab)
        for file, key in [('model.ot', 'model_sha256'), ('optimizer.ot', 'optimizer_sha256'),
                          ('optimizer.json', 'optimizer_metadata_sha256'), ('config.json', 'config_sha256')]:
            state[key] = self.put(name + '/' + file, {'checkpoint':name})['sha256']
        return self.put(name + '/state.json', state)

    def fixture(self, improvement=.03):
        artifacts = {k:self.put(k + '.json', {'kind':k}) for k in ('corpus', 'vocabulary', 'panel', 'contexts')}
        arms = {k:self.checkpoint(k, artifacts['corpus']['sha256'], artifacts['vocabulary']['sha256'])
                for k in ('challenger', 'recovery', 'serving', 'original_serving', 'cycle25')}
        artifacts['nomination'] = self.put('nomination.json', dict(training_only=True,
            validation_results_used=False, checkpoint_sha256=arms['challenger']['sha256']))
        contexts = [dict(family='family'+str(f), partition='meta_validation', cut=cut,
                    prepared_sha256='prepared'+str(f), training_sha256='train'+str(cut),
                    validation_sha256='val'+str(cut), evaluator_sha256='evaluator',
                    vocabulary_sha256=artifacts['vocabulary']['sha256']) for f in range(8) for cut in range(2)]
        manifest = dict(schema='parallel.promotion_experiment.v1', policy=gate.POLICY,
            final_tests_opened=False, frozen_at=1, proposal_protocol='training_context_neural_pool_v1',
            metric='rmse', units='original_target', artifacts=artifacts, arms=arms,
            evaluator_sha256='evaluator', validation_contexts=contexts, rng_seeds=[1,2,3],
            population_slots=dict(neural=64, retained=16), panel_sha256='panel',
            monitor_corpus_sha256='monitor-corpus', monitor_executable_sha256='monitor-exe')
        manifest_ref = self.put('manifest.json', manifest)
        comparisons = []
        for c in contexts:
            for seed in manifest['rng_seeds']:
                for arm in sorted({a['sha256'] for a in arms.values()} | {'generic', 'shuffled:'+arms['challenger']['sha256']}):
                    i = len(comparisons)
                    rmse = 1-improvement if arm == arms['challenger']['sha256'] else 1
                    receipt = dict(context=c, rng_seed=seed, arm=arm, metric='rmse', units='original_target',
                                   status='ok', rmse=rmse, program_sha256=hashlib.sha256(b'program').hexdigest())
                    outcome = dict(status='ok', rmse=rmse, receipt=self.put(f'result-{i}.json', receipt))
                    proposal = self.put(f'proposal-{i}.json', dict(schema='parallel.pure_best_predicted_program.v1',
                        protocol=manifest['proposal_protocol'], teacher_population_consulted=False,
                        prediction=dict(program='program',program_sha256=receipt['program_sha256']),
                        origin='neural_policy', checkpoint_sha256=arm.removeprefix('shuffled:')))
                    row = dict(schema='parallel.promotion_comparison.v1', manifest_sha256=manifest_ref['sha256'],
                        context=c, rng_seed=seed, arm=arm, started_at=2, search_budget=dict(seconds=600, evaluations=1000),
                        population_slots=manifest['population_slots'], metric='rmse', units='original_target',
                        complete=True, direct=dict(**outcome, proposal=proposal), search=outcome)
                    comparisons.append(self.put(f'comparison-{i}.json', row))
        monitoring = {name:self.put('monitor-'+name+'.json', dict(binding=dict(
            checkpoint_sha256=arms[name]['sha256'], panel_sha256='panel', corpus_sha256='monitor-corpus',
            executable_sha256='monitor-exe'), sampled_examples=2048, family_balanced=True,
            final_tests_opened=False, sequence_nll=1, ranking_bt=1, ranking_listwise=1)) for name in ('challenger','recovery')}
        evidence = dict(schema='parallel.promotion_evidence.v1', manifest_sha256=manifest_ref['sha256'],
            complete=True, final_tests_opened=False, resources=dict(gpu_node_seconds=100,
            cpu_core_seconds=1000, max_concurrent_searches=16), comparisons=comparisons, monitoring=monitoring)
        candidate = dict(checkpoint='challenger', corpus='corpus', checkpoint_sha256=arms['challenger']['sha256'],
                         corpus_sha256=artifacts['corpus']['sha256'])
        return manifest_ref, evidence, candidate, arms['serving']['sha256']

    def evaluate(self, fixture):
        manifest, evidence, candidate, serving = fixture
        return gate.evaluate(self.root, manifest, self.put('evidence.json', evidence), candidate, serving)

    def test_complete_improvement_and_family_bootstrap(self):
        result = self.evaluate(self.fixture())
        self.assertEqual(result['outcome'], 'accepted', result)
        self.assertGreater(result['comparisons']['serving']['confidence_interval'][0], 0)
        self.assertGreater(result['comparisons']['original_serving']['confidence_interval'][0], 0)

    def test_self_hashed_legacy_evidence_cannot_authorize_production_promotion(self):
        self.execution_verifier.stop()
        result = self.evaluate(self.fixture())
        self.assertEqual(result['outcome'], 'inconclusive', result)
        self.assertIn('original registered evaluation', result['reasons'][0])

    def test_original_serving_reference_is_mandatory(self):
        manifest_ref, _, _, _ = self.fixture()
        manifest = gate.artifact(self.root, manifest_ref)
        del manifest['arms']['original_serving']
        with self.assertRaisesRegex(RuntimeError, 'reference checkpoints missing'):
            gate.validate_manifest(self.root, manifest)

    def test_non_regression_execution_stays_blocked(self):
        manifest_ref, _, _, _ = self.fixture()
        for task in ('binary', 'multiclass', 'timeseries'):
            manifest = gate.artifact(self.root, manifest_ref)
            manifest['task'] = task
            with self.assertRaisesRegex(RuntimeError, 'separately qualified task contract'):
                gate.validate_manifest(self.root, manifest)

    def test_zero_monitor_reference_is_inconclusive(self):
        for metric in ('sequence_nll', 'ranking_bt', 'ranking_listwise'):
            manifest, evidence, candidate, serving = self.fixture()
            for name in ('challenger', 'recovery'):
                monitor = gate.artifact(self.root, evidence['monitoring'][name])
                monitor[metric] = 0
                evidence['monitoring'][name] = self.put('zero-' + name + '.json', monitor)
            result = self.evaluate((manifest, evidence, candidate, serving))
            self.assertEqual(result['outcome'], 'inconclusive', result)
            self.assertIn('zero diagnostic reference', result['reasons'][0])

    def test_gain_against_recovery_cannot_hide_loss_against_original_serving(self):
        manifest, evidence, candidate, serving = self.fixture()
        original = gate.artifact(self.root, manifest)['arms']['original_serving']['sha256']
        for i, ref in enumerate(evidence['comparisons']):
            row = gate.artifact(self.root, ref)
            if row['arm'] != original:
                continue
            for name in ('direct', 'search'):
                receipt = gate.artifact(self.root, row[name]['receipt'])
                receipt['rmse'] = .9
                row[name]['rmse'] = .9
                row[name]['receipt'] = self.put(f'original-{name}-{i}.json', receipt)
            evidence['comparisons'][i] = self.put(f'original-comparison-{i}.json', row)
        result = self.evaluate((manifest, evidence, candidate, serving))
        self.assertEqual(result['outcome'], 'rejected', result)
        self.assertIn('insufficient direct improvement versus original_serving', result['reasons'])

    def direct_only_references(self):
        manifest_ref,evidence,candidate,serving=self.fixture()
        manifest=gate.artifact(self.root,manifest_ref)
        manifest.update(schema='parallel.promotion_experiment.v2',matched_search_arms=['challenger','serving'])
        manifest_ref=self.put('manifest-v2.json',manifest)
        evidence['manifest_sha256']=manifest_ref['sha256']
        required={candidate['checkpoint_sha256'],serving}
        for i,ref in enumerate(evidence['comparisons']):
            row=gate.artifact(self.root,ref);row['manifest_sha256']=manifest_ref['sha256']
            if row['arm'] not in required:row.update(search=None,search_budget=None,population_slots=None)
            evidence['comparisons'][i]=self.put(f'v2-comparison-{i}.json',row)
        return manifest_ref,evidence,candidate,serving

    def test_unapproved_reduced_search_scope_cannot_promote(self):
        fixture=self.direct_only_references()
        result=self.evaluate(fixture)
        self.assertEqual(result['outcome'],'inconclusive',result)

    def test_v1_still_requires_full_historical_search_panel(self):
        fixture=self.fixture();manifest,evidence,candidate,serving=fixture
        i=next(i for i,ref in enumerate(evidence['comparisons']) if gate.artifact(self.root,ref)['arm']=='generic')
        row=gate.artifact(self.root,evidence['comparisons'][i]);row['search']=None
        evidence['comparisons'][i]=self.put('missing-v1-search.json',row)
        self.assertEqual(self.evaluate((manifest,evidence,candidate,serving))['outcome'],'inconclusive')

    def test_regression_and_tie_rejected(self):
        fixture = self.fixture(-.03)
        result = self.evaluate(fixture)
        self.assertEqual(result['outcome'], 'rejected', result)

    def test_missing_changed_cross_cut_and_nonfinite_evidence_cannot_promote(self):
        fixture = self.fixture()
        manifest, original, candidate, serving = fixture
        challenger_index=next(i for i,ref in enumerate(original['comparisons'])
                              if gate.artifact(self.root,ref)['arm']==candidate['checkpoint_sha256'])
        for mutation in ('missing', 'cross-cut', 'units', 'fallback', 'nan', 'budget', 'hash'):
            evidence = copy.deepcopy(original)
            row = gate.artifact(self.root, evidence['comparisons'][challenger_index])
            if mutation == 'missing':
                evidence['comparisons'].pop()
            elif mutation == 'budget':
                evidence['resources']['max_concurrent_searches'] = 17
            elif mutation == 'hash':
                evidence['comparisons'][challenger_index]['sha256'] = 'changed'
            else:
                if mutation == 'cross-cut': row['context']['cut'] = 99
                if mutation == 'units': row['units'] = 'cut_normalized_target'
                if mutation == 'nan': row['direct']['rmse'] = float('nan')
                if mutation == 'fallback':
                    p = gate.artifact(self.root, row['direct']['proposal'])
                    p['teacher_population_consulted'] = True
                    row['direct']['proposal'] = self.put('bad-proposal.json', p)
                evidence['comparisons'][challenger_index] = self.put('bad-comparison.json', row)
            result = self.evaluate((manifest, evidence, candidate, serving))
            self.assertEqual(result['outcome'], 'inconclusive', (mutation, result))

    def supervisor(self, candidate, serving):
        s = controller.Supervisor.__new__(controller.Supervisor)
        s.root = s.output = self.root
        s.manifest_sha = 'manifest'
        s.state_path, s.journal = self.root/'state.json', self.root/'journal.jsonl'
        s.s = dict(checkpoint='serving', serving_checkpoint='serving', serving_checkpoint_sha256=serving,
                   serving_status='provisional_unqualified', candidates=[candidate], promotion_decisions=[])
        s.save('fixture')
        return s

    def test_crash_after_decision_before_pointer_and_after_pointer_is_idempotent(self):
        manifest, evidence, candidate, serving = self.fixture()
        evidence_ref = self.put('evidence.json', evidence)
        self.put('qualification/'+candidate['checkpoint_sha256']+'/request.json', dict(manifest=manifest, evidence=evidence_ref))
        supervisor = self.supervisor(candidate, serving)
        with patch.object(supervisor, 'save', side_effect=OSError('crash')):
            with self.assertRaises(OSError): supervisor.promotion_work()
        self.assertEqual(controller.read(supervisor.state_path)['checkpoint'], 'serving')
        self.assertEqual(len(list(self.root.glob('qualification/*/decision-*.json'))), 1)
        supervisor.s = controller.read(supervisor.state_path)
        supervisor.promotion_work()
        self.assertEqual(controller.read(supervisor.state_path)['checkpoint'], 'challenger')
        supervisor.s = controller.read(supervisor.state_path)
        supervisor.promotion_work()
        self.assertEqual(len(supervisor.s['promotion_decisions']), 1)

    def test_rejected_decision_preserves_serving(self):
        manifest, evidence, candidate, serving = self.fixture(-.1)
        self.put('qualification/'+candidate['checkpoint_sha256']+'/request.json',
                 dict(manifest=manifest, evidence=self.put('evidence.json', evidence)))
        supervisor = self.supervisor(candidate, serving)
        supervisor.promotion_work()
        self.assertEqual(supervisor.s['checkpoint'], 'serving')
        self.assertEqual(supervisor.s['candidates'][0]['qualification_status'], 'rejected')

    def test_versioned_proof_can_follow_an_immutable_missing_evidence_request(self):
        manifest,evidence,candidate,serving=self.fixture()
        directory='qualification/'+candidate['checkpoint_sha256']+'/'
        placeholder=self.put(directory+'request.json',dict(manifest=None,evidence=None))
        supervisor=self.supervisor(candidate,serving)
        supervisor.promotion_work()
        self.assertEqual(supervisor.s['checkpoint'],'serving')
        self.assertEqual(supervisor.s['promotion_decisions'][0]['outcome'],'inconclusive')
        self.put(directory+'request-proof-v1.json',dict(manifest=manifest,evidence=self.put('evidence.json',evidence)))
        supervisor.s=controller.read(supervisor.state_path)
        supervisor.promotion_work();supervisor.promotion_work()
        self.assertEqual(supervisor.s['checkpoint'],'challenger')
        self.assertEqual(len(supervisor.s['promotion_decisions']),2)
        self.assertEqual(gate.sha(self.root/placeholder['path']),placeholder['sha256'])

    def test_candidate_publication_crash_and_consumed_snapshot_prevent_retraining(self):
        corpus = self.put('corpus/index.json', {'records_sha256':'records'})['sha256']
        cp = self.checkpoint('candidate', corpus, 'vocab')
        parent = self.checkpoint('serving', 'old-corpus', 'vocab')
        self.put('train/latest.json', dict(checkpoint='candidate', state_sha256=cp['sha256']))
        self.put('train/complete.json', {'complete':True})
        supervisor = self.supervisor({}, parent['sha256'])
        supervisor.s.update(candidates=[], candidate_checkpoint=None, completed_blocks=0,
            corpus='old-corpus', trained_snapshots=[], consumed_training_snapshots=[],
            snapshots=[dict(corpus='corpus', timestamp=1, records_sha256='records')],
            tasks=dict(train=dict(output='train', receipt='train/complete.json', corpus='corpus', checkpoint='serving')))
        supervisor.m = dict(steps=3000, vocabulary_sha256='vocab', neural_executable='exe', bindings={'exe':'exe-sha'})
        supervisor.save('setup')
        with patch.object(supervisor, 'save', side_effect=OSError('crash')):
            with self.assertRaises(OSError): supervisor.finish_task('train')
        supervisor.s = controller.read(supervisor.state_path)
        supervisor.finish_task('train')
        self.assertEqual(supervisor.s['checkpoint'], 'serving')
        self.assertEqual(supervisor.s['candidate_checkpoint'], 'candidate')
        self.assertEqual(supervisor.s['consumed_training_snapshots'], ['corpus'])
        self.assertEqual(len(supervisor.s['candidates']), 1)
        supervisor.training_work()
        self.assertEqual(supervisor.s['tasks'], {})


if __name__ == '__main__':
    unittest.main()
