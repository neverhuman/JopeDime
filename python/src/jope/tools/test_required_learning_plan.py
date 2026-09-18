import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import required_learning_plan as plan
from scientific_metadata import immutable


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write(root, name, value):
    path = root / name
    return dict(path=name, sha256=immutable(path, value))


def selected():
    return dict(schema='parallel.required_learning_selection.v1', seeds=[20260909, 20260910, 20260911],
        conditions=[dict(identity=name, support='narrow' if name in 'AB' else 'broader',
                         replay='broad' if name in 'AC' else 'fresh_balanced') for name in 'ABCD'])


def fixture(root, broader=True, fresh=True):
    parent = write(root, 'parent.json', dict(schema='parallel.checkpoint.v1', step=3000,
                                          vocabulary_sha256=sha('vocabulary')))
    template = dict(schema='parallel.checkpoint_continuation.v1', mode='optimizer_carry_new_window',
        optimizer='carry', exact_resume=False, sampling_seed=20260909,
        parent_checkpoint_sha256=parent['sha256'], corpus_sha256=sha('old corpus'),
        vocabulary_sha256=sha('vocabulary'), requested_step_increment=1000, parent_state=parent,
        execution_partition=dict(schema='parallel.training_partition.v1', world=8, microbatch=256,
                                 global_batch=2048, accumulation=1))
    for field in ('original_run', 'original_completion', 'original_initialization'):
        template[field] = write(root, field + '.json', dict(schema='synthetic metadata fixture'))
    supports = {}
    for name, count in (('narrow', 1), ('broader', 2 if broader else 1)):
        rows, shards = [], []
        for group in range(count):
            rows.append(dict(family='family-' + str(group), prepared_sha256=sha(str(group)),
                             cut=0, partition='meta_train'))
            shards.append(dict(context=group, records=8, sha256=sha('shard' + str(group)),
                               records_sha256=sha('records' + str(group)), labels_sha256=sha('labels' + str(group))))
        index = dict(schema='parallel.teaching_shards.v1', label_policy='strict_midrank_v4',
            vocabulary_sha256=sha('vocabulary'), context_index_sha256=sha('contexts'),
            contexts=rows, shards=shards, records=count * 8,
            admission_policy=dict(schema='parallel.label_admission.v2',
                evaluator_protocol='featmath_training_fold_normalized_v1', evaluator_sha256=sha('evaluator'),
                runtime_sha256=sha('runtime'), numerical_contract_sha256=sha('numerics'),
                authority_instance=sha('authority')))
        index_ref = write(root, name + '-index.json', index)
        policy = dict(schema='parallel.replay_policy.v2', optimizer='carry',
            parent_checkpoint_sha256=parent['sha256'], corpus_sha256=index_ref['sha256'],
            vocabulary_sha256=sha('vocabulary'), final_tests_opened=False, eligible=True, blocked_reasons=[],
            fresh_records={'0': [0]} if fresh else {}, fresh_groups=[0] if fresh else [],
            sampling_seed=7, replay='fresh_balanced', fresh_fraction=.5 if fresh else 0,
            sampling_version='lineage_fresh_program_v2')
        supports[name] = dict(corpus_index=index_ref, replay_policy=write(root, name + '-policy.json', policy))
    inputs = dict(schema='parallel.required_learning_plan_inputs.v1',
                  continuation_template=write(root, 'continuation-template.json', template), **supports)
    write(root, 'inputs.json', inputs)
    return inputs


class RequiredLearningPlanTests(unittest.TestCase):
    def prepare(self, root):
        with patch.object(plan, 'recover_selection', return_value=selected()):
            return plan.prepare(root, root / 'custody', 'inputs.json', 'study')

    def test_twelve_independent_original_branches_carry_to_four_thousand(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            result = self.prepare(root)
            self.assertEqual(len(result['branches']), 12)
            paths = set()
            for branch in result['branches']:
                self.assertEqual(branch['preparation_status'], 'metadata_prepared')
                self.assertEqual(branch['target_optimizer_step'], 4000)
                self.assertEqual(branch['new_committed_steps'], 1000)
                self.assertEqual(branch['training_ceiling_seconds'], 300)
                continuation = plan.metadata_reference(root, branch['continuation'])
                replay = plan.metadata_reference(root, continuation['replay_policy_ref'])
                self.assertEqual(continuation['sampling_seed'], branch['training_seed'])
                self.assertEqual(replay['sampling_seed'], branch['training_seed'])
                self.assertEqual(replay['optimizer'], 'carry')
                self.assertNotIn('fresh_records', continuation)
                self.assertEqual(replay['fresh_fraction'], .5 if branch['condition'] in 'BD' else 0)
                paths.add(branch['output'])
                self.assertFalse(branch['launch_allowed'])
            self.assertEqual(len(paths), 12)
            for key in ('launch_allowed', 'funding_verified', 'committed_exposure_verified',
                        'support_admission_independently_verified', 'references_opened', 'numerical_work_started'):
                self.assertFalse(result[key])

    def test_no_broader_support_preserves_unavailable_c_and_d(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root, broader=False)
            result = self.prepare(root)
            for branch in result['branches']:
                self.assertEqual(branch['preparation_status'],
                                 'unavailable' if branch['condition'] in 'CD' else 'metadata_prepared')
            self.assertFalse((root / 'study/C').exists())
            self.assertFalse((root / 'study/D').exists())

    def test_empty_fresh_pool_cannot_masquerade_as_fifty_fifty_condition(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root, fresh=False)
            result = self.prepare(root)
            self.assertTrue(all(branch['preparation_status'] == 'unavailable'
                                for branch in result['branches'] if branch['condition'] in 'BD'))
            self.assertFalse((root / 'study/B').exists())

    def test_changed_policy_rejected_before_any_branch_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            (root / 'narrow-policy.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                self.prepare(root)
            self.assertFalse((root / 'study').exists())

    def test_bool_and_cross_group_fresh_record_aliases_reject(self):
        changes = [lambda p: p.update(fresh_records={'0': [True]}),
                   lambda p: p.update(fresh_records={'0': [8]}),
                   lambda p: p.update(fresh_groups=[False]),
                   lambda p: p.update(optimizer='reset')]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                inputs = fixture(root)
                original = plan.metadata_reference(root, inputs['narrow']['replay_policy'])
                change(original)
                inputs['narrow']['replay_policy'] = write(root, 'changed-policy.json', original)
                (root / 'inputs.json').write_text(json.dumps(inputs))
                with self.assertRaises(ValueError):
                    self.prepare(root)
                self.assertFalse((root / 'study').exists())

    def test_sealed_reference_rejects_before_open(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(plan, 'custody_json', side_effect=AssertionError('payload opened')):
                with self.assertRaisesRegex(ValueError, 'before open'):
                    plan.metadata_reference(root, dict(path='sealed/targets.json', sha256=sha('targets')))

    def test_existing_manifest_and_branches_remain_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = fixture(root)
            self.prepare(root)
            original = (root / 'study/plan.json').read_bytes()
            inputs['broader'] = None
            (root / 'inputs.json').write_text(json.dumps(inputs))
            with self.assertRaisesRegex(ValueError, 'immutable'):
                self.prepare(root)
            self.assertEqual((root / 'study/plan.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
