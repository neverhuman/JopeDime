import copy
import json
import tempfile
import unittest
from pathlib import Path

from protocol import BASELINES, comparable, digest, next_action, publish, validate_plan, validate_result
from scoreboard import scoreboard


def result():
    identity = dict(task='historical', input_archive_sha256='1' * 64, features_sha256='2' * 64,
                    split_sha256='3' * 64, evaluation_role='development', evaluator_sha256='4' * 64,
                    vocabulary_sha256='5' * 64, budget=4)
    return dict(schema='jope.research-result/v1', state='complete', run_id='one', branch='branch',
                source_commit='a' * 40, preregistration_sha256='b' * 64, concept='control',
                comparison_identity=identity, comparison_identity_sha256=digest(identity),
                counts=dict(optimizer_updates=1, fresh_gp_fits=0, historical_replay_slots=2, cache_reads=2, failures=0),
                scope='historical_replay', hwm_promotion=False,
                parameter_hashes={'member': {'before': '6' * 64, 'after': '7' * 64}},
                metrics={b: dict(parent_mean_regret=.1, parent_mean_quality=.2) for b in BASELINES},
                primary=dict(parents=10))


class ProtocolTests(unittest.TestCase):
    def test_complete_both_native_checkpoints(self):
        self.assertEqual(validate_result(result())['state'], 'complete')

    def test_native429_missing_refuses(self):
        r = result(); del r['metrics']['native429_qhead']
        with self.assertRaises(ValueError): validate_result(r)

    def test_nan_metric_refuses(self):
        r = result(); r['metrics']['native17773_qhead']['parent_mean_quality'] = float('nan')
        with self.assertRaises(ValueError): validate_result(r)

    def test_counter_boolean_not_number(self):
        r = result(); r['counts']['fresh_gp_fits'] = False
        with self.assertRaises(ValueError): validate_result(r)

    def test_historical_cannot_be_fresh_fits(self):
        r = result(); r['counts']['fresh_gp_fits'] = 1
        with self.assertRaises(ValueError): validate_result(r)

    def test_no_hwm_promotion(self):
        r = result(); r['hwm_promotion'] = True
        with self.assertRaises(ValueError): validate_result(r)

    def test_update_without_weight_change_refuses(self):
        r = result(); r['parameter_hashes']['member']['after'] = r['parameter_hashes']['member']['before']
        with self.assertRaises(ValueError): validate_result(r)

    def test_one_parent_not_uncertainty(self):
        r = result(); r['primary']['parents'] = 1
        with self.assertRaises(ValueError): validate_result(r)

    def test_context_field_drift_is_incompatible(self):
        a = result(); b = copy.deepcopy(a)
        b['comparison_identity']['native_comparison_context'] = {'checkpoint_role': 'other'}
        b['comparison_identity_sha256'] = digest(b['comparison_identity'])
        with self.assertRaises(ValueError): comparable(a, b)

    def test_cut_prefix_not_normalized(self):
        a = result(); b = copy.deepcopy(a)
        a['comparison_identity']['cut'] = 'trainonly:abc'
        b['comparison_identity']['cut'] = 'abc'
        for r in (a, b): r['comparison_identity_sha256'] = digest(r['comparison_identity'])
        with self.assertRaises(ValueError): comparable(a, b)

    def test_failed_result_is_retained_not_quality(self):
        r = result(); r['state'] = 'failed'; r['metrics'] = {}; r['counts']['failures'] = 1
        validate_result(r)
        with self.assertRaises(ValueError): comparable(r, result())

    def test_create_only_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'result.json'; publish(p, result())
            with self.assertRaises(FileExistsError): publish(p, result())

    def test_two_identities_stay_two_groups(self):
        a = result(); b = copy.deepcopy(a)
        b['comparison_identity']['evaluation_role'] = 'other'
        b['comparison_identity_sha256'] = digest(b['comparison_identity'])
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d) / n for n in ('a.json', 'b.json')]
            for p, r in zip(paths, (a, b)): publish(p, r)
            self.assertEqual(len(scoreboard(paths)['comparison_groups']), 2)

    def test_not_admitted_never_schedules_launch(self):
        self.assertEqual(next_action('planned', admitted=False, reviewed=True), 'blocked_name_owner_and_exact_dependency')

    def test_failed_attempt_not_retried(self):
        self.assertEqual(next_action('failed', admitted=True, reviewed=True), 'study_immutable_results_and_freeze_new_hypothesis')

    def test_running_checks_weights_and_guard(self):
        self.assertEqual(next_action('running', admitted=True, reviewed=True), 'verify_optimizer_checkpoint_guard_and_cost_progress')

    def test_plan_controls_are_fixed(self):
        p = json.loads((Path(__file__).parent / 'concept.json').read_text())
        p.update(schema='jope.research-wave/v1', input_pins={'a': 'b'}, source_sha256={'a': 'b'}, native_checkpoint_parameters={'17773': 'a', '429': 'b'})
        validate_plan(p)
        p['seeds'] = [1]
        with self.assertRaises(ValueError): validate_plan(p)

    def test_no_sealed_assessment_repurposing(self):
        p = json.loads((Path(__file__).parent / 'concept.json').read_text())
        p.update(schema='jope.research-wave/v1', input_pins={'a': 'b'}, source_sha256={'a': 'b'}, native_checkpoint_parameters={'17773': 'a', '429': 'b'})
        p['evaluation_partition'] = 'assessment'
        with self.assertRaises(ValueError): validate_plan(p)


if __name__ == '__main__': unittest.main()
