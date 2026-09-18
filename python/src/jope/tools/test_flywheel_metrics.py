#!/usr/bin/env python3
"""flywheel_metrics unit tests: hand-built fixtures plus the real local monitoring receipt."""
import hashlib
import math
import statistics
import tempfile
import unittest
from pathlib import Path

import continuous_quality_report as q
import flywheel_metrics as fm
from promotion_gate import canonical
from resume_expired_cycle import read, sha

FIXTURE = Path('/home/ubuntu/jope-dime/parallel/runs/registry-integration-20260910/health-v1/qualification-status/latest.json')
NEW_KEYS = ('best_initial_arm', 'gp_improves_on_seed', 'best_final_excluding_shared_tail')
CUT = dict(index=0, training_sha256='a' * 64, validation_sha256='b' * 64, preprocessing_sha256='c' * 64)
SCALE = 2.0
# (origin, initial_seed_slot, cut-normalized rmse | None, failure). Teachers occupy slots >= 64 as in the
# deployed populations (64 emissions + retained tail at candidates[64:74]); GP offspring carry no slot.
ARM_OBSERVATIONS = {
    'neural': [('neural_policy', 0, 0.5, None), ('neural_rejection_fallback', 1, 0.4, None),
               ('neural_policy', 2, None, 'timeout'), ('incumbent_cut_teacher', 64, 0.3, None),
               ('retained_cut_teacher', 65, 0.45, None), ('gp_evolution', None, 0.35, None),
               ('cycle_champion', None, 0.32, None)],
    'generic': [('generic_comparison', 0, 0.55, None), ('generic_comparison', 1, 0.6, None),
                ('incumbent_cut_teacher', 64, 0.3, None), ('gp_evolution', None, 0.5, None)],
    'shuffled': [('neural_policy', 0, 0.7, None), ('incumbent_cut_teacher', 64, 0.3, None),
                 ('gp_evolution', None, 0.33, None)],
}
TAIL_ONLY = [('neural_policy', 0, None, 'timeout'), ('incumbent_cut_teacher', 64, 0.3, None)]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b'\n')


def observation(prepared_sha, dataset, origin, slot, rmse, failure):
    program = f'(pipe X {dataset} {origin} {slot} {rmse})'
    return dict(schema='parallel.gp_observation.v1', dataset=dataset, family='fam-' + dataset,
                partition='meta_validation', cut=0, prepared_sha256=prepared_sha,
                training_sha256=CUT['training_sha256'], validation_sha256=CUT['validation_sha256'],
                origin=origin, initial_seed_slot=slot, program_sha256=hashlib.sha256(program.encode()).hexdigest(),
                evaluator_sha256='e' * 64, vocabulary_sha256='v' * 64, target_scale=SCALE, failure=failure,
                metrics=None if failure else dict(rmse=rmse, units='cut_normalized_target'))


def build_fixture(root):
    """One generation, three arms on ds1 plus a tail-only neural search on ds2."""
    generation = root / 'gen-0001'
    jobs = []
    for dataset, arms in (('ds1', ARM_OBSERVATIONS), ('ds2', {'neural': TAIL_ONLY})):
        prepared = dict(dataset=dataset, family='fam-' + dataset, meta_partition='meta_validation',
                        task='regression', cuts=[CUT])
        write(root / 'data' / dataset / 'prepared.json', prepared)
        prepared_sha = sha(root / 'data' / dataset / 'prepared.json')
        for arm, spec in arms.items():
            output = f'gen-0001/{arm}/{dataset}/cut-0'
            rows = [observation(prepared_sha, dataset, *entry) for entry in spec]
            write(root / output / 'observations.json', rows)
            best = min(r['metrics']['rmse'] * SCALE for r in rows if r['failure'] is None)
            write(root / output / 'search.json', dict(rng_seed=7, seeds_sha256='s' * 64, best_validation_rmse=best,
                  completed_evaluations=len(rows), completed_at=1789000000,
                  observations_sha256=sha(root / output / 'observations.json')))
            jobs.append(dict(output=output, prepared='data/' + dataset, prepared_sha256=prepared_sha, cut=0,
                             rng_seed=7, seeds=f'gen-0001/{arm}/{dataset}/seeds.json', seeds_sha256='s' * 64))
    write(generation / 'jobs.json', dict(jobs=jobs))
    return generation, jobs


def projection(row):
    return {k: v for k, v in row.items() if k not in NEW_KEYS}


# Rows produced by search_row BEFORE the additive keys existed, on build_fixture (canonical bytes must match).
GOLDEN = {
    'neural': {
        'arm': 'neural', 'baseline_available': False, 'baseline_rmse': None, 'baseline_rule': 'train_mean_on_validation',
        'baseline_status': 'original_baseline_evidence_unavailable', 'baseline_units': 'original_target',
        'beats_cut_baseline': None, 'beats_unit_one': None, 'best_evaluated_pure_pool': None, 'best_final_gp': 0.6,
        'best_final_origin': 'incumbent_cut_teacher', 'best_initial_neural_seed': 1.0, 'best_later_gp': 0.64,
        'best_retained_teacher': 0.6, 'completed_at': 1789000000, 'completed_evaluations': 7, 'cut': 0, 'dataset': 'ds1',
        'direct_measurement_scope': 'current search observations; unevaluated pure proposals remain missing',
        'evaluator_sha256': 'e' * 64, 'family': 'fam-ds1', 'metric': 'rmse', 'metrics_units': 'original_target',
        'observation_metric_units': 'cut_normalized_target',
        'observation_sha256': 'ba64c69ca0bf450875a59732e99adc6e043b9235b4cb222797fd59d41b198a79',
        'output': 'gen-0001/neural/ds1/cut-0', 'partition': 'meta_validation', 'prediction_contract': None,
        'prepared_sha256': '91437830e1a4d0ee2c5eb403d0f3d52e285366aef686b70d9ebaf80b7a5ed6b5',
        'pure_predicted_best': {'rmse': None, 'status': 'protocol_unavailable'}, 'ratio_status': 'unavailable',
        'relative_rmse': None, 'rmse_skill': None, 'rng_seed': 7, 'search_best_units': 'original_target',
        'search_sha256': 'e9e0f116bab3e9cc4a7d0e1fd25db207b12a9eee927d0ad78f94ba246338c6b4', 'squared_error_skill': None,
        'task': 'regression', 'training_sha256': 'a' * 64, 'unit_vs_baseline_disagrees': None, 'units': 'original_target',
        'validation_sha256': 'b' * 64, 'vocabulary_sha256': 'v' * 64,
    },
    'generic': {
        'arm': 'generic', 'baseline_available': False, 'baseline_rmse': None, 'baseline_rule': 'train_mean_on_validation',
        'baseline_status': 'original_baseline_evidence_unavailable', 'baseline_units': 'original_target',
        'beats_cut_baseline': None, 'beats_unit_one': None, 'best_evaluated_pure_pool': None, 'best_final_gp': 0.6,
        'best_final_origin': 'incumbent_cut_teacher', 'best_initial_neural_seed': None, 'best_later_gp': 1.0,
        'best_retained_teacher': 0.6, 'completed_at': 1789000000, 'completed_evaluations': 4, 'cut': 0, 'dataset': 'ds1',
        'direct_measurement_scope': 'current search observations; unevaluated pure proposals remain missing',
        'evaluator_sha256': 'e' * 64, 'family': 'fam-ds1', 'metric': 'rmse', 'metrics_units': 'original_target',
        'observation_metric_units': 'cut_normalized_target',
        'observation_sha256': 'd2a40300d4ea0ab50b24ad173f5db8f0a4db48b18f4529ad0caaae61966aa40c',
        'output': 'gen-0001/generic/ds1/cut-0', 'partition': 'meta_validation', 'prediction_contract': None,
        'prepared_sha256': '91437830e1a4d0ee2c5eb403d0f3d52e285366aef686b70d9ebaf80b7a5ed6b5',
        'pure_predicted_best': {'rmse': None, 'status': 'protocol_unavailable'}, 'ratio_status': 'unavailable',
        'relative_rmse': None, 'rmse_skill': None, 'rng_seed': 7, 'search_best_units': 'original_target',
        'search_sha256': '2862d96b706a2f4f4487a12f3f1a2e494f3f9d3d1e9b377f6a5e438f93d141a1', 'squared_error_skill': None,
        'task': 'regression', 'training_sha256': 'a' * 64, 'unit_vs_baseline_disagrees': None, 'units': 'original_target',
        'validation_sha256': 'b' * 64, 'vocabulary_sha256': 'v' * 64,
    },
}


def quality_rows(n_cuts=16, generation=None, partition='meta_validation', legacy_generic=False):
    """Three matched arms per cut: neural seeds win 12, lose 3, tie 1 against generic; finals all tie."""
    rows = []
    for i in range(n_cuts):
        dataset, cut = 'ds%02d' % (i // 2), i % 2
        generic = 1.0 + 0.01 * i
        neural = generic * (0.9 if i < 12 else 1.05 if i < 15 else 1.0)
        shuffled = generic * 1.02
        final = 0.5 + 0.001 * i
        for arm, seed in (('neural', neural), ('generic', generic), ('shuffled', shuffled)):
            row = dict(arm=arm, dataset=dataset, family='fam-' + dataset, partition=partition, cut=cut, rng_seed=11,
                       output=f'gen/{arm}/{dataset}/cut-{cut}', metric='rmse', units='original_target',
                       best_initial_neural_seed=None if arm == 'generic' else seed, best_initial_arm=seed,
                       best_final_gp=final, best_retained_teacher=final, best_final_origin='incumbent_cut_teacher')
            if legacy_generic and arm == 'generic':
                del row['best_initial_arm']
            if generation is not None:
                row['generation'] = generation
            rows.append(row)
    return rows


class ClusterBootstrapTests(unittest.TestCase):
    def test_three_cluster_fixture_has_known_mean_and_deterministic_interval(self):
        clusters = {'a': [1.0, 2.0, 3.0], 'b': [4.0], 'c': [5.0, 6.0]}
        ci = fm.cluster_bootstrap_ci(clusters)
        self.assertAlmostEqual(ci['mean'], statistics.fmean([2.0, 4.0, 5.5]), places=12)
        self.assertEqual((ci['n_clusters'], ci['n_values'], ci['resamples'], ci['seed'], ci['alpha']), (3, 6, 2000, 20260912, 0.10))
        self.assertLessEqual(2.0, ci['ci_low'])
        self.assertLessEqual(ci['ci_low'], ci['mean'])
        self.assertLessEqual(ci['mean'], ci['ci_high'])
        self.assertLessEqual(ci['ci_high'], 5.5)
        self.assertEqual(ci, fm.cluster_bootstrap_ci(clusters))
        self.assertEqual(ci, fm.cluster_bootstrap_ci({'c': clusters['c'], 'a': clusters['a'], 'b': clusters['b']}))

    def test_single_cluster_interval_collapses_and_bad_input_fails_closed(self):
        ci = fm.cluster_bootstrap_ci({'only': [0.25, 0.75]})
        self.assertEqual((ci['mean'], ci['ci_low'], ci['ci_high'], ci['n_clusters']), (0.5, 0.5, 0.5, 1))
        with self.assertRaisesRegex(RuntimeError, 'at least one cluster'):
            fm.cluster_bootstrap_ci({})
        with self.assertRaisesRegex(RuntimeError, 'finite'):
            fm.cluster_bootstrap_ci({'a': [1.0, float('nan')]})
        with self.assertRaisesRegex(RuntimeError, 'finite'):
            fm.cluster_bootstrap_ci({'a': []})


class MatchedSeedContrastTests(unittest.TestCase):
    def test_all_ties_at_final_level(self):
        result = fm.matched_seed_contrast(quality_rows(), 'meta_validation', level='final')
        self.assertEqual((result['n_cuts'], result['wins'], result['ties'], result['losses']), (16, 0, 16, 0))
        self.assertEqual(result['mean_log_ratio'], 0.0)
        self.assertEqual(result['ci90'], [0.0, 0.0])
        self.assertEqual(result['tail_caveat'], fm.TAIL_CAVEAT)
        self.assertEqual(len(result['per_cut']), 16)
        self.assertEqual((result['vs_shuffled']['ties'], result['vs_shuffled']['wins']), (16, 0))
        self.assertFalse(result['final_tests_opened'])

    def test_seed_only_wins_and_shuffled_control(self):
        result = fm.matched_seed_contrast(quality_rows(), 'meta_validation')
        self.assertEqual((result['n_cuts'], result['n_pairs'], result['wins'], result['ties'], result['losses']), (16, 16, 12, 1, 3))
        expected = (12 * math.log(0.9) + 3 * math.log(1.05)) / 16
        self.assertAlmostEqual(result['mean_log_ratio'], expected, places=12)
        low, high = result['ci90']
        self.assertLessEqual(low, result['mean_log_ratio'])
        self.assertLessEqual(result['mean_log_ratio'], high)
        self.assertLess(high, 0.0)
        self.assertIsNone(result['tail_caveat'])
        self.assertEqual(result['cluster'], '(dataset, cut)')
        shuffled = result['vs_shuffled']
        self.assertEqual((shuffled['control'], shuffled['wins'], shuffled['ties'], shuffled['losses']), ('shuffled', 13, 0, 3))
        self.assertNotIn('vs_shuffled', shuffled)
        cut = next(c for c in result['per_cut'] if c['dataset'] == 'ds00' and c['cut'] == 0)
        self.assertAlmostEqual(cut['mean_log_ratio'], math.log(0.9), places=12)
        self.assertEqual(cut['wins'], 1)

    def test_generations_pool_into_clusters_and_partition_filter(self):
        rows = quality_rows(generation=1) + quality_rows(generation=2)
        result = fm.matched_seed_contrast(rows, 'meta_validation')
        self.assertEqual((result['n_cuts'], result['n_pairs'], result['wins']), (16, 32, 24))
        self.assertEqual(fm.matched_seed_contrast(rows, 'meta_train')['n_cuts'], 0)
        with self.assertRaisesRegex(RuntimeError, 'duplicate comparison arm'):
            fm.matched_seed_contrast(quality_rows() + quality_rows(), 'meta_validation')
        with self.assertRaisesRegex(RuntimeError, 'unknown contrast level'):
            fm.matched_seed_contrast(rows, 'meta_validation', level='direct')

    def test_legacy_rows_without_best_initial_arm_are_not_imputed(self):
        result = fm.matched_seed_contrast(quality_rows(legacy_generic=True), 'meta_validation')
        self.assertEqual((result['n_cuts'], result['missing_values']), (0, 16))
        self.assertIsNone(result['mean_log_ratio'])
        self.assertIn('no matched', result['reason'])
        self.assertEqual(result['vs_shuffled']['wins'], 13)


class GpImprovesOnSeedTests(unittest.TestCase):
    def test_counts_only_neural_rows_with_both_values(self):
        rows = [dict(arm='neural', best_initial_neural_seed=1.0, best_final_gp=0.8),
                dict(arm='neural', best_initial_neural_seed=1.0, best_final_gp=1.0),
                dict(arm='neural', best_initial_neural_seed=None, best_final_gp=0.5),
                dict(arm='generic', best_initial_arm=1.0, best_final_gp=0.1)]
        result = fm.gp_improves_on_seed(rows)
        self.assertEqual((result['n'], result['improved'], result['rate']), (2, 1, 0.5))
        self.assertAlmostEqual(result['mean_relative_gain'], 0.1, places=12)
        generic = fm.gp_improves_on_seed(rows, arm='generic')
        self.assertEqual((generic['n'], generic['improved'], generic['seed_key']), (1, 1, 'best_initial_arm'))
        empty = fm.gp_improves_on_seed([])
        self.assertEqual((empty['n'], empty['rate'], empty['mean_relative_gain']), (0, None, None))


class SharedTailTests(unittest.TestCase):
    def test_tail_origins_are_excluded_and_failures_skipped(self):
        def row(origin, rmse, failure=None):
            return dict(origin=origin, failure=failure, program_sha256=origin,
                        metrics=None if failure else dict(rmse=rmse, units='original_target'))
        observations = [row('incumbent_cut_teacher', 0.3), row('retained_cut_teacher', 0.31), row('cycle_champion', 0.32),
                        row('neural_policy', 0.1, failure='timeout'), row('gp_evolution', 0.35),
                        row('neural_policy', 0.5), row('generic_comparison', 0.6)]
        result = fm.best_final_excluding_shared_tail(observations)
        self.assertEqual((result['best_rmse'], result['origin'], result['program_sha256']), (0.35, 'gp_evolution', 'gp_evolution'))
        self.assertTrue(result['tail_contaminated'])
        self.assertEqual(result['excluded_origins'], ['cycle_champion', 'incumbent_cut_teacher', 'retained_cut_teacher'])
        self.assertEqual(result['origin_histogram']['neural_policy'], 1)
        none = fm.best_final_excluding_shared_tail(observations[:3])
        self.assertEqual((none['best_rmse'], none['origin']), (None, None))
        self.assertTrue(none['tail_contaminated'])
        self.assertIn('no finite', none['reason'])


class RegressionSignatureTests(unittest.TestCase):
    def setUp(self):
        if not FIXTURE.is_file():
            self.skipTest('local qualification-status fixture absent: ' + str(FIXTURE))
        self.status = read(FIXTURE)

    def test_real_quality_table_shows_the_collapse(self):
        signature = fm.regression_signature(self.status['quality'])
        self.assertEqual(len(signature['generations']), len(self.status['quality']))
        self.assertEqual(signature['generations'][:3], [0, 1, 2])
        self.assertEqual(signature['neural_exclusive_wins'][:3], [9, 10, 11])
        self.assertLess(statistics.fmean(signature['neural_exclusive_wins'][-10:]), 2.0)
        self.assertLessEqual(max(signature['neural_exclusive_wins'][-10:]), 3)
        self.assertEqual(signature['direct_seed_beats_teacher'][:3], [2, 7, 6])
        self.assertEqual(set(signature['direct_seed_beats_teacher'][-30:]), {0})
        self.assertEqual(signature['partition'], 'meta_train')
        validation = signature['by_partition']['meta_validation']
        self.assertEqual(len(validation['neural_exclusive_wins']), len(signature['generations']))
        self.assertEqual(validation['neural_exclusive_wins'][:3],
                         [row['partitions']['meta_validation']['neural_exclusive_wins'] for row in self.status['quality'][:3]])
        as_dict = fm.regression_signature({row['generation']: row for row in self.status['quality']})
        self.assertEqual(as_dict, signature)

    def test_full_quality_reports_are_summarized_from_rows(self):
        rows = quality_rows(n_cuts=2, partition='meta_train')
        neural = next(r for r in rows if r['arm'] == 'neural' and r['dataset'] == 'ds00')
        neural['best_final_gp'] = 0.4        # exclusive win against min(generic, shuffled) = 0.5
        neural['best_initial_neural_seed'] = 0.45   # direct seed beats the retained teacher (0.5)
        report = dict(schema='parallel.continuous_quality.v1', final_tests_opened=False, independent_evaluation=False, rows=rows)
        signature = fm.regression_signature({7: report})
        self.assertEqual((signature['generations'], signature['neural_exclusive_wins'], signature['direct_seed_beats_teacher']),
                         ([7], [1], [1]))
        self.assertEqual(signature['by_partition']['meta_train']['neural_best_ties'], [1])
        with self.assertRaisesRegex((RuntimeError, ValueError), 'incomplete matched triplet'):
            fm.regression_signature({7: dict(report, rows=rows[:-1])})


class PanelNullTests(unittest.TestCase):
    def test_hand_built_receipts_order_by_step_within_binding(self):
        def receipt(step, timestamp, executable, nll, bt):
            return dict(checkpoint=f'ckpt-{step}', evaluated_step=step, timestamp=timestamp, sequence_nll=nll, ranking_bt=bt,
                        ranking_listwise=2.0, runtime_huber=0.05,
                        binding=dict(panel_sha256='p', corpus_sha256='c', executable_sha256=executable))
        receipts = [receipt(9000, 30, 'x', 1.1, 0.55), receipt(3000, 10, 'x', 1.0, 0.5), receipt(6000, 40, 'x', 1.2, 0.6),
                    receipt(3000, 5, 'y', 2.0, 0.8), receipt(6000, 6, 'y', 2.2, 0.8)]
        null = fm.panel_null(receipts)
        self.assertEqual((null['n_receipts'], null['n_segments']), (5, 2))
        nll = null['metrics']['sequence_nll']
        self.assertEqual(nll['n_pairs'], 3)
        # x: 1.0 -> 1.2 -> 1.1 (step order, not timestamp order); y: 2.0 -> 2.2
        self.assertAlmostEqual(nll['max'], 0.2, places=12)
        self.assertAlmostEqual(nll['p50'], 0.1, places=12)
        self.assertEqual(nll['extreme']['previous_checkpoint'], 'ckpt-3000')
        self.assertEqual(nll['extreme']['current_checkpoint'], 'ckpt-6000')
        self.assertEqual(null['guards']['g_nll'], max(0.005, 2 * nll['p90']))
        self.assertEqual(null['guards']['g_bt'], max(0.01, 2 * null['metrics']['ranking_bt']['p90']))
        self.assertFalse(null['guards']['pairs_sufficient'])
        empty = fm.panel_null([])
        self.assertEqual((empty['guards']['g_nll'], empty['guards']['g_bt']), (None, None))
        self.assertIn('no consecutive', empty['guards']['reason'])
        with self.assertRaisesRegex(RuntimeError, 'nonfinite panel metric'):
            fm.panel_null([dict(receipts[0], ranking_bt=None)])

    def test_real_fixture_reproduces_regression_pair_and_guards(self):
        if not FIXTURE.is_file():
            self.skipTest('local qualification-status fixture absent: ' + str(FIXTURE))
        status = read(FIXTURE)
        monitoring = status['monitoring']
        rows = fm.monitoring_rows(monitoring)
        self.assertEqual(len(rows), 133)
        by_cycle = {}
        for row in rows:
            for cycle in ('cycle-0000/', 'cycle-000000/', 'cycle-000073/', 'cycle-000107/'):
                if cycle in row['checkpoint']:
                    by_cycle.setdefault(cycle, set()).add(round(row['ranking_bt'], 4))
        self.assertEqual(by_cycle['cycle-0000/'], {0.5337})       # frozen baseline (current binding)
        self.assertEqual(by_cycle['cycle-000107/'], {0.5421})     # latest (current binding)
        self.assertEqual(by_cycle['cycle-000000/'], {0.5708})     # regressed run start (archived binding)
        self.assertEqual(by_cycle['cycle-000073/'], {0.6796})     # regressed run end (both bindings)
        current = [row for row in rows if row['binding'] == monitoring['binding']]
        self.assertEqual(len(current), len(monitoring['history']))
        null = fm.panel_null(current)
        self.assertEqual((null['n_receipts'], null['n_segments']), (49, 1))
        for name in fm.PANEL_METRICS:
            self.assertEqual(null['metrics'][name]['n_pairs'], 48)
        self.assertAlmostEqual(null['metrics']['sequence_nll']['p90'], 0.0028906602986982655, places=15)
        self.assertAlmostEqual(null['metrics']['ranking_bt']['p90'], 0.012310723132958484, places=15)
        self.assertAlmostEqual(null['guards']['g_nll'], 0.005781320597396531, places=15)
        self.assertAlmostEqual(null['guards']['g_bt'], 0.024621446265916967, places=15)
        self.assertTrue(null['guards']['pairs_sufficient'])
        self.assertEqual(null['guards']['floors'], dict(g_nll=0.005, g_bt=0.01))
        pooled = fm.panel_null(rows)
        self.assertEqual((pooled['n_receipts'], pooled['n_segments'], pooled['metrics']['ranking_bt']['n_pairs']), (133, 2, 131))
        self.assertAlmostEqual(pooled['guards']['g_nll'], 0.008367943278040718, places=15)
        self.assertAlmostEqual(pooled['guards']['g_bt'], 0.032801057162668575, places=15)


class DiversityTests(unittest.TestCase):
    def test_unique_emissions_tokens_and_fresh_programs(self):
        genome = '(pipe X (rff_sin append=true gamma=1.0 n=32 seed=29) (cheb append=false degree=6))'
        populations = [dict(candidates=[dict(genome=genome, origin='neural_policy'), dict(genome=genome, origin='neural_policy'),
                                        dict(genome='(pipe X)', origin='neural_policy'), dict(genome='(g)', origin='generic_bootstrap')]),
                       dict(candidates=[dict(genome='(pipe X (cheb degree=2))', origin='neural_policy')])]
        snapshots = [dict(fresh_records={'1': [10, 11], '2': [11, 12]}), dict(unique_fresh_programs=7), dict(fresh_records={})]
        result = fm.diversity(populations, snapshots)
        self.assertEqual(result['unique_neural_per_context'], [2, 1])
        self.assertEqual(result['unique_neural_per_context_median'], 1.5)
        self.assertEqual(fm.tokens('(pipe X)'), ['(', 'pipe', 'X', ')'])
        lengths = [len(fm.tokens(genome))] * 2 + [4, len(fm.tokens('(pipe X (cheb degree=2))'))]
        self.assertAlmostEqual(result['mean_token_length'], statistics.fmean(lengths), places=12)
        self.assertEqual((result['unique_fresh_programs'], result['unique_fresh_programs_median']), ([3, 7, 0], 3))
        with self.assertRaisesRegex(RuntimeError, 'fresh_records'):
            fm.diversity([], [dict()])


class SearchRowExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.generation, self.jobs = build_fixture(self.root)
        self.rows = {Path(j['output']).parts[1] + '/' + Path(j['output']).parts[2]: q.search_row(self.root, self.generation, j)
                     for j in self.jobs}

    def tearDown(self):
        self.tmp.cleanup()

    def test_existing_keys_and_values_are_byte_identical(self):
        for arm in ('neural', 'generic'):
            projected = projection(self.rows[arm + '/ds1'])
            self.assertEqual(canonical(projected), canonical(GOLDEN[arm]), arm + ' row drifted: ' + canonical(projected).decode())

    def test_additive_keys(self):
        neural, generic, shuffled, tail = (self.rows[k] for k in ('neural/ds1', 'generic/ds1', 'shuffled/ds1', 'neural/ds2'))
        self.assertEqual((neural['best_initial_neural_seed'], neural['best_initial_arm']), (1.0, 0.8))
        self.assertIs(neural['gp_improves_on_seed'], True)
        self.assertEqual(neural['best_final_excluding_shared_tail'], dict(rmse=0.7, origin='gp_evolution', tail_contaminated=True))
        self.assertEqual((generic['best_initial_neural_seed'], generic['best_initial_arm'], generic['gp_improves_on_seed']), (None, 1.1, None))
        self.assertEqual(generic['best_final_excluding_shared_tail'], dict(rmse=1.0, origin='gp_evolution', tail_contaminated=True))
        self.assertEqual((shuffled['best_initial_arm'], shuffled['gp_improves_on_seed']), (1.4, True))
        self.assertEqual(shuffled['best_final_excluding_shared_tail']['rmse'], 0.66)
        self.assertEqual((tail['best_initial_arm'], tail['gp_improves_on_seed'], tail['best_final_excluding_shared_tail']), (None, None, None))
        self.assertEqual(tail['best_final_gp'], 0.6)

    def test_rows_feed_the_contrast_and_producers(self):
        rows = [row for key, row in self.rows.items() if key.endswith('/ds1')]
        seed_only = fm.matched_seed_contrast(rows, 'meta_validation')
        self.assertEqual((seed_only['n_cuts'], seed_only['wins'], seed_only['vs_shuffled']['wins']), (1, 1, 1))
        self.assertAlmostEqual(seed_only['mean_log_ratio'], math.log(0.8 / 1.1), places=12)
        final = fm.matched_seed_contrast(rows, 'meta_validation', level='final')
        self.assertEqual((final['ties'], final['wins'], final['losses']), (1, 0, 0))
        improves = fm.gp_improves_on_seed(list(self.rows.values()))
        self.assertEqual((improves['n'], improves['improved']), (1, 1))
        self.assertAlmostEqual(improves['mean_relative_gain'], 0.4, places=12)


if __name__ == '__main__':
    unittest.main()
