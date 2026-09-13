import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import flywheel_series as series
import flywheel_verdict as verdict
import successor_targets as st
from resume_expired_cycle import atomic, sha
from test_flywheel_series import baseline_fixture, monitor_row, per_cut
from test_successor_targets import EVALUATOR, context, corpus, encoded, median_of, scenario

TOOLS = Path(__file__).resolve().parent
REAL_PREREGISTRATION = TOOLS.parent / 'recovery/eight-a100-20260912/PREREGISTRATION.md'
# Same dotted paths as the fenced block of PREREGISTRATION.md (parallel.flywheel_preregistration.v1).
BLOCK = dict(schema='parallel.flywheel_preregistration.v1', final_tests_opened=False,
             statistics=dict(n_clusters=16, window_generations=10, resamples=2000, seed=20260912, alpha=0.10,
                             secondary=dict(win_rate_delta=0.13, n_pairs=160),
                             mde_n16=dict(multiplier_of_sigma_cut=0.879, power=0.8)),
             falsifiers=dict(F1=dict(secondary=dict(win_rate_delta=0.13)),
                             F2=dict(consecutive_rejects=3, distinct_checkpoints=True),
                             F3=dict(unique_fresh_programs_fraction_of_baseline_median=0.5, unique_emissions_per_context_min=32,
                                     unique_emissions_per_context_of=64, mean_tokens_over_baseline=1.3, consecutive_generations=3),
                             F4=dict(minimum_generations=5, p_flip_min=0.10, a_own_ci_alpha=0.10),
                             F8=dict(accuracy_gain_min=0.02, by_block=10, hold_below_baseline=0.01, hold_consecutive_blocks=3)),
             verdict=dict(generations=[5, 10, 20], brake_fires_for_harmful=2))


def document(block):
    return '# Pre-registration\n\nThresholds:\n\n```json\n%s\n```\n\nAfter.\n' % json.dumps(block, indent=2, sort_keys=True)


def generation_row(number, offset, wins=120, losses=40, diversity=True, unique=40):
    cuts = per_cut(offset)
    mean = sum(c['mean_log_ratio'] for c in cuts) / len(cuts)
    window = dict(mean_log_ratio=mean, ci90=[mean - 0.01, mean + 0.01], wins=wins, ties=0, losses=losses, n_pairs=wins + losses,
                  n_cuts=len(cuts), per_cut=cuts, generations=list(range(max(0, number - 9), number + 1)))
    row = dict(schema=series.SCHEMA, kind='generation', id='generation-%06d' % number, generation=number, timestamp=1789240000 + number,
               matched=dict(val=dict(trailing10=window)), final_tests_opened=False)
    if diversity:
        row['diversity'] = dict(unique_neural_per_context_median=unique, mean_token_length=12.0, unique_fresh_programs=[500],
                                unique_fresh_programs_median=500)
    return row


def critic_row(step, accuracy, groups=True, seed_value=None, by_context=True, comparable=True):
    row = monitor_row(step, 1.4, comparable=comparable)
    row['panel']['nontie_pairwise_accuracy'] = accuracy
    if groups:
        row['panel']['nontie_pairwise_accuracy_by_group'] = [accuracy + d for d in (-0.005, 0.0, 0.005, 0.0)]
    if seed_value is not None:
        # parallel.seed_value_monitoring.v1 joined by flywheel_series.monitor_rows to the checkpoint's row.
        row['panel']['seed_value_monitoring'] = dict(nontie_pairwise_accuracy_vs_S=seed_value, ndcg10_vs_S=0.8, n_groups=48, contexts=16,
                                                     by_context=[seed_value + d for d in (-0.01, 0.0, 0.01, 0.005) * 4] if by_context else None,
                                                     receipt_sha256='r' * 64)
    return row


def episodes_row(number, p_flip=0.2, ci=(0.05, 0.3), mismatches=0):
    return dict(schema=series.SCHEMA, kind='episodes', id='seed-values-%d' % number, generation=number,
                successor=dict(p_flip=p_flip, a_own_ci=(list(ci) if isinstance(ci, tuple) else ci), row0_mismatches=mismatches), final_tests_opened=False)


def block_row(index, consumed=None):
    seed_values = dict(consumed='yes', reason='seed_values block abc') if consumed else \
        dict(consumed='no', reason='seed_values_absent: no export')
    return dict(schema=series.SCHEMA, kind='block', id='cycle-%06d' % index, block_index=index, seed_values=seed_values,
                seed_value_bt=0.3 if consumed else None, seed_value_groups_seen=12 if consumed else None, final_tests_opened=False)


def scenario_root(directory, rows, block=BLOCK, baseline=None):
    root = Path(directory)
    atomic(root / 'baseline.json', baseline or baseline_fixture(panel=dict(sequence_nll=1.4, ranking_bt=0.5, ranking_listwise=2.2,
                                                                          runtime_huber=0.07, nontie_pairwise_accuracy=0.60)))
    (root / 'PREREGISTRATION.md').write_text(document(block))
    series.append(root / 'flywheel-series.jsonl', rows)
    return root


def smarter_rows(seed_value=0.58):
    rows = [generation_row(n, -0.05) for n in (8, 9, 10)]
    rows += [critic_row(3000 * k, 0.63, seed_value=seed_value) for k in range(1, 4)]
    rows += [episodes_row(n) for n in range(1, 6)] + [block_row(1, consumed=True)]
    return rows


class Preregistration(unittest.TestCase):
    def test_thresholds_come_from_the_block_and_digest_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = scenario_root(directory, smarter_rows())
            registration = verdict.preregistration(root / 'PREREGISTRATION.md')
            self.assertEqual(registration['block'], BLOCK)
            self.assertEqual(registration['sha256'], series.digest(BLOCK))
            self.assertEqual(len(registration['block_bytes_sha256']), 64)
            self.assertNotEqual(registration['block_bytes_sha256'], registration['sha256'])
            self.assertEqual(verdict.preregistration(root / 'PREREGISTRATION.md', registration['block_bytes_sha256'])['matched_by'], 'block_text')
            self.assertEqual(verdict.preregistration(root / 'PREREGISTRATION.md', registration['sha256'])['matched_by'], 'canonical_json')
            with self.assertRaises(RuntimeError):
                verdict.preregistration(root / 'PREREGISTRATION.md', 'f' * 64)
            changed = copy.deepcopy(BLOCK)
            changed['falsifiers']['F4']['p_flip_min'] = 0.5
            (root / 'PREREGISTRATION.md').write_text(document(changed))
            with self.assertRaises(RuntimeError):
                verdict.preregistration(root / 'PREREGISTRATION.md', registration['sha256'])
            rows = series.series_rows(root / 'flywheel-series.jsonl')
            result = verdict.evaluate(rows, root / 'baseline.json', verdict.preregistration(root / 'PREREGISTRATION.md'), 10)
            self.assertEqual(result['falsifiers']['F4']['status'], 'fail')
            self.assertEqual(result['falsifiers']['F4']['p_flip_min'], 0.5)
            (root / 'PREREGISTRATION.md').write_text('# no block\n')
            with self.assertRaises(RuntimeError):
                verdict.preregistration(root / 'PREREGISTRATION.md')

    def test_missing_threshold_makes_that_falsifier_not_evaluable(self):
        block = copy.deepcopy(BLOCK)
        del block['falsifiers']['F4']
        del block['verdict']['brake_fires_for_harmful']
        with tempfile.TemporaryDirectory() as directory:
            root = scenario_root(directory, smarter_rows(), block)
            rows = series.series_rows(root / 'flywheel-series.jsonl')
            result = verdict.evaluate(rows, root / 'baseline.json', verdict.preregistration(root / 'PREREGISTRATION.md'), 10)
            self.assertEqual(result['not_evaluable']['F4'], 'preregistration block lacks falsifiers.F4.p_flip_min | F4.p_flip_min')
            self.assertNotIn('F4', result['evaluable'])
            self.assertEqual(result['outcome'], 'smarter_confirmed')
            self.assertEqual(result['outcome_inputs']['final_generation'], 20)
            self.assertIn('not evaluable', result['outcome_inputs']['fires_clause'])
            with self.assertRaises(RuntimeError):
                verdict.evaluate(rows, root / 'baseline.json', verdict.preregistration(root / 'PREREGISTRATION.md'), 7)
            rejects = [generation_row(10, -0.05)] + [monitor_row(3000 * k, 1.5) for k in range(1, 7)]
            series.append(root / 'fires.jsonl', rejects)
            result = verdict.evaluate(series.series_rows(root / 'fires.jsonl'), root / 'baseline.json',
                                      verdict.preregistration(root / 'PREREGISTRATION.md'), 10)
            self.assertEqual(result['falsifiers']['F2']['fires'], 2)
            self.assertEqual(result['outcome'], 'inconclusive_underpowered')  # the fires clause needs a numeric key in the block

    @unittest.skipUnless(REAL_PREREGISTRATION.is_file(), 'PREREGISTRATION.md not mirrored locally')
    def test_real_preregistration_block_resolves_every_threshold(self):
        registration = verdict.preregistration(REAL_PREREGISTRATION)
        block = registration['block']
        self.assertEqual(block.get('schema'), verdict.PREREGISTRATION_SCHEMA)
        self.assertEqual(verdict.preregistration(REAL_PREREGISTRATION, registration['sha256'])['matched_by'], 'canonical_json')
        with self.assertRaises(RuntimeError):
            verdict.preregistration(REAL_PREREGISTRATION, 'f' * 64)
        optional = {'final_generation', 'harmful_fires', 'mde_absolute'}
        for name in verdict.PATHS:
            if name not in optional:
                self.assertIsNotNone(verdict.threshold(block, name), name)
        self.assertEqual(verdict.threshold(block, 'f2_consecutive'), 3)
        self.assertEqual(verdict.threshold(block, 'generations'), [5, 10, 20])
        self.assertEqual(verdict.threshold(block, 'harmful_fires'), 2)
        self.assertAlmostEqual(verdict.threshold(block, 'mde_multiplier'), 0.879)
        self.assertIn('two-sample', block['statistics']['bootstrap'])


class Outcomes(unittest.TestCase):
    def run_scenario(self, rows, generation=10, block=BLOCK, since=None, baseline=None):
        with tempfile.TemporaryDirectory() as directory:
            root = scenario_root(directory, rows, block, baseline)
            return verdict.evaluate(series.series_rows(root / 'flywheel-series.jsonl'), root / 'baseline.json',
                                    verdict.preregistration(root / 'PREREGISTRATION.md'), generation, since)

    def test_smarter_confirmed(self):
        result = self.run_scenario(smarter_rows())
        self.assertEqual(result['outcome'], 'smarter_confirmed')
        f1 = result['falsifiers']['F1']
        self.assertEqual(f1['status'], 'pass')
        self.assertEqual(f1['difference']['method'], 'independent_cluster_resampling')   # the pre-registered two-sample interval
        self.assertEqual(f1['paired_diagnostic']['method'], 'paired_cut_difference_cluster_bootstrap')
        self.assertTrue(f1['paired_diagnostic']['available'])
        self.assertIn('diagnostic', f1['paired_diagnostic']['role'])
        self.assertAlmostEqual(f1['difference']['mean'], -0.05)
        self.assertLess(f1['difference']['ci_high'], 0)
        self.assertEqual(f1['n_cuts'], dict(window=16, baseline=16, common=16))
        self.assertAlmostEqual(f1['secondary']['delta'], 0.75 - 1.0)  # baseline fixture already wins every pair
        self.assertFalse(f1['secondary']['passed'])
        f8 = result['falsifiers']['F8']
        self.assertEqual(f8['status'], 'pass')
        self.assertEqual(f8['seed_value_monitoring']['status'], 'pass')
        self.assertGreater(f8['seed_value_monitoring']['ci']['ci_low'], 0)
        self.assertEqual(result['statuses']['F2'], 'silent')
        self.assertEqual(result['statuses']['F3'], 'silent')
        self.assertEqual(result['statuses']['F5'], 'silent')
        self.assertEqual(result['statuses']['F7'], 'pass')
        self.assertEqual(result['statuses']['F4'], 'pass')
        self.assertEqual(result['not_evaluable'], dict(F6=result['falsifiers']['F6']['reason']))
        self.assertEqual(result['generations'], dict(verdict=10, origin=0, origin_source='baseline receipt generation_range.last', run_generation=10,
                                                     run_range=[1, 10], rows_in_range=8, rule=result['generations']['rule']))
        self.assertEqual(result['outcome_inputs']['brake_fires'], 0)
        self.assertFalse(result['outcome_inputs']['f5_fired'])
        self.assertIs(result['final_tests_opened'], False)
        self.assertEqual(result['schema'], verdict.SCHEMA)

    def test_f8_cannot_pass_without_the_seed_value_monitoring_clause(self):
        absent = self.run_scenario(smarter_rows(seed_value=None))
        f8 = absent['falsifiers']['F8']
        self.assertEqual(f8['status'], 'pending')
        self.assertEqual(f8['seed_value_monitoring']['status'], 'pending')
        self.assertIn('no parallel.seed_value_monitoring.v1 receipt', f8['reason'])
        self.assertEqual(absent['outcome'], 'inconclusive_underpowered')
        rows = [generation_row(n, -0.05) for n in (8, 9, 10)] + [critic_row(3000 * k, 0.63, seed_value=0.5) for k in range(1, 11)]
        rows += [episodes_row(n) for n in range(1, 6)]
        failing = self.run_scenario(rows)
        self.assertEqual(failing['falsifiers']['F8']['status'], 'fail')
        self.assertEqual(failing['falsifiers']['F8']['seed_value_monitoring']['status'], 'fail')
        self.assertEqual(failing['outcome'], 'inconclusive_underpowered')
        early = self.run_scenario(rows[:3] + rows[3:6] + rows[13:])   # three blocks: the failing clause is still pending
        self.assertEqual(early['falsifiers']['F8']['status'], 'pending')
        no_contexts = self.run_scenario([generation_row(10, -0.05)] + [critic_row(3000 * k, 0.63, seed_value=0.6, by_context=False) for k in range(1, 4)])
        self.assertEqual(no_contexts['falsifiers']['F8']['status'], 'underpowered')
        self.assertEqual(no_contexts['falsifiers']['F8']['seed_value_monitoring']['status'], 'underpowered')

    def test_no_learning_detected_only_at_the_final_generation_with_f2_f3_silent(self):
        def rows_at(last):
            rows = [generation_row(n, 0.06, wins=40, losses=120) for n in (last - 2, last - 1, last)]
            return rows + [critic_row(3000 * k, 0.605) for k in range(1, 11)] + [episodes_row(n) for n in range(1, 6)]
        result = self.run_scenario(rows_at(20), 20)
        self.assertEqual(result['outcome'], 'no_learning_detected')
        self.assertEqual(result['statuses']['F1'], 'fail')
        self.assertEqual(result['falsifiers']['F8']['reason'], 'accuracy below baseline + delta at or after block 10')
        self.assertTrue(result['outcome_inputs']['at_final_generation'])
        self.assertTrue(result['outcome_inputs']['f2_f3_silent'])
        for generation in (5, 10):
            early = self.run_scenario(rows_at(generation), generation)
            self.assertEqual(early['statuses']['F1'], 'fail')
            self.assertEqual(early['statuses']['F8'], 'fail')
            self.assertEqual(early['outcome'], 'inconclusive_underpowered', generation)
            self.assertFalse(early['outcome_inputs']['at_final_generation'])
        # F2/F3 not evaluable is not "silent": no monitor row is comparable, no generation row carries diversity.
        blind = [generation_row(n, 0.06, wins=40, losses=120, diversity=False) for n in (18, 19, 20)]
        blind += [critic_row(3000 * k, 0.605, comparable=False) for k in range(1, 11)] + [episodes_row(n) for n in range(1, 6)]
        result = self.run_scenario(blind, 20)
        self.assertEqual((result['statuses']['F1'], result['statuses']['F8']), ('fail', 'fail'))
        self.assertEqual(result['statuses']['F2'], 'not_evaluable')
        self.assertEqual(result['statuses']['F3'], 'not_evaluable')
        self.assertEqual(result['outcome'], 'inconclusive_underpowered')
        self.assertFalse(result['outcome_inputs']['f2_f3_silent'])

    def test_harmful_stopped_is_the_brake_firing_twice(self):
        rejects = [monitor_row(3000 * k, 1.5) for k in range(1, 7)]
        result = self.run_scenario([generation_row(10, -0.05)] + rejects)
        self.assertEqual(result['outcome'], 'harmful_stopped')
        self.assertEqual((result['falsifiers']['F2']['fires'], result['outcome_inputs']['brake_fires']), (2, 2))
        # F3 diagnostic rejects feed the same hold: two windows of three alarmed generations.
        starved = [generation_row(n, -0.05, unique=10) for n in range(5, 11)] + [monitor_row(3000, 1.4)]
        result = self.run_scenario(starved)
        self.assertEqual(result['outcome'], 'harmful_stopped')
        self.assertEqual((result['outcome_inputs']['f2_fires'], result['outcome_inputs']['f3_fires']), (0, 2))
        one_each = [generation_row(n, -0.05, unique=10) for n in range(8, 11)] + [monitor_row(3000 * k, 1.5) for k in range(1, 4)]
        result = self.run_scenario(one_each)
        self.assertEqual(result['outcome'], 'harmful_stopped')
        self.assertEqual(result['outcome_inputs']['brake_fires'], 2)
        # F5 and an F8 refutation are flags: they block smarter_confirmed but never extend the pre-registered enum.
        mismatch = smarter_rows() + [episodes_row(6, mismatches=2)]
        result = self.run_scenario(mismatch)
        self.assertEqual(result['statuses']['F5'], 'fired')
        self.assertTrue(result['outcome_inputs']['f5_fired'])
        self.assertEqual(result['outcome'], 'inconclusive_underpowered')
        refuted = [generation_row(10, -0.05)] + [critic_row(3000 * k, 0.58, seed_value=0.58) for k in range(1, 4)]
        result = self.run_scenario(refuted)
        self.assertTrue(result['falsifiers']['F8']['refuted'])
        self.assertTrue(result['outcome_inputs']['f8_refuted'])
        self.assertEqual(result['outcome'], 'inconclusive_underpowered')

    def test_inconclusive_underpowered(self):
        rows = [generation_row(10, 0.0)]
        rows[0]['matched']['val']['trailing10']['per_cut'] = [dict(c, mean_log_ratio=c['mean_log_ratio'] + (0.004 if i % 2 else -0.004))
                                                            for i, c in enumerate(rows[0]['matched']['val']['trailing10']['per_cut'])]
        result = self.run_scenario(rows + [monitor_row(3000, 1.4)])
        self.assertEqual(result['outcome'], 'inconclusive_underpowered')
        f1 = result['falsifiers']['F1']
        self.assertEqual(f1['status'], 'underpowered')
        self.assertEqual(f1['difference']['method'], 'independent_cluster_resampling')
        self.assertLessEqual(f1['difference']['ci_low'], 0)
        self.assertGreaterEqual(f1['difference']['ci_high'], 0)
        self.assertEqual(f1['mde']['multiplier'], 0.879)
        self.assertAlmostEqual(f1['mde']['value'], 0.879 * f1['mde']['sigma_cut'])
        self.assertLess(abs(f1['difference']['mean']), f1['mde']['value'])
        self.assertEqual(result['not_evaluable']['F8'], 'nontie_pairwise_accuracy absent from monitor receipts (pre-B1 monitor identity)')
        self.assertIn('F4', result['not_evaluable'])
        self.assertIn('F7', result['not_evaluable'])
        empty = self.run_scenario([generation_row(10, -0.05)])
        self.assertEqual(empty['outcome'], 'inconclusive_underpowered')
        self.assertEqual(empty['not_evaluable']['F2'], 'no comparable monitored learning checkpoints in the baseline monitor segment')

    def test_paired_diagnostic_is_unavailable_when_cut_sets_differ(self):
        rows = smarter_rows()
        window = rows[2]['matched']['val']['trailing10']
        window['per_cut'] = window['per_cut'][:12] + [dict(c, dataset='extra%d' % i) for i, c in enumerate(window['per_cut'][:3])]
        result = self.run_scenario(rows)
        f1 = result['falsifiers']['F1']
        self.assertEqual(f1['difference']['method'], 'independent_cluster_resampling')
        self.assertFalse(f1['paired_diagnostic']['available'])
        self.assertEqual(f1['n_cuts']['common'], 12)

    def test_verdict_generations_count_from_the_run_origin(self):
        rows = [generation_row(n, -0.05) for n in (73, 74, 75)] + [episodes_row(n) for n in range(71, 76)] + [monitor_row(3000, 1.4)]
        # Origin from the baseline receipt (its window ended at run generation 70) ...
        result = self.run_scenario(rows, 5, baseline=baseline_fixture(last_generation=70))
        self.assertEqual(result['generations']['origin'], 70)
        self.assertEqual(result['generations']['origin_source'], 'baseline receipt generation_range.last')
        self.assertEqual((result['generations']['run_generation'], result['generations']['run_range']), (75, [71, 75]))
        self.assertEqual(result['falsifiers']['F1']['generation'], 75)
        self.assertEqual(result['falsifiers']['F1']['status'], 'pass')
        self.assertEqual((result['falsifiers']['F4']['generations'], result['falsifiers']['F4']['generation']), (5, 75))
        # ... or explicitly from the CLI; the origin wins over the receipt.
        explicit = self.run_scenario(rows, 5, since=70, baseline=baseline_fixture(last_generation=0))
        self.assertEqual(explicit['generations']['origin_source'], 'cli:--since-generation')
        self.assertEqual(explicit['falsifiers']['F1']['status'], 'pass')
        # The run counter alone never satisfies a verdict generation: with origin 0 nothing lies in (0, 5].
        counter = self.run_scenario(rows, 5)
        self.assertEqual(counter['not_evaluable']['F1'], 'no generation row with run generation in (0, 5] (verdict generation 5 from origin 0)')
        self.assertIn('0 generation(s) with a measured P_flip in run generations (0, 5]', counter['not_evaluable']['F4'])
        # Pre-change rows never feed F1/F4: at verdict generation 5 from origin 72 only 73..75 count.
        partial = self.run_scenario(rows, 5, since=72)
        self.assertEqual(partial['generations']['run_range'], [73, 75])
        self.assertIn('3 generation(s) with a measured P_flip', partial['not_evaluable']['F4'])
        unknown = baseline_fixture()
        del unknown['generation_range']
        with self.assertRaisesRegex(RuntimeError, 'verdict origin unknown'):
            self.run_scenario(rows, 5, baseline=unknown)

    def test_f4_reads_the_interval_successor_targets_produces(self):
        # The producer's {low, high, resamples} dict, the series' [low, high] list and a straddling interval.
        dict_ci = [episodes_row(n, ci=dict(low=0.05, high=0.3, resamples=2000)) for n in range(1, 6)]
        result = self.run_scenario([generation_row(10, -0.05)] + dict_ci)
        self.assertEqual(result['falsifiers']['F4']['status'], 'pass')
        self.assertEqual(result['falsifiers']['F4']['a_own_ci'], [0.05, 0.3])
        self.assertEqual(result['falsifiers']['F4']['a_own_ci_source'], dict(low=0.05, high=0.3, resamples=2000))
        straddle = [episodes_row(n, ci=dict(low=-0.05, high=0.3, resamples=2000)) for n in range(1, 6)]
        self.assertEqual(self.run_scenario([generation_row(10, -0.05)] + straddle)['falsifiers']['F4']['reason'], 'P_flip meets floor 0.1; A_own CI straddles 0')
        missing = [episodes_row(n, ci=None) for n in range(1, 6)]
        self.assertEqual(self.run_scenario([generation_row(10, -0.05)] + missing)['falsifiers']['F4']['reason'], 'P_flip meets floor 0.1; A_own CI unavailable')

    def test_f4_through_the_real_producer_chain(self):
        """successor_targets.build -> policy block on disk -> flywheel_series.episode_rows -> flywheel_verdict.f4_information."""
        with tempfile.TemporaryDirectory(dir='/dev/shm') as directory:
            root = Path(directory)
            c0, c1 = context('p0'), context('p1')
            corpus_dir = corpus(root, [c0, c1])
            rows = scenario(c0, improvement=0.55) + scenario(c1, improvement=0.55)
            outcomes = {}
            for r in rows:
                if r['role'] == 'candidate':
                    median = median_of(c0)
                    outcomes[(r['context_key'], r['seed_program_sha256'])] = 10.0 * (st.value(r['S_rmse'], median) - st.value(r['r0'], median)) - 0.05 * r['r0']
            successor = root / 'run/successor'
            (successor / 'seed-values').mkdir(parents=True)
            export = successor / 'seed-values-73.json'
            export.write_bytes(encoded(dict(schema='parallel.seed_value.v1', generation=73, protocol='onestep_forced_mutation_v2',
                                            evaluator_sha256=EVALUATOR, rows=rows, counts={}, final_tests_opened=False)))
            block = st.build(corpus_dir, export, evaluator_sha256=EVALUATOR, next_outcome=outcomes)
            self.assertTrue(block['info_test']['passed'])
            self.assertIsInstance(block['info_test']['a_own_ci'], dict)   # the producer's shape
            atomic(successor / 'seed-values' / (block['corpus_sha256'] + '.json'), block)
            manifest = dict(output='run', bindings={}, cpu_executable='bin/gp', monitor_executable='bin/monitor')
            episode = series.episode_rows(root, manifest, successor)[0]
            self.assertEqual(episode['successor']['a_own_ci'], [block['info_test']['a_own_ci']['low'], block['info_test']['a_own_ci']['high']])
            self.assertGreater(episode['successor']['a_own_ci'][0], 0)
            self.assertEqual(episode['successor']['a_ctx']['generic'], block['a_ctx']['by_arm']['generic'])
            self.assertAlmostEqual(episode['successor']['p_flip'], 24 / 132)
            series_rows = [dict(episode, id='seed-values-%d' % g, generation=g) for g in range(71, 76)]
            f4 = verdict.f4_information(series_rows, BLOCK, 5, origin=70)
            self.assertEqual(f4['status'], 'pass')
            self.assertEqual(f4['reason'], 'P_flip meets floor 0.1; A_own CI excludes 0')
            self.assertEqual(f4['generations'], 5)


class Cli(unittest.TestCase):
    def test_writes_immutable_receipt_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = scenario_root(directory, smarter_rows())
            registration = verdict.preregistration(root / 'PREREGISTRATION.md')
            command = [sys.executable, str(TOOLS / 'flywheel_verdict.py'), '--preregistration', str(root / 'PREREGISTRATION.md'),
                       '--preregistration-sha256', registration['sha256'], '--series', str(root / 'flywheel-series.jsonl'),
                       '--baseline', str(root / 'baseline.json'), '--generation', '10', '--since-generation', '0']
            dry = subprocess.run(command + ['--dry-run'], text=True, capture_output=True)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertEqual(json.loads(dry.stdout)['outcome'], 'smarter_confirmed')
            self.assertEqual(json.loads(dry.stdout)['generations']['origin_source'], 'cli:--since-generation')
            self.assertFalse((root / 'successor/verdicts/generation-10.json').exists())
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            path = root / 'successor/verdicts/generation-10.json'
            written = json.loads(path.read_text())
            self.assertEqual(written['outcome'], 'smarter_confirmed')
            self.assertEqual(written['preregistration']['matched_by'], 'canonical_json')
            self.assertEqual(written['preregistration']['reader'], 'flywheel_baseline.preregistration')
            self.assertEqual(json.loads(result.stdout)['sha256'], series.sha(path))
            self.assertEqual(subprocess.run(command, text=True, capture_output=True).returncode, 0)
            refused = subprocess.run(command[:5] + ['f' * 64] + command[6:], text=True, capture_output=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('differs from the supplied', refused.stderr)
            rows, latest = series.verdict_rows(root / 'successor')
            self.assertEqual(latest['outcome'], 'smarter_confirmed')
            self.assertEqual(rows[0]['verdict_sha256'], series.sha(path))


if __name__ == '__main__':
    unittest.main()
