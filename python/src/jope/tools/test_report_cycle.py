import bisect
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import itertools
import unittest

from report_cycle import (ASSEMBLY_PROTOCOL, PURE_PROTOCOL, SELECTION_ID, best_prefix, compact_digest,
                          digest, family_ratios, search_timing, verify_generation, wall_prefix)


class MeasuredReportTests(unittest.TestCase):
    def test_source_families_receive_equal_weight(self):
        views = [dict(dataset='a', cut=i, family='large', matched_evaluations=dict(best_rmse=dict(neural=2., generic=1.))) for i in range(10)]
        views.append(dict(dataset='b', cut=0, family='small', matched_evaluations=dict(best_rmse=dict(neural=.5, generic=1.))))
        result = family_ratios(views, 'matched_evaluations', 'neural', 'generic')
        self.assertEqual(result['families'], 2)
        self.assertAlmostEqual(result['family_balanced_rmse_ratio'], 1.)

    def test_semantic_dedup_and_failed_candidates_match_gp_budget(self):
        rows = [
            dict(evaluation_index=0, runtime_seconds=1., failure=None, metrics=dict(rmse=2., semantic_sha256='same-predictions'), target_scale=2.),
            dict(evaluation_index=1, runtime_seconds=2., failure=None, metrics=dict(rmse=1., semantic_sha256='same-predictions'), target_scale=2.),
            dict(evaluation_index=2, runtime_seconds=1., failure='timeout', metrics=None),
        ]
        best, cumulative, failures = best_prefix(rows)
        self.assertEqual(best, [None, 4., 4., 4.])
        self.assertEqual(cumulative, [1., 3., 4.])
        self.assertEqual(failures, 1)

    def exact_clock_fixture(self):
        rows = [dict(completed_search_seconds=t, search_clock='parallel.gp_search_clock.v1', search_clock_accuracy='exact_active_elapsed') for t in [2., 6., 8.]]
        summary = dict(wall_seconds=9., resumed=False, search_clock=dict(schema='parallel.gp_search_clock.v1', exact_wall_prefix_available=True, accuracy='exact_active_elapsed', resume_offset_seconds=0., resume_unobserved_allowance_seconds=0., legacy_observations_without_clock=0))
        return summary, rows

    def test_exact_prefix_uses_measured_overhead_not_candidate_sum(self):
        summary, rows = self.exact_clock_fixture()
        timing = search_timing(summary, rows, [1., 3., 4.])
        result = wall_prefix(dict(timing=timing, cumulative=[1., 3., 4.], best=[None, 9., 2., 1.]), 5.)
        self.assertTrue(result['exact_prefix_available'])
        self.assertEqual(result['possibly_completed_evaluations'], 1)
        self.assertEqual(result['guaranteed_completed_evaluations'], 1)
        self.assertEqual(result['optimistic_best_rmse'], 9.)
        self.assertEqual(result['guaranteed_best_rmse'], 9.)

    def test_unqualified_or_resumed_clocks_cannot_be_called_exact(self):
        for mode in ['resumed_summary', 'nonexact_row', 'missing_clock', 'offset', 'unknown_summary']:
            with self.subTest(mode=mode):
                summary, rows = self.exact_clock_fixture()
                if mode == 'resumed_summary':
                    summary['resumed'] = True
                elif mode == 'nonexact_row':
                    rows[-1]['search_clock_accuracy'] = 'conservative_resume_offset'
                elif mode == 'missing_clock':
                    rows[0].pop('completed_search_seconds')
                elif mode == 'offset':
                    summary['search_clock']['resume_offset_seconds'] = 1.
                else:
                    summary.pop('search_clock')
                timing = search_timing(summary, rows, [1., 3., 4.])
                self.assertFalse(timing['exact'])
                self.assertTrue(timing['fallback_reasons'])
                bounds = wall_prefix(dict(timing=timing, cumulative=[1., 3., 4.], best=[None, 9., 2., 1.]), 6.)
                self.assertEqual(bounds['optimistic_best_rmse'], 1.)
                self.assertEqual(bounds['guaranteed_best_rmse'], 9.)

    def test_corrupt_completion_timestamps_fail_closed(self):
        for values in [[2., 1., 8.], [2., float('nan'), 8.], [2., 6., 10.], [2., 2.5, 8.]]:
            with self.subTest(values=values):
                summary, rows = self.exact_clock_fixture()
                for row, value in zip(rows, values):
                    row['completed_search_seconds'] = value
                with self.assertRaises(ValueError):
                    search_timing(summary, rows, [1., 3., 4.])

    def test_legacy_ledger_preserves_conservative_bounds(self):
        timing = search_timing(dict(wall_seconds=7.), [{}, {}, {}], [1., 3., 4.])
        bounds = wall_prefix(dict(timing=timing, cumulative=[1., 3., 4.], best=[None, 9., 2., 1.]), 5.)
        self.assertFalse(bounds['exact_prefix_available'])
        self.assertEqual(bounds['optimistic_best_rmse'], 1.)
        self.assertEqual(bounds['guaranteed_best_rmse'], 9.)

    def test_wall_bounds_cover_every_possible_overhead_allocation(self):
        cumulative, best = [1., 3., 4.], [None, 9., 2., 1.]
        slack, budget = 3., 5.
        possible = bisect.bisect_right(cumulative, budget)
        guaranteed = bisect.bisect_right(cumulative, budget-slack)
        for overhead in itertools.product(range(4), repeat=4):
            if sum(overhead) != slack:
                continue
            completion = [value+sum(overhead[:i+1]) for i, value in enumerate(cumulative)]
            actual = best[bisect.bisect_right(completion, budget)]
            self.assertGreaterEqual(actual, best[possible])
            self.assertLessEqual(actual, best[guaranteed])


class ProposalProtocolReportTests(unittest.TestCase):
    def fixture(self, mutation=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        directory = root/'cycle/neural/dataset/cut-0'
        directory.mkdir(parents=True)
        checkpoint, context, vocabulary = ('a'*64, 'b'*64, 'c'*64)
        snapshot = dict(schema='parallel.champion_snapshot.v1', vocabulary_sha256=vocabulary,
                        request=dict(checkpoint_sha256=checkpoint, vocabulary_sha256=vocabulary))
        snapshot_path = root/'cycle/champion-snapshot.json'
        snapshot_path.write_text(json.dumps(snapshot))
        snapshot_sha = digest(snapshot_path.read_bytes())
        pure_programs = [f'pure-{i}' for i in range(64)]
        gp_programs = ['refill'] + pure_programs[1:]
        def ranking(programs):
            rows = [dict(program=p, program_sha256=digest(p.encode()), predicted_quality=float(100-i),
                         predicted_runtime_seconds=1., reward_heads=[float(100-i)]*4) for i,p in enumerate(programs)]
            return dict(schema='parallel.proposal_quality_ranking.v1', selection_id=SELECTION_ID,
                        bindings=dict(checkpoint_sha256=checkpoint, conditioning_context_sha256=context,
                                      vocabulary_sha256=vocabulary), training_context_only=True,
                        heldout_labels_read=False, gp_scores_read=False, ranked=rows,
                        pool_size=64, chosen_count=64,
                        pool_sha256=compact_digest(sorted(r['program_sha256'] for r in rows)))
        tail = [dict(genome=pure_programs[0] if i == 0 else f'teacher-{i}',
                     origin='incumbent_cut_teacher' if i == 0 else 'retained_cut_teacher' if i < 10 else 'random_exploration')
                for i in range(16)]
        seeds = dict(schema='parallel.seed_population.v1', cut=0, prepared_sha256='d'*64,
                     training_sha256='e'*64, context_sha256=context, checkpoint_sha256=checkpoint,
                     candidates=[dict(genome=p, origin='neural_overlap_fallback' if p == 'refill' else 'neural_policy')
                                 for p in gp_programs]+tail)
        first = dict(dataset='dataset', cut=0, prepared_sha256=seeds['prepared_sha256'],
                     training_sha256=seeds['training_sha256'], vocabulary_sha256=vocabulary)
        job = dict(cut=0, prepared_sha256=seeds['prepared_sha256'])
        pure = dict(schema='parallel.pure_neural_ranking.v1', protocol=PURE_PROTOCOL,
                    ranking=ranking(pure_programs), teacher_population_consulted=False,
                    sampling_seed=42, fallback_seed=77, attempted=128, valid=128,
                    within_neural_duplicates=5, unique_neural=64)
        best = dict(schema='parallel.pure_best_predicted_program.v1', protocol=PURE_PROTOCOL,
                    dataset='dataset', cut=0, prepared_sha256=seeds['prepared_sha256'],
                    training_context_sha256=context, conditioning_context_sha256=context,
                    checkpoint_sha256=checkpoint, vocabulary_sha256=vocabulary,
                    context_mode='matching_training_cut', selection_id=SELECTION_ID,
                    prediction=copy.deepcopy(pure['ranking']['ranked'][0]),
                    training_context_only_end_to_end=True, teacher_population_consulted=False,
                    gp_scores_used_for_prediction=False, heldout_targets_used_for_prediction=False,
                    final_tests_opened=False)
        overlap = dict(pure_rank=0, program_sha256=digest(pure_programs[0].encode()),
                       retained_slot=64, retained_origin='incumbent_cut_teacher')
        refill = dict(program_sha256=digest(b'refill'), generator_seed=123, origin='neural_overlap_fallback')
        assembly = dict(schema='parallel.gp_population_assembly.v1', protocol=ASSEMBLY_PROTOCOL,
                        pure_programs_sha256=compact_digest(pure_programs), retained_tail_sha256=compact_digest(tail),
                        overlaps=[overlap], refills=[refill], champion_snapshot_sha256=snapshot_sha,
                        checkpoint_sha256=checkpoint, conditioning_context_sha256=context)
        gp = dict(schema='parallel.gp_population_ranking.v1', protocol=ASSEMBLY_PROTOCOL,
                  ranking=ranking(gp_programs), champion_snapshot_sha256=snapshot_sha,
                  teacher_population_consulted=True, training_context_only_end_to_end=False)
        gen = dict(schema='parallel.proposal_generation.v2', pure_prediction_protocol=PURE_PROTOCOL,
                   gp_assembly_protocol=ASSEMBLY_PROTOCOL, checkpoint_sha256=checkpoint,
                   conditioning_context_sha256=context, context_mode='matching_training_cut', seed=42, selection_id=SELECTION_ID,
                   champion_snapshot_sha256=snapshot_sha, attempted=128, valid=128,
                   within_neural_duplicates=5, pure_unique_neural=64, unique_neural=63,
                   filled_population=80, teacher_overlaps=[copy.deepcopy(overlap)],
                   overlap_refills=[copy.deepcopy(refill)], final_tests_opened=False)
        state = dict(pure=pure, best=best, gp=gp, assembly=assembly, gen=gen, seeds=seeds)
        if mutation:
            mutation(state)
        def save(name, value):
            path = directory/name
            path.write_text(json.dumps(value))
            return digest(path.read_bytes())
        pure_sha = save('pure-ranking.json', pure)
        for artifact in (best, assembly, gp):
            artifact['pure_ranking_sha256'] = pure_sha
        assembly_sha = save('gp-population-assembly.json', assembly)
        gp['population_assembly_sha256'] = assembly_sha
        gen.update(pure_ranking_sha256=pure_sha, pure_best_predicted_sha256=save('pure-best-predicted.json', best),
                   gp_population_assembly_sha256=assembly_sha, gp_ranking_sha256=save('gp-ranking.json', gp),
                   seeds_sha256=save('seeds.json', seeds))
        job['seeds_sha256'] = gen['seeds_sha256']
        reads = []
        def read(path):
            reads.append(path)
            return json.loads(path.read_text())
        return directory, gen, seeds, job, first, read, reads

    def test_v2_teacher_winner_overlap_preserves_pure_prediction_and_separate_counters(self):
        *args, reads = self.fixture()
        counts, scope = verify_generation(*args)
        self.assertEqual(scope, 'pure_prediction_then_adaptive_gp_assembly')
        self.assertEqual(counts, dict(attempted=128, valid=128, unique_neural=63, pure_unique_neural=64,
                                     within_neural_duplicates=5, teacher_overlaps=1, overlap_refills=1))
        self.assertNotIn('duplicates', counts)
        self.assertEqual({p.name for p in reads}, {'pure-ranking.json', 'pure-best-predicted.json',
                         'gp-ranking.json', 'gp-population-assembly.json', 'champion-snapshot.json'})
        self.assertFalse(any('test.csv' in str(p) for p in reads))

    def test_v2_no_overlap_preserves_entire_ranked_neural_pool(self):
        def no_overlap(v):
            programs = [r['program'] for r in v['pure']['ranking']['ranked']]
            v['seeds']['candidates'][64]['genome'] = 'different-teacher'
            v['seeds']['candidates'][:64] = [dict(genome=p, origin='neural_policy') for p in programs]
            v['gp']['ranking'] = copy.deepcopy(v['pure']['ranking'])
            v['assembly'].update(overlaps=[], refills=[], retained_tail_sha256=compact_digest(v['seeds']['candidates'][64:]))
            v['gen'].update(teacher_overlaps=[], overlap_refills=[], unique_neural=64)
        *args, _ = self.fixture(no_overlap)
        counts, _ = verify_generation(*args)
        self.assertEqual(counts['unique_neural'], 64)
        self.assertEqual(counts['teacher_overlaps'], 0)
        self.assertEqual(counts['overlap_refills'], 0)

    def test_v2_shuffled_prediction_binds_donor_context_separately_from_training_cut(self):
        def shuffled(v):
            v['seeds']['context_sha256'] = 'f'*64
            v['best']['training_context_sha256'] = 'f'*64
            v['best']['context_mode'] = v['gen']['context_mode'] = 'shuffled_between_source_families'
        *args, _ = self.fixture(shuffled)
        counts, scope = verify_generation(*args)
        self.assertEqual(counts['teacher_overlaps'], 1)
        self.assertEqual(scope, 'pure_prediction_then_adaptive_gp_assembly')

    def test_v2_artifact_byte_corruption_is_rejected(self):
        for name in ('pure-ranking.json', 'pure-best-predicted.json', 'gp-ranking.json', 'gp-population-assembly.json'):
            with self.subTest(name=name):
                *args, _ = self.fixture()
                path = args[0]/name
                path.write_text(path.read_text()+' ')
                with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                    verify_generation(*args)

    def test_consistently_rehashed_wrong_scope_identity_or_overlap_still_rejected(self):
        mutations = {
            'gp_claims_pure_input': lambda v: v['gp'].update(training_context_only_end_to_end=True),
            'pure_consults_teachers': lambda v: v['pure'].update(teacher_population_consulted=True),
            'wrong_conditioning': lambda v: v['gp']['ranking']['bindings'].update(conditioning_context_sha256='f'*64),
            'best_uses_gp_winner': lambda v: v['best'].update(prediction=v['gp']['ranking']['ranked'][0]),
            'false_overlap_slot': lambda v: v['assembly']['overlaps'][0].update(retained_slot=65),
            'unmarked_overlap_fallback': lambda v: v['seeds']['candidates'][0].update(origin='neural_policy'),
            'mixed_duplicate_counter': lambda v: v['gen'].update(duplicates=6),
            'unsorted_pure_scores': lambda v: v['pure']['ranking']['ranked'].reverse(),
            'changed_tail': lambda v: v['seeds']['candidates'][-1].update(genome='changed'),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                *args, _ = self.fixture(mutation)
                with self.assertRaises(ValueError):
                    verify_generation(*args)

    def test_legacy_numeric_counts_preserved_without_inventing_collision_frequency(self):
        *args, _ = self.fixture()
        directory, _, seeds, job, first, _ = args
        seeds['candidates'][0]['origin'] = 'neural_policy'
        gen = dict(schema='parallel.proposal_generation.v1', seeds_sha256=job['seeds_sha256'],
                   attempted=256, valid=256, duplicates=189, unique_neural=64)
        def no_new_artifacts(path):
            self.fail(f'Legacy inputs unexpectedly require a v2 artifact: {path}')
        counts, scope = verify_generation(directory, gen, seeds, job, first, no_new_artifacts)
        previous = Counter()
        for field in ('attempted', 'valid', 'duplicates', 'unique_neural'):
            previous[field] += gen[field]
        self.assertEqual(dict(previous), counts)
        self.assertEqual(scope, 'legacy_filtered_pool')
        self.assertNotIn('teacher_overlaps', counts)
        self.assertNotIn('within_neural_duplicates', counts)


if __name__ == '__main__':
    unittest.main()
