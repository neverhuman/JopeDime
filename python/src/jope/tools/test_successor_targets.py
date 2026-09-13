"""successor_targets tests on a tempdir corpus shaped like replay_policy.groups() expects."""
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import successor_targets as st
from resume_expired_cycle import atomic, sha

EVALUATOR = '9' * 64
CHECKPOINT = 'c' * 64


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def context(prepared, cut=0, family=None, partition='meta_train', n=20, base=1.0, step=0.1):
    """Programs 0..12 at base + step*i (0 = champion, 1..12 = candidates); 13.. interleave into the same range (controls)."""
    return dict(prepared=prepared, cut=cut, family=family or 'family-' + prepared, partition=partition,
                programs=[('program-%s-%d' % (prepared, i), base + step * i if i <= 12 else base + step * (i - 12) + 0.03) for i in range(n)])


def median_of(ctx):
    import statistics
    return statistics.median(rmse for _, rmse in ctx['programs'])


def corpus(root, contexts):
    path = root / 'corpus'
    path.mkdir(parents=True)
    records, labels, index, number = bytearray(), [], dict(record_bytes=532, vocabulary_sha256='v', context_index_sha256='c', contexts=[], groups=[]), 0
    for g, ctx in enumerate(contexts):
        ids = []
        for name, rmse in ctx['programs']:
            record = bytearray(532)
            struct.pack_into('<H', record, 0, number + 1)
            struct.pack_into('<H', record, 512, 1)
            struct.pack_into('<fff', record, 520, .4, .5, 1)
            records.extend(record)
            labels.append(dict(record=number, context=g, program=name, rmse_cut_normalized=rmse))
            ids.append(number)
            number += 1
        index['contexts'].append(dict(prepared_sha256=ctx['prepared'], cut=ctx['cut'], family=ctx['family'], partition=ctx['partition']))
        index['groups'].append(dict(context=g, family=ctx['family'], partition=ctx['partition'], records=ids))
    (path / 'records.bin').write_bytes(records)
    (path / 'labels.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in labels))
    index.update(records=number, records_sha256=sha(path / 'records.bin'))
    atomic(path / 'index.json', index)
    atomic(path / 'provenance.json', dict(index_sha256=sha(path / 'index.json'), labels_sha256=sha(path / 'labels.jsonl')))
    return path


def row(ctx, role, program, r0, child, arm='neural', replicate=0, replicate_pair=None, status='measured', evaluator=EVALUATOR, partition='meta_train'):
    key = '%s:%d' % (ctx['prepared'], ctx['cut'])
    identity = digest(('%s|%s|%s|%d' % (key, role, program, replicate)).encode())
    return dict(schema='parallel.seed_value.v1', generation=73, dataset='ds-' + ctx['prepared'], family=ctx['family'], prepared_sha256=ctx['prepared'],
                cut=ctx['cut'], context_key=key, episode_id=identity, identity_sha256=identity, protocol='onestep_forced_mutation_v2', role=role,
                arm=arm, replicate=replicate, seed_origin='generic_comparison' if arm == 'generic' else 'neural_policy',
                seed_program_sha256=digest(program.encode()), seed_genome=program, producing_checkpoint_sha256=CHECKPOINT, evaluator_sha256=evaluator,
                driver_sha256=None, r0=r0, child=dict(program_sha256='x' * 64, rmse=child, failure=None if child is not None else 'candidate_or_view_timeout',
                                                       genetic_operator='forced_seed_mutation'),
                flip=int(child is not None and child < r0), S_rmse=(min(r0, child) if child is not None else r0), replicate_pair=replicate_pair,
                status=status, completed_at=1789237312, partition=partition)


def scenario(ctx, improvement=0.25, controls=4, panel=3, spread=0.001, candidates=12):
    """12 candidates (odd: child better by `improvement`, even: worse by 0.2), controls, and a replicate panel."""
    rows = []
    programs = dict(ctx['programs'])
    for i in range(1, candidates + 1):
        name = 'program-%s-%d' % (ctx['prepared'], i)
        r0 = programs[name]
        child = r0 - improvement if i % 2 else r0 + 0.2
        pair = dict(child_rmse_rep0=child, child_rmse_rep1=child + spread) if i <= panel else None
        rows.append(row(ctx, 'candidate', name, r0, child, replicate_pair=pair))
        if i <= panel:
            rows.append(row(ctx, 'replicate_panel', name, r0, child + spread, replicate=1, replicate_pair=pair))
    for i in range(candidates + 1, candidates + 1 + controls):
        name = 'program-%s-%d' % (ctx['prepared'], i)
        rows.append(row(ctx, 'control_generic', name, programs[name], programs[name] + 0.3, arm='generic'))
    return rows


class TargetsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/dev/shm')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.c0 = context('p0')
        self.c1 = context('p1')
        self.validation = context('p2', partition='meta_validation')
        self.corpus = corpus(self.root, [self.c0, self.c1, self.validation])

    def export(self, rows, name='seed-values-73.json', **extra):
        document = dict(schema='parallel.seed_value.v1', generation=73, protocol='onestep_forced_mutation_v2', evaluator_sha256=EVALUATOR,
                        rows=rows, counts={}, final_tests_opened=False)
        document.update(extra)
        path = self.root / name
        path.write_bytes(encoded(document))
        return path

    def test_value_conversion_ranks_and_percentile(self):
        self.assertAlmostEqual(st.value(1.95, 1.95), 0.5)
        self.assertAlmostEqual(st.value(0.0, 1.95), 1.0)
        self.assertGreater(st.value(1.0, 1.95), st.value(1.1, 1.95))
        with self.assertRaises(ValueError):
            st.value(float('nan'), 1.95)
        with self.assertRaises(ValueError):
            st.value(1.0, 0.0)
        self.assertEqual(st.ranks([3.0, 1.0, 2.0, 2.0]), [4.0, 1.0, 2.5, 2.5])
        self.assertAlmostEqual(st.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(st.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(st.spearman([1, 1, 1], [1, 2, 3]))
        self.assertEqual(st.percentile([5, 1, 3, 4, 2], 0.75), 4)
        self.assertIsNone(st.percentile([], 0.75))
        fit = st.ols3([1.5, 2.0, 3.5, 4.0, 6.0], [0, 1, 2, 3, 4], [1, 0, 1, 0, 2])   # y = 1 + x1 + 0.5 x2 exactly
        self.assertAlmostEqual(fit[0], 1.0, places=6)
        self.assertAlmostEqual(fit[1], 1.0, places=6)
        self.assertAlmostEqual(fit[2], 0.5, places=6)
        self.assertIsNone(st.ols3([1, 2, 3, 4], [1, 2, 3, 4], [2, 4, 6, 8]))

    def test_eps_floor_and_replicate_spread(self):
        median = 1.95
        tiny = st.eps_for(scenario(self.c0, spread=0.001), median)
        self.assertEqual(tiny, st.EPS_FLOOR)
        wide = st.eps_for(scenario(self.c0, spread=0.5), median)
        spreads = sorted(abs(st.value(c, median) - st.value(c + 0.5, median)) for c in (0.85, 1.4, 1.05))
        self.assertAlmostEqual(wide, spreads[2])
        self.assertIsNone(st.eps_for(scenario(self.c0, panel=0), median))

    def test_block_schema_groups_and_info_test(self):
        block = st.build(self.corpus, self.export(scenario(self.c0)), evaluator_sha256=EVALUATOR)
        self.assertEqual(block['schema'], 'parallel.seed_values.v1')
        self.assertEqual(block['corpus_sha256'], sha(self.corpus / 'index.json'))
        self.assertEqual(block['evaluator_sha256'], EVALUATOR)
        self.assertEqual((block['protocol'], block['generation'], block['tie_rule'], block['per_context_cap']),
                         ('onestep_forced_mutation_v2', 73, 'abs(dS_value) <= eps_c', 8))
        self.assertIs(block['final_tests_opened'], False)
        self.assertEqual(block['seed_values_export_sha256'], sha(self.root / 'seed-values-73.json'))
        for key in ('groups', 'values', 'floors', 'roles', 'arms', 'counts', 'info_test'):
            self.assertIn(key, block)
        self.assertEqual(block['counts']['contexts'], 1)
        self.assertEqual(block['counts']['groups'], 2)      # 16 eligible seeds -> two full groups of 7 + champion, 2 leftovers
        self.assertEqual(block['counts']['incomplete_groups'], 1)
        self.assertEqual(block['counts']['ties_excluded'], 0)
        self.assertEqual(block['counts']['refused_contexts'], 0)
        self.assertEqual(block['counts']['label_mismatch_rows'], 0)
        median = median_of(self.c0)
        self.assertAlmostEqual(block['contexts']['p0:0']['median_rmse'], median)
        for index, group in enumerate(block['groups']):
            self.assertEqual(len(group), 8)
            self.assertEqual(len(set(group)), 8)
            self.assertEqual(group[0], 0)                   # champion record (program-p0-0, rmse 1.0) first
            self.assertEqual(block['roles'][str(group[0])], 'champion')
            values = [block['values'][str(r)] for r in group[1:]]
            self.assertEqual(values, sorted(values, reverse=True))
            self.assertEqual(block['floors'][str(index)], st.EPS_FLOOR)
        self.assertAlmostEqual(block['values']['0'], st.value(1.0, median))
        first_seed = block['groups'][0][1]
        self.assertAlmostEqual(block['values'][str(first_seed)], st.value(0.85, median))   # program-p0-1: r0 1.1, child 0.85
        self.assertEqual(block['roles'][str(first_seed)], 'candidate')
        self.assertEqual(block['arms'][str(first_seed)], 'neural')
        self.assertIn('generic', block['arms'].values())
        info = block['info_test']
        self.assertEqual((info['p_flip_flips'], info['p_flip_pairs']), (5, 66))
        self.assertAlmostEqual(info['p_flip'], 5 / 66)
        self.assertFalse(info['passed_a'])
        self.assertFalse(info['passed_b'])
        self.assertFalse(info['passed'])
        self.assertIsNone(info['a_own_ci'])
        self.assertEqual(info['a_own']['reason'], 'next_outcome_unavailable')
        contrast = info['conditional_contrast']
        self.assertEqual(contrast['control_role'], 'control_generic')
        self.assertEqual(contrast['contexts'], 1)
        self.assertGreater(contrast['cells'], 0)
        self.assertIsNotNone(contrast['overall']['mean'])
        self.assertIsNone(contrast['overall']['ci90'])   # one cluster: no bootstrap interval
        # A_ctx (plan D1): value(S) - median over the context's control_generic seeds of value(S), per record and per arm.
        a_ctx = block['a_ctx']
        self.assertEqual(a_ctx['control_role'], 'control_generic')
        self.assertEqual((a_ctx['contexts_with_control'], a_ctx['contexts_without_control']), (1, []))
        programs = dict(self.c0['programs'])
        controls = sorted(st.value(programs['program-p0-%d' % i], median) for i in range(13, 17))   # S = r0: children are worse
        reference = (controls[1] + controls[2]) / 2
        self.assertAlmostEqual(block['contexts']['p0:0']['a_ctx_control_median'], reference)
        self.assertAlmostEqual(a_ctx['by_arm']['generic'], 0.0)
        neural = sorted(st.value(min(programs['program-p0-%d' % i], programs['program-p0-%d' % i] - 0.25 if i % 2 else programs['program-p0-%d' % i] + 0.2), median) - reference
                        for i in range(1, 13))
        self.assertAlmostEqual(a_ctx['by_arm']['neural'], (neural[5] + neural[6]) / 2)
        self.assertIsNone(a_ctx['by_arm']['shuffled'])
        self.assertEqual(a_ctx['n_by_arm'], dict(neural=12, generic=4, shuffled=0))
        self.assertAlmostEqual(a_ctx['per_record'][str(first_seed)], st.value(0.85, median) - reference)
        self.assertEqual(len(a_ctx['per_record']), 16)   # every joined seed, consumed or not (two leftovers never enter a group)
        self.assertTrue({str(r) for g in block['groups'] for r in g[1:]} <= set(a_ctx['per_record']))
        uncontrolled = st.build(self.corpus, self.export(scenario(self.c0, controls=0)))
        self.assertEqual(uncontrolled['a_ctx']['contexts_without_control'], ['p0:0'])
        self.assertEqual(uncontrolled['a_ctx']['by_arm'], dict(neural=None, generic=None, shuffled=None))
        self.assertNotIn('a_ctx_control_median', uncontrolled['contexts']['p0:0'])
        drifted = scenario(self.c0)
        drifted[0]['r0'] = drifted[0]['r0'] + 1e-9          # export r0 no longer bit-equal to the corpus label
        block = st.build(self.corpus, self.export(drifted))
        self.assertEqual(block['counts']['label_mismatch_rows'], 1)
        self.assertEqual(block['label_mismatches'][0]['seed_program_sha256'], drifted[0]['seed_program_sha256'])

    def test_ties_are_excluded_and_thin_groups_dropped(self):
        rows = scenario(self.c0)
        twin = next(r for r in rows if r['seed_genome'] == 'program-p0-4')
        source = next(r for r in rows if r['seed_genome'] == 'program-p0-2')
        twin['child']['rmse'], twin['S_rmse'] = source['S_rmse'], source['S_rmse']   # exact tie in S; own label (r0) untouched
        block = st.build(self.corpus, self.export(rows))
        self.assertEqual(block['counts']['ties_excluded'], 1)
        self.assertEqual(block['counts']['groups'], 2)
        flat = [r for g in block['groups'] for r in g[1:]]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertLessEqual(sum(1 for r in flat if block['values'][str(r)] == st.value(source['S_rmse'], median_of(self.c0))), 1)
        collapsed = scenario(self.c0)
        for r in collapsed:
            if r['role'] != 'replicate_panel':
                r['r0'], r['child']['rmse'], r['S_rmse'] = 1.5, 1.5, 1.5
        with self.assertRaisesRegex(ValueError, 'no context produced a consumable group'):
            st.build(self.corpus, self.export(collapsed))

    def test_cap_of_eight_groups_per_context(self):
        big = context('p0', n=80)
        self.corpus = corpus(self.root / 'big', [big])
        rows = scenario(big, controls=60)
        block = st.build(self.corpus, self.export(rows))
        self.assertEqual(block['counts']['groups'], 8)
        self.assertEqual(len(block['floors']), 8)

    def test_refusal_paths(self):
        rows = scenario(self.c0)
        with self.assertRaisesRegex(ValueError, 'corpus mismatch'):
            st.build(self.corpus, self.export(rows), corpus_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'evaluator mismatch'):
            st.build(self.corpus, self.export(rows), evaluator_sha256='1' * 64)
        with self.assertRaisesRegex(ValueError, 'replicate panel unavailable'):
            st.build(self.corpus, self.export(scenario(self.c0, panel=0)))
        with self.assertRaisesRegex(ValueError, 'non-meta_train corpus group'):
            st.build(self.corpus, self.export(scenario(self.validation)))
        sealed = scenario(self.c0)
        sealed[0]['partition'] = 'meta_validation'
        with self.assertRaisesRegex(ValueError, 'non-meta_train seed-value rows'):
            st.build(self.corpus, self.export(sealed))
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            st.build(self.corpus, self.export(rows, schema='parallel.other.v1'))
        with self.assertRaisesRegex(ValueError, 'exactly one evaluator'):
            st.build(self.corpus, self.export(rows + [row(self.c1, 'candidate', 'program-p1-1', 1.1, 1.0, evaluator='2' * 64)]))
        with self.assertRaisesRegex(ValueError, 'control role'):
            st.build(self.corpus, self.export(rows), control_role='candidate')
        tied = scenario(self.c1)
        for r in tied:
            if r['role'] != 'replicate_panel':
                r['r0'], r['child']['rmse'], r['S_rmse'] = 1.5, 1.6, 1.5
        block = st.build(self.corpus, self.export(rows + tied))
        self.assertEqual(block['counts']['contexts'], 1)
        self.assertEqual(block['counts']['refused_contexts'], 1)
        self.assertEqual(block['refused'][0]['context_key'], 'p1:0')
        self.assertTrue(block['refused'][0]['reason'].startswith('tie_fraction 1.000'))
        self.assertEqual(block['tie_fraction']['p1:0']['tie_fraction'], 1.0)
        unknown = scenario(self.c0) + [row(context('p9'), 'candidate', 'program-p9-1', 1.1, 1.0, replicate_pair=dict(child_rmse_rep0=1.0, child_rmse_rep1=1.0)),
                                       row(context('p9'), 'replicate_panel', 'program-p9-1', 1.1, 1.0, replicate=1, replicate_pair=dict(child_rmse_rep0=1.0, child_rmse_rep1=1.0))]
        block = st.build(self.corpus, self.export(unknown))
        self.assertEqual(block['refused'], [dict(context_key='p9:0', reason='context_not_in_corpus')])
        unjoined = scenario(self.c0) + [row(self.c0, 'candidate', 'program-not-in-corpus', 1.15, 1.0)]
        block = st.build(self.corpus, self.export(unjoined))
        self.assertEqual(block['counts']['unjoined_rows'], 1)

    def test_information_tests_pass_when_flips_and_a_own_signal_exist(self):
        rows = scenario(self.c0, improvement=0.55) + scenario(self.c1, improvement=0.55)
        outcomes = {}
        for r in rows:
            if r['role'] == 'candidate':
                median = median_of(self.c0)
                a_own = st.value(r['S_rmse'], median) - st.value(r['r0'], median)
                outcomes[(r['context_key'], r['seed_program_sha256'])] = 10.0 * a_own - 0.05 * r['r0']
        block = st.build(self.corpus, self.export(rows), next_outcome=outcomes)
        info = block['info_test']
        self.assertEqual((info['p_flip_flips'], info['p_flip_pairs']), (24, 132))
        self.assertTrue(info['passed_a'])
        self.assertEqual(info['a_own']['n_rows'], 24)
        self.assertEqual(info['a_own']['n_contexts'], 2)
        self.assertGreater(info['a_own']['coefficient'], 0)
        self.assertGreater(info['a_own_ci']['low'], 0)
        self.assertTrue(info['passed_b'])
        self.assertTrue(info['passed'])
        self.assertEqual(block['counts']['contexts'], 2)
        self.assertEqual(block['bootstrap'], dict(resamples=2000, seed=20260912, alpha=0.10, clusters='contexts'))
        again = st.build(self.corpus, self.export(rows), next_outcome=outcomes)
        self.assertEqual(again['info_test']['a_own_ci'], info['a_own_ci'])
        listed = self.root / 'next-outcome.json'
        listed.write_bytes(encoded(dict(schema='parallel.successor_next_outcome.v1',
                                        rows=[dict(context_key=k[0], seed_program_sha256=k[1], outcome=v) for k, v in outcomes.items()])))
        from_file = st.build(self.corpus, self.export(rows), next_outcome=listed)
        self.assertEqual(from_file['info_test']['a_own'], info['a_own'])

    def test_cli_writes_immutable_block_or_refusal(self):
        export = self.export(scenario(self.c0))
        output = self.root / 'policy/seed-values.json'
        argv = ['successor_targets', '--corpus', str(self.corpus), '--export', str(export), '--output', str(output), '--evaluator-sha256', EVALUATOR]
        sys.argv = argv
        st.main()
        first = output.read_bytes()
        sys.argv = argv
        st.main()
        self.assertEqual(output.read_bytes(), first)
        self.assertEqual(json.loads(first)['schema'], 'parallel.seed_values.v1')
        refusal = self.root / 'policy/refusal.json'
        sys.argv = argv[:-2] + ['--evaluator-sha256', '1' * 64, '--refusal-output', str(refusal), '--output', str(self.root / 'policy/other.json')]
        with self.assertRaises(SystemExit) as stop:
            st.main()
        self.assertEqual(stop.exception.code, 1)
        body = json.loads(refusal.read_bytes())
        self.assertEqual(body['schema'], 'parallel.seed_values_refusal.v1')
        self.assertTrue(body['refused'])
        self.assertIn('evaluator mismatch', body['reason'])
        self.assertFalse((self.root / 'policy/other.json').exists())


if __name__ == '__main__':
    unittest.main()
