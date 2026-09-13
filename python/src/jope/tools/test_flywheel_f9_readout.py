import copy
import json
from pathlib import Path
import tempfile
import unittest

import flywheel_baseline
import flywheel_f9_readout as readout
from resume_expired_cycle import atomic, read, sha

TOOLS = Path(__file__).resolve().parent
PREREGISTRATION = TOOLS.parent / 'recovery/eight-a100-20260912/PREREGISTRATION.md'
FAMILIES = ['family%d' % i for i in range(8)]
NEUTRAL, REPAIR, WORSE = (0.0, 0.004), (-0.02, 0.005), (0.02, 0.004)


def padded(prefix):
    """A 64-hex digest starting with the pre-registered prefix."""
    return (prefix + '0' * 64)[:64]


SHARED = dict(protocol_sha256=padded('d8d57069'), parent_state_sha256=padded('d715c47a'), parent_model_sha256=padded('3253c386'),
              parent_optimizer_sha256=padded('da43282d'), corpus_sha256=padded('ddbca1d3'))
CHECKPOINTS = dict(A=padded('aaaa0001'), B=padded('bbbb0002'))
PANEL_BINDINGS = dict(reserved=dict(name='reserved', panel_sha256=padded('961a2d52'), corpus_sha256=padded('c0c0c0c0')),
                      old_frozen=dict(name='old_frozen', panel_sha256=padded('fbff7a04'), corpus_sha256=padded('11e60a43')))
EXECUTABLE = padded('87995977')


def contexts(panel):
    """Reserved: 8 families x 32 contexts, one group each. Old frozen: 226 distinct contexts over 256 draws (30 drawn twice)."""
    if panel == 'reserved':
        return [(FAMILIES[f], f * 32 + c, 0) for f in range(8) for c in range(32)]
    return [(FAMILIES[c % 8], 1000 + c, 0) for c in range(226)] + [(FAMILIES[c % 8], 1000 + c, 1) for c in range(30)]


def base_group(family, context, draw):
    weight = 8.0 + 8.0 * (context % 7)   # batch weight rises with the loss: sum(num)/sum(den) sits ~0.02 above the mean of group ratios
    bt = 0.40 + 0.02 * (context % 7)
    return dict(context_index=context, dataset='dataset-%d' % context, cut='cut-%d' % (context % 3), family=family, partition='meta_validation',
                record_ids=[context * 100 + draw * 8 + k for k in range(8)], bt_numerator=bt * weight, bt_denominator=weight, ranking_bt=bt,
                sequence_nll=1.40 + 0.001 * (context % 11), ranking_listwise=2.2 + 0.001 * (context % 5), runtime_huber=0.07, n_pairs=28)


def shifted(group, index, offset, noise):
    """Arm-B copy: ranking_bt and sequence_nll moved by offset plus an alternating +-noise (symmetric, so offset 0 stays centred)."""
    delta = offset + noise * (1 if index % 2 == 0 else -1)
    moved = dict(group, record_ids=list(group['record_ids']), ranking_bt=group['ranking_bt'] + delta, sequence_nll=group['sequence_nll'] + delta)
    moved['bt_numerator'] = moved['ranking_bt'] * group['bt_denominator']
    return moved


def aggregate(groups):
    weights = sum(g['bt_denominator'] for g in groups)
    weighted = lambda key: sum(g[key] * g['bt_denominator'] for g in groups) / weights
    return dict(ranking_bt=sum(g['bt_numerator'] for g in groups) / weights, sequence_nll=weighted('sequence_nll'),
                ranking_listwise=weighted('ranking_listwise'), runtime_huber=weighted('runtime_huber'))


def panel_file(arm, panel, groups):
    return dict(schema=readout.INPUT_SCHEMA, arm=arm, **SHARED, checkpoint_sha256=CHECKPOINTS[arm], panel=dict(PANEL_BINDINGS[panel]),
                executable_sha256=EXECUTABLE, sampled_examples=8 * len(groups), groups=groups, legacy_aggregate=aggregate(groups),
                final_tests_opened=False)


def files(reserved=NEUTRAL, old=NEUTRAL, panels=readout.PANELS):
    """(offset, noise) shapes B - A on each panel."""
    table = {}
    for panel in panels:
        offset, noise = reserved if panel == 'reserved' else old
        a_groups = [base_group(*c) for c in contexts(panel)]
        table[('A', panel)] = panel_file('A', panel, a_groups)
        table[('B', panel)] = panel_file('B', panel, [shifted(g, i, offset, noise) for i, g in enumerate(a_groups)])
    return table


def write(directory, table):
    paths = []
    for (arm, panel), value in sorted(table.items()):
        path = Path(directory) / ('%s-%s.json' % (panel, arm))
        atomic(path, value)
        paths.append(path)
    return paths


def legacy_receipt(value, **overrides):
    binding = dict(checkpoint_sha256=value['checkpoint_sha256'], corpus_sha256=value['panel']['corpus_sha256'],
                   panel_sha256=value['panel']['panel_sha256'], executable_sha256=value['executable_sha256'])
    receipt = dict(schema=readout.LEGACY_SCHEMA, checkpoint='arm', evaluated_step=66000, timestamp=1789300000, binding=binding,
                   **value['legacy_aggregate'], final_tests_opened=False)
    receipt.update(overrides)
    return receipt


def run(directory, table, legacy=(), preregistration=PREREGISTRATION):
    registration = readout.preregistration(preregistration)
    return readout.evaluate([readout.validate_input(p) for p in write(directory, table)], registration, legacy)


class Fixture(unittest.TestCase):
    def test_fixture_geometry_and_bridge_distinction(self):
        table = files()
        reserved, old = table[('A', 'reserved')], table[('A', 'old_frozen')]
        self.assertEqual(len(reserved['groups']), 256)
        self.assertEqual(len({g['context_index'] for g in reserved['groups']}), 256)
        self.assertEqual(len(old['groups']), 256)
        self.assertEqual(len({g['context_index'] for g in old['groups']}), 226)
        self.assertEqual(old['sampled_examples'], 2048)
        bridge = readout.numerical_bridge(reserved['groups'], reserved['legacy_aggregate']['ranking_bt'])
        self.assertTrue(bridge['within_1e-6'])
        self.assertGreater(abs(bridge['reconstructed'] - bridge['mean_of_group_means']), 1e-4)   # the fixture tells the two aggregates apart


class Verdicts(unittest.TestCase):
    def test_repair_signal_when_reserved_improves_and_old_panel_is_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(reserved=REPAIR, old=NEUTRAL))
        self.assertEqual(receipt['verdict'], 'repair_signal')
        self.assertEqual(receipt['reason'], 'reserved_delta_clears_threshold_and_old_panel_within_tolerance')
        reserved = receipt['panels']['reserved']
        self.assertEqual(reserved['n_families'], 8)
        self.assertEqual(set(reserved['contexts_per_family'].values()), {32})
        self.assertEqual(reserved['distinct_contexts'], 256)
        self.assertEqual(reserved['role'], 'primary')
        bt = reserved['metrics']['ranking_bt']
        self.assertAlmostEqual(bt['equal_family_mean'], -0.02, places=12)
        self.assertEqual(len(bt['family_means']), 8)
        self.assertLess(bt['ci95'][1], 0)
        self.assertLessEqual(bt['ci95'][0], bt['equal_family_mean'])
        self.assertLessEqual(bt['equal_family_mean'], bt['ci95'][1])
        self.assertEqual((bt['resamples'], bt['seed'], bt['interval']), (10000, 20260912, 0.95))
        self.assertAlmostEqual(reserved['metrics']['sequence_nll']['equal_family_mean'], -0.02, places=12)
        self.assertLess(reserved['metrics']['sequence_nll']['ci95'][1], 0)
        self.assertTrue(reserved['metrics']['ranking_listwise']['ci95'].startswith('unavailable:'))
        old = receipt['panels']['old_frozen']
        self.assertEqual(old['distinct_contexts'], 226)
        self.assertTrue(old['distinct_contexts_match_preregistration'])
        self.assertEqual(old['role'], 'tolerance')
        self.assertLessEqual(abs(old['metrics']['ranking_bt']['equal_family_mean']), 0.001)
        for panel in receipt['panels'].values():
            for arm in ('A', 'B'):
                self.assertTrue(panel['bridge'][arm]['within_1e-6'])
        self.assertEqual(receipt['verdict_inputs']['reserved_delta'], bt['equal_family_mean'])
        self.assertEqual(receipt['interpretation'], readout.INTERPRETATION)
        self.assertEqual(receipt['hold_rule'], readout.HOLD_RULE)
        self.assertIn('cannot release the 66000 hold', receipt['hold_rule'])
        self.assertIs(receipt['final_tests_opened'], False)
        self.assertEqual([(i['panel'], i['arm']) for i in receipt['inputs']], [('old_frozen', 'A'), ('old_frozen', 'B'), ('reserved', 'A'), ('reserved', 'B')])
        self.assertTrue(receipt['legacy_receipts'].startswith('unavailable:'))
        self.assertTrue(all(c['matches'] for c in receipt['preregistered_prefixes']))
        self.assertEqual({c['field'] for c in receipt['preregistered_prefixes'] if not c['enforced']}, {'panels.old_frozen.executable_sha256'})

    def test_inconclusive_when_the_interval_crosses_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(reserved=(-0.006, 0.08), old=NEUTRAL))
        self.assertEqual(receipt['verdict'], 'inconclusive')
        self.assertEqual(receipt['reason'], 'no_demonstrated_repair')
        self.assertIn('interval reaches 0', receipt['reason_detail'])
        bt = receipt['panels']['reserved']['metrics']['ranking_bt']
        self.assertAlmostEqual(bt['equal_family_mean'], -0.006, places=12)
        self.assertLess(bt['ci95'][0], 0)
        self.assertGreater(bt['ci95'][1], 0)
        self.assertFalse(receipt['verdict_inputs']['reserved_clears_threshold'])
        self.assertTrue(receipt['verdict_inputs']['old_panel_within_tolerance'])

    def test_inconclusive_when_the_mean_misses_the_threshold_with_a_tight_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(reserved=(-0.003, 0.0005), old=NEUTRAL))
        self.assertEqual(receipt['verdict'], 'inconclusive')
        self.assertEqual(receipt['reason'], 'no_demonstrated_repair')
        self.assertIn('above reserved_delta_max', receipt['reason_detail'])
        self.assertLess(receipt['panels']['reserved']['metrics']['ranking_bt']['ci95'][1], 0)

    def test_inconclusive_when_the_old_panel_is_worse_than_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(reserved=REPAIR, old=WORSE))
        self.assertEqual(receipt['verdict'], 'inconclusive')
        self.assertEqual(receipt['reason'], 'old_panel_worse_than_tolerance')
        self.assertTrue(receipt['verdict_inputs']['reserved_clears_threshold'])
        self.assertFalse(receipt['verdict_inputs']['old_panel_within_tolerance'])
        self.assertGreater(receipt['verdict_inputs']['old_panel_delta'], 0.005)

    def test_missing_reserved_panel_is_inconclusive_with_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(old=NEUTRAL, panels=('old_frozen',)))
        self.assertEqual((receipt['verdict'], receipt['reason']), ('inconclusive', 'reserved_panel_missing'))
        self.assertIsNone(receipt['verdict_inputs']['reserved_delta'])
        self.assertEqual(sorted(receipt['panels']), ['old_frozen'])


class Refusals(unittest.TestCase):
    def refuse(self, table, fragment, legacy=()):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError) as caught:
                run(directory, table, legacy)
        self.assertIn(fragment, str(caught.exception))

    def test_bridge_failure_is_refused(self):
        table = files()
        table[('B', 'reserved')]['legacy_aggregate']['ranking_bt'] += 1e-4
        self.refuse(table, 'bridge_failed')

    def test_ranking_bt_must_equal_its_ratio(self):
        table = files()
        table[('A', 'old_frozen')]['groups'][3]['ranking_bt'] += 1e-6
        self.refuse(table, 'ranking_bt_not_ratio')

    def test_identity_mismatch_is_refused(self):
        table = files()
        table[('B', 'reserved')]['groups'][5]['record_ids'][0] = 999999
        self.refuse(table, 'identity_mismatch')
        table = files()
        table[('B', 'old_frozen')]['groups'][100]['cut'] = 'cut-other'   # context 1100 is drawn once
        self.refuse(table, 'identity_mismatch')
        table = files()
        table[('B', 'old_frozen')]['groups'][7]['cut'] = 'cut-other'   # context 1007 is drawn twice: its two groups now disagree
        self.refuse(table, 'context_index 1007 is bound to two different (dataset, cut, family, partition)')

    def test_reserved_family_with_wrong_context_count_is_refused(self):
        table = files()
        for arm in ('A', 'B'):
            value = table[(arm, 'reserved')]
            del value['groups'][0]   # family0 drops to 31 contexts; the aggregate follows so the bridge still holds
            value['legacy_aggregate'] = aggregate(value['groups'])
        self.refuse(table, 'family context counts differ from the pre-registered 32')
        table = files()
        for arm in ('A', 'B'):
            value = table[(arm, 'reserved')]
            value['groups'] = [g for g in value['groups'] if g['family'] != 'family7']
            value['legacy_aggregate'] = aggregate(value['groups'])
        self.refuse(table, 'reserved panel has 7 families; 8 pre-registered')

    def test_duplicate_arm_panel_and_missing_arm_are_refused(self):
        table = files()
        with tempfile.TemporaryDirectory() as directory:
            paths = write(directory, table)
            extra = Path(directory) / 'again.json'
            atomic(extra, table[('A', 'reserved')])
            registration = readout.preregistration(PREREGISTRATION)
            with self.assertRaises(RuntimeError) as caught:
                readout.evaluate([readout.validate_input(p) for p in paths + [extra]], registration)
            self.assertIn('duplicate input for arm A on panel reserved', str(caught.exception))
        del table[('B', 'old_frozen')]
        self.refuse(table, 'panel old_frozen lacks arm B')

    def test_shared_digests_must_agree_and_match_the_preregistration(self):
        table = files()
        table[('B', 'reserved')]['protocol_sha256'] = padded('d8d57069ffff')
        self.refuse(table, 'protocol_sha256')
        table = files()
        for value in table.values():
            value['protocol_sha256'] = padded('0badf00d')
        self.refuse(table, 'not_preregistered: protocol_sha256')
        table = files()
        for arm in ('A', 'B'):
            table[(arm, 'reserved')]['panel']['panel_sha256'] = padded('deadbeef')
        self.refuse(table, 'not_preregistered: panels.reserved.panel_sha256')
        table = files()
        table[('B', 'reserved')]['checkpoint_sha256'] = padded('bbbb0009')
        self.refuse(table, 'arm B carries two checkpoints across panels')
        table = files()
        for panel in readout.PANELS:
            table[('B', panel)]['checkpoint_sha256'] = CHECKPOINTS['A']
        self.refuse(table, 'arms A and B share one checkpoint')

    def test_schema_and_shape_refusals(self):
        table = files()
        table[('A', 'reserved')]['schema'] = 'parallel.frozen_monitoring.v1'
        self.refuse(table, 'schema is not ' + readout.INPUT_SCHEMA)
        table = files()
        table[('A', 'reserved')]['groups'][0]['record_ids'] = list(range(7))
        self.refuse(table, 'record_ids is not exactly 8 distinct integers')
        table = files()
        table[('A', 'reserved')]['groups'][1]['sequence_nll'] = float('nan')
        self.refuse(table, 'sequence_nll is not finite')
        table = files()
        table[('A', 'reserved')]['groups'].append(copy.deepcopy(table[('A', 'reserved')]['groups'][0]))
        self.refuse(table, 'duplicate group identity')
        table = files()
        table[('A', 'reserved')]['final_tests_opened'] = True
        self.refuse(table, 'final_tests_opened is not false')


class Legacy(unittest.TestCase):
    def test_agreeing_legacy_receipt_is_recorded_and_disagreeing_one_refused(self):
        table = files(reserved=REPAIR)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / 'legacy-old-A.json'
            atomic(good, legacy_receipt(table[('A', 'old_frozen')]))
            receipt = run(directory, table, [good])
            self.assertEqual(len(receipt['legacy_receipts']), 1)
            row = receipt['legacy_receipts'][0]
            self.assertEqual((row['arm'], row['panel'], row['sha256']), ('A', 'old_frozen', sha(good)))
            self.assertTrue(row['within_1e-6'])
            self.assertLessEqual(row['abs_diff']['reconstructed_ranking_bt'], 1e-6)
            bad = root / 'legacy-old-B.json'
            atomic(bad, legacy_receipt(table[('B', 'old_frozen')], ranking_bt=table[('B', 'old_frozen')]['legacy_aggregate']['ranking_bt'] + 1e-4))
            with self.assertRaises(RuntimeError) as caught:
                run(directory, table, [good, bad])
            self.assertIn('legacy_disagree', str(caught.exception))
            orphan = root / 'legacy-orphan.json'
            atomic(orphan, legacy_receipt(table[('B', 'old_frozen')], binding=dict(legacy_receipt(table[('B', 'old_frozen')])['binding'],
                                                                                   checkpoint_sha256=padded('cccc0003'))))
            with self.assertRaises(RuntimeError) as caught:
                run(directory, table, [orphan])
            self.assertIn('matches 0 inputs', str(caught.exception))


class Preregistration(unittest.TestCase):
    def test_thresholds_come_from_the_real_block(self):
        registration = readout.preregistration(PREREGISTRATION)
        block = registration['block']
        f9 = block['falsifiers']['F9']
        self.assertEqual(f9['thresholds'], dict(reserved_delta_max=-0.005, old_panel_tolerance=0.005, interval=0.95, resamples=10000))
        self.assertEqual(sorted(f9['verdicts']), sorted(readout.VERDICTS))
        self.assertEqual(block['statistics']['seed'], 20260912)
        policy = readout.thresholds(block)
        self.assertEqual({k: policy[k] for k in ('reserved_delta_max', 'old_panel_tolerance', 'interval', 'resamples', 'seed')},
                         dict(reserved_delta_max=-0.005, old_panel_tolerance=0.005, interval=0.95, resamples=10000, seed=20260912))
        self.assertEqual(policy['seed_source'], 'statistics.seed')
        self.assertEqual(readout.readout_panel(block, 'reserved')['contexts_per_family'], 32)
        self.assertEqual(readout.readout_panel(block, 'reserved')['families'], 8)
        self.assertEqual(readout.readout_panel(block, 'old_frozen')['distinct_contexts'], 226)
        self.assertEqual(registration['sha256'], flywheel_baseline.preregistration(PREREGISTRATION)['sha256'])
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(reserved=REPAIR))
        self.assertEqual(receipt['preregistration_sha256'], registration['sha256'])
        self.assertEqual(receipt['thresholds'], policy)
        self.assertEqual(receipt['verdict_rules'], f9['verdicts'])
        for name in readout.BOOTSTRAPPED:
            self.assertEqual((receipt['panels']['reserved']['metrics'][name]['resamples'], receipt['panels']['reserved']['metrics'][name]['seed']),
                             (10000, 20260912))

    def test_pinned_digest_and_missing_thresholds(self):
        registration = readout.preregistration(PREREGISTRATION)
        self.assertEqual(readout.preregistration(PREREGISTRATION, registration['sha256'])['matched_by'], 'canonical_json')
        with self.assertRaises(RuntimeError):
            readout.preregistration(PREREGISTRATION, 'f' * 64)
        block = copy.deepcopy(registration['block'])
        del block['falsifiers']['F9']['thresholds']['resamples']
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / 'PREREGISTRATION.md'
            document.write_text('# Pre-registration\n\n```json\n%s\n```\n' % json.dumps(block, indent=2, sort_keys=True))
            with self.assertRaises(RuntimeError) as caught:
                run(directory, files(), preregistration=document)
            self.assertIn('lacks a finite falsifiers.F9.thresholds.resamples', str(caught.exception))
            block['falsifiers']['F9']['thresholds']['resamples'] = 10000
            del block['statistics']['seed']
            document.write_text('# Pre-registration\n\n```json\n%s\n```\n' % json.dumps(block, indent=2, sort_keys=True))
            policy = readout.thresholds(readout.preregistration(document)['block'])
            self.assertEqual(policy['seed'], readout.DEFAULT_SEED)
            self.assertIn('default', policy['seed_source'])


class Immutability(unittest.TestCase):
    def test_receipt_is_immutable_and_cli_summary_is_one_line(self):
        table = files(reserved=REPAIR)
        with tempfile.TemporaryDirectory() as directory:
            paths = write(directory, table)
            output = Path(directory) / 'receipts' / 'f9.json'
            argv = ['--preregistration', str(PREREGISTRATION), '--inputs', *map(str, paths), '--output', str(output)]
            first = readout.main(argv)
            first_bytes = output.read_bytes()
            self.assertEqual(json.loads(first_bytes), first)
            self.assertEqual(first['verdict'], 'repair_signal')
            readout.main(argv)
            self.assertEqual(output.read_bytes(), first_bytes)
            pinned = argv + ['--preregistration-sha256', readout.preregistration(PREREGISTRATION)['sha256']]
            readout.main(pinned)
            self.assertEqual(output.read_bytes(), first_bytes)   # the pin changes no receipt byte
            with self.assertRaises(RuntimeError):
                readout.main(argv + ['--preregistration-sha256', 'e' * 64])
            changed = files(reserved=(-0.021, 0.005))
            write(directory, changed)
            with self.assertRaises(RuntimeError) as caught:
                readout.main(argv)
            self.assertIn('immutable receipt changed', str(caught.exception))
            self.assertEqual(output.read_bytes(), first_bytes)
            dry = readout.main(['--preregistration', str(PREREGISTRATION), '--inputs', *map(str, paths), '--dry-run'])
            self.assertEqual(dry['verdict'], 'repair_signal')
            self.assertEqual(sorted(p.name for p in (Path(directory) / 'receipts').iterdir()), ['f9.json'])


if __name__ == '__main__':
    unittest.main()
