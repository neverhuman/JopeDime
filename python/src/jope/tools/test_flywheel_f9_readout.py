import copy
import json
from pathlib import Path
import statistics
import tempfile
import unittest

import flywheel_baseline
import flywheel_f9_readout as readout
from resume_expired_cycle import atomic, read, sha

TOOLS = Path(__file__).resolve().parent
PREREGISTRATION = TOOLS.parent / 'recovery/eight-a100-20260912/PREREGISTRATION.md'
PANELS = read(TOOLS / 'fixtures/f9-panels.json')['panels']   # the real batch_records of both panels
FAMILIES = ['family%d' % i for i in range(8)]
NEUTRAL, REPAIR, WORSE = (0.0, 0.004), (-0.02, 0.005), (0.02, 0.004)
PAIRS_PER_DRAW = 28


def padded(prefix):
    """A 64-hex digest starting with the pre-registered prefix."""
    return (prefix + '0' * 64)[:64]


SHARED = dict(protocol_sha256=padded('d8d57069'), parent_state_sha256=padded('d715c47a'), parent_model_sha256=padded('3253c386'),
              parent_optimizer_sha256=padded('da43282d'), corpus_sha256=padded('ddbca1d3'))
CHECKPOINTS = dict(A=padded('aaaa0001'), B=padded('bbbb0002'), parent=SHARED['parent_state_sha256'])
EXECUTABLE = padded('5eabdc94')
REQUEST = dict(A=padded('0a0a0a0a'), B=padded('0b0b0b0b'), parent=padded('0c0c0c0c'))


def contexts(panel):
    """Reserved: 8 families x 32 contexts, one draw each. Old frozen: 226 distinct contexts over 256 draws (30 drawn twice)."""
    if panel == 'reserved':
        return [(FAMILIES[f], f * 32 + c) for f in range(8) for c in range(32)]
    return [(FAMILIES[c % 8], 1000 + c) for c in range(226)] + [(FAMILIES[c % 8], 1000 + c) for c in range(30)]


def batch_records(panel):
    return PANELS[panel]['records']


def draw_values(context, index, offset, noise):
    """Arm shift: ranking_bt and sequence_nll moved by offset plus an alternating +-noise (symmetric, so offset 0 stays centred)."""
    delta = offset + noise * (1 if index % 2 == 0 else -1)
    weight = 8.0 + 8.0 * (context % 7)   # raw batch weight rises with the loss: pooled ratios sit above the mean of draw ratios
    return weight, 0.40 + 0.02 * (context % 7) + delta, 1.40 + 0.001 * (context % 11) + delta


def sidecar(arm, panel, offset=0.0, noise=0.0, undefined=()):
    """A parallel.panel_group_metrics.v2 groups.json on the real batch_records with synthetic contexts and losses."""
    records, draws = batch_records(panel), contexts(panel)
    groups, batches = [], []
    for b in range(readout.BATCHES):
        rows = draws[readout.DRAWS_PER_BATCH * b:readout.DRAWS_PER_BATCH * (b + 1)]
        defined = [readout.DRAWS_PER_BATCH * b + g not in undefined for g in range(len(rows))]
        weights = [draw_values(c, readout.DRAWS_PER_BATCH * b + g, offset, noise)[0] for g, (_, c) in enumerate(rows)]
        batch_weight = sum(w for w, d in zip(weights, defined) if d)
        pair_count = PAIRS_PER_DRAW * sum(defined)
        scale = batch_weight / pair_count if pair_count else 0.0
        batch_groups = []
        for g, (family, context) in enumerate(rows):
            index = readout.DRAWS_PER_BATCH * b + g
            weight, bt, nll = draw_values(context, index, offset, noise)
            ids = records[b][8 * g:8 * g + 8]
            live = defined[g]
            pairs = [dict(winner_record_id=ids[i], loser_record_id=ids[j], winner_row_in_batch=8 * g + i, loser_row_in_batch=8 * g + j,
                          raw_weight=weight / PAIRS_PER_DRAW, batch_normalized_weight_f32=weight / PAIRS_PER_DRAW / scale, reward_margin=0.1,
                          softplus_loss_f32=bt, raw_weighted_loss=weight / PAIRS_PER_DRAW * bt, effective_weighted_loss_f32=weight / PAIRS_PER_DRAW / scale * bt,
                          sign_accuracy=1.0, sigmoid_brier=0.2) for i in range(8) for j in range(i + 1, 8)] if live else []
            effective_numerator = weight / scale * bt if live else 0.0
            batch_groups.append(dict(
                draw_index=index, batch_index=b, draw_in_batch=g, context_index=context, dataset='dataset-%d' % context, cut='cut-%d' % (context % 3),
                family=family, partition='meta_validation', record_ids=list(ids), n_pairs=len(pairs), pairs=pairs,
                ranking_bt=bt if live else None, bt_defined=live, bt_raw_numerator=bt * weight if live else 0.0, bt_raw_denominator=weight if live else 0.0,
                bt_numerator=bt * weight / batch_weight if live else 0.0, bt_denominator=weight / batch_weight if live else 0.0,
                bt_legacy_contribution=effective_numerator / pair_count if live else 0.0, bt_effective_numerator=effective_numerator,
                bt_effective_denominator=weight / scale if live else 0.0,
                sequence_nll=nll, nll_raw_numerator=8 * nll, nll_raw_denominator=8.0, nll_numerator=8 * nll / 256, nll_denominator=8 / 256,
                nll_legacy_contribution=8 * nll / 256, ranking_listwise=2.2 + 0.001 * (context % 5), runtime_huber=0.07,
                pairwise_sign_accuracy=0.6 + 0.01 * (context % 4) if live else None, pairwise_sigmoid_brier=0.2 if live else None))
        legacy = [sum(g['nll_legacy_contribution'] for g in batch_groups), sum(g['bt_legacy_contribution'] for g in batch_groups),
                  statistics.fmean(g['ranking_listwise'] for g in batch_groups), statistics.fmean(g['runtime_huber'] for g in batch_groups)]
        batches.append(dict(batch_index=b, record_ids=list(records[b]), padded_sequence_length=64, draws=readout.DRAWS_PER_BATCH,
                            bt_raw_denominator=batch_weight, bt_batch_mean_raw_weight=scale, bt_pair_count_divisor=pair_count,
                            bt_effective_weight_sum=sum(g['bt_effective_denominator'] for g in batch_groups), bt_effective_reconstructed=legacy[1],
                            nll_batch_denominator_f32=256.0, legacy_values=legacy, effective_group_reconstructed=list(legacy),
                            raw_bt_ratio_reconstructed=sum(g['bt_numerator'] for g in batch_groups), bridge_absolute_errors=[0.0] * 4,
                            bridge_tolerance=1e-6, zero_pair_draws=sum(1 for g in batch_groups if not g['bt_defined'])))
        groups.extend(batch_groups)
    aggregate = {m: readout.batch_mean(batch['legacy_values'][i] for batch in batches) for i, m in enumerate(readout.LEGACY_METRICS)}
    return dict(schema=readout.INPUT_SCHEMA, arm=arm, **SHARED, checkpoint_sha256=CHECKPOINTS[arm], executable_sha256=EXECUTABLE, request_sha256=REQUEST[arm],
                panel=dict(name=panel, panel_sha256=PANELS[panel]['panel_sha256'], file_sha256=padded('f11e0001'), corpus_sha256=PANELS[panel]['corpus_sha256']),
                batch_records=[list(b) for b in records], batches=batches, groups=groups, sampled_examples=2048, draw_count=256,
                distinct_contexts=len({c for _, c in draws}), legacy_aggregate=aggregate, pair_semantics='fixture', bt_semantics='fixture',
                duplicate_draw_semantics='fixture', external_879_numerical_bridge_qualified=False, independent_evaluation=False, final_tests_opened=False)


def legacy_receipt(value, **overrides):
    binding = dict(checkpoint_sha256=value['checkpoint_sha256'], corpus_sha256=value['panel']['corpus_sha256'],
                   panel_sha256=value['panel']['panel_sha256'], executable_sha256=value['executable_sha256'])
    receipt = dict(schema=readout.LEGACY_SCHEMA, binding=binding, checkpoint='arm', evaluated_step=66000, timestamp=1789300000,
                   **value['legacy_aggregate'], sampled_examples=2048, family_balanced=True, training_selection_used=False,
                   evaluation_bias='frozen adaptive teaching validation panel', independent_evaluation=False, final_tests_opened=False)
    receipt.update(overrides)
    return receipt


def files(reserved=NEUTRAL, old=NEUTRAL, panels=readout.PANELS, undefined=(), parent=()):
    """(offset, noise) shapes B - A on each panel; `undefined` lists draw indices without eligible pairs on arm B of the named panels."""
    table, spec = {}, dict(undefined) if undefined else {}
    for panel in panels:
        offset, noise = reserved if panel == 'reserved' else old
        table[('A', panel)] = sidecar('A', panel)
        table[('B', panel)] = sidecar('B', panel, offset, noise, undefined=spec['draws'] if panel in spec.get('panels', ()) else ())
        if panel in parent:
            table[('parent', panel)] = sidecar('parent', panel, 0.03, 0.0)
    return table


def write(directory, table, legacy=None):
    """One sidecar directory per (arm, panel): groups.json beside legacy.json (the emitter's published pair)."""
    paths = []
    for (arm, panel), value in sorted(table.items()):
        folder = Path(directory) / ('%s-%s' % (panel, arm))
        folder.mkdir(exist_ok=True)
        atomic(folder / readout.GROUPS_FILE, value)
        atomic(folder / readout.LEGACY_FILE, (legacy or {}).get((arm, panel)) or legacy_receipt(value))
        paths.append(folder)
    return paths


def run(directory, table, legacy=None, preregistration=PREREGISTRATION):
    registration = readout.preregistration(preregistration)
    return readout.evaluate([readout.validate_input(p) for p in write(directory, table, legacy)], registration)


class Fixture(unittest.TestCase):
    def test_real_panels_and_bridge_distinction(self):
        for name, prefix in (('reserved', '961a2d52'), ('old_frozen', 'fbff7a04')):
            self.assertTrue(PANELS[name]['panel_sha256'].startswith(prefix))
            self.assertEqual(readout.compact_sha256(PANELS[name]['records']), PANELS[name]['panel_sha256'])
        table = files()
        reserved, old = table[('A', 'reserved')], table[('A', 'old_frozen')]
        self.assertEqual((len(reserved['groups']), reserved['distinct_contexts']), (256, 256))
        self.assertEqual((len(old['groups']), old['distinct_contexts']), (256, 226))
        self.assertEqual(len({i for b in old['batch_records'] for i in b}), 2018)
        bridge = readout.numerical_bridge(reserved['groups'], reserved['batches'], reserved['legacy_aggregate'])
        self.assertTrue(bridge['within_1e-6'])
        self.assertEqual(len(bridge['per_batch']), 8)
        self.assertGreater(abs(bridge['batch_mean']['ranking_bt'] - bridge['mean_of_defined_draw_ranking_bt']), 1e-4)   # pooled vs mean of ratios
        self.assertAlmostEqual(bridge['raw_bt_ratio_mean_of_batches'], bridge['batch_mean']['ranking_bt'], places=9)   # uniform losses within a draw


class Verdicts(unittest.TestCase):
    def test_repair_signal_when_reserved_improves_and_old_panel_is_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(reserved=REPAIR, old=NEUTRAL))
        self.assertEqual(receipt['schema'], readout.SCHEMA)
        self.assertEqual(receipt['verdict'], 'repair_signal')
        self.assertEqual(receipt['reason'], 'reserved_delta_clears_threshold_and_old_panel_within_tolerance')
        reserved = receipt['panels']['reserved']
        self.assertEqual((reserved['n_families'], reserved['distinct_contexts'], reserved['n_draws'], reserved['role']), (8, 256, 256, 'primary'))
        self.assertEqual(set(reserved['contexts_per_family'].values()), {32})
        self.assertEqual(reserved['contexts_drawn_more_than_once'], 0)
        bt = reserved['metrics']['ranking_bt']
        self.assertTrue(bt['defined'])
        self.assertAlmostEqual(bt['equal_family_mean'], -0.02, places=12)
        self.assertEqual(len(bt['family_means']), 8)
        self.assertLess(bt['ci95'][1], 0)
        self.assertLessEqual(bt['ci95'][0], bt['equal_family_mean'])
        self.assertLessEqual(bt['equal_family_mean'], bt['ci95'][1])
        self.assertEqual((bt['resamples'], bt['seed'], bt['interval']), (10000, 20260912, 0.95))
        self.assertAlmostEqual(reserved['metrics']['sequence_nll']['equal_family_mean'], -0.02, places=12)
        self.assertLess(reserved['metrics']['sequence_nll']['ci95'][1], 0)
        for name in ('ranking_listwise', 'runtime_huber', 'pairwise_sign_accuracy', 'pairwise_sigmoid_brier'):
            self.assertTrue(reserved['metrics'][name]['ci95'].startswith('unavailable:'))
            self.assertEqual(reserved['metrics'][name]['role'], 'reported')
        self.assertAlmostEqual(reserved['metrics']['pairwise_sign_accuracy']['equal_family_mean'], 0.0, places=12)
        old = receipt['panels']['old_frozen']
        self.assertEqual((old['distinct_contexts'], old['n_draws'], old['contexts_drawn_more_than_once'], old['role']), (226, 256, 30, 'tolerance'))
        self.assertTrue(old['distinct_contexts_match_preregistration'])
        self.assertLessEqual(abs(old['metrics']['ranking_bt']['equal_family_mean']), 0.001)
        for panel in receipt['panels'].values():
            for arm in ('A', 'B'):
                self.assertTrue(panel['bridge'][arm]['within_1e-6'])
                self.assertTrue(panel['arms'][arm]['legacy_json']['within_1e-6'])
                self.assertEqual(panel['arms'][arm]['undefined_draws'], [])
            self.assertTrue(panel['parent_reference'].startswith('unavailable:'))
        self.assertEqual(receipt['verdict_inputs']['reserved_delta'], bt['equal_family_mean'])
        self.assertEqual(receipt['interpretation'], readout.INTERPRETATION)
        self.assertEqual(receipt['hold_rule'], readout.HOLD_RULE)
        self.assertIn('cannot release the 66000 hold', receipt['hold_rule'])
        self.assertIs(receipt['final_tests_opened'], False)
        self.assertIs(receipt['external_879_numerical_bridge_qualified'], False)
        self.assertEqual([(i['panel'], i['arm']) for i in receipt['inputs']], [('old_frozen', 'A'), ('old_frozen', 'B'), ('reserved', 'A'), ('reserved', 'B')])
        self.assertTrue(all(i['legacy_path'].endswith(readout.LEGACY_FILE) for i in receipt['inputs']))
        self.assertTrue(all(c['matches'] for c in receipt['preregistered_prefixes'] if c['enforced']))
        self.assertEqual({c['field'] for c in receipt['preregistered_prefixes'] if not c['enforced']}, {'panels.old_frozen.executable_sha256'})
        self.assertFalse([c for c in receipt['preregistered_prefixes'] if c['field'] == 'panels.old_frozen.executable_sha256'][0]['matches'])   # v2 emitter, reported only

    def test_inconclusive_when_the_interval_crosses_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(reserved=(-0.006, 0.08), old=NEUTRAL))
        self.assertEqual((receipt['verdict'], receipt['reason']), ('inconclusive', 'no_demonstrated_repair'))
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
        self.assertEqual((receipt['verdict'], receipt['reason']), ('inconclusive', 'no_demonstrated_repair'))
        self.assertIn('above reserved_delta_max', receipt['reason_detail'])
        self.assertLess(receipt['panels']['reserved']['metrics']['ranking_bt']['ci95'][1], 0)

    def test_inconclusive_when_the_old_panel_is_worse_than_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(reserved=REPAIR, old=WORSE))
        self.assertEqual((receipt['verdict'], receipt['reason']), ('inconclusive', 'old_panel_worse_than_tolerance'))
        self.assertTrue(receipt['verdict_inputs']['reserved_clears_threshold'])
        self.assertFalse(receipt['verdict_inputs']['old_panel_within_tolerance'])
        self.assertGreater(receipt['verdict_inputs']['old_panel_delta'], 0.005)

    def test_missing_reserved_panel_is_inconclusive_with_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(old=NEUTRAL, panels=('old_frozen',)))
        self.assertEqual((receipt['verdict'], receipt['reason']), ('inconclusive', 'reserved_panel_missing'))
        self.assertIsNone(receipt['verdict_inputs']['reserved_delta'])
        self.assertEqual(sorted(receipt['panels']), ['old_frozen'])

    def test_undefined_ranking_evidence_is_reported_never_zeroed(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(reserved=REPAIR, undefined=dict(panels=('reserved',), draws=(5, 70))))
        self.assertEqual((receipt['verdict'], receipt['reason']), ('inconclusive', 'primary_statistic_undefined'))
        self.assertIn('[5, 70]', receipt['reason_detail'])
        reserved = receipt['panels']['reserved']
        for name in ('ranking_bt', 'pairwise_sign_accuracy', 'pairwise_sigmoid_brier'):
            self.assertFalse(reserved['metrics'][name]['defined'])
            self.assertEqual(reserved['metrics'][name]['undefined_draws'], [5, 70])
            self.assertTrue(reserved['metrics'][name]['equal_family_mean'].startswith('unavailable:undefined_ranking_evidence'))
        self.assertTrue(reserved['metrics']['sequence_nll']['defined'])
        self.assertLess(reserved['metrics']['sequence_nll']['ci95'][1], 0)
        self.assertEqual(reserved['distinct_contexts'], 256)   # contexts are never dropped
        self.assertEqual(reserved['arms']['B']['undefined_draws'], [5, 70])
        self.assertEqual(reserved['bridge']['B']['per_batch'][0]['zero_pair_draws'], 1)
        self.assertIsNone(receipt['verdict_inputs']['reserved_delta'])
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(reserved=REPAIR, undefined=dict(panels=('old_frozen',), draws=(255,))))
        self.assertEqual((receipt['verdict'], receipt['reason']), ('inconclusive', 'tolerance_statistic_undefined'))
        self.assertAlmostEqual(receipt['verdict_inputs']['reserved_delta'], -0.02, places=12)
        self.assertIsNone(receipt['verdict_inputs']['old_panel_delta'])

    def test_parent_sidecar_is_reported_and_never_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, files(reserved=REPAIR, parent=('reserved',)))
        self.assertEqual(receipt['verdict'], 'repair_signal')
        reference = receipt['panels']['reserved']['parent_reference']
        self.assertEqual(reference['checkpoint_sha256'], SHARED['parent_state_sha256'])
        self.assertTrue(reference['bridge']['within_1e-6'])
        self.assertAlmostEqual(reference['arm_minus_parent']['A']['ranking_bt'], -0.03, places=9)
        self.assertLess(reference['arm_minus_parent']['B']['ranking_bt'], reference['arm_minus_parent']['A']['ranking_bt'])
        self.assertEqual([(i['panel'], i['arm']) for i in receipt['inputs']], [('old_frozen', 'A'), ('old_frozen', 'B'), ('reserved', 'A'), ('reserved', 'B'), ('reserved', 'parent')])
        self.assertEqual(sorted(receipt['arms']), ['A', 'B'])

    def test_groups_json_path_is_accepted_like_its_directory(self):
        table = files(reserved=REPAIR)
        with tempfile.TemporaryDirectory() as directory:
            folders = write(directory, table)
            registration = readout.preregistration(PREREGISTRATION)
            by_folder = readout.evaluate([readout.validate_input(p) for p in folders], registration)
            by_file = readout.evaluate([readout.validate_input(p / readout.GROUPS_FILE) for p in folders], registration)
        self.assertEqual(by_folder, by_file)


class Refusals(unittest.TestCase):
    def refuse(self, table, fragment, legacy=None):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError) as caught:
                run(directory, table, legacy)
        self.assertIn(fragment, str(caught.exception))

    def test_bridge_failure_is_refused(self):
        table = files()
        table[('B', 'reserved')]['groups'][3]['bt_legacy_contribution'] += 1e-5   # the batch no longer reconstructs
        self.refuse(table, 'bridge_failed')
        table = files()
        table[('B', 'reserved')]['legacy_aggregate']['ranking_bt'] += 1e-4   # the eight-batch mean no longer matches
        self.refuse(table, 'bridge_failed')

    def test_ranking_bt_must_equal_its_raw_ratio(self):
        table = files()
        table[('A', 'old_frozen')]['groups'][3]['ranking_bt'] += 1e-6
        self.refuse(table, 'ranking_bt_not_ratio')

    def test_draw_identities_are_bound_to_the_panel(self):
        table = files()
        table[('B', 'reserved')]['groups'][5]['record_ids'].reverse()   # same records, wrong order: the ordered tuple is the identity
        self.refuse(table, 'record_ids are not batch_records[0][40:48]')
        table = files()
        table[('B', 'reserved')]['groups'][5]['record_ids'][0] = 999999
        self.refuse(table, 'pairs[0] names a record outside the draw')
        table = files()
        value = table[('B', 'reserved')]
        value['groups'][1], value['groups'][2] = value['groups'][2], value['groups'][1]
        self.refuse(table, 'draws must be listed in forward order')
        table = files()
        table[('B', 'reserved')]['groups'][40]['draw_in_batch'] = 9
        self.refuse(table, 'draw_index 40 is not 32 * batch_index + draw_in_batch')
        table = files()
        table[('B', 'reserved')]['batch_records'][0][0] = 999999
        self.refuse(table, 'panel_sha256 is not the sha256 of the compact json of batch_records')
        table = files()
        table[('B', 'reserved')]['batches'][2]['record_ids'][0] = 999999
        self.refuse(table, 'batches[2]: record_ids differ from batch_records[2]')

    def test_identity_mismatch_is_refused(self):
        table = files()
        table[('B', 'old_frozen')]['groups'][100]['cut'] = 'cut-other'   # context 1100 is drawn once
        self.refuse(table, 'identity_mismatch')
        table = files()
        table[('B', 'old_frozen')]['groups'][7]['cut'] = 'cut-other'   # context 1007 is drawn twice: its two draws now disagree
        self.refuse(table, 'context_index 1007 is bound to two different (dataset, cut, family, partition)')

    def test_reserved_family_with_wrong_context_count_is_refused(self):
        table = files()
        for arm in ('A', 'B'):
            for group in table[(arm, 'reserved')]['groups']:
                if group['context_index'] == 31:
                    group.update(context_index=30, dataset='dataset-30', cut='cut-0')   # family0 drops to 31 contexts; every draw stays paired
            table[(arm, 'reserved')]['distinct_contexts'] = 255
        self.refuse(table, 'family context counts differ from the pre-registered 32')
        table = files()
        for arm in ('A', 'B'):
            for group in table[(arm, 'reserved')]['groups']:
                if group['family'] == 'family7':
                    group['family'] = 'family6'
            table[(arm, 'reserved')]['distinct_contexts'] = 256
        self.refuse(table, 'reserved panel has 7 families; 8 pre-registered')

    def test_duplicate_arm_panel_and_missing_arm_are_refused(self):
        table = files()
        with tempfile.TemporaryDirectory() as directory:
            paths = write(directory, table)
            extra = Path(directory) / 'again'
            extra.mkdir()
            atomic(extra / readout.GROUPS_FILE, table[('A', 'reserved')])
            atomic(extra / readout.LEGACY_FILE, legacy_receipt(table[('A', 'reserved')]))
            registration = readout.preregistration(PREREGISTRATION)
            with self.assertRaises(RuntimeError) as caught:
                readout.evaluate([readout.validate_input(p) for p in paths + [extra]], registration)
            self.assertIn('duplicate input for arm A on panel reserved', str(caught.exception))
        del table[('B', 'old_frozen')]
        self.refuse(table, 'panel old_frozen lacks arm B')
        table = files(parent=('old_frozen',))
        del table[('A', 'old_frozen')]
        del table[('B', 'old_frozen')]
        self.refuse(table, 'panel old_frozen carries a parent sidecar but no arm')

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
        self.refuse(table, 'panel_sha256 is not the sha256')   # the digest is the records' own; it cannot be renamed to another prefix
        table = files()
        table[('B', 'reserved')]['checkpoint_sha256'] = padded('bbbb0009')
        self.refuse(table, 'arm B carries two checkpoints across panels')
        table = files()
        for panel in readout.PANELS:
            table[('B', panel)]['checkpoint_sha256'] = CHECKPOINTS['A']
        self.refuse(table, 'arms A and B share one checkpoint')
        table = files()
        for panel in readout.PANELS:
            table[('B', panel)]['checkpoint_sha256'] = SHARED['parent_state_sha256']
        self.refuse(table, 'arm B carries the parent checkpoint')
        table = files(parent=('reserved',))
        table[('parent', 'reserved')]['checkpoint_sha256'] = padded('cccc0003')
        self.refuse(table, 'parent sidecar checkpoint is not parent_state_sha256')

    def test_schema_and_shape_refusals(self):
        table = files()
        table[('A', 'reserved')]['schema'] = 'parallel.panel_group_metrics.v1'
        self.refuse(table, 'schema is not ' + readout.INPUT_SCHEMA)
        table = files()
        table[('A', 'reserved')]['groups'][0]['record_ids'] = list(range(7))
        self.refuse(table, 'record_ids is not exactly 8 distinct integers')
        table = files()
        table[('A', 'reserved')]['groups'][1]['sequence_nll'] = float('nan')
        self.refuse(table, 'sequence_nll is not finite')
        table = files()
        table[('A', 'reserved')]['final_tests_opened'] = True
        self.refuse(table, 'final_tests_opened is not false')
        table = files()
        table[('A', 'reserved')]['sampled_examples'] = 2040
        self.refuse(table, 'sampled_examples is not 2048')
        table = files()
        table[('A', 'old_frozen')]['distinct_contexts'] = 256
        self.refuse(table, 'distinct_contexts 256 differs from the 226 contexts drawn')
        table = files()
        table[('A', 'reserved')]['groups'][4]['bt_defined'] = False
        self.refuse(table, 'bt_defined is not the boolean (n_pairs > 0)')
        table = files(undefined=dict(panels=('reserved',), draws=(4,)))
        table[('B', 'reserved')]['groups'][4]['ranking_bt'] = 0.0
        self.refuse(table, 'ranking_bt is not null although no pair is eligible')
        table = files(undefined=dict(panels=('reserved',), draws=(4,)))
        table[('B', 'reserved')]['batches'][0]['zero_pair_draws'] = 0
        self.refuse(table, 'zero_pair_draws 0 differs from the 1 draws without pairs')
        table = files()
        table[('A', 'reserved')]['groups'][2]['pairs'][0]['loser_record_id'] = 999999
        self.refuse(table, 'pairs[0] names a record outside the draw')
        table = files()
        del table[('A', 'reserved')]['external_879_numerical_bridge_qualified']
        self.refuse(table, 'external_879_numerical_bridge_qualified is not a boolean')


class Legacy(unittest.TestCase):
    def test_sibling_legacy_json_is_required_and_must_agree(self):
        table = files(reserved=REPAIR)
        key = ('A', 'old_frozen')
        with tempfile.TemporaryDirectory() as directory:
            receipt = run(directory, table)
            row = receipt['panels']['old_frozen']['arms']['A']['legacy_json']
            self.assertEqual(row['sha256'], sha(Path(directory) / 'old_frozen-A' / readout.LEGACY_FILE))
            self.assertTrue(row['within_1e-6'])
            self.assertEqual(row['values'], table[key]['legacy_aggregate'])
        with tempfile.TemporaryDirectory() as directory:
            folders = write(directory, table)
            (Path(directory) / 'old_frozen-A' / readout.LEGACY_FILE).unlink()
            with self.assertRaises(RuntimeError) as caught:
                readout.validate_input(folders[0])
            self.assertIn('legacy.json is missing beside groups.json', str(caught.exception))
        disagreeing = {key: legacy_receipt(table[key], ranking_bt=table[key]['legacy_aggregate']['ranking_bt'] + 1e-4)}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError) as caught:
                run(directory, table, disagreeing)
            self.assertIn('legacy_disagree', str(caught.exception))
        wrong = legacy_receipt(table[key])
        wrong['binding']['checkpoint_sha256'] = padded('cccc0003')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError) as caught:
                run(directory, table, {key: wrong})
            self.assertIn('legacy_binding_mismatch', str(caught.exception))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError) as caught:
                run(directory, table, {key: legacy_receipt(table[key], final_tests_opened=True)})
            self.assertIn('final_tests_opened is not false', str(caught.exception))


def certificate(value, legacy_sha, **overrides):
    binding = dict(checkpoint_sha256=value['checkpoint_sha256'], corpus_sha256=value['panel']['corpus_sha256'], panel_sha256=value['panel']['panel_sha256'])
    doc = dict(schema=readout.CERTIFICATE_SCHEMA, original_executable_sha256=padded('87995977'), emitter_executable_sha256=value['executable_sha256'],
               binding=binding, original_receipt_sha256=padded('0e0e0e0e'), emitter_legacy_sha256=legacy_sha,
               abs_diff={m: 1e-8 for m in readout.LEGACY_METRICS}, tolerance=1e-6, within_tolerance=True, final_tests_opened=False)
    doc.update(overrides)
    return doc


class Certificates(unittest.TestCase):
    def test_bridge_certificate_is_recorded_beside_the_sidecar_flag_never_substituted(self):
        table = files(reserved=REPAIR)
        key = ('A', 'old_frozen')
        with tempfile.TemporaryDirectory() as directory:
            folders = write(directory, table)
            legacy_sha = sha(Path(directory) / 'old_frozen-A' / readout.LEGACY_FILE)
            good = Path(directory) / 'certificate-old-A.json'
            atomic(good, certificate(table[key], legacy_sha))
            registration = readout.preregistration(PREREGISTRATION)
            inputs = [readout.validate_input(p) for p in folders]
            receipt = readout.evaluate(inputs, registration, [good])
            self.assertEqual(receipt['verdict'], 'repair_signal')
            self.assertIs(receipt['external_879_numerical_bridge_qualified'], False)   # the sidecar flag is reported as emitted
            comparability = receipt['legacy_comparability']
            self.assertEqual(comparability['rule'], readout.LEGACY_COMPARABILITY)
            self.assertEqual(len(comparability['certificates']), 1)
            row = comparability['certificates'][0]
            self.assertEqual((row['arm'], row['panel'], row['sha256'], row['certified_input']), ('A', 'old_frozen', sha(good), inputs[0]['path']))
            self.assertTrue(row['within_tolerance'])
            self.assertEqual(comparability['certified_inputs'], [inputs[0]['path']])
            self.assertEqual(len(comparability['uncertified_inputs']), 3)
            self.assertFalse(comparability['all_inputs_certified'])
            without = readout.evaluate(inputs, registration)
            self.assertEqual(without['legacy_comparability']['certificates'], [])
            self.assertEqual(without['verdict_inputs'], receipt['verdict_inputs'])   # the certificate never changes the A/B statistic
            for name, overrides, fragment in (
                    ('orphan', dict(emitter_legacy_sha256=padded('cccc0003')), 'matches 0 inputs'),
                    ('wide', dict(abs_diff=dict(certificate(table[key], legacy_sha)['abs_diff'], ranking_bt=2e-6)), 'certificate_not_within_tolerance'),
                    ('claimed', dict(within_tolerance=False), 'certificate_not_within_tolerance'),
                    ('foreign', dict(original_executable_sha256=padded('deadbeef')), 'pre-registered frozen monitor prefix'),
                    ('emitter', dict(emitter_executable_sha256=padded('0bad0bad')), 'emitter_executable_sha256 differs'),
                    ('binding', dict(binding=dict(certificate(table[key], legacy_sha)['binding'], panel_sha256=padded('0bad0bad'))), 'certificate binding differs'),
                    ('loose', dict(tolerance=1e-3), 'tolerance is not in'),
                    ('opened', dict(final_tests_opened=True), 'final_tests_opened is not false')):
                bad = Path(directory) / ('certificate-%s.json' % name)
                atomic(bad, certificate(table[key], legacy_sha, **overrides))
                with self.assertRaises(RuntimeError, msg=name) as caught:
                    readout.evaluate(inputs, registration, [bad])
                self.assertIn(fragment, str(caught.exception), name)
            twice = Path(directory) / 'certificate-twice.json'
            atomic(twice, certificate(table[key], legacy_sha))
            with self.assertRaises(RuntimeError) as caught:
                readout.evaluate(inputs, registration, [good, twice])
            self.assertIn('two bridge certificates name one input', str(caught.exception))
            output = Path(directory) / 'receipts' / 'f9.json'
            argv = ['--preregistration', str(PREREGISTRATION), '--inputs', *map(str, folders), '--bridge-certificate', str(good), '--output', str(output)]
            self.assertEqual(readout.main(argv)['legacy_comparability']['certified_inputs'], [inputs[0]['path']])


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
        self.assertEqual(readout.readout_panel(block, 'old_frozen')['draws'], 2048)
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
            block['falsifiers']['F9']['readout_panels'][1]['draws'] = 2040
            document.write_text('# Pre-registration\n\n```json\n%s\n```\n' % json.dumps(block, indent=2, sort_keys=True))
            with self.assertRaises(RuntimeError) as caught:
                run(directory, files(), preregistration=document)
            self.assertIn('old_frozen panel presents 2048 records; 2040 pre-registered as draws', str(caught.exception))


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
            write(directory, files(reserved=(-0.021, 0.005)))
            with self.assertRaises(RuntimeError) as caught:
                readout.main(argv)
            self.assertIn('immutable receipt changed', str(caught.exception))
            self.assertEqual(output.read_bytes(), first_bytes)
            dry = readout.main(['--preregistration', str(PREREGISTRATION), '--inputs', *map(str, paths), '--dry-run'])
            self.assertEqual(dry['verdict'], 'repair_signal')
            self.assertEqual(sorted(p.name for p in (Path(directory) / 'receipts').iterdir()), ['f9.json'])


if __name__ == '__main__':
    unittest.main()
