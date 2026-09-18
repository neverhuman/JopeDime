#!/usr/bin/env python3
"""F9 read-out: the two-arm replay-concentration contrast under root's frozen protocol d8d57069.

Reads one parallel.panel_group_metrics.v2 sidecar per (arm, panel) - the
groups.json written by the isolated concentration_monitor emitter together with
its sibling legacy.json (parallel.frozen_monitoring.v1 from the same forward) -
for arm A (the deployed fresh_balanced sampler) and arm B (the d8d57069 quota
schedule), each trained one window from the same fixed parent under one shared
seed and read out on the reserved panel (primary) and the old frozen panel
(tolerance), and writes an immutable parallel.flywheel_f9_readout.v2 receipt
whose verdict is one of {repair_signal, inconclusive}. Every threshold, the
resample count and the interval come from the fenced json block of
PREREGISTRATION.md (falsifiers.F9.thresholds; geometry from
falsifiers.F9.readout_panels); no constant in this file is a threshold.
--preregistration-sha256 pins the block. A `parent` sidecar (the protocol's
shared-parent diagnostic) may be supplied per panel; it is reported, never paired.
--bridge-certificate supplies root's frozen parallel.original_monitor_numerical_bridge.v1
receipts (one per certified legacy.json, original-parent-old.json beside each):
they are verified and recorded under legacy_comparability; the sidecar's own
external_879_numerical_bridge_qualified flag is reported as emitted, never rewritten,
and no certificate changes the A/B statistic.

Numerical bridge (SCHEMA.md of concentration-monitor-metrics-v1): the legacy
ranking_bt is the mean of eight per-batch values, each the f32 dense
softplus-times-normalized-weight sum divided by the batch pair count, so per
batch sum(draw.bt_legacy_contribution) must equal batches[b].legacy_values[1],
sum(draw.nll_legacy_contribution) the sequence_nll, and the means of the draws'
ranking_listwise / runtime_huber the other two, each within 1e-6; the mean of the
eight batch values must equal legacy_aggregate and the sibling legacy.json. The
raw per-draw ranking_bt (N_g / D_g over raw pair weights, context-normalized) is
the scientific statistic; raw bt_numerator / bt_denominator are checked as a
ratio and reported, never substituted for the effective bridge.

Identities: draw_index == 32 * batch_index + draw_in_batch; a draw's record_ids
are exactly batch_records[batch_index][8 * draw_in_batch : +8]; the panel digest
is sha256 of the compact json of batch_records. Arms are paired by draw position
AND the scientific identity (context index, dataset, cut, family, partition,
ordered record ids); A and B must carry identical batch_records. Draws are then
clustered by context (the old panel repeats draws: 256 positions over 226
contexts; both occurrences of a repeated draw contribute to the context mean) and
contexts by family; draw positions are never resampled as contexts.

Undefined ranking evidence: a draw with no eligible pair (bt_defined false,
ranking_bt null) makes ranking_bt undefined for its whole panel - the receipt
says so and the verdict is inconclusive; contexts are never dropped and missing
evidence is never zero.

Statistic: per-draw delta = metric(B) - metric(A); per-context delta = mean over
the context's paired draws; family mean = mean over the family's contexts;
primary statistic = equal-family mean. Interval: percentile interval of the
equal-family mean over stratified resamples drawing contexts with replacement
within each family (one draw serves every metric). ranking_bt primary,
sequence_nll secondary; ranking_listwise, runtime_huber, pairwise_sign_accuracy
and pairwise_sigmoid_brier reported without an interval. The verdict never
authorizes a live block under B and never releases the 66000 hold.
"""
import argparse
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path

import promotion_gate
from flywheel_verdict import lookup, preregistration
from resume_expired_cycle import read, require, sha

SCHEMA = 'parallel.flywheel_f9_readout.v2'
INPUT_SCHEMA = 'parallel.panel_group_metrics.v2'
LEGACY_SCHEMA = 'parallel.frozen_monitoring.v1'
CERTIFICATE_SCHEMA = 'parallel.original_monitor_numerical_bridge.v1'
GROUPS_FILE, LEGACY_FILE, ORIGINAL_RECEIPT_FILE = 'groups.json', 'legacy.json', 'original-parent-old.json'
ARMS = ('A', 'B')
PARENT = 'parent'
PANELS = ('reserved', 'old_frozen')
LEGACY_METRICS = ('sequence_nll', 'ranking_bt', 'ranking_listwise', 'runtime_huber')
REPORTED = ('pairwise_sign_accuracy', 'pairwise_sigmoid_brier')
METRICS = LEGACY_METRICS + REPORTED
PRIMARY, SECONDARY = 'ranking_bt', 'sequence_nll'
BOOTSTRAPPED = (PRIMARY, SECONDARY)
NULLABLE = (PRIMARY,) + REPORTED   # null exactly when a draw has no eligible pair
IDENTITY = ('context_index', 'dataset', 'cut', 'family', 'partition', 'record_ids')
RECORDS_PER_DRAW, DRAWS_PER_BATCH, BATCHES = 8, 32, 8
ROWS_PER_BATCH = RECORDS_PER_DRAW * DRAWS_PER_BATCH
DRAWS = DRAWS_PER_BATCH * BATCHES
SAMPLED_EXAMPLES = ROWS_PER_BATCH * BATCHES
SHARED = ('protocol_sha256', 'parent_state_sha256', 'parent_model_sha256', 'parent_optimizer_sha256', 'corpus_sha256')
PANEL_BINDING = ('panel_sha256', 'corpus_sha256', 'executable_sha256')
LEGACY_BINDING = ('checkpoint_sha256', 'corpus_sha256', 'panel_sha256', 'executable_sha256')
LEGACY_CONTRIBUTION = {'sequence_nll': 'nll_legacy_contribution', 'ranking_bt': 'bt_legacy_contribution'}
LEGACY_MEAN = ('ranking_listwise', 'runtime_huber')
BRIDGE_TOLERANCE = 1e-6
RATIO_TOLERANCE = 1e-9
DEFAULT_SEED = 20260912
THRESHOLDS = 'falsifiers.F9.thresholds'
THRESHOLD_KEYS = ('reserved_delta_max', 'old_panel_tolerance', 'interval', 'resamples')
READOUT_PANELS = 'falsifiers.F9.readout_panels'
VERDICTS = ('repair_signal', 'inconclusive')
INTERPRETATION = 'engineering signal consistent with reduced concentration, not proof of sole cause'
HOLD_RULE = 'cannot authorize the next live block under B and cannot release the 66000 hold'
STATISTIC = ('per-draw delta = metric(B) - metric(A) at one draw position with one scientific identity; per-context delta = mean over '
             'the context\'s paired draws (repeated draws of one context all count); family mean = mean over the family\'s contexts; '
             'equal-family mean = mean of family means; interval = percentile interval of the equal-family mean over stratified resamples '
             'drawing contexts with replacement within each family (one draw per resample serves every metric; draw positions are never '
             'resampled as contexts); ranking_bt (raw per-draw N_g / D_g) primary, sequence_nll secondary')
BRIDGE_RULE = ('per batch: sum(draw.bt_legacy_contribution) = legacy ranking_bt of that batch (f32 dense softplus * normalized weight, '
               'divided by the batch pair count), sum(draw.nll_legacy_contribution) = legacy sequence_nll, mean(draw.ranking_listwise) '
               'and mean(draw.runtime_huber) = the other two, each within 1e-6 of batches[b].legacy_values; the legacy aggregate is the '
               'mean of the eight batch values (sum of value / 8) and must equal legacy_aggregate and the sibling legacy.json within 1e-6. '
               'The raw ratio sum(bt_numerator) / sum(bt_denominator) is reported for interpretation only and never substituted.')
UNDEFINED_RULE = ('a draw with no eligible pair (bt_defined false) makes ranking_bt, pairwise_sign_accuracy and pairwise_sigmoid_brier '
                  'undefined for its panel; contexts are never excluded and missing ranking evidence is never zero')
LEGACY_COMPARABILITY = ('legacy.json is a receipt of the v2 emitter, not an execution of the frozen monitor 87995977; absolute levels are '
                        'comparable with the historical frozen-panel series only once external_879_numerical_bridge_qualified is true')
HEX64 = re.compile(r'^[0-9a-f]{64}$')


# ---------------------------------------------------------------- helpers

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def hex64(value):
    return isinstance(value, str) and bool(HEX64.match(value))


def unavailable(reason):
    return 'unavailable:' + reason


def identity(group):
    return (group['context_index'], group['dataset'], group['cut'], group['family'], group['partition'], tuple(group['record_ids']))


def pair_key(group):
    """Arms are paired by draw position and scientific identity together."""
    return (group['draw_index'],) + identity(group)


def panel_binding(item):
    panel = item['value']['panel']
    return dict(panel_sha256=panel['panel_sha256'], corpus_sha256=panel['corpus_sha256'], executable_sha256=item['value']['executable_sha256'])


def compact_sha256(value):
    """sha256 of serde_json::to_vec-style compact json (the emitter's panel_sha256 of batch_records)."""
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def batch_mean(values):
    """The emitter accumulates value / 8 per batch; reproduce that expression exactly."""
    total = 0.0
    for value in values:
        total += value / BATCHES
    return total


# ---------------------------------------------------------------- preregistration

def thresholds(block):
    """falsifiers.F9.thresholds verbatim plus the seed (statistics.seed, else the documented default)."""
    table = lookup(block, THRESHOLDS)
    require(isinstance(table, dict), 'preregistration block lacks ' + THRESHOLDS)
    for key in THRESHOLD_KEYS:
        require(finite(table.get(key)), 'preregistration block lacks a finite %s.%s' % (THRESHOLDS, key))
    require(type(table['resamples']) is int and table['resamples'] >= 1, '%s.resamples is not a positive integer' % THRESHOLDS)
    require(0 < table['interval'] < 1, '%s.interval is not in (0, 1)' % THRESHOLDS)
    seed = lookup(block, 'statistics.seed')
    pinned = type(seed) is int
    return dict({key: table[key] for key in THRESHOLD_KEYS}, seed=seed if pinned else DEFAULT_SEED,
                seed_source='statistics.seed' if pinned else 'default %d (block fixes no statistics.seed)' % DEFAULT_SEED,
                source=THRESHOLDS)


def readout_panel(block, name):
    entries = lookup(block, READOUT_PANELS)
    require(isinstance(entries, list), 'preregistration block lacks ' + READOUT_PANELS)
    found = [entry for entry in entries if isinstance(entry, dict) and entry.get('name') == name]
    require(len(found) == 1, 'preregistration block lacks exactly one %s entry named %s' % (READOUT_PANELS, name))
    return found[0]


def preregistered_prefixes(block, shared, panels):
    """Every digest prefix the block fixes must agree with the inputs; the old-panel monitor executable is reported only."""
    checks = []

    def check(field, prefix, observed, enforced=True):
        if isinstance(prefix, str) and prefix:
            checks.append(dict(field=field, expected_prefix=prefix, observed=observed, matches=observed.startswith(prefix), enforced=enforced))

    parent = lookup(block, 'falsifiers.F9.parent') or {}
    check('protocol_sha256', lookup(block, 'falsifiers.F9.protocol_sha256_prefix'), shared['protocol_sha256'])
    check('parent_state_sha256', parent.get('state_sha256_prefix'), shared['parent_state_sha256'])
    check('parent_model_sha256', parent.get('model_tensor_sha256_prefix'), shared['parent_model_sha256'])
    check('parent_optimizer_sha256', parent.get('optimizer_sha256_prefix'), shared['parent_optimizer_sha256'])
    check('corpus_sha256', parent.get('corpus_sha256_prefix'), shared['corpus_sha256'])
    for name in sorted(panels):
        entry, binding = readout_panel(block, name), panels[name]['binding']
        check('panels.%s.panel_sha256' % name, entry.get('panel_sha256_prefix'), binding['panel_sha256'])
        check('panels.%s.corpus_sha256' % name, entry.get('corpus_sha256_prefix'), binding['corpus_sha256'])
        check('panels.%s.executable_sha256' % name, entry.get('monitor_executable_sha256_prefix'), binding['executable_sha256'], enforced=False)
    failed = [c for c in checks if c['enforced'] and not c['matches']]
    require(not failed, 'not_preregistered: ' + '; '.join('%s %s... does not start with the pre-registered prefix %s'
                                                          % (c['field'], c['observed'][:12], c['expected_prefix']) for c in failed))
    return checks


# ---------------------------------------------------------------- inputs

def numerical_bridge(groups, batches, aggregate):
    """Per batch, the legacy values reconstructed from the draws' effective contributions; then the eight-batch mean against the aggregate."""
    rows = []
    for batch in batches:
        draws = [g for g in groups if g['batch_index'] == batch['batch_index']]
        reconstructed = {m: sum(g[LEGACY_CONTRIBUTION[m]] for g in draws) for m in LEGACY_CONTRIBUTION}
        reconstructed.update({m: statistics.fmean(g[m] for g in draws) for m in LEGACY_MEAN})
        legacy = dict(zip(LEGACY_METRICS, batch['legacy_values']))
        rows.append(dict(batch_index=batch['batch_index'], draws=len(draws), legacy=legacy, reconstructed=reconstructed,
                         abs_diff={m: abs(reconstructed[m] - legacy[m]) for m in LEGACY_METRICS},
                         emitter_abs_diff=dict(zip(LEGACY_METRICS, batch['bridge_absolute_errors'])),
                         raw_bt_ratio=sum(g['bt_numerator'] for g in draws) / sum(g['bt_denominator'] for g in draws)
                         if sum(g['bt_denominator'] for g in draws) > 0 else None,
                         zero_pair_draws=sum(1 for g in draws if not g['bt_defined'])))
    means = {m: batch_mean(row['legacy'][m] for row in rows) for m in LEGACY_METRICS}
    reconstructed_means = {m: batch_mean(row['reconstructed'][m] for row in rows) for m in LEGACY_METRICS}
    differences = {m: max(row['abs_diff'][m] for row in rows) for m in LEGACY_METRICS}
    aggregate_differences = {m: abs(means[m] - aggregate[m]) for m in LEGACY_METRICS}
    raw = [row['raw_bt_ratio'] for row in rows]
    return {'per_batch': rows, 'max_batch_abs_diff': differences, 'batch_mean': means, 'reconstructed_mean': reconstructed_means,
            'aggregate': dict(aggregate), 'aggregate_abs_diff': aggregate_differences, 'tolerance': BRIDGE_TOLERANCE,
            'within_1e-6': max(differences.values()) <= BRIDGE_TOLERANCE and max(aggregate_differences.values()) <= BRIDGE_TOLERANCE,
            'raw_bt_ratio_mean_of_batches': statistics.fmean(raw) if all(r is not None for r in raw) else None,
            'mean_of_defined_draw_ranking_bt': statistics.fmean(g['ranking_bt'] for g in groups if g['bt_defined'])
            if any(g['bt_defined'] for g in groups) else None, 'rule': BRIDGE_RULE}


def validate_group(group, where):
    require(isinstance(group, dict), where + ': group is not an object')
    for key, bound in (('draw_index', DRAWS), ('batch_index', BATCHES), ('draw_in_batch', DRAWS_PER_BATCH)):
        require(type(group.get(key)) is int and 0 <= group[key] < bound, where + ': %s is not an integer in [0, %d)' % (key, bound))
    require(group['draw_index'] == DRAWS_PER_BATCH * group['batch_index'] + group['draw_in_batch'],
            where + ': draw_index %d is not 32 * batch_index + draw_in_batch' % group['draw_index'])
    require(type(group.get('context_index')) is int and group['context_index'] >= 0, where + ': context_index is not a non-negative integer')
    for key in ('dataset', 'family', 'partition'):
        require(isinstance(group.get(key), str) and group[key], where + ': %s is not a non-empty string' % key)
    require(type(group.get('cut')) in (str, int), where + ': cut is neither a string nor an integer')
    ids = group.get('record_ids')
    require(isinstance(ids, list) and len(ids) == RECORDS_PER_DRAW and all(type(i) is int for i in ids) and len(set(ids)) == RECORDS_PER_DRAW,
            where + ': record_ids is not exactly %d distinct integers' % RECORDS_PER_DRAW)
    require(type(group.get('n_pairs')) is int and group['n_pairs'] >= 0, where + ': n_pairs is not a non-negative integer')
    pairs = group.get('pairs')
    require(isinstance(pairs, list) and len(pairs) == group['n_pairs'], where + ': pairs does not list exactly n_pairs entries')
    members = set(ids)
    for index, pair in enumerate(pairs):
        require(isinstance(pair, dict) and pair.get('winner_record_id') in members and pair.get('loser_record_id') in members,
                where + ': pairs[%d] names a record outside the draw' % index)
    require(type(group.get('bt_defined')) is bool and group['bt_defined'] == (group['n_pairs'] > 0),
            where + ': bt_defined is not the boolean (n_pairs > 0)')
    for key in ('sequence_nll', 'ranking_listwise', 'runtime_huber', 'bt_raw_numerator', 'bt_raw_denominator', 'bt_numerator', 'bt_denominator',
                'bt_legacy_contribution', 'bt_effective_numerator', 'bt_effective_denominator', 'nll_raw_numerator', 'nll_raw_denominator',
                'nll_legacy_contribution'):
        require(finite(group.get(key)), where + ': %s is not finite' % key)
    require(group['nll_raw_denominator'] > 0, where + ': nll_raw_denominator is not positive')
    if group['bt_defined']:
        for key in NULLABLE:
            require(finite(group.get(key)), where + ': %s is not finite although bt_defined' % key)
        require(group['bt_raw_denominator'] > 0 and group['bt_denominator'] > 0, where + ': bt denominators are not positive although bt_defined')
        require(abs(group['ranking_bt'] - group['bt_raw_numerator'] / group['bt_raw_denominator']) <= RATIO_TOLERANCE,
                where + ': ranking_bt_not_ratio (ranking_bt differs from bt_raw_numerator / bt_raw_denominator by more than %g)' % RATIO_TOLERANCE)
        require(abs(group['ranking_bt'] - group['bt_numerator'] / group['bt_denominator']) <= RATIO_TOLERANCE,
                where + ': ranking_bt_not_ratio (ranking_bt differs from bt_numerator / bt_denominator by more than %g)' % RATIO_TOLERANCE)
        require(0 <= group['pairwise_sign_accuracy'] <= 1 and 0 <= group['pairwise_sigmoid_brier'] <= 1, where + ': pairwise accuracy / Brier outside [0, 1]')
    else:
        for key in NULLABLE:
            require(group.get(key) is None, where + ': %s is not null although no pair is eligible' % key)
        for key in ('bt_raw_numerator', 'bt_legacy_contribution', 'bt_effective_numerator'):
            require(group[key] == 0, where + ': %s is not zero although no pair is eligible' % key)


def validate_batch(batch, where):
    require(isinstance(batch, dict), where + ': batch is not an object')
    require(type(batch.get('batch_index')) is int and 0 <= batch['batch_index'] < BATCHES, where + ': batch_index is not an integer in [0, 8)')
    ids = batch.get('record_ids')
    require(isinstance(ids, list) and len(ids) == ROWS_PER_BATCH and all(type(i) is int for i in ids), where + ': record_ids is not 256 integers')
    require(batch.get('draws') == DRAWS_PER_BATCH, where + ': draws is not 32')
    for key in ('legacy_values', 'effective_group_reconstructed', 'bridge_absolute_errors'):
        values = batch.get(key)
        require(isinstance(values, list) and len(values) == len(LEGACY_METRICS) and all(finite(v) for v in values), where + ': %s is not four finite numbers' % key)
    require(batch.get('bridge_tolerance') == BRIDGE_TOLERANCE, where + ': bridge_tolerance is not %g' % BRIDGE_TOLERANCE)
    require(type(batch.get('zero_pair_draws')) is int and batch['zero_pair_draws'] >= 0, where + ': zero_pair_draws is not a non-negative integer')


def validate_legacy_sibling(path, value, where):
    """legacy.json: the frozen_monitoring.v1 receipt published with groups.json, same binding, same four numbers within 1e-6."""
    legacy = read(path)
    require(isinstance(legacy, dict) and legacy.get('schema') == LEGACY_SCHEMA, '%s: schema is not %s' % (path, LEGACY_SCHEMA))
    require(legacy.get('final_tests_opened') is False, '%s: final_tests_opened is not false' % path)
    binding = legacy.get('binding')
    require(isinstance(binding, dict) and all(hex64(binding.get(k)) for k in LEGACY_BINDING), '%s: binding lacks 64-hex %s' % (path, ', '.join(LEGACY_BINDING)))
    expected = dict(checkpoint_sha256=value['checkpoint_sha256'], corpus_sha256=value['panel']['corpus_sha256'],
                    panel_sha256=value['panel']['panel_sha256'], executable_sha256=value['executable_sha256'])
    require({k: binding[k] for k in LEGACY_BINDING} == expected, '%s: legacy_binding_mismatch (binding differs from %s)' % (path, where))
    require(all(finite(legacy.get(m)) for m in LEGACY_METRICS), '%s: legacy receipt lacks finite %s' % (path, ', '.join(LEGACY_METRICS)))
    require(legacy.get('sampled_examples') == value['sampled_examples'], '%s: sampled_examples differs from %s' % (path, where))
    differences = {m: abs(legacy[m] - value['legacy_aggregate'][m]) for m in LEGACY_METRICS}
    disagreeing = sorted(m for m, d in differences.items() if d > BRIDGE_TOLERANCE)
    require(not disagreeing, '%s: legacy_disagree (%s differ from %s legacy_aggregate by more than %g: %s)'
            % (path, ', '.join(disagreeing), where, BRIDGE_TOLERANCE, {m: differences[m] for m in disagreeing}))
    return dict(path=str(path), sha256=sha(path), evaluated_step=legacy.get('evaluated_step'), checkpoint=legacy.get('checkpoint'),
                values={m: legacy[m] for m in LEGACY_METRICS}, abs_diff=differences, tolerance=BRIDGE_TOLERANCE, **{'within_1e-6': True})


def locate(path):
    """A sidecar directory (groups.json + legacy.json) or the groups.json itself; legacy.json must sit beside it."""
    path = Path(path)
    groups = path / GROUPS_FILE if path.is_dir() else path
    require(groups.is_file(), '%s: %s is missing' % (path, GROUPS_FILE))
    legacy = groups.parent / LEGACY_FILE
    require(legacy.is_file(), '%s: %s is missing beside %s (the emitter publishes both together)' % (path, LEGACY_FILE, GROUPS_FILE))
    return groups, legacy


def validate_input(path):
    groups_path, legacy_path = locate(path)
    where = str(groups_path)
    value = read(groups_path)
    require(isinstance(value, dict) and value.get('schema') == INPUT_SCHEMA, where + ': schema is not ' + INPUT_SCHEMA)
    require(value.get('final_tests_opened') is False, where + ': final_tests_opened is not false')
    require(value.get('arm') in ARMS + (PARENT,), where + ': arm is not one of %s' % (ARMS + (PARENT,),))
    for key in SHARED + ('checkpoint_sha256', 'executable_sha256', 'request_sha256'):
        require(hex64(value.get(key)), where + ': %s is not a 64-hex digest' % key)
    if value['arm'] == PARENT:
        require(value['checkpoint_sha256'] == value['parent_state_sha256'], where + ': parent sidecar checkpoint is not parent_state_sha256')
    panel = value.get('panel')
    require(isinstance(panel, dict) and panel.get('name') in PANELS, where + ': panel.name is not one of %s' % (PANELS,))
    for key in ('panel_sha256', 'file_sha256', 'corpus_sha256'):
        require(hex64(panel.get(key)), where + ': panel.%s is not a 64-hex digest' % key)
    require(value.get('sampled_examples') == SAMPLED_EXAMPLES, where + ': sampled_examples is not %d' % SAMPLED_EXAMPLES)
    require(value.get('draw_count') == DRAWS, where + ': draw_count is not %d' % DRAWS)
    require(type(value.get('external_879_numerical_bridge_qualified')) is bool, where + ': external_879_numerical_bridge_qualified is not a boolean')
    records = value.get('batch_records')
    require(isinstance(records, list) and len(records) == BATCHES and all(isinstance(b, list) and len(b) == ROWS_PER_BATCH and all(type(i) is int for i in b) for b in records),
            where + ': batch_records is not 8 lists of 256 integers')
    require(compact_sha256(records) == panel['panel_sha256'], where + ': panel_sha256 is not the sha256 of the compact json of batch_records')
    batches = value.get('batches')
    require(isinstance(batches, list) and len(batches) == BATCHES, where + ': batches is not a list of 8')
    for index, batch in enumerate(batches):
        validate_batch(batch, '%s batches[%d]' % (where, index))
        require(batch['batch_index'] == index, '%s batches[%d]: batch_index is %d' % (where, index, batch['batch_index']))
        require(batch['record_ids'] == records[index], '%s batches[%d]: record_ids differ from batch_records[%d]' % (where, index, index))
    groups = value.get('groups')
    require(isinstance(groups, list) and len(groups) == DRAWS, where + ': groups is not a list of %d draws' % DRAWS)
    seen, contexts, keys = {}, {}, {}
    for index, group in enumerate(groups):
        validate_group(group, '%s groups[%d]' % (where, index))
        require(group['draw_index'] == index, '%s groups[%d]: draw_index is %d (draws must be listed in forward order)' % (where, index, group['draw_index']))
        expected = records[group['batch_index']][RECORDS_PER_DRAW * group['draw_in_batch']:RECORDS_PER_DRAW * (group['draw_in_batch'] + 1)]
        require(group['record_ids'] == expected, '%s groups[%d]: record_ids are not batch_records[%d][%d:%d]'
                % (where, index, group['batch_index'], RECORDS_PER_DRAW * group['draw_in_batch'], RECORDS_PER_DRAW * (group['draw_in_batch'] + 1)))
        seen.setdefault(identity(group), []).append(index)
        keys[pair_key(group)] = group
        bound = (group['dataset'], group['cut'], group['family'], group['partition'])
        require(contexts.setdefault(group['context_index'], bound) == bound,
                '%s groups[%d]: context_index %d is bound to two different (dataset, cut, family, partition)' % (where, index, group['context_index']))
    for batch in batches:
        zero = sum(1 for g in groups if g['batch_index'] == batch['batch_index'] and not g['bt_defined'])
        require(zero == batch['zero_pair_draws'], '%s batches[%d]: zero_pair_draws %d differs from the %d draws without pairs'
                % (where, batch['batch_index'], batch['zero_pair_draws'], zero))
    require(value.get('distinct_contexts') == len(contexts), where + ': distinct_contexts %r differs from the %d contexts drawn' % (value.get('distinct_contexts'), len(contexts)))
    aggregate = value.get('legacy_aggregate')
    require(isinstance(aggregate, dict) and all(finite(aggregate.get(m)) for m in LEGACY_METRICS), where + ': legacy_aggregate lacks finite ' + ', '.join(LEGACY_METRICS))
    bridge = numerical_bridge(groups, batches, aggregate)
    require(bridge['within_1e-6'], '%s: bridge_failed (max per-batch abs diff %s, aggregate abs diff %s exceed %g)'
            % (where, bridge['max_batch_abs_diff'], bridge['aggregate_abs_diff'], BRIDGE_TOLERANCE))
    legacy = validate_legacy_sibling(legacy_path, value, where)
    repeated = {k: v for k, v in seen.items() if len(v) > 1}
    return dict(path=where, sha256=sha(groups_path), value=value, groups=keys, bridge=bridge, legacy=legacy,
                distinct_contexts=len(contexts), repeated_draws=sorted(v for v in repeated.values()),
                undefined_draws=[g['draw_index'] for g in groups if not g['bt_defined']])


def optional_binding(value, name, where):
    """A certificate may carry original_binding / emitter_binding objects; when present they must be well-formed and are cross-checked."""
    binding = value.get(name + '_binding')
    if binding is None:
        return None
    require(isinstance(binding, dict) and all(hex64(binding.get(k)) for k in LEGACY_BINDING), where + ': %s_binding lacks 64-hex ' % name + ', '.join(LEGACY_BINDING))
    require(binding.get('final_tests_opened') is not True, where + ': %s_binding final_tests_opened is true' % name)
    return {k: binding[k] for k in LEGACY_BINDING}


def validate_certificate(path, inputs, block):
    """Root's frozen native bridge certificate (parallel.original_monitor_numerical_bridge.v1): names exactly one input by the
    sha256 of its legacy.json (and its groups.json), carries per-metric original / emitter values with their absolute
    difference within tolerance <= 1e-6, passed true, training_updates 0, science_execution false and no improvement
    claim. original-parent-old.json beside it (the original binary's frozen_monitoring.v1 receipt) must hash to
    original_sha256, carry the certified original values and a binding on the input's checkpoint, corpus and panel whose
    executable carries the pre-registered frozen-monitor prefix; the emitter binding is the certified input's own
    (sidecar and legacy.json, already equal). Optional original_binding / emitter_binding objects in the certificate must
    agree. The sidecar's external_879_numerical_bridge_qualified flag is never rewritten; the certificate is recorded beside it."""
    path = Path(path)
    where = str(path)
    value = read(path)
    require(isinstance(value, dict) and value.get('schema') == CERTIFICATE_SCHEMA, where + ': schema is not ' + CERTIFICATE_SCHEMA)
    require(value.get('final_tests_opened') is not True, where + ': final_tests_opened is true')
    require(value.get('passed') is True, where + ': certificate_not_passed (passed is not true)')
    require(value.get('training_updates') == 0, where + ': training_updates is not 0')
    require(value.get('science_execution') is False, where + ': science_execution is not false')
    require(value.get('improvement_claim') is not True, where + ': improvement_claim is true (a bridge certificate claims no improvement)')
    require('parent_files_before_after' in value, where + ': parent_files_before_after missing')
    for key in ('original_sha256', 'emitter_legacy_sha256', 'emitter_groups_sha256'):
        require(hex64(value.get(key)), where + ': %s is not a 64-hex digest' % key)
    require(finite(value.get('tolerance')) and 0 < value['tolerance'] <= BRIDGE_TOLERANCE, where + ': tolerance is not in (0, %g]' % BRIDGE_TOLERANCE)
    deltas = value.get('deltas')
    require(isinstance(deltas, dict) and all(isinstance(deltas.get(m), dict) and all(finite(deltas[m].get(k)) for k in ('original', 'emitter', 'absolute_difference'))
                                             for m in LEGACY_METRICS), where + ': deltas lack finite original/emitter/absolute_difference for ' + ', '.join(LEGACY_METRICS))
    for m in LEGACY_METRICS:
        require(abs(abs(deltas[m]['original'] - deltas[m]['emitter']) - deltas[m]['absolute_difference']) <= 1e-12,
                where + ': deltas.%s.absolute_difference is not |original - emitter|' % m)
    exceeding = sorted(m for m in LEGACY_METRICS if deltas[m]['absolute_difference'] > value['tolerance'])
    require(not exceeding, where + ': certificate_not_within_tolerance (%s exceed %g)' % (exceeding, value['tolerance']))
    matches = [i for i in inputs if i['legacy']['sha256'] == value['emitter_legacy_sha256']]
    require(len(matches) == 1, '%s: certificate matches %d inputs by emitter_legacy_sha256; exactly one required' % (where, len(matches)))
    match = matches[0]
    require(value['emitter_groups_sha256'] == match['sha256'], where + ': emitter_groups_sha256 differs from the certified input\'s groups.json')
    disagreeing = sorted(m for m in LEGACY_METRICS if abs(deltas[m]['emitter'] - match['legacy']['values'][m]) > 1e-12)
    require(not disagreeing, where + ': deltas.%s.emitter differ from the certified legacy.json values' % disagreeing)
    emitter = dict(checkpoint_sha256=match['value']['checkpoint_sha256'], corpus_sha256=match['value']['panel']['corpus_sha256'],
                   panel_sha256=match['value']['panel']['panel_sha256'], executable_sha256=match['value']['executable_sha256'])
    expected = {k: emitter[k] for k in ('checkpoint_sha256', 'corpus_sha256', 'panel_sha256')}
    original_path = path.parent / ORIGINAL_RECEIPT_FILE
    require(original_path.is_file(), where + ': %s is missing beside the certificate' % ORIGINAL_RECEIPT_FILE)
    require(sha(original_path) == value['original_sha256'], where + ': %s does not hash to original_sha256' % ORIGINAL_RECEIPT_FILE)
    original_receipt = read(original_path)
    require(isinstance(original_receipt, dict) and original_receipt.get('schema') == LEGACY_SCHEMA and original_receipt.get('final_tests_opened') is False,
            where + ': %s is not a %s receipt with final_tests_opened false' % (ORIGINAL_RECEIPT_FILE, LEGACY_SCHEMA))
    original = original_receipt.get('binding')
    require(isinstance(original, dict) and all(hex64(original.get(k)) for k in LEGACY_BINDING),
            where + ': %s binding lacks 64-hex ' % ORIGINAL_RECEIPT_FILE + ', '.join(LEGACY_BINDING))
    original = {k: original[k] for k in LEGACY_BINDING}
    prefix = readout_panel(block, 'old_frozen').get('monitor_executable_sha256_prefix')
    require(isinstance(prefix, str) and prefix and original['executable_sha256'].startswith(prefix),
            where + ': %s executable_sha256 does not start with the pre-registered frozen monitor prefix %r' % (ORIGINAL_RECEIPT_FILE, prefix))
    require({k: original[k] for k in expected} == expected, where + ': %s binding checkpoint/corpus/panel differ from the certified input (%s)' % (ORIGINAL_RECEIPT_FILE, match['path']))
    mismatching = sorted(m for m in LEGACY_METRICS if not finite(original_receipt.get(m)) or abs(original_receipt[m] - deltas[m]['original']) > 1e-12)
    require(not mismatching, where + ': %s values %s differ from deltas.<metric>.original' % (ORIGINAL_RECEIPT_FILE, mismatching))
    for name, binding, known in (('original', optional_binding(value, 'original', where), original), ('emitter', optional_binding(value, 'emitter', where), emitter)):
        require(binding is None or binding == known, where + ': %s_binding differs from the %s' % (name, ORIGINAL_RECEIPT_FILE if name == 'original' else 'certified input'))
    identities = {name: sha(path.parent / name) for name in ('original-identity.json', 'emitter-identity.json') if (path.parent / name).is_file()}
    return dict(path=where, sha256=sha(path), certified_input=match['path'], arm=match['value']['arm'], panel=match['value']['panel']['name'],
                original_receipt=dict(path=str(original_path), sha256=value['original_sha256'], binding=original),
                emitter_binding=emitter, deltas={m: dict(deltas[m]) for m in LEGACY_METRICS}, tolerance=value['tolerance'],
                passed=True, training_updates=0, science_execution=False, improvement_claim=value.get('improvement_claim'),
                attempts=value.get('attempts'), identity_files=identities)


def assemble(inputs):
    """One sidecar per (arm, panel); shared digests identical; each panel carries both arms on one binding with identical draws."""
    require(inputs, 'no inputs')
    table = {}
    for item in inputs:
        key = (item['value']['arm'], item['value']['panel']['name'])
        require(key not in table, 'duplicate input for arm %s on panel %s: %s and %s' % (key[0], key[1], table.get(key, {}).get('path'), item['path']))
        table[key] = item
    first = inputs[0]
    for item in inputs[1:]:
        for key in SHARED:
            require(item['value'][key] == first['value'][key], '%s: %s %s... differs from %s... in %s'
                    % (item['path'], key, item['value'][key][:12], first['value'][key][:12], first['path']))
    checkpoints = {}
    for (arm, name), item in sorted(table.items()):
        checkpoint = item['value']['checkpoint_sha256']
        require(checkpoints.setdefault(arm, checkpoint) == checkpoint, 'arm %s carries two checkpoints across panels (%s... and %s...)'
                % (arm, checkpoints[arm][:12], checkpoint[:12]))
    require(len({checkpoints[a] for a in ARMS if a in checkpoints}) == len([a for a in ARMS if a in checkpoints]),
            'arms A and B share one checkpoint %s...' % checkpoints.get('A', '')[:12])
    for arm in ARMS:
        require(checkpoints.get(arm) != first['value']['parent_state_sha256'], 'arm %s carries the parent checkpoint' % arm)
    panels = {}
    for name in PANELS:
        present = {arm: table[(arm, name)] for arm in ARMS if (arm, name) in table}
        parent = table.get((PARENT, name))
        if not present:
            require(parent is None, 'panel %s carries a parent sidecar but no arm' % name)
            continue
        missing = [arm for arm in ARMS if arm not in present]
        require(not missing, 'panel %s lacks arm %s' % (name, ', '.join(missing)))
        a, b = present['A'], present['B']
        require(panel_binding(a) == panel_binding(b), 'panel %s binding (panel, corpus, executable) differs between %s and %s' % (name, a['path'], b['path']))
        require(a['value']['batch_records'] == b['value']['batch_records'], 'panel %s: batch_records differ between %s and %s' % (name, a['path'], b['path']))
        only_a, only_b = set(a['groups']) - set(b['groups']), set(b['groups']) - set(a['groups'])
        require(not only_a and not only_b, 'panel %s: identity_mismatch (%d draws only in A, %d only in B; pairing = draw_index plus %s with exactly %d record ids)'
                % (name, len(only_a), len(only_b), ', '.join(IDENTITY), RECORDS_PER_DRAW))
        if parent is not None:
            require(panel_binding(parent) == panel_binding(a) and parent['value']['batch_records'] == a['value']['batch_records'],
                    'panel %s: parent sidecar %s is not on the arms\' binding and batch_records' % (name, parent['path']))
        panels[name] = dict(A=a, B=b, parent=parent, binding=panel_binding(a))
    require(len(panels) < 2 or panels['reserved']['binding']['panel_sha256'] != panels['old_frozen']['binding']['panel_sha256'],
            'reserved and old_frozen inputs name one panel_sha256')
    return dict(shared={k: first['value'][k] for k in SHARED}, checkpoints=checkpoints, panels=panels)


# ---------------------------------------------------------------- statistic

def panel_deltas(a, b):
    """Per-context deltas grouped by family; a metric undefined on any paired draw is undefined for the panel."""
    by_context, undefined = {}, {m: [] for m in METRICS}
    for key in sorted(a['groups'], key=lambda k: k[0]):
        group_a, group_b = a['groups'][key], b['groups'][key]
        context = by_context.setdefault(key[1], dict(family=key[4], draws=0, deltas={m: [] for m in METRICS}))
        context['draws'] += 1
        for m in METRICS:
            if group_a[m] is None or group_b[m] is None:
                undefined[m].append(key[0])
            else:
                context['deltas'][m].append(group_b[m] - group_a[m])
    defined = [m for m in METRICS if not undefined[m]]
    families = {}
    for index in sorted(by_context):
        context = by_context[index]
        family = families.setdefault(context['family'], dict(contexts=[], deltas={m: [] for m in defined}))
        family['contexts'].append(index)
        for m in defined:
            family['deltas'][m].append(statistics.fmean(context['deltas'][m]))
    repeated = sum(1 for c in by_context.values() if c['draws'] > 1)
    return families, dict(contexts=len(by_context), draws=sum(c['draws'] for c in by_context.values()), contexts_drawn_more_than_once=repeated,
                          undefined={m: sorted(v) for m, v in undefined.items() if v})


def stratified_bootstrap(families, metrics, resamples, seed, interval):
    """Percentile interval of the equal-family mean; contexts resampled with replacement within each family, one draw for all metrics."""
    names = sorted(families)
    require(names and all(families[f]['contexts'] for f in names), 'stratified bootstrap requires at least one context per family')
    rows = [[tuple(families[f]['deltas'][m][i] for m in metrics) for i in range(len(families[f]['contexts']))] for f in names]
    rng = random.Random(seed)
    stats = [[] for _ in metrics]
    for _ in range(resamples):
        totals = [0.0] * len(metrics)
        for values in rows:
            draw = rng.choices(values, k=len(values))
            for j in range(len(metrics)):
                totals[j] += sum(d[j] for d in draw) / len(draw)
        for j in range(len(metrics)):
            stats[j].append(totals[j] / len(rows))
    alpha = 1 - interval
    result = {}
    for j, m in enumerate(metrics):
        ordered = sorted(stats[j])
        last = len(ordered) - 1
        result[m] = [ordered[min(last, int(len(ordered) * alpha / 2))], ordered[min(last, int(len(ordered) * (1 - alpha / 2)))]]
    return result


def arm_summary(item):
    value = item['value']
    return dict(path=item['path'], sha256=item['sha256'], checkpoint_sha256=value['checkpoint_sha256'], executable_sha256=value['executable_sha256'],
                request_sha256=value['request_sha256'], sampled_examples=value['sampled_examples'], draws=len(item['groups']),
                distinct_contexts=item['distinct_contexts'], undefined_draws=item['undefined_draws'], legacy_aggregate=dict(value['legacy_aggregate']),
                legacy_json=item['legacy'], external_879_numerical_bridge_qualified=value['external_879_numerical_bridge_qualified'])


def panel_readout(name, pair, block, policy):
    families, counts = panel_deltas(pair['A'], pair['B'])
    entry = readout_panel(block, name)
    per_family = {f: len(families[f]['contexts']) for f in sorted(families)}
    if name == 'reserved':
        require(type(entry.get('families')) is int and type(entry.get('contexts_per_family')) is int,
                'preregistration block lacks integer families / contexts_per_family for the reserved panel')
        require(len(families) == entry['families'], 'reserved panel has %d families; %d pre-registered' % (len(families), entry['families']))
        wrong = {f: n for f, n in per_family.items() if n != entry['contexts_per_family']}
        require(not wrong, 'reserved panel family context counts differ from the pre-registered %d: %s' % (entry['contexts_per_family'], wrong))
    if type(entry.get('draws')) is int:
        require(entry['draws'] == pair['A']['value']['sampled_examples'], '%s panel presents %d records; %d pre-registered as draws'
                % (name, pair['A']['value']['sampled_examples'], entry['draws']))
    bootstrapped = [m for m in BOOTSTRAPPED if m not in counts['undefined']]
    intervals = stratified_bootstrap(families, bootstrapped, policy['resamples'], policy['seed'], policy['interval']) if bootstrapped else {}
    metrics = {}
    for m in METRICS:
        role = 'primary' if m == PRIMARY else 'secondary' if m == SECONDARY else 'reported'
        if m in counts['undefined']:
            metrics[m] = dict(role=role, defined=False, undefined_draws=counts['undefined'][m],
                              equal_family_mean=unavailable('undefined_ranking_evidence: %d paired draws with no eligible pair' % len(counts['undefined'][m])),
                              rule=UNDEFINED_RULE)
            continue
        family_means = {f: statistics.fmean(families[f]['deltas'][m]) for f in sorted(families)}
        context_deltas = [d for f in sorted(families) for d in families[f]['deltas'][m]]
        with_interval = m in intervals
        metrics[m] = dict(role=role, defined=True, family_means=family_means, equal_family_mean=statistics.fmean(family_means.values()),
                          pooled_context_mean=statistics.fmean(context_deltas), n_contexts=len(context_deltas),
                          fraction_negative_contexts=sum(d < 0 for d in context_deltas) / len(context_deltas),
                          ci95=intervals[m] if with_interval else unavailable('interval pre-registered for %s only' % ' and '.join(BOOTSTRAPPED)),
                          interval=policy['interval'] if with_interval else None, resamples=policy['resamples'] if with_interval else None,
                          seed=policy['seed'] if with_interval else None)
    preregistered = {k: entry.get(k) for k in ('families', 'contexts_per_family', 'distinct_contexts', 'draws') if k in entry}
    parent = pair.get('parent')
    reference = unavailable('no parent sidecar supplied for this panel')
    if parent is not None:
        aggregate = parent['value']['legacy_aggregate']
        reference = dict(arm_summary(parent), arm_minus_parent={arm: {m: pair[arm]['value']['legacy_aggregate'][m] - aggregate[m] for m in LEGACY_METRICS}
                                                                  for arm in ARMS}, bridge=parent['bridge'])
    return dict(role=entry.get('role'), binding=pair['binding'], n_draws=counts['draws'], n_families=len(families), contexts_per_family=per_family,
                distinct_contexts=counts['contexts'], contexts_drawn_more_than_once=counts['contexts_drawn_more_than_once'],
                repeated_draw_positions={arm: pair[arm]['repeated_draws'] for arm in ARMS}, preregistered=preregistered,
                distinct_contexts_match_preregistration=(counts['contexts'] == entry['distinct_contexts']) if type(entry.get('distinct_contexts')) is int else None,
                arms={arm: arm_summary(pair[arm]) for arm in ARMS}, bridge={arm: pair[arm]['bridge'] for arm in ARMS},
                parent_reference=reference, metrics=metrics)


# ---------------------------------------------------------------- verdict

def decide(panels, policy):
    """repair_signal iff reserved delta <= reserved_delta_max with the interval entirely below 0 and old-panel delta <= old_panel_tolerance."""
    inputs = dict(reserved_delta_max=policy['reserved_delta_max'], old_panel_tolerance=policy['old_panel_tolerance'], interval=policy['interval'],
                  reserved_delta=None, reserved_ci95=None, reserved_clears_threshold=None, old_panel_delta=None, old_panel_within_tolerance=None)
    reserved, old = panels.get('reserved'), panels.get('old_frozen')
    if reserved is None:
        return 'inconclusive', 'reserved_panel_missing', 'no reserved-panel read-out for both arms; the primary statistic is undefined', inputs
    r = reserved['metrics'][PRIMARY]
    if not r['defined']:
        return 'inconclusive', 'primary_statistic_undefined', 'reserved ranking_bt is undefined: draws %s have no eligible pair' % r['undefined_draws'], inputs
    inputs.update(reserved_delta=r['equal_family_mean'], reserved_ci95=r['ci95'],
                  reserved_clears_threshold=r['equal_family_mean'] <= policy['reserved_delta_max'] and r['ci95'][1] < 0)
    reserved_text = 'reserved ranking_bt delta %.6f with %g%% interval [%.6f, %.6f]' % (r['equal_family_mean'], 100 * policy['interval'], *r['ci95'])
    if old is None:
        return 'inconclusive', 'old_panel_missing', reserved_text + '; no old-frozen-panel read-out for both arms, the tolerance clause is undefined', inputs
    o = old['metrics'][PRIMARY]
    if not o['defined']:
        return 'inconclusive', 'tolerance_statistic_undefined', reserved_text + '; old-panel ranking_bt is undefined: draws %s have no eligible pair' % o['undefined_draws'], inputs
    inputs.update(old_panel_delta=o['equal_family_mean'], old_panel_within_tolerance=o['equal_family_mean'] <= policy['old_panel_tolerance'])
    old_text = 'old-panel ranking_bt delta %.6f against tolerance %+g' % (o['equal_family_mean'], policy['old_panel_tolerance'])
    if inputs['reserved_clears_threshold'] and inputs['old_panel_within_tolerance']:
        return 'repair_signal', 'reserved_delta_clears_threshold_and_old_panel_within_tolerance', '%s <= %g and entirely below 0; %s' % (
            reserved_text, policy['reserved_delta_max'], old_text), inputs
    if inputs['reserved_clears_threshold']:
        return 'inconclusive', 'old_panel_worse_than_tolerance', '%s clears the threshold but %s' % (reserved_text, old_text), inputs
    why = []
    if r['equal_family_mean'] > policy['reserved_delta_max']:
        why.append('delta above reserved_delta_max %g' % policy['reserved_delta_max'])
    if r['ci95'][1] >= 0:
        why.append('interval reaches 0 (never evidence of not causal)')
    if not inputs['old_panel_within_tolerance']:
        why.append('and the old panel is worse than tolerance')
    return 'inconclusive', 'no_demonstrated_repair', '%s: %s; %s' % (reserved_text, '; '.join(why), old_text), inputs


def evaluate(inputs, registration, certificate_paths=()):
    block = registration['block']
    policy = thresholds(block)
    rules = lookup(block, 'falsifiers.F9.verdicts')
    require(isinstance(rules, dict) and set(rules) == set(VERDICTS), 'preregistration block falsifiers.F9.verdicts must name exactly %s' % (VERDICTS,))
    assembled = assemble(inputs)
    prefixes = preregistered_prefixes(block, assembled['shared'], assembled['panels'])
    panels = {name: panel_readout(name, pair, block, policy) for name, pair in assembled['panels'].items()}
    verdict, reason, detail, verdict_inputs = decide(panels, policy)
    require(verdict in VERDICTS, 'unknown verdict')
    ordered = sorted(inputs, key=lambda item: (item['value']['panel']['name'], item['value']['arm']))
    qualified = all(item['value']['external_879_numerical_bridge_qualified'] for item in inputs)
    certificates = sorted((validate_certificate(path, inputs, block) for path in certificate_paths), key=lambda c: (c['panel'], c['arm']))
    certified = {c['certified_input'] for c in certificates}
    require(len(certified) == len(certificates), 'two bridge certificates name one input')
    comparability = dict(rule=LEGACY_COMPARABILITY, sidecar_flag_all_true=qualified, certificates=certificates,
                         certified_inputs=sorted(certified), uncertified_inputs=[i['path'] for i in ordered if i['path'] not in certified],
                         all_inputs_certified=bool(inputs) and all(i['path'] in certified for i in inputs))
    return dict(schema=SCHEMA, preregistration_sha256=registration['sha256'],
                # The --preregistration-sha256 pin is a refusal gate, not receipt content: a pinned retry must reproduce identical bytes.
                preregistration={k: registration.get(k) for k in ('path', 'document_sha256', 'block_bytes_sha256', 'reader')},
                **assembled['shared'], arms={arm: dict(checkpoint_sha256=assembled['checkpoints'].get(arm)) for arm in ARMS},
                thresholds=policy, preregistered_prefixes=prefixes,
                inputs=[dict(path=item['path'], sha256=item['sha256'], legacy_path=item['legacy']['path'], legacy_sha256=item['legacy']['sha256'],
                             arm=item['value']['arm'], panel=item['value']['panel']['name'], checkpoint_sha256=item['value']['checkpoint_sha256'],
                             executable_sha256=item['value']['executable_sha256'], sampled_examples=item['value']['sampled_examples'],
                             draws=len(item['groups']), distinct_contexts=item['distinct_contexts']) for item in ordered],
                panels=panels, verdict=verdict, reason=reason, reason_detail=detail, verdict_inputs=verdict_inputs, verdict_rules=rules,
                external_879_numerical_bridge_qualified=qualified, legacy_comparability=comparability, statistic=STATISTIC,
                bridge_rule=BRIDGE_RULE, undefined_rule=UNDEFINED_RULE, interpretation=INTERPRETATION, hold_rule=HOLD_RULE, final_tests_opened=False)


VALIDATION_SCHEMA = 'parallel.flywheel_f9_validation.v1'


def validation(inputs, registration, certificate_paths=()):
    """Every input and certificate check of the read-out, without assembling arms or judging: the receipt for a lone
    parent sidecar (or any subset) so a real emitter output is authenticated before the arms exist."""
    block = registration['block']
    certificates = sorted((validate_certificate(path, inputs, block) for path in certificate_paths), key=lambda c: (c['panel'], c['arm']))
    certified = {c['certified_input'] for c in certificates}
    require(len(certified) == len(certificates), 'two bridge certificates name one input')
    rows = []
    for item in sorted(inputs, key=lambda i: (i['value']['panel']['name'], i['value']['arm'])):
        entry = readout_panel(block, item['value']['panel']['name'])
        prefixes = [dict(field=field, expected_prefix=entry.get(key), observed=observed, matches=observed.startswith(entry[key]))
                    for field, key, observed in (('panel_sha256', 'panel_sha256_prefix', item['value']['panel']['panel_sha256']),
                                                 ('corpus_sha256', 'corpus_sha256_prefix', item['value']['panel']['corpus_sha256']),
                                                 ('executable_sha256', 'monitor_executable_sha256_prefix', item['value']['executable_sha256']))
                    if isinstance(entry.get(key), str) and entry.get(key)]
        rows.append(dict(arm_summary(item), arm=item['value']['arm'], panel=item['value']['panel']['name'], legacy_aggregate=dict(item['value']['legacy_aggregate']),
                         bridge=dict(max_batch_abs_diff=item['bridge']['max_batch_abs_diff'], aggregate_abs_diff=item['bridge']['aggregate_abs_diff'],
                                     within_1e_6=item['bridge']['within_1e-6'], raw_bt_ratio_mean_of_batches=item['bridge']['raw_bt_ratio_mean_of_batches'],
                                     mean_of_defined_draw_ranking_bt=item['bridge']['mean_of_defined_draw_ranking_bt']),
                         repeated_draw_positions=item['repeated_draws'], preregistered_prefixes=prefixes,
                         distinct_contexts_match_preregistration=(item['distinct_contexts'] == entry['distinct_contexts'])
                         if type(entry.get('distinct_contexts')) is int else None, certified=item['path'] in certified))
    return dict(schema=VALIDATION_SCHEMA, preregistration_sha256=registration['sha256'], inputs=rows, certificates=certificates,
                bridge_rule=BRIDGE_RULE, undefined_rule=UNDEFINED_RULE, legacy_comparability=LEGACY_COMPARABILITY,
                note='validation only: no arms assembled, no statistic, no verdict', final_tests_opened=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preregistration', type=Path, required=True, help='PREREGISTRATION.md carrying the fenced json block')
    parser.add_argument('--preregistration-sha256', help='refuse unless the block digest (canonical json or block text) equals this')
    parser.add_argument('--inputs', type=Path, nargs='+', required=True,
                        help='parallel.panel_group_metrics.v2 sidecar directories (groups.json + legacy.json) or groups.json paths, one per (arm, panel)')
    parser.add_argument('--bridge-certificate', type=Path, nargs='*', default=[],
                        help='authenticated %s receipts (one per certified legacy.json); recorded beside the sidecar flag, never substituted for it' % CERTIFICATE_SCHEMA)
    parser.add_argument('--output', type=Path, help='immutable %s receipt (required unless --dry-run)' % SCHEMA)
    parser.add_argument('--dry-run', action='store_true', help='print the receipt without writing it')
    parser.add_argument('--validate-only', action='store_true',
                        help='authenticate the inputs and certificates only (any subset, e.g. a lone parent sidecar): %s, no verdict' % VALIDATION_SCHEMA)
    args = parser.parse_args(argv)
    require(args.output is not None or args.dry_run, '--output is required unless --dry-run')
    registration = preregistration(args.preregistration, args.preregistration_sha256)
    inputs = [validate_input(path) for path in args.inputs]
    receipt = validation(inputs, registration, args.bridge_certificate) if args.validate_only else evaluate(inputs, registration, args.bridge_certificate)
    if args.dry_run:
        print(json.dumps(receipt, sort_keys=True))
        return receipt
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_sha = promotion_gate.immutable(args.output, receipt)
    if args.validate_only:
        print(json.dumps(dict(receipt=str(args.output), sha256=receipt_sha, schema=VALIDATION_SCHEMA, inputs=[(r['panel'], r['arm']) for r in receipt['inputs']],
                              certified=[c['certified_input'] for c in receipt['certificates']])))
        return receipt
    print(json.dumps(dict(receipt=str(args.output), sha256=receipt_sha, verdict=receipt['verdict'], reason=receipt['reason'],
                          reserved_delta=receipt['verdict_inputs']['reserved_delta'], reserved_ci95=receipt['verdict_inputs']['reserved_ci95'],
                          old_panel_delta=receipt['verdict_inputs']['old_panel_delta'], panels=sorted(receipt['panels']))))
    return receipt


if __name__ == '__main__':
    main()
