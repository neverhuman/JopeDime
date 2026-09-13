#!/usr/bin/env python3
"""F9 read-out: the two-arm replay-concentration contrast under root's frozen protocol d8d57069.

Reads one parallel.panel_group_metrics.v1 file per (arm, panel) - arm A is the
deployed fresh_balanced sampler, arm B the d8d57069 quota schedule, each trained
one window from the same fixed parent under one shared seed and read out on the
reserved panel (primary) and the old frozen panel (tolerance) - and writes an
immutable parallel.flywheel_f9_readout.v1 receipt whose verdict is one of
{repair_signal, inconclusive}. Every threshold, the resample count and the
interval come from the fenced json block of PREREGISTRATION.md
(falsifiers.F9.thresholds; the reserved geometry from falsifiers.F9.readout_panels);
no constant in this file is a threshold. --preregistration-sha256 pins the block.

Refusals are explicit RuntimeErrors: schema or digest shape, more than one file
per (arm, panel), a panel missing an arm, arms sharing one checkpoint, a group
identity (context index, dataset, cut, family, partition and the exact eight
record ids) not matching across arms, ranking_bt differing from its retained
bt_numerator / bt_denominator, the legacy batch-weighted aggregate not
reconstructing from sum(numerator) / sum(denominator) within 1e-6 (the
numerical bridge; a supplied parallel.frozen_monitoring.v1 receipt must agree
within 1e-6 as well), a reserved family with the wrong context count, or a
digest that does not start with its pre-registered prefix (the old-panel
monitor executable is reported only, since F6 may rebind it).

Statistic: per-group delta = metric(B) - metric(A); per-context delta = mean over
that context's paired groups; family mean = mean over the family's contexts; the
primary statistic is the equal-family mean (mean of family means). Interval:
percentile interval of the equal-family mean over stratified resamples that draw
contexts with replacement within each family (one draw serves every metric).
ranking_bt is primary and sequence_nll secondary; ranking_listwise and
runtime_huber are reported without an interval. The verdict never authorizes a
live block under B and never releases the 66000 hold.
"""
import argparse
import json
import math
from pathlib import Path
import random
import re
import statistics

from flywheel_verdict import lookup, preregistration
import promotion_gate
from resume_expired_cycle import read, require, sha

SCHEMA = 'parallel.flywheel_f9_readout.v1'
INPUT_SCHEMA = 'parallel.panel_group_metrics.v1'
LEGACY_SCHEMA = 'parallel.frozen_monitoring.v1'
ARMS = ('A', 'B')
PANELS = ('reserved', 'old_frozen')
METRICS = ('sequence_nll', 'ranking_bt', 'ranking_listwise', 'runtime_huber')
PRIMARY, SECONDARY = 'ranking_bt', 'sequence_nll'
BOOTSTRAPPED = (PRIMARY, SECONDARY)
IDENTITY = ('context_index', 'dataset', 'cut', 'family', 'partition', 'record_ids')
RECORDS_PER_GROUP = 8
SHARED = ('protocol_sha256', 'parent_state_sha256', 'parent_model_sha256', 'parent_optimizer_sha256', 'corpus_sha256')
PANEL_BINDING = ('panel_sha256', 'corpus_sha256', 'executable_sha256')
LEGACY_BINDING = ('checkpoint_sha256', 'corpus_sha256', 'panel_sha256', 'executable_sha256')
BRIDGE_TOLERANCE = 1e-6
RATIO_TOLERANCE = 1e-9
DEFAULT_SEED = 20260912
THRESHOLDS = 'falsifiers.F9.thresholds'
THRESHOLD_KEYS = ('reserved_delta_max', 'old_panel_tolerance', 'interval', 'resamples')
READOUT_PANELS = 'falsifiers.F9.readout_panels'
VERDICTS = ('repair_signal', 'inconclusive')
INTERPRETATION = 'engineering signal consistent with reduced concentration, not proof of sole cause'
HOLD_RULE = 'cannot authorize the next live block under B and cannot release the 66000 hold'
STATISTIC = ('per-group delta = metric(B) - metric(A); per-context delta = mean over the context\'s paired groups; '
             'family mean = mean over the family\'s contexts; equal-family mean = mean of family means; interval = percentile '
             'interval of the equal-family mean over stratified resamples drawing contexts with replacement within each family '
             '(one draw per resample serves every metric); ranking_bt primary, sequence_nll secondary')
BRIDGE_RULE = 'sum(bt_numerator) / sum(bt_denominator) over all groups of the file; never the mean of per-group means'
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


def panel_binding(item):
    panel = item['value']['panel']
    return dict(panel_sha256=panel['panel_sha256'], corpus_sha256=panel['corpus_sha256'], executable_sha256=item['value']['executable_sha256'])


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

def numerical_bridge(groups, legacy):
    """The legacy batch-weighted aggregate reconstructed from the retained per-group numerators and denominators."""
    reconstructed = sum(g['bt_numerator'] for g in groups) / sum(g['bt_denominator'] for g in groups)
    difference = abs(reconstructed - legacy)
    return {'reconstructed': reconstructed, 'legacy': legacy, 'abs_diff': difference, 'tolerance': BRIDGE_TOLERANCE,
            'within_1e-6': difference <= BRIDGE_TOLERANCE,
            'mean_of_group_means': statistics.fmean(g['ranking_bt'] for g in groups), 'rule': BRIDGE_RULE}


def validate_group(group, where):
    require(isinstance(group, dict), where + ': group is not an object')
    require(type(group.get('context_index')) is int and group['context_index'] >= 0, where + ': context_index is not a non-negative integer')
    for key in ('dataset', 'family', 'partition'):
        require(isinstance(group.get(key), str) and group[key], where + ': %s is not a non-empty string' % key)
    require(type(group.get('cut')) in (str, int), where + ': cut is neither a string nor an integer')
    ids = group.get('record_ids')
    require(isinstance(ids, list) and len(ids) == RECORDS_PER_GROUP and all(type(i) is int for i in ids) and len(set(ids)) == RECORDS_PER_GROUP,
            where + ': record_ids is not exactly %d distinct integers' % RECORDS_PER_GROUP)
    for key in METRICS + ('bt_numerator', 'bt_denominator'):
        require(finite(group.get(key)), where + ': %s is not finite' % key)
    require(group['bt_denominator'] > 0, where + ': bt_denominator is not positive')
    require(abs(group['ranking_bt'] - group['bt_numerator'] / group['bt_denominator']) <= RATIO_TOLERANCE,
            where + ': ranking_bt_not_ratio (ranking_bt differs from bt_numerator / bt_denominator by more than %g)' % RATIO_TOLERANCE)
    require(type(group.get('n_pairs')) is int and group['n_pairs'] >= 0, where + ': n_pairs is not a non-negative integer')


def validate_input(path):
    path = Path(path)
    where = str(path)
    value = read(path)
    require(isinstance(value, dict) and value.get('schema') == INPUT_SCHEMA, where + ': schema is not ' + INPUT_SCHEMA)
    require(value.get('final_tests_opened') is False, where + ': final_tests_opened is not false')
    require(value.get('arm') in ARMS, where + ': arm is not one of %s' % (ARMS,))
    for key in SHARED + ('checkpoint_sha256', 'executable_sha256'):
        require(hex64(value.get(key)), where + ': %s is not a 64-hex digest' % key)
    panel = value.get('panel')
    require(isinstance(panel, dict) and panel.get('name') in PANELS, where + ': panel.name is not one of %s' % (PANELS,))
    for key in ('panel_sha256', 'corpus_sha256'):
        require(hex64(panel.get(key)), where + ': panel.%s is not a 64-hex digest' % key)
    require(type(value.get('sampled_examples')) is int and value['sampled_examples'] >= 0, where + ': sampled_examples is not a non-negative integer')
    groups = value.get('groups')
    require(isinstance(groups, list) and groups, where + ': groups is empty or not a list')
    seen, contexts = {}, {}
    for index, group in enumerate(groups):
        validate_group(group, '%s groups[%d]' % (where, index))
        key = identity(group)
        require(key not in seen, '%s groups[%d]: duplicate group identity (same as groups[%s])' % (where, index, seen.get(key)))
        seen[key] = index
        bound = (group['dataset'], group['cut'], group['family'], group['partition'])
        require(contexts.setdefault(group['context_index'], bound) == bound,
                '%s groups[%d]: context_index %d is bound to two different (dataset, cut, family, partition)' % (where, index, group['context_index']))
    aggregate = value.get('legacy_aggregate')
    require(isinstance(aggregate, dict) and all(finite(aggregate.get(m)) for m in METRICS),
            where + ': legacy_aggregate lacks finite ' + ', '.join(METRICS))
    bridge = numerical_bridge(groups, aggregate['ranking_bt'])
    require(bridge['within_1e-6'], '%s: bridge_failed (sum(bt_numerator)/sum(bt_denominator) = %.12g differs from legacy_aggregate.ranking_bt = %.12g by %.3g > %g)'
            % (where, bridge['reconstructed'], aggregate['ranking_bt'], bridge['abs_diff'], BRIDGE_TOLERANCE))
    return dict(path=str(path), sha256=sha(path), value=value, groups={identity(g): g for g in groups}, bridge=bridge)


def validate_legacy(path, inputs):
    """A parallel.frozen_monitoring.v1 receipt must name exactly one input by (checkpoint, panel) and agree with its aggregate within 1e-6."""
    path = Path(path)
    where = str(path)
    value = read(path)
    require(isinstance(value, dict) and value.get('schema') == LEGACY_SCHEMA, where + ': schema is not ' + LEGACY_SCHEMA)
    require(value.get('final_tests_opened') is False, where + ': final_tests_opened is not false')
    binding = value.get('binding')
    require(isinstance(binding, dict) and all(hex64(binding.get(k)) for k in LEGACY_BINDING), where + ': binding lacks 64-hex ' + ', '.join(LEGACY_BINDING))
    require(all(finite(value.get(m)) for m in METRICS), where + ': legacy receipt lacks finite ' + ', '.join(METRICS))
    matches = [i for i in inputs if i['value']['checkpoint_sha256'] == binding['checkpoint_sha256']
               and i['value']['panel']['panel_sha256'] == binding['panel_sha256']]
    require(len(matches) == 1, '%s: legacy receipt matches %d inputs by (checkpoint_sha256, panel_sha256); exactly one required' % (where, len(matches)))
    match = matches[0]
    bound = panel_binding(match)
    require(binding['corpus_sha256'] == bound['corpus_sha256'] and binding['executable_sha256'] == bound['executable_sha256'],
            '%s: legacy_binding_mismatch (corpus/executable differ from %s)' % (where, match['path']))
    differences = {m: abs(value[m] - match['value']['legacy_aggregate'][m]) for m in METRICS}
    differences['reconstructed_ranking_bt'] = abs(value['ranking_bt'] - match['bridge']['reconstructed'])
    disagreeing = sorted(k for k, d in differences.items() if d > BRIDGE_TOLERANCE)
    require(not disagreeing, '%s: legacy_disagree (%s differ from %s by more than %g: %s)'
            % (where, ', '.join(disagreeing), match['path'], BRIDGE_TOLERANCE, {k: differences[k] for k in disagreeing}))
    return {'path': str(path), 'sha256': sha(path), 'arm': match['value']['arm'], 'panel': match['value']['panel']['name'],
            'checkpoint_sha256': binding['checkpoint_sha256'], 'matched_input': match['path'], 'abs_diff': differences,
            'tolerance': BRIDGE_TOLERANCE, 'within_1e-6': True}


def assemble(inputs):
    """One file per (arm, panel); shared digests identical; each panel carries both arms on one binding with equal group identities."""
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
    require(len(checkpoints) < 2 or checkpoints['A'] != checkpoints['B'], 'arms A and B share one checkpoint %s...' % checkpoints['A'][:12])
    panels = {}
    for name in PANELS:
        present = {arm: table[(arm, name)] for arm in ARMS if (arm, name) in table}
        if not present:
            continue
        missing = [arm for arm in ARMS if arm not in present]
        require(not missing, 'panel %s lacks arm %s' % (name, ', '.join(missing)))
        a, b = present['A'], present['B']
        require(panel_binding(a) == panel_binding(b), 'panel %s binding (panel, corpus, executable) differs between %s and %s' % (name, a['path'], b['path']))
        only_a, only_b = set(a['groups']) - set(b['groups']), set(b['groups']) - set(a['groups'])
        require(not only_a and not only_b, 'panel %s: identity_mismatch (%d groups only in A, %d only in B; identity = %s with exactly %d record ids)'
                % (name, len(only_a), len(only_b), ', '.join(IDENTITY), RECORDS_PER_GROUP))
        panels[name] = dict(A=a, B=b, binding=panel_binding(a))
    require(len(panels) < 2 or panels['reserved']['binding']['panel_sha256'] != panels['old_frozen']['binding']['panel_sha256'],
            'reserved and old_frozen inputs name one panel_sha256')
    return dict(shared={k: first['value'][k] for k in SHARED}, checkpoints=checkpoints, panels=panels)


# ---------------------------------------------------------------- statistic

def panel_deltas(a, b):
    """Per-context deltas grouped by family: a context's delta is the mean over its paired groups of metric(B) - metric(A)."""
    by_context = {}
    for key, group_a in a['groups'].items():
        group_b = b['groups'][key]
        context = by_context.setdefault(key[0], dict(family=key[3], groups=0, deltas={m: [] for m in METRICS}))
        context['groups'] += 1
        for m in METRICS:
            context['deltas'][m].append(group_b[m] - group_a[m])
    families = {}
    for index in sorted(by_context):
        context = by_context[index]
        family = families.setdefault(context['family'], dict(contexts=[], deltas={m: [] for m in METRICS}))
        family['contexts'].append(index)
        for m in METRICS:
            family['deltas'][m].append(statistics.fmean(context['deltas'][m]))
    return families, len(by_context), sum(c['groups'] for c in by_context.values())


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


def panel_readout(name, pair, block, policy):
    families, n_contexts, n_groups = panel_deltas(pair['A'], pair['B'])
    entry = readout_panel(block, name)
    counts = {f: len(families[f]['contexts']) for f in sorted(families)}
    if name == 'reserved':
        require(type(entry.get('families')) is int and type(entry.get('contexts_per_family')) is int,
                'preregistration block lacks integer families / contexts_per_family for the reserved panel')
        require(len(families) == entry['families'], 'reserved panel has %d families; %d pre-registered' % (len(families), entry['families']))
        wrong = {f: n for f, n in counts.items() if n != entry['contexts_per_family']}
        require(not wrong, 'reserved panel family context counts differ from the pre-registered %d: %s' % (entry['contexts_per_family'], wrong))
    intervals = stratified_bootstrap(families, BOOTSTRAPPED, policy['resamples'], policy['seed'], policy['interval'])
    metrics = {}
    for m in METRICS:
        family_means = {f: statistics.fmean(families[f]['deltas'][m]) for f in sorted(families)}
        context_deltas = [d for f in sorted(families) for d in families[f]['deltas'][m]]
        bootstrapped = m in intervals
        metrics[m] = dict(family_means=family_means, equal_family_mean=statistics.fmean(family_means.values()),
                          pooled_context_mean=statistics.fmean(context_deltas), n_contexts=len(context_deltas),
                          fraction_negative_contexts=sum(d < 0 for d in context_deltas) / len(context_deltas),
                          ci95=intervals[m] if bootstrapped else unavailable('interval pre-registered for %s only' % ' and '.join(BOOTSTRAPPED)),
                          interval=policy['interval'] if bootstrapped else None, resamples=policy['resamples'] if bootstrapped else None,
                          seed=policy['seed'] if bootstrapped else None, role='primary' if m == PRIMARY else 'secondary' if m == SECONDARY else 'reported')
    preregistered = {k: entry.get(k) for k in ('families', 'contexts_per_family', 'distinct_contexts', 'draws') if k in entry}
    return dict(role=entry.get('role'), binding=pair['binding'], n_groups=n_groups, n_families=len(families), contexts_per_family=counts,
                distinct_contexts=n_contexts, preregistered=preregistered,
                distinct_contexts_match_preregistration=(n_contexts == entry['distinct_contexts']) if type(entry.get('distinct_contexts')) is int else None,
                arms={arm: dict(path=pair[arm]['path'], sha256=pair[arm]['sha256'], checkpoint_sha256=pair[arm]['value']['checkpoint_sha256'],
                                sampled_examples=pair[arm]['value']['sampled_examples'], n_groups=len(pair[arm]['groups'])) for arm in ARMS},
                bridge={arm: pair[arm]['bridge'] for arm in ARMS}, metrics=metrics)


# ---------------------------------------------------------------- verdict

def decide(panels, policy):
    """repair_signal iff reserved delta <= reserved_delta_max with the interval entirely below 0 and old-panel delta <= old_panel_tolerance."""
    inputs = dict(reserved_delta_max=policy['reserved_delta_max'], old_panel_tolerance=policy['old_panel_tolerance'], interval=policy['interval'],
                  reserved_delta=None, reserved_ci95=None, reserved_clears_threshold=None, old_panel_delta=None, old_panel_within_tolerance=None)
    reserved, old = panels.get('reserved'), panels.get('old_frozen')
    if reserved is None:
        return 'inconclusive', 'reserved_panel_missing', 'no reserved-panel read-out for both arms; the primary statistic is undefined', inputs
    r = reserved['metrics'][PRIMARY]
    inputs.update(reserved_delta=r['equal_family_mean'], reserved_ci95=r['ci95'],
                  reserved_clears_threshold=r['equal_family_mean'] <= policy['reserved_delta_max'] and r['ci95'][1] < 0)
    reserved_text = 'reserved ranking_bt delta %.6f with %g%% interval [%.6f, %.6f]' % (r['equal_family_mean'], 100 * policy['interval'], *r['ci95'])
    if old is None:
        return 'inconclusive', 'old_panel_missing', reserved_text + '; no old-frozen-panel read-out for both arms, the tolerance clause is undefined', inputs
    o = old['metrics'][PRIMARY]
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


def evaluate(inputs, registration, legacy_paths=()):
    block = registration['block']
    policy = thresholds(block)
    rules = lookup(block, 'falsifiers.F9.verdicts')
    require(isinstance(rules, dict) and set(rules) == set(VERDICTS), 'preregistration block falsifiers.F9.verdicts must name exactly %s' % (VERDICTS,))
    assembled = assemble(inputs)
    legacy = sorted((validate_legacy(path, inputs) for path in legacy_paths), key=lambda row: (row['panel'], row['arm'])) \
        or unavailable('no parallel.frozen_monitoring.v1 receipt supplied; identity rests on each input\'s legacy_aggregate bridge')
    prefixes = preregistered_prefixes(block, assembled['shared'], assembled['panels'])
    panels = {name: panel_readout(name, pair, block, policy) for name, pair in assembled['panels'].items()}
    verdict, reason, detail, verdict_inputs = decide(panels, policy)
    require(verdict in VERDICTS, 'unknown verdict')
    ordered = sorted(inputs, key=lambda item: (item['value']['panel']['name'], item['value']['arm']))
    return dict(schema=SCHEMA, preregistration_sha256=registration['sha256'],
                # The --preregistration-sha256 pin is a refusal gate, not receipt content: a pinned retry must reproduce identical bytes.
                preregistration={k: registration.get(k) for k in ('path', 'document_sha256', 'block_bytes_sha256', 'reader')},
                **assembled['shared'], arms={arm: dict(checkpoint_sha256=assembled['checkpoints'].get(arm)) for arm in ARMS},
                thresholds=policy, preregistered_prefixes=prefixes,
                inputs=[dict(path=item['path'], sha256=item['sha256'], arm=item['value']['arm'], panel=item['value']['panel']['name'],
                             checkpoint_sha256=item['value']['checkpoint_sha256'], sampled_examples=item['value']['sampled_examples'],
                             n_groups=len(item['groups'])) for item in ordered],
                legacy_receipts=legacy, panels=panels, verdict=verdict, reason=reason, reason_detail=detail, verdict_inputs=verdict_inputs,
                verdict_rules=rules, statistic=STATISTIC, bridge_rule=BRIDGE_RULE, interpretation=INTERPRETATION, hold_rule=HOLD_RULE,
                final_tests_opened=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preregistration', type=Path, required=True, help='PREREGISTRATION.md carrying the fenced json block')
    parser.add_argument('--preregistration-sha256', help='refuse unless the block digest (canonical json or block text) equals this')
    parser.add_argument('--inputs', type=Path, nargs='+', required=True, help='parallel.panel_group_metrics.v1 files, one per (arm, panel)')
    parser.add_argument('--legacy', type=Path, nargs='*', default=[], help='optional parallel.frozen_monitoring.v1 receipts for the identity check')
    parser.add_argument('--output', type=Path, help='immutable parallel.flywheel_f9_readout.v1 receipt (required unless --dry-run)')
    parser.add_argument('--dry-run', action='store_true', help='print the receipt without writing it')
    args = parser.parse_args(argv)
    require(args.output is not None or args.dry_run, '--output is required unless --dry-run')
    registration = preregistration(args.preregistration, args.preregistration_sha256)
    receipt = evaluate([validate_input(path) for path in args.inputs], registration, args.legacy)
    if args.dry_run:
        print(json.dumps(receipt, sort_keys=True))
        return receipt
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_sha = promotion_gate.immutable(args.output, receipt)
    print(json.dumps(dict(receipt=str(args.output), sha256=receipt_sha, verdict=receipt['verdict'], reason=receipt['reason'],
                          reserved_delta=receipt['verdict_inputs']['reserved_delta'], reserved_ci95=receipt['verdict_inputs']['reserved_ci95'],
                          old_panel_delta=receipt['verdict_inputs']['old_panel_delta'], panels=sorted(receipt['panels']))))
    return receipt


if __name__ == '__main__':
    main()
