#!/usr/bin/env python3
"""Append the flywheel evidence series from saved receipts; joins evidence, never claims learning.

Rows are keyed by (kind, id) and appended once, under a lock and fsync'd;
runs/flywheel-latest.json is rewritten atomically for the console. Numbers come
from receipts and from flywheel_metrics; a missing input is recorded as
unavailable with a reason, never as a zero. Frozen-panel deltas are computed
only inside the baseline receipt's monitor segment; anything else is
'incomparable'.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
import re
import statistics
import time

from promotion_gate import canonical
from resume_expired_cycle import require, sha

try:
    import flywheel_metrics
except ImportError:  # ships separately; rows then record the gap explicitly
    flywheel_metrics = None

SCHEMA = 'parallel.flywheel_series.v1'
LATEST_SCHEMA = 'parallel.flywheel_latest.v1'
BASELINE_SCHEMA = 'parallel.flywheel_baseline.v1'
BRAKE_SCHEMA = 'parallel.flywheel_brake.v1'
VERDICT_SCHEMA = 'parallel.flywheel_verdict.v1'
MONITOR_SCHEMA = 'parallel.frozen_monitoring.v1'
SEED_VALUE_MONITOR_SCHEMA = 'parallel.seed_value_monitoring.v1'
QUALITY_SCHEMA = 'parallel.continuous_quality.v1'
SUCCESSOR_ROW0_MISMATCH = 'row0_label_mismatch'
PANEL = ('sequence_nll', 'ranking_bt', 'ranking_listwise', 'runtime_huber')
PANEL_EXTRA = ('nontie_pairwise_accuracy', 'ndcg10', 'tie_fraction', 'n_pairs')
SEGMENT_KEYS = ('monitor_executable_sha256', 'metric_definitions_sha256', 'driver_sha256',
                'evaluator_sha256', 'ranking_backward', 'nll_row_weight')
# The frozen panel is a function of the monitor definitions only (binary + metric definitions):
# PREREGISTRATION F2 counts rejects across objective identities ("from the first monitored block
# after the first block of a new objective identity"), so ranking_backward/nll_row_weight stay in
# the row's segment for display but never make a checkpoint incomparable. Evaluator/driver changes
# segment the search rows.
PANEL_SEGMENT_KEYS = ('monitor_executable_sha256', 'metric_definitions_sha256')
KINDS = ('monitor', 'generation', 'block', 'episodes', 'brake', 'verdict')
ARMS = ('neural', 'generic', 'shuffled')
WINDOW = 10
TAIL = 200
METRIC_ERRORS = (ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError,
                 RuntimeError, statistics.StatisticsError, OverflowError)


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    """None when absent or malformed; callers record the gap instead of guessing."""
    try:
        return read(path)
    except (OSError, ValueError):
        return None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def clean(value):
    """JSON-safe copy: non-finite floats become null, tuples become lists."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def usable(value):
    """flywheel_baseline marks missing inputs as 'unavailable:<reason>' strings; treat them as absent."""
    return None if isinstance(value, str) and value.startswith('unavailable:') else value


def ci_bounds(value):
    """[low, high] from either interval shape: successor_targets emits {low, high, resamples}; rows carry [low, high]."""
    if isinstance(value, dict):
        value = [value.get('low'), value.get('high')]
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(finite(v) for v in value):
        return [value[0], value[1]]
    return None


def monitor_order(row):
    step = row.get('evaluated_step')
    return (step if type(step) is int else -1, row.get('timestamp') if finite(row.get('timestamp')) else -1, row['id'])


def binding_sha(value):
    if isinstance(value, dict):
        return usable(value.get('sha256'))
    return value if isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) else None


def strip_per_cut(value):
    if isinstance(value, dict):
        return {k: strip_per_cut(v) for k, v in value.items() if k != 'per_cut'}
    if isinstance(value, list):
        return [strip_per_cut(v) for v in value]
    return value


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def measure(unavailable, name, function, *args, **kwargs):
    if flywheel_metrics is None:
        unavailable[name] = 'flywheel_metrics unavailable'
        return None
    try:
        return clean(getattr(flywheel_metrics, function)(*args, **kwargs))
    except METRIC_ERRORS as error:
        unavailable[name] = '%s: %s: %s' % (function, type(error).__name__, error)
        return None


def tail_record(path, size=65536):
    """Last JSON object of a metrics.jsonl without reading the whole file."""
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            stream.seek(max(0, end - size))
            lines = [line for line in stream.read().splitlines() if line.strip()]
        value = json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------- baseline

def baseline_receipt(path):
    """Digest of the immutable parallel.flywheel_baseline.v1 receipt (bindings nest under 'binding')."""
    receipt = read(path)
    require(receipt.get('schema') == BASELINE_SCHEMA, 'unknown baseline schema: ' + str(path))
    require(receipt.get('final_tests_opened') is False, 'baseline final-test scope changed')
    binding = receipt.get('binding') if isinstance(receipt.get('binding'), dict) else {}
    metrics = receipt.get('metrics') or {}
    # flywheel_baseline publishes executables as {label: sha256} at the top level (paths under binding.executable_paths).
    executables = receipt.get('executables') or binding.get('executables') or binding.get('executable_paths') or {}
    checkpoints = receipt.get('checkpoints') or binding.get('checkpoints') or {}
    learning = usable(checkpoints.get('learning')) or {}
    legacy = metrics.get('panel_legacy') or {}
    panel = usable(legacy.get('learning')) or {}
    null = usable(metrics.get('panel_null')) or {}
    # PREREGISTRATION F2.guards.source: metrics.panel_null.guards (borrowed floors until >= 10 pre-change blocks).
    guards = null.get('guards') or null.get('effective_guards') or null.get('guards_derived') or {}
    segment = dict(monitor_executable_sha256=binding_sha(executables.get('monitor_legacy')),
                   metric_definitions_sha256=panel.get('metric_definitions_sha256', legacy.get('metric_definitions_sha256',
                                                                                             receipt.get('metric_definitions_sha256'))),
                   driver_sha256=binding_sha(executables.get('driver')),
                   evaluator_sha256=binding_sha(executables.get('cpu_evaluator')),
                   ranking_backward=usable(learning.get('ranking_backward')),
                   nll_row_weight=usable(learning.get('nll_row_weight')))
    return dict(schema=BASELINE_SCHEMA, path=str(path), sha256=sha(path),
                manifest_sha256=binding.get('manifest_sha256', receipt.get('manifest_sha256')),
                frozen_at=usable(receipt.get('frozen_at')), segment=segment,
                checkpoint=dict(path=learning.get('path'), sha256=learning.get('sha256'), step=usable(learning.get('step'))),
                panel={k: panel[k] for k in PANEL + PANEL_EXTRA if finite(panel.get(k))},
                guards=dict(g_nll=guards.get('g_nll'), g_bt=guards.get('g_bt'), borrowed=null.get('guards_borrowed'),
                            receipts_on_binding=null.get('receipts_on_binding')),
                matched_heldout=strip_per_cut(usable(metrics.get('matched_heldout'))),
                matched_train=strip_per_cut(usable(metrics.get('matched_train'))),
                gp_improves_on_seed=usable(metrics.get('gp_improves_on_seed')),
                best_final_origin=usable(metrics.get('best_final_origin')),
                regression_signature=usable(metrics.get('regression_signature')), diversity=usable(metrics.get('diversity')),
                discoveries_per_hour=usable(metrics.get('discoveries_per_hour')), throughput=usable(metrics.get('throughput')),
                queue=usable(metrics.get('queue')), final_tests_opened=False)


def run_segment(root, manifest, monitor_executable_sha256=None, metric_definitions_sha256=None,
                checkpoint=None, evaluator_sha256=None, driver_sha256=None):
    bindings = manifest.get('bindings') or {}
    state = load(root / checkpoint / 'state.json') if checkpoint else None
    state = state if isinstance(state, dict) else {}
    driver = manifest.get('continuation_driver') or manifest.get('successor_driver')
    return dict(monitor_executable_sha256=monitor_executable_sha256 or bindings.get(manifest.get('monitor_executable')),
                metric_definitions_sha256=metric_definitions_sha256,
                driver_sha256=driver_sha256 or (bindings.get(driver) if driver else None),
                evaluator_sha256=evaluator_sha256 or bindings.get(manifest.get('cpu_executable')),
                ranking_backward=state.get('ranking_backward'), nll_row_weight=state.get('nll_row_weight'))


def panel_segment(segment):
    return {k: (segment or {}).get(k) for k in PANEL_SEGMENT_KEYS}


def baseline_delta(panel, segment, baseline):
    """Deltas only inside the baseline's panel segment; anything else is incomparable."""
    if panel_segment(segment) != panel_segment(baseline['segment']):
        return 'incomparable'
    reference = baseline['panel']
    if not all(finite(reference.get(k)) for k in PANEL):
        return 'incomparable'
    return {k: panel[k] - reference[k] for k in PANEL if finite(panel.get(k))}


# ---------------------------------------------------------------- monitor rows

def checkpoint_lineage(state, checkpoint):
    if not isinstance(state, dict) or not checkpoint:
        return 'unknown'
    if checkpoint == state.get('serving_checkpoint'):
        return 'serving'
    candidates = {c.get('checkpoint') for c in state.get('candidates', []) if isinstance(c, dict)}
    if checkpoint == state.get('learning_checkpoint') or checkpoint in candidates:
        return 'learning'
    return 'unknown'


def seed_value_monitoring(receipt, path):
    """F8 second clause input: the critic's accuracy against realized S on the reporting contexts (never packed)."""
    by_context = receipt.get('nontie_pairwise_accuracy_by_context', receipt.get('by_context'))
    return dict(nontie_pairwise_accuracy_vs_S=receipt.get('nontie_pairwise_accuracy_vs_S'), ndcg10_vs_S=receipt.get('ndcg10_vs_S'),
                n_groups=receipt.get('n_groups'), contexts=receipt.get('contexts'),
                by_context=[v for v in by_context if finite(v)] if isinstance(by_context, list) else None,
                generation=receipt.get('generation'), receipt=path.name, receipt_sha256=sha(path), final_tests_opened=False)


def monitor_rows(root, run, manifest, baseline):
    state = load(run / 'state.json')
    rows, seed_value = [], {}
    receipts = []
    for path in sorted((run / 'monitoring').glob('*.json')):
        receipt = load(path)
        if not isinstance(receipt, dict) or receipt.get('final_tests_opened') is not False:
            continue
        if receipt.get('schema') == SEED_VALUE_MONITOR_SCHEMA and finite(receipt.get('nontie_pairwise_accuracy_vs_S')):
            key = (receipt.get('binding') or {}).get('checkpoint_sha256') or receipt.get('checkpoint')
            seed_value[key] = seed_value_monitoring(receipt, path)
        elif receipt.get('schema') == MONITOR_SCHEMA and all(finite(receipt.get(k)) for k in PANEL):
            receipts.append((path, receipt))
    for path, receipt in receipts:
        binding = receipt.get('binding') or {}
        segment = run_segment(root, manifest, binding.get('executable_sha256'),
                              receipt.get('metric_definitions_sha256'), receipt.get('checkpoint'))
        panel = {k: receipt[k] for k in PANEL}
        panel.update({k: receipt[k] for k in PANEL_EXTRA if finite(receipt.get(k))})
        if isinstance(receipt.get('nontie_pairwise_accuracy_by_group'), list):
            panel['nontie_pairwise_accuracy_by_group'] = [v for v in receipt['nontie_pairwise_accuracy_by_group'] if finite(v)]
        panel['baseline_delta'] = baseline_delta(panel, segment, baseline)
        joined = seed_value.get(binding.get('checkpoint_sha256')) or seed_value.get(receipt.get('checkpoint'))
        if joined is not None:
            panel['seed_value_monitoring'] = joined
        rows.append(dict(schema=SCHEMA, kind='monitor', id=path.stem, timestamp=receipt.get('timestamp'),
             generation=None, block_index=None, evaluated_step=receipt.get('evaluated_step'),
             checkpoint=receipt.get('checkpoint'), checkpoint_sha256=binding.get('checkpoint_sha256'),
             lineage=checkpoint_lineage(state, receipt.get('checkpoint')),
             corpus_sha256=binding.get('corpus_sha256'), panel_sha256=binding.get('panel_sha256'),
             segment=segment, panel=panel, receipt_sha256=sha(path), final_tests_opened=False))
    # File names are <policy>-<checkpoint sha>: lexical order is not evaluation order.
    return sorted(rows, key=monitor_order)


# ---------------------------------------------------------------- generation rows

def contrast(unavailable, name, rows, partition, level, keep_per_cut=False):
    value = measure(unavailable, name, 'matched_seed_contrast', rows, partition, level)
    return value if keep_per_cut else strip_per_cut(value)


def neural_seed_populations(root, generation_dir, jobs):
    populations = []
    for job in (jobs or {}).get('jobs') or []:
        if not isinstance(job, dict) or not isinstance(job.get('seeds'), str):
            continue
        try:
            arm = Path(job['output']).relative_to(generation_dir.relative_to(root)).parts[0]
        except (ValueError, KeyError, TypeError, IndexError):
            continue
        population = load(root / job['seeds']) if arm == 'neural' else None
        if isinstance(population, dict) and population.get('schema') == 'parallel.seed_population.v1':
            populations.append(population)
    return populations


def replay_snapshots(run, limit=WINDOW):
    """Replay policies as the deployed controller stores them: replay-membership/<identity>/policy.json (same source as
    flywheel_baseline.replay_discoveries), ordered by mtime then identity; nothing under cycle-*/train-2 is ever written."""
    policies = []
    directory = run / 'replay-membership'
    for path in sorted(directory.glob('*/policy.json')) if directory.is_dir() else []:
        policy = load(path)
        if isinstance(policy, dict) and policy.get('schema') == 'parallel.replay_policy.v2':
            policies.append((int(path.stat().st_mtime), path.parent.name, policy))
    policies.sort(key=lambda item: item[:2])
    return [(name, policy) for _, name, policy in policies[-limit:]]


def quality_reports(run):
    reports = {}
    for directory in sorted(run.glob('generation-*')):
        match = re.fullmatch(r'generation-(\d+)', directory.name)
        report = load(directory / 'quality-report.json')
        jobs_path = directory / 'jobs.json'
        if not match or not isinstance(report, dict) or not jobs_path.is_file():
            continue
        if report.get('schema') != QUALITY_SCHEMA or report.get('final_tests_opened') is not False:
            continue
        if report.get('jobs_sha256') != sha(jobs_path) or not isinstance(report.get('rows'), list):
            continue
        spec = load(directory / 'generation.json')
        spec = spec if isinstance(spec, dict) else {}
        number = spec.get('number') if type(spec.get('number')) is int else int(match.group(1))
        reports[number] = dict(directory=directory, spec=spec, report=report, jobs=load(jobs_path),
                               jobs_sha256=report['jobs_sha256'], report_sha256=sha(directory / 'quality-report.json'))
    return reports


def numbered_rows(report, number):
    return [dict(r, generation=r.get('generation', number)) for r in report.get('rows') or [] if isinstance(r, dict)]


def generation_rows(root, run, manifest, baseline, window=WINDOW):
    reports = quality_reports(run)
    snapshots = replay_snapshots(run)
    rows = []
    for number in sorted(reports):
        entry = reports[number]
        quality = numbered_rows(entry['report'], number)
        window_numbers = [n for n in sorted(reports) if number - window < n <= number]
        pooled = [r for n in window_numbers for r in numbered_rows(reports[n]['report'], n)]
        evaluators = {r.get('evaluator_sha256') for r in quality if isinstance(r.get('evaluator_sha256'), str)}
        evaluator = next(iter(evaluators)) if len(evaluators) == 1 else ('mixed' if evaluators else None)
        segment = run_segment(root, manifest, checkpoint=entry['spec'].get('checkpoint'), evaluator_sha256=evaluator)
        unavailable = {}
        trailing = contrast(unavailable, 'matched.val.trailing10', pooled, 'meta_validation', 'seed_only', keep_per_cut=True)
        if isinstance(trailing, dict):
            trailing['generations'] = window_numbers
        matched = dict(val=dict(seed_only=contrast(unavailable, 'matched.val.seed_only', quality, 'meta_validation', 'seed_only'),
                                final=contrast(unavailable, 'matched.val.final', quality, 'meta_validation', 'final'),
                                trailing10=trailing),
                       train=dict(seed_only=contrast(unavailable, 'matched.train.seed_only', quality, 'meta_train', 'seed_only'),
                                  final=contrast(unavailable, 'matched.train.final', quality, 'meta_train', 'final')))
        populations = neural_seed_populations(root, entry['directory'], entry['jobs'])
        if populations:
            diversity = measure(unavailable, 'diversity', 'diversity', populations, [p for _, p in snapshots])
        else:
            diversity = None
            unavailable['diversity'] = 'no neural seed populations readable for this generation'
        signature = measure(unavailable, 'regression_signature', 'regression_signature',
                            {n: reports[n]['report'] for n in window_numbers})
        improves = {arm: measure(unavailable, 'gp_improves_on_seed.' + arm, 'gp_improves_on_seed', quality, arm)
                    for arm in ('neural', 'generic')}
        rows.append(dict(schema=SCHEMA, kind='generation', id=entry['directory'].name, timestamp=entry['report'].get('timestamp'),
             generation=number, block_index=None, checkpoint=entry['spec'].get('checkpoint'),
             checkpoint_sha256=entry['spec'].get('checkpoint_sha256'), corpus=entry['spec'].get('corpus'),
             segment=segment, evaluator_sha256=evaluator, contexts=len({(r.get('dataset'), r.get('cut')) for r in quality}),
             partitions=dict(Counter(str(r.get('partition')) for r in quality)), matched=matched,
             gp_improves_on_seed=improves,
             best_final_origin=dict(Counter(str(r.get('best_final_origin')) for r in quality)),
             diversity=diversity, snapshots=[name for name, _ in snapshots], regression_signature=signature,
             discoveries_per_hour=None, unavailable=dict(unavailable, discoveries_per_hour='no producer in this ship'),
             report_sha256=entry['report_sha256'], jobs_sha256=entry['jobs_sha256'], codegen='pending',
             baseline_comparable=(evaluator is not None and evaluator == baseline['segment'].get('evaluator_sha256')),
             final_tests_opened=False))
    return rows


# ---------------------------------------------------------------- block rows

def seed_values_status(complete):
    """F7: a block must say whether it consumed a seed_values block; silence is recorded as such."""
    if complete.get('seed_values_sha256'):
        return dict(consumed='yes', reason='seed_values block %s' % str(complete['seed_values_sha256'])[:12])
    if complete.get('seed_values_absent'):
        return dict(consumed='no', reason=str(complete['seed_values_absent']))
    return dict(consumed='no', reason='block receipt carries no seed_values field (trainer identity without seed values)')


def block_rows(run):
    rows = []
    for path in sorted(run.glob('cycle-*/train-2/complete.json')):
        complete = load(path)
        match = re.fullmatch(r'cycle-(\d+)', path.parent.parent.name)
        if not match or not isinstance(complete, dict) or complete.get('final_tests_opened') is not False:
            continue
        latest = complete.get('latest') if isinstance(complete.get('latest'), dict) else {}
        metrics = tail_record(path.parent / 'metrics.jsonl')
        stamped = finite(complete.get('timestamp'))
        rows.append(dict(schema=SCHEMA, kind='block', id=path.parent.parent.name,
             timestamp=complete['timestamp'] if stamped else int(path.stat().st_mtime),
             timestamp_source='receipt' if stamped else 'receipt_mtime',
             generation=None, block_index=int(match.group(1)),
             checkpoint=latest.get('checkpoint'), checkpoint_sha256=latest.get('state_sha256'),
             corpus_sha256=complete.get('corpus_sha256'),
             steps=dict(initial=complete.get('initial_optimizer_step'), target=complete.get('target_optimizer_step'),
                        requested=complete.get('steps', complete.get('requested_step_increment'))),
             replay_policy_sha256=complete.get('replay_policy_sha256'), seed_values=seed_values_status(complete),
             seed_value_bt=metrics.get('seed_value_bt') if finite(metrics.get('seed_value_bt')) else None,
             seed_value_groups_seen=metrics.get('seed_value_groups_seen') if type(metrics.get('seed_value_groups_seen')) is int else None,
             receipt_sha256=sha(path), final_tests_opened=False))
    return rows


# ---------------------------------------------------------------- episode rows

def arm_outcomes(records):
    arms = {}
    for record in records:
        arm = str(record.get('arm'))
        a = arms.setdefault(arm, dict(n=0, flips=0, better=0, tied=0, worse=0, failed=0, superseded=0, child_beats_incumbent=0))
        a['n'] += 1
        if record.get('status') == 'superseded':
            a['superseded'] += 1
            continue
        child = record.get('child') if isinstance(record.get('child'), dict) else {}
        rmse, r0 = child.get('rmse'), record.get('r0')
        if finite(rmse) and finite(r0):
            a['better' if rmse < r0 else 'worse' if rmse > r0 else 'tied'] += 1
        else:
            a['failed'] += 1
        a['flips'] += int(record.get('flip') in (True, 1))
        a['child_beats_incumbent'] += int(record.get('child_beats_incumbent') is True)
    return arms


def median_or_none(values):
    values = [v for v in values if finite(v)]
    return statistics.median(values) if values else None


def single(values):
    values = {v for v in values if isinstance(v, str)}
    return next(iter(values)) if len(values) == 1 else ('mixed' if values else None)


def policy_block(successor, generation):
    for path in sorted((successor / 'seed-values').glob('*.json')):
        block = load(path)
        if isinstance(block, dict) and block.get('schema') == 'parallel.seed_values.v1' and block.get('generation') == generation:
            return block, sha(path)
    return None, None


def row0_mismatches(records, successor):
    """F5 evidence in the successor ledger's own shape: export rows with row0_label_verified false (measured and not
    bit-equal to the admitted label) plus admitted-sources exclusions whose nested reasons start with
    row0_label_mismatch; episodes are counted once by id. Unverifiable rows (no evidence, seed timeout) are not mismatches."""
    mismatched = {r.get('episode_id') or r.get('identity_sha256') or ('row-%d' % i)
                  for i, r in enumerate(records) if r.get('row0_label_verified') is False}
    sources = load(successor / 'admitted-sources.json')
    if isinstance(sources, dict):
        for entry in sources.get('excluded') or []:
            if not isinstance(entry, dict):
                continue
            for item in entry.get('unverified') or []:
                if isinstance(item, dict) and any(str(reason).startswith(SUCCESSOR_ROW0_MISMATCH) for reason in item.get('reasons') or []):
                    mismatched.add(item.get('episode_id') or ('excluded-%s-%s' % (entry.get('generation'), entry.get('context_key'))))
        declared = sources.get('row0_mismatches')
        if type(declared) is int and declared > len(mismatched):
            return declared
    return len(mismatched)


def episode_rows(root, manifest, successor):
    rows = []
    for path in sorted(successor.glob('seed-values-*.json')):
        match = re.fullmatch(r'seed-values-(\d+)\.json', path.name)
        if not match:
            continue
        generation = int(match.group(1))
        export = load(path)
        records = export if isinstance(export, list) else export.get('rows') if isinstance(export, dict) else None
        unavailable = {}
        if not isinstance(records, list):
            records = []
            unavailable['records'] = 'seed-values export unreadable or without rows'
        records = [r for r in records if isinstance(r, dict)]
        economics = load(successor / 'economics' / ('generation-%d.json' % generation))
        economics = economics if isinstance(economics, dict) else {}
        if not economics:
            unavailable['economics'] = 'economics receipt absent'
        block, block_sha = policy_block(successor, generation)
        info = (block or {}).get('info_test') or {}
        counts = (block or {}).get('counts') or {}
        floors = (block or {}).get('floors') or {}
        if block is None:
            unavailable['info_test'] = 'seed_values policy block absent for generation %d' % generation
        # A_ctx is produced by successor_targets.build (policy block a_ctx.by_arm), never by export rows.
        by_arm = ((block or {}).get('a_ctx') or {}).get('by_arm')
        a_ctx = {arm: (by_arm.get(arm) if isinstance(by_arm, dict) and finite(by_arm.get(arm)) else None) for arm in ARMS}
        if not isinstance(by_arm, dict):
            unavailable['a_ctx'] = 'policy block carries no a_ctx.by_arm' if block is not None else 'seed_values policy block absent'
        elif all(v is None for v in a_ctx.values()):
            unavailable['a_ctx'] = 'no context with a measured control_generic seed'
        a_own_ci = ci_bounds(info.get('a_own_ci'))
        if info and a_own_ci is None:
            unavailable['a_own_ci'] = 'info_test.a_own_ci unavailable: ' + str(((info.get('a_own') or {}).get('reason')) or 'no interval')
        evaluator = single(r.get('evaluator_sha256') for r in records)
        driver = single(r.get('driver_sha256') for r in records)
        successor_block = dict(tier=economics.get('tier'), episodes=len(records),
            roles=dict(Counter(str(r.get('role')) for r in records)),
            statuses=dict(Counter(str(r.get('status')) for r in records)),
            completed=sum(r.get('status') == 'measured' for r in records), failed=sum(r.get('status') == 'failed' for r in records),
            deadline=sum(r.get('status') == 'deadline' for r in records), superseded=sum(r.get('status') == 'superseded' for r in records),
            core_seconds=economics.get('reserved_core_seconds'),
            fits_per_core_second=economics.get('unique_fits_per_core_second', economics.get('unique_successful_fits_per_reserved_core_second')),
            attempted_fits=economics.get('attempted_fits'), successful_fits=economics.get('successful_fits'),
            timeouts=economics.get('timeouts'), completed_within_generation=economics.get('completed_within_generation'),
            arms=arm_outcomes(records), eps_median=median_or_none(list(floors.values())),
            tie_fraction_median=counts.get('tie_fraction_median', info.get('tie_fraction_median')),
            p_flip=info.get('p_flip'), a_own_ci=a_own_ci,
            a_own_ci_resamples=(info.get('a_own_ci') or {}).get('resamples') if isinstance(info.get('a_own_ci'), dict) else None,
            info_test_passed=info.get('passed'), a_ctx=a_ctx, a_ctx_contexts_with_control=((block or {}).get('a_ctx') or {}).get('contexts_with_control'),
            credited_groups=counts.get('groups'), refused_contexts=counts.get('refused_contexts'),
            row0_mismatches=row0_mismatches(records, successor), row0_verified_rows=sum(r.get('row0_label_verified') is True for r in records),
            policy_block_sha256=block_sha)
        rows.append(dict(schema=SCHEMA, kind='episodes', id='seed-values-%d' % generation,
             timestamp=max([r.get('completed_at') for r in records if finite(r.get('completed_at'))], default=None),
             generation=generation, block_index=None, successor=successor_block, export_sha256=sha(path),
             segment=run_segment(root, manifest, evaluator_sha256=evaluator, driver_sha256=driver),
             evaluator_sha256=evaluator, driver_sha256=driver, unavailable=unavailable, final_tests_opened=False))
    return rows


# ---------------------------------------------------------------- brake / verdict rows

def brake_rows(run):
    brake = load(run / 'brake.json')
    if not isinstance(brake, dict) or brake.get('schema') != BRAKE_SCHEMA:
        return [], None
    identity = digest({k: v for k, v in brake.items() if k not in ('timestamp', 'pause', 'series_rows')})[:16]
    row = dict(schema=SCHEMA, kind='brake', id=identity, timestamp=brake.get('timestamp'), generation=None, block_index=None,
               brake=dict(held=brake.get('held'), reason=brake.get('reason'), tripped_by=brake.get('tripped_by'),
                          rejects=brake.get('rejects'), consecutive=brake.get('consecutive'), fires=brake.get('fires')),
               manifest_sha256=brake.get('manifest_sha256'), final_tests_opened=False)
    return [row], brake


def verdict_rows(successor):
    rows, latest = [], None
    for path in sorted((successor / 'verdicts').glob('generation-*.json')):
        verdict = load(path)
        match = re.fullmatch(r'generation-(\d+)', path.stem)
        if not match or not isinstance(verdict, dict) or verdict.get('schema') != VERDICT_SCHEMA:
            continue
        summary = dict(generation=int(match.group(1)), outcome=verdict.get('outcome'), evaluable=verdict.get('evaluable'),
                       not_evaluable=verdict.get('not_evaluable'), sha256=sha(path), path=str(path))
        rows.append(dict(schema=SCHEMA, kind='verdict', id=path.stem, timestamp=None, generation=summary['generation'],
                         block_index=None, verdict=summary, verdict_sha256=summary['sha256'], final_tests_opened=False))
        latest = summary if latest is None or summary['generation'] > latest['generation'] else latest
    return rows, latest


# ---------------------------------------------------------------- series file

def series_rows(path):
    rows = []
    if not Path(path).is_file():
        return rows
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                require(row.get('schema') == SCHEMA and row.get('kind') in KINDS, 'foreign row in flywheel series')
                rows.append(row)
    return rows


def append(series, rows):
    """Append-only under a lock; (kind, id) pairs already present are skipped."""
    series = Path(series)
    series.parent.mkdir(parents=True, exist_ok=True)
    with series.with_name(series.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        seen = {(r['kind'], r['id']) for r in series_rows(series)}
        fresh = []
        for row in rows:
            key = (row['kind'], row['id'])
            if key not in seen:
                seen.add(key)
                fresh.append(clean(row))
        with series.open('a') as stream:
            for row in fresh:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    return fresh


def publish_latest(path, manifest_sha, baseline, series, brake, verdict):
    rows = series_rows(series)
    value = dict(schema=LATEST_SCHEMA, timestamp=int(time.time()), manifest_sha256=manifest_sha, series=str(series),
                 series_rows=len(rows), rows=rows[-TAIL:], baseline=baseline, brake=brake, verdict=verdict,
                 flywheel_metrics='available' if flywheel_metrics else 'unavailable', final_tests_opened=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.%s.%d.partial' % (path.name, os.getpid()))
    with temporary.open('w') as stream:
        json.dump(clean(value), stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return value


def produce(root, manifest_path, baseline_path, series, latest, successor=None, window=WINDOW):
    root = Path(root).resolve()
    manifest_path = resolve(root, manifest_path)
    manifest = read(manifest_path)
    require(isinstance(manifest.get('output'), str), 'manifest lacks output')
    run = root / manifest['output']
    successor = resolve(root, successor) if successor else run / 'successor'
    baseline = baseline_receipt(resolve(root, baseline_path))
    rows = monitor_rows(root, run, manifest, baseline) + generation_rows(root, run, manifest, baseline, window)
    rows += block_rows(run) + episode_rows(root, manifest, successor)
    brake_row, brake = brake_rows(run)
    verdicts, verdict = verdict_rows(successor)
    fresh = append(series, rows + brake_row + verdicts)
    publish_latest(latest, sha(manifest_path), baseline, series, brake, verdict)
    return fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='immutable parallel.flywheel_baseline.v1 receipt')
    parser.add_argument('--series', type=Path, help='default <run>/flywheel-series.jsonl')
    parser.add_argument('--latest', type=Path, help='default runs/flywheel-latest.json under --root')
    parser.add_argument('--successor', type=Path, help='default <run>/successor')
    parser.add_argument('--window', type=int, default=WINDOW)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = read(resolve(root, args.manifest))
    run = root / manifest['output']
    series = resolve(root, args.series) if args.series else run / 'flywheel-series.jsonl'
    latest = resolve(root, args.latest) if args.latest else root / 'runs/flywheel-latest.json'
    fresh = produce(root, args.manifest, args.baseline, series, latest, args.successor, args.window)
    print(json.dumps(dict(appended=len(fresh), kinds=dict(Counter(r['kind'] for r in fresh)), series=str(series),
                          latest=str(latest), flywheel_metrics='available' if flywheel_metrics else 'unavailable')))


if __name__ == '__main__':
    main()
