#!/usr/bin/env python3
"""Freeze the pre-change flywheel baseline receipt; bound to the live release, never to a wall clock.

Every falsifier in PREREGISTRATION.md is a delta against this receipt, so it is
written once through promotion_gate.immutable and re-derived byte-for-byte by
--verify.  Required bindings (manifest, executables) fail closed; every optional
input that is absent yields an explicit 'unavailable:<reason>' field instead of
a fabricated number.  Wall-clock time never enters the receipt: frozen_at is the
latest timestamp carried by the bound evidence.

Receipt shape (parallel.flywheel_baseline.v1): the appendix keys live at the top
level (manifest_sha256, executables, checkpoints, panel_sha256, metrics, ...);
`binding` carries the path-level detail behind each digest.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import sys

import continuous_status
import experiment_runtime
import flywheel_metrics
import promotion_gate
import continuous_quality_report
import resume_expired_cycle
import scientific_metadata
import selfplay_milestones
from resume_expired_cycle import read, require, sha
from scientific_metadata import finite

SCHEMA = 'parallel.flywheel_baseline.v1'
PREREGISTRATION_SCHEMA = 'parallel.flywheel_preregistration.v1'
PREREGISTRATION = 'parallel/recovery/eight-a100-20260912/PREREGISTRATION.md'
MONITOR_SCHEMA = 'parallel.frozen_monitoring.v1'
QUALITY_SCHEMA = 'parallel.continuous_quality.v1'
MANIFEST_SCHEMA = 'parallel.continuous_learning.v1'
METRICS = ('sequence_nll', 'ranking_bt', 'ranking_listwise', 'runtime_huber')
PERCENTILE_FORMULA = '1 - i/(n-1)'
REFERENCE_LABEL_POLICY = 'reference_v1'
TAIL_CAVEAT = flywheel_metrics.TAIL_CAVEAT
TAIL_FREE_ORIGINS = flywheel_metrics.OWN_ORIGINS
DEFAULT_REGRESSION_RECEIPT = 'runs/registry-integration-20260910/health-v1/qualification-status/latest.json'
BORROWED_GUARDS = dict(g_nll=selfplay_milestones.NLL_GUARD, g_bt=selfplay_milestones.BT_GUARD)
PANEL_NULL_MINIMUM_BLOCKS = 10
EXECUTABLES = (('cpu_evaluator', 'cpu_executable'), ('monitor_legacy', 'monitor_executable'),
               ('training', 'training_executable'), ('packing', 'packing_executable'))
CHECKPOINT_FIELDS = ('step', 'ranking_backward', 'runtime_objective', 'nll_row_weight', 'corpus_sha256', 'vocabulary_sha256')
CATCH = (RuntimeError, ValueError, KeyError, TypeError, ZeroDivisionError, OSError, AttributeError, IndexError)
FENCE = re.compile(r'^```json[ \t]*\n(.*?)^```[ \t]*$', re.S | re.M)


def unavailable(reason):
    return 'unavailable:' + reason


def is_unavailable(value):
    return isinstance(value, str) and value.startswith('unavailable:')


def within(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root), 'receipt input outside workspace: ' + str(name))
    return path


def relative(root, path):
    path = Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def strictly_better(a, b):
    """Same tolerance as continuous_status.better; ties never count as discoveries."""
    return a < b and not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def metric_call(name, *args, **kwargs):
    """Route to flywheel_metrics; a failing producer is an explicit gap, never a crash."""
    producer = getattr(flywheel_metrics, name, None)
    if producer is None:
        return unavailable('flywheel_metrics lacks ' + name)
    try:
        return producer(*args, **kwargs)
    except CATCH as error:
        return unavailable(f'{name} raised {type(error).__name__}: {error}')


# ---------------------------------------------------------------- bindings

def manifest_binding(root, manifest_path):
    manifest = read(manifest_path)
    require(manifest.get('schema') == MANIFEST_SCHEMA, 'unknown continuous manifest schema')
    require(isinstance(manifest.get('bindings'), dict), 'manifest lacks bindings')
    parent = manifest.get('parent_manifest')
    if isinstance(parent, dict) and isinstance(parent.get('sha256'), str):
        parent_sha, parent_path = parent['sha256'], parent.get('path')
    elif isinstance(manifest.get('parent_manifest_sha256'), str):
        parent_sha, parent_path = manifest['parent_manifest_sha256'], manifest.get('parent_manifest_path')
    else:
        parent_sha, parent_path = unavailable('manifest has no parent chain key'), None
    for key in ('contexts_sha256', 'vocabulary_sha256', 'output', 'monitor_corpus', 'monitor_panel'):
        require(isinstance(manifest.get(key), str) and manifest[key], 'manifest lacks ' + key)
    return manifest, dict(manifest_path=relative(root, manifest_path), manifest_sha256=sha(manifest_path),
        parent_manifest_sha256=parent_sha, parent_manifest_path=parent_path,
        release_revision=manifest.get('release_revision', unavailable('manifest has no release_revision')),
        run_identity=manifest.get('run_identity', unavailable('manifest has no run_identity')),
        output=manifest['output'], contexts_sha256=manifest['contexts_sha256'],
        vocabulary_sha256=manifest['vocabulary_sha256'])


def executable_binding(root, manifest, key):
    """The on-disk bytes must equal the manifest binding: this is the identity every metric hangs on."""
    name = manifest.get(key)
    require(isinstance(name, str) and name, 'manifest lacks executable key: ' + key)
    expected = manifest['bindings'].get(name)
    require(isinstance(expected, str) and len(expected) == 64, 'manifest does not bind executable: ' + name)
    path = within(root, name)
    require(path.is_file(), 'bound executable missing: ' + name)
    actual = sha(path)
    require(actual == expected, 'executable binding changed: ' + name)
    return dict(path=name, sha256=actual)


def store_binding(root, manifest):
    store = manifest.get('teaching_store')
    if not isinstance(store, str) or not store:
        return unavailable('manifest has no teaching_store')
    path = within(root, store) / 'store.sqlite'
    if not path.is_file():
        return unavailable('store.sqlite missing: ' + relative(root, path))
    try:
        connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        try:
            rows = [row[0] for row in connection.execute('SELECT identity FROM binding ORDER BY identity')]
        finally:
            connection.close()
    except sqlite3.Error as error:
        return unavailable(f'store binding unreadable: {error}')
    return dict(path=relative(root, path), identities=rows, rows=len(rows), table='binding(identity TEXT)')


def checkpoint_binding(root, name, expected_sha=None):
    """State fields are copied raw (None when absent) so series segment keys compare equal."""
    if not isinstance(name, str) or not name:
        return unavailable('checkpoint path absent from controller state')
    state_path = within(root, name) / 'state.json'
    if not state_path.is_file():
        return unavailable('checkpoint state missing: ' + name)
    digest = sha(state_path)
    if expected_sha is not None:
        require(digest == expected_sha, 'checkpoint identity changed: ' + name)
    state = read(state_path)
    return dict(path=name, sha256=digest, absent_fields=[f for f in CHECKPOINT_FIELDS if f not in state],
                **{f: state.get(f) for f in CHECKPOINT_FIELDS})


def corpus_binding(root, name):
    if not isinstance(name, str) or not name:
        return unavailable('learning corpus absent from controller state')
    index = within(root, name) / 'index.json'
    if not index.is_file():
        return unavailable('learning corpus index missing: ' + name)
    value = read(index)
    return dict(path=name, sha256=sha(index), schema=value.get('schema'),
                label_policy=value.get('label_policy', unavailable('label_policy absent from corpus index')))


def monitor_binding(root, manifest, monitor_executable_sha):
    """Receipts bind the panel's declared digests (continuous_status.observe:390-392); file digests are recorded beside them."""
    corpus_index = within(root, manifest['monitor_corpus']) / 'index.json'
    require(corpus_index.is_file(), 'monitor corpus index missing: ' + manifest['monitor_corpus'])
    panel_path = within(root, manifest['monitor_panel'])
    require(panel_path.is_file(), 'monitor panel missing: ' + manifest['monitor_panel'])
    panel = read(panel_path)
    corpus_index_sha = sha(corpus_index)
    panel_file_sha = sha(panel_path)
    panel_sha = panel.get('panel_sha256') if isinstance(panel.get('panel_sha256'), str) else panel_file_sha
    corpus_sha = panel.get('corpus_sha256') if isinstance(panel.get('corpus_sha256'), str) else corpus_index_sha
    # Same digest the controller uses to name monitoring/<policy>-<checkpoint>.json (runtime-v4 :967-974).
    policy = hashlib.sha256((monitor_executable_sha + ':' + panel_file_sha + ':' + corpus_index_sha).encode()).hexdigest()
    return dict(monitor_corpus=manifest['monitor_corpus'], monitor_corpus_sha256=corpus_index_sha,
        monitor_panel=manifest['monitor_panel'], panel_sha256=panel_sha, panel_file_sha256=panel_file_sha,
        panel_declares_corpus_sha256=panel.get('corpus_sha256', unavailable('panel has no corpus_sha256')),
        corpus_binding_matches_index=corpus_sha == corpus_index_sha, monitor_policy_sha256=policy,
        receipt_binding=dict(panel_sha256=panel_sha, corpus_sha256=corpus_sha, executable_sha256=monitor_executable_sha))


# ---------------------------------------------------------------- monitoring

def monitoring_receipts(root, run):
    directory = run / 'monitoring'
    receipts, ignored = [], []
    for path in sorted(directory.glob('*.json')) if directory.is_dir() else []:
        value = read(path)
        if value.get('schema') != MONITOR_SCHEMA or not isinstance(value.get('binding'), dict):
            ignored.append(relative(root, path))
            continue
        receipts.append((path, value))
    return receipts, ignored


def compact_monitor(root, path, receipt):
    row = dict(checkpoint=receipt.get('checkpoint'), checkpoint_sha256=receipt['binding'].get('checkpoint_sha256'),
               evaluated_step=receipt.get('evaluated_step'), timestamp=receipt.get('timestamp'),
               receipt=relative(root, path), receipt_sha256=sha(path))
    for key in METRICS:
        row[key] = receipt.get(key)
    return row


def panel_legacy(root, receipts, ignored, binding, checkpoints):
    keys = ('panel_sha256', 'corpus_sha256', 'executable_sha256')
    current, others = [], {}
    for path, receipt in receipts:
        if all(receipt['binding'].get(k) == binding[k] for k in keys):
            current.append(compact_monitor(root, path, receipt))
        else:
            label = ':'.join(str(receipt['binding'].get(k))[:16] for k in keys)
            others[label] = others.get(label, 0) + 1
    current.sort(key=lambda r: (r['evaluated_step'] if type(r['evaluated_step']) is int else -1, r['timestamp'] or 0, r['receipt']))
    result = dict(binding=dict(binding), series=current, receipts_on_binding=len(current),
                  receipts_other_bindings=others, ignored_receipts=ignored, metric_definitions_sha256=None,
                  metric_definitions_note='bin-v3 legacy monitor: sums at monitoring.rs:84-93; no metric_definitions_sha256 before B1')
    for role in ('learning', 'serving'):
        checkpoint = checkpoints[role]
        if is_unavailable(checkpoint):
            result[role] = checkpoint
            continue
        rows = [r for r in current if r['checkpoint_sha256'] == checkpoint['sha256']]
        if not rows:
            result[role] = unavailable(f'no frozen-panel receipt for the {role} checkpoint {checkpoint["sha256"][:12]} under the bound monitor policy')
            continue
        row = rows[-1]
        require(all(finite(row[k]) for k in METRICS), f'nonfinite frozen-panel metric for the {role} checkpoint')
        result[role] = row
    return result


def on_binding(raw_receipts, binding):
    keys = ('panel_sha256', 'corpus_sha256', 'executable_sha256')
    return [receipt for _, receipt in raw_receipts if all(receipt['binding'].get(k) == binding[k] for k in keys)]


def panel_null(raw_receipts, binding):
    """Empirical block-to-block null of the bound monitor policy only (same panel, corpus and executable as the receipt);
    guards stay borrowed until ten pre-change blocks exist ON THAT BINDING. Other bindings on the node (the run was rebound
    cc248228/db0aabe9 -> v4/87995977) contribute only to a labelled secondary that never feeds a guard."""
    current = on_binding(raw_receipts, binding)
    null = metric_call('panel_null', current) if current else unavailable('no monitored receipts on the bound monitor policy')
    result = dict(n_rows=len(current), n_receipts=len(raw_receipts), receipts_on_binding=len(current), binding=dict(binding),
                  minimum_blocks=PANEL_NULL_MINIMUM_BLOCKS, borrowed_guards=dict(BORROWED_GUARDS), producer='flywheel_metrics.panel_null',
                  rule='g_nll = max(0.005, 2*p90 rel|delta nll|); g_bt = max(0.01, 2*p90 rel|delta bt|) over consecutive receipts of the '
                       f'bound monitor policy; guards stay borrowed until >= {PANEL_NULL_MINIMUM_BLOCKS} pre-change monitored receipts on it')
    candidate = None
    if isinstance(null, dict):
        for metric in METRICS:
            result[metric] = (null.get('metrics') or {}).get(metric, unavailable('metric absent from panel null'))
        guards = null.get('guards') or {}
        if finite(guards.get('g_nll')) and finite(guards.get('g_bt')):
            candidate = dict(g_nll=guards['g_nll'], g_bt=guards['g_bt'])
        result.update(n_segments=null.get('n_segments'), segments=null.get('segments'),
                      relative_delta=null.get('relative_delta'), order=null.get('order'))
    else:
        result['reason'] = null
        for metric in METRICS:
            result[metric] = null
    borrowed = candidate is None or len(current) < PANEL_NULL_MINIMUM_BLOCKS
    result.update(guards_borrowed=borrowed, guards_derived=None if borrowed else candidate,
                  guards_from_insufficient_null=candidate if borrowed else None,
                  guards=dict(BORROWED_GUARDS) if borrowed else candidate)
    pooled = metric_call('panel_null', [receipt for _, receipt in raw_receipts]) if raw_receipts else unavailable('no monitored receipts')
    if isinstance(pooled, dict):
        result['pooled_all_bindings'] = dict(label='secondary: pooled across every monitor binding on the node; never a guard source',
                                             n_receipts=pooled.get('n_receipts'), n_segments=pooled.get('n_segments'),
                                             p90={m: (pooled.get('metrics') or {}).get(m, {}).get('p90') for m in METRICS},
                                             guards_if_pooled=pooled.get('guards'))
    else:
        result['pooled_all_bindings'] = pooled
    return result


CYCLE = re.compile(r'^(?P<run>.+?)/cycle-(?P<index>\d+)/')


def regression_pair(root, path):
    """The pre-registered regression pair of the chained run: ranking_bt at the run's first monitored block and at the last
    block before the rollback (the largest single-step drop), read across every monitor segment of the receipt (current +
    archived_policies) so a segment boundary inside the run cannot hide either end. On the local registry-integration
    receipt this is cycle-000000 0.5708 -> cycle-000073 0.6796 (+19%; ranking_bt is a loss), the 74-block chain."""
    if not path.is_file():
        return unavailable('regression receipt missing: ' + relative(root, path))
    receipt = read(path)
    monitoring = receipt.get('monitoring')
    if not isinstance(monitoring, dict) or not isinstance(monitoring.get('history'), list) or not monitoring['history']:
        return unavailable('regression receipt carries no monitoring history: ' + relative(root, path))
    try:
        rows = [row for row in flywheel_metrics.monitoring_rows(monitoring) if finite(row.get('ranking_bt'))]
    except CATCH as error:
        return unavailable(f'regression receipt monitoring blocks unreadable: {type(error).__name__}: {error}')
    if not rows:
        return unavailable('regression receipt history has no finite ranking_bt rows')
    latest = monitoring.get('latest') if isinstance(monitoring.get('latest'), dict) else {}
    match = CYCLE.match(str(latest.get('checkpoint') or ''))
    if match is None:
        return unavailable('regression receipt latest checkpoint is not a <run>/cycle-<n>/ path: ' + str(latest.get('checkpoint')))
    run = match.group('run')
    cycles = {}
    for row in sorted(rows, key=lambda r: (r.get('timestamp') or 0, str(r.get('checkpoint')))):
        found = CYCLE.match(str(row.get('checkpoint') or ''))
        if found and found.group('run') == run:
            cycles.setdefault(int(found.group('index')), []).append(row)
    if not cycles:
        return unavailable('regression receipt carries no monitored cycle of the run ' + run)
    def compact(row, index=None):
        value = {k: row.get(k) for k in ('checkpoint', 'timestamp', *METRICS)}
        value.update(cycle=index, executable_sha256=(row.get('binding') or {}).get('executable_sha256'))
        return value
    ordered = sorted(cycles)
    value = {index: cycles[index][0]['ranking_bt'] for index in ordered}
    disagreements = [index for index in ordered if len({r['ranking_bt'] for r in cycles[index]}) > 1]
    drops = [(value[b] - value[a], a, b) for a, b in zip(ordered, ordered[1:])]
    largest = min(drops) if drops else None
    first_index = ordered[0]
    end_index = largest[1] if largest is not None and largest[0] < 0 else ordered[-1]
    first, end = cycles[first_index][0], cycles[end_index][0]
    peak_index = max(ordered, key=lambda i: value[i])
    segments = {}
    for index in ordered:
        for row in cycles[index]:
            executable = (row.get('binding') or {}).get('executable_sha256')
            entry = segments.setdefault(executable, dict(executable_sha256=executable, first_cycle=index, last_cycle=index, rows=0))
            entry.update(first_cycle=min(entry['first_cycle'], index), last_cycle=max(entry['last_cycle'], index), rows=entry['rows'] + 1)
    return dict(receipt=relative(root, path), receipt_sha256=sha(path), schema=receipt.get('schema'), run=run,
        binding=monitoring.get('binding'), baseline={k: (monitoring.get('baseline') or {}).get(k) for k in ('checkpoint', 'timestamp', *METRICS)},
        latest={k: latest.get(k) for k in ('checkpoint', 'timestamp', *METRICS)},
        first=compact(first, first_index), end=compact(end, end_index), peak=compact(cycles[peak_index][0], peak_index),
        rollback=(dict(from_cycle=largest[1], to_cycle=largest[2], drop=largest[0], to_ranking_bt=value[largest[2]]) if largest is not None and largest[0] < 0
                  else unavailable('no single-step drop in ranking_bt within the run')),
        history_rows=len(rows), cycles=len(ordered), cycle_range=[ordered[0], ordered[-1]], shared_cycles=sum(len(cycles[i]) > 1 for i in ordered),
        cross_binding_disagreements=disagreements, segments=sorted(segments.values(), key=lambda s: s['first_cycle']),
        ranking_bt_from=first['ranking_bt'], ranking_bt_to=end['ranking_bt'], pair=[first['ranking_bt'], end['ranking_bt']],
        end_over_first=end['ranking_bt'] / first['ranking_bt'] if first['ranking_bt'] > 0 else unavailable('zero first ranking_bt'),
        statement='ranking_bt (a loss) from the run\'s first monitored cycle to the last cycle before the rollback (largest single-step drop), '
                  'across current + archived monitor segments; the pair the brake exists to catch')


# ---------------------------------------------------------------- generations

def generation_window(root, run, count):
    """Last `count` generations that carry a quality report bound to their jobs manifest."""
    require(type(count) is int and count > 0, 'generation window must be a positive integer')
    found = []
    for directory in sorted(run.glob('generation-*')):
        report_path, jobs_path = directory / 'quality-report.json', directory / 'jobs.json'
        if not (directory.is_dir() and report_path.is_file() and jobs_path.is_file()):
            continue
        suffix = directory.name.split('-', 1)[1]
        if not suffix.isdigit():
            continue
        report = read(report_path)
        require(report.get('schema') == QUALITY_SCHEMA, 'unknown quality schema: ' + relative(root, report_path))
        require(report.get('jobs_sha256') == sha(jobs_path), 'quality jobs binding changed: ' + relative(root, directory))
        require(report.get('final_tests_opened') is False, 'quality report opened final tests: ' + relative(root, directory))
        spec_path = directory / 'generation.json'
        spec = read(spec_path) if spec_path.is_file() else {}
        jobs = read(jobs_path)
        checkpoint = spec.get('checkpoint') or next((j.get('parent_checkpoint') for j in jobs.get('jobs', [])), None)
        found.append(dict(number=int(suffix), path=directory, relative=relative(root, directory), report=report,
                          jobs=jobs, report_sha256=sha(report_path), jobs_sha256=report['jobs_sha256'],
                          checkpoint=checkpoint, checkpoint_sha256=spec.get('checkpoint_sha256')))
    found.sort(key=lambda g: g['number'])
    return found[-count:]


def derive_best_initial_arm(root, row):
    """`best_initial_arm` for a quality row written before the producer existed.

    Same rule as continuous_quality_report.search_row: the best original-target score over this
    arm's own initial population (slots 0..63, whatever the origin string), failures skipped. Read
    from the row's own observation ledger; never imputed from another arm or from the teacher.
    """
    output = row.get('output')
    ledger = within(root, output) / 'observations.json' if isinstance(output, str) else None
    if ledger is None or not ledger.is_file():
        return None, 'ledger_missing'
    scores = []
    for observation in read(ledger):
        slot = observation.get('initial_seed_slot')
        if type(slot) is int and 0 <= slot < 64 and observation.get('failure') is None:
            score = continuous_quality_report.metric(observation)
            if score is not None:
                scores.append(score)
    return (min(scores) if scores else None), 'observations_ledger'


def window_rows(window, root=None, derived=None):
    rows = []
    for generation in window:
        for row in generation['report'].get('rows', []):
            row = dict(row, generation=generation['number'])
            if root is not None and 'best_initial_arm' not in row:
                value, source = derive_best_initial_arm(root, row)
                if source == 'observations_ledger':
                    row['best_initial_arm'] = value
                if derived is not None:
                    derived[source] = derived.get(source, 0) + 1
            rows.append(row)
    return rows


def quality_summaries(window):
    """Per-generation partition counts through the status observer's own reducer."""
    summaries, failures = {}, {}
    for generation in window:
        try:
            partitions = continuous_status.summarize_quality(generation['report'], generation['jobs'])
        except CATCH as error:
            failures[str(generation['number'])] = f'{type(error).__name__}: {error}'
            continue
        summaries[generation['number']] = dict(checkpoint=generation['checkpoint'], partitions=partitions)
    return summaries, failures


def origin_histogram(rows):
    result = dict(all={}, by_arm={})
    for row in rows:
        origin = row.get('best_final_origin', 'unavailable')
        result['all'][origin] = result['all'].get(origin, 0) + 1
        arm = result['by_arm'].setdefault(row.get('arm', 'unknown'), {})
        arm[origin] = arm.get(origin, 0) + 1
    return result


def excluding_tail_tally(root, rows):
    """Best RMSE over tail-free origins per search; from the row field when produced, else from the ledger."""
    if not rows:
        return unavailable('no quality rows in the generation window')
    tally = dict(n_rows=0, by_origin={}, differs_from_best_final=0, ledger_missing=0, unavailable_rows=0,
                 tail_contaminated=True, tail_free_origins=list(TAIL_FREE_ORIGINS), caveat=TAIL_CAVEAT, source=None)
    sources = set()
    for row in rows:
        value = row.get('best_final_excluding_shared_tail')
        if isinstance(value, dict):
            sources.add('quality_row')
        else:
            output = row.get('output')
            ledger = within(root, output) / 'observations.json' if isinstance(output, str) else None
            if ledger is None or not ledger.is_file():
                tally['ledger_missing'] += 1
                continue
            value = metric_call('best_final_excluding_shared_tail', read(ledger))
            sources.add('observations_ledger')
        if not isinstance(value, dict):
            tally['unavailable_rows'] += 1
            tally.setdefault('first_unavailable_reason', value)
            continue
        tally['n_rows'] += 1
        origin = value.get('origin')
        # A search whose best tail-free row does not exist (every survivor descends from the shared tail)
        # is a real outcome, counted under an explicit bucket so the receipt stays canonical JSON.
        origin = origin if isinstance(origin, str) else 'no_tail_free_row'
        tally['by_origin'][origin] = tally['by_origin'].get(origin, 0) + 1
        best, final = value.get('best_rmse'), row.get('best_final_gp')
        if finite(best) and finite(final) and not math.isclose(best, final, rel_tol=1e-12, abs_tol=1e-12):
            tally['differs_from_best_final'] += 1
    tally['source'] = sorted(sources)
    if tally['n_rows'] == 0:
        return unavailable('best_final_excluding_shared_tail derivable for no row: '
                           + str(tally.get('first_unavailable_reason', 'ledgers missing')))
    return tally


def teacher_beats(rows, window):
    per_generation = {}
    for row in rows:
        if row.get('arm') != 'neural':
            continue
        key = str(row['generation'])
        per_generation.setdefault(key, 0)
        final, teacher = row.get('best_final_gp'), row.get('best_retained_teacher')
        if finite(final) and finite(teacher) and strictly_better(final, teacher):
            per_generation[key] += 1
    timestamps = sorted(g['report'].get('timestamp') for g in window if finite(g['report'].get('timestamp')))
    total = sum(per_generation.values())
    result = dict(per_generation=per_generation, total=total,
                  rule='neural arm: best_final_gp < best_retained_teacher (continuous_status.better tolerance)')
    if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
        result['per_hour'] = unavailable('fewer than two timestamped quality reports; wall-hour denominator undefined')
    else:
        hours = (timestamps[-1] - timestamps[0]) / 3600
        result.update(window_hours=hours, per_hour=total / hours)
    return result


def replay_discoveries(root, run):
    directory = run / 'replay-membership'
    policies = sorted(directory.glob('*/policy.json')) if directory.is_dir() else []
    if not policies:
        return unavailable('no replay policies under ' + relative(root, directory)), []
    loaded, rows = [], []
    for path in policies:
        policy = read(path)
        fresh = policy.get('fresh_records')
        if not isinstance(fresh, dict):
            continue
        loaded.append(policy)
        rows.append(dict(policy=relative(root, path), fresh_records=sum(len(r) for r in fresh.values()),
                         fresh_groups=len(fresh), mtime=int(path.stat().st_mtime),
                         discoveries=policy.get('discoveries', 'any_new_or_improved_program')))
    if not rows:
        return unavailable('replay policies carry no fresh_records'), loaded
    total = sum(r['fresh_records'] for r in rows)
    result = dict(policies=rows, fresh_records_total=total, wall_clock_source='policy.json mtime',
                  teacher_improved_policies=sum(r['discoveries'] == 'teacher_improved' for r in rows))
    times = sorted(r['mtime'] for r in rows)
    if len(times) < 2 or times[-1] <= times[0]:
        result['per_hour'] = unavailable('fewer than two distinct replay policy times; wall-hour denominator undefined')
    else:
        hours = (times[-1] - times[0]) / 3600
        result.update(wall_hours=hours, per_hour=total / hours)
    return result, loaded


def emission_census(root, window):
    """Unique neural emissions per context and their token length from each generation's neural populations."""
    populations, per_generation, tokens, missing, all_uniques = [], {}, [], 0, []
    for generation in window:
        uniques = []
        for job in generation['jobs'].get('jobs', []):
            try:
                arm = Path(job['output']).relative_to(generation['relative']).parts[0]
            except (ValueError, KeyError):
                continue
            if arm != 'neural' or not job.get('seeds'):
                continue
            seeds = within(root, job['seeds'])
            if not seeds.is_file():
                missing += 1
                continue
            require(sha(seeds) == job.get('seeds_sha256'), 'population changed: ' + job['seeds'])
            population = read(seeds)
            genomes = {c['genome'] for c in population.get('candidates', [])
                       if c.get('origin') == 'neural_policy' and isinstance(c.get('genome'), str)}
            uniques.append(len(genomes))
            tokens.extend(len(flywheel_metrics.tokens(g)) for g in genomes)
            populations.append(dict(population, generation=generation['number'], arm=arm, output=job['output']))
        all_uniques.extend(uniques)
        per_generation[str(generation['number'])] = dict(contexts=len(uniques),
            unique_neural_per_context_median=statistics.median(uniques) if uniques else None)
    if not all_uniques:
        return unavailable('no neural populations in the generation window'), populations
    census = dict(per_generation=per_generation, populations=len(populations), missing_populations=missing,
                  token_rule=flywheel_metrics.TOKEN_RULE, unique_neural_per_context_median=statistics.median(all_uniques),
                  unique_neural_per_context_min=min(all_uniques),
                  mean_token_length=statistics.fmean(tokens) if tokens else unavailable('no neural emissions'))
    return census, populations


def train_block_seconds(root, run):
    values = []
    for path in sorted(run.glob('cycle-*/train-2/complete.json')):
        seconds = read(path).get('seconds')
        if finite(seconds):
            values.append(seconds)
    if not values:
        return unavailable('no training completion receipts under ' + relative(root, run))
    return dict(blocks=len(values), median=statistics.median(values), last=values[-1])


def cadence_seconds(window):
    timestamps = sorted(g['report'].get('timestamp') for g in window if finite(g['report'].get('timestamp')))
    if len(timestamps) < 2:
        return unavailable('fewer than two timestamped quality reports in the window')
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    return dict(intervals=len(gaps), median=statistics.median(gaps), mean=statistics.fmean(gaps))


def throughput(root, run, window):
    gp_path = root / 'runs/gp-service.json'
    result = dict(train_block_seconds=train_block_seconds(root, run), cadence_seconds=cadence_seconds(window))
    if not gp_path.is_file():
        result['gp'] = unavailable('runs/gp-service.json missing')
        return result
    gp = read(gp_path)
    row = {k: gp.get(k) for k in ('schema', 'timestamp', 'phase', 'workers', 'active', 'completed_views',
                                  'completed_evaluations', 'evaluations_per_second', 'elapsed_seconds', 'executable_sha256')}
    elapsed, views = gp.get('elapsed_seconds'), gp.get('completed_views')
    row['views_per_hour'] = (views / elapsed * 3600 if finite(elapsed) and elapsed > 0 and finite(views)
                             else unavailable('completed_views/elapsed_seconds absent or zero'))
    row['scope'] = 'service counters since the GP service started; not a controller-generation rate'
    result['gp'] = row
    return result


def ledger_evaluator(root, output):
    directory = within(root, output)
    streamed = directory / 'observations.jsonl'
    if streamed.is_file():
        with streamed.open() as stream:
            for line in stream:
                if line.strip():
                    return json.loads(line).get('evaluator_sha256')
        return None
    committed = directory / 'observations.json'
    if committed.is_file():
        rows = read(committed)
        return rows[0].get('evaluator_sha256') if rows else None
    return None


def queue(root, evaluator_sha):
    gp_path = root / 'runs/gp-service.json'
    if not gp_path.is_file():
        return unavailable('runs/gp-service.json missing')
    gp = read(gp_path)
    queued, retry = gp.get('queued'), gp.get('retry_pending')
    result = dict(queued=queued, retry_pending=retry, failed_views=gp.get('failed_views'),
                  ready=queued - retry if type(queued) is int and type(retry) is int else None,
                  evaluator_reference=evaluator_sha,
                  rule='failed_rows outputs whose ledger evaluator_sha256 differs from bindings[cpu_executable]')
    failed = gp.get('failed_rows')
    if not isinstance(failed, list):
        result['dead_letter_candidates'] = unavailable('gp-service.json carries no failed_rows')
        return result
    outputs = sorted({row.get('output') for row in failed if isinstance(row.get('output'), str)})
    candidates, missing, foreign = 0, 0, {}
    for output in outputs:
        try:
            found = ledger_evaluator(root, output)
        except CATCH:
            found = None
        if found is None:
            missing += 1
        elif found != evaluator_sha:
            candidates += 1
            foreign[found] = foreign.get(found, 0) + 1
    result.update(failed_rows=len(failed), failed_outputs=len(outputs), dead_letter_candidates=candidates,
                  ledger_missing=missing, foreign_evaluators=foreign)
    return result


def producers(root):
    result = {}
    for module in (sys.modules[__name__], flywheel_metrics, continuous_status, promotion_gate, selfplay_milestones,
                   resume_expired_cycle, experiment_runtime, scientific_metadata):
        path = Path(module.__file__).resolve()
        # Keyed by file name, not module name: the CLI runs this file as __main__ while --verify may
        # import it, and the receipt bytes must not depend on how the producer was invoked.
        result[path.stem] = dict(path=relative(root, path), sha256=sha(path))
    return result


LIVE_SECTIONS = ('throughput', 'queue')


def split_live(receipt):
    """Move the sections read from moving service files out of the immutable bytes.

    runs/gp-service.json is rewritten every two seconds, so throughput and queue can never be
    re-derived byte-for-byte; they are recorded once in a timestamped sidecar beside the receipt,
    and the receipt keeps only the pointer. Everything else in the receipt is a saved-evidence read.
    """
    live = {}
    metrics = receipt.get('metrics', {})
    for name in LIVE_SECTIONS:
        if name in metrics:
            live[name] = metrics[name]
            metrics[name] = dict(moved_to='live-snapshot sidecar', section=name,
                                 reason='read from runs/gp-service.json, a moving input; excluded from the immutable receipt bytes')
    return live


# ---------------------------------------------------------------- receipt

def build(root, manifest_path, generations, regression_receipt=None, quality_run=None):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest, binding = manifest_binding(root, manifest_path)
    executables = {label: executable_binding(root, manifest, key) for label, key in EXECUTABLES}
    monitor = monitor_binding(root, manifest, executables['monitor_legacy']['sha256'])
    run = within(root, manifest['output'])
    # The matched-contrast window may come from an earlier run of the same evaluator/contexts when the
    # live run has not yet reported enough generations; the provenance is recorded, never implied.
    quality_source = within(root, quality_run) if quality_run else run
    require(quality_source.is_dir(), 'quality run directory missing: ' + relative(root, quality_source))
    state_path = run / 'state.json'
    if state_path.is_file():
        state = read(state_path)
        checkpoints = dict(learning=checkpoint_binding(root, state.get('learning_checkpoint', state.get('checkpoint'))),
                           serving=checkpoint_binding(root, state.get('serving_checkpoint'), state.get('serving_checkpoint_sha256')))
        corpus = corpus_binding(root, state.get('learning_corpus', state.get('corpus')))
        completed_blocks = state.get('completed_blocks', unavailable('completed_blocks absent from controller state'))
    else:
        missing = unavailable('controller state missing: ' + relative(root, state_path))
        checkpoints, corpus, completed_blocks = dict(learning=missing, serving=missing), missing, missing
    label_policy = corpus['label_policy'] if isinstance(corpus, dict) else corpus
    percentile = (PERCENTILE_FORMULA if label_policy == REFERENCE_LABEL_POLICY
                  else unavailable(f'percentile formula is defined for {REFERENCE_LABEL_POLICY} only; found {label_policy}'))
    window = generation_window(root, quality_source, generations)
    derived_initial_arm = {}
    rows = window_rows(window, root, derived_initial_arm)
    raw_receipts, ignored = monitoring_receipts(root, run)
    legacy = panel_legacy(root, raw_receipts, ignored, monitor['receipt_binding'], checkpoints)
    regression_path = Path(regression_receipt) if regression_receipt else root / DEFAULT_REGRESSION_RECEIPT
    if not regression_path.is_absolute():
        regression_path = root / regression_path
    census, populations = emission_census(root, window)
    discoveries, policies = replay_discoveries(root, run)
    if window:
        contrast = lambda partition, level: metric_call('matched_seed_contrast', rows, partition, level)
        heldout = dict(n_generations=len(window), seed_only=contrast('meta_validation', 'seed_only'),
                       final=contrast('meta_validation', 'final'), tail_caveat=TAIL_CAVEAT)
        train = dict(label='secondary: meta_train contexts are the search-adapted surface, not held out',
                     n_generations=len(window), seed_only=contrast('meta_train', 'seed_only'),
                     final=contrast('meta_train', 'final'), tail_caveat=TAIL_CAVEAT)
        for block in (heldout, train):
            block['n_cuts'] = block['seed_only'].get('n_cuts') if isinstance(block['seed_only'], dict) else block['seed_only']
        improves = dict(neural=metric_call('gp_improves_on_seed', rows),
                        generic=metric_call('gp_improves_on_seed', rows, 'generic'))
        summaries, summary_failures = quality_summaries(window)
        signature = (metric_call('regression_signature', summaries) if summaries
                     else unavailable('no generation summarizes: ' + '; '.join(f'{k}: {v}' for k, v in summary_failures.items())))
        if isinstance(signature, dict):
            signature['summary_failures'] = summary_failures
        partitions = {str(k): v['partitions'] for k, v in summaries.items()}
        origins = origin_histogram(rows)
        tail = excluding_tail_tally(root, rows)
        beats = teacher_beats(rows, window)
        win_rates = selfplay_milestones.win_rates([r for r in rows if r.get('arm') == 'neural'])
    else:
        empty = unavailable('no reported generations under ' + relative(root, run))
        heldout = train = improves = signature = partitions = origins = tail = beats = win_rates = empty
    diversity = (metric_call('diversity', populations, policies) if populations
                 else unavailable('no neural populations in the generation window'))
    evidence_times = [g['report'].get('timestamp') for g in window] + [r['timestamp'] for r in legacy['series']]
    evidence_times = [t for t in evidence_times if finite(t)]
    generation_range = (dict(requested=generations, count=len(window), first=window[0]['number'], last=window[-1]['number'],
                             directories=[g['relative'] for g in window], report_sha256=[g['report_sha256'] for g in window],
                             checkpoints=[g['checkpoint'] for g in window])
                        if window else dict(requested=generations, count=0, first=None, last=None, directories=[],
                                            report_sha256=[], checkpoints=[]))
    generation_range.update(quality_source_run=relative(root, quality_source),
                            quality_source_is_live_run=quality_source == run,
                            best_initial_arm_derived=dict(derived_initial_arm,
                                rule='legacy rows without best_initial_arm take the min original-target score over their own '
                                     'initial slots 0..63 from their observation ledger (continuous_quality_report.search_row rule)'),
                            quality_source_rule='matched contrast, origins, win rates and emission census come from quality_source_run; '
                                                'frozen-panel, checkpoints, corpus and replay discoveries come from the live run')
    return dict(schema=SCHEMA,
        frozen_at=max(evidence_times) if evidence_times else unavailable('no timestamped evidence bound'),
        frozen_at_rule='max timestamp over the bound quality reports and frozen-panel receipts; wall clock never enters the receipt',
        manifest_sha256=binding['manifest_sha256'], parent_manifest_sha256=binding['parent_manifest_sha256'],
        release_revision=binding['release_revision'], run_identity=binding['run_identity'],
        executables={label: executables[label]['sha256'] for label, _ in EXECUTABLES},
        checkpoints=checkpoints,
        learning_corpus_sha256=corpus['sha256'] if isinstance(corpus, dict) else corpus,
        monitor_corpus_sha256=monitor['monitor_corpus_sha256'], panel_sha256=monitor['panel_sha256'],
        contexts_sha256=binding['contexts_sha256'], vocabulary_sha256=binding['vocabulary_sha256'],
        label_policy=label_policy, percentile_formula=percentile, metric_definitions_sha256=None,
        teaching_store_binding=store_binding(root, manifest), generation_range=generation_range,
        producers=producers(root),
        binding=dict(manifest_path=binding['manifest_path'], manifest_sha256=binding['manifest_sha256'],
            parent_manifest_path=binding['parent_manifest_path'], output=binding['output'],
            executable_paths=executables, learning_corpus=corpus, completed_blocks=completed_blocks,
            controller_state=relative(root, state_path),
            **{k: monitor[k] for k in ('monitor_corpus', 'monitor_panel', 'panel_file_sha256', 'panel_declares_corpus_sha256',
                                        'corpus_binding_matches_index', 'monitor_policy_sha256', 'receipt_binding')}),
        metrics=dict(panel_legacy=legacy,
            recovery_reference=dict(nll=selfplay_milestones.RECOVERY_NLL, bt=selfplay_milestones.RECOVERY_BT,
                                    nll_guard=selfplay_milestones.NLL_GUARD, bt_guard=selfplay_milestones.BT_GUARD,
                                    source='selfplay_milestones.py constants'),
            regression_pair=regression_pair(root, regression_path),
            panel_null=panel_null(raw_receipts, monitor['receipt_binding']),
            matched_heldout=heldout, matched_train=train, gp_improves_on_seed=improves,
            best_final_origin=origins, best_final_excluding_shared_tail=tail, regression_signature=signature,
            quality_partitions=partitions, win_rates=win_rates,
            discoveries_per_hour=dict(replay_policies=discoveries, new_teacher_beats=beats),
            diversity=diversity, emission_census=census,
            throughput=throughput(root, run, window), queue=queue(root, executables['cpu_evaluator']['sha256'])),
        scope='Saved receipts of the live release only; no fits, no training, no queue or controller writes.',
        independent_evaluation=False, final_tests_opened=False)


# ---------------------------------------------------------------- preregistration

def preregistration(path):
    """The single fenced JSON block of PREREGISTRATION.md; sha256 is the canonical-JSON digest."""
    path = Path(path)
    require(path.is_file(), 'preregistration missing: ' + str(path))
    blocks = FENCE.findall(path.read_text())
    require(len(blocks) == 1, f'PREREGISTRATION.md must carry exactly one fenced json block; found {len(blocks)}')
    block = json.loads(blocks[0])
    require(block.get('schema') == PREREGISTRATION_SCHEMA, 'unknown preregistration schema')
    require(block.get('final_tests_opened') is False, 'preregistration opened final tests')
    return dict(path=str(path), block=block, sha256=scientific_metadata.digest(block),
                block_bytes_sha256=hashlib.sha256(blocks[0].encode()).hexdigest(),
                digest_rule='sha256 of the canonical JSON (sorted keys, compact separators) of the block')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--generations', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--regression-receipt', type=Path, default=None,
                        help='parallel.continuous_status.v1 receipt carrying the monitoring history (default: ' + DEFAULT_REGRESSION_RECEIPT + ')')
    parser.add_argument('--verify', action='store_true', help='re-derive and byte-compare; never writes')
    parser.add_argument('--quality-run', type=Path, default=None,
                        help='run directory whose generation-*/quality-report.json supply the matched-contrast window '
                             '(default: the manifest output run); recorded as generation_range.quality_source_run')
    args = parser.parse_args(argv)
    receipt = build(args.root, args.manifest, args.generations, args.regression_receipt, args.quality_run)
    live = split_live(receipt)
    data = promotion_gate.canonical(receipt) + b'\n'
    if args.verify:
        require(args.output.is_file(), 'baseline receipt missing: ' + str(args.output))
        verified = args.output.read_bytes() == data
        print(json.dumps(dict(verified=verified, path=str(args.output), sha256=sha(args.output),
                              rederived_sha256=hashlib.sha256(data).hexdigest())))
        return 0 if verified else 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = promotion_gate.immutable(args.output, receipt)
    snapshot = args.output.with_name(args.output.stem + '.live-snapshot.json')
    resume_expired_cycle.atomic(snapshot, dict(schema='parallel.flywheel_baseline_live_snapshot.v1', receipt=str(args.output),
                                               receipt_sha256=digest, sections=live, final_tests_opened=False))
    print(json.dumps(dict(path=str(args.output), sha256=digest, frozen_at=receipt['frozen_at'],
                          generation_range={k: receipt['generation_range'][k] for k in ('count', 'first', 'last')},
                          manifest_sha256=receipt['manifest_sha256'], live_snapshot=str(snapshot))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
