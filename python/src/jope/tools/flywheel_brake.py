#!/usr/bin/env python3
"""Advisory flywheel brake: F2 frozen-panel guards, F3 diversity, F5 evaluator identity.

Reads the flywheel series and the immutable baseline receipt, never a live
service, and writes learning/brake.json (parallel.flywheel_brake.v1). F2 and F3
trip on evidence, not on invocation timing: any window of `consecutive` (3)
rejects on distinct monitored learning checkpoints / generations among the
decisions an operator has not acknowledged trips the brake, even when a later
accept follows before the tool runs; F5 trips on the first row-0 mismatch. A hold
latches until an operator releases it with --release, which acknowledges exactly
the rejects that produced it. The documented run-specific pause (OPERATIONS.md,
"Pause and resume") is invoked only on F2/F5 trips, only once per hold, and only
when --pause-command is given; otherwise the tool prints what it would run.
"""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from flywheel_series import baseline_receipt, clean, digest, finite, load, read, resolve, series_rows
from resume_expired_cycle import atomic, require, sha

SCHEMA = 'parallel.flywheel_brake.v1'
CONSECUTIVE = 3
F3_FRESH_FRACTION, F3_MIN_UNIQUE, F3_TOKEN_FACTOR = 0.5, 32, 1.3
PAUSE_TRIPS = ('F2', 'F5')
F2_RULE = 'reject iff sequence_nll > ref*(1+g_nll) or ranking_bt > ref*(1+g_bt) or ranking_listwise > ref*(1+g_bt); ' \
          'counted on distinct monitored learning checkpoints under the baseline monitor definitions (executable + metric ' \
          'definitions), across objective identities; trips on any window of `limit` consecutive unacknowledged rejects'
F3_RULE = 'alarm iff unique neural emissions per context median < unique_min, or unique fresh programs (latest snapshot) ' \
          '< fresh_fraction * baseline median, or mean emission tokens > token_factor * baseline mean; ' \
          'trips on any window of `limit` consecutive unacknowledged alarms'
FRESH_UNAVAILABLE = 'unavailable:no replay snapshot readable for this generation'


def learning_checkpoints(rows, ignore=()):
    """Latest monitor row per distinct non-serving checkpoint, in evaluation order."""
    latest = {}
    for row in rows:
        if row.get('kind') != 'monitor' or row.get('lineage') == 'serving' or row.get('id') in ignore:
            continue
        key = row.get('checkpoint_sha256') or row.get('checkpoint') or row['id']
        previous = latest.get(key)
        if previous is None or (row.get('timestamp') or 0) >= (previous.get('timestamp') or 0):
            latest[key] = row
    return sorted(latest.values(), key=lambda r: (r['evaluated_step'] if type(r.get('evaluated_step')) is int else -1,
                                                  r.get('timestamp') or 0, r['id']))


def tally(decisions, consecutive):
    """Trailing run of rejects, the number of non-overlapping `consecutive`-long reject windows (fires) and their ids."""
    run, fires, windows, pending = 0, 0, [], []
    for decision in decisions:
        if decision['decision'] == 'rejected':
            run += 1
            pending.append(decision['id'])
            if run % consecutive == 0:
                fires += 1
                windows.append(list(pending))
                pending = []
        else:
            run, pending = 0, []
    return run, fires, windows


def evaluate_f2(rows, baseline, consecutive=CONSECUTIVE, ignore=()):
    guards, reference = baseline.get('guards') or {}, baseline.get('panel') or {}
    g_nll, g_bt = guards.get('g_nll'), guards.get('g_bt')
    result = dict(falsifier='F2', evaluable=False, reason=None, tripped=False, fires=0, consecutive=0, windows=[], rejects=[],
                  decisions=[], skipped_incomparable=0, limit=consecutive, rule=F2_RULE,
                  guards=dict(g_nll=g_nll, g_bt=g_bt, borrowed=guards.get('borrowed')),
                  reference={k: reference.get(k) for k in ('sequence_nll', 'ranking_bt', 'ranking_listwise')})
    if not (finite(g_nll) and finite(g_bt)):
        return dict(result, reason='baseline receipt carries no finite g_nll/g_bt guards')
    if not all(finite(v) for v in result['reference'].values()):
        return dict(result, reason='baseline receipt carries no finite frozen-panel reference for the learning checkpoint')
    limits = (('sequence_nll', g_nll), ('ranking_bt', g_bt), ('ranking_listwise', g_bt))
    for row in learning_checkpoints(rows, ignore):
        panel = row.get('panel') or {}
        if not isinstance(panel.get('baseline_delta'), dict):
            result['skipped_incomparable'] += 1  # other monitor definitions (executable / metric definitions): never judged here
            continue
        exceeded = [name for name, guard in limits if finite(panel.get(name)) and panel[name] > reference[name] * (1 + guard)]
        decision = dict(id=row['id'], checkpoint_sha256=row.get('checkpoint_sha256'), evaluated_step=row.get('evaluated_step'),
                        decision='rejected' if exceeded else 'accepted', exceeded=exceeded,
                        panel={name: panel.get(name) for name, _ in limits})
        result['decisions'].append(decision)
        if exceeded:
            result['rejects'].append(dict(falsifier='F2', **decision))
    result['consecutive'], result['fires'], result['windows'] = tally(result['decisions'], consecutive)
    result['tripped'] = result['fires'] > 0   # evidence, not invocation timing: a later accept does not erase a window
    result['evaluable'] = bool(result['decisions'])
    if not result['evaluable']:
        result['reason'] = 'no comparable monitored learning checkpoints in the baseline monitor segment'
    return result


def evaluate_f3(rows, baseline, consecutive=CONSECUTIVE, fresh_fraction=F3_FRESH_FRACTION,
                unique_min=F3_MIN_UNIQUE, token_factor=F3_TOKEN_FACTOR, ignore=()):
    reference = baseline.get('diversity') if isinstance(baseline.get('diversity'), dict) else {}
    fresh_ref, tokens_ref = reference.get('unique_fresh_programs_median'), reference.get('mean_token_length')
    result = dict(falsifier='F3', evaluable=False, reason=None, tripped=False, fires=0, consecutive=0, windows=[], rejects=[],
                  decisions=[], skipped_unavailable=0, fresh_unavailable=0, limit=consecutive, rule=F3_RULE,
                  thresholds=dict(unique_min=unique_min, fresh_fraction=fresh_fraction, token_factor=token_factor),
                  reference=dict(unique_fresh_programs_median=fresh_ref, mean_token_length=tokens_ref),
                  checks=dict(unique_emissions=True, fresh_programs=finite(fresh_ref), token_length=finite(tokens_ref)))
    generations = sorted((r for r in rows if r.get('kind') == 'generation' and r.get('id') not in ignore),
                         key=lambda r: (r.get('generation') if type(r.get('generation')) is int else -1, r['id']))
    for row in generations:
        diversity = row.get('diversity')
        if not isinstance(diversity, dict):
            result['skipped_unavailable'] += 1
            continue
        unique, tokens = diversity.get('unique_neural_per_context_median'), diversity.get('mean_token_length')
        per_snapshot = diversity.get('unique_fresh_programs') or []
        fresh = per_snapshot[-1] if per_snapshot and finite(per_snapshot[-1]) else diversity.get('unique_fresh_programs_median')
        alarms = []
        if finite(unique) and unique < unique_min:
            alarms.append('unique emissions per context median %.1f < %s' % (unique, unique_min))
        if finite(fresh) and finite(fresh_ref) and fresh < fresh_fraction * fresh_ref:
            alarms.append('unique fresh programs %.1f < %.2f x baseline median %.1f' % (fresh, fresh_fraction, fresh_ref))
        if finite(tokens) and finite(tokens_ref) and tokens > token_factor * tokens_ref:
            alarms.append('mean emission tokens %.2f > %.2f x baseline %.2f' % (tokens, token_factor, tokens_ref))
        if not finite(fresh):
            result['fresh_unavailable'] += 1   # the check is active but this generation carried no snapshot value: say so
        decision = dict(id=row['id'], generation=row.get('generation'), decision='rejected' if alarms else 'accepted',
                        alarms=alarms, values=dict(unique_emissions_per_context_median=unique,
                                                   unique_fresh_programs=fresh if finite(fresh) else FRESH_UNAVAILABLE,
                                                   mean_token_length=tokens))
        result['decisions'].append(decision)
        if alarms:
            result['rejects'].append(dict(falsifier='F3', **decision))
    result['consecutive'], result['fires'], result['windows'] = tally(result['decisions'], consecutive)
    result['tripped'] = result['fires'] > 0
    result['evaluable'] = bool(result['decisions'])
    if not result['evaluable']:
        result['reason'] = 'no generation rows with diversity measurements'
    return result


def evaluate_f5(rows, ignore=()):
    episodes = [r for r in rows if r.get('kind') == 'episodes' and r.get('id') not in ignore]
    rejects = [dict(falsifier='F5', id=r['id'], generation=r.get('generation'), row0_mismatches=n)
               for r in episodes for n in [(r.get('successor') or {}).get('row0_mismatches')] if finite(n) and n > 0]
    return dict(falsifier='F5', evaluable=bool(episodes), reason=None if episodes else 'no successor episode rows in the series',
                rejects=rejects, tripped=bool(rejects), fires=len(rejects), episodes_rows=len(episodes),
                rule='any admitted episode whose row-0 RMSE is not bit-equal to the seed label trips immediately')


def decide(rows, baseline, previous, manifest_sha, now, consecutive=CONSECUTIVE):
    previous = previous if isinstance(previous, dict) and previous.get('schema') == SCHEMA else {}
    ignore = set(previous.get('acknowledged_ids') or [])
    f2 = evaluate_f2(rows, baseline, consecutive, ignore)
    f3 = evaluate_f3(rows, baseline, consecutive, ignore=ignore)
    f5 = evaluate_f5(rows, ignore)
    trips = [name for name, f in (('F5', f5), ('F2', f2), ('F3', f3)) if f['tripped']]
    last_window = lambda f: (f['windows'] or [[]])[-1]
    reasons = dict(
        F5='F5 evaluator identity: %d episode row(s) with row-0 RMSE mismatch (%s)' % (
            len(f5['rejects']), ', '.join(r['id'] for r in f5['rejects'][:3])),
        F2='F2 frozen panel: %d consecutive rejected learning checkpoints (%s), last %s exceeded %s' % (
            consecutive, ','.join(last_window(f2)), (f2['rejects'] or [{}])[-1].get('id'),
            ','.join((f2['rejects'] or [{}])[-1].get('exceeded', []))),
        F3='F3 diversity: %d consecutive generations below thresholds (%s), last %s' % (
            consecutive, ','.join(last_window(f3)), (f3['rejects'] or [{}])[-1].get('id')))
    latched = previous.get('held') is True
    if trips:
        held, reason = True, reasons[trips[0]]
        hold_id = digest(dict(trips=trips, ids=[r['id'] for f in (f5, f2, f3) for r in f['rejects']]))[:16]
    elif latched:
        held, reason, hold_id = True, 'latched: ' + str(previous.get('reason')).removeprefix('latched: '), previous.get('hold_id')
    else:
        held, reason, hold_id = False, None, None
    return dict(schema=SCHEMA, held=held, reason=reason, tripped_by=trips, hold_id=hold_id,
                latched_from_previous=bool(latched and not trips),
                rejects=f2['rejects'] + f3['rejects'] + f5['rejects'], windows=dict(F2=f2['windows'], F3=f3['windows']),
                consecutive=dict(F2=f2['consecutive'], F3=f3['consecutive']), fires=dict(F2=f2['fires'], F3=f3['fires'], F5=f5['fires']),
                falsifiers=dict(F2=f2, F3=f3, F5=f5), limit=consecutive,
                acknowledged_ids=sorted(ignore), released=previous.get('released'),
                manifest_sha256=manifest_sha, baseline_sha256=baseline.get('sha256'), series_rows=len(rows),
                timestamp=now, final_tests_opened=False)


def release(previous, note, now):
    """Operator-only: clear the hold and acknowledge every reject that produced it."""
    require(isinstance(previous, dict) and previous.get('schema') == SCHEMA, 'no brake state to release')
    require(previous.get('held') is True, 'brake is not held')
    require(isinstance(note, str) and note.strip(), 'release requires a non-empty operator note')
    acknowledged = sorted(set(previous.get('acknowledged_ids') or []) | {r['id'] for r in previous.get('rejects') or []})
    return dict(previous, held=False, reason=None, tripped_by=[], hold_id=None, latched_from_previous=False,
                acknowledged_ids=acknowledged, timestamp=now,
                released=dict(note=note, timestamp=now, previous_reason=previous.get('reason'), hold_id=previous.get('hold_id')))


def pause(command, brake, previous, timeout=600):
    """Invoke the documented run-specific pause once per F2/F5 hold; F3 alone never pauses."""
    previous_pause = (previous or {}).get('pause') if isinstance(previous, dict) else None
    previous_pause = previous_pause if isinstance(previous_pause, dict) else {}
    due = brake['held'] and any(t in PAUSE_TRIPS for t in brake['tripped_by'])
    if not due:
        reason = 'no F2/F5 trip' if not brake['held'] else 'hold is F3-only or latched; the pause is not repeated'
        return dict(due=False, invoked=False, dry_run=not command, hold_id=brake['hold_id'], reason=reason)
    if previous_pause.get('invoked') and previous_pause.get('hold_id') == brake['hold_id']:
        return dict(previous_pause, repeated=False, reason='already invoked for this hold')
    if not command:
        return dict(due=True, invoked=False, dry_run=True, hold_id=brake['hold_id'],
                    reason='no --pause-command given; would invoke the documented controller --pause for this run')
    argv = shlex.split(command)
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return dict(due=True, invoked=True, dry_run=False, hold_id=brake['hold_id'], command=argv,
                returncode=completed.returncode, stdout=completed.stdout[-2000:], stderr=completed.stderr[-2000:],
                timestamp=int(time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='immutable parallel.flywheel_baseline.v1 receipt')
    parser.add_argument('--series', type=Path, help='default <run>/flywheel-series.jsonl')
    parser.add_argument('--brake', type=Path, help='default <run>/brake.json')
    parser.add_argument('--consecutive', type=int, default=CONSECUTIVE)
    parser.add_argument('--pause-command', help='documented pause, e.g. "python3 <controller> --root R --manifest M --pause"')
    parser.add_argument('--release', metavar='NOTE', help='operator release of a held brake; acknowledges its rejects')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = resolve(root, args.manifest)
    manifest = read(manifest_path)
    run = root / manifest['output']
    series = resolve(root, args.series) if args.series else run / 'flywheel-series.jsonl'
    brake_path = resolve(root, args.brake) if args.brake else run / 'brake.json'
    previous = load(brake_path)
    now = int(time.time())
    if args.release:
        brake = release(previous, args.release, now)
        brake['pause'] = dict(due=False, invoked=False, reason='released by operator')
    else:
        baseline = baseline_receipt(resolve(root, args.baseline))
        brake = decide(series_rows(series), baseline, previous, sha(manifest_path), now, args.consecutive)
        brake['pause'] = pause(args.pause_command, brake, previous)
    atomic(brake_path, clean(brake))
    print(json.dumps(dict(brake=str(brake_path), held=brake['held'], reason=brake['reason'], tripped_by=brake['tripped_by'],
                          consecutive=brake.get('consecutive'), pause=brake['pause'])))


if __name__ == '__main__':
    main()
