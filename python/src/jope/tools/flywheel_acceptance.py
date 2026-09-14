#!/usr/bin/env python3
"""Acceptance receipt for the recovery flywheel: pre-registered pass/fail gates over live evidence only.

The owner's "final approval" is the first receipt whose verdict is `approved`; it prints `approved` only when EVERY gate
passes on fresh evidence, otherwise `not_approved` with the failing gates named. Gates (rules in GATE_RULES): LOOP,
CADENCE, THROUGHPUT, SCIENCE, PUBLICATION, SYNC_TRUST, TUI. Evidence is collected read-only: one `python3 -` snippet on the
node (files by sha256/bytes/mtime, small JSON parsed, a bounded 8 MiB journal tail, the needed fields of state.json read
once, tmux/systemctl facts for the console) plus local facts (receipt mirror, node mirror receipts, git remote head).
Narration (.md, chat) is never evidence. `pending` and `not_applicable` never count as pass except the GPU checks while
the active manifest has training disabled AND progress.idle_explanation says so. Receipts are immutable
(parallel.flywheel_acceptance.v1) under <output>/acceptance-<utc>.json; latest.json binds the newest by sha256.

  flywheel_acceptance.py --root /home/ubuntu/jope-dime/parallel --output parallel/recovery/<run-id>/successor/acceptance [--since-generation N] [--dry-run] [--publish-node]
"""
import argparse
import datetime
import json
import os
import stat
import subprocess
import time
from pathlib import Path

import mirror_learning_receipts as mirror
import promotion_gate
from resume_expired_cycle import atomic, read, require, sha

SCHEMA = 'parallel.flywheel_acceptance.v1'
LATEST_SCHEMA = 'parallel.flywheel_acceptance_latest.v1'
GATES = ('LOOP', 'CADENCE', 'THROUGHPUT', 'SCIENCE', 'PUBLICATION', 'SYNC_TRUST', 'TUI')
# every mirror class whose newest receipt is evidence (the first three bound SYNC_TRUST freshness; the rest are the 2026-09-14 migration classes)
MIRROR_CLASSES = ('receipts', 'champion-git', 'champions', 'weights', 'dataset', 'prepared', 'contexts', 'teaching-corpus', 'results', 'observations', 'node-source', 'node-release', 'node-home',
                  'kernel-mirror', 'recovery-20260908', 'corpus', 'assets', 'kernel-work', 'runs-census', 'recovery-run-census')
GATE_RULES = dict(
    LOOP='rollback reparented (learning head == re-parent target, no f11_deviation); progress.automatic_training == manifest; once training is enabled a train commit + monitor verdict within the newest 3 generations and >= 2 connected lineage transitions',
    CADENCE='every generation older than the newest two has all its searches and a quality report; no generation older than 90 min without a report',
    THROUGHPUT='rolling-hour audit <= 2 h old: GP allocated-core busy while ready >= 0.80, no ready starvation; with training enabled: GPU task occupancy >= 0.90, mean util >= 0.80 (fraction), handoff p95 < 30 s; every idle interval explained',
    SCIENCE='F2 silent (ranking_bt <= baseline * (1 + r), not held); verdict receipts at post-resume generations 5/10/20; no improvement claim without a smarter_confirmed verdict; <= 1 hold in the 20 post-release generations',
    PUBLICATION='every strict high mark committed on the node <= 5 min, mirrored to xbabe2 <= 10 min, pushed + remote-verified <= 10 min; code repo local head == remote head',
    SYNC_TRUST='receipt mirror verifies locally and on the node (both-host), pull receipt <= 30 min old with no refusals; node mirror receipts fresh with no drift; restore drill <= 24 h',
    TUI='console unit active, exactly one live pane, renderer --watch, frame age < 30 s, headline feed <= 15 min old and bound to its receipt, CI90 / high marks / git rows visible natively',
)
THRESHOLDS = dict(report_age_seconds=5400, audit_age_seconds=7200, gp_busy=0.80, gpu_occupancy=0.90, gpu_util=0.80, handoff_p95=30.0, guard_ratio=0.01,
                  commit_seconds=300, mirror_seconds=600, publish_seconds=600, pull_age_seconds=1800, restore_age_seconds=86400, frame_age_seconds=30,
                  headline_age_seconds=900, node_mirror_age_seconds=900)
JOURNAL_TAIL_BYTES = 8 << 20

REMOTE_SCRIPT = r'''
import glob, hashlib, json, os, subprocess, time
root = %(root)r; run = %(run)r; tail_bytes = %(tail)d; small = 1 << 20
now = time.time()
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for piece in iter(lambda: f.read(1 << 20), b''): h.update(piece)
    return h.hexdigest()
def describe(rel, parse=True):
    p = os.path.join(root, rel)
    try: st = os.stat(p)
    except OSError: return None
    row = dict(sha256=digest(p), bytes=st.st_size, mtime=st.st_mtime)
    if parse and st.st_size <= small:
        try: row['json'] = json.load(open(p))
        except Exception: pass
    return row
out = dict(now=now, files={}, run=run)
for rel in %(allow)r:
    for p in sorted(glob.glob(os.path.join(root, rel))):
        r = os.path.relpath(p, root); d = describe(r)
        if d: out['files'][r] = d
# state.json once, only the fields the gates need
sp = os.path.join(root, run, 'learning/state.json')
try:
    st = json.load(open(sp)); gens = st.get('generations') or []
    def gen_row(g):
        path = os.path.join(root, g['path']); spec = os.path.join(path, 'generation.json')
        jobs = os.path.join(path, 'jobs.json'); n_jobs = None
        if os.path.exists(jobs):
            try: n_jobs = len(json.load(open(jobs)).get('jobs', []))
            except Exception: n_jobs = None
        searches = sum(1 for _ in glob.glob(os.path.join(path, '*/*/cut-*/search/search.json')))
        return dict(number=g.get('number'), stage=g.get('stage'), checkpoint_sha256=g.get('checkpoint_sha256'), jobs=n_jobs, searches=searches,
                    report=os.path.exists(os.path.join(path, 'quality-report.json')), frozen_at=os.path.getmtime(spec) if os.path.exists(spec) else None)
    lc = st.get('learning_checkpoint'); lcs = os.path.join(root, lc, 'state.json') if isinstance(lc, str) else None
    out['state'] = dict(manifest_sha256=st.get('manifest_sha256'), learning_checkpoint=lc, learning_checkpoint_state_sha256=digest(lcs) if lcs and os.path.exists(lcs) else None,
                        learning_corpus=st.get('learning_corpus'), training_diagnostic=st.get('training_diagnostic'), training_growth=st.get('training_growth'),
                        completed_blocks=st.get('completed_blocks'), serving_status=st.get('serving_status'), generations=[gen_row(g) for g in gens[-12:]], generation_count=len(gens))
    out['files'][os.path.join(run, 'learning/state.json')] = dict(sha256=None, bytes=os.path.getsize(sp), mtime=os.path.getmtime(sp), note='fields extracted, not hashed (large)')
except Exception as e:
    out['state_error'] = str(e)[:200]
# bounded journal tail
jp = os.path.join(root, run, 'learning/journal.jsonl'); events = []
try:
    size = os.path.getsize(jp)
    with open(jp, 'rb') as f:
        off = max(0, size - tail_bytes); f.seek(off)
        if off: f.readline()
        for raw in f:
            if not raw.endswith(b'\n'): break
            try: e = json.loads(raw)
            except ValueError: continue
            events.append({k: e.get(k) for k in ('timestamp', 'event', 'kind', 'checkpoint_sha256', 'new_manifest_sha256', 'reason', 'generation') if k in e})
    out['journal'] = dict(bytes=size, window=min(size, tail_bytes), events=events)
except OSError as e:
    out['journal'] = dict(error=str(e)[:200], events=[])
# champion store: newest event mtime (one directory walk)
newest = 0.0; count = 0
try:
    with os.scandir(os.path.join(root, 'runs/champions')) as it:
        for d in it:
            if not d.is_dir(): continue
            try:
                with os.scandir(d.path) as inner:
                    for f in inner:
                        if f.name.startswith('event-'):
                            count += 1; m = f.stat().st_mtime
                            if m > newest: newest = m
            except OSError: pass
except OSError: pass
out['champions'] = dict(newest_event_mtime=newest or None, events=count)
# console facts
def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout
    except Exception as e: return ''
out['console'] = dict(unit=sh('systemctl show jope-eight-a100-console.service -p ActiveState -p MainPID -p InvocationID'),
                      panes=sh("tmux -S /run/jope-eight-a100-console/tmux.sock list-panes -a -F '#{session_name}\t#{pane_dead}\t#{pane_pid}' 2>/dev/null"),
                      frame=sh('tmux -S /run/jope-eight-a100-console/tmux.sock capture-pane -p -t jope-recovery 2>/dev/null'))
# both-host re-hash of the mirrored small files (list supplied by the caller)
out['rehash'] = {rel: (digest(os.path.join(root, rel)) if os.path.exists(os.path.join(root, rel)) else None) for rel in %(rehash)r}
print(json.dumps(out))
'''

ALLOW = ('{run}/learning/progress.json', '{run}/learning/control.json', 'runs/gp-service.json', '{run}/health/latest.json', '{run}/health/summary.json',
         '{run}/review/measured-performance-audit-latest.json', '{run}/review/flywheel-acceptance.json', '{run}/successor/decisions/f11-first-monitor-readout.json',
         '{run}/successor/decisions/f11-rollback-status.json', '{run}/successor/scoreboard/latest.json', '{run}/successor/scoreboard/headline/*.json',
         '{run}/successor/scoreboard/champion-git.json', '{run}/successor/verdicts/*.json', '{run}/successor/publication/gp-publication.json', '{run}/learning/monitoring/*.json')


def collect_node(node, key, root, run, rehash_list):
    script = REMOTE_SCRIPT % dict(root=root, run=run, tail=JOURNAL_TAIL_BYTES, allow=[a.format(run=run) for a in ALLOW], rehash=rehash_list)
    result = mirror.remote_python(node, key, script, timeout=900)
    require(result.returncode == 0, 'node evidence collection failed: ' + result.stderr[-400:])
    return json.loads(result.stdout)


def collect_local(root, mirror_dir, node_mirror, repo, credentials):
    """Local facts: receipt-mirror manifest verification, pull receipt, node-mirror receipts, code repo remote head, credential mode."""
    out = dict(mirror=None, pull=None, node_mirror={}, git=None, credentials=None, files={})
    manifest = Path(mirror_dir) / 'learning-receipts/MANIFEST.json'
    if manifest.is_file():
        verify = mirror.verify(mirror_dir)
        out['mirror'] = dict(files=verify['files'], mismatched=verify['mismatched'], manifest_sha256=sha(manifest))
        out['files'][str(manifest)] = dict(host='xbabe2', sha256=sha(manifest), bytes=manifest.stat().st_size, mtime=manifest.stat().st_mtime)
        small = [rel for rel, entry in read(manifest)['files'].items() if not entry.get('stub') and not entry.get('tail_lines') and entry.get('bytes', 0) <= mirror.NODE_VERIFY_LIMIT_BYTES and not mirror.mutable(rel)]
        out['mirror']['rehash_expected'] = {rel: read(manifest)['files'][rel]['sha256'] for rel in small}
    pull = Path(mirror_dir) / 'learning-receipts/last-pull.json'
    if pull.is_file():
        out['pull'] = read(pull)
        out['files'][str(pull)] = dict(host='xbabe2', sha256=sha(pull), bytes=pull.stat().st_size, mtime=pull.stat().st_mtime)
    receipts = Path(node_mirror) / 'receipts'
    if receipts.is_dir():
        # mirror receipts live under receipts/mirror/<utc>-<class>-<sha12>.json (one per class per run); restore drills under receipts/restore/
        for cls in MIRROR_CLASSES:
            newest = sorted((receipts / 'mirror').glob('*-%s-*.json' % cls), key=lambda p: p.stat().st_mtime) if (receipts / 'mirror').is_dir() else []
            if newest:
                p = newest[-1]
                out['node_mirror'][cls] = dict(path=str(p), mtime=p.stat().st_mtime, value=read(p))
                out['files'][str(p)] = dict(host='xbabe2', sha256=sha(p), bytes=p.stat().st_size, mtime=p.stat().st_mtime)
        # lane import receipts (captures into the jain-datasets Store): newest per class under receipts/lane/<utc>-<class>[-<sha12>].json
        out['lane'] = {}
        if (receipts / 'lane').is_dir():
            for p in sorted((receipts / 'lane').glob('*.json'), key=lambda p: p.stat().st_mtime):
                value = read(p)
                cls = value.get('class')
                if value.get('schema') == 'jain.champion_capture_receipt.v1':   # the champion stream receipt carries identity counts, not a class
                    cls = 'champion-events'
                    value = dict(value, units=value.get('identities'), counts=dict(captured=value.get('captured', 0), already_bound=value.get('already_bound', 0), failed=value.get('failed', 0),
                                                                                    refused_partition=value.get('refused_partition', 0)))
                if cls:
                    out['lane'][cls] = dict(path=str(p), mtime=p.stat().st_mtime, value=value)
                    out['files'][str(p)] = dict(host='xbabe2', sha256=sha(p), bytes=p.stat().st_size, mtime=p.stat().st_mtime)
        if out['node_mirror']:
            out['node_mirror']['mirror'] = min(out['node_mirror'].values(), key=lambda r: r['mtime'])   # the oldest class receipt bounds freshness
        restore = sorted((receipts / 'restore').glob('*.json'), key=lambda p: p.stat().st_mtime) if (receipts / 'restore').is_dir() else []
        if restore:
            p = restore[-1]
            out['node_mirror']['restore'] = dict(path=str(p), mtime=p.stat().st_mtime, value=read(p))
            out['files'][str(p)] = dict(host='xbabe2', sha256=sha(p), bytes=p.stat().st_size, mtime=p.stat().st_mtime)
    try:
        head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'workspace-canonical'], capture_output=True, text=True, timeout=30).stdout.strip()
        remotes = subprocess.run(['git', '-C', str(repo), 'remote'], capture_output=True, text=True, timeout=30).stdout.split()
        remote_head = None
        if 'forge' in remotes:
            listed = subprocess.run(['git', '-C', str(repo), '-c', 'credential.interactive=false', 'ls-remote', 'forge', 'refs/heads/workspace-canonical'],
                                    capture_output=True, text=True, timeout=60, env=dict(os.environ, GIT_TERMINAL_PROMPT='0')).stdout.split()
            remote_head = listed[0] if listed else None
        out['git'] = dict(head=head or None, remotes=remotes, remote_head=remote_head)
    except Exception as e:   # git absent or timing out is evidence, not a crash
        out['git'] = dict(error=str(e)[:200])
    if credentials:
        try:
            info = os.stat(credentials)
            out['credentials'] = dict(path=str(credentials), exists=True, mode_0600=stat.S_IMODE(info.st_mode) == 0o600, owner_ok=info.st_uid == os.getuid())
        except OSError:
            out['credentials'] = dict(path=str(credentials), exists=False)
    return out


def check(name, ok, value=None, threshold=None, evidence=(), note=None):
    status = 'pass' if ok is True else 'fail' if ok is False else str(ok)   # 'pending' / 'not_applicable'
    return dict(name=name, status=status, value=value, threshold=threshold, evidence=list(evidence), note=note)


def gate(checks):
    statuses = {c['status'] for c in checks}
    if 'fail' in statuses:
        status = 'fail'
    elif 'pending' in statuses:
        status = 'pending'
    else:
        status = 'pass'   # only pass / not_applicable remain
    return dict(status=status, checks=checks)


def numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def node_file(ev, rel):
    return ((ev.get('node') or {}).get('files') or {}).get(rel)


def node_json(ev, rel):
    f = node_file(ev, rel)
    return f.get('json') if f else None


def gate_loop(ev, run):
    node = ev['node']; state = node.get('state') or {}; now = node['now']
    status = node_json(ev, f'{run}/successor/decisions/f11-rollback-status.json') or {}
    readout = node_json(ev, f'{run}/successor/decisions/f11-first-monitor-readout.json') or {}
    progress = node_json(ev, f'{run}/learning/progress.json') or {}
    checks = []
    target = (readout.get('reparent_to') or {}).get('checkpoint_sha256')
    head = state.get('learning_checkpoint_state_sha256')
    if readout.get('outcome') == 'rollback_required':
        checks.append(check('reparented', status.get('status') == 'reparented' and head == target, dict(status=status.get('status'), head=head, target=target),
                            'status reparented and learning head == target', [f'{run}/successor/decisions/f11-rollback-status.json', f'{run}/learning/state.json']))
        checks.append(check('no_f11_deviation', not status.get('deviations'), len(status.get('deviations') or []), 0, [f'{run}/successor/decisions/f11-rollback-status.json']))
        checks.append(check('status_fresh', (now - (node_file(ev, f'{run}/successor/decisions/f11-rollback-status.json') or {}).get('mtime', 0)) <= THRESHOLDS['pull_age_seconds'],
                            None, THRESHOLDS['pull_age_seconds'], [f'{run}/successor/decisions/f11-rollback-status.json'], 'the rollback feed must be recomputed within 30 min'))
    else:
        checks.append(check('rollback_readout', 'pending', readout.get('outcome'), 'a decided read-out', [f'{run}/successor/decisions/f11-first-monitor-readout.json'], 'no decided rollback read-out found'))
    manifest_auto = (status.get('manifest') or {}).get('automatic_training')
    checks.append(check('automatic_training_consistent', progress.get('automatic_training') == manifest_auto if manifest_auto is not None else 'pending',
                        dict(progress=progress.get('automatic_training'), manifest=manifest_auto), 'equal', [f'{run}/learning/progress.json']))
    events = (node.get('journal') or {}).get('events') or []
    frozen = [e for e in events if e.get('event') == 'generation_frozen']
    since = frozen[-3]['timestamp'] if len(frozen) >= 3 else None
    if manifest_auto is True:
        trains = [e for e in events if e.get('event') == 'stage_committed' and e.get('kind') == 'train' and since is not None and e['timestamp'] >= since]
        verdicts = [e for e in events if e.get('event') in ('diagnostic_accepted', 'diagnostic_rejected') and since is not None and e['timestamp'] >= since]
        checks.append(check('block_within_3_generations', bool(trains) and bool(verdicts), dict(trains=len(trains), verdicts=len(verdicts)), '>= 1 each', [f'{run}/learning/journal.jsonl']))
        acceptance = node_json(ev, f'{run}/review/flywheel-acceptance.json') or {}
        transitions = acceptance.get('connected_transitions', acceptance.get('transitions'))
        checks.append(check('connected_transitions', (transitions >= 2) if numeric(transitions) else 'pending', transitions, '>= 2', [f'{run}/review/flywheel-acceptance.json']))
    else:
        checks.append(check('training_enabled', 'pending', manifest_auto, True, [f'{run}/successor/decisions/f11-rollback-status.json'], 'training disabled by the F11 revision; loop gate cannot pass until v11'))
    return gate(checks)


def gate_cadence(ev, run):
    node = ev['node']; now = node['now']; gens = (node.get('state') or {}).get('generations') or []
    checks = []
    older = gens[:-2] if len(gens) > 2 else []
    incomplete = [g['number'] for g in older if g.get('jobs') is None or g.get('searches', 0) < g['jobs'] or not g.get('report')]
    checks.append(check('older_generations_complete', not incomplete, incomplete, [], [f'{run}/learning/generation-*/quality-report.json'], 'generations older than the newest two lacking all searches or a report'))
    stale = [g['number'] for g in gens if not g.get('report') and g.get('frozen_at') and now - g['frozen_at'] > THRESHOLDS['report_age_seconds']]
    checks.append(check('no_report_older_than_90min', not stale, stale, [], [f'{run}/learning/generation-*/generation.json']))
    if not gens:
        checks.append(check('generations_visible', 'pending', 0, '>= 1', [f'{run}/learning/state.json']))
    return gate(checks)


def gate_throughput(ev, run):
    node = ev['node']; now = node['now']
    audit_file = node_file(ev, f'{run}/review/measured-performance-audit-latest.json') or {}
    audit = audit_file.get('json') or {}
    hour = audit.get('rolling_full_hour') or {}
    progress = node_json(ev, f'{run}/learning/progress.json') or {}
    gp = (node_json(ev, 'runs/gp-service.json') or {})
    checks = []
    age = now - audit_file.get('mtime', 0) if audit_file else None
    checks.append(check('audit_fresh', (age is not None and age <= THRESHOLDS['audit_age_seconds']), age, THRESHOLDS['audit_age_seconds'], [f'{run}/review/measured-performance-audit-latest.json']))
    busy = hour.get('allocated_core_utilization_while_ready')
    checks.append(check('gp_busy_while_ready', (busy >= THRESHOLDS['gp_busy']) if numeric(busy) else 'pending', busy, THRESHOLDS['gp_busy'], [f'{run}/review/measured-performance-audit-latest.json']))
    checks.append(check('gp_active_now', (gp.get('active', 0) >= 100) if numeric(gp.get('active')) else 'pending', gp.get('active'), '>= 100 of 104', ['runs/gp-service.json']))
    status = node_json(ev, f'{run}/successor/decisions/f11-rollback-status.json') or {}
    training_off = (status.get('manifest') or {}).get('automatic_training') is False
    explanation = str(progress.get('idle_explanation') or '')
    if training_off and ('await' in explanation or 'disabled' in explanation or 'qualified replay' in explanation):
        checks.append(check('gpu_targets', 'not_applicable', explanation[:120], None, [f'{run}/learning/progress.json'], 'training disabled by the active manifest; GPU targets apply once training resumes'))
    else:
        occ, util, p95 = hour.get('occupancy'), hour.get('mean_gpu_utilization_missing_zero'), hour.get('handoff_p95_sampled_upper_gap_seconds')
        checks.append(check('gpu_occupancy', (occ >= THRESHOLDS['gpu_occupancy']) if numeric(occ) else 'pending', occ, THRESHOLDS['gpu_occupancy'], [f'{run}/review/measured-performance-audit-latest.json']))
        checks.append(check('gpu_mean_util', (util >= THRESHOLDS['gpu_util']) if numeric(util) else 'pending', util, THRESHOLDS['gpu_util'], [f'{run}/review/measured-performance-audit-latest.json']))
        checks.append(check('handoff_p95', (p95 < THRESHOLDS['handoff_p95']) if numeric(p95) else 'pending', p95, THRESHOLDS['handoff_p95'], [f'{run}/review/measured-performance-audit-latest.json']))
    events = (node.get('journal') or {}).get('events') or []
    opened = sum(1 for e in events if e.get('event') == 'unexplained_gpu_idle_incident'); resolved = sum(1 for e in events if e.get('event') == 'gpu_idle_incident_resolved')
    checks.append(check('idle_explained', opened <= resolved or bool(progress.get('idle_explanation')), dict(opened=opened, resolved=resolved, explanation=explanation[:80]), 'no open unexplained incident', [f'{run}/learning/journal.jsonl', f'{run}/learning/progress.json']))
    return gate(checks)


def gate_science(ev, run, since_generation):
    node = ev['node']; state = node.get('state') or {}; guard = state.get('training_diagnostic') or {}
    checks = []
    bt, base = guard.get('ranking_bt'), guard.get('baseline_ranking_bt')
    limit = base * (1 + THRESHOLDS['guard_ratio']) if numeric(base) else None
    checks.append(check('f2_silent', (bt <= limit and not guard.get('held')) if numeric(bt) and limit is not None else 'pending', dict(ranking_bt=bt, limit=limit, held=guard.get('held')), 'bt <= baseline*(1+r), not held', [f'{run}/learning/state.json']))
    events = (node.get('journal') or {}).get('events') or []
    holds = [e for e in events if e.get('event') == 'diagnostic_hold_acknowledged']
    checks.append(check('at_most_one_hold_in_window', len(holds) <= 1, len(holds), '<= 1 (journal tail window)', [f'{run}/learning/journal.jsonl']))
    verdict_files = {rel: f for rel, f in (node.get('files') or {}).items() if rel.startswith(f'{run}/successor/verdicts/')}
    gens = state.get('generation_count') or 0
    if since_generation is None:
        checks.append(check('verdict_milestones', 'pending', None, 'origin generation unset', [], 'pass --since-generation once the post-resume origin exists'))
    else:
        due = [since_generation + k for k in (5, 10, 20) if since_generation + k <= gens - 1]
        missing = [g for g in due if f'{run}/successor/verdicts/generation-{g:06d}.json' not in verdict_files]
        checks.append(check('verdict_milestones', not missing if due else 'pending', dict(due=due, missing=missing), 'receipt per due milestone', [f'{run}/successor/verdicts/']))
    latest = node_json(ev, f'{run}/successor/scoreboard/latest.json') or {}
    claim = str((latest.get('claims') or {}).get('independent_improvement') or 'not_established')
    if claim.startswith('established'):
        digest = claim.split(':', 1)[-1]
        checks.append(check('improvement_claim_backed', any(f.get('sha256') == digest and (f.get('json') or {}).get('outcome') == 'smarter_confirmed' for f in verdict_files.values()), claim, 'verdict receipt with smarter_confirmed', list(verdict_files)))
    else:
        checks.append(check('improvement_claim_backed', True, claim, 'no claim or a backed claim', [f'{run}/successor/scoreboard/latest.json']))
    return gate(checks)


def gate_publication(ev, run):
    node = ev['node']; now = node['now']; local = ev['local']
    feed = node_json(ev, f'{run}/successor/scoreboard/champion-git.json') or {}
    newest = (node.get('champions') or {}).get('newest_event_mtime')
    checks = []
    synced_at = (feed.get('last_sync') or {}).get('at') or feed.get('at')
    committed_lag = (newest - synced_at) if numeric(newest) and numeric(synced_at) and newest > synced_at else 0.0
    checks.append(check('committed_within_5min', feed.get('uncommitted_events') == 0 and committed_lag <= THRESHOLDS['commit_seconds'] and (now - (synced_at or 0)) <= THRESHOLDS['commit_seconds'] * 2,
                        dict(uncommitted=feed.get('uncommitted_events'), lag=committed_lag, sync_age=(now - synced_at) if synced_at else None), THRESHOLDS['commit_seconds'], [f'{run}/successor/scoreboard/champion-git.json', 'runs/champions']))
    mirror_receipt = (local.get('node_mirror') or {}).get('champion-git') or (local.get('node_mirror') or {}).get('mirror')
    checks.append(check('mirrored_within_10min', (now - mirror_receipt['mtime'] <= THRESHOLDS['mirror_seconds']) if mirror_receipt else 'pending', (now - mirror_receipt['mtime']) if mirror_receipt else None, THRESHOLDS['mirror_seconds'], [mirror_receipt['path']] if mirror_receipt else [], None if mirror_receipt else 'no node-mirror receipt on xbabe2 yet'))
    remote = (feed.get('remote') or {}).get('status')
    publication = node_json(ev, f'{run}/successor/publication/gp-publication.json') or {}
    verified = publication.get('scope') == 'live' and publication.get('lag_p95_24h') is not None and publication['lag_p95_24h'] <= THRESHOLDS['publish_seconds']
    checks.append(check('pushed_and_verified', True if verified else ('pending' if remote in (None, 'no_remote', 'no_credentials') else False), dict(champion_remote=remote, publication_scope=publication.get('scope'), lag_p95_24h=publication.get('lag_p95_24h')), THRESHOLDS['publish_seconds'],
                        [f'{run}/successor/publication/gp-publication.json', f'{run}/successor/scoreboard/champion-git.json'], None if verified else 'no live publication feed; forge access pending'))
    git = local.get('git') or {}
    checks.append(check('code_repo_published', (git.get('remote_head') == git.get('head')) if git.get('remote_head') else 'pending', dict(head=(git.get('head') or '')[:12], remote_head=(git.get('remote_head') or '')[:12], remotes=git.get('remotes')), 'remote head == local head', ['git ls-remote forge'], None if git.get('remote_head') else 'no forge remote / not reachable'))
    return gate(checks)


def gate_sync_trust(ev, run):
    node = ev['node']; now = node['now']; local = ev['local']
    checks = []
    m = local.get('mirror')
    checks.append(check('local_mirror_verified', (m['mismatched'] == []) if m else 'pending', (len(m['mismatched']) if m else None), 0, [str(Path(ev['mirror_dir']) / 'learning-receipts/MANIFEST.json')]))
    expected = (m or {}).get('rehash_expected') or {}
    rehash = node.get('rehash') or {}
    mismatched = [rel for rel, digest in expected.items() if rehash.get(rel) is not None and rehash[rel] != digest]
    checks.append(check('node_rehash_matches', (not mismatched) if expected else 'pending', dict(checked=len(expected), mismatched=mismatched[:10]), 0, [f'{run}/learning/monitoring', f'{run}/successor']))
    pull = local.get('pull')
    # the Rust receipts-class mirror receipt (both-host verified, no drift) is an equivalent pull receipt; the fresher of the two counts
    rust_receipts = ((local.get('node_mirror') or {}).get('receipts') or {})
    rv = rust_receipts.get('value') or {}
    if rust_receipts and rv.get('node_rehash_verified') is True and (rv.get('drift_refused') or []) == [] and (not pull or rv.get('at', 0) > pull.get('at', 0)):
        pull = dict(at=rv.get('at'), refused=[], source=rust_receipts.get('path'))
    checks.append(check('pull_receipt_fresh', (now - pull['at'] <= THRESHOLDS['pull_age_seconds'] and pull.get('refused') == []) if pull else 'pending', (now - pull['at']) if pull else None, THRESHOLDS['pull_age_seconds'], [str(Path(ev['mirror_dir']) / 'learning-receipts/last-pull.json')]))
    classes = {k: v for k, v in (local.get('node_mirror') or {}).items() if k not in ('mirror', 'restore')}
    fast = [classes[k] for k in ('receipts', 'champion-git', 'champions') if k in classes]
    drift = sum(len((r['value'] or {}).get('drift_refused') or []) for r in classes.values())
    unverified = [k for k, r in classes.items() if (r['value'] or {}).get('node_rehash_verified') is not True]
    fresh = bool(fast) and len(fast) == 3 and all(now - r['mtime'] <= THRESHOLDS['node_mirror_age_seconds'] for r in fast)
    checks.append(check('node_mirror_fresh_no_drift', (fresh and drift == 0 and not unverified) if classes else 'pending',
                        dict(classes=sorted(classes), fast_ages=[round(now - r['mtime']) for r in fast], drift=drift, unverified=unverified), THRESHOLDS['node_mirror_age_seconds'],
                        [r['path'] for r in classes.values()], None if classes else 'no node-mirror receipt yet'))
    restore = (local.get('node_mirror') or {}).get('restore')
    # only a forge-scope drill (fresh clone + LFS fetch + re-hash) passes; a local_store drill proves the Store can give bytes back and stays pending
    forge_scope = bool(restore) and (restore['value'].get('scope') or 'forge') == 'forge'
    checks.append(check('restore_drill', (now - restore['mtime'] <= THRESHOLDS['restore_age_seconds'] and restore['value'].get('passed') is True) if (restore and forge_scope) else 'pending', (now - restore['mtime']) if restore else None, THRESHOLDS['restore_age_seconds'], [restore['path']] if restore else [], None if restore else 'no restore drill yet'))
    return gate(checks)


def gate_tui(ev, run):
    node = ev['node']; now = node['now']; console = node.get('console') or {}
    checks = []
    unit = dict(line.split('=', 1) for line in console.get('unit', '').split('\n') if '=' in line)
    checks.append(check('unit_active', unit.get('ActiveState') == 'active', unit.get('ActiveState'), 'active', ['systemctl show jope-eight-a100-console.service']))
    panes = [line.split('\t') for line in console.get('panes', '').split('\n') if line.strip()]
    checks.append(check('one_live_pane', len(panes) == 1 and panes[0][0] == 'jope-recovery' and panes[0][1] == '0', panes, 'exactly one live pane in jope-recovery', ['tmux list-panes']))
    frame = console.get('frame', '')
    stamp = None
    for line in frame.split('\n'):
        if line.startswith('Frame generated at epoch'):
            try: stamp = float(line.rsplit(' ', 1)[-1])
            except ValueError: stamp = None
    age = (now - stamp) if stamp else None
    checks.append(check('frame_fresh', (age is not None and age <= THRESHOLDS['frame_age_seconds']), age, THRESHOLDS['frame_age_seconds'], ['tmux capture-pane']))
    latest_file = node_file(ev, f'{run}/successor/scoreboard/latest.json') or {}
    latest = latest_file.get('json') or {}
    receipt_rel = (latest.get('receipt') or {}).get('path'); receipt_sha = (latest.get('receipt') or {}).get('sha256')
    bound = None
    if receipt_rel:
        rel = os.path.relpath(receipt_rel, ev['node_root']) if receipt_rel.startswith('/') else receipt_rel
        bound = (node_file(ev, rel) or {}).get('sha256') == receipt_sha
    checks.append(check('headline_fresh_and_bound', bool(latest_file) and (now - latest_file.get('mtime', 0) <= THRESHOLDS['headline_age_seconds']) and bound is True,
                        dict(age=(now - latest_file['mtime']) if latest_file else None, bound=bound), THRESHOLDS['headline_age_seconds'], [f'{run}/successor/scoreboard/latest.json']))
    native_rows = all(token in frame for token in ('CI90', 'Local records', 'Git status'))
    checks.append(check('kpi_rows_native', native_rows, dict(ci90='CI90' in frame, records='Local records' in frame, git='Git status' in frame), 'held-out CI90, high-mark and git rows rendered natively', ['tmux capture-pane'], None if native_rows else 'native dashboard cutover pending'))
    checks.append(check('no_optimistic_live', 'STALE / UNVERIFIED' not in frame or 'Controller LIVE' not in frame, None, 'no LIVE label alongside a stale/unverified state', ['tmux capture-pane']))
    return gate(checks)


def screaming(ev, run):
    node = ev['node']; gp = node_json(ev, 'runs/gp-service.json') or {}; audit = (node_json(ev, f'{run}/review/measured-performance-audit-latest.json') or {}).get('rolling_full_hour') or {}
    latest = node_json(ev, f'{run}/successor/scoreboard/latest.json') or {}; marks = (latest.get('rows') or {}).get('high_marks') or {}
    numbers = dict(gp_evaluations_per_second=gp.get('evaluations_per_second'), gp_active=gp.get('active'), gp_busy_while_ready=audit.get('allocated_core_utilization_while_ready'),
                   gpu_occupancy=audit.get('occupancy'), gpu_mean_util=audit.get('mean_gpu_utilization_missing_zero'), handoff_p95=audit.get('handoff_p95_sampled_upper_gap_seconds'),
                   high_marks_per_hour=marks.get('per_hour_24h'), events_committed=marks.get('all_time'))
    targets = dict(gp_busy_while_ready=THRESHOLDS['gp_busy'], gpu_occupancy=THRESHOLDS['gpu_occupancy'], gpu_mean_util=THRESHOLDS['gpu_util'], handoff_p95=THRESHOLDS['handoff_p95'])
    met = all(numeric(numbers.get(k)) and (numbers[k] < v if k == 'handoff_p95' else numbers[k] >= v) for k, v in targets.items())
    return dict(numbers=numbers, targets=targets, met=met, note='green only when every target is met on fresh evidence (Codex §5)')


def migration(ev):
    """Informational (not a gate): what the xbabe2 mirror and the lane Store hold, per class, from their newest receipts."""
    local = ev.get('local') or {}
    mirror = {}
    for cls, r in (local.get('node_mirror') or {}).items():
        if cls in ('mirror', 'restore'):
            continue
        v = r.get('value') or {}
        mirror[cls] = dict(files=(v.get('census') or {}).get('files'), bytes=(v.get('census') or {}).get('bytes'), verified=v.get('node_rehash_verified') is True and not v.get('drift_refused'),
                           census_only=bool(v.get('census_only')), at=v.get('at'))
    lane = {}
    for cls, r in (local.get('lane') or {}).items():
        v = r.get('value') or {}
        counts = v.get('counts') or {}
        lane[cls] = dict(units=v.get('units'), captured=counts.get('captured', 0), already_bound=counts.get('already_bound', 0), failed=counts.get('failed', 0), captured_bytes=v.get('captured_bytes'), at=v.get('at'))
    git = local.get('git') or {}
    restore = (local.get('node_mirror') or {}).get('restore')
    drill = None
    if restore:
        v = restore.get('value') or {}
        drill = dict(scope=v.get('scope') or 'forge', passed=v.get('passed') is True, sample=v.get('sample'), acknowledged=v.get('acknowledged'), mismatches=v.get('mismatches'), at=v.get('at'))
    return dict(mirror_classes=mirror, lane_classes=lane, restore_drill=drill, mirror_verified_bytes=sum(m['bytes'] or 0 for m in mirror.values() if m['verified'] and not m['census_only']),
                lane_units_captured=sum(l['captured'] + l['already_bound'] for l in lane.values()), lane_failed=sum(l['failed'] for l in lane.values()),
                code_published=bool(git.get('remote_head')) and git.get('remote_head') == git.get('head'), note='informational; publication to the forge is gated in PUBLICATION')


def evaluate(ev, run, since_generation=None):
    gates = dict(LOOP=gate_loop(ev, run), CADENCE=gate_cadence(ev, run), THROUGHPUT=gate_throughput(ev, run), SCIENCE=gate_science(ev, run, since_generation),
                 PUBLICATION=gate_publication(ev, run), SYNC_TRUST=gate_sync_trust(ev, run), TUI=gate_tui(ev, run))
    failing = [name for name in GATES if gates[name]['status'] != 'pass']
    evidence = {}
    for rel, f in ((ev['node'].get('files') or {})).items():
        evidence[rel] = dict(host='node', sha256=f.get('sha256'), bytes=f.get('bytes'), mtime=f.get('mtime'))
    evidence.update({p: dict(v) for p, v in (ev['local'].get('files') or {}).items()})
    require(not any(p.endswith('.md') for p in evidence), 'narration is never evidence')
    return dict(schema=SCHEMA, at=ev['node']['now'], run=run, node=ev.get('node_name'), verdict='approved' if not failing else 'not_approved', failing=failing,
                gates=gates, rules=GATE_RULES, thresholds=THRESHOLDS, screaming=screaming(ev, run), migration=migration(ev), since_generation=since_generation,
                journal_window_bytes=(ev['node'].get('journal') or {}).get('window'), evidence=evidence, final_tests_opened=False)


def write_receipt(output, receipt):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.fromtimestamp(receipt['at'], datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = output / ('acceptance-%s.json' % stamp)
    digest = promotion_gate.immutable(path, receipt)
    atomic(output / 'latest.json', dict(schema=LATEST_SCHEMA, receipt=dict(path=str(path), sha256=digest), verdict=receipt['verdict'], failing=receipt['failing'], at=receipt['at'], final_tests_opened=False))
    return path, digest


def publish_node(node, key, node_root, run, path, digest, latest):
    """Copy the receipt and latest.json to the node's successor/acceptance/ and require the remote sha256 to match."""
    target = f'{node_root}/{run}/successor/acceptance'
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-i', key, node, 'mkdir -p ' + target], check=True, timeout=60)
    for local in (path, latest):
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-i', key, str(local), f'{node}:{target}/{Path(local).name}'], check=True, timeout=300)
    remote = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-i', key, node, f'sha256sum {target}/{Path(path).name}'], capture_output=True, text=True, timeout=60).stdout.split()
    require(remote and remote[0] == digest, 'node copy of the acceptance receipt does not match')
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path('/home/ubuntu/jope-dime/parallel'))
    parser.add_argument('--run', default='runs/eight-a100-recovery-20260912')
    parser.add_argument('--node', default='ubuntu@158.101.20.226')
    parser.add_argument('--key', default=str(Path.home() / '.ssh/lambda_cloud_a100_20260907_ed25519'))
    parser.add_argument('--node-root', default='/home/ubuntu/jope-dime/parallel')
    parser.add_argument('--mirror', type=Path, default=Path('/home/ubuntu/jope-dime/parallel/recovery/eight-a100-20260912'))
    parser.add_argument('--node-mirror', type=Path, default=Path('/home/ubuntu/jope-node-mirror'))   # real directory (symlink under the lane's materialized/)
    parser.add_argument('--repo', type=Path, default=Path('/home/ubuntu/jope-dime'))
    parser.add_argument('--credentials', default=str(Path.home() / '.git-credentials'))
    parser.add_argument('--output', type=Path, default=Path('/home/ubuntu/jope-dime/parallel/recovery/eight-a100-20260912/successor/acceptance'))
    parser.add_argument('--since-generation', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--publish-node', action='store_true')
    args = parser.parse_args(argv)
    local = collect_local(args.root, args.mirror, args.node_mirror, args.repo, args.credentials)
    node = collect_node(args.node, args.key, args.node_root, args.run, sorted(((local.get('mirror') or {}).get('rehash_expected') or {}).keys()))
    ev = dict(node=node, local=local, node_root=args.node_root, node_name=args.node, mirror_dir=str(args.mirror))
    receipt = evaluate(ev, args.run, args.since_generation)
    summary = dict(verdict=receipt['verdict'], failing=receipt['failing'], gates={g: receipt['gates'][g]['status'] for g in GATES}, screaming=receipt['screaming']['met'])
    if args.dry_run:
        print(json.dumps(receipt if args.json else summary, sort_keys=True, default=str))
        return receipt
    path, digest = write_receipt(args.output, receipt)
    summary.update(receipt=str(path), sha256=digest)
    if args.publish_node:
        summary['node_copy'] = publish_node(args.node, args.key, args.node_root, args.run, path, digest, args.output / 'latest.json')
    print(json.dumps(summary, sort_keys=True))
    return receipt


if __name__ == '__main__':
    main()
