#!/usr/bin/env python3
"""Flywheel plots: a self-contained terminal viewer (stdlib only, read-only) for the recovery run.

Four panels, redrawn every --loop seconds (default 60) with Unicode sparklines and block charts:
  1 FROZEN PANEL   per monitored checkpoint step: ranking_bt and sequence_nll (lower is better), the diagnostic guard
                   line (baseline * (1 + maximum_relative_regression)), accepted / rejected marks from the journal.
  2 GENERATIONS    per generation: neural-vs-generic FINAL-level wins/ties/losses on the 124 training cuts and the 16
                   held-out cuts (best_final_gp of the paired searches; shared tail not excludable), win-rate sparkline
                   and trailing-10 rate; seed-only rows appear when the scoreboard cache exists (scoreboard/derived/).
  3 HIGH MARKS     new per-dataset-fold record highs per hour over the last 24 h (live evaluator identities), by
                   partition, and the champion git state (commits, last sync, push status).
  4 THROUGHPUT     GP slots / evaluations / continuation statistics (seed-coverage/health.json), GPU task occupancy
                   (learning/progress.json), nvidia-smi utilization, learning state (phase, hold, growth gate, blocks,
                   serving vs learning step).
Every number is labelled with its source; nothing here claims model improvement (that is the verdict receipt's job).
Usage: flywheel_plots.py --root /home/ubuntu/jope-dime/parallel [--run runs/eight-a100-recovery-20260912] [--loop 60]
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BLOCKS = '▁▂▃▄▅▆▇█'
NUMBER = re.compile(r'generation-(\d+)')


def read(path):
    try:
        with open(path, 'rb') as stream:
            return json.loads(stream.read())
    except (OSError, ValueError):
        return None


def sparkline(values, width):
    values = [v for v in values if v is not None][-width:]
    if not values:
        return ''
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return BLOCKS[3] * len(values)
    return ''.join(BLOCKS[min(7, int((v - low) / (high - low) * 7.999))] for v in values)


def chart(values, width, height, marks=None):
    """A block chart: rows top to bottom; marks[i] = char drawn above column i (e.g. 'x' for rejected)."""
    values = values[-width:]
    marks = (marks or [])[-width:]
    if not values:
        return ['(no data)']
    finite = [v for v in values if v is not None]
    low, high = min(finite), max(finite)
    span = (high - low) or 1e-12
    rows = []
    for level in range(height, 0, -1):
        threshold_low = low + span * (level - 1) / height
        line = ''
        for v in values:
            if v is None:
                line += ' '
            elif v >= threshold_low:
                line += '█' if v >= low + span * level / height else BLOCKS[min(7, int((v - threshold_low) / (span / height) * 7.999))]
            else:
                line += ' '
        label = '%9.4f' % (low + span * level / height) if level in (height, max(1, height // 2)) else ' ' * 9
        rows.append(label + ' |' + line)
    rows.append(' ' * 9 + ' +' + '-' * len(values))
    if marks:
        rows.append(' ' * 11 + ''.join((m or ' ') for m in marks))
    return rows


def fmt_time(ts):
    if not ts:
        return '-'
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime('%m-%d %H:%M')


# ---------------------------------------------------------------- panel 1: frozen panel

def frozen_panel(run):
    receipts = [r for r in (read(p) for p in glob.glob(str(run / 'learning/monitoring/*.json'))) if r and r.get('schema') == 'parallel.frozen_monitoring.v1']
    receipts.sort(key=lambda r: (r.get('evaluated_step') or 0, r.get('timestamp') or 0))
    verdicts = {}
    journal = run / 'learning/journal.jsonl'
    if journal.is_file():
        try:
            with open(journal, 'rb') as stream:
                for raw in stream:   # streamed: the journal grows without bound and the diagnostic events are sparse
                    if b'"diagnostic_' not in raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except ValueError:
                        continue
                    if row.get('event') in ('diagnostic_accepted', 'diagnostic_rejected'):
                        verdicts[row.get('checkpoint_sha256')] = row['event']
        except OSError:
            pass
    progress = read(run / 'learning/progress.json') or {}
    guard = progress.get('training_diagnostic') or {}
    baseline = guard.get('baseline_ranking_bt')
    return receipts, verdicts, guard, baseline


def render_frozen(run, width):
    receipts, verdicts, guard, baseline = frozen_panel(run)
    lines = ['1 FROZEN PANEL (learning/monitoring/*.json; same 2048 examples; lower is better) — %d receipts' % len(receipts)]
    if not receipts:
        return lines + ['   no monitoring receipts']
    steps = [r.get('evaluated_step') for r in receipts]
    bt = [r.get('ranking_bt') for r in receipts]
    nll = [r.get('sequence_nll') for r in receipts]
    marks = []
    for r in receipts:
        v = verdicts.get((r.get('binding') or {}).get('checkpoint_sha256'))
        marks.append('x' if v == 'diagnostic_rejected' else 'o' if v == 'diagnostic_accepted' else '.')
    limit = baseline * 1.01 if baseline else None
    lines.append('   ranking_bt  steps %s..%s  last %.4f  min %.4f (step %s)  guard %s  marks: o accepted  x rejected  . unmonitored'
                 % (steps[0], steps[-1], bt[-1], min(bt), steps[bt.index(min(bt))], ('%.4f (baseline %.4f * 1.01)' % (limit, baseline)) if limit else 'n/a'))
    lines += ['   ' + row for row in chart(bt, width - 14, 6, marks)]
    lines.append('   sequence_nll last %.4f  min %.4f   %s' % (nll[-1], min(nll), sparkline(nll, width - 40)))
    held = guard.get('held')
    lines.append('   diagnostic guard: held=%s consecutive_rejections=%s last_step=%s' % (held, guard.get('consecutive_rejections'), guard.get('last_step')))
    return lines


# ---------------------------------------------------------------- panel 2: generations

def generation_rows(run):
    out = []
    for path in glob.glob(str(run / 'learning/generation-*/quality-report.json')):
        number = int(NUMBER.search(path).group(1))
        report = read(path)
        if not report or report.get('schema') != 'parallel.continuous_quality.v1':
            continue
        out.append((number, report))
    out.sort()
    return out


def wtl(report, partition, level='best_final_gp'):
    by_key = {}
    for row in report.get('rows', []):
        if row.get('partition') != partition or row.get('task') != 'regression':
            continue
        by_key.setdefault((row.get('dataset'), row.get('cut')), {})[row.get('arm')] = row.get(level)
    wins = ties = losses = 0
    for arms in by_key.values():
        a, b = arms.get('neural'), arms.get('generic')
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        if abs(a - b) <= 1e-12 * max(1.0, abs(b)):
            ties += 1
        elif a < b:
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


def render_generations(run, width):
    rows = generation_rows(run)
    lines = ['2 GENERATIONS (generation-*/quality-report.json; FINAL-level best_final_gp neural vs generic; shared tail not excludable; NOT held-out unless marked)']
    if not rows:
        return lines + ['   no quality reports']
    train_rates, val_rows = [], []
    table = []
    for number, report in rows:
        w, t, l = wtl(report, 'meta_train')
        vw, vt, vl = wtl(report, 'meta_validation')
        n = w + t + l
        rate = w / n if n else None
        train_rates.append(rate)
        val_rows.append((vw, vt, vl))
        table.append((number, w, t, l, rate, vw, vt, vl, (report.get('rows') or [{}])[0].get('completed_at')))
    trailing = [r for r in train_rates[-10:] if r is not None]
    lines.append('   training cuts (124 contexts): neural win rate per generation %s  last %s  trailing-10 mean %s'
                 % (sparkline(train_rates, width - 60), ('%.2f' % train_rates[-1]) if train_rates[-1] is not None else '-', ('%.2f' % (sum(trailing) / len(trailing))) if trailing else '-'))
    recent = table[-8:]
    lines.append('   %-5s %-18s %-8s %-22s %-8s' % ('gen', 'train W/T/L', 'rate', 'HELD-OUT 16 cuts W/T/L', 'completed'))
    for number, w, t, l, rate, vw, vt, vl, done in recent:
        lines.append('   %-5d %-18s %-8s %-22s %-8s' % (number, '%d/%d/%d' % (w, t, l), ('%.2f' % rate) if rate is not None else '-', '%d/%d/%d' % (vw, vt, vl), fmt_time(done)))
    cum = [sum(r[i] for r in val_rows[-10:]) for i in range(3)]
    lines.append('   held-out trailing-10 W/T/L %d/%d/%d (final level, underpowered n=16 per generation; seed-only rows come from the scoreboard producer)' % tuple(cum))
    return lines


# ---------------------------------------------------------------- panel 3: high marks

def live_evaluator(root, manifest):
    doc = read(root / manifest) if manifest else None
    if not doc:
        return None
    return (doc.get('bindings') or {}).get(doc.get('cpu_executable'))


def render_high_marks(root, run, evaluator, width):
    now = time.time()
    hours = [0] * 24
    by_partition = {'meta_train': 0, 'meta_validation': 0}
    total = 0
    store = root / 'runs/champions'
    for current_path in glob.glob(str(store / '*/current.json')):
        current = read(current_path)
        if not current:
            continue
        identity = current.get('identity') or {}
        if evaluator and identity.get('evaluator_sha256') != evaluator:
            continue
        if identity.get('partition') not in by_partition:
            continue
        directory = Path(current_path).parent
        for event in glob.glob(str(directory / 'event-*.json')):
            try:
                age = now - os.path.getmtime(event)
            except OSError:
                continue
            total += 1
            if age < 86400:
                hours[min(23, int(age // 3600))] += 1
                by_partition[identity['partition']] += 1
    series = list(reversed(hours))
    lines = ['3 HIGH MARKS (runs/champions event files, live evaluator %s; local record highs on the adaptive cut-heldout teacher — not held-out wins)' % ((evaluator or '?')[:8])]
    lines.append('   last 24 h per hour (oldest→newest): %s  total 24 h: %d (train %d, held-out cuts %d)  last hour: %d  all-time events (live identities): %d'
                 % (sparkline(series, 24), sum(hours), by_partition['meta_train'], by_partition['meta_validation'], hours[0], total))
    repo = run / 'successor/champion-git/repo'
    state = read(repo / '.champion-git-state.json')
    if state:
        remote = state.get('remote') or {}
        receipts = sorted(glob.glob(str(run / 'successor/champion-git/receipts/*.json')))
        last = read(receipts[-1]) if receipts else {}
        lines.append('   champion git: %d events committed (%d identities)  last sync %s  committed then %s  push: %s %s'
                     % (len(state.get('published') or {}), len(state.get('identities') or {}), fmt_time((last or {}).get('at')), (last or {}).get('committed', '-'),
                        remote.get('status', '-'), ('(%s)' % remote.get('last_error')[:60]) if remote.get('last_error') else ''))
    else:
        lines.append('   champion git: no repository yet')
    return lines


# ---------------------------------------------------------------- panel 4: throughput

def gpu_utilization():
    if not shutil.which('nvidia-smi'):
        return None
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [line.split(',') for line in out.stdout.strip().splitlines() if ',' in line]
    return [(int(a.strip()), int(b.strip())) for a, b in rows]


def render_throughput(run, width):
    health = read(run / 'seed-coverage/health.json') or {}
    progress = read(run / 'learning/progress.json') or {}
    service = read(run.parent / 'gp-service.json') or {}
    lines = ['4 THROUGHPUT / LEARNING STATE (seed-coverage/health.json, learning/progress.json, runs/gp-service.json, nvidia-smi)']
    gate = health.get('gp_gate') or {}
    counts = health.get('counts') or {}
    cont = health.get('continuation_statistics') or {}
    lines.append('   GP: workers %s active %s retry_pending %s | ordinary active %s ready %s | continuation active %s ready %s | evals/s %s'
                 % (service.get('workers'), service.get('active'), service.get('retry_pending'), gate.get('ordinary_active'), gate.get('ordinary_ready'),
                    gate.get('continuation_active'), gate.get('continuation_ready'), ('%.1f' % service['evaluations_per_second']) if isinstance(service.get('evaluations_per_second'), (int, float)) else '-'))
    lines.append('   episodes: completed %s pending %s failed %s | continuation fits attempted %s successful %s timeouts %s | ingester %s'
                 % (counts.get('completed'), counts.get('pending'), counts.get('failed'), cont.get('attempted_fits'), cont.get('successful_fits'), cont.get('failed_fits') or cont.get('timeout_fits'),
                    ('active' if (health.get('ingester') or {}).get('active') else 'idle')))
    occ = progress.get('gpu_occupancy') or {}
    util = gpu_utilization()
    lines.append('   GPU: task occupancy (1 h) %s  full-hour %s  paid idle %s s | utilization now %s'
                 % (('%.3f' % occ['task_occupancy']) if isinstance(occ.get('task_occupancy'), (int, float)) else '-', ('%.3f' % occ['full_hour_occupancy']) if isinstance(occ.get('full_hour_occupancy'), (int, float)) else '-',
                    progress.get('paid_gpu_idle_seconds'), ' '.join('%d%%' % u for u, _ in util) if util else 'n/a'))
    growth = progress.get('training_growth') or {}
    lines.append('   learning: phase %s automatic_training %s hold=%s (%s) | blocks %s | growth eligible %s (programs %s, searches %s) | next %s'
                 % (progress.get('phase'), progress.get('automatic_training'), (progress.get('training_diagnostic') or {}).get('held'), progress.get('training_hold_reason'),
                    progress.get('completed_training_blocks'), growth.get('eligible'), growth.get('newly_admitted_training_programs'), growth.get('newly_admitted_current_run_searches'),
                    progress.get('next_scheduled_work')))
    learning = progress.get('last_committed_checkpoint') or ''
    serving = progress.get('serving_checkpoint') or ''
    lines.append('   checkpoints: learning %s | serving %s (%s) | idle: %s' % (learning.rsplit('/', 1)[-1], serving.rsplit('/', 1)[-1], progress.get('serving_status'), progress.get('idle_explanation')))
    return lines


# ---------------------------------------------------------------- main

def render(root, run, manifest, width):
    evaluator = live_evaluator(root, manifest)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    lines = ['FLYWHEEL PLOTS  %s  run %s  (read-only viewer; independent model improvement: NOT ESTABLISHED unless a verdict receipt says so)' % (stamp, run.name), '']
    for section in (render_frozen(run, width), render_generations(run, width), render_high_marks(root, run, evaluator, width), render_throughput(run, width)):
        lines += section + ['']
    return '\n'.join(line[:width] for line in lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', default='runs/eight-a100-recovery-20260912')
    parser.add_argument('--manifest', default='runs/eight-a100-recovery-20260912/recovery-manifest-v9.json')
    parser.add_argument('--loop', type=int, default=0, help='redraw every N seconds (0 = once)')
    parser.add_argument('--width', type=int, default=0)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    run = root / args.run
    while True:
        width = args.width or shutil.get_terminal_size((160, 50)).columns
        text = render(root, run, args.manifest, width)
        if args.loop:
            sys.stdout.write('\x1b[2J\x1b[H')
        sys.stdout.write(text + '\n')
        sys.stdout.flush()
        if not args.loop:
            return text
        time.sleep(args.loop)


if __name__ == '__main__':
    main()
