#!/usr/bin/env python3
"""H76 FIFO coordinator with a durable, non-resettable GPU wall ledger."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from runner import ARMS, HERE, ROOT, SEEDS, h59, load, summarize

fair = load('h76_fair_coordinator', HERE.parent / 'h61/run_wave.py')
CAP = 2700.0
FAIRNESS_MERGE = '9f5d290e188269de50168ffd28fa84e40165810a'


def check_dependencies(cpu_pr):
    subprocess.run(['git', 'fetch', 'origin', 'main'], cwd=ROOT, check=True)
    subprocess.run(['git', 'merge-base', '--is-ancestor', FAIRNESS_MERGE, 'origin/main'], cwd=ROOT, check=True)
    for pr in (832, cpu_pr):
        data = json.loads(subprocess.check_output(
            ['/home/ubuntu/.local/bin/gh', 'pr', 'view', str(pr), '--repo', 'neverhuman/JopeDime',
             '--json', 'mergedAt,headRefOid'], text=True))
        if not data['mergedAt']:
            raise RuntimeError(f'PR {pr} must merge before starting waiter')
        if pr == cpu_pr:
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            if data['headRefOid'] != head:
                raise RuntimeError('CPU PR head differs from execution source')
    for path in HERE.glob('*'):
        if path.suffix not in ('.py', '.md'):
            continue
        relative = str(path.relative_to(ROOT))
        landed = subprocess.check_output(['git', 'show', f'origin/main:{relative}'], cwd=ROOT)
        if landed != path.read_bytes():
            raise RuntimeError(f'execution source differs from merged main: {relative}')
    if subprocess.check_output(['git', 'status', '--porcelain', '--', str(HERE)], cwd=ROOT, text=True):
        raise RuntimeError('dirty H76 source')


def remaining(ledger):
    if ledger.get('in_progress'):
        raise RuntimeError('unaccounted prior launch; budget cannot safely resume')
    value = CAP - sum(row['wall_seconds'] for row in ledger['cells'])
    if value <= 0:
        raise RuntimeError('2700-second cumulative wall exhausted')
    return value


def feasible(timing):
    return timing * 13 * 1.25 <= CAP


def preflight():
    status = fair.safe_preflight()
    load1 = os.getloadavg()[0]
    if load1 > 85:
        raise RuntimeError(f'host CPU above 85% load budget: {load1:.2f}/102 cores')
    status['load1'] = load1
    return status


def child_setup():
    os.nice(10)
    # A coordinator crash cannot leave its own GPU child running unaccounted.
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)


def run_child(command, log_path, timeout, lease):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    started = time.monotonic()
    max_rss = 0
    with log_path.open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                pass_fds=(lease.fileno(),), preexec_fn=child_setup)
        try:
            while proc.poll() is None:
                rss = fair.process_rss_kib(proc.pid) or 0
                max_rss = max(max_rss, rss)
                if rss >= fair.RSS_LIMIT_KIB:
                    raise RuntimeError('child RSS reached 6 GiB')
                if time.monotonic() - started >= timeout:
                    raise RuntimeError('remaining GPU wall expired')
                preflight()
                time.sleep(1)
            if proc.returncode:
                raise RuntimeError(f'cell exit {proc.returncode}; inspect {log_path}')
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    return {'max_rss_kib': max_rss}


def execute(root, ledger, arm, seed, timing, wait_total):
    status = root / 'H76_STATUS.log'
    timeout = remaining(ledger) - 10.0  # reserve watchdog polling + own-child shutdown
    if timeout <= 0:
        raise RuntimeError('insufficient wall remaining for safe child shutdown')
    yielded = 30 if ledger['cells'] else 0
    if yielded:
        time.sleep(30)  # mandatory yield before a fresh ticket, outside GPU cell wall
        wait_total += 30
    lease, waited, free, snapshot = fair.acquire_lock(
        status, wait_total, after_release=True, wave_name='H76', preflight_fn=preflight)
    started = None
    record = {'arm': arm, 'seed': seed, 'timing': timing, 'wait_seconds': waited + yielded,
              'gpu_free_mib': free, 'preflight': snapshot}
    name = 'timing-K8-seed12' if timing else f'{arm}-seed{seed}'
    out = root / f'{name}.json'
    error = None
    try:
        ledger['in_progress'] = dict(record)
        h59.atomic_json(root / 'budget.json', ledger)
        command = [sys.executable, str(HERE / 'runner.py'), '--arm', arm, '--seed', str(seed),
                   '--out', str(out), '--lock-fd', str(lease.fileno())]
        if timing:
            command.append('--timing')
        fair.note(status, f'RUN {name}; remaining_gpu_wall={timeout:.2f}s')
        started = time.monotonic()
        record.update(run_child(command, root / f'{name}.log', timeout, lease))
    except BaseException as exc:
        error = exc
        record['error'] = repr(exc)
    finally:
        record['wall_seconds'] = time.monotonic() - started if started is not None else 0.0
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()
    ledger['cells'].append(record)
    ledger['in_progress'] = None
    h59.atomic_json(root / 'budget.json', ledger)
    fair.note(status, f'RELEASE {name}; wall={record["wall_seconds"]:.2f}s; error={record.get("error")}')
    if error is not None:
        raise error
    return json.loads(out.read_text()), waited + yielded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-pr', type=int, required=True)
    parser.add_argument('--root', type=Path, default=Path.home() / 'jope-data/h76/h76-wave')
    args = parser.parse_args()
    check_dependencies(args.cpu_pr)  # before ticket, GPU query, or lock attempt
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'coordinator.lock').open('a+') as coordinator_lock:
        fcntl.flock(coordinator_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.root / 'budget.json').exists():
            raise RuntimeError('wave already launched: do not reset timing, outcomes or wall budget')
        ledger = {'cap_seconds': CAP, 'cells': [], 'in_progress': None,
                  'prereg_sha256': h59.sha256(HERE / 'PREREG.md'), 'cpu_pr': args.cpu_pr}
        h59.atomic_json(args.root / 'budget.json', ledger)
        status = args.root / 'H76_STATUS.log'
        try:
            timing, wait = execute(args.root, ledger, 'K8', 12, True, 0.0)
            timing_wall = ledger['cells'][0]['wall_seconds']
            if not feasible(timing_wall):
                raise RuntimeError(f'timing infeasible: {timing_wall:.2f} * 13 * 1.25 > {CAP}')
            cells = []
            for seed in SEEDS:
                for arm in ARMS:
                    cell, waited = execute(args.root, ledger, arm, seed, False, wait)
                    wait += waited
                    cells.append(cell)
            result = summarize(cells)
            result['gpu_wall_seconds'] = sum(c['wall_seconds'] for c in ledger['cells'])
            result['ledger'] = ledger
            h59.atomic_json(args.root / 'metrics.json', result)
            fair.note(status, 'COMPLETE all12 paired cells; metrics.json written; no serving promotion')
        except BaseException as exc:
            h59.atomic_json(args.root / 'incomplete.json', {'complete': False, 'reason': repr(exc), 'ledger': ledger})
            fair.note(status, f'STOP {exc}; outcomes unknown where missing; no admission')
            raise


if __name__ == '__main__':
    main()
