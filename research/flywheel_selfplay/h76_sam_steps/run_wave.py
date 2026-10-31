#!/usr/bin/env python3
"""H76 FIFO coordinator with a durable, non-resettable GPU wall ledger."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import json
import math
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
             '--json', 'mergedAt,headRefOid,mergeCommit'], text=True))
        if not data['mergedAt']:
            raise RuntimeError(f'PR {pr} must merge before starting waiter')
        if pr == cpu_pr:
            subprocess.run(['git', 'merge-base', '--is-ancestor', data['mergeCommit']['oid'], 'HEAD'],
                           cwd=ROOT, check=True)
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


def feasible(timing, prior_wall=0.0):
    return (math.isfinite(timing) and math.isfinite(prior_wall) and timing > 0 and
            prior_wall >= 0 and prior_wall + timing * 13 * 1.25 <= CAP)


def preflight():
    status = fair.safe_preflight()
    load1 = os.getloadavg()[0]
    if load1 > 85:
        raise RuntimeError(f'host CPU above 85% load budget: {load1:.2f}/102 cores')
    status['load1'] = load1
    return status


def wait_for_cpu_gate(status, *, load_fn=os.getloadavg, sleep_fn=time.sleep,
                      clock_fn=time.monotonic, max_wait=1800.0):
    """Two consecutive <=86.7 load checks, >=30s apart, before any FIFO ticket."""
    started = clock_fn()
    consecutive = 0
    while True:
        load1 = float(load_fn()[0])
        consecutive = consecutive + 1 if math.isfinite(load1) and load1 <= 86.7 else 0
        fair.note(status, f'CPU_GATE load1={load1:.2f}/102; consecutive={consecutive}/2')
        if consecutive == 2:
            return clock_fn() - started
        if clock_fn() - started + 30 > max_wait:
            raise RuntimeError('CPU launch gate did not stabilize within 30 minutes')
        sleep_fn(30)


def retry_ledger(root):
    """Authorize only the owner-specified, fully charged first timing infra void."""
    ledger = json.loads((root / 'budget.json').read_text())
    cells = ledger.get('cells', [])
    if (ledger.get('in_progress') or ledger.get('cap_seconds') != CAP or len(cells) != 1 or
        ledger.get('timing_retry_cpu_pr') or ledger.get('prereg_sha256') != h59.sha256(HERE / 'PREREG.md')):
        raise RuntimeError('not the single accounted timing attempt authorized for retry')
    row = cells[0]
    if (row.get('timing') is not True or row.get('arm') != 'K8' or row.get('seed') != 12 or
        'host CPU above 85%' not in row.get('error', '') or
        not math.isfinite(row.get('wall_seconds', float('nan'))) or
        not 0 < row['wall_seconds'] < CAP or
        not math.isfinite(row.get('wait_seconds', float('nan'))) or row['wait_seconds'] < 0):
        raise RuntimeError('retry requires a fully accounted CPU-guard timing infra void')
    if any(p.name != 'timing-K8-seed12.json' for p in root.glob('*-seed*.json')):
        raise RuntimeError('outcome/retry artifacts already exist; refusing retry')
    if list(root.glob('*.adapter.pt')):
        raise RuntimeError('outcome adapter artifacts already exist; refusing retry')
    if (root / 'metrics.json').exists():
        raise RuntimeError('outcome metrics already exist; refusing retry')
    receipt = root / 'timing-K8-seed12.json'
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record.get('timing_only') is not True or 'train' in record or 'dev' in record:
            raise RuntimeError('prior artifact is not timing-only')
    remaining(ledger)
    return ledger


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
    yielded = wait_for_cpu_gate(status)  # outside lock; also enforces the >=30s per-cell yield
    wait_total += yielded
    if wait_total >= fair.MAX_LOCK_WAIT_SECONDS:
        raise RuntimeError('separate cumulative 3-hour wait budget exhausted before ticket')
    lease, waited, free, snapshot = fair.acquire_lock(
        status, wait_total, after_release=True, wave_name='H76', preflight_fn=preflight)
    started = None
    record = {'arm': arm, 'seed': seed, 'timing': timing, 'wait_seconds': waited + yielded, 'cpu_gate_seconds': yielded,
              'gpu_free_mib': free, 'preflight': snapshot}
    name = ('timing-K8-seed12-attempt2' if ledger['cells'] else 'timing-K8-seed12') if timing else f'{arm}-seed{seed}'
    out = root / f'{name}.json'
    error = None
    try:
        ledger['in_progress'] = dict(record)
        h59.atomic_json(root / 'budget.json', ledger)
        command = [sys.executable, str(HERE / 'runner.py'), '--arm', arm, '--seed', str(seed),
                   '--out', str(out), '--lock-fd', str(lease.fileno())]
        if timing:
            command.append('--timing')
        ledger['in_progress']['command'] = command
        ledger['in_progress']['started_utc'] = datetime.now(timezone.utc).isoformat()
        h59.atomic_json(root / 'budget.json', ledger)
        record.update(command=command, started_utc=ledger['in_progress']['started_utc'])
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
    parser.add_argument('--retry-timing-infra', action='store_true')
    parser.add_argument('--root', type=Path, default=Path.home() / 'jope-data/h76/h76-wave')
    args = parser.parse_args()
    check_dependencies(args.cpu_pr)  # before ticket, GPU query, or lock attempt
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'coordinator.lock').open('a+') as coordinator_lock:
        fcntl.flock(coordinator_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.retry_timing_infra:
            ledger = retry_ledger(args.root)
            ledger['timing_retry_cpu_pr'] = args.cpu_pr
            ledger['retry_addendum_sha256'] = h59.sha256(HERE / 'ADDENDUM-cpu-guard-t2335.md')
        else:
            if (args.root / 'budget.json').exists():
                raise RuntimeError('wave already launched: do not reset timing, outcomes or wall budget')
            ledger = {'cap_seconds': CAP, 'cells': [], 'in_progress': None,
                      'prereg_sha256': h59.sha256(HERE / 'PREREG.md'), 'cpu_pr': args.cpu_pr}
        prior_wall = sum(c['wall_seconds'] for c in ledger['cells'])
        prior_wait = sum(c['wait_seconds'] for c in ledger['cells'])
        h59.atomic_json(args.root / 'budget.json', ledger)
        status = args.root / 'H76_STATUS.log'
        try:
            timing, waited = execute(args.root, ledger, 'K8', 12, True, prior_wait)
            wait = prior_wait + waited
            timing_wall = ledger['cells'][-1]['wall_seconds']
            if not feasible(timing_wall, prior_wall):
                raise RuntimeError(f'timing infeasible: {prior_wall:.2f} + {timing_wall:.2f} * 13 * 1.25 > {CAP}')
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
            h59.atomic_json(args.root / ('incomplete-timing-retry.json' if args.retry_timing_infra else 'incomplete.json'),
                             {'complete': False, 'reason': repr(exc), 'ledger': ledger})
            fair.note(status, f'STOP {exc}; outcomes unknown where missing; no admission')
            raise


if __name__ == '__main__':
    main()
