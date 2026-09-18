#!/usr/bin/env python3
"""Recorded continuation of a bound cycle; never edits its qualification schedule."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate(root, manifest):
    m = read(manifest)
    require(m['schema'] == 'parallel.expired_cycle_continuation.v1', 'bad continuation schema')
    for path, expected in m['bindings'].items():
        require(sha(root / path) == expected, f'changed continuation input: {path}')
    command = m['command']
    require(command[0] == str(root / m['executable']) and command[1:4] == ['--root', str(root), 'cycle'],
            'recovery must use the bound original cycle command')
    require(command[command.index('--initialize') + 1] == str(root / m['original_initialize']),
            'original cycle initialization changed')
    cycle = root / m['cycle']
    selection = read(cycle / 'auxiliary/complete.json')['selection']
    require(selection == m['selected_initialization'], 'auxiliary selection changed')
    require(selection['step'] == 300 and selection['state_sha256'].startswith('80dc36c5'),
            'wrong recovered train-2 initialization')
    jobs = read(cycle / 'search-jobs.json')['jobs']
    require(len(jobs) == 420 and all((root / j['output'] / 'search.json').is_file() for j in jobs),
            'recovery would repeat a GP search')
    exports = read(cycle / 'predictor-exports.json')
    require(exports['completed'] == exports['total'] == 140, 'predictor exports incomplete')
    return m


def run(root, manifest, output, check=False):
    m = validate(root, manifest)
    if check:
        print(json.dumps({'valid': True, 'command': m['command'], 'selected_initialization': m['selected_initialization']}))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with (root / 'runs/continuous-learning.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = root / m['cycle'] / 'complete.json'
        if completed.exists():
            require(read(completed)['dl_gp_dl_complete'] is True, 'invalid cycle completion')
            atomic(output / 'complete.json', {'timestamp': int(time.time()), 'cycle_complete_sha256': sha(completed)})
            return 0
        attempt = output / f'attempt-{time.time_ns()}'
        attempt.mkdir()
        atomic(attempt / 'launch.json', {'timestamp': int(time.time()), 'manifest_sha256': sha(manifest),
               'command': m['command'], 'historical_deadline': m['historical_deadline'],
               'continuation': True, 'complete_learning_cycle': False})
        env = os.environ.copy()
        env.pop('JOPE_CAMPAIGN_DEADLINE', None)
        env['JOPE_GP_EXECUTABLE'] = str(root / m['gp_executable'])
        stopping = False

        def stop(signum, frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        with (attempt / 'cycle.log').open('ab') as log:
            child = subprocess.Popen(m['command'], cwd=root, env=env, stdout=log, stderr=log)
            try:
                while child.poll() is None:
                    atomic(output / 'progress.json', {'timestamp': int(time.time()), 'phase': 'recovery',
                           'pid': child.pid, 'cycle': m['cycle'], 'historical_deadline': m['historical_deadline'],
                           'continuation_started_at': read(attempt / 'launch.json')['timestamp'],
                           'complete_learning_cycle': False})
                    if stopping:
                        child.terminate()
                        break
                    time.sleep(2)
                child.wait(timeout=25)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
            ok = child.returncode == 0 and completed.exists() and read(completed)['dl_gp_dl_complete'] is True
            atomic(attempt / 'result.json', {'timestamp': int(time.time()), 'exit_code': child.returncode,
                   'phase': 'paused' if stopping else 'completed learning cycle' if ok else 'failed',
                   'complete_learning_cycle': ok})
            if ok:
                atomic(output / 'complete.json', {'timestamp': int(time.time()), 'cycle_complete_sha256': sha(completed)})
            return 0 if ok or stopping else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run(args.root, args.manifest, args.output, args.check))
