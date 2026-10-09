"""Cold remote preflight then one bounded new service. No historical restart."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-trace-recovery-20261009-v2')
UNIT = 'jopedime-hpc-archive-recovery-20261009-v2'
OLD = 'jopedime-hpc-archive-recovery-20261009-v1'


def show(unit):
    proc = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '-p', 'ControlGroup', '-p', 'ActiveState', '-p', 'LoadState'], capture_output=True, text=True, timeout=5)
    assert proc.returncode == 0 and len(proc.stdout) < 16384
    return dict(line.split('=', 1) for line in proc.stdout.splitlines() if '=' in line)


def create(name, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with (ROOT / name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config-sha', required=True); args = parser.parse_args()
    assert os.uname().nodename == 'xbabe3'
    path = ROOT / 'config.json'; assert all(not p.is_symlink() for p in (path, *path.parents))
    raw = path.read_bytes(); assert hashlib.sha256(raw).hexdigest() == args.config_sha
    config = json.loads(raw)
    assert config['schema'] == 'jope.trace-archive-recovery/v2' and config['out'] == str(ROOT)
    assert config['maximum_global_put_attempts'] == 5000 and config['maximum_put_attempts'] == 3943 and config['maximum_segments'] == 2500
    assert config['created_unix'] <= time.time() and 180 < config['deadline_unix'] - time.time() <= 3600 and config['deadline_unix'] - config['created_unix'] == 3600 and config['global_deadline_unix'] == 1791592146.380285
    for key in ('source', 'base', 'prior_capture'):
        ref = config[key]; source = Path(ref['path']); assert source.parent == ROOT and not source.is_symlink()
        body = source.read_bytes(); assert len(body) == ref['bytes'] and hashlib.sha256(body).hexdigest() == ref['sha256']
    old = show(OLD); new = show(UNIT)
    assert old['MainPID'] == '0' and old['ControlGroup'] == '' and old['ActiveState'] in ('inactive', 'failed')
    assert new['LoadState'] == 'not-found' and new['MainPID'] == '0' and new['ControlGroup'] == ''
    assert not (ROOT / 'execution-started.json').exists() and not (ROOT / 'receipts').exists()
    mono = time.monotonic() + config['deadline_unix'] - time.time()
    def remaining():
        value = math.floor(min(config['deadline_unix'] - time.time(), mono - time.monotonic()))
        assert value > 5
        return value
    common = ['systemd-run', '--user', '-p', 'CPUQuota=100%', '-p', 'MemoryMax=512M', '-p', 'MemorySwapMax=0', '-p', 'TasksMax=24', '-p', 'Nice=19', '-p', 'Restart=no', '-p', 'KillMode=control-group', '--setenv=CUDA_VISIBLE_DEVICES=', '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=PYTHONDONTWRITEBYTECODE=1']
    command = ['python3', '-B', str(ROOT / 'trace_recovery_v2.py'), '--config', str(path), '--config-sha', args.config_sha]
    pre = [*common, '--unit=' + UNIT + '-preflight', '--collect', '--wait', '--pipe', '-p', 'RuntimeMaxSec=' + str(min(180, remaining())), *command, '--preflight-only']
    create('preflight-intent.json', {'config_sha256': args.config_sha, 'at_utc_ns': time.time_ns(), 'old': old, 'argv': pre})
    result = subprocess.run(pre, capture_output=True, text=True, timeout=min(185, remaining() + 5))
    assert len(result.stdout) + len(result.stderr) <= 1024**2
    create('preflight-reply.json', {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'at_utc_ns': time.time_ns()})
    assert result.returncode == 0
    proof = json.loads(result.stdout.splitlines()[-1])
    timing = proof.pop('qualification_timing')
    assert timing['duration_monotonic_ns'] >= 0 and timing['ended_utc_ns'] >= timing['started_utc_ns'] and all(x >= 0 for x in timing['cpu_delta'].values())
    assert proof == {'state': 'passed', 'config_sha256': args.config_sha, 'fresh_first_part': 529, 'prior_verified_segments': 528, 'prior_global_put_attempts': 1057, 'remaining_put_attempts': 3943, 'gap_parts': [153], 's3_requests': 0}
    proof['qualification_timing'] = timing
    old = show(OLD); assert old['MainPID'] == '0' and old['ControlGroup'] == ''
    mem = dict((k, int(value.split()[0]) * 1024) for k, value in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines()))
    assert mem['MemAvailable'] >= 26 * 2**30 and os.getloadavg()[0] < 100
    launch = [*common, '--unit=' + UNIT, '-p', 'RuntimeMaxSec=' + str(remaining()), *command]
    create('launch-intent.json', {'config_sha256': args.config_sha, 'at_utc_ns': time.time_ns(), 'deadline_unix': config['deadline_unix'], 'argv': launch, 'prior_physical_state': old, 'preflight': proof})
    result = subprocess.run(launch, capture_output=True, text=True, timeout=min(15, remaining()))
    receipt = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'at_utc_ns': time.time_ns(), 'service': show(UNIT), 'config_sha256': args.config_sha}
    create('launch-receipt.json', receipt); print(json.dumps(receipt)); assert result.returncode == 0


if __name__ == '__main__':
    main()
