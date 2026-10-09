"""Cold read-only capture of a physically retired recovery1 archive.

No network, SQL, raw training data, source writes, or systemd lifecycle changes.
Only metadata/content hashes of the fixed telemetry archive are read.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-trace-recovery-20261009-v1')
UNIT = 'jopedime-hpc-archive-recovery-20261009-v1'
CONFIG_SHA = '142aede3c9186516cd85ca0942f077cd3421c0f07d6c2987105611719b6bc3b6'
SOURCE_SHA = 'c8a13033c6a93bb97d14d3ccb4de0d2fae01be0928a75f654cfd79f32d835c69'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def show(unit):
    proc = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '-p', 'ControlGroup', '-p', 'ActiveState', '-p', 'Result', '-p', 'LoadState'], capture_output=True, text=True, timeout=5)
    need(proc.returncode == 0 and len(proc.stdout) < 16384, 'bounded systemd read')
    result = dict(line.split('=', 1) for line in proc.stdout.splitlines() if '=' in line)
    need(result.get('MainPID') == '0' and result.get('ControlGroup') == '' and result.get('ActiveState') in ('inactive', 'failed'), 'previous archive still physically active')
    return result


def files(root):
    need(root.is_dir() and all(not p.is_symlink() for p in (root, *root.parents)), 'physical prior root')
    result = []
    for path in sorted(root.iterdir()):
        need(not path.is_symlink(), 'symlink in prior root')
        if path.is_file():
            result.append(path)
        elif path.name == 'receipts' and path.is_dir():
            for child in sorted(path.iterdir()):
                need(child.is_file() and not child.is_symlink() and child.suffix == '.json', 'physical receipt only')
                result.append(child)
        else:
            need(path.name == '__pycache__' and path.is_dir(), 'unexpected prior directory')
    need(len(result) <= 22000, 'fixed prior file count cap')
    return result


def capture():
    need(os.uname().nodename == 'xbabe3', 'single host')
    started = time.monotonic()
    before = show(UNIT)
    names = files(ROOT)
    references, total = [], 0
    for path in names:
        need(time.monotonic() - started < 60, 'bounded capture')
        limit = 64 * 1024**2 if path.name == 'stage-timing.jsonl' else 4 * 1024**2
        need(0 < path.stat().st_size <= limit, 'bounded nonempty artifact')
        raw = path.read_bytes()
        need(len(raw) <= limit, 'file grew during capture')
        total += len(raw)
        need(total <= 256 * 1024**2, 'total old evidence cap')
        references.append({'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    by_name = {Path(ref['path']).relative_to(ROOT).as_posix(): ref for ref in references}
    need(by_name['config.json']['sha256'] == CONFIG_SHA and by_name['trace_archive.py']['sha256'] == SOURCE_SHA, 'exact previous source/config')
    progress = json.loads((ROOT / 'progress.json').read_bytes())
    need(hashlib.sha256((ROOT / 'progress.json').read_bytes()).hexdigest() == by_name['progress.json']['sha256'], 'progress moved')
    need(files(ROOT) == names and show(UNIT) == before, 'old archive changed during capture')
    result = {'schema': 'jope.trace-archive-cold-snapshot/v2', 'host': 'xbabe3', 'root': str(ROOT), 'unit': UNIT, 'service': before, 'checked_at_utc_ns': time.time_ns(), 'files': references, 'total_bytes': total, 'progress': progress, 'read_only': True, 's3_requests': 0}
    body = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    need(len(body) <= 16 * 1024**2, 'snapshot output cap')
    return body


if __name__ == '__main__':
    print(capture().decode())
