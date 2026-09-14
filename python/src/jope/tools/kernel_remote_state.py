"""Read-only observations of the running node; never edit services or queues."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

UNITS = ('jope-parallel-gp', 'jope-parallel-campaign', 'jope-parallel-monitor')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def capture(root):
    root = Path(root).resolve()
    stable, telemetry = {}, {}
    for unit in UNITS:
        properties = subprocess.check_output([
            'systemctl', 'show', unit,
            '--property=MainPID,InvocationID,NRestarts,ActiveState,SubState,ControlGroup,ExecStart,Environment',
        ], text=True)
        props = dict(line.split('=', 1) for line in properties.splitlines() if '=' in line)
        if props.get('ActiveState') != 'active' or int(props.get('MainPID', 0)) <= 0:
            raise ValueError(f'Required live service is unavailable: {unit}')
        pid = props['MainPID']
        proc = Path('/proc') / pid
        command = (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0')
        executable = str((proc / 'exe').resolve())
        # Only the child-executable route is read from the live environment.
        route = [v for v in (proc / 'environ').read_bytes().split(b'\0')
                 if v.startswith(b'JOPE_GP_EXECUTABLE=')]
        stable[unit] = {'properties': props, 'definition': subprocess.check_output(
            ['systemctl', 'cat', unit], text=True), 'exe': executable,
            'exe_sha256': sha(proc / 'exe'), 'command': command,
            'child_route': [v.decode() for v in route]}
        for flag in ('--manifest', '--spec'):
            if flag in command:
                path = Path(command[command.index(flag) + 1])
                path = path if path.is_absolute() else root / path
                stable[str(path)] = sha(path)
                doc = json.loads(path.read_text())
                if isinstance(doc.get('vocabulary'), str):
                    p = root / doc['vocabulary']
                    stable[str(p)] = sha(p)
        for value in route:
            path = Path(value.decode().split('=', 1)[1])
            stable[str(path)] = sha(path)
    # Snapshot all live source paths, including grammar and the vendored kernel.
    # Mutable campaign progress is observed separately, not required to freeze.
    for name in ('Cargo.toml', 'Cargo.lock', 'build.rs', 'src', 'native', 'tools',
                 'assets/grammar', 'assets/source/jope-dime/vendor/feat-math'):
        base = root / name
        if not base.exists():
            continue
        paths = [base] if base.is_file() else base.rglob('*')
        for p in paths:
            relative = p.relative_to(root)
            if p.is_file() and not any(x in relative.parts for x in ('target', 'build', '__pycache__', '.git')):
                stable[str(p)] = sha(p)
    for name in ('hardware.json', 'gp-service.json', 'campaign-progress.json', 'cycle-progress.json'):
        path = root / 'runs' / name
        telemetry[name] = json.loads(path.read_text()) if path.exists() else None
    return {'schema': 'kernel.live-state.v1', 'timestamp': time.time(),
            'stable': stable, 'telemetry': telemetry}


def compare(before, after):
    old, new = before['stable'], after['stable']
    changes = [key for key in sorted(old.keys() | new.keys()) if old.get(key) != new.get(key)]
    return {'stable_identities_unchanged': not changes, 'changed': changes,
            'scope': 'Live source, service invocation, executable routes, fixed manifests and vocabulary; mutable progress may advance'}


def compare_health(before, after, max_age=10):
    """Check observed operational health, without inferring performance or cause.

    Unavailable/stale evidence is inconclusive, never a healthy observation.
    Cumulative counters are comparable only within the same service invocation.
    """
    failures, unavailable, observations = [], [], {}
    if not isinstance(max_age, (int, float)) or not math.isfinite(max_age) or max_age <= 0:
        raise ValueError('A positive finite telemetry age limit is required')

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    gp = []
    for label, snapshot in (('before', before), ('after', after)):
        stamp = snapshot.get('timestamp')
        telemetry = snapshot.get('telemetry')
        if not number(stamp) or not isinstance(telemetry, dict):
            unavailable.append(label + ': missing snapshot timestamp or telemetry')
            gp.append(None)
            continue
        for name in ('gp-service.json', 'cycle-progress.json', 'hardware.json'):
            record = telemetry.get(name)
            if not isinstance(record, dict) or not number(record.get('timestamp')):
                unavailable.append(label + ': missing timestamp for ' + name)
                continue
            age = stamp - record['timestamp']
            # Integer-second producer timestamps permit up to one second of skew.
            if not -1 <= age <= max_age:
                unavailable.append(label + ': stale or future ' + name)
            if name == 'hardware.json':
                if not isinstance(record.get('errors'), list):
                    unavailable.append(label + ': hardware errors unavailable')
                elif record['errors']:
                    failures.append(label + ': hardware telemetry reports errors')
        record = telemetry.get('gp-service.json')
        counters = ('completed_views', 'completed_evaluations', 'failed_views', 'active', 'queued')
        if (not isinstance(record, dict) or not isinstance(record.get('phase'), str)
                or any(type(record.get(k)) is not int or record[k] < 0 for k in counters)):
            unavailable.append(label + ': invalid GP counters')
            gp.append(None)
        else:
            gp.append(record)
    if number(before.get('timestamp')) and number(after.get('timestamp')) and after['timestamp'] < before['timestamp']:
        unavailable.append('snapshot time moved backwards')
    invocations = [s.get('stable', {}).get('jope-parallel-gp', {}).get('properties', {}).get('InvocationID')
                   for s in (before, after)]
    if not all(isinstance(value, str) and value for value in invocations) or invocations[0] != invocations[1]:
        unavailable.append('GP invocation changed or is unavailable; counters cannot be compared')
    elif all(record is not None for record in gp):
        for name in ('completed_views', 'completed_evaluations', 'failed_views'):
            delta = gp[1][name] - gp[0][name]
            observations[name + '_delta'] = delta
            if delta < 0:
                failures.append(name + ' moved backwards within one service invocation')
            elif name == 'failed_views' and delta > 0:
                failures.append('new failed views observed')
        observations['active_workers_after'] = gp[1]['active']
        observations['failed_views_after'] = gp[1]['failed_views']
        # A legitimate empty/draining queue need not keep all workers active.
        if gp[1]['phase'] == 'running' and gp[1]['active'] == 0 and gp[1].get('queued', 0) > 0:
            failures.append('GP reports queued work with no active workers')
    return {'schema': 'kernel.operational-health.v1',
            'status': 'failed' if failures else 'inconclusive' if unavailable else 'passed',
            'failures': failures, 'unavailable': unavailable, 'observations': observations,
            'scope': 'Observed health over this interval; not causal attribution or quality/throughput qualification'}
