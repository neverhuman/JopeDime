#!/usr/bin/env python3
"""Strict Node fixture bridge for Python CI; no browser or service execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def write_new(path, value):
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    require(not output.exists() and not output.is_symlink(), 'fresh output; no historical receipt replacement')
    require(output.parent.is_dir(), 'existing output parent')
    destination = output.parent.resolve() / output.name
    require(destination != ROOT and ROOT not in destination.parents and destination not in ROOT.parents,
            'evidence outside dashboard source')
    output.mkdir(mode=0o700)
    start, mono = time.time_ns(), time.monotonic_ns()
    write_new(output / 'intent.json', {'schema': 'jope.dashboard-fixture-bridge-intent/v1',
              'started_utc_ns': start, 'runtime_browser_or_services': False})
    receipt = {'schema': 'jope.dashboard-fixture-bridge/v1', 'passed': False, 'collected': 0,
               'executed': 0, 'identities': [], 'scripts': {}, 'production_service_runs': 0,
               'optimizer_updates': 0, 'actual_runtime_qualified': False}
    exitcode = 1
    try:
        require(__debug__, 'non-optimized wrapper')
        node = shutil.which('node')
        require(node is not None, 'Node is required; missing runtime is a failure, never a skip')
        plan_path = ROOT / 'DASHBOARD_FIXTURE_ROSTER.json'
        require(not plan_path.is_symlink(), 'regular dashboard roster')
        plan_raw = plan_path.read_bytes(); plan = json.loads(plan_raw)
        receipt['roster_sha256'] = hashlib.sha256(plan_raw).hexdigest()
        def verify():
            observed = {}
            for name, ref in plan['source_pins'].items():
                require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                        'contained dashboard source member')
                path = ROOT / name
                require(path.is_file() and not path.is_symlink(), 'regular pinned dashboard member')
                body = path.read_bytes()
                actual = {'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)}
                require(actual == ref, 'exact reviewed dashboard source ' + name)
                observed[name] = actual
            return observed
        receipt['source_before'] = verify()
        require(len(plan['scripts']) >= 4 and len({s['path'] for s in plan['scripts']}) == len(plan['scripts']),
                'closed unique dashboard and optimizer-display script roster')
        for entry in plan['scripts']:
            script = entry['path']
            require(script in plan['source_pins'] and script.endswith('.cjs'), 'pinned CJS only')
            command = [node, str(ROOT / script)]
            if entry.get('receipt_style') == 'liveness':
                require(entry['renderer'] in plan['source_pins'] and entry['original_html'] in plan['source_pins'],
                        'renderer and original source belong to exact source roster')
                html = (ROOT / 'index.html').read_text()
                prefix = 'function optimizerDisplay('
                suffix = '\nfunction render('
                require(html.count(prefix) == 1 and html.count(suffix) == 1,
                        'unique inline optimizer display and render boundary')
                begin, end = html.index(prefix), html.index(suffix)
                require(end > begin, 'ordered exact optimizer function range')
                function = html[begin:end].rstrip() + '\n'
                renderer = (ROOT / entry['renderer']).read_text()
                require(renderer.count(prefix) == 1 and
                        function == renderer[renderer.index(prefix):].rstrip() + '\n',
                        'actual final HTML function equals frozen qualified renderer')
                extracted = output / 'optimizer-display.js'
                fd = os.open(extracted, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(function.encode()); stream.flush(); os.fsync(stream.fileno())
                receipt['actual_inline_optimizer_function'] = {
                    'sha256': hashlib.sha256(function.encode()).hexdigest(), 'bytes': len(function.encode())}
                command.extend([str(extracted), str(ROOT / entry['original_html']), str(ROOT / 'index.html')])
            child = subprocess.run(command, check=False, timeout=45,
                                   capture_output=True, cwd=ROOT)
            require(len(child.stdout) <= 65536 and len(child.stderr) <= 65536, 'bounded fixture output')
            require(child.returncode == 0 and not child.stderr.strip(), 'CJS successful exit, no stderr')
            result = json.loads(child.stdout)
            require(result['schema'] == entry['schema'], 'known success receipt schema')
            if entry.get('receipt_style') == 'liveness':
                require(result['state'] == 'passed' and result['tests_failed'] == 0 and
                        result['tests_skipped'] == 0 and result['tests_expected_failed'] == 0 and
                        result['production_imports'] == 0 and result['browser_or_server_execution'] == 0,
                        'strict liveness fixture success without runtime execution')
                executed = result['tests_executed']
                require(result['tests_collected'] == executed, 'liveness collection count')
            else:
                require(result['failed'] == 0 and result['production_service_runs'] == 0, 'no service execution')
                if 'optimizer_updates' in result:
                    require(result['optimizer_updates'] == 0, 'no optimizer execution')
                executed = result['executed']
            cases = result['cases']
            if entry['case_shape'] in ('named_states', 'ids_states'):
                require(all(type(row) is dict and row.get('state') == 'passed' for row in cases),
                        'each CJS case passed')
                names = [row['id' if entry['case_shape'] == 'ids_states' else 'name'] for row in cases]
            else:
                require(all(type(name) is str for name in cases), 'exact string cases')
                names = cases
            require(executed == len(entry['cases']) and sorted(names) == sorted(entry['cases'])
                    and len(names) == len(set(names)), 'exact unique script fixture roster')
            receipt['scripts'][script] = result
            receipt['collected'] += len(names); receipt['executed'] += executed
            receipt['identities'].extend(script + ':' + name for name in names)
        expected = sum(len(entry['cases']) for entry in plan['scripts'])
        require(expected == plan['wired_fixture_count'] and expected >= 34 and
                receipt['collected'] == receipt['executed'] == expected and
                len(set(receipt['identities'])) == expected, 'complete concrete dashboard identities')
        receipt['source_after'] = verify()
        require(receipt['source_before'] == receipt['source_after'], 'source unchanged after fixtures')
        receipt.update(passed=True, state='passed_synthetic_cjs_only')
        exitcode = 0
    except BaseException as error:
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)[:4096]}
    receipt.update(ended_utc_ns=time.time_ns(), elapsed_monotonic_ns=time.monotonic_ns() - mono)
    write_new(output / 'receipt.json', receipt)
    print(json.dumps({'passed': receipt['passed'], 'executed': receipt['executed']}))
    raise SystemExit(exitcode)


if __name__ == '__main__':
    main()
