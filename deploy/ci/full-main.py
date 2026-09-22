#!/usr/bin/env python3
"""Run independent CPU families and report every outcome without turning failures green."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import receipt
import results

ROOT = Path(__file__).resolve().parents[2]


def families(mapping):
    return {name: lane for name, lane in mapping['lanes'].items() if name.startswith('full-')}


def run(mapping, output, execute=receipt.run_lane):
    output.mkdir(parents=True, exist_ok=False)
    failed = False
    for name, lane in families(mapping).items():
        destination = output / name
        backend = lane['result_backend']
        junit = destination / 'junit.xml' if backend in ('nextest', 'pytest') else None
        try:
            code = execute(name, lane['command'], destination, backend=backend, junit=junit)
            failed |= code != 0
        except (OSError, ValueError, KeyError) as exc:
            failed = True
            (output / f'{name}-error.txt').write_text(str(exc) + '\n')
        # Keep going after failures, collection errors, unavailable capabilities and timeouts.
    return int(failed)


def summarize(mapping, output, expected):
    inventory, errors = {}, []
    for name, lane in families(mapping).items():
        path = output / name / 'receipt.json'
        try:
            data = json.loads(path.read_text())
            for field in ('source_sha', 'checkout_sha', 'run_id', 'attempt', 'event'):
                if data.get(field) != expected[field]:
                    raise ValueError(f'{field} mismatch')
            if data.get('command') != lane['command'] or data.get('lane') != name:
                raise ValueError('command/lane mismatch')
            if data.get('dirty_source') != [] or data.get('dirty_source_after') != []:
                raise ValueError('dirty or incomplete source state')
            actual = {str(p.relative_to(path.parent)) for p in path.parent.rglob('*') if p.is_file() and p != path}
            if actual != set(data['evidence']):
                raise ValueError('missing or undeclared evidence')
            for filename, digest in data['evidence'].items():
                file = path.parent / filename
                if file.is_symlink() or not file.resolve().is_relative_to(path.parent.resolve()) or receipt.digest(file) != digest:
                    raise ValueError(f'unsafe or changed evidence: {filename}')
            if data.get('schema') != 'jope.ci-receipt/v1':
                raise ValueError('unexpected receipt schema')
            for filename in receipt.INPUTS:
                if data.get('inputs', {}).get(filename) != receipt.digest(ROOT / filename):
                    raise ValueError(f'input mismatch: {filename}')
            counts = data['counts']
            receipt.validate_counts(counts)
            status = data['status']
            if data['exit_status'] == 78:
                status = 'unavailable'
            if status == 'passed':
                if data['exit_status'] != 0 or data.get('diagnostic') or not data.get('finished_at'):
                    raise ValueError('invalid passing receipt')
                backend = lane['result_backend']
                junit = path.parent / 'junit.xml' if backend in ('nextest', 'pytest') else None
                if results.summarize((path.parent / 'command.log').read_text(), backend, junit) != counts:
                    raise ValueError('counts differ from execution evidence')
                if backend == 'nextest':
                    listing = json.loads((path.parent / 'collection.json').read_text())
                    listed = sum(len(suite.get('testcases', {})) for suite in listing['rust-suites'].values())
                    if listing['test-count'] != listed or listed != counts['collected']:
                        raise ValueError('collection and execution disagree')
                if lane['requires_nonzero_tests'] and (any(value is None for value in counts.values()) or counts['executed'] <= 0 or counts['failed'] != 0):
                    raise ValueError('zero, unknown or failing tests')
            if status != 'passed':
                errors.append(f'{name}: {status} (exit {data["exit_status"]})')
            inventory[name] = {'status': status, 'exit_status': data['exit_status'], 'tests': counts,
                               'diagnostic': data.get('diagnostic'), 'receipt_sha256': receipt.digest(path)}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f'{name}: {exc}')
            inventory[name] = {'status': 'missing-or-invalid', 'tests': None, 'diagnostic': str(exc)}
    return {'schema': 'jope.full-cpu/v1', **expected, 'diagnostic_only': True,
            'status': 'failed' if errors else 'passed', 'families': inventory, 'errors': errors,
            'excluded_dispositions': mapping['dispositions'],
            'note': 'Ignored tests and Python xbabe2-marked tests remain explicit non-execution; research scripts are not collected tests. Qualification and soak are unstarted.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'summarize'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mapping = json.loads((ROOT / 'deploy/ci/coverage.json').read_text())
    if args.command == 'run':
        return run(mapping, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    expected = receipt.identity(os.environ, subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    report = summarize(mapping, args.output, expected)
    with (args.output / 'inventory.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write(f"Full CPU diagnostic at `{expected['source_sha']}`: **{report['status']}**\n\n")
            stream.write('| Family | Status | Passed | Failed | Skipped | Excluded |\n|---|---|---:|---:|---:|---:|\n')
            for name, result in report['families'].items():
                counts = result.get('tests') or {}
                values = ['unknown' if counts.get(k) is None else str(counts[k]) for k in ('passed','failed','skipped','excluded')]
                stream.write('| ' + ' | '.join([name, result['status'], *values]) + ' |\n')
    return report['status'] != 'passed'


if __name__ == '__main__':
    sys.exit(main())
