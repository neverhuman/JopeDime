#!/usr/bin/env python3
"""Run independent CPU families and report every outcome without turning failures green."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

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


VERSION = '0.15.0'
ARCHIVE = f'sccache-v{VERSION}-x86_64-unknown-linux-musl.tar.gz'
ARCHIVE_SHA256 = '782d2b5dd7ae0a55ebe368ab258114d0928d019ac2d949ab85d5d02f3926709e'
BINARY_SHA256 = 'fcfae9d4f44c5ad74b73953bc052d385ec08f437f3c02f3be10590b863503359'
URL = f'https://github.com/mozilla/sccache/releases/download/v{VERSION}/{ARCHIVE}'


def cache_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def install_compiler_cache(target):
    tools = target / 'compiler-cache-tools'
    version_dir = tools / f'v{VERSION}'
    if tools.is_symlink() or version_dir.is_symlink():
        raise ValueError('unsafe compiler cache tool directory')
    tools.mkdir(exist_ok=True)
    archive = tools / ARCHIVE
    # sccache uses argv[0] to recognize its compiler-wrapper mode. Keep its name.
    version_dir.mkdir(exist_ok=True)
    binary = version_dir / 'sccache'
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open('xb') as stream:
            while data := response.read(1024 * 1024):
                stream.write(data)
    if archive.is_symlink() or cache_digest(archive) != ARCHIVE_SHA256:
        raise ValueError('compiler cache archive checksum mismatch')
    if not binary.exists():
        with tarfile.open(archive) as contents:
            member = contents.getmember(ARCHIVE.removesuffix('.tar.gz') + '/sccache')
            if not member.isfile():
                raise ValueError('compiler cache executable is not a regular file')
            data = contents.extractfile(member).read()
        if hashlib.sha256(data).hexdigest() != BINARY_SHA256:
            raise ValueError('compiler cache executable checksum mismatch')
        with binary.open('xb') as stream:
            stream.write(data)
        binary.chmod(0o755)
    if binary.is_symlink() or cache_digest(binary) != BINARY_SHA256:
        raise ValueError('compiler cache executable checksum mismatch')
    return binary


def setup_compiler_cache(output):
    # PR and trusted-main targets are separate. Never share this cache with PRs.
    if os.environ['JOPE_RUNNER_LANE'] != 'full' or os.environ['GITHUB_REF'] != 'refs/heads/main':
        raise ValueError('compiler cache is restricted to the trusted full lane on main')
    if os.environ['CARGO_INCREMENTAL'] != '0':
        raise ValueError('compiler cache requires incremental compilation disabled')
    target = Path(os.environ['CARGO_TARGET_DIR']).resolve(strict=True)
    if (target / 'compiler-cache').is_symlink():
        raise ValueError('unsafe compiler cache data directory')
    output.mkdir(parents=True, exist_ok=False)
    binary = install_compiler_cache(target)
    conf = output.resolve() / 'sccache.toml'
    conf.write_text('[cache.disk]\ndir = ' + json.dumps(str(target / 'compiler-cache')) + '\nsize = 21474836480\n')
    environment = dict(RUSTC_WRAPPER=str(binary), SCCACHE_CONF=str(conf),
                       SCCACHE_DIR=str(target / 'compiler-cache'), SCCACHE_CACHE_SIZE='20G',
                       SCCACHE_IDLE_TIMEOUT='0',
                       SCCACHE_SERVER_UDS=str(Path(os.environ['RUNNER_TEMP']) / 'full-sccache.sock'))
    subprocess.run([str(binary), '--start-server'], env=dict(os.environ, **environment), check=True, timeout=30)
    with Path(os.environ['GITHUB_ENV']).open('a') as stream:
        for key, value in environment.items():
            stream.write(f'{key}={value}\n')
    (output / 'setup.json').write_text(json.dumps(dict(
        source_sha=os.environ['GITHUB_SHA'], version=VERSION, archive_sha256=ARCHIVE_SHA256,
        executable_sha256=BINARY_SHA256, environment=environment,
        note='Library compile cache only; executable links, test execution and feature families remain unchanged.'), indent=2) + '\n')


def compiler_cache_stats(output):
    binary = Path(os.environ['RUSTC_WRAPPER'])
    if binary.is_symlink() or cache_digest(binary) != BINARY_SHA256:
        raise ValueError('compiler cache executable checksum mismatch')
    data = json.loads(subprocess.check_output([str(binary), '--show-stats', '--stats-format=json'], text=True))
    # Keep the existing family artifact layout intact; publish cache evidence below it.
    setup_dir = Path(os.environ['SCCACHE_CONF']).parent
    record = json.loads((setup_dir / 'setup.json').read_text())
    if record['source_sha'] != os.environ['GITHUB_SHA']:
        raise ValueError('compiler cache setup source mismatch')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'setup.json').write_text(json.dumps(record, indent=2) + '\n')
    (output / 'sccache.toml').write_bytes((setup_dir / 'sccache.toml').read_bytes())
    (output / 'stats.json').write_text(json.dumps(dict(source_sha=os.environ['GITHUB_SHA'], **data), indent=2) + '\n')
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'summarize', 'cache-setup', 'cache-stats'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'cache-setup':
        setup_compiler_cache(args.output)
        return 0
    if args.command == 'cache-stats':
        compiler_cache_stats(args.output)
        return 0
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
