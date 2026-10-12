#!/usr/bin/env python3
"""Run one explicitly selected synthetic family in an isolated interpreter."""
import argparse
import contextlib
import io
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

sys.dont_write_bytecode = True
from offline_adapter import (FAMILIES, flatten, imported_origins, load,
                             prepare_output, require, verify_source, write_new)

ROOT = Path(__file__).resolve().parent



def verify_external_ci(require_import=False):
    declaration = json.loads((ROOT / 'CI_SOURCE_PIN.json').read_bytes())
    require(set(declaration) == {'schema', 'path', 'sha256', 'bytes'} and
            declaration['schema'] == 'jope.standalone-monitor-ci-source-pin/v1' and
            declaration['path'] == 'deploy/ci/ci-plan.py', 'closed owning CI source pin')
    path = ROOT.parents[3] / declaration['path']
    require(path.is_file() and not path.is_symlink(), 'regular external CI source')
    raw = path.read_bytes()
    require(len(raw) == declaration['bytes'] and hashlib.sha256(raw).hexdigest() == declaration['sha256'],
            'exact owning CI source')
    if require_import:
        module = sys.modules.get('monitor_ci_plan_adapter_v1')
        require(module is not None and Path(module.__file__).resolve() == path.resolve(),
                'actual guarded CI module origin')
    return {'path': declaration['path'], 'absolute_origin': str(path.resolve()),
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True, choices=FAMILIES)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(__debug__, 'non-optimized interpreter required for frozen synthetic assertions')
    output = prepare_output(args.output, ROOT, args.family)
    start, mono, cpu = time.time_ns(), time.monotonic_ns(), time.process_time_ns()
    write_new(output / 'intent.json', {'schema': 'jope.standalone-paper-monitor-offline-intent/v1',
              'family': args.family, 'started_utc_ns': start, 'monotonic_start_ns': mono,
              'scope': 'synthetic CPU only; no runtime admission, GP, services or network'})
    receipt = {'schema': 'jope.standalone-paper-monitor-offline-family/v1', 'family': args.family,
               'passed': False, 'state': 'failed_before_tests', 'collected': 0, 'executed': 0,
               'identities': [], 'runtime_or_remote_execution': False, 'optimizer_updates': 0}
    exitcode = 1
    try:
        receipt['source_before'] = verify_source(ROOT)
        if args.family == 'adapter_v1':
            receipt['external_ci_before'] = verify_external_ci()
        roster = json.loads((ROOT / 'FIXTURE_ROSTER.json').read_bytes())['families'][args.family]
        if args.family == 'incremental_v1':
            base = ROOT / args.family
            load('monitor_incremental_custody_v1', base / 'package/__init__.py', True)
            load('monitor_incremental_custody_v1_tests', base / 'tests/__init__.py', True)
            module = load('monitor_incremental_custody_v1_tests.test_custody', base / 'tests/test_custody.py')
        elif args.family == 'capture_reuse_v3':
            base = ROOT / args.family
            load('monitor_capture_reuse_v1', base / 'package/__init__.py', True)
            load('monitor_capture_reuse_v1_tests', base / 'tests/__init__.py', True)
            module = load('monitor_capture_reuse_v1_tests.test_reuse', base / 'tests/test_reuse.py')
        elif args.family == 'budget_preflight_v1':
            base = ROOT / args.family
            load('preflight', base / 'preflight.py')
            module = load('test_preflight', base / 'test_preflight.py')
        else:
            load('monitor_portable_adapter_v1_tests', ROOT / 'tests/__init__.py', True)
            module = load('monitor_portable_adapter_v1_tests.test_adapter', ROOT / 'tests/test_adapter.py')
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        identities = [case.id() for case in flatten(suite)]
        require(len(identities) == len(set(identities)) and sorted(identities) == roster,
                'exact unique fixture roster; no failed import or implicit collection')
        receipt['identities'], receipt['collected'] = identities, len(identities)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        receipt.update(executed=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                       skipped=len(result.skipped), expected_failures=len(result.expectedFailures),
                       unexpected_successes=len(result.unexpectedSuccesses), log=stream.getvalue())
        require(result.testsRun == len(roster) and result.wasSuccessful() and not
                (result.skipped or result.expectedFailures or result.unexpectedSuccesses),
                'all exact fixtures execute without skip or expected failure')
        receipt['imported_origins'] = imported_origins(ROOT)
        if args.family == 'adapter_v1':
            receipt['external_ci_after'] = verify_external_ci(require_import=True)
            require(receipt['external_ci_before'] == receipt['external_ci_after'], 'CI source unchanged')
        receipt['source_after'] = verify_source(ROOT)
        require(receipt['source_after'] == receipt['source_before'], 'immutable source after fixture run')
        receipt.update(passed=True, state='passed_synthetic_only')
        exitcode = 0
    except BaseException as error:
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)[:4096]}
    receipt.update(ended_utc_ns=time.time_ns(), elapsed_monotonic_ns=time.monotonic_ns() - mono,
                   process_cpu_ns=time.process_time_ns() - cpu)
    write_new(output / 'receipt.json', receipt)
    print(json.dumps({'family': args.family, 'passed': receipt['passed'], 'executed': receipt['executed']}))
    raise SystemExit(exitcode)


if __name__ == '__main__':
    main()
