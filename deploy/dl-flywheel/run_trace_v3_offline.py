"""Isolated synthetic trace-recovery-v3 fixtures; no live archive or network."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'pipelines/telemetry/trace_recovery_v3'
FIXTURE_WALLCLOCK = 1791564000.0
REAL_TIME_NS = time.time_ns
EXPECTED_COUNT = 34


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from identities(item)
        else:
            yield item.id()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    need(sys.flags.optimize == 0, 'fixture assertions must remain enabled')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    need(not {'trace_recovery_v3', 'trace_recovery_v2_frozen', 'test_trace_recovery_v3', 'prepare_config', 'launch_v3'}.intersection(sys.modules),
         'new isolated fixture process required')
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()}
    expected = json.loads((SOURCE / 'test-identities.json').read_bytes())
    need(len(expected) == len(set(expected)) == EXPECTED_COUNT, 'exact fixture roster required')
    receipt = {
        'schema': 'jope.trace-v3-offline-tests/v1',
        'source_sha': os.environ.get('JOPE_SOURCE_SHA'),
        'started_utc_ns': REAL_TIME_NS(),
        'fixture_wallclock': FIXTURE_WALLCLOCK,
        'fixture_wallclock_scope': 'This isolated synthetic fixture process only; production GLOBAL and all source bytes unchanged. Only time.time is frozen; time.time_ns, monotonic and CPU clocks remain real.',
        'fixture_path_scope': 'Owned temporary copy of exact source and synthetic fixtures; no prior live captures, raw data, credentials or private deployment paths required.',
        'sources': source_hashes,
        'prior687_tests_executed_by_this_runner': 0,
        'live_archives': 0,
        's3_requests': 0,
        'gpu_execution': False,
        'scope': 'Synthetic CPU/file fixtures only. No live source qualification, upload, execution-window renewal, replay, optimizer admission or release claim.',
    }
    with tempfile.TemporaryDirectory(prefix='trace-v3-', dir=output) as tmp:
        target = Path(tmp) / 'source'
        shutil.copytree(SOURCE, target)
        with patch('time.time', return_value=FIXTURE_WALLCLOCK):
            sys.path.insert(0, str(target))
            suite = unittest.defaultTestLoader.loadTestsFromName('test_trace_recovery_v3')
            names = sorted(identities(suite))
            need(names == expected and len(names) == EXPECTED_COUNT, 'collected fixture identities differ')
            tests = sys.modules['test_trace_recovery_v3']
            need(Path(tests.__file__).resolve() == target / 'test_trace_recovery_v3.py', 'test import escaped private tree')
            production = sys.modules['trace_recovery_v3']
            for module, name in ((production, 'trace_recovery_v3.py'),
                                 (production.v, 'trace_recovery_v2.py'),
                                 (production.a, 'trace_archive_v1_c8a13033.py'),
                                 (tests.freeze_config, 'prepare_config.py'),
                                 (tests.launch, 'launch_v3.py')):
                imported = Path(module.__file__).resolve()
                need(imported == target / name and digest(imported) == digest(SOURCE / name), 'production source/import changed')
            receipt.update(collected=len(names), identities=names)
            if args.collect_only:
                receipt.update(executed=0, passed=None, execution_state='not_run_collection_only')
            else:
                with (output / 'tests.log').open('x') as log:
                    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
                receipt.update(executed=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                               skipped=len(result.skipped), expected_failures=len(result.expectedFailures),
                               passed=result.wasSuccessful() and result.testsRun == EXPECTED_COUNT and not result.skipped and not result.expectedFailures,
                               execution_state='executed', log_sha256=digest(output / 'tests.log'))
    need(source_hashes == {str(p.relative_to(ROOT)): digest(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()},
         'offline fixtures changed packaged source')
    receipt['finished_utc_ns'] = REAL_TIME_NS()
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('identities', 'sources')}))
    if not args.collect_only and receipt['passed'] is not True:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
