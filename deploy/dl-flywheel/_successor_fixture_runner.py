"""Portable fixture custody for the exact frozen next128/trace-v2 sources.

Production modules and deadlines are not edited. A private temporary tree holds
derived fixture JSON references, and only this isolated test process uses a fixed
wallclock. It cannot create a production queue or acquire deployment authority.
"""
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
FIXTURE_WALLCLOCK = 1791550000.0
REAL_TIME_NS = time.time_ns
KINDS = {
    'next128': ('pipelines/encoding/next128', 'test_next128', 25),
    'trace_v2': ('pipelines/telemetry/trace_recovery_v2', 'test_trace_recovery_v2', 13),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False))


def stage_next128(source, scratch):
    target = scratch / 'kumo-xbabe3-next128/revision2'
    shutil.copytree(source, target)
    original = ROOT / 'pipelines/encoding/fresh32'
    comparison = scratch / 'kumo-xbabe3-fresh32'
    comparison.mkdir()
    for name in ('kumo_controller.py', 'kumo_queue.py', 'selection.json'):
        shutil.copyfile(original / name, comparison / name)
        need(digest(comparison / name) == digest(original / name), 'fresh32 comparison bytes changed')
    top = json.loads((target / 'config.template.json').read_bytes())
    old_root = Path(top['orchestrator']['path']).parent

    def rebased(old):
        path = Path(old['path'])
        need(path.is_relative_to(old_root), 'only pinned local fixture refs may be rebased')
        new = target / path.relative_to(old_root)
        return ref(new)

    for packet in top['packets']:
        folder = target / ('packet' + str(packet['ordinal']))
        path = folder / 'config.template.json'
        child = json.loads(path.read_bytes())
        child['local_source_pins'] = [rebased(item) for item in child['local_source_pins']]
        child['local_publisher'] = str(folder / 'kumo_publish.py')
        child['local_base'] = str(folder / 'pilot_publish_core.py')
        # Remote source pins/identities/selection rows/production time ceilings
        # stay byte-for-byte represented. This JSON is private test data only.
        write_json(path, child)
        packet['template'] = ref(path)
        packet['controller'] = ref(folder / 'kumo_controller.py')
    top['selection'] = rebased(top['selection'])
    top['source_pins'] = [rebased(item) for item in top['source_pins']]
    top['orchestrator'] = rebased(top['orchestrator'])
    top['freezer'] = rebased(top['freezer'])
    write_json(target / 'config.template.json', top)
    for original_path in source.rglob('*.py'):
        need(digest(target / original_path.relative_to(source)) == digest(original_path), 'fixture stage modified Python source')
    return target


def identities(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from identities(item)
        else:
            yield item.id()


def main(kind):
    source_rel, module_name, expected_count = KINDS[kind]
    source = ROOT / source_rel
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    imported = {'run_next128', 'freeze_next128', 'campaign', 'kumo_queue', 'trace_recovery_v2', 'trace_archive_v1_c8a13033', 'test_next128', 'test_trace_recovery_v2'}
    need(not imported.intersection(sys.modules), 'new fixtures require isolated imports')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
    expected = json.loads((source / 'test-identities.json').read_bytes())
    receipt = {'schema': 'jope.' + kind.replace('_', '-') + '-offline-tests/v1',
               'source_sha': os.environ.get('JOPE_SOURCE_SHA'), 'started_utc_ns': REAL_TIME_NS(),
               'fixture_wallclock': FIXTURE_WALLCLOCK,
               'fixture_wallclock_scope': 'Isolated fixture process only; production GLOBAL and all Python source bytes unchanged. Monotonic clocks remain real.',
               'fixture_path_scope': 'Owned temporary source copy; only derived local fixture JSON references are rebased, never remote/runtime pins or production source.',
               'sources': source_hashes, 'gp_executions': 0, 's3_requests': 0,
               'gpu_execution': False, 'production_queue_mutations': 0,
               'prior540_tests_executed_by_this_runner': 0,
               'scope': 'New offline fixtures only. No live preflight, deployment, archive, training admission, source promotion, or serving claim.'}
    with tempfile.TemporaryDirectory(prefix='fixture-', dir=args.output) as tmp:
        scratch = Path(tmp)
        if kind == 'next128':
            target = stage_next128(source, scratch)
        else:
            target = scratch / 'trace_recovery_v2'
            shutil.copytree(source, target)
        with patch('time.time', return_value=FIXTURE_WALLCLOCK):
            sys.path.insert(0, str(target))
            suite = unittest.defaultTestLoader.loadTestsFromName(module_name)
            names = sorted(identities(suite))
            need(names == expected and len(names) == len(set(names)) == expected_count, 'exact additive fixture roster differs')
            need(Path(sys.modules[module_name].__file__).resolve() == target / (module_name + '.py'), 'test import escaped private tree')
            modules = ('run_next128', 'freeze_next128') if kind == 'next128' else ('trace_recovery_v2',)
            for name in modules:
                module = sys.modules[name]
                need(Path(module.__file__).resolve() == target / (name + '.py') and digest(Path(module.__file__)) == digest(source / (name + '.py')), 'production module import/source changed')
            receipt.update(collected=len(names), identities=names)
            if args.collect_only:
                receipt.update(executed=0, passed=None, execution_state='not_run_collection_only')
            else:
                with (args.output / 'tests.log').open('x') as log:
                    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
                receipt.update(executed=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                               skipped=len(result.skipped), expected_failures=len(result.expectedFailures),
                               passed=result.wasSuccessful() and result.testsRun == expected_count and not result.skipped and not result.expectedFailures,
                               execution_state='executed', log_sha256=digest(args.output / 'tests.log'))
    need(source_hashes == {str(p.relative_to(ROOT)): digest(p) for p in sorted(source.rglob('*')) if p.is_file()}, 'offline fixtures changed packaged source')
    receipt['finished_utc_ns'] = REAL_TIME_NS()
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('identities', 'sources')}))
    if not args.collect_only and receipt['passed'] is not True:
        raise SystemExit(1)
