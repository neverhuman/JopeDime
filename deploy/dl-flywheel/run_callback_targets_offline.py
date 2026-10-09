"""Isolated, exact-roster callback revisions and reference-compatibility fixtures.

The two repeated revision2 configuration identities are not executed again.
This wrapper never changes frozen package roots or their original runners.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import time
import types
import unittest

ROOT = Path(__file__).resolve().parent
ROSTER = ROOT / 'pipelines/generated_target_callbacks_v1/integration-fixtures.json'
ROSTER_SHA256 = 'b54c059aebd34e750da06f0617469ee47e629da82bf79f00e08177c3abd03874'
EXPECTED_COUNTS = {'revision1': 37, 'revision2_new': 7, 'revision3': 11, 'compatibility': 8}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_roster():
    need(digest(ROSTER) == ROSTER_SHA256, 'frozen integration roster hash')
    value = json.loads(ROSTER.read_bytes())
    need(set(value['groups']) == set(EXPECTED_COUNTS), 'exact group set')
    ids = []
    for name, group in value['groups'].items():
        current = group['fixture_ids']
        need(current == sorted(current) and len(current) == len(set(current)) == EXPECTED_COUNTS[name], 'exact group roster')
        ids.extend(group['identity_prefix'] + x for x in current)
    need(len(ids) == len(set(ids)) == 63, 'exact unique 55 callback plus 8 compatibility identities')
    return value


def closure(group):
    for rel, expected in group['source_pins'].items():
        path = ROOT / rel
        need(path.is_file() and not path.is_symlink() and digest(path) == expected, 'source pin: ' + rel)
    for prefix in group['python_roots']:
        need(not any((ROOT / prefix).rglob('*.pyc')), 'preexisting bytecode refused: ' + prefix)
        actual = {str(p.relative_to(ROOT)) for p in (ROOT / prefix).rglob('*.py')}
        expected = {x for x in group['source_pins'] if x.startswith(prefix + '/') and x.endswith('.py')}
        need(actual == expected, 'closed Python root: ' + prefix)


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def child(name, output, collect_only):
    group = read_roster()['groups'][name]
    closure(group)
    namespace = '_generated_target_callbacks_qualification' if name != 'compatibility' else '_target_reference_compatibility'
    need(not any(x == namespace or x.startswith(namespace + '.') for x in sys.modules), 'fresh child namespace')
    if name == 'compatibility':
        parent = types.ModuleType(namespace)
        parent.__path__ = [str(ROOT / 'pipelines')]
        sys.modules[namespace] = parent
        load_prefix = namespace + '.generated_target_reference_export_v2.'
    else:
        package_root = ROOT / group['package_root']
        spec = importlib.util.spec_from_file_location(namespace, package_root / '__init__.py', submodule_search_locations=[str(package_root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = module
        spec.loader.exec_module(module)
        load_prefix = namespace + '.'
    suite = unittest.TestSuite()
    for identity in group['fixture_ids']:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(load_prefix + identity))
    actual = sorted(t.id()[len(load_prefix):] for t in flatten(suite))
    need(actual == group['fixture_ids'], 'exact collected fixture identities')
    started = time.time_ns()
    mono = time.monotonic_ns()
    stream = io.StringIO()
    if collect_only:
        result = None
        passed = None
    else:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        passed = result.testsRun == len(actual) and not any((result.failures, result.errors, result.skipped, result.expectedFailures, result.unexpectedSuccesses))
    closure(group)
    origins = {}
    for key, module in list(sys.modules.items()):
        if key == namespace or key.startswith(namespace + '.'):
            if name == 'compatibility' and key == namespace:
                continue  # The one explicit namespace container, not a source module.
            path = Path(module.__file__).resolve()
            need(path.is_relative_to(ROOT), 'foreign package origin')
            rel = str(path.relative_to(ROOT))
            need(rel in group['source_pins'] and digest(path) == group['source_pins'][rel], 'import source pin: ' + rel)
            origins[key] = rel
    text = stream.getvalue().encode()
    receipt = {'schema': 'jope.callback-integration-child/v1', 'group': name,
               'collected': len(actual), 'executed': 0 if collect_only else result.testsRun,
               'fixture_ids': actual, 'passed': passed, 'collection_only': collect_only,
               'source_pins': group['source_pins'], 'loaded_package_origins': origins,
               'roster_sha256': ROSTER_SHA256, 'started_utc_ns': started,
               'ended_utc_ns': time.time_ns(), 'monotonic_elapsed_ns': time.monotonic_ns() - mono,
               'test_output': {'bytes': len(text), 'sha256': hashlib.sha256(text).hexdigest(), 'base64': base64.b64encode(text).decode()},
               'live_qualification': False, 'optimizer_admission': False}
    with (output / (name + '.json')).open('x') as f:
        json.dump(receipt, f, sort_keys=True, indent=2)
        f.write('\n')
    return 0 if collect_only or passed else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--collect-only', action='store_true')
    ap.add_argument('--child', choices=sorted(EXPECTED_COUNTS))
    args = ap.parse_args()
    need(sys.flags.optimize == 0, 'fixture assertions must remain enabled')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'
    output = args.output.resolve()
    if args.child:
        need(output.is_dir(), 'parent-created output directory')
        return child(args.child, output, args.collect_only)
    roster = read_roster()
    output.mkdir(parents=True, exist_ok=False)
    receipts = {}
    passed = True
    identities = []
    for name in EXPECTED_COUNTS:
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--output', str(output), '--child', name]
        if args.collect_only:
            command.append('--collect-only')
        process = subprocess.run(command, check=False)
        need(process.returncode == 0, 'isolated fixture child failed: ' + name)
        receipt = json.loads((output / (name + '.json')).read_bytes())
        group = roster['groups'][name]
        need(receipt['group'] == name and receipt['fixture_ids'] == group['fixture_ids'] and receipt['collected'] == EXPECTED_COUNTS[name], 'child receipt roster mismatch')
        need(receipt['executed'] == (0 if args.collect_only else EXPECTED_COUNTS[name]), 'child execution count mismatch')
        need(receipt['collection_only'] is args.collect_only and receipt['roster_sha256'] == ROSTER_SHA256, 'child mode/roster mismatch')
        if not args.collect_only:
            passed = passed and receipt['passed'] is True
        receipts[name] = receipt
        identities.extend(group['identity_prefix'] + x for x in group['fixture_ids'])
    need(len(identities) == len(set(identities)) == 63, 'combined exact unique identities')
    final = {'schema': 'jope.callback-integration-offline/v1', 'collected': 63,
             'executed': sum(x['executed'] for x in receipts.values()), 'identities': identities,
             'passed': None if args.collect_only else passed, 'collection_only': args.collect_only,
             'children': receipts, 'prior749_executed_by_this_runner': 0,
             'callback_unique': 55, 'compatibility_unique': 8,
             'live_qualification': False, 'optimizer_admission': False}
    with (output / 'receipt.json').open('x') as f:
        json.dump(final, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in final.items() if k not in ('children', 'identities')}))
    return 0 if args.collect_only or passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
