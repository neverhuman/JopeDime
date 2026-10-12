"""Shared pinned offline runner for the C10 revision-2 packages (generated_target_preparation_v2, boundary_admission_v2).

Each package's run_offline.py calls ``run(package_root, expected_ids)``.  Before importing fixture code it checks the
exact package Python roster against SOURCE_PINS.json and the base export against its git-verified BASE_MANIFEST.json
(``JOPE_C10_BASE_ROOT`` selects the export; its manifest digest must equal ``JOPE_C10_BASE_MANIFEST_SHA256``).  It
then runs exactly the registered fixture identities in this process with the successor-layout sys.path and, after
the run, proves that every imported module file under the base or candidate root is a pinned byte-exact file
(review finding: the imported base origin must be independently hashed, never trusted from the environment).
Revision 6: ``pipelines/hpc_verified_v5`` is appended last (the native-result overlay fixture world needs it), and an
imported candidate file must be pinned by the SOURCE_PINS.json of the candidate package that contains it (the
byte-exact W3 overlay copy included); the receipt names every candidate package pin digest.
Skips, expected failures, unexpected successes, a changed roster or a changed count are failures.  CPU only; no
network, S3, GPU or GP execution.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import unittest
from pathlib import Path

PINS_SCHEMA = 'jope.c10-offline-source-pins/v2'
RECEIPT_SCHEMA = 'jope.c10-offline-qualification/v2'
BASE_SCHEMA = 'jope.c10-base-export-manifest/v1'


def _require(ok, why):
    if not ok:
        raise RuntimeError(why)


def source_pins(root):
    paths = sorted(p for p in root.rglob('*.py') if '__pycache__' not in p.parts)
    _require(not list(root.rglob('*.pyc')), 'no precompiled package files')
    out = {}
    for path in paths:
        _require(path.is_file() and not path.is_symlink(), 'regular source file: ' + str(path))
        raw = path.read_bytes()
        out[str(path.relative_to(root))] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return out


def _ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _ids(item)
        else:
            yield item.id()


def run(package_root, expected_ids, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args(argv)
    package_root = Path(package_root).resolve()
    name = package_root.name
    df = package_root.parents[1]
    pins_raw = (package_root / 'SOURCE_PINS.json').read_bytes()
    pins = json.loads(pins_raw)
    _require(pins['schema'] == PINS_SCHEMA and pins['package'] == name and
             pins['fixture_ids'] == list(expected_ids), 'closed pin manifest and fixture roster')
    _require(source_pins(package_root) == pins['source_pins'], 'pinned package source bytes changed')
    sys.dont_write_bytecode = True
    base = Path(os.environ['JOPE_C10_BASE_ROOT']).resolve()
    base_raw = (base.parents[1] / 'BASE_MANIFEST.json').read_bytes()
    _require(hashlib.sha256(base_raw).hexdigest() == os.environ['JOPE_C10_BASE_MANIFEST_SHA256'],
             'pinned base export manifest')
    base_manifest = json.loads(base_raw)
    _require(base_manifest['schema'] == BASE_SCHEMA and base_manifest['subtree'] == 'deploy/dl-flywheel',
             'base export manifest schema')
    sys.path[:0] = [str(base / 'pipelines/prepared'), str(base / 'pipelines/continuous'),
                    str(base / 'pipelines/continuous/consumer'), str(base / 'pipelines/continuous/runtime'),
                    str(base), str(df), str(base / 'tests'), str(base / 'tests_native'),
                    str(base / 'pipelines/hpc_verified_v5')]
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    started, cpu0 = time.time(), os.times()
    suite = unittest.defaultTestLoader.loadTestsFromNames(sorted({i.rsplit('.', 1)[0] for i in expected_ids}))
    ids = sorted(_ids(suite))
    _require(ids == sorted(expected_ids), 'exact fixture identities')
    log = args.receipt.with_suffix('.log')
    with log.open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    cpu1 = os.times()
    closure, foreign = {}, []
    package_pins, candidate_packages = {}, {}
    for pin_path in sorted((df / 'pipelines').glob('*/SOURCE_PINS.json')):
        raw = pin_path.read_bytes()
        candidate_packages[pin_path.parent.name] = hashlib.sha256(raw).hexdigest()
        for k, v in json.loads(raw)['source_pins'].items():
            package_pins[str((pin_path.parent / k).resolve())] = v['sha256']
    for mod in list(sys.modules.values()):
        path = getattr(mod, '__file__', None)
        if not path or not path.endswith('.py'):
            continue
        path = Path(path).resolve()
        if base in path.parents:
            rel = 'deploy/dl-flywheel/' + str(path.relative_to(base))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            _require(base_manifest['files'].get(rel, {}).get('sha256') == digest, 'imported base file not pinned: ' + rel)
            closure[rel] = digest
        elif df in path.parents:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            _require(package_pins.get(str(path)) == digest, 'imported candidate file not pinned: ' + str(path))
            closure['candidate:' + str(path.relative_to(df))] = digest
    import numpy
    import torch
    passed = (result.wasSuccessful() and result.testsRun == len(expected_ids) and not result.skipped and
              not result.expectedFailures and not result.unexpectedSuccesses)
    receipt = {'schema': RECEIPT_SCHEMA, 'package': name, 'passed': passed, 'collected': len(ids),
               'executed': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'expected_failures': len(result.expectedFailures),
               'fixture_ids': ids, 'source_pins_sha256': hashlib.sha256(pins_raw).hexdigest(),
               'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(), 'started_unix': started,
               'finished_unix': time.time(), 'cpu_user_seconds': cpu1.user - cpu0.user,
               'cpu_system_seconds': cpu1.system - cpu0.system, 'python': platform.python_version(),
               'numpy': numpy.__version__, 'torch': torch.__version__, 'base_root': str(base),
               'base_manifest_sha256': hashlib.sha256(base_raw).hexdigest(), 'base_commit': base_manifest['commit'],
               'base_subtree_tree_id': base_manifest['subtree_tree_id'], 'import_closure': dict(sorted(closure.items())),
               'candidate_package_pins_sha256': candidate_packages,
               'scope': 'CPU fixtures only; no live optimizer, planner, outbox, S3, GPU or GP execution.'}
    with args.receipt.open('x') as stream:
        stream.write(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: receipt[k] for k in ('package', 'passed', 'collected', 'executed', 'failures', 'errors')}))
    if not passed:
        raise SystemExit(1)
