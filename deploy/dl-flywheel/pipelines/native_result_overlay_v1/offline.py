"""Pinned offline runner for native_result_overlay_v1 (derived from the C10 r2 runner, generated_target_preparation_v2).

Before importing fixture code it checks the exact package Python roster against SOURCE_PINS.json, the base export
against its git-verified BASE_MANIFEST.json (``JOPE_C10_BASE_ROOT``; manifest digest ``JOPE_C10_BASE_MANIFEST_SHA256``),
the C10 r4 candidate root (``JOPE_W3_C10_R4_ROOT``) against r4's own SOURCE_PINS files pinned here, and the real R6
preregistration (``JOPE_W3_R6_PATH``) against its pinned digest. It then runs exactly the registered fixture identities
with the successor sys.path (+ ``pipelines/hpc_verified_v5``, + the r4 candidate root) and afterwards proves every
imported module file under the base, this package or the r4 root is a pinned byte-exact file. Skips, expected
failures, unexpected successes, a changed roster or a changed count fail. CPU only; no network, S3, GPU or GP.
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

PINS_SCHEMA = 'jope.w3-offline-source-pins/v1'
RECEIPT_SCHEMA = 'jope.w3-offline-qualification/v1'
BASE_SCHEMA = 'jope.c10-base-export-manifest/v1'


def _require(ok, why):
    if not ok:
        raise RuntimeError(why)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_pins(root):
    paths = sorted(p for p in root.rglob('*.py') if '__pycache__' not in p.parts)
    _require(not list(root.rglob('*.pyc')), 'no precompiled package files')
    out = {}
    for path in paths:
        _require(path.is_file() and not path.is_symlink(), 'regular source file: ' + str(path))
        raw = path.read_bytes()
        out[str(path.relative_to(root))] = {'sha256': _sha(raw), 'bytes': len(raw)}
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
    _require(pins['schema'] == PINS_SCHEMA and pins['package'] == name and pins['fixture_ids'] == list(expected_ids),
             'closed pin manifest and fixture roster')
    _require(source_pins(package_root) == pins['source_pins'], 'pinned package source bytes changed')
    sys.dont_write_bytecode = True
    base = Path(os.environ['JOPE_C10_BASE_ROOT']).resolve()
    base_raw = (base.parents[1] / 'BASE_MANIFEST.json').read_bytes()
    _require(_sha(base_raw) == os.environ['JOPE_C10_BASE_MANIFEST_SHA256'] == pins['base_manifest_sha256'],
             'pinned base export manifest')
    base_manifest = json.loads(base_raw)
    _require(base_manifest['schema'] == BASE_SCHEMA and base_manifest['subtree'] == 'deploy/dl-flywheel',
             'base export manifest schema')
    r4 = Path(os.environ['JOPE_W3_C10_R4_ROOT']).resolve()
    r4_pins = {}
    for package, pin in pins['c10_r4_source_pins'].items():
        raw = (r4 / 'pipelines' / package / 'SOURCE_PINS.json').read_bytes()
        _require(_sha(raw) == pin, 'pinned C10 r4 SOURCE_PINS: ' + package)
        for rel, v in json.loads(raw)['source_pins'].items():
            r4_pins[str((r4 / 'pipelines' / package / rel).resolve())] = v['sha256']
    r6_raw = Path(os.environ['JOPE_W3_R6_PATH']).read_bytes()
    _require(_sha(r6_raw) == pins['r6_preregistration_sha256'], 'pinned R6 preregistration bytes')
    sys.path[:0] = [str(base / 'pipelines/prepared'), str(base / 'pipelines/continuous'),
                    str(base / 'pipelines/continuous/consumer'), str(base / 'pipelines/continuous/runtime'),
                    str(base), str(df), str(r4), str(base / 'tests'), str(base / 'tests_native'),
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
    closure = {}
    package_pins = {str((package_root / k).resolve()): v['sha256'] for k, v in pins['source_pins'].items()}
    for mod in list(sys.modules.values()):
        path = getattr(mod, '__file__', None)
        if not path or not path.endswith('.py'):
            continue
        path = Path(path).resolve()
        if base in path.parents:
            digest = _sha(path.read_bytes())
            rel = 'deploy/dl-flywheel/' + str(path.relative_to(base))
            _require(base_manifest['files'].get(rel, {}).get('sha256') == digest, 'imported base file not pinned: ' + rel)
            closure[rel] = digest
        elif package_root in path.parents:
            digest = _sha(path.read_bytes())
            _require(package_pins.get(str(path)) == digest, 'imported package file not pinned: ' + str(path))
            closure['candidate:' + str(path.relative_to(df))] = digest
        elif r4 in path.parents:
            digest = _sha(path.read_bytes())
            _require(r4_pins.get(str(path)) == digest, 'imported C10 r4 file not pinned: ' + str(path))
            closure['c10-r4:' + str(path.relative_to(r4))] = digest
        elif df in path.parents:
            _require(False, 'imported candidate file outside the pinned package: ' + str(path))
    import numpy
    import torch
    passed = (result.wasSuccessful() and result.testsRun == len(expected_ids) and not result.skipped and
              not result.expectedFailures and not result.unexpectedSuccesses)
    receipt = {'schema': RECEIPT_SCHEMA, 'package': name, 'passed': passed, 'collected': len(ids),
               'executed': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'expected_failures': len(result.expectedFailures),
               'unexpected_successes': len(result.unexpectedSuccesses), 'fixture_ids': ids,
               'source_pins_sha256': _sha(pins_raw), 'log_sha256': _sha(log.read_bytes()), 'started_unix': started,
               'finished_unix': time.time(), 'cpu_user_seconds': cpu1.user - cpu0.user,
               'cpu_system_seconds': cpu1.system - cpu0.system, 'python': platform.python_version(),
               'numpy': numpy.__version__, 'torch': torch.__version__, 'host': platform.node(), 'base_root': str(base),
               'base_manifest_sha256': _sha(base_raw), 'base_commit': base_manifest['commit'],
               'base_subtree_tree_id': base_manifest['subtree_tree_id'], 'c10_r4_root': str(r4),
               'c10_r4_source_pins': pins['c10_r4_source_pins'], 'r6_preregistration_sha256': _sha(r6_raw),
               'import_closure': dict(sorted(closure.items())),
               'scope': 'CPU fixtures only on a synthetic world; no live optimizer, planner, outbox, S3, GPU or GP.'}
    with args.receipt.open('x') as stream:
        stream.write(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(args.receipt, 0o444)
    os.chmod(log, 0o444)
    print(json.dumps({k: receipt[k] for k in ('package', 'passed', 'collected', 'executed', 'failures', 'errors')}))
    if not passed:
        raise SystemExit(1)
