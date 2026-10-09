"""Portable isolated, pin-bound runner for this source-only fixture family.

Use python3 -B run_offline.py --receipt /absolute/new/receipt.json. The receipt
is create-only. Expected failures, unexpected successes and skips are failures.
SOURCE_PINS.json is a separately packet-pinned closed manifest, not a signature.
"""
import argparse
import datetime
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

PREFIX = 'generated_target_admission_v1'
EXPECTED_FIXTURE_COUNT = 28
EXPECTED_FIXTURE_IDS = (
    'generated_target_admission_v1.tests.test_export.ExportTests.test_artifact_budget_exhaustion_and_deadline_refuse_partial_manifest',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_bare_verifier_success_and_wrong_identity_refused',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_callback_source_and_dependency_drift',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_canonical_alias_dedup_after_selection_cap',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_complete_pending_refusal_validation_census',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_conflicting_alias_ref_refused_even_after_cap',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_duplicate_weight_in_predecessor_refused',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_loaded_callback_code_must_match_current_pinned_file',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_missing_version_and_missing_object_refused',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_new_unique_limit_retains_complete_pending_denominator',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_no_sql_network_subprocess_or_source_mutation_during_export',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_normalizer_roster_drift_refused',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_old_ref_retained_and_existing_alias_not_double_weighted',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_old_reference_removal_reordering_and_replacement_rejected',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_parity_rejection_no_target_manifest',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_snapshot_event_and_materialized_ack_tampering',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_snapshot_no_mutation_and_exact_head',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_snapshot_sidecars_hash_config_and_row_bounds',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_stale_role_proof_wrong_objective_and_drift_refused',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_stale_version_and_corrupt_bytes_refused',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_success_all_seven_and_no_numerical_or_admission_claim',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_valid_but_unapproved_kumo_weight_refused',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_validation_alias_cannot_claim_training_role',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_wrong_objective_and_normalizer_refused',
    'generated_target_admission_v1.tests.test_export.ExportTests.test_wrong_wrapper_context_program_and_protocol_refused',
    'generated_target_admission_v1.tests.test_runner.RunnerTests.test_expected_and_unexpected_failures_skips_and_counts_are_not_passes',
    'generated_target_admission_v1.tests.test_runner.RunnerTests.test_missing_extra_duplicate_replaced_or_reordered_fixtures_refused',
    'generated_target_admission_v1.tests.test_runner.RunnerTests.test_source_pin_and_import_origin_tampering_refused',
)
EXPECTED_IMPORTS = {
    PREFIX: '__init__.py',
    PREFIX + '.common': 'common.py',
    PREFIX + '.cutoff': 'cutoff.py',
    PREFIX + '.export_verified': 'export_verified.py',
    PREFIX + '.transition': 'transition.py',
    PREFIX + '.run_offline': 'run_offline.py',
    PREFIX + '.tests': 'tests/__init__.py',
    PREFIX + '.tests.fixture_callbacks': 'tests/fixture_callbacks.py',
    PREFIX + '.tests.reference_outbox': 'tests/reference_outbox.py',
    PREFIX + '.tests.test_export': 'tests/test_export.py',
    PREFIX + '.tests.test_runner': 'tests/test_runner.py',
}


def _require(ok, why):
    if not ok:
        raise RuntimeError(why)


def _source_pins(root):
    root = Path(root)
    _require(root.is_absolute() and all(not p.is_symlink() for p in (root, *root.parents)), 'absolute non-symlink source root')
    paths = sorted(root.rglob('*.py'))
    _require({str(p.relative_to(root)) for p in paths} == set(EXPECTED_IMPORTS.values()), 'exact Python source roster')
    _require(not list(root.rglob('*.pyc')), 'no precompiled fixture imports')
    result = {}
    for path in paths:
        _require(path.is_file() and not path.is_symlink() and all(not p.is_symlink() for p in path.parents), 'regular owned source file')
        raw = path.read_bytes()
        _require(len(raw) <= 2 * 1024**2, 'source file byte bound')
        result[str(path.relative_to(root))] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return result


def _pinned_sources(root):
    manifest_path = root / 'SOURCE_PINS.json'
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), 'regular source-pin manifest')
    raw = manifest_path.read_bytes()
    _require(len(raw) <= 65536, 'source-pin manifest bound')
    manifest = json.loads(raw)
    _require(set(manifest) == {'schema', 'fixture_ids', 'fixture_count', 'source_pins', 'import_origins'} and
             manifest['schema'] == 'jope.generated-target-export-offline-pins/v2' and
             manifest['fixture_count'] == EXPECTED_FIXTURE_COUNT and
             manifest['fixture_ids'] == list(EXPECTED_FIXTURE_IDS) and manifest['import_origins'] == EXPECTED_IMPORTS,
             'closed frozen source/import/fixture manifest')
    pins = _source_pins(root)
    _require(pins == manifest['source_pins'], 'pinned source bytes changed')
    return pins, hashlib.sha256(raw).hexdigest()


def _check_fixture_roster(ids):
    _require(len(ids) == EXPECTED_FIXTURE_COUNT and len(set(ids)) == EXPECTED_FIXTURE_COUNT and
             tuple(ids) == EXPECTED_FIXTURE_IDS, 'exact frozen fixture roster/count required')


def _check_import_origins(root, modules=None):
    modules = sys.modules if modules is None else modules
    scoped = {name: mod for name, mod in modules.items() if name == PREFIX or name.startswith(PREFIX + '.')}
    _require(set(scoped) == set(EXPECTED_IMPORTS), 'exact isolated import module roster')
    observed = {}
    for name, mod in scoped.items():
        expected = root / EXPECTED_IMPORTS[name]
        origin = getattr(getattr(mod, '__spec__', None), 'origin', None)
        filename = getattr(mod, '__file__', None)
        _require(filename == str(expected) and origin == str(expected) and not expected.is_symlink(),
                 'pinned import origin mismatch: ' + name)
        observed[name] = EXPECTED_IMPORTS[name]
    return observed


def _result_is_clean(result):
    return (result.testsRun == EXPECTED_FIXTURE_COUNT and result.wasSuccessful() and
            not result.failures and not result.errors and not result.skipped and
            not result.expectedFailures and not result.unexpectedSuccesses)


def run():
    root = Path(__file__).resolve().parent
    before, manifest_sha = _pinned_sources(root)  # Before importing any fixture code.
    old_path, old_write = sys.path[:], sys.dont_write_bytecode
    saved = {k: v for k, v in sys.modules.items() if k == PREFIX or k.startswith(PREFIX + '.')}
    for key in saved:
        del sys.modules[key]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root.parent))
    output = io.StringIO()
    started = time.monotonic()
    try:
        modules = [importlib.import_module(PREFIX + '.tests.' + name) for name in ('test_export', 'test_runner')]
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
        def identities(node):
            if isinstance(node, unittest.TestSuite):
                return [x for child in node for x in identities(child)]
            return [node.id()]
        ids = identities(suite)
        _check_fixture_roster(ids)
        before_imports = _check_import_origins(root)
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
        after, final_manifest_sha = _pinned_sources(root)
        imports = _check_import_origins(root)
        return {'schema': 'jope.generated-target-export-offline-qualification/v2',
                'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'passed': _result_is_clean(result) and before == after and manifest_sha == final_manifest_sha and imports == before_imports,
                'tests_run': result.testsRun, 'fixture_ids': ids, 'skipped': result.skipped,
                'expected_failures': [x.id() for x, _ in result.expectedFailures],
                'unexpected_successes': [x.id() for x in result.unexpectedSuccesses],
                'failures': [x.id() for x, _ in result.failures], 'errors': [x.id() for x, _ in result.errors],
                'wall_seconds': time.monotonic() - started, 'python': sys.version,
                'source_pin_manifest_sha256': manifest_sha, 'source_pins': before,
                'source_unchanged': before == after, 'imports': imports,
                'test_output': output.getvalue(),
                'scope': 'Corrected source-only exporter family; synthetic callbacks. No production callback/native evaluator qualification.',
                'remote_actions': 0, 'gp_executions': 0, 'live_database_access': 0, 'admission_commits': 0}
    finally:
        for key in list(sys.modules):
            if key == PREFIX or key.startswith(PREFIX + '.'):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    receipt = run()
    raw = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if args.receipt:
        p = args.receipt
        if not p.is_absolute() or any(x.is_symlink() for x in (p, *p.parents)):
            raise ValueError('absolute non-symlink create-only receipt path required')
        with p.open('xb') as f:
            f.write(raw)
    print(receipt['test_output'], end='')
    print(json.dumps({k: receipt[k] for k in ('passed', 'tests_run', 'wall_seconds', 'source_unchanged')}))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
