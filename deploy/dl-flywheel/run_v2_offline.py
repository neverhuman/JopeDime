"""Isolated, exact-roster CPU fixtures for the v2.0.0 learner packages; additive to the prior812 identities.

Families (168 identities): continuous_v6 40, boundary_admission_v2 24, generated_target_preparation_v2 36,
native_result_overlay_v1 36 and final_capture_v1 32.  Each family runs in its own fresh interpreter because the
fixtures replace module globals and clocks.  final_capture_v1 runs through its own pinned runner, which supports the
integrated tree.  The other four run through the child mode below: the boundary/preparation/overlay runners shipped
inside those packages bind the lane's external base export manifest and an older candidate root, which are not part
of this repository, so this wrapper applies the same checks to the integrated tree instead.  Before import it
verifies each package's pinned SOURCE_PINS.json (or the continuous_v6 pins below) and the closed package roster; it
runs exactly the registered identities with the successor search path; afterwards every imported module under
deploy/dl-flywheel must be byte-pinned by SOURCE_MANIFEST.json, and every imported module inside a v2 package must
also be pinned by that package's own pins.  Skips, expected failures, unexpected successes or a changed roster fail.
CPU only: CUDA is hidden; no network, S3, GPU, live campaign, outbox, planner or program-search execution.  The
frozen packages, their original runners and the prior812 runners are unchanged.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parent
PIPELINES = ROOT / 'pipelines'
MANIFEST = ROOT / 'SOURCE_MANIFEST.json'
R6_PREREGISTRATION = ROOT / 'tests/fixtures/r6-dev-split-preregistration-v1.json'
R6_PREREGISTRATION_SHA256 = 'd228d41366e5e313da63fc64ea3c76bcbfebe6899c4e0bb015f9da22d741725a'

# Execution order follows the import dependencies (continuous_v6 imports the other four packages).
SOURCE_PINS_SHA256 = {
    'boundary_admission_v2': '25d4910ffddbd1c66cbaa6e01dd2d39181b106da0bb2caffaadc1d774ab003fc',
    'generated_target_preparation_v2': 'e13111f7166e55036364b42b74132c972c1dc5f1deb54d21b1f2ce1384d01f5a',
    'native_result_overlay_v1': '3b43721a17cb6d41486012f4b311b20be78a6d337fa5676fee1433cd08fea31d',
    'final_capture_v1': '7b5f44511c66d6d4c3d5bd54f547a76c0c9d59c87f82b69f74feebd861af9ca4',
}
CONTINUOUS_V6_PINS = {
    '__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'cli_v6.py': '41f4b2fc38ac1ac4658c1d30f25521b97cd5811d43dd7796e55617d72aab9c7d',
    'custody_v6.py': '6d2ea58d4e0fc8eb1c6b855a711a6f3324a5448ab759d0b321a429ae812d356e',
    'genesis.py': '6cd7a44566c692e656e740d547c2f01137718cb002174efa93463940a1d1a04e',
    'parity_probe.py': 'a4a20a03298ba4a2f4eb06e746c023b5b9471ff9d4e4d915a2e222919d6af142',
    'resume_v6.py': '9029f2f44a9136945a4c1f3007515640f26e1b7a8bca03f17655e0158bc94181',
    'runner_v6.py': 'db87374f1a2222c4d43a926b89a756969dc5fe927ba0d80a0e7f89d6f58c0b0a',
    'tests/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'tests/test_r2_custody.py': '6eedd89a39074b9108663a894f2fac634f3d9c053f0821e94fa437644a297ab0',
    'tests/test_runner_v6.py': 'ca070ad813ad2a702df585244b4fd5506290e67a916636f7c580d1a765438b42',
}
CONTINUOUS_V6_IDS = tuple('pipelines.continuous_v6.tests.test_runner_v6.RunnerV6Tests.' + name for name in (
    'test_genesis_inherits_lineage_indexes_and_logs_r14_after_parity',
    'test_memory_gate_waits_then_exits_cleanly_without_reservation',
    'test_own_row_cap_waits_without_reservation',
    'test_real_empty_inbox_boundary_reports_idle',
    'test_restart_continues_bitwise_from_own_state',
    'test_restart_refuses_failure_or_mismatched_state',
    'test_runner_refuses_planner_without_bounded_verify_index',
    'test_runner_v6_trains_from_genesis_with_origin_proofs_and_bounded_closure',
    'test_tick_safe_stop_never_reserves_inside_the_budget',
)) + tuple('pipelines.continuous_v6.tests.test_r2_custody.R2CustodyTests.' + name for name in (
    'test_f1_probe_root_keeps_v1_control_and_has_no_r14',
    'test_f1_r14_requires_exact_cuda_next_update_parity',
    'test_f2_clean_closure_resumes_and_binds_every_file',
    'test_f2_failed_closure_writes_failure_and_no_clean_result',
    'test_f2_fatal_and_markers_refuse_and_runner_rechecks_basis',
    'test_f2_fully_rebound_unpassed_or_unclean_closure_refuses',
    'test_f2_missing_or_interrupted_verification_refuses',
    'test_f2_rebound_or_failed_verification_refuses',
    'test_f2_stale_closure_of_an_earlier_segment_refuses',
    'test_f3_ack_without_commit_refuses_closure',
    'test_f3_changed_earlier_proof_after_watermark_refuses',
    'test_f3_closure_rehashes_a_proof_changed_after_its_epoch_watermark',
    'test_f3_missing_or_tampered_proof_refuses',
    'test_f3_stale_control_or_checkpoint_refuses',
    'test_f3_symlinked_proof_refuses_even_with_identical_bytes',
    'test_f3_truncated_suffix_orphan_and_prefix_tamper_refuse',
    'test_f3_wrong_ack_and_old_outbox_refuse',
    'test_f4_cli_joins_capture_and_ancestry_to_genesis',
    'test_f4_runner_refuses_ancestry_not_stored_by_the_planner',
    'test_f5_genesis_records_v1_epoch_and_fresh_schedule',
    'test_g32_deferred_epoch_checked_once_then_adopts',
    'test_g32_deferred_epoch_is_checked_before_adoption_and_trips',
    'test_g32_deferred_epoch_with_too_little_time_defers_again_without_reservation',
    'test_g32_guard_state_is_sealed_in_the_closure',
    'test_g32_guard_update_transitions',
    'test_g32_learning_guard_holds_adoptions_and_is_sticky',
    'test_g32_runner_refuses_malformed_policy_and_genesis_sidecar',
)) + tuple('pipelines.continuous_v6.tests.test_r2_custody.R2ClockAndOwnerTests.' + name for name in (
    'test_f6_gpu_exclusive_and_pinned',
    'test_f6_stopped_owners_must_be_closed',
    'test_f6_unit_membership_and_inclusive_allocation',
    'test_f6_window_uses_boot_clock_and_fails_closed',
))
EXPECTED_COUNTS = {'boundary_admission_v2': 24, 'generated_target_preparation_v2': 36,
                   'native_result_overlay_v1': 36, 'final_capture_v1': 32, 'continuous_v6': 40}
OWN_RUNNER = {'final_capture_v1'}
TOTAL = 168
CHILD_SCHEMA = 'jope.v2-learner-offline-child/v1'
RECEIPT_SCHEMA = 'jope.v2-learner-offline/v1'
SCOPE = ('Offline CPU fixtures on private temporary worlds only. No live campaign, optimizer launch, planner, '
         'outbox, S3, GPU, network or program-search execution, and no launch permission.')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def manifest_pins():
    pins = {}
    for item in json.loads(MANIFEST.read_bytes())['files']:
        need(item['path'] not in pins, 'duplicate SOURCE_MANIFEST path: ' + item['path'])
        pins[item['path']] = item['sha256']
    return pins


def package_pins(name):
    """Return ({relative .py path: sha256}, fixture ids, pins file sha256) for one v2 package."""
    if name == 'continuous_v6':
        return dict(CONTINUOUS_V6_PINS), list(CONTINUOUS_V6_IDS), None
    path = PIPELINES / name / 'SOURCE_PINS.json'
    need(path.is_file() and not path.is_symlink(), 'regular SOURCE_PINS.json: ' + name)
    raw = path.read_bytes()
    need(digest(raw) == SOURCE_PINS_SHA256[name], 'pinned SOURCE_PINS.json: ' + name)
    pins = json.loads(raw)
    need(pins['package'] == name, 'SOURCE_PINS package name: ' + name)
    return {k: v['sha256'] for k, v in pins['source_pins'].items()}, list(pins['fixture_ids']), digest(raw)


def closed_package(name, repo_pins):
    """The package's Python roster equals its pins, nothing is precompiled, and every file is manifest-pinned."""
    root = PIPELINES / name
    pins, ids, pins_sha = package_pins(name)
    need(not any(root.rglob('*.pyc')) and not any(p.name == '__pycache__' for p in root.rglob('*')),
         'preexisting bytecode refused: ' + name)
    actual = {}
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink(), 'symlink refused: ' + str(path))
        if not path.is_file():
            continue
        raw = path.read_bytes()
        rel = str(path.relative_to(ROOT.parents[1]))
        need(repo_pins.get(rel) == digest(raw), 'SOURCE_MANIFEST pin: ' + rel)
        if path.suffix == '.py':
            actual[str(path.relative_to(root))] = digest(raw)
    need(actual == pins, 'closed pinned Python roster: ' + name)
    need(len(ids) == len(set(ids)) == EXPECTED_COUNTS[name], 'exact fixture roster: ' + name)
    return pins, ids, pins_sha


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def child(name, output):
    need(sys.flags.optimize == 0, 'fixture assertions must remain enabled')
    need(name not in OWN_RUNNER, 'family runs through its own pinned runner')
    need(not any(k == 'pipelines' or k.startswith('pipelines.') for k in sys.modules), 'fresh child namespace')
    repo_pins = manifest_pins()
    _, ids, pins_sha = closed_package(name, repo_pins)
    v2_pins = {}
    for other in EXPECTED_COUNTS:
        pins = closed_package(other, repo_pins)[0]
        v2_pins.update({str((PIPELINES / other / rel).resolve()): sha for rel, sha in pins.items()})
    env = {}
    if name == 'native_result_overlay_v1':
        need(not R6_PREREGISTRATION.is_symlink() and digest(R6_PREREGISTRATION.read_bytes()) == R6_PREREGISTRATION_SHA256,
             'pinned R6 preregistration fixture input')
        os.environ['JOPE_W3_R6_PATH'] = env['JOPE_W3_R6_PATH'] = str(R6_PREREGISTRATION)
    sys.path[:0] = [str(PIPELINES / 'prepared'), str(PIPELINES / 'continuous'), str(PIPELINES / 'continuous/consumer'),
                    str(PIPELINES / 'continuous/runtime'), str(ROOT), str(ROOT / 'tests'), str(ROOT / 'tests_native'),
                    str(PIPELINES / 'hpc_verified_v5')]
    suite = unittest.defaultTestLoader.loadTestsFromNames(sorted({i.rsplit('.', 1)[0] for i in ids}))
    collected = sorted(t.id() for t in flatten(suite))
    need(collected == sorted(ids), 'exact collected fixture identities: ' + name)
    started, mono, cpu0 = time.time_ns(), time.monotonic_ns(), os.times()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed, cpu1 = time.monotonic_ns() - mono, os.times()
    passed = (result.wasSuccessful() and result.testsRun == len(ids) and not result.skipped and
              not result.expectedFailures and not result.unexpectedSuccesses)
    closure = {}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if not path or not path.endswith('.py'):
            continue
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT):
            continue
        raw = path.read_bytes()
        rel = str(path.relative_to(ROOT.parents[1]))
        need(repo_pins.get(rel) == digest(raw), 'imported module not pinned by SOURCE_MANIFEST: ' + rel)
        if any(path.is_relative_to(PIPELINES / other) for other in EXPECTED_COUNTS):
            need(v2_pins.get(str(path)) == digest(raw), 'imported v2 module not pinned by its package: ' + rel)
        closure[rel] = digest(raw)
    import numpy
    import torch
    log = stream.getvalue().encode()
    (output / (name + '.log')).write_bytes(log)
    receipt = {'schema': CHILD_SCHEMA, 'family': name, 'runner': 'run_v2_offline.py --child',
               'collected': len(collected), 'executed': result.testsRun, 'fixture_ids': collected,
               'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
               'expected_failures': len(result.expectedFailures),
               'unexpected_successes': len(result.unexpectedSuccesses), 'passed': passed,
               'source_pins_sha256': pins_sha, 'environment': env, 'import_closure': dict(sorted(closure.items())),
               'log': {'bytes': len(log), 'sha256': digest(log)}, 'started_utc_ns': started,
               'ended_utc_ns': time.time_ns(), 'monotonic_elapsed_ns': elapsed,
               'cpu_user_seconds': cpu1.user - cpu0.user, 'cpu_system_seconds': cpu1.system - cpu0.system,
               'python': platform.python_version(), 'numpy': numpy.__version__, 'torch': str(torch.__version__),
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'live_qualification': False,
               'scope': SCOPE}
    with (output / (name + '.json')).open('x') as f:
        json.dump(receipt, f, sort_keys=True, indent=2)
        f.write('\n')
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--child', choices=sorted(set(EXPECTED_COUNTS) - OWN_RUNNER))
    args = ap.parse_args()
    need(sys.flags.optimize == 0, 'fixture assertions must remain enabled')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'
    for key in ('JOPE_W10_BASE_ROOT', 'JOPE_C10_BASE_ROOT', 'JOPE_C10_BASE_MANIFEST_SHA256',
                'JOPE_W3_C10_R4_ROOT', 'JOPE_W3_R6_PATH'):
        os.environ.pop(key, None)
    output = args.output.resolve()
    if args.child:
        need(output.is_dir(), 'parent-created output directory')
        return child(args.child, output)
    repo_pins = manifest_pins()
    rosters = {name: closed_package(name, repo_pins)[1] for name in EXPECTED_COUNTS}
    output.mkdir(parents=True, exist_ok=False)
    started, mono = time.time_ns(), time.monotonic_ns()
    children, identities, passed = {}, [], True
    for name, count in EXPECTED_COUNTS.items():
        if name in OWN_RUNNER:
            command = [sys.executable, '-B', str(PIPELINES / name / 'run_offline.py'),
                       '--receipt', str(output / (name + '.json'))]
        else:
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--output', str(output), '--child', name]
        family_mono = time.monotonic_ns()
        process = subprocess.run(command, check=False)
        receipt = json.loads((output / (name + '.json')).read_bytes())
        receipt['family_wall_monotonic_ns'] = time.monotonic_ns() - family_mono
        ids = sorted(receipt['fixture_ids'])
        need(ids == sorted(rosters[name]) and receipt['collected'] == count, 'child receipt roster mismatch: ' + name)
        ok = (process.returncode == 0 and receipt['passed'] is True and receipt['executed'] == count and
              receipt['failures'] == receipt['errors'] == receipt['skipped'] == 0)
        passed = passed and ok
        children[name] = receipt
        identities.extend(ids)
    need(len(identities) == len(set(identities)) == TOTAL, 'combined exact unique identities')
    final = {'schema': RECEIPT_SCHEMA, 'source_sha': os.environ.get('JOPE_SOURCE_SHA'), 'collected': TOTAL,
             'executed': sum(x['executed'] for x in children.values()), 'identities': identities, 'passed': passed,
             'family_counts': dict(EXPECTED_COUNTS), 'children': children, 'prior812_executed_by_this_runner': 0,
             'r6_preregistration_sha256': R6_PREREGISTRATION_SHA256, 'started_utc_ns': started,
             'ended_utc_ns': time.time_ns(), 'monotonic_elapsed_ns': time.monotonic_ns() - mono,
             'live_qualification': False, 'optimizer_admission': False, 'scope': SCOPE}
    with (output / 'receipt.json').open('x') as f:
        json.dump(final, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in final.items() if k not in ('children', 'identities')}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
