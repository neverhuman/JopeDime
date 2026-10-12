"""Pinned offline runner for final_capture_v1 (W10 r2).

Before importing any package code it checks the exact package Python roster and bytes, the pinned fixture inputs
(the byte copy of the sealed R17 owner receipt) and the pinned base modules against SOURCE_PINS.json. It then runs
exactly the registered fixture identities (r1's 14 + r2's 18 runtime_limit_stopped_idle) in this process with the
successor sys.path and writes a create-only receipt and log. Skips, expected failures, unexpected successes, a
changed roster or a changed count are failures. ``--measure`` writes the capacity-measurement receipt instead,
after the same pin checks. CPU only; no network, S3, GPU, GP execution or live journal access.
The base root is JOPE_W10_BASE_ROOT (a lane mirror of the pinned base) or, in an integrated tree, deploy/dl-flywheel.
"""
import argparse, hashlib, json, os, platform, sys, time, unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DF = PACKAGE.parents[1]
PINS_SCHEMA = 'jope.w10-offline-source-pins/v1'
RECEIPT_SCHEMA = 'jope.w10-offline-qualification/v1'
T = 'pipelines.final_capture_v1.tests.test_final_capture.FinalCaptureTests.'
EXPECTED_FIXTURE_IDS = tuple(T + name for name in (
    'test_a_missing_result_is_captured_and_fresh_journal_inherits',
    'test_a_present_result_must_join_exactly',
    'test_b_checkpoint_bytes_tamper_refused',
    'test_b_failed_or_uncertain_terminal_state_refused',
    'test_b_owner_active_or_cgroup_nonempty_refused',
    'test_b_parameter_sha_disagreement_refused',
    'test_b_pending_reservation_or_orphan_state_is_a_hold',
    'test_b_proof_or_ack_evidence_mismatch_refused',
    'test_c_capture_bytes_are_deterministic',
    'test_d_next_update_parity_on_isolated_scratch',
    'test_e_private_copy_only_pinned_runtime_and_create_only_output',
    'test_f_tensor_key_classes_synthetic_headers',
    'test_g_chunked_parts_boundaries_tamper_missing_and_reorder',
    'test_h_capacity_at_v1_and_v2_scale')) + tuple(
    'pipelines.final_capture_v1.tests.test_runtime_limit.RuntimeLimitTests.' + name for name in (
        'test_r_live_shape_runtime_limit_stopped_idle_is_captured',
        'test_r_a_timeout_with_unresolved_reservation_is_a_hold',
        'test_r_b_write_after_in_loop_deadline_refused',
        'test_r_b_in_loop_deadline_comes_from_window_and_config',
        'test_r_b_in_loop_deadline_must_have_passed',
        'test_r_c_owner_cgroup_or_pid_refused',
        'test_r_c_restarted_unit_refused',
        'test_r_d_systemd_result_other_than_timeout_refused',
        'test_r_e_running_failed_or_unknown_state_refused',
        'test_r_f_last_commit_progress_and_rolling_must_agree',
        'test_r_g_missing_or_forged_r17_receipt_refused',
        'test_r_g_clean_class_refuses_r17_inputs',
        'test_r_h_symlink_or_special_file_in_root_refused',
        'test_r_h_walk_entry_and_byte_bounds_refused',
        'test_r_i_live_walk_must_equal_private_copy_walk',
        'test_r_i_live_walk_changed_during_capture_refused',
        'test_r_j_status_or_result_json_present_refused',
        'test_r_k_walk_parts_reader_refuses_tamper'))
R = 'pipelines.final_capture_v1.tests.test_runtime_limit.RuntimeLimitTests.'
FIXTURE_INPUTS = ('tests/owner-askuserquestion-receipt-R17-v1.json',)


def require(ok, why):
    if not ok: raise RuntimeError(why)


def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_pins(root):
    require(not list(root.rglob('*.pyc')), 'no precompiled package files')
    out = {}
    for path in sorted(p for p in root.rglob('*.py') if '__pycache__' not in p.parts):
        require(path.is_file() and not path.is_symlink(), 'regular source file: ' + str(path))
        raw = path.read_bytes(); out[str(path.relative_to(root))] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return out


def base_root(): return Path(os.environ.get('JOPE_W10_BASE_ROOT', DF)).resolve()


def search_path(base, df):
    return [str(base / 'pipelines/prepared'), str(base / 'pipelines/continuous'), str(base / 'pipelines/continuous/consumer'),
            str(base / 'pipelines/continuous/runtime'), str(base), str(df), str(base / 'tests'), str(base / 'tests_native')]


def checked_pins(package=PACKAGE):
    raw = (PACKAGE / 'SOURCE_PINS.json').read_bytes(); pins = json.loads(raw); base = base_root()
    require(pins['schema'] == PINS_SCHEMA and pins['package'] == 'final_capture_v1' and pins['fixture_ids'] == list(EXPECTED_FIXTURE_IDS),
            'closed pin manifest and fixture roster')
    require(source_pins(package) == pins['source_pins'], 'pinned package source bytes changed')
    require(sorted(pins['fixture_inputs']) == sorted(FIXTURE_INPUTS) and
            all(not (package / rel).is_symlink() and file_sha(package / rel) == h for rel, h in pins['fixture_inputs'].items()),
            'pinned fixture inputs changed')
    require({rel: file_sha(base / rel) for rel in pins['base_pins']} == pins['base_pins'], 'pinned base modules changed')
    return pins, raw, base


def create_only(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, 'w') as f: f.write(text); f.flush(); os.fsync(f.fileno())


def ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite): yield from ids(item)
        else: yield item.id()


def main(argv=None):
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--receipt', type=Path); group.add_argument('--measure', type=Path); args = parser.parse_args(argv)
    pins, pins_raw, base = checked_pins()
    sys.dont_write_bytecode = True; sys.path[:0] = search_path(base, DF); os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.measure:
        from pipelines.final_capture_v1 import measure
        return measure.main(['--receipt', str(args.measure)])
    started, cpu0 = time.time(), os.times()
    suite = unittest.defaultTestLoader.loadTestsFromNames(sorted({i.rsplit('.', 1)[0] for i in EXPECTED_FIXTURE_IDS}))
    found = sorted(ids(suite)); require(found == sorted(EXPECTED_FIXTURE_IDS), 'exact fixture identities')
    log = args.receipt.with_suffix('.log'); fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite); stream.flush(); os.fsync(stream.fileno())
    os.chmod(log, 0o444); cpu1 = os.times()
    import numpy, safetensors, torch
    passed = (result.wasSuccessful() and result.testsRun == len(EXPECTED_FIXTURE_IDS) and not result.skipped and
              not result.expectedFailures and not result.unexpectedSuccesses)
    receipt = {'schema': RECEIPT_SCHEMA, 'package': 'final_capture_v1', 'revision': 'r2', 'passed': passed, 'collected': len(found),
               'executed': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'expected_failures': len(result.expectedFailures),
               'unexpected_successes': len(result.unexpectedSuccesses), 'fixture_ids': found,
               'source_pins_sha256': hashlib.sha256(pins_raw).hexdigest(), 'log_sha256': file_sha(log),
               'started_unix': started, 'finished_unix': time.time(), 'cpu_user_seconds': cpu1.user - cpu0.user,
               'cpu_system_seconds': cpu1.system - cpu0.system, 'host': platform.node(), 'python': platform.python_version(),
               'numpy': numpy.__version__, 'torch': str(torch.__version__), 'safetensors': safetensors.__version__,
               'base_root': str(base), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
               'scope': 'CPU fixtures on private temporary worlds only; no live optimizer, planner, outbox, S3, GPU or GP execution.'}
    create_only(args.receipt, json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: receipt[k] for k in ('package', 'passed', 'collected', 'executed', 'failures', 'errors', 'skipped')}))
    if not passed: raise SystemExit(1)


if __name__ == '__main__':
    main()
