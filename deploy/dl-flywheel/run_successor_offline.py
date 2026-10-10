"""Qualify prepared admission and transfer in isolated CPU-only test shards."""
import argparse, hashlib, json, os, subprocess, sys, time, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = {
    'transfer': ['test_resume_transfer.TransferTests',
                 'test_resume_transfer.OpenEpochTests.test_open_epoch_transfer_defers_admission_and_keeps_bit_exact_sequence',
                 'test_resume_transfer.OpenEpochTests.test_open_capture_stale_progress_and_unknown_intent_refused',
                 'test_resume_transfer.OpenEpochTests.test_changed_authority_at_boundary_refuses_admission_without_replaying_commit',
                 'test_transfer_watch.WatchTests', 'test_transfer_watch.TransitionTests',
                 'test_transfer_watch.RetirementTests',
                 'test_cold_recovery.ColdTests.test_same_reserved_batch_finishes_once_and_is_bit_exact',
                 'test_cold_recovery.ColdTests.test_ack_and_changed_trace_refuse',
                 'test_cold_recovery.ColdTests.test_later_optimizer_stage_refuses',
                 'test_cold_recovery.ColdTests.test_changed_membership_or_unrelated_pending_refuses',
                 'test_cold_recovery.ColdTests.test_rehashed_changed_membership_cannot_reuse_certified_batch_id',
                 'test_cold_recovery.ColdTests.test_sealed_capture_roundtrip_matches_fresh_cold_custody'],
    'prepared': ['test_prepared_snapshot'],
    'custody': ['test_live_snapshot.LiveSnapshotTests'],
}
COUNTS = {'transfer': 21, 'prepared': 12, 'custody': 18}


# Fixture-only wall clock (PR #283 independent review finding B2).  The frozen campaign modules cap every window at
# the historical global deadline 1791592146.380285 (2026-10-10T00:29:06Z) while these fixtures build their windows
# from time.time(), so the suite started failing as real time crossed that constant.  Inside this fixture process
# only, the wall clock is shifted to a fixed epoch at which the whole suite passed (2026-10-09T23:00:00Z); elapsed
# time, monotonic clocks and file times are untouched and the shift is recorded in the receipt.
FIXTURE_WALL_EPOCH = 1791586800.0


def shift_fixture_wall_clock():
    real_time, real_time_ns = time.time, time.time_ns
    shift_ns = int(round((FIXTURE_WALL_EPOCH - real_time()) * 1e9))
    time.time = lambda: real_time() + shift_ns / 1e9
    time.time_ns = lambda: real_time_ns() + shift_ns
    return {'fixture_wall_epoch': FIXTURE_WALL_EPOCH, 'fixture_wall_shift_ns': shift_ns}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=CASES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.suite is None:
        receipts = {}
        for name in CASES:
            subprocess.run([sys.executable, '-B', __file__, '--suite', name,
                            '--output', str(args.output / name)], check=True)
            receipts[name] = json.loads((args.output / name / 'receipt.json').read_bytes())
        receipt = {'schema': 'jope.successor-offline-tests/v1', 'source_sha': os.environ.get('JOPE_SOURCE_SHA'),
                   'collected': sum(r['collected'] for r in receipts.values()),
                   'executed': sum(r['executed'] for r in receipts.values()),
                   'passed': all(r['passed'] for r in receipts.values()), 'shards': receipts}
    else:
        continuous = ROOT / 'pipelines/continuous'
        pipeline = ROOT / ('pipelines/hpc_verified_v4' if args.suite == 'custody' else 'pipelines/prepared')
        sys.path[:0] = [str(pipeline), str(continuous), str(continuous / 'consumer'),
                       str(continuous / 'runtime'), str(ROOT)]
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        clock = shift_fixture_wall_clock() if args.suite == 'transfer' else {}
        suite = unittest.defaultTestLoader.loadTestsFromNames(CASES[args.suite])
        count = suite.countTestCases()
        if count != COUNTS[args.suite]:
            raise ValueError('exact test roster changed: ' + args.suite)
        started = time.time()
        with (args.output / 'tests.log').open('x') as log:
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        receipt = {'schema': 'jope.successor-offline-shard/v1', 'source_sha': os.environ.get('JOPE_SOURCE_SHA'),
                   'suite': args.suite, 'started': started, 'finished': time.time(), **clock, 'collected': count,
                   'executed': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
                   'skipped': len(result.skipped), 'expected_failures': len(result.expectedFailures),
                   'log_sha256': hashlib.sha256((args.output / 'tests.log').read_bytes()).hexdigest(),
                   'passed': result.wasSuccessful() and result.testsRun == count and not result.skipped and not result.expectedFailures,
                   'scope': 'CPU fixtures only; no live optimizer, S3, process control or GPU execution.'}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'shards'}))
    if not receipt['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
