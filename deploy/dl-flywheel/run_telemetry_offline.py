"""Seal the new21 CPU telemetry custody fixtures; no live/network execution."""
import argparse
import hashlib
import json
import os
import sys
import time
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent


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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['JOPE_TELEMETRY_FIXTURE_ROOT'] = str(args.output.resolve())
    clock = shift_fixture_wall_clock()
    sys.path.insert(0, str(ROOT/'pipelines/telemetry'))
    suite = unittest.defaultTestLoader.loadTestsFromName('test_trace_archive.Tests')
    def ids(s):
        for x in s:
            if isinstance(x, unittest.TestSuite): yield from ids(x)
            else: yield x.id()
    identities = sorted(ids(suite))
    if len(identities) != len(set(identities)) or len(identities) != 21:
        raise ValueError('exact21 telemetry fixture roster')
    started = time.time_ns()
    with (args.output/'tests.log').open('x') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    paths = ['pipelines/telemetry/trace_archive.py', 'pipelines/telemetry/test_trace_archive.py',
             'run_telemetry_offline.py']
    receipt = {'schema': 'jope.trace-custody-offline/v1', 'started_utc_ns': started, **clock,
               'finished_utc_ns': time.time_ns(), 'collected': len(identities), 'executed': result.testsRun,
               'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
               'identities': identities, 'sources': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
               'gp_executions': 0, 's3_requests': 0, 'optimizer_updates': 0,
               'passed': result.wasSuccessful() and not result.skipped and result.testsRun == 21}
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if not result.wasSuccessful() or result.skipped: raise SystemExit(1)
    print('21 telemetry custody fixtures PASS/0skip; no live writes or execution')


if __name__ == '__main__': main()
