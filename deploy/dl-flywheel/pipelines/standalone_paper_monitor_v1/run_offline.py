#!/usr/bin/env python3
"""Portable CI aggregation; qualification evidence is outside immutable source."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from offline_adapter import FAMILIES, prepare_output, require, verify_source, write_new

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(__debug__, 'non-optimized synthetic qualification')
    output = prepare_output(args.output, ROOT, 'aggregate')
    start, mono = time.time_ns(), time.monotonic_ns()
    write_new(output / 'intent.json', {'schema': 'jope.standalone-paper-monitor-offline-aggregate-intent/v1',
              'started_utc_ns': start, 'scope': 'isolated synthetic fixtures only; one dispatch per family'})
    receipt = {'schema': 'jope.standalone-paper-monitor-offline-aggregate/v1', 'passed': False,
               'collected': 0, 'executed': 0, 'identities': [], 'families': {},
               'actual_runtime_qualified': False, 'runtime_or_remote_execution': False}
    exitcode = 1
    try:
        receipt['source_before'] = verify_source(ROOT)
        roster = json.loads((ROOT / 'FIXTURE_ROSTER.json').read_bytes())['families']
        for family in FAMILIES:
            child = subprocess.run([sys.executable, '-B', str(ROOT / 'run_family.py'),
                                    '--family', family, '--output', str(output / family)],
                                   check=False, timeout=180)
            path = output / family / 'receipt.json'
            require(path.is_file() and not path.is_symlink(), 'durable family receipt required; no retry')
            result = json.loads(path.read_bytes())
            receipt['families'][family] = result
            require(child.returncode == 0 and result['passed'] is True and result['family'] == family,
                    'family success and exact identity')
            require(result['collected'] == result['executed'] == len(roster[family]) and
                    sorted(result['identities']) == roster[family], 'exact family roster and count')
            receipt['collected'] += result['collected']
            receipt['executed'] += result['executed']
            receipt['identities'].extend(result['identities'])
        expected = sorted(identity for family in FAMILIES for identity in roster[family])
        require(len(receipt['identities']) == len(set(receipt['identities'])) and
                sorted(receipt['identities']) == expected and receipt['executed'] == len(expected),
                'complete unique aggregate roster')
        receipt['source_after'] = verify_source(ROOT)
        require(receipt['source_before'] == receipt['source_after'], 'source unchanged')
        receipt.update(passed=True, state='passed_synthetic_only')
        exitcode = 0
    except BaseException as error:
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)[:4096]}
    receipt.update(ended_utc_ns=time.time_ns(), elapsed_monotonic_ns=time.monotonic_ns() - mono)
    write_new(output / 'receipt.json', receipt)
    print(json.dumps({'passed': receipt['passed'], 'collected': receipt['collected'], 'executed': receipt['executed']}))
    raise SystemExit(exitcode)


if __name__ == '__main__':
    main()
