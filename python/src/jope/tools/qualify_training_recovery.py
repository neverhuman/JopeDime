#!/usr/bin/env python3
"""Compare eight-rank training with injected failure against uninterrupted training."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root, binary, corpus, output = [p.resolve() for p in [args.root, args.binary, args.corpus, args.output]]
    assert not output.exists(), 'Qualification output already exists'
    output.mkdir(parents=True)
    results = []
    begun = time.monotonic()
    for label, failure in [('uninterrupted', None), ('recovery', 6)]:
        run = output / label
        cmd = [str(binary), '--root', str(root), 'train', '--corpus', str(corpus), '--output', str(run),
               '--world', '8', '--microbatch', '256', '--steps', '30', '--checkpoint-steps', '10']
        if failure is not None:
            cmd += ['--fail-rank', str(failure)]
        start = time.monotonic()
        with (output / f'{label}.log').open('x') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        complete = json.loads((run / 'complete.json').read_text())
        checkpoint = run / 'checkpoints/step-30'
        state = json.loads((checkpoint / 'state.json').read_text())
        result = dict(label=label, seconds=time.monotonic()-start, attempts=complete['attempts'],
                      rank_failure_injected=complete['rank_failure_injected'],
                      parameter_sha256=state['parameter_sha256'], sampler_id=state['sampler_id'],
                      checkpoint_state_sha256=digest(checkpoint / 'state.json'),
                      completion_sha256=digest(run / 'complete.json'))
        results.append(result)
        print(json.dumps(result), flush=True)
        (output / 'progress.json').write_text(json.dumps(results, indent=2) + '\n')
    identical = results[0]['parameter_sha256'] == results[1]['parameter_sha256']
    passed = identical and results[1]['rank_failure_injected'] and results[1]['attempts'] > 1
    report = dict(schema='parallel.distributed_recovery.v3', passed=passed, world=8, global_batch=2048,
                  microbatch=256, steps=30, binary_sha256=digest(binary), corpus_sha256=digest(corpus/'index.json'),
                  driver_sha256=digest(Path(__file__)), results=results, parameter_hashes_identical=identical,
                  maximum_absolute_error=0 if identical else None, seconds=time.monotonic()-begun,
                  qualification=True, final_tests_opened=False)
    (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    assert passed, 'Distributed recovery parameters differ'


if __name__ == '__main__':
    main()
