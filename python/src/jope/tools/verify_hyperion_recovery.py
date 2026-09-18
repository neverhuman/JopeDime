#!/usr/bin/env python3
"""Verify saved recovery encodings and their equivalence without reading CSVs."""
import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    old_spec_path = root / 'manifests/hyperion-concurrent-spec.json'
    new_spec_path = root / 'manifests/hyperion-reviewed4-safe-spec.json'
    old_spec, new_spec = read(old_spec_path), read(new_spec_path)
    changed = {k for k in old_spec.keys() | new_spec.keys() if old_spec.get(k) != new_spec.get(k)}
    assert changed == {'persistent_binary', 'persistent_binary_sha256'}, changed
    assert sha(root / new_spec['persistent_binary']) == new_spec['persistent_binary_sha256']
    snapshot_path = root / 'runs/parity/hyperion-reviewed4-stall/saved-hashes.json'
    snapshot = read(snapshot_path)
    for path, expected in snapshot['files'].items():
        assert sha(root / path) == expected, f'Preserved artifact changed: {path}'
    job_path = root / 'manifests/generated/gp-reviewed4-jobs.json'
    jobs = read(job_path)
    prepared_cache = {}
    verified, comparisons = [], []
    new_root = root / 'data/encodings/hyperion-reviewed4-safe'
    old_root = root / 'data/encodings/hyperion-batched-cuts'
    for job in jobs['jobs']:
        prepared_path = root / job['prepared'] / 'prepared.json'
        if job['prepared'] not in prepared_cache:
            assert sha(prepared_path) == job['prepared_sha256']
            prepared_cache[job['prepared']] = read(prepared_path)
        p = prepared_cache[job['prepared']]
        cut = next(c for c in p['cuts'] if c['index'] == job['cut'])
        relative = Path(p['dataset']) / f"cut-{job['cut']}"
        path = new_root / relative / 'hyperion.f32'
        receipt_path = path.with_suffix('.json')
        receipt = read(receipt_path)
        assert receipt['prepared_sha256'] == job['prepared_sha256']
        assert receipt['training_sha256'] == cut['training_sha256']
        assert receipt['preprocessing_sha256'] == cut['preprocessing_sha256']
        assert receipt['spec_sha256'] == sha(new_spec_path)
        assert receipt['signature_sha256'] == sha(path)
        assert receipt['max_forward_cells'] == 24000
        values = struct.unpack('<512f', path.read_bytes())
        assert all(map(math.isfinite, values))
        verified.append({'dataset': p['dataset'], 'cut': job['cut'], 'path': str(path.relative_to(root)),
                         'sha256': sha(path), 'receipt_sha256': sha(receipt_path)})
        old_path = old_root / relative / 'hyperion.f32'
        if str(old_path.relative_to(root)) in snapshot['files']:
            old_receipt = read(old_path.with_suffix('.json'))
            for key in ['prepared_sha256', 'training_sha256', 'preprocessing_sha256']:
                assert old_receipt[key] == receipt[key]
            assert old_receipt['spec_sha256'] == sha(old_spec_path)
            reference = struct.unpack('<512f', old_path.read_bytes())
            error = max(abs(a-b) for a,b in zip(values, reference))
            # Same declared tolerance as the original CPU/CUDA qualification.
            assert error <= 0.0003, (relative, error)
            comparisons.append({'dataset': p['dataset'], 'cut': job['cut'],
                                'max_abs_error': error, 'reference_sha256': sha(old_path),
                                'candidate_sha256': sha(path)})
    assert len(verified) == 300
    assert len(comparisons) == 71, len(comparisons)
    result = {'schema': 'parallel.hyperion_recovery_parity.v1', 'passed': True,
              'timestamp': time.time(), 'jobs_sha256': sha(job_path),
              'reference_spec_sha256': sha(old_spec_path), 'candidate_spec_sha256': sha(new_spec_path),
              'frozen_contract_unchanged': True, 'changed_spec_fields': sorted(changed),
              'runtime_changes': ['24000-cell forward cap restored', '24000-cell coalescing cap',
                                  '60-second stalled CUDA forward deadline', 'in-flight batch receipts',
                                  'stricter saved-cut identity checks'],
              'preserved_vectors_unchanged': snapshot['saved_vectors'],
              'preserved_artifacts_unchanged': len(snapshot['files']),
              'preservation_snapshot_sha256': sha(snapshot_path),
              'shared_input_comparisons': len(comparisons), 'tolerance_max_abs': 0.0003,
              'max_abs_error': max(c['max_abs_error'] for c in comparisons),
              'comparisons': comparisons, 'verified_encodings': verified,
              'qualification_only': True, 'original_final_test_targets_opened': False}
    out = root / 'runs/parity/hyperion-reviewed4-recovery-parity.json'
    assert not out.exists(), 'Immutable recovery parity receipt already exists'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'verified': 300, 'shared_input_comparisons': 71,
                      'max_abs_error': result['max_abs_error'], 'receipt': str(out), 'sha256': sha(out)}))


if __name__ == '__main__':
    main()
