"""Freeze a new wave before numerical work. Stdlib only; no device/model import."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import digest, file_hash, publish, validate_plan


def freeze(config, inputs, output, source_commit, custody, node):
    original = json.loads((inputs / 'preregistration.json').read_text())
    receipt = json.loads((inputs / 'feature-receipt.json').read_text())
    root = HERE.parent
    sources = list(HERE.glob('*.py')) + [root / n for n in ('feedback.py', 'run.py', 'prepare.py')]
    sources += list((root / 'dltrain').rglob('*.py'))
    plan = {**json.loads(config.read_text()), 'schema': 'jope.research-wave/v1',
            'source_commit': source_commit,
            'source_sha256': {str(p.relative_to(root)): file_hash(p) for p in sorted(sources)},
            'input_pins': {n: file_hash(inputs / n) for n in ('inputs.tar.zst', 'features.npz', 'controls.json',
                           'preregistration.json', 'feature-receipt.json')},
            'native_checkpoint_parameters': original['checkpoint_parameters'],
            'vocabulary_sha256': 'fd64472eaa5bfca4196f54b4208a39dfd1d697631f420e946f5920990112f8d9',
            'custody_receipt_sha256': file_hash(custody),
            'custody_receipt_name': str(custody),
            'feature_receipt_sha256': file_hash(inputs / 'feature-receipt.json'),
            'selection_rule': 'fixed final updates; no checkpoint/threshold choice on development outcomes',
            'resource_admission': 'external exact protected job/whole-tree/CPU70 proof REQUIRED; not supplied by freeze',
            'fresh_gp_fits': 0}
    if node not in ('xbabe2', 'xbabe3') or (plan['assigned_node'] in ('xbabe2', 'xbabe3') and plan['assigned_node'] != node):
        raise ValueError('concept must remain on its assigned node')
    plan['assigned_node'] = node
    plan['run_id'] += '-' + node
    plan['comparison_identity'] = {
        'task': 'historical_finite_pool_budget4', 'input_archive_sha256': plan['input_pins']['inputs.tar.zst'],
        'features_sha256': receipt['features_sha256'], 'split_sha256': original['split_sha256'],
        'evaluation_role': 'calibration_as_development',
        'evaluator_sha256': digest({n: file_hash(root / n) for n in ('feedback.py', 'campaign/runner.py')}),
        'vocabulary_sha256': plan['vocabulary_sha256'], 'budget': 4}
    if plan['input_pins']['features.npz'] != receipt['features_sha256']:
        raise ValueError('feature receipt mismatch')
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('full source commit required')
    validate_plan(plan)
    publish(output, plan)
    return plan


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ('config', 'inputs', 'out', 'custody'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--node', choices=['xbabe2', 'xbabe3'], required=True)
    a = ap.parse_args()
    freeze(a.config, a.inputs, a.out, a.source_commit, a.custody, a.node)
    print(json.dumps({'state': 'preregistered_not_launched', 'sha256': file_hash(a.out)}))
