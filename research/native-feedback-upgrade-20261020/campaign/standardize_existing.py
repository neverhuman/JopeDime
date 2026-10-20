"""Re-express the immutable completed trial; performs no fit or new assessment."""
import argparse
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import digest, file_hash, publish, validate_result


def standardize(root, out):
    old = json.loads((root / 'experiment/comparison.json').read_text())
    plan = json.loads((root / 'preregistration.json').read_text())
    receipt = json.loads((root / 'feature-receipt.json').read_text())
    identity = {'task': 'historical_finite_pool_budget4', 'input_archive_sha256': file_hash(root / 'inputs.tar.zst'),
                'features_sha256': receipt['features_sha256'], 'split_sha256': plan['split_sha256'],
                'evaluation_role': 'original_214_context_assessment',
                'evaluator_sha256': digest({n: plan['source_sha256'][n] for n in ('feedback.py', 'run.py')}),
                'vocabulary_sha256': 'fd64472eaa5bfca4196f54b4208a39dfd1d697631f420e946f5920990112f8d9', 'budget': 4}
    hashes = {}
    for path in sorted((root / 'experiment').glob('*-consumption.json.gz')):
        receipts = json.loads(gzip.decompress(path.read_bytes()))
        for mi, r in enumerate(receipts):
            if isinstance(r, dict) and 'parameter_before' in r and 'parameter_after' in r:
                hashes[f'{path.name}:{mi}'] = {'before': r['parameter_before'], 'after': r['parameter_after']}
    if not hashes:
        raise ValueError('locate actual original optimizer receipts; do not fabricate ancestry')
    primary = dict(old['paired_99pct_intervals_budget4']['uniform_feedback-continuation'])
    primary['decision'] = 'inconclusive' if primary['lower'] <= 0 <= primary['upper'] else ('positive' if primary['lower'] > 0 else 'negative')
    metrics = {n.split('/')[0]: {'parent_mean_regret': m['regret'], 'parent_mean_quality': m['quality'],
               'parent_mean_beats_incumbent': m['beats_incumbent']}
               for n, m in old['metrics'].items() if n.endswith('/budget4')}
    logical = sum(v['charged_logical_replay_slots'] for v in old['acquisition'].values())
    r = {'schema': 'jope.research-result/v1', 'state': 'complete', 'run_id': 'original-historical-trial-20261020',
         'branch': old['branch'], 'source_commit': 'ca019e7c2c514f2875d64e12db1f28e4760d60a4',
         'source_commit_scope': 'Post-run source publication; immutable preregistration pins actual original trial source.',
         'preregistration_sha256': old['preregistration_sha256'], 'concept': 'original_composite_control',
         'scope': 'historical_replay', 'comparison_identity': identity, 'comparison_identity_sha256': digest(identity),
         'metrics': metrics, 'primary': primary, 'parameter_hashes': hashes, 'hwm_promotion': False,
         'counts': {'optimizer_updates': old['actual_adapter_optimizer_updates'], 'fresh_gp_fits': 0,
                    'historical_replay_slots': logical, 'cache_reads': logical, 'failures': 0},
         'counter_scope': 'Unmodified original per-arm charged replay-slot receipts; shared initialization is repeated per arm in that historical report. New wave uses explicit once-only shared initialization accounting; do not compare these cost counters as identical.',
         'cost': {'wall_seconds': old['elapsed_seconds'], 'scope': 'Original reported process, not a new whole-fleet measurement'},
         'original_comparison_sha256': file_hash(root / 'experiment/comparison.json'),
         'limitations': old['limitations'] + ['Reformatted only. Incompatible with new calibration-development identity; no new lift or HWM.']}
    validate_result(r)
    publish(out, r)
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    standardize(a.inputs, a.out)
