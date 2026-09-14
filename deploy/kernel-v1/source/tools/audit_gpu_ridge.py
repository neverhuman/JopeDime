#!/usr/bin/env python3
"""Read completed GP ledgers; estimate CUDA shape eligibility without evaluations.

Shape rules mirror the pinned feat-math ops.rs. Unknown syntax/operators stay
explicitly unknown. Eligibility is for a ridge screening problem, never a claim
that a rounded surrogate score can replace the original program's f64 reward.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

SAME = set('standardize norm_features robust_scale rank blom quantile_bins winsor_clip winsor_moments target_bin_encode sigmoid step norm_row polar lag roll symlog map'.split())
HEADS = set('ridge ridge_gcv logistic elasticnet histgbm histgbm2 gbm gpr krr ksmooth ensemble stack ensemble_ridge_gbm portfolio'.split())


def stages(genome):
    found = re.findall(r'\([^()]*\)', genome)
    if '(pipe X ' + ' '.join(found) + ')' != genome:
        return None
    return [(s[1:-1].split()[0], dict(re.findall(r'(\w+)=([^ ]+)', s[1:-1]))) for s in found]


def width(parts, p, n):
    for op, params in parts:
        def k(key, default, lo, hi):
            raw = params.get(key, str(default))
            value = int(raw) if raw.isdigit() else default
            return min(hi, max(lo, value))
        if op in SAME:
            q = p
        elif op == 'bin_onehot': q = p*k('bins', 6, 2, 32)
        elif op == 'hinge': q = p*k('knots', 4, 1, 16)
        elif op == 'cheb': q = p*k('degree', 4, 1, 8)
        elif op == 'poly_interact': q = min(512, p*(p+1)//2)
        elif op in ('rff_cos', 'rff_sin', 'oblique_hinge'): q = k('n', 32, 1, 256)
        elif op == 'rand_proj': q = k('k', 16, 1, 256)
        elif op == 'landmark_dist': q = k('k', 16, 1, 128)
        elif op == 'nystrom': q = min(n, k('k', 16, 1, 128))
        elif op == 'row_stats': q = 10
        elif op == 'row_spectral': q = 5
        elif op == 'row_diff_stats': q = 8
        elif op == 'fft_mag': q = k('k', 8, 1, max(1, p//2))
        elif op == 'svd_project': q = min(p, k('k', 8, 1, 64))
        elif op == 'pair_prod':
            selected = min(p, k('topk', 8, 2, 24))
            q = selected*(selected-1)//2
        elif op == 'select_corr': q = min(p, k('k', 32, 1, 512))
        elif op == 'derive': q = 1
        else: return None, op
        p = max(1, min(512, q + (p if params.get('append') in ('true', '1') else 0)))
    return p, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cycle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    assert not a.output.exists(), 'Use a new immutable report path'
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(a.root))] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return json.loads(raw)

    groups = defaultdict(lambda: dict(count=0, failures=0, runtime_seconds=0., fit_predict_metric_seconds=0.))
    unknown = Counter()
    view_shapes = {}
    widths = Counter()
    programs = set()
    for arm in ('neural', 'generic', 'shuffled'):
        jobs = read(a.cycle/arm/'jobs.json')['jobs']
        for job in jobs:
            prepared = read(a.root/job['prepared']/'prepared.json')
            cut = next(c for c in prepared['cuts'] if c['index'] == job['cut'])
            n, vn, p = len(cut['train_indices']), len(cut['validation_indices']), len(prepared['storage']['feature_columns'])
            view_shapes[(prepared['dataset'], job['cut'])] = dict(dataset=prepared['dataset'], cut=job['cut'], train_rows=n, heldout_rows=vn, input_columns=p)
            rows = read(a.root/job['output']/'observations.json')
            for row in rows:
                parts = stages(row['genome'])
                head = parts[-1][0] if parts and parts[-1][0] in HEADS else 'unknown'
                transformed, reason = width(parts[:-1], p, n) if head != 'unknown' else (None, 'parse_or_head')
                keys = ['all', 'head:'+head, 'arm:'+arm]
                if transformed is None:
                    unknown[reason] += 1
                    keys.append('shape:unknown')
                else:
                    widths[transformed] += 1
                    keys.append('shape:1_to_96' if transformed <= 96 else 'shape:over_96')
                    if transformed <= 96:
                        keys.append('screen_shape_eligible')
                        if head == 'ridge': keys.append('real_ridge_shape_eligible')
                if head == 'ridge': keys.append('real_ridge_all_shapes')
                programs.add(row['program_sha256'])
                for key in keys:
                    v = groups[key]
                    v['count'] += 1
                    v['failures'] += row['failure'] is not None
                    v['runtime_seconds'] += row['runtime_seconds']
                    v['fit_predict_metric_seconds'] += row['metrics']['seconds'] if row['metrics'] else 0.
    for value in groups.values():
        value['observation_percent'] = 100*value['count']/groups['all']['count']
        value['runtime_percent'] = 100*value['runtime_seconds']/groups['all']['runtime_seconds']
        value['fit_predict_metric_percent'] = 100*value['fit_predict_metric_seconds']/groups['all']['fit_predict_metric_seconds']
    source_files = ['src/gp.rs', 'assets/source/jope-dime/vendor/feat-math/src/ops.rs', 'assets/source/jope-dime/vendor/feat-math/src/heads.rs', 'assets/source/jope-dime/vendor/feat-math/src/evolve.rs', 'assets/source/jope-dime/vendor/battle-gpu/src/lib.rs', 'assets/source/jope-dime/vendor/battle-gpu/kernels/ridge_screen.cu']
    for name in source_files:
        path = a.root/name
        if path.exists():
            raw = path.read_bytes()
            sources[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    result = dict(schema='parallel.cuda_ridge_workload_audit.v1', read_only=True, gpu_work_launched=False,
        evaluation_bias='adaptive_cut_heldout_teacher', final_tests_opened=False,
        count=len(view_shapes), unique_programs=len(programs), groups=dict(sorted(groups.items())),
        transformed_width_inference='static_rules_for_recognized_ops_no_data_or_model_execution',
        unknown_width_reasons=dict(unknown), inferred_width_histogram=dict(sorted(widths.items())), views=list(view_shapes.values()),
        limitations=['No encoder, feature transform, head, or GPU was run.',
            'runtime_seconds includes isolated worker overhead; metrics.seconds combines fit, prediction, and metric computation.',
            'No per-stage timing exists, so head-only or GPU-speedup claims are unavailable.',
            'Shape eligibility is necessary, not sufficient: successful CPU transform preparation and numerical GPU parity are still required.',
            'Existing GPU scorer casts features and train targets to f32 and returns surrogate R2; original f64 rewards must be recomputed.',
            'Shape inference uses original program parameters, not the separate search_surrogate parameter clamps.'],
        sources=sources)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(groups=result['groups'], unknown=result['unknown_width_reasons'], views=len(view_shapes)), indent=2))


if __name__ == '__main__': main()
