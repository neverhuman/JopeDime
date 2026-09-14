#!/usr/bin/env python3
"""Attribute tournament outcomes to initial seeds, retained teachers and later GP."""
import argparse
import math
from pathlib import Path
import time
from resume_expired_cycle import atomic, read, require, sha


def metric(row):
    value = row.get('metrics') or {}
    # Native evaluation receipts use validation_rmse (older receipts may use rmse).
    for name in ('validation_rmse', 'rmse'):
        score = value.get(name)
        if isinstance(score, (int, float)) and math.isfinite(score) and score >= 0:
            return score
    return None


def report(root, generation):
    manifest = read(generation / 'jobs.json')
    rows = []
    for job in manifest['jobs']:
        directory = root / job['output']
        search = read(directory / 'search.json')
        require(search['rng_seed'] == job['rng_seed'] and search['seeds_sha256'] == job['seeds_sha256'],
                'GP result identity changed')
        observations = read(directory / 'observations.json')
        initial_neural, retained, later = [], [], []
        for observation in observations:
            score = metric(observation)
            if score is None:
                continue
            origin = observation.get('origin', '')
            slot = observation.get('initial_seed_slot')
            if isinstance(slot, int) and slot < 64 and 'neural' in origin:
                initial_neural.append(score)
            if 'retained' in origin or 'champion' in origin:
                retained.append(score)
            if slot is None:
                later.append(score)
        prepared = read(root / job['prepared'] / 'prepared.json')
        rows.append(dict(output=job['output'], dataset=prepared['dataset'], family=prepared['family'],
             partition=prepared['meta_partition'], cut=job['cut'], rng_seed=job['rng_seed'],
             arm=Path(job['output']).relative_to(generation.relative_to(root)).parts[0],
             best_initial_neural_seed=min(initial_neural, default=None),
             best_retained_teacher=min(retained, default=None), best_later_gp=min(later, default=None),
             best_final_gp=search.get('best_validation_rmse'),
             completed_evaluations=search['completed_evaluations'], search_sha256=sha(directory / 'search.json'),
             observation_sha256=sha(directory / 'observations.json')))
    atomic(generation / 'quality-report.json', dict(schema='parallel.continuous_quality.v1',
         timestamp=int(time.time()), jobs_sha256=sha(generation / 'jobs.json'), rows=rows,
         direct_neural_seed_scope='Initial neural slots actually evaluated within the matched GP budget; missing values are not imputed.',
         retained_teacher_scope='Evaluated retained/champion origins, distinct from subsequent GP improvements.',
         evaluation_bias='adaptive_cut_heldout_teacher', independent_evaluation=False, final_tests_opened=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--generation', type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.generation)
