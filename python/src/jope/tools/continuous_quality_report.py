#!/usr/bin/env python3
"""Attribute tournament outcomes to initial seeds, retained teachers and later GP."""
import argparse
import math
from pathlib import Path
import time
from cut_baseline import annotate_observation
from resume_expired_cycle import atomic, read, require, sha
from scored_family_allowlist import inspect_prepared, quality_rows_on_allowlist, report as allowlist_report

# Origins whose evaluated outcome descends from this search's own population or its GP offspring.
# Retained teachers/champions are the shared tail across arms (their descendants are not separable).
OWN_LINEAGE_ORIGINS = ('neural_policy', 'generic_comparison', 'gp_evolution')


def metric(row):
    value = row.get('metrics') or {}
    task = row.get('task') or 'regression'
    if task != 'regression':
        primary = value.get('primary')
        if type(primary) in (int, float) and math.isfinite(primary) and primary >= 0:
            return primary
    # Native evaluation receipts use validation_rmse (older receipts may use rmse).
    for name in ('validation_rmse', 'rmse'):
        score = value.get(name)
        if type(score) in (int, float) and math.isfinite(score) and score >= 0:
            scale = row.get('target_scale')
            if value.get('units') == 'cut_normalized_target':
                require(type(scale) in (int, float) and math.isfinite(scale) and scale > 0,
                        'normalized teacher score lacks target scale')
                return score * scale
            require(value.get('units') == 'original_target', 'teacher score units missing or unknown')
            return score
    return None


def search_row(root, generation, job):
    directory = root / job['output']
    search = read(directory / 'search.json')
    require(search['rng_seed'] == job['rng_seed'] and search['seeds_sha256'] == job['seeds_sha256'],
            'GP result identity changed')
    observations = read(directory / 'observations.json')
    if 'observations_sha256' in search:
        require(search['observations_sha256'] == sha(directory / 'observations.json'), 'committed observation digest changed')
    prepared = read(root / job['prepared'] / 'prepared.json')
    require(sha(root / job['prepared'] / 'prepared.json') == job['prepared_sha256'], 'teacher prepared data changed')
    cut = next(c for c in prepared['cuts'] if c['index'] == job['cut'])
    initial_neural, retained, later, scored = [], [], [], []
    initial_arm, own_lineage = [], []
    for observation in observations:
        require(observation.get('prepared_sha256') == job['prepared_sha256']
                and observation.get('cut') == job['cut']
                and observation.get('training_sha256') == cut['training_sha256']
                and observation.get('validation_sha256') == cut['validation_sha256'], 'teacher observation cut conflict')
        score = metric(observation)
        if score is None:
            continue
        origin = observation.get('origin', '')
        if observation.get('failure') is not None:
            continue
        slot = observation.get('initial_seed_slot')
        if type(slot) is int and 0 <= slot < 64 and origin == 'neural_policy':
            initial_neural.append(score)
        if type(slot) is int and 0 <= slot < 64:
            initial_arm.append(score)  # this arm's own population, whatever its origin string
        if origin in OWN_LINEAGE_ORIGINS:
            own_lineage.append((score, origin))
        if origin in ('incumbent_cut_teacher', 'retained_cut_teacher') or 'champion' in origin:
            retained.append(score)
        if slot is None:
            later.append(score)
        scored.append((score, origin, observation))
    best = min(scored, key=lambda r: r[0], default=None)
    if best:
        require(type(search.get('best_validation_rmse')) in (int, float)
                and math.isclose(search['best_validation_rmse'], best[0], rel_tol=1e-12, abs_tol=1e-12),
                'search summary disagrees with evaluated outcomes')
    task = prepared.get('task') or 'regression'
    metric_name, units = {
        'regression': ('rmse', 'original_target'),
        'binary': ('logloss', 'class_index'),
        'multiclass': ('logloss', 'class_index'),
        'timeseries': ('mase', 'original_target'),
    }.get(task, ('rmse', 'original_target'))
    row = dict(output=job['output'], dataset=prepared['dataset'], family=prepared['family'],
         partition=prepared['meta_partition'], cut=job['cut'], rng_seed=job['rng_seed'],
         arm=Path(job['output']).relative_to(generation.relative_to(root)).parts[0],
         best_initial_neural_seed=min(initial_neural, default=None),
         best_retained_teacher=min(retained, default=None), best_later_gp=min(later, default=None),
         best_final_gp=search.get('best_validation_primary', search.get('best_validation_rmse')),
         metric=metric_name, units=units, task=task,
         completed_evaluations=search['completed_evaluations'], search_sha256=sha(directory / 'search.json'),
         observation_sha256=sha(directory / 'observations.json'))
    row['best_final_origin'] = best[1] if best else 'unavailable'
    if best:
        for key in ('prepared_sha256', 'training_sha256', 'validation_sha256', 'evaluator_sha256', 'vocabulary_sha256'):
            row[key] = best[2].get(key)
    row["completed_at"] = search.get("completed_at", (directory / "search.json").stat().st_mtime)
    row['search_best_units'] = 'original_target'
    row['observation_metric_units'] = 'cut_normalized_target'
    attach_cut_baseline(row, prepared, cut, best[2] if best else None)
    row['pure_predicted_best'] = dict(status='protocol_unavailable', rmse=None)
    row['best_evaluated_pure_pool'] = None
    prediction_path = (root / job['seeds']).parent / 'pure-best-predicted.json'
    if prediction_path.is_file():
        prediction = read(prediction_path)
        ranking_path = prediction_path.with_name('pure-ranking.json')
        require(sha(ranking_path) == prediction['pure_ranking_sha256'], 'pure ranking identity changed')
        ranking = read(ranking_path)['ranking']['ranked']
        pool = {p['program_sha256'] for p in ranking}
        selected = prediction['prediction']['program_sha256']
        require(selected in pool and prediction['teacher_population_consulted'] is False,
                'pure prediction population changed')
        evaluated = [o for o in observations if o.get('program_sha256') == selected]
        require(len(evaluated) <= 1, 'duplicate selected-program evaluation')
        score = metric(evaluated[0]) if evaluated and evaluated[0].get('failure') is None else None
        row['pure_predicted_best'] = dict(program_sha256=selected, rmse=score,
            status='evaluated_ok' if score is not None else 'failed' if evaluated else 'not_evaluated_within_gp_budget',
            prediction_sha256=sha(prediction_path))
        pool_scores = [score for score,_,o in scored if o.get('program_sha256') in pool]
        row['best_evaluated_pure_pool'] = min(pool_scores, default=None)
    row['direct_measurement_scope'] = 'current search observations; unevaluated pure proposals remain missing'
    # Additive flywheel producers (parallel.flywheel_baseline.v1); the keys above are untouched.
    row['best_initial_arm'] = min(initial_arm, default=None)
    final, seed = row['best_final_gp'], row['best_initial_neural_seed']
    row['gp_improves_on_seed'] = ((final < seed) if all(type(v) in (int, float) and math.isfinite(v) for v in (final, seed))
                                  else None)
    own = min(own_lineage, key=lambda r: r[0], default=None)
    row['best_final_excluding_shared_tail'] = dict(rmse=own[0], origin=own[1], tail_contaminated=True) if own else None
    return row


def attach_cut_baseline(row, prepared, cut, observation):
    """Compare E and B in original units using retained, cut-bound evidence.

    A reporting pass cannot reconstruct missing baselines from target payloads.
    This checks producer bindings/units; independent custody verification remains
    a separate release gate, not a claim made by this diagnostic.
    """
    row.update(annotate_observation(None, None, units='original_target'))
    row['prediction_contract'] = prepared.get('prediction_contract')
    evidence = ((observation or {}).get('metrics') or {}).get('baseline_evidence')
    if not evidence:
        row['baseline_status'] = 'original_baseline_evidence_unavailable'
        return row
    require(evidence.get('schema') == 'parallel.cut_baseline.v1'
            and evidence.get('rule') == 'training_target_mean_on_scoring_rows'
            and evidence.get('training_sha256') == cut.get('training_sha256')
            and evidence.get('scoring_sha256') == cut.get('validation_sha256')
            and evidence.get('preprocessing_sha256') == cut.get('preprocessing_sha256')
            and all(isinstance(cut.get(key), str) and len(cut[key]) == 64
                    for key in ('training_sha256', 'validation_sha256', 'preprocessing_sha256')),
            'baseline cut or preprocessing identity changed')
    require(all(type(evidence.get(key)) is int and evidence[key] > 0
                for key in ('training_rows', 'scoring_rows')), 'baseline row denominator unavailable')
    baseline = evidence.get('rmse')
    require(type(baseline) in (int, float) and math.isfinite(baseline) and baseline >= 0,
            'invalid baseline RMSE')
    if evidence.get('units') == 'cut_normalized_target':
        scale = observation.get('target_scale')
        require(type(scale) in (int, float) and math.isfinite(scale) and scale > 0,
                'normalized baseline lacks target scale')
        baseline *= scale
    else:
        require(evidence.get('units') == 'original_target', 'baseline units unavailable')
    require(math.isfinite(baseline), 'baseline rescaling overflow')
    row.update(annotate_observation(metric(observation), baseline, units='original_target'))
    row['baseline_status'] = 'producer_bound_units_checked; independent_custody_not_established_here'
    return row


def apply_allowlist_filter(payload):
    """Filter overlay. Does not rewrite family names or scores."""
    inspected = []
    for row in payload.get('rows') or []:
        inspected.append(inspect_prepared({
            'family': row.get('family'),
            'task': row.get('task') or 'regression',
            'prediction_contract': row.get('prediction_contract'),
            'cuts': row.get('cuts') or [],
        }))
    allow = allowlist_report(inspected)
    filtered = quality_rows_on_allowlist(payload, allow)
    out = dict(payload)
    out['allowlist'] = {'schema': allow['schema'], 'kept': filtered['kept'],
                        'skipped': filtered['skipped'], 'scored_families': allow['scored_families']}
    out['allowlisted_rows'] = filtered['rows']
    return out


def report(root, generation):
    manifest = read(generation / 'jobs.json')
    rows = [search_row(root, generation, job) for job in manifest['jobs']]
    payload = apply_allowlist_filter(dict(schema='parallel.continuous_quality.v1',
         timestamp=int(time.time()), jobs_sha256=sha(generation / 'jobs.json'), rows=rows,
         direct_neural_seed_scope='Initial neural slots actually evaluated within the matched GP budget; missing values are not imputed.',
         retained_teacher_scope='Evaluated retained/champion origins, distinct from subsequent GP improvements.',
         evaluation_bias='adaptive_cut_heldout_teacher', independent_evaluation=False, final_tests_opened=False,
         metric_unit_note='search best_validation_rmse is original_target; observation metrics.rmse is cut_normalized_target'))
    atomic(generation / 'quality-report.json', payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--generation', type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.generation)
