#!/usr/bin/env python3
"""Fail-closed, receipt-bound adaptive qualification; never writes serving state."""
import hashlib
import math
from pathlib import Path
import random
import statistics

import experiment_runtime as runtime
from resume_expired_cycle import read, require, sha


POLICY = dict(primary_improvement=0.01, nll_regression=0.005,
              ranking_regression=0.01, bootstrap_seed=20260909,
              bootstrap_replicates=10000, confidence=0.95,
              validation_families=8, cuts_per_family=2, seeds_per_cut=3,
              gpu_node_seconds=7200, cpu_core_seconds=230400,
              concurrent_searches=16, search_seconds=600, search_evaluations=1000,
              fit_seconds=30)
TASK_METRICS = {
    'regression': ('rmse', 'original_target'),
    'binary': ('logloss', 'class_index'),
    'multiclass': ('logloss', 'class_index'),
    'timeseries': ('mase', 'original_target'),
}
IDENTITY = ('family', 'prepared_sha256', 'cut', 'training_sha256',
            'validation_sha256', 'evaluator_sha256', 'vocabulary_sha256')


def canonical(value):
    import json
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def immutable(path, value):
    """Atomic durable object publication; retries must reproduce identical bytes."""
    data = canonical(value) + b'\n'
    if path.exists():
        require(path.read_bytes() == data, 'immutable receipt changed: ' + str(path))
    else:
        runtime.atomic_bytes(path, data)
    return sha(path)


def artifact(root, ref):
    require(set(ref) == {'path', 'sha256'}, 'artifact requires path and digest')
    path = (root / ref['path']).resolve()
    require(path.is_relative_to(root.resolve()), 'qualification path escapes workspace')
    require(sha(path) == ref['sha256'], 'qualification artifact changed: ' + ref['path'])
    return read(path)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def interval(values):
    """Resample families, preserving every matched cut/seed inside each cluster."""
    rng = random.Random(POLICY['bootstrap_seed'])
    means = sorted(statistics.fmean(rng.choices(values, k=len(values)))
                   for _ in range(POLICY['bootstrap_replicates']))
    return [means[int(len(means) * .025)], means[int(len(means) * .975)]]


def verify_execution_protocol(root, manifest, evidence):
    """Refuse legacy score-only evidence until the native outcome protocol lands.

    The v1 producer never bound query evaluation, pre-outcome selection, GP slot
    consumption, or monitor execution to original committed process receipts.
    Self-consistent artifact hashes cannot establish those facts. This gate has
    no environment/manifest override; integrating and qualifying that producer
    and verifier is required before any production decision can be accepted.
    """
    raise RuntimeError('promotion unavailable: original registered evaluation, selection, '
                       'search-slot and monitor execution receipts are not yet qualified')


def validate_manifest(root, manifest):
    require(manifest['schema'] == 'parallel.promotion_experiment.v1', 'unknown qualification manifest')
    require(manifest['policy'] == POLICY, 'promotion policy differs from predeclared rule')
    require(manifest['final_tests_opened'] is False, 'sealed tests are forbidden')
    require(manifest['proposal_protocol'] == 'training_context_neural_pool_v1', 'pure proposal protocol required')
    task = manifest.get('task', 'regression')
    require(task == 'regression', 'non-regression promotion awaits its separately qualified task contract')
    metric, units = TASK_METRICS[task]
    require(manifest['metric'] == metric and manifest['units'] == units, 'metric identity changed')
    require(manifest['frozen_at'] > 0, 'experiment must be frozen before outcomes')
    for ref in manifest['artifacts'].values():
        artifact(root, ref)
    required = {'corpus', 'vocabulary', 'panel', 'contexts', 'nomination'}
    require(required <= set(manifest['artifacts']), 'qualification inputs missing')
    nomination = artifact(root, manifest['artifacts']['nomination'])
    require(nomination['training_only'] is True and nomination['validation_results_used'] is False
            and nomination['checkpoint_sha256'] == manifest['arms']['challenger']['sha256'],
            'challenger nomination used validation or changed checkpoint')
    for ref in manifest['arms'].values():
        state = artifact(root, ref)
        directory = (root / ref['path']).parent
        for name, key in [('model.ot', 'model_sha256'), ('optimizer.ot', 'optimizer_sha256'),
                          ('optimizer.json', 'optimizer_metadata_sha256'), ('config.json', 'config_sha256')]:
            require(sha(directory / name) == state[key], 'checkpoint payload changed: ' + name)
        require(state['vocabulary_sha256'] == manifest['artifacts']['vocabulary']['sha256'], 'checkpoint vocabulary changed')
    require({'challenger', 'recovery', 'serving', 'original_serving', 'cycle25'} <= set(manifest['arms']),
            'reference checkpoints missing: recovery, original_serving, current serving and cycle25 are required')
    contexts = manifest['validation_contexts']
    families = {}
    require(len(contexts) == 16, 'validation requires sixteen cuts')
    for context in contexts:
        require(context['partition'] == 'meta_validation', 'non-validation qualification context')
        require(all(context.get(k) is not None for k in IDENTITY), 'context identity missing')
        require(context['evaluator_sha256'] == manifest['evaluator_sha256']
                and context['vocabulary_sha256'] == manifest['artifacts']['vocabulary']['sha256'],
                'context evaluator or vocabulary changed')
        families.setdefault(context['family'], set()).add(tuple(context[k] for k in IDENTITY))
    require(len(families) == 8 and all(len(cuts) == 2 for cuts in families.values()), 'validation family/cut coverage changed')
    require(len(set(manifest['rng_seeds'])) == 3, 'three matched seeds required')


def evaluate(root, manifest_ref, evidence_ref, candidate, serving_sha):
    """Return a durable decision body. Malformed/missing evidence is inconclusive."""
    decision = dict(schema='parallel.promotion_decision.v1', outcome='inconclusive',
                    checkpoint=candidate['checkpoint'], checkpoint_sha256=candidate['checkpoint_sha256'],
                    corpus_sha256=candidate['corpus_sha256'], serving_before_sha256=serving_sha,
                    manifest=manifest_ref, evidence=evidence_ref, reasons=[],
                    independent_evaluation=False, final_tests_opened=False)
    try:
        require(manifest_ref is not None and evidence_ref is not None, 'qualification evidence not supplied')
        manifest = artifact(root, manifest_ref)
        validate_manifest(root, manifest)
        evidence = artifact(root, evidence_ref)
        require(evidence['schema'] == 'parallel.promotion_evidence.v1'
                and evidence['manifest_sha256'] == manifest_ref['sha256'], 'evidence manifest changed')
        require(manifest['arms']['challenger']['sha256'] == candidate['checkpoint_sha256']
                and manifest['arms']['serving']['sha256'] == serving_sha
                and manifest['artifacts']['corpus']['sha256'] == candidate['corpus_sha256'],
                'candidate, corpus or captured serving identity changed')
        require(evidence['complete'] is True and evidence['final_tests_opened'] is False, 'incomplete or sealed evidence')
        verify_execution_protocol(root, manifest, evidence)
        for key, ceiling in [('gpu_node_seconds', 7200), ('cpu_core_seconds', 230400), ('max_concurrent_searches', 16)]:
            require(finite(evidence['resources'][key]) and evidence['resources'][key] <= ceiling, 'qualification budget exceeded: ' + key)
        # Every expected checkpoint arm is evaluated; identical references are deduplicated.
        arms = {v['sha256'] for v in manifest['arms'].values()} | {'generic', 'shuffled:' + candidate['checkpoint_sha256']}
        expected = {(tuple(c[k] for k in IDENTITY), seed, arm)
                    for c in manifest['validation_contexts'] for seed in manifest['rng_seeds'] for arm in arms}
        results = {}
        for ref in evidence['comparisons']:
            row = artifact(root, ref)
            require(row['schema'] == 'parallel.promotion_comparison.v1'
                    and row['manifest_sha256'] == manifest_ref['sha256'], 'comparison binding changed')
            key = (tuple(row['context'][k] for k in IDENTITY), row['rng_seed'], row['arm'])
            require(key in expected and key not in results, 'cross-cut, extra or duplicate comparison')
            require(row['started_at'] >= manifest['frozen_at'], 'outcome predates frozen experiment')
            require(row['search_budget'] == dict(seconds=600, evaluations=1000)
                    and row['population_slots'] == manifest['population_slots'], 'unequal comparison budgets/population')
            require(row['metric'] == 'rmse' and row['units'] == 'original_target', 'comparison units changed')
            require(row['complete'] is True, 'comparison unfinished')
            for name in ('direct', 'search'):
                outcome = row[name]
                receipt = artifact(root, outcome['receipt'])
                require(receipt['context'] == row['context'] and receipt['rng_seed'] == row['rng_seed']
                        and receipt['arm'] == row['arm'] and receipt['metric'] == 'rmse'
                        and receipt['units'] == 'original_target', 'evaluation receipt identity changed')
                require(outcome['status'] == receipt['status'] and outcome['rmse'] == receipt['rmse'], 'evaluation result changed')
                require(outcome['status'] in ('ok', 'failed'), 'missing evaluation cannot count as tie')
                require(outcome['rmse'] is None if outcome['status'] == 'failed' else finite(outcome['rmse']), 'nonfinite outcome')
                if name == 'direct' and row['arm'] != 'generic':
                    selected = artifact(root, outcome['proposal'])
                    require(selected['schema'] == 'parallel.pure_best_predicted_program.v1'
                            and selected['protocol'] == manifest['proposal_protocol']
                            and selected['teacher_population_consulted'] is False,
                            'teacher/fallback cannot be a pure neural proposal')
                    prediction = selected['prediction']
                    require(prediction['program_sha256'] == hashlib.sha256(prediction['program'].encode()).hexdigest(),
                            'selected program digest changed')
                    require(prediction['program_sha256'] == receipt['program_sha256']
                            and selected['checkpoint_sha256'] == row['arm'].removeprefix('shuffled:'),
                            'selected proposal evaluation changed')
                    require(selected['origin'] in ('neural_policy', 'neural_rejection_fallback'), 'selected origin missing or unknown')
                    if selected['origin'] != 'neural_policy' and row['arm'] == candidate['checkpoint_sha256']:
                        decision['reasons'].append('challenger selected generic fallback; neural gain not established')
            results[key] = row
        require(set(results) == expected, 'missing required matched comparisons')
        monitors = {}
        for name in ('challenger', 'recovery'):
            row = artifact(root, evidence['monitoring'][name])
            require(row['binding']['checkpoint_sha256'] == manifest['arms'][name]['sha256'], 'monitor checkpoint changed')
            require(row['binding']['panel_sha256'] == manifest['panel_sha256']
                    and row['binding']['corpus_sha256'] == manifest['monitor_corpus_sha256']
                    and row['binding']['executable_sha256'] == manifest['monitor_executable_sha256'], 'monitor panel/evaluator changed')
            require(row['sampled_examples'] == 2048 and row['family_balanced'] is True
                    and row['final_tests_opened'] is False, 'monitor scope changed')
            require(all(finite(row[k]) for k in ('sequence_nll', 'ranking_bt', 'ranking_listwise')), 'nonfinite monitor')
            monitors[name] = row
        reasons = list(decision['reasons'])
        for metric, tolerance in [('sequence_nll', .005), ('ranking_bt', .01), ('ranking_listwise', .01)]:
            require(monitors['recovery'][metric] > 0, 'zero diagnostic reference: ' + metric)
            if monitors['challenger'][metric] > monitors['recovery'][metric] * (1 + tolerance):
                reasons.append('diagnostic regression: ' + metric)
        challenger = candidate['checkpoint_sha256']
        comparisons = {}
        for reference in ('recovery', 'original_serving', 'serving'):
            ref_sha = manifest['arms'][reference]['sha256']
            values, final = {}, {}
            failures = {challenger: 0, ref_sha: 0}
            for context in manifest['validation_contexts']:
                identity = tuple(context[k] for k in IDENTITY)
                for seed in manifest['rng_seeds']:
                    a, b = [results[(identity, seed, arm)] for arm in (challenger, ref_sha)]
                    # Selected failures count against the challenger, never drop from means.
                    if a['direct']['status'] == 'failed':
                        reasons.append('challenger selected proposal failed')
                        continue
                    require(b['direct']['status'] == 'ok' and b['direct']['rmse'] > 0, 'reference direct outcome unavailable or zero')
                    values.setdefault(context['family'], []).append(1 - a['direct']['rmse'] / b['direct']['rmse'])
                    for arm, row in ((challenger, a), (ref_sha, b)):
                        failures[arm] += row['search']['status'] == 'failed'
                    if a['search']['status'] == 'failed':
                        reasons.append('challenger matched search failed')
                    else:
                        require(b['search']['status'] == 'ok' and b['search']['rmse'] > 0, 'reference search outcome unavailable or zero')
                        final.setdefault(context['family'], []).append(a['search']['rmse'] / b['search']['rmse'] - 1)
            if len(values) != 8 or any(len(v) != 6 for v in values.values()):
                reasons.append('challenger direct failures prevent complete family means')
                continue
            means = [statistics.fmean(values[k]) for k in sorted(values)]
            ci = interval(means)
            comparisons[reference] = dict(relative_improvement=statistics.fmean(means), confidence_interval=ci)
            if statistics.fmean(means) < .01 or ci[0] <= 0:
                reasons.append('insufficient direct improvement versus ' + reference)
            if reference == 'serving':
                if failures[challenger] > failures[ref_sha]:
                    reasons.append('matched search failure rate worsened')
                if len(final) != 8 or statistics.fmean(statistics.fmean(v) for v in final.values()) > 0:
                    reasons.append('family-balanced final GP RMSE worsened')
        decision.update(outcome='rejected' if reasons else 'accepted', reasons=sorted(set(reasons)), comparisons=comparisons)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ZeroDivisionError) as error:
        decision['reasons'] = [str(error)]
    return decision
