#!/usr/bin/env python3
"""Report a completed adaptive-teaching cycle from immutable search ledgers.

No source datasets or final-test targets are read. Exact evaluation prefixes
use qualified cumulative completion clocks where available. Older or resumed
ledgers retain conservative timing bounds; they are never called exact.
"""
import argparse
from experiment_runtime import require
import bisect
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ('neural', 'generic', 'shuffled')
PAIRS = (('neural', 'generic'), ('neural', 'shuffled'), ('shuffled', 'generic'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


PURE_PROTOCOL = 'training_context_neural_pool_v1'
ASSEMBLY_PROTOCOL = 'preserve_shared_tail_dedup_v1'
SELECTION_ID = 'mean_reward_heads_desc_program_hash_v1'


def compact_digest(value):
    # Matches serde_json::to_vec for these string lists / SeedCandidate structs.
    return digest(json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())


def verify_ranking(ranking, checkpoint, context, vocabulary):
    require((ranking['schema'] == 'parallel.proposal_quality_ranking.v1'), "Invalid artifact: ranking['schema'] == 'parallel.proposal_quality_ranking.v1'")
    require((ranking['selection_id'] == SELECTION_ID), "Invalid artifact: ranking['selection_id'] == SELECTION_ID")
    require((ranking['bindings'] == dict(checkpoint_sha256=checkpoint,
                                      conditioning_context_sha256=context,
                                      vocabulary_sha256=vocabulary)), "Invalid artifact: ranking['bindings'] == dict(checkpoint_sha256=checkpoint,                                       conditioning_context_sha256=context,                                       vocabulary_sha256=vocabulary)")
    require((ranking['training_context_only'] is True), "Invalid artifact: ranking['training_context_only'] is True")
    require((ranking['heldout_labels_read'] is False and ranking['gp_scores_read'] is False), "Invalid artifact: ranking['heldout_labels_read'] is False and ranking['gp_scores_read'] is False")
    rows = ranking['ranked']
    require((ranking['pool_size'] == ranking['chosen_count'] == len(rows) == 64), "Invalid artifact: ranking['pool_size'] == ranking['chosen_count'] == len(rows) == 64")
    require((len({r['program'] for r in rows}) == 64), "Invalid artifact: len({r['program'] for r in rows}) == 64")
    for row in rows:
        require((digest(row['program'].encode()) == row['program_sha256']), "Invalid artifact: digest(row['program'].encode()) == row['program_sha256']")
        require((math.isfinite(row['predicted_quality'])), "Invalid artifact: math.isfinite(row['predicted_quality'])")
        require((math.isfinite(row['predicted_runtime_seconds']) and row['predicted_runtime_seconds'] >= 0), "Invalid artifact: math.isfinite(row['predicted_runtime_seconds']) and row['predicted_runtime_seconds'] >= 0")
        require((len(row['reward_heads']) == 4 and all(math.isfinite(x) for x in row['reward_heads'])), "Invalid artifact: len(row['reward_heads']) == 4 and all(math.isfinite(x) for x in row['reward_heads'])")
    require((rows == sorted(rows, key=lambda r: (-r['predicted_quality'], r['program_sha256'], r['program']))), "Invalid artifact: rows == sorted(rows, key=lambda r: (-r['predicted_quality'], r['program_sha256'], r['program']))")
    require((ranking['pool_sha256'] == compact_digest(sorted(r['program_sha256'] for r in rows))), "Invalid artifact: ranking['pool_sha256'] == compact_digest(sorted(r['program_sha256'] for r in rows))")
    return [r['program'] for r in rows]


def verify_generation(directory, gen, seeds, job, first, read):
    """Verify saved proposal provenance without reading source data or targets.

    A ranking's training_context_only flag qualifies scoring a fixed pool.
    Only the separate v2 pure receipt establishes pool independence from the
    current retained teachers; no GP population receives that claim.
    """
    origins = Counter(c['origin'] for c in seeds['candidates'][:64])
    schema = gen['schema']
    require((gen['seeds_sha256'] == job['seeds_sha256']), "Invalid artifact: gen['seeds_sha256'] == job['seeds_sha256']")
    if schema == 'parallel.proposal_generation.v1':
        require((origins['neural_policy'] + origins['neural_rejection_fallback'] == 64), "Invalid artifact: origins['neural_policy'] + origins['neural_rejection_fallback'] == 64")
        # Preserve legacy numerical counters exactly. Its duplicate field mixes
        # within-model duplicates and heldout-selected population collisions.
        return {k: gen[k] for k in ('attempted', 'valid', 'duplicates', 'unique_neural')}, 'legacy_filtered_pool'
    require((schema == 'parallel.proposal_generation.v2'), 'Unknown generation protocol')
    require((gen['pure_prediction_protocol'] == PURE_PROTOCOL and gen['gp_assembly_protocol'] == ASSEMBLY_PROTOCOL), "Invalid artifact: gen['pure_prediction_protocol'] == PURE_PROTOCOL and gen['gp_assembly_protocol'] == ASSEMBLY_PROTOCOL")
    require((origins['neural_policy'] + origins['neural_rejection_fallback'] + origins['neural_overlap_fallback'] == 64), "Invalid artifact: origins['neural_policy'] + origins['neural_rejection_fallback'] + origins['neural_overlap_fallback'] == 64")
    require((gen['unique_neural'] == origins['neural_policy']), "Invalid artifact: gen['unique_neural'] == origins['neural_policy']")
    require(('duplicates' not in gen), 'v2 cannot relabel mixed legacy duplicates')
    for field in ('attempted', 'valid', 'unique_neural', 'pure_unique_neural', 'within_neural_duplicates'):
        require((isinstance(gen[field], int) and gen[field] >= 0), 'Invalid artifact: isinstance(gen[field], int) and gen[field] >= 0')
    require((gen['within_neural_duplicates'] <= gen['valid'] <= gen['attempted']), "Invalid artifact: gen['within_neural_duplicates'] <= gen['valid'] <= gen['attempted']")
    require((gen['pure_unique_neural'] <= 64 and gen['filled_population'] == 80), "Invalid artifact: gen['pure_unique_neural'] <= 64 and gen['filled_population'] == 80")
    require((gen['final_tests_opened'] is False), "Invalid artifact: gen['final_tests_opened'] is False")
    expected = {
        'pure_ranking_sha256': 'pure-ranking.json',
        'pure_best_predicted_sha256': 'pure-best-predicted.json',
        'gp_ranking_sha256': 'gp-ranking.json',
        'gp_population_assembly_sha256': 'gp-population-assembly.json',
    }
    artifacts = {}
    for key, name in expected.items():
        path = directory/name
        artifacts[name] = read(path)
        require((digest(path.read_bytes()) == gen[key]), f'Proposal artifact hash mismatch: {name}')
    pure, best, gp, assembly = (artifacts[name] for name in expected.values())
    snapshot_path = directory.parents[2]/'champion-snapshot.json'
    snapshot = read(snapshot_path)
    snapshot_sha = digest(snapshot_path.read_bytes())
    require((snapshot_sha == gen['champion_snapshot_sha256']), "Invalid artifact: snapshot_sha == gen['champion_snapshot_sha256']")
    require((snapshot['schema'] == 'parallel.champion_snapshot.v1'), "Invalid artifact: snapshot['schema'] == 'parallel.champion_snapshot.v1'")
    checkpoint, context = gen['checkpoint_sha256'], gen['conditioning_context_sha256']
    vocabulary = snapshot['vocabulary_sha256']
    require((snapshot['request']['checkpoint_sha256'] == checkpoint == seeds['checkpoint_sha256']), "Invalid artifact: snapshot['request']['checkpoint_sha256'] == checkpoint == seeds['checkpoint_sha256']")
    require((snapshot['request']['vocabulary_sha256'] == vocabulary == first['vocabulary_sha256']), "Invalid artifact: snapshot['request']['vocabulary_sha256'] == vocabulary == first['vocabulary_sha256']")
    require((gen['selection_id'] == SELECTION_ID), "Invalid artifact: gen['selection_id'] == SELECTION_ID")
    if gen['context_mode'] == 'matching_training_cut':
        require((context == seeds['context_sha256']), "Invalid artifact: context == seeds['context_sha256']")
    else:
        require((gen['context_mode'] == 'shuffled_between_source_families'), "Invalid artifact: gen['context_mode'] == 'shuffled_between_source_families'")
    require((seeds['prepared_sha256'] == job['prepared_sha256'] == first['prepared_sha256']), "Invalid artifact: seeds['prepared_sha256'] == job['prepared_sha256'] == first['prepared_sha256']")
    require((seeds['cut'] == job['cut'] == first['cut']), "Invalid artifact: seeds['cut'] == job['cut'] == first['cut']")
    require((seeds['training_sha256'] == first['training_sha256']), "Invalid artifact: seeds['training_sha256'] == first['training_sha256']")
    require((pure['schema'] == 'parallel.pure_neural_ranking.v1' and pure['protocol'] == PURE_PROTOCOL), "Invalid artifact: pure['schema'] == 'parallel.pure_neural_ranking.v1' and pure['protocol'] == PURE_PROTOCOL")
    require((pure['teacher_population_consulted'] is False), "Invalid artifact: pure['teacher_population_consulted'] is False")
    require((pure['sampling_seed'] == gen['seed']), "Invalid artifact: pure['sampling_seed'] == gen['seed']")
    for key in ('attempted', 'valid', 'within_neural_duplicates'):
        require((pure[key] == gen[key]), 'Invalid artifact: pure[key] == gen[key]')
    require((pure['unique_neural'] == gen['pure_unique_neural']), "Invalid artifact: pure['unique_neural'] == gen['pure_unique_neural']")
    pure_programs = verify_ranking(pure['ranking'], checkpoint, context, vocabulary)
    gp_programs = verify_ranking(gp['ranking'], checkpoint, context, vocabulary)
    require((gp['schema'] == 'parallel.gp_population_ranking.v1' and gp['protocol'] == ASSEMBLY_PROTOCOL), "Invalid artifact: gp['schema'] == 'parallel.gp_population_ranking.v1' and gp['protocol'] == ASSEMBLY_PROTOCOL")
    require((gp['teacher_population_consulted'] is True and gp['training_context_only_end_to_end'] is False), "Invalid artifact: gp['teacher_population_consulted'] is True and gp['training_context_only_end_to_end'] is False")
    require((gp['pure_ranking_sha256'] == gen['pure_ranking_sha256']), "Invalid artifact: gp['pure_ranking_sha256'] == gen['pure_ranking_sha256']")
    require((gp['population_assembly_sha256'] == gen['gp_population_assembly_sha256']), "Invalid artifact: gp['population_assembly_sha256'] == gen['gp_population_assembly_sha256']")
    require((gp['champion_snapshot_sha256'] == snapshot_sha), "Invalid artifact: gp['champion_snapshot_sha256'] == snapshot_sha")
    require((assembly['schema'] == 'parallel.gp_population_assembly.v1' and assembly['protocol'] == ASSEMBLY_PROTOCOL), "Invalid artifact: assembly['schema'] == 'parallel.gp_population_assembly.v1' and assembly['protocol'] == ASSEMBLY_PROTOCOL")
    require((assembly['pure_ranking_sha256'] == gen['pure_ranking_sha256']), "Invalid artifact: assembly['pure_ranking_sha256'] == gen['pure_ranking_sha256']")
    require((assembly['champion_snapshot_sha256'] == snapshot_sha), "Invalid artifact: assembly['champion_snapshot_sha256'] == snapshot_sha")
    require((assembly['checkpoint_sha256'] == checkpoint and assembly['conditioning_context_sha256'] == context), "Invalid artifact: assembly['checkpoint_sha256'] == checkpoint and assembly['conditioning_context_sha256'] == context")
    require((assembly['pure_programs_sha256'] == compact_digest(pure_programs)), "Invalid artifact: assembly['pure_programs_sha256'] == compact_digest(pure_programs)")
    candidates = seeds['candidates']
    require((len(candidates) == len({c['genome'] for c in candidates}) == 80), "Invalid artifact: len(candidates) == len({c['genome'] for c in candidates}) == 80")
    require((gp_programs == [c['genome'] for c in candidates[:64]]), "Invalid artifact: gp_programs == [c['genome'] for c in candidates[:64]]")
    tail = [dict(genome=c['genome'], origin=c['origin']) for c in candidates[64:]]
    require((assembly['retained_tail_sha256'] == compact_digest(tail)), "Invalid artifact: assembly['retained_tail_sha256'] == compact_digest(tail)")
    require((all(c['origin'] in ('incumbent_cut_teacher', 'retained_cut_teacher', 'retained_slot_fallback') for c in tail[:10])), "Invalid artifact: all(c['origin'] in ('incumbent_cut_teacher', 'retained_cut_teacher', 'retained_slot_fallback') for c in tail[:10])")
    require((all(c['origin'] == 'random_exploration' for c in tail[10:])), "Invalid artifact: all(c['origin'] == 'random_exploration' for c in tail[10:])")
    require((assembly['overlaps'] == gen['teacher_overlaps'] and assembly['refills'] == gen['overlap_refills']), "Invalid artifact: assembly['overlaps'] == gen['teacher_overlaps'] and assembly['refills'] == gen['overlap_refills']")
    overlap_programs = set()
    for overlap in assembly['overlaps']:
        require((0 <= overlap['pure_rank'] < 64 and 64 <= overlap['retained_slot'] < 80), "Invalid artifact: 0 <= overlap['pure_rank'] < 64 and 64 <= overlap['retained_slot'] < 80")
        program = pure_programs[overlap['pure_rank']]
        retained = candidates[overlap['retained_slot']]
        require((program == retained['genome'] and overlap['retained_origin'] == retained['origin']), "Invalid artifact: program == retained['genome'] and overlap['retained_origin'] == retained['origin']")
        require((overlap['program_sha256'] == digest(program.encode()) and program not in overlap_programs), "Invalid artifact: overlap['program_sha256'] == digest(program.encode()) and program not in overlap_programs")
        overlap_programs.add(program)
    require((overlap_programs == set(pure_programs) & {c['genome'] for c in tail}), "Invalid artifact: overlap_programs == set(pure_programs) & {c['genome'] for c in tail}")
    refills = assembly['refills']
    require((len(refills) == len(overlap_programs) == origins['neural_overlap_fallback']), "Invalid artifact: len(refills) == len(overlap_programs) == origins['neural_overlap_fallback']")
    refill_hashes = {r['program_sha256'] for r in refills}
    require((len(refill_hashes) == len(refills)), 'Invalid artifact: len(refill_hashes) == len(refills)')
    require((all(r['origin'] == 'neural_overlap_fallback' and isinstance(r['generator_seed'], int)
               and 0 <= r['generator_seed'] < 2**64 for r in refills)), "Invalid artifact: all(r['origin'] == 'neural_overlap_fallback' and isinstance(r['generator_seed'], int)                and 0 <= r['generator_seed'] < 2**64 for r in refills)")
    require((refill_hashes == {digest(c['genome'].encode()) for c in candidates[:64] if c['origin'] == 'neural_overlap_fallback'}), "Invalid artifact: refill_hashes == {digest(c['genome'].encode()) for c in candidates[:64] if c['origin'] == 'neural_overlap_fallback'}")
    require((set(pure_programs)-set(gp_programs) == overlap_programs), 'Invalid artifact: set(pure_programs)-set(gp_programs) == overlap_programs')
    require(({digest(p.encode()) for p in set(gp_programs)-set(pure_programs)} == refill_hashes), 'Invalid artifact: {digest(p.encode()) for p in set(gp_programs)-set(pure_programs)} == refill_hashes')
    require((best['schema'] == 'parallel.pure_best_predicted_program.v1' and best['protocol'] == PURE_PROTOCOL), "Invalid artifact: best['schema'] == 'parallel.pure_best_predicted_program.v1' and best['protocol'] == PURE_PROTOCOL")
    require((best['pure_ranking_sha256'] == gen['pure_ranking_sha256']), "Invalid artifact: best['pure_ranking_sha256'] == gen['pure_ranking_sha256']")
    require((best['dataset'] == first['dataset'] and best['cut'] == job['cut']), "Invalid artifact: best['dataset'] == first['dataset'] and best['cut'] == job['cut']")
    require((best['prepared_sha256'] == job['prepared_sha256']), "Invalid artifact: best['prepared_sha256'] == job['prepared_sha256']")
    require((best['checkpoint_sha256'] == checkpoint and best['conditioning_context_sha256'] == context), "Invalid artifact: best['checkpoint_sha256'] == checkpoint and best['conditioning_context_sha256'] == context")
    require((best['vocabulary_sha256'] == vocabulary and best['training_context_sha256'] == seeds['context_sha256']), "Invalid artifact: best['vocabulary_sha256'] == vocabulary and best['training_context_sha256'] == seeds['context_sha256']")
    require((best['context_mode'] == gen['context_mode'] and best['selection_id'] == SELECTION_ID), "Invalid artifact: best['context_mode'] == gen['context_mode'] and best['selection_id'] == SELECTION_ID")
    require((best['prediction'] == pure['ranking']['ranked'][0]), "Invalid artifact: best['prediction'] == pure['ranking']['ranked'][0]")
    require((best['training_context_only_end_to_end'] is True and best['teacher_population_consulted'] is False), "Invalid artifact: best['training_context_only_end_to_end'] is True and best['teacher_population_consulted'] is False")
    require((all(best[k] is False for k in ('gp_scores_used_for_prediction', 'heldout_targets_used_for_prediction', 'final_tests_opened'))), "Invalid artifact: all(best[k] is False for k in ('gp_scores_used_for_prediction', 'heldout_targets_used_for_prediction', 'final_tests_opened'))")
    counts = {k: gen[k] for k in ('attempted', 'valid', 'unique_neural', 'pure_unique_neural', 'within_neural_duplicates')}
    counts.update(teacher_overlaps=len(overlap_programs), overlap_refills=len(refills))
    return counts, 'pure_prediction_then_adaptive_gp_assembly'


def finite_positive(value):
    return value is not None and math.isfinite(value) and value > 0


def best_prefix(observations):
    best = [None]
    cumulative = []
    semantic = set()
    elapsed = 0.0
    failures = 0
    for i, row in enumerate(observations):
        require((row['evaluation_index'] == i), 'Non-contiguous observation ledger')
        runtime = row['runtime_seconds']
        require((math.isfinite(runtime) and runtime >= 0), 'Invalid artifact: math.isfinite(runtime) and runtime >= 0')
        elapsed += runtime
        cumulative.append(elapsed)
        value = best[-1]
        if row['failure'] is not None or row['metrics'] is None:
            failures += 1
        else:
            metrics = row['metrics']
            score = metrics['rmse'] * row['target_scale']
            require((math.isfinite(score) and score >= 0 and row['target_scale'] > 0), "Invalid artifact: math.isfinite(score) and score >= 0 and row['target_scale'] > 0")
            # Reproduce the GP's semantic deduplication: the first observation
            # for a prediction signature enters the retained population.
            key = metrics['semantic_sha256']
            if key not in semantic:
                semantic.add(key)
                value = score if value is None else min(value, score)
        best.append(value)
    return best, cumulative, failures



def search_timing(summary, observations, cumulative):
    """Qualify exact active-search clocks; keep older timer bounds separately."""
    wall = summary['wall_seconds']
    require((math.isfinite(wall) and wall >= 0), 'Invalid artifact: math.isfinite(wall) and wall >= 0')
    slack = wall-cumulative[-1]
    require((slack >= -1e-6), 'Candidate durations exceed search wall time; timing bounds invalid')
    clock = summary.get('search_clock') or {}
    completion = []
    previous = 0.0
    for row, runtime_sum in zip(observations, cumulative):
        value = row.get('completed_search_seconds')
        if value is None:
            continue
        require((row.get('search_clock') == 'parallel.gp_search_clock.v1'), 'Unknown observation search clock')
        require((math.isfinite(value) and previous <= value <= wall+1e-6), 'Invalid cumulative completion timestamp')
        require((value+1e-6 >= runtime_sum), 'Completion timestamp precedes completed candidate runtime')
        previous = value
        completion.append(value)
    reasons = []
    if clock.get('schema') != 'parallel.gp_search_clock.v1':
        reasons.append('missing_or_unknown_summary_clock')
    if clock.get('exact_wall_prefix_available') is not True or clock.get('accuracy') != 'exact_active_elapsed':
        reasons.append('summary_does_not_qualify_exact_active_elapsed')
    if summary.get('resumed') or any(clock.get(k) != 0 for k in ('resume_offset_seconds', 'resume_unobserved_allowance_seconds', 'legacy_observations_without_clock')):
        reasons.append('resumed_legacy_or_unqualified_offset')
    if len(completion) != len(observations) or any(row.get('search_clock_accuracy') != 'exact_active_elapsed' for row in observations):
        reasons.append('incomplete_or_nonexact_observation_clocks')
    return dict(exact=not reasons, completion=completion if not reasons else None,
                slack=max(0, slack), mode='exact_active_elapsed' if not reasons else 'conservative_duration_bounds',
                fallback_reasons=reasons, recorded_clock=clock)


def wall_prefix(info, budget):
    timing = info['timing']
    if timing['exact']:
        possible = guaranteed = bisect.bisect_right(timing['completion'], budget+1e-9)
    else:
        possible = bisect.bisect_right(info['cumulative'], budget+1e-9)
        guaranteed = bisect.bisect_right(info['cumulative'], budget-timing['slack']+1e-9)
    return dict(mode=timing['mode'], exact_prefix_available=timing['exact'],
                possibly_completed_evaluations=possible, guaranteed_completed_evaluations=guaranteed,
                optimistic_best_rmse=info['best'][possible], guaranteed_best_rmse=info['best'][guaranteed],
                unattributed_search_seconds=timing['slack'], fallback_reasons=timing['fallback_reasons'])


def family_ratios(views, budget, left, right):
    families = defaultdict(list)
    excluded = []
    for view in views:
        a, b = view[budget]['best_rmse'][left], view[budget]['best_rmse'][right]
        if finite_positive(a) and finite_positive(b):
            families[view['family']].append(math.log(a/b))
        else:
            excluded.append(dict(dataset=view['dataset'], cut=view['cut'], left=a, right=b, status='undefined_ratio_zero_or_missing_rmse'))
    per_family = {k:dict(views=len(v), log_rmse_ratio=statistics.mean(v), rmse_ratio=math.exp(statistics.mean(v))) for k, v in sorted(families.items())}
    means = [v['log_rmse_ratio'] for v in per_family.values()]
    ratio = math.exp(statistics.mean(means)) if means else None
    return dict(numerator=left, denominator=right, families=len(means), views=sum(len(v) for v in families.values()), family_balanced_rmse_ratio=ratio, relative_change_percent=(ratio-1)*100 if ratio is not None else None, family_wins=sum(v < -1e-12 for v in means), family_ties=sum(abs(v) <= 1e-12 for v in means), family_losses=sum(v > 1e-12 for v in means), per_family=per_family, excluded=excluded)


def family_wall_bounds(views, left, right):
    families = defaultdict(list)
    excluded = []
    for view in views:
        arms = view['matched_wall_bounds']['arms']
        a, b = arms[left], arms[right]
        vals = (a['optimistic_best_rmse'], a['guaranteed_best_rmse'], b['optimistic_best_rmse'], b['guaranteed_best_rmse'])
        if all(finite_positive(v) for v in vals):
            lo, hi = vals[0]/vals[3], vals[1]/vals[2]
            require((lo <= hi + 1e-10), 'Invalid artifact: lo <= hi + 1e-10')
            families[view['family']].append((math.log(lo), math.log(hi)))
        else:
            excluded.append(dict(dataset=view['dataset'], cut=view['cut'], status='no_finite_positive_guaranteed_bound'))
    rows = {k:dict(views=len(v), rmse_ratio_lower=math.exp(statistics.mean(x[0] for x in v)), rmse_ratio_upper=math.exp(statistics.mean(x[1] for x in v))) for k, v in sorted(families.items())}
    lo = math.exp(statistics.mean(math.log(v['rmse_ratio_lower']) for v in rows.values())) if rows else None
    hi = math.exp(statistics.mean(math.log(v['rmse_ratio_upper']) for v in rows.values())) if rows else None
    conclusion = 'unavailable' if lo is None else 'numerator_lower_within_timing_bounds' if hi < 1 else 'denominator_lower_within_timing_bounds' if lo > 1 else 'timing_bounds_overlap_equal_rmse'
    return dict(numerator=left, denominator=right, families=len(rows), rmse_ratio_lower=lo, rmse_ratio_upper=hi, descriptive_bound_result=conclusion, per_family=rows, excluded=excluded)


def describe(values):
    return dict(sum=sum(values), minimum=min(values), median=statistics.median(values), maximum=max(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cycle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contexts', type=Path, required=True, help='Exact context index bound by this cycle.json')
    a = p.parse_args()
    root = a.root.resolve()
    a.cycle, a.output, a.contexts = (path.resolve() for path in (a.cycle, a.output, a.contexts))
    require((all(path.is_relative_to(root) for path in (a.cycle, a.output, a.contexts))), 'Invalid artifact: all(path.is_relative_to(root) for path in (a.cycle, a.output, a.contexts))')
    require((not a.output.exists()), 'Use a new report version; original artifacts are immutable')
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(root))] = dict(sha256=digest(raw), bytes=len(raw))
        return json.loads(raw)

    complete = read(a.cycle/'complete.json')
    require((complete['dl_gp_dl_complete'] and complete['final_tests_opened'] is False), "Invalid artifact: complete['dl_gp_dl_complete'] and complete['final_tests_opened'] is False")
    comparisons = read(a.cycle/'comparisons.json')
    require((sources[str((a.cycle/'comparisons.json').relative_to(root))]['sha256'] == complete['comparisons_sha256']), "Invalid artifact: sources[str((a.cycle/'comparisons.json').relative_to(root))]['sha256'] == complete['comparisons_sha256']")
    require((comparisons['evaluation_bias'] == 'adaptive_cut_heldout_teacher'), "Invalid artifact: comparisons['evaluation_bias'] == 'adaptive_cut_heldout_teacher'")
    reference_views = {(v['dataset'], v['cut']):v for v in comparisons['comparisons']}
    jobs = {arm:read(a.cycle/arm/'jobs.json')['jobs'] for arm in ARMS}
    cycle_config = read(a.cycle/'cycle.json')
    contexts = read(a.contexts)
    require((sources[str(a.contexts.relative_to(root))]['sha256'] == cycle_config['contexts_sha256']), 'Context index does not match the frozen cycle')
    families_by_context = defaultdict(set)
    for row in contexts['encodings']:
        families_by_context[row['sha256']].add(row['family'])
    indexed = {arm:{} for arm in ARMS}
    generations = {arm:Counter() for arm in ('neural', 'shuffled')}
    proposal_scopes = {arm:Counter() for arm in generations}
    summaries = {arm:[] for arm in ARMS}
    evaluator_hashes = set()
    original_biases = Counter()
    for arm in ARMS:
        for job in jobs[arm]:
            path = root/job['output']
            require((path.is_relative_to(a.cycle/arm)), 'Arm path mismatch')
            summary = read(path/'search.json')
            observations = read(path/'observations.json')
            seeds = read(root/job['seeds'])
            require((sources[job['seeds']]['sha256'] == job['seeds_sha256'] == summary['seeds_sha256']), "Invalid artifact: sources[job['seeds']]['sha256'] == job['seeds_sha256'] == summary['seeds_sha256']")
            require((len(observations) == summary['completed_evaluations'] and observations), "Invalid artifact: len(observations) == summary['completed_evaluations'] and observations")
            key = (observations[0]['dataset'], job['cut'])
            require((key not in indexed[arm]), 'Invalid artifact: key not in indexed[arm]')
            seed_origins = Counter(v['origin'] for v in seeds['candidates'])
            require((len(seeds['candidates']) == 80), "Invalid artifact: len(seeds['candidates']) == 80")
            if arm == 'generic':
                require((seed_origins['generic_comparison'] == 64 and seeds['comparison_arm'] == 'generic_and_retained'), "Invalid artifact: seed_origins['generic_comparison'] == 64 and seeds['comparison_arm'] == 'generic_and_retained'")
            else:
                gen = read(path.parent/'generation.json')
                expected_mode = 'matching_training_cut' if arm == 'neural' else 'shuffled_between_source_families'
                require((gen['context_mode'] == expected_mode and gen['seeds_sha256'] == job['seeds_sha256']), "Invalid artifact: gen['context_mode'] == expected_mode and gen['seeds_sha256'] == job['seeds_sha256']")
                if arm == 'shuffled':
                    donor_families = families_by_context[gen['conditioning_context_sha256']]
                    require((donor_families and observations[0]['family'] not in donor_families), "Invalid artifact: donor_families and observations[0]['family'] not in donor_families")
                counts, scope = verify_generation(path.parent, gen, seeds, job, observations[0], read)
                generations[arm].update(counts)
                proposal_scopes[arm][scope] += 1
            for row in observations:
                require((row['dataset'] == key[0] and row['cut'] == key[1] and row['prepared_sha256'] == job['prepared_sha256'] and row['seeds_sha256'] == job['seeds_sha256']), "Invalid artifact: row['dataset'] == key[0] and row['cut'] == key[1] and row['prepared_sha256'] == job['prepared_sha256'] and row['seeds_sha256'] == job['seeds_sha256']")
                require((digest(row['genome'].encode()) == row['program_sha256']), "Invalid artifact: digest(row['genome'].encode()) == row['program_sha256']")
                require((row['qualification'] and row['independent_evaluation'] is False), "Invalid artifact: row['qualification'] and row['independent_evaluation'] is False")
                require((row['evaluation_bias'] == 'adaptive_cut_heldout_teacher'), "Invalid artifact: row['evaluation_bias'] == 'adaptive_cut_heldout_teacher'")
                evaluator_hashes.add(row['evaluator_sha256'])
                original_biases[row['evaluation_bias']] += 1
            best, cumulative, failures = best_prefix(observations)
            actual = summary['best_validation_rmse']
            require((best[-1] is not None and math.isclose(best[-1], actual, rel_tol=1e-9, abs_tol=1e-8)), 'Reconstructed best does not match immutable search summary')
            wall = summary['wall_seconds']
            timing = search_timing(summary, observations, cumulative)
            indexed[arm][key] = dict(summary=summary, best=best, cumulative=cumulative, timing=timing, failures=failures, seeds=seeds, first=observations[0])
            summaries[arm].append(dict(evaluations=len(observations), wall_seconds=wall, candidate_seconds=cumulative[-1], unattributed_seconds=timing['slack'], timing_mode=timing['mode'], failures=failures, seed_origins=dict(seed_origins), summary_neural_seeded=summary.get('neural_seeded')))
    require((all(set(indexed[arm]) == set(reference_views) for arm in ARMS)), 'Invalid artifact: all(set(indexed[arm]) == set(reference_views) for arm in ARMS)')
    views = []
    for key, original in sorted(reference_views.items()):
        arms = {arm:indexed[arm][key] for arm in ARMS}
        tails = [[(s['genome'], s['origin']) for s in arms[arm]['seeds']['candidates'][64:]] for arm in ARMS]
        require((tails[0] == tails[1] == tails[2]), 'Retained/exploration slots differ across arms')
        for field in ['training_sha256', 'validation_sha256', 'target_scale', 'family', 'partition']:
            require((all(arms[arm]['first'][field] == arms['neural']['first'][field] for arm in ARMS)), "Invalid artifact: all(arms[arm]['first'][field] == arms['neural']['first'][field] for arm in ARMS)")
        count = min(len(v['best'])-1 for v in arms.values())
        initial = min(80, count)
        wall = min(v['summary']['wall_seconds'] for v in arms.values())
        wall_arms = {arm:wall_prefix(info, wall) for arm, info in arms.items()}
        exact = all(info['timing']['exact'] for info in arms.values())
        views.append(dict(dataset=key[0], cut=key[1], family=original['family'], partition=original['partition'], rmse_units='original_target_units', actual_completed={arm:dict(evaluations=len(v['best'])-1, wall_seconds=v['summary']['wall_seconds'], failures=v['failures'], best_rmse=v['best'][-1]) for arm,v in arms.items()}, matched_evaluations=dict(evaluations_each_arm=count, best_rmse={arm:v['best'][count] for arm,v in arms.items()}), matched_initial_population=dict(evaluations_each_arm=initial, best_rmse={arm:v['best'][initial] for arm,v in arms.items()}), matched_wall_bounds=dict(seconds_each_arm=wall, exact_prefix_available=exact, arms=wall_arms), matched_wall_exact=dict(available=exact, best_rmse={arm:info['optimistic_best_rmse'] for arm,info in wall_arms.items()} if exact else {})))
    grouped = {}
    for partition in sorted({v['partition'] for v in views}):
        selected = [v for v in views if v['partition'] == partition]
        exact_views = [v for v in selected if v['matched_wall_exact']['available']]
        grouped[partition] = dict(views=len(selected), source_families=len({v['family'] for v in selected}), evaluation_prefix={f'{l}_over_{r}':family_ratios(selected,'matched_evaluations',l,r) for l,r in PAIRS}, initial_population_prefix={f'{l}_over_{r}':family_ratios(selected,'matched_initial_population',l,r) for l,r in PAIRS}, wall_prefix_bounds={f'{l}_over_{r}':family_wall_bounds(selected,l,r) for l,r in PAIRS}, exact_wall_prefix=dict(views=len(exact_views), source_families=len({v['family'] for v in exact_views}), excluded_views=len(selected)-len(exact_views), comparisons={f'{l}_over_{r}':family_ratios(exact_views,'matched_wall_exact',l,r) for l,r in PAIRS}))
    budgets = {arm:dict(searches=len(v), timing_modes=dict(Counter(x['timing_mode'] for x in v)), evaluations=describe([x['evaluations'] for x in v]), search_wall_seconds=describe([x['wall_seconds'] for x in v]), candidate_seconds=describe([x['candidate_seconds'] for x in v]), unattributed_seconds=describe([x['unattributed_seconds'] for x in v]), failed_candidates=sum(x['failures'] for x in v), failure_rate=sum(x['failures'] for x in v)/sum(x['evaluations'] for x in v), evaluations_per_worker_second=sum(x['evaluations'] for x in v)/sum(x['wall_seconds'] for x in v), mislabeled_neural_seeded_summaries=sum(arm=='generic' and x['summary_neural_seeded'] is True for x in v)) for arm,v in summaries.items()}
    corpus = read(a.cycle/'corpus-2/index.json')
    require((sources[str((a.cycle/'corpus-2/index.json').relative_to(root))]['sha256'] == complete['teaching_corpus_sha256']), "Invalid artifact: sources[str((a.cycle/'corpus-2/index.json').relative_to(root))]['sha256'] == complete['teaching_corpus_sha256']")
    exports = read(a.cycle/'predictor-exports.json')
    require((sources[str((a.cycle/'predictor-exports.json').relative_to(root))]['sha256'] == complete['predictor_exports_sha256']), "Invalid artifact: sources[str((a.cycle/'predictor-exports.json').relative_to(root))]['sha256'] == complete['predictor_exports_sha256']")
    train_receipts = {arm:read(a.cycle/arm/'complete.json') for arm in ['train-1', 'train-2']}
    exact_searches = sum(x['timing_mode'] == 'exact_active_elapsed' for rows in summaries.values() for x in rows)
    exact_views = sum(v['matched_wall_exact']['available'] for v in views)
    validation_families = grouped.get('meta_validation', {}).get('source_families', 0)
    mislabeled = sum(v['mislabeled_neural_seeded_summaries'] for v in budgets.values())
    limitations = [
        'All scores are adaptive cut-held-out teaching objectives repeatedly optimized by GP. They are not independent test scores; original final tests were not opened.',
        f'Source-family grouping remains qualification-scoped, with {validation_families} meta-validation families. These descriptive comparisons do not establish generalization or statistical significance.',
        f'Exact active-search prefixes qualify for {exact_searches} searches and {exact_views} complete three-arm views. Exact-only aggregates describe that subset, which may differ from the full comparison cohort.',
        'Older or resumed searches retain duration-based conservative bounds. Resume clocks include an allowance for interrupted unobserved work and exclude process downtime; they are not exact elapsed wall clocks.',
        'Recorded search wall time excludes frozen encoding, neural proposal generation, and preparation before the search timer. This is not an end-to-end cost comparison.',
    ]
    if mislabeled:
        limitations.append(f'{mislabeled} generic summaries incorrectly mark neural_seeded=true. Report arms use frozen seed origins, generation receipts, and paths; original summaries remain unchanged.')
    report = dict(schema='parallel.measured_cycle_report.v2' if any(v['pure_prediction_then_adaptive_gp_assembly'] for v in proposal_scopes.values()) else 'parallel.measured_cycle_report.v1', cycle=str(a.cycle.relative_to(root)), evaluation_bias='adaptive_cut_heldout_teacher', qualification=True, independent_evaluation=False, final_tests_opened=False, pilot_started=complete.get('pilot_started', False), dl_gp_dl_complete=True, teaching_records=corpus['records'], fitted_predictors_passed=sum(x['passed'] is True for x in exports['exports']), fitted_predictors_failed=sum(x['passed'] is not True for x in exports['exports']), training=train_receipts, comparison_views=len(views), comparison_searches=sum(len(x) for x in indexed.values()), source_families=len({v['family'] for v in views}), evaluator_sha256=sorted(evaluator_hashes), arm_identification='Immutable arm paths + seed origins + generation context_mode; never summary.neural_seeded', shared_retained_and_exploration_candidates_verified=True, matched_evaluation_budget=describe([v['matched_evaluations']['evaluations_each_arm'] for v in views]), matched_wall_budget_seconds=describe([v['matched_wall_bounds']['seconds_each_arm'] for v in views]), aggregation='Mean paired log(RMSE numerator / RMSE denominator) within each source family, then equal-weight mean across families; exponentiate. No averaging RMSE across incompatible target units.', contexts=str(a.contexts.relative_to(root)), contexts_sha256=cycle_config['contexts_sha256'], clock_coverage=dict(exact_searches=exact_searches, bounded_searches=sum(len(x) for x in summaries.values())-exact_searches, all_arms_exact_views=exact_views, mixed_or_bounded_views=len(views)-exact_views), exact_wall_derivation='Only when summary and every observation qualify exact_active_elapsed with no resume or legacy offset: completed_search_seconds <= common budget determines the exact prefix. Exact arms become degenerate intervals in the full-cohort bounds.', wall_bound_derivation='For candidate i, C_i=sum(runtime_seconds through i), S=search.wall_seconds-C_last. Without exact clocks, recorded sequential-search completion is bounded by [C_i,C_i+S], treating timer overhead and conservative resume allowances as unattributed. At budget B=min(arm search wall_seconds), C_i<=B defines the optimistic prefix; C_i+S<=B defines the guaranteed prefix.', budgets=budgets, generation={k:dict(v) for k,v in generations.items()}, partitions=grouped, views=views, limitations=limitations)
    legacy_views = sum(v['legacy_filtered_pool'] for v in proposal_scopes.values())
    pure_views = sum(v['pure_prediction_then_adaptive_gp_assembly'] for v in proposal_scopes.values())
    report['proposal_input_scope'] = dict(
        arm_views={arm:dict(counts) for arm,counts in proposal_scopes.items()},
        verified_pure_prediction_views=pure_views,
        legacy_filtered_pool_views=legacy_views,
        gp_population_training_only_end_to_end=False,
        fixed_pool_scoring_scope='Frozen model scoring uses the declared saved training context; the GP candidate pool also depends on adaptive teachers.',
        legacy_duplicates_scope='Mixed within-neural and retained-population collisions; teacher-overlap frequency unavailable.',
        v2_duplicates_scope='within_neural_duplicates excludes teachers; teacher_overlaps and overlap_refills count the separate GP assembly.',
        pure_prediction_scope='For the same frozen checkpoint, vocabulary, saved context and RNG, the v2 pure pool and prediction exclude current teacher identities and scores.')
    if legacy_views:
        limitations.append(f'{legacy_views} legacy proposal pools use a mixed duplicate counter. Their fixed-pool model scores use training encodings, but end-to-end best-prediction independence from heldout-selected teacher identities is not established; teacher-overlap frequency is unavailable.')
        amendment_path = root/'runs/acceptance/proposal-training-input-scope-v44.json'
        if amendment_path.exists():
            amendment = read(amendment_path)
            require((amendment['schema'] == 'parallel.proposal_training_input_scope_amendment.v1'), "Invalid artifact: amendment['schema'] == 'parallel.proposal_training_input_scope_amendment.v1'")
            report['proposal_input_scope']['v44_scope_amendment'] = dict(path=str(amendment_path.relative_to(root)), sha256=digest(amendment_path.read_bytes()))
    if pure_views:
        limitations.append(f'{pure_views} v2 pure predictions are separately verified before teacher-based GP assembly. Overlap removal/refills can change the GP population ranking; that ranking is not the pure model prediction.')
    a.output.mkdir(parents=True)
    inputs = dict(schema='parallel.report_inputs.v1', source_script_sha256=digest(Path(__file__).read_bytes()), files=sources)
    (a.output/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    report['inputs_sha256'] = digest((a.output/'inputs.json').read_bytes())
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = [
        f'# {a.cycle.name}: measured teaching results', '',
        f'The DL → GP → DL cycle completed: {report["comparison_searches"]} searches across {len(views)} cut views, {report["fitted_predictors_passed"]} fitted predictors, and {corpus["records"]:,} records in the second teaching corpus. Neural blocks completed ' + ' and '.join(f'{v["steps"]} steps on {v["world"]} GPUs' for v in train_receipts.values()) + '.', '',
        '**These are adaptive teaching scores.** GP repeatedly optimized each cut’s held-out targets. Original final tests remain sealed. This report describes the qualification cohort recorded by the cycle receipts.', '',
        '## Matched evaluation counts', '',
        'Each view uses the minimum completed evaluation count across its three arms. Failed candidates consume budget. Scores are best observed RMSE in original target units; ratios average log differences within each family, then give each family equal weight. A ratio below 1 means the numerator had lower teaching RMSE.', '',
        '| Partition | Families / views | Neural ÷ generic | Neural ÷ shuffled | Shuffled ÷ generic |', '|---|---:|---:|---:|---:|',
    ]
    fmt = lambda n: 'undefined' if n is None else f'{n:.4f}'
    for partition, results in grouped.items():
        ratios = [results['evaluation_prefix'][f'{l}_over_{r}']['family_balanced_rmse_ratio'] for l,r in PAIRS]
        lines.append(f'| {partition} | {results["source_families"]} / {results["views"]} | '+ ' | '.join(fmt(x) for x in ratios)+' |')
    lines += ['', f'Per-view matched counts ranged from {report["matched_evaluation_budget"]["minimum"]} to {report["matched_evaluation_budget"]["maximum"]}; median {report["matched_evaluation_budget"]["median"]:.0f}. Initial-population comparisons at min(80, matched count), per-family results, and every view’s raw metrics are in [report.json](report.json).', '', f'## Meta-validation: {validation_families} source families', '', '| Family | Views | Neural ÷ generic | Neural ÷ shuffled | Shuffled ÷ generic |', '|---|---:|---:|---:|---:|']
    validation = grouped.get('meta_validation', {})
    if validation:
        for family in sorted(validation['evaluation_prefix']['neural_over_generic']['per_family']):
            counts = validation['evaluation_prefix']['neural_over_generic']['per_family'][family]['views']
            vals = [validation['evaluation_prefix'][f'{l}_over_{r}']['per_family'].get(family,{}).get('rmse_ratio') for l,r in PAIRS]
            lines.append(f'| {family} | {counts} | '+' | '.join(fmt(v) for v in vals)+' |')
    lines += ['', 'These few family-level observations describe this teaching run. They do not establish a reliable model advantage or generalization.', '', '## Matched search-wall budget', '', f'Qualified cumulative completion timestamps provide exact prefixes for {exact_searches} searches and {exact_views} complete three-arm views. Each view uses the minimum recorded search duration across arms. The full-cohort intervals below use exact prefixes where qualified and conservative duration-based bounds for older or resumed ledgers. Resumed timers contain allowances for interrupted work and exclude downtime.', '', '| Partition | Neural ÷ generic interval | Neural ÷ shuffled interval | Shuffled ÷ generic interval |', '|---|---:|---:|---:|']
    for partition, results in grouped.items():
        intervals = [results['wall_prefix_bounds'][f'{l}_over_{r}'] for l,r in PAIRS]
        lines.append(f'| {partition} | '+' | '.join(f'[{fmt(v["rmse_ratio_lower"])}, {fmt(v["rmse_ratio_upper"])}]' for v in intervals)+' |')
    lines += ['', 'These timers cover GP search after input preparation and initial population construction. Encoding and neural-proposal generation costs are excluded. Exact-only results and their smaller cohort coverage are provided separately in report.json; they must not be substituted for the full cohort.', '', '## Actual completed work', '', '| Arm | Evaluations | Per-view min / median / max | Candidate failures | Sum of worker search seconds |', '|---|---:|---:|---:|---:|']
    for arm, budget in budgets.items():
        e=budget['evaluations']
        lines.append(f'| {arm} | {e["sum"]:,} | {e["minimum"]} / {e["median"]:.0f} / {e["maximum"]} | {budget["failed_candidates"]:,} | {budget["search_wall_seconds"]["sum"]:,.1f} |')
    lines += ['', 'All arms shared the same ten retained and six random-exploration candidates for each cut. The other 64 slots differed by arm. Arm identities use frozen seed origins, context receipts, and paths. Original search summaries were preserved.', '', 'The machine also ran the background GP corpus campaign. Summed worker seconds are aggregate search work, not campaign elapsed time.', '', '## Reproducibility', '', 'Every consumed ledger, seed file, generation receipt, job manifest, and completion receipt is SHA-256 bound in [inputs.json](inputs.json). [report.json](report.json) contains the full calculations and limitations. This report reads saved supervision and metadata only; it does not open source data or final-test targets.', '']
    lines += ['## Proposal input scope', '', report['proposal_input_scope']['fixed_pool_scoring_scope'], '']
    if legacy_views:
        lines += [f'{legacy_views} legacy proposal pools retain their original mixed duplicate counts. Their best predictions are conditional on teacher-filtered pools; the actual teacher-overlap frequency is unknown.', '']
    if pure_views:
        lines += [f'{pure_views} pure prediction receipts were verified separately from GP assembly. Pure rankings and best predictions use the declared training context and frozen model; overlap/refill counts describe only assembly of the GP population.', '']
    (a.output/'report.md').write_text('\n'.join(lines))
    (a.output/'receipt.json').write_text(json.dumps(dict(schema='parallel.report_receipt.v1',passed=True,report_sha256=digest((a.output/'report.json').read_bytes()),markdown_sha256=digest((a.output/'report.md').read_bytes()),inputs_sha256=report['inputs_sha256'],source_files=len(sources),original_artifacts_modified=False),indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['comparison_views','comparison_searches','source_families','teaching_records','fitted_predictors_passed','budgets','partitions']},indent=2))


if __name__ == '__main__':
    main()
