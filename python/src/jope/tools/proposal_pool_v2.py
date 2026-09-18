"""Versioned saved v2 proposal verification; no GP evaluation or data targets read."""
import hashlib
import json
import math
from collections import Counter


def digest(data):
    return hashlib.sha256(data).hexdigest()

PURE_PROTOCOL = 'training_context_neural_pool_v1'
ASSEMBLY_PROTOCOL = 'preserve_shared_tail_dedup_v1'
SELECTION_ID = 'mean_reward_heads_desc_program_hash_v1'


def compact_digest(value):
    # Matches serde_json::to_vec for these string lists / SeedCandidate structs.
    return digest(json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())


def verify_ranking(ranking, checkpoint, context, vocabulary):
    if not (ranking['schema'] == 'parallel.proposal_quality_ranking.v1'):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 21')
    if not (ranking['selection_id'] == SELECTION_ID):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 22')
    if not (ranking['bindings'] == dict(checkpoint_sha256=checkpoint,
                                      conditioning_context_sha256=context,
                                      vocabulary_sha256=vocabulary)):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 23')
    if ranking['training_context_only'] is not True:
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 26')
    if not (ranking['heldout_labels_read'] is False and ranking['gp_scores_read'] is False):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 27')
    rows = ranking['ranked']
    if not (ranking['pool_size'] == ranking['chosen_count'] == len(rows) == 64):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 29')
    if not (len({r['program'] for r in rows}) == 64):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 30')
    for row in rows:
        if not (digest(row['program'].encode()) == row['program_sha256']):
            raise ValueError('proposal_pool_v2: artifact validation failed at original line 32')
        if not (math.isfinite(row['predicted_quality'])):
            raise ValueError('proposal_pool_v2: artifact validation failed at original line 33')
        if not (math.isfinite(row['predicted_runtime_seconds']) and row['predicted_runtime_seconds'] >= 0):
            raise ValueError('proposal_pool_v2: artifact validation failed at original line 34')
        if not (len(row['reward_heads']) == 4 and all(math.isfinite(x) for x in row['reward_heads'])):
            raise ValueError('proposal_pool_v2: artifact validation failed at original line 35')
    if not (rows == sorted(rows, key=lambda r: (-r['predicted_quality'], r['program_sha256'], r['program']))):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 36')
    if not (ranking['pool_sha256'] == compact_digest(sorted(r['program_sha256'] for r in rows))):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 37')
    return [r['program'] for r in rows]


def verify_generation(directory, gen, seeds, job, first, read):
    """Verify saved proposal provenance without reading source data or targets.

    A ranking's training_context_only flag qualifies scoring a fixed pool.
    Only the separate v2 pure receipt establishes pool independence from the
    current retained teachers; no GP population receives that claim.
    """
    origins = Counter(c['origin'] for c in seeds['candidates'][:64])
    schema = gen['schema']
    if not (gen['seeds_sha256'] == job['seeds_sha256']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 50')
    if schema == 'parallel.proposal_generation.v1':
        if not (origins['neural_policy'] + origins['neural_rejection_fallback'] == 64):
            raise ValueError('proposal_pool_v2: artifact validation failed at original line 52')
        # Preserve legacy numerical counters exactly. Its duplicate field mixes
        # within-model duplicates and heldout-selected population collisions.
        return {k: gen[k] for k in ('attempted', 'valid', 'duplicates', 'unique_neural')}, 'legacy_filtered_pool'
    if not (schema == 'parallel.proposal_generation.v2'):
        raise ValueError('Unknown generation protocol')
    if not (gen['pure_prediction_protocol'] == PURE_PROTOCOL and gen['gp_assembly_protocol'] == ASSEMBLY_PROTOCOL):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 57')
    if not (origins['neural_policy'] + origins['neural_rejection_fallback'] + origins['neural_overlap_fallback'] == 64):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 58')
    if not (gen['unique_neural'] == origins['neural_policy']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 59')
    if not ('duplicates' not in gen):
        raise ValueError('v2 cannot relabel mixed legacy duplicates')
    for field in ('attempted', 'valid', 'unique_neural', 'pure_unique_neural', 'within_neural_duplicates'):
        if not (isinstance(gen[field], int) and gen[field] >= 0):
            raise ValueError('proposal_pool_v2: artifact validation failed at original line 62')
    if not (gen['within_neural_duplicates'] <= gen['valid'] <= gen['attempted']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 63')
    if not (gen['pure_unique_neural'] <= 64 and gen['filled_population'] == 80):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 64')
    if gen['final_tests_opened'] is not False:
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 65')
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
        if not (digest(path.read_bytes()) == gen[key]):
            raise ValueError(f'Proposal artifact hash mismatch: {name}')
    pure, best, gp, assembly = (artifacts[name] for name in expected.values())
    snapshot_path = directory.parents[2]/'champion-snapshot.json'
    snapshot = read(snapshot_path)
    snapshot_sha = digest(snapshot_path.read_bytes())
    if not (snapshot_sha == gen['champion_snapshot_sha256']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 81')
    if not (snapshot['schema'] == 'parallel.champion_snapshot.v1'):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 82')
    checkpoint, context = gen['checkpoint_sha256'], gen['conditioning_context_sha256']
    vocabulary = snapshot['vocabulary_sha256']
    if not (snapshot['request']['checkpoint_sha256'] == checkpoint == seeds['checkpoint_sha256']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 85')
    if not (snapshot['request']['vocabulary_sha256'] == vocabulary == first['vocabulary_sha256']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 86')
    if not (gen['selection_id'] == SELECTION_ID):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 87')
    if gen['context_mode'] == 'matching_training_cut':
        if not (context == seeds['context_sha256']):
            raise ValueError('proposal_pool_v2: artifact validation failed at original line 89')
    else:
        if not (gen['context_mode'] == 'shuffled_between_source_families'):
            raise ValueError('proposal_pool_v2: artifact validation failed at original line 91')
    if not (seeds['prepared_sha256'] == job['prepared_sha256'] == first['prepared_sha256']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 92')
    if not (seeds['cut'] == job['cut'] == first['cut']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 93')
    if not (seeds['training_sha256'] == first['training_sha256']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 94')
    if not (pure['schema'] == 'parallel.pure_neural_ranking.v1' and pure['protocol'] == PURE_PROTOCOL):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 95')
    if pure['teacher_population_consulted'] is not False:
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 96')
    if not (pure['sampling_seed'] == gen['seed']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 97')
    for key in ('attempted', 'valid', 'within_neural_duplicates'):
        if not (pure[key] == gen[key]):
            raise ValueError('proposal_pool_v2: artifact validation failed at original line 99')
    if not (pure['unique_neural'] == gen['pure_unique_neural']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 100')
    pure_programs = verify_ranking(pure['ranking'], checkpoint, context, vocabulary)
    gp_programs = verify_ranking(gp['ranking'], checkpoint, context, vocabulary)
    if not (gp['schema'] == 'parallel.gp_population_ranking.v1' and gp['protocol'] == ASSEMBLY_PROTOCOL):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 103')
    if not (gp['teacher_population_consulted'] is True and gp['training_context_only_end_to_end'] is False):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 104')
    if not (gp['pure_ranking_sha256'] == gen['pure_ranking_sha256']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 105')
    if not (gp['population_assembly_sha256'] == gen['gp_population_assembly_sha256']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 106')
    if not (gp['champion_snapshot_sha256'] == snapshot_sha):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 107')
    if not (assembly['schema'] == 'parallel.gp_population_assembly.v1' and assembly['protocol'] == ASSEMBLY_PROTOCOL):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 108')
    if not (assembly['pure_ranking_sha256'] == gen['pure_ranking_sha256']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 109')
    if not (assembly['champion_snapshot_sha256'] == snapshot_sha):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 110')
    if not (assembly['checkpoint_sha256'] == checkpoint and assembly['conditioning_context_sha256'] == context):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 111')
    if not (assembly['pure_programs_sha256'] == compact_digest(pure_programs)):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 112')
    candidates = seeds['candidates']
    if not (len(candidates) == len({c['genome'] for c in candidates}) == 80):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 114')
    if not (gp_programs == [c['genome'] for c in candidates[:64]]):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 115')
    tail = [dict(genome=c['genome'], origin=c['origin']) for c in candidates[64:]]
    if not (assembly['retained_tail_sha256'] == compact_digest(tail)):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 117')
    if not (all(c['origin'] in ('incumbent_cut_teacher', 'retained_cut_teacher', 'retained_slot_fallback') for c in tail[:10])):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 118')
    if not (all(c['origin'] == 'random_exploration' for c in tail[10:])):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 119')
    if not (assembly['overlaps'] == gen['teacher_overlaps'] and assembly['refills'] == gen['overlap_refills']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 120')
    overlap_programs = set()
    for overlap in assembly['overlaps']:
        if not (0 <= overlap['pure_rank'] < 64 and 64 <= overlap['retained_slot'] < 80):
            raise ValueError('proposal_pool_v2: artifact validation failed at original line 123')
        program = pure_programs[overlap['pure_rank']]
        retained = candidates[overlap['retained_slot']]
        if not (program == retained['genome'] and overlap['retained_origin'] == retained['origin']):
            raise ValueError('proposal_pool_v2: artifact validation failed at original line 126')
        if not (overlap['program_sha256'] == digest(program.encode()) and program not in overlap_programs):
            raise ValueError('proposal_pool_v2: artifact validation failed at original line 127')
        overlap_programs.add(program)
    if not (overlap_programs == set(pure_programs) & {c['genome'] for c in tail}):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 129')
    refills = assembly['refills']
    if not (len(refills) == len(overlap_programs) == origins['neural_overlap_fallback']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 131')
    refill_hashes = {r['program_sha256'] for r in refills}
    if not (len(refill_hashes) == len(refills)):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 133')
    if not (all(r['origin'] == 'neural_overlap_fallback' and isinstance(r['generator_seed'], int)
               and 0 <= r['generator_seed'] < 2**64 for r in refills)):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 134')
    if not (refill_hashes == {digest(c['genome'].encode()) for c in candidates[:64] if c['origin'] == 'neural_overlap_fallback'}):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 136')
    if not (set(pure_programs)-set(gp_programs) == overlap_programs):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 137')
    if not ({digest(p.encode()) for p in set(gp_programs)-set(pure_programs)} == refill_hashes):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 138')
    if not (best['schema'] == 'parallel.pure_best_predicted_program.v1' and best['protocol'] == PURE_PROTOCOL):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 139')
    if not (best['pure_ranking_sha256'] == gen['pure_ranking_sha256']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 140')
    if not (best['dataset'] == first['dataset'] and best['cut'] == job['cut']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 141')
    if not (best['prepared_sha256'] == job['prepared_sha256']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 142')
    if not (best['checkpoint_sha256'] == checkpoint and best['conditioning_context_sha256'] == context):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 143')
    if not (best['vocabulary_sha256'] == vocabulary and best['training_context_sha256'] == seeds['context_sha256']):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 144')
    if not (best['context_mode'] == gen['context_mode'] and best['selection_id'] == SELECTION_ID):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 145')
    if not (best['prediction'] == pure['ranking']['ranked'][0]):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 146')
    if not (best['training_context_only_end_to_end'] is True and best['teacher_population_consulted'] is False):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 147')
    if not (all(best[k] is False for k in ('gp_scores_used_for_prediction', 'heldout_targets_used_for_prediction', 'final_tests_opened'))):
        raise ValueError('proposal_pool_v2: artifact validation failed at original line 148')
    counts = {k: gen[k] for k in ('attempted', 'valid', 'unique_neural', 'pure_unique_neural', 'within_neural_duplicates')}
    counts.update(teacher_overlaps=len(overlap_programs), overlap_refills=len(refills))
    return counts, 'pure_prediction_then_adaptive_gp_assembly'

