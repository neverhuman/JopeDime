#!/usr/bin/env python3
"""Build the inert policy['seed_values'] block (parallel.seed_values.v1) from a seed-value export.

Consumes one immutable successor/seed-values-<generation>.json export (parallel.seed_value.v1
rows written by successor_ledger.export) and the bound teaching corpus, and resolves every seed
program to its record id through replay_policy.groups()/sharded_groups (the same map
membership.precompute uses). Value units: value(x) = 1/(1 + x/median_c) with median_c the median
of the context's labels.jsonl rmse_cut_normalized. S = min(r0, r_child) (failed child -> r0).
A_own = value(S) - value(r0) is diagnostic only and never a weight. A_ctx = value(S) - median over
the context's control_generic seeds of value(S) is the reported advantage (plan D1); it is written per
record and as per-arm medians, 'unavailable' where no control_generic seed was measured. The tie
floor eps_c comes from the replicate panel (p75 of |value(child_rep0) - value(child_rep1)| over
finite pairs, floored at 0.002); a context without a replicate panel is 'unavailable' and refused. Groups are 8 record ids
(champion record first, then 7 seeds ordered by S_value; adjacent ties excluded; groups with fewer
than 2 non-tie pairs dropped; at most 8 groups per context). Refusals: corpus/evaluator mismatch,
non-meta_train records, tie_fraction > 0.7 (per context), no usable eps anywhere.

The block is attached to policy['seed_values'] by the runtime after membership.bind and before
replay_policy.immutable; the policy schema string stays parallel.replay_policy.v2 and the deployed
trainer ignores the key. Nothing here writes labels, training state or serving state.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import replay_policy
from promotion_gate import immutable
from resume_expired_cycle import sha


SCHEMA = 'parallel.seed_values.v1'
EXPORT_SCHEMA = 'parallel.seed_value.v1'
REFUSAL_SCHEMA = 'parallel.seed_values_refusal.v1'
NEXT_OUTCOME_SCHEMA = 'parallel.successor_next_outcome.v1'
PROTOCOL = 'onestep_forced_mutation_v2'
TIE_RULE = 'abs(dS_value) <= eps_c'
ROLE_PRIORITY = ('candidate', 'control_generic', 'control_shuffled', 'gp_final')
CONTROL_ROLES = ('control_generic', 'control_shuffled')
A_CTX_CONTROL = 'control_generic'
ARMS = ('neural', 'generic', 'shuffled')
GROUP_SIZE = 8
PER_CONTEXT_CAP = 8
EPS_FLOOR = 0.002
EPS_QUANTILE = 0.75
TIE_REFUSAL = 0.7
P_FLIP_FLOOR = 0.10
RESAMPLES = 2000
BOOTSTRAP_SEED = 20260912
ALPHA = 0.10
EXACT_TIE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def value(x, median):
    """Commensurable value units: 1/(1 + x/median_c) with the bound corpus's context median."""
    require(finite(x) and x >= 0 and finite(median) and median > 0, 'value requires finite nonnegative x and positive median')
    return 1.0 / (1.0 + x / median)


def percentile(values, q):
    """Nearest-rank percentile; None on empty input."""
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def ranks(values):
    """1-based average ranks (ties share the mean rank)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return result


def spearman(a, b):
    """Spearman rank correlation with average ranks; None when either side is constant."""
    if len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 0 else None


def ols3(y, x1, x2):
    """Plain OLS y ~ 1 + x1 + x2 by the normal equations; None when singular."""
    n = len(y)
    if n < 4:
        return None
    s = dict(x1=sum(x1), x2=sum(x2), x11=sum(a * a for a in x1), x12=sum(a * b for a, b in zip(x1, x2)),
             x22=sum(b * b for b in x2), y=sum(y), x1y=sum(a * c for a, c in zip(x1, y)), x2y=sum(b * c for b, c in zip(x2, y)))
    matrix = [[n, s['x1'], s['x2'], s['y']], [s['x1'], s['x11'], s['x12'], s['x1y']], [s['x2'], s['x12'], s['x22'], s['x2y']]]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(matrix[r][col]))
        if abs(matrix[pivot][col]) < 1e-12:
            return None
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        for r in range(3):
            if r != col:
                factor = matrix[r][col] / matrix[col][col]
                matrix[r] = [a - factor * b for a, b in zip(matrix[r], matrix[col])]
    return [matrix[r][3] / matrix[r][r] for r in range(3)]


def cluster_bootstrap(clusters, statistic, resamples=RESAMPLES, seed=BOOTSTRAP_SEED, alpha=ALPHA):
    """Resample clusters with replacement; return (low, high, valid resamples) of a statistic over the pooled samples."""
    keys = sorted(clusters)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        pooled = []
        for key in rng.choices(keys, k=len(keys)):
            pooled.extend(clusters[key])
        result = statistic(pooled)
        if result is not None and math.isfinite(result):
            draws.append(result)
    if len(draws) < resamples // 2:
        return None
    draws.sort()
    low = draws[max(0, math.ceil(alpha / 2 * len(draws)) - 1)]
    high = draws[min(len(draws) - 1, math.ceil((1 - alpha / 2) * len(draws)) - 1)]
    return dict(low=low, high=high, resamples=len(draws))


def rank_regression_coefficient(samples):
    """A_own coefficient of rank(outcome) ~ 1 + rank(r0) + rank(A_own) over (outcome, r0, A_own) triples."""
    if len(samples) < 4:
        return None
    fit = ols3(ranks([s[0] for s in samples]), ranks([s[1] for s in samples]), ranks([s[2] for s in samples]))
    return None if fit is None else fit[2]


def load_next_outcome(source):
    """Map (context_key, seed_program_sha256) -> next-generation outcome; None when unsupplied."""
    if source is None:
        return None
    if isinstance(source, dict):
        require(all(isinstance(k, tuple) and len(k) == 2 and finite(v) for k, v in source.items()),
                'next outcome mapping requires (context_key, seed_program_sha256) -> finite outcome')
        return dict(source)
    document = read(source)
    rows = document['rows'] if isinstance(document, dict) else document
    if isinstance(document, dict):
        require(document.get('schema') == NEXT_OUTCOME_SCHEMA, 'unknown next-outcome schema')
    result = {}
    for row in rows:
        key = (row['context_key'], row['seed_program_sha256'])
        require(key not in result and finite(row['outcome']), 'duplicate or nonfinite next outcome row')
        result[key] = row['outcome']
    return result


def corpus_contexts(corpus):
    """(prepared_sha256, cut) -> corpus group with programs -> {record, rmse}; refuses sealed partitions."""
    index, groups = replay_policy.groups(Path(corpus))
    result = {}
    for (prepared, cut, family, partition), group in groups.items():
        entry = result.setdefault((prepared, cut), [])
        entry.append(dict(family=family, partition=partition, index=group['index'], programs=group['programs']))
    return index, result


def context_median(programs):
    scores = [p['rmse'] for p in programs.values()]
    require(scores, 'context has no labelled programs')
    return statistics.median(scores)


def eps_for(rows, median):
    """eps_c from the replicate panel; None when no finite replicate pair exists in the context."""
    spreads = []
    for row in rows:
        pair = row.get('replicate_pair')
        if row.get('role') != 'replicate_panel' or not pair:
            continue
        a, b = pair.get('child_rmse_rep0'), pair.get('child_rmse_rep1')
        if finite(a) and finite(b):
            spreads.append(abs(value(a, median) - value(b, median)))
    spread = percentile(spreads, EPS_QUANTILE)
    return None if spread is None else max(EPS_FLOOR, spread)


def pairs_of(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


def tie_fraction(entries, eps):
    ties = total = 0
    for a, b in pairs_of(entries):
        total += 1
        ties += abs(a['S_value'] - b['S_value']) <= eps
    return (ties / total if total else None), ties, total


def p_flip(entries, eps):
    """Flips among non-tie candidate pairs: S-order differs from r0-order."""
    flips = nontie = 0
    for a, b in pairs_of(entries):
        if abs(a['S_value'] - b['S_value']) <= eps or abs(a['r0'] - b['r0']) <= EXACT_TIE:
            continue
        nontie += 1
        flips += (a['S_rmse'] < b['S_rmse']) != (a['r0'] < b['r0'])
    return flips, nontie


def make_groups(champion, entries, eps, cap=PER_CONTEXT_CAP):
    """[champion] + 7 seeds by S_value desc; adjacent ties excluded; groups with < 2 non-tie pairs dropped."""
    ordered = sorted((e for e in entries if e['record'] != champion['record']),
                     key=lambda e: (-e['S_value'], e['record']))
    admitted, ties_excluded, seen = [], 0, set()
    for entry in ordered:
        if entry['record'] in seen:
            continue
        if admitted and admitted[-1]['S_value'] - entry['S_value'] <= eps:
            ties_excluded += 1
            continue
        admitted.append(entry)
        seen.add(entry['record'])
    groups, dropped, incomplete = [], 0, 0
    for start in range(0, len(admitted), GROUP_SIZE - 1):
        chunk = admitted[start:start + GROUP_SIZE - 1]
        if len(chunk) < GROUP_SIZE - 1:
            incomplete += 1
            break
        members = [champion] + chunk
        nontie = sum(abs(a['S_value'] - b['S_value']) > eps for a, b in pairs_of(members))
        if nontie < 2:
            dropped += 1
            continue
        if len(groups) >= cap:
            break
        groups.append(members)
    return groups, ties_excluded, dropped, incomplete


def context_advantage(entries):
    """A_ctx per entry: value(S) - median_{control_generic in the context} value(S); None without a measured control."""
    controls = [e['S_value'] for e in entries if e['role'] == A_CTX_CONTROL]
    if not controls:
        return None, [None] * len(entries)
    reference = statistics.median(controls)
    return reference, [e['S_value'] - reference for e in entries]


def a_ctx_report(by_context_entries):
    """Per-record A_ctx (joined records) and per-arm medians over every measured entry; unavailable contexts are named."""
    per_record, by_arm, without = {}, {arm: [] for arm in ARMS}, []
    references = {}
    for context_key, entries in sorted(by_context_entries.items()):
        reference, advantages = context_advantage(entries)
        if reference is None:
            without.append(context_key)
            continue
        references[context_key] = reference
        for entry, advantage in zip(entries, advantages):
            if entry['record'] is not None:
                per_record[str(entry['record'])] = advantage
            if entry['arm'] in by_arm:
                by_arm[entry['arm']].append(advantage)
    return dict(definition='value(S) - median over the context\'s %s seeds of value(S); reported advantage, never a weight' % A_CTX_CONTROL,
                control_role=A_CTX_CONTROL, per_record=per_record,
                by_arm={arm: (statistics.median(v) if v else None) for arm, v in by_arm.items()},
                n_by_arm={arm: len(v) for arm, v in by_arm.items()}, control_median_by_context=references,
                contexts_with_control=len(references), contexts_without_control=without)


def conditional_contrast(by_context, control_role):
    """E[S_value | neural, r0 decile] - E[S_value | control, r0 decile] within context, pooled with a cluster bootstrap."""
    cells = {}
    for context_key, entries in by_context.items():
        pool = [e for e in entries if e['role'] in ('candidate', control_role)]
        if len(pool) < 2:
            continue
        deciles = [min(9, int(10 * (r - 1) / len(pool))) for r in ranks([e['r0'] for e in pool])]
        buckets = {}
        for entry, decile in zip(pool, deciles):
            buckets.setdefault(decile, {'candidate': [], control_role: []})[entry['role']].append(entry['S_value'])
        for decile, bucket in buckets.items():
            if bucket['candidate'] and bucket[control_role]:
                cells.setdefault(context_key, []).append((decile, statistics.fmean(bucket['candidate']) - statistics.fmean(bucket[control_role])))
    per_decile = {}
    for rows in cells.values():
        for decile, diff in rows:
            per_decile.setdefault(decile, []).append(diff)
    every = [diff for rows in cells.values() for _, diff in rows]
    interval = cluster_bootstrap({k: [d for _, d in v] for k, v in cells.items()},
                                 lambda pooled: statistics.fmean(pooled) if pooled else None)
    return dict(control_role=control_role, contexts=len(cells), cells=len(every),
                overall=dict(mean=statistics.fmean(every) if every else None, ci90=interval),
                per_decile={str(d): dict(mean=statistics.fmean(v), n_cells=len(v)) for d, v in sorted(per_decile.items())})


def build(corpus_dir, seed_values_export_path, control_role='control_generic', *, evaluator_sha256=None,
          corpus_sha256=None, next_outcome=None, per_context_cap=PER_CONTEXT_CAP):
    """Policy block parallel.seed_values.v1; raises ValueError with the refusal reason when it cannot be built."""
    require(control_role in CONTROL_ROLES, 'control role must be control_generic or control_shuffled')
    export_path = Path(seed_values_export_path)
    export = read(export_path)
    require(export.get('schema') == EXPORT_SCHEMA, 'seed-value export schema mismatch: ' + str(export.get('schema')))
    require(export.get('final_tests_opened') is False, 'seed-value export must state final_tests_opened false')
    rows = export['rows']
    evaluators = sorted({r.get('evaluator_sha256') for r in rows})
    require(len(evaluators) == 1 and evaluators[0], 'export rows must carry exactly one evaluator identity')
    require(evaluator_sha256 is None or evaluator_sha256 == evaluators[0],
            'evaluator mismatch: export ' + str(evaluators[0]) + ' expected ' + str(evaluator_sha256))
    corpus = Path(corpus_dir)
    bound_corpus = sha(corpus / 'index.json')
    require(corpus_sha256 is None or corpus_sha256 == bound_corpus, 'corpus mismatch: bound ' + bound_corpus + ' expected ' + str(corpus_sha256))
    index, contexts = corpus_contexts(corpus)
    outcomes = load_next_outcome(next_outcome)
    measured = [r for r in rows if r.get('status') == 'measured' and finite(r.get('r0')) and finite(r.get('S_rmse'))]
    require(measured, 'export carries no measured seed values')
    require(all(r.get('partition', 'meta_train') == 'meta_train' for r in rows), 'non-meta_train seed-value rows are never consumed')
    by_context = {}
    for row in measured:
        by_context.setdefault(row['context_key'], []).append(row)
    refused, eps_by, ties_by, groups, values, roles, arms, floors = [], {}, {}, [], {}, {}, {}, {}
    counts = dict(contexts=0, groups=0, ties_excluded=0, refused_contexts=0, unjoined_rows=0, measured_rows=len(measured),
                  eligible_rows=0, dropped_groups=0, incomplete_groups=0, duplicate_records=0, label_mismatch_rows=0)
    label_mismatches = []
    flips = nontie = 0
    per_context_info, regression, contrast_entries = {}, {}, {}
    for context_key in sorted(by_context):
        entries_raw = by_context[context_key]
        prepared, cut = entries_raw[0]['prepared_sha256'], entries_raw[0]['cut']
        candidates = contexts.get((prepared, cut))
        if not candidates:
            refused.append(dict(context_key=context_key, reason='context_not_in_corpus'))
            continue
        for candidate in candidates:
            require(candidate['partition'] == 'meta_train', 'non-meta_train corpus group for ' + context_key)
        group = next((c for c in candidates if c['family'] == entries_raw[0]['family']), None)
        if group is None:
            refused.append(dict(context_key=context_key, reason='family_mismatch'))
            continue
        median = context_median(group['programs'])
        eps = eps_for(entries_raw, median)
        if eps is None:
            refused.append(dict(context_key=context_key, reason='eps_unavailable: no finite replicate pair'))
            continue
        entries = []
        for row in entries_raw:
            if row['role'] not in ROLE_PRIORITY or row.get('replicate', 0) != 0:
                continue
            s_value, r0_value = value(row['S_rmse'], median), value(row['r0'], median)
            joined = group['programs'].get(row['seed_genome'])
            if joined is not None and joined['rmse'] != row['r0']:
                # F5 at consumption: the seed's own row-0 RMSE must be bit-equal to its corpus label.
                counts['label_mismatch_rows'] = counts.get('label_mismatch_rows', 0) + 1
                label_mismatches.append(dict(context_key=context_key, seed_program_sha256=row['seed_program_sha256'], r0=row['r0'], label=joined['rmse']))
                joined = None
            entries.append(dict(record=None if joined is None else joined['record'], role=row['role'], arm=row['arm'],
                                r0=row['r0'], S_rmse=row['S_rmse'], S_value=s_value, r0_value=r0_value, A_own=s_value - r0_value,
                                seed=row['seed_program_sha256']))
        fraction, ties, total = tie_fraction(entries, eps)
        eps_by[context_key] = eps
        ties_by[context_key] = dict(tie_fraction=fraction, ties=ties, pairs=total)
        candidate_entries = [e for e in entries if e['role'] == 'candidate']
        context_flips, context_nontie = p_flip(candidate_entries, eps)
        flips += context_flips
        nontie += context_nontie
        per_context_info[context_key] = dict(candidates=len(candidate_entries), flips=context_flips, nontie_pairs=context_nontie, eps=eps,
                                             tie_fraction=fraction, median_rmse=median)
        if outcomes is not None:
            samples = [(outcomes[(context_key, e['seed'])], e['r0'], e['A_own']) for e in candidate_entries if (context_key, e['seed']) in outcomes]
            if samples:
                regression[context_key] = samples
        contrast_entries[context_key] = entries
        if fraction is not None and fraction > TIE_REFUSAL:
            refused.append(dict(context_key=context_key, reason='tie_fraction %.3f > %.1f' % (fraction, TIE_REFUSAL)))
            continue
        joined_entries = [e for e in entries if e['record'] is not None]
        counts['unjoined_rows'] += len(entries) - len(joined_entries)
        champion_program = min(group['programs'].items(), key=lambda item: (item[1]['rmse'], item[1]['record']))
        champion = dict(record=champion_program[1]['record'], role='champion', arm='corpus', S_value=value(champion_program[1]['rmse'], median))
        ordered = sorted(joined_entries, key=lambda e: (ROLE_PRIORITY.index(e['role']), e['record']))
        unique, seen = [], set()
        for entry in ordered:
            if entry['record'] in seen:
                counts['duplicate_records'] += 1
                continue
            seen.add(entry['record'])
            unique.append(entry)
        counts['eligible_rows'] += len(unique)
        context_groups, ties_excluded, dropped, incomplete = make_groups(champion, unique, eps, per_context_cap)
        counts['ties_excluded'] += ties_excluded
        counts['dropped_groups'] += dropped
        counts['incomplete_groups'] += incomplete
        if not context_groups:
            refused.append(dict(context_key=context_key, reason='no group with two non-tie pairs'))
            continue
        counts['contexts'] += 1
        for members in context_groups:
            floors[str(len(groups))] = eps
            groups.append([m['record'] for m in members])
            for member in members:
                values[str(member['record'])] = member['S_value']
                roles[str(member['record'])] = member['role']
                arms[str(member['record'])] = member['arm']
    require(eps_by, 'replicate panel unavailable in every context: eps_c cannot be derived; refusing to build seed values')
    counts['groups'] = len(groups)
    counts['refused_contexts'] = len(refused)
    require(groups, 'no context produced a consumable group: ' + '; '.join(r['reason'] for r in refused[:5]))
    p_flip_value = flips / nontie if nontie else None
    passed_a = p_flip_value is not None and p_flip_value >= P_FLIP_FLOOR
    a_own = dict(coefficient=None, ci90=None, n_rows=0, n_contexts=len(regression), reason=None)
    if outcomes is None:
        a_own['reason'] = 'next_outcome_unavailable'
    else:
        pooled = [s for samples in regression.values() for s in samples]
        a_own['n_rows'] = len(pooled)
        a_own['coefficient'] = rank_regression_coefficient(pooled)
        a_own['ci90'] = cluster_bootstrap(regression, rank_regression_coefficient)
        if a_own['coefficient'] is None or a_own['ci90'] is None:
            a_own['reason'] = 'insufficient rows or clusters for the rank regression'
    passed_b = a_own['ci90'] is not None and (a_own['ci90']['low'] > 0 or a_own['ci90']['high'] < 0)
    generation = export.get('generation')
    a_ctx = a_ctx_report(contrast_entries)
    for context_key, reference in a_ctx['control_median_by_context'].items():
        per_context_info[context_key]['a_ctx_control_median'] = reference
    return dict(schema=SCHEMA, corpus_sha256=bound_corpus, evaluator_sha256=evaluators[0], protocol=PROTOCOL,
                generation=generation, tie_rule=TIE_RULE, groups=groups, values=values, floors=floors, roles=roles, arms=arms,
                per_context_cap=per_context_cap, group_size=GROUP_SIZE, counts=counts, refused=refused, label_mismatches=label_mismatches[:16],
                seed_values_export_sha256=sha(export_path), control_role=control_role, a_ctx=a_ctx,
                eps=eps_by, tie_fraction=ties_by, contexts=per_context_info,
                info_test=dict(p_flip=p_flip_value, p_flip_flips=flips, p_flip_pairs=nontie, p_flip_floor=P_FLIP_FLOOR,
                               a_own_ci=a_own['ci90'], a_own=a_own, passed_a=passed_a, passed_b=passed_b,
                               passed=bool(passed_a and passed_b),
                               conditional_contrast=conditional_contrast(contrast_entries, control_role)),
                value_definition='1/(1 + x/median_c); median_c = median labels.jsonl rmse_cut_normalized of the context',
                a_own_definition='value(S) - value(r0); diagnostic only, never a weight',
                a_ctx_definition=a_ctx['definition'],
                bootstrap=dict(resamples=RESAMPLES, seed=BOOTSTRAP_SEED, alpha=ALPHA, clusters='contexts'),
                final_tests_opened=False)


def refusal(reason, corpus_dir=None, export_path=None):
    return dict(schema=REFUSAL_SCHEMA, refused=True, reason=str(reason),
                corpus=None if corpus_dir is None else str(corpus_dir),
                seed_values_export=None if export_path is None else str(export_path), final_tests_opened=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='immutable policy block path')
    parser.add_argument('--control-role', default='control_generic', choices=CONTROL_ROLES)
    parser.add_argument('--evaluator-sha256')
    parser.add_argument('--corpus-sha256')
    parser.add_argument('--next-outcome', type=Path)
    parser.add_argument('--refusal-output', type=Path, help='where to record the refusal reason (never immutable)')
    args = parser.parse_args()
    try:
        block = build(args.corpus, args.export, args.control_role, evaluator_sha256=args.evaluator_sha256,
                      corpus_sha256=args.corpus_sha256, next_outcome=args.next_outcome)
    except (ValueError, RuntimeError, KeyError, OSError) as error:
        body = refusal(error, args.corpus, args.export)
        if args.refusal_output:
            args.refusal_output.parent.mkdir(parents=True, exist_ok=True)
            args.refusal_output.write_bytes(canonical(body) + b'\n')
        print(json.dumps(body, sort_keys=True))
        raise SystemExit(1)
    digest = immutable(args.output, block)
    print(json.dumps(dict(output=str(args.output), sha256=digest, groups=block['counts']['groups'],
                          contexts=block['counts']['contexts'], passed=block['info_test']['passed']), sort_keys=True))


if __name__ == '__main__':
    main()
