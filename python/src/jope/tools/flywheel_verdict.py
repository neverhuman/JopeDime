#!/usr/bin/env python3
"""Pre-registered flywheel verdict at a generation; thresholds come only from PREREGISTRATION.md.

Reads the fenced ```json block of the pre-registration document (through
flywheel_baseline.preregistration when available; refusing it when its digest
differs from --preregistration-sha256), the flywheel series and the immutable
baseline receipt; evaluates F1-F8 as far as the inputs exist and writes an
immutable successor/verdicts/generation-<G>.json with an outcome in
{smarter_confirmed, no_learning_detected, harmful_stopped,
inconclusive_underpowered}. Nothing is imputed: a falsifier without inputs or
without its threshold is reported as not evaluable, and no constant in this file
is a threshold.

Verdict generations (5/10/20) count from an explicit origin: --since-generation
is the last pre-change run generation (the run counter is already ~70+ on the
node); without it the origin is the baseline receipt's generation_range.last.
F1/F4 read rows with origin < run_generation <= origin + G.

Outcome rules (PREREGISTRATION.md verdict.rules): harmful_stopped iff the brake
(F2 + F3 reject windows) fired >= verdict.brake_fires_for_harmful times;
smarter_confirmed iff F1 and F8 pass with F2/F3 evaluable and silent and F5 not
fired; no_learning_detected iff F1 and F8 both fail at the final generation with
F2/F3 evaluable and silent; else inconclusive_underpowered. F5 trips and an F8
refutation are reported as flags (they block smarter_confirmed) and never
extend the pre-registered enum.

Dotted paths read from the block (parallel.flywheel_preregistration.v1; the
first existing alternative wins):
  F1  statistics.{alpha,resamples,seed,window_generations,n_clusters}
      falsifiers.F1.secondary.win_rate_delta
      statistics.mde_n16.multiplier_of_sigma_cut (MDE = multiplier * stdev of the baseline per-cut means)
  F2  falsifiers.F2.consecutive_rejects
  F3  falsifiers.F3.{consecutive_generations,unique_fresh_programs_fraction_of_baseline_median,
                     unique_emissions_per_context_min,mean_tokens_over_baseline}
  F4  falsifiers.F4.{p_flip_min,minimum_generations}
  F8  falsifiers.F8.{accuracy_gain_min,hold_below_baseline,hold_consecutive_blocks,by_block}
  verdict.generations; optional verdict.final_generation (default: max of generations) and
  verdict.brake_fires_for_harmful (without it the "brake fires twice" clause is not evaluable)
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

import flywheel_brake
import flywheel_metrics
from flywheel_series import baseline_receipt, ci_bounds, clean, digest, finite, read, series_rows, usable
import promotion_gate
from resume_expired_cycle import require, sha

try:
    import flywheel_baseline
except ImportError:  # B's reader ships separately; the fallback below reads the same block
    flywheel_baseline = None

SCHEMA = 'parallel.flywheel_verdict.v1'
PREREGISTRATION_SCHEMA = 'parallel.flywheel_preregistration.v1'
OUTCOMES = ('smarter_confirmed', 'no_learning_detected', 'harmful_stopped', 'inconclusive_underpowered')
FALSIFIERS = ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8')
FENCE = re.compile(r'^```json[ \t]*\r?\n(.*?)\r?\n```[ \t]*$', re.S | re.M)
PATHS = dict(
    alpha=('statistics.alpha', 'F1.alpha'), resamples=('statistics.resamples', 'F1.resamples'),
    seed=('statistics.seed', 'F1.seed'), window=('statistics.window_generations', 'F1.window_generations'),
    n_cuts=('statistics.n_clusters', 'F1.n_cuts'),
    win_rate_delta=('falsifiers.F1.secondary.win_rate_delta', 'statistics.secondary.win_rate_delta', 'F1.win_rate_delta'),
    mde_multiplier=('statistics.mde_n16.multiplier_of_sigma_cut',), mde_absolute=('F1.mde_log_ratio',),
    f2_consecutive=('falsifiers.F2.consecutive_rejects', 'F2.consecutive'),
    f3_consecutive=('falsifiers.F3.consecutive_generations', 'F3.consecutive'),
    f3_fresh=('falsifiers.F3.unique_fresh_programs_fraction_of_baseline_median', 'F3.fresh_fraction'),
    f3_unique=('falsifiers.F3.unique_emissions_per_context_min', 'F3.unique_emissions_min'),
    f3_tokens=('falsifiers.F3.mean_tokens_over_baseline', 'F3.token_length_factor'),
    f4_p_flip=('falsifiers.F4.p_flip_min', 'F4.p_flip_min'), f4_generations=('falsifiers.F4.minimum_generations', 'F4.min_generations'),
    f8_delta=('falsifiers.F8.accuracy_gain_min', 'F8.accuracy_delta'), f8_regression=('falsifiers.F8.hold_below_baseline', 'F8.regression_delta'),
    f8_consecutive=('falsifiers.F8.hold_consecutive_blocks', 'F8.consecutive'), f8_by_block=('falsifiers.F8.by_block', 'F8.by_block'),
    generations=('verdict.generations',), final_generation=('verdict.final_generation',),
    harmful_fires=('verdict.brake_fires_for_harmful',))


class Missing(RuntimeError):
    """A threshold the pre-registration block does not fix."""


# ---------------------------------------------------------------- preregistration

def fallback_reader(path):
    path = Path(path)
    blocks = FENCE.findall(path.read_text())
    require(len(blocks) == 1, 'PREREGISTRATION.md must carry exactly one fenced json block; found %d' % len(blocks))
    block = json.loads(blocks[0])
    require(isinstance(block, dict) and block.get('schema') == PREREGISTRATION_SCHEMA, 'unknown preregistration schema')
    require(block.get('final_tests_opened') is False, 'preregistration opened final tests')
    return dict(path=str(path), block=block, sha256=digest(block), block_bytes_sha256=hashlib.sha256(blocks[0].encode()).hexdigest(),
                digest_rule='sha256 of the canonical JSON (sorted keys, compact separators) of the block')


def preregistration(path, expected_sha=None):
    reader = flywheel_baseline.preregistration if flywheel_baseline is not None else fallback_reader
    found = reader(path)
    matched = None
    if expected_sha is not None:
        matched = 'canonical_json' if expected_sha == found['sha256'] else 'block_text' if expected_sha == found.get('block_bytes_sha256') else None
        require(matched, 'preregistration block digest %s (block text %s) differs from the supplied %s' % (
            found['sha256'][:12], str(found.get('block_bytes_sha256'))[:12], str(expected_sha)[:12]))
    return dict(found, document_sha256=sha(path), supplied_sha256=expected_sha, matched_by=matched,
                reader='flywheel_baseline.preregistration' if flywheel_baseline is not None else 'flywheel_verdict.fallback_reader')


def lookup(block, path):
    value = block
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def threshold(block, name, optional=False):
    for path in PATHS[name]:
        value = lookup(block, path)
        if value is not None:
            return value
    if optional:
        return None
    raise Missing('preregistration block lacks ' + ' | '.join(PATHS[name]))


# ---------------------------------------------------------------- helpers

def latest(rows, kind, generation=None, key='generation', after=None):
    """Latest row of a kind with after < row[key] <= generation (either bound optional)."""
    chosen = [r for r in rows if r.get('kind') == kind
              and (generation is None or (type(r.get(key)) is int and r[key] <= generation))
              and (after is None or (type(r.get(key)) is int and r[key] > after))]
    return max(chosen, key=lambda r: (r.get(key) if type(r.get(key)) is int else -1, r.get('timestamp') or 0, r['id']), default=None)


def origin_generation(receipt, since=None):
    """Last pre-change run generation: --since-generation, else the baseline receipt's generation_range.last."""
    if type(since) is int:
        return since, 'cli:--since-generation'
    generation_range = usable(receipt.get('generation_range'))
    last = generation_range.get('last') if isinstance(generation_range, dict) else None
    if type(last) is int:
        return last, 'baseline receipt generation_range.last'
    raise RuntimeError('verdict origin unknown: pass --since-generation (the last pre-change run generation) '
                       'or a baseline receipt whose generation_range.last is an integer')


def not_evaluable(name, reason, **extra):
    return dict(falsifier=name, status='not_evaluable', evaluable=False, reason=reason, **extra)


def guarded(name, function, *args):
    try:
        return function(*args)
    except Missing as missing:
        return not_evaluable(name, str(missing))


def cut_means(per_cut):
    return {'%s:%s' % (c.get('dataset'), c.get('cut')): c['mean_log_ratio']
            for c in per_cut or [] if isinstance(c, dict) and finite(c.get('mean_log_ratio'))}


def two_sample_ci(a, b, resamples, seed, alpha):
    """Independent cluster resampling of two per-cut samples; statistic = mean(a) - mean(b)."""
    a = [a[k] for k in sorted(a)]
    b = [b[k] for k in sorted(b)]
    require(a and b, 'two-sample bootstrap needs both samples')
    rng = random.Random(seed)
    stats = sorted(statistics.fmean(rng.choices(a, k=len(a))) - statistics.fmean(rng.choices(b, k=len(b))) for _ in range(resamples))
    last = len(stats) - 1
    return dict(mean=statistics.fmean(a) - statistics.fmean(b), ci_low=stats[min(last, int(len(stats) * alpha / 2))],
                ci_high=stats[min(last, int(len(stats) * (1 - alpha / 2)))], n_clusters=[len(a), len(b)],
                resamples=resamples, seed=seed, alpha=alpha, method='independent_cluster_resampling')


def difference_ci(window, base, resamples, seed, alpha):
    """The pre-registered primary interval: two-sample cluster bootstrap of window minus baseline, always."""
    return two_sample_ci(window, base, resamples, seed, alpha)


def paired_diagnostic(window, base, resamples, seed, alpha):
    """Labelled diagnostic only (never the decision statistic): paired per-cut differences when both cover the same cuts."""
    common = sorted(set(window) & set(base))
    if not common or len(common) != len(window) or len(common) != len(base):
        return dict(available=False, reason='window and baseline do not cover the same cuts', n_common=len(common))
    ci = flywheel_metrics.cluster_bootstrap_ci({k: [window[k] - base[k]] for k in common}, resamples, seed, alpha)
    return dict(ci, available=True, method='paired_cut_difference_cluster_bootstrap', n_clusters=len(common),
                role='diagnostic: not pre-registered; the primary interval is the two-sample cluster bootstrap')


def minimum_detectable(block, base_cuts):
    """MDE per the block: an absolute log-ratio, or multiplier * stdev of the baseline per-cut means."""
    absolute = threshold(block, 'mde_absolute', optional=True)
    if finite(absolute):
        return dict(value=absolute, rule='absolute log-ratio from the block')
    multiplier = threshold(block, 'mde_multiplier', optional=True)
    if not finite(multiplier):
        return dict(value=None, rule='block fixes no MDE')
    if len(base_cuts) < 2:
        return dict(value=None, rule='baseline receipt has fewer than two per-cut means; sigma_cut undefined', multiplier=multiplier)
    sigma = statistics.stdev(base_cuts.values())
    return dict(value=multiplier * sigma, multiplier=multiplier, sigma_cut=sigma, n_cuts=len(base_cuts),
                rule='multiplier * sample standard deviation of the baseline per-cut mean log-ratios')


# ---------------------------------------------------------------- falsifiers

def f1_heldout(rows, receipt, block, generation, origin=0):
    alpha, resamples, seed = threshold(block, 'alpha'), threshold(block, 'resamples'), threshold(block, 'seed')
    window_size, n_cuts, win_rate_delta = threshold(block, 'window'), threshold(block, 'n_cuts'), threshold(block, 'win_rate_delta')
    run_generation = origin + generation
    row = latest(rows, 'generation', run_generation, after=origin)
    if row is None:
        return not_evaluable('F1', 'no generation row with run generation in (%d, %d] (verdict generation %d from origin %d)'
                             % (origin, run_generation, generation, origin))
    window = ((row.get('matched') or {}).get('val') or {}).get('trailing10')
    if not isinstance(window, dict) or not finite(window.get('mean_log_ratio')):
        return not_evaluable('F1', 'trailing window contrast unavailable in generation row ' + row['id'])
    heldout = usable((receipt.get('metrics') or {}).get('matched_heldout'))
    base = heldout.get('seed_only') if isinstance(heldout, dict) else None
    if not isinstance(base, dict) or not finite(base.get('mean_log_ratio')):
        return not_evaluable('F1', 'baseline receipt carries no matched held-out seed-only contrast')
    window_cuts, base_cuts = cut_means(window.get('per_cut')), cut_means(base.get('per_cut'))
    if not window_cuts or not base_cuts:
        return not_evaluable('F1', 'per-cut log-ratios missing from the window or the baseline receipt')
    ci = difference_ci(window_cuts, base_cuts, resamples, seed, alpha)
    paired = paired_diagnostic(window_cuts, base_cuts, resamples, seed, alpha)
    mde = minimum_detectable(block, base_cuts)
    wins, losses, n_pairs = window.get('wins'), window.get('losses'), window.get('n_pairs')
    win_rate = wins / n_pairs if type(wins) is int and type(n_pairs) is int and n_pairs else None
    base_rate = base['wins'] / base['n_pairs'] if type(base.get('wins')) is int and type(base.get('n_pairs')) is int and base['n_pairs'] else None
    secondary = dict(win_rate=win_rate, baseline_win_rate=base_rate,
                     delta=(win_rate - base_rate) if win_rate is not None and base_rate is not None else None,
                     threshold=win_rate_delta, caveat='pooled pairs are cluster-correlated; overstated ~2-3x')
    secondary['passed'] = secondary['delta'] is not None and secondary['delta'] >= win_rate_delta
    excludes = not (ci['ci_low'] <= 0 <= ci['ci_high'])
    if excludes and ci['ci_high'] < 0:
        status, reason = ('pass', 'CI excludes 0 on the improving side and wins > losses') if (type(wins) is int and type(losses) is int and wins > losses) \
            else ('fail', 'CI excludes 0 but wins do not exceed losses')
    elif excludes:
        status, reason = 'fail', 'CI excludes 0 on the worsening side'
    elif finite(mde['value']) and abs(ci['mean']) >= mde['value']:
        status, reason = 'fail', 'CI straddles 0 with |mean| at or above the pre-registered MDE'
    else:
        status, reason = 'underpowered', 'CI straddles 0' + (' with |mean| below the MDE' if finite(mde['value']) else '; block fixes no MDE')
    return dict(falsifier='F1', status=status, evaluable=True, reason=reason, generation=row.get('generation'),
                verdict_generation=generation, origin=origin,
                window_generations=window.get('generations'), window_size=window_size, expected_n_cuts=n_cuts,
                n_cuts=dict(window=len(window_cuts), baseline=len(base_cuts), common=len(set(window_cuts) & set(base_cuts))),
                window=dict(mean_log_ratio=window['mean_log_ratio'], ci90=window.get('ci90'), wins=wins, ties=window.get('ties'), losses=losses),
                baseline=dict(mean_log_ratio=base['mean_log_ratio'], ci90=base.get('ci90'), wins=base.get('wins'), ties=base.get('ties'), losses=base.get('losses')),
                difference=ci, paired_diagnostic=paired, mde=mde, secondary=secondary,
                statistic='window per-cut mean minus baseline per-cut mean of ln(best_initial_neural / best_initial_generic); '
                          'two-sample cluster bootstrap (pre-registered), MDE = two-sample formula')


def f2_brake(rows, baseline, block):
    result = flywheel_brake.evaluate_f2(rows, baseline, threshold(block, 'f2_consecutive'))
    if not result['evaluable']:
        return not_evaluable('F2', result['reason'], fires=0)
    return dict(result, status='fired' if result['fires'] else 'silent')


def f3_diversity(rows, baseline, block):
    result = flywheel_brake.evaluate_f3(rows, baseline, threshold(block, 'f3_consecutive'), threshold(block, 'f3_fresh'),
                                        threshold(block, 'f3_unique'), threshold(block, 'f3_tokens'))
    if not result['evaluable']:
        return not_evaluable('F3', result['reason'], fires=0)
    return dict(result, status='fired' if result['fires'] else 'silent')


def f4_information(rows, block, generation, origin=0):
    floor, minimum = threshold(block, 'f4_p_flip'), threshold(block, 'f4_generations')
    run_generation = origin + generation
    episodes = sorted((r for r in rows if r.get('kind') == 'episodes' and type(r.get('generation')) is int
                       and origin < r['generation'] <= run_generation), key=lambda r: r['generation'])
    measured = [r for r in episodes if finite((r.get('successor') or {}).get('p_flip'))]
    if len(measured) < minimum:
        return not_evaluable('F4', '%d generation(s) with a measured P_flip in run generations (%d, %d]; %d required'
                             % (len(measured), origin, run_generation, minimum), generations=[r['generation'] for r in measured])
    row = measured[-1]['successor']
    ci = ci_bounds(row.get('a_own_ci'))   # successor_targets emits {low, high, resamples}; series rows carry [low, high]
    ci_ok = ci is not None
    a = ci_ok and not (ci[0] <= 0 <= ci[1])
    p_flip_ok = row['p_flip'] >= floor
    return dict(falsifier='F4', status='pass' if p_flip_ok and a else 'fail', evaluable=True,
                reason='P_flip %s floor %s; A_own CI %s' % ('meets' if p_flip_ok else 'below', floor, 'excludes 0' if a else ('straddles 0' if ci_ok else 'unavailable')),
                generation=measured[-1]['generation'], verdict_generation=generation, origin=origin, p_flip=row['p_flip'], p_flip_min=floor,
                a_own_ci=ci, a_own_ci_source=row.get('a_own_ci'),
                p_flip_series=[(r['generation'], r['successor']['p_flip']) for r in measured], generations=len(measured))


F4_RECEIPT_SCHEMA = 'parallel.successor_f4_information_test.v1'


def f4_from_receipt(path, block):
    """F4 from an immutable successor F4 information-test receipt (successor ledger driver): the measured P_flip and the
    A_own cluster-bootstrap interval are copied verbatim and re-judged against the block's floors (the receipt's own
    pass flags are never trusted; its floor must equal the block's). A pass additionally needs at least
    falsifiers.F4.minimum_generations measured generations; a fail is a fail on any number of generations."""
    floor, minimum = threshold(block, 'f4_p_flip'), threshold(block, 'f4_generations')
    path = Path(path)
    value = read(path)
    require(isinstance(value, dict) and value.get('schema') == F4_RECEIPT_SCHEMA, '%s: schema is not %s' % (path, F4_RECEIPT_SCHEMA))
    require(value.get('final_tests_opened') is False, '%s: final_tests_opened is not false' % path)
    p_flip, a_own = value.get('p_flip') or {}, value.get('a_own') or {}
    require(finite(p_flip.get('value')) and type(p_flip.get('pairs')) is int and type(p_flip.get('flips')) is int, '%s: p_flip lacks value/pairs/flips' % path)
    require(finite(p_flip.get('floor')) and abs(p_flip['floor'] - floor) < 1e-12,
            '%s: receipt P_flip floor %r differs from the pre-registered %r' % (path, p_flip.get('floor'), floor))
    ci = ci_bounds(a_own.get('ci90'))
    coefficient = a_own.get('coefficient') if finite(a_own.get('coefficient')) else None
    generations = value.get('inputs', {}).get('generations')
    generations = [g for g in generations if type(g) is int] if isinstance(generations, list) else []
    per_generation = p_flip.get('per_generation') if isinstance(p_flip.get('per_generation'), dict) else {}
    a = ci is not None and not (ci[0] <= 0 <= ci[1])
    p_flip_ok = p_flip['value'] >= floor
    enough = len(generations) >= minimum
    status = 'pass' if p_flip_ok and a and enough else 'fail'
    reason = 'P_flip %s floor %s; A_own CI %s' % ('meets' if p_flip_ok else 'below', floor,
                                                 'excludes 0' if a else ('straddles 0' if ci is not None else 'unavailable'))
    if p_flip_ok and a and not enough:
        reason += '; measured on %d generation(s), %d required for a pass' % (len(generations), minimum)
    contrast = value.get('conditional_contrast') or {}
    return dict(falsifier='F4', status=status, evaluable=True, reason=reason, source='receipt', receipt=dict(path=str(path), sha256=sha(path)),
                p_flip=p_flip['value'], p_flip_min=floor, p_flip_pairs=p_flip['pairs'], p_flip_flips=p_flip['flips'],
                a_own_coefficient=coefficient, a_own_ci=ci, a_own_ci_source=a_own.get('ci90'), a_own_n_contexts=a_own.get('n_contexts'),
                a_own_n_rows=a_own.get('n_rows'), generations=len(generations), generation_list=generations, minimum_generations=minimum,
                insufficient_generations=not enough,
                p_flip_series=[(int(g), per_generation[g]['value']) for g in sorted(per_generation, key=lambda g: int(g))
                               if isinstance(per_generation[g], dict) and finite(per_generation[g].get('value'))],
                conditional_contrast=dict(contexts=contrast.get('contexts'), cells=contrast.get('cells'), reason=contrast.get('reason'))
                if isinstance(contrast, dict) else None,
                counts={k: (value.get('counts') or {}).get(k) for k in ('completed_with_generation', 'ingested', 'verified', 'unverified')})


def f5_identity(rows):
    result = flywheel_brake.evaluate_f5(rows)
    if not result['evaluable']:
        return not_evaluable('F5', result['reason'])
    return dict(result, status='fired' if result['tripped'] else 'silent')


def f6_bridge(rows):
    bridges = [r for r in rows if r.get('kind') == 'bridge']
    return not_evaluable('F6', 'bridge receipts are not part of the flywheel series in this ship (F6 is evaluated before the manifest revision that rebinds a rebuilt monitor activates)') \
        if not bridges else dict(falsifier='F6', status='pass', evaluable=True, reason='bridge rows present', rows=len(bridges))


def f7_consumption(rows):
    row = latest(rows, 'block', key='block_index')
    if row is None:
        return not_evaluable('F7', 'no completed training block rows in the series')
    status = row.get('seed_values') or {}
    bt, groups = row.get('seed_value_bt'), row.get('seed_value_groups_seen')
    if status.get('consumed') == 'yes':
        recorded = finite(bt) or type(groups) is int
        ok = (not recorded) or (finite(bt) and type(groups) is int and groups > 0)
        verdict, reason = ('pass', 'block consumed a seed_values block' + ('' if recorded else '; trainer records no seed_value_bt yet')) if ok \
            else ('fail', 'block claims consumption but seed_value_bt/seed_value_groups_seen are not finite/positive')
    elif status.get('consumed') == 'no' and 'carries no seed_values field' in str(status.get('reason')):
        verdict, reason = 'silent', 'block receipt is silent about seed values: ' + str(status.get('reason'))
    else:
        verdict, reason = 'not_applicable', 'journaled: ' + str(status.get('reason'))
    return dict(falsifier='F7', status=verdict, evaluable=True, reason=reason, block=row['id'], block_index=row.get('block_index'),
                consumed=status.get('consumed'), seed_value_bt=bt, seed_value_groups_seen=groups)


def f8_critic(rows, baseline, block):
    delta, regression = threshold(block, 'f8_delta'), threshold(block, 'f8_regression')
    consecutive, by_block = threshold(block, 'f8_consecutive'), threshold(block, 'f8_by_block')
    base = (baseline.get('panel') or {}).get('nontie_pairwise_accuracy')
    series = [r for r in flywheel_brake.learning_checkpoints(rows) if finite((r.get('panel') or {}).get('nontie_pairwise_accuracy'))]
    if not series:
        return not_evaluable('F8', 'nontie_pairwise_accuracy absent from monitor receipts (pre-B1 monitor identity)')
    if not finite(base):
        return not_evaluable('F8', 'baseline receipt lacks nontie_pairwise_accuracy for the learning checkpoint')
    values = [r['panel']['nontie_pairwise_accuracy'] for r in series]
    run = refuted = 0
    for value in values:
        run = run + 1 if value < base - regression else 0
        refuted = refuted or run >= consecutive
    current = series[-1]
    accuracy = current['panel']['nontie_pairwise_accuracy']
    groups = current['panel'].get('nontie_pairwise_accuracy_by_group')
    ci = flywheel_metrics.cluster_bootstrap_ci({str(i): [v - base] for i, v in enumerate(groups)}) if isinstance(groups, list) and groups else None
    seed_value = seed_value_clause(current['panel'].get('seed_value_monitoring'))
    if refuted:
        status, reason = 'fail', '%d consecutive blocks below baseline - %s' % (consecutive, regression)
    elif accuracy >= base + delta and ci is not None and ci['ci_low'] > 0:
        # Clause 1 (frozen panel) passes; the pre-registered pass also needs clause 2 (seed-value monitoring vs realized S).
        if seed_value['status'] == 'pass':
            status, reason = 'pass', 'accuracy >= baseline + delta with group cluster-bootstrap CI excluding 0; ' + seed_value['reason']
        elif seed_value['status'] == 'underpowered':
            status, reason = 'underpowered', 'frozen-panel clause passes; ' + seed_value['reason']
        elif seed_value['status'] == 'fail' and len(series) >= by_block:
            status, reason = 'fail', 'frozen-panel clause passes but ' + seed_value['reason'] + ' at or after block %d' % by_block
        else:
            status, reason = 'pending', 'frozen-panel clause passes; ' + seed_value['reason']
    elif accuracy >= base + delta:
        status, reason = 'underpowered', 'accuracy meets baseline + delta but no per-group values for a CI'
    elif len(series) >= by_block:
        status, reason = 'fail', 'accuracy below baseline + delta at or after block %d' % by_block
    else:
        status, reason = 'pending', 'block %d of %d' % (len(series), by_block)
    return dict(falsifier='F8', status=status, evaluable=True, reason=reason, refuted=bool(refuted), baseline=base,
                accuracy=accuracy, delta=accuracy - base, threshold_delta=delta, regression_delta=regression, ci=ci,
                seed_value_monitoring=seed_value,
                blocks=len(series), by_block=by_block, series=values, checkpoint=current.get('checkpoint_sha256'))


def seed_value_clause(monitoring, chance=0.5):
    """F8 clause 2: seed-value monitoring accuracy vs realized S on the reporting contexts > 0.5, per-context CI excluding 0.5."""
    if not isinstance(monitoring, dict):
        return dict(status='pending', reason='no parallel.seed_value_monitoring.v1 receipt joined to this checkpoint', accuracy=None, ci=None)
    accuracy, by_context = monitoring.get('nontie_pairwise_accuracy_vs_S'), monitoring.get('by_context')
    if not finite(accuracy):
        return dict(status='pending', reason='seed_value_monitoring receipt carries no finite accuracy', accuracy=None, ci=None)
    ci = flywheel_metrics.cluster_bootstrap_ci({str(i): [v - chance] for i, v in enumerate(by_context)}) \
        if isinstance(by_context, list) and by_context else None
    if accuracy > chance and ci is not None and ci['ci_low'] > 0:
        status, reason = 'pass', 'seed-value monitoring accuracy %.4f > %.1f with per-context CI excluding %.1f' % (accuracy, chance, chance)
    elif accuracy > chance and ci is None:
        status, reason = 'underpowered', 'seed-value monitoring accuracy %.4f > %.1f but no per-context values for a CI' % (accuracy, chance)
    else:
        status, reason = 'fail', 'seed-value monitoring accuracy %.4f does not exceed %.1f with CI excluding it' % (accuracy, chance)
    return dict(status=status, reason=reason, accuracy=accuracy, ci=ci, contexts=monitoring.get('contexts'),
                n_groups=monitoring.get('n_groups'), receipt_sha256=monitoring.get('receipt_sha256'))


# ---------------------------------------------------------------- outcome

def outcome(f, block, generation):
    harmful_fires = threshold(block, 'harmful_fires', optional=True)
    generations = threshold(block, 'generations')
    final = threshold(block, 'final_generation', optional=True)
    final = final if type(final) is int else max(generations)
    f2_fires, f3_fires = f['F2'].get('fires', 0) or 0, f['F3'].get('fires', 0) or 0
    fires = f2_fires + f3_fires   # the brake: F2 rejects and F3 diagnostic rejects feed the same hold
    silent = lambda name: f[name]['status'] == 'silent'   # evaluable and silent; not_evaluable is never silent
    inputs = dict(brake_fires=fires, f2_fires=f2_fires, f3_fires=f3_fires, brake_fires_for_harmful=harmful_fires,
                  final_generation=final, at_final_generation=generation >= final,
                  f2_f3_silent=silent('F2') and silent('F3'), f5_fired=f['F5']['status'] == 'fired', f8_refuted=bool(f['F8'].get('refuted')),
                  fires_clause='evaluable' if type(harmful_fires) is int else 'not evaluable: block lacks verdict.brake_fires_for_harmful')
    if type(harmful_fires) is int and fires >= harmful_fires:
        return 'harmful_stopped', inputs
    if f['F1']['status'] == 'pass' and f['F8']['status'] == 'pass' and inputs['f2_f3_silent'] and not inputs['f5_fired']:
        return 'smarter_confirmed', inputs
    if f['F1']['status'] == 'fail' and f['F8']['status'] == 'fail' and inputs['f2_f3_silent'] and inputs['at_final_generation']:
        return 'no_learning_detected', inputs
    return 'inconclusive_underpowered', inputs


def evaluate(rows, receipt_path, registration, generation, since_generation=None, f4_receipt=None):
    block = registration['block']
    generations = threshold(block, 'generations')
    require(isinstance(generations, list) and generation in generations,
            'generation %d is not a pre-registered verdict generation %s' % (generation, generations))
    receipt = read(receipt_path)
    baseline = baseline_receipt(receipt_path)
    origin, origin_source = origin_generation(receipt, since_generation)
    run_generation = origin + generation
    considered = [r['generation'] for r in rows if type(r.get('generation')) is int and origin < r['generation'] <= run_generation]
    # F4 comes from the immutable information-test receipt when one is given (the measured numbers verbatim); otherwise
    # from the series rows' successor block.
    f4 = guarded('F4', f4_from_receipt, f4_receipt, block) if f4_receipt is not None else guarded('F4', f4_information, rows, block, generation, origin)
    falsifiers = dict(F1=guarded('F1', f1_heldout, rows, receipt, block, generation, origin),
                      F2=guarded('F2', f2_brake, rows, baseline, block),
                      F3=guarded('F3', f3_diversity, rows, baseline, block),
                      F4=f4,
                      F5=f5_identity(rows), F6=f6_bridge(rows), F7=f7_consumption(rows),
                      F8=guarded('F8', f8_critic, rows, baseline, block))
    for name in ('F2', 'F3'):
        falsifiers[name] = {k: v for k, v in falsifiers[name].items() if k != 'decisions'}
    result, inputs = outcome(falsifiers, block, generation)
    require(result in OUTCOMES, 'unknown outcome')
    return clean(dict(schema=SCHEMA, generation=generation, outcome=result, outcome_inputs=inputs,
                      generations=dict(verdict=generation, origin=origin, origin_source=origin_source, run_generation=run_generation,
                                       run_range=[min(considered), max(considered)] if considered else None, rows_in_range=len(considered),
                                       rule='rows with origin < run generation <= origin + verdict generation feed F1/F4'),
                      evaluable=[n for n in FALSIFIERS if falsifiers[n]['evaluable']],
                      not_evaluable={n: falsifiers[n]['reason'] for n in FALSIFIERS if not falsifiers[n]['evaluable']},
                      statuses={n: falsifiers[n]['status'] for n in FALSIFIERS}, falsifiers=falsifiers,
                      thresholds=block, preregistration={k: v for k, v in registration.items() if k != 'block'},
                      baseline=dict(path=baseline['path'], sha256=baseline['sha256'], manifest_sha256=baseline['manifest_sha256']),
                      series=dict(rows_considered=len(rows), last_row_id=rows[-1]['id'] if rows else None),
                      outcome_rule='harmful_stopped: brake fires (F2 + F3 reject windows) >= verdict.brake_fires_for_harmful; '
                                   'smarter_confirmed: F1 pass and F8 pass with F2/F3 evaluable and silent and F5 not fired; '
                                   'no_learning_detected: F1 fail and F8 fail at the final generation with F2/F3 evaluable and silent; '
                                   'else inconclusive_underpowered. F5 trips and F8 refutation are flags (outcome_inputs), not outcomes',
                      final_tests_opened=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preregistration', type=Path, required=True, help='PREREGISTRATION.md carrying the fenced json block')
    parser.add_argument('--preregistration-sha256', help='refuse unless the block digest (canonical json or block text) equals this')
    parser.add_argument('--series', type=Path, required=True, help='flywheel-series.jsonl')
    parser.add_argument('--baseline', type=Path, required=True, help='immutable parallel.flywheel_baseline.v1 receipt')
    parser.add_argument('--generation', type=int, required=True, help='pre-registered verdict generation (5, 10 or 20), counted from the origin')
    parser.add_argument('--since-generation', type=int, help='origin: the last pre-change run generation (default: baseline generation_range.last)')
    parser.add_argument('--output', type=Path, help='default <series dir>/successor/verdicts/generation-<G>.json')
    parser.add_argument('--f4-receipt', type=Path, help='immutable %s receipt; F4 is then judged from its measured numbers' % F4_RECEIPT_SCHEMA)
    parser.add_argument('--dry-run', action='store_true', help='print the verdict without writing it')
    args = parser.parse_args()
    registration = preregistration(args.preregistration, args.preregistration_sha256)
    rows = series_rows(args.series)
    require(rows, 'flywheel series is empty or missing: ' + str(args.series))
    verdict = evaluate(rows, args.baseline, registration, args.generation, args.since_generation, f4_receipt=args.f4_receipt)
    output = args.output or args.series.resolve().parent / 'successor' / 'verdicts' / ('generation-%d.json' % args.generation)
    if args.dry_run:
        print(json.dumps(verdict, sort_keys=True))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_sha = promotion_gate.immutable(output, verdict)
    print(json.dumps(dict(verdict=str(output), sha256=receipt_sha, generation=args.generation, outcome=verdict['outcome'],
                          evaluable=verdict['evaluable'], not_evaluable=verdict['not_evaluable'])))


if __name__ == '__main__':
    main()
