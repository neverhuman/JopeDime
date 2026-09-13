#!/usr/bin/env python3
"""Promotion tournament: the F10 verifier and statistic over parallel.promotion_evidence.v2.

A learning checkpoint (challenger) is compared with the current serving checkpoint on the 16 meta_validation cuts
(primary) and a rotating 32-context meta_train subset (secondary): proposals sampled from each checkpoint with three
matched rng seeds, each proposal fitted directly once (row 0, ordinary evaluator, 30 s deadline) as GP qualification
jobs whose process receipts bind evaluator, program, slot and deadline. This tool never runs anything: it validates the
evidence file the native producer wrote, recomputes every aggregate from the raw fits, applies the pre-registered rule
F10 from PREREGISTRATION.md (thresholds are read from the block, never from this file) and writes an immutable
parallel.promotion_decision.v2. Codex's native verifier must produce byte-identical decisions on the shared fixture.

Evidence (parallel.promotion_evidence.v2):
  run_identity, manifest_sha256, evaluator_sha256, resource_protocol, fit_deadline_seconds, rng_seeds[3],
  challenger{checkpoint, sha256, step, monitor_receipt_sha256, diagnostic}, serving{checkpoint, sha256, step, ...},
  contexts[{context_key, dataset, family, partition, cut, prepared_sha256, role: held_out|training_subset}],
  outcomes[{arm: challenger|serving|shuffled, context_key, seed_index, proposals, fits[{program_sha256, rmse|null,
  failure|null, receipt_sha256}], best_initial|null, failures}], consistency{generation, program_sha256,
  in_search_rmse_bits, direct_rmse_bits, equal}, final_tests_opened false.

Statistic (F10): per held-out cut, mean over seeds of ln(best_initial_challenger / best_initial_serving); wins/ties/
losses by cut (tie when |ln ratio| <= 1e-12); cluster bootstrap over cuts (flywheel_metrics.cluster_bootstrap_ci with the
block's resamples/seed/alpha). accepted iff mean < 0 AND wins >= losses AND challenger diagnostic_accepted AND challenger
failures <= serving failures; a held-out tie (wins == losses) is decided by the training subset's CI90 excluding 0 in
the challenger's favour. Anything malformed is `inconclusive` with reasons; nothing here is an improvement claim.
"""
import argparse
import json
import math
from pathlib import Path
import re

import flywheel_metrics
from flywheel_verdict import lookup, preregistration
import promotion_gate
from resume_expired_cycle import read, require, sha

SCHEMA = 'parallel.promotion_decision.v2'
EVIDENCE_SCHEMA = 'parallel.promotion_evidence.v2'
ARMS = ('challenger', 'serving', 'shuffled')
ROLES = ('held_out', 'training_subset')
OUTCOMES = ('accepted', 'rejected', 'inconclusive')
HELD_OUT_CUTS = 16
SEEDS = 3
TIE = 1e-12
HEX64 = re.compile(r'^[0-9a-f]{64}$')
RULE = 'falsifiers.F10'
CLAIM = 'a promotion is an operational decision under F10; it is never an improvement claim (F1/F8 are the only such claims)'


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def hex64(value):
    return isinstance(value, str) and bool(HEX64.match(value))


# ---------------------------------------------------------------- validation

def validate_evidence(path):
    where = str(path)
    value = read(path)
    require(isinstance(value, dict) and value.get('schema') == EVIDENCE_SCHEMA, where + ': schema is not ' + EVIDENCE_SCHEMA)
    require(value.get('final_tests_opened') is False, where + ': final_tests_opened is not false')
    for key in ('manifest_sha256', 'evaluator_sha256'):
        require(hex64(value.get(key)), where + ': %s is not a 64-hex digest' % key)
    require(isinstance(value.get('resource_protocol'), str) and value['resource_protocol'], where + ': resource_protocol missing')
    require(finite(value.get('fit_deadline_seconds')) and value['fit_deadline_seconds'] > 0, where + ': fit_deadline_seconds missing')
    seeds = value.get('rng_seeds')
    require(isinstance(seeds, list) and len(seeds) == SEEDS and len(set(map(str, seeds))) == SEEDS, where + ': rng_seeds must list %d distinct seeds' % SEEDS)
    for arm in ('challenger', 'serving'):
        entry = value.get(arm)
        require(isinstance(entry, dict) and isinstance(entry.get('checkpoint'), str) and hex64(entry.get('sha256')) and type(entry.get('step')) is int,
                where + ': %s lacks checkpoint/sha256/step' % arm)
    require(value['challenger']['sha256'] != value['serving']['sha256'], where + ': challenger and serving are one checkpoint')
    require(hex64(value['challenger'].get('monitor_receipt_sha256')), where + ': challenger monitor_receipt_sha256 missing')
    require(value['challenger'].get('diagnostic') in ('diagnostic_accepted', 'diagnostic_rejected'), where + ': challenger diagnostic must be diagnostic_accepted or diagnostic_rejected')
    contexts = value.get('contexts')
    require(isinstance(contexts, list) and contexts, where + ': contexts empty')
    by_key = {}
    for index, context in enumerate(contexts):
        require(isinstance(context, dict) and isinstance(context.get('context_key'), str) and context.get('role') in ROLES
                and isinstance(context.get('partition'), str) and isinstance(context.get('family'), str), where + ': contexts[%d] malformed' % index)
        require(context['context_key'] not in by_key, where + ': duplicate context %s' % context['context_key'])
        expected_partition = 'meta_validation' if context['role'] == 'held_out' else 'meta_train'
        require(context['partition'] == expected_partition, where + ': context %s role %s but partition %s' % (context['context_key'], context['role'], context['partition']))
        by_key[context['context_key']] = context
    held = [c for c in by_key.values() if c['role'] == 'held_out']
    require(len(held) == HELD_OUT_CUTS, where + ': %d held-out contexts, %d required' % (len(held), HELD_OUT_CUTS))
    families = {}
    for c in held:
        families.setdefault(c['family'], 0)
        families[c['family']] += 1
    require(len(families) == 8 and all(n == 2 for n in families.values()), where + ': held-out coverage must be 8 families x 2 cuts')
    outcomes = value.get('outcomes')
    require(isinstance(outcomes, list) and outcomes, where + ': outcomes empty')
    table = {}
    for index, outcome in enumerate(outcomes):
        prefix = '%s outcomes[%d]' % (where, index)
        require(isinstance(outcome, dict) and outcome.get('arm') in ARMS and outcome.get('context_key') in by_key
                and type(outcome.get('seed_index')) is int and 0 <= outcome['seed_index'] < SEEDS, prefix + ': arm/context/seed malformed')
        fits = outcome.get('fits')
        require(isinstance(fits, list) and fits and type(outcome.get('proposals')) is int and outcome['proposals'] >= len(fits), prefix + ': fits/proposals malformed')
        scores, failures = [], 0
        for j, fit in enumerate(fits):
            require(isinstance(fit, dict) and hex64(fit.get('program_sha256')) and hex64(fit.get('receipt_sha256')) and 'failure' in fit,
                    prefix + ' fits[%d]: program/receipt/failure missing' % j)
            if fit['failure'] is None:
                require(finite(fit.get('rmse')) and fit['rmse'] >= 0, prefix + ' fits[%d]: rmse not finite' % j)
                scores.append(fit['rmse'])
            else:
                require(fit.get('rmse') is None, prefix + ' fits[%d]: failed fit carries an rmse' % j)
                failures += 1
        best = min(scores) if scores else None
        require(outcome.get('best_initial') == best, prefix + ': best_initial %r is not the minimum finite fit %r' % (outcome.get('best_initial'), best))
        require(outcome.get('failures') == failures, prefix + ': failures %r differ from the %d failed fits' % (outcome.get('failures'), failures))
        key = (outcome['arm'], outcome['context_key'], outcome['seed_index'])
        require(key not in table, prefix + ': duplicate (arm, context, seed)')
        table[key] = dict(best=best, failures=failures, fits=len(fits))
    for arm in ('challenger', 'serving'):
        for context in by_key.values():
            for seed in range(SEEDS):
                require((arm, context['context_key'], seed) in table, '%s: missing outcome for %s on %s seed %d' % (where, arm, context['context_key'], seed))
    consistency = value.get('consistency')
    require(isinstance(consistency, dict) and consistency.get('equal') is True and hex64(consistency.get('program_sha256')),
            where + ': consistency (F5-style bit-equality of the challenger direct fit with the live in-search observation) not established')
    return dict(path=where, sha256=sha(path), value=value, contexts=by_key, table=table)


# ---------------------------------------------------------------- statistic

def cut_rows(evidence, role):
    """Per-context rows: mean over seeds of ln(challenger / serving) when both arms fitted finitely in that seed."""
    rows = []
    for key, context in evidence['contexts'].items():
        if context['role'] != role:
            continue
        ratios = []
        for seed in range(SEEDS):
            a = evidence['table'][('challenger', key, seed)]['best']
            b = evidence['table'][('serving', key, seed)]['best']
            if a is not None and b is not None and a > 0 and b > 0:
                ratios.append(math.log(a / b))
        if not ratios:
            rows.append(dict(context_key=key, dataset=context.get('dataset'), cut=context.get('cut'), family=context['family'], seeds=0, mean_log_ratio=None))
            continue
        mean = sum(ratios) / len(ratios)
        rows.append(dict(context_key=key, dataset=context.get('dataset'), cut=context.get('cut'), family=context['family'], seeds=len(ratios),
                         log_ratios=ratios, mean_log_ratio=mean, result='tie' if abs(mean) <= TIE else ('win' if mean < 0 else 'loss')))
    return rows


def contrast(rows, statistics_block):
    scored = [r for r in rows if r.get('mean_log_ratio') is not None]
    if not scored:
        return dict(n_cuts=len(rows), scored=0, wins=0, ties=0, losses=0, mean_log_ratio=None, ci=None, reason='no cut with finite fits in both arms')
    values = [r['mean_log_ratio'] for r in scored]
    # clusters = cuts; each cluster's values are its per-seed log-ratios (the cluster mean is the per-cut mean above)
    ci = flywheel_metrics.cluster_bootstrap_ci({r['context_key']: r['log_ratios'] for r in scored}, resamples=statistics_block['resamples'],
                                               seed=statistics_block['seed'], alpha=statistics_block['alpha'])
    return dict(n_cuts=len(rows), scored=len(scored), wins=sum(r['result'] == 'win' for r in scored), ties=sum(r['result'] == 'tie' for r in scored),
                losses=sum(r['result'] == 'loss' for r in scored), mean_log_ratio=sum(values) / len(values), ci=[ci['ci_low'], ci['ci_high']],
                bootstrap=ci, per_cut=scored, unscored=[r['context_key'] for r in rows if r.get('mean_log_ratio') is None])


def failures(evidence, arm, role):
    return sum(v['failures'] for (a, key, _), v in evidence['table'].items() if a == arm and evidence['contexts'][key]['role'] == role)


def ci_low_high(ci):
    if isinstance(ci, dict):
        return ci.get('ci_low'), ci.get('ci_high')
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        return ci[0], ci[1]
    return None, None


def decide(evidence, block):
    rule = lookup(block, RULE)
    require(isinstance(rule, dict) and rule.get('accepted_iff'), 'preregistration block lacks ' + RULE)
    stats = lookup(block, 'statistics') or {}
    statistics_block = dict(resamples=stats.get('resamples', 2000), seed=stats.get('seed', 20260912), alpha=stats.get('alpha', 0.10))
    held = contrast(cut_rows(evidence, 'held_out'), statistics_block)
    training = contrast(cut_rows(evidence, 'training_subset'), statistics_block)
    value = evidence['value']
    challenger_failures, serving_failures = failures(evidence, 'challenger', 'held_out'), failures(evidence, 'serving', 'held_out')
    reasons = []
    if value['challenger']['diagnostic'] != 'diagnostic_accepted':
        reasons.append('challenger frozen-panel monitor is %s' % value['challenger']['diagnostic'])
    if challenger_failures > serving_failures:
        reasons.append('challenger direct failures %d exceed serving failures %d' % (challenger_failures, serving_failures))
    if held['mean_log_ratio'] is None:
        reasons.append('held-out statistic undefined: ' + held.get('reason', ''))
    else:
        if not held['mean_log_ratio'] < 0:
            reasons.append('held-out mean log-ratio %.6f is not below 0' % held['mean_log_ratio'])
        if held['wins'] < held['losses']:
            reasons.append('held-out wins %d < losses %d' % (held['wins'], held['losses']))
        elif held['wins'] == held['losses']:
            low, high = ci_low_high(training.get('ci'))
            if training['mean_log_ratio'] is None or high is None or not (high < 0):
                reasons.append('held-out tie (%d/%d/%d) and the training-subset CI90 %s does not exclude 0 in the challenger\'s favour'
                               % (held['wins'], held['ties'], held['losses'], [low, high]))
    outcome = 'accepted' if not reasons else 'rejected'
    inputs = dict(held_out=dict(wins=held['wins'], ties=held['ties'], losses=held['losses'], mean_log_ratio=held['mean_log_ratio'], ci90=held.get('ci'), n_cuts=held['n_cuts'], scored=held['scored']),
                  training_subset=dict(wins=training['wins'], ties=training['ties'], losses=training['losses'], mean_log_ratio=training['mean_log_ratio'], ci90=training.get('ci'), n_cuts=training['n_cuts']),
                  failures=dict(challenger=challenger_failures, serving=serving_failures), challenger_diagnostic=value['challenger']['diagnostic'], statistics=statistics_block)
    return outcome, reasons, inputs, held, training


def evaluate(evidence, registration, serving_before_sha256=None):
    block = registration['block']
    value = evidence['value']
    if serving_before_sha256 is not None:
        require(value['serving']['sha256'] == serving_before_sha256, 'evidence serving checkpoint %s... is not the current serving %s...' % (value['serving']['sha256'][:12], serving_before_sha256[:12]))
    outcome, reasons, inputs, held, training = decide(evidence, block)
    require(outcome in OUTCOMES, 'unknown outcome')
    rule = lookup(block, RULE)
    return dict(schema=SCHEMA, outcome=outcome, reasons=reasons, rule=RULE, rule_text=dict(accepted_iff=rule.get('accepted_iff'), otherwise=rule.get('otherwise')),
                preregistration_sha256=registration['sha256'], preregistration={k: registration.get(k) for k in ('path', 'document_sha256', 'block_bytes_sha256', 'reader')},
                evidence=dict(path=evidence['path'], sha256=evidence['sha256']), run_identity=value.get('run_identity'), manifest_sha256=value['manifest_sha256'],
                evaluator_sha256=value['evaluator_sha256'], resource_protocol=value['resource_protocol'], fit_deadline_seconds=value['fit_deadline_seconds'],
                rng_seeds=value['rng_seeds'], challenger=value['challenger'], serving=value['serving'], serving_before_sha256=value['serving']['sha256'],
                inputs=inputs, held_out_per_cut=held.get('per_cut', []), training_subset_per_cut=training.get('per_cut', []),
                unscored=dict(held_out=held.get('unscored', []), training_subset=training.get('unscored', [])),
                consistency=value.get('consistency'), claim=CLAIM, independent_evaluation=False, final_tests_opened=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--preregistration-sha256')
    parser.add_argument('--evidence', type=Path, required=True, help=EVIDENCE_SCHEMA + ' file written by the native producer')
    parser.add_argument('--serving-before', help='sha256 of the current serving checkpoint; the evidence must name it')
    parser.add_argument('--output', type=Path, help='immutable %s (required unless --dry-run)' % SCHEMA)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    require(args.output is not None or args.dry_run, '--output is required unless --dry-run')
    registration = preregistration(args.preregistration, args.preregistration_sha256)
    decision = evaluate(validate_evidence(args.evidence), registration, args.serving_before)
    if args.dry_run:
        print(json.dumps(decision, sort_keys=True))
        return decision
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = promotion_gate.immutable(args.output, decision)
    print(json.dumps(dict(decision=str(args.output), sha256=digest, outcome=decision['outcome'], reasons=decision['reasons'], held_out=decision['inputs']['held_out'])))
    return decision


if __name__ == '__main__':
    main()
