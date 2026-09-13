import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import promotion_tournament as pt
from resume_expired_cycle import atomic, sha

TOOLS = Path(__file__).resolve().parent
PREREGISTRATION = TOOLS.parent / 'recovery/eight-a100-20260912/PREREGISTRATION.md'


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def contexts(training=4):
    out = []
    for f in range(8):
        for c in range(2):
            out.append(dict(context_key='%s:%d' % (h('val%d' % f), c), dataset='val%d' % f, family='family%d' % f, partition='meta_validation', cut=c,
                            prepared_sha256=h('val%d' % f), role='held_out'))
    for i in range(training):
        out.append(dict(context_key='%s:0' % h('train%d' % i), dataset='train%d' % i, family='tfam%d' % (i % 3), partition='meta_train', cut=0,
                        prepared_sha256=h('train%d' % i), role='training_subset'))
    return out


def outcome(arm, context, seed, scores, failures=0):
    fits = [dict(program_sha256=h('%s%s%d%d' % (arm, context['context_key'], seed, j)), rmse=s, failure=None, receipt_sha256=h('r%s%s%d%d' % (arm, context['context_key'], seed, j)))
            for j, s in enumerate(scores)]
    fits += [dict(program_sha256=h('f%s%s%d%d' % (arm, context['context_key'], seed, j)), rmse=None, failure='candidate_or_view_timeout', receipt_sha256=h('fr%d%d' % (seed, j)))
             for j in range(failures)]
    return dict(arm=arm, context_key=context['context_key'], seed_index=seed, proposals=64, fits=fits, best_initial=min(scores) if scores else None, failures=failures)


def evidence(challenger_factor=0.9, training_factor=0.9, held_scores=None, training=4, challenger_diagnostic='diagnostic_accepted', challenger_failures=0, serving_failures=0):
    """Serving best_initial 1.0 on every cut; challenger = factor * 1.0 (per-cut override via held_scores[index])."""
    ctx = contexts(training)
    outcomes = []
    for index, context in enumerate(ctx):
        for seed in range(3):
            base = 1.0 + 0.01 * seed
            outcomes.append(outcome('serving', context, seed, [base, base * 1.2], serving_failures))
            factor = (held_scores[index] if held_scores and index < len(held_scores) else challenger_factor) if context['role'] == 'held_out' else training_factor
            outcomes.append(outcome('challenger', context, seed, [base * factor, base * factor * 1.3], challenger_failures))
    return dict(schema=pt.EVIDENCE_SCHEMA, run_identity='fixture', manifest_sha256=h('manifest'), evaluator_sha256=h('evaluator'), resource_protocol='single_candidate_2s_8gib_v1',
                fit_deadline_seconds=30, rng_seeds=[1, 2, 3], challenger=dict(checkpoint='learning/cycle-1/step-69000', sha256=h('challenger'), step=69000, monitor_receipt_sha256=h('monitor'),
                                                                                diagnostic=challenger_diagnostic),
                serving=dict(checkpoint='old/step-3000', sha256=h('serving'), step=3000), contexts=ctx, outcomes=outcomes,
                consistency=dict(generation=37, program_sha256=h('selected'), in_search_rmse_bits='3ff0000000000000', direct_rmse_bits='3ff0000000000000', equal=True),
                final_tests_opened=False)


def run(directory, doc, serving_before=None):
    path = Path(directory) / 'evidence.json'
    atomic(path, doc)
    return pt.evaluate(pt.validate_evidence(path), pt.preregistration(PREREGISTRATION), serving_before)


class Decisions(unittest.TestCase):
    def test_challenger_better_everywhere_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            decision = run(directory, evidence(), serving_before=h('serving'))
        self.assertEqual((decision['schema'], decision['outcome'], decision['reasons']), (pt.SCHEMA, 'accepted', []))
        held = decision['inputs']['held_out']
        self.assertEqual((held['wins'], held['ties'], held['losses'], held['n_cuts']), (16, 0, 0, 16))
        self.assertAlmostEqual(held['mean_log_ratio'], -0.1053605, places=6)   # ln(0.9)
        self.assertLess(held['ci90'][1], 0)
        self.assertEqual(decision['inputs']['training_subset']['wins'], 4)
        self.assertEqual(decision['inputs']['failures'], dict(challenger=0, serving=0))
        self.assertEqual(decision['rule'], 'falsifiers.F10')
        self.assertIn('wins >= losses', decision['rule_text']['accepted_iff'])
        self.assertIs(decision['final_tests_opened'], False)
        self.assertEqual(len(decision['held_out_per_cut']), 16)
        self.assertEqual(decision['preregistration_sha256'], pt.preregistration(PREREGISTRATION)['sha256'])

    def test_worse_challenger_is_rejected_with_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            decision = run(directory, evidence(challenger_factor=1.1))
        self.assertEqual(decision['outcome'], 'rejected')
        self.assertTrue(any('not below 0' in r for r in decision['reasons']))
        self.assertTrue(any('wins 0 < losses 16' in r for r in decision['reasons']))

    def test_monitor_rejection_and_failures_block_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            decision = run(directory, evidence(challenger_diagnostic='diagnostic_rejected'))
        self.assertEqual((decision['outcome'], decision['reasons']), ('rejected', ['challenger frozen-panel monitor is diagnostic_rejected']))
        with tempfile.TemporaryDirectory() as directory:
            decision = run(directory, evidence(challenger_failures=2, serving_failures=1))
        self.assertEqual(decision['outcome'], 'rejected')
        self.assertIn('challenger direct failures 96 exceed serving failures 48', decision['reasons'][0])

    def test_held_out_tie_is_decided_by_the_training_subset(self):
        # 8 wins / 8 losses on held-out with a negative mean; the training subset decides
        scores = [0.8] * 8 + [1.1] * 8
        with tempfile.TemporaryDirectory() as directory:
            better = run(directory, evidence(held_scores=scores, training_factor=0.9))
        self.assertEqual((better['outcome'], better['inputs']['held_out']['wins'], better['inputs']['held_out']['losses']), ('accepted', 8, 8))
        with tempfile.TemporaryDirectory() as directory:
            worse = run(directory, evidence(held_scores=scores, training_factor=1.05))
        self.assertEqual(worse['outcome'], 'rejected')
        self.assertTrue(any('held-out tie (8/0/8)' in r for r in worse['reasons']))

    def test_exact_ties_count_as_ties_not_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            decision = run(directory, evidence(challenger_factor=1.0))
        self.assertEqual((decision['outcome'], decision['inputs']['held_out']['ties']), ('rejected', 16))


class Refusals(unittest.TestCase):
    def refuse(self, doc, fragment, serving_before=None):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError) as caught:
                run(directory, doc, serving_before)
        self.assertIn(fragment, str(caught.exception))

    def test_shape_and_consistency_refusals(self):
        doc = evidence()
        self.refuse(dict(doc, schema='other'), 'schema is not')
        self.refuse(dict(doc, consistency=dict(doc['consistency'], equal=False)), 'consistency')
        self.refuse(dict(doc, rng_seeds=[1, 1, 2]), 'distinct seeds')
        self.refuse(dict(doc, contexts=doc['contexts'][1:]), 'held-out contexts')
        bad = copy.deepcopy(doc)
        bad['outcomes'][0]['best_initial'] = 0.0
        self.refuse(bad, 'is not the minimum finite fit')
        bad = copy.deepcopy(doc)
        bad['outcomes'][0]['fits'][0]['rmse'] = None
        self.refuse(bad, 'rmse not finite')
        bad = copy.deepcopy(doc)
        del bad['outcomes'][0]
        self.refuse(bad, 'missing outcome')
        bad = copy.deepcopy(doc)
        bad['contexts'][0]['partition'] = 'meta_train'
        self.refuse(bad, 'role held_out but partition meta_train')
        bad = copy.deepcopy(doc)
        bad['serving']['sha256'] = bad['challenger']['sha256']
        self.refuse(bad, 'one checkpoint')
        self.refuse(doc, 'is not the current serving', serving_before='0' * 64)
        self.refuse(dict(doc, final_tests_opened=True), 'final_tests_opened')


class Cli(unittest.TestCase):
    def test_immutable_decision_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.json'
            atomic(path, evidence())
            output = Path(directory) / 'decisions' / 'd.json'
            argv = ['--preregistration', str(PREREGISTRATION), '--evidence', str(path), '--serving-before', h('serving'), '--output', str(output)]
            first = pt.main(argv)
            self.assertEqual(first['outcome'], 'accepted')
            first_bytes = output.read_bytes()
            self.assertEqual(json.loads(first_bytes), first)
            pt.main(argv)
            self.assertEqual(output.read_bytes(), first_bytes)
            self.assertEqual(json.loads(first_bytes)['evidence']['sha256'], sha(path))
            dry = pt.main(['--preregistration', str(PREREGISTRATION), '--evidence', str(path), '--dry-run'])
            self.assertEqual(dry['outcome'], 'accepted')
            atomic(path, evidence(challenger_factor=1.2))
            with self.assertRaises(RuntimeError):
                pt.main(argv)   # a different decision may not overwrite the immutable one


if __name__ == '__main__':
    unittest.main()
