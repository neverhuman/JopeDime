"""Data-free checks of the exact pinned BASE path; no model imports or scores."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).parent


def node(file, name):
    tree = ast.parse((HERE / file).read_text())
    return next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)


def function(file, name):
    original = node(file, name)
    namespace = dict(np=np, math=math, TAU=.1)
    exec(compile(ast.Module(body=[original], type_ignores=[]), file, 'exec'), namespace)
    return namespace[name]


vote = function('base_bagged.py', 'vote_counts')
switch = function('base_bagged.py', 'switch_by_votes')
consensus = function('base_ensemble.py', 'vote_consensus_pick')
member = function('base_run_h303.py', 'member_picks')
select = function('base_h265_core.py', 'select_from_top4_scores')
mask = function('base_veto.py', 'mask')

REACHABLE = {
    'base_run_h303.py': ['score_members', 'member_picks', 'ens_picks'],
    'base_ensemble.py': ['score_single_member_from_disk', 'vote_consensus_pick'],
    'base_bagged.py': ['vote_counts', 'switch_by_votes'],
    'base_h265_core.py': ['select_from_top4_scores'],
    'base_veto.py': ['mask'],
}


def scores(positive, magnitude=1.):
    out = np.zeros((8, 1, 2))
    out[:, 0, 1] = -1
    out[:positive, 0, 1] = magnitude
    return out


def pick(bags, q=6, p_invalid=0.):
    out = dict(_strict_margin=.1, raw_scores=np.array([[0., 2.]]),
               p_inv=np.array([[0., p_invalid]]), bag_scores=bags)
    return member(out, .1, q, SimpleNamespace(mask=mask),
                  SimpleNamespace(select_from_top4_scores=select),
                  SimpleNamespace(vote_counts=vote, switch_by_votes=switch))


class SourceContract(unittest.TestCase):
    def test_pinned_sources(self):
        pins = json.loads((HERE / 'source_pins.json').read_text())
        for file, pin in pins.items():
            self.assertEqual(hashlib.sha256((HERE / file).read_bytes()).hexdigest(), pin['sha256'])

    def test_no_raw_mean_in_reachable_path(self):
        for file, names in REACHABLE.items():
            for name in names:
                calls = [ast.unparse(x.func) for x in ast.walk(node(file, name)) if isinstance(x, ast.Call)]
                self.assertFalse(any(x.split('.')[-1] in {'mean', 'average', 'median'} for x in calls), (file, name, calls))

    def test_full_bag_tensor_reaches_vote(self):
        source = ast.unparse(node('base_ensemble.py', 'score_single_member_from_disk'))
        self.assertIn('np.stack(bag_scores, axis=0)', source)
        self.assertIn('bagged.vote_counts(bag_scores_arr, j_ref, strict_margin)', source)
        self.assertIn("bagged.vote_counts(np.asarray(out['bag_scores']), j_ref, margin)", ast.unparse(node('base_run_h303.py', 'member_picks')))

    def test_other_mean_helper_is_separate(self):
        helper = ast.unparse(node('base_ensemble.py', 'aggregate_ensemble_picks'))
        self.assertIn("m['raw_scores']", helper)
        self.assertIn('np.mean', helper)
        for name in REACHABLE['base_run_h303.py']:
            self.assertNotIn('aggregate_ensemble_picks', ast.unparse(node('base_run_h303.py', name)))

    def test_hot_bag_is_only_one_vote(self):
        for magnitude in [1., 1e3, 1e12]:
            self.assertEqual(vote(scores(1, magnitude), [1], .1).tolist(), [1])
            self.assertEqual(pick(scores(1, magnitude)), [0])

    def test_q_thresholds_are_counts(self):
        for q in [5, 6, 7]:
            self.assertEqual(pick(scores(q - 1), q), [0])
            self.assertEqual(pick(scores(q), q), [1])

    def test_strict_margin_equality(self):
        self.assertEqual(vote(scores(8, .1), [1], .1).tolist(), [0])
        self.assertEqual(vote(scores(8, np.nextafter(.1, 1.)), [1], .1).tolist(), [8])

    def test_infinite_margin_and_slot_zero(self):
        self.assertEqual(vote(scores(8), [1], math.inf).tolist(), [0])
        self.assertEqual(vote(scores(8), [0], 0.).tolist(), [0])

    def test_bag_permutation_and_offsets(self):
        bags = scores(6)
        shifted = bags + np.arange(8)[:, None, None] * 100.
        np.testing.assert_array_equal(vote(bags, [1], .1), vote(shifted[::-1], [1], .1))

    def test_main_veto_and_member_consensus(self):
        self.assertEqual(pick(scores(8), p_invalid=.1), [0])
        self.assertEqual(pick(scores(8), p_invalid=.09), [1])
        self.assertEqual(consensus([1, 1, 0], 2), 1)
        self.assertEqual(consensus([1, 1, 0], 3), 0)

    def test_mean_dominance_is_not_base_behavior(self):
        bags = scores(1, 1000.)
        self.assertGreater(float(bags.mean(axis=0)[0, 1]), .1)
        self.assertLess(float(np.median(bags, axis=0)[0, 1]), .1)
        self.assertEqual(pick(bags), [0])

    def test_median_broadcast_changes_unchanged_k_rule(self):
        bags = scores(5)
        broadcast = np.repeat(np.median(bags, axis=0, keepdims=True), 8, axis=0)
        self.assertEqual([pick(bags, q)[0] for q in [5, 6, 7]], [1, 0, 0])
        self.assertEqual([pick(broadcast, q)[0] for q in [5, 6, 7]], [1, 1, 1])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SourceContract))
    report = dict(status='PASS' if result.wasSuccessful() else 'FAIL', collected=result.testsRun,
                  executed=result.testsRun, failed=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  scope='Data-free source/selection fixtures only; no model or scientific data access',
                  scientific_regression='NOT_RUN', model_hash_audit='NOT_RUN', medbag_scoring='NOT_RUN')
    (HERE / 'tests_receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
