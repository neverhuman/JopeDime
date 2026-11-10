import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import softvote
import score

HERE = Path(__file__).parent


def original(file, name):
    node = next(n for n in ast.parse((HERE / file).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(np=np, math=math, TAU=.1)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(HERE / file), 'exec'), namespace)
    return namespace[name]


def fixture():
    bagged = SimpleNamespace(vote_counts=original('base_bagged.py', 'vote_counts'),
                             switch_by_votes=original('base_bagged.py', 'switch_by_votes'))
    softvote.install(bagged)
    core = SimpleNamespace(select_from_top4_scores=original('base_h265_core.py', 'select_from_top4_scores'))
    veto = SimpleNamespace(mask=original('base_veto.py', 'mask'))
    calls = []
    raw = np.zeros((1, 8)); raw[0, 1] = 2.
    bags = np.zeros((8, 1, 8)); bags[:7, 0, 1] = .2; bags[7, 0, 1] = -1.
    inv = np.zeros((1, 8))

    def frozen(directory, member, X, n, k):
        calls.append((directory, n, k))
        margin = math.inf if member['strict_margin'] == 'inf' else member['strict_margin']
        bagged.vote_counts(bags, [1], margin)
        return dict(raw_scores=raw.copy(), bag_scores=bags.copy(), p_inv=inv.copy())

    G = dict(bagged=bagged, core=core, veto=veto, ens=SimpleNamespace(K=8, score_single_member_from_disk=frozen))
    return G, calls, raw, bags, inv


class ScoreRouteTests(unittest.TestCase):
    def test_temperature_applies_to_both_vote_calls(self):
        G, calls, *_ = fixture()
        out, picks, trace = score.score_member_outputs(G, '.', dict(strict_margin=.1), np.zeros((8, 568)), 1,
                                                      dict(tau=.1, k=5), np.ones(8))
        self.assertEqual(len(calls), 1)
        self.assertEqual(picks.tolist(), [0])
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0], trace[1])
        self.assertIsNone(softvote._CURRENT)

    def test_different_member_temperature_is_not_leaked(self):
        G, *_ = fixture()
        args = (G, '.', dict(strict_margin=.1), np.zeros((8, 568)), 1, dict(tau=.1, k=5))
        self.assertEqual(score.score_member_outputs(*args, np.full(8, .001))[1].tolist(), [1])
        self.assertEqual(score.score_member_outputs(*args, np.ones(8))[1].tolist(), [0])

    def test_main_gate_bag_predictions_unchanged(self):
        G, _, raw, bags, inv = fixture()
        out, _, _ = score.score_member_outputs(G, '.', dict(strict_margin=.1), np.zeros((8, 568)), 1,
                                              dict(tau=.1, k=5), np.ones(8))
        for key, original_value in [('raw_scores', raw), ('bag_scores', bags), ('p_inv', inv)]:
            np.testing.assert_array_equal(out[key], original_value)

    def test_infinity_and_invalid_temperature(self):
        G, calls, *_ = fixture()
        args = (G, '.', dict(strict_margin='inf'), np.zeros((8, 568)), 1, dict(tau=.1, k=5))
        _, picks, trace = score.score_member_outputs(*args, np.ones(8))
        self.assertEqual(picks.tolist(), [0]); self.assertEqual(trace, [])
        before = len(calls)
        with self.assertRaises(AssertionError):
            score.score_member_outputs(*args, np.zeros(8))
        self.assertEqual(len(calls), before)


if __name__ == '__main__':
    unittest.main()
