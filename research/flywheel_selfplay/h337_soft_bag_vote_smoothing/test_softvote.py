import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import softvote as sv

BASE = Path(__file__).with_name('base_bagged.py').read_text()
NODE = next(n for n in ast.parse(BASE).body if isinstance(n, ast.FunctionDef) and n.name == 'vote_counts')
SOURCE = ast.get_source_segment(BASE, NODE)
TREE, RECEIPT = sv.transform(SOURCE)
NS = dict(np=np, math=math, _h337_indicator=sv.indicator)
exec(compile(TREE, '<soft-fixture>', 'exec'), NS)
soft = NS['vote_counts']
NS_BASE = dict(np=np, math=math)
exec(compile(ast.Module(body=[NODE], type_ignores=[]), '<hard-fixture>', 'exec'), NS_BASE)
hard = NS_BASE['vote_counts']


def bags(delta):
    result = np.zeros((8, 1, 2))
    result[:, 0, 1] = delta
    return result


class SoftVoteTests(unittest.TestCase):
    def test_exact_single_node_inverse(self):
        self.assertTrue(RECEIPT['inverse_exact'])
        self.assertEqual(RECEIPT['replacement_nodes'], 1)
        self.assertIn('dtype=np.int32', RECEIPT['source'])
        self.assertIn('int(np.sum(', RECEIPT['source'])

    def test_zero_centered_margin_is_half_vote(self):
        with sv.temperatures(np.ones(8)) as trace:
            self.assertEqual(soft(bags(np.full(8, .2)), [1], .2).tolist(), [4])
            self.assertEqual(trace, [4.])
        self.assertEqual(hard(bags(np.full(8, .2)), [1], .2).tolist(), [0])

    def test_full_strict_margin_is_subtracted(self):
        with sv.temperatures(np.ones(8)):
            got = sv.indicator(np.full(8, .7), .2)
        np.testing.assert_allclose(got, np.full(8, 1 / (1 + math.exp(-.5))), rtol=0, atol=1e-15)

    def test_bag_temperatures_are_individual(self):
        t = np.arange(1, 9) * .1
        with sv.temperatures(t):
            got = sv.indicator(np.ones(8), .1)
        np.testing.assert_allclose(got, [1 / (1 + math.exp(-.9 / x)) for x in t], rtol=0, atol=1e-15)

    def test_saturation_has_no_overflow(self):
        with sv.temperatures(np.ones(8)):
            np.testing.assert_array_equal(sv.indicator(np.array([-1e6, 1e6] * 4), 0), [0, 1] * 4)

    def test_infinity_and_slot0_preserve_early_returns(self):
        with sv.temperatures(np.ones(8)) as trace:
            self.assertEqual(soft(bags(np.ones(8)), [1], math.inf).tolist(), [0])
            self.assertEqual(soft(bags(np.ones(8)), [0], 0).tolist(), [0])
            self.assertEqual(trace, [])

    def test_floor_sum_is_equivalent_to_integer_k(self):
        for k in [5, 6, 7]:
            for value in [0., np.nextafter(float(k), 0.), float(k), np.nextafter(float(k), 8.), 8.]:
                self.assertEqual(int(value) >= k, value >= k)

    def test_temperature_population_and_factor(self):
        x = np.arange(28, dtype=np.float64).reshape(4, 7)
        expected = .5 * math.sqrt(sum((v - 13.5)**2 for v in range(28)) / 28)
        self.assertEqual(sv.temperature_from_deltas(x), expected)
        self.assertEqual(sv.temperature_from_deltas(x + 100.), expected)

    def test_degenerate_and_nonfinite_temperatures_fail(self):
        for data in [np.zeros((3, 7)), np.full((3, 7), np.nan)]:
            with self.assertRaises(AssertionError):
                sv.temperature_from_deltas(data)
        for t in [np.zeros(8), np.full(8, -1.), np.full(8, np.inf), np.ones(7)]:
            with self.assertRaises(AssertionError):
                with sv.temperatures(t):
                    pass

    def test_unbound_indicator_fails_and_scope_clears(self):
        with self.assertRaises(AssertionError):
            sv.indicator(np.ones(8), .1)
        with sv.temperatures(np.ones(8)):
            pass
        self.assertIsNone(sv._CURRENT)

    def test_candidate_delta_controls_only_its_vote(self):
        with sv.temperatures(np.ones(8)):
            a = sv.indicator(np.zeros(8), 0)
            b = sv.indicator(np.array([100.] + [0.] * 7), 0)
        np.testing.assert_array_equal(a[1:], b[1:])
        self.assertAlmostEqual(float(b.sum() - a.sum()), .5)

    def test_call_site_mismatch_refused(self):
        with self.assertRaises(AssertionError):
            sv.transform(SOURCE.replace('> margin', '>= margin'))


if __name__ == '__main__':
    unittest.main()
