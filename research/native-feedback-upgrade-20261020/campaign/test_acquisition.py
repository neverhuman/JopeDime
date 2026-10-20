import random
import unittest
from acquisition import parent_bootstrap_weights, thompson_pick, thompson_probabilities


class AcquisitionTests(unittest.TestCase):
    def test_marginal_distribution_includes_ensemble_and_uniform(self):
        p = thompson_probabilities([0, 1, 2], [[3, 1, 0], [2, 1, 0], [0, 3, 1]])
        self.assertAlmostEqual(sum(p.values()), 1.)
        self.assertAlmostEqual(p[0], .25 / 3 + .75 * 2 / 3)
        self.assertAlmostEqual(p[2], .25 / 3)

    def test_only_available_candidates_and_exact_recorded_propensity(self):
        scores = [[0, 3, 2], [0, 1, 4]]
        for seed in range(30):
            picked, p, member = thompson_pick([1, 2], scores, random.Random(seed))
            self.assertIn(picked, [1, 2]); self.assertIn(member, [0, 1])
            self.assertAlmostEqual(p, thompson_probabilities([1, 2], scores)[picked])

    def test_nonfinite_or_exhausted_pool_refused(self):
        with self.assertRaises(ValueError): thompson_probabilities([], [[1]])
        with self.assertRaises(ValueError): thompson_probabilities([0], [[float('nan')]])

    def test_parent_bootstrap_is_same_across_order_and_repeat(self):
        a = parent_bootstrap_weights(['a', 'b', 'a'], 41)
        self.assertEqual(a, parent_bootstrap_weights(['b', 'a'], 41))
        self.assertTrue(all(type(v) is int and v >= 0 for v in a.values()))

    def test_tied_scores_do_not_invent_random_priority(self):
        p = thompson_probabilities([1, 0], [[1, 1]])
        self.assertAlmostEqual(p[0], .875)


if __name__ == '__main__': unittest.main()
