"""Scientific acceptance checks for the frozen H307 protocol."""

import unittest
import numpy as np
import reselection as a
from run_h307 import r0_replay


class H307Tests(unittest.TestCase):
    def test_tie_is_relative_to_global_max_not_chained(self):
        scores = np.array([1.0, 1.00009, 1.00018])
        rounds = np.array([10, 11, 12])
        self.assertEqual(a.earliest_max(scores, rounds), 1)
        self.assertEqual(a.earliest_max(scores[[2, 0, 1]], rounds[[2, 0, 1]]), 2)

    def test_r0_chronological_ties_never_promote(self):
        x = np.array([[1.0], [1.00009], [1.00018]])
        r = r0_replay(np.array([10, 11, 12]), x, x)
        self.assertEqual(r["n_promotions"], 1)
        self.assertEqual(r["final_incumbent_round"], 12)

    def test_mean_and_median_rank_different_spike(self):
        x = np.array([[0.0, 0.0, 15.0], [4.0, 4.0, 4.0], [3.0, 3.0, 3.0]])
        picks, scores, eb = a.select_candidates(x, np.array([100, 200, 300]), 5)
        self.assertEqual(picks[0], 0)
        self.assertEqual(picks[1], 1)
        np.testing.assert_array_equal(scores[0], x.mean(1))

    def test_lcb_matches_explicit_whole_family_resampling(self):
        x = np.array([[0.0, 2.0, 5.0], [2.0, 3.0, 1.0], [4.0, 1.0, 2.0]])
        indices = np.random.default_rng(23).integers(0, 3, (1000, 3))
        direct = x[:, indices].mean(2)
        picks, scores, eb = a.select_candidates(x, np.array([1, 2, 3]), 23)
        np.testing.assert_allclose(
            scores[2], np.quantile(direct, 0.05, axis=1), atol=1e-14
        )

    def test_shrinkage_zero_between_candidate_variance(self):
        means = np.array([1.0, 1.0001, 1.0002])
        variances = np.ones(3)
        scores, eb = a.shrunk_scores(means, variances)
        self.assertEqual(eb["tau2"], 0.0)
        np.testing.assert_array_equal(scores, np.full(3, means.mean()))
        self.assertEqual(a.earliest_max(scores, np.array([2, 1, 3])), 1)

    def test_shrinkage_known_moment_formula_and_zero_variance(self):
        means = np.array([0.0, 2.0, 4.0])
        v = np.array([0.0, 1.0, 2.0])
        scores, eb = a.shrunk_scores(means, v)
        self.assertEqual(eb["tau2"], 3.0)
        np.testing.assert_allclose(scores, 2 + 3 / (3 + v) * (means - 2))
        self.assertEqual(scores[0], 0.0)

    def test_smooth_window_is_numeric_round_not_adjacent_index(self):
        np.testing.assert_array_equal(
            a.smooth_scores(np.array([10.0, 0.0, 9.0]), np.array([100, 101, 107])),
            [5.0, 5.0, 9.0],
        )

    def test_family_equal_weights_not_context_counts(self):
        names, m = a.family_matrix(
            np.array([[1.0, 1.0, 1.0, 9.0]]), np.array(["a", "a", "a", "b"])
        )
        np.testing.assert_array_equal(names, ["a", "b"])
        self.assertEqual(m.mean(), 5.0)

    def test_balanced_splits_exact_h304_permutation_stream(self):
        splits = a.balanced_splits()
        rng = np.random.default_rng(304150)
        np.testing.assert_array_equal(splits[0][0], np.sort(rng.permutation(19)[:9]))
        self.assertEqual(len({tuple(s) for s, e in splits}), 200)
        self.assertTrue(
            all(len(s) == 9 and len(e) == 10 and not set(s) & set(e) for s, e in splits)
        )

    def test_eval_does_not_change_frozen_select_picks(self):
        rng = np.random.default_rng(99)
        full = rng.normal(size=(12, 8))
        rounds = np.arange(12)
        before = a.select_candidates(full[:, :4], rounds, 22)[0]
        full[:, 4:] = rng.normal(1000, 100, size=(12, 4))
        after = a.select_candidates(full[:, :4], rounds, 22)[0]
        np.testing.assert_array_equal(before, after)

    def test_all_ties_zero_gain_and_elo1500(self):
        m = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        results = a.eval_split(
            m, np.array([0, 1, 1, 0, 0]), np.array([0, 1]), np.array([2, 3])
        )
        for r in results.values():
            self.assertEqual(r["gain_vs_r3175_interval"]["ci95"], [0.0, 0.0])
            self.assertEqual(r["family_wins_ties_losses"], [0, 2, 0])
            self.assertEqual(r["elo_interval"]["ci95"], [1500.0, 1500.0])

    def test_mm_pairwise_solution_and_tie_invariance(self):
        expected = 1500 + 400 * np.log10(3.01 / 1.01)
        self.assertAlmostEqual(float(a.elo_mm_counts(3, 1)), expected)
        self.assertAlmostEqual(float(a.elo_mm_counts(1, 3)), 3000 - expected)
        self.assertAlmostEqual(
            a.elo_family(np.array([1.0, 1.0, 1.0, -1.0, 0.0, 0.0])), expected
        )

    def test_common_resampling_preserves_identical_eval_sets(self):
        sets = [np.array([0, 2, 3]), np.array([0, 2, 3]), np.array([1, 2])]
        weights, rejected = a.common_family_weights(4, sets, draws=100)
        np.testing.assert_array_equal(weights[0], weights[1])
        for w in weights:
            np.testing.assert_allclose(w.sum(1), 1.0)
        self.assertGreaterEqual(rejected, 0)

    def test_decision_requires_every_gate_strictly(self):
        base = dict(
            median_gain_vs_r3175=0.002, positive_lcb95_seeds=3, beats_s0_seeds=2
        )
        self.assertEqual(a.decision("S1", {"S1": base})["verdict"], "RESELECT_ADVANCES")
        for field, value in [
            ("median_gain_vs_r3175", 0.001),
            ("positive_lcb95_seeds", 2),
            ("beats_s0_seeds", 1),
        ]:
            self.assertEqual(
                a.decision("S1", {"S1": dict(base, **{field: value})})["verdict"],
                "RESELECT_NOT_ADVANCED",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
