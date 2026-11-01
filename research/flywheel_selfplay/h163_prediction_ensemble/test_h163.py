"""Data-free tests for H163 score direction, ID alignment, ties and gates."""

import unittest
from pathlib import Path
import numpy as np
import analysis


class H163Tests(unittest.TestCase):
    def test_argmin_and_measured_utility_direction_are_separate(self):
        costs = np.array([[4.0, 1.0, 2.0]])
        self.assertEqual(analysis.choose_indices(costs).tolist(), [1])

    def test_alignment_preserves_first_model_native_order(self):
        ids = [["a", "b", "c"], ["c", "a", "b"], ["a", "b", "c"], ["b", "c", "a"]]
        vals = [np.array([[1., 2., 3.]])] * 4
        aligned, aligned_ids = analysis.aligned_scores(vals, ids)
        self.assertEqual(aligned_ids, ids[0])
        np.testing.assert_array_equal(aligned[1, 0], [2., 3., 1.])

    def test_bad_id_sets_rejected(self):
        ids = [["a", "b"], ["a", "x"], ["a", "b"], ["a", "b"]]
        with self.assertRaises(ValueError):
            analysis.aligned_scores([np.ones((1, 2))] * 4, ids)

    def test_zscore_constant_and_native_tie(self):
        np.testing.assert_array_equal(analysis.zscore_costs([[7., 7.]]), [[0., 0.]])
        self.assertEqual(analysis.choose_indices([[0., 0.]]).tolist(), [0])

    def test_rank_average_and_ens4_w(self):
        panel = np.array([[[0., 3.]], [[1., 0.]], [[2., 1.]], [[3., 2.]]])
        self.assertEqual(analysis.arm_costs(panel, "RANKAVG4").shape, (1, 2))
        self.assertEqual(analysis.arm_costs(panel, "ENS4_W", .7).shape, (1, 2))
        with self.assertRaises(ValueError):
            analysis.arm_costs(panel, "ENS4_W", .6)

    def test_strict_gate_rejects_margin_equality(self):
        self.assertFalse(analysis.admit({"delta": .001 + 1e-12, "ci_low_95": .1}))
        self.assertFalse(analysis.admit({"delta": .01, "ci_low_95": 0.}))
        self.assertTrue(analysis.admit({"delta": .001001, "ci_low_95": .0001}))

    def test_median_seed_bootstrap_uses_shared_family_draws(self):
        rows = [{"family_id": f"f{i}"} for i in range(5) for _ in range(2)]
        base = np.zeros((3, len(rows)))
        candidate = base + .002
        result = analysis.median_seed_bootstrap(rows, candidate, base, draws=100, seed=17)
        self.assertAlmostEqual(result["delta"], .002)
        self.assertAlmostEqual(result["ci_low_95"], .002)
        self.assertAlmostEqual(result["ci_high_95"], .002)

    def test_seed_aggregation_precedes_family_bootstrap(self):
        # Per-seed family means are [0, 0, +.004], so the median seed gain is zero.
        # A family-median-then-mean estimator would incorrectly report +.004.
        rows = [{"family_id": "A"}, {"family_id": "B"}]
        baseline = np.zeros((3, 2))
        candidate = np.array([[.004, -.004], [-.004, .004], [.004, .004]])
        result = analysis.median_seed_bootstrap(rows, candidate, baseline, draws=100, seed=19)
        self.assertAlmostEqual(result["delta"], 0.0)

    def test_canonical_h145_elo_dominance_matches_tracker(self):
        root = Path(__file__).resolve().parents[3]
        tracker = root / "python/src/jope/north_star/elo_tracker.py"
        rows = [
            {"context_sha256": "d1", "family_id": "f1"},
            {"context_sha256": "d2", "family_id": "f2"},
        ]
        ratings = analysis.canonical_elo(
            rows,
            np.array([[0., 1.], [0., 1.]]),
            ["R882", "ALT"],
            tracker,
            "9ece98326c32453179b0c765eb45944b68ca9de944d2828a1d42dee1401d1835",
        )
        self.assertEqual(ratings, {"ALT": 6300.0, "R882": 1500.0})

    def test_fixed_family_folds_are_balanced_and_deterministic(self):
        rows = [{"family_id": f"fam-{i:03d}"} for i in range(205)]
        a = analysis.folds(rows)
        b = analysis.folds(list(reversed(rows)))
        self.assertEqual(a, b)
        self.assertEqual([len(g) for g in a], [41] * 5)
        self.assertEqual(len(set().union(*(set(g) for g in a))), 205)


if __name__ == "__main__":
    unittest.main()
