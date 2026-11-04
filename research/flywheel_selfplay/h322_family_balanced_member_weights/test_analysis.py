import unittest
import numpy as np
import statistics_h322 as s


class AnalysisTests(unittest.TestCase):
    def test_exact_tie_never_advances(self):
        a = dict(median_gain=0.1, median_harmed_rate=0)
        r = s.decide({"median": 0.0, "lcb95": 0.0}, a, a)
        self.assertEqual(r["verdict"], "FAMW_NOT_ADVANCED")

    def test_all_three_clauses_required(self):
        base = dict(median_gain=0.1, median_harmed_rate=0.2)
        for p, b in [
            (
                {"median": 0.01, "lcb95": 0.0},
                dict(median_gain=0.1, median_harmed_rate=0.2),
            ),
            (
                {"median": 0.01, "lcb95": 0.001},
                dict(median_gain=0.07, median_harmed_rate=0.2),
            ),
            (
                {"median": 0.01, "lcb95": 0.001},
                dict(median_gain=0.1, median_harmed_rate=0.3),
            ),
        ]:
            self.assertEqual(s.decide(p, b, base)["verdict"], "FAMW_NOT_ADVANCED")

    def test_retention_boundary_inclusive(self):
        base = dict(median_gain=0.1, median_harmed_rate=0.2)
        famw = dict(median_gain=0.9 * 0.1, median_harmed_rate=0.2)
        self.assertEqual(
            s.decide({"median": 0.01, "lcb95": 0.001}, famw, base)["verdict"],
            "FAMW_ADVANCE",
        )

    def test_ties_excluded_from_elo(self):
        self.assertEqual(s.elo(0, 0), 1500.0)
        self.assertGreater(s.elo(2, 1), 1500.0)

    def test_paired_constant_shift_and_shared_indices(self):
        base = np.arange(60, dtype=float).reshape(3, 20) / 1000
        famw = base + 0.02
        stats, p, secondary, ix = s.analyze(base, famw, draws=100)
        self.assertAlmostEqual(p["median"], 0.02)
        np.testing.assert_allclose(p["ci95"], [0.02, 0.02], atol=1e-15)
        expected = np.random.default_rng(322191).integers(0, 20, size=(100, 20))
        np.testing.assert_array_equal(ix, expected)

    def test_drop_difference_distinct_from_drop_delta(self):
        b = np.array([[0.0] * 10 + [1.0] * 10])
        m = np.array([[1.0] * 10 + [0.0] * 10])
        stats, p, other, ix = s.analyze(b, m, draws=40)
        self.assertEqual(p["median"], 0)
        self.assertEqual(other["median"], -1)

    def test_whole_family_mean_equal_weight(self):
        b = np.zeros((3, 20))
        m = np.tile(np.arange(20), (3, 1)).astype(float)
        stats, p, other, ix = s.analyze(b, m, draws=40)
        self.assertEqual(stats["FAMW"]["aggregate"]["median_gain"], 9.5)
        self.assertEqual(stats["FAMW"]["cells"][0]["drop_top"]["10"], 4.5)

    def test_interval_matches_direct_resample(self):
        b = np.zeros((3, 20))
        m = np.tile(np.linspace(-0.03, 0.08, 20), (3, 1))
        stats, p, other, ix = s.analyze(b, m, draws=100)
        direct = np.sort(m[0][ix], axis=1)[:, :-10].mean(axis=1)
        np.testing.assert_array_equal(p["ci95"], np.quantile(direct, [0.025, 0.975]))
        self.assertEqual(p["lcb95"], np.quantile(direct, 0.05))

    def test_wtl_tol_and_harm(self):
        b = np.zeros((3, 20))
        m = np.zeros_like(b)
        m[:, 0] = 1e-4
        m[:, 1] = -1e-4
        m[:, 2] = -0.01
        m[:, 3] = -0.01001
        stats, *_ = s.analyze(b, m, draws=10)
        r = stats["FAMW"]["cells"][0]
        self.assertEqual(
            (r["wins"], r["ties"], r["losses"], r["harmed"]), (0, 18, 2, 1)
        )

    def test_signed_concentration_can_exceed_one(self):
        from evaluate import concentration

        x = np.array([1.0] * 10 + [-0.1] * 10)
        r = concentration(x)
        self.assertAlmostEqual(r["top10_share"], 10 / 9)
        self.assertEqual(r["top10_gross_positive_share"], 1)

    def test_concentration_nonpositive_denominator(self):
        from evaluate import concentration

        self.assertIsNone(concentration(np.zeros(20))["top10_share"])
        self.assertIsNone(concentration(-np.ones(20))["top10_share"])

    def test_jaccard_empty_and_partial_overlap(self):
        from evaluate import jaccard

        self.assertEqual(jaccard(set(), set()), 1)
        self.assertEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)
        self.assertEqual(jaccard({("a", 1)}, {("a", 2)}), 0)

    def test_lofo_equals_remove_best_family(self):
        x = np.array([-0.2, 0.1, 0.3, 0.4])
        self.assertAlmostEqual(float(((x.sum() - x) / 3).min()), float(s.drop(x, 1)))


if __name__ == "__main__":
    unittest.main()
