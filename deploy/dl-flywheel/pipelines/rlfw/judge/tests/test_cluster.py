"""Cluster intervals stay beside the context interval and can detect a large effect."""

import unittest

from rlfw.judge.cluster import cluster_bootstrap_ci, hierarchical_power, variance_components


class ClusterIntervalTest(unittest.TestCase):
    def test_identical_clusters_match_the_context_mean(self):
        groups = [[0.0, 0.0], [0.0, 0.0], [4.0, 4.0], [4.0, 4.0]]
        report = cluster_bootstrap_ci(groups, draws=200, seed=1)
        self.assertEqual(report["n_contexts"], 8)
        self.assertEqual(report["n_clusters"], 4)
        self.assertAlmostEqual(report["mean"], 2.0)
        self.assertAlmostEqual(report["context"]["mean"], 2.0)
        self.assertLess(report["cluster_bootstrap"]["low"], report["mean"])
        self.assertGreater(report["cluster_bootstrap"]["high"], report["mean"])
        self.assertGreater(
            report["cluster_bootstrap"]["high"] - report["cluster_bootstrap"]["low"],
            report["context"]["high"] - report["context"]["low"],
        )

    def test_variance_components_separate_a_parent_shift(self):
        groups = [[0.0, 0.1], [0.0, 0.1], [10.0, 10.1], [10.0, 10.1]]
        parts = variance_components(groups)
        self.assertGreater(parts["sigma_between"], parts["sigma_within"])
        self.assertGreater(parts["icc"], 0.9)

    def test_hierarchical_power_is_higher_for_a_large_effect_than_for_zero(self):
        sizes = [8] * 20
        null = hierarchical_power(sizes, 1.0, 0.25, 0.0, sims=80, draws=200, seed=3)
        alt = hierarchical_power(sizes, 1.0, 0.25, 1.5, sims=80, draws=200, seed=3)
        self.assertLessEqual(null["cluster_reject_rate"], 0.15)
        self.assertGreaterEqual(alt["cluster_reject_rate"], 0.80)
        self.assertFalse(alt["qualifying"])


if __name__ == "__main__":
    unittest.main()
