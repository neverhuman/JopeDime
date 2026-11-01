"""Data-free controls for H157's family partition and strict candidate gate."""

import unittest

import numpy as np


def partition(families, seed=157140):
    ordered = sorted(families)
    groups = np.array_split(
        np.random.default_rng(seed).permutation(ordered), 5
    )
    select = set(groups[0].tolist())
    fit = set().union(*(set(group.tolist()) for group in groups[1:]))
    return fit, select


def admit(median_gain, lcb95):
    if abs(median_gain - 0.001) <= 1e-12:
        return False
    return median_gain > 0.001 and lcb95 > 0.0


def choose(candidates):
    """Candidate map: ID -> (median gain, bootstrap LCB95)."""
    ranked = sorted(candidates.items(), key=lambda item: (-item[1][0], item[0]))
    for candidate_id, (gain, lcb) in ranked:
        if admit(gain, lcb):
            return candidate_id
    return "R882"


class H157Tests(unittest.TestCase):
    def test_partition_is_complete_disjoint_and_deterministic(self):
        families = [f"fam-{i:03d}" for i in range(205)]
        fit, select = partition(families)
        self.assertEqual((len(fit), len(select)), (164, 41))
        self.assertFalse(fit & select)
        self.assertEqual(fit | select, set(families))
        self.assertEqual((fit, select), partition(reversed(families)))

    def test_strict_margin_and_lcb_gates(self):
        self.assertFalse(admit(0.001, 0.01))
        self.assertFalse(admit(0.001 + 5e-13, 0.01))
        self.assertFalse(admit(0.002, 0.0))
        self.assertFalse(admit(0.0, 0.1))
        self.assertTrue(admit(0.001 + 2e-12, 1e-12))

    def test_candidate_ties_are_stable_and_failed_candidates_reject(self):
        self.assertEqual(
            choose({"c02": (0.003, 0.001), "c01": (0.003, 0.002)}),
            "c01",
        )
        self.assertEqual(choose({"c01": (0.003, 0.0)}), "R882")


if __name__ == "__main__":
    unittest.main()
