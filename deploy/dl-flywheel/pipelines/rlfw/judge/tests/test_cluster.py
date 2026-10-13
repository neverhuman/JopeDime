"""Cluster intervals stay beside the context interval and can detect a large effect."""

import io
import json
import unittest

from rlfw.judge.cluster import (
    cluster_bootstrap_ci,
    context_interval,
    hierarchical_power,
    variance_components,
)
from rlfw.judge.gap_close_report import (
    complete_series,
    paired_groups,
    require_power_sizes,
    validation_cluster_sizes,
)


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

    def test_nonfinite_power_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            hierarchical_power([2, 2], float("nan"), 0.1, 0, sims=3, draws=10)
        with self.assertRaises(ValueError):
            hierarchical_power([2, 2], 0.1, float("inf"), 0, sims=3, draws=10)
        with self.assertRaises(ValueError):
            hierarchical_power([2, 2], 0.1, 0.1, float("nan"), sims=3, draws=10)
        with self.assertRaises(ValueError):
            context_interval([0.0, float("nan")])
        with self.assertRaises(ValueError):
            variance_components([[0.0, 0.1], [float("inf"), 1.0]])
        with self.assertRaises(ValueError):
            cluster_bootstrap_ci([[0.0, 1.0], [float("nan")]], draws=5, seed=1)


def _roster(prefix):
    return [prefix + f"{i:03d}" for i in range(263)]


def _parents(ids):
    return {context_id: ("p000" if i < 4 else f"p{i:03d}") for i, context_id in enumerate(ids)}


def _rows(ids, cost=1.0, updates=3, seed=7, epoch=1):
    rows = []
    for arm in ("r23_control", "rloo", "pkpo_b8"):
        arm_cost = cost if arm == "r23_control" else cost + 0.25
        for context_id in ids:
            rows.append({
                "arm": arm,
                "epoch": epoch,
                "context_id": context_id,
                "l6_cost": arm_cost,
                "split": "full_val",
                "updates": updates,
                "seed": seed,
            })
    return rows


def _handle(rows):
    return io.StringIO("".join(json.dumps(row) + "\n" for row in rows))


class PinnedRosterTest(unittest.TestCase):
    def setUp(self):
        self.val = _roster("v")
        self.parents = _parents(self.val)

    def test_pinned_ids_pair_and_power_sizes_match_those_groups(self):
        epoch, arms = complete_series(_handle(_rows(self.val)), self.val)
        self.assertEqual(epoch, 1)
        groups = paired_groups(arms, self.parents, "rloo", self.val)
        sizes = validation_cluster_sizes(self.val, self.parents)
        self.assertEqual(require_power_sizes(groups, sizes, self.val, self.parents), sizes)
        self.assertEqual(sum(sizes), 263)
        self.assertEqual(sizes[0], 4)

    def test_train_ids_with_the_right_count_are_refused(self):
        with self.assertRaisesRegex(ValueError, "pinned validation roster"):
            complete_series(_handle(_rows(_roster("t"))), self.val)

    def test_one_wrong_id_is_refused(self):
        wrong = list(self.val)
        wrong[0] = "not-a-validation-id"
        with self.assertRaisesRegex(ValueError, "pinned validation roster"):
            complete_series(_handle(_rows(wrong)), self.val)

    def test_a_different_arm_roster_is_refused(self):
        rows = _rows(self.val)
        for row in rows:
            if row["arm"] == "rloo" and row["context_id"] == self.val[-1]:
                row["context_id"] = "other-arm-id"
        with self.assertRaisesRegex(ValueError, "pinned validation roster"):
            complete_series(_handle(rows), self.val)

    def test_conflicting_duplicate_cost_is_refused(self):
        rows = _rows(self.val)
        duplicate = dict(rows[0])
        duplicate["l6_cost"] = 9.0
        rows.append(duplicate)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            complete_series(_handle(rows), self.val)

    def test_conflicting_duplicate_provenance_is_refused(self):
        rows = _rows(self.val)
        duplicate = dict(rows[0])
        duplicate["updates"] = 99
        rows.append(duplicate)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            complete_series(_handle(rows), self.val)

    def test_identical_repeat_dedupes(self):
        rows = _rows(self.val)
        rows.append(dict(rows[0]))
        _epoch, arms = complete_series(_handle(rows), self.val)
        self.assertEqual(set(arms["r23_control"]), set(self.val))
        self.assertEqual(arms["r23_control"][self.val[0]], 1.0)

    def test_nonfinite_cost_is_refused(self):
        rows = _rows(self.val)
        rows[0]["l6_cost"] = float("nan")
        with self.assertRaisesRegex(ValueError, "not finite"):
            complete_series(_handle(rows), self.val)

    def test_power_sizes_must_match_the_paired_groups(self):
        _epoch, arms = complete_series(_handle(_rows(self.val)), self.val)
        groups = paired_groups(arms, self.parents, "pkpo_b8", self.val)
        sizes = [len(group) for group in groups]
        shifted = list(sizes)
        shifted[0] += 1
        with self.assertRaisesRegex(ValueError, "power cluster sizes"):
            require_power_sizes(groups, shifted, self.val, self.parents)


if __name__ == "__main__":
    unittest.main()
