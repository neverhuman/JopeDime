from __future__ import annotations

import unittest

import numpy as np

from research.flywheel_selfplay.h86_w3_outcome_status_census.census import (
    ANOMALOUS,
    FAIL,
    NONFINITE,
    PASS,
    SENTINEL,
    classify,
    order_sha,
    quantiles,
    top_resolved,
)


class ClassifyTest(unittest.TestCase):
    def test_signature_rule_and_precedence(self) -> None:
        fitness = np.array(
            [-0.25, 99.9941, 100.0, 100.0, np.nan, 0.5, 100.5, 3.0, 100.0]
        )
        logratio = np.array([-0.3, -0.0031, 0.0, 0.0, 0.1, np.inf, 0.2, 0.1, 0.25])
        gate = np.array([True, False, False, True, False, True, True, False, False])
        self.assertEqual(
            classify(fitness, logratio, gate).tolist(),
            [
                PASS,
                FAIL,
                SENTINEL,
                SENTINEL,
                NONFINITE,
                NONFINITE,
                ANOMALOUS,
                ANOMALOUS,
                FAIL,
            ],
        )

    def test_finite_sentinel_is_never_measured(self) -> None:
        status = classify(np.array([100.0]), np.array([0.0]), np.array([False]))
        self.assertEqual(status.tolist(), [SENTINEL])
        self.assertEqual(status.dtype, np.uint8)


class TopResolvedTest(unittest.TestCase):
    def test_no_passer_path(self) -> None:
        self.assertEqual(
            top_resolved(np.array([])), {"path": "no_passer", "passers": 0}
        )

    def test_single_passer_uses_the_floor(self) -> None:
        result = top_resolved(np.array([0.3]))
        self.assertEqual(result["path"], "last_passing_order_statistic")
        self.assertTrue(result["floor_binds"])
        self.assertEqual(result["clipped"], 0)

    def test_sixteenth_order_statistic_sets_the_scale(self) -> None:
        fitness = np.concatenate([np.linspace(0.0, 0.15, 16), [0.5, 1.0, 2.0]])
        result = top_resolved(fitness[::-1])
        self.assertEqual(result["path"], "top16")
        self.assertFalse(result["floor_binds"])
        # Scale .15: regrets 3.33, 6.67 and 13.3 for the three tail rows, so two exceed the clip.
        self.assertEqual(result["clipped"], 2)

    def test_fewer_than_sixteen_uses_last_passer(self) -> None:
        result = top_resolved(np.array([0.0, 0.01, 0.02]))
        self.assertEqual(result["path"], "last_passing_order_statistic")
        self.assertTrue(result["floor_binds"])


class HelperTest(unittest.TestCase):
    def test_order_sha_binds_order(self) -> None:
        ids = np.array([b"a" * 64, b"b" * 64])
        self.assertNotEqual(order_sha(ids), order_sha(ids[::-1]))
        self.assertEqual(len(order_sha(ids)), 64)

    def test_quantiles_handle_empty(self) -> None:
        self.assertIsNone(quantiles(np.array([])))
        self.assertEqual(quantiles(np.array([1.0, 3.0]))["q0.5"], 2.0)


if __name__ == "__main__":
    unittest.main()
