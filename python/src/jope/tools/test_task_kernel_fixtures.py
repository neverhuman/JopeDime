"""Isolated four-task kernel fixtures. No live queue jobs."""
import unittest

from four_task_campaign import PRIMARY, isolate
from scored_family_allowlist import inspect_prepared
import promotion_gate as gate


class TaskKernelFixtures(unittest.TestCase):
    def test_each_task_has_a_primary_and_cannot_share_recovery(self):
        for task, (metric, units) in PRIMARY.items():
            camp = isolate(task, "fixture")
            self.assertEqual(camp["primary_metric"], metric)
            self.assertEqual(camp["units"], units)
            self.assertFalse(camp["may_merge_into_recovery"])
            self.assertEqual(gate.TASK_METRICS[task], (metric, units))

    def test_promotion_manifest_keeps_regression_rmse(self):
        # Existing paper gate identity.
        self.assertEqual(gate.TASK_METRICS["regression"], ("rmse", "original_target"))

    def test_regression_screen_does_not_invent_target_or_forecast_claims(self):
        self.assertTrue(inspect_prepared({
            "family": "datasets-penguins", "task": "regression",
            "source_target_column_original": "bill_len",
            "preprocessing": {"target_mean": 43.8877},
        })["scored"])
        self.assertTrue(inspect_prepared({
            "family": "sidc-sunspots", "task": "regression", "prediction_contract": "interpolation",
            "cuts": [{"train_indices": [0, 2, 9], "validation_indices": [1, 3, 8]}],
        })["scored"])



if __name__ == "__main__":
    unittest.main()
