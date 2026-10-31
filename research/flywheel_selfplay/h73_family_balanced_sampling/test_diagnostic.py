from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from diagnostic import aggregate_metrics, read_train_rows_only


class TrainingOnlyReaderTests(unittest.TestCase):
    def test_stops_before_dev_value_and_returns_only_train_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.json"
            path.write_text(
                '{"train": [{"context_sha256": "train-1", "nested": [1, 2]}], '
                '"dev": NOT_JSON_AND_MUST_NOT_BE_PARSED'
            )
            rows, prefix_sha = read_train_rows_only(path)
        self.assertEqual(rows, [{"context_sha256": "train-1", "nested": [1, 2]}])
        self.assertEqual(len(prefix_sha), 64)

    def test_requires_train_to_be_the_first_selected_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.json"
            path.write_text('{"dev": [], "train": []}')
            with self.assertRaisesRegex(ValueError, "train must be the first"):
                read_train_rows_only(path)


class AggregationTests(unittest.TestCase):
    def test_family_balance_and_minus_two_failure_share(self):
        rows = [
            {"context_sha256": "a1", "family_group": "fam-a", "fold": 0},
            {"context_sha256": "a2", "family_group": "fam-a", "fold": 0},
            {"context_sha256": "b1", "family_group": "fam-b", "fold": 1},
        ]
        results = {
            "a1": {"objective_loss": 2.0, "selected_failure": True, "failure_candidate_loss_mass": 0.5, "all_candidate_loss_mass": 1.0},
            "a2": {"objective_loss": 1.0, "selected_failure": False, "failure_candidate_loss_mass": 0.0, "all_candidate_loss_mass": 1.0},
            "b1": {"objective_loss": 1.0, "selected_failure": False, "failure_candidate_loss_mass": 0.0, "all_candidate_loss_mass": 1.0},
        }
        metrics = aggregate_metrics(rows, results)
        self.assertEqual(metrics["resolved_families"], 2)
        self.assertEqual(metrics["context_count_histogram"], {"1": 1, "2": 1})
        self.assertAlmostEqual(metrics["contexts_per_family"]["top10_family_context_share"], 1.0)
        self.assertAlmostEqual(metrics["r882_training_objective"]["selected_minus_two_failure_contexts"]["share_of_unweighted_context_loss_mass"], 0.5)
        self.assertEqual(len(metrics["r882_training_objective"]["per_family_loss_profiles"]), 2)
        self.assertAlmostEqual(metrics["r882_training_objective"]["raw_family_loss_share"]["family_size_vs_total_loss_spearman_rho"], 1.0)
        self.assertEqual(len(metrics["fold_summaries"]), 5)
        self.assertFalse(json.dumps(metrics, allow_nan=False).startswith("NaN"))

    def test_full_cohort_family_profile_stays_below_size_limit(self):
        rows = [
            {"context_sha256": f"ctx-{i}", "family_group": f"fam-{i}", "fold": i % 5}
            for i in range(205)
        ]
        results = {
            row["context_sha256"]: {
                "objective_loss": 1.0 + i / 100.0,
                "selected_failure": i % 20 == 0,
                "failure_candidate_loss_mass": 0.1,
                "all_candidate_loss_mass": 1.0,
            }
            for i, row in enumerate(rows)
        }
        metrics = aggregate_metrics(rows, results)
        encoded = json.dumps(metrics, allow_nan=False).encode()
        self.assertLess(len(encoded), 50 * 1024)
        self.assertEqual(len(metrics["r882_training_objective"]["per_family_loss_profiles"]), 205)


if __name__ == "__main__":
    unittest.main()
