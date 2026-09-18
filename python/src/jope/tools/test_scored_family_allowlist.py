import copy
import unittest

import scored_family_allowlist as a

PENGUINS_METADATA_SHA256 = "af97e181ae6adda3e1f1b333ca9166a3b6bb920feb1a3569474c9c18d306c980"

class AllowlistTests(unittest.TestCase):
    def test_penguins_bill_length_with_year_feature_is_not_quarantined(self):
        row = a.inspect_prepared({
            "family": "datasets-penguins", "task": "regression",
            "source_target_column_original": "bill_len",
            "metadata_sha256": PENGUINS_METADATA_SHA256,
            "preprocessing": {"target_mean": 43.8877,
                              "columns": [{"name": "year", "mean": 2008.02}]},
        })
        self.assertTrue(row["scored"])
        self.assertEqual(row["reasons"], [])

    def test_numeric_year_target_is_not_automatically_invalid_regression(self):
        row = a.inspect_prepared({"family": "declared-numeric-target", "task": "regression",
                                  "preprocessing": {"target_mean": 2008.02}})
        self.assertTrue(row["scored"])

    def test_interleaved_regression_interpolation_is_not_forecast_claim(self):
        row = a.inspect_prepared({
            "family": "sidc-sunspots", "task": "regression",
            "prediction_contract": "interpolation",
            "cuts": [{"train_indices": [0, 2, 4, 800], "validation_indices": [1, 3, 5, 200]}],
        })
        self.assertTrue(row["scored"])

    def test_explicit_forecasting_interleave_is_rejected_on_any_cut(self):
        row = a.inspect_prepared({
            "family": "series", "task": "regression", "prediction_contract": "forecasting",
            "cuts": [{"train_indices": [0, 1], "validation_indices": [2, 3]},
                     {"train_indices": [0, 2], "validation_indices": [1, 3]}],
        })
        self.assertFalse(row["scored"])
        self.assertEqual(row["reasons"], ["cut-1 train/val indices interleave under forecasting contract"])

    def test_nonregression_stays_out_of_regression_panel(self):
        for task in ("binary", "multiclass", "timeseries", "unknown"):
            with self.subTest(task=task):
                self.assertFalse(a.inspect_prepared({"family": "synthetic", "task": task})["scored"])

    def test_quality_filter_keeps_only_explicit_screen_members(self):
        allow = a.report([a.inspect_prepared({"family": "islr-auto", "task": "regression"})])
        out = a.quality_rows_on_allowlist(
            {"rows": [{"family": "islr-auto"}, {"family": "datasets-penguins"}]}, allow)
        self.assertEqual((out["kept"], out["skipped"]), (1, 1))
        self.assertFalse(allow["independent_evaluation"])
        self.assertIn("qualification required", allow["eligibility_scope"])
        self.assertNotIn("datasets-penguins", allow["quarantined_families"])

    def test_explicit_quarantine_does_not_mutate_input_rows(self):
        rows = [a.inspect_prepared({"family": "synthetic", "task": "regression"})]
        original = copy.deepcopy(rows)
        result = a.report(rows, {"synthetic": "explicit unresolved identity"})
        self.assertEqual(rows, original)
        self.assertEqual(result["quarantined_families"], ["synthetic"])

if __name__ == "__main__":
    unittest.main()
