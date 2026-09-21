import unittest

import continuous_quality_report as q
import scored_family_allowlist as a


class QualityUnitsTests(unittest.TestCase):
    def test_units_and_baseline_stay_on_the_row(self):
        cut = dict(training_sha256="a" * 64, validation_sha256="b" * 64, preprocessing_sha256="c" * 64)
        evidence = dict(
            schema="parallel.cut_baseline.v1",
            rule="training_target_mean_on_scoring_rows",
            training_sha256=cut["training_sha256"],
            scoring_sha256=cut["validation_sha256"],
            preprocessing_sha256=cut["preprocessing_sha256"],
            training_rows=2,
            scoring_rows=2,
            rmse=0.5,
            units="cut_normalized_target",
        )
        row = q.attach_cut_baseline(
            {"family": "islr-auto", "task": "regression"},
            {},
            cut,
            {
                "metrics": {"rmse": 0.8, "units": "cut_normalized_target", "baseline_evidence": evidence},
                "target_scale": 10.0,
            },
        )
        self.assertEqual(row["metrics_units"], "original_target")
        self.assertEqual(row["search_best_units"], "original_target")
        self.assertAlmostEqual(row["baseline_rmse"], 5.0)
        self.assertAlmostEqual(row["relative_rmse"], 1.6)
        self.assertAlmostEqual(row["rmse_skill"], -0.6)
        self.assertAlmostEqual(row["squared_error_skill"], -1.56)
        for field in ("training_sha256", "validation_sha256", "preprocessing_sha256"):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                q.attach_cut_baseline(
                    {},
                    {},
                    dict(cut, **{field: "d" * 64}),
                    dict(metrics=dict(rmse=0.8, units="original_target", baseline_evidence=evidence)),
                )

    def test_missing_baseline_is_not_reconstructed_from_unbound_targets_or_scalars(self):
        row = q.attach_cut_baseline(
            {},
            {},
            dict(train_targets=[0.0, 1.0], validation_targets=[0.0, 1.0]),
            dict(metrics=dict(rmse=0.0, units="original_target"), baseline_rmse=0.5),
        )
        self.assertFalse(row["baseline_available"])
        self.assertIsNone(row["relative_rmse"])

    def test_boolean_score_is_not_perfect_evidence(self):
        self.assertIsNone(q.metric(dict(metrics=dict(rmse=False, units="original_target"))))

    def test_allowlist_is_a_filter_not_a_rewrite(self):
        payload = {
            "rows": [
                {"family": "islr-auto", "task": "regression", "best_final_gp": 1.2},
                {
                    "family": "series",
                    "task": "regression",
                    "prediction_contract": "forecasting",
                    "cuts": [{"train_indices": [0, 2], "validation_indices": [1, 3]}],
                    "best_final_gp": 0.4,
                },
            ]
        }
        out = q.apply_allowlist_filter(payload)
        self.assertEqual(out["rows"][1]["best_final_gp"], 0.4)
        self.assertEqual(out["allowlist"]["skipped"], 1)
        self.assertEqual([r["family"] for r in out["allowlisted_rows"]], ["islr-auto"])
        self.assertEqual(a.KNOWN_QUARANTINE, {})


if __name__ == "__main__":
    unittest.main()
