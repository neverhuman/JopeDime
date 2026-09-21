import unittest

import cut_baseline as b


class CutBaselineTests(unittest.TestCase):
    def test_skill_uses_a_training_mean_baseline_and_preserves_zero_error(self):
        baseline = b.baseline_rmse([0.0, 0.0], [2.0, 4.0])
        self.assertAlmostEqual(baseline, 10.0**0.5)
        result = b.annotate_observation(0.0, baseline, "original_target")
        self.assertEqual(result["relative_rmse"], 0.0)
        self.assertEqual(result["rmse_skill"], 1.0)
        self.assertEqual(result["squared_error_skill"], 1.0)
        zero = b.annotate_observation(0.0, 0.0, "original_target")
        self.assertIsNone(zero["relative_rmse"])
        self.assertTrue(zero["baseline_available"])
        self.assertEqual(zero["ratio_status"], "zero_baseline")

    def test_target_rescaling_preserves_all_skills(self):
        expected = b.annotate_observation(2.0, 4.0, "original_target")
        for scale in (1e-100, 1e100, 7.0):
            result = b.annotate_observation(2.0 * scale, 4.0 * scale, "original_target")
            for field in ("relative_rmse", "rmse_skill", "squared_error_skill"):
                self.assertEqual(result[field], expected[field])
        for invalid in (None, True, float("inf"), float("nan"), -1.0):
            self.assertIsNone(b.annotate_observation(invalid, 1.0)["relative_rmse"])
            self.assertIsNone(b.disagreement_vs_unit(invalid, 1.0))

    def test_perfect_mean_predictor(self):
        self.assertEqual(b.baseline_rmse([2.0, 2.0, 2.0], [2.0, 2.0]), 0.0)

    def test_known_rmse(self):
        # train mean 0; val [-1, 1] => RMSE 1
        self.assertAlmostEqual(b.baseline_rmse([-1.0, 1.0], [-1.0, 1.0]), 1.0)

    def test_unit_one_disagrees_with_cut_baseline(self):
        self.assertTrue(b.disagreement_vs_unit(0.8, 0.5))
        self.assertFalse(b.disagreement_vs_unit(0.8, 1.2))
        note = b.annotate_observation(0.8, 0.5)
        self.assertTrue(note["beats_unit_one"])
        self.assertFalse(note["beats_cut_baseline"])
        self.assertTrue(note["unit_vs_baseline_disagrees"])
        self.assertEqual(note["search_best_units"], "original_target")
        self.assertEqual(note["metrics_units"], "cut_normalized_target")


if __name__ == "__main__":
    unittest.main()
