"""Focused controls for whole displayed-row GP sensitivity, without fitting models."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("input_influence", Path(__file__).parents[1] / "input_influence.py")
influence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(influence)


class VisibleRowSensitivity(unittest.TestCase):
    def test_all_seventeen_rows_share_one_actual_maximum(self):
        baseline = [100. + row for row in range(17)]
        perturbed = [[baseline[row] + row * 4 + column + 1 for column in range(4)] for row in range(17)]
        deltas, maximum, scores = influence.score_arrays(baseline, perturbed)
        self.assertEqual((len(scores), len(scores[0])), (17, 4))
        self.assertEqual(maximum, 68.)
        self.assertEqual(deltas[-1][-1], 68.)
        self.assertEqual(scores[0][0], 1. / 68.)
        self.assertEqual(scores[4][-1], 20. / 68.)
        self.assertEqual(scores[-1][-1], 1.)

    def test_zero_sensitivity_stays_zero_for_old_and_extended_counts(self):
        for count in (5, 17):
            deltas, maximum, scores = influence.score_arrays([2.] * count, [[2.] * 4 for _ in range(count)])
            self.assertEqual(maximum, 0.)
            self.assertEqual(deltas, [[0.] * 4 for _ in range(count)])
            self.assertEqual(scores, [[0.] * 4 for _ in range(count)])

    def test_extended_perturbations_change_only_one_named_feature(self):
        columns = ["x2", "x0", "x3", "x1"]
        rows = [[row + 10. * column for column in range(4)] for row in range(17)]
        ctx = {"visible_ids": list(range(17)), "features": ["x0", "x1", "x2", "x3"],
               "medians": [101., 102., 103., 104.], "rows": rows,
               "bindings": {"model_columns": columns}}
        changed = influence.perturbations(ctx)
        self.assertEqual(len(changed), 68)
        for row in range(17):
            for feature in range(4):
                result = changed[row * 4 + feature]
                at = columns.index(ctx["features"][feature])
                self.assertEqual(len(result), 4)  # The target is never an input.
                self.assertEqual(result[at], ctx["medians"][feature])
                self.assertEqual([v for j, v in enumerate(result) if j != at],
                                 [v for j, v in enumerate(rows[row]) if j != at])
        self.assertEqual(rows[0], [0., 10., 20., 30.])  # Originals are untouched.

    def test_target_column_or_missing_additional_row_is_refused(self):
        for perturbed in ([[1.] * 5 for _ in range(17)], [[1.] * 4 for _ in range(16)]):
            with self.assertRaisesRegex(ValueError, "shape"):
                influence.score_arrays([1.] * 17, perturbed)


if __name__ == "__main__":
    unittest.main()
