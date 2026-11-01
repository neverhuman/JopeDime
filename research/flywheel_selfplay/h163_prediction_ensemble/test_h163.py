"""Data-free tests for H163 score direction, ID alignment, ties and gates."""

import unittest
import numpy as np
import analysis


class H163Tests(unittest.TestCase):
    def test_argmin_and_measured_utility_direction_are_separate(self):
        costs = np.array([[4.0, 1.0, 2.0]])
        self.assertEqual(analysis.choose_indices(costs).tolist(), [1])

    def test_alignment_preserves_first_model_native_order(self):
        ids = [["a", "b", "c"], ["c", "a", "b"], ["a", "b", "c"], ["b", "c", "a"]]
        vals = [np.array([[1., 2., 3.]])] * 4
        aligned, aligned_ids = analysis.aligned_scores(vals, ids)
        self.assertEqual(aligned_ids, ids[0])
        np.testing.assert_array_equal(aligned[1, 0], [2., 3., 1.])

    def test_bad_id_sets_rejected(self):
        ids = [["a", "b"], ["a", "x"], ["a", "b"], ["a", "b"]]
        with self.assertRaises(ValueError):
            analysis.aligned_scores([np.ones((1, 2))] * 4, ids)

    def test_zscore_constant_and_native_tie(self):
        np.testing.assert_array_equal(analysis.zscore_costs([[7., 7.]]), [[0., 0.]])
        self.assertEqual(analysis.choose_indices([[0., 0.]]).tolist(), [0])

    def test_rank_average_and_ens4_w(self):
        panel = np.array([[[0., 3.]], [[1., 0.]], [[2., 1.]], [[3., 2.]]])
        self.assertEqual(analysis.arm_costs(panel, "RANKAVG4").shape, (1, 2))
        self.assertEqual(analysis.arm_costs(panel, "ENS4_W", .7).shape, (1, 2))
        with self.assertRaises(ValueError):
            analysis.arm_costs(panel, "ENS4_W", .6)

    def test_strict_gate_rejects_margin_equality(self):
        self.assertFalse(analysis.admit({"delta": .001 + 1e-12, "ci_low_95": .1}))
        self.assertFalse(analysis.admit({"delta": .01, "ci_low_95": 0.}))
        self.assertTrue(analysis.admit({"delta": .001001, "ci_low_95": .0001}))


if __name__ == "__main__":
    unittest.main()
