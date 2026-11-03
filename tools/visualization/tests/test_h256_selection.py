"""Selection and source-execution controls without private model dependencies."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h256_gp_source import choose
from h256_gp_validate import prediction_vector
from h256_validate import check_readouts


class SourceSelectionSafety(unittest.TestCase):
    def case(self, costs):
        return {"programs": [f"program-{i}" for i in range(6)], "outputs": {"final_k0_cost": costs}}

    def test_prior_probes_cannot_replace_an_actual_proposal(self):
        index, program, _, _ = choose(self.case([-100., -99., -98., -97., 2., 1.]))
        self.assertEqual((index, program), (5, "program-5"))

    def test_equal_proposal_costs_keep_recorded_order(self):
        self.assertEqual(choose(self.case([0.] * 6))[0], 4)

    def test_residual_without_complete_final_cost_is_not_a_selection(self):
        case = self.case([0.] * 6)
        case["outputs"] = {"ensemble_weighted_residual": [0.] * 6}
        with self.assertRaisesRegex(ValueError, "complete G cost"):
            choose(case)

    def test_nonfinite_final_cost_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid observed"):
            choose(self.case([0., 0., 0., 0., float("nan"), 1.]))

    def test_real_driver_stdout_encodings_are_both_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "prediction.stdout"
            for body in ("[1.25,-2.5]\n", "1.25\n-2.5\n"):
                file.write_text(body)
                self.assertEqual(prediction_vector(file).tolist(), [1.25, -2.5])

    def test_nonfinite_driver_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "prediction.stdout"
            file.write_text("nan\n")
            with self.assertRaisesRegex(ValueError, "invalid prediction"):
                prediction_vector(file)

    def test_float64_cost_readout_cannot_hide_rounding_in_float32(self):
        case = {"outputs": {"cost": [1. + 2. ** -40]}, "outputs_tensor_ids": {"cost": "cost"}}
        tensors = {"cost": {"dtype": "float64", "values": [1.]}}
        with self.assertRaisesRegex(ValueError, "differs from observed tensor"):
            check_readouts(case, tensors)


if __name__ == "__main__":
    unittest.main()
