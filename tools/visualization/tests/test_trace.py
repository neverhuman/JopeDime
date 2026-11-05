"""Tiny arithmetic controls only; never used to render inference evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("validate_trace", Path(__file__).parents[1] / "validate_trace.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def control():
    layer = {"sequence": 0, "network": "dope", "name": "control",
             "input_shape": [1, 2], "weight_shape": [1, 2], "output_shape": [1, 1],
             "input": [2, -3], "weights": [0.5, -2], "bias": [1], "output": [8]}
    jope = {**layer, "sequence": 1, "network": "jope"}
    case = {"id": "control-a", "dataset_sha256": "0" * 64, "diagnostic": True,
            "source": None, "tokens_per_second": None,
            "validation": {"trace_on_off_bitwise_equal": True, "closure_negative_control_rejected": True},
            "outputs": {"tokens": [0] * 320},
            "trace": {"affine": [layer, jope], "tensors": [
                {"sequence": 2, "network": "bridge", "name": "stage1b_tokens", "shape": [1, 10, 32], "values": [0] * 320}]}}
    return {"schema": validator.SCHEMA, "bundle_digest": "control-only", "cases": [case]}


class TraceSafety(unittest.TestCase):
    def check_data(self, data):
        with tempfile.TemporaryDirectory(prefix="trace-test-", dir=Path(__file__).parents[1]) as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(data))
            return validator.validate(path)

    def test_known_signed_sum(self):
        _, report = self.check_data(control())
        self.assertEqual(report["status"], "PASS")

    def test_corrupted_native_output_is_rejected(self):
        data = control()
        data["cases"][0]["trace"]["affine"][0]["output"] = [9]
        with self.assertRaisesRegex(ValueError, "closure failed"):
            self.check_data(data)

    def test_dope_only_cannot_claim_fused(self):
        data = control()
        data["cases"][0]["trace"]["affine"].pop()
        with self.assertRaisesRegex(ValueError, "both fused constituents"):
            self.check_data(data)

    def test_weight_change_between_cases_is_rejected(self):
        data = control()
        second = copy.deepcopy(data["cases"][0])
        second["id"] = "control-b"
        second["trace"]["affine"][0]["weights"][0] = 1
        second["trace"]["affine"][0]["output"][0] = 9
        data["cases"].append(second)
        with self.assertRaisesRegex(ValueError, "weights changed"):
            self.check_data(data)

    def test_nonfinite_value_is_rejected(self):
        data = control()
        data["cases"][0]["trace"]["affine"][0]["input"][0] = float("nan")
        with self.assertRaises(ValueError):
            self.check_data(data)

    def test_unmeasured_rate_is_rejected(self):
        data = control()
        data["cases"][0]["tokens_per_second"] = 123
        with self.assertRaisesRegex(ValueError, "unqualified token rate"):
            self.check_data(data)

    def test_trace_divergence_is_rejected(self):
        data = control()
        data["cases"][0]["validation"]["trace_on_off_bitwise_equal"] = False
        with self.assertRaisesRegex(ValueError, "changes outputs"):
            self.check_data(data)

    def test_changed_layer_identity_is_rejected(self):
        data = control()
        second = copy.deepcopy(data["cases"][0])
        second["id"] = "control-b"
        second["trace"]["affine"][1]["name"] = "another-layer"
        data["cases"].append(second)
        with self.assertRaisesRegex(ValueError, "inventory/order differs"):
            self.check_data(data)

    def test_corrupted_bridge_output_is_rejected(self):
        data = control()
        data["cases"][0]["outputs"]["tokens"][0] = 1
        with self.assertRaisesRegex(ValueError, "differs from native readout"):
            self.check_data(data)

    def test_gp_preview_preserves_actual_syntax_and_parameters(self):
        self.assertEqual(validator.gp_sexpression("(pipe X (scale q=0.05) (ridge))"),
                         ["pipe", "X", ["scale", "q=0.05"], ["ridge"]])

    def test_gp_preview_rejects_unbalanced_and_multiple_roots(self):
        for text in ("(pipe X", "(pipe X))", "(pipe)(ridge)", "()", "ridge"):
            with self.assertRaises(ValueError):
                validator.gp_sexpression(text)


if __name__ == "__main__":
    unittest.main()
