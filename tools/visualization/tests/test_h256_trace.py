"""Portable rejection controls for the new encoder's explicit tensor graph."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h256_validate import check_graph, safe_file


def graph():
    return {
        "tensors": [
            {"id": "x", "name": "input", "role": "problem_encoder", "dtype": "float32",
             "shape": [1, 2], "values": [1., -2.], "producer_op_id": None},
            {"id": "y", "name": "observed", "role": "solution_encoder", "dtype": "float32",
             "shape": [1, 1], "values": [3.], "producer_op_id": "linear"},
        ],
        "operations": [{"id": "linear", "name": "saved.linear", "role": "solution_encoder",
                        "kind": "aten.addmm", "input_tensor_ids": ["x", "w"],
                        "output_tensor_ids": ["y"], "parameter_ids": ["w"], "parent_op_ids": []}],
    }


class EncoderGraphSafety(unittest.TestCase):
    def setUp(self):
        self.case, self.parameters = graph(), {"w": {}}

    def test_explicit_producer_graph_accepts_known_input(self):
        tensors, operations = check_graph(self.case, self.parameters)
        self.assertEqual(set(tensors), {"x", "y"})
        self.assertEqual(set(operations), {"linear"})

    def test_matching_dimensions_do_not_authorize_foreign_tensor(self):
        self.case["operations"][0]["input_tensor_ids"][0] = "other_run_same_shape"
        with self.assertRaisesRegex(ValueError, "foreign operation input"):
            check_graph(self.case, self.parameters)

    def test_output_cannot_claim_an_unobserved_producer(self):
        self.case["tensors"][1]["producer_op_id"] = "unobserved"
        with self.assertRaisesRegex(ValueError, "output producer"):
            check_graph(self.case, self.parameters)

    def test_future_or_missing_parent_is_rejected(self):
        self.case["operations"][0]["parent_op_ids"] = ["later"]
        with self.assertRaisesRegex(ValueError, "not topological"):
            check_graph(self.case, self.parameters)

    def test_foreign_weights_and_nonfinite_activation_are_rejected(self):
        changed = copy.deepcopy(self.case)
        changed["operations"][0]["parameter_ids"] = ["fresh_random_weights"]
        with self.assertRaisesRegex(ValueError, "foreign operation parameter"):
            check_graph(changed, self.parameters)
        self.case["tensors"][0]["values"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            check_graph(self.case, self.parameters)

    def test_source_inventory_cannot_escape_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            root.mkdir()
            (root.parent / "outside.py").write_text("outside")
            with self.assertRaisesRegex(ValueError, "escapes"):
                safe_file(root, "../outside.py")


if __name__ == "__main__":
    unittest.main()
