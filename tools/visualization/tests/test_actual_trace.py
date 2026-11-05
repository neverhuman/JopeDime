"""Portable tiny controls for same-run journal binding, not runtime evidence.

These invented two-input tensors exercise the validator's identity rules. They
contain no private model, dataset, source export, or production capture assets.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("actual_trace_validator", Path(__file__).parents[1] / "validate_trace.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actual_control(root):
    run_id = "synthetic-control-run"
    dataset_id = "synthetic-control-dataset"
    identity = {"run_id": run_id, "dataset_id": dataset_id,
                "fit_matrix_sha256": "f" * 64, "bundle_digest": "synthetic-control-only"}
    layer = {"sequence": 0, "network": "dope", "name": "control-affine",
             "weight_layout": "out_in", "input_shape": [1, 2], "weight_shape": [1, 2],
             "output_shape": [1, 1], "input": [2.0, -3.0], "weights": [0.5, -2.0],
             "bias": [1.0], "output": [8.0], "closure_max_normalized_error": 0.0}
    bridge = {"sequence": 1, "network": "bridge", "name": "stage1b_tokens",
              "shape": [1, 10, 32], "values": [0.0] * 320}
    continuation = {**copy.deepcopy(layer), "network": "jope", "name": "control-continuation"}
    fit = {"call_id": "conditioning", "identity": {**identity, "phase": "fit_conditioning"},
           "trace": {"affine": [layer], "tensors": [bridge]},
           "outputs": {"dope629": [0.0] * 629, "tokens": [0.0] * 320}}
    admission = {"call_id": "round-1", "identity": {**identity, "phase": "ranker_admission"},
                 "trace": {"affine": [continuation], "tensors": []}, "outputs": {"total_cost": [8.0]}}
    journal = {"schema": "jope.actual-run-visual-journal/v1", "run_id": run_id,
               "dropped_records": 0, "calls": [fit, admission]}
    source = {"run_id": run_id}
    for key, filename in (("proof", "execution-control.json"), ("training_proof", "training-control.json")):
        path = root / filename
        path.write_text(json.dumps({"scope": "synthetic validator control only", "run_id": run_id}))
        source[key + "_path"] = filename
        source[key + "_sha256"] = sha(path)
    case = {"id": "control-a", "run_id": run_id, "dataset_id": dataset_id, "diagnostic": False,
            "dataset_sha256": "c" * 64, "source": source, "tokens_per_second": None,
            "native_journal_path": "native-control.json", "native_journal_sha256": "",
            "display_call_ids": ["conditioning", "round-1"],
            "validation": {"actual_run_binding": True, "observed_affine_closure_passed": True,
                           "source_execution_parity": True, "source_self_checks_passed": True, "dropped_records": 0},
            "trace": {family: [{**copy.deepcopy(item), "call_id": call["call_id"]}
                                for call in journal["calls"] for item in call["trace"][family]]
                      for family in ("affine", "tensors")},
            "outputs": {**copy.deepcopy(fit["outputs"]), "costs": [8.0]}}
    data = {"schema": validator.RUN_SCHEMA, "source_commit": "a" * 40,
            "runtime_checkout_head": "a" * 40, "build_base_commit": "a" * 40,
            "source_identity_verified": True, "runtime_service_source_dirty": False,
            "diagnostic_slate_substitution": False, "bundle_digest": "synthetic-control-only", "cases": [case]}
    return data, journal


class ActualRunTraceBinding(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="actual-trace-control-", dir=Path(__file__).parents[1])
        self.root = Path(self.scratch.name)
        self.data, self.journal = actual_control(self.root)

    def tearDown(self):
        self.scratch.cleanup()

    def write(self, data=None, journal=None):
        data = copy.deepcopy(self.data if data is None else data)
        journal = self.journal if journal is None else journal
        journal_path = self.root / "native-control.json"
        journal_path.write_text(json.dumps(journal))
        data["cases"][0]["native_journal_sha256"] = sha(journal_path)
        path = self.root / "trace.json"
        path.write_text(json.dumps(data))
        return path

    def test_valid_tiny_actual_run_binding(self):
        _, report = validator.validate(self.write())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["cases"][0]["affine_calls"], 2)
        self.assertIn("same actual Engine", report["scope"])

    def test_missing_native_call_is_rejected(self):
        self.journal["calls"].pop()
        with self.assertRaisesRegex(ValueError, "native call inventory"):
            validator.validate(self.write())

    def test_display_cannot_drop_native_observations(self):
        self.data["cases"][0]["trace"]["affine"].pop()
        with self.assertRaisesRegex(ValueError, "drops native observations"):
            validator.validate(self.write())

    def test_changed_fp32_bit_is_rejected_even_within_closure_tolerance(self):
        self.data["cases"][0]["trace"]["affine"][0]["input"][0] = 2.000000238418579
        with self.assertRaisesRegex(ValueError, "FP32 bits"):
            validator.validate(self.write())

    def test_changed_operation_metadata_is_rejected(self):
        for key, value in (("name", "invented-operation"), ("sequence", 99), ("input_shape", [2])):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["cases"][0]["trace"]["affine"][0][key] = value
                with self.assertRaisesRegex(ValueError, "native metadata"):
                    validator.validate(self.write(data))

    def test_short_and_promoted_json_decimals_with_identical_fp32_are_accepted(self):
        native = self.journal["calls"][0]["trace"]["affine"][0]
        displayed = self.data["cases"][0]["trace"]["affine"][0]
        native["input"][0], native["output"][0] = 0.1, 7.05
        displayed["input"][0], displayed["output"][0] = 0.10000000149011612, 7.050000190734863
        self.assertEqual(validator.validate(self.write())[1]["status"], "PASS")

    def test_opposite_zero_sign_is_rejected(self):
        native = self.journal["calls"][0]["trace"]["affine"][0]
        displayed = self.data["cases"][0]["trace"]["affine"][0]
        native["input"][0], native["output"][0] = -0.0, 7.0
        displayed["input"][0], displayed["output"][0] = 0.0, 7.0
        with self.assertRaisesRegex(ValueError, "FP32 bits"):
            validator.validate(self.write())

    def test_source_from_another_run_is_rejected(self):
        self.data["cases"][0]["source"]["run_id"] = "another-control-run"
        with self.assertRaisesRegex(ValueError, "source comes from a different run"):
            validator.validate(self.write())

    def test_corrupt_journal_hash_is_rejected(self):
        path = self.write()
        with (self.root / "native-control.json").open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "journal identity differs"):
            validator.validate(path)

    def test_changed_execution_proof_is_rejected(self):
        path = self.write()
        (self.root / "execution-control.json").write_text("changed control proof")
        with self.assertRaisesRegex(ValueError, "execution proof identity differs"):
            validator.validate(path)

    def add_non_displayed_call(self):
        later = copy.deepcopy(self.journal["calls"][1])
        later["call_id"] = "round-2-not-displayed"
        self.journal["calls"].append(later)
        return later

    def test_non_displayed_calls_keep_dataset_fit_and_bundle_identity(self):
        for field in ("dataset_id", "fit_matrix_sha256", "bundle_digest"):
            with self.subTest(field=field):
                journal = copy.deepcopy(self.journal)
                later = copy.deepcopy(journal["calls"][1])
                later["call_id"] = "round-2-not-displayed"
                later["identity"][field] = "another-control-identity"
                journal["calls"].append(later)
                with self.assertRaisesRegex(ValueError, "dataset/FIT/bundle identity"):
                    validator.validate(self.write(journal=journal))

    def test_non_displayed_call_corrupt_affine_is_rejected(self):
        later = self.add_non_displayed_call()
        later["trace"]["affine"][0]["output"][0] += 1.0
        with self.assertRaisesRegex(ValueError, "later native journal affine closure"):
            validator.validate(self.write())

    def test_non_displayed_call_nonfinite_readout_is_rejected(self):
        later = self.add_non_displayed_call()
        later["trace"]["tensors"] = [{"sequence": 1, "network": "jope", "name": "later-readout",
                                         "shape": [1], "values": [float("inf")]}]
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validator.validate(self.write())

    def test_unverified_dirty_or_mismatched_source_stamp_is_rejected(self):
        for key, value in (("source_identity_verified", False), ("runtime_service_source_dirty", True),
                           ("build_base_commit", "b" * 40), ("diagnostic_slate_substitution", True)):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaisesRegex(ValueError, "source|diagnostic slate"):
                    validator.validate(self.write(data))


if __name__ == "__main__":
    unittest.main()
