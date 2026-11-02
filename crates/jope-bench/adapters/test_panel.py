"""Panel contract tests and bounded, explicitly synthetic external-engine fit checks."""
import copy
import importlib
import json
import math
import os
import tempfile
from unittest.mock import patch
import hashlib
import sys
import unittest
from pathlib import Path

# Keep this synthetic suite within one CPU even when run directly. Constructors
# also pin estimator threads, and the smoke uses threadpoolctl for preloaded BLAS.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _panel


class ToyRegressor:
    def fit(self, x, y):
        self.mean = float(np.mean(y))
        self.width = x.shape[1]
        return self

    def predict(self, x):
        assert x.shape[1] == self.width
        return np.full(len(x), self.mean)


class ReversedClassifier:
    def fit(self, x, y):
        self.classes_ = np.unique(y)[::-1]
        self.width = x.shape[1]
        return self

    def predict_proba(self, x):
        assert x.shape[1] == self.width
        # Distinct column values make an accidental class-column permutation visible.
        weights = self.classes_ + 1.0
        return np.tile(weights / weights.sum(), (len(x), 1))


def request(task="multiclass"):
    classes = [] if task == "regression" else (["yes", "no"] if task == "binary" else ["zebra", "alpha", "middle"])
    return {"contract": _panel.CONTRACT, "task": task,
            "columns": [{"name": "numeric", "kind": "numeric"}, {"name": "category", "kind": "categorical"}],
            "classes": classes,
            "train": {"features": [["1", None], [None, "NA"], ["9", "null"], ["3", ""], ["5", "a"], ["7", "a"]],
                      "targets": (["1", "3", "5", "7", "9", "11"] if not classes else [classes[i % len(classes)] for i in range(6)])},
            "predict": {"features": [[None, "NA"], ["1000000", "unseen"], ["4", None]]}}


class PanelTests(unittest.TestCase):
    def test_probability_matrix_budget_is_checked_before_model_fit_or_predict(self):
        data = request("multiclass")
        data["columns"] = data["columns"][:1]
        data["train"]["features"] = [row[:1] for row in data["train"]["features"]]
        data["predict"]["features"] = [["1"]] * 30
        # Input matrices total 72 cells, but 30 x 3 probabilities exceed 80.
        with patch.object(_panel, "MAX_TRANSFORMED_CELLS", 80, create=True):
            create = unittest.mock.Mock(side_effect=lambda _: ReversedClassifier())
            with self.assertRaisesRegex(ValueError, "transformed matrix"):
                _panel.fit(data, create, "toy 1")
            create.assert_not_called()
            model = unittest.mock.Mock()
            with self.assertRaisesRegex(ValueError, "transformed matrix"):
                _panel.predictions(model, np.zeros((30, 2)), "multiclass", data["classes"])
            model.predict_proba.assert_not_called()

    def test_run_rejects_duplicate_json_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path, result_path = root / "request.json", root / "result.json"
            raw = json.dumps(request("regression"))
            raw = raw.replace('"contract":', '"contract":"untrusted-first-value","contract":', 1)
            request_path.write_text(raw)
            create = unittest.mock.Mock(side_effect=lambda _: ToyRegressor())
            with patch.object(sys, "argv", ["adapter", "--panel-v2", str(request_path), str(result_path)]):
                self.assertEqual(_panel.run(create, lambda: "toy 1"), 1)
            create.assert_not_called()
            self.assertFalse(result_path.exists())
            self.assertFalse(result_path.with_name("result.json.fitted.pkl").exists())

    def test_json_document_rejects_duplicate_fields_and_nonfinite_constants(self):
        for raw in (b'{"contract":"first","contract":"second"}',
                    b'{"train":{"targets":[],"targets":["replacement"]}}',
                    b'{"unknown":NaN}', b'{"unknown":Infinity}', b'{"unknown":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _panel.parse_request(raw)
        data = request("regression")
        self.assertEqual(_panel.parse_request(json.dumps(data).encode()), data)

    def test_transformed_matrix_budget_is_checked_before_model_construction(self):
        data = request("regression")
        # Two numeric outputs + five selected categories, across nine total rows.
        with patch.object(_panel, "MAX_TRANSFORMED_CELLS", 62, create=True), \
                patch.object(np, "zeros", wraps=np.zeros) as allocate:
            create = unittest.mock.Mock(side_effect=lambda _: ToyRegressor())
            with self.assertRaisesRegex(ValueError, "transformed matrix"):
                _panel.fit(data, create, "toy 1")
            create.assert_not_called()
            allocate.assert_not_called()
        with patch.object(_panel, "MAX_TRANSFORMED_CELLS", 63, create=True):
            result, artifact, checksum = _panel.fit(data, lambda _: ToyRegressor(), "toy 1")
            self.assertEqual(_panel._replay_trusted_artifact(artifact, checksum, data["predict"]["features"]), result)

    def test_transform_refuses_oversized_or_invalid_width_before_numpy_allocation(self):
        state = _panel.fit_preprocessing(request()["columns"], request()["train"]["features"])
        for width in (0, -1, True, getattr(_panel, "MAX_TRANSFORMED_CELLS", 16 * 1024 * 1024) + 1):
            with self.subTest(width=width), patch.object(np, "zeros") as allocate:
                invalid = dict(state, width=width)
                with self.assertRaisesRegex(ValueError, "transformed matrix"):
                    _panel.transform(invalid, [["1", "a"]])
                allocate.assert_not_called()

    def test_train_only_preprocessing_preserves_literal_categories_and_null(self):
        data = request()
        _panel.validate_request(data)
        state = _panel.fit_preprocessing(data["columns"], data["train"]["features"])
        self.assertEqual(state["fitted"][0]["fill"], 5.0)
        self.assertEqual(len(state["fitted"][1]["vocabulary"]), 5)
        self.assertNotIn(_panel._category("unseen"), state["fitted"][1]["vocabulary"])
        values = _panel.transform(state, [[None, v] for v in (None, "NA", "null", "", "unseen")])
        self.assertTrue(np.all(values[:, 0] == 5.0))
        self.assertTrue(np.all(values[:, 1] == 1.0))
        self.assertEqual(len({tuple(row) for row in values}), 5)
        self.assertTrue(np.all(values[-1, 2:] == 0.0))
        # Scoring features cannot change either vocabulary or numeric imputation.
        before = copy.deepcopy(state)
        _panel.transform(state, data["predict"]["features"])
        self.assertEqual(state, before)

    def test_median_matches_native_quantile_for_subnormal_and_signed_zero(self):
        columns = [{"name": "numeric", "kind": "numeric"}]
        smallest = math.ulp(0.0)
        state = _panel.fit_preprocessing(columns, [[str(smallest)]])
        self.assertEqual(state["fitted"][0]["fill"], smallest)
        state = _panel.fit_preprocessing(columns, [["0.0"], ["-0.0"], ["-0.0"]])
        # Rust total_cmp sorts negative zero first; the interpolation then adds +0.
        self.assertEqual(math.copysign(1.0, state["fitted"][0]["fill"]), 1.0)

    def test_probability_columns_match_declared_original_class_order(self):
        for task in ("binary", "multiclass"):
            data = request(task)
            result, artifact, checksum = _panel.fit(data, lambda _: ReversedClassifier(), "toy 1")
            self.assertEqual(result["data"]["classes"], data["classes"])
            expected = np.arange(1, len(data["classes"]) + 1, dtype=float)
            expected /= expected.sum()
            np.testing.assert_array_equal(result["data"]["values"][0], expected)
            self.assertEqual(_panel._replay_trusted_artifact(artifact, checksum, data["predict"]["features"]), result)
            self.assertEqual(hashlib.sha256(artifact).hexdigest(), checksum)
            saved = _panel._load_trusted_artifact(artifact)
            self.assertEqual(saved["classes"], data["classes"])
            self.assertEqual(saved["preprocessing"]["policy"], _panel.INPUT_POLICY)
            with self.assertRaisesRegex(ValueError, "checksum"):
                _panel._replay_trusted_artifact(artifact + b"changed", checksum, data["predict"]["features"])

    def test_regression_reload_preserves_complete_preprocessing(self):
        data = request("regression")
        result, artifact, checksum = _panel.fit(data, lambda _: ToyRegressor(), "toy 1")
        self.assertEqual(result, {"kind": "values", "data": [6.0, 6.0, 6.0]})
        self.assertEqual(_panel._replay_trusted_artifact(artifact, checksum, data["predict"]["features"]), result)

    def test_replay_good_pin_loads_and_mismatch_refuses_before_pickle(self):
        data = request("regression")
        result, artifact, checksum = _panel.fit(data, lambda _: ToyRegressor(), "toy 1")
        with patch.object(_panel, "_load_trusted_artifact", wraps=_panel._load_trusted_artifact) as loader:
            self.assertEqual(_panel._replay_trusted_artifact(artifact, checksum, data["predict"]["features"]), result)
            loader.assert_called_once()
        for pin in ("0" * 64, None, "not-a-sha256"):
            with self.subTest(pin=pin), patch.object(_panel, "_load_trusted_artifact") as loader:
                with self.assertRaisesRegex(ValueError, "checksum|SHA-256 pin"):
                    _panel._replay_trusted_artifact(artifact, pin, data["predict"]["features"])
                loader.assert_not_called()

    def test_replay_deserializes_verified_snapshot_of_mutable_buffer(self):
        data = request("regression")
        result, artifact, checksum = _panel.fit(data, lambda _: ToyRegressor(), "toy 1")
        buffer = bytearray(artifact)
        original_loads = _panel._load_trusted_artifact

        def mutate_then_load(snapshot):
            buffer[:] = b"replaced"
            return original_loads(snapshot)

        with patch.object(_panel, "_load_trusted_artifact", side_effect=mutate_then_load):
            self.assertEqual(_panel._replay_trusted_artifact(buffer, checksum, data["predict"]["features"]), result)

    def test_leaked_labels_unknown_fields_and_missing_classes_are_refused(self):
        for mutate in (lambda r: r["predict"].update(targets=["leak"]),
                       lambda r: r.update(heldout_targets=["leak"]),
                       lambda r: r["classes"].append("missing"),
                       lambda r: r["classes"].__setitem__(0, ""),
                       lambda r: r["classes"].append(r["classes"][0]),
                       lambda r: r["train"]["features"][0].__setitem__(0, "nan"),
                       lambda r: r["train"].__setitem__("targets", ["zebra"] * 6)):
            data = request()
            mutate(data)
            with self.assertRaises(ValueError):
                _panel.validate_request(data)

    def test_outputs_are_private_immutable_and_bind_exact_request_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "request.json"
            result_path = root / "result.json"
            source = (json.dumps(request("regression"), indent=2) + "\n").encode()
            request_path.write_bytes(source)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                _panel.read_request_bytes(request_path, limit=8)
            with patch.object(sys, "argv", ["adapter", "--panel-v2", str(request_path), str(result_path)]):
                self.assertEqual(_panel.run(lambda _: ToyRegressor(), lambda: "toy 1"), 0)
                saved_result = result_path.read_bytes()
                self.assertEqual(_panel.run(lambda _: ToyRegressor(), lambda: "toy 1"), 1)
                self.assertEqual(saved_result, result_path.read_bytes())
            result = json.loads(saved_result)
            self.assertEqual(result["input_sha256"], hashlib.sha256(source).hexdigest())
            artifact = root / result["fitted"]["path"]
            self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), result["fitted"]["sha256"])
            if os.name == "posix":
                self.assertEqual(result_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)

    def test_all_external_engines_fit_and_reload_each_task_on_synthetic_rows(self):
        from threadpoolctl import threadpool_limits

        adapters = ("xgboost_adapter", "lightgbm_adapter", "catboost_adapter", "sklearn_rf_adapter")
        completed = 0
        with threadpool_limits(limits=1):
            for adapter_name in adapters:
                # Exercise the real panel constructor. Required benchmark dependencies
                # are deliberately not skipped if unavailable in the benchmark CI lane.
                with patch.object(sys, "argv", ["adapter", "--panel-v2"]):
                    adapter = importlib.import_module(adapter_name)
                engine_version = adapter.version()
                self.assertTrue(engine_version.strip())
                for task in ("regression", "binary", "multiclass"):
                    with self.subTest(engine=engine_version, task=task):
                        data = request(task)  # Six synthetic training rows, null/unseen predictions.
                        output, artifact, checksum = _panel.fit(
                            data, adapter.create_panel_model, engine_version)
                        self.assertEqual(hashlib.sha256(artifact).hexdigest(), checksum)
                        self.assertEqual(_panel._replay_trusted_artifact(
                            artifact, checksum, data["predict"]["features"]), output)
                        saved = _panel._load_trusted_artifact(artifact)
                        self.assertEqual(saved["schema"], _panel.ARTIFACT)
                        self.assertEqual(saved["engine_version"], engine_version)
                        self.assertEqual(saved["task"], task)
                        self.assertEqual(saved["classes"], data["classes"])
                        self.assertEqual(saved["preprocessing"], _panel.fit_preprocessing(
                            data["columns"], data["train"]["features"]))
                        if task == "regression":
                            self.assertEqual(output["kind"], "values")
                            values = np.asarray(output["data"])
                            self.assertEqual(values.shape, (3,))
                        else:
                            self.assertEqual(output["kind"], "probabilities")
                            self.assertEqual(output["data"]["classes"], data["classes"])
                            values = np.asarray(output["data"]["values"])
                            self.assertEqual(values.shape, (3, len(data["classes"])))
                            self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))
                            self.assertTrue(np.all(np.abs(values.sum(axis=1) - 1.0) <= 1e-6))
                        self.assertTrue(np.all(np.isfinite(values)))
                        completed += 1
                        print(f"synthetic panel fit/reload: {engine_version}; task={task}; "
                              f"train_rows=6; prediction_rows=3; artifact_bytes={len(artifact)}")
        self.assertEqual(completed, 12)

    def test_onehot_cap_retains_full_vocabulary_and_records_selection(self):
        columns = [{"name": "category", "kind": "categorical"}]
        rows = [[f"level-{i:03}"] for i in range(260)] + [["level-259"]] * 2
        state = _panel.fit_preprocessing(columns, rows)
        self.assertEqual(len(state["fitted"][0]["vocabulary"]), 260)
        self.assertEqual(state["width"], 256)
        self.assertIn(_panel._category("level-259"), state["fitted"][0]["selected"])


if __name__ == "__main__":
    unittest.main()
