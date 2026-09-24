#!/usr/bin/env python3
"""Metadata-only tamper and independent score tests. No jope/model execution."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("demo_qualifier", Path(__file__).with_name("qualify-demo.py"))
qualifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualifier)


class DemoQualificationTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / "scratch"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name).resolve()
        self.source = "a" * 40
        self.binary = "b" * 64

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return qualifier.sha256(path)

    def fixture(self, task):
        classes = (
            []
            if task == "regression"
            else ["negative", "positive"]
            if task == "binary"
            else ["zebra", "alpha", "middle"]
        )
        if task == "regression":
            mean = sum(2.0 * (i / 8.0 - 4.0) + 1.0 for i in range(60)) / 60
            predictions = {"kind": "values", "data": [mean] * 15}
            state = {"kind": "train_mean", "value": mean}
        else:
            predictions = {
                "kind": "probabilities",
                "data": {
                    "classes": classes,
                    "values": [[1.0 / len(classes)] * len(classes) for _ in range(15)],
                },
            }
            state = {
                "kind": "train_class_priors",
                "classes": classes,
                "probabilities": [1.0 / len(classes)] * len(classes),
            }
        fitted_sha = self.write("fitted.json", {"synthetic_metadata_only": True})
        worker_sha = self.write(
            "journal/jobs/fixture/result.json", {"artifact_sha256": fitted_sha, "predictions": predictions}
        )
        execution = {
            "command_status": "completed",
            "exit_code": 0,
            "outputs": [{"path": str(self.root / "journal/jobs/fixture/result.json"), "sha256": worker_sha}],
        }
        score = qualifier.fixture_score(task, classes, predictions)
        plan_sha = self.write("demo-plan.json", {"experiment": {"source_sha": self.source}})
        hashes = {
            name: self.write(name, value)
            for name, value in [
                ("execution.json", execution),
                ("predictions.json", predictions),
                ("baseline.json", {"state": state, "predictions": predictions, "score": score}),
            ]
        }
        result = {
            "schema": "jope.warm-demo/v1",
            "ok": True,
            "task": task,
            "input": {"kind": "synthetic", "task": task, "version": "jope.synthetic-demo/v1"},
            "classes": classes,
            "source_sha": self.source,
            "plan_sha256": "c" * 64,
            "artifact_sha256": hashes,
            "worker_fit_attempts": 1,
            "fitted_artifacts": 1,
            "neural_consumed": False,
            "sealed_inputs_opened": False,
            "fitted": {"path": str(self.root / "fitted.json"), "sha256": fitted_sha, "reload_verified": True},
            "model_score": score,
            "baseline_score": score,
        }
        result_sha = self.write("demo-result.json", result)
        invocation = {
            "schema": "jope.warm-demo-invocation/v1",
            "ok": True,
            "result": result,
            "result_file_sha256": result_sha,
            "plan_file_sha256": plan_sha,
            "plan_sha256": "c" * 64,
            "program": {"sha256": self.binary},
            "source_sha": self.source,
            "warm_elapsed_seconds": 3.0,
            "under_60_seconds": True,
        }
        self.write("invocation-receipt.json", invocation)
        return invocation

    def verify(self, task):
        return qualifier.verify_demo(self.root, task, self.source, self.binary)

    def test_metadata_fixture_three_tasks_and_independent_scores(self):
        for task in ["regression", "binary", "multiclass"]:
            with self.subTest(task=task):
                self.fixture(task)
                self.assertTrue(self.verify(task)["ok"])

    def test_byte_tampering_and_wrong_installed_binary_are_rejected(self):
        for filename in [
            "demo-plan.json",
            "demo-result.json",
            "predictions.json",
            "baseline.json",
            "execution.json",
            "fitted.json",
            "journal/jobs/fixture/result.json",
        ]:
            with self.subTest(file=filename):
                self.fixture("regression")
                with (self.root / filename).open("a") as stream:
                    stream.write(" ")
                with self.assertRaises(ValueError):
                    self.verify("regression")
        self.fixture("binary")
        with self.assertRaises(ValueError):
            qualifier.verify_demo(self.root, "binary", self.source, "d" * 64)

    def test_self_consistent_false_score_is_still_rejected(self):
        invocation = self.fixture("regression")
        invocation["result"]["model_score"] = {"metric": "rmse", "value": 0.0, "loss": 0.0}
        invocation["result_file_sha256"] = self.write("demo-result.json", invocation["result"])
        self.write("invocation-receipt.json", invocation)
        with self.assertRaisesRegex(ValueError, "independently"):
            self.verify("regression")

    def test_class_order_and_invalid_probability_rows_are_rejected(self):
        classes = ["negative", "positive"]
        for rows in [[[math.nan, 0.0]] * 15, [[0.2, 0.2]] * 15, [[1.0]] * 15]:
            with self.assertRaises(ValueError):
                qualifier.fixture_score(
                    "binary", classes, {"kind": "probabilities", "data": {"classes": classes, "values": rows}}
                )
        with self.assertRaisesRegex(ValueError, "ordering"):
            qualifier.fixture_score(
                "binary",
                classes[::-1],
                {"kind": "probabilities", "data": {"classes": classes[::-1], "values": [[0.5, 0.5]] * 15}},
            )


if __name__ == "__main__":
    unittest.main()
