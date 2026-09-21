import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import label_admission as admission
from promotion_gate import canonical, immutable


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "evaluator").write_bytes(b"synthetic native verifier fixture")
        preprocessing = self.put("preprocessing.json", dict(target_scale=2))
        self.context = dict(
            family="f",
            cut=1,
            partition="meta_train",
            **{
                key: digest(key)
                for key in (
                    "prepared_sha256",
                    "training_sha256",
                    "validation_sha256",
                    "evaluator_sha256",
                    "vocabulary_sha256",
                    "fit_training_sha256",
                    "feature_view_sha256",
                )
            },
            preprocessing_sha256=preprocessing["sha256"],
            evaluator="evaluator",
            authority_root="authority",
            authority_witness="witness",
            authority_instance=digest("authority"),
            runtime_sha256=digest("runtime"),
            numerical_contract_sha256=digest("contract"),
        )
        self.context["evaluator_sha256"] = hashlib.sha256((self.root / "evaluator").read_bytes()).hexdigest()
        program = "(pipe X (ridge))"
        self.candidate = dict(
            transfer_eligible=True,
            admission_track="transfer_qualification",
            canonical_program=program,
            program_sha256=digest(program),
            historical_information={"score": 0.00001},
        )
        self.row = dict(
            schema="parallel.current_program_evaluation.v1",
            status="ok",
            context=self.context,
            program_sha256=self.candidate["program_sha256"],
            final_tests_opened=False,
            target_scale=2,
            evaluation_origin="execution",
            metrics=self.put(
                "metric.json", dict(rmse=3, units="cut_normalized_target", evaluation_origin="execution")
            ),
            preprocessing=preprocessing,
            evaluation_identity_sha256=digest("evaluation"),
            runtime_seconds=0.1,
            numerical_runtime_seconds=0.05,
        )
        model = self.put("model.json", dict(synthetic_model=True))
        fit = dict(
            schema="parallel.fit_identity.v2",
            canonical_program=program,
            program_sha256=self.candidate["program_sha256"],
            training_sha256=self.context["fit_training_sha256"],
            preprocessing_sha256=preprocessing["sha256"],
            feature_view_sha256=self.context["feature_view_sha256"],
            executable_sha256=self.context["evaluator_sha256"],
            runtime_sha256=digest("runtime"),
            fit_seed=0,
            numerical_contract_sha256=digest("contract"),
        )
        fit_key = hashlib.sha256(json.dumps(fit, separators=(",", ":")).encode()).hexdigest()
        self.native = dict(
            schema="parallel.fit_completion.v2",
            fit_identity=fit,
            fit_key=fit_key,
            training_sha256=self.context["fit_training_sha256"],
            program_sha256=self.candidate["program_sha256"],
            executable_sha256=self.context["evaluator_sha256"],
            outcome="success",
            observed_exit_code=0,
            observed_signal=None,
            elapsed_seconds=0.05,
            deadline_seconds=30.0,
            model_sha256=model["sha256"],
        )
        self.native["clean_completion_sha256"] = hashlib.sha256(canonical(self.native)).hexdigest()
        attempt = dict(
            evaluation_identity_sha256=self.row["evaluation_identity_sha256"],
            program_sha256=self.candidate["program_sha256"],
            evaluator_sha256=self.context["evaluator_sha256"],
        )
        self.completion = dict(
            schema="parallel.clean_completion.v3",
            **attempt,
            attempt=self.put("attempt.json", attempt),
            fit_completion=self.put("native.json", self.native),
            outcome="success",
            exit_observed=True,
            exit_code=0,
            signal=None,
            deadline_exceeded=False,
            runtime_seconds=0.1,
            deadline_seconds=30.0,
            output=self.row["metrics"],
            model=model,
        )
        self.row["clean_completion"] = self.put("completion.json", self.completion)
        self.receipt = self.put("receipt.json", self.row)
        self.verifier = patch(
            "label_admission.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, json.dumps(self.native), ""),
        )
        self.verify = self.verifier.start()
        self.addCleanup(self.verifier.stop)

    def put(self, name, value):
        return dict(path=name, sha256=immutable(self.root / name, value))

    def test_cross_identity_units_historical_and_conflicting_labels_fail_closed(self):
        label = admission.admit(self.root, self.candidate, self.receipt, self.context)
        self.assertEqual(label["rmse"], 6)
        command = self.verify.call_args.args[0]
        self.assertIn("verify-fit-completion", command)
        self.assertIn(self.context["authority_instance"], command)
        self.assertIn(self.completion["fit_completion"]["sha256"], command)
        for key in ("cut", "evaluator_sha256", "vocabulary_sha256", "training_sha256"):
            expected = dict(self.context)
            expected[key] = "wrong"
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                admission.admit(self.root, self.candidate, self.receipt, expected)
        with self.assertRaisesRegex(RuntimeError, "current evaluation receipt"):
            admission.admit(self.root, self.candidate, None, self.context)
        with self.assertRaisesRegex(RuntimeError, "collection-only"):
            admission.admit(
                self.root, dict(self.candidate, transfer_eligible=False), self.receipt, self.context
            )
        other = copy.deepcopy(label)
        other["rmse"] = 5
        with self.assertRaisesRegex(RuntimeError, "conflicting duplicate"):
            admission.deduplicate([label, other])
        self.assertEqual(len(admission.deduplicate([label, label])), 1)
        badrow = dict(self.row, metrics=self.put("bad-metric.json", dict(rmse=3, units="original_target")))
        with self.assertRaisesRegex(RuntimeError, "conflicting evaluator units"):
            admission.admit(self.root, self.candidate, self.put("bad-receipt.json", badrow), self.context)

    def test_result_file_cannot_clear_original_exit_deadline_or_missing_evidence(self):
        for field, value in [
            ("exit_code", 1),
            ("exit_observed", False),
            ("signal", 9),
            ("deadline_exceeded", True),
            ("runtime_seconds", None),
            ("fit_completion", None),
        ]:
            completion = dict(self.completion)
            completion[field] = value
            row = dict(self.row, clean_completion=self.put("bad-" + field + ".json", completion))
            with self.assertRaises((RuntimeError, TypeError)):
                admission.admit(
                    self.root, self.candidate, self.put("row-" + field + ".json", row), self.context
                )
        row = dict(self.row)
        row.pop("clean_completion")
        with self.assertRaisesRegex(RuntimeError, "original clean completion"):
            admission.admit(self.root, self.candidate, self.put("no-completion.json", row), self.context)
        (self.root / "model.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "fitted model changed"):
            admission.admit(self.root, self.candidate, self.receipt, self.context)

    def test_uncommitted_receipt_and_cache_reuse_cannot_teach(self):
        self.verify.side_effect = subprocess.CalledProcessError(1, ["verify-fit-completion"])
        with self.assertRaises(subprocess.CalledProcessError):
            admission.admit(self.root, self.candidate, self.receipt, self.context)
        self.verify.side_effect = None
        row = dict(self.row, evaluation_origin="reuse")
        with self.assertRaisesRegex(RuntimeError, "cache reuse"):
            admission.admit(self.root, self.candidate, self.put("reuse.json", row), self.context)
        row = dict(self.row, numerical_runtime_seconds=0.001)
        with self.assertRaisesRegex(RuntimeError, "numerical runtime changed"):
            admission.admit(self.root, self.candidate, self.put("wrong-fit-runtime.json", row), self.context)

    def test_tied_scores_have_equal_targets_at_original_precision(self):
        for actual, expected in zip(admission.tied_percentiles([1, 1, 2, 3]), [5 / 6, 5 / 6, 1 / 3, 0]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(admission.tied_percentiles([1, 1, 1]), [0.5, 0.5, 0.5])
        self.assertEqual(admission.tied_percentiles([0, 0, 1]), [0.75, 0.75, 0])
        self.assertEqual(admission.tied_percentiles([1.0, 1.0 + 2**-52]), [1.0, 0.0])
