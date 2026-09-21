import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from direct_proposal_evaluation import evaluate, finish, identity, reference
from promotion_gate import immutable
from qualification_budget import Budget
from resume_expired_cycle import atomic, read, sha


class IdentityTests(unittest.TestCase):
    def setUp(self):
        from test_qualification_budget import synthetic_domain

        self.domain_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.domain_directory.cleanup)
        self.enterContext(synthetic_domain(self.domain_directory.name))

    def policy(self, binary):
        return dict(
            schema="parallel.label_admission.v2",
            evaluator="evaluator",
            evaluator_sha256=sha(binary),
            evaluator_protocol="featmath_training_fold_normalized_v1",
            authority_root="authority",
            authority_witness="witness",
            authority_instance="a" * 64,
            runtime_sha256="b" * 64,
            numerical_contract_sha256="c" * 64,
        )

    def test_actual_fake_processes_cannot_publish_quality_or_claim_physical_fits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cut = root / "prepared/cut-0"
            cut.mkdir(parents=True)
            (cut / "train.csv").write_text("1,2\n")
            (cut / "validation.csv").write_text("2,3\n")
            atomic(cut / "preprocessing.json", dict(target_scale=2.5))
            frozen = dict(
                index=0,
                training_sha256=sha(cut / "train.csv"),
                validation_sha256=sha(cut / "validation.csv"),
                preprocessing_sha256=sha(cut / "preprocessing.json"),
            )
            atomic(
                root / "prepared/prepared.json",
                dict(task="regression", meta_partition="meta_train", cuts=[frozen]),
            )
            binary = root / "evaluator"
            binary.write_text(
                '#!/usr/bin/python3\nimport sys,json,pathlib\na=sys.argv\np=pathlib.Path(a[a.index("--genome")+1]).read_text()\nif p=="failed":sys.exit(1)\npathlib.Path(a[a.index("--output")+1]).write_text(json.dumps({"rmse":0.4}))\n'
            )
            binary.chmod(0o755)
            vocabulary = root / "vocab"
            vocabulary.write_text("{}")
            context = dict(
                prepared="prepared",
                prepared_sha256=sha(root / "prepared/prepared.json"),
                partition="meta_train",
                family="f",
                cut=0,
            )
            cases = [dict(context=context, program=name) for name in ("success", "failed", "success")]
            result = evaluate(
                root,
                cases,
                binary,
                vocabulary,
                root / "fits",
                [min(os.sched_getaffinity(0))],
                Budget(root / "budget"),
                "test",
                admission_policy=self.policy(binary),
            )
            self.assertIsNone(result["physical_evaluations"])
            self.assertEqual(result["logical_slots"], 3)
            self.assertEqual(result["dispatched_requests"], 2)
            self.assertEqual(result["verified_physical_starts"], 0)
            self.assertEqual(result["unknown_consumption_requests"], 2)
            rows = result["evaluations"]
            self.assertEqual(rows[0]["receipt"], rows[2]["receipt"])
            self.assertTrue(rows[2]["evaluation_reused"])
            unverified = read(root / rows[0]["receipt"]["path"])
            self.assertIsNone(unverified["rmse"])
            self.assertEqual(unverified["status"], "unknown")
            failed = read(root / rows[1]["receipt"]["path"])
            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(failed["rmse"])

    def test_cut_units_evaluator_and_program_order_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared = root / "prepared"
            cut = prepared / "cut-0"
            cut.mkdir(parents=True)
            (cut / "train.csv").write_text("1,2\n")
            (cut / "validation.csv").write_text("2,3\n")
            atomic(cut / "preprocessing.json", dict(target_scale=2.5))
            value = dict(
                index=0,
                training_sha256=sha(cut / "train.csv"),
                validation_sha256=sha(cut / "validation.csv"),
                preprocessing_sha256=sha(cut / "preprocessing.json"),
            )
            atomic(
                prepared / "prepared.json", dict(task="regression", meta_partition="meta_train", cuts=[value])
            )
            (root / "native").write_bytes(b"frozen")
            (root / "vocab").write_bytes(b"vocabulary")
            context = dict(
                prepared="prepared",
                prepared_sha256=sha(prepared / "prepared.json"),
                partition="meta_train",
                family="f",
                cut=0,
            )
            a = identity(root, context, "(pipe X (rank) (ridge))", root / "native", root / "vocab")
            b = identity(root, context, "(pipe X (ridge) (rank))", root / "native", root / "vocab")
            self.assertNotEqual(a["program_sha256"], b["program_sha256"])
            self.assertEqual(a["target_scale"], 2.5)
            self.assertEqual(a["units"], "original_target")
            (root / "native").write_bytes(b"another evaluator")
            c = identity(root, context, a["program"], root / "native", root / "vocab")
            self.assertNotEqual(a["evaluator_sha256"], c["evaluator_sha256"])
            (cut / "validation.csv").write_text("tampered")
            with self.assertRaisesRegex(RuntimeError, "content changed"):
                identity(root, context, a["program"], root / "native", root / "vocab")

    def test_missing_approved_policy_blocks_before_dispatch(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("direct_proposal_evaluation.runtime.popen") as spawn,
        ):
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "frozen native admission policy"):
                evaluate(
                    root,
                    [{}],
                    root / "evaluator",
                    root / "vocab",
                    root / "fits",
                    [0],
                    Budget(root / "budget"),
                    "test",
                )
            spawn.assert_not_called()
            self.assertFalse((root / "fits").exists())

    def test_compact_input_without_independent_native_identity_is_not_dispatched(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("direct_proposal_evaluation.runtime.popen") as spawn,
        ):
            root = Path(folder)
            cut = root / "prepared/cut-0"
            cut.mkdir(parents=True)
            atomic(cut / "preprocessing.json", dict(target_scale=2.5))
            atomic(
                root / "prepared/prepared.json",
                dict(
                    task="regression",
                    meta_partition="meta_train",
                    storage={},
                    cuts=[
                        dict(
                            index=0,
                            training_sha256="a" * 64,
                            validation_sha256="b" * 64,
                            preprocessing_sha256=sha(cut / "preprocessing.json"),
                        )
                    ],
                ),
            )
            binary = root / "evaluator"
            binary.write_text("no executable needed")
            vocabulary = root / "vocab"
            vocabulary.write_text("{}")
            context = dict(
                prepared="prepared",
                prepared_sha256=sha(root / "prepared/prepared.json"),
                partition="meta_train",
                family="f",
                cut=0,
            )
            with self.assertRaisesRegex(RuntimeError, "compact direct view"):
                evaluate(
                    root,
                    [dict(context=context, program="(pipe X (ridge))")],
                    binary,
                    vocabulary,
                    root / "fits",
                    [0],
                    Budget(root / "budget"),
                    "test",
                    admission_policy=self.policy(binary),
                )
            spawn.assert_not_called()


class CompletionTests(unittest.TestCase):
    """The positive branch mocks committed verification, never numerical fitting."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "request"
        self.directory.mkdir()
        self.bound = dict(program_sha256="d" * 64, evaluator_sha256="e" * 64, target_scale=2.5)
        self.policy = dict(
            evaluator="evaluator",
            authority_root="authority",
            authority_witness="witness",
            authority_instance="a" * 64,
            runtime_sha256="b" * 64,
            numerical_contract_sha256="c" * 64,
        )
        self.native = dict(task_id=str(self.directory / "fit.json"), elapsed_seconds=0.2)
        self.native_ref = self.put("native.json", self.native)
        self.model = self.put("model.json", dict(synthetic=True))
        self.metrics = dict(
            task="regression",
            primary_metric="rmse",
            units="cut_normalized_target",
            rmse=0.4,
            fit_completion=self.native_ref,
            fitted_model=self.model,
            evaluation_origin="execution",
            physical_fit_starts=1,
        )
        self.attempt = self.put("attempt.json", dict(synthetic_attempt=True))

    def put(self, name, value):
        immutable(self.root / name, value)
        return reference(self.root, self.root / name)

    def done(self, *, code=0, elapsed=0.5, timed_out=False):
        immutable(self.directory / "fit.json", self.metrics)
        return finish(
            self.root, self.directory, self.bound, self.policy, self.attempt, code, elapsed, timed_out, 12345
        )

    def test_verified_original_and_registry_reuse_have_distinct_physical_counts(self):
        with patch(
            "direct_proposal_evaluation.admission.clean_completion", return_value=({}, self.native)
        ) as verify:
            result = self.done()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rmse"], 1.0)
        self.assertEqual(result["verified_physical_starts"], 1)
        self.assertEqual(result["unknown_consumption_requests"], 0)
        self.assertFalse(result["registered_query_receipt_qualified"])
        verify.assert_called_once()
        self.directory = self.root / "reuse"
        self.directory.mkdir()
        self.metrics.update(evaluation_origin="reuse", physical_fit_starts=0)
        with patch("direct_proposal_evaluation.admission.clean_completion", return_value=({}, self.native)):
            reused = self.done()
        self.assertEqual(reused["status"], "ok")
        self.assertEqual(reused["verified_physical_starts"], 0)
        self.assertEqual(reused["registry_reuse_requests"], 1)
        self.assertEqual(reused["unknown_consumption_requests"], 0)

    def test_late_successful_exit_cannot_clear_timeout(self):
        with patch("direct_proposal_evaluation.admission.clean_completion") as verify:
            result = self.done(code=0, elapsed=30.1, timed_out=True)
        verify.assert_not_called()
        self.assertEqual(result["status"], "timeout")
        self.assertIsNone(result["rmse"])
        wrapper = read(self.root / result["clean_completion"]["path"])
        self.assertEqual(wrapper["exit_code"], 0)
        self.assertTrue(wrapper["deadline_exceeded"])
        self.assertEqual(result["unknown_consumption_requests"], 1)

    def test_signal_with_valid_looking_result_remains_failed(self):
        with patch("direct_proposal_evaluation.admission.clean_completion") as verify:
            result = self.done(code=-9)
        verify.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["rmse"])
        self.assertEqual(read(self.root / result["clean_completion"]["path"])["signal"], 9)

    def test_uncommitted_or_missing_receipt_never_becomes_quality(self):
        with patch(
            "direct_proposal_evaluation.admission.clean_completion",
            side_effect=RuntimeError("uncommitted native receipt"),
        ):
            result = self.done()
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["rmse"])
        self.assertEqual(result["verified_physical_starts"], 0)
        self.assertEqual(result["unknown_consumption_requests"], 1)
        self.directory = self.root / "missing"
        self.directory.mkdir()
        self.metrics.pop("fit_completion")
        missing = self.done()
        self.assertEqual(missing["status"], "unknown")
        self.assertIsNone(missing["rmse"])

    def test_wrong_units_or_start_attribution_block_quality(self):
        self.metrics["units"] = "original_target"
        with patch("direct_proposal_evaluation.admission.clean_completion") as verify:
            result = self.done()
        verify.assert_not_called()
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["rmse"])
        self.directory = self.root / "wrong-start"
        self.directory.mkdir()
        self.metrics["units"] = "cut_normalized_target"
        with patch("direct_proposal_evaluation.admission.clean_completion", return_value=({}, self.native)):
            result = self.done()
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["rmse"])
