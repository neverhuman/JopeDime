"""Offline renderer contracts for the sealed train-only SCQ campaign."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from train_only_redteam_scq import (
    EXACT,
    JITTER,
    NODES,
    OWNERSHIP,
    PHASE_ORDER,
    SCQ,
    add_retry_attempt,
    immutable_import_tree,
    is_capacity_failure,
    render_spec,
    submit_attempt,
    unique_package_file,
    validate_selection_freeze,
    write_attempt_spec,
)


class TrainOnlyRedteamScqTests(unittest.TestCase):
    def test_renderer_uses_enforced_capability_placement(self) -> None:
        body = render_spec(
            job_id="redteam-test-v1",
            node="xbabe2",
            command="true",
            inputs=[
                {
                    "name": "manifest",
                    "source": "/tmp/manifest.json",
                    "target": "manifest.json",
                    "sha256": "a" * 64,
                }
            ],
            outputs=[("receipt", "receipt", True)],
        )
        self.assertNotIn("node =", body)
        self.assertIn("[[dependencies.smoke]]", body)
        self.assertIn("nvidia-smi -i GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b", body)
        self.assertIn("ram_mb = 40960", body)
        self.assertIn("[[input]]", body)
        self.assertIn('sha256 = "' + "a" * 64 + '"', body)
        self.assertIn("[[output]]", body)

    def test_sealed_placement_and_dataset_cardinality(self) -> None:
        self.assertEqual(len(EXACT), 4)
        self.assertEqual(len(JITTER), 3)
        self.assertEqual(set(OWNERSHIP), set(NODES))
        self.assertEqual(
            sum(len(datasets) for datasets in OWNERSHIP.values()),
            10,
        )
        self.assertEqual(
            sum(len(datasets) for datasets in OWNERSHIP.values()),
            len({dataset for datasets in OWNERSHIP.values() for dataset in datasets}),
        )
        self.assertEqual(
            PHASE_ORDER,
            (
                "build",
                "tests",
                "prepare",
                "proposals",
                "aggregate-proposals",
                "smoke",
                "full",
                "validate",
                "aggregate",
                "score",
                "report",
            ),
        )

    def test_launch_ready_capacity_is_retryable(self) -> None:
        payload = {
            "Error": {
                "code": "dependency-unsatisfied",
                "diagnostics": [
                    {
                        "kind": "eligible_node",
                        "requirement": "launch-ready-capacity",
                        "message": "wait for exact capacity",
                    }
                ],
            }
        }
        self.assertTrue(is_capacity_failure(payload))
        self.assertFalse(
            is_capacity_failure(
                {
                    "Error": {
                        "diagnostics": [
                            {
                                "requirement": "GPU UUID mismatch",
                                "message": "scientific placement failure",
                            }
                        ]
                    }
                }
            )
        )

    def test_worker_uses_staged_input_and_declared_output(self) -> None:
        body = render_spec(
            job_id="redteam-wiring-v1",
            node="xbabe4",
            command="{input:release_binary} --help > {output:receipt}/help.txt",
            inputs=[
                {
                    "name": "release_binary",
                    "source": "/controller/release",
                    "target": "release",
                    "sha256": "b" * 64,
                    "executable": True,
                }
            ],
            outputs=[("receipt", "receipt", True)],
        )
        self.assertIn("{input:release_binary}", body)
        self.assertIn("{output:receipt}", body)
        self.assertIn("executable = true", body)
        self.assertNotIn("/controller/release} --help", body)

    def test_immutable_import_rejects_unequal_existing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "receipt.json").write_text("new")
            (destination / "receipt.json").write_text("old")
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                immutable_import_tree(source, destination)

    def test_duplicate_unequal_fetch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a/result.json").write_text("one")
            (root / "b/result.json").write_text("two")
            with self.assertRaisesRegex(RuntimeError, "duplicate-unequal"):
                unique_package_file(root, "result.json")

    def test_retry_lineage_preserves_scientific_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = {
                "logical_id": "tests-xbabe1",
                "id": "redteam-tests-xbabe1-launch1",
                "node": "xbabe1",
                "command": "true",
                "inputs": [],
                "outputs": [("receipt", "receipt", True)],
                "upstream_package_sha256": {"build": "c" * 64},
            }
            first = write_attempt_spec(output, "tests", job, 0, [], None)
            state = {
                "phases": {
                    "tests": {
                        "jobs": {
                            "tests-xbabe1": {
                                "scientific_job_sha256": first["scientific_job_sha256"],
                                "attempts": [first],
                            }
                        }
                    }
                }
            }
            add_retry_attempt(
                output,
                state,
                "tests",
                job,
                "rpc-error: transport vanished",
            )
            retry = state["phases"]["tests"]["jobs"]["tests-xbabe1"]["attempts"][1]
            self.assertEqual(
                retry["scientific_job_sha256"],
                first["scientific_job_sha256"],
            )
            self.assertEqual(retry["predecessor_job_ids"], [first["job_id"]])
            self.assertEqual(retry["retry_index"], 1)
            self.assertNotEqual(retry["job_id"], first["job_id"])

    def test_stale_id_without_submission_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            meta = output / "attempt.meta.json"
            meta.write_text('{"complete_spec_hash":"abc"}')
            attempt = {
                "job_id": "stale-id",
                "spec_path": str(output / "attempt.toml"),
                "meta_path": str(meta),
                "state": "rendered",
            }
            with mock.patch(
                "train_only_redteam_scq.show_job",
                return_value={"state": "completed"},
            ):
                with self.assertRaisesRegex(RuntimeError, "stale SCQ ID"):
                    submit_attempt(output, attempt)

    def test_score_before_valid_freeze_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "validation-success.v1.json"
            freeze = root / "SELECTION_FREEZE.v1.json"
            validation.write_text(
                '{"schema":"TrainOnlyRedteamGlobalValidation.v1",'
                '"dataset_validation_sha256":{},"development_only":true,'
                '"confirmatory_valid":false}'
            )
            freeze.write_text(
                '{"schema":"TrainOnlyRedteamSelectionFreeze.v1",'
                '"run_id":"golden42-train-only-redteam-20260730-v1",'
                '"validation_sha256":"wrong","validated_dataset_count":10,'
                '"selection_frozen_before_test_access":true,'
                '"development_only":true,"confirmatory_valid":false}'
            )
            with self.assertRaisesRegex(RuntimeError, "score-before-freeze"):
                validate_selection_freeze(freeze, validation)

    @unittest.skipUnless(
        os.environ.get("REDTEAM_SCQ_INTEGRATION") == "1",
        "set REDTEAM_SCQ_INTEGRATION=1 for live SCQ placement checks",
    )
    def test_every_rendered_phase_passes_scq_check(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="redteam-scq-check-") as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts/train_only_redteam_scq.py"),
                    "plan",
                    "--output",
                    directory,
                    "--tag",
                    "integration-test-v1",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertTrue(SCQ.exists())


if __name__ == "__main__":
    unittest.main()
