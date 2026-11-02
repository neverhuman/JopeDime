#!/usr/bin/env python3

import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import smart_v3_cpu_scq as dispatcher


class SmartV3CpuScqTests(unittest.TestCase):
    def test_fixed_eleven_job_topology_and_dataset_assignment(self) -> None:
        progress = dispatcher.new_progress()
        self.assertEqual(len(progress["jobs"]), 11)
        assigned = [
            dataset
            for node in dispatcher.NODES
            for dataset in dispatcher.SHARDS[node]
        ]
        self.assertEqual(len(assigned), 10)
        self.assertEqual(len(set(assigned)), 10)
        self.assertEqual(
            dispatcher.SHARDS["xbabe1"], ("55f06d147bcc2a32",)
        )
        self.assertEqual(
            dispatcher.SHARDS["xbabe4"],
            ("7af85a04de0c7018", "f8536dfeef471c2f"),
        )
        self.assertEqual(
            {progress["jobs"][f"evolve-{dataset}"]["node"] for dataset in assigned},
            set(dispatcher.NODES),
        )
        self.assertEqual(progress["schema"], dispatcher.PROGRESS_SCHEMA)

    def test_evolution_command_is_smart_only_and_test_blind(self) -> None:
        command = dispatcher.evolution_command("xbabe2", "bd49e740a79bcb6c")
        self.assertEqual(command.count("run-train-only-redteam"), 1)
        self.assertEqual(command.count("--selection-mode smart-only"), 1)
        self.assertIn("--dataset bd49e740a79bcb6c", command)
        self.assertNotIn("g20-extension", command)
        self.assertNotIn("test.csv", command)
        self.assertNotIn("baseline", command.lower())
        self.assertNotIn("score-smart-v3-redteam", command)

    def test_evolution_and_score_resource_specs_parse(self) -> None:
        manifest = {
            "receipt_sha256": "a" * 64,
            "proposal_bundle": {"sha256": "b" * 64},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            workspace = root / "workspace.tar"
            source = root / "source.tar"
            for path in (binary, workspace, source):
                path.write_bytes(path.name.encode())
            inputs = [
                dispatcher.input_pin(
                    "release_binary", binary, "binary", executable=True
                ),
                dispatcher.input_pin(
                    "training_workspace", workspace, "workspace.tar"
                ),
                dispatcher.input_pin("source_bundle", source, "source.tar"),
            ]
            evolution = tomllib.loads(
                dispatcher.render_spec(
                    job_id="evolution",
                    node="xbabe1",
                    command=dispatcher.evolution_command("xbabe1", "55f06d147bcc2a32"),
                    inputs=inputs,
                    output_name="evolution-results",
                    output_source="evolution-results",
                    cpus=4,
                    ram_gb=16,
                    manifest=manifest,
                    source_sha256="c" * 64,
                    attempt=0,
                )
            )
            score = tomllib.loads(
                dispatcher.render_spec(
                    job_id="score",
                    node="xbabe4",
                    command=dispatcher.scoring_command(
                        [f"result_{index}" for index in range(4)]
                    ),
                    inputs=inputs,
                    output_name="smart-v3-results",
                    output_source="smart-v3-results",
                    cpus=2,
                    ram_gb=8,
                    manifest=manifest,
                    source_sha256="c" * 64,
                    attempt=0,
                )
            )
        self.assertEqual(evolution["resource"]["cpus_reserved"], 4)
        self.assertEqual(evolution["resource"]["ram_mb"], 16 * 1024)
        self.assertEqual(evolution["resource"]["gpus_per_task"], 0)
        self.assertIn("hostname pin xbabe1", evolution["dependencies"]["smoke"][0]["name"])
        self.assertNotIn("nvidia-smi", str(evolution))
        self.assertNotIn("GPU UUID", str(evolution))
        self.assertFalse(evolution["preempt"]["preemptible"])
        self.assertEqual(evolution["limits"]["max_runtime_hours"], 24)
        self.assertEqual(score["resource"]["cpus_reserved"], 2)
        self.assertEqual(score["resource"]["ram_mb"], 8 * 1024)
        self.assertIn("score-smart-v3-redteam", score["job"]["args"][2])
        self.assertNotIn("test.csv", evolution["job"]["args"][2])

    def test_retry_uses_one_new_identifier(self) -> None:
        logical = "evolve-xbabe3"
        self.assertNotEqual(
            dispatcher.job_id(logical, 0), dispatcher.job_id(logical, 1)
        )
        self.assertTrue(dispatcher.job_id(logical, 1).endswith("-r1"))

    def test_submission_receipt_adopts_job_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "job.toml"
            spec.write_text("job")
            attempt = {
                "job_id": dispatcher.job_id("evolve-bd49e740a79bcb6c", 0),
                "retry": 0,
                "spec_path": str(spec),
                "state": "rendered",
            }
            payload = {"Submitted": {"id": attempt["job_id"], "job": None}}
            dispatcher.write_submission_receipt(root, attempt, payload)
            with mock.patch.object(
                dispatcher,
                "shown_job",
                return_value={"state": "queued", "node": "xbabe2"},
            ):
                self.assertTrue(dispatcher.submit_attempt(root, attempt))
            self.assertEqual(attempt["state"], "queued")

    def test_null_job_after_accepted_submission_is_transient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "job.toml"
            spec.write_text("job")
            attempt = {
                "job_id": dispatcher.job_id("evolve-bd49e740a79bcb6c", 0),
                "retry": 0,
                "spec_path": str(spec),
                "state": "queued",
            }
            dispatcher.write_submission_receipt(
                root,
                attempt,
                {"Submitted": {"id": attempt["job_id"], "job": None}},
            )
            progress = {
                "jobs": {
                    "evolve-bd49e740a79bcb6c": {
                        "node": "xbabe2",
                        "dataset": "bd49e740a79bcb6c",
                        "attempts": [attempt],
                        "state": "queued",
                    }
                }
            }
            with mock.patch.object(dispatcher, "shown_job", return_value=None):
                with mock.patch.object(
                    dispatcher, "accepted_submission_vanished", return_value=False
                ):
                    self.assertEqual(
                        dispatcher.reconcile_job(
                            root, "evolve-bd49e740a79bcb6c", progress
                        ),
                        "accepted_invisible",
                    )

    def test_only_infrastructure_failure_allows_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "job.toml"
            spec.write_text("job")
            attempt = {
                "job_id": dispatcher.job_id("evolve-bd49e740a79bcb6c", 0),
                "retry": 0,
                "spec_path": str(spec),
                "state": "queued",
            }
            dispatcher.write_submission_receipt(
                root,
                attempt,
                {"Submitted": {"id": attempt["job_id"]}},
            )
            progress = {
                "jobs": {
                    "evolve-bd49e740a79bcb6c": {
                        "node": "xbabe2",
                        "dataset": "bd49e740a79bcb6c",
                        "attempts": [attempt],
                        "state": "queued",
                    }
                }
            }
            with mock.patch.object(
                dispatcher,
                "shown_job",
                return_value={"state": "failed", "failure_class": "user-error"},
            ):
                with self.assertRaisesRegex(RuntimeError, "non-retryable"):
                    dispatcher.reconcile_job(
                        root, "evolve-bd49e740a79bcb6c", progress
                    )
            with mock.patch.object(
                dispatcher,
                "shown_job",
                return_value={"state": "failed", "failure_class": "storage"},
            ):
                self.assertEqual(
                    dispatcher.reconcile_job(
                        root, "evolve-bd49e740a79bcb6c", progress
                    ),
                    "retry",
                )

    def test_wrong_node_marker_diagnostic_is_capacity_wait(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.toml"
            path.write_text(
                """
[env]
REDTEAM_EXPECTED_NODE = "xbabe4"
[[dependencies.smoke]]
name = "xbabe4 provisioned SSH capability"
"""
            )
            completed = mock.Mock(
                returncode=1,
                stdout="""{"Error":{"code":"dependency-unsatisfied","diagnostics":[
                    {"node":"xbabe2","requirement":"xbabe4 provisioned SSH capability"},
                    {"node":"xbabe3","requirement":"xbabe4 provisioned SSH capability"}
                ]}}""",
                stderr="",
            )
            with mock.patch.object(
                dispatcher.subprocess, "run", return_value=completed
            ):
                self.assertEqual(dispatcher.check_spec(path), "capacity_wait")


if __name__ == "__main__":
    unittest.main()
