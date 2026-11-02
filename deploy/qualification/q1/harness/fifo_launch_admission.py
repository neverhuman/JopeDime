"""FIFO launch unittest mixins, imported by ``test_fifo_launch``."""
from pathlib import Path
import fcntl
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fifo_launch as launch
import q1
import q1_cut


class FIFOAdmissionTests:
    def test_signed_metadata_source_is_checked_without_relabeling_compiled_source(self):
        import json
        import q1_bindings

        compiled = json.loads(q1_bindings.EVALUATOR_PIN.read_text())["source_commit"]
        self.plan["source_commit"] = compiled
        self.plan["metadata_source_commit"] = "b" * 40
        self.resign()
        seen = []
        launch.admit(
            self.args,
            self.paths,
            source_check=lambda paths, source: seen.append(source),
            load=lambda: 0,
        )
        self.assertEqual(seen, ["b" * 40])
        self.assertEqual(self.plan["source_commit"], compiled)
        self.plan["metadata_source_commit"] = "c" * 40
        self.planfile.write_bytes(q1.canonical(self.plan))
        with self.assertRaises(ValueError):
            launch.admit(
                self.args, self.paths, source_check=lambda *_: None, load=lambda: 0
            )
        self.assertFalse(self.paths.queue.exists() and list(self.paths.queue.iterdir()))

    def test_preparation_is_not_a_live_ticket_or_lock_claim(self):
        sentinel = self.queue / "w1-owner.json"
        sentinel.write_text('{"owner":"fixture-W1"}')
        before = sentinel.read_bytes()
        with patch.object(fcntl, "flock", side_effect=AssertionError("lock touched")):
            intent = launch.prepare(self.paths, "unit-fixture")
        self.assertEqual(intent["status"], "prepared_not_enqueued")
        self.assertIsNone(intent["active_ticket"])
        self.assertIsNone(intent["planning"]["expected_wall_seconds"])
        self.assertEqual(intent["planning"]["serial_timeout_seconds"], 144000)
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertEqual(list(self.queue.iterdir()), [sentinel])

    def test_synthetic_dry_run_writes_unknown_decision_without_lock_or_payload(self):
        sentinel = self.queue / "w1-owner.json"
        sentinel.write_text('{"owner":"fixture-W1"}')
        before = sentinel.read_bytes()
        output = self.root / "decision-record.json"
        with patch.object(fcntl, "flock", side_effect=AssertionError("lock touched")):
            manifest = launch.load_input_manifest(self.paths.input_manifest, self.paths)
            record = launch.dry_run(self.paths.input_manifest, output, self.paths)
        self.assertTrue(manifest["datasets"])
        self.assertEqual(
            set(manifest["datasets"][0]),
            {"dataset_id", "sha256", "s3_manifest_sha256"},
        )
        self.assertEqual(record["schema"], "jope.q1-decision-record/v1")
        self.assertEqual(record["q1_status"], "NOT_RUN")
        self.assertEqual(record["decision"], "UNKNOWN_NOT_RUN")
        self.assertIsNone(record["primary_effect"])
        self.assertEqual(record["failure_inventory"], "NOT_RUN")
        self.assertFalse(record["controls"]["qualification_lock_acquired"])
        self.assertFalse(record["controls"]["fifo_ticket_created"])
        self.assertFalse(record["controls"]["backend_invoked"])
        self.assertFalse(record["controls"]["dataset_payloads_opened"])
        self.assertFalse(record["controls"]["default_changed"])
        self.assertEqual(q1.load(output), record)
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertEqual(list(self.queue.iterdir()), [sentinel])
        with self.assertRaises(FileExistsError):
            launch.dry_run(self.paths.input_manifest, output, self.paths)

    def test_synthetic_dry_run_consumes_amended_roster_three_fold_cut(self):
        cut_root = self.root / "cuts"
        cut_root.mkdir()
        first_dataset_sha = f"{1:016x}" + "a" * 48
        cut = {
            "schema": q1_cut.SCHEMA,
            "dataset_sha256": first_dataset_sha,
            "train_blob_sha256": "b" * 64,
            "train_rows": 6,
            "target": "target",
            "feature_columns": ["x"],
            "folds": [
                {"repeat": 0, "train": [2, 3, 4, 5], "validation": [0, 1]},
                {"repeat": 0, "train": [0, 1, 4, 5], "validation": [2, 3]},
                {"repeat": 0, "train": [0, 1, 2, 3], "validation": [4, 5]},
            ],
        }
        cut_sha = q1_cut.cut_sha256(cut)
        (cut_root / f"{cut_sha}.json").write_bytes(q1_cut.canonical_bytes(cut))
        datasets = []
        for index in range(40):
            dataset_sha = f"{index + 1:016x}" + "a" * 48
            row = {
                "dataset_id": dataset_sha[:16],
                "sha256": dataset_sha,
                "s3_manifest_sha256": "b" * 64,
            }
            if index == 0:
                row["cut_sha256"] = cut_sha
            datasets.append(row)
        cohort = self.root / "amended-roster.json"
        cohort.write_bytes(
            q1.canonical(
                {
                    "schema": "jope.q1.held-out-cohort/v1",
                    "dataset_count": len(datasets),
                    "datasets": datasets,
                    "source": {"catalog_sha256": "c" * 64},
                }
            )
        )
        paths = replace(
            self.paths,
            cohort=cohort,
            input_manifest=self.root / "inputs-manifest.json",
            cut_definitions=cut_root,
        )
        manifest = launch.build_input_manifest(paths)
        self.assertEqual(manifest["datasets"][0]["cut_sha256"], cut_sha)
        self.assertNotIn("cut_sha256", manifest["datasets"][1])
        paths.input_manifest.write_bytes(q1.canonical(manifest))
        output = self.root / "amended-dry-run.json"
        record = launch.dry_run(paths.input_manifest, output, paths)
        self.assertEqual(record["q1_status"], "NOT_RUN")
        self.assertEqual(record["decision"], "UNKNOWN_NOT_RUN")
        self.assertFalse(record["qualification_eligible"])
        self.assertFalse(record["controls"]["dataset_payloads_opened"])

    def test_input_manifest_rejects_a_non_three_fold_roster_cut(self):
        cut_root = self.root / "cuts"
        cut_root.mkdir()
        first_dataset_sha = f"{1:016x}" + "a" * 48
        definition = {
            "schema": q1_cut.SCHEMA,
            "dataset_sha256": first_dataset_sha,
            "train_blob_sha256": "b" * 64,
            "train_rows": 4,
            "target": "target",
            "feature_columns": ["x"],
            "folds": [
                {"repeat": 0, "train": [0, 1], "validation": [2, 3]},
                {"repeat": 0, "train": [2, 3], "validation": [0, 1]},
            ],
        }
        cut_sha = q1_cut.cut_sha256(definition)
        (cut_root / f"{cut_sha}.json").write_bytes(q1_cut.canonical_bytes(definition))
        datasets = []
        for index in range(40):
            dataset_sha = f"{index + 1:016x}" + "a" * 48
            row = {
                "dataset_id": dataset_sha[:16],
                "sha256": dataset_sha,
                "s3_manifest_sha256": "b" * 64,
            }
            if index == 0:
                row["cut_sha256"] = cut_sha
            datasets.append(row)
        paths = replace(
            self.paths,
            cohort=self.root / "bad-amended-roster.json",
            cut_definitions=cut_root,
        )
        paths.cohort.write_bytes(
            q1.canonical(
                {
                    "schema": "jope.q1.held-out-cohort/v1",
                    "dataset_count": len(datasets),
                    "datasets": datasets,
                    "source": {"catalog_sha256": "c" * 64},
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "exactly three FIT folds"):
            launch.build_input_manifest(paths)

    def test_consumes_real_t0610_amended_roster_cut_packet_and_cuts(self):
        summary = launch.load_amended_roster_packet(self.paths)
        self.assertEqual(summary["schema"], "jope.q1-amended-roster-summary/v1")
        self.assertEqual(summary["amendment_id"], "QLT-04-T0610-R1-R2")
        self.assertEqual(
            summary["roster_cut_packet_sha256"],
            "ab434d145b8cec60399b35c35a1ce9551a881843e8136b1c757e9c93ae62bd9a",
        )
        self.assertEqual(
            summary["fold_manifest_sha256"],
            "fdc243193d9436449cf74aa92b05ff7842c6c88bf4dce0b29ef7edf453b74987",
        )
        self.assertEqual(summary["official_roster_count"], 153)
        self.assertEqual(len(summary["datasets"]), 153)
        self.assertFalse(summary["qualification_eligible"])
        self.assertEqual(summary["q1_status"], "NOT_RUN")
        datasets = [
            {
                "dataset_id": row["dataset_id"],
                "sha256": row["sha256"],
                "s3_manifest_sha256": row["train_blob_sha256"],
                "cut_sha256": row["cut_sha256"],
            }
            for row in summary["datasets"]
        ]
        cohort = self.root / "real-t0610-roster.json"
        cohort.write_bytes(
            q1.canonical(
                {
                    "schema": "jope.q1.held-out-cohort/v1",
                    "dataset_count": len(datasets),
                    "datasets": datasets,
                    "source": {"catalog_sha256": summary["fold_manifest_sha256"]},
                }
            )
        )
        paths = replace(
            self.paths,
            cohort=cohort,
            input_manifest=self.root / "real-t0610-inputs-manifest.json",
        )
        manifest = launch.build_input_manifest(paths)
        self.assertEqual(len(manifest["datasets"]), 153)
        self.assertEqual(
            manifest["datasets"][0]["cut_sha256"],
            "8b686be059e981dab5cf3245d8c7e2f33e7af2cca0a93935b5e7a7083d91645c",
        )
        paths.input_manifest.write_bytes(q1.canonical(manifest))
        output = self.root / "real-t0610-dry-run.json"
        record = launch.dry_run(paths.input_manifest, output, paths)
        self.assertEqual(record["real_s3_metadata_entries"], 153)
        self.assertEqual(record["q1_status"], "NOT_RUN")
        self.assertFalse(record["controls"]["dataset_payloads_opened"])

    def test_synthetic_dry_run_refuses_a_changed_dope_manifest(self):
        source = q1.load(self.paths.input_manifest)
        source["dope_sha256"] = "f" * 64
        manifest = self.root / "bad-input-manifest.json"
        manifest.write_bytes(q1.canonical(source))
        output = self.root / "must-not-exist.json"
        with self.assertRaisesRegex(ValueError, "frozen DOPE differs"):
            launch.dry_run(manifest, output, self.paths)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_unsigned_preregistration_refuses_before_queue_output_or_ledger(self):
        self.prereg.write_text("Owner allowed-signers SHA-256: PENDING\n")
        self.plan["preregistration_sha256"] = q1.sha_file(self.prereg)
        self.planfile.write_bytes(q1.canonical(self.plan))
        with self.assertRaisesRegex(ValueError, "OWNER_SIGNATURE_REQUIRED"):
            self.run_launch()
        self.assertFalse(self.args.run_dir.exists())
        self.assertFalse(self.args.ledger.exists())
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_explicit_intent_and_finite_caps_are_required_before_effects(self):
        self.args.authorized_launch = False
        with self.assertRaisesRegex(ValueError, "authorized-launch"):
            self.run_launch()
        self.args.authorized_launch = True
        for field, value in [
            ("wait_seconds", 0),
            ("wait_seconds", 10801),
            ("wait_seconds", math.nan),
            ("wall_seconds", math.inf),
            ("wall_seconds", -1),
            ("wall_seconds", 144001),
        ]:
            prior = getattr(self.args, field)
            setattr(self.args, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.run_launch()
            setattr(self.args, field, prior)
        with self.assertRaises(ValueError):
            launch.limits(10**1000, 10800, "test")
        self.assertFalse(self.args.run_dir.exists())
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_bad_signature_backend_and_dope_never_enqueue(self):
        self.args.signature.write_text("not a signature")
        with self.assertRaisesRegex(ValueError, "signature"):
            self.admit()
        self.backend.write_text("changed")
        with self.assertRaisesRegex(ValueError, "backend hash"):
            self.admit()
        self.plan["pins"]["dope_sha256"] = "f" * 64
        self.planfile.write_bytes(q1.canonical(self.plan))
        with self.assertRaisesRegex(ValueError, "DOPE"):
            self.admit()
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_source_must_be_clean_exact_and_merged(self):
        with patch.object(subprocess, "check_output", return_value=b"b" * 40):
            with self.assertRaisesRegex(ValueError, "signed source differs"):
                launch.source_identity(self.paths, "a" * 40)
        with patch.object(
            subprocess, "check_output", side_effect=[b"a" * 40, b" M file"]
        ):
            with self.assertRaisesRegex(ValueError, "clean checkout"):
                launch.source_identity(self.paths, "a" * 40)
        with patch.object(
            subprocess,
            "check_output",
            side_effect=[
                b"a" * 40,
                b"",
                subprocess.CalledProcessError(1, ["git", "merge-base"]),
            ],
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                launch.source_identity(self.paths, "a" * 40)
