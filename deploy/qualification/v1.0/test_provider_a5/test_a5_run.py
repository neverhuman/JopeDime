"""Tests for the A5 command adapter and its non-qualifying dev receipt."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a5_run


class A5RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "checkout"
        self.root.mkdir()
        self.candidate = Path(self.temp.name) / "candidate.bin"
        self.candidate.write_bytes(b"synthetic main DEV-BUILD candidate")
        self.evidence = Path(self.temp.name) / "evidence"
        self.evidence.mkdir()
        self.manifest = self.evidence / "manifest.json"
        self.manifest.write_text('{"synthetic":true}\n', encoding="utf-8")
        self.manifest_sha = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.result = self.evidence / "result.json"
        self.source_sha = "a" * 40
        self.env = patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin:/bin",
                "HOME": self.temp.name,
                "CARGO_BUILD_JOBS": "4",
                "CARGO_TARGET_DIR": str(Path(self.temp.name) / "target"),
                "AWS_SECRET_ACCESS_KEY": "must-not-be-forwarded",
                "JOPE_REL07_GATE_ID": "A5",
            },
            clear=True,
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.root_patch = patch.object(a5_run, "ROOT", self.root)
        self.provider_patch = patch.object(a5_run, "PROVIDER", self.root / "a5.py")
        self.root_patch.start()
        self.provider_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.provider_patch.stop)

    def args(self, *extra):
        return [
            "--candidate",
            str(self.candidate),
            "--result",
            str(self.result),
            "--source-sha",
            self.source_sha,
            "--manifest",
            str(self.manifest),
            "--manifest-sha256",
            self.manifest_sha,
            *extra,
        ]

    def test_provider_gets_candidate_and_trusted_manifest_bindings(self):
        observed = {}

        def provider(command, *, cwd, env, check):
            observed.update(command=command, cwd=cwd, env=env, check=check)
            receipt = {
                "schema": "jope.rel07.gate-execution/v1",
                "id": "A5",
                "status": "PASS",
                "source_sha": self.source_sha,
                "artifact_sha256": hashlib.sha256(
                    self.candidate.read_bytes()
                ).hexdigest(),
                "evidence_manifest_sha256": self.manifest_sha,
            }
            Path(env["JOPE_REL07_RESULT"]).write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0)

        runtime_inputs = {
            "JOPE_REL07_ARTIFACT": str(self.candidate),
            "JOPE_REL07_RESULT": str(self.result),
            "JOPE_REL07_SOURCE_SHA": self.source_sha,
            "JOPE_REL07_A5_MANIFEST": str(self.manifest),
            "JOPE_REL07_A5_MANIFEST_SHA256": self.manifest_sha,
        }
        with (
            patch.dict(os.environ, runtime_inputs),
            patch.object(a5_run.subprocess, "run", side_effect=provider),
        ):
            # This is the exact zero-argument invocation used by release-gates.
            self.assertEqual(a5_run.main([]), 0)
        env = observed["env"]
        self.assertEqual(observed["cwd"], self.root)
        self.assertEqual(env["JOPE_REL07_ARTIFACT"], str(self.candidate))
        self.assertEqual(env["JOPE_REL07_RESULT"], str(self.result))
        self.assertEqual(env["JOPE_REL07_SOURCE_SHA"], self.source_sha)
        self.assertEqual(env["JOPE_REL07_A5_MANIFEST"], str(self.manifest))
        self.assertEqual(env["JOPE_REL07_A5_MANIFEST_SHA256"], self.manifest_sha)
        self.assertEqual(env["CARGO_BUILD_JOBS"], "4")
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)

    def test_dev_build_writes_a_distinct_non_gate_receipt(self):
        with patch.object(a5_run.subprocess, "run") as run:
            self.assertEqual(
                a5_run.main(
                    [
                        "--candidate",
                        str(self.candidate),
                        "--result",
                        str(self.result),
                        "--source-sha",
                        self.source_sha,
                        "--dev-build",
                    ]
                ),
                0,
            )
            run.assert_not_called()
        receipt = json.loads(self.result.read_text(encoding="utf-8"))
        self.assertEqual(receipt["schema"], "jopedime.rel07.a5-dev-build/v1")
        self.assertEqual(receipt["status"], "DEV-BUILD")
        self.assertEqual(receipt["qualification_status"], "NOT_RUN")
        self.assertIs(receipt["not_a_gate_receipt"], True)
        self.assertNotEqual(receipt["schema"], "jope.rel07.gate-execution/v1")

    def test_manifest_hash_mismatch_refuses_before_provider(self):
        with patch.object(a5_run.subprocess, "run") as run:
            self.assertEqual(a5_run.main(self.args("--manifest-sha256", "0" * 64)), 2)
            run.assert_not_called()
        self.assertFalse(self.result.exists())

    def test_provider_cannot_report_success_without_result(self):
        with patch.object(
            a5_run.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(["provider"], 0),
        ):
            self.assertEqual(a5_run.main(self.args()), 2)

    def test_candidate_symlink_is_refused(self):
        link = Path(self.temp.name) / "candidate-link"
        link.symlink_to(self.candidate)
        with patch.object(a5_run.subprocess, "run") as run:
            self.assertEqual(
                a5_run.main(
                    [
                        "--candidate",
                        str(link),
                        "--result",
                        str(self.result),
                        "--source-sha",
                        self.source_sha,
                        "--manifest",
                        str(self.manifest),
                        "--manifest-sha256",
                        self.manifest_sha,
                    ]
                ),
                2,
            )
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
