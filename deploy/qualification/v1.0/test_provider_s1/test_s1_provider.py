"""Data-free controls for the S1 provider; mocks are not release results."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
PROVIDERS = HERE.parent / "providers"
sys.path.insert(0, str(PROVIDERS))
s1 = importlib.import_module("s1")
common = importlib.import_module("common")
Context, ProviderError = common.Context, common.ProviderError


class S1ProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        self.artifact = self.root / "candidate"
        self.artifact.write_bytes(b"synthetic provider-control bytes; not a release")
        self.ctx = Context(
            "S1",
            self.artifact,
            hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
            "a" * 40,
            self.root / "result.json",
        )
        self.source = self.ctx.source_sha
        self.dirty = ""
        self.test_output = (
            f"running 1 test\ntest {s1.TEST_NAME} ... ok\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; "
            "482 filtered out\n"
        )
        self.returncode = 0
        self.result_mutation = None
        self.seen_test_env = None
        self.addCleanup(patch.stopall)
        patch.object(s1, "ROOT", self.repo).start()
        patch.object(common, "ROOT", self.repo).start()

    def run_child(self, args, **kwargs):
        if args[:2] == ["git", "rev-parse"]:
            output = self.source
        elif args[:2] == ["git", "status"]:
            output = self.dirty
        elif args[:3] == ["cargo", "run", "--locked"] and "inspect" in args:
            output = json.dumps(
                {
                    "manifest": {
                        "profile": "full",
                        "scope": "release",
                        "source": {"sha40": self.ctx.source_sha},
                    }
                }
            )
        elif args[:3] == ["cargo", "run", "--locked"] and "verify" in args:
            output = json.dumps({"schema": "jope.appliance-verify/v1", "ok": True})
        elif args[:2] == ["cargo", "test"]:
            self.seen_test_env = kwargs.get("env")
            if self.result_mutation is not None:
                self.result_mutation()
            return subprocess.CompletedProcess(
                args, self.returncode, self.test_output, ""
            )
        else:
            raise AssertionError(f"unexpected child command: {args}")
        return subprocess.CompletedProcess(args, 0, output, "")

    def invoke(self):
        return s1.qualify(self.ctx, run=self.run_child)

    def context_environment(self, artifact, result):
        return {
            "JOPE_REL07_GATE_ID": "S1",
            "JOPE_REL07_ARTIFACT": str(artifact),
            "JOPE_REL07_ARTIFACT_SHA256": self.ctx.artifact_sha256,
            "JOPE_REL07_SOURCE_SHA": self.ctx.source_sha,
            "JOPE_REL07_RESULT": str(result),
        }

    def test_exact_surface_test_writes_bound_single_case_pass(self):
        self.assertEqual(self.invoke(), 0)
        result = json.loads(self.ctx.result.read_text())
        self.assertEqual(result["id"], "S1")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["source_sha"], self.ctx.source_sha)
        self.assertEqual(result["artifact_sha256"], self.ctx.artifact_sha256)
        self.assertEqual(result["counts"], s1.SINGLE_PASS)
        self.assertEqual(s1.command()[6:9], ["e2e", "--features", "engine,mcp"])
        self.assertEqual(s1.command()[-2:], ["--exact", "--test-threads=1"])

    def test_environment_forces_committed_fixture_and_lane_target(self):
        supplied = {
            "JOPE_INT01_QUALIFY_HOME": "/sealed/alternate",
            "JOPE_INT01_CSV": "/tmp/alternate.csv",
            "JOPE_INT01_RECEIPTS": "/tmp/receipts",
            "JOPE_GOLDEN_DIR": "/sealed/goldens",
            "CARGO_TARGET_DIR": "/tmp/target",
            "CARGO_BUILD_JOBS": "99",
            "CARGO_INCREMENTAL": "1",
            "CARGO_PROFILE_DEV_DEBUG": "2",
        }
        with patch.dict(os.environ, supplied, clear=False):
            self.assertEqual(self.invoke(), 0)
        for name in s1.UNTRUSTED_TEST_INPUTS:
            self.assertNotIn(name, self.seen_test_env)
        self.assertEqual(self.seen_test_env["CARGO_BUILD_JOBS"], "4")
        self.assertEqual(self.seen_test_env["CARGO_INCREMENTAL"], "0")
        self.assertEqual(
            self.seen_test_env["CARGO_PROFILE_DEV_DEBUG"], "line-tables-only"
        )
        self.assertNotIn("CARGO_TARGET_DIR", self.seen_test_env)

    def test_context_rejects_symlink_nonfile_and_hash_mismatch(self):
        link = self.root / "candidate-link"
        link.symlink_to(self.artifact)
        directory = self.root / "candidate-dir"
        directory.mkdir()
        for path in (link, directory):
            env = self.context_environment(path, self.root / "result.json")
            with patch.dict(os.environ, env, clear=False):
                with self.subTest(path=path), self.assertRaises(ProviderError):
                    common.context("S1")

        env = self.context_environment(self.artifact, self.root / "result.json")
        env["JOPE_REL07_ARTIFACT_SHA256"] = "0" * 64
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaisesRegex(ProviderError, "declared artifact SHA"):
                common.context("S1")

    def test_context_rejects_existing_and_symlink_result_paths(self):
        existing = self.root / "existing.json"
        existing.write_text("prior")
        link = self.root / "result-link.json"
        link.symlink_to(existing)
        for path in (existing, link):
            env = self.context_environment(self.artifact, path)
            with patch.dict(os.environ, env, clear=False):
                with self.subTest(path=path), self.assertRaises(ProviderError):
                    common.context("S1")

    def test_zero_or_wrong_case_never_writes_pass(self):
        self.test_output = (
            "running 1 test\ntest unrelated::case ... ok\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
        )
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(json.loads(self.ctx.result.read_text())["status"], "FAIL")

    def test_nonzero_child_exit_preserves_command_failure(self):
        self.returncode = 17
        self.assertEqual(self.invoke(), 17)
        self.assertFalse(self.ctx.result.exists())

    def test_source_drift_and_dirty_checkout_refuse(self):
        for source, dirty in (("b" * 40, ""), (self.ctx.source_sha, " M source.rs")):
            self.source, self.dirty = source, dirty
            with (
                self.subTest(source=source, dirty=dirty),
                self.assertRaises(ProviderError),
            ):
                self.invoke()
            self.assertFalse(self.ctx.result.exists())

    def test_artifact_drift_during_run_refuses_result(self):
        self.result_mutation = lambda: self.artifact.write_bytes(b"changed artifact")
        with self.assertRaisesRegex(ProviderError, "candidate bytes changed"):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_existing_or_in_checkout_result_is_refused(self):
        self.ctx.result.write_text("keep")
        with self.assertRaises(ProviderError):
            self.invoke()
        self.assertEqual(self.ctx.result.read_text(), "keep")

        self.ctx.result.unlink()
        self.ctx = Context(
            "S1",
            self.artifact,
            hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
            "a" * 40,
            self.repo / "inside.json",
        )
        with self.assertRaisesRegex(ProviderError, "outside the source checkout"):
            self.invoke()

    def test_wrong_gate_id_refuses(self):
        self.ctx = Context(
            "S2",
            self.artifact,
            hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
            "a" * 40,
            self.root / "result.json",
        )
        with self.assertRaisesRegex(ProviderError, "requires gate S1"):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
