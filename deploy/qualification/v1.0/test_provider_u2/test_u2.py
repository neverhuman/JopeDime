"""Data-free controls for the candidate-bound U2 provider."""

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

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "providers"))
u2 = importlib.import_module("u2")
common = importlib.import_module("common")
Context, ProviderError = common.Context, common.ProviderError


class U2Tests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(
            os.environ.get("JOPE_QUALIFICATION_TEST_TMP", HERE.parents[2] / "target")
        )
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="u2-", dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        self.artifact = self.root / "candidate"
        self.artifact.write_bytes(b"synthetic U2 provider fixture")
        self.source = "a" * 40
        self.ctx = Context(
            "U2",
            self.artifact,
            hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
            self.source,
            self.root / "result.json",
        )
        self.calls = []
        self.tests = []
        self.dirty = ""
        self.output_override = None
        self.returncode = 0
        self.wrong_inspect_source = False
        self.mutate_source_after_test = False
        self.mutate_artifact_after_test = False
        self.addCleanup(patch.stopall)
        patch.object(u2, "ROOT", self.repo).start()
        patch.object(common, "ROOT", self.repo).start()

    def run_child(self, args, **kwargs):
        self.calls.append(args)
        if args[:2] == ["git", "rev-parse"]:
            output = (
                "b" * 40
                if self.mutate_source_after_test and self.tests
                else self.source
            )
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args, 0, self.dirty, "")
        if args[:2] == ["cargo", "run"]:
            if "inspect" in args:
                source = "b" * 40 if self.wrong_inspect_source else self.ctx.source_sha
                output = json.dumps(
                    {
                        "manifest": {
                            "profile": "full",
                            "scope": "release",
                            "source": {"sha40": source},
                        }
                    }
                )
            else:
                output = json.dumps({"schema": "jope.appliance-verify/v1", "ok": True})
            return subprocess.CompletedProcess(args, 0, output, "")
        test = next(value for value in u2.TESTS if value in args)
        self.tests.append(test)
        if self.mutate_artifact_after_test:
            self.artifact.write_bytes(b"changed U2 candidate")
        output = self.output_override
        if output is None:
            output = (
                f"test {test} ... ok\n"
                "test result: ok. 1 passed; 0 failed; 0 ignored; "
                "0 measured; 500 filtered out\n"
            )
        return subprocess.CompletedProcess(args, self.returncode, output, "")

    def invoke(self, system="Linux"):
        return u2.qualify(self.ctx, run=self.run_child, system=system)

    def test_all_required_cases_produce_candidate_bound_counts(self):
        self.assertEqual(self.invoke(), 0)
        result = json.loads(self.ctx.result.read_text())
        self.assertEqual(result["id"], "U2")
        self.assertEqual(result["source_sha"], self.ctx.source_sha)
        self.assertEqual(result["artifact_sha256"], self.ctx.artifact_sha256)
        self.assertEqual(
            result["counts"],
            {
                "collected": len(u2.TESTS),
                "executed": len(u2.TESTS),
                "passed": len(u2.TESTS),
                "failed": 0,
                "skipped": 0,
                "excluded": 0,
            },
        )
        self.assertEqual(self.tests, list(u2.TESTS))
        self.assertEqual(
            u2.TESTS[4:],
            (
                "terminal::tests::detects_each_color_level_and_disables_color_when_requested",
                "terminal::tests::strips_escape_sequences_and_controls_without_losing_plain_text",
                "terminal::tests::uses_unicode_symbols_only_for_utf8_locales",
            ),
        )
        self.assertTrue(all("--bin" in u2.command(test) for test in u2.TESTS))
        self.assertTrue(all("jopedime" in u2.command(test) for test in u2.TESTS))
        self.assertTrue(all("--exact" in u2.command(test) for test in u2.TESTS))

    def test_zero_wrong_skipped_measured_or_multiple_cases_refuse(self):
        outputs = (
            "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 500 filtered out\n",
            "test other::case ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 500 filtered out\n",
            f"test {u2.TESTS[0]} ... ignored\ntest result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 500 filtered out\n",
            f"test {u2.TESTS[0]} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 1 measured; 500 filtered out\n",
            f"test {u2.TESTS[0]} ... ok\ntest {u2.TESTS[0]} ... ok\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 500 filtered out\n",
        )
        for output in outputs:
            with self.subTest(output=output[:70]):
                self.output_override = output
                with self.assertRaises(ProviderError):
                    self.invoke()
                self.assertFalse(self.ctx.result.exists())
                self.tests.clear()

    def test_failed_child_stops_and_preserves_nonzero_exit(self):
        self.returncode = 23
        self.assertEqual(self.invoke(), 23)
        self.assertEqual(len(self.tests), 1)
        self.assertFalse(self.ctx.result.exists())

    def test_source_or_artifact_drift_prevents_a_result(self):
        for attr in ("mutate_source_after_test", "mutate_artifact_after_test"):
            setattr(self, attr, True)
            with self.subTest(attr=attr), self.assertRaises(ProviderError):
                self.invoke()
            self.assertFalse(self.ctx.result.exists())
            self.tests.clear()
            setattr(self, attr, False)

    def test_dirty_wrong_source_and_unavailable_platform_refuse_before_tests(self):
        self.dirty = " M source.py"
        with self.assertRaises(ProviderError):
            self.invoke()
        self.assertEqual(self.tests, [])
        self.dirty = ""
        self.source = "b" * 40
        with self.assertRaises(ProviderError):
            self.invoke()
        self.assertEqual(self.tests, [])
        self.source = self.ctx.source_sha
        with self.assertRaisesRegex(ProviderError, "only on Linux"):
            self.invoke(system="Darwin")
        self.assertEqual(self.tests, [])

    def test_wrong_candidate_pin_and_existing_or_in_checkout_result_refuse(self):
        self.wrong_inspect_source = True
        with self.assertRaises(ProviderError):
            self.invoke()
        self.assertEqual(self.tests, [])
        self.wrong_inspect_source = False
        self.ctx = Context(
            "U2",
            self.artifact,
            self.ctx.artifact_sha256,
            self.source,
            self.repo / "result.json",
        )
        with self.assertRaises(ProviderError):
            self.invoke()
        self.ctx = Context(
            "U2",
            self.artifact,
            self.ctx.artifact_sha256,
            self.source,
            self.root / "existing.json",
        )
        self.ctx.result.write_text("{}")
        with self.assertRaises(ProviderError):
            self.invoke()

    def test_candidate_context_rejects_wrong_gate_symlinks_nonfiles_and_hashes(self):
        link = self.root / "candidate-link"
        link.symlink_to(self.artifact)
        values = {
            "JOPE_REL07_GATE_ID": "U2",
            "JOPE_REL07_ARTIFACT": str(link),
            "JOPE_REL07_ARTIFACT_SHA256": self.ctx.artifact_sha256,
            "JOPE_REL07_SOURCE_SHA": self.source,
            "JOPE_REL07_RESULT": str(self.root / "result.json"),
        }
        with patch.dict(os.environ, values, clear=True):
            with self.assertRaises(ProviderError):
                u2.context("U2")
        values["JOPE_REL07_ARTIFACT"] = str(self.root)
        with patch.dict(os.environ, values, clear=True):
            with self.assertRaises(ProviderError):
                u2.context("U2")
        values["JOPE_REL07_ARTIFACT"] = str(self.artifact)
        values["JOPE_REL07_ARTIFACT_SHA256"] = "0" * 64
        with patch.dict(os.environ, values, clear=True):
            with self.assertRaises(ProviderError):
                u2.context("U2")
        values["JOPE_REL07_ARTIFACT_SHA256"] = self.ctx.artifact_sha256
        values["JOPE_REL07_GATE_ID"] = "U1"
        with patch.dict(os.environ, values, clear=True):
            with self.assertRaises(ProviderError):
                u2.context("U2")
