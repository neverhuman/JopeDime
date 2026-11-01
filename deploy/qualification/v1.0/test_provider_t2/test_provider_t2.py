from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "providers"))
import common
import t2


class T2ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.artifact = self.root / "candidate.bin"
        self.artifact.write_bytes(b"controlled T2 fixture; not a release candidate")
        self.ctx = common.Context(
            "T2",
            self.artifact,
            hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
            "a" * 40,
            self.root / "result.json",
        )
        self.calls: list[tuple[list[str], dict]] = []
        self.dirty = ""
        self.source_sha = self.ctx.source_sha
        self.build_code = 0
        self.positive_output: str | None = None
        self.negative_output = (
            "test bounded_real_run_http_status_and_sse_do_not_leak_internal_sentinels ... FAILED\n"
            'Leak { surface: Http, sentinel: "T2_DSL_SENTINEL_8f27" }\n'
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 30 filtered out\n"
        )
        self.negative_code = 101
        self.after_test = lambda: None
        self.addCleanup(patch.stopall)
        patch.object(t2, "ROOT", self.checkout).start()
        patch.object(common, "ROOT", self.checkout).start()

    def run_child(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[:2] == ["git", "rev-parse"]:
            output, code = self.source_sha, 0
        elif args[:2] == ["git", "status"]:
            output, code = self.dirty, 0
        elif args[:2] == ["cargo", "run"]:
            if "inspect" in args:
                output = json.dumps(
                    {
                        "manifest": {
                            "profile": "full",
                            "scope": "release",
                            "source": {"sha40": self.ctx.source_sha},
                        }
                    }
                )
            else:
                output = json.dumps({"schema": "jope.appliance-verify/v1", "ok": True})
            code = 0
        elif args == t2.build_command():
            output, code = "", self.build_code
        elif args[:8] == [
            "cargo",
            "test",
            "--locked",
            "-p",
            "jope-service",
            "--features",
            "failpoints",
            "--test",
        ]:
            test = args[-3]
            leak = kwargs["env"].get("JOPE_PUB06_TEST_LEAK") == "1"
            self.after_test()
            if leak and test == t2.REAL_RUN_TEST:
                output, code = self.negative_output, self.negative_code
            elif self.positive_output is not None:
                output, code = self.positive_output, 0
            else:
                output = (
                    f"test {test} ... ok\ntest result: ok. 1 passed; 0 failed; "
                    "0 ignored; 0 measured; 30 filtered out\n"
                )
                code = 0
        else:
            self.fail(f"unexpected child command: {args}")
        return subprocess.CompletedProcess(args, code, output, "")

    def invoke(self):
        return t2.qualify(self.ctx, run=self.run_child)

    def test_pass_result_is_candidate_bound_and_counts_only_positive_cases(
        self,
    ):
        self.assertEqual(self.invoke(), 0)
        payload = json.loads(self.ctx.result.read_text())
        self.assertEqual(payload["schema"], "jope.rel07.gate-execution/v1")
        self.assertEqual(payload["id"], "T2")
        self.assertEqual(payload["artifact_sha256"], self.ctx.artifact_sha256)
        self.assertEqual(payload["source_sha"], self.ctx.source_sha)
        self.assertEqual(payload["counts"]["collected"], 4)
        self.assertEqual(payload["counts"]["passed"], 4)

    def test_only_exact_cpu_surface_tests_and_release_build_are_selected(self):
        self.assertEqual(self.invoke(), 0)
        commands = [args for args, _ in self.calls]
        self.assertIn(t2.build_command(), commands)
        for name in t2.POSITIVE_TESTS:
            args = t2.test_command(name)
            self.assertIn(args, commands)
            self.assertIn("--exact", args)
            self.assertNotIn("--ignored", args)
            self.assertNotIn("cuda", args)
        self.assertEqual(commands.count(t2.test_command(t2.REAL_RUN_TEST)), 2)

    def test_negative_control_must_fail_at_expected_http_sentinel(self):
        self.negative_output = self.negative_output.replace(
            "surface: Http", "surface: Sse"
        )
        with self.assertRaisesRegex(common.ProviderError, "negative control"):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_negative_control_must_not_pass_or_fail_for_an_unrelated_reason(self):
        self.negative_code = 0
        with self.assertRaisesRegex(common.ProviderError, "negative control"):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())
        self.negative_code = 101
        self.negative_output = "error: failed to build package\n"
        with self.assertRaises(common.ProviderError):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_missing_or_skipped_positive_case_cannot_pass(self):
        self.positive_output = "test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 31 filtered out\n"
        with self.assertRaises(common.ProviderError):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_failed_build_or_positive_suite_never_writes_result(self):
        self.build_code = 17
        self.assertEqual(self.invoke(), 17)
        self.assertFalse(self.ctx.result.exists())
        self.build_code = 0
        self.positive_output = "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 30 filtered out\n"
        with self.assertRaises(common.ProviderError):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_wrong_or_dirty_source_refuses_before_product_commands(self):
        self.source_sha = "b" * 40
        with self.assertRaises(common.ProviderError):
            self.invoke()
        self.assertFalse(any(args[0] == "cargo" for args, _ in self.calls))
        self.source_sha = self.ctx.source_sha
        self.dirty = "?? untracked.rs"
        with self.assertRaises(common.ProviderError):
            self.invoke()
        self.assertFalse(any(args[0] == "cargo" for args, _ in self.calls))

    def test_candidate_drift_during_run_refuses_result(self):
        self.after_test = lambda: self.artifact.write_bytes(b"changed candidate")
        with self.assertRaisesRegex(common.ProviderError, "candidate bytes changed"):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_result_must_be_new_and_outside_checkout(self):
        self.ctx = common.Context(
            "T2",
            self.artifact,
            self.ctx.artifact_sha256,
            self.ctx.source_sha,
            self.checkout / "result.json",
        )
        with self.assertRaisesRegex(
            common.ProviderError, "outside the source checkout"
        ):
            self.invoke()
        self.assertFalse(any(args[0] == "cargo" for args, _ in self.calls))

    def test_context_rejects_symlink(self):
        link = self.root / "link.bin"
        link.symlink_to(self.artifact)
        with patch.dict(
            "os.environ",
            {
                "JOPE_REL07_GATE_ID": "T2",
                "JOPE_REL07_ARTIFACT": str(link),
                "JOPE_REL07_ARTIFACT_SHA256": self.ctx.artifact_sha256,
                "JOPE_REL07_SOURCE_SHA": self.ctx.source_sha,
                "JOPE_REL07_RESULT": str(self.root / "link-result.json"),
            },
            clear=True,
        ):
            with self.assertRaisesRegex(common.ProviderError, "non-symlink"):
                common.context("T2")


if __name__ == "__main__":
    unittest.main()
