"""Controller fixtures only; these tests do not qualify a release candidate."""

import contextlib
from dataclasses import replace
import importlib
import io
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
a4 = importlib.import_module("a4")
common = importlib.import_module("common")


class A4Tests(unittest.TestCase):
    def setUp(self):
        folder = Path(os.environ.get("JOPE_QUALIFICATION_TEST_TMP", HERE / "target"))
        folder.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix="a4-control-", dir=folder)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        self.artifact = self.root / "candidate"
        self.artifact.write_bytes(b"data-free controller fixture, not a release")
        self.ctx = common.Context(
            "A4",
            self.artifact,
            common.sha256(self.artifact),
            "a" * 40,
            self.root / "result.json",
        )
        self.source, self.dirty = self.ctx.source_sha, ""
        self.calls, self.tests = [], []
        self.output, self.code, self.profile = None, 0, "full"
        self.after_test = lambda: None
        self.addCleanup(patch.stopall)
        patch.object(a4, "ROOT", self.repo).start()

    def run_child(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[:2] == ["git", "rev-parse"]:
            output = self.source
        elif args[:2] == ["git", "status"]:
            output = self.dirty
        elif args[:2] == ["cargo", "run"]:
            output = json.dumps(
                {
                    "manifest": {
                        "profile": self.profile,
                        "scope": "public",
                        "source": {"sha40": self.ctx.source_sha},
                    }
                }
                if "inspect" in args
                else {"schema": "jope.appliance-verify/v1", "ok": True}
            )
        else:
            test = args[args.index("jopedime") + 1]
            self.tests.append(test)
            self.after_test()
            output = self.output if self.output is not None else self.success(test)
            return subprocess.CompletedProcess(args, self.code, output, "")
        return subprocess.CompletedProcess(args, 0, output, "")

    @staticmethod
    def success(test):
        return f"running 1 test\ntest {test} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 156 filtered out\n"

    def invoke(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return a4.qualify(self.ctx, run=self.run_child)

    def test_each_product_selector_passes_once_and_emits_bound_counts(self):
        self.assertEqual(self.invoke(), 0)
        result = json.loads(self.ctx.result.read_text())
        self.assertEqual(len(a4.TESTS), 23)
        self.assertEqual(self.tests, list(a4.TESTS))
        self.assertEqual(len(set(self.tests)), 23)
        self.assertEqual(
            result["counts"], {k: 23 * v for k, v in a4.SINGLE_PASS.items()}
        )
        self.assertEqual(result["artifact_sha256"], self.ctx.artifact_sha256)
        self.assertEqual(result["source_sha"], self.ctx.source_sha)
        self.assertEqual(len(result["protocols"]), 4)
        self.assertIn("not measured", result["timing"])
        for args, kwargs in self.calls:
            self.assertEqual(kwargs["env"]["CARGO_INCREMENTAL"], "0")
            self.assertEqual(
                kwargs["env"]["CARGO_PROFILE_DEV_DEBUG"], "line-tables-only"
            )
            self.assertEqual(kwargs["env"]["CARGO_BUILD_JOBS"], "4")
            if args[:2] == ["cargo", "test"]:
                self.assertIn("--exact", args)
                self.assertIn("--color=never", args)

    def test_zero_wrong_ignored_measured_multiple_and_duplicate_cases_refuse(self):
        for output in [
            "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 157 filtered out\n",
            self.success("other::test"),
            self.success(a4.TESTS[0]).replace("1 passed", "0 passed; 0 failed"),
            self.success(a4.TESTS[0]).replace("0 ignored", "1 ignored"),
            self.success(a4.TESTS[0]).replace("0 measured", "1 measured"),
            self.success(a4.TESTS[0]).replace("1 passed", "2 passed"),
            self.success(a4.TESTS[0]) * 2,
        ]:
            self.output = output
            with self.subTest(output=output), self.assertRaises(common.ProviderError):
                self.invoke()
            self.assertFalse(self.ctx.result.exists())

    def test_command_failure_stops_and_preserves_exit_code(self):
        self.code = 37
        self.assertEqual(self.invoke(), 37)
        self.assertEqual(len(self.tests), 1)
        self.assertFalse(self.ctx.result.exists())

    def test_source_or_candidate_drift_prevents_receipt(self):
        for kind in ["source", "artifact", "symlink"]:
            if kind == "source":
                self.after_test = lambda: setattr(self, "source", "b" * 40)
            elif kind == "artifact":
                self.after_test = lambda: self.artifact.write_bytes(b"changed")
            else:
                original = self.root / "original"
                original.write_bytes(self.artifact.read_bytes())

                def swap():
                    self.artifact.unlink()
                    self.artifact.symlink_to(original)

                self.after_test = swap
            with self.subTest(kind=kind), self.assertRaises(common.ProviderError):
                self.invoke()
            self.assertFalse(self.ctx.result.exists())
            if self.artifact.is_symlink():
                self.artifact.unlink()
            self.artifact.write_bytes(b"data-free controller fixture, not a release")
            self.source = self.ctx.source_sha

    def test_dirty_wrong_source_or_wrong_profile_refuses_before_tests(self):
        for source, dirty, profile in [
            ("b" * 40, "", "full"),
            (self.ctx.source_sha, " M source", "full"),
            (self.ctx.source_sha, "", "lite"),
        ]:
            self.source, self.dirty, self.profile = source, dirty, profile
            with (
                self.subTest(source=source, dirty=dirty, profile=profile),
                self.assertRaises(common.ProviderError),
            ):
                self.invoke()
        self.assertFalse(self.tests)

    def test_directories_and_symlinks_are_not_candidates(self):
        directory = self.root / "directory"
        directory.mkdir()
        link = self.root / "link"
        link.symlink_to(self.artifact)
        for path in [directory, link]:
            self.ctx = replace(self.ctx, artifact=path)
            with self.subTest(path=path), self.assertRaises(common.ProviderError):
                self.invoke()
        self.assertFalse(self.tests)

    def test_existing_relative_symlink_or_in_checkout_result_refuses(self):
        existing = self.root / "existing.json"
        existing.write_bytes(b"retain")
        link = self.root / "dangling.json"
        link.symlink_to(self.root / "absent")
        for path in [
            existing,
            link,
            Path("relative.json"),
            self.repo / "result.json",
            self.root / ".." / self.root.name / "checkout" / "result.json",
        ]:
            self.ctx = replace(self.ctx, result=path)
            with self.subTest(path=path), self.assertRaises(common.ProviderError):
                self.invoke()
        self.assertEqual(existing.read_bytes(), b"retain")
        self.assertFalse(self.tests)

    def test_racing_result_is_never_overwritten(self):
        self.after_test = lambda: self.ctx.result.write_bytes(b"concurrent receipt")
        with self.assertRaises(common.ProviderError):
            self.invoke()
        self.assertEqual(self.ctx.result.read_bytes(), b"concurrent receipt")


if __name__ == "__main__":
    unittest.main()
