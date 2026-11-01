"""Data-free A2 controller controls; child fixtures are not release evidence."""

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

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "providers"))
a2 = importlib.import_module("a2")
common = importlib.import_module("common")
Context, ProviderError = common.Context, common.ProviderError


class A2Tests(unittest.TestCase):
    def setUp(self):
        root = Path(
            os.environ.get(
                "JOPE_QUALIFICATION_TEST_TMP",
                common.ROOT / "target" / "provider-controls",
            )
        )
        root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="a2-", dir=root)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        self.artifact = self.root / "candidate"
        self.artifact.write_bytes(b"A2 controller fixture, not a release")
        self.ctx = Context(
            "A2",
            self.artifact,
            common.sha256(self.artifact),
            "a" * 40,
            self.root / "result.json",
        )
        self.source = self.ctx.source_sha
        self.dirty = ""
        self.git_rc = 0
        self.profile = "full"
        self.scope = "release"
        self.manifest_source = self.ctx.source_sha
        self.verify_rc = 0
        self.verified = {"schema": "jope.appliance-verify/v1", "ok": True}
        self.test_rc = 0
        self.override = None
        self.after_test = lambda: None
        self.calls = []
        self.tests = []
        self.addCleanup(patch.stopall)
        patch.object(a2, "ROOT", self.repo).start()
        patch.object(common, "ROOT", self.repo).start()

    @staticmethod
    def passing(name):
        return f"running 1 test\ntest {name} ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 500 filtered out\n"

    def child(self, args, **kwargs):
        self.calls.append(args)
        self.assertEqual(kwargs["cwd"], self.repo)
        self.assertEqual(kwargs["env"]["CARGO_INCREMENTAL"], "0")
        self.assertEqual(kwargs["env"]["CARGO_PROFILE_DEV_DEBUG"], "line-tables-only")
        if args[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(args, self.git_rc, self.source, "")
        if args[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args, self.git_rc, self.dirty, "")
        if args[:2] == ["cargo", "run"]:
            if "inspect" in args:
                value = {
                    "manifest": {
                        "profile": self.profile,
                        "scope": self.scope,
                        "source": {"sha40": self.manifest_source},
                    }
                }
                rc = 0
            else:
                value, rc = self.verified, self.verify_rc
            return subprocess.CompletedProcess(args, rc, json.dumps(value), "")
        self.assertEqual(args[:2], ["cargo", "test"])
        name = args[args.index("--") - 1]
        self.tests.append(name)
        self.after_test()
        return subprocess.CompletedProcess(
            args,
            self.test_rc,
            self.passing(name) if self.override is None else self.override,
            "",
        )

    def invoke(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return a2.qualify(self.ctx, run=self.child)

    def test_all_required_clauses_run_once_with_bound_counts(self):
        self.assertEqual(self.invoke(), 0)
        result = json.loads(self.ctx.result.read_text())
        self.assertEqual(result["id"], "A2")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["source_sha"], self.ctx.source_sha)
        self.assertEqual(result["artifact_sha256"], self.ctx.artifact_sha256)
        self.assertEqual(
            result["counts"], {k: len(a2.CHECKS) * v for k, v in a2.SINGLE_PASS.items()}
        )
        self.assertEqual(self.tests, [check[2] for check in a2.CHECKS])
        for check in a2.CHECKS:
            self.assertIn(a2.command(check), self.calls)
            self.assertIn("--exact", a2.command(check))

    def test_empty_wrong_duplicate_and_nonpassing_cases_refuse(self):
        name = a2.CHECKS[0][2]
        for output in [
            "",
            "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 500 filtered out\n",
            self.passing("other::test"),
            self.passing(name) * 2,
            self.passing(name).replace(" ... ok", " ... ignored"),
            self.passing(name).replace(
                "1 passed; 0 failed; 0 ignored", "0 passed; 0 failed; 1 ignored"
            ),
            self.passing(name).replace(
                "1 passed; 0 failed; 0 ignored; 0 measured",
                "0 passed; 0 failed; 0 ignored; 1 measured",
            ),
            self.passing(name).replace("1 passed", "2 passed"),
            self.passing(name).replace("result: ok", "result: FAILED"),
        ]:
            with self.subTest(output=output), self.assertRaises(ProviderError):
                self.override = output
                self.invoke()
            self.assertFalse(self.ctx.result.exists())

    def test_child_failure_preserves_exit_and_stops(self):
        self.test_rc = 19
        self.assertEqual(self.invoke(), 19)
        self.assertEqual(len(self.tests), 1)
        self.assertFalse(self.ctx.result.exists())

    def test_dirty_wrong_and_unreadable_source_refuse_before_children(self):
        for source, dirty, rc in [
            ("b" * 40, "", 0),
            (self.ctx.source_sha, " M file", 0),
            (self.ctx.source_sha, "", 1),
        ]:
            self.source, self.dirty, self.git_rc = source, dirty, rc
            with (
                self.subTest(source=source, dirty=dirty, rc=rc),
                self.assertRaises(ProviderError),
            ):
                self.invoke()
        self.assertEqual(self.tests, [])

    def test_source_and_artifact_drift_during_case_refuse(self):
        for change in [
            lambda: setattr(self, "source", "b" * 40),
            lambda: self.artifact.write_bytes(b"changed"),
        ]:
            self.source = self.ctx.source_sha
            self.artifact.write_bytes(b"A2 controller fixture, not a release")
            self.tests.clear()
            self.after_test = change
            with self.subTest(change=change), self.assertRaises(ProviderError):
                self.invoke()
            self.assertEqual(len(self.tests), 1)
            self.assertFalse(self.ctx.result.exists())

    def test_candidate_envelope_and_full_byte_verification_are_required(self):
        for attribute, value in [
            ("profile", "lite"),
            ("scope", "development"),
            ("manifest_source", "b" * 40),
            ("verify_rc", 1),
            ("verified", {"schema": "wrong", "ok": True}),
            ("verified", {"schema": "jope.appliance-verify/v1", "ok": False}),
        ]:
            old = getattr(self, attribute)
            setattr(self, attribute, value)
            with (
                self.subTest(attribute=attribute, value=value),
                self.assertRaises(ProviderError),
            ):
                self.invoke()
            setattr(self, attribute, old)
        self.assertEqual(self.tests, [])

    def test_same_byte_candidate_symlink_replacement_refuses_after_case(self):
        saved = self.root / "same-bytes"
        saved.write_bytes(self.artifact.read_bytes())

        def substitute():
            self.artifact.unlink()
            self.artifact.symlink_to(saved)

        self.after_test = substitute
        with self.assertRaisesRegex(ProviderError, "no symlinks"):
            self.invoke()
        self.assertEqual(len(self.tests), 1)
        self.assertFalse(self.ctx.result.exists())

    def test_initial_artifact_drift_is_refused(self):
        self.artifact.write_bytes(b"changed before gate")
        with self.assertRaises(ProviderError):
            self.invoke()
        self.assertEqual(self.tests, [])

    def test_wrong_gate_and_noncanonical_hashes_refuse(self):
        for fields in [
            dict(gate_id="A1"),
            dict(source_sha="A" * 40),
            dict(artifact_sha256="0" * 63),
        ]:
            with self.subTest(fields=fields), self.assertRaises(ProviderError):
                a2.check_paths(replace(self.ctx, **fields))

    def test_candidate_nonfiles_relative_and_symlink_paths_refuse(self):
        directory = self.root / "directory"
        directory.mkdir()
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        link = self.root / "linked-candidate"
        link.symlink_to(self.artifact)
        parent_link = self.root / "linked-parent"
        parent_link.symlink_to(self.root, target_is_directory=True)
        for path in [
            directory,
            fifo,
            link,
            parent_link / self.artifact.name,
            Path("relative"),
        ]:
            with self.subTest(path=path), self.assertRaises(ProviderError):
                a2.check_paths(replace(self.ctx, artifact=path))

    def test_evidence_paths_refuse_symlinks_nondirectory_parents_and_checkout(self):
        link = self.root / "dangling-result"
        link.symlink_to(self.root / "absent")
        parent = self.root / "linked-parent"
        parent.symlink_to(self.root, target_is_directory=True)
        other = self.root / "other"
        other.mkdir()
        for path in [
            link,
            parent / "result",
            self.artifact / "child",
            Path("relative"),
            other / ".." / "checkout" / "result",
        ]:
            with self.subTest(path=path), self.assertRaises(ProviderError):
                a2.check_paths(replace(self.ctx, result=path))

    def test_existing_or_concurrently_created_result_is_not_overwritten(self):
        self.ctx.result.write_bytes(b"prior")
        with self.assertRaises(ProviderError):
            self.invoke()
        self.assertEqual(self.tests, [])
        self.ctx.result.unlink()
        self.after_test = lambda: self.ctx.result.write_bytes(b"concurrent")
        with self.assertRaises(ProviderError):
            self.invoke()
        self.assertEqual(self.ctx.result.read_bytes(), b"concurrent")
        self.assertEqual(len(self.tests), 1)

    def test_timeout_cannot_emit_a_receipt(self):
        def timeout():
            raise subprocess.TimeoutExpired("cargo test", 1)

        self.after_test = timeout
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())


if __name__ == "__main__":
    unittest.main()
