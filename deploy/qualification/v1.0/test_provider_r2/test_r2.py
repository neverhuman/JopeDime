#!/usr/bin/env python3
"""Data-free controller controls; these fixtures never qualify a release."""

import copy
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
r2 = importlib.import_module("r2")


class R2Tests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("JOPE_QUALIFICATION_TEST_TMP", HERE / "target"))
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="r2-", dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        artifact = self.root / "candidate"
        artifact.write_bytes(b"data-free release-controller fixture")
        self.ctx = r2.Context(
            "R2",
            artifact,
            hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "a" * 40,
            self.root / "result.json",
        )
        self.calls = []
        self.source = self.ctx.source_sha
        self.dirty = ""
        self.target = "x86_64-unknown-linux-gnu"
        self.build_code = self.test_code = 0
        self.override = None
        self.after_test = lambda: None
        self.rlib = self.root / "libjope_service.rlib"
        self.rlib.write_bytes(b"fixture library; never linked or executed")
        self.build_output = json.dumps(
            dict(
                reason="compiler-artifact",
                target=dict(name="jope_service", kind=["lib"]),
                filenames=[str(self.rlib)],
            )
        )
        self.rows = [
            dict(
                schema="jope.rel07.worker-isolation/v1",
                fault=fault,
                failure=failure,
                source_sha=self.ctx.source_sha,
                train_sha256=r2.TRAIN_SHA,
                wall_limit_ms=1000,
                elapsed_ms=1020,
                baseline_rss_bytes=10000,
                peak_rss_bytes=11000,
                rss_samples=20,
                completed=True,
                champion_present=True,
                workers_started=2,
                workers_remaining=0,
            )
            for fault, failure in [("timeout", "Timeout"), ("memory", "MemoryLimit")]
        ]
        self.addCleanup(patch.stopall)
        patch.object(r2, "ROOT", self.repo).start()

    def output(self):
        return (
            f"running 1 test\ntest {r2.TEST} ... "
            + "\n".join("REL07_R2 " + json.dumps(row) for row in self.rows)
            + "\nok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 499 filtered out; finished in 1.25s\n"
        )

    def run_child(self, args, **kwargs):
        self.calls.append(args)
        code = 0
        if args[:2] == ["git", "rev-parse"]:
            output = self.source
        elif args[:2] == ["git", "status"]:
            output = self.dirty
        elif args[0] == "rustc":
            output = "rustc fixture\nhost: x86_64-unknown-linux-gnu\n"
        elif args[:2] == ["cargo", "run"]:
            if "inspect" in args:
                output = json.dumps(
                    dict(
                        manifest=dict(
                            profile="full",
                            scope="release",
                            source=dict(sha40=self.ctx.source_sha),
                            target=self.target,
                        )
                    )
                )
            else:
                output = json.dumps(dict(schema="jope.appliance-verify/v1", ok=True))
        elif args[:2] == ["cargo", "build"]:
            code, output = self.build_code, self.build_output
        elif args[:2] == ["cargo", "test"]:
            code = self.test_code
            output = self.output() if self.override is None else self.override
            self.assertIn("--exact", args)
            self.assertIn("--ignored", args)
            self.assertEqual(kwargs["env"]["CARGO_BUILD_JOBS"], "4")
            self.assertTrue(Path(kwargs["env"]["JOPE_DSC_SERVICE_RLIB"]).is_file())
            self.assertEqual(
                kwargs["env"]["JOPE_BUILD_SOURCE_SHA"], self.ctx.source_sha
            )
            self.assertFalse(Path(kwargs["env"]["TMPDIR"]).is_relative_to(self.repo))
            self.after_test()
        else:
            self.fail(f"unexpected command: {args}")
        return subprocess.CompletedProcess(args, code, output, "")

    def invoke(self):
        return r2.qualify(self.ctx, run=self.run_child)

    def refused(self):
        with self.assertRaises((r2.ProviderError, OSError)):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_both_faults_and_exact_interleaved_test_produce_bound_result(self):
        self.assertEqual(self.invoke(), 0)
        report = json.loads(self.ctx.result.read_text())
        self.assertEqual(report["counts"], r2.COUNTS)
        self.assertEqual(report["measurements"], self.rows)
        self.assertEqual(report["source_sha"], self.ctx.source_sha)
        self.assertEqual(report["artifact_sha256"], self.ctx.artifact_sha256)
        self.assertEqual(report["service_rlib_sha256"], r2.sha256(self.rlib))

    def test_missing_duplicate_or_mixed_result_summaries_refused(self):
        good = self.output()
        for text in [
            "",
            good + good,
            good
            + "\ntest result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n",
        ]:
            with self.subTest(text=text), self.assertRaises(r2.ProviderError):
                r2.measurements(text, self.ctx.source_sha)

    def test_zero_ignored_failed_measured_and_extra_cases_refused(self):
        good = self.output()
        for counts in [
            "0 passed; 0 failed; 0 ignored; 0 measured",
            "0 passed; 0 failed; 1 ignored; 0 measured",
            "0 passed; 1 failed; 0 ignored; 0 measured",
            "1 passed; 0 failed; 0 ignored; 1 measured",
            "2 passed; 0 failed; 0 ignored; 0 measured",
        ]:
            with self.subTest(counts=counts), self.assertRaises(r2.ProviderError):
                r2.measurements(
                    good.replace("1 passed; 0 failed; 0 ignored; 0 measured", counts),
                    self.ctx.source_sha,
                )

    def test_wrong_native_test_name_refused(self):
        self.override = self.output().replace(r2.TEST, "unrelated::passing_test")
        self.refused()

    def test_missing_duplicate_or_wrong_fault_rows_refused(self):
        good = copy.deepcopy(self.rows)
        for rows in [
            [],
            good[:1],
            good + good[:1],
            [good[0], good[0]],
            [dict(good[0], fault="other"), good[1]],
        ]:
            self.rows = rows
            with self.subTest(rows=rows), self.assertRaises(r2.ProviderError):
                r2.measurements(self.output(), self.ctx.source_sha)

    def test_wrong_identity_or_failure_or_no_successful_completion_refused(self):
        good = copy.deepcopy(self.rows)
        for key, value in [
            ("schema", "other"),
            ("source_sha", "b" * 40),
            ("train_sha256", "0" * 64),
            ("failure", "Timeout"),
            ("completed", False),
            ("champion_present", False),
        ]:
            self.rows = copy.deepcopy(good)
            self.rows[1][key] = value
            with self.subTest(key=key), self.assertRaises(r2.ProviderError):
                r2.measurements(self.output(), self.ctx.source_sha)

    def test_all_numeric_measurements_are_required_typed_and_bounded(self):
        good = copy.deepcopy(self.rows)
        for key, value in [
            ("wall_limit_ms", 1001),
            ("elapsed_ms", 3001),
            ("baseline_rss_bytes", 0),
            ("peak_rss_bytes", 11001),
            ("peak_rss_bytes", 9999),
            ("rss_samples", 0),
            ("workers_started", 1),
            ("workers_remaining", 1),
            ("rss_samples", True),
            ("elapsed_ms", float("nan")),
            ("peak_rss_bytes", None),
        ]:
            self.rows = copy.deepcopy(good)
            self.rows[0][key] = value
            with (
                self.subTest(key=key, value=value),
                self.assertRaises(r2.ProviderError),
            ):
                r2.measurements(self.output(), self.ctx.source_sha)

    def test_wrong_or_dirty_source_refused_before_build(self):
        self.source = "b" * 40
        self.refused()
        self.assertFalse(any(args[:2] == ["cargo", "build"] for args in self.calls))

    def test_dirty_source_refused(self):
        self.dirty = " M src/lib.rs"
        self.refused()

    def test_source_drift_during_test_refused(self):
        self.after_test = lambda: setattr(self, "source", "b" * 40)
        self.refused()

    def test_candidate_byte_drift_refused(self):
        self.after_test = lambda: self.ctx.artifact.write_bytes(b"changed")
        self.refused()

    def test_same_byte_candidate_symlink_replacement_refused(self):
        def replace():
            real = self.root / "other-candidate"
            real.write_bytes(self.ctx.artifact.read_bytes())
            self.ctx.artifact.unlink()
            self.ctx.artifact.symlink_to(real)

        self.after_test = replace
        self.refused()

    def test_missing_library_refused(self):
        self.build_output = "{}"
        self.refused()

    def test_ambiguous_service_library_artifacts_refused(self):
        with self.assertRaises(r2.ProviderError):
            r2.library(self.build_output + "\n" + self.build_output)

    def test_wrong_gate_and_non_linux_host_refused_before_children(self):
        wrong = r2.Context(
            "T1",
            self.ctx.artifact,
            self.ctx.artifact_sha256,
            self.ctx.source_sha,
            self.ctx.result,
        )
        with self.assertRaises(r2.ProviderError):
            r2.qualify(wrong, run=self.run_child)
        with (
            patch.object(r2.sys, "platform", "darwin"),
            self.assertRaises(r2.ProviderError),
        ):
            self.invoke()
        self.assertEqual(self.calls, [])

    def test_dependency_snapshot_detects_mutation(self):
        def replace():
            snapshot = (
                self.ctx.result.with_name(self.ctx.result.name + ".r2")
                / "dependencies"
                / self.rlib.name
            )
            snapshot.write_bytes(b"changed")

        self.after_test = replace
        self.refused()

    def test_build_and_test_failures_never_write_pass(self):
        self.build_code = 1
        self.refused()

    def test_owned_test_timeout_never_writes_pass(self):
        self.test_code = 124
        self.refused()

    def test_other_candidate_target_refused(self):
        self.target = "aarch64-apple-darwin"
        self.refused()

    def test_existing_result_or_result_symlink_refused_without_overwrite(self):
        self.ctx.result.write_text("preserved")
        with self.assertRaises(r2.ProviderError):
            self.invoke()
        self.assertEqual(self.ctx.result.read_text(), "preserved")
        self.ctx.result.unlink()
        self.ctx.result.symlink_to(self.root / "absent")
        with self.assertRaises(r2.ProviderError):
            self.invoke()

    def test_result_inside_source_or_symlink_parent_refused(self):
        for result in [self.repo / "result.json", self.root / "link/result.json"]:
            link = self.root / "link"
            if not link.exists():
                link.symlink_to(self.repo, target_is_directory=True)
            ctx = r2.Context(
                "R2",
                self.ctx.artifact,
                self.ctx.artifact_sha256,
                self.ctx.source_sha,
                result,
            )
            with self.subTest(result=result), self.assertRaises(r2.ProviderError):
                r2.qualify(ctx, run=self.run_child)


if __name__ == "__main__":
    unittest.main()
