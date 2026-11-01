"""Data-free controls for the candidate-bound M2 provider."""

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
m2 = importlib.import_module("m2")
common = importlib.import_module("common")
Context, ProviderError = common.Context, common.ProviderError


class M2Tests(unittest.TestCase):
    def setUp(self):
        root = HERE.parents[2] / "target"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="m2-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        self.artifact = self.root / "candidate"
        self.artifact.write_bytes(b"synthetic M2 provider fixture; not a release")
        self.source = "a" * 40
        self.ctx = Context(
            "M2",
            self.artifact,
            hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
            self.source,
            self.root / "result.json",
        )
        self.golden = self.root / "golden"
        self.golden.mkdir()
        (self.golden / "index.json").write_text("index sentinel", encoding="utf-8")
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        (self.bundle / "dope.safetensors").write_bytes(
            b"synthetic frozen DOPE placeholder"
        )
        (self.bundle / "jope.safetensors").write_bytes(
            b"synthetic frozen JOPE placeholder"
        )
        self.report_dir = self.root / "report"
        self.report = self.good_report()
        self.calls = []
        self.dirty = ""
        self.returncode = 0
        self.output_override = None
        self.mutate_source = False
        self.mutate_artifact = False
        self.bad_candidate_source = False
        self._start_patch(patch.object(m2, "ROOT", self.repo))
        self._start_patch(patch.object(common, "ROOT", self.repo))
        self._start_patch(
            patch.object(
                m2,
                "sha256",
                lambda path: (
                    m2.INDEX_SHA256
                    if path.name == "index.json"
                    else common.sha256(path)
                ),
            )
        )

    def _start_patch(self, patcher):
        self.addCleanup(patcher.stop)
        patcher.start()

    def good_report(self):
        stages = {
            stage: {
                "values": 1,
                "failed_values": 0,
                "max_tolerance_ratio": 0.0,
            }
            for stage in m2.REQUIRED_STAGES
        }
        return {
            "source": self.source,
            "consumer": "RNK-12",
            "fixture_index_sha256": m2.INDEX_SHA256,
            "contexts": 50,
            "slates": 100,
            "passed": True,
            "failed_values": 0,
            "trace_errors": [],
            "admitted_set_mismatches_outside_tie_band": 0,
            "decisive_admission_slates": 100,
            "tie_band_slates": 0,
            "stages": stages,
        }

    def run_child(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[:2] == ["git", "rev-parse"]:
            source = "b" * 40 if self.mutate_source else self.source
            return subprocess.CompletedProcess(args, 0, source, "")
        if args[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args, 0, self.dirty, "")
        if args[:2] == ["cargo", "run"]:
            if "inspect" in args:
                source = "b" * 40 if self.bad_candidate_source else self.ctx.source_sha
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
        if self.mutate_artifact:
            self.artifact.write_bytes(b"changed candidate")
        report_dir = Path(kwargs["env"]["JOPE_PARITY_REPORT_DIR"])
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "RNK-12.json").write_text(
            json.dumps(self.report), encoding="utf-8"
        )
        output = self.output_override or (
            f"test {m2.TEST} ... ok\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 500 filtered out\n"
        )
        return subprocess.CompletedProcess(args, self.returncode, output, "")

    def invoke(self, *, cargo_build_jobs="4"):
        values = {
            "JOPE_GOLDEN_DIR": str(self.golden),
            "JOPE_GOLDEN_INDEX_SHA256": m2.INDEX_SHA256,
            "JOPE_PARITY_BUNDLE_DIR": str(self.bundle),
            "JOPE_PARITY_REPORT_DIR": str(self.report_dir),
            "CARGO_BUILD_JOBS": cargo_build_jobs,
        }
        with patch.dict(os.environ, values, clear=True):
            return m2.qualify(self.ctx, run=self.run_child)

    def test_success_records_only_exact_full_stage_cpu_evidence(self):
        caller_cwd = os.getcwd()
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(os.getcwd(), caller_cwd)
        self.assertTrue(self.root.is_relative_to(HERE.parents[2] / "target"))
        result = json.loads(self.ctx.result.read_text())
        self.assertEqual(result["id"], "M2")
        self.assertEqual(result["source_sha"], self.source)
        self.assertEqual(result["artifact_sha256"], self.ctx.artifact_sha256)
        self.assertEqual(result["counts"], m2.SINGLE_PASS)
        cargo = next(
            args
            for args, _ in self.calls
            if args[:1] == ["cargo"] and args[1:2] == ["test"]
        )
        self.assertIn("--ignored", cargo)
        self.assertIn("--exact", cargo)
        self.assertNotIn("--nocapture", cargo)
        self.assertNotIn("-j8", cargo)
        self.assertNotIn("fused-cuda", cargo)
        child_env = next(
            kwargs["env"]
            for args, kwargs in self.calls
            if args[:1] == ["cargo"] and args[1:2] == ["test"]
        )
        self.assertEqual(child_env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(child_env["JOPE_SOURCE_SHA"], self.source)
        self.assertEqual(child_env["CARGO_BUILD_JOBS"], "4")
        self.assertTrue(all(kwargs["cwd"] == self.repo for _, kwargs in self.calls))

    def test_rejects_build_job_override_above_runner_limit(self):
        with self.assertRaisesRegex(ProviderError, "1..4"):
            self.invoke(cargo_build_jobs="8")
        self.assertFalse(any(args[:1] == ["cargo"] for args, _ in self.calls))

    def test_caller_environment_does_not_leak_into_replay(self):
        with patch.dict(
            os.environ,
            {
                "JOPE_QUALIFICATION_TEST_TMP": "/shared/unsafe/temp-root",
                "JOPE_UNRELATED_SENTINEL": "must-not-reach-child",
            },
            clear=False,
        ):
            self.assertEqual(self.invoke(), 0)
        child_env = next(
            kwargs["env"]
            for args, kwargs in self.calls
            if args[:1] == ["cargo"] and args[1:2] == ["test"]
        )
        self.assertNotIn("JOPE_UNRELATED_SENTINEL", child_env)
        self.assertNotIn("JOPE_QUALIFICATION_TEST_TMP", child_env)

    def test_interleaved_libtest_stdout_cannot_emit_pass(self):
        self.output_override = (
            f"test {m2.TEST} ... {{replay receipt}}\nok\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 500 filtered out\n"
        )
        with self.assertRaisesRegex(ProviderError, "exactly its one ignored"):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_existing_report_directory_preserves_old_files_and_refuses(self):
        self.report_dir.mkdir()
        old_stage = self.report_dir / "stages-existing.json"
        old_stage.write_text("previous evidence", encoding="utf-8")
        with self.assertRaisesRegex(ProviderError, "must not already exist"):
            self.invoke()
        self.assertEqual(old_stage.read_text(encoding="utf-8"), "previous evidence")
        self.assertFalse(any(args[:1] == ["cargo"] for args, _ in self.calls))

    def test_manifest_arguments_bind_parity_inputs_for_current_child(self):
        values = [
            "--golden-dir",
            "/approved/golden",
            "--golden-index-sha256",
            m2.INDEX_SHA256,
            "--parity-bundle-dir",
            "/approved/bundle",
            "--parity-report-dir",
            "/new/report",
        ]
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(m2, "context", return_value=self.ctx),
            patch.object(m2, "qualify", return_value=0) as qualify,
        ):
            self.assertEqual(m2.main(values), 0)
            self.assertEqual(
                [
                    os.environ["JOPE_GOLDEN_DIR"],
                    os.environ["JOPE_GOLDEN_INDEX_SHA256"],
                    os.environ["JOPE_PARITY_BUNDLE_DIR"],
                    os.environ["JOPE_PARITY_REPORT_DIR"],
                ],
                [
                    "/approved/golden",
                    m2.INDEX_SHA256,
                    "/approved/bundle",
                    "/new/report",
                ],
            )
            qualify.assert_called_once_with(self.ctx)

    def test_missing_stage_bad_coverage_or_admission_mismatch_refuse(self):
        for change in (
            lambda report: report["stages"].pop("tokens"),
            lambda report: report.update(contexts=49),
            lambda report: report.update(admitted_set_mismatches_outside_tie_band=1),
        ):
            with self.subTest(change=change):
                report = self.good_report()
                change(report)
                report_path = self.root / "report-input.json"
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises(ProviderError):
                    m2.validate_parity_report(report_path, self.source, m2.INDEX_SHA256)

    def test_dirty_or_wrong_source_candidate_refuses_before_replay(self):
        self.dirty = " M changed.rs"
        with self.assertRaises(ProviderError):
            self.invoke()
        self.dirty = ""
        self.bad_candidate_source = True
        with self.assertRaises(ProviderError):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_child_failure_and_source_or_artifact_drift_never_emit_pass(self):
        self.returncode = 23
        self.assertEqual(self.invoke(), 23)
        self.assertFalse(self.ctx.result.exists())
        self.returncode = 0
        self.mutate_source = True
        with self.assertRaises(ProviderError):
            self.invoke()
        self.mutate_source = False
        self.mutate_artifact = True
        with self.assertRaises(ProviderError):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_wrong_index_and_existing_result_are_refused(self):
        values = {
            "JOPE_GOLDEN_DIR": str(self.golden),
            "JOPE_GOLDEN_INDEX_SHA256": "0" * 64,
            "JOPE_PARITY_BUNDLE_DIR": str(self.bundle),
            "JOPE_PARITY_REPORT_DIR": str(self.report_dir),
        }
        with (
            patch.dict(os.environ, values, clear=False),
            self.assertRaises(ProviderError),
        ):
            m2.check_binding(self.ctx, self.run_child)
        self.ctx.result.write_text("prior", encoding="utf-8")
        with patch.dict(
            os.environ,
            {**values, "JOPE_GOLDEN_INDEX_SHA256": m2.INDEX_SHA256},
            clear=False,
        ):
            with self.assertRaises(ProviderError):
                m2.check_binding(self.ctx, self.run_child)
        self.assertEqual(self.ctx.result.read_text(), "prior")


if __name__ == "__main__":
    unittest.main()
