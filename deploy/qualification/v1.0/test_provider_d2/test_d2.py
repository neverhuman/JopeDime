"""Data-free contract tests for the REL-07 D2 size-only provider."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "providers"))
import d2  # noqa: E402
from common import ProviderError  # noqa: E402


SOURCE = "a" * 40


def size_report(*, final=8_000_000, base=7_800_000, checks=None):
    if checks is None:
        checks = [
            {"what": "base_executable_bytes", "pass": True},
            {"what": "final_executable_bytes", "pass": True},
        ]
    return {
        "schema": "jope.appliance-size/v1",
        "profile": "fused-cpu",
        "base_executable_bytes": base,
        "final_executable_bytes": final,
        "budget_checks": checks,
        "ok": all(row["pass"] for row in checks),
    }


class D2ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="d2-provider-")
        self.root = Path(self.temp.name)
        self.artifact = self.root / "candidate.bin"
        self.artifact.write_bytes(b"synthetic candidate bytes; not a release")
        self.result = self.root / "result.json"
        self.ctx = SimpleNamespace(
            gate_id="D2",
            artifact=self.artifact,
            artifact_sha256=hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
            source_sha=SOURCE,
            result=self.result,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_size_command_uses_budget_and_no_speed_command(self):
        argv = d2.command("size-report", self.artifact)
        self.assertEqual(argv[-2:], ["--budget", "deploy/appliance/size-budgets.toml"])
        self.assertIn("size-report", argv)
        self.assertNotIn("bench", argv)
        self.assertNotIn("latency", " ".join(argv))

    def test_cargo_environment_uses_reused_lane_target_and_four_jobs(self):
        with mock.patch.dict(
            "os.environ", {"CARGO_TARGET_DIR": "/home/ubuntu/foreign-target"}
        ):
            env = d2.cargo_env()
        self.assertNotIn("CARGO_TARGET_DIR", env)
        self.assertEqual(env["CARGO_BUILD_JOBS"], "4")
        self.assertEqual(env["CARGO_INCREMENTAL"], "0")
        self.assertEqual(env["CARGO_PROFILE_DEV_DEBUG"], "line-tables-only")

    def test_report_passes_with_both_budget_checks_and_under_40_mb(self):
        _, passed, passed_count = d2.parse_size_report(json.dumps(size_report()))
        self.assertTrue(passed)
        self.assertEqual(passed_count, 3)

    def test_report_fails_if_either_budget_check_fails(self):
        report = size_report(
            checks=[
                {"what": "base_executable_bytes", "pass": False},
                {"what": "final_executable_bytes", "pass": True},
            ]
        )
        _, passed, passed_count = d2.parse_size_report(json.dumps(report))
        self.assertFalse(passed)
        self.assertEqual(passed_count, 2)

    def test_report_fails_above_absolute_40_mb_target_even_when_budget_passes(self):
        _, passed, passed_count = d2.parse_size_report(
            json.dumps(size_report(final=40_000_001))
        )
        self.assertFalse(passed)
        self.assertEqual(passed_count, 2)

    def test_report_rejects_missing_duplicate_or_extra_budget_checks(self):
        for checks in (
            [{"what": "final_executable_bytes", "pass": True}],
            [
                {"what": "base_executable_bytes", "pass": True},
                {"what": "base_executable_bytes", "pass": True},
            ],
            [
                {"what": "base_executable_bytes", "pass": True},
                {"what": "final_executable_bytes", "pass": True},
                {"what": "latency_ms", "pass": True},
            ],
        ):
            with self.subTest(checks=checks), self.assertRaises(ProviderError):
                d2.parse_size_report(json.dumps(size_report(checks=checks)))

    def test_report_rejects_wrong_profile_schema_and_bad_size(self):
        report = size_report()
        report["profile"] = "full"
        with self.assertRaises(ProviderError):
            d2.parse_size_report(json.dumps(report))
        report = size_report()
        report["schema"] = "unknown"
        with self.assertRaises(ProviderError):
            d2.parse_size_report(json.dumps(report))
        report = size_report(final=True)
        with self.assertRaises(ProviderError):
            d2.parse_size_report(json.dumps(report))

    def test_context_rejects_symlink_and_non_file_candidate(self):
        link = self.root / "candidate-link"
        link.symlink_to(self.artifact)
        directory = self.root / "candidate-dir"
        directory.mkdir()
        for candidate in (link, directory):
            env = {
                "JOPE_REL07_GATE_ID": "D2",
                "JOPE_REL07_ARTIFACT": str(candidate),
                "JOPE_REL07_ARTIFACT_SHA256": self.ctx.artifact_sha256,
                "JOPE_REL07_SOURCE_SHA": SOURCE,
                "JOPE_REL07_RESULT": str(self.root / (candidate.name + ".result")),
            }
            with (
                self.subTest(candidate=candidate),
                mock.patch.dict("os.environ", env),
                self.assertRaises(ProviderError),
            ):
                d2.context("D2")

    def test_context_rejects_candidate_digest_mismatch(self):
        env = {
            "JOPE_REL07_GATE_ID": "D2",
            "JOPE_REL07_ARTIFACT": str(self.artifact),
            "JOPE_REL07_ARTIFACT_SHA256": "0" * 64,
            "JOPE_REL07_SOURCE_SHA": SOURCE,
            "JOPE_REL07_RESULT": str(self.result),
        }
        with mock.patch.dict("os.environ", env), self.assertRaises(ProviderError):
            d2.context("D2")

    def test_inspection_requires_fused_cpu_source_pin_and_non_development(self):
        cases = (
            {"profile": "full", "scope": "release", "source": {"sha40": SOURCE}},
            {
                "profile": "fused-cpu",
                "scope": "development",
                "source": {"sha40": SOURCE},
            },
            {"profile": "fused-cpu", "scope": "release", "source": {"sha40": "b" * 40}},
        )
        for manifest in cases:

            def run(args, **kwargs):
                if "inspect" in args:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=json.dumps({"manifest": manifest}),
                        stderr="",
                    )
                raise AssertionError("verify should not run after wrong identity")

            with self.subTest(manifest=manifest), self.assertRaises(ProviderError):
                d2.verify_candidate(self.ctx, run)

    def test_verify_requires_full_byte_success(self):
        inspect = {
            "manifest": {
                "profile": "fused-cpu",
                "scope": "release",
                "source": {"sha40": SOURCE},
            }
        }

        def run(args, **kwargs):
            if "inspect" in args:
                return SimpleNamespace(
                    returncode=0, stdout=json.dumps(inspect), stderr=""
                )
            if "verify" in args:
                return SimpleNamespace(
                    returncode=0, stdout=json.dumps({"ok": True}), stderr=""
                )
            raise AssertionError(args)

        with self.assertRaises(ProviderError):
            d2.verify_candidate(self.ctx, run)

    def test_qualify_writes_pass_from_size_report_only(self):
        inspect = {
            "manifest": {
                "profile": "fused-cpu",
                "scope": "release",
                "source": {"sha40": SOURCE},
            }
        }
        verify = {"schema": "jope.appliance-verify/v1", "ok": True}
        report = size_report()

        def run(args, **kwargs):
            if args[:3] == ["git", "rev-parse", "HEAD"]:
                return SimpleNamespace(returncode=0, stdout=SOURCE + "\n", stderr="")
            if args[:3] == ["git", "status", "--porcelain"]:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if "inspect" in args:
                return SimpleNamespace(
                    returncode=0, stdout=json.dumps(inspect), stderr=""
                )
            if "verify" in args:
                return SimpleNamespace(
                    returncode=0, stdout=json.dumps(verify), stderr=""
                )
            if "size-report" in args:
                return SimpleNamespace(
                    returncode=0, stdout=json.dumps(report), stderr=""
                )
            raise AssertionError(args)

        self.assertEqual(d2.qualify(self.ctx, run), 0)
        payload = json.loads(self.result.read_text())
        self.assertEqual(payload["id"], "D2")
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["artifact_sha256"], self.ctx.artifact_sha256)

    def test_qualify_emits_size_failure_without_latency_verdict(self):
        inspect = {
            "manifest": {
                "profile": "fused-cpu",
                "scope": "release",
                "source": {"sha40": SOURCE},
            }
        }
        verify = {"schema": "jope.appliance-verify/v1", "ok": True}
        report = size_report(final=40_000_001)

        def run(args, **kwargs):
            if args[:3] == ["git", "rev-parse", "HEAD"]:
                return SimpleNamespace(returncode=0, stdout=SOURCE + "\n", stderr="")
            if args[:3] == ["git", "status", "--porcelain"]:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if "inspect" in args:
                return SimpleNamespace(
                    returncode=0, stdout=json.dumps(inspect), stderr=""
                )
            if "verify" in args:
                return SimpleNamespace(
                    returncode=0, stdout=json.dumps(verify), stderr=""
                )
            if "size-report" in args:
                return SimpleNamespace(
                    returncode=0, stdout=json.dumps(report), stderr=""
                )
            raise AssertionError(args)

        self.assertEqual(d2.qualify(self.ctx, run), 1)
        payload = json.loads(self.result.read_text())
        self.assertEqual(payload["status"], "FAIL")
        self.assertEqual(payload["counts"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
