#!/usr/bin/env python3
"""Data-free E2 provider controls; synthetic reports are not release evidence."""

from copy import deepcopy
from dataclasses import replace
import contextlib
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
e2 = importlib.import_module("e2")
common = importlib.import_module("common")


class E2Tests(unittest.TestCase):
    def setUp(self):
        temporary = Path(os.environ.get("JOPE_QUALIFICATION_TEST_TMP", HERE / "target"))
        temporary.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="e2-", dir=temporary)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        (self.repo / e2.HARNESS).parent.mkdir(parents=True)
        (self.repo / e2.HARNESS).write_bytes(b"synthetic harness identity")
        (self.repo / "Cargo.lock").write_bytes(b"synthetic lock identity")
        self.artifact = self.root / "candidate"
        self.artifact.write_bytes(b"synthetic candidate, not a release")
        self.ctx = common.Context(
            "E2",
            self.artifact,
            common.sha256(self.artifact),
            "a" * 40,
            self.root / "result.json",
        )
        self.calls = []
        self.source = self.ctx.source_sha
        self.dirty = ""
        self.code = 0
        self.profile = "full"
        self.scope = "release"
        self.manifest_source = self.ctx.source_sha
        self.verify_ok = True
        self.edit_report = lambda report, folder: None
        self.after_harness = lambda: None
        self.harness_calls = 0
        self.addCleanup(patch.stopall)
        patch.object(e2, "ROOT", self.repo).start()
        patch.object(common, "ROOT", self.repo).start()
        patch.dict(os.environ, {"CARGO_BUILD_JOBS": "4"}).start()

    def report(self, folder):
        folder.mkdir()
        cases = [
            {
                "key": "routing/tree",
                "emissions": [
                    {"language": language} for language in sorted(e2.LANGUAGES)
                ],
            }
        ]
        calibrated = [
            {
                "key": "calibration/" + capability,
                "capability": capability,
                "emissions": [
                    {"language": language} for language in sorted(e2.LANGUAGES)
                ],
            }
            for capability in ["affine_floor", "platt"]
        ]
        cells = [
            {
                "key": "routing/tree",
                "language": language,
                "supported": True,
                "status": "pending",
            }
            for language in sorted(e2.LANGUAGES)
        ] + [
            {
                "key": "unsupported",
                "language": "cuda",
                "supported": False,
                "status": "unsupported_registry",
            }
        ]
        fixtures = {"cases": cases, "calibration": calibrated, "cells": cells}
        (folder / "fixtures.json").write_text(json.dumps(fixtures))
        (folder / "registry.json").write_text("{}")
        (folder / "harness.sh").write_bytes((self.repo / e2.HARNESS).read_bytes())

        def records(items):
            return [
                {
                    "key": case["key"],
                    "language": emission["language"],
                    "status": "pass",
                    "rows": 64,
                    "self_check_rows": 64,
                }
                for case in items
                for emission in case["emissions"]
            ]

        observed = deepcopy(cells)
        for cell in observed:
            if cell["supported"]:
                cell["status"] = "pass"
        return {
            "schema": "jope.export-parity/v1",
            "status": "pass",
            "passed": True,
            "source_sha": self.ctx.source_sha,
            "source_end": self.ctx.source_sha,
            "source_stable": True,
            "source_dirty": "",
            "source_dirty_end": "",
            "fault_injected": False,
            "harness_sha256": common.sha256(self.repo / e2.HARNESS),
            "cargo_lock_sha256": common.sha256(self.repo / "Cargo.lock"),
            "registry_sha256": common.sha256(folder / "registry.json"),
            "fixture_sha256": common.sha256(folder / "fixtures.json"),
            "files": {
                name: {
                    "bytes": (folder / name).stat().st_size,
                    "sha256": common.sha256(folder / name),
                }
                for name in ["fixtures.json", "registry.json", "harness.sh"]
            },
            "cells": observed,
            "expected_supported_cells": 3,
            "executed_supported_cells": 3,
            "unsupported_cells": 1,
            "cases": records(cases),
            "expected_emitted_modules": 3,
            "executed_emitted_modules": 3,
            "calibration": {
                "strict": True,
                "expected_modules": 6,
                "executed_modules": 6,
                "passed_modules": 6,
                "failed": [],
                "cases": records(calibrated),
            },
            "raw_numeric_contract": [
                {"language": language, "status": "pass", "rows": 72}
                for language in sorted(e2.LANGUAGES)
            ],
            "commands": [{"argv": ["fixture-child"], "exit": 0}],
            "toolchains": {
                "python": "Python 3.12.3",
                "rustc": "rustc 1.97.1 (fixture)",
                "node": "v26.1.0",
            },
        }

    def run_child(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[:2] == ["git", "rev-parse"]:
            text = self.source
        elif args[:2] == ["git", "status"]:
            text = self.dirty
        elif "inspect" in args:
            text = json.dumps(
                {
                    "manifest": {
                        "profile": self.profile,
                        "scope": self.scope,
                        "source": {"sha40": self.manifest_source},
                    }
                }
            )
        elif "verify" in args:
            text = json.dumps(
                {"schema": "jope.appliance-verify/v1", "ok": self.verify_ok}
            )
        else:
            self.harness_calls += 1
            folder = Path(args[-1])
            report = self.report(folder)
            self.edit_report(report, folder)
            (folder / "report.json").write_text(json.dumps(report))
            self.after_harness()
            return subprocess.CompletedProcess(
                args, self.code, "harness stdout", "harness stderr"
            )
        return subprocess.CompletedProcess(args, 0, text, "")

    def invoke(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return e2.qualify(self.ctx, run=self.run_child)

    def refuses(self):
        with self.assertRaises(
            (common.ProviderError, OSError, KeyError, TypeError, ValueError)
        ):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_complete_report_produces_candidate_bound_counts_and_retained_evidence(
        self,
    ):
        self.assertEqual(self.invoke(), 0)
        result = json.loads(self.ctx.result.read_text())
        self.assertEqual(result["id"], "E2")
        self.assertEqual(result["artifact_sha256"], self.ctx.artifact_sha256)
        self.assertEqual(result["source_sha"], self.ctx.source_sha)
        self.assertEqual(
            result["counts"],
            dict(collected=12, executed=12, passed=12, failed=0, skipped=0, excluded=0),
        )
        self.assertEqual(result["parity"]["unsupported_registry_cells"], 1)
        self.assertEqual(
            common.sha256(Path(result["parity"]["report"])), result["parity"]["sha256"]
        )
        args, kwargs = next((a, k) for a, k in self.calls if e2.HARNESS in a)
        self.assertEqual(
            args[:6], ["nice", "-n", "10", "bash", e2.HARNESS, "--require-calibration"]
        )
        self.assertEqual(kwargs["env"]["CARGO_BUILD_JOBS"], "4")
        self.assertEqual(kwargs["env"]["CARGO_INCREMENTAL"], "0")
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(self.harness_calls, 1)
        for args, kwargs in self.calls:
            if "inspect" in args or "verify" in args:
                self.assertEqual(args[:3], ["nice", "-n", "10"])
                self.assertEqual(kwargs["env"]["CARGO_BUILD_JOBS"], "4")

    def test_failed_child_preserves_exit_and_report_without_pass_receipt(self):
        self.code = 42
        self.assertEqual(self.invoke(), 42)
        self.assertFalse(self.ctx.result.exists())
        self.assertEqual(len(list(self.root.glob("*.parity-*/report/report.json"))), 1)

    def test_zero_missing_duplicate_failed_or_unchecked_modules_refuse(self):
        changes = [
            lambda r: r.update(cases=[]),
            lambda r: r["cases"].pop(),
            lambda r: r["cases"].append(deepcopy(r["cases"][0])),
            lambda r: r["cases"][0].update(status="fail"),
            lambda r: r["cases"][0].update(self_check_rows=0),
            lambda r: r["cases"][0].update(rows=0),
        ]
        for change in changes:
            self.edit_report = lambda r, f, change=change: change(r)
            with self.subTest(change=change):
                self.refuses()

    def test_summary_identity_dirty_fault_and_status_errors_refuse(self):
        for field, value in [
            ("schema", "wrong"),
            ("passed", False),
            ("status", "fail"),
            ("source_sha", "b" * 40),
            ("source_end", "b" * 40),
            ("source_stable", False),
            ("source_dirty", " M file"),
            ("source_dirty_end", "?? file"),
            ("fault_injected", True),
            ("harness_sha256", "0" * 64),
            ("cargo_lock_sha256", "0" * 64),
            ("registry_sha256", "0" * 64),
            ("fixture_sha256", "0" * 64),
        ]:
            self.edit_report = lambda r, f, field=field, value=value: r.update(
                {field: value}
            )
            with self.subTest(field=field):
                self.refuses()

    def test_registry_counts_and_inventory_cannot_shrink_or_relabel_unsupported(self):
        changes = [
            lambda r: r.update(expected_supported_cells=0),
            lambda r: r.update(executed_supported_cells=2),
            lambda r: r.update(unsupported_cells=0),
            lambda r: r["cells"].pop(),
            lambda r: r["cells"][-1].update(status="pass"),
            lambda r: r["cells"][0].update(supported=False),
        ]
        for change in changes:
            self.edit_report = lambda r, f, change=change: change(r)
            with self.subTest(change=change):
                self.refuses()

    def test_strict_calibration_and_each_language_are_required(self):
        changes = [
            lambda r: r["calibration"].update(strict=False),
            lambda r: r["calibration"].update(passed_modules=5),
            lambda r: r["calibration"]["cases"].pop(),
            lambda r: r["calibration"]["cases"][0].update(status="fail"),
            lambda r: r["raw_numeric_contract"].pop(),
            lambda r: r["raw_numeric_contract"][0].update(status="fail"),
            lambda r: r["commands"][0].update(exit=1),
        ]
        for change in changes:
            self.edit_report = lambda r, f, change=change: change(r)
            with self.subTest(change=change):
                self.refuses()

    def test_evidence_hash_drift_traversal_and_symlinks_refuse(self):
        for change in [
            lambda r, f: (f / "harness.sh").write_bytes(b"changed"),
            lambda r, f: r["files"].update(
                {"../escape": {"bytes": 1, "sha256": "0" * 64}}
            ),
            lambda r, f: (
                (f / "harness.sh").unlink(),
                (f / "harness.sh").symlink_to(self.repo / e2.HARNESS),
            ),
            lambda r, f: r.update(files={}),
        ]:
            self.edit_report = change
            with self.subTest(change=change):
                self.refuses()

    def test_candidate_must_verify_full_release_profile_and_source(self):
        for attribute, value in [
            ("profile", "lite"),
            ("scope", "development"),
            ("manifest_source", "b" * 40),
            ("verify_ok", False),
        ]:
            old = getattr(self, attribute)
            setattr(self, attribute, value)
            with self.subTest(attribute=attribute):
                self.refuses()
            setattr(self, attribute, old)
        self.assertEqual(self.harness_calls, 0)

    def test_source_binding_is_required_before_harness(self):
        for source, dirty in [
            ("b" * 40, ""),
            (self.ctx.source_sha, " M tracked"),
            (self.ctx.source_sha, "?? untracked"),
        ]:
            self.source, self.dirty = source, dirty
            with self.subTest(source=source, dirty=dirty):
                self.refuses()
        self.assertEqual(self.harness_calls, 0)

    def test_artifact_and_source_drift_after_harness_refuse(self):
        self.after_harness = lambda: setattr(self, "source", "b" * 40)
        self.refuses()
        self.source = self.ctx.source_sha
        self.after_harness = lambda: self.artifact.write_bytes(b"changed")
        self.refuses()

    def test_symlink_or_nonfile_candidate_is_refused_before_commands(self):
        for path in [self.root / "link", self.root / "directory"]:
            if path.name == "link":
                path.symlink_to(self.artifact)
            else:
                path.mkdir()
            self.ctx = replace(self.ctx, artifact=path)
            with self.subTest(path=path):
                self.refuses()
        self.assertEqual(self.calls, [])

    def test_result_inside_checkout_existing_or_symlinked_never_overwrites(self):
        outside = self.root / "outside"
        outside.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(outside, target_is_directory=True)
        original = self.ctx
        for result in [
            self.repo / "result.json",
            outside / ".." / "checkout" / "result.json",
            alias / "result.json",
        ]:
            self.ctx = replace(original, result=result)
            with self.subTest(result=result):
                self.refuses()
        self.ctx = original
        self.ctx.result.write_bytes(b"prior receipt")
        with self.assertRaises(common.ProviderError):
            self.invoke()
        self.assertEqual(self.ctx.result.read_bytes(), b"prior receipt")
        self.assertEqual(self.calls, [])

    def test_bad_job_budget_refuses_without_harness(self):
        with patch.dict(os.environ, {"CARGO_BUILD_JOBS": "32"}):
            self.refuses()
        self.assertEqual(self.harness_calls, 0)

    def test_missing_report_and_wrong_toolchains_never_pass(self):
        for tool in ["python", "node", "rustc"]:
            self.edit_report = lambda r, f, tool=tool: r["toolchains"].update(
                {tool: "wrong"}
            )
            with self.subTest(tool=tool):
                self.refuses()
        self.edit_report = lambda r, f: r.pop("cells")
        self.refuses()

    def test_exclusive_result_creation_preserves_a_racing_receipt(self):
        self.after_harness = lambda: self.ctx.result.write_bytes(b"racing receipt")
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(self.ctx.result.read_bytes(), b"racing receipt")

    def test_environment_context_refuses_wrong_gate_symlink_and_nonfile_inputs(self):
        values = dict(
            JOPE_REL07_GATE_ID="E2",
            JOPE_REL07_ARTIFACT=str(self.artifact),
            JOPE_REL07_ARTIFACT_SHA256=self.ctx.artifact_sha256,
            JOPE_REL07_SOURCE_SHA=self.ctx.source_sha,
            JOPE_REL07_RESULT=str(self.ctx.result),
        )
        with patch.dict(os.environ, values):
            self.assertEqual(common.context("E2"), self.ctx)
            for key, value in [
                ("JOPE_REL07_GATE_ID", "E1"),
                ("JOPE_REL07_ARTIFACT_SHA256", "0" * 64),
                ("JOPE_REL07_SOURCE_SHA", "abbreviated"),
                ("JOPE_REL07_ARTIFACT", str(self.repo)),
            ]:
                with self.subTest(key=key), patch.dict(os.environ, {key: value}):
                    with self.assertRaises(common.ProviderError):
                        common.context("E2")


if __name__ == "__main__":
    unittest.main()
