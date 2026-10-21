#!/usr/bin/env python3
"""Synthetic positive/negative receipt controls; no browser or hosted CI proof."""
import base64
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import zlib

spec = importlib.util.spec_from_file_location("rendered_receipt", Path(__file__).with_name("rendered-ux-receipt.py"))
receipt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipt)
SHA, RUN, ATTEMPT = "a" * 40, "123", "1"


def png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\0\0\0\0\0")) + chunk(b"IEND", b""))


def synthetic_report():
    box = {"x": 0, "y": 0, "width": 100, "height": 100, "right": 100}
    geometry = {"viewport": 100, "viewportHeight": 100, "documentWidth": 100,
                "main": box, "heading": box, "navigation": box, "panels": [box], "dialog": None}
    axe = {"testEngine": {"name": "axe-core", "version": "4.14.0"}, "violations": [], "passes": [{"id": "fixture"}]}
    attachments = [{"name": name, "contentType": kind, "body": base64.b64encode(raw).decode()}
                   for name, kind, raw in (("state.png", "image/png", png()),
                       ("state-geometry.json", "application/json", json.dumps(geometry).encode()),
                       ("state-accessibility.json", "application/json", json.dumps(axe).encode()))]
    specs = []
    for project in ("desktop", "mobile"):
        specs.append({"id": "fixture-" + project, "ok": True, "tests": [{
            "projectId": project, "expectedStatus": "passed", "status": "expected", "results": [{
                "status": "passed", "retry": 0, "errors": [], "attachments": copy.deepcopy(attachments)}]}]})
    return {"config": {"metadata": {"source_sha": SHA, "run_id": RUN, "attempt": ATTEMPT},
                       "projects": [{"id": "desktop"}, {"id": "mobile"}]},
            "suites": [{"specs": specs}], "errors": [],
            "stats": {"expected": 2, "skipped": 0, "unexpected": 0, "flaky": 0}}


class ReceiptControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rendered-receipt-control-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "evidence"
        (self.root / "html/data").mkdir(parents=True)
        (self.root / "results").mkdir()
        (self.root / "html/index.html").write_text("<!doctype html><html>synthetic fixture</html>")
        (self.root / "html/data/state.png").write_bytes(png())
        (self.root / "results/.last-run.json").write_text('{"status":"passed","failedTests":[]}')
        (self.root / "junit.xml").write_text('<testsuites tests="2" failures="0" errors="0" skipped="0"><testsuite><testcase/><testcase/></testsuite></testsuites>')
        self.report = synthetic_report()
        self.save()

    def save(self):
        (self.root / "report.json").write_text(json.dumps(self.report))

    def build(self):
        return receipt.build(self.root, SHA, RUN, ATTEMPT)

    def cli(self, mode, root=None):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return receipt.main([mode, "--root", str(root or self.root), "--source-sha", SHA,
                                 "--run-id", RUN, "--attempt", ATTEMPT])

    def result(self):
        return self.report["suites"][0]["specs"][0]["tests"][0]["results"][0]

    def mutate_json_attachment(self, suffix, change):
        item = next(a for a in self.result()["attachments"] if a["name"].endswith(suffix))
        value = json.loads(base64.b64decode(item["body"]))
        change(value)
        item["body"] = base64.b64encode(json.dumps(value).encode()).decode()
        self.save()

    def test_playwright_actual_workers_is_accepted_without_weakening_identity(self):
        self.report["config"]["metadata"]["actualWorkers"] = 1
        self.report["config"]["workers"] = 1
        self.save()
        self.assertEqual(self.cli("create"), 0)
        self.assertEqual(self.cli("verify"), 0)
        self.report["config"]["metadata"]["source_sha"] = "b" * 40
        self.save()
        self.assertEqual(self.cli("verify"), 1)

    def test_actual_workers_must_be_positive_integer_within_configured_bound(self):
        self.report["config"]["workers"] = 1
        for value in (0, -1, True, "1", 2):
            with self.subTest(value=value):
                self.report["config"]["metadata"]["actualWorkers"] = value
                self.save()
                self.assertEqual(self.cli("create"), 1)
        self.report["config"]["metadata"]["actualWorkers"] = 1
        for value in (0, -1, True, "1"):
            with self.subTest(configured=value):
                self.report["config"]["workers"] = value
                self.save()
                self.assertEqual(self.cli("create"), 1)

    def test_unknown_metadata_stays_refused(self):
        self.report["config"]["metadata"]["authority"] = "approved"
        self.save()
        self.assertEqual(self.cli("create"), 1)

    def test_create_verify_and_relocation_preserve_identity(self):
        self.assertEqual(self.cli("create"), 0)
        self.assertEqual(self.cli("verify"), 0)
        self.assertEqual(self.build()["checks"], 2)
        moved = self.root.with_name("downloaded")
        shutil.copytree(self.root, moved)
        self.assertEqual(self.cli("verify", moved), 0)

    def test_existing_receipt_is_never_overwritten(self):
        self.assertEqual(self.cli("create"), 0)
        before = (self.root / receipt.RECEIPT).read_bytes()
        self.assertEqual(self.cli("create"), 1)
        self.assertEqual(before, (self.root / receipt.RECEIPT).read_bytes())

    def test_stale_source_run_and_attempt_are_refused(self):
        for key in ("source_sha", "run_id", "attempt"):
            with self.subTest(key=key):
                old = self.report["config"]["metadata"][key]
                self.report["config"]["metadata"][key] = "wrong"
                self.save()
                self.assertEqual(self.cli("create"), 1)
                self.report["config"]["metadata"][key] = old

    def test_missing_execution_metadata_is_refused(self):
        self.report["config"]["metadata"] = {}
        self.save()
        self.assertEqual(self.cli("create"), 1)

    def test_zero_skip_flaky_and_failed_counts_are_refused(self):
        for key, value in (("expected", 0), ("skipped", 1), ("flaky", 1), ("unexpected", 1)):
            with self.subTest(key=key):
                original = self.report["stats"][key]
                self.report["stats"][key] = value
                self.save()
                self.assertEqual(self.cli("create"), 1)
                self.report["stats"][key] = original

    def test_missing_project_is_refused_without_a_fixed_test_count(self):
        self.report["suites"][0]["specs"].pop()
        self.report["stats"]["expected"] = 1
        (self.root / "junit.xml").write_text('<testsuites tests="1" failures="0" errors="0" skipped="0"><testcase/></testsuites>')
        self.save()
        self.assertEqual(self.cli("create"), 1)

    def test_skipped_expected_failure_and_retry_are_refused(self):
        for path, value in (("expectedStatus", "failed"), ("status", "skipped")):
            with self.subTest(path=path):
                test = self.report["suites"][0]["specs"][0]["tests"][0]
                old = test[path]
                test[path] = value
                self.save()
                self.assertEqual(self.cli("create"), 1)
                test[path] = old
        self.result()["retry"] = 1
        self.save()
        self.assertEqual(self.cli("create"), 1)

    def test_each_test_requires_all_three_attachment_types(self):
        original = copy.deepcopy(self.result()["attachments"])
        for index in range(3):
            with self.subTest(index=index):
                self.result()["attachments"] = copy.deepcopy(original)
                self.result()["attachments"].pop(index)
                self.save()
                self.assertEqual(self.cli("create"), 1)

    def test_mismatched_state_names_and_duplicate_attachments_are_refused(self):
        self.result()["attachments"][0]["name"] = "different.png"
        self.save()
        self.assertEqual(self.cli("create"), 1)
        self.result()["attachments"][0]["name"] = "state.png"
        self.result()["attachments"].append(copy.deepcopy(self.result()["attachments"][0]))
        self.save()
        self.assertEqual(self.cli("create"), 1)

    def test_axe_violations_and_empty_scan_are_refused(self):
        for field, value in (("violations", [{"id": "color-contrast"}]), ("passes", [])):
            with self.subTest(field=field):
                self.report = synthetic_report()
                self.mutate_json_attachment("-accessibility.json", lambda a: a.update({field: value}))
                self.assertEqual(self.cli("create"), 1)

    def test_overflow_nonfinite_and_invalid_panel_geometry_are_refused(self):
        changes = (lambda g: g.update(documentWidth=101.5),
                   lambda g: g.update(viewport=float("nan")),
                   lambda g: g["panels"][0].update(width=-1),
                   lambda g: g.update(dialog={"x": 0, "y": 0, "right": 101, "bottom": 100}))
        for change in changes:
            with self.subTest(change=change):
                self.report = synthetic_report()
                self.mutate_json_attachment("-geometry.json", change)
                self.assertEqual(self.cli("create"), 1)

    def test_corrupt_png_and_wrong_content_type_are_refused(self):
        self.result()["attachments"][0]["body"] = base64.b64encode(png()[:-1]).decode()
        self.save()
        self.assertEqual(self.cli("create"), 1)
        self.report = synthetic_report()
        self.result()["attachments"][0]["contentType"] = "image/jpeg"
        self.save()
        self.assertEqual(self.cli("create"), 1)

    def test_invalid_base64_is_refused(self):
        self.result()["attachments"][0]["body"] = "not base64!"
        self.save()
        self.assertEqual(self.cli("create"), 1)

    def test_junit_zero_test_or_skipped_case_is_refused(self):
        for xml in ('<testsuites tests="0" failures="0" errors="0" skipped="0"/>',
                    '<testsuites tests="2" failures="0" errors="0" skipped="0"><testcase><skipped/></testcase><testcase/></testsuites>'):
            (self.root / "junit.xml").write_text(xml)
            self.assertEqual(self.cli("create"), 1)

    def test_missing_report_junit_and_portable_images_are_refused(self):
        for name in ("report.json", "junit.xml", "html/data/state.png"):
            with self.subTest(name=name):
                path = self.root / name
                before = path.read_bytes()
                path.unlink()
                self.assertEqual(self.cli("create"), 1)
                path.write_bytes(before)

    def test_tampered_report_or_auxiliary_artifact_is_refused(self):
        self.assertEqual(self.cli("create"), 0)
        original = (self.root / "report.json").read_bytes()
        self.report["extra"] = "different retained bytes"
        self.save()
        self.assertEqual(self.cli("verify"), 1)
        (self.root / "report.json").write_bytes(original)
        (self.root / "html/index.html").write_text("tampered")
        self.assertEqual(self.cli("verify"), 1)

    def test_tampered_receipt_is_refused(self):
        self.assertEqual(self.cli("create"), 0)
        path = self.root / receipt.RECEIPT
        value = json.loads(path.read_text())
        value["source_sha"] = "b" * 40
        path.write_text(json.dumps(value))
        self.assertEqual(self.cli("verify"), 1)

    def test_missing_and_added_pinned_artifacts_are_refused(self):
        self.assertEqual(self.cli("create"), 0)
        (self.root / "html/extra.txt").write_text("not in original inventory")
        self.assertEqual(self.cli("verify"), 1)
        (self.root / "html/extra.txt").unlink()
        (self.root / "results/.last-run.json").unlink()
        self.assertEqual(self.cli("verify"), 1)

    def test_symlink_and_traversal_attachment_are_refused(self):
        path = self.root / "html/index.html"
        path.unlink()
        path.symlink_to(self.root / "junit.xml")
        self.assertEqual(self.cli("create"), 1)
        path.unlink()
        path.write_text("html")
        item = self.result()["attachments"][0]
        item.pop("body")
        item["path"] = "../outside.png"
        self.save()
        self.assertEqual(self.cli("create"), 1)


if __name__ == "__main__":
    unittest.main()
