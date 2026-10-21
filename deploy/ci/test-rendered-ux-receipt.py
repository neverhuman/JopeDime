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


ACTUAL_MANUAL_CONFIG = {'metadata': {'source_sha': '9bf1037f73e98b083e72e4e373780c393a7fc4ec', 'run_id': '39641157469', 'attempt': '1', 'ci': {'commitHref': 'https://github.com/neverhuman/JopeDime/commit/9bf1037f73e98b083e72e4e373780c393a7fc4ec', 'commitHash': '9bf1037f73e98b083e72e4e373780c393a7fc4ec', 'buildHref': 'https://github.com/neverhuman/JopeDime/actions/runs/39641157469'}, 'gitCommit': {'shortHash': '9bf1037f7', 'hash': '9bf1037f73e98b083e72e4e373780c393a7fc4ec', 'subject': 'test: classify audit controls and pin source inventory', 'body': 'test: classify audit controls and pin source inventory\n\nAgent-Session: JopeDime\nAgent-Model: gpt-6-luna medium\nAgent-Task: codex/pr395-integration-20261021\n', 'author': {'name': 'Jepson Taylor', 'email': '130782313+jepsontaylor@users.noreply.github.com', 'time': 1792602430000}, 'committer': {'name': 'Jepson Taylor', 'email': '130782313+jepsontaylor@users.noreply.github.com', 'time': 1792602430000}, 'branch': 'HEAD'}, 'actualWorkers': 1}, 'workers': 1}


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

    def cli(self, mode, root=None, checkout=None, event=None):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return receipt.main([mode, "--root", str(root or self.root), "--source-sha", SHA,
                                 "--run-id", RUN, "--attempt", ATTEMPT]
                                + (["--checkout-sha", checkout] if checkout is not None else [])
                                + (["--event-sha", event] if event is not None else []))

    def result(self):
        return self.report["suites"][0]["specs"][0]["tests"][0]["results"][0]

    def mutate_json_attachment(self, suffix, change):
        item = next(a for a in self.result()["attachments"] if a["name"].endswith(suffix))
        value = json.loads(base64.b64decode(item["body"]))
        change(value)
        item["body"] = base64.b64encode(json.dumps(value).encode()).decode()
        self.save()

    def hosted(self, event="c" * 40):
        metadata = self.report["config"]["metadata"]
        repository = "https://github.com/neverhuman/JopeDime"
        metadata.update(ci={"commitHash": event, "commitHref": repository + "/commit/" + event,
                            "buildHref": repository + "/actions/runs/" + RUN,
                            "prHref": repository + "/pull/395", "prTitle": "synthetic control",
                            "prBaseHash": "b" * 40, "branch": "feature"},
                        gitCommit={"hash": SHA, "shortHash": SHA[:9], "subject": "synthetic",
                                   "body": "synthetic", "author": {"name": "fixture", "email": "fixture@example.invalid", "time": 0},
                                   "committer": {"name": "fixture", "email": "fixture@example.invalid", "time": 0}, "branch": "HEAD"},
                        gitDiff="diff --git synthetic nonbinding annotation", actualWorkers=1)
        self.report["config"]["workers"] = 1
        self.save()
        return metadata

    def test_actual_hosted_manual_report_metadata_is_accepted(self):
        # Verbatim config.metadata from retained raw manual report 39641157469/a1;
        # this checks real annotations, not the original rendered artifact qualification.
        config = ACTUAL_MANUAL_CONFIG
        sha = "9bf1037f73e98b083e72e4e373780c393a7fc4ec"
        before = copy.deepcopy(config)
        bound = receipt.report_identity(config, sha, "39641157469", "1", sha, sha)
        self.assertEqual(bound["checkout_sha"], sha)
        self.assertEqual(config, before)

    def test_pr_checkout_and_event_are_distinct_bound_and_portable(self):
        self.hosted()
        before = (self.root / "report.json").read_bytes()
        self.assertEqual(self.cli("create", checkout=SHA, event="c" * 40), 0)
        self.assertEqual(self.cli("verify", checkout=SHA, event="c" * 40), 0)
        value = json.loads((self.root / receipt.RECEIPT).read_text())
        self.assertEqual(value["schema"], "jope.rendered-ux-ci/v2")
        self.assertEqual((value["source_sha"], value["checkout_sha"], value["event_sha"]), (SHA, SHA, "c" * 40))
        self.assertEqual((self.root / "report.json").read_bytes(), before)
        moved = self.root.with_name("downloaded")
        shutil.copytree(self.root, moved)
        self.assertEqual(self.cli("verify", moved, SHA, "c" * 40), 0)
        self.assertEqual(self.cli("verify", moved, "c" * 40, "c" * 40), 1)
        self.assertEqual(self.cli("verify", moved, SHA, SHA), 1)

    def test_hosted_links_hashes_and_attempts_cannot_substitute_for_core(self):
        for where, key, value in (
            ("ci", "commitHash", SHA), ("ci", "commitHref", "https://github.com/neverhuman/JopeDime/commit/" + SHA),
            ("ci", "buildHref", "https://github.com/neverhuman/JopeDime/actions/runs/124"),
            ("ci", "buildHref", "https://github.com/neverhuman/JopeDime/actions/runs/123/attempts/2"),
            ("ci", "buildHref", "https://github.com/other/repository/actions/runs/123"),
            ("gitCommit", "hash", "c" * 40), ("gitCommit", "shortHash", "bbbbbbb")):
            with self.subTest(where=where, key=key, value=value):
                self.report = synthetic_report()
                metadata = self.hosted()
                metadata[where][key] = value
                self.save()
                self.assertEqual(self.cli("create", checkout=SHA, event="c" * 40), 1)
        for key in ("source_sha", "run_id", "attempt"):
            self.report = synthetic_report()
            metadata = self.hosted()
            metadata.pop(key)
            self.save()
            self.assertEqual(self.cli("create", checkout=SHA, event="c" * 40), 1)

    def test_known_hosted_annotations_remain_typed_and_unknown_fields_refused(self):
        changes = (
            lambda m: m["ci"].update(unknown="x"), lambda m: m["gitCommit"].update(unknown="x"),
            lambda m: m["gitCommit"]["author"].update(time=True),
            lambda m: m["gitCommit"]["committer"].update(unknown="x"),
            lambda m: m["ci"].update(prTitle=1), lambda m: m["ci"].update(branch=[]),
            lambda m: m["ci"].update(prBaseHash="short"), lambda m: m["ci"].update(prHref="https://example.invalid/pull/395"),
            lambda m: m.update(gitDiff={}), lambda m: m.update(gitDiff="x" * 100001))
        for change in changes:
            with self.subTest(change=change):
                self.report = synthetic_report()
                metadata = self.hosted()
                change(metadata)
                self.save()
                self.assertEqual(self.cli("create", checkout=SHA, event="c" * 40), 1)

    def test_matching_attempt_link_is_allowed_and_v1_receipt_not_relabeled(self):
        metadata = self.hosted()
        metadata["ci"]["buildHref"] += "/attempts/1"
        self.save()
        self.assertEqual(self.cli("create", checkout=SHA, event="c" * 40), 0)
        path = self.root / receipt.RECEIPT
        value = json.loads(path.read_text())
        value["schema"] = "jope.rendered-ux-ci/v1"
        value.pop("checkout_sha")
        value.pop("event_sha")
        path.write_text(json.dumps(value))
        self.assertEqual(self.cli("verify", checkout=SHA, event="c" * 40), 1)

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
