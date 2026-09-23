#!/usr/bin/env python3
"""Negative acceptance cases for static mapping and real collection failures."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("coverage_map", ROOT / "deploy/ci/coverage.py")
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.mapping = json.loads((ROOT / "deploy/ci/coverage.json").read_text())
        self.discovered = {key: item["discovery"] for key, item in self.mapping["items"].items()}

    def test_current_mapping(self):
        self.assertEqual([], coverage.validate(self.mapping, self.discovered))

    def test_each_unassigned_kind_fails(self):
        for kind in ("package", "feature", "target", "python-family", "workflow", "workflow-job", "benchmark-source"):
            with self.subTest(kind=kind):
                discovered = dict(self.discovered, **{f"new:{kind}": {"kind": kind}})
                self.assertIn(f"unassigned addition: new:{kind}", coverage.validate(self.mapping, discovered))

    def test_changed_target_requirements_fail(self):
        discovered = copy.deepcopy(self.discovered)
        key = next(k for k in discovered if k.startswith("target:"))
        discovered[key]["required_features"] = ["new-capability"]
        self.assertTrue(any("changed discovery" in e for e in coverage.validate(self.mapping, discovered)))

    def test_required_feature_target_cannot_claim_default_execution(self):
        key = "target:jope-policy/test/jope_dime"
        for lane, disposition in (("rust-check", None), ("rust-test", None), (None, "full-cpu-default")):
            with self.subTest(lane=lane, disposition=disposition):
                mapping = copy.deepcopy(self.mapping)
                if lane:
                    # Exercise a default-only lane even when production CI has
                    # gained an explicit feature selection for other targets.
                    mapping["lanes"][lane]["feature_selection"] = "package-defaults"
                mapping["items"][key]["lanes"] = [lane] if lane else []
                mapping["items"][key]["dispositions"] = [disposition] if disposition else []
                self.assertTrue(any("non-default required features" in e for e in coverage.validate(mapping, self.discovered)))
        self.assertEqual({"default", "hyperion-cpu", "batch-worker"}, coverage.default_features(self.discovered, "hyperion"))

    def test_workflow_comments_flow_values_and_indentation(self):
        text = 'name: proof\njobs: # comment\n    build: # comment\n      runs-on: ubuntu-latest\n    "test": {runs-on: ubuntu-latest}\n'
        self.assertEqual(["build", "test"], coverage.workflow_jobs(text))
        for text in ('jobs: {build: {runs-on: ubuntu-latest}}', 'jobs:\n  <<: *other\n', 'jobs:\n  build:\n  build:\n', 'jobs:\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                coverage.workflow_jobs(text)

    def test_unassigned_and_stale_items_fail(self):
        key = next(iter(self.discovered))
        self.mapping["items"][key]["lanes"] = []
        self.mapping["items"][key]["dispositions"] = []
        self.assertTrue(any("unassigned item" in e for e in coverage.validate(self.mapping, self.discovered)))
        del self.discovered[key]
        self.assertTrue(any("stale assignment" in e for e in coverage.validate(self.mapping, self.discovered)))

    def test_incomplete_disposition_and_lane_fail(self):
        next(iter(self.mapping["dispositions"].values()))["dependencies"] = []
        next(iter(self.mapping["lanes"].values())).pop("requires_nonzero_tests")
        self.assertGreaterEqual(len(coverage.validate(self.mapping, self.discovered)), 2)

    def test_every_lane_declares_feature_selection(self):
        self.mapping["lanes"]["rust-check"].pop("feature_selection")
        self.assertTrue(any("feature selection" in error for error in coverage.validate(self.mapping, self.discovered)))

    def test_collection_failure_preserves_exit_without_static_fallback(self):
        for phase, code in (("sync", 23), ("run", 2)):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                (root / "deploy/ci").mkdir(parents=True)
                (root / "python").mkdir()
                (root / "python/uv.lock").touch()
                shutil.copyfile(ROOT / "deploy/ci/test-inventory.sh", root / "deploy/ci/test-inventory.sh")
                uv = root / "uv"
                uv.write_text(f'#!/bin/sh\nif [ "$1" = {phase} ]; then echo collection-fixture >&2; exit {code}; fi\n')
                uv.chmod(0o755)
                env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"])
                result = subprocess.run(["bash", "deploy/ci/test-inventory.sh", "--no-rust"], cwd=root, env=env, capture_output=True, text=True)
                self.assertEqual(code, result.returncode)
                self.assertIn("collection-fixture", result.stderr)
                self.assertNotIn("static", result.stdout)
                self.assertFalse((root / "docs/parity/test-inventory.json").exists())


if __name__ == "__main__":
    unittest.main()
