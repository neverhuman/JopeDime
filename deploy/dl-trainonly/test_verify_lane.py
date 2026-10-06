#!/usr/bin/env python3
"""Offline positive and substitution/refusal checks for pinned source custody."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import verify_lane as v


class Custody(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dl-source-custody-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        current = b"x = 1\n"
        historical = b"x = 0\n"
        supplement = b"old_test = True\n"
        self.supplement_name = "prior.py." + v.sha(supplement)
        self.history_name = "one.py." + v.sha(historical)
        self.old = {"schema": "jope.dl-lane-handoff/v1", "files": {
            "src/dl_training_v1/one.py": self.record(historical),
            "src/dl_training_v1/prior.py": self.record(supplement)},
            "historical_versions": {self.history_name: self.record(historical)}}
        self.old_sha = self.put_json("predecessor/MANIFEST.json", self.old)
        self.manifest = {"schema": "jope.dl-lane-handoff/v2", "supersedes_sha256": self.old_sha,
            "files": {"src/dl_training_v1/one.py": self.record(current)},
            "historical_versions": {self.history_name: self.record(historical)}}
        self.pin = self.put_json("MANIFEST.json", self.manifest)
        for name, data in (("src/dl_training_v1/one.py", current), ("historical/" + self.history_name, historical),
                           ("predecessor/" + self.supplement_name, supplement)):
            self.put(name, data)

    @staticmethod
    def record(data):
        return {"sha256": v.sha(data), "bytes": len(data)}

    def put(self, name, data):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def put_json(self, name, data):
        body = (json.dumps(data, sort_keys=True) + "\n").encode()
        self.put(name, body)
        return v.sha(body)

    def verify(self):
        return v.verify(self.root, self.pin, self.old_sha, self.supplement_name)

    def test_authorized_exact_closure_reconstructs_predecessor(self):
        report = self.verify()
        self.assertEqual(report["current_files"], 1)
        self.assertEqual(report["historical_versions"], 1)
        self.assertEqual(report["supplemental_versions"], 1)
        self.assertEqual(set(report["predecessor_restoration"]), set(self.old["files"]))

    def test_actual_import_matches_frozen_producer_pins(self):
        report = v.verify(Path(v.__file__).with_name("lane"))
        self.assertEqual(report["handoff_sha256"], v.HANDOFF_SHA)
        self.assertEqual(report["current_files"], 147)
        self.assertEqual(report["python_syntax_files"], 137)
        self.assertEqual(len(report["predecessor_restoration"]), 143)

    def test_same_width_current_edit_refused(self):
        self.put("src/dl_training_v1/one.py", b"x = 2\n")
        with self.assertRaisesRegex(ValueError, "source bytes mismatch"):
            self.verify()

    def test_same_width_history_edit_refused(self):
        self.put("historical/" + self.history_name, b"x = 3\n")
        with self.assertRaisesRegex(ValueError, "source bytes mismatch"):
            self.verify()

    def test_self_sidecar_cannot_reauthorize_changed_source(self):
        self.put("src/dl_training_v1/one.py", b"x = 2\n")
        self.manifest["files"]["src/dl_training_v1/one.py"] = self.record(b"x = 2\n")
        self.put_json("MANIFEST.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "trusted manifest pin mismatch"):
            self.verify()

    def test_missing_source_refused(self):
        (self.root / "src/dl_training_v1/one.py").rename(self.root / "saved.py")
        with self.assertRaisesRegex(ValueError, "missing/escaping"):
            self.verify()

    def test_unlisted_current_file_refused(self):
        self.put("src/dl_training_v1/unlisted.py", b"x = 1\n")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            self.verify()

    def test_symlink_source_refused(self):
        p = self.root / "src/dl_training_v1/one.py"
        p.rename(self.root / "saved.py")
        p.symlink_to(self.root / "saved.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.verify()

    def test_unlisted_directory_symlink_refused(self):
        (self.root / "src/escape").symlink_to(self.root / "historical", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.verify()

    def test_predecessor_manifest_replacement_refused(self):
        changed = copy.deepcopy(self.old)
        changed["schema"] = "other"
        self.put_json("predecessor/MANIFEST.json", changed)
        with self.assertRaisesRegex(ValueError, "trusted manifest pin mismatch"):
            self.verify()

    def test_supplement_replacement_refused(self):
        self.put("predecessor/" + self.supplement_name, b"old_test = False\n")
        with self.assertRaisesRegex(ValueError, "supplemental predecessor bytes mismatch"):
            self.verify()

    def test_unlisted_predecessor_symlink_refused(self):
        (self.root / "predecessor/escape").symlink_to(self.root / "src", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "predecessor symlink"):
            self.verify()

    def test_unsafe_manifest_path_refused_even_for_authenticated_fixture(self):
        self.manifest["files"]["src/dl_training_v1/../../escape.py"] = self.record(b"x = 1\n")
        self.pin = self.put_json("MANIFEST.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "unsafe source path"):
            self.verify()

    def test_duplicate_json_key_refused_even_for_authenticated_fixture(self):
        body = b'{"schema":"jope.dl-lane-handoff/v2","schema":"other"}'
        self.put("MANIFEST.json", body)
        self.pin = v.sha(body)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.verify()

    def test_predecessor_cannot_be_silently_rebased(self):
        self.manifest["supersedes_sha256"] = "0" * 64
        self.pin = self.put_json("MANIFEST.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "predecessor identity mismatch"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
