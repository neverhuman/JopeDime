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
        middle = b"x = 5\n"
        self.supplement_name = "predecessor/prior.py." + v.sha(supplement)
        self.middle_supplement = "predecessor/mid/one.py." + v.sha(middle)
        self.history_name = "one.py." + v.sha(historical)
        self.old = {"schema": "jope.dl-lane-handoff/v1", "files": {
            "src/dl_training_v1/one.py": self.record(historical),
            "src/dl_training_v1/prior.py": self.record(supplement)},
            "historical_versions": {self.history_name: self.record(historical)}}
        self.old_sha = self.put_json("predecessor/MANIFEST.json", self.old)
        self.mid = {"schema": "jope.dl-lane-handoff/v2", "supersedes_sha256": self.old_sha,
            "files": {"src/dl_training_v1/one.py": self.record(middle)},
            "historical_versions": {self.history_name: self.record(historical)}}
        self.mid_sha = self.put_json("predecessor/mid/MANIFEST.json", self.mid)
        self.abandoned_sha = v.sha(b"abandoned staging bytes\n")
        self.manifest = {"schema": "jope.dl-lane-handoff/v2", "supersedes_sha256": self.mid_sha,
            "files": {"src/dl_training_v1/one.py": self.record(current)},
            "historical_versions": {self.history_name: self.record(historical)},
            v.ABANDONED: [{"manifest": "gone/MANIFEST.json", "manifest_sha256": self.abandoned_sha, "reason": "fixture"}]}
        self.pin = self.put_json("MANIFEST.json", self.manifest)
        for name, data in (("src/dl_training_v1/one.py", current), ("historical/" + self.history_name, historical),
                           (self.supplement_name, supplement), (self.middle_supplement, middle)):
            self.put(name, data)
        self.supplements = (self.supplement_name, self.middle_supplement)

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

    def chain(self):
        return (("predecessor/mid/MANIFEST.json", self.mid_sha), ("predecessor/MANIFEST.json", self.old_sha))

    def repin_chain(self):
        self.old_sha = self.put_json("predecessor/MANIFEST.json", self.old)
        self.mid["supersedes_sha256"] = self.old_sha
        self.mid_sha = self.put_json("predecessor/mid/MANIFEST.json", self.mid)
        self.manifest["supersedes_sha256"] = self.mid_sha
        self.pin = self.put_json("MANIFEST.json", self.manifest)

    def verify(self):
        return v.verify(self.root, self.pin, self.chain(), self.supplements)

    def test_authorized_exact_closure_reconstructs_predecessor(self):
        report = self.verify()
        self.assertEqual(report["current_files"], 1)
        self.assertEqual(report["historical_versions"], 1)
        self.assertEqual(report["supplemental_versions"], 2)
        self.assertEqual(report["predecessor_sha256"], [self.mid_sha, self.old_sha])
        restored = report["predecessor_restoration"]
        self.assertEqual(set(restored[self.old_sha]), set(self.old["files"]))
        self.assertEqual(restored[self.mid_sha], {"src/dl_training_v1/one.py": self.middle_supplement})
        self.assertEqual(restored[self.old_sha]["src/dl_training_v1/prior.py"], self.supplement_name)
        self.assertEqual(report["abandoned_unpushed_stagings"], [self.abandoned_sha])

    def test_actual_import_matches_frozen_producer_pins(self):
        report = v.verify(Path(v.__file__).with_name("lane"))
        self.assertEqual(report["handoff_sha256"], v.HANDOFF_SHA)
        self.assertEqual(report["predecessor_sha256"], [expected for _, expected in v.PREDECESSORS])
        self.assertEqual(report["current_files"], 177)
        self.assertEqual(report["historical_versions"], 284)
        self.assertEqual(report["supplemental_versions"], 3)
        self.assertEqual(report["python_syntax_files"], 167)
        self.assertEqual([len(report["predecessor_restoration"][expected]) for _, expected in v.PREDECESSORS],
                         [176, 170, 169, 168, 165, 158, 158, 156, 155, 147, 143])
        self.assertEqual(report["abandoned_unpushed_stagings"],
                         ["0388e6a75c0f54038661c61f35e18e1225f1386a2b40abbb43b2ff25b3381987",
                          "be5e5f600a15d01d83a13c2745d37748c772c0317ee3986153e236fc0fea0d7f",
                          "e1ca4e355a5092bc8a7d97a90352ffd35c5d96f1a902c7a411ddca23c9a57ca5"])

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
        self.put("src/dl_training_v1/__pycache__/unlisted.cpython-312.pyc", b"cache")
        self.assertEqual(self.verify()["current_files"], 1)
        self.put("src/dl_training_v1/unlisted.py", b"x = 1\n")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            self.verify()

    def test_extra_historical_file_refused(self):
        self.put("historical/one.py." + v.sha(b"x = 9\n"), b"x = 9\n")
        with self.assertRaisesRegex(ValueError, "source inventory mismatch"):
            self.verify()

    def test_extra_predecessor_chain_file_refused(self):
        self.put("predecessor/mid/extra.py", b"x = 1\n")
        with self.assertRaisesRegex(ValueError, "predecessor inventory mismatch"):
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

    def test_intermediate_manifest_replacement_refused(self):
        changed = copy.deepcopy(self.mid)
        changed["files"] = {}
        self.put_json("predecessor/mid/MANIFEST.json", changed)
        with self.assertRaisesRegex(ValueError, "trusted manifest pin mismatch"):
            self.verify()

    def test_supplement_replacement_refused(self):
        self.put(self.supplement_name, b"old_test = False\n")
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

    def test_unsafe_supplement_path_refused(self):
        self.supplements = (self.supplement_name, "predecessor/../" + self.middle_supplement)
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

    def test_broken_intermediate_chain_link_refused(self):
        self.mid["supersedes_sha256"] = "0" * 64
        self.mid_sha = self.put_json("predecessor/mid/MANIFEST.json", self.mid)
        self.manifest["supersedes_sha256"] = self.mid_sha
        self.pin = self.put_json("MANIFEST.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "predecessor identity mismatch"):
            self.verify()

    def test_reordered_chain_refused(self):
        with self.assertRaisesRegex(ValueError, "unsupported custody schema"):
            v.verify(self.root, self.pin, tuple(reversed(self.chain())), self.supplements)

    def test_truncated_chain_refused(self):
        with self.assertRaisesRegex(ValueError, "unsupported custody schema"):
            v.verify(self.root, self.pin, self.chain()[:1], self.supplements)
        with self.assertRaisesRegex(ValueError, "predecessor chain missing"):
            v.verify(self.root, self.pin, (), self.supplements)

    def test_chain_must_end_at_original_handoff(self):
        self.old["supersedes_sha256"] = "0" * 64
        self.repin_chain()
        with self.assertRaisesRegex(ValueError, "predecessor chain incomplete"):
            self.verify()

    def test_lost_predecessor_version_refused(self):
        (self.root / self.middle_supplement).unlink()
        self.supplements = (self.supplement_name,)
        with self.assertRaisesRegex(ValueError, "predecessor version lost: src/dl_training_v1/one.py"):
            self.verify()

    def test_lost_predecessor_historical_version_refused(self):
        dropped = b"x = 7\n"
        self.mid["historical_versions"]["one.py." + v.sha(dropped)] = self.record(dropped)
        self.repin_chain()
        with self.assertRaisesRegex(ValueError, "historical predecessor version lost"):
            self.verify()

    def test_abandoned_staging_inside_chain_refused(self):
        self.manifest[v.ABANDONED][0]["manifest_sha256"] = self.mid_sha
        self.pin = self.put_json("MANIFEST.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "abandoned staging inside custody chain"):
            self.verify()

    def test_abandoned_staging_path_inside_chain_refused(self):
        self.manifest[v.ABANDONED][0]["manifest"] = "mid/MANIFEST.json"
        self.pin = self.put_json("MANIFEST.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "abandoned staging inside custody chain"):
            self.verify()

    def test_abandoned_staging_recorded_by_predecessor_refused(self):
        self.mid[v.ABANDONED] = [{"manifest": "MANIFEST.json", "manifest_sha256": v.sha(
            (json.dumps(self.old, sort_keys=True) + "\n").encode())}]
        self.repin_chain()
        with self.assertRaisesRegex(ValueError, "abandoned staging inside custody chain"):
            self.verify()

    def test_malformed_abandoned_staging_record_refused(self):
        for bad in ({"manifest": "gone/MANIFEST.json"}, [{"manifest": "gone/MANIFEST.json", "manifest_sha256": "e1ca4e35"}],
                    ["e1ca4e35"], [{"manifest_sha256": self.abandoned_sha}]):
            self.manifest[v.ABANDONED] = bad
            self.pin = self.put_json("MANIFEST.json", self.manifest)
            with self.assertRaisesRegex(ValueError, "invalid abandoned staging record"):
                self.verify()

    def test_redundant_or_unreferenced_supplement_refused(self):
        name = "predecessor/mid/one.py." + v.sha(b"x = 1\n")
        self.put(name, b"x = 1\n")
        self.supplements += (name,)
        with self.assertRaisesRegex(ValueError, "redundant supplemental predecessor"):
            self.verify()
        stray = "predecessor/mid/one.py." + v.sha(b"x = 8\n")
        (self.root / name).unlink()
        self.put(stray, b"x = 8\n")
        self.supplements = self.supplements[:-1] + (stray,)
        with self.assertRaisesRegex(ValueError, "unreferenced supplemental predecessor"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
