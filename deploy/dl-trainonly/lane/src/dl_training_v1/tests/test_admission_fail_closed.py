"""Admission must fail closed on an unknown or undecided P3 holdout assignment (owner ENG:watcher 07:35Z).

The S1 catalog join fed contexts whose P3 role was pending an owner decision with assignment='no_family'; Admission.check let
every string other than sealed_excluded/heldout through. These fixtures reproduce that fail-open (they fail on the previous
admission.py) and pin that every decided role still behaves exactly as before. Synthetic denylists only; no private input.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dltrain.admission import Admission, AdmissionRefused

PURPOSES = ("train", "selection", "teacher")
UNKNOWN = ("no_family", "roleless_bridge", "ROLELESS_BRIDGE", "unknown", "", "TRAIN_FIT", "train", "pending")


class AdmissionFailClosed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        doc = {"contexts": [{"view_id": "denied", "assignment": "heldout"}], "object_keys": ["k/denied"],
               "assignment_by_view": {"v-fit": "train_fit", "v-nonpool": "train_nonpool", "v-dev": "dev",
                                      "v-held": "heldout", "v-sealed": "sealed_excluded", "v-nofam": "no_family"}}
        self.path = Path(self.tmp.name) / "deny.json"
        self.path.write_bytes(json.dumps(doc).encode())
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def adm(self, purpose):
        a = Admission(self.path, self.sha, purpose=purpose)
        self.assertTrue(a.ok)
        return a

    def test_unknown_explicit_assignment_refused_for_every_purpose(self):
        # MUST-FAIL on the previous admission.py: these returned the string instead of refusing.
        for purpose in PURPOSES:
            for value in UNKNOWN:
                with self.subTest(purpose=purpose, value=value):
                    with self.assertRaises(AdmissionRefused):
                        self.adm(purpose).check("ctx", assignment=value)

    def test_unknown_assignment_from_the_denylist_refused(self):
        for purpose in PURPOSES:
            with self.subTest(purpose=purpose), self.assertRaises(AdmissionRefused):
                self.adm(purpose).check("v-nofam")

    def test_missing_assignment_still_refused(self):
        for purpose in PURPOSES:
            with self.subTest(purpose=purpose), self.assertRaises(AdmissionRefused):
                self.adm(purpose).check("not-in-denylist")

    def test_control_decided_roles_unchanged(self):
        # CONTROL: every decided role behaves exactly as before the fix.
        for purpose in PURPOSES:
            a = self.adm(purpose)
            self.assertEqual(a.check("v-fit"), "train_fit")
            self.assertEqual(a.check("v-nonpool"), "train_nonpool")
            self.assertEqual(a.check("ctx", assignment="train_fit"), "train_fit")
            self.assertEqual(a.check("ctx", assignment="train_nonpool"), "train_nonpool")
            for value in ("heldout", "sealed_excluded"):
                with self.subTest(purpose=purpose, value=value), self.assertRaises(AdmissionRefused):
                    a.check("ctx", assignment=value)
            with self.assertRaises(AdmissionRefused):
                a.check("denied")
            with self.assertRaises(AdmissionRefused):
                a.check("v-fit", key="k/denied")
            if purpose == "train":
                with self.assertRaises(AdmissionRefused):
                    a.check("v-dev")
            else:
                self.assertEqual(a.check("v-dev"), "dev")
                self.assertEqual(a.check("ctx", assignment="dev"), "dev")

    def test_wrong_denylist_hash_still_fails_closed(self):
        a = Admission(self.path, "0" * 64, purpose="selection")
        self.assertFalse(a.ok)
        with self.assertRaises(AdmissionRefused):
            a.check("v-fit")


if __name__ == "__main__":
    unittest.main()
