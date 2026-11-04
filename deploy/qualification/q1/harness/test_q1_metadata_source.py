"""Protocol/statistical fixtures only; no learned model or sealed payload is executed."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import q1
import q1_bindings


from _q1_test_support import fixture_receipts, qualification_plan

class MetadataSourceTests(unittest.TestCase):
    def plan(self):
        plan = qualification_plan()
        plan["source_commit"] = json.loads(q1_bindings.EVALUATOR_PIN.read_text())[
            "source_commit"
        ]
        plan["metadata_source_commit"] = "b" * 40
        return plan

    def test_explicit_metadata_and_actual_compiled_source_are_separate(self):
        plan = self.plan()
        q1.validate_plan(plan)
        self.assertNotEqual(plan["source_commit"], plan["metadata_source_commit"])

    def test_wrong_compiled_source_and_malformed_metadata_are_refused(self):
        plan = self.plan()
        plan["source_commit"] = "a" * 40
        with self.assertRaisesRegex(ValueError, "compiled evaluator source"):
            q1.validate_plan(plan)
        for value in [None, False, "x" * 40, "b" * 39]:
            plan = self.plan()
            plan["metadata_source_commit"] = value
            with self.assertRaisesRegex(ValueError, "metadata source commit"):
                q1.validate_plan(plan)

    def test_omitting_metadata_field_cannot_hide_wrong_compiled_source(self):
        plan = self.plan()
        plan.pop("metadata_source_commit")
        plan["source_commit"] = "a" * 40
        with self.assertRaisesRegex(ValueError, "compiled evaluator source"):
            q1.validate_plan(plan)

    def test_metadata_source_is_covered_by_owner_approval_payload(self):
        plan = self.plan()
        before = q1.approval_payload(plan)
        plan["metadata_source_commit"] = "c" * 40
        self.assertNotEqual(q1.approval_payload(plan), before)


if __name__ == "__main__":
    unittest.main()
