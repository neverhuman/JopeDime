#!/usr/bin/env python3
"""Static controls for the non-signable REL-07 evidence draft."""

import json
from pathlib import Path
import unittest

import jsonschema


HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "qualification_receipt.json"
SCHEMA = HERE / "qualification_receipt.draft.schema.json"
RELEASE_NOTES = HERE / "release_notes.md"
GATE_IDS = {
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "S1",
    "S2",
    "S3",
    "S4",
    "R1",
    "R2",
    "R3",
    "R4",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "E1",
    "E2",
    "E3",
    "U1",
    "U2",
    "T1",
    "T2",
    "T3",
    "D1",
    "D2",
}


class QualificationReceiptDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(cls.schema)

    def test_draft_is_schema_complete_and_not_signable(self):
        jsonschema.Draft202012Validator(self.schema).validate(self.receipt)
        self.assertFalse(self.receipt["qualification"]["release_qualified"])
        self.assertIsNone(self.receipt["qualification"]["owner_release_approval"])
        self.assertIsNone(self.receipt["qualification"]["aggregate_signature"])
        self.assertEqual(
            self.receipt["current_evidence"]["final_controller_rows_promoted"], 0
        )

    def test_all_final_controller_rows_are_not_run_without_borrowed_results(self):
        observations = self.receipt["current_evidence"]["observations"]
        by_gate = {row["gate"]: row for row in observations}
        self.assertEqual(set(by_gate), GATE_IDS)
        self.assertEqual(len(observations), len(GATE_IDS))
        for row in observations:
            with self.subTest(gate=row["gate"]):
                self.assertEqual(row["observed_status"], "NOT_RUN")
                self.assertEqual(row["final_controller_status"], "NOT_RUN")
                self.assertEqual(row["evidence"], [])
                self.assertEqual(row["qualification_use"], "NO_RESULT_PROMOTED")
        self.assertEqual(self.receipt["blockers"]["A5"]["status"], "PENDING")
        self.assertEqual(self.receipt["current_evidence"]["final_controller_rows_promoted"], 0)

    def test_m3_raw_failure_and_waiver_are_distinct_and_version_limited(self):
        m3 = self.receipt["owner_decisions"]["M3"]
        self.assertEqual(m3["raw_status"], "FAIL")
        self.assertEqual(m3["disposition"], "WAIVED_V1_0_0")
        self.assertEqual(m3["ruling_id"], "M3-WAIVER-V1.0.0-T1955")
        self.assertEqual(m3["expires_at"], "v1.0.1")
        self.assertEqual(m3["future_tolerance_pp"], 2.0)
        self.assertFalse(m3["future_tolerance_changed"])

    def test_release_notes_keep_open_items_and_nonqualification_clear(self):
        notes = RELEASE_NOTES.read_text(encoding="utf-8")
        for required in (
            "DRAFT / NOT RELEASED / NOT QUALIFIED",
            "25 PASS, 2 FAIL, and 1 BLOCKED",
            "standalone M2 provider",
            "raw operator-mix verdict remains **FAIL**",
            "WAIVED_V1_0_0",
            "A5 is **PENDING**",
            "owner-recorded",
            "guidance default decision also remain pending",
        ):
            with self.subTest(required=required):
                self.assertIn(required, notes)


if __name__ == "__main__":
    unittest.main()
