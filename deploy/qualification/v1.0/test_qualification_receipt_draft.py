#!/usr/bin/env python3
"""Structural controls for the delivered REL-07 receipt and its raw waiver."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import jsonschema

HERE = Path(__file__).resolve().parent
GATES = "M1 M2 M3 M4 M5 S1 S2 S3 S4 R1 R2 R3 R4 A1 A2 A3 A4 A5 E1 E2 E3 U1 U2 T1 T2 T3 D1 D2".split()
SOURCE = "ccaf14b210b3b55a350218e60f6b49ef76524cf3"


class QualificationReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads((HERE / "qualification_receipt.json").read_text())
        cls.schema = json.loads(
            (HERE / "qualification_receipt.final.schema.json").read_text()
        )
        jsonschema.Draft202012Validator.check_schema(cls.schema)
        cls.validator = jsonschema.Draft202012Validator(
            cls.schema, format_checker=jsonschema.FormatChecker()
        )

    def assert_valid(self, receipt):
        self.validator.validate(receipt)
        self.assertEqual([row["id"] for row in receipt["gates"]], GATES)
        self.assertEqual(
            [row["id"] for row in receipt["provenance"]["gate_results"]], GATES
        )
        self.assertEqual(
            [
                row["id"]
                for row in receipt["provenance"]["controller_execution"]["checks"]
            ],
            GATES,
        )
        self.assertEqual(
            {a["id"] for a in receipt["candidate"]["artifacts"]},
            {"full-linux-x86_64", "full-darwin-arm64", "fused-cpu-linux-x86_64"},
        )

    def test_schema_and_maintained_assembler_contract_are_preserved(self):
        self.assert_valid(self.receipt)
        minimal_path = HERE / "release-gates-receipt.json"
        minimal = json.loads(minimal_path.read_text())
        self.assertEqual({key: self.receipt[key] for key in minimal}, minimal)
        assembly = self.receipt["provenance"]["assembly"]
        self.assertEqual(
            hashlib.sha256(minimal_path.read_bytes()).hexdigest(),
            assembly["output"]["sha256"],
        )
        self.assertEqual(
            json.loads(assembly["stdout"]), {"gates": 28, "all_pass": False}
        )
        self.assertEqual(sum(r["status"] == "PASS" for r in minimal["gates"]), 27)
        self.assertEqual(
            [r["id"] for r in minimal["gates"] if r["status"] == "FAIL"], ["M3"]
        )

    def test_all_results_and_commands_belong_to_one_completed_run(self):
        provenance = self.receipt["provenance"]
        execution = provenance["controller_execution"]
        run = provenance["controller_output"]
        self.assertFalse(execution["interrupted"])
        self.assertTrue(execution["finished_utc"])
        self.assertEqual(execution["source_sha"], SOURCE)
        self.assertEqual(
            execution["candidate_inputs_sha256"],
            provenance["core_files"]["inputs"]["sha256"],
        )
        self.assertEqual(
            execution["manifest_sha256"],
            provenance["core_files"]["command_manifest"]["sha256"],
        )
        self.assertEqual(
            set(provenance["provider_inputs"]["gates"]), {"A5", "D1", "M2", "M3", "M5"}
        )
        self.assertEqual(provenance["previous_attempts_promoted"], 0)
        for raw, command in zip(
            provenance["gate_results"], execution["checks"], strict=True
        ):
            with self.subTest(gate=raw["id"]):
                self.assertEqual(
                    raw["result"]["path"], f"{run}/{raw['id']}/result.json"
                )
                self.assertEqual(
                    raw["command_log"]["path"], f"{run}/{raw['id']}/command.log"
                )
                self.assertEqual(raw["command_log"]["sha256"], command["log_sha256"])
                self.assertFalse(command["timed_out"])
                self.assertEqual(command["exit_code"], 1 if raw["id"] == "M3" else 0)
                counts = raw["counts"]
                self.assertEqual(counts["collected"], counts["executed"])
                self.assertGreater(counts["executed"], 0)
                if raw["id"] != "M3":
                    self.assertEqual(raw["result"]["sha256"], command["result_sha256"])
                    self.assertEqual(counts, command["counts"])
                    self.assertEqual(counts["passed"], counts["executed"])
                    self.assertEqual(
                        counts["failed"] + counts["skipped"] + counts["excluded"], 0
                    )
        argv = provenance["assembly"]["argv"]
        supplied = [
            argv[i + 1] for i, arg in enumerate(argv) if arg == "--gate-receipt"
        ]
        self.assertEqual(supplied, [f"{run}/{gate}/result.json" for gate in GATES])

    def test_raw_m3_waiver_default_off_and_pending_owner_approval_are_distinct(self):
        m3 = next(row for row in self.receipt["gates"] if row["id"] == "M3")
        self.assertEqual(m3["status"], "FAIL")
        self.assertEqual(
            m3["counts"],
            {
                "collected": 4,
                "executed": 4,
                "passed": 3,
                "failed": 1,
                "skipped": 0,
                "excluded": 0,
            },
        )
        waiver = self.receipt["waivers"][0]
        self.assertEqual(waiver["raw_status"], "FAIL")
        self.assertEqual(waiver["disposition"], "WAIVED_V1_0_0")
        self.assertEqual(waiver["expires_at"], "v1.0.1")
        self.assertEqual(
            [v["mode"] for v in waiver["guard_views"]], ["all", False, True]
        )
        self.assertEqual(self.receipt["q1"]["guidance_default"], "OFF")
        self.assertEqual(self.receipt["q1"]["owner_decision"], "OWNER-R21")
        self.assertNotIn("Q1", GATES)
        qualification = self.receipt["qualification"]
        self.assertFalse(qualification["raw_all_pass"])
        self.assertFalse(qualification["release_qualified"])
        self.assertIsNone(qualification["owner_release_approval"])
        self.assertIsNone(qualification["aggregate_signature"])

    def test_incomplete_forged_or_out_of_scope_receipts_are_rejected(self):
        def gate(receipt, name):
            return next(r for r in receipt["gates"] if r["id"] == name)

        mutations = [
            ("M3 relabelled", lambda r: gate(r, "M3").update(status="PASS")),
            ("other gate failed", lambda r: gate(r, "D1").update(status="FAIL")),
            ("missing gate", lambda r: r["gates"].pop()),
            (
                "duplicate gate",
                lambda r: r["gates"].__setitem__(17, copy.deepcopy(r["gates"][16])),
            ),
            (
                "wrong source",
                lambda r: r["candidate"].update(runtime_source_sha="0" * 40),
            ),
            (
                "wrong D2 carrier",
                lambda r: gate(r, "D2").update(artifact_sha256="0" * 64),
            ),
            ("future release", lambda r: r.update(release_version="v1.0.1")),
            (
                "head guard exceeded",
                lambda r: r["waivers"][0]["guard_views"][0]["deltas_pp"].update(
                    edit_replace_head=3.01
                ),
            ),
            (
                "other operator guard exceeded",
                lambda r: r["waivers"][0]["guard_views"][0]["deltas_pp"].update(
                    edit_delete_stage=2.01
                ),
            ),
            (
                "interrupted",
                lambda r: r["provenance"]["controller_execution"].update(
                    interrupted=True
                ),
            ),
            (
                "borrowed pass",
                lambda r: r["provenance"].update(previous_attempts_promoted=1),
            ),
            (
                "forged approval",
                lambda r: r["qualification"].update(
                    owner_release_approval={"approved": True}
                ),
            ),
            (
                "forged release qualification",
                lambda r: r["qualification"].update(release_qualified=True),
            ),
        ]
        for name, mutate in mutations:
            with self.subTest(control=name):
                changed = copy.deepcopy(self.receipt)
                mutate(changed)
                with self.assertRaises((jsonschema.ValidationError, AssertionError)):
                    self.assert_valid(changed)

    def test_release_notes_keep_current_evidence_and_limits(self):
        notes = (HERE / "release_notes.md").read_text()
        for required in (
            "27 PASS",
            "raw **FAIL**",
            "WAIVED_V1_0_0",
            "v1.0.1",
            "±2.0 pp",
            "A5 **PASS**",
            "guidance default is **OFF**",
            "OWNER-R21",
            "OWNER RELEASE APPROVAL PENDING",
            "Q1 outcome",
            "internal-restricted",
            "58/1,024",
        ):
            with self.subTest(required=required):
                self.assertIn(required, notes)


if __name__ == "__main__":
    unittest.main()
