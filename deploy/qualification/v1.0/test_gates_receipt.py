"""Fail-closed tests for the signer_watch release-gate receipt adapter."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import gates_receipt as runner


class GatesReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rel07-gates-receipt-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def gate_rows(self, status: str = "NOT_RUN") -> list[dict[str, str]]:
        return [
            {"id": gate, "status": status, "evidence": f"evidence/{gate}.json"}
            for gate in runner.RELEASE_GATE_IDS
        ]

    def test_release_inventory_is_exactly_28_and_excludes_q1(self) -> None:
        self.assertEqual(len(runner.RELEASE_GATE_IDS), 28)
        self.assertEqual(len(set(runner.RELEASE_GATE_IDS)), 28)
        self.assertNotIn("Q1", runner.RELEASE_GATE_IDS)

    def test_cpu_prestage_projects_all_gates_as_not_run_for_signer(self) -> None:
        path = self.root / "qualification_receipt.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "jope.qualification-cpu-prestage/v1",
                    "source": {"main_sha": "a" * 40},
                    "release_qualified": False,
                    "signed_candidate": None,
                    "release_gates": {
                        gate: {"status": "not-qualified", "reason": "prep only"}
                        for gate in runner.RELEASE_GATE_IDS + ["Q1"]
                    },
                }
            )
        )

        result = runner.from_cpu_prestage(path)

        self.assertEqual(len(result["gates"]), 28)
        self.assertTrue(all(row["status"] == "NOT_RUN" for row in result["gates"]))
        self.assertTrue(
            all(set(row) == {"id", "status", "evidence"} for row in result["gates"])
        )
        self.assertTrue(all("prep only" in row["evidence"] for row in result["gates"]))

    def test_cpu_prestage_cannot_claim_signed_release(self) -> None:
        path = self.root / "qualification_receipt.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "jope.qualification-cpu-prestage/v1",
                    "source": {"main_sha": "a" * 40},
                    "release_qualified": True,
                    "signed_candidate": {"sha256": "b" * 64},
                    "release_gates": {
                        gate: {"status": "PASS"}
                        for gate in runner.RELEASE_GATE_IDS + ["Q1"]
                    },
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "cannot claim signed release"):
            runner.from_cpu_prestage(path)

    def test_pass_results_require_matching_artifact_bytes(self) -> None:
        artifact = self.root / "candidate.bin"
        artifact.write_bytes(b"signed candidate bytes")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        results = self.root / "results.json"
        results.write_text(
            json.dumps(
                {
                    "schema": "jope.rel07.gate-results/v1",
                    "signed_bytes_sha256": digest,
                    "gates": self.gate_rows("PASS"),
                }
            )
        )

        receipt = runner.from_gate_results(results, artifact)

        self.assertEqual(receipt["signed_bytes_sha256"], digest)
        self.assertEqual(
            [row["id"] for row in receipt["gates"]], runner.RELEASE_GATE_IDS
        )
        self.assertTrue(all(row["status"] == "PASS" for row in receipt["gates"]))

    def test_pass_results_without_artifact_are_refused(self) -> None:
        results = self.root / "results.json"
        results.write_text(
            json.dumps(
                {
                    "schema": "jope.rel07.gate-results/v1",
                    "gates": self.gate_rows("PASS"),
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "require --artifact"):
            runner.from_gate_results(results, None)

    def test_wrong_artifact_hash_is_refused(self) -> None:
        artifact = self.root / "candidate.bin"
        artifact.write_bytes(b"candidate")
        results = self.root / "results.json"
        results.write_text(
            json.dumps(
                {
                    "schema": "jope.rel07.gate-results/v1",
                    "signed_bytes_sha256": "0" * 64,
                    "gates": self.gate_rows("PASS"),
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "not bound"):
            runner.from_gate_results(results, artifact)

    def test_missing_extra_duplicate_or_malformed_gate_fails_closed(self) -> None:
        cases = [
            self.gate_rows()[:-1],
            self.gate_rows() + [{"id": "Q1", "status": "PASS", "evidence": "x"}],
            self.gate_rows()[:-1] + [self.gate_rows()[0]],
            self.gate_rows()[:-1] + [{"id": "D2", "status": "MAYBE", "evidence": "x"}],
        ]
        for gates in cases:
            with self.subTest(gates=len(gates)):
                source = self.root / "results.json"
                source.write_text(
                    json.dumps({"schema": "jope.rel07.gate-results/v1", "gates": gates})
                )
                with self.assertRaises(ValueError):
                    runner.from_gate_results(source, None)


if __name__ == "__main__":
    unittest.main()
