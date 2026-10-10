"""Lifecycle proposals, the byte/PUT guard, and the ledger export. No S3 calls."""
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s3_lifecycle import (
    BYTE_CAP,
    BudgetLedger,
    artifact_bytes,
    classify_object,
    export_ledger,
    proposal_from_rows,
    refuse_apply,
)


EVAL = "626d93a5cd11830e1b7cd43bf46f3fbde42c06273e750faa6d64b69a0c037526"
OTHER = "a" * 64
SOURCE = "b" * 64


def row(key, size, **extra):
    body = {
        "key": key,
        "size": size,
        "storage_class": "STANDARD",
        "legal_hold_checked": True,
        "legal_hold": "OFF",
    }
    body.update(extra)
    return body


class LifecycleTests(unittest.TestCase):
    def test_protected_and_unchecked_are_retained(self):
        held = classify_object(
            row("experiments/gp-policy-dl/v2-trainonly/gp-hwm/current.json", 10),
            current_evaluator=EVAL,
        )
        self.assertEqual(held["action"], "untouched")
        judge = classify_object(
            row("bucket/judge/family.json", 10),
            current_evaluator=EVAL,
        )
        self.assertEqual(judge["action"], "untouched")
        locked = classify_object(
            row("experiments/x/score_records", 200000, legal_hold="ON"),
            current_evaluator=EVAL,
        )
        self.assertEqual(locked["action"], "untouched")
        unknown = classify_object(
            {"key": "experiments/x/score_records", "size": 200000},
            current_evaluator=EVAL,
        )
        self.assertEqual(unknown["reason"], "legal_hold_unmeasured")

    def test_stale_score_and_small_object_and_executable(self):
        stale_small = classify_object(
            row(
                "experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/objects/ab/score_records",
                2357,
                evaluator_sha256=OTHER,
            ),
            current_evaluator=EVAL,
        )
        self.assertEqual(stale_small["reason"], "stale_score_below_128kib_no_individual_transition")
        stale_large = classify_object(
            row(
                "experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/objects/ab/evaluation",
                200000,
                evaluator_sha256=OTHER,
            ),
            current_evaluator=EVAL,
        )
        self.assertEqual(stale_large["action"], "propose_glacier_ir")
        fresh = classify_object(
            row(
                "experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/objects/ab/evaluation",
                200000,
                evaluator_sha256=EVAL,
            ),
            current_evaluator=EVAL,
        )
        self.assertEqual(fresh["action"], "retain")
        pending = classify_object(
            row(
                "experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/objects/aa/rust_executable",
                4448192,
            ),
            current_evaluator=EVAL,
            superseded_source_shas={SOURCE},
        )
        self.assertEqual(pending["reason"], "executable_not_yet_superseded")
        ready = classify_object(
            row(
                "experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/objects/aa/rust_executable",
                4448192,
                rust_source_sha256=SOURCE,
            ),
            current_evaluator=EVAL,
            superseded_source_shas={SOURCE},
        )
        self.assertEqual(ready["action"], "propose_expire")

    def test_proposal_is_not_applied_and_mixed_prefix_has_no_rule(self):
        with self.assertRaises(RuntimeError):
            refuse_apply(True)
        refuse_apply(False)
        rows = [
            row(
                "experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/objects/aa/rust_executable",
                4448192,
                rust_source_sha256=SOURCE,
            ),
            row(
                "experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/objects/aa/score_records",
                2357,
                evaluator_sha256=EVAL,
            ),
            row("experiments/gp-policy-dl/v2-trainonly/gp-hwm/current.json", 40),
        ]
        proposal = proposal_from_rows(
            rows,
            current_evaluator=EVAL,
            superseded_source_shas={SOURCE},
            sample_note="unit fixture",
        )
        self.assertFalse(proposal["applied"])
        self.assertEqual(proposal["proposed_rules"], [])
        self.assertEqual(proposal["totals"]["untouched"]["count"], 1)
        self.assertEqual(proposal["totals"]["propose_expire"]["bytes"], 4448192)
        record = {
            "artifacts": {
                "rust_executable": {"bytes": 4448192},
                "input": {"bytes": 40017},
            }
        }
        self.assertEqual(artifact_bytes(record)["rust_executable"], 4448192)

    def test_budget_guard_chain_alarm_and_refusal(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            ledger = BudgetLedger(Path(tmp) / "ledger.jsonl", byte_cap=1000, put_cap=10)
            moment = datetime(2026, 10, 10, tzinfo=timezone.utc)
            first = ledger.admit(700, 1, kind="test", now=moment)
            self.assertEqual(first["decision"], "admit")
            self.assertFalse(first["alarm"])
            second = ledger.admit(200, 1, kind="test", now=moment)
            self.assertEqual(second["decision"], "admit")
            self.assertTrue(second["alarm"])
            self.assertEqual(second["prev"], first["sha256"])
            refused = ledger.admit(200, 1, kind="test", now=moment)
            self.assertEqual(refused["decision"], "refuse")
            reloaded = BudgetLedger(ledger.path, byte_cap=1000, put_cap=10)
            self.assertEqual(len(reloaded.events), 2)
            tampered = ledger.path.read_text().replace(first["sha256"], "0" * 64)
            ledger.path.write_text(tampered)
            with self.assertRaises(ValueError):
                BudgetLedger(ledger.path, byte_cap=1000, put_cap=10)

    def test_parquet_ledger_when_duckdb_is_present(self):
        try:
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest("duckdb not installed")
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            rows = [row("experiments/x/record.json", 10), row("experiments/x/rust_executable", 20)]
            report = export_ledger(rows, root / "ledger.parquet", root / "ledger.duckdb")
            self.assertEqual(report["rows"], 2)
            self.assertGreater(report["parquet_bytes"], 0)
            again = export_ledger(rows[:1], root / "ledger.parquet", root / "ledger.duckdb")
            self.assertEqual(again["rows"], 1)

    def test_second_client_reloads_before_admit(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / "budget.jsonl"
            first = BudgetLedger(path, byte_cap=100, put_cap=10)
            second = BudgetLedger(path, byte_cap=100, put_cap=10)
            moment = datetime(2026, 10, 10, 14, tzinfo=timezone.utc)
            self.assertEqual(first.admit(60, 1, kind="fixture", now=moment)["decision"], "admit")
            refused = second.admit(60, 1, kind="fixture", now=moment)
            self.assertEqual(refused["decision"], "refuse")
            self.assertEqual(refused["projected_bytes"], 120)
            reloaded = BudgetLedger(path, byte_cap=100, put_cap=10)
            self.assertEqual(len(reloaded.events), 1)
            self.assertEqual(reloaded.events[0]["prev"], "0" * 64)

    def test_sample_lists_object_candidates_without_a_prefix_rule(self):
        key = "research/objects/abc/rust_executable"
        proposal = proposal_from_rows(
            [row(key, 200000, rust_source_sha256="a" * 64)],
            current_evaluator="b" * 64,
            superseded_source_shas={"a" * 64},
            sample_note="bounded sample, not a census",
            census=True,
        )
        self.assertEqual(proposal["proposed_rules"], [])
        self.assertFalse(proposal["listing_census"])
        self.assertEqual(proposal["object_candidates"], [
            {"key": key, "action": "propose_expire", "size": 200000,
             "reason": "superseded_by_reproducible_executable", "scope": "object"},
        ])
        versioned = proposal_from_rows(
            [row(key, 200000, rust_source_sha256="a" * 64, version_id="v1")],
            current_evaluator="b" * 64,
            superseded_source_shas={"a" * 64},
            sample_note="complete listing",
            census=True,
        )
        self.assertEqual(versioned["proposed_rules"], [])
        self.assertTrue(versioned["listing_census"])
        self.assertTrue(versioned["listing_versioned"])
        self.assertEqual(versioned["object_candidates"][0]["key"], key)
        self.assertIn("prefix", versioned["prefix_rules_withheld"])

    def test_default_caps(self):
        self.assertEqual(BYTE_CAP, 200 * 10**9)


if __name__ == "__main__":
    unittest.main()
