"""Budget contract tests for the train-only CPU lane."""

import concurrent.futures
import datetime as dt
import sqlite3
import tempfile
import unittest
from pathlib import Path

import gpu_gate as gate
import outputs as out

NOW = dt.datetime(2026, 10, 5, 18, tzinfo=dt.timezone.utc)

class Scratch(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[2] / "target" / "dl-trainonly-tests"
        target.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)


class BudgetTests(Scratch):
    def setUp(self):
        super().setUp()
        self.path = self.root / "budget.sqlite"
        self.budget = gate.Budget(self.path, initialize=True)

    def job(self, identity="a", host="xbabe2", hours=1):
        return {
            "job_id": identity,
            "host": host,
            "gpu_seconds": hours * 3600,
            "output_ids": [out.digest(identity.encode())],
        }

    def reserve(self, job, **kw):
        return self.budget.reserve(job, now=NOW, **kw)

    def test_all_three_hosts_valid_but_execution_unadmitted(self):
        for host in sorted(gate.HOSTS):
            result = self.reserve(self.job(host, host))
            self.assertTrue(result["budget_reserved"])
            self.assertFalse(result["execution_admitted"])

    def test_wrong_hosts_refused(self):
        for host in (
            "xbabe0",
            "xbabe4",
            "localhost",
            "xbabe3.evil",
            "XBABE2",
            "xbabe2 ",
        ):
            with self.subTest(host=host), self.assertRaises(gate.Refusal):
                self.reserve(self.job(host=host))

    def test_plan_boundary_and_contingency(self):
        self.reserve(self.job(hours=16))
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job("b", hours=1))
        result = self.reserve(self.job("b", hours=8), contingency=True)
        self.assertEqual(result["charged_gpu_seconds"], 24 * 3600)
        self.assertEqual(result["contingency_seconds"], 8 * 3600)
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job("c"), contingency=True)

    def test_single_job_over_hard_cap(self):
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job(hours=25), contingency=True)

    def test_retry_restart_never_refunds(self):
        request = self.job(hours=16)
        self.reserve(request)
        self.budget.finish("a", 1, "cancelled")
        self.budget = gate.Budget(self.path)
        with self.assertRaisesRegex(gate.Refusal, "terminal job"):
            self.reserve(request)
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job("b"))

    def test_terminal_replay_refused_for_every_outcome(self):
        for outcome in ("complete", "failed", "cancelled", "unknown"):
            request = self.job(outcome)
            self.reserve(request)
            self.budget.finish(outcome, 1, outcome)
            with self.assertRaisesRegex(gate.Refusal, "terminal job"):
                self.reserve(request)

    def test_duplicate_delivery_and_dispatch_are_separate(self):
        request = self.job()
        self.assertTrue(self.reserve(request)["new_reservation"])
        repeated = self.reserve(request)
        self.assertFalse(repeated["new_reservation"])
        self.assertEqual(repeated["charged_gpu_seconds"], 3600)
        result = self.budget.claim_dispatch(request, now=NOW)
        self.assertTrue(result["dispatch_claimed"])
        self.assertFalse(result["execution_admitted"])
        with self.assertRaisesRegex(gate.Refusal, "already claimed"):
            gate.Budget(self.path).claim_dispatch(request, now=NOW)

    def test_two_attempts_include_failed_and_unknown_restarts(self):
        original = self.job()
        self.reserve(original)
        self.budget.finish("a", 1, "failed")
        retry = original | {"job_id": "retry"}
        result = gate.Budget(self.path).reserve(retry, now=NOW)
        self.assertEqual(result["charged_gpu_seconds"], 7200)
        self.budget.finish("retry", 0, "unknown")
        with self.assertRaisesRegex(gate.Refusal, "two-attempt"):
            gate.Budget(self.path).reserve(original | {"job_id": "third"}, now=NOW)

    def test_concurrent_output_attempt_cap(self):
        original = self.job()
        self.reserve(original)

        def attempt(identity):
            try:
                gate.Budget(self.path).reserve(original | {"job_id": identity}, now=NOW)
                return True
            except gate.Refusal:
                return False

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            self.assertEqual(sorted(pool.map(attempt, ("b", "c"))), [False, True])

    def test_batched_attempt_reservation_rolls_back(self):
        original = self.job()
        self.reserve(original)
        self.reserve(original | {"job_id": "second"})
        fresh = self.job("fresh")
        with self.assertRaises(gate.Refusal):
            self.reserve(
                fresh | {"output_ids": fresh["output_ids"] + original["output_ids"]}
            )
        with self.budget.connect() as db:
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM attempts WHERE output_id=?",
                    (fresh["output_ids"][0],),
                ).fetchone()[0],
                0,
            )
        self.assertTrue(self.reserve(fresh)["new_reservation"])

    def test_output_identity_and_dispatch_scope(self):
        original = self.job()
        for ids in ([], ["short"], [original["output_ids"][0]] * 2, "not-list", [None]):
            with self.assertRaises(gate.Refusal):
                self.reserve(original | {"output_ids": ids})
        with self.assertRaises(gate.Refusal):
            self.budget.claim_dispatch(original, now=NOW)
        self.reserve(original)
        with self.assertRaises(gate.Refusal):
            self.reserve(original | {"output_ids": ["f" * 64]})
        for request, now in (
            (original | {"host": "xbabe3"}, NOW),
            (original, gate.DEADLINE),
        ):
            with self.assertRaises(gate.Refusal):
                self.budget.claim_dispatch(request, now=now)

    def test_changed_retry_and_terminal_refused(self):
        self.reserve(self.job())
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job(host="xbabe3"))
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job(hours=2))
        self.budget.finish("a", 3, "failed")
        self.budget.finish("a", 3, "failed")
        with self.assertRaises(gate.Refusal):
            self.budget.finish("a", 0, "complete")

    def test_overrun_retained_and_blocks_new_jobs(self):
        self.reserve(self.job())
        self.budget.finish("a", 25 * 3600, "failed")
        with self.assertRaises(gate.Refusal):
            gate.Budget(self.path).reserve(self.job("b"), contingency=True, now=NOW)

    def test_expired_crossing_and_naive_clock(self):
        for now in (
            gate.DEADLINE,
            gate.DEADLINE - dt.timedelta(seconds=1),
            NOW.replace(tzinfo=None),
        ):
            with self.subTest(now=now), self.assertRaises(gate.Refusal):
                self.budget.reserve(self.job(), now=now)

    def test_finish_at_deadline(self):
        result = self.budget.reserve(
            self.job(), now=gate.DEADLINE - dt.timedelta(hours=1)
        )
        self.assertTrue(result["budget_reserved"])

    def test_invalid_seconds_and_id(self):
        for seconds in (0, -1, True, float("nan"), float("inf"), 1.5, "3600"):
            with self.subTest(seconds=seconds), self.assertRaises(gate.Refusal):
                self.reserve(self.job() | {"gpu_seconds": seconds})
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job("../escape"))
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job() | {"deadline": "2027-01-01"})

    def test_concurrent_reservations(self):
        # Separate SQLite connections race for 20 hours in a 16-hour plan.
        def attempt(identity):
            try:
                return gate.Budget(self.path).reserve(
                    self.job(identity, hours=10), now=NOW
                )["budget_reserved"]
            except gate.Refusal:
                return False

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            self.assertEqual(sorted(pool.map(attempt, ("a", "b"))), [False, True])

    def test_missing_ledger_never_resets_charges(self):
        self.reserve(self.job(hours=16))
        saved = self.path.read_bytes()
        self.path.unlink()
        with self.assertRaisesRegex(gate.Refusal, "restore custody"):
            gate.Budget(self.path)
        self.assertFalse(self.path.exists())
        with self.assertRaises(FileExistsError):
            gate.Budget(self.path, initialize=True)
        self.assertFalse(self.path.exists())
        self.path.write_bytes(saved)
        with self.assertRaises(gate.Refusal):
            gate.Budget(self.path).reserve(self.job("b"), now=NOW)

    def test_first_initialization_is_explicit_and_exclusive(self):
        path = self.root / "another.sqlite"
        with self.assertRaisesRegex(gate.Refusal, "restore custody"):
            gate.Budget(path)
        self.assertFalse(path.exists())
        budget = gate.Budget(path, initialize=True)
        budget.reserve(self.job(), now=NOW)
        with self.assertRaises(FileExistsError):
            gate.Budget(path, initialize=True)
        self.assertEqual(
            gate.Budget(path).reserve(self.job(), now=NOW)["charged_gpu_seconds"], 3600
        )

    def test_missing_after_open_never_recreates_database(self):
        self.path.unlink()
        with self.assertRaises(sqlite3.OperationalError):
            self.reserve(self.job())
        self.assertFalse(self.path.exists())

    def test_missing_or_replaced_marker_refuses_resume(self):
        marker = self.path.with_name(self.path.name + ".initialization.json")
        saved = marker.read_bytes()
        marker.unlink()
        with self.assertRaisesRegex(gate.Refusal, "restore custody"):
            gate.Budget(self.path)
        marker.write_text("different plan\n")
        with self.assertRaisesRegex(gate.Refusal, "marker differs"):
            gate.Budget(self.path)
        marker.write_bytes(saved)
        self.assertTrue(
            gate.Budget(self.path).reserve(self.job(), now=NOW)["new_reservation"]
        )

    def test_changed_persisted_policy_refused(self):
        with self.budget.connect() as db:
            db.execute("UPDATE policy SET value='renewed'")
        with self.assertRaises(gate.Refusal):
            gate.Budget(self.path)

    def test_unreserved_finish_and_bad_outcome(self):
        with self.assertRaises(gate.Refusal):
            self.budget.finish("missing", 0, "failed")
        self.reserve(self.job())
        with self.assertRaises(gate.Refusal):
            self.budget.finish("a", 0, "passed")
