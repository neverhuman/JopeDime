"""Public synthetic conformance; not live teacher runtime qualification."""

import concurrent.futures
import datetime as dt
import tempfile
import unittest
from pathlib import Path

import s1_budget as s1

NOW = dt.datetime(2026, 10, 5, 18, tzinfo=dt.timezone.utc)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[4] / "target" / "s1-tests"
        root.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "attempts.sqlite"
        self.plan = {
            "contexts": [f"{i:064x}" for i in range(s1.CONTEXTS)],
            "engine_sha256": "a" * 64,
            "worker_sha256": "b" * 64,
            "input_manifest_sha256": "c" * 64,
        }
        self.ledger = s1.Ledger(self.path, self.plan)

    def reserve(self, identity=1, ordinal=0):
        return self.ledger.reserve(
            f"{identity:064x}", self.plan["contexts"][0], ordinal, "d" * 64, now=NOW
        )

    def test_fixed_equal_ignores_outer_test_for_every_stage(self):
        views = [
            ("a", {"test_rmse": 0, "ratio": 0, "beats_with_margin": True}),
            (
                "b",
                {"test_rmse": float("nan"), "ratio": 1000, "beats_with_margin": False},
            ),
        ]
        for stage in range(1, 19):
            self.assertEqual(s1.contexts_for_stage(stage, views), ["a", "b"])
            self.assertEqual(
                s1.contexts_for_stage(stage, [(v, object()) for v, _ in views]),
                ["a", "b"],
            )
        for stage in (0, 19, True, 1.5):
            with self.assertRaises(ValueError):
                s1.contexts_for_stage(stage, views)

    def test_exact_budget_and_phase_boundaries(self):
        self.assertEqual(s1.TOTAL, 6081504)
        for identity, ordinal, phase, generation in (
            (1, 95, "initialization", 0),
            (2, 96, "evolution", 1),
            (3, 1247, "evolution", 18),
        ):
            result = self.reserve(identity, ordinal)
            self.assertEqual(
                (result["phase"], result["generation"]), (phase, generation)
            )
            self.assertFalse(result["execution_admitted"])
        for ordinal in (-1, 1248, True):
            with self.assertRaises(ValueError):
                self.reserve(10, ordinal)

    def test_duplicate_delivery_never_dispatches_twice(self):
        self.assertTrue(self.reserve()["new_reservation"])
        self.assertFalse(self.reserve()["new_reservation"])
        attempt = f"{1:064x}"
        self.ledger.claim_dispatch(attempt, now=NOW)
        restarted = s1.Ledger(self.path, self.plan)
        with self.assertRaises(ValueError):
            restarted.claim_dispatch(attempt, now=NOW)
        self.assertEqual(restarted.status()["charged_attempts"], 1)

    def test_failure_retry_charged_inside_ceiling(self):
        self.reserve()
        self.ledger.finish(f"{1:064x}", "failed")
        with self.assertRaises(ValueError):
            self.reserve()
        with self.assertRaises(ValueError):
            self.reserve(2, 0)
        self.reserve(2, 1)
        self.ledger.finish(f"{2:064x}", "unknown")
        status = s1.Ledger(self.path, self.plan).status()
        self.assertEqual(status["charged_attempts"], 2)
        self.assertEqual(status["counts"]["initialization"]["failed"], 1)
        self.assertEqual(status["counts"]["initialization"]["unknown"], 1)

    def test_concurrent_same_slot_has_one_winner(self):
        def reserve(identity):
            try:
                s1.Ledger(self.path, self.plan).reserve(
                    f"{identity:064x}", self.plan["contexts"][0], 0, "d" * 64, now=NOW
                )
                return True
            except ValueError:
                return False

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            self.assertEqual(sorted(pool.map(reserve, (1, 2))), [False, True])

    def test_changed_plan_or_reservation_refused(self):
        self.reserve()
        with self.assertRaises(ValueError):
            self.reserve(1, 1)
        for field in ("engine_sha256", "worker_sha256", "input_manifest_sha256"):
            with self.assertRaises(ValueError):
                s1.Ledger(self.path, self.plan | {field: "e" * 64})
        with self.assertRaises(ValueError):
            s1.Ledger(self.path, self.plan | {"contexts": self.plan["contexts"][:-1]})

    def test_original_deadline_and_terminal_identity(self):
        for now in (
            s1.DEADLINE,
            s1.DEADLINE - dt.timedelta(seconds=1),
            NOW.replace(tzinfo=None),
        ):
            with self.assertRaises(ValueError):
                self.ledger.reserve(
                    "a" * 64, self.plan["contexts"][0], 0, "d" * 64, now=now
                )
        self.reserve()
        self.ledger.finish(f"{1:064x}", "cancelled")
        self.ledger.finish(f"{1:064x}", "cancelled")
        with self.assertRaises(ValueError):
            self.ledger.finish(f"{1:064x}", "complete")
        with self.assertRaises(ValueError):
            self.ledger.claim_dispatch(f"{1:064x}", now=NOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
