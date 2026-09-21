import json
import tempfile
import time
import unittest
from pathlib import Path

import funding_recovery as f


class FundingRecoveryTests(unittest.TestCase):
    def snapshot(self, root, name, **updates):
        row = dict(core_seconds=1, node_seconds=2, searches=0, complete=False)
        row.update(updates)
        path = root / name
        path.write_text(
            json.dumps(dict(schema="parallel.qualification_budget.v1", reservations={"same-name": row}))
        )
        return path

    def test_overlapping_snapshots_preserve_full_charges_and_unknowns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.snapshot(root, "a.json")
            b = self.snapshot(root, "b.json", complete=True, receipt=dict(path="/original", sha256="a" * 64))
            result = f.reconcile([f.inspect([a, b], deadline=time.time() + 10)])
            self.assertEqual(len(result["reservation_names"][0]["variants"]), 2)
            self.assertEqual(result["observations"][0]["unfinished"], ["same-name"])
            self.assertIsNone(result["aggregate_consumption"])
            self.assertIsNone(result["remaining_balance"])
            self.assertFalse(result["production_authorized"])

    def test_identical_snapshots_are_not_proof_of_same_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = f.inspect(
                [self.snapshot(root, "a"), self.snapshot(root, "b")], deadline=time.time() + 10
            )
            result = f.reconcile([observation, observation])
            self.assertEqual(len(result["observations"]), 2)
            self.assertEqual(len(result["identical_snapshot_bytes"][0]), 2)
            self.assertEqual(len(result["reservation_names"][0]["variants"][0]["origins"]), 2)

    def test_missing_symlink_and_expired_paths_remain_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = self.snapshot(root, "actual")
            alias = root / "alias"
            alias.symlink_to(actual)
            rows = f.inspect([alias, root / "missing"], deadline=time.time() + 10)["records"]
            self.assertEqual([r["status"] for r in rows], ["unresolved", "missing"])
            self.assertEqual(f.inspect([actual], deadline=0)["records"][0]["status"], "unvisited")

    def test_invalid_charges_and_duplicate_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in (True, -1, float("nan"), float("inf"), 10**400):
                path = self.snapshot(root, "invalid", core_seconds=value)
                self.assertEqual(
                    f.inspect([path], deadline=time.time() + 10)["records"][0]["status"], "unresolved"
                )
            path.write_text('{"schema":"a","schema":"b"}')
            self.assertIn(
                "duplicate JSON field", f.inspect([path], deadline=time.time() + 10)["records"][0]["reason"]
            )

    def test_overflowed_snapshot_retains_later_path_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.snapshot(root, "a", core_seconds=1e308)
            doc = json.loads(path.read_text())
            doc["reservations"]["another"] = dict(doc["reservations"]["same-name"])
            path.write_text(json.dumps(doc))
            result = f.inspect([path, self.snapshot(root, "b")], deadline=time.time() + 10)
            self.assertEqual([r["status"] for r in result["records"]], ["unresolved", "legacy_snapshot"])
            self.assertIn("overflow", result["records"][0]["reason"])
            json.dumps(result, allow_nan=False)
            doc["reservations"]["same-name"]["core_seconds"] = 10**308
            doc["reservations"]["another"]["core_seconds"] = 10**308
            doc["reservations"]["third"] = dict(doc["reservations"]["same-name"], core_seconds=1.0)
            path.write_text(json.dumps(doc))
            self.assertEqual(
                f.inspect([path], deadline=time.time() + 10)["records"][0]["status"], "unresolved"
            )
            doc["reservations"]["third"]["extra"] = float("nan")
            path.write_text(json.dumps(doc))
            self.assertIn(
                "non-finite JSON", f.inspect([path], deadline=time.time() + 10)["records"][0]["reason"]
            )
            path.write_text(path.read_text().replace("NaN", "1e400"))
            self.assertIn(
                "non-finite JSON number", f.inspect([path], deadline=time.time() + 10)["records"][0]["reason"]
            )

    def test_self_declared_authorization_never_becomes_a_grant(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authorization.json"
            path.write_text(
                json.dumps(dict(schema="parallel.budget_authorization.v2", approved=True, core_seconds=100))
            )
            result = f.inspect([path], deadline=time.time() + 10)
            self.assertEqual(result["records"][0]["status"], "unresolved")
            self.assertFalse(result["production_authorized"])


if __name__ == "__main__":
    unittest.main()
