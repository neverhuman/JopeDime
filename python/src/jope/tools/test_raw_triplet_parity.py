import unittest

import raw_triplet_parity as r


class RawTripletParityTests(unittest.TestCase):
    def test_plans_only_missing_real_triplets(self):
        plan = r.plan_copy(
            [
                "regression/aa/meta.json",
                "regression/aa/train.csv",
                "regression/aa/reports/skip.json",
                "binary/3c288e95bcdaf2fa/meta.json",
                "expansion_v9_manifest.json",
                "synth_foo/train.csv",
            ],
            remote_hashes={"a" * 64},
            local_sizes={
                "regression/aa/meta.json": {
                    "sha256": "b" * 64,
                    "size": 10,
                    "mode": 0o644,
                    "classification": "retained_non_synthetic_triplet",
                },
                "regression/aa/train.csv": {
                    "sha256": "a" * 64,
                    "size": 20,
                    "mode": 0o644,
                    "classification": "retained_non_synthetic_triplet",
                },
            },
        )
        self.assertEqual(plan["selected"], ["regression/aa/meta.json"])
        self.assertEqual(plan["synthetics_copied"], 0)
        self.assertTrue(plan["expansion_v9_preserved"])
        self.assertFalse(plan["execute"])
        self.assertEqual(plan["n_skipped_present_hash"], 1)

    def test_traversal_missing_hash_and_duplicate_paths_are_rejected(self):
        for path in ("regression/../train.csv", "/regression/a/train.csv", "regression//train.csv"):
            self.assertFalse(r.is_triplet(path))
        with self.assertRaisesRegex(ValueError, "complete original object record"):
            r.plan_copy(["regression/a/train.csv"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            r.plan_copy(["regression/a/train.csv"] * 2)

    def test_booleans_cannot_authorize_transfer(self):
        ok, reasons = r.execute_allowed(
            dict(synthetics_copied=0),
            dict(backfill_allowed=True),
            dict(certified=False, missing_roots=["deployment_conformance"]),
            True,
        )
        self.assertFalse(ok)
        self.assertTrue(any("verified workspace transfer" in row for row in reasons))
        self.assertFalse(
            r.execute_allowed(
                dict(synthetics_copied=0), dict(backfill_allowed=True), dict(certified=True), True
            )[0]
        )


if __name__ == "__main__":
    unittest.main()
