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

class OwnerSelectionTests(unittest.TestCase):
    def test_frozen_family_minimum_sha_selection(self):
        import q1_dedup

        amendment = q1.load(q1_dedup.AMENDMENT)
        packet = q1_dedup.bound(amendment["source_roster"])
        membership = q1_dedup.bound(amendment["membership"])
        before = copy.deepcopy(packet)
        selected, excluded = q1_dedup.select(packet, membership)
        self.assertEqual(packet, before)
        self.assertEqual((len(selected), len(excluded)), (97, 56))
        self.assertEqual(len({d["family"] for d in selected}), 97)
        reversed_packet = copy.deepcopy(packet)
        reversed_packet["official_roster"].reverse()
        self.assertEqual(
            q1_dedup.select(reversed_packet, membership), (selected, excluded)
        )
        original = {d["dataset_id"]: d for d in packet["official_roster"]}
        for family in membership["families"]:
            expected = min(
                (original[d] for d in family["dataset_ids"]),
                key=lambda d: d["dataset_sha256"],
            )
            actual = next(d for d in selected if d["family"] == family["family_id"][4:])
            self.assertEqual(
                (actual["id"], actual["cut_sha256"]),
                (expected["dataset_id"], expected["cut_sha256"]),
            )
        self.assertEqual(q1_dedup.load_selection(), selected)
        self.assertEqual(
            q1_dedup.subgroup_counts(selected),
            {"small": 70, "large": 27, "narrow": 94, "wide": 3},
        )

    def test_invalid_or_ambiguous_owner_membership_is_refused(self):
        import q1_dedup

        amendment = q1.load(q1_dedup.AMENDMENT)
        packet = q1_dedup.bound(amendment["source_roster"])
        original = q1_dedup.bound(amendment["membership"])
        cases = []
        membership = copy.deepcopy(original)
        membership["families"][0]["dataset_ids"].append(
            membership["families"][1]["dataset_ids"][0]
        )
        cases.append(membership)
        membership = copy.deepcopy(original)
        membership["families"][0]["family_id"] = membership["families"][1]["family_id"]
        cases.append(membership)
        membership = copy.deepcopy(original)
        membership["families"][0]["partition"] = "dl_holdout"
        cases.append(membership)
        membership = copy.deepcopy(original)
        membership["families"][0]["dataset_ids"] = []
        cases.append(membership)
        membership = copy.deepcopy(original)
        membership["families"][0]["family_id"] = "srcfile:unreviewed"
        cases.append(membership)
        for membership in cases:
            with self.assertRaises(ValueError):
                q1_dedup.select(packet, membership)

    def test_bound_metadata_hash_and_scope_are_required(self):
        import q1_dedup

        reference = copy.deepcopy(q1.load(q1_dedup.AMENDMENT)["membership"])
        reference["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            q1_dedup.bound(reference)
        with self.assertRaisesRegex(ValueError, "outside Q1"):
            q1_dedup.asset("Cargo.toml")

    def test_regenerated_plan_preserves_fixed_execution_identity(self):
        import q1_dedup

        selected = q1_dedup.load_selection()
        plan = q1_dedup.make_plan(selected, ["f" * 40], "a" * 40)
        self.assertEqual(set(plan), q1.PLAN_KEYS | {"wide_census_sha256"})
        self.assertEqual(
            (plan["budget"], plan["seeds"], plan["timeout_seconds"]),
            (24, [0, 1, 2], 600),
        )
        self.assertEqual(plan["pins"]["bundle_sha256"], q1_dedup.BUNDLE)
        self.assertEqual(plan["pins"]["dope_sha256"], q1_dedup.DOPE)
        self.assertIsNone(plan["pins"]["evaluator_sha256"])
        self.assertEqual(plan["ledger_root"], q1_dedup.LEDGER)
        self.assertNotIn("minimum_wide_families", plan)
        self.assertEqual(selected, plan["datasets"])
        with self.assertRaisesRegex(ValueError, "all qualification identities"):
            q1.validate_plan(plan)

    def test_five_distinct_wide_families_cannot_be_waived(self):
        plan = qualification_plan()  # Entirely invented unsigned metadata.
        for i, dataset in enumerate(plan["datasets"]):
            dataset["features"] = 30 if i < 5 else 10
        q1.validate_plan(plan)
        plan["datasets"][4]["features"] = 10
        with self.assertRaisesRegex(ValueError, "subgroup wide"):
            q1.validate_plan(plan)
        plan["minimum_wide_families"] = 4
        with self.assertRaisesRegex(ValueError, "unsupported/missing plan field"):
            q1.validate_plan(plan)

    def test_r20_requires_every_pinned_wide_family_and_exact_census(self):
        import q1_dedup

        plan = q1_dedup.make_plan(
            q1_dedup.load_selection(),
            ["f" * 40],
            qualification_plan()["source_commit"],
            qualification_plan()["pins"]["evaluator_sha256"],
        )
        q1.validate_plan(plan)  # Unsigned metadata fixture, no evaluator execution.
        self.assertEqual(q1.subgroup_minimum(plan, "wide"), 3)
        self.assertEqual(q1.subgroup_minimum(plan, "small"), 5)
        cases = []
        changed = copy.deepcopy(plan)
        changed["wide_census_sha256"] = "0" * 64
        cases.append(changed)
        changed = copy.deepcopy(plan)
        changed["datasets"] = [
            d for d in changed["datasets"] if d["id"] != "12fd2dc4af621833"
        ]
        cases.append(changed)
        changed = copy.deepcopy(plan)
        next(d for d in changed["datasets"] if d["features"] > 20)["family"] = "0" * 64
        cases.append(changed)
        changed = copy.deepcopy(plan)
        next(d for d in changed["datasets"] if d["features"] > 20)["features"] = 20
        cases.append(changed)
        for changed in cases:
            with self.subTest(changed=q1.digest(changed)):
                with self.assertRaisesRegex(ValueError, "R20"):
                    q1.validate_plan(changed)
        del plan["wide_census_sha256"]
        with self.assertRaisesRegex(ValueError, "subgroup wide"):
            q1.validate_plan(plan)

    def test_r20_three_family_statistics_still_enforce_wide_noninferiority(self):
        import q1_dedup

        plan = q1_dedup.make_plan(
            q1_dedup.load_selection(),
            ["f" * 40],
            qualification_plan()["source_commit"],
            qualification_plan()["pins"]["evaluator_sha256"],
        )
        rows = fixture_receipts(plan, 0.5)
        positive = q1.analyze(plan, rows)
        self.assertEqual(positive["decision"], "PASS")
        self.assertEqual(positive["subgroups"]["wide"]["datasets"], 3)
        for row in rows:
            if (
                row["event"] == "finished"
                and row["guidance"] == "learned"
                and row["dataset"]["features"] > 20
            ):
                row["response"]["loss"] = 1.1
        result = q1.analyze(plan, rows)
        self.assertGreater(result["primary"]["lower"], 0)
        self.assertEqual(result["decision"], "FAIL")
        self.assertLess(result["subgroups"]["wide"]["lower"], -0.02)
