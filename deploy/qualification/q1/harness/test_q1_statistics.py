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

class StatisticsTests(unittest.TestCase):
    def test_percentile_interpolation_and_exact_bootstrap(self):
        self.assertEqual(q1.quantile([0, 10, 20, 30], 0.25), 7.5)
        result = q1.bootstrap([0.1] * 40)
        self.assertEqual(
            (result["median"], result["lower"], result["upper"]), (0.1, 0.1, 0.1)
        )
        self.assertEqual(result["resamples"], 10000)
        self.assertEqual(q1.bootstrap([0, 1, 2, 3]), q1.bootstrap([0, 1, 2, 3]))

    def test_superiority_direction_and_zero_effect(self):
        plan = qualification_plan()
        result = q1.analyze(plan, fixture_receipts(plan, 0.9))
        self.assertEqual(result["decision"], "PASS")
        self.assertAlmostEqual(result["primary"]["median"], 0.1)
        self.assertEqual(
            q1.analyze(plan, fixture_receipts(plan, 1.0))["decision"], "FAIL"
        )
        self.assertEqual(
            q1.analyze(plan, fixture_receipts(plan, 1.1))["decision"], "FAIL"
        )

    def test_complete_panel_and_immutable_identity_are_required(self):
        plan = qualification_plan()
        valid = fixture_receipts(plan)
        bad_start = copy.deepcopy(valid)
        bad_start[0]["request_sha256"] = "e" * 64
        cases = [valid[:-1], valid + [valid[-1]], bad_start]
        for field, value in [
            ("budget", 23),
            ("source_commit", "b" * 40),
            ("request_sha256", "0" * 64),
        ]:
            rows = copy.deepcopy(valid)
            rows[1][field] = value
            cases.append(rows)
        rows = copy.deepcopy(valid)
        rows[1]["response"]["status"] = "failed"
        rows[1]["response"]["loss"] = None
        cases.append(rows)
        rows = copy.deepcopy(valid)
        rows[1]["response"]["completed"] = 0
        cases.append(rows)
        for rows in cases:
            with self.assertRaises(ValueError):
                q1.analyze(plan, rows)

    def test_subgroup_regression_blocks_positive_overall_result(self):
        plan = qualification_plan()
        for i, dataset in enumerate(plan["datasets"]):
            dataset["features"] = 30 if i < 10 else 10
        rows = fixture_receipts(plan, 0.5)
        # Regress 10 wide datasets; overall superiority still passes, subgroup must block.
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
        smoke = copy.deepcopy(plan)
        smoke["mode"] = "heuristic_smoke"
        with self.assertRaises(ValueError):
            q1.analyze(smoke, rows)

    def test_complete_panel_requires_program_and_adapter_identities(self):
        plan = qualification_plan()
        for field in ("champion_program", "adapter"):
            for value in (None, "", "  ", False, {}):
                with self.subTest(field=field, value=value):
                    rows = fixture_receipts(plan)
                    rows[1]["response"][field] = value
                    with self.assertRaises(ValueError):
                        q1.analyze(plan, rows)

    def test_failure_inventory_has_unique_charged_attempts_and_reasons(self):
        plan = qualification_plan()
        bad = [
            [{"attempt": 24, "reason": "out of range"}],
            [{"attempt": -1, "reason": "negative"}],
            [{"attempt": True, "reason": "Boolean"}],
            [{"attempt": 0}],
            [{"attempt": 0, "reason": "  "}],
            ["unknown"],
            [{"attempt": 1, "reason": "x"}, {"attempt": 1, "reason": "duplicate"}],
        ]
        for failures in bad:
            with self.subTest(failures=failures):
                rows = fixture_receipts(plan)
                rows[1]["response"].update(
                    completed=24 - len(failures), candidate_failures=failures
                )
                with self.assertRaises(ValueError):
                    q1.analyze(plan, rows)
        # A real, identified failed attempt remains charged and does not poison
        # the other completed candidates' finite champion loss.
        rows = fixture_receipts(plan)
        rows[1]["response"].update(
            completed=23,
            fold_scores_completed=46,
            candidate_failures=[{"attempt": 0, "reason": "fixture candidate failure"}],
        )
        self.assertEqual(q1.analyze(plan, rows)["decision"], "PASS")

    def test_boolean_seed_alias_is_not_an_approved_integer_seed(self):
        plan = qualification_plan()
        rows = fixture_receipts(plan)
        for row in rows:
            if row["seed"] != 1:
                continue
            row["seed"] = True
            request = {
                k: row[k]
                for k in (
                    "mode",
                    "plan_sha256",
                    "source_commit",
                    "pins",
                    "dataset",
                    "guidance",
                    "seed",
                    "budget",
                )
            }
            request.update(schema="jope.q1-request/v1", metric="rmse")
            row["request_sha256"] = q1.digest(request)
            if row["event"] == "finished":
                row["response"]["request_sha256"] = row["request_sha256"]
        with self.assertRaises(ValueError):
            q1.analyze(plan, rows)

    def test_finished_panel_requires_real_compute_accounting(self):
        plan = qualification_plan()
        for value in (-1, float("nan"), q1.decode("1e309"), True, None):
            with self.subTest(wall_seconds=value):
                rows = fixture_receipts(plan)
                rows[1]["wall_seconds"] = value
                with self.assertRaises(ValueError):
                    q1.analyze(plan, rows)
        rows = fixture_receipts(plan)
        del rows[1]["wall_seconds"]
        with self.assertRaises(ValueError):
            q1.analyze(plan, rows)
