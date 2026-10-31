"""Synthetic-only tests for the gated W5 plan validator."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from runner import build_plan


ROOT = Path(__file__).resolve().parent


class PlanTests(unittest.TestCase):
    def test_plan_retains_fixed_protocol_and_stays_gated(self):
        plan = build_plan()
        self.assertEqual(plan["arms"], [
            "r882-absolute-cost-ranking",
            "program-feature-gbm-transition-baseline",
            "neural-parent-child-head",
            "parent-operator-shuffled-placebo",
        ])
        self.assertEqual(plan["seeds"], [701, 702, 703])
        self.assertEqual(plan["family_folds"], 5)
        self.assertIsNone(plan["planned_pair_count"])
        self.assertIsNone(plan["planned_fit_count"])
        self.assertEqual(plan["provenance_status"], "not_checked")
        self.assertFalse(plan["execution_admitted"])
        self.assertEqual(plan["data_access"], "none")
        self.assertEqual(plan["model_fits_executed"], 0)

    def test_dry_run_json_is_explicitly_non_executing(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "runner.py"), "--dry-run", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["claim"], "H71")
        self.assertFalse(plan["execution_admitted"])
        self.assertEqual(plan["data_access"], "none")

    def test_non_dry_run_refuses_execution(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "runner.py")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("W5 execution is provenance-gated", result.stderr)


if __name__ == "__main__":
    unittest.main()
