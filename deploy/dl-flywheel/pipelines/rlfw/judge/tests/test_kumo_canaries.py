"""PR-14 canaries A1–A5. Live mode is refused here; no weights are opened."""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from rlfw.judge.canaries import main, run_canaries
from rlfw.judge.encoding import NOT_DECONTAMINATION

PIPELINES = Path(__file__).resolve().parents[3]


class CanaryContractTest(unittest.TestCase):
    def test_a1_through_a5_on_the_fixture_table(self):
        report = run_canaries()
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["fit_row_ids"]), 8)
        self.assertEqual(len(report["judge_row_ids"]), 2)
        canaries = report["canaries"]
        self.assertEqual(canaries["A1"]["diagnostic_label"], NOT_DECONTAMINATION)
        self.assertEqual(canaries["A1"]["diagnostic_label"], "not decontamination")
        self.assertTrue(canaries["A2"]["passed"])
        self.assertTrue(canaries["A3"]["passed"])
        self.assertTrue(canaries["A4"]["passed"])
        self.assertEqual(canaries["A5"]["overlap"], "quarantine")
        self.assertEqual(canaries["A5"]["missing_disclosure"], "overlap_unknown")
        self.assertTrue(canaries["A5"]["unseen_refused_for_overlap"])
        self.assertIn("XGBoost was not run", report["selection_implementation"])

    def test_live_flag_refuses_without_opening_a_weights_path(self):
        weight = Path("/tmp/g-judge-weights-must-not-be-opened.pt")
        self.assertFalse(weight.exists())
        with patch("builtins.open", side_effect=AssertionError("weights path was opened")):
            self.assertEqual(main(["--live", "--weights", str(weight)]), 2)
            self.assertEqual(main(["--live", "--owner-live", "--weights", str(weight)]), 3)
        self.assertFalse(weight.exists())

    def test_module_cli_exits_are_2_and_3(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PIPELINES)
        command = [sys.executable, "-m", "rlfw.judge.canaries"]
        refused = subprocess.run(
            [*command, "--live"],
            cwd=PIPELINES,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        accepted = subprocess.run(
            [*command, "--live", "--owner-live"],
            cwd=PIPELINES,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(refused.returncode, 2)
        self.assertEqual(accepted.returncode, 3)
        self.assertIn("Kumo weights were not loaded", accepted.stderr)


if __name__ == "__main__":
    unittest.main()
