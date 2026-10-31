from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from .gate import CONTRASTS, Refusal, main, verdict


def evidence(lcb: float = 0.002, median: float = 0.001, harmed: int = 10) -> dict:
    return {
        "schema": "jope.w1-gate-evidence/v1",
        "bootstrap": {
            "unit": "family",
            "draws": 10000,
            "seed": 20261031,
            "paired_seeds_together": True,
            "all_registered_contrasts_adjusted": True,
        },
        "contrasts": {
            name: {
                "adjusted_family_mean_lcb": lcb,
                "median_family_delta": median,
                "families_loss_gt_0_01": harmed,
                "family_count": 205,
            }
            for name in CONTRASTS
        },
    }


class GateTests(unittest.TestCase):
    def test_search_only_requires_u4_and_legacy_no_harm(self):
        row = evidence()
        row["contrasts"]["legacy_vs_program_gbm"]["adjusted_family_mean_lcb"] = -0.002
        row["contrasts"]["legacy_vs_program_context_gbm"][
            "adjusted_family_mean_lcb"
        ] = -0.004
        result = verdict(row)
        self.assertEqual(result["decision"], "ADVANCE_SEARCH_ONLY")
        self.assertFalse(result["scientific_result"])

    def test_legacy_advance_needs_both_legacy_contrasts(self):
        row = evidence(lcb=-0.01)
        for name in CONTRASTS[2:]:
            row["contrasts"][name]["adjusted_family_mean_lcb"] = 0.003
        self.assertEqual(verdict(row)["decision"], "ADVANCE_LEGACY")

    def test_search_does_not_select_only_one_passing_comparator(self):
        row = evidence()
        row["contrasts"]["u4_vs_program_context_gbm"]["adjusted_family_mean_lcb"] = 0.0
        for name in CONTRASTS[2:]:
            row["contrasts"][name]["adjusted_family_mean_lcb"] = -0.01
        self.assertEqual(verdict(row)["decision"], "NO_ADVANCE")

    def test_exact_lcb_boundary_does_not_pass(self):
        row = evidence(lcb=0.001)
        self.assertEqual(verdict(row)["decision"], "NO_ADVANCE")

    def test_more_than_ten_percent_harm_does_not_pass(self):
        row = evidence(harmed=21)
        self.assertEqual(verdict(row)["decision"], "NO_ADVANCE")

    def test_ten_percent_harm_boundary_passes(self):
        row = evidence(harmed=20)
        self.assertEqual(verdict(row)["decision"], "ADVANCE_SEARCH_ONLY")

    def test_negative_family_median_does_not_pass(self):
        row = evidence(median=-0.0001)
        self.assertEqual(verdict(row)["decision"], "NO_ADVANCE")

    def test_missing_registered_contrast_refuses(self):
        row = evidence()
        del row["contrasts"][CONTRASTS[-1]]
        with self.assertRaises(Refusal):
            verdict(row)

    def test_unadjusted_or_wrong_bootstrap_refuses(self):
        row = evidence()
        row["bootstrap"]["all_registered_contrasts_adjusted"] = False
        with self.assertRaises(Refusal):
            verdict(row)
        row = evidence()
        row["bootstrap"]["all_registered_contrasts_adjusted"] = 1
        with self.assertRaises(Refusal):
            verdict(row)

    def test_nonfinite_or_invalid_harm_counts_refuse(self):
        row = evidence()
        row["contrasts"][CONTRASTS[0]]["adjusted_family_mean_lcb"] = math.nan
        with self.assertRaises(Refusal):
            verdict(row)
        row = evidence()
        row["contrasts"][CONTRASTS[0]]["families_loss_gt_0_01"] = 206
        with self.assertRaises(Refusal):
            verdict(row)
        row = evidence()
        row["contrasts"][CONTRASTS[0]]["family_count"] = 204
        with self.assertRaises(Refusal):
            verdict(row)

    def test_cli_marks_synthetic_decision_as_non_scientific(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text(json.dumps(evidence()), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(["--evidence", str(path)])
        self.assertEqual(result, 0)
        self.assertFalse(json.loads(output.getvalue())["scientific_result"])

    def test_cli_refuses_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text('{"schema":"wrong","schema":"wrong"}', encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stderr(output):
                result = main(["--evidence", str(path)])
        self.assertEqual(result, 2)
        self.assertIn("duplicate JSON key", output.getvalue())


if __name__ == "__main__":
    unittest.main()
