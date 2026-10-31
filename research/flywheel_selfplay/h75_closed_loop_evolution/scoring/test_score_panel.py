"""Tiny CPU-only development fixtures; no dataset, model or experiment execution."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import score_panel as scoring


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(panel="legacy"):
    manifest = {
        "schema": "jope.h75-closed-loop.score-admission/v1",
        "fixture_only": True,
        "run_id": "tiny-development-fixture",
        "source_git_sha": "a" * 40,
        "prereg_sha256": "b" * 64,
        "execution_manifest_sha256": "c" * 64,
        "panel": panel,
        "dope_sha256": scoring.DOPE_SHA256,
        "roster": [],
    }
    rows = []
    for cid, family, fold in [("a", "large", 0), ("b", "large", 0), ("c", "small", 1)]:
        for seed in scoring.SEEDS:
            cell = {
                "context_sha256": sha(cid),
                "family_id": family,
                "fold": fold,
                "role": "DL-Train",
                "seed": seed,
                "evaluation_identity_sha256": sha(cid + "evaluation"),
            }
            if panel == "confirmed_U8":
                cell["confirmation_cut_sha256s"] = [sha(cid + str(cut)) for cut in (801, 802, 803)]
            manifest["roster"].append(cell)
            for policy in scoring.PANELS[panel]:
                utility = (1.0 if family == "large" else -1.0) if policy == "D" else 0.0
                rows.append(
                    {
                        "schema": "jope.h75-closed-loop.score/v1",
                        **{k: manifest[k] for k in scoring.IDENTITIES},
                        "panel": panel,
                        "policy_id": policy,
                        **copy.deepcopy(cell),
                        "selected_program_sha256": sha(cid + policy),
                        "status": "completed",
                        "utility": utility,
                    }
                )
    return {"manifest": manifest, "rows": rows}


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.elo_source = Path(os.environ.get("H75_ELO_SOURCE", "python/src/jope/north_star/elo_tracker.py"))
        cls.reference = scoring.load_reference(cls.elo_source)

    def test_complete_panel_has_equal_family_weight_and_no_admission(self):
        result = scoring.score(fixture(), self.reference)
        self.assertEqual(result["status"], "COMPLETE_POINT_DIAGNOSTICS")
        self.assertEqual(result["elo"]["ratings"]["A"], 1500.0)
        self.assertEqual(result["elo"]["ratings"]["D"], 1500.0)
        self.assertEqual((result["n_contexts"], result["n_families"], result["expected_rows"]), (3, 2, 54))
        raw = result["raw_utility"]["D"]
        self.assertAlmostEqual(raw["val_reg_inner_context_mean"], 1 / 3)
        self.assertEqual(raw["equal_family_mean_utility"], 0.0)
        self.assertEqual(raw["fraction_families_delta_below_minus_0_01"], 0.5)
        self.assertIsNone(result["admission_decision"])
        self.assertEqual(result["bootstrap"]["executed_draws"], 0)

    def test_seed_averaging_precedes_elo_ties(self):
        packet = fixture()
        for row in packet["rows"]:
            if row["policy_id"] == "D":
                row["utility"] = {701: 0.3, 702: 0.3, 703: -0.6}[row["seed"]]
        result = scoring.score(packet, self.reference)
        self.assertEqual(result["elo"]["ratings"]["D"], 1500.0)
        self.assertEqual(result["elo"]["pairwise_win_tie_loss_vs_anchor"]["D"]["ties"], 3)

    def test_actual_canonical_reference_panel_parity(self):
        packet = fixture()
        for row in packet["rows"]:
            if row["policy_id"] == "B":
                row["utility"] = 0.0001
        result = scoring.score(packet, self.reference)
        records = [
            self.reference.DatasetEvalRecord(
                cid, "small" if cid == sha("c") else "large", "regression", values
            )
            for cid, values in result["per_context_seed_mean_utility"].items()
        ]
        expected = self.reference.compute_family_grouped_bradley_terry_elo(
            records, "A", tie_margin=1e-4, prior=0.01, n_iters=400
        )
        self.assertEqual(result["elo"], expected)
        self.assertEqual(result["elo"]["pairwise_win_tie_loss_vs_anchor"]["B"]["ties"], 3)

    def test_missing_and_failed_cells_return_counts_without_elo(self):
        for status in [None, "failed", "timeout", "unknown"]:
            packet = fixture()
            if status is None:
                packet["rows"].pop()
            else:
                packet["rows"][0].update(status=status, utility=None)
            result = scoring.score(packet, self.reference)
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertIsNone(result["elo"])
            self.assertIsNone(result["raw_utility"])
            self.assertEqual(result["missing_rows"], 1 if status is None else 0)

    def test_duplicate_foreign_incomplete_roster_and_identity_rejection(self):
        changes = [
            lambda p: p["rows"].append(copy.deepcopy(p["rows"][0])),
            lambda p: p["manifest"]["roster"].pop(),
            lambda p: p["rows"][0].update(seed=704),
            lambda p: p["rows"][0].update(panel="confirmed_U8"),
            lambda p: p["rows"][0].update(policy_id="r882"),
            lambda p: p["rows"][0].update(evaluation_identity_sha256="f" * 64),
            lambda p: p["rows"][0].update(run_id="other"),
            lambda p: p["manifest"]["roster"][0].update(role="DL-Val"),
            lambda p: p["rows"][0].update(role="sealed"),
            lambda p: p["manifest"]["roster"][0].update(family_id="unknown"),
            lambda p: p["manifest"]["roster"][3].update(fold=2),
            lambda p: p["manifest"].update(dope_sha256="wrong"),
        ]
        for change in changes:
            packet = fixture()
            change(packet)
            with self.subTest(change=change), self.assertRaises(ValueError):
                scoring.admit(packet)

    def test_nonfinite_boolean_and_completed_null_are_rejected(self):
        for value in [float("nan"), float("inf"), True, "1", None]:
            packet = fixture()
            packet["rows"][0]["utility"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                scoring.admit(packet)

    def test_confirmed_panel_requires_all_three_exact_cuts(self):
        packet = fixture("confirmed_U8")
        result = scoring.score(packet, self.reference)
        self.assertEqual(result["status"], "COMPLETE_POINT_DIAGNOSTICS")
        self.assertIsNone(result["raw_utility"]["D"]["val_reg_inner_context_mean"])
        packet["rows"][0]["confirmation_cut_sha256s"].reverse()
        with self.assertRaisesRegex(ValueError, "confirmation"):
            scoring.admit(packet)

    def test_hash_refusal_precedes_module_execution(self):
        with tempfile.TemporaryDirectory(prefix="h75-reference-") as directory:
            path = Path(directory) / "helper.py"
            marker = Path(directory) / "executed"
            path.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                scoring.load_reference(path)
            self.assertFalse(marker.exists())

    def test_cli_fixture_dryrun_and_duplicate_json_refusal(self):
        with tempfile.TemporaryDirectory(prefix="h75-fixture-") as directory:
            path = Path(directory) / "packet.json"
            path.write_text(json.dumps(fixture()))
            command = [
                sys.executable,
                str(Path(scoring.__file__)),
                str(path),
                "--elo-source",
                str(self.elo_source),
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["fixture_only"])
            self.assertEqual(report["observed_rows"], 54)
            self.assertEqual(len(report["packet_sha256"]), 64)
            path.write_text('{"manifest": 1, "manifest": 2}')
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
