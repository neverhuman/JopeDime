"""CPU-only bootstrap proofs using the actual pinned Elo solver and tiny fixtures."""

import copy
import os
import unittest
from pathlib import Path

import bootstrap
import numpy as np
import score_panel
from test_score_panel import fixture


class BootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = score_panel.load_reference(
            Path(os.environ.get("H75_ELO_SOURCE", "python/src/jope/north_star/elo_tracker.py"))
        )

    def test_full_prescribed_draws_with_paired_utility_and_no_decision(self):
        result = bootstrap.family_bootstrap(fixture(), self.reference)
        stats = result["bootstrap"]
        self.assertEqual(stats["status"], "COMPLETE_DESCRIPTIVE_BOOTSTRAP")
        self.assertEqual((stats["executed_draws"], stats["seed"], stats["n_families"]), (10000, 20261031, 2))
        self.assertEqual(stats["elo_interval95"]["A"], [1500.0, 1500.0])
        self.assertEqual(stats["contrasts"]["D-A"]["mean_lcb_adjusted"], -1.0)
        self.assertEqual(stats["contrasts"]["D-A"]["median_interval95"], [-1.0, 1.0])
        self.assertEqual(stats["contrasts"]["replay_D-replay_B"]["median_interval95"], [0.0, 0.0])
        self.assertIsNone(result["admission_decision"])

    def test_batch_solver_matches_actual_canonical_cluster_resamples(self):
        packet = fixture()
        for row in packet["rows"]:
            if row["policy_id"] == "B":
                row["utility"] = 0.1 if row["fold"] == 0 else -0.2
        point = score_panel.score(packet, self.reference)
        families = sorted(point["per_family_mean_utility"])
        family_by_context = {
            cell["context_sha256"]: cell["family_id"] for cell in packet["manifest"]["roster"]
        }
        indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
        pairs = bootstrap._family_pair_totals(point, family_by_context, families)
        actual = bootstrap._elo_samples(pairs, indices, point["contenders"].index("A"))
        for i, sampled in enumerate(indices):
            records = []
            for occurrence, index in enumerate(sampled):
                for cid, values in point["per_context_seed_mean_utility"].items():
                    if family_by_context[cid] == families[index]:
                        # Bootstrap occurrences only, never an admitted experimental roster.
                        records.append(
                            self.reference.DatasetEvalRecord(
                                cid, f"resample-{occurrence}", "regression", values
                            )
                        )
            expected = self.reference.compute_family_grouped_bradley_terry_elo(
                records, "A", tie_margin=1e-4, prior=0.01, n_iters=400
            )
            np.testing.assert_allclose(
                actual[i], [expected["ratings"][p] for p in point["contenders"]], rtol=0, atol=0.005
            )

    def test_shared_pairing_keeps_identical_policies_zero_in_every_draw(self):
        packet = fixture()
        for row in packet["rows"]:
            if row["policy_id"] in {"D", "C"}:
                row["utility"] = 0.3 if row["fold"] == 0 else -0.7
        stats = bootstrap.family_bootstrap(packet, self.reference, draws=37)["bootstrap"]
        contrast = stats["contrasts"]["D-C"]
        self.assertEqual(contrast["mean_lcb_adjusted"], 0.0)
        self.assertEqual(contrast["median_interval95"], [0.0, 0.0])
        self.assertEqual(stats["status"], "SMOKE_ONLY")

    def test_order_independence_and_input_immutability(self):
        packet = fixture()
        original = copy.deepcopy(packet)
        first = bootstrap.family_bootstrap(packet, self.reference, draws=19)
        self.assertEqual(packet, original)
        packet["rows"].reverse()
        packet["manifest"]["roster"].reverse()
        second = bootstrap.family_bootstrap(packet, self.reference, draws=19)
        self.assertEqual(first["bootstrap"], second["bootstrap"])

    def test_seed_means_precede_family_sampling(self):
        packet = fixture()
        for row in packet["rows"]:
            if row["policy_id"] == "D":
                row["utility"] = {701: 0.3, 702: 0.3, 703: -0.6}[row["seed"]]
        result = bootstrap.family_bootstrap(packet, self.reference, draws=23)
        self.assertEqual(result["bootstrap"]["elo_interval95"]["D"], [1500.0, 1500.0])
        self.assertEqual(result["bootstrap"]["contrasts"]["D-A"]["mean_lcb_adjusted"], 0.0)

    def test_incomplete_execution_produces_no_intervals_or_draws(self):
        packet = fixture()
        packet["rows"].pop()
        result = bootstrap.family_bootstrap(packet, self.reference)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["bootstrap"]["executed_draws"], 0)
        self.assertNotIn("elo_interval95", result["bootstrap"])
        self.assertIsNone(result["elo"])

    def test_fixed_confirmed_panel_has_four_prescribed_contrasts(self):
        result = bootstrap.family_bootstrap(fixture("confirmed_U8"), self.reference, draws=11)
        self.assertEqual(set(result["bootstrap"]["contrasts"]), {"D-B", "D-C", "D-G", "D-R"})
        self.assertIsNone(result["raw_utility"]["D"]["val_reg_inner_context_mean"])

    def test_invalid_or_excessive_draws_and_seeds_are_refused(self):
        for draws in [True, 0, -1, 10001, float("nan"), 1.5]:
            with self.subTest(draws=draws), self.assertRaises(ValueError):
                bootstrap.family_bootstrap(fixture(), self.reference, draws=draws)
        for seed in [True, -1, 2**32, float("nan")]:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                bootstrap.family_bootstrap(fixture(), self.reference, seed=seed)
        changed = bootstrap.family_bootstrap(fixture(), self.reference, draws=2, seed=1)
        self.assertEqual(changed["bootstrap"]["status"], "SMOKE_ONLY")


if __name__ == "__main__":
    unittest.main()
