"""Data-free controls for H157's family partition and strict candidate gate."""

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from science import admits, bootstrap_median, choose_winner, folds, primary_summary
from runner import h142_loss, training_plan, validate_launch


class H157Tests(unittest.TestCase):
    def test_partition_is_complete_disjoint_and_deterministic(self):
        families = [f"fam-{i:03d}" for i in range(205)]
        groups = folds(families)
        select = set(groups[0])
        fit = set().union(*(set(g) for g in groups[1:]))
        self.assertEqual((len(fit), len(select)), (164, 41))
        self.assertFalse(fit & select)
        self.assertEqual(fit | select, set(families))
        self.assertEqual(groups, folds(reversed(families)))

    def test_strict_margin_and_lcb_gates(self):
        tie = bootstrap_median(np.full(41, 0.001), 10)
        self.assertLessEqual(abs(tie["median"] - 0.001), 1e-12)
        self.assertFalse(admits(tie))
        self.assertTrue(admits(bootstrap_median(np.full(41, 0.001 + 2e-12), 10)))

    def test_candidate_ties_and_failed_winner_keep_r882(self):
        tied = {"c02": np.full(41, 0.003), "c01": np.full(41, 0.003)}
        tied_decision = choose_winner(tied, 20)
        self.assertTrue(tied_decision["tied_for_best"])
        self.assertEqual(tied_decision["selected"], "R882")
        candidates = {
            "bok-01": np.array([0.003] * 21 + [-0.001] * 20),
            "bok-02": np.full(41, 0.002),
        }
        decision = choose_winner(candidates, 30)
        self.assertEqual(decision["winner"], "bok-01")
        self.assertFalse(decision["admitted"])
        self.assertFalse(decision["tied_for_best"])
        self.assertGreater(decision["candidates"]["bok-02"]["ci95"][0], 0)
        self.assertEqual(decision["selected"], "R882")

    def test_h142_plan_is_40_updates_fit_only_and_candidate_seeded(self):
        rows = [
            {"family_id": f"f{i:03d}", "context_sha256": f"c{i:03d}"}
            for i in range(164)
        ]
        pools = {
            row["context_sha256"]: {"fitness": np.arange(220, dtype=np.float64)}
            for row in rows
        }
        first = training_plan(rows, pools, 15710100)
        again = training_plan(rows, pools, 15710100)
        other = training_plan(rows, pools, 15710101)
        def fingerprint(plan):
            return [
                (stream, cids, tuple(ix.tobytes() for ix in indexes))
                for stream, cids, indexes in plan
            ]
        self.assertEqual(len(first), 40)
        self.assertEqual(fingerprint(first), fingerprint(again))
        self.assertNotEqual(fingerprint(first), fingerprint(other))
        allowed = {row["context_sha256"] for row in rows}
        for _, cids, indexes in first:
            self.assertTrue(set(cids) <= allowed)
            self.assertEqual(len(cids), 16)
            self.assertTrue(all(len(ix) == 192 for ix in indexes))
            self.assertTrue(all(np.array_equal(ix[:48], np.arange(48)) for ix in indexes))

    def test_primary_bootstrap_and_holm_are_paired_family_level(self):
        base = np.zeros((3, 19))
        positive = np.full((3, 19), 0.01)
        summary = primary_summary(base, {"BOK8": positive, "TR1": positive})
        self.assertEqual(summary["draws"], 10000)
        for item in summary["treatments"].values():
            self.assertGreater(item["holm_lcb95"], 0)
            self.assertLess(item["holm_adjusted_p"], 0.05)

    def test_manifest_source_pins_match_checkout(self):
        root = Path(__file__).resolve().parent
        manifest = json.loads((root / "manifest.json").read_text())
        for name, expected in manifest["source_sha256"].items():
            self.assertEqual(
                hashlib.sha256((root / name).read_bytes()).hexdigest(), expected, name
            )

    def test_launch_requires_merged_source_and_assigned_cross_review(self):
        launch = dict(
            source_commit="a" * 40, manifest_sha256="b" * 64,
            merged_source_verified=True, cross_review_pass=True,
            reviewer="luna_v1_rnk11", review_pr=1400,
            reviewed_source_commit="c" * 40, merged_pr=1401,
        )
        validate_launch(launch, "a" * 40, "b" * 64)
        for change in (
            {"cross_review_pass": False},
            {"reviewer": "another-agent"},
            {"source_commit": "d" * 40},
        ):
            with self.assertRaises(RuntimeError):
                validate_launch(launch | change, "a" * 40, "b" * 64)

    def test_h142_loss_uses_tau_point_two_and_strict_gate_cost(self):
        import torch
        import torch.nn.functional as F

        prediction = torch.tensor([[0.4, 1.1, -0.2]], dtype=torch.float32)
        fitness = torch.tensor([[-1.0, 0.5, -3.0]], dtype=torch.float32)
        gate = torch.tensor([[True, False, True]])
        target = torch.tensor([[-1.0, 2.0, -3.0]], dtype=torch.float32)
        expected = (
            F.smooth_l1_loss(prediction, target, reduction="none", beta=1.0).mean(1)
            + F.kl_div(
                F.log_softmax(-prediction / 0.2, dim=-1),
                F.softmax(-target / 0.2, dim=-1),
                reduction="none",
            ).sum(1)
        ).mean()
        self.assertTrue(torch.allclose(h142_loss(prediction, fitness, gate), expected))


if __name__ == "__main__":
    unittest.main()
