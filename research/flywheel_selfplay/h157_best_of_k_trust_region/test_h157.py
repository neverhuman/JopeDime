"""Data-free controls for H157's family partition and strict candidate gate."""

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from science import admits, bootstrap_median, choose_winner, folds, primary_summary
from runner import (
    BOK_BOOTSTRAP_BASE_SEED,
    HERE,
    REPO_ROOT,
    disk_used_percent,
    h142_loss,
    import_parent,
    input_path,
    restore_initial_state,
    training_plan,
    validate_baseline,
    validate_launch,
)


class H157Tests(unittest.TestCase):
    def test_parent_runner_uses_h146_modules_and_restores_h157_imports(self):
        h157_science = sys.modules["science"]
        parent, runtime = import_parent()
        self.assertEqual(set(parent.ARMS), {"CTRL", "LA", "LA_K10"})
        self.assertEqual(runtime.HERE.name, "h146_lookahead_exec")
        self.assertIs(sys.modules["science"], h157_science)

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
            self.assertTrue(
                all(np.array_equal(ix[:48], np.arange(48)) for ix in indexes)
            )

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
            source_commit="a" * 40,
            manifest_sha256="b" * 64,
            merged_source_verified=True,
            cross_review_pass=True,
            reviewer="luna_v1_rnk11",
            review_pr=1400,
            reviewed_source_commit="c" * 40,
            merged_pr=1401,
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

    def test_manifest_input_paths_resolve_to_pinned_files_not_hashes(self):
        self.assertEqual(
            input_path("/home/ubuntu/jope-data/h50/input/selected.json"),
            Path("/home/ubuntu/jope-data/h50/input/selected.json"),
        )
        self.assertEqual(
            input_path(
                "research/flywheel_selfplay/h150_w5_provenance_unblock/sealed_family_manifest.json"
            ),
            REPO_ROOT
            / "research/flywheel_selfplay/h150_w5_provenance_unblock/sealed_family_manifest.json",
        )

    def test_disk_guard_uses_user_available_blocks(self):
        percent = disk_used_percent(1000, 70, 40)
        self.assertAlmostEqual(percent, 100 * 930 / 970)
        self.assertGreaterEqual(percent, 95.0)

    def test_baseline_requires_exact_91_context_reference(self):
        expected = np.linspace(-1.0, 1.0, 91, dtype=np.float64)
        self.assertEqual(validate_baseline(expected.copy(), expected), 0.0)
        near_miss = expected.copy()
        near_miss[37] += 1e-12
        with self.assertRaisesRegex(RuntimeError, "exact native regression"):
            validate_baseline(near_miss, expected)
        with self.assertRaisesRegex(RuntimeError, "exactly 91"):
            validate_baseline(expected[:-1], expected[:-1])

    def test_bok8_bootstrap_seeds_match_preregistered_roster(self):
        gains = {f"bok-{index:02d}": np.zeros(41) for index in range(1, 9)}
        for seed_index in range(3):
            decision = choose_winner(gains, BOK_BOOTSTRAP_BASE_SEED + 100 * seed_index)
            observed = {
                name: score["bootstrap_seed"]
                for name, score in decision["candidates"].items()
            }
            expected = {
                f"bok-{index:02d}": 157141 + 100 * seed_index + index
                for index in range(1, 9)
            }
            self.assertEqual(observed, expected)

    def test_descriptive_r882_baseline_restores_frozen_initial_state(self):
        import torch

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

        class Runtime:
            @staticmethod
            def check_frozen(*_args):
                return None

        net = Net()
        initial = {
            name: value.detach().clone() for name, value in net.state_dict().items()
        }
        with torch.no_grad():
            net.adapter.add_(4.0)
        restore_initial_state(net, initial, Runtime(), None, [], None)
        self.assertTrue(net.adapter.detach().equal(initial["adapter"]))

    def test_final_pin_verification_allows_only_exact_identity_metadata(self):
        script = """
import json, os, sys
from pathlib import Path
import runner
_, runtime = runner.import_parent()
manifest = json.loads((runner.HERE / 'manifest.json').read_text())
path = (runner.HERE / manifest['h150_sealed_identity_manifest']).resolve()
expected = manifest['h150_sealed_identity_sha256']
pins = {
    'inputs': {str(path.relative_to(runner.REPO_ROOT)): expected},
    'assets': {}, 'packages': {}, 'source_sha256': {},
    'parent_source_sha256': {}, 'reference_source_sha256': {},
    'h142_runner_reference': str(runner.H142_RUNNER),
    'h142_runner_reference_sha256': runner.H142_RUNNER_SHA,
    'h150_sealed_identity_manifest': manifest['h150_sealed_identity_manifest'],
    'h150_sealed_identity_sha256': expected,
}
audit = runner.identity_manifest_audit(runtime, pins, runner.HERE, {str(path)})
sys.addaudithook(audit.check)
runner.verify(pins)
for args in ((str(path) + '.lookalike', 'r', 0), (str(path), 'w', 0)):
    try:
        audit.check('open', args)
    except RuntimeError:
        continue
    raise AssertionError('unapproved identity path access was admitted')
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=HERE,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
