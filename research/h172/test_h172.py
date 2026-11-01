#!/usr/bin/env python3
"""Data-free unit tests for H172 (Anchored Convex Output-Layer L-BFGS Refit + Bagging + Placebo)."""

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import psutil
import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import analysis  # noqa: E402
import monitor  # noqa: E402
import run as run_mod  # noqa: E402


class TestH172(unittest.TestCase):
    def test_outer_and_inner_cv_family_partition_disjointness(self) -> None:
        rows = [
            {"context_sha256": f"c_{i:04d}", "family_id": f"fam_{i % 205:03d}"}
            for i in range(615)
        ]
        groups = analysis.folds(rows, seed=analysis.FOLD_SEED)
        self.assertEqual([len(g) for g in groups], [41, 41, 41, 41, 41])
        self.assertEqual(len(set(sum(groups, []))), 205)

        for outer in range(5):
            fit, assess = analysis.partition(rows, outer, groups=groups)
            fit_fams = {r["family_id"] for r in fit}
            assess_fams = {r["family_id"] for r in assess}
            self.assertEqual(len(fit_fams), 164)
            self.assertEqual(len(assess_fams), 41)
            self.assertEqual(fit_fams & assess_fams, set())

            val_fams_by_seed = {}
            for seed in analysis.SEEDS:
                icv = analysis.inner_cv_folds(fit, seed=seed, outer=outer, n_inner=5)
                self.assertEqual([len(vf) for _, _, vf in icv], [33, 33, 33, 33, 32])
                for itr, iva, vf in icv:
                    itr_fams = {r["family_id"] for r in itr}
                    iva_fams = {r["family_id"] for r in iva}
                    self.assertEqual(itr_fams & iva_fams, set())
                    self.assertEqual(iva_fams & assess_fams, set())
                    self.assertEqual(sorted(iva_fams), vf)
                val_fams_by_seed[seed] = [vf for _, _, vf in icv]
            self.assertNotEqual(val_fams_by_seed[172101], val_fams_by_seed[172102])
            self.assertNotEqual(val_fams_by_seed[172102], val_fams_by_seed[172103])

    def test_per_context_live_loss_matches_rt_loss(self) -> None:
        _, rt = run_mod.parent_executor()
        g = torch.Generator().manual_seed(172999)
        b, k = 11, 24
        pred = torch.randn((b, k), generator=g, dtype=torch.float64)
        fitness = torch.randn((b, k), generator=g, dtype=torch.float64) * 5.0
        fitness[0, 0] = -12.0
        fitness[0, 1] = 9.5
        gate = torch.rand((b, k), generator=g) > 0.2
        gate[1, :] = True

        target = run_mod.resolve_slate_target_tensor(fitness, gate)
        vec = run_mod.per_context_live_losses(pred, target)
        self.assertEqual(tuple(vec.shape), (b,))
        ref = rt.loss(pred.to(torch.float32), fitness.to(torch.float32), gate)
        self.assertAlmostEqual(float(vec.mean().item()), float(ref.item()), places=6)

        w_unif = torch.ones((b,), dtype=torch.float64) / float(b)
        self.assertAlmostEqual(
            float((w_unif * vec).sum().item()), float(vec.mean().item()), places=12
        )

    def test_anchored_convex_lbfgs_convergence_and_monotone_drift(self) -> None:
        g = torch.Generator().manual_seed(172555)
        n_ctx, n_cand = 24, 32
        c0 = torch.randn((n_ctx, n_cand), generator=g, dtype=torch.float64) * 0.2
        x_raw = torch.randn((n_ctx, n_cand, 48), generator=g, dtype=torch.float64) * 0.1
        x48 = x_raw - x_raw.mean(dim=1, keepdim=True)
        true_dw = torch.randn((48,), generator=g, dtype=torch.float64) * 0.15
        tgt = (c0 + torch.matmul(x48, true_dw)).clamp(-4.0, 2.0)
        ptgt = F.softmax(-tgt / 0.2, dim=-1)
        ws = torch.ones((n_ctx,), dtype=torch.float64) / float(n_ctx)
        size_batches = [(c0, x48, tgt, ptgt, ws)]

        sols = run_mod.solve_lambda_path(size_batches)
        self.assertTrue(sols["inf"]["converged"])
        self.assertEqual(sols["inf"]["l2_drift"], 0.0)
        self.assertTrue(np.all(sols["inf"]["delta99"] == 0.0))

        prev_l2 = -1.0
        for lbl in analysis.FINITE_LAMBDA_LABELS:
            res = sols[lbl]
            self.assertTrue(res["converged"], msg=f"L-BFGS did not converge for {lbl}")
            self.assertLessEqual(res["grad_norm"], analysis.LBFGS_GRAD_TOL)
            self.assertLessEqual(res["final_obj"], res["initial_obj"] + 1e-12)
            d99 = res["delta99"]
            self.assertEqual(d99.shape, (99,))
            self.assertEqual(float(d99[24]), 0.0)
            self.assertEqual(float(d99[49]), 0.0)
            self.assertTrue(np.all(d99[50:] == 0.0))
            self.assertGreater(res["l2_drift"], prev_l2)
            prev_l2 = res["l2_drift"]

    def test_none_grad_handling_data_free(self) -> None:
        p_active = torch.nn.Parameter(torch.ones((1, 24), dtype=torch.float32))
        p_bias = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        p_unused = torch.nn.Parameter(torch.ones((1, 48), dtype=torch.float32))
        loss = (p_active * 2.0).sum() + (p_bias - p_bias.mean()).sum()
        loss.backward()
        self.assertIsNone(p_unused.grad)
        gn, missing = run_mod.sanitize_and_verify_head_grads(
            [p_active, p_bias, p_unused],
            [
                "adapter_10tok.mm_prior_out.2.weight",
                "adapter_10tok.mm_prior_out.2.bias",
                "adapter_10tok.parent_delta_net.3.weight",
            ],
        )
        self.assertEqual(missing, ["adapter_10tok.parent_delta_net.3.weight"])
        self.assertAlmostEqual(gn, math.sqrt(24.0 * 4.0), places=5)

        p_bad = torch.nn.Parameter(torch.ones((1, 4), dtype=torch.float32))
        p_bad.grad = torch.tensor([[1.0, float("nan"), 0.0, 0.0]], dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            run_mod.sanitize_and_verify_head_grads([p_bad], ["bad"])

    def test_strict_lambda_selection_and_tie_rejection(self) -> None:
        rows = [
            {"context_sha256": f"c_{i:02d}", "family_id": f"fam_{i:02d}"}
            for i in range(25)
        ]
        ctrl = [0.50 for _ in rows]
        cands_tie = {
            "1e4": [0.50 for _ in rows],
            "1e3": [0.5005 for _ in rows],  # +0.0005 <= +0.001 margin
            "1e2": [
                0.520 if i < 13 else 0.480 for i in range(25)
            ],  # mean +0.0008, ci_low_95 <= 0
            "1e1": [0.50 for _ in rows],
            "1": [0.50 for _ in rows],
            "1e-1": [0.50 for _ in rows],
        }
        sel_tie = analysis.select_lambda(ctrl, cands_tie, rows, seed=172101, outer=0)
        self.assertEqual(sel_tie["selected_lambda"], "inf")
        self.assertEqual(sel_tie["unconstrained_selected_lambda"], "1e2")

        cands_win = dict(cands_tie)
        cands_win["1e1"] = [0.505 for _ in rows]
        sel_win = analysis.select_lambda(ctrl, cands_win, rows, seed=172101, outer=0)
        self.assertEqual(sel_win["selected_lambda"], "1e1")
        self.assertEqual(sel_win["best_finite_lambda"], "1e1")

    def test_crossing_seed_control_and_holm2_and_placebo_gate(self) -> None:
        # 20 families, 1 context per family.
        # Crossing seed matrix on REFIT_CV: seed0=[+0.004,-0.004], seed1=[-0.004,+0.004], seed2=[+0.004,+0.004]
        # Wait: per-seed equal-family means are [0.0, 0.0, +0.004] -> median over seeds is 0.0!
        # And if we test family-crossing where per-family medians across seeds cancel:
        rows = [
            {"context_sha256": f"c_{i:02d}", "family_id": f"fam_{i:02d}"}
            for i in range(20)
        ]
        panel_cross = np.full((3, 20, 5), 0.50, dtype=np.float64)
        pattern = [(+0.004, -0.004), (-0.004, +0.004), (+0.004, +0.004)]
        for s in range(3):
            pos, neg = pattern[s]
            for i in range(20):
                panel_cross[s, i, 2] = 0.50 + (pos if (i % 2 == 0) else neg)
        rep_cross = analysis.summarize(rows, panel_cross, seed=analysis.BOOT_SEED)
        self.assertAlmostEqual(rep_cross["arms"]["REFIT_CV"]["delta"], 0.0, places=12)
        self.assertFalse(rep_cross["arms"]["REFIT_CV"]["advance"])

        # Strong real gain on REFIT_CV passes when REFIT_SHUF is neutral.
        panel_win = np.full((3, 20, 5), 0.50, dtype=np.float64)
        panel_win[:, :, 2] = 0.505  # REFIT_CV +0.005
        rep_win = analysis.summarize(rows, panel_win, seed=analysis.BOOT_SEED)
        self.assertTrue(rep_win["arms"]["REFIT_CV"]["advance"])
        self.assertFalse(rep_win["arms"]["REFIT_BAG"]["advance"])
        self.assertFalse(rep_win["placebo_advance_leak_flag"])

        # If REFIT_SHUF placebo also passes, placebo_advance_leak_flag triggers and advance is blocked.
        panel_leak = panel_win.copy()
        panel_leak[:, :, 4] = 0.505  # REFIT_SHUF +0.005
        rep_leak = analysis.summarize(rows, panel_leak, seed=analysis.BOOT_SEED)
        self.assertTrue(rep_leak["placebo_advance_leak_flag"])
        self.assertFalse(rep_leak["arms"]["REFIT_CV"]["advance"])

    def test_within_family_target_permutation_and_pseudoreplication_report(
        self,
    ) -> None:
        rng = np.random.default_rng(172777)
        fitness = rng.standard_normal(32).astype(np.float32)
        gate = rng.random(32) > 0.25
        f_p, g_p, perm = analysis.permute_targets_within_family(
            fitness, gate, seed=172101, outer=0, cid="ctx_0"
        )
        self.assertEqual(f_p.shape, fitness.shape)
        self.assertFalse(np.array_equal(perm, np.arange(32)))
        self.assertTrue(np.allclose(np.sort(fitness), np.sort(f_p)))
        self.assertEqual(int(gate.sum()), int(g_p.sum()))

        fit_rows = [
            {"context_sha256": f"c_{i}", "family_id": f"f_{i // 2}"} for i in range(10)
        ]
        w_ctx, fam_mult = analysis.family_bootstrap_context_weights(
            fit_rows, seed=172101, outer=0, bag_idx=0
        )
        self.assertAlmostEqual(float(w_ctx.sum()), 1.0, places=12)
        self.assertEqual(sum(fam_mult.values()), 5)

        recs = []
        for s_idx, seed in enumerate(analysis.SEEDS):
            recs.append(
                dict(
                    outer_fold=0,
                    seed=seed,
                    decisions={
                        "REFIT_CV": {"selected_lambda": "inf"},
                        "REFIT_BAG": {"selected_lambda": "1e2"},
                        "REFIT_SHUF": {"selected_lambda": "inf"},
                    },
                    head_sha256={
                        "REFIT_CV": {"strict": "sha_zero"},
                        "REFIT_BAG": {"strict": f"sha_bag_{s_idx}"},
                        "REFIT_SHUF": {"strict": "sha_zero"},
                    },
                    assess_choices={
                        "REFIT_CV": {
                            "strict": [0, 1],
                            "unconstrained": [0, 1],
                            "best_finite": [1, 1],
                        },
                        "REFIT_BAG": {
                            "strict": [s_idx, 1],
                            "unconstrained": [s_idx, 1],
                            "best_finite": [s_idx, 1],
                        },
                        "REFIT_SHUF": {
                            "strict": [0, 1],
                            "unconstrained": [0, 1],
                            "best_finite": [0, s_idx],
                        },
                    },
                )
            )
        pseudo = analysis.pseudoreplication_report(recs)
        self.assertTrue(pseudo["REFIT_CV"]["all_seeds_identical_strict_choices"])
        self.assertFalse(pseudo["REFIT_BAG"]["all_seeds_identical_strict_choices"])

    def test_monitor_sample_includes_supervisor_process(self) -> None:
        dummy_proc = mock.MagicMock()
        dummy_proc.pid = os.getpid()
        dummy_proc.children.return_value = []
        dummy_proc.memory_info.return_value = psutil.Process(os.getpid()).memory_info()
        dummy_proc.num_threads.return_value = psutil.Process(os.getpid()).num_threads()
        s = monitor.sample(dummy_proc)
        self.assertGreaterEqual(s["total_threads"], 1)
        self.assertGreater(s["aggregate_rss_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
