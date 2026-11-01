"""Data-free unit tests for H166 C-Mixup kernel, gate-resolved target mixing, None-grad guard, and H148 cross-fit statistics."""

from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as stat  # noqa: E402
import monitor  # noqa: E402
import run as run_mod  # noqa: E402


class TestH166(unittest.TestCase):
    def test_cmixup_geometry_and_partner_kernel(self):
        # 4 synthetic fit-fold contexts: c0 and c1 have nearly identical utility distributions; c2 and c3 are far away.
        raw = {
            "c0": stat.context_target_summary(
                np.array([-1.2, -1.1, -1.0, -0.9]), np.array([True, True, True, True])
            ),
            "c1": stat.context_target_summary(
                np.array([-1.21, -1.09, -1.01, -0.89]),
                np.array([True, True, True, True]),
            ),
            "c2": stat.context_target_summary(
                np.array([0.5, 0.8, 1.1, 1.4]), np.array([True, True, True, True])
            ),
            "c3": stat.context_target_summary(
                np.array([1.5, 1.7, 1.9, 2.0]), np.array([True, True, True, True])
            ),
        }
        geom = stat.fit_fold_cmixup_geometry(list(raw.keys()), raw)
        self.assertGreater(geom["bandwidth"], 0.0)
        self.assertEqual(geom["pair_count"], 6)

        batch = ["c0", "c1", "c2", "c3"]
        p_gauss = stat.cmixup_partner_probs(
            batch, geom["standardized"], geom["bandwidth"], "gaussian"
        )
        self.assertEqual(p_gauss.shape, (4, 4))
        np.testing.assert_allclose(np.diag(p_gauss), np.zeros(4), atol=0.0)
        np.testing.assert_allclose(p_gauss.sum(axis=1), np.ones(4), atol=1e-12)
        # c0 is much closer to c1 than to c2 or c3
        self.assertGreater(p_gauss[0, 1], p_gauss[0, 2])
        self.assertGreater(p_gauss[0, 2], p_gauss[0, 3])

        p_unif = stat.cmixup_partner_probs(
            batch, geom["standardized"], geom["bandwidth"], "uniform"
        )
        np.testing.assert_allclose(np.diag(p_unif), np.zeros(4), atol=0.0)
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertAlmostEqual(p_unif[i, j], 1.0 / 3.0, places=12)

        # MIX02 and RANDMIX02 share identical Beta(0.2, 0.2) lambda draws at the same (seed, outer, step)
        plan_mix02 = stat.sample_cmixup_plan(
            batch, geom["standardized"], geom["bandwidth"], "MIX02", 166101, 0, 1
        )
        plan_rand02 = stat.sample_cmixup_plan(
            batch, geom["standardized"], geom["bandwidth"], "RANDMIX02", 166101, 0, 1
        )
        np.testing.assert_allclose(
            plan_mix02["lambdas"], plan_rand02["lambdas"], rtol=0.0, atol=0.0
        )
        self.assertTrue(np.all(plan_mix02["partners"] != np.arange(4)))
        self.assertTrue(np.all(plan_rand02["partners"] != np.arange(4)))

        plan_ctrl = stat.sample_cmixup_plan(
            batch, geom["standardized"], geom["bandwidth"], "CTRL", 166101, 0, 1
        )
        np.testing.assert_array_equal(plan_ctrl["partners"], np.arange(4))
        np.testing.assert_allclose(plan_ctrl["lambdas"], np.ones(4), atol=0.0)

    def test_valid_invalid_gate_resolved_convex_target_mixup(self):
        import torch

        # Red-team counterexample: slate 0 has valid cost -1.0 (gate=True -> target -1.0),
        # slate 1 has failed-gate raw fitness 99.0 (gate=False -> effective loss target 2.0).
        # With lambda = 0.5 and partners = [1, 0], convex effective target must be [[0.5], [0.5]].
        progs = np.zeros((2, 1, 4), dtype=np.float32)
        sos = np.zeros((2, 1, 2), dtype=np.float32)
        tok = np.zeros((2, 10, 32), dtype=np.float32)
        mask = np.ones((2, 10), dtype=bool)
        rev_raw = np.array([[-1.0], [99.0]], dtype=np.float32)
        rev_gate = np.array([[1.0], [0.0]], dtype=np.float32)
        a_tot = np.zeros((2, 1), dtype=np.float32)
        a_eff = np.zeros((2, 1), dtype=np.float32)
        fitness = np.array([[-1.0], [99.0]], dtype=np.float32)
        gate = np.array([[True], [False]], dtype=bool)
        partners = np.array([1, 0], dtype=np.int64)
        lambdas = np.array([0.5, 0.5], dtype=np.float64)

        np_mix = stat.mix_slate_arrays(
            progs,
            sos,
            tok,
            mask,
            rev_raw,
            rev_gate,
            a_tot,
            a_eff,
            fitness,
            gate,
            partners,
            lambdas,
        )
        np.testing.assert_allclose(
            np_mix["fitness"], np.array([[0.5], [0.5]]), atol=1e-7
        )
        np.testing.assert_allclose(
            np_mix["rev_raw"], np.array([[0.5], [0.5]]), atol=1e-7
        )
        self.assertTrue(np.all(np_mix["gate"]))

        t_mix = run_mod.mix_slate_tensors(
            torch.as_tensor(progs),
            torch.as_tensor(sos),
            torch.as_tensor(tok),
            torch.as_tensor(mask),
            torch.as_tensor(rev_raw),
            torch.as_tensor(rev_gate),
            torch.as_tensor(a_tot),
            torch.as_tensor(a_eff),
            torch.as_tensor(fitness),
            torch.as_tensor(gate),
            partners,
            lambdas,
        )
        fit_m, gate_m = t_mix[8], t_mix[9]
        # Verify the effective target inside rt.loss: torch.where(gate_m, fit_m.clamp(-4.0, 2.0), 2.0)
        eff_in_loss = torch.where(
            gate_m, fit_m.clamp(-4.0, 2.0), torch.full_like(fit_m, 2.0)
        )
        np.testing.assert_allclose(
            eff_in_loss.numpy(), np.array([[0.5], [0.5]], dtype=np.float32), atol=1e-7
        )

        # Clipping-crossing control: valid fitness -10.0 (clamped to -4.0) and valid fitness +10.0 (clamped to +2.0)
        # Mixed at lambda = 0.5 must yield effective target -1.0 (not 0.0 from unclipped raw mixing).
        fitness_clip = np.array([[-10.0], [10.0]], dtype=np.float32)
        gate_clip = np.array([[True], [True]], dtype=bool)
        rev_raw_clip = np.array([[-10.0], [10.0]], dtype=np.float32)
        rev_gate_clip = np.array([[1.0], [1.0]], dtype=np.float32)
        np_clip = stat.mix_slate_arrays(
            progs,
            sos,
            tok,
            mask,
            rev_raw_clip,
            rev_gate_clip,
            a_tot,
            a_eff,
            fitness_clip,
            gate_clip,
            partners,
            lambdas,
        )
        np.testing.assert_allclose(
            np_clip["fitness"], np.array([[-1.0], [-1.0]]), atol=1e-7
        )
        t_clip = run_mod.mix_slate_tensors(
            torch.as_tensor(progs),
            torch.as_tensor(sos),
            torch.as_tensor(tok),
            torch.as_tensor(mask),
            torch.as_tensor(rev_raw_clip),
            torch.as_tensor(rev_gate_clip),
            torch.as_tensor(a_tot),
            torch.as_tensor(a_eff),
            torch.as_tensor(fitness_clip),
            torch.as_tensor(gate_clip),
            partners,
            lambdas,
        )
        np.testing.assert_allclose(
            t_clip[8].numpy(), np.array([[-1.0], [-1.0]], dtype=np.float32), atol=1e-7
        )

    def test_crossing_seed_control_median_of_seed_gains(self):
        # Red-team crossing-seed counterexample:
        # 2 families, 3 seeds; family deltas are [[+0.004, -0.004], [-0.004, +0.004], [+0.004, +0.004]].
        # True per-seed equal-family mean gains are [0.0, 0.0, +0.004], with median = 0.0 and advance = False.
        rows = [
            {"context_sha256": "c0", "family_id": "fam_0"},
            {"context_sha256": "c1", "family_id": "fam_1"},
        ]
        panel = np.zeros((3, 2, 5), dtype=np.float64)
        # Seed 0: fam_0 +0.004, fam_1 -0.004 -> mean 0.0
        panel[0, 0, 2] = 0.004
        panel[0, 1, 2] = -0.004
        # Seed 1: fam_0 -0.004, fam_1 +0.004 -> mean 0.0
        panel[1, 0, 2] = -0.004
        panel[1, 1, 2] = 0.004
        # Seed 2: fam_0 +0.004, fam_1 +0.004 -> mean +0.004
        panel[2, 0, 2] = 0.004
        panel[2, 1, 2] = 0.004

        summary = stat.summarize(rows, panel)
        mix02 = summary["arms"]["MIX02"]
        np.testing.assert_allclose(
            mix02["per_seed_equal_family_delta"], [0.0, 0.0, 0.004], atol=1e-12
        )
        self.assertAlmostEqual(mix02["delta"], 0.0, places=12)
        self.assertLessEqual(mix02["ci_low_95"], 0.0)
        self.assertFalse(mix02["advance"])
        self.assertEqual(stat.verdict(summary, summary, []), "NO_ADVANCE")

    def test_slate_level_rank_aligned_cmixup_and_step0_unmixed_parity(self):
        import torch
        import torch.nn.functional as F

        # Check batch_plan rank-orders both head (0..3) and tail (4..23) by ascending fitness
        fit_cids = [f"cid_{i:02d}" for i in range(12)]
        rng = np.random.default_rng(42)
        fitness_by_cid = {c: rng.normal(size=40) for c in fit_cids}
        orders = {c: np.argsort(fitness_by_cid[c]) for c in fit_cids}
        plan = stat.batch_plan(fit_cids, orders, 166101, steps=5, batch=8, top=24)
        self.assertEqual(len(plan), 5)
        for _, chosen, ixs in plan:
            self.assertEqual(len(chosen), 8)
            for cid, ix in zip(chosen, ixs, strict=True):
                self.assertEqual(ix.shape, (24,))
                slate_fit = fitness_by_cid[cid][ix]
                # Revealed 0..3 and unrevealed 4..23 are monotonically non-decreasing in cost (fitness)
                self.assertTrue(np.all(np.diff(slate_fit[:4]) >= -1e-12))
                self.assertTrue(np.all(np.diff(slate_fit[4:]) >= -1e-12))
                self.assertLessEqual(slate_fit[3], slate_fit[4] + 1e-12)

        # Verify mix_slate_tensors matches mix_slate_arrays
        b, k = 4, 24
        progs = rng.normal(size=(b, k, 16)).astype(np.float32)
        sos = np.zeros((b, k, 2), dtype=np.float32)
        tok = rng.normal(size=(b, 10, 32)).astype(np.float32)
        mask = rng.integers(0, 2, size=(b, 10)).astype(bool)
        rev_raw = rng.normal(size=(b, 4)).astype(np.float32)
        rev_gate = np.ones((b, 4), dtype=np.float32)
        a_tot = rng.normal(size=(b, k)).astype(np.float32)
        a_eff = rng.normal(size=(b, k)).astype(np.float32)
        fitness = rng.normal(size=(b, k)).astype(np.float32)
        gate = rng.integers(0, 2, size=(b, k)).astype(bool)
        gate[:, 0] = True
        partners = np.array([1, 2, 3, 0], dtype=np.int64)
        lambdas = np.array([0.25, 0.75, 0.4, 0.9], dtype=np.float64)

        np_mix = stat.mix_slate_arrays(
            progs,
            sos,
            tok,
            mask,
            rev_raw,
            rev_gate,
            a_tot,
            a_eff,
            fitness,
            gate,
            partners,
            lambdas,
        )
        t_mix = run_mod.mix_slate_tensors(
            torch.as_tensor(progs),
            torch.as_tensor(sos),
            torch.as_tensor(tok),
            torch.as_tensor(mask),
            torch.as_tensor(rev_raw),
            torch.as_tensor(rev_gate),
            torch.as_tensor(a_tot),
            torch.as_tensor(a_eff),
            torch.as_tensor(fitness),
            torch.as_tensor(gate),
            partners,
            lambdas,
        )
        np.testing.assert_allclose(t_mix[0].numpy(), np_mix["progs"], atol=1e-6)
        np.testing.assert_allclose(t_mix[2].numpy(), np_mix["tok"], atol=1e-6)
        np.testing.assert_array_equal(t_mix[3].numpy(), np_mix["mask"])
        np.testing.assert_allclose(t_mix[4].numpy(), np_mix["rev_raw"], atol=1e-6)
        np.testing.assert_allclose(t_mix[6].numpy(), np_mix["a_tot"], atol=1e-6)
        np.testing.assert_allclose(t_mix[8].numpy(), np_mix["fitness"], atol=1e-6)
        np.testing.assert_array_equal(t_mix[9].numpy(), np_mix["gate"])

        # Step-0 / unmixed forward parity with direct adapter forward
        class DummyAnchor:
            def predict_costs(self, prog_412, proj_16, scale, **kwargs):
                tot = prog_412[:, :, :4].sum(dim=-1) + proj_16[:, :, :2].sum(dim=-1)
                eff = prog_412[:, :, 4:8].sum(dim=-1) * scale.unsqueeze(-1)
                return {"total_cost": tot, "eff_prior": eff}

        class DummyAdapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = DummyAnchor()
                self.w = torch.nn.Linear(8, 1, bias=False)

            def encode_multimodal_context_and_candidates(
                self,
                mm_tokens_10x32,
                mm_mask_10,
                prog_412,
                sos_prior,
                active_modality_override_mask=None,
            ):
                ctx_pooled = mm_tokens_10x32.mean(dim=1)
                cand_q = prog_412[:, :, :8]
                sem24 = prog_412[:, :, :24]
                mm_prior_adj = self.w(cand_q).squeeze(-1) + sos_prior[:, :, 0] * 0.0
                return ctx_pooled, cand_q, sem24, mm_prior_adj

            def compute_adapter_delta_from_static_terms(
                self,
                static_terms,
                scale,
                revealed_indices,
                revealed_raw_f,
                revealed_gate,
            ):
                return (
                    static_terms["mm_prior_adj"]
                    + 0.1 * revealed_raw_f.mean(dim=-1, keepdim=True)
                    + 0.0 * scale.unsqueeze(-1)
                )

            def forward(
                self,
                prog_412,
                proj_16,
                mm_tokens_10x32,
                mm_mask_10,
                scale,
                sos_prior,
                revealed_indices,
                revealed_raw_f,
                revealed_gate,
            ):
                a_out = self.anchor.predict_costs(
                    prog_412=prog_412, proj_16=proj_16, scale=scale
                )
                ctx_pooled, cand_q, sem24, mm_prior_adj = (
                    self.encode_multimodal_context_and_candidates(
                        mm_tokens_10x32=mm_tokens_10x32,
                        mm_mask_10=mm_mask_10,
                        prog_412=prog_412,
                        sos_prior=sos_prior,
                    )
                )
                static = {
                    "eff_prior": a_out["eff_prior"] + mm_prior_adj,
                    "mm_prior_adj": mm_prior_adj,
                    "sem24": sem24,
                    "cand_q": cand_q,
                    "cand_q_norm": F.normalize(cand_q, p=2, dim=-1, eps=1e-8),
                    "ctx_pooled": ctx_pooled,
                }
                delta = self.compute_adapter_delta_from_static_terms(
                    static_terms=static,
                    scale=scale,
                    revealed_indices=revealed_indices,
                    revealed_raw_f=revealed_raw_f,
                    revealed_gate=revealed_gate,
                )
                return (a_out["total_cost"] + delta).float()

        class DummyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter_10tok = DummyAdapter()

        torch.manual_seed(166101)
        net = DummyNet()
        prog_t = torch.randn(3, 12, 412)
        proj_t = torch.randn(3, 12, 16)
        tok_t = torch.randn(3, 10, 32)
        mask_t = torch.ones(3, 10, dtype=torch.bool)
        scale_t = torch.full((3,), 0.4208)
        sos_t = torch.zeros(3, 12, 2)
        rev_idx = torch.arange(4).reshape(1, 4).expand(3, 4)
        rev_raw_f = torch.randn(3, 4)
        rev_gate = torch.ones(3, 4)

        with torch.no_grad():
            a_out = net.adapter_10tok.anchor.predict_costs(
                prog_412=prog_t, proj_16=proj_t, scale=scale_t
            )
            a_tot = a_out["total_cost"].detach().float()
            a_eff = a_out["eff_prior"].detach().float()
            expected = net.adapter_10tok.forward(
                prog_t,
                proj_t,
                tok_t,
                mask_t,
                scale_t,
                sos_t,
                rev_idx,
                rev_raw_f,
                rev_gate,
            )
            actual = run_mod.forward_adapter_with_anchor(
                net,
                prog_t,
                sos_t,
                tok_t,
                mask_t,
                scale_t,
                rev_idx,
                rev_raw_f,
                rev_gate,
                a_tot,
                a_eff,
            )
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(float((expected - actual).abs().max()), 0.0)

    def test_none_grad_handling_data_free(self):
        import torch

        p_active = torch.nn.Parameter(
            torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
        )
        p_unused = torch.nn.Parameter(torch.tensor([0.5, -0.5], dtype=torch.float32))
        params = [p_active, p_unused]
        names = ["adapter.active_w", "adapter.unused_w"]
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        opt.zero_grad(set_to_none=True)
        loss = (p_active * torch.tensor([3.0, 4.0, 0.0])).sum()
        loss.backward()
        self.assertIsNotNone(p_active.grad)
        self.assertIsNone(p_unused.grad)

        norm, missing = run_mod.sanitize_and_clip_grads(params, names, max_norm=1.0)
        self.assertAlmostEqual(norm, 5.0, places=5)
        self.assertEqual(missing, ["adapter.unused_w"])
        self.assertIsNone(p_unused.grad)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(p_active.grad)), 1.0, places=5
        )
        opt.step()

        # Non-finite gradient must fail closed
        p_active.grad = torch.tensor([float("nan"), 0.0, 0.0])
        with self.assertRaises(RuntimeError):
            run_mod.sanitize_and_clip_grads(params, names, max_norm=1.0)

    def test_family_folds_and_partition_disjoint(self):
        rows = [
            {"context_sha256": f"c_{i}_{j}", "family_id": f"fam_{i:03d}"}
            for i in range(205)
            for j in range(3)
        ]
        groups = stat.folds(rows, seed=stat.FOLD_SEED)
        self.assertEqual(len(groups), 5)
        for g in groups:
            self.assertEqual(len(g), 41)
        all_fams = [f for g in groups for f in g]
        self.assertEqual(len(set(all_fams)), 205)
        for outer in range(5):
            fit, sel, out = stat.partition(rows, outer, seed=stat.FOLD_SEED)
            f_fit = {r["family_id"] for r in fit}
            f_sel = {r["family_id"] for r in sel}
            f_out = {r["family_id"] for r in out}
            self.assertEqual(len(f_fit), 123)
            self.assertEqual(len(f_sel), 41)
            self.assertEqual(len(f_out), 41)
            self.assertFalse(f_fit & f_sel)
            self.assertFalse(f_fit & f_out)
            self.assertFalse(f_sel & f_out)

    def test_selector_gate_strict_unconstrained_and_best_nonzero(self):
        rows = [
            {"context_sha256": f"c_{i}", "family_id": f"fam_{i:02d}"}
            for i in range(41)
        ]
        base = np.zeros(41, dtype=np.float64)
        # Boundary tie (+0.001) rejected by strict selector, accepted by unconstrained selector
        panel_tie = np.column_stack(
            [base, base + 0.001, base + 0.0005, base, base]
        )
        res_tie = stat.choose(rows, panel_tie, (0, 10, 20, 30, 40), seed=166141)
        self.assertEqual(res_tie["selected_step"], 0)
        self.assertEqual(res_tie["unconstrained_selected_step"], 10)
        self.assertEqual(res_tie["best_nonzero_step"], 10)

        # Negative across all continuation steps: strict and unconstrained pick 0, best_nonzero picks least negative (20)
        panel_neg = np.column_stack(
            [base, base - 0.010, base - 0.002, base - 0.008, base - 0.015]
        )
        res_neg = stat.choose(rows, panel_neg, (0, 10, 20, 30, 40), seed=166141)
        self.assertEqual(res_neg["selected_step"], 0)
        self.assertEqual(res_neg["unconstrained_selected_step"], 0)
        self.assertEqual(res_neg["best_nonzero_step"], 20)

        # Clear positive gain (+0.015) accepted by strict selector
        panel_pos = np.column_stack(
            [base, base + 0.005, base + 0.015, base + 0.012, base + 0.010]
        )
        res_pos = stat.choose(rows, panel_pos, (0, 10, 20, 30, 40), seed=166141)
        self.assertEqual(res_pos["selected_step"], 20)
        self.assertEqual(res_pos["unconstrained_selected_step"], 20)
        self.assertEqual(res_pos["best_nonzero_step"], 20)

    def test_summarize_and_verdict_rule(self):
        rows = [
            {"context_sha256": f"c_{i}", "family_id": f"fam_{i:02d}"}
            for i in range(41)
        ]
        # Shape: (3 seeds, 41 contexts, 5 arms: r882, CTRL, MIX02, MIX04, RANDMIX02)
        panel = np.zeros((3, 41, 5), dtype=np.float64)
        panel[:, :, 0] = 1.0
        panel[:, :, 1] = 1.001  # CTRL
        panel[:, :, 2] = 1.015  # MIX02 (+0.015 vs r882)
        panel[:, :, 3] = 1.000  # MIX04
        panel[:, :, 4] = 1.002  # RANDMIX02
        summary = stat.summarize(rows, panel)
        self.assertTrue(summary["arms"]["MIX02"]["advance"])
        self.assertLess(summary["arms"]["MIX02"]["p_holm"], 0.05)
        self.assertFalse(summary["arms"]["MIX04"]["advance"])
        self.assertAlmostEqual(summary["r882_elo"], 1500.0, places=9)
        self.assertGreater(summary["arms"]["MIX02"]["elo"], 1500.0)
        self.assertEqual(stat.verdict(summary, summary, []), "ADVANCE")

        panel_null = np.ones((3, 41, 5), dtype=np.float64)
        summary_null = stat.summarize(rows, panel_null)
        self.assertEqual(stat.verdict(summary_null, summary_null, []), "NO_ADVANCE")

    def test_monitor_sample_includes_supervisor_process(self):
        import os
        import psutil

        self.assertAlmostEqual(monitor.disk_used_pct(95, 5), 95.0)
        self.assertAlmostEqual(monitor.disk_used_pct(50, 50), 50.0)
        cur = psutil.Process(os.getpid())
        snap = monitor.sample(cur)
        self.assertIn(os.getpid(), snap["pids"])
        self.assertGreater(snap["rss_bytes"], 0)
        self.assertGreaterEqual(snap["threads"], 1)


if __name__ == "__main__":
    unittest.main()
