"""Data-free unit tests for H162 cosine warm-restart LR schedule, strict selector, and adapter forward parity."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as stat  # noqa: E402
import monitor  # noqa: E402
import run as h162_run  # noqa: E402


class TestH162(unittest.TestCase):
    def test_cosine_warm_restart_schedules_and_exact_lrmatch(self):
        r2 = stat.lr_schedule("COS_R2", 40)
        r4 = stat.lr_schedule("COS_R4", 40)
        ctrl = stat.lr_schedule("CTRL", 40)
        lrm = stat.lr_schedule("CTRL_LRMATCH", 40)
        self.assertEqual(len(r2), 40)
        self.assertEqual(len(r4), 40)
        self.assertEqual(len(ctrl), 40)
        self.assertEqual(len(lrm), 40)

        # Cycle starts at 2e-4 and cycle ends at 1e-6
        for start_idx in (0, 20):
            self.assertAlmostEqual(r2[start_idx], 2e-4, places=15)
        for end_idx in (19, 39):
            self.assertAlmostEqual(r2[end_idx], 1e-6, places=15)

        for start_idx in (0, 10, 20, 30):
            self.assertAlmostEqual(r4[start_idx], 2e-4, places=15)
        for end_idx in (9, 19, 29, 39):
            self.assertAlmostEqual(r4[end_idx], 1e-6, places=15)

        # Exact mean LR parity between COS_R2, COS_R4, and CTRL_LRMATCH
        mean_r2 = float(np.mean(r2))
        mean_r4 = float(np.mean(r4))
        mean_lrm = float(np.mean(lrm))
        self.assertAlmostEqual(mean_r2, 1.005e-4, places=15)
        self.assertAlmostEqual(mean_r4, 1.005e-4, places=15)
        self.assertAlmostEqual(mean_lrm, 1.005e-4, places=15)
        self.assertAlmostEqual(float(np.mean(ctrl)), 1.0e-4, places=15)

        self.assertEqual(stat.ARM_STEPS["COS_R2"], (0, 20, 40))
        self.assertEqual(stat.ARM_STEPS["COS_R4"], (0, 10, 20, 30, 40))
        self.assertEqual(stat.ARM_STEPS["CTRL"], (0, 10, 20, 30, 40))
        self.assertEqual(stat.ARM_STEPS["CTRL_LRMATCH"], (0, 10, 20, 30, 40))

    def test_fold_partition_disjointness_and_seed_162140(self):
        rows = [
            {"context_sha256": f"c_{i}_{j}", "family_id": f"fam_{i:03d}"}
            for i in range(205)
            for j in range(3)
        ]
        groups = stat.folds(rows)
        self.assertEqual(len(groups), 5)
        for g in groups:
            self.assertEqual(len(g), 41)
        for outer in range(5):
            fit, sel, out = stat.partition(rows, outer)
            f_fit = {r["family_id"] for r in fit}
            f_sel = {r["family_id"] for r in sel}
            f_out = {r["family_id"] for r in out}
            self.assertEqual(len(f_fit), 123)
            self.assertEqual(len(f_sel), 41)
            self.assertEqual(len(f_out), 41)
            self.assertFalse(f_fit & f_sel)
            self.assertFalse(f_fit & f_out)
            self.assertFalse(f_sel & f_out)

    def test_strict_inner_selector_margin_and_lcb_gate(self):
        rows = [{"context_sha256": f"c_{i}", "family_id": f"f_{i}"} for i in range(41)]
        base = np.ones((41, 1), dtype=np.float64)
        # Step 20 has +0.0005 (< 0.001 threshold), Step 40 has +0.01 (> 0.001 threshold, LCB > 0)
        panel = np.hstack([base, base + 0.0005, base + 0.01])
        dec = stat.choose(rows, panel, (0, 20, 40), seed=162141)
        self.assertEqual(dec["selected_step"], 40)
        self.assertFalse(dec["candidates"]["20"]["eligible_strict"])
        self.assertTrue(dec["candidates"]["20"]["eligible_unconstrained"])
        self.assertTrue(dec["candidates"]["40"]["eligible_strict"])

        # Sub-threshold gain everywhere falls back to step 0
        panel_sub = np.hstack([base, base + 0.0005, base + 0.0009])
        dec_sub = stat.choose(rows, panel_sub, (0, 20, 40), seed=162141)
        self.assertEqual(dec_sub["selected_step"], 0)
        self.assertEqual(dec_sub["unconstrained_selected_step"], 40)

    def test_summarize_and_advance_requires_beating_ctrl_lrmatch(self):
        rows = [{"context_sha256": f"c_{i}", "family_id": f"f_{i}"} for i in range(41)]
        # Case 1: COS_R2 beats r882 by +0.01, but ties CTRL_LRMATCH (+0.01) -> NO_ADVANCE
        panel_tie = np.zeros((3, 41, 5), dtype=np.float64)
        panel_tie[:, :, 1] = 0.005  # CTRL
        panel_tie[:, :, 2] = 0.010  # COS_R2
        panel_tie[:, :, 3] = 0.008  # COS_R4
        panel_tie[:, :, 4] = 0.010  # CTRL_LRMATCH
        s_tie = stat.summarize(rows, panel_tie)
        self.assertFalse(s_tie["arms"]["COS_R2"]["advance"])
        self.assertFalse(s_tie["arms"]["COS_R4"]["advance"])
        self.assertEqual(stat.verdict(s_tie, s_tie, []), "NO_ADVANCE")

        # Case 2: COS_R2 beats r882 by +0.015 AND beats CTRL_LRMATCH (+0.005) -> ADVANCE
        panel_win = np.zeros((3, 41, 5), dtype=np.float64)
        panel_win[:, :, 1] = 0.005  # CTRL
        panel_win[:, :, 2] = 0.015  # COS_R2
        panel_win[:, :, 3] = 0.002  # COS_R4
        panel_win[:, :, 4] = 0.005  # CTRL_LRMATCH
        s_win = stat.summarize(rows, panel_win)
        self.assertTrue(s_win["arms"]["COS_R2"]["advance"])
        self.assertFalse(s_win["arms"]["CTRL_LRMATCH"]["advance"])
        self.assertEqual(stat.verdict(s_win, s_win, []), "ADVANCE")

    def test_invalid_minus2_counts_and_elo_anchor(self):
        rows = [{"context_sha256": f"c_{i}", "family_id": f"f_{i}"} for i in range(20)]
        panel = np.ones((3, 20, 5), dtype=np.float64)
        panel[0, :2, 0] = -2.0
        panel[1, :3, 0] = -2.0
        panel[2, :1, 0] = -2.0
        panel[:, :4, 2] = -2.0
        s = stat.summarize(rows, panel)
        self.assertAlmostEqual(s["r882_elo"], 1500.0, places=9)
        self.assertEqual(s["r882_minus2_invalid_counts"]["per_seed"], [2, 3, 1])
        self.assertEqual(s["r882_minus2_invalid_counts"]["median"], 2.0)
        self.assertEqual(
            s["arms"]["COS_R2"]["minus2_invalid_counts"]["per_seed"],
            [4, 4, 4],
        )

    def test_forward_adapter_with_anchor_exact_parity_data_free(self):
        _, rt = h162_run.parent_executor()
        pins = __import__("json").loads((HERE / "manifest.json").read_text())
        net, guard, names, frozen, _ = rt.build(pins["checkpoint"])
        params = [dict(net.named_parameters())[n] for n in names]
        self.assertEqual(sum(p.numel() for p in params), 29211)

        torch.manual_seed(162101)
        b, k = 4, 12
        prog_b = torch.randn(b, k, 412, dtype=torch.float32)
        proj_b = torch.randn(b, k, 16, dtype=torch.float32)
        sos_b = torch.zeros(b, k, 2, dtype=torch.float32)
        tok_b = torch.randn(b, 10, 32, dtype=torch.float32)
        mask_b = torch.ones(b, 10, dtype=torch.bool)
        scale_b = torch.full((b,), 0.4208, dtype=torch.float32)
        rev_idx = torch.arange(4).reshape(1, -1).expand(b, -1)
        rev_raw_b = torch.randn(b, 4, dtype=torch.float32)
        rev_gate_b = torch.ones(b, 4, dtype=torch.float32)

        with torch.no_grad():
            full_out = net.adapter_10tok.predict_costs(
                ctx_feat=torch.zeros((b, 584), dtype=torch.float32),
                prog_412=prog_b,
                proj_16=proj_b,
                base_prior_cost=torch.zeros((b, k), dtype=torch.float32),
                scale=scale_b,
                sos_prior=sos_b,
                mm_tokens_10x32=tok_b,
                mm_mask_10=mask_b,
                revealed_indices=rev_idx,
                revealed_raw_f=rev_raw_b,
                revealed_gate=rev_gate_b,
            )["total_cost"]
            a_out = net.adapter_10tok.anchor.predict_costs(
                ctx_feat=torch.zeros((b, 584), dtype=torch.float32),
                prog_412=prog_b,
                proj_16=proj_b,
                base_prior_cost=torch.zeros((b, k), dtype=torch.float32),
                scale=scale_b,
                sos_prior=sos_b,
                revealed_indices=rev_idx,
                revealed_raw_f=rev_raw_b,
                revealed_gate=rev_gate_b,
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )
            cached_out = h162_run.forward_adapter_with_anchor(
                net,
                prog_b,
                sos_b,
                tok_b,
                mask_b,
                scale_b,
                rev_idx,
                rev_raw_b,
                rev_gate_b,
                a_out["total_cost"].detach().float(),
                a_out["eff_prior"].detach().float(),
            )
        max_diff = float(torch.max(torch.abs(full_out - cached_out)).item())
        self.assertEqual(max_diff, 0.0)
        rt.check_frozen(net, guard, names, frozen)

    def test_monitor_caps_and_disk_used_pct(self):
        self.assertEqual(monitor.MAX_THREADS, 4)
        self.assertEqual(monitor.MAX_RSS, 6_000_000_000)
        self.assertEqual(monitor.MAX_LOAD, 85.0)
        self.assertEqual(monitor.MIN_MEM, 16 * 1024**3)
        self.assertEqual(monitor.MIN_DISK, 40_000_000_000)
        self.assertEqual(monitor.MAX_DISK_USED_PCT, 95.0)
        self.assertAlmostEqual(monitor.disk_used_pct(90, 10), 90.0)


if __name__ == "__main__":
    unittest.main()
