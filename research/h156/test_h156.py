"""Data-free unit tests for H156 family-grouped cross-fit early stopping and capacity sweep."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as stat
import monitor as mon


def synthetic_rows(n_fams=205, per_fam=2):
    rows = []
    for f in range(n_fams):
        for c in range(per_fam):
            rows.append(
                dict(
                    context_sha256=f"cid_{f:03d}_{c}",
                    family_id=f"fam_{f:03d}",
                )
            )
    return rows


class TestH156(unittest.TestCase):
    def test_nested_family_partition_has_no_overlap(self):
        rows = synthetic_rows(205, 2)
        all_fams = {r["family_id"] for r in rows}
        assessed = set()
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
            self.assertEqual(f_fit | f_sel | f_out, all_fams)
            assessed |= f_out
        self.assertEqual(assessed, all_fams)

    def test_width_arms_and_multipliers(self):
        self.assertEqual(stat.WIDTH_ARMS, ("W05", "W10", "W20"))
        self.assertEqual(stat.WIDTH_MULTS, {"W05": 0.5, "W10": 1.0, "W20": 2.0})
        self.assertEqual(stat.SEEDS, (156101, 156102, 156103))
        self.assertEqual(stat.FOLD_SEED, 156140)
        self.assertEqual(stat.BOOT_SEED, 156141)

    def test_choose_cross_fit_and_lcb_fallback(self):
        rows = synthetic_rows(25, 2)
        base = np.ones((len(rows), len(stat.STEPS)), dtype=np.float64)
        res_flat = stat.choose(rows, base, seed=156141, r882_col=base[:, 0])
        self.assertEqual(res_flat["selected_step"], 0)
        self.assertEqual(res_flat["lcb_selected_step"], 0)

        panel = base.copy()
        panel[:, 2] += 0.025
        res_pos = stat.choose(rows, panel, seed=156141, r882_col=base[:, 0])
        self.assertEqual(res_pos["selected_step"], 20)
        self.assertEqual(res_pos["lcb_selected_step"], 20)

    def test_choose_joint_selects_best_width_and_step(self):
        rows = synthetic_rows(25, 2)
        r882_col = np.ones(len(rows), dtype=np.float64)
        panels = {
            "W05": np.ones((len(rows), 5), dtype=np.float64) - 0.01,
            "W10": np.ones((len(rows), 5), dtype=np.float64),
            "W20": np.ones((len(rows), 5), dtype=np.float64),
        }
        j_flat = stat.choose_joint(rows, panels, r882_col, seed=156141)
        self.assertEqual(j_flat["selected_arm"], "W10")
        self.assertEqual(j_flat["selected_step"], 0)

        panels["W05"][:, 1] = 1.015
        panels["W20"][:, 3] = 1.040
        j_pos = stat.choose_joint(rows, panels, r882_col, seed=156141)
        self.assertEqual(j_pos["selected_arm"], "W20")
        self.assertEqual(j_pos["selected_step"], 30)
        self.assertAlmostEqual(j_pos["selected_gain"], 0.040, places=9)

    def test_holm_stepdown_adjustment(self):
        pvals = {
            "W05_XFIT": 0.04,
            "W10_XFIT": 0.01,
            "W20_XFIT": 0.02,
            "XFIT_JOINT": 0.20,
        }
        adj = stat.holm(pvals)
        self.assertAlmostEqual(adj["W10_XFIT"], 0.04)
        self.assertAlmostEqual(adj["W20_XFIT"], 0.06)
        self.assertAlmostEqual(adj["W05_XFIT"], 0.08)
        self.assertAlmostEqual(adj["XFIT_JOINT"], 0.20)

    def test_elo_anchor_and_ordering(self):
        rows = synthetic_rows(20, 2)
        ties = np.ones((len(rows), 5), dtype=np.float64)
        elos_tie = stat.elo(rows, ties)
        for val in elos_tie:
            self.assertAlmostEqual(val, 1500.0, places=6)

        mat = np.column_stack(
            [
                np.zeros(len(rows)),
                np.full(len(rows), -0.05),
                np.full(len(rows), 0.01),
                np.full(len(rows), 0.05),
                np.full(len(rows), 0.05),
            ]
        )
        elos = stat.elo(rows, mat)
        self.assertAlmostEqual(elos[0], 1500.0, places=6)
        self.assertLess(elos[1], 1500.0)
        self.assertGreater(elos[2], 1500.0)
        self.assertGreater(elos[3], elos[2])

    def test_batch_plan_closure_and_reproducibility(self):
        fit_cids = {f"cid_{i:03d}" for i in range(30)}
        orders = {c: np.arange(32, dtype=np.int64) for c in fit_cids}
        p1 = stat.batch_plan(fit_cids, orders, 156101)
        p2 = stat.batch_plan(fit_cids, orders, 156101)
        self.assertEqual(len(p1), 40)
        for (s1, c1, i1), (s2, c2, i2) in zip(p1, p2):
            self.assertEqual(s1, s2)
            self.assertEqual(c1, c2)
            self.assertTrue(set(c1) <= fit_cids)
            for a, b in zip(i1, i2):
                np.testing.assert_array_equal(a[:4], np.arange(4))
                np.testing.assert_array_equal(a, b)

    def test_summarize_and_verdict_gates(self):
        rows = synthetic_rows(25, 2)
        base = np.ones((3, len(rows), 5), dtype=np.float64)
        s_neutral = stat.summarize(rows, base)
        sels_neutral = [
            {
                "W05": {"selected_step": 0},
                "W10": {"selected_step": 0},
                "W20": {"selected_step": 0},
                "XFIT_JOINT": {"selected_step": 0},
            }
        ]
        self.assertEqual(
            stat.verdict(s_neutral, s_neutral, sels_neutral), "RULE_NEUTRAL"
        )

        pos = base.copy()
        pos[:, :, 3] += 0.05
        pos[:, :, 4] += 0.05
        s_pos = stat.summarize(rows, pos)
        sels_pos = [
            {
                "W05": {"selected_step": 0},
                "W10": {"selected_step": 0},
                "W20": {"selected_step": 20},
                "XFIT_JOINT": {"selected_step": 20},
            }
        ]
        self.assertEqual(stat.verdict(s_pos, s_pos, sels_pos), "ADVANCE")

    def test_disk_used_pct_includes_reserved_blocks_and_95_boundary(self):
        # Match user-visible df Use% = used / (used + bavail)
        pct = mon.disk_used_pct(950_000_000_000, 50_000_000_000)
        self.assertAlmostEqual(pct, 95.0, places=9)
        self.assertTrue(pct >= mon.MAX_DISK_USED_PCT)
        pct_ok = mon.disk_used_pct(949_999_999_000, 50_000_001_000)
        self.assertLess(pct_ok, mon.MAX_DISK_USED_PCT)

    def test_configure_adapter_width_preserves_silu_and_exact_w10_parity(self):
        import torch
        import torch.nn as nn
        import run as h156_run

        class MockAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Linear(4, 4)
                for p in self.anchor.parameters():
                    p.requires_grad_(False)
                self.prog_412_bottleneck = nn.Sequential(
                    nn.Linear(412, 16), nn.LayerNorm(16), nn.SiLU()
                )
                self.cand_query_proj = nn.Sequential(
                    nn.Linear(66, 32), nn.LayerNorm(32), nn.SiLU()
                )
                self.ctx_bilinear_u = nn.Linear(32, 16, bias=False)
                self.prog_bilinear_v = nn.Linear(24, 16, bias=False)
                self.mm_prior_out = nn.Sequential(
                    nn.Linear(50, 24), nn.SiLU(), nn.Linear(24, 1)
                )
                self.traj_delta_head = nn.Sequential(
                    nn.Linear(67, 24), nn.SiLU(), nn.Linear(24, 1)
                )
                self.parent_delta_net = nn.Sequential(
                    nn.Linear(114, 48), nn.LayerNorm(48), nn.SiLU(), nn.Linear(48, 1)
                )

            def forward_probe(self, x_prog, x_sem, x_ctx, x_traj, x_par):
                b16 = self.prog_412_bottleneck(x_prog)
                cq = self.cand_query_proj(
                    torch.cat([x_sem, x_sem, b16, x_sem[:, :2]], dim=-1)
                )
                bi = self.ctx_bilinear_u(x_ctx) * self.prog_bilinear_v(x_sem)
                mm = self.mm_prior_out(torch.cat([cq, bi, x_sem[:, :2]], dim=-1))
                tr = self.traj_delta_head(x_traj)
                pa = self.parent_delta_net(x_par)
                return mm + tr + pa

        class MockNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter_10tok = MockAdapter()

        torch.manual_seed(156)
        net = MockNet()
        net.eval()
        r882_sd = {n: t.detach().clone() for n, t in net.state_dict().items()}
        names = [
            n
            for n, p in net.named_parameters()
            if n.startswith("adapter_10tok.")
            and not n.startswith("adapter_10tok.anchor.")
        ]
        x_prog = torch.randn(4, 412)
        x_sem = torch.randn(4, 24)
        x_ctx = torch.randn(4, 32)
        x_traj = torch.randn(4, 67)
        x_par = torch.randn(4, 114)
        with torch.no_grad():
            y_ref = net.adapter_10tok.forward_probe(x_prog, x_sem, x_ctx, x_traj, x_par)
            for wm in (0.5, 2.0, 1.0):
                h156_run.configure_adapter_width(net, r882_sd, names, wm, 156101)
                ad = net.adapter_10tok
                self.assertIsInstance(ad.prog_412_bottleneck[2], nn.SiLU)
                self.assertIsInstance(ad.cand_query_proj[2], nn.SiLU)
                self.assertIsInstance(ad.mm_prior_out[1], nn.SiLU)
                self.assertIsInstance(ad.traj_delta_head[1], nn.SiLU)
                self.assertIsInstance(ad.parent_delta_net[2], nn.SiLU)
            y_w10 = net.adapter_10tok.forward_probe(x_prog, x_sem, x_ctx, x_traj, x_par)
        self.assertEqual(float((y_ref - y_w10).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
