"""Data-free unit tests for H161: robust per-context normalized/Huber loss + family-balanced sampling."""

from pathlib import Path
import sys
import unittest
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import monitor  # noqa: E402
import runner  # noqa: E402
import runtime  # noqa: E402
import science  # noqa: E402


def _synthetic_rows(n_fams=25, ctx_per_fam=2):
    rows = []
    for f in range(n_fams):
        for c in range(ctx_per_fam):
            rows.append(
                dict(
                    context_sha256=f"cid_{f:03d}_{c:02d}",
                    family_id=f"fam_{f:03d}",
                    dl_split="DL-Train",
                    task="regression",
                    has_meta_test=False,
                    source_split="train",
                )
            )
    return rows


class TestH161(unittest.TestCase):
    def test_folds_and_partition_disjoint(self):
        rows = _synthetic_rows(25, 2)
        groups = science.folds(rows)
        self.assertEqual(len(groups), 5)
        self.assertEqual(sum(len(g) for g in groups), 25)
        for outer in range(5):
            fit, select, assess = science.partition(rows, outer)
            f_fit = {r["family_id"] for r in fit}
            f_sel = {r["family_id"] for r in select}
            f_ass = {r["family_id"] for r in assess}
            self.assertEqual(len(f_fit), 15)
            self.assertEqual(len(f_sel), 5)
            self.assertEqual(len(f_ass), 5)
            self.assertFalse(f_fit & f_sel)
            self.assertFalse(f_fit & f_ass)
            self.assertFalse(f_sel & f_ass)
        bad = _synthetic_rows(10, 1)
        bad[0]["has_meta_test"] = True
        with self.assertRaises(ValueError):
            science.folds(bad)

    def test_batch_plan_and_fambal_batch_plan(self):
        rows = _synthetic_rows(20, 3)
        orders = {r["context_sha256"]: np.arange(256) for r in rows}
        fit_ids = [r["context_sha256"] for r in rows]
        plan = science.batch_plan(fit_ids, orders, science.SEEDS[0])
        self.assertEqual(len(plan), 40)
        for _, cids, ixs in plan:
            self.assertEqual(len(cids), 16)
            self.assertEqual(len(ixs), 16)
            self.assertTrue(all(len(ix) == 192 for ix in ixs))

        fplan = science.fambal_batch_plan(rows, orders, science.SEEDS[0])
        self.assertEqual(len(fplan), 40)
        fam_counts = {f"fam_{i:03d}": 0 for i in range(20)}
        for _, cids, ixs, step_fams, mbs in fplan:
            self.assertEqual(len(cids), 16)
            self.assertEqual(len(set(step_fams)), 16)
            self.assertEqual(len(mbs), 2)
            self.assertEqual(len(mbs[0][0]), 8)
            self.assertEqual(len(mbs[1][0]), 8)
            for f in step_fams:
                fam_counts[f] += 1
        self.assertLessEqual(max(fam_counts.values()) - min(fam_counts.values()), 1)

    def test_fold_median_scale_and_robust_loss_gradient_capping(self):
        base_losses = {"c1": 0.5, "c2": 1.0, "c3": 2.0, "c_excluded": 100.0}
        scale = science.fold_median_scale(["c1", "c2", "c3"], base_losses)
        self.assertAlmostEqual(scale, 1.0)

        pred = torch.tensor(
            [[0.0, 0.2, 0.4, 0.6], [10.0, -10.0, 5.0, -5.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        fitness = torch.tensor(
            [[0.0, 0.2, 0.4, 0.6], [-2.0, 2.0, -2.0, 2.0]], dtype=torch.float32
        )
        gate = torch.tensor([[True, True, True, True], [False, True, False, True]])
        l_vec = runtime.per_context_loss(pred, fitness, gate)
        self.assertEqual(l_vec.shape, (2,))
        self.assertGreater(float(l_vec[1]), 5.0)
        # Verify Huber derivative wrt normalized loss u_c is capped at 1.0 when u_c > 1.0
        u = torch.tensor([0.4, 8.0], dtype=torch.float32, requires_grad=True)
        h = torch.nn.functional.huber_loss(
            u, torch.zeros_like(u), reduction="mean", delta=1.0
        )
        h.backward()
        self.assertAlmostEqual(float(u.grad[0]), 0.4 / 2.0, places=6)
        self.assertAlmostEqual(float(u.grad[1]), 1.0 / 2.0, places=6)
        rob = runtime.robust_loss(pred, fitness, gate, fold_median_scale=1.0, delta=1.0)
        self.assertTrue(torch.isfinite(rob))

    def test_choose_strict_selection_and_tie_rejection(self):
        rows = _synthetic_rows(10, 2)
        base = np.ones((len(rows), 5), dtype=float)
        # Exact tie at 0.001 + 0.5e-12 is rejected
        near = base.copy()
        near[:, 2] += 0.001 + 0.5e-12
        d_near = science.choose(rows, near, 161141)
        self.assertFalse(d_near["accepted"])
        self.assertEqual(d_near["selected_step"], 0)
        # Uniform +0.01 gain across all families is accepted
        good = base.copy()
        good[:, 3] += 0.01
        d_good = science.choose(rows, good, 161141)
        self.assertTrue(d_good["accepted"])
        self.assertEqual(d_good["selected_step"], 30)

    def test_summarize_bt_elo_and_invalid_counts(self):
        rows = _synthetic_rows(10, 2)
        panels = np.ones((3, len(rows), 5), dtype=float)
        panels[:, :, 2] += 0.02  # CLIP better than r882
        inv = np.zeros((3, len(rows), 5), dtype=bool)
        inv[:, 0, 0] = True
        s = science.summarize(rows, panels, inv)
        self.assertAlmostEqual(s["median_elo"]["r882"], 1500.0, places=6)
        self.assertGreater(s["median_elo"]["CLIP"], 1500.0)
        self.assertEqual(s["invalid_counts_by_seed"]["r882"], [1, 1, 1])
        self.assertEqual(s["invalid_counts_by_seed"]["CLIP"], [0, 0, 0])

    def test_decide_holm_and_post_hoc_drift_matched_ctrl(self):
        rows = _synthetic_rows(15, 2)
        panels = np.ones((3, len(rows), 5), dtype=float)
        inv = np.zeros((3, len(rows), 5), dtype=bool)
        s0 = science.summarize(rows, panels, inv)
        d0 = science.decide(s0, {a: 0 for a in science.ARMS})
        self.assertEqual(d0["verdict"], "RULE_NEUTRAL")

        # Make CLIP win strongly vs r882 across all families and seeds
        panels_win = np.ones((3, len(rows), 6), dtype=float)
        panels_win[:, :, 2] += 0.02  # CLIP
        panels_win[:, :, 5] += 0.005  # CTRL_DRIFT
        inv_win = np.zeros((3, len(rows), 6), dtype=bool)
        arms_ext = list(science.ARMS) + ["CTRL_DRIFT"]
        s_win = science.summarize(rows, panels_win, inv_win, arms=arms_ext)
        nonzero = {"CTRL": 0, "CLIP": 5, "FAMBAL": 0, "CLIP_FAMBAL": 0, "CTRL_DRIFT": 5}
        # Without drift match valid -> NO_ADVANCE
        d_invalid = science.decide(
            s_win, nonzero, drift_match_receipt={"valid": False}, winner_arm="CLIP"
        )
        self.assertEqual(d_invalid["verdict"], "NO_ADVANCE")
        # With drift match valid and CLIP > CTRL_DRIFT -> ADVANCE
        d_valid = science.decide(
            s_win, nonzero, drift_match_receipt={"valid": True}, winner_arm="CLIP"
        )
        self.assertEqual(d_valid["verdict"], "ADVANCE")
        self.assertTrue(d_valid["treatment_arms"]["CLIP"]["advance"])

    def test_drift_lr_and_drift_match(self):
        lr = science.drift_lr([0.2, 0.22, 0.24], [0.4, 0.44, 0.48], base_lr=1e-4)
        self.assertAlmostEqual(lr, 5e-5)
        dm = science.drift_match([0.2, 0.2, 0.2], [0.21, 0.21, 0.21])
        self.assertTrue(dm["valid"])
        dm_bad = science.drift_match([0.2, 0.2, 0.2], [0.3, 0.3, 0.3])
        self.assertFalse(dm_bad["valid"])

    def test_baseline_gate_exact(self):
        vals = np.linspace(0.5, 1.5, 91)
        ref = {
            "utility": {"per_context": vals.tolist()},
            "best_utility": float(vals.mean()),
        }
        res = runner.baseline_gate(vals, ref)
        self.assertEqual(res["status"], "PASS")
        bad = vals.copy()
        bad[0] += 1e-12
        with self.assertRaises(RuntimeError):
            runner.baseline_gate(bad, ref)

    def test_audit_hook_refuses_protected_and_unlisted_paths(self):
        out_dir = Path("/home/ubuntu/jope-data/h161/custody/run")
        launch_path = str((out_dir.parent / "launch.json").resolve())
        allowed = {"/home/ubuntu/jope-data/h50/input/selected.json", launch_path}
        audit = runtime.Audit(allowed, out_dir)
        audit.check("open", (launch_path, "r", 0))
        self.assertIn(launch_path, audit.opened)
        for blocked in [
            "/home/ubuntu/jope-data/sealed/x.npz",
            "/home/ubuntu/jope-data/holdout/x.npz",
            "/home/ubuntu/jope-data/meta-test/x.npz",
            "/home/ubuntu/jope_plan/qualification.lock",
            "/home/ubuntu/jope-data/h50/input/unlisted.npz",
        ]:
            with self.assertRaises(RuntimeError):
                audit.check("open", (blocked, "r", 0))
        with self.assertRaises(RuntimeError):
            audit.check(
                "open", ("/home/ubuntu/jope-data/h50/input/selected.json", "w", 0)
            )

    def test_monitor_classify_boundaries(self):
        ok = dict(
            rss=3_000_000_000,
            threads=3,
            free=100_000_000_000,
            disk_used=0.90,
            scratch=100_000_000,
            load=40.0,
            available=32 * 1024**3,
        )
        self.assertEqual(monitor.classify(ok), ("run", None))
        for k, v, expected in [
            ("rss", 6_000_000_000, "RSS"),
            ("threads", 5, "TOTAL_THREADS"),
            ("free", 39_999_999_999, "DISK"),
            ("disk_used", 0.95, "DISK_USED"),
            ("scratch", 3_000_000_000, "SCRATCH"),
            ("load", 85.1, "HOST_PRESSURE"),
            ("available", 16 * 1024**3 - 1, "HOST_PRESSURE"),
        ]:
            bad = dict(ok, **{k: v})
            self.assertEqual(monitor.classify(bad), ("abort", expected))


if __name__ == "__main__":
    unittest.main()
