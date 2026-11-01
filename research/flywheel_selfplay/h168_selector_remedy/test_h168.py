"""Data-free unit tests for H168 prospective selector remedy, H156 W10/W20 width config, None-grad clipping, and resource guards."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

import monitor as mon
import runner as rn
import runtime as rt
import science as sc

HERE = Path(__file__).resolve().parent


def make_rows(n_fams=205, ctx_per_fam=1):
    out = []
    idx = 0
    for f in range(n_fams):
        for _ in range(ctx_per_fam):
            out.append(
                dict(
                    context_sha256=f"c{idx:04d}",
                    family_id=f"f{f:03d}",
                    dl_split="DL-Train",
                    has_meta_test=False,
                )
            )
            idx += 1
    return out


class MockAnchor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)


class MockAdapter10Tok(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = MockAnchor()
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


class MockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_10tok = MockAdapter10Tok()


class H168Tests(unittest.TestCase):
    def test_five_fold_family_exclusion_and_exact_assessment_cover(self):
        rows = make_rows(205, 2)
        assessed = []
        for outer in range(5):
            fit, sel, out = sc.partition(rows, outer)
            f_fit = {r["family_id"] for r in fit}
            f_sel = {r["family_id"] for r in sel}
            f_out = {r["family_id"] for r in out}
            self.assertEqual([len(f_fit), len(f_sel), len(f_out)], [123, 41, 41])
            self.assertFalse(f_fit & f_sel or f_fit & f_out or f_sel & f_out)
            assessed.extend(f_out)
        self.assertEqual(len(assessed), 205)
        self.assertEqual(len(set(assessed)), 205)

        for bad in (
            rows + [rows[0]],
            [dict(r, dl_split="DL-Val") for r in rows],
            [dict(r, has_meta_test=True) for r in rows],
            make_rows(204, 1),
        ):
            with self.assertRaises(ValueError):
                sc.folds(bad)

    def test_batch_plan_common_random_numbers_and_fit_closure(self):
        fit, _, _ = sc.partition(make_rows(205, 1), 0)
        cids = [r["context_sha256"] for r in fit]
        orders = {c: np.arange(48) for c in cids}
        p1 = sc.batch_plan(cids, orders, sc.SEEDS[0])
        p2 = sc.batch_plan(list(reversed(cids)), orders, sc.SEEDS[0])
        self.assertEqual(len(p1), 40)
        for (s1, ids1, ix1), (s2, ids2, ix2) in zip(p1, p2, strict=True):
            self.assertEqual(s1, s2)
            self.assertEqual(ids1, ids2)
            self.assertEqual(len(ids1), 8)
            self.assertLessEqual(set(ids1), set(cids))
            for a, b in zip(ix1, ix2, strict=True):
                self.assertTrue(np.array_equal(a, b))
                self.assertTrue(np.array_equal(a[:4], np.arange(4)))
                self.assertEqual(len(set(a)), 24)

    def test_old_vs_new_selector_diffuse_sparse_and_ties(self):
        r41 = make_rows(41, 1)

        # 1. Diffuse +0.005 effect: both OLD and NEW select step 10 (earliest tie-breaker)
        panel_diff = np.ones((41, 5))
        panel_diff[:, 1] += 0.005
        panel_diff[:, 2] += 0.005 + 5e-13
        d_diff = sc.choose_dual(r41, panel_diff, seed=580061)
        self.assertEqual(d_diff["old_selected_step"], 10)
        self.assertEqual(d_diff["new_selected_step"], 10)
        self.assertEqual(d_diff["relaxed_selected_step"], 10)
        self.assertEqual(d_diff["nonzero_selected_step"], 10)

        # 2. Sparse +0.005 mean (+0.205 on 1 family, 0 on 40 families):
        # OLD rejects (ci_low_95 == 0.0 because 36.33% of draws omit family 0),
        # NEW selects step 10 (mean gain == +0.005 > 0.001 + 1e-12)!
        panel_sparse = np.ones((41, 5))
        panel_sparse[0, 1] += 0.205
        d_sparse = sc.choose_dual(r41, panel_sparse, seed=580061)
        self.assertEqual(d_sparse["old_selected_step"], 0)
        self.assertAlmostEqual(d_sparse["candidates"]["10"]["ci_low_95"], 0.0)
        self.assertEqual(d_sparse["new_selected_step"], 10)
        self.assertAlmostEqual(d_sparse["new_selected_gain"], 0.005)
        self.assertEqual(d_sparse["relaxed_selected_step"], 10)
        self.assertEqual(d_sparse["nonzero_selected_step"], 10)

        # 3. Exact margin tie (+0.001) and within-TIE (+0.001 + 5e-13): both OLD and NEW reject (step 0)
        for gain in (0.0, 0.001, 0.001 + 5e-13):
            panel_tie = np.ones((41, 5))
            panel_tie[:, 1:] += gain
            d_tie = sc.choose_dual(r41, panel_tie, seed=580061)
            self.assertEqual(d_tie["old_selected_step"], 0)
            self.assertEqual(d_tie["new_selected_step"], 0)
            if gain > sc.TIE:
                self.assertEqual(d_tie["relaxed_selected_step"], 10)
            else:
                self.assertEqual(d_tie["relaxed_selected_step"], 0)

        # 4. All negative non-zero steps: OLD, NEW, RELAXED pick 0; NONZERO picks best non-zero step (10)
        panel_neg = np.zeros((41, 5))
        panel_neg[:, 1:] = [-0.01, -0.02, -0.03, -0.04]
        d_neg = sc.choose_dual(r41, panel_neg, seed=580061)
        self.assertEqual(d_neg["old_selected_step"], 0)
        self.assertEqual(d_neg["new_selected_step"], 0)
        self.assertEqual(d_neg["relaxed_selected_step"], 0)
        self.assertEqual(d_neg["nonzero_selected_step"], 10)
        self.assertAlmostEqual(d_neg["nonzero_selected_gain"], -0.01)

    def test_pinned_median_seed_estimand_and_equal_family_weighting(self):
        # 2 families: "big" (9 contexts) and "small" (1 context) -> equal family weight 0.5 each
        rows = [
            dict(
                context_sha256=f"c{i}",
                family_id="big" if i < 9 else "small",
                dl_split="DL-Train",
                has_meta_test=False,
            )
            for i in range(10)
        ]
        panel = np.zeros((3, 10, 5))
        # Seed 0: big +0.02, small -0.02 -> equal-family mean 0.0
        # Seed 1: big +0.01, small +0.01 -> equal-family mean +0.01
        # Seed 2: big -0.04, small +0.08 -> equal-family mean +0.02
        panel[0, :9, 2] = 0.02
        panel[0, 9, 2] = -0.02
        panel[1, :9, 2] = 0.01
        panel[1, 9, 2] = 0.01
        panel[2, :9, 2] = -0.04
        panel[2, 9, 2] = 0.08
        s = sc.summarize(rows, panel)
        c = s["contrasts"]["W10_NEW_minus_r882"]
        np.testing.assert_allclose(c["per_seed_gain"], [0.0, 0.01, 0.02])
        # Median of per-seed gains is 0.01, whereas legacy mean-of-family-seed-medians is mean(0.01, 0.01) = 0.01
        self.assertAlmostEqual(c["median_gain"], 0.01)
        self.assertEqual(c["positive_seeds"], 2)

    def test_canonical_bt_elo_anchor_and_dominance(self):
        ties = np.ones((5, 5)) - np.eye(5)
        np.testing.assert_allclose(sc.bt_ratings(ties, anchor=0), 1500.0)
        # Strict dominance with prior=0 clips strength at 1e6 -> 1500 + 400*log10(1e6) = 3900 vs anchor,
        # and anchor vs dominated -> 1500 + 400*6 = 3900 / 6300 span
        dom = np.zeros((2, 2))
        dom[1, 0] = 205.0
        r_prior0 = sc.bt_ratings(dom, anchor=0, prior=0.0)
        self.assertAlmostEqual(r_prior0[0], 1500.0)
        self.assertGreater(r_prior0[1], 3800.0)

    def test_holm_four_arm_and_mechanism_classification(self):
        rows = make_rows(41, 1)
        panel = np.zeros((3, 41, 5))
        # W10_NEW (col 2) gains +0.005 across all families & seeds; others stay at 0
        panel[:, :, 2] = 0.005
        summary = sc.summarize(rows, panel)
        nz = {"W10_OLD": 0, "W10_NEW": 5, "W20_OLD": 0, "W20_NEW": 0}
        dec = sc.decide(summary, nz)
        self.assertEqual(dec["verdict"], "ADVANCE")
        self.assertTrue(dec["primary_arms"]["W10_NEW"]["advance"])
        self.assertFalse(dec["primary_arms"]["W10_OLD"]["advance"])
        self.assertEqual(
            dec["selector_mechanism"]["W10"]["classification"],
            "NEW_SELECTOR_IMPROVES_OOF",
        )
        self.assertEqual(
            dec["selector_mechanism"]["W20"]["classification"],
            "IDENTICAL_ZERO_SELECTION",
        )

    def test_safe_grad_norm_and_clip_handles_none_gradients(self):
        p_active = nn.Parameter(torch.tensor([3.0, 4.0]))
        p_none = nn.Parameter(torch.tensor([1.0, 2.0]))
        p_active.grad = torch.tensor([3.0, 4.0])
        raw_norm, none_ix = rt.safe_grad_norm_and_clip([p_active, p_none], max_norm=1.0)
        self.assertAlmostEqual(raw_norm, 5.0, places=5)
        self.assertEqual(none_ix, [1])
        self.assertIsNone(p_none.grad)
        torch.testing.assert_close(
            p_active.grad, torch.tensor([0.6, 0.8]), atol=1e-5, rtol=1e-5
        )
        p_active.grad = None
        with self.assertRaises(RuntimeError):
            rt.safe_grad_norm_and_clip([p_active, p_none], max_norm=1.0)

    def test_configure_adapter_width_silu_and_step0_parity(self):
        net = MockNet()
        names = [
            n
            for n, p in net.named_parameters()
            if n.startswith("adapter_10tok.")
            and not n.startswith("adapter_10tok.anchor.")
        ]
        sd = {n: t.detach().clone() for n, t in net.state_dict().items()}
        p_w10, _, _ = rt.configure_adapter_width(net, sd, names, 1.0, 168101)
        self.assertEqual(sum(p.numel() for p in p_w10), 18315)
        self.assertIsInstance(net.adapter_10tok.prog_412_bottleneck[2], nn.SiLU)

        x_tr = torch.randn(4, 67)
        x_pa = torch.randn(4, 114)
        with torch.no_grad():
            y10_tr = net.adapter_10tok.traj_delta_head(x_tr).clone()
            y10_pa = net.adapter_10tok.parent_delta_net(x_pa).clone()

        p_w20, _, _ = rt.configure_adapter_width(net, sd, names, 2.0, 168101)
        self.assertEqual(sum(p.numel() for p in p_w20), 35699)
        self.assertIsInstance(net.adapter_10tok.prog_412_bottleneck[2], nn.SiLU)
        with torch.no_grad():
            y20_tr = net.adapter_10tok.traj_delta_head(x_tr)
            y20_pa = net.adapter_10tok.parent_delta_net(x_pa)
        torch.testing.assert_close(y10_tr, y20_tr, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(y10_pa, y20_pa, atol=1e-6, rtol=1e-6)

    def test_audit_hook_allows_launch_json_and_blocks_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "custody" / "run"
            out.mkdir(parents=True)
            pinned = Path("/home/ubuntu/jope-data/h168/inputs/round-882.pt")
            audit = rt.Audit(
                [pinned, out.parent / "launch.json", out / "launch.json"], out
            )
            audit.check("open", (str(out.parent / "launch.json"), "w", 0))
            audit.check("open", (str(out / "launch.json"), "w", 0))
            audit.check("open", (str(pinned), "r", 0))
            for bad_path, mode in (
                ("/home/ubuntu/jope-data/sealed/x.npz", "r"),
                ("/home/ubuntu/jope-data/meta-test/x.npz", "r"),
                ("/home/ubuntu/jope-data/holdout/x.npz", "r"),
                ("/home/ubuntu/jope-data/h168/unpinned.npz", "r"),
                (str(pinned), "w"),
            ):
                with self.assertRaises(RuntimeError):
                    audit.check("open", (bad_path, mode, 0))
            self.assertEqual(audit.receipt()["protected_reads"], 0)
            self.assertEqual(len(audit.receipt()["refused"]), 5)

    def test_supervisor_hard_stop_boundaries(self):
        ok = dict(
            rss=5_999_999_999,
            threads=4,
            free=40_000_000_000,
            disk_used=0.949,
            scratch=2_999_999_999,
            load=85.0,
            available=16 * 1024**3,
            elapsed=43200.0,
        )
        self.assertEqual(mon.classify(ok), ("run", None))
        for k, bad, expected_reason in (
            ("rss", 6_000_000_000, "RSS_LIMIT"),
            ("threads", 5, "THREAD_LIMIT"),
            ("free", 39_999_999_999, "DISK_LIMIT"),
            ("disk_used", 0.95, "DISK_LIMIT"),
            ("scratch", 3_000_000_000, "DISK_LIMIT"),
            ("load", 85.01, "HOST_LOAD"),
            ("available", 16 * 1024**3 - 1, "MEMORY_PRESSURE"),
            ("elapsed", 43200.01, "WALL_TIMEOUT"),
        ):
            self.assertEqual(
                mon.classify(dict(ok, **{k: bad})), ("abort", expected_reason)
            )

    def test_baseline_gate_and_subpanel(self):
        vals = np.ones(91, dtype=np.float64)
        ref = {
            "utility": {"per_context": [{"utility": 1.0} for _ in range(91)]},
            "best_utility": 1.0,
        }
        self.assertEqual(rn.baseline_gate(vals, ref)["status"], "PASS")
        vals_bad = vals.copy()
        vals_bad[0] += 0.01
        vals_bad[1] -= 0.01
        with self.assertRaises(RuntimeError):
            rn.baseline_gate(vals_bad, ref)

        full = dict(
            context_ids=["c2", "c1"],
            values=[2.0, 1.0],
            choices=[5, 3],
            invalid=[False, True],
        )
        sub = rn.subpanel(full, [{"context_sha256": "c1"}])
        self.assertEqual(sub["values"], [1.0])
        self.assertEqual(sub["choices"], [3])
        self.assertEqual(sub["invalid"], [True])

    def test_h156_parent_loss_exact_parity(self):
        import importlib.util

        h146_path = HERE.parent / "h146_lookahead_exec" / "runtime.py"
        self.assertEqual(
            rt.sha(h146_path),
            "1fbb2c89222efb22001ce7fdc785e5d7dc8cee1c3be3b754ba7c4eb057fbcbc4",
        )
        spec = importlib.util.spec_from_file_location("h146_runtime_ref", h146_path)
        h146_rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h146_rt)

        gen = torch.Generator(device="cpu").manual_seed(168101)
        pred = torch.randn((8, 24), generator=gen, dtype=torch.float32)
        fitness = torch.randn((8, 24), generator=gen, dtype=torch.float32) * 5.0
        fitness[0, 0] = -10.0
        fitness[0, 1] = 10.0
        gate = torch.randint(0, 2, (8, 24), generator=gen, dtype=torch.int64).bool()

        l_h168 = rt.loss(pred, fitness, gate)
        l_h156 = h146_rt.loss(pred, fitness, gate)
        self.assertTrue(torch.equal(l_h168, l_h156))

    def test_native_score_bit_identity_gate_and_external_launch_binding(self):
        rows = [{"context_sha256": f"ctx_{i}"} for i in range(91)]
        uncached = {
            f"ctx_{i}": np.array([-1.5, 0.25, 1.75], dtype=np.float32)
            for i in range(91)
        }
        cached = {k: v.copy() for k, v in uncached.items()}
        res = rn.verify_native_score_bit_identity(uncached, cached, rows)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["bit_identical"])
        self.assertEqual(res["candidates_checked"], 91 * 3)

        # Perturb a non-selected candidate by 1e-6 (argmin choice remains index 0, but raw float32 bytes differ)
        cached_bad = {k: v.copy() for k, v in uncached.items()}
        cached_bad["ctx_0"][2] += np.float32(1e-6)
        self.assertEqual(
            int(np.argmin(uncached["ctx_0"])), int(np.argmin(cached_bad["ctx_0"]))
        )
        with self.assertRaises(RuntimeError):
            rn.verify_native_score_bit_identity(uncached, cached_bad, rows)

        with tempfile.TemporaryDirectory() as tmp:
            lp = Path(tmp) / "launch.json"
            with self.assertRaises(RuntimeError):
                rn.verify_external_launch_binding(lp, "abc", "m1", "p1", "f1")
            lp.write_text(
                json.dumps(
                    dict(
                        merged_commit="abc",
                        manifest_sha256="m1",
                        prereg_sha256="p1",
                        fold_manifest_sha256="f1",
                        review_id=5727048283,
                    )
                )
            )
            ext = rn.verify_external_launch_binding(lp, "abc", "m1", "p1", "f1")
            self.assertEqual(ext["review_id"], 5727048283)
            with self.assertRaises(RuntimeError):
                rn.verify_external_launch_binding(lp, "wrong_head", "m1", "p1", "f1")

    def test_monitor_sample_includes_supervisor_process_and_boundary(self):
        class FakeMem:
            def __init__(self, rss):
                self.rss = rss

        class FakeProc:
            def __init__(self, pid, rss, threads, kids=None):
                self.pid = pid
                self._rss = rss
                self._threads = threads
                self._kids = kids or []

            def memory_info(self):
                return FakeMem(self._rss)

            def num_threads(self):
                return self._threads

            def children(self, recursive=True):
                return list(self._kids)

        sup = FakeProc(pid=100, rss=2_000_000_000, threads=2)
        worker_child = FakeProc(pid=201, rss=1_500_000_000, threads=1)
        worker = FakeProc(pid=200, rss=2_000_000_000, threads=1, kids=[worker_child])
        s_ok = mon.sample(proc=worker, sup_proc=sup)
        self.assertEqual(s_ok["rss"], 5_500_000_000)
        self.assertEqual(s_ok["threads"], 4)

        # Adding 1 more thread in worker_child pushes aggregate supervisor+worker threads to 5 -> THREAD_LIMIT abort
        worker_child_bad = FakeProc(pid=201, rss=1_500_000_000, threads=2)
        worker_bad = FakeProc(
            pid=200, rss=2_000_000_000, threads=1, kids=[worker_child_bad]
        )
        s_bad = mon.sample(proc=worker_bad, sup_proc=sup)
        self.assertEqual(s_bad["threads"], 5)
        self.assertEqual(
            mon.classify(
                dict(
                    s_bad,
                    free=50_000_000_000,
                    disk_used=0.5,
                    scratch=0,
                    load=10.0,
                    available=32 * 1024**3,
                    elapsed=1.0,
                )
            ),
            ("abort", "THREAD_LIMIT"),
        )

    def test_predict_uncached_adapter_costs_uses_adapter_10tok(self):
        class FakeAdapter:
            def __init__(self):
                self.called = False

            def predict_costs(self, **kwargs):
                self.called = True
                b, k = kwargs["prog_412"].shape[:2]
                return {"total_cost": torch.ones((b, k), dtype=torch.float32) * -0.5}

        class FakeFusedNet:
            def __init__(self):
                self.adapter_10tok = FakeAdapter()

        net = FakeFusedNet()
        self.assertFalse(hasattr(net, "predict_with_10tok_adapter"))
        progs = np.zeros((2, 6, 412), dtype=np.float32)
        proj = np.zeros((2, 6, 16), dtype=np.float32)
        fit = np.zeros((2, 6), dtype=np.float64)
        gate_arr = np.ones((2, 6), dtype=bool)
        tok_arr = np.zeros((2, 10, 32), dtype=np.float32)
        msk_arr = np.ones((2, 10), dtype=bool)
        scores = rn.predict_uncached_adapter_costs(
            net, progs, proj, fit, gate_arr, tok_arr, msk_arr
        )
        self.assertTrue(net.adapter_10tok.called)
        self.assertEqual(scores.shape, (2, 6))
        np.testing.assert_allclose(scores, -0.5)


if __name__ == "__main__":
    unittest.main()
