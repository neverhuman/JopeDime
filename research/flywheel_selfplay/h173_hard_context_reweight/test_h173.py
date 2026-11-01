"""Data-free unit tests for H173 FIT-only regret reweighting, weight-leak guard, strict inner-B selector, pseudoreplication check, and resource guards."""

import importlib.util
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


class H173Tests(unittest.TestCase):
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

    def test_fit_regret_weights_depend_only_on_fit_ids_and_reject_leakage(self):
        rows = make_rows(205, 1)
        fit, sel, assess = sc.partition(rows, 0)
        fit_ids = [r["context_sha256"] for r in fit]
        b_ids = [r["context_sha256"] for r in sel]
        a_ids = [r["context_sha256"] for r in assess]
        val_ids = [f"val_{i:03d}" for i in range(91)]
        excluded = set(b_ids + a_ids + val_ids)

        rng = np.random.default_rng(173101)
        sources_clean = {}
        for cid in fit_ids + b_ids + a_ids + val_ids:
            fit_arr = rng.uniform(-1.5, 0.5, size=12)
            gate_arr = np.ones(12, dtype=bool)
            gate_arr[-1] = False
            sources_clean[cid] = dict(
                fitness=fit_arr,
                gate=gate_arr,
                r882_choice=int(rng.integers(0, 12)),
            )

        # Wildly corrupt B, A, and Val contexts in a second dictionary
        sources_corrupted_nonfit = {
            k: dict(
                fitness=v["fitness"].copy(),
                gate=v["gate"].copy(),
                r882_choice=v["r882_choice"],
            )
            for k, v in sources_clean.items()
        }
        for cid in excluded:
            sources_corrupted_nonfit[cid] = dict(
                fitness=np.array([-1e6, 1e6] + [0.0] * 10, dtype=np.float64),
                gate=np.array([True, False] + [True] * 10, dtype=bool),
                r882_choice=1,
            )

        res1 = sc.compute_fit_regret_weights(
            fit_ids, sources_clean, seed=173101, outer=0, excluded_ids=excluded
        )
        res2 = sc.compute_fit_regret_weights(
            list(reversed(fit_ids)),
            sources_corrupted_nonfit,
            seed=173101,
            outer=0,
            excluded_ids=excluded,
        )

        self.assertEqual(res1["eps"], res2["eps"])
        self.assertEqual(res1["raw_median_fit_regret"], res2["raw_median_fit_regret"])
        self.assertEqual(res1["regret_by_cid"], res2["regret_by_cid"])
        self.assertEqual(res1["weights_by_arm"], res2["weights_by_arm"])
        self.assertEqual(res1["ess_by_arm"], res2["ess_by_arm"])

        # Assert passing any B, A, or Val context ID in fit_ids raises ValueError
        for leak_cid in (b_ids[0], a_ids[0], val_ids[0]):
            with self.assertRaises(ValueError):
                sc.compute_fit_regret_weights(
                    fit_ids + [leak_cid],
                    sources_clean,
                    seed=173101,
                    outer=0,
                    excluded_ids=excluded,
                )

    def test_regret_weight_formula_clipping_renormalization_and_shuffle_placebo(self):
        # 5 FIT contexts with known regrets:
        # c0: choice 0 (cost -1.0), min cost -1.0 -> regret 0.0
        # c1: choice 1 (cost -0.8), min cost -1.0 -> regret 0.2
        # c2: choice 1 (cost -0.5), min cost -1.0 -> regret 0.5 (median = 0.5)
        # c3: choice 1 (cost  0.0), min cost -1.0 -> regret 1.0
        # c4: choice 2 (invalid -> cost 2.0), min cost -1.0 -> regret 3.0
        fit_ids = ["c0", "c1", "c2", "c3", "c4"]
        sources = {
            "c0": dict(
                fitness=[-1.0, -0.5, 0.0], gate=[True, True, False], r882_choice=0
            ),
            "c1": dict(
                fitness=[-1.0, -0.8, 0.0], gate=[True, True, False], r882_choice=1
            ),
            "c2": dict(
                fitness=[-1.0, -0.5, 0.0], gate=[True, True, False], r882_choice=1
            ),
            "c3": dict(
                fitness=[-1.0, 0.0, 0.5], gate=[True, True, False], r882_choice=1
            ),
            "c4": dict(
                fitness=[-1.0, -0.5, -9.0], gate=[True, True, False], r882_choice=2
            ),
        }
        out = sc.compute_fit_regret_weights(fit_ids, sources, seed=173101, outer=2)
        self.assertAlmostEqual(out["eps"], 0.5)
        np.testing.assert_allclose(
            [out["regret_by_cid"][c] for c in fit_ids],
            [0.0, 0.2, 0.5, 1.0, 3.0],
        )

        for arm in sc.ARMS:
            w = np.asarray([out["weights_by_arm"][arm][c] for c in fit_ids])
            self.assertAlmostEqual(float(np.mean(w)), 1.0, places=12)
            self.assertTrue(np.all(w > 0.0))

        w_ctrl = np.asarray([out["weights_by_arm"]["CTRL"][c] for c in fit_ids])
        np.testing.assert_allclose(w_ctrl, np.ones(5))
        self.assertAlmostEqual(out["ess_by_arm"]["CTRL"]["ess"], 5.0)
        self.assertAlmostEqual(out["ess_by_arm"]["CTRL"]["ess_ratio"], 1.0)

        w_rw10 = np.asarray([out["weights_by_arm"]["RW10"][c] for c in fit_ids])
        w_shuf = np.asarray([out["weights_by_arm"]["RW10_SHUF"][c] for c in fit_ids])
        # RW10 is strictly increasing with regret
        self.assertTrue(np.all(np.diff(w_rw10) > 0.0))
        # RW10_SHUF has exact same multiset and ESS as RW10
        np.testing.assert_allclose(np.sort(w_rw10), np.sort(w_shuf))
        self.assertAlmostEqual(
            out["ess_by_arm"]["RW10"]["ess"],
            out["ess_by_arm"]["RW10_SHUF"]["ess"],
            places=12,
        )

        # Verify extreme outlier clipping to [0.2, 5.0] followed by renormalization to mean 1.0
        fit20 = [f"x{i:02d}" for i in range(20)]
        src20 = {}
        for i, cid in enumerate(fit20):
            reg = 1000.0 if i == 19 else 0.01
            src20[cid] = dict(
                fitness=[-1.0, -1.0 + reg], gate=[True, True], r882_choice=1
            )
        out20 = sc.compute_fit_regret_weights(fit20, src20, seed=173101, outer=0)
        w20 = np.asarray([out20["weights_by_arm"]["RW10"][c] for c in fit20])
        self.assertAlmostEqual(float(np.mean(w20)), 1.0, places=12)
        # Before final renorm, ratio max/min is at most 5.0 / 0.2 = 25.0, which is preserved by positive scalar renorm
        self.assertLessEqual(float(np.max(w20) / np.min(w20)), 25.0 + 1e-9)

        # Verify zero-median FIT regret floor (eps = max(median(FIT regret), 1e-6) = 1e-6 for [0, 0, 1])
        # and exact registered shuf_seed = seed + 1000 * outer + 77 (173178 for seed=173101, outer=0)
        fit3 = ["z0", "z1", "z2"]
        src3 = {
            "z0": dict(fitness=[-1.0, 0.0], gate=[True, True], r882_choice=0),
            "z1": dict(fitness=[-1.0, 0.0], gate=[True, True], r882_choice=0),
            "z2": dict(fitness=[-1.0, 0.0], gate=[True, True], r882_choice=1),
        }
        out3 = sc.compute_fit_regret_weights(fit3, src3, seed=173101, outer=0)
        self.assertAlmostEqual(out3["raw_median_fit_regret"], 0.0, places=12)
        self.assertAlmostEqual(out3["eps"], 1e-6, places=12)
        self.assertEqual(out3["shuf_seed"], 173178)
        self.assertEqual(sc.selector_seed_for(0, 173101, 173141), 173242)

    def test_weighted_loss_exact_h146_parity_and_scaling(self):
        h146_path = HERE.parent / "h146_lookahead_exec" / "runtime.py"
        if not h146_path.exists():
            h146_path = Path(
                "/home/ubuntu/JopeDime/research/flywheel_selfplay/h146_lookahead_exec/runtime.py"
            )
        self.assertEqual(
            rt.sha(h146_path),
            "1fbb2c89222efb22001ce7fdc785e5d7dc8cee1c3be3b754ba7c4eb057fbcbc4",
        )
        spec = importlib.util.spec_from_file_location("h146_runtime_ref", h146_path)
        h146_rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h146_rt)

        gen = torch.Generator(device="cpu").manual_seed(173101)
        pred = torch.randn((8, 24), generator=gen, dtype=torch.float32)
        fitness = torch.randn((8, 24), generator=gen, dtype=torch.float32) * 5.0
        fitness[0, 0] = -10.0
        fitness[0, 1] = 10.0
        gate = torch.randint(0, 2, (8, 24), generator=gen, dtype=torch.int64).bool()

        l_ref = h146_rt.loss(pred, fitness, gate)
        l_unw = rt.loss(pred, fitness, gate)
        l_w1 = rt.weighted_loss(pred, fitness, gate, torch.ones(8, dtype=torch.float32))
        self.assertTrue(torch.equal(l_unw, l_ref))
        self.assertTrue(torch.equal(l_w1, l_ref))

        # Verify custom positive weights match explicit dot product
        w = torch.tensor([0.5, 1.5, 0.2, 2.8, 1.0, 0.8, 1.2, 1.0], dtype=torch.float32)
        ctx_vec = rt.per_context_loss(pred, fitness, gate)
        torch.testing.assert_close(
            rt.weighted_loss(pred, fitness, gate, w), (ctx_vec * w).mean()
        )

        with self.assertRaises(ValueError):
            rt.weighted_loss(pred, fitness, gate, torch.zeros(8))
        with self.assertRaises(ValueError):
            rt.weighted_loss(pred, fitness, gate, torch.ones(7))

    def test_strict_selector_and_diagnostics_ties_and_lcb(self):
        r41 = make_rows(41, 1)

        # 1. Diffuse +0.005 effect: strict, unconstrained, and nonzero all pick step 10 (earliest tie-breaker)
        panel_diff = np.ones((41, 5))
        panel_diff[:, 1] += 0.005
        panel_diff[:, 2] += 0.005 + 5e-13
        d_diff = sc.choose_strict_and_diagnostics(r41, panel_diff, seed=173141)
        self.assertEqual(d_diff["selected_step"], 10)
        self.assertEqual(d_diff["unconstrained_selected_step"], 10)
        self.assertEqual(d_diff["nonzero_selected_step"], 10)

        # 2. Sparse +0.005 mean (+0.205 on 1 family, 0 on 40 families):
        # strict rejects (LCB95 == 0.0 -> step 0); unconstrained & nonzero pick step 10
        panel_sparse = np.ones((41, 5))
        panel_sparse[0, 1] += 0.205
        d_sparse = sc.choose_strict_and_diagnostics(r41, panel_sparse, seed=173141)
        self.assertEqual(d_sparse["selected_step"], 0)
        self.assertEqual(d_sparse["unconstrained_selected_step"], 10)
        self.assertEqual(d_sparse["nonzero_selected_step"], 10)

        # 3. All negative non-zero steps: strict & unconstrained pick 0; nonzero picks best non-zero step (10)
        panel_neg = np.zeros((41, 5))
        panel_neg[:, 1:] = [-0.01, -0.02, -0.03, -0.04]
        d_neg = sc.choose_strict_and_diagnostics(r41, panel_neg, seed=173141)
        self.assertEqual(d_neg["selected_step"], 0)
        self.assertEqual(d_neg["unconstrained_selected_step"], 0)
        self.assertEqual(d_neg["nonzero_selected_step"], 10)

    def test_summarize_holm_m3_decide_and_pseudoreplication_report(self):
        rows = make_rows(41, 1)
        panel = np.zeros((3, 41, 5))
        # RW10 (col 3) gains +0.005 across all families & seeds; others stay at 0
        panel[:, :, 3] = 0.005
        summary = sc.summarize(rows, panel, cols=sc.COLUMNS)
        nz = {"CTRL": 0, "RW05": 0, "RW10": 5, "RW10_SHUF": 0}
        dec = sc.decide(summary, nz)
        self.assertEqual(dec["verdict"], "ADVANCE")
        self.assertTrue(dec["primary_arms"]["RW10"]["advance"])
        self.assertFalse(dec["primary_arms"]["RW10_SHUF"]["advance"])
        self.assertEqual(
            dec["mechanism_classification"],
            "HARDNESS_SIGNAL_ADVANCES_OVER_SHUFFLED_PLACEBO",
        )

        # Placebo-only (RW10_SHUF-only) win must NOT advance (RW10_SHUF is in m=3 Holm family, not a qualifying hardness arm)
        panel_shuf = np.zeros((3, 41, 5))
        panel_shuf[:, :, 4] = 0.005
        summary_shuf = sc.summarize(rows, panel_shuf, cols=sc.COLUMNS)
        nz_shuf = {"CTRL": 0, "RW05": 0, "RW10": 0, "RW10_SHUF": 5}
        dec_shuf = sc.decide(summary_shuf, nz_shuf)
        self.assertEqual(dec_shuf["verdict"], "NO_ADVANCE")
        self.assertFalse(dec_shuf["primary_arms"]["RW10_SHUF"]["advance"])
        self.assertFalse(
            dec_shuf["primary_arms"]["RW10_SHUF"]["gates"]["qualifies_as_hardness_arm"]
        )
        self.assertTrue(dec_shuf["primary_arms"]["RW10_SHUF"]["gates"]["holm_m3"])
        self.assertEqual(
            dec_shuf["mechanism_classification"],
            "NOISE_NEUTRAL_NO_OOF_GAIN",
        )

        steps_map = {
            "CTRL": [[0] * 5, [0] * 5, [0] * 5],
            "RW05": [[0] * 5, [0] * 5, [0] * 5],
            "RW10": [[10, 0, 10, 0, 10], [10, 20, 0, 0, 10], [0, 10, 10, 0, 20]],
            "RW10_SHUF": [[0] * 5, [0] * 5, [0] * 5],
        }
        pr = sc.pseudoreplication_report(panel, steps_map)
        self.assertTrue(pr["CTRL"]["all_baseline_selected"])
        self.assertTrue(pr["CTRL"]["identical_oof_panel_all_3_seeds"])
        self.assertFalse(pr["RW10"]["all_baseline_selected"])
        self.assertFalse(pr["RW10"]["identical_choices_all_3_seeds"])
        self.assertEqual(
            pr["RW10"]["identical_oof_panel_to_r882_by_seed"], [False, False, False]
        )

    def test_canonical_bt_elo_anchor_and_dominance(self):
        ties = np.ones((5, 5)) - np.eye(5)
        np.testing.assert_allclose(sc.bt_ratings(ties, anchor=0), 1500.0)
        dom = np.zeros((2, 2))
        dom[1, 0] = 205.0
        r_prior0 = sc.bt_ratings(dom, anchor=0, prior=0.0)
        self.assertAlmostEqual(r_prior0[0], 1500.0)
        self.assertGreater(r_prior0[1], 3800.0)

        # Near-tie (< 1e-4) is a 0.5 tie under canonical H148 1e-4 context outcome rule
        near_tie = np.array([[0.0, 5e-5]])
        np.testing.assert_allclose(
            sc._pair_wins(near_tie), np.array([[0.0, 0.5], [0.5, 0.0]])
        )

        # Within-family cancellation: 2 contexts in the same family with +0.01 + 5e-5 and -0.01 vs r882
        # Family mean is +2.5e-5 > 1e-12, but context-level outcomes are 1.0 and 0.0 -> family average 0.5 tie
        rows_2ctx = [
            dict(context_sha256="c0", family_id="fam_0000"),
            dict(context_sha256="c1", family_id="fam_0000"),
        ]
        vals_2ctx = np.array([[0.0, 0.01 + 5e-5], [0.0, -0.01]])
        fw = sc.family_wins(vals_2ctx, rows_2ctx)
        np.testing.assert_allclose(fw[0], np.array([[0.0, 0.5], [0.5, 0.0]]))
        panel_3seed = np.stack([vals_2ctx, vals_2ctx, vals_2ctx])
        elo_out = sc.elo_panel(rows_2ctx, panel_3seed, cols=("r882", "RW10"))
        self.assertAlmostEqual(elo_out["median_elo"]["r882"], 1500.0, places=6)
        self.assertAlmostEqual(elo_out["median_elo"]["RW10"], 1500.0, places=6)

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

    def test_audit_hook_and_supervisor_hard_stop_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "custody" / "run"
            out.mkdir(parents=True)
            pinned = Path("/home/ubuntu/jope-data/h173/inputs/round-882.pt")
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
                ("/home/ubuntu/jope-data/h173/unpinned.npz", "r"),
                (str(pinned), "w"),
            ):
                with self.assertRaises(RuntimeError):
                    audit.check("open", (bad_path, mode, 0))
            self.assertEqual(audit.receipt()["protected_reads"], 0)
            self.assertEqual(len(audit.receipt()["refused"]), 5)

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

    def test_native_score_bit_identity_and_external_launch_binding(self):
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

        cached_bad = {k: v.copy() for k, v in uncached.items()}
        cached_bad["ctx_0"][2] += np.float32(1e-6)
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
                        review_id=12345,
                    )
                )
            )
            ext = rn.verify_external_launch_binding(lp, "abc", "m1", "p1", "f1")
            self.assertEqual(ext["review_id"], 12345)

    def test_predict_uncached_adapter_costs_uses_real_fused_jope_network(self):
        fused = rt.load_fused()
        net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
        net.eval()
        self.assertFalse(hasattr(net, "predict_with_10tok_adapter"))
        self.assertTrue(hasattr(net, "adapter_10tok"))
        self.assertTrue(hasattr(net.adapter_10tok, "predict_costs"))
        progs = np.zeros((2, 6, 412), dtype=np.float32)
        proj = np.zeros((2, 6, 16), dtype=np.float32)
        fit = np.zeros((2, 6), dtype=np.float64)
        gate_arr = np.ones((2, 6), dtype=bool)
        tok_arr = np.zeros((2, 10, 32), dtype=np.float32)
        msk_arr = np.ones((2, 10), dtype=bool)
        with torch.no_grad():
            scores = rn.predict_uncached_adapter_costs(
                net, progs, proj, fit, gate_arr, tok_arr, msk_arr
            )
        self.assertEqual(scores.shape, (2, 6))
        self.assertTrue(np.isfinite(scores).all())
        with self.assertRaises(TypeError):
            net.adapter_10tok.predict_costs(
                ctx_feat=torch.zeros((2, 584), dtype=torch.float32),
                prog_412=torch.as_tensor(progs),
                proj_16=torch.as_tensor(proj),
                base_prior_cost=torch.zeros((2, 6), dtype=torch.float32),
                scale=torch.full((2,), 0.4208, dtype=torch.float32),
                sos_prior=torch.zeros((2, 6, 2), dtype=torch.float32),
                mm_tokens_10x32=torch.as_tensor(tok_arr),
                mm_mask_10=torch.as_tensor(msk_arr),
                enable_prior_correction=True,
            )


if __name__ == "__main__":
    unittest.main()
