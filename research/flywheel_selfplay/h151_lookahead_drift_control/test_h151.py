"""Data-free controls for H151 drift-matched control selection, Lookahead schedule, bounds, and tensor bits (reused from H146)."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import unittest
from pathlib import Path
import numpy as np
from science import (
    contrast,
    decide,
    folds,
    batch_plan,
    match_drift_control,
    mechanism_verdict,
    SEEDS,
    ARMS,
    HOLM_M,
    LA_ALPHA,
)
from runtime import Audit, LookaheadState, tensor_hash, loss
from monitor import classify, environment


def rows(n=19):
    return [
        dict(
            context_sha256=f"c{i}",
            family_id=f"f{i}",
            dl_split="DL-Train",
            source_split="train",
        )
        for i in range(n)
    ]


class H151Tests(unittest.TestCase):
    def test_ties_reject(self):
        c = contrast(rows(), np.zeros((3, 19)), np.zeros(19))
        d = decide(
            {"CTRL": c, "CTRL_HALF": c, "CTRL_QTR": c, "LA": c},
            dict(drift_match_valid=True),
        )
        self.assertTrue(all(not v["advance"] for v in d.values()))
        self.assertEqual(c["elo"], 1500.0)

    def test_positive_spread_advances_only_on_three_seeds(self):
        c3 = contrast(rows(), np.full((3, 19), 0.01), np.zeros(19))
        d3 = decide(
            {"CTRL": c3, "CTRL_HALF": c3, "CTRL_QTR": c3, "LA": c3},
            dict(drift_match_valid=True),
        )
        self.assertTrue(all(v["advance"] for v in d3.values()))
        self.assertLess(c3["top3_positive_share"], 0.8)
        c2 = contrast(
            rows(),
            np.array([[0.01] * 19, [0.01] * 19, [-0.001] * 19]),
            np.zeros(19),
        )
        d2 = decide(
            {"CTRL": c2, "CTRL_HALF": c2, "CTRL_QTR": c2, "LA": c2},
            dict(drift_match_valid=True),
        )
        self.assertTrue(all(not v["advance"] for v in d2.values()))

    def test_drift_match_selection_and_inclusive_15pct_boundary(self):
        drifts_ok = {
            "CTRL": [0.443, 0.443, 0.438],
            "CTRL_HALF": [0.221, 0.222, 0.219],
            "CTRL_QTR": [0.111, 0.111, 0.110],
            "LA": [0.219, 0.219, 0.217],
        }
        dm = match_drift_control(drifts_ok)
        self.assertEqual(dm["matched_control_arm"], "CTRL_HALF")
        self.assertTrue(dm["drift_match_valid"])
        self.assertEqual(dm["status"], "DRIFT_MATCH_VALID")

        # Exact lower endpoint 0.85 * LA (reproduces rd_astra boundary_case)
        dm_low = match_drift_control(
            {
                "CTRL": [2.0, 2.0, 2.0],
                "CTRL_HALF": [0.85, 0.85, 0.85],
                "CTRL_QTR": [0.3, 0.3, 0.3],
                "LA": [1.0, 1.0, 1.0],
            }
        )
        self.assertTrue(dm_low["drift_match_valid"])
        self.assertEqual(dm_low["status"], "DRIFT_MATCH_VALID")

        # Exact upper endpoint 1.15 * LA
        dm_high = match_drift_control(
            {
                "CTRL": [2.0, 2.0, 2.0],
                "CTRL_HALF": [1.15, 1.15, 1.15],
                "CTRL_QTR": [0.3, 0.3, 0.3],
                "LA": [1.0, 1.0, 1.0],
            }
        )
        self.assertTrue(dm_high["drift_match_valid"])
        self.assertEqual(dm_high["status"], "DRIFT_MATCH_VALID")

        # Just outside lower and upper bounds
        for bad_val in (0.85 - 1e-9, 1.15 + 1e-9):
            dm_out = match_drift_control(
                {
                    "CTRL": [2.0, 2.0, 2.0],
                    "CTRL_HALF": [bad_val, bad_val, bad_val],
                    "CTRL_QTR": [0.3, 0.3, 0.3],
                    "LA": [1.0, 1.0, 1.0],
                }
            )
            self.assertFalse(dm_out["drift_match_valid"])
            self.assertEqual(dm_out["status"], "DRIFT_MATCH_INVALID")

    def test_ctrl_governed_by_holm_four_and_cannot_bypass(self):
        # Reproduces rd_astra ctrl_admission_case: CTRL p=0.02 with 3 failing arms -> Holm 4*0.02 = 0.08 >= 0.05 -> NO_ADVANCE
        ctrl_c = dict(
            p_one_sided=0.02,
            median_gain=0.01,
            ci95=[0.001, 0.02],
            seeds_beating=3,
            top3_positive_share=0.2,
        )
        fail_c = dict(
            p_one_sided=1.0,
            median_gain=-0.01,
            ci95=[-0.02, -0.001],
            seeds_beating=0,
            top3_positive_share=1.0,
        )
        dm_valid = dict(drift_match_valid=True)
        d = decide(
            {"CTRL": ctrl_c, "CTRL_HALF": fail_c, "CTRL_QTR": fail_c, "LA": fail_c},
            dm_valid,
        )
        self.assertAlmostEqual(d["CTRL"]["holm_adjusted_p"], 0.08)
        self.assertFalse(d["CTRL"]["gates"]["holm"])
        self.assertFalse(d["CTRL"]["advance"])
        self.assertEqual(
            mechanism_verdict(fail_c, dm_valid, d),
            ("STEP_SIZE_ONLY", "STEP_SIZE_ONLY", "NO_ADVANCE"),
        )

    def test_invalid_drift_match_bars_la_per_arm_advance_and_aggregate(self):
        # Reproduces rd_astra invalid_match_positive_la_case
        pos_c = contrast(rows(), np.full((3, 19), 0.01), np.zeros(19))
        neg_c = contrast(rows(), np.full((3, 19), -0.01), np.zeros(19))
        dm_invalid = dict(drift_match_valid=False)
        d = decide(
            {"CTRL": neg_c, "CTRL_HALF": neg_c, "CTRL_QTR": neg_c, "LA": pos_c},
            dm_invalid,
        )
        self.assertTrue(d["LA"]["efficacy_only_advance"])
        self.assertFalse(d["LA"]["gates"]["drift_match"])
        self.assertFalse(d["LA"]["advance"])
        self.assertEqual(
            mechanism_verdict(pos_c, dm_invalid, d),
            ("DRIFT_MATCH_INVALID", "DRIFT_MATCH_INVALID", "NO_ADVANCE"),
        )

    def test_mechanism_verdicts(self):
        c_zero = contrast(rows(), np.zeros((3, 19)), np.zeros(19))
        c_pos = contrast(rows(), np.full((3, 19), 0.01), np.zeros(19))
        c_neg = contrast(rows(), np.full((3, 19), -0.01), np.zeros(19))
        dm_valid = dict(drift_match_valid=True)
        dm_invalid = dict(drift_match_valid=False)
        d_no_adv = decide(
            {"CTRL": c_neg, "CTRL_HALF": c_neg, "CTRL_QTR": c_neg, "LA": c_neg},
            dm_valid,
        )
        d_no_adv_inv = decide(
            {"CTRL": c_neg, "CTRL_HALF": c_neg, "CTRL_QTR": c_neg, "LA": c_neg},
            dm_invalid,
        )

        self.assertEqual(
            mechanism_verdict(c_pos, dm_valid, d_no_adv),
            ("LOOKAHEAD_SPECIFIC", "LOOKAHEAD_SPECIFIC", "NO_ADVANCE"),
        )
        self.assertEqual(
            mechanism_verdict(c_zero, dm_valid, d_no_adv),
            ("STEP_SIZE_ONLY", "STEP_SIZE_ONLY", "NO_ADVANCE"),
        )
        self.assertEqual(
            mechanism_verdict(c_neg, dm_valid, d_no_adv),
            ("STEP_SIZE_ONLY", "STEP_SIZE_ONLY", "NO_ADVANCE"),
        )
        self.assertEqual(
            mechanism_verdict(c_pos, dm_invalid, d_no_adv_inv),
            ("DRIFT_MATCH_INVALID", "DRIFT_MATCH_INVALID", "NO_ADVANCE"),
        )

    def test_median_not_mean(self):
        c = contrast(
            rows(), np.array([[-0.01] * 19, [-0.01] * 19, [1.0] * 19]), np.zeros(19)
        )
        self.assertLess(c["median_gain"], 0)
        self.assertLess(c["ci95"][1], 0)

    def test_equal_family_not_context(self):
        r = rows(4) + [dict(context_sha256=f"x{i}", family_id="f0") for i in range(30)]
        v = np.array([[1, -1, -1, -1] + [1] * 30] * 3, dtype=float)
        c = contrast(r, v, np.zeros(34))
        self.assertLess(c["median_gain"], 0)

    def test_concentration_refusal(self):
        v = np.zeros((3, 19))
        v[:, 0] = 1
        c = contrast(rows(), v, np.zeros(19))
        self.assertEqual(c["top3_positive_share"], 1.0)

    def test_missing_seed_refusal(self):
        with self.assertRaises(ValueError):
            contrast(rows(), np.zeros((2, 19)), np.zeros(19))

    def test_nonfinite_refusal(self):
        with self.assertRaises(ValueError):
            contrast(rows(), np.full((3, 19), np.nan), np.zeros(19))

    def test_missing_arm_refusal(self):
        with self.assertRaises(ValueError):
            decide({}, dict(drift_match_valid=True))
        with self.assertRaises(ValueError):
            match_drift_control({"CTRL": [0.1, 0.1, 0.1]})

    def test_holm_four(self):
        c = dict(
            p_one_sided=0.013,
            median_gain=0.01,
            ci95=[0.001, 0.02],
            seeds_beating=3,
            top3_positive_share=0.3,
        )
        dm_valid = dict(drift_match_valid=True)
        d = decide(
            {"CTRL": c, "CTRL_HALF": c, "CTRL_QTR": c, "LA": c},
            dm_valid,
        )
        self.assertFalse(
            any(v["advance"] for v in d.values())
        )  # 0.013 x 4 = 0.052 > 0.05
        d2 = decide(
            {
                "CTRL": dict(c, p_one_sided=0.011),
                "CTRL_HALF": dict(c, p_one_sided=0.015),
                "CTRL_QTR": dict(c, p_one_sided=0.020),
                "LA": dict(c, p_one_sided=0.040),
            },
            dm_valid,
        )
        self.assertTrue(all(v["advance"] for v in d2.values()))

    def test_fold_disjoint(self):
        r = rows(205)
        f = folds(r)
        self.assertEqual([len(x) for x in f], [41] * 5)
        self.assertEqual(len(set(sum(f, []))), 205)
        self.assertEqual(f, folds(r))

    def test_protected_fold_refusal(self):
        r = rows()
        r[0]["dl_split"] = "DL-Val"
        with self.assertRaises(ValueError):
            folds(r)

    def test_paired_batch_plan(self):
        orders = {"a": np.arange(250), "b": np.arange(250)}
        a = batch_plan(["a", "b"], orders, SEEDS[0])
        b = batch_plan(["b", "a"], orders, SEEDS[0])
        self.assertEqual(len(a), 40)
        for x, y in zip(a, b):
            self.assertEqual(x[:2], y[:2])
            self.assertTrue(all(np.array_equal(i, j) for i, j in zip(x[2], y[2])))

    def test_resource_boundaries(self):
        s = dict(
            rss=1,
            threads=2,
            free=40_000_000_000,
            scratch=0,
            load=85,
            available=12 * 1024**3,
        )
        self.assertEqual(classify(s)[0], "run")
        for k, v in [
            ("rss", 6_000_000_000),
            ("threads", 5),
            ("free", 39_999_999_999),
            ("scratch", 3_000_000_000),
        ]:
            self.assertEqual(classify(dict(s, **{k: v}))[0], "abort")
        self.assertEqual(classify(dict(s, load=85.01))[0], "pause")
        self.assertEqual(classify(dict(s, available=12 * 1024**3 - 1))[0], "pause")

    def test_environment_one_thread(self):
        self.assertEqual(environment()["OMP_NUM_THREADS"], "1")

    def test_read_audit(self):
        a = Audit(
            ["/home/ubuntu/jope-data/h50/input/selected.json"],
            Path("/home/ubuntu/jope-data/h151/test"),
        )
        a.check("open", ("/home/ubuntu/jope-data/h50/input/selected.json", "r", 0))
        for path, mode in [
            ("/home/ubuntu/jope-data/holdout/data", "r"),
            ("/home/ubuntu/jope-data/h50/input/pools/unknown.npz", "r"),
            ("/home/ubuntu/JopeDime/data/flywheel_selfplay/live", "w"),
            ("/home/ubuntu/jope-data/h50/input/selected.json", "w"),
        ]:
            with self.assertRaises(RuntimeError):
                a.check("open", (path, mode, 0))

    def test_signed_zero_byte_guard(self):
        import torch

        self.assertNotEqual(
            tensor_hash({"x": torch.tensor([0.0])}),
            tensor_hash({"x": torch.tensor([-0.0])}),
        )

    def test_buffer_change_guard(self):
        import torch

        self.assertNotEqual(
            tensor_hash({"buffer": torch.tensor([1.0])}),
            tensor_hash({"buffer": torch.tensor([2.0])}),
        )

    def test_control_loss(self):
        import torch

        torch.set_num_threads(1)
        p = torch.tensor([[0.2, 0.4]], requires_grad=True)
        f = torch.tensor([[0.3, 100.0]])
        g = torch.tensor([[True, False]])
        value = loss(p, f, g)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_baseline_exact_refusal(self):
        from runner import baseline_gate

        ref = {"utility": {"per_context": [1.0] * 91}, "best_utility": 1.0}
        self.assertEqual(baseline_gate([1.0] * 91, ref)["matches"], 91)
        with self.assertRaises(RuntimeError):
            baseline_gate([1.00000001] + [1.0] * 90, ref)

    def test_lookahead_schedule_and_slow_update(self):
        import torch

        p = torch.nn.Parameter(torch.zeros(2))
        la = LookaheadState([p], 5, 0.5)
        synced = []
        for step in range(1, 41):
            with torch.no_grad():
                p.add_(1.0)
            if la.after_fast_step():
                synced.append(step)
        self.assertEqual(synced, list(range(5, 41, 5)))
        p2 = torch.nn.Parameter(torch.zeros(1))
        la2 = LookaheadState([p2], 5, 0.5)
        for _ in range(5):
            with torch.no_grad():
                p2.add_(2.0)
            la2.after_fast_step()
        self.assertEqual(float(p2), 5.0)
        self.assertEqual(float(la2.slow[0]), 5.0)

    def test_registered_arms(self):
        self.assertEqual(
            ARMS,
            {
                "CTRL": (1e-4, None),
                "CTRL_HALF": (5e-5, None),
                "CTRL_QTR": (2.5e-5, None),
                "LA": (1e-4, 5),
            },
        )
        self.assertEqual((HOLM_M, LA_ALPHA), (4, 0.5))
        self.assertEqual(SEEDS, (151101, 151102, 151103))


if __name__ == "__main__":
    unittest.main()
