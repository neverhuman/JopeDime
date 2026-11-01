"""Data-free controls for H149 inference, bounds, input isolation, and tensor bits."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import unittest
from pathlib import Path
import numpy as np
from science import contrast, decide, folds, batch_plan, SEEDS, dose_check
from runtime import Audit, tensor_hash, loss
from monitor import classify, environment


def valid_drifts():
    return {"CTRL": [1.0] * 3, "WD_5": [0.8] * 3, "WD_25": [1.3] * 3}


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


class H149Tests(unittest.TestCase):
    def test_dose_inclusive_five_percent(self):
        for v in (0.95, 1.0, 1.05):
            d = valid_drifts()
            d["WD_5"] = [v] * 3
            self.assertFalse(dose_check(d)["WD_5"]["dose_valid"])

    def test_dose_outside_five_percent(self):
        for v in (0.949, 1.051):
            d = valid_drifts()
            d["WD_5"] = [v] * 3
            self.assertTrue(dose_check(d)["WD_5"]["dose_valid"])

    def test_dose_uses_median(self):
        d = valid_drifts()
        d["WD_5"] = [1.0, 1.0, 20.0]
        self.assertFalse(dose_check(d)["WD_5"]["dose_valid"])

    def test_dose_invalid_cannot_advance(self):
        c = contrast(rows(), np.full((3, 19), 0.01), np.zeros(19))
        d = decide({"WD_5": c, "WD_25": c}, {a: [1.0] * 3 for a in valid_drifts()})
        self.assertTrue(
            all(not x["advance"] and not x["gates"]["dose"] for x in d.values())
        )

    def test_dose_missing_or_nonfinite_refusal(self):
        for values in ([1.0, 1.0], [1.0, np.nan, 1.0], [0.0] * 3):
            d = valid_drifts()
            d["CTRL"] = values
            with self.assertRaises(ValueError):
                dose_check(d)

    def test_ties_reject(self):
        c = contrast(rows(), np.zeros((3, 19)), np.zeros(19))
        d = decide({"WD_5": c, "WD_25": c}, valid_drifts())
        self.assertTrue(all(not v["advance"] for v in d.values()))
        self.assertEqual(c["elo"], 1500.0)

    def test_positive_spread_advances(self):
        c = contrast(rows(), np.full((3, 19), 0.01), np.zeros(19))
        d = decide({"WD_5": c, "WD_25": c}, valid_drifts())
        self.assertTrue(all(v["advance"] for v in d.values()))
        self.assertLess(c["top3_positive_share"], 0.8)

    def test_negative_rejects(self):
        c = contrast(rows(), np.full((3, 19), -0.01), np.zeros(19))
        self.assertLess(c["ci95"][1], 0)

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
            decide({}, valid_drifts())

    def test_holm_two(self):
        c = dict(
            p_one_sided=0.026,
            median_gain=0.01,
            ci95=[0.001, 0.02],
            seeds_beating=3,
            top3_positive_share=0.3,
        )
        d = decide({"WD_5": c, "WD_25": c}, valid_drifts())
        self.assertFalse(any(v["advance"] for v in d.values()))

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
            available=16 * 1024**3,
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
        self.assertEqual(classify(dict(s, available=16 * 1024**3 - 1))[0], "pause")

    def test_environment_one_thread(self):
        self.assertEqual(environment()["OMP_NUM_THREADS"], "1")

    def test_read_audit(self):
        a = Audit(
            ["/home/ubuntu/jope-data/h50/input/selected.json"],
            Path("/home/ubuntu/jope-data/h149/test"),
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


if __name__ == "__main__":
    unittest.main()
