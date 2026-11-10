"""Focused negative controls for loader isolation, objective, inference and gates."""

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import robust_objective as r
import evaluate_h309 as ev
import trainonly_loader as tl


class H309Tests(unittest.TestCase):
    def test_train_prefix_never_reads_poisoned_tail(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "context_features.npz"
            a = np.arange(21, dtype=np.float32).reshape(7, 3)
            np.savez(
                p, cids=np.array(["t0", "t1", "t2", "t3", "t4", "v0", "v1"]), lm48_z=a
            )
            off, shape, dtype = tl._member_layout(p, "lm48_z.npy")
            with p.open("r+b") as f:
                f.seek(off + 5 * 12)
                f.write(np.full((2, 3), np.nan, dtype=np.float32).tobytes())
            with self.assertRaises(zipfile.BadZipFile):
                with np.load(p) as z:
                    z["lm48_z"]
            calls = []
            original = tl.os.pread

            def trace(fd, n, offset):
                calls.append((n, offset))
                return original(fd, n, offset)

            with (
                patch.object(tl.os, "pread", trace),
                tl.TrainOnlyNumpy({f"t{i}" for i in range(5)}).load(p) as z,
            ):
                result = z["lm48_z"]
            np.testing.assert_array_equal(result, a[:5])
            self.assertIn((60, off), calls)
            self.assertFalse(any(pos == off and n > 60 for n, pos in calls))

    def test_ragged_train_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "pool_priors.npz"
            a = np.arange(20, dtype=np.float32).reshape(10, 2)
            np.savez(
                p,
                cids=np.array(["t0", "t1", "v0"]),
                offsets=np.array([0, 2, 7, 10]),
                proj16=a,
            )
            with tl.TrainOnlyNumpy({"t0", "t1"}).load(p) as z:
                np.testing.assert_array_equal(z["proj16"], a[:7])

    def test_invalid_train_prefix_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "context_features.npz"
            np.savez(p, cids=np.array(["t0", "v0", "t1"]), lm48_z=np.zeros((3, 2)))
            with self.assertRaises(RuntimeError):
                tl.TrainOnlyNumpy({"t0", "t1"}).load(p)

    def test_outer_test_guard_refuses_get_and_index(self):
        g = tl.GuardedPackets({"train": 1, "test": 2}, {"test"})
        self.assertEqual(g["train"], 1)
        for fn in [lambda: g["test"], lambda: g.get("test")]:
            with self.assertRaises(KeyError):
                fn()
        self.assertEqual(g.denied, 2)

    def test_objective_trims_families_not_contexts(self):
        gains = [10, 9, 8, 7, 6, 2, 0]
        packets = {}
        fams = {}
        cids = []
        for i, gain in enumerate(gains):
            for j in range(4 if i == 0 else 1):
                c = f"{i}:{j}"
                cids.append(c)
                packets[c] = {"utility_k": np.array([0, gain])}
                fams[c] = f"f{i}"
        out = r.family_profile([1] * len(cids), cids, packets, fams)
        self.assertEqual(out["trimmed_gain"], 1.0)
        self.assertEqual(out["family_median"], 7.0)
        self.assertEqual(out["dropped_families"], ["f0", "f1", "f2", "f3", "f4"])

    def test_robust_objective_can_reject_mean_winner(self):
        a = dict(
            tau=0.05,
            k=5,
            a=2,
            inner_gain=10,
            trimmed_gain=-0.1,
            family_median=0,
            inner_switches=5,
        )
        b = dict(a, inner_gain=2, trimmed_gain=0.1, k=6)
        self.assertEqual(r.choose([a, b])[0], b)

    def test_tie_median_then_conservatism(self):
        a = dict(
            tau=0.05, k=5, a=2, trimmed_gain=0.2, family_median=0.001, inner_switches=5
        )
        b = dict(a, trimmed_gain=0.19995, family_median=0.002, inner_switches=10)
        self.assertEqual(r.choose([a, b])[0], b)
        c = dict(b, k=7, inner_switches=9)
        self.assertEqual(r.choose([a, b, c])[0], c)

    def test_no_chained_tolerance_selection(self):
        rows = [
            dict(tau=0.05, k=5, a=2, trimmed_gain=v, family_median=0, inner_switches=i)
            for i, v in enumerate([0.2, 0.20009, 0.20018])
        ]
        self.assertEqual(r.choose(rows)[0]["inner_switches"], 1)

    def test_paired_identical_arms_exact_zero(self):
        x = np.arange(30, dtype=float).reshape(3, 10)
        got = ev.paired_statistics(x, x, draws=200)
        for name in ["mean", "drop_top5", "family_median"]:
            self.assertEqual(got[name]["per_seed"], [0.0, 0.0, 0.0])
            self.assertEqual(got[name]["pooled"]["ci95"], [0.0, 0.0])

    def test_paired_trim_is_difference_of_arm_trims(self):
        base = np.array([[0, 0, 0, 0, 0, 10, 20, 30, 40, 50]] * 3, float)
        robust = np.array([[1, 1, 1, 1, 1, 10, 20, 30, 40, 50]] * 3, float)
        out = ev.paired_statistics(robust, base, draws=200)
        self.assertEqual(out["drop_top5"]["per_seed"], [1.0, 1.0, 1.0])
        self.assertEqual(float(ev.trim5(robust[0] - base[0])), 0.0)

    def test_paired_median_is_difference_of_medians(self):
        x = np.array([[0, 0, 0, 0, 0, 10, 10, 10, 10]] * 3, float)
        y = np.array([[0, 0, 0, 0, 5, 5, 5, 5, 5]] * 3, float)
        out = ev.paired_statistics(x, y, draws=100)
        self.assertEqual(out["family_median"]["per_seed"], [-5.0, -5.0, -5.0])
        self.assertEqual(
            out["median_familywise_paired_delta_descriptive"], [0.0, 0.0, 0.0]
        )

    def test_decision_gates_and_no_change_override(self):
        pair = {
            "mean": {"median": 0.0},
            "drop_top5": {"median": 0.002, "pooled": {"lcb95": 0.0001}},
        }
        arm = {
            "concentration": [{"drop_top5_lcb95_one_sided": 0.001}] * 3,
            "summary": {"harmed_share_by_seed": [0.1] * 3},
            "invalid_detail": [{"excess": 0}] * 3,
        }
        self.assertEqual(r.decide(pair, arm, arm, 11)["verdict"], "ROBUSTSEL_ADVANCE")
        self.assertEqual(r.decide(pair, arm, arm, 12)["verdict"], "ROBUSTSEL_NO_CHANGE")
        import copy

        for path, value in [
            (("mean", "median"), -0.00501),
            (("drop_top5", "median"), 0.001),
            (("drop_top5", "pooled", "lcb95"), 0.0),
        ]:
            p = copy.deepcopy(pair)
            d = p
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = value
            self.assertEqual(
                r.decide(p, arm, arm, 11)["verdict"], "ROBUSTSEL_NOT_ADVANCED"
            )
        for field in ["harm", "invalid", "drop"]:
            bad = copy.deepcopy(arm)
            if field == "harm":
                bad["summary"]["harmed_share_by_seed"][0] = 0.11001
            elif field == "invalid":
                bad["invalid_detail"][0]["excess"] = 1
            else:
                bad["concentration"][0]["drop_top5_lcb95_one_sided"] = 0
            self.assertEqual(
                r.decide(pair, bad, arm, 11)["verdict"], "ROBUSTSEL_NOT_ADVANCED"
            )
        self.assertEqual(
            ev.ELO_LABEL,
            "reference = r882 slot0 (DL-Train OOF); r3175-anchored Elo unmeasured",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
