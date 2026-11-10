import unittest, itertools, sys, tempfile, subprocess, os, json
from pathlib import Path
import numpy as np
from scipy.stats import t
import inference as a

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "vendor/h308_ensvote2_trainonly_repartition"))
import trainonly_loader as tl


class Tests(unittest.TestCase):
    def test_exact_against_brute(self):
        d = np.array([[1.0, -2.0, 3.0, 0.0], [2.0, 0.0, -1.0, 4.0]])
        p, pp = a.exact_flip(d)
        vals = np.array(
            [
                np.mean(d * np.array(s), axis=1)
                for s in itertools.product([-1, 1], repeat=4)
            ]
        )
        np.testing.assert_equal(p, (vals >= d.mean(axis=1) - 1e-14).mean(axis=0))
        self.assertEqual(
            pp,
            float(
                (np.median(vals, axis=1) >= np.median(d.mean(axis=1)) - 1e-14).mean()
            ),
        )

    def test_all_zero_ties(self):
        p, pp = a.exact_flip(np.zeros((6, 19)))
        np.testing.assert_equal(p, np.ones(6))
        self.assertEqual(pp, 1)

    def test_all_positive_exact(self):
        p, pp = a.exact_flip(np.ones((2, 19)))
        np.testing.assert_equal(p, np.full(2, 1 / 2**19))
        self.assertEqual(pp, 1 / 2**19)

    def test_jackknife_linear_se(self):
        d = np.arange(19.0)
        v = (d.sum() - d) / 18
        j = a.jk(v, d.mean(), 18)
        self.assertAlmostEqual(j["se"], d.std(ddof=1) / np.sqrt(19))
        self.assertEqual(j["min_gain"], 8.5)

    def test_t18(self):
        d = np.vstack([np.linspace(-0.01, 0.08, 19)] * 6)
        rows, p = a.analyze(d)
        se = d[0].std(ddof=1) / np.sqrt(19)
        self.assertAlmostEqual(rows[0]["t_lcb95"], d[0].mean() - t.ppf(0.95, 18) * se)
        self.assertAlmostEqual(p["t_se"], se)

    def test_decision_strict(self):
        r = [
            dict(
                gain=0.002,
                exact_signflip_p=0.01,
                t_lcb95=0.001,
                jackknife={"min_gain": 0.001},
            )
            for _ in range(6)
        ]
        self.assertEqual(a.decision(r, 0, None)["verdict"], "H310_SMALLCLUSTER_ROBUST")
        r[3]["gain"] = r[4]["gain"] = 0.001
        self.assertIn(
            "a_fresh3_median_gt_001", a.decision(r, 0, None)["failed_clauses"]
        )

    def test_each_gate(self):
        for field, val, key in [
            ("exact_signflip_p", 0.05, "b_exact_p_lt05_ge4"),
            ("t_lcb95", 0, "c_t18_lcb_positive_ge4"),
        ]:
            r = [
                dict(
                    gain=0.002,
                    exact_signflip_p=0.01,
                    t_lcb95=0.001,
                    jackknife={"min_gain": 0.001},
                )
                for _ in range(6)
            ]
            for x in r[:3]:
                x[field] = val
            self.assertIn(key, a.decision(r, 0, None)["failed_clauses"])

    def test_overlap_empty_fails(self):
        r = [
            dict(
                gain=0.002,
                exact_signflip_p=0.01,
                t_lcb95=0.001,
                jackknife={"min_gain": 0.001},
            )
            for _ in range(6)
        ]
        self.assertIn("e_nonoverlap_positive", a.decision(r, 1, None)["failed_clauses"])

    def test_dev_open_refused(self):
        code = (
            "import sys;sys.path.insert(0,"
            + repr(str(P / "vendor/h310_ensvote2_dev91_vs_r3175"))
            + ");import run_h310 as h;h.install_dev_guard(['fake_dev_cid']);\ntry: open('/nonexistent/fake_dev_cid.npz')\nexcept RuntimeError: assert h.DEV_GUARD['denied']==1\nelse: raise AssertionError('open allowed')"
        )
        self.assertEqual(
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True
            ).returncode,
            0,
        )

    def test_packets_refuse(self):
        p = tl.GuardedPackets({"train": 1, "dev": 2}, {"dev"})
        self.assertEqual(p["train"], 1)
        with self.assertRaises(KeyError):
            p["dev"]
        self.assertEqual(p.denied, 1)

    def test_prefix_does_not_decode_tail(self):
        # Deliberately poison trailing feature bytes; prefix read excludes them physically.
        scratch = Path("/home/ubuntu/jope-data/h312-astra/tests")
        scratch.mkdir(parents=True, exist_ok=True)
        path = scratch / "context_features.npz"
        np.savez(
            path,
            cids=np.array(["train", "dev"]),
            lm48_z=np.array([[1.0, 2.0], [999.0, 888.0]]),
        )
        off, shape, dtype = tl._member_layout(path, "lm48_z.npy")
        with path.open("r+b") as f:
            f.seek(off + 2 * dtype.itemsize)
            f.write(b"\xff" * 2 * dtype.itemsize)
        view = tl.TrainOnlyNpz(path, {"train"})
        np.testing.assert_equal(view["lm48_z"], [[1.0, 2.0]])
        self.assertEqual(tl.COUNTERS["dlval_feature_rows_read"], 0)


if __name__ == "__main__":
    unittest.main()
