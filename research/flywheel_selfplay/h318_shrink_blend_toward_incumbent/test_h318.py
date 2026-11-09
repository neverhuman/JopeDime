import unittest, tempfile, zipfile, sys, json, os
from pathlib import Path
import numpy as np
import runtime as r


class BoundaryTests(unittest.TestCase):
    def test_score_blend_endpoints_and_exact_half(self):
        p = np.arange(8, dtype=np.int32)
        for a in [0, 0.25, 0.5]:
            np.testing.assert_array_equal(
                r.blend_picks(p, a), np.zeros(8, dtype=np.int32)
            )
        for a in [0.75, 1]:
            np.testing.assert_array_equal(r.blend_picks(p, a), p)

    def test_incorrect_alpha_rejected(self):
        with self.assertRaises(AssertionError):
            r.blend_picks([1], 0.3)

    def test_pick_out_of_range_rejected(self):
        with self.assertRaises(AssertionError):
            r.blend_picks([8], 1)

    def test_alpha_tie_incumbent(self):
        ch, t = r.choose_alpha(
            [1, 1],
            ["a", "b"],
            {"a": {"utility_k": [1, 1]}, "b": {"utility_k": [1, 1]}},
            {"a": "x", "b": "y"},
        )
        self.assertEqual(ch["alpha"], 0)

    def test_alpha_positive_ties_to_point75(self):
        ch, t = r.choose_alpha(
            [1, 1],
            ["a", "b"],
            {"a": {"utility_k": [1, 1.1]}, "b": {"utility_k": [1, 1.2]}},
            {"a": "x", "b": "y"},
        )
        self.assertEqual(ch["alpha"], 0.75)
        self.assertAlmostEqual(ch["objective"], 0.15)

    def test_family_equal_median_and_harm(self):
        cids = ["a", "b", "c", "d"]
        pp = [1] * 4
        P = {c: {"utility_k": [0, v]} for c, v in zip(cids, [1, 1, 1, -0.02])}
        f = {"a": "x", "b": "x", "c": "x", "d": "y"}
        ch, table = r.choose_alpha(pp, cids, P, f)
        row = table[-1]
        self.assertAlmostEqual(row["family_median_gain"], 0.49)
        self.assertEqual(row["harmed_rate"], 0.5)
        self.assertAlmostEqual(row["objective"], 0.24)

    def test_harm_strict_point01(self):
        ch, t = r.choose_alpha([1], ["a"], {"a": {"utility_k": [0, -0.01]}}, {"a": "x"})
        self.assertEqual(t[-1]["harmed_rate"], 0)
        self.assertEqual(ch["alpha"], 0)

    def test_discontiguous_ranges_do_not_read_poisoned_outer_rows(self):
        with tempfile.TemporaryDirectory(dir=r.P) as td:
            p = Path(td) / "cache.npz"
            a = np.arange(24, dtype=np.float32).reshape(6, 4)
            np.savez(p, x=a)
            off, _, _ = r.tl._member_layout(p, "x.npy")
            with p.open("r+b") as f:
                for i in [1, 3, 4, 5]:
                    f.seek(off + i * 16)
                    f.write(np.full(4, np.nan, dtype=np.float32).tobytes())
            with np.load(p) as z, self.assertRaises(zipfile.BadZipFile):
                z["x"]
            s = r.Scope({"t0", "t2"}, {"t0", "t1", "t2", "t3"})
            s.expected = {(str(p), "x", "t0"): (0, 1), (str(p), "x", "t2"): (2, 3)}
            np.testing.assert_array_equal(s.read(p, "x", 0, 1, "t0"), a[:1])
            np.testing.assert_array_equal(s.read(p, "x", 2, 3, "t2"), a[2:3])
            self.assertEqual(s.bytes, 32)
            with self.assertRaises(PermissionError):
                s.read(p, "x", 1, 2, "t1")
            with self.assertRaises(PermissionError):
                s.read(p, "x", 1, 2, "t0")
            self.assertEqual(s.bytes, 32)

    def test_pool_open_denied_before_os(self):
        s = r.Scope({"train"}, {"train", "outer"}, True)
        sys.addaudithook(s.hook)
        try:
            with self.assertRaises(PermissionError):
                open("/home/ubuntu/jope-data/h50/input/pools/outer.npz", "rb")
            with self.assertRaises(PermissionError):
                open("/home/ubuntu/jope-data/h318-astra/references/unknown.npz", "rb")
        finally:
            s.active = False

    def test_preregistered_fold_disjointness(self):
        fs = ["fam" + str(i) for i in range(205)]
        for p in r.PARTS:
            buckets = [{f for f in fs if r.fold(p, f) == k} for k in range(4)]
            self.assertEqual(set.union(*buckets), set(fs))
            self.assertEqual(sum(map(len, buckets)), len(fs))

    def test_durable_lock_json(self):
        with tempfile.TemporaryDirectory(dir=r.P) as td:
            p = Path(td) / "fit_lock.json"
            r.durable(p, {"alpha": 0.75})
            self.assertEqual(json.loads(p.read_text()), {"alpha": 0.75})
            self.assertFalse(p.with_suffix(".new").exists())


if __name__ == "__main__":
    unittest.main()
