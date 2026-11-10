import unittest, tempfile, zipfile, sys, json, os
from pathlib import Path
import numpy as np
import runtime as r


class BoundaryTests(unittest.TestCase):
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
                open("/home/ubuntu/jope-data/h327-astra/references/unknown.npz", "rb")
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
