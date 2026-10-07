"""LINEAR-DETERMINED gate (dltrain/target_linear.py; owner-decisions-20261006T112914Z.json). Every refusal is paired with a
control that must not refuse.

  Ols        exact multi-column linear y (with intercept) refused; noisy near-linear flagged for review; random admitted;
             constant y / too few rows give no verdict; non-finite rows dropped; the method equals scan_lindet.py's
  TrainOnly  check_jtt1 decodes outer-train rows only: linear test rows never refuse, changed test bytes never change
             the verdict, linear train rows refuse whatever the test rows hold
  Control    the must-fail control context 2a5429bf (a423c0d1ed051081, PUBLISH-rowsel-raw-nonnumeric-v2) is refused
             (read-only; skipped where that private tree is absent)
"""
import gzip
import json
import unittest
from pathlib import Path

import numpy as np

from dltrain import jtt1, target_linear as tl, trainonly

CONTROL = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/build-v2/PUBLISH-rowsel-raw-nonnumeric-v2")


def data(n=400, p=6, seed=1, noise=0.0, linear=True):
    rs = np.random.default_rng(seed)
    X = rs.uniform(0, 1, (n, p))
    y = (X @ rs.normal(size=p) + 0.7) if linear else rs.normal(size=n)
    return X, y + (rs.normal(0, noise, n) if noise else 0.0)


def payload(Xtr, ytr, Xte, yte):
    labels = np.zeros(len(ytr), dtype=np.uint8)
    return trainonly.jtt1_bytes(np.column_stack([Xtr, ytr]), np.column_stack([Xte, yte]), labels)


class Ols(unittest.TestCase):
    def test_exact_multi_column_linear_target_is_refused(self):
        X, y = data()
        r = tl.check_train_rows(X, y)
        self.assertEqual(r["verdict"], tl.REFUSE)
        self.assertLess(r["ols_rel_rmse"], 1e-8)
        Xs = X.copy(); Xs[:, 2] = 0.0                                   # a constant column: rank-deficient, still exact
        self.assertEqual(tl.check_train_rows(Xs, Xs[:, :2].sum(axis=1) * 3 - 1)["verdict"], tl.REFUSE)

    def test_near_linear_is_flagged_and_random_is_admitted(self):
        X, y = data(noise=1e-5)
        r = tl.check_train_rows(X, y)
        self.assertEqual(r["verdict"], tl.REVIEW, r["ols_rel_rmse"])
        X2, y2 = data(noise=5e-4)                                        # rel ~ 4e-4: still below 1e-3
        self.assertEqual(tl.check_train_rows(X2, y2)["verdict"], tl.REVIEW, tl.check_train_rows(X2, y2)["ols_rel_rmse"])
        X3, y3 = data(linear=False)
        self.assertIsNone(tl.check_train_rows(X3, y3)["verdict"])
        self.assertEqual((tl.verdict(1e-8), tl.verdict(9.99e-9), tl.verdict(1e-3), tl.verdict(9.99e-4), tl.verdict(None)),
                         (tl.REVIEW, tl.REFUSE, None, tl.REVIEW, None))

    def test_the_intercept_is_part_of_the_fit(self):
        X, y = data()
        self.assertEqual(tl.check_train_rows(X, y + 1000.0)["verdict"], tl.REFUSE)

    def test_no_verdict_without_enough_finite_rows_or_variance(self):
        X, y = data(n=8, p=6)
        self.assertIsNone(tl.check_train_rows(X, y)["ols_rel_rmse"])       # 8 rows: not more than p + 2
        X, y = data(n=9, p=6)
        self.assertEqual(tl.check_train_rows(X, y)["verdict"], tl.REFUSE)  # 9 rows: fitted (as the scan)
        X, _ = data()
        self.assertIsNone(tl.check_train_rows(X, np.full(len(X), 3.0))["verdict"])

    def test_non_finite_rows_are_dropped(self):
        X, y = data()
        X[::7, 1] = np.nan
        y[::11] = np.inf
        r = tl.check_train_rows(X, y)
        self.assertEqual(r["verdict"], tl.REFUSE)
        self.assertEqual(r["rows_finite"], int(np.sum(np.isfinite(X).all(axis=1) & np.isfinite(y))))

    def test_same_numbers_as_the_scan(self):
        """scan_lindet.py's computation inline (66c8b37d) gives the same report on the same rows."""
        X, y = data(noise=1e-6, seed=4)
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        A = np.c_[X[ok], np.ones(int(ok.sum()))]
        coef, _, rank, _ = np.linalg.lstsq(A, y[ok], rcond=None)
        rel = float(np.sqrt(np.mean((y[ok] - A @ coef) ** 2)) / float(np.std(y[ok])))
        r = tl.ols_rel_rmse(X, y)
        self.assertEqual((r["ols_rel_rmse"], r["ols_rank"]), (rel, int(rank)))


class TrainOnly(unittest.TestCase):
    def test_linear_test_rows_never_refuse(self):
        Xtr, _ = data(linear=False, seed=5)
        ytr = np.random.default_rng(6).normal(size=len(Xtr))
        Xte, yte = data(n=100, seed=7)                                    # exactly linear OUTER-TEST rows
        self.assertIsNone(tl.check_jtt1(payload(Xtr, ytr, Xte, yte))["verdict"])

    def test_linear_train_rows_refuse_whatever_the_test_rows_hold(self):
        Xtr, ytr = data(seed=8)
        Xte = np.random.default_rng(9).uniform(0, 1, (100, 6)); yte = np.random.default_rng(10).normal(size=100)
        raw = payload(Xtr, ytr, Xte, yte)
        r = tl.check_jtt1(raw)
        self.assertEqual(r["verdict"], tl.REFUSE)
        start, stop = jtt1.train_range(jtt1.parse_header(raw))
        flipped = raw[:stop] + bytes(b ^ 0xFF for b in raw[stop:])        # every outer-test byte changed
        self.assertEqual(tl.check_jtt1(flipped), r)
        self.assertEqual(r["n_train"], len(ytr))


class Control(unittest.TestCase):
    def test_2a5429bf_is_refused(self):
        hits = [p for p in sorted(CONTROL.glob("shard-*/*/context.json"))
                if json.loads(p.read_bytes()).get("context_sha256", "").startswith("2a5429bf")] if CONTROL.is_dir() else []
        if not hits:
            self.skipTest("private control tree not present")
        self.assertEqual(len(hits), 1)
        r = tl.check_jtt1(gzip.decompress((hits[0].parent / "encoder.jtt1.gz").read_bytes()))
        self.assertEqual(r["verdict"], tl.REFUSE)
        self.assertLess(r["ols_rel_rmse"], 1e-12)


if __name__ == "__main__":
    unittest.main()
