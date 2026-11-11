#!/usr/bin/env python3
"""Tests of deploy_h423.py (follow-up 3 of H422): the combiner. They read two stored FIT-SEL contexts (a design cohort; H420
TBOTH matrices and records) and synthetic data; no DEV, CONFIRM, P201b or dev91 file is opened and no worker runs.

  test_h423.py [-v]          environment: H423_TEST_RUN_DIR (default: the pulled H420 FIT-SEL copy on xbabe2)
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_h423 as D23  # noqa: E402
import meta_h416 as M  # noqa: E402
import sel_h422 as S  # noqa: E402
import train_h416 as T16  # noqa: E402

RUN = Path(os.environ.get("H423_TEST_RUN_DIR",
                          "/home/ubuntu/jope_plan/evidence/H420/pulled/H420_FIT-SEL/H420_FIT-SEL"))
TAG, SXB = "TBOTH", "P3+LIB3.SXB"


def stored_contexts(n=2):
    import apex_core_meta as AM

    rows = AM.rows()
    cids = json.loads((RUN / f"config_{TAG}.json").read_text())["cids"][:n]
    out = []
    for c in cids:
        if rows[c]["cohort"] != "FIT-SEL":
            raise AssertionError("the tests read FIT-SEL contexts only")
        rec = T16._last_result(RUN / "stages" / f"H416_{TAG}" / f"{c}__confirm.jsonl")
        out.append((c, rec, T16.load_npz(np, rec, RUN)))
    return out


class Flags:
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kw}
        for k, v in self.kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


OFF = {D23.FLAG_DESIGNS: None, D23.FLAG_ADMISSION: None}


class Combiner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = stored_contexts()

    def test_flags_default_off(self):
        with Flags(**OFF):
            self.assertFalse(D23.enabled(D23.FLAG_DESIGNS))
            self.assertFalse(D23.enabled(D23.FLAG_ADMISSION))
        for v in ("0", "true", "yes", " 1", ""):
            with Flags(**{D23.FLAG_DESIGNS: v}):
                self.assertFalse(D23.enabled(D23.FLAG_DESIGNS))

    def test_new_names_refused_when_off(self):
        c, rec, z = self.ctx[0]
        with Flags(**OFF):
            with self.assertRaises(ValueError):
                D23.design_pred("BCAR.mixw6", z["Q"], z["y"], c, z["bag"], z["refit"], z["refs"], z["ym"], z["na"])

    def test_stored_designs_unchanged_and_replay_record(self):
        for flags in (OFF, {D23.FLAG_DESIGNS: "1", D23.FLAG_ADMISSION: "1"}):
            with Flags(**flags):
                for c, rec, z in self.ctx:
                    for d in ("BCAR.mix", "KCAR100.bag", "CONS2"):
                        a = D23.design_pred(d, z["Q"], z["y"], c, z["bag"], z["refit"], z["refs"], z["ym"], z["na"])
                        b = M.design_pred(d, z["Q"], z["y"], c, z["bag"], z["refit"], z["refs"], z["ym"], z["na"])
                        self.assertTrue(np.array_equal(a[0], b[0]))
                        self.assertEqual(a[1], b[1])
                    p, _ = D23.design_pred("BCAR.mix", z["Q"], z["y"], c, z["bag"], z["refit"], z["refs"], z["ym"], z["na"])
                    self.assertLessEqual(abs(M.utility(p, z["ya"], z["ba"])[0] - float(rec["outputs"][SXB]["value"])), 1e-9)

    def test_winsor_equals_h422_offline_code(self):
        with Flags(**{D23.FLAG_DESIGNS: "1"}):
            for c, rec, z in self.ctx:
                w = M.meta_weights("BCAR", z["Q"], z["y"], c)
                mean = np.full(z["na"], z["ym"])
                REF = np.column_stack([r if r is not None else mean for r in z["refit"]])
                BAG = np.column_stack([b if b is not None else REF[:, i] for i, b in enumerate(z["bag"])])
                st = S.context_stats({"Q": z["Q"], "y": z["y"], "BAG": BAG, "REF": REF, "ya": z["ya"], "ba": z["ba"]}, w)
                for j, k in enumerate(S.KGRID[:-1]):
                    name = f"BCAR.mixw{k:g}"
                    p, info = D23.design_pred(name, z["Q"], z["y"], c, z["bag"], z["refit"], z["refs"], z["ym"], z["na"])
                    self.assertEqual(info["winsor_k"], k)
                    self.assertLessEqual(float(np.abs(p - st["P"][j]).max()), 1e-9 * (1 + float(np.abs(p).max())))
                    self.assertLessEqual(abs(M.utility(p, z["ya"], z["ba"])[0] - st["U"][j]), 1e-9)

    def test_huge_k_is_the_stored_design(self):
        with Flags(**{D23.FLAG_DESIGNS: "1"}):
            for c, rec, z in self.ctx:
                p, _ = D23.design_pred("BCAR.mixw1000000000000", z["Q"], z["y"], c, z["bag"], z["refit"], z["refs"],
                                       z["ym"], z["na"])
                self.assertLessEqual(abs(M.utility(p, z["ya"], z["ba"])[0] - float(rec["outputs"][SXB]["value"])), 1e-6)


class WinsorSynthetic(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(423001)
        self.na, m = 50, 5
        self.bag = [rng.normal(size=self.na) for _ in range(m)]
        self.refit = [b + 0.1 * rng.normal(size=self.na) for b in self.bag]
        self.w = rng.random(m) + 0.1
        self.scales = rng.random(m) + 0.5

    def test_k_zero_is_the_weighted_median(self):
        p, _ = D23.winsor_pred(self.w, self.bag, self.refit, self.scales, 0.0, self.na, 0.0)
        cols = np.column_stack(self.bag + self.refit)
        w2 = np.concatenate([self.w, self.w]) / (2 * self.w.sum())
        self.assertTrue(np.allclose(p, D23.wquant(cols, w2, 0.5), atol=1e-12))

    def test_within_member_range_and_monotone_to_mean(self):
        cols = np.column_stack(self.bag + self.refit)
        w2 = np.concatenate([self.w, self.w]) / (2 * self.w.sum())
        for k in (0.5, 2.0, 6.0):
            p, _ = D23.winsor_pred(self.w, self.bag, self.refit, self.scales, 0.0, self.na, k)
            self.assertTrue(np.all(p >= cols.min(1) - 1e-12) and np.all(p <= cols.max(1) + 1e-12))
        p, _ = D23.winsor_pred(self.w, self.bag, self.refit, self.scales, 0.0, self.na, 1e12)
        self.assertTrue(np.allclose(p, cols @ w2, atol=1e-12))

    def test_fallbacks_match_the_stored_mix(self):
        bag = list(self.bag)
        refit = list(self.refit)
        bag[1], refit[2], bag[3], refit[3] = None, None, None, None
        p, info = D23.winsor_pred(self.w, bag, refit, self.scales, 0.7, self.na, 1e12)
        q, qi = M.deploy_pred("mix", self.w, bag, refit, 0.7, self.na)
        self.assertTrue(np.allclose(p, q, atol=1e-12))
        for k in ("bag_full_refit_fallback", "bag_mean_substituted", "refit_mean_substituted"):
            self.assertEqual(info[k], qi[k])

    def test_one_outlying_member_is_pulled_back(self):
        bag = [np.zeros(self.na) for _ in range(5)]
        bag[4] = np.full(self.na, 100.0)
        w = np.full(5, 0.2)
        p, _ = D23.winsor_pred(w, bag, bag, np.ones(5), 0.0, self.na, 3.0)
        self.assertTrue(np.allclose(p, 0.2 * 3.0))   # the stored mean would give 20


if __name__ == "__main__":
    unittest.main()
