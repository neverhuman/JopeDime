#!/usr/bin/env python3
"""Tests of deploy_h423's pooled admission against the stored rule (deploy_h415.hf_candidates), on synthetic search
results. A separate process from test_h423.py: importing deploy_h415 installs the H317 guard, which then refuses to
open files under the evidence tree. No file of any cohort is opened and no worker runs.

  test_admission_h423.py [-v]
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_h423 as D23  # noqa: E402
from test_h423 import OFF, Flags  # noqa: E402


class Admission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import deploy_h415 as DP15

        cls.DP15 = DP15
        heads = ("ridge_gcv", "elasticnet", "krr", "histgbm", "xtrees")

        def res(i, fit, head, tag):
            dsl = f"(pipe X (standardize append=false) ({head}))"
            return DP15.E.SearchResult(cid="c", sha=f"{tag}{i:04d}".ljust(64, "0"), dsl=dsl, status="completed",
                                       utility=-fit, fitness=fit, gate=True, lr0=0.0, lr1=0.0, mean_lr=0.0, tail_lr=0.0,
                                       sd_lr=0.0, complexity=1.0, p0=None, p1=None, wall_s=0.0, error="")

        rng = np.random.default_rng(423002)
        cls.a = [res(i, float(rng.random()), "ridge_gcv" if i < 70 else heads[i % 5], "a") for i in range(90)]
        cls.b = [res(i, float(rng.random()), heads[i % 5], "b") for i in range(60)] + cls.a[:20]

    def test_off_is_the_stored_rule(self):
        hf = self.DP15.hf_candidates
        for flags in (OFF, {D23.FLAG_ADMISSION: "0"}):
            with Flags(**flags):
                for sub in (None, [self.a, self.b]):
                    H, n = D23.hf_candidates(self.a + self.b, hf, sub)
                    H0, n0 = hf(self.a + self.b)
                    self.assertEqual([r.sha for r in H], [r.sha for r in H0])
                    self.assertEqual(n, n0)

    def test_on_without_sub_pools_is_the_stored_rule(self):
        hf = self.DP15.hf_candidates
        with Flags(**{D23.FLAG_ADMISSION: "1"}):
            H, n = D23.hf_candidates(self.a, hf)
            self.assertEqual(([r.sha for r in H], n), ([r.sha for r in hf(self.a)[0]], hf(self.a)[1]))

    def test_on_is_the_ordered_union(self):
        hf = self.DP15.hf_candidates
        with Flags(**{D23.FLAG_ADMISSION: "1"}):
            H, n = D23.hf_candidates(self.a + self.b, hf, [self.a, self.b])
        Ha, Hb = [r.sha for r in hf(self.a)[0]], [r.sha for r in hf(self.b)[0]]
        got = [r.sha for r in H]
        self.assertEqual(got[:len(Ha)], Ha)
        self.assertEqual(len(got), len(set(got)))
        self.assertEqual(set(got), set(Ha) | set(Hb))
        self.assertEqual(n, len(got) - len(Ha))
        self.assertEqual([s for s in got if s not in set(Ha)], [s for s in Hb if s not in set(Ha)])
        with Flags(**{D23.FLAG_ADMISSION: "1"}):
            H1, n1 = D23.hf_candidates(self.a, hf, [self.a])
        self.assertEqual(([r.sha for r in H1], n1), (Ha, 0))


if __name__ == "__main__":
    unittest.main()
