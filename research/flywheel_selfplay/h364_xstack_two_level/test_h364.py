"""H364 data-free tests (synthetic arrays only). Run from the staged code root (stage_h364.sh), which also holds the
H354 modules: PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest test_h364 -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h364_core as H  # noqa: E402

K, ST, E, C = H.K, H.ST, H.E, H.C


def brute_qp(A, g, c, n=60):
    """Dense search over the feasible set {b >= 0, sum b <= c} (J <= 3) followed by local refinement."""
    J = len(g)
    best, bf = np.zeros(J), 0.0
    pts = np.linspace(0, c, n)
    for b in np.array(np.meshgrid(*([pts] * J))).reshape(J, -1).T:
        if b.sum() <= c:
            f = b @ A @ b + 2 * g @ b
            if f < bf:
                best, bf = b, f
    return best, bf


def synth(seed=0, n0=150, n1=140, na=120, k=3, lib_noise=(0.6, 0.8, 1.0, 1.4)):
    """A context whose stored X0 spec is the frozen S1-direct stack of k GP members, plus 4 library learners."""
    import test_h354 as T

    ctx = T.synth_ctx(seed, n0, n1, na, k)
    w0, w1, wa = ST.weights_s1_direct(ctx.members, ctx.fd)
    P0, P1, _, comps = ST._mats(ctx.members)
    spec = ST.StackSpec("CV3@12", [m.sha for m in ctx.members], [m.dsl for m in ctx.members], [-1] * k,
                        np.asarray(wa), np.asarray(w0), np.asarray(w1), ST.crossfit_search_s1(P0, P1, ctx.fd, w0, w1,
                                                                                               comps))
    raw = ST.stack_audit_utility(spec, ctx.audit_by_sha, ctx.ya, ctx.ba, ctx.y_search_mean)
    ctx.stored = {"spec": spec.to_json(), "raw": {"utility": raw["utility"]}}
    rng = np.random.default_rng(seed + 100)
    fd = ctx.fd
    ls, la = {}, {}
    for n, s in zip(H.CLASSICAL, lib_noise):
        p0 = fd.y0 + rng.normal(size=len(fd.y0)) * s
        p1 = fd.y1 + rng.normal(size=len(fd.y1)) * s
        pa = ctx.ya + rng.normal(size=len(ctx.ya)) * s
        ls[n] = K.lib_search_result(ctx, n, p0, p1)
        la[f"lib:{n}"] = K.lib_audit_result(ctx, n, pa, "completed")
    return ctx, ls, la


class QPTests(unittest.TestCase):
    def test_exact_vs_brute(self):
        rng = np.random.default_rng(3)
        for t in range(40):
            J = 1 + t % 3
            M = rng.normal(size=(8, J))
            A = M.T @ M / 8 + (0.0 if t % 2 else 0.05) * np.eye(J)
            g = rng.normal(size=J)
            c = float(rng.choice([0.1, 0.35, 1.0]))
            b = H.qp_capped(A, g, c)
            self.assertTrue(np.all(b >= 0) and b.sum() <= c * (1 + 1e-9))
            _, bf = brute_qp(A, g, c)
            self.assertLessEqual(b @ A @ b + 2 * g @ b, bf + 1e-9)

    def test_zero_when_no_descent(self):
        A = np.eye(2)
        self.assertTrue(np.all(H.qp_capped(A, np.array([0.3, 0.1]), 0.5) == 0))
        self.assertTrue(np.all(H.qp_capped(A, np.array([-1.0, -1.0]), 0.0) == 0))


class SecondLevelTests(unittest.TestCase):
    def test_x0_check_and_exact_tie(self):
        ctx, ls, la = synth(1)
        d = H.Design(0.35, 0.0, 0.0, math.inf)
        rec = H.evaluate(ctx, ls, la, [d, H.Design(0.0, 0.0, 0.0, math.inf)])
        self.assertLessEqual(rec["x0_check"]["dlr"], 1e-12)
        self.assertLessEqual(rec["x0_check"]["du"], 1e-12)
        zero = rec["designs"][H.Design(0.0, 0.0, 0.0, math.inf).key()]
        self.assertFalse(zero["active"])
        self.assertEqual(zero["u"], rec["x0_u"])  # beta = 0 -> X0's utility exactly

    def test_useful_library_is_used_and_capped(self):
        ctx, ls, la = synth(2, lib_noise=(0.05, 0.05, 0.05, 0.05))  # library far better than the GP members
        for c in H.GRID_C:
            r = H.evaluate(ctx, ls, la, [H.Design(c, 0.0, 0.0, math.inf)])["designs"]
            v = next(iter(r.values()))
            self.assertTrue(v["active"])
            self.assertLessEqual(v["beta_sum"], c * (1 + 1e-9))
            self.assertGreater(v["u"], H.evaluate(ctx, ls, la, [])["x0_u"])

    def test_near_perfect_guard_and_gate(self):
        ctx, ls, la = synth(4, lib_noise=(0.05, 0.05, 0.05, 0.05))
        spec = H.x0_spec(ctx)
        xs = H.x0_search(ctx, spec)
        g = H.second_level(ctx, xs, ls, H.Design(0.5, 0.0, 0.0, xs.u - 1e-9))
        self.assertEqual((g["active"], g["reason"]), (False, "near_perfect_guard"))
        g = H.second_level(ctx, xs, ls, H.Design(0.5, 0.0, 5.0, math.inf))  # impossible margin
        self.assertEqual((g["active"], g["reason"]), (False, "acceptance_gate"))

    def test_beta_ignores_audit_arrays(self):
        ctx, ls, la = synth(5, lib_noise=(0.3, 0.5, 0.7, 0.9))
        d = H.Design(0.35, 0.02, 0.0, math.inf)
        spec = H.x0_spec(ctx)
        a = H.second_level(ctx, H.x0_search(ctx, spec), ls, d)
        import dataclasses

        ctx.ya = ctx.ya * 3.0 + 1.0
        ctx.audit_by_sha = {k: dataclasses.replace(s, pa=s.pa * -2.0) for k, s in ctx.audit_by_sha.items()}
        b = H.second_level(ctx, H.x0_search(ctx, spec), ls, d)
        self.assertEqual(a["beta"], b["beta"])
        self.assertEqual(a["reason"], b["reason"])

    def test_gp_relative_weights_unchanged(self):
        """2L's AUDIT prediction is x + D beta with x = the stored-weight X0 prediction: the GP members enter only
        through x, so their relative weights are the stored ones."""
        ctx, ls, la = synth(6, lib_noise=(0.2, 0.2, 0.2, 0.2))
        spec = H.x0_spec(ctx)
        xa = H.x0_audit(ctx, spec)
        sl = H.second_level(ctx, H.x0_search(ctx, spec), ls, H.Design(0.5, 0.0, 0.0, math.inf))
        self.assertTrue(sl["active"])
        x = np.asarray(xa["pred"])
        pred = x + sum(b * (la[f"lib:{n}"].pa - x) for n, b in zip(sl["cols"], sl["beta"]))
        u = C.audit_utility_from_rmse(float(np.sqrt(np.mean((pred - ctx.ya) ** 2))), ctx.ba)
        self.assertAlmostEqual(H.audit_utility(ctx, xa, sl, la)["utility"], u, places=12)


class DesignTests(unittest.TestCase):
    def test_grid_is_the_prereg_grid(self):
        self.assertEqual(len(H.grid()), 144)
        self.assertEqual(H.GRID_C, (0.10, 0.20, 0.35, 0.50))
        self.assertEqual(H.GRID_LAMBDA, (0.0, 0.02, 0.10))
        self.assertEqual(H.GRID_DELTA, (0.0, 0.005, 0.02))
        self.assertEqual(H.GRID_TAU, (math.inf, 6.0, 4.0, 3.0))
        self.assertEqual(H.CLASSICAL, ("catboost.v1", "lightgbm.v1", "xgboost.v1", "random_forest.v1"))

    def test_select_tie_break(self):
        a, b = H.Design(0.2, 0.0, 0.0, math.inf), H.Design(0.1, 0.1, 0.0, 4.0)
        rows = [{"design": a, "lcb": 0.010}, {"design": b, "lcb": 0.0100000005}]
        self.assertEqual(H.select(rows)["design"], b)  # within 1e-6 -> smaller c
        rows = [{"design": a, "lcb": 0.012}, {"design": b, "lcb": 0.010}]
        self.assertEqual(H.select(rows)["design"], a)

    def test_design_json_roundtrip(self):
        for d in H.grid():
            self.assertEqual(H.Design.from_json(d.as_json()), d)

    def test_family_lcb(self):
        st = H.family_lcb({"a": 1.0, "b": 3.0, "c": 2.0}, {"a": "f", "b": "f", "c": "g"})
        self.assertEqual(st["families"], 2)
        self.assertAlmostEqual(st["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
