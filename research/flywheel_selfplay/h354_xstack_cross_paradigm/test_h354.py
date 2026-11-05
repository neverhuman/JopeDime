"""H354 data-free tests (no TRAIN or DL-Val file is opened; synthetic arrays only).

Run: PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest test_h354 -v   (from this directory)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h354_core as K  # noqa: E402
import lib_oof_h354 as LO  # noqa: E402
import phase_d_h354 as PD  # noqa: E402

ST, E = K.ST, K.E


def synth_ctx(seed: int = 0, n0: int = 120, n1: int = 110, na: int = 90, k: int = 4) -> "K.Context":
    rng = np.random.default_rng(seed)
    y0, y1, ya = rng.normal(size=n0) * 3, rng.normal(size=n1) * 3, rng.normal(size=na) * 3
    fd = ST.FoldData(y0, y1, float(np.sqrt(np.mean((y0 - y1.mean()) ** 2))),
                     float(np.sqrt(np.mean((y1 - y0.mean()) ** 2))))
    members, audit = [], {}
    for i in range(k):
        p0, p1 = y0 + rng.normal(size=n0) * (0.5 + 0.2 * i), y1 + rng.normal(size=n1) * (0.5 + 0.2 * i)
        s = ST.s1_from_lr(ST._lr(p0, y0, fd.b0), ST._lr(p1, y1, fd.b1), 3.0)
        sha = f"m{i}"
        members.append(E.SearchResult("c", sha, f"(pipe X (ridge) {i})", "completed", s["utility"], s["fitness"],
                                      s["gate"], s["lr0"], s["lr1"], s["mean_lr"], s["tail_lr"], s["sd_lr"], 3, p0, p1))
        pa = ya + rng.normal(size=na) * (0.5 + 0.2 * i)
        audit[sha] = E.AuditResult("c", sha, "", "completed", 0.0, float(np.sqrt(np.mean((pa - ya) ** 2))), 1.0, pa)
    members.sort(key=lambda r: (r.fitness, r.sha))
    env = K.Envelope(np.zeros(1, np.uint8), np.zeros((1, 1)), np.zeros(1), "", 0)
    ysm = 0.0
    ba = float(np.sqrt(np.mean((ya - ysm) ** 2)))
    return K.Context("c", Path("."), "f", env, env, fd, members, audit, {}, ya, ba, ysm)


class EnvelopeTests(unittest.TestCase):
    def test_roundtrip_and_worker_form(self):
        rng = np.random.default_rng(1)
        X, y, folds = rng.normal(size=(7, 3)), rng.normal(size=7), np.array([0, 1, 0, 1, 1, 0, 0], np.uint8)
        env = K.Envelope(folds, X, y, "", 0)
        raw = K.train_only_bytes(env, folds)
        self.assertEqual(raw, LO.env_bytes(X, y, folds))
        tmp = K.DATA / "test" / (self.id().replace(".", "_") + ".jtt1")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_bytes(raw + b"")
            got = K.read_envelope(tmp)
            np.testing.assert_array_equal(got.folds, folds)
            np.testing.assert_array_equal(got.X, X)
            np.testing.assert_array_equal(got.y, y)
            # worker form: nq = 10 all-zero query records are accepted and ignored
            import struct
            w = bytearray(raw)
            struct.pack_into("<I", w, 8, 10)
            w.extend(bytes(10 * 4 * 8))
            tmp.write_bytes(bytes(w))
            self.assertEqual(K.read_envelope(tmp).nq, 10)
            w[-1] = 1  # a non-zero query cell is refused
            tmp.write_bytes(bytes(w))
            with self.assertRaises(ValueError):
                K.read_envelope(tmp)
        finally:
            tmp.write_bytes(b"")  # left empty under $H354_DATA/test, never deleted (no-delete rule)

    def test_llm_fold_relabel(self):
        """p0 envelope: SEARCH fold 1 rows are the fit rows (byte 0), fold 0 rows are predicted (byte 1), same order."""
        folds = np.array([0, 1, 1, 0, 1], np.uint8)
        relab = np.where(folds == 1, 0, 1)
        np.testing.assert_array_equal(relab, [1, 0, 0, 1, 0])
        self.assertEqual(int((relab == 1).sum()), int((folds == 0).sum()))


class StackTests(unittest.TestCase):
    def test_x0_equals_frozen_stacker(self):
        ctx = synth_ctx()
        spec = K.stack_arm(ctx, {}, "X0")
        w0, w1, wa = ST.weights_s1_direct(ctx.members, ctx.fd)
        np.testing.assert_array_equal(spec.w_audit, wa)
        np.testing.assert_array_equal(spec.w_fold0, w0)
        np.testing.assert_array_equal(spec.w_fold1, w1)

    def test_library_gate_and_failures(self):
        ctx = synth_ctx()
        good = K.lib_search_result(ctx, "catboost.v1", ctx.fd.y0 + 0.1, ctx.fd.y1 + 0.1)
        bad_nan = K.lib_search_result(ctx, "lightgbm.v1", np.full(len(ctx.fd.y0), np.nan), ctx.fd.y1)
        failed = K.lib_search_result(ctx, "xgboost.v1", None, None, "failed", "boom")
        worse = K.lib_search_result(ctx, "random_forest.v1", ctx.fd.y0 * 0 + 50.0, ctx.fd.y1 * 0 + 50.0)
        self.assertTrue(good.ok and good.gate)
        self.assertFalse(bad_nan.ok)
        self.assertFalse(failed.ok)
        self.assertFalse(worse.gate)  # worse than the fold mean -> fails the S1 gate -> never a candidate
        ls = {"catboost.v1": good, "lightgbm.v1": bad_nan, "xgboost.v1": failed, "random_forest.v1": worse}
        spec = K.stack_arm(ctx, ls, "X_CL")
        self.assertEqual([s for s in spec.members if s.startswith("lib:")], ["lib:catboost.v1"])
        # a near-perfect library member takes most of the weight
        self.assertGreater(float(spec.w_audit[-1]), 0.5)

    def test_weights_ignore_audit(self):
        """The stacker sees SEARCH only: perturbing every AUDIT array leaves the weights bit-identical."""
        ctx = synth_ctx(3)
        ls = {n: K.lib_search_result(ctx, n, ctx.fd.y0 + 0.3, ctx.fd.y1 - 0.2) for n in K.CLASSICAL}
        a = K.stack_arm(ctx, ls, "X_CL")
        ctx.ya = ctx.ya[::-1] * 7.0
        ctx.audit_by_sha = {s: E.AuditResult("c", s, "", "failed", -2.0, None, None, None) for s in ctx.audit_by_sha}
        b = K.stack_arm(ctx, ls, "X_CL")
        np.testing.assert_array_equal(a.w_audit, b.w_audit)
        np.testing.assert_array_equal(a.w_fold0, b.w_fold0)

    def test_audit_substitution(self):
        ctx = synth_ctx(4)
        ls = {n: K.lib_search_result(ctx, n, ctx.fd.y0 + 0.05, ctx.fd.y1 + 0.05) for n in K.CLASSICAL}
        spec = K.stack_arm(ctx, ls, "X_CL")
        abs_ = dict(ctx.audit_by_sha)
        for n in K.CLASSICAL:
            abs_[f"lib:{n}"] = K.lib_audit_result(ctx, n, None, "failed")
        out = K.audit_arm(ctx, spec, abs_)
        self.assertGreater(out["substituted_weight_mass"], 0.0)
        self.assertEqual(out["status"], "completed")
        self.assertAlmostEqual(out["library_weight"], float(sum(
            w for s, w in zip(spec.members, spec.w_audit) if s.startswith("lib:"))))


class DecisionTests(unittest.TestCase):
    def test_one_se_rule(self):
        g = {"X_ALL": {"gap": 0.0, "se": 0.0}, "X_CL": {"gap": 0.004, "se": 0.005}, "X_LLM": {"gap": 0.02, "se": 0.005}}
        elig, pick = PD.pick_arm(g)
        self.assertEqual(sorted(elig), ["X_ALL", "X_CL"])
        self.assertEqual(pick, "X_CL")  # fewer library members than X_ALL
        g["X_LLM"]["gap"] = 0.001
        self.assertEqual(PD.pick_arm(g)[1], "X_LLM")  # tie in size with X_CL -> the cheaper library

    def test_load_context_defaults_are_phase_d(self):
        import inspect
        sig = inspect.signature(K.load_context)
        self.assertEqual(sig.parameters["runs"].default, K.H335_RUNS)
        self.assertEqual((sig.parameters["stage"].default, sig.parameters["seed"].default), ("T_AGG", 335101))

    def test_arms_are_fixed(self):
        self.assertEqual(K.ARMS["X0"], ())
        self.assertEqual(set(K.ARMS["X_ALL"]), set(K.CLASSICAL) | set(K.LLM))
        self.assertEqual(K.STACK_KEY, "AGG.g50.CV3@12")
        self.assertEqual(K.C_LIB, 0.0)


if __name__ == "__main__":
    unittest.main()
