"""Synthetic tests for dltrain.scale_check (UPSTREAM-SCALE-01 counterfactual check). numpy and fractions only."""
import collections
import decimal
import math
import sys
import unittest
from fractions import Fraction as Fr
from pathlib import Path
from unittest import mock

import numpy as np

from dltrain import common
from dltrain import scale_check as sc


# --------------------------------------------------------------------------------------------------- synthetic data
def synth_integer_matrix(n=240, F=4, n_test=60, seed=0):
    """Integer raw values (exact rationals), target last, with test-only extremes planted in column 0 (max) and in the
    target (min). Returns (M int matrix, R, T)."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 40, size=(n, F)).astype(np.int64)
    y = X @ rng.integers(1, 4, size=F) + rng.integers(0, 7, size=n)
    M = np.column_stack([X, y]).astype(np.int64)
    T = sorted(int(t) for t in rng.choice(n, n_test, replace=False))
    tset = set(T)
    R = [i for i in range(n) if i not in tset]
    M[T[0], 0] = 55                       # a test-only maximum of feature 0
    M[T[1], F] = int(M[:, F].min()) - 9   # a test-only minimum of the target
    return M, R, T


def exact_scale(M):
    """The upstream all-rows min-max scaling, in exact rational arithmetic."""
    cols = M.shape[1]
    lo = [Fr(int(M[:, j].min())) for j in range(cols)]
    hi = [Fr(int(M[:, j].max())) for j in range(cols)]
    return [[(Fr(int(M[i, j])) - lo[j]) / (hi[j] - lo[j]) for j in range(cols)] for i in range(M.shape[0])]


def exact_phi(rows):
    """phi over the given rows (exact): (x - min)/(max - min), divisor 1 for a constant column."""
    cols = len(rows[0])
    out = []
    lo = [min(r[j] for r in rows) for j in range(cols)]
    hi = [max(r[j] for r in rows) for j in range(cols)]
    for r in rows:
        out.append([(r[j] - lo[j]) / ((hi[j] - lo[j]) or 1) for j in range(cols)])
    return out


def to_f64(rows):
    return np.asarray([[float(x) for x in r] for r in rows], dtype=np.float64)


def exact_percentile(col, q, n):
    """Linear-method percentile with Codex's virtual index fl64((n-1)*fl64(q/100)), taken exactly as a rational."""
    s = sorted(col)
    vi = Fr(float((n - 1) * np.true_divide(q, 100)))
    i = math.floor(vi)
    if i >= n - 1:
        return s[-1]
    return s[i] + (s[i + 1] - s[i]) * (vi - i)


def exact_maps(rows, kept):
    """Exact-arithmetic versions of the AI component maps on train rows (list of rational rows, target last)."""
    n = len(rows)
    out = {"hyperion_t": [], "foundation_v": [], "target_v": None}
    for j in kept:
        col = [r[j] for r in rows]
        lo, hi = exact_percentile(col, 1, n), exact_percentile(col, 99, n)
        mn, mx = min(col), max(col)
        out["hyperion_t"].append([(x - lo) / (hi - lo) * 255 if hi != lo else None for x in col])
        out["foundation_v"].append([(x - mn) / (mx - mn) if mx != mn else Fr(0) for x in col])
    y = [r[-1] for r in rows]
    out["target_v"] = [(x - min(y)) / (max(y) - min(y)) for x in y]
    return out


def yz_decimal(y, ctx):
    """yz = (cy - mean)/(std + 1e-8) in 60-digit decimal arithmetic (the std is irrational)."""
    with decimal.localcontext() as dc:
        dc.prec = 60
        cy = [decimal.Decimal(y[i].numerator) / decimal.Decimal(y[i].denominator) for i in ctx]
        mu = sum(cy) / len(cy)
        sd = (sum((v - mu) ** 2 for v in cy) / len(cy)).sqrt()
        return [(v - mu) / (sd + decimal.Decimal("1e-8")) for v in cy]


def synthetic_ci(A, R, T, kept_jtt1=None):
    F = A.shape[1] - 1
    kept_jtt1 = list(range(F)) if kept_jtt1 is None else list(kept_jtt1)
    raw, _labels = sc.jtt1_rebuild(A, R, T, kept_jtt1, False)
    return {"A": A, "R": list(R), "T": list(T), "F": F, "kept_jtt1": kept_jtt1, "labels_from_parent": False,
            "refs": {"input_raw_sha256": common.sha256_bytes(raw)}, "packet_dataset": None, "packet_fold": None,
            "subsampled": False, "cohort": "frozen", "fold": 0}


def walk_positions(tc, doc, train_rows):
    """Assert every listed position names an outer-train row; return how many were seen."""
    seen = 0
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k == "positions":
                for p in v:
                    tc.assertIn(p[0], train_rows)
                    seen += 1
            else:
                seen += walk_positions(tc, v, train_rows)
    elif isinstance(doc, list):
        for v in doc:
            seen += walk_positions(tc, v, train_rows)
    return seen


# ------------------------------------------------------------------------------------------------- Codex replica
class TestCodexReplica(unittest.TestCase):
    def test_percentile_np126_equals_list_q_float64_lerp(self):
        """The 1.26.4 emulation equals np.percentile with a list q (a float64 gamma, so a float64 lerp), with ties."""
        rng = np.random.default_rng(7)
        for n in (1, 2, 3, 5, 17, 100, 1600, 1614, 2000):
            X = rng.random((n, 6)).astype(np.float32)
            X[:, 1] = np.round(X[:, 1] * 10) / 10            # lattice values
            X[: max(1, n // 2), 2] = X[0, 2]                   # heavy ties
            X[:, 3] = 0.25                                     # constant
            for q in (1, 99):
                got = sc.percentile_np126(X, q)
                want = np.percentile(X, [q], axis=0)[0].astype("float32")
                self.assertEqual(got.dtype, np.float32)
                self.assertTrue(np.array_equal(got.view(np.uint32), want.view(np.uint32)), (n, q))

    @unittest.skipIf(np.lib.NumpyVersion(np.__version__) < "2.0.0", "the version trap needs numpy 2")
    def test_numpy2_scalar_q_is_the_version_trap(self):
        """numpy 2 with a scalar q lerps in float32 and differs from Codex's numpy 1.26.4 on some columns."""
        rng = np.random.default_rng(11)
        X = rng.random((1614, 4000)).astype(np.float32)
        native = np.percentile(X, 1, axis=0).astype("float32")
        mine = sc.percentile_np126(X, 1)
        self.assertGreater(int(np.count_nonzero(native.view(np.uint32) != mine.view(np.uint32))), 0)

    def test_mean_std_np126(self):
        """mean_np126 is numpy's float32 mean bit for bit; std_np126 follows numpy 1.26.4 (_var: a float32 division of
        the keepdims sum, float32 squares and sum, a float64 division by n, float32 sqrt) and agrees with numpy 2's
        cy.std() except on double-rounding ties."""
        rng = np.random.default_rng(23)
        disagreements = 0
        for trial in range(400):
            n = int(rng.choice([2, 3, 7, 84, 252, 768]))
            a = (rng.random(n) * rng.choice([1e-3, 1.0, 37.0])).astype(np.float32)
            if trial % 7 == 0:
                a = np.round(a * 8) / 8                           # lattice values
            m = sc.mean_np126(a)
            self.assertEqual(m.dtype, np.float32)
            self.assertEqual(m.view(np.uint32), np.float32(a.mean()).view(np.uint32))
            s = sc.std_np126(a)
            self.assertEqual(s.dtype, np.float32)
            mean32 = np.float32(np.add.reduce(a)) / np.float32(n)            # float32 scalar division
            d = (a - np.float32(mean32)).astype(np.float32)
            v = np.float32(np.float64(np.add.reduce(d * d)) / n)
            self.assertEqual(s.view(np.uint32), np.float32(np.sqrt(v)).view(np.uint32))
            disagreements += int(s.view(np.uint32) != np.float32(a.std()).view(np.uint32))
        self.assertLessEqual(disagreements, 2)

    def test_target_span_is_the_numpy126_float64_difference(self):
        y = np.asarray([np.float32(0.1), np.float32(1.0 / 3.0), np.float32(0.7)], dtype=np.float32)
        t = sc.target_inputs(np.tile(y, 20))
        self.assertEqual(t["span"], float(np.float64(np.float32(0.7)) - float(np.float32(0.1))))
        self.assertEqual(t["v"].dtype, np.float32)
        self.assertEqual(t["classical"].shape, (1, 2, 32))
        self.assertEqual(t["neural"].shape, (1, 2, 32))

    def test_small_n_target_windows_duplicate(self):
        t = sc.target_inputs(np.linspace(0, 1, 20, dtype=np.float32))
        self.assertTrue(np.array_equal(t["q"][0], t["q"][1]))   # one window (n < 32) is duplicated
        self.assertEqual(t["c"].shape, (2, 32))                  # sampled with replacement

    def test_sampling_sizes(self):
        for n, nc, nq in ((2, 1, 1), (10, 7, 3), (1600, 768, 256), (336, 252, 84)):
            ctx, qry = sc.sampling(n)
            self.assertEqual((len(ctx), len(qry)), (nc, nq))
            self.assertFalse(set(ctx.tolist()) & set(qry.tolist()))

    def test_records_drops_an_exact_affine_target_copy(self):
        rng = np.random.default_rng(3)
        X = rng.random((50, 3))
        y = rng.random(50)
        raw = np.column_stack([X[:, 0], 2.0 * y + 0.5, X[:, 2], y])
        X32, y32, b = sc.records(raw)
        self.assertEqual(b["removed"], [1])
        self.assertEqual(b["kept"], [0, 2])
        self.assertEqual(X32.dtype, np.float32)
        self.assertEqual(X32.shape, (50, 2))

    def test_stable_seed_job_id_and_shard_owner(self):
        self.assertEqual(sc.stable_seed("a", 1), sc.stable_seed("a", 1))
        self.assertLess(sc.stable_seed("encode", 17, 1614, 32, 32), 2 ** 63)
        self.assertEqual(sc.indices_hash([0, 1]), common.sha256_bytes((0).to_bytes(8, "little") + (1).to_bytes(8, "little")))
        idx = [0, 5, 7, 2 ** 40]
        self.assertEqual(sc.indices_hash(idx), common.sha256_bytes(b"".join(int(x).to_bytes(8, "little") for x in idx)))
        self.assertEqual(sc.indices_hash([]), common.sha256_bytes(b""))
        with self.assertRaises(sc.ScaleCheckError):
            sc.indices_hash([3, -1])
        identity = ["014b93f43dbaa1a4", "c" * 64, "d" * 64]
        want = sc.CODEX_NODES[int(common.sha256_bytes(
            b'[["014b93f43dbaa1a4","' + b"c" * 64 + b'","' + b"d" * 64 + b'"],0]')[:16], 16) % 3]
        self.assertEqual(sc.codex_shard_owner(identity, 0), want)


# ------------------------------------------------------------------------------------------- exact arithmetic
class TestExactArithmeticInvariance(unittest.TestCase):
    """Exact rationals separate true non-invariance (an exact difference) from float rounding (exact values equal)."""

    @classmethod
    def setUpClass(cls):
        cls.M, cls.R, cls.T = synth_integer_matrix()
        cls.A_exact = exact_scale(cls.M)
        cls.AR_exact = [cls.A_exact[i] for i in cls.R]
        cls.B_exact = exact_phi(cls.AR_exact)
        cls.kept = list(range(cls.M.shape[1] - 1))

    def test_ai_components_are_exactly_invariant(self):
        ea, eb = exact_maps(self.AR_exact, self.kept), exact_maps(self.B_exact, self.kept)
        self.assertNotEqual(self.AR_exact, self.B_exact)           # the counterfactual really moves the bytes
        self.assertEqual(ea["foundation_v"], eb["foundation_v"])
        self.assertEqual(ea["target_v"], eb["target_v"])
        self.assertEqual(ea["hyperion_t"], eb["hyperion_t"])

    def test_raw_pass_through_and_absolute_floor_are_exactly_non_invariant(self):
        """Planted ops: the tabpfn35 raw arrays, an absolute span floor that fires in one variant only, and an
        additive mixture of raw values; each differs in exact arithmetic."""
        col_a = [r[0] for r in self.AR_exact]
        col_b = [r[0] for r in self.B_exact]
        self.assertNotEqual(col_a, col_b)                            # tabpfn35: raw values
        floor = Fr(999, 1000)                                        # an absolute span floor (Codex's are 1e-9, 1e-12)

        def floored(col):
            span = max(col) - min(col)
            span = span if span > floor else Fr(1)
            return [(x - min(col)) / span for x in col]
        span_a, span_b = max(col_a) - min(col_a), max(col_b) - min(col_b)
        self.assertTrue(span_a <= floor < span_b)                    # fires on A (a test row held the max), not on B
        self.assertNotEqual(floored(col_a), floored(col_b))

        def mixed(col):
            mn, mx = min(col), max(col)
            return [(x - mn) / (mx - mn) + Fr(1, 100) * x for x in col]
        self.assertNotEqual(mixed(col_a), mixed(col_b))

    def test_yz_epsilon_is_a_tiny_exact_non_invariance(self):
        ctx, _ = sc.sampling(len(self.R))
        ya = [r[-1] for r in self.AR_exact]
        yb = [r[-1] for r in self.B_exact]
        za, zb = yz_decimal(ya, ctx), yz_decimal(yb, ctx)
        diff = max(abs(a - b) for a, b in zip(za, zb))
        self.assertGreater(diff, decimal.Decimal("1e-40"))           # not invariant even exactly
        self.assertLess(diff, decimal.Decimal("1e-6"))               # but of the order of 1e-8 / std

    def test_float_pipeline_residual_is_rounding_only(self):
        """Exact values are equal, so every float difference between A and B on AI components is rounding: small,
        never structural; the raw tabpfn35 arrays are structural."""
        A64 = to_f64(self.A_exact)
        AR = A64[np.asarray(self.R)]
        B64 = sc.phi(AR, AR.min(0), AR.max(0))
        ia, ib = sc.codex_inputs(AR), sc.codex_inputs(B64)
        cmp = sc.compare_inputs(ia, ib, np.asarray(self.R), 0)
        self.assertEqual(cmp["binding"]["class"], "IDENTICAL")
        for comp in ("foundation", "target"):
            self.assertIn(cmp["components"][comp]["class"], ("IDENTICAL", "NUMERICAL"), comp)
        for arr in ("x_ctx", "x_qry"):
            rec = cmp["components"]["foundation"][arr]
            self.assertEqual(rec["exceed_2^-12"], 0)
            self.assertLessEqual(rec["max_abs_x2^24"], 16.0)
        self.assertEqual(cmp["components"]["tabpfn35"]["class"], "STRUCTURAL")
        # each variant's float v sits within rounding of the common exact value
        ea = exact_maps(self.AR_exact, self.kept)
        for name, inp in (("A", ia), ("B", ib)):
            ctx = inp["foundation"]["ctx"]
            exact = np.asarray([[float(ea["foundation_v"][j][i]) for j in range(len(self.kept))] for i in ctx])
            self.assertLess(float(np.abs(inp["foundation"]["x_ctx"].astype(np.float64) - exact).max()), 2.0 ** -20, name)

    def test_hyperion_flips_sit_at_exact_integer_boundaries(self):
        """Every non-floor u flip between A and B is rounding: the common exact t = 255*(x - q01)/(q99 - q01) lies within
        float32 rounding of an integer (where truncation is discontinuous), never far from one. Lattice columns (k/K
        with 255/K an integer, test-only maxima so A's scale is not the train range) put t exactly on integers, so
        flips do occur."""
        rng = np.random.default_rng(13)
        n, lattice = 400, (15, 17, 51, 85, 5, 3, 17, 15)
        M = np.zeros((n, len(lattice) + 1), dtype=np.int64)
        T = list(range(60))
        R = list(range(60, n))
        for j, K in enumerate(lattice):
            M[:, j] = rng.integers(0, K + 1, size=n)
            M[R[: 2 * (j + 4)], j] = 0                     # enough rows at both ends: q01 = 0 and q99 = K exactly
            M[R[2 * (j + 4): 4 * (j + 4)], j] = K
            M[T[j], j] = K + 2 + j                         # a test-only maximum
        M[:, -1] = rng.integers(0, 1000, size=n)
        A_exact = exact_scale(M)
        AR_exact = [A_exact[i] for i in R]
        kept = list(range(len(lattice)))
        A64 = to_f64(A_exact)
        AR = A64[np.asarray(R)]
        B64 = sc.phi(AR, AR.min(0), AR.max(0))
        ha, hb = sc.codex_inputs(AR)["hyperion"], sc.codex_inputs(B64)["hyperion"]
        ea = exact_maps(AR_exact, kept)["hyperion_t"]
        flips = np.argwhere(ha["u"] != hb["u"])
        self.assertGreater(len(flips), 0)
        for i, j in flips:
            self.assertFalse(bool(ha["floor"][j] or hb["floor"][j]))
            t = ea[j][i]
            self.assertIsNotNone(t)
            self.assertLess(abs(t - round(t)), Fr(1, 2 ** 12), (int(i), int(j), float(t)))
            self.assertLessEqual(abs(int(ha["u"][i, j]) - int(hb["u"][i, j])), 1)

    def test_module_flags_a_planted_non_invariant_op(self):
        """A foundation replica that skips the train-only re-normalization is classified STRUCTURAL."""
        A64 = to_f64(self.A_exact)
        AR = A64[np.asarray(self.R)]
        B64 = sc.phi(AR, AR.min(0), AR.max(0))
        real = sc.foundation_inputs

        def planted(X32, y32):
            out = real(X32, y32)
            out["x_ctx"], out["x_qry"] = X32[out["ctx"]].copy(), X32[out["qry"]].copy()
            return out
        honest = sc.compare_inputs(sc.codex_inputs(AR), sc.codex_inputs(B64), np.asarray(self.R), 0)
        with mock.patch.object(sc, "foundation_inputs", planted):
            bad = sc.compare_inputs(sc.codex_inputs(AR), sc.codex_inputs(B64), np.asarray(self.R), 0)
        self.assertNotEqual(honest["components"]["foundation"]["class"], "STRUCTURAL")
        self.assertEqual(bad["components"]["foundation"]["class"], "STRUCTURAL")
        self.assertGreater(bad["components"]["foundation"]["x_ctx"]["exceed_2^-12"], 0)
        self.assertEqual(bad["class"], "STRUCTURAL")

    def test_module_flags_a_planted_absolute_floor(self):
        """A hyperion replica whose span floor is absolute and large (planted: 0.5 instead of 1e-9) fires on A's narrow
        column but not on B's [0,1] column: its u flips by far more than 1, so hyperion is STRUCTURAL."""
        A64 = to_f64(self.A_exact)
        AR = A64[np.asarray(self.R)]
        AR[:, 1] = AR[:, 1] * 0.25                                  # a narrow upstream band for column 1
        B64 = sc.phi(AR, AR.min(0), AR.max(0))
        real = sc.hyperion_inputs

        def planted(X32):
            out = real(X32)
            floor = (out["hi"] - out["lo"]) < 0.5
            span = np.where(floor, np.float32(1), out["hi"] - out["lo"]).astype(np.float32)
            t = (X32 - out["lo"]) / span * 255
            out.update(span=span, floor=floor, t=t, u=np.clip(t, 0, 255).astype("uint8"))
            return out
        with mock.patch.object(sc, "hyperion_inputs", planted):
            bad = sc.compare_inputs(sc.codex_inputs(AR), sc.codex_inputs(B64), np.asarray(self.R), 0)
        u = bad["components"]["hyperion"]["u"]
        self.assertGreater(u["flips_ge2"], 0)
        self.assertGreater(u["floor_column_flips"], 0)
        self.assertEqual(bad["components"]["hyperion"]["floor_status_flips"], 1)
        self.assertEqual(bad["components"]["hyperion"]["class"], "STRUCTURAL")


# ------------------------------------------------------------------------------------------------- uint8 and ULP
class TestUint8Flips(unittest.TestCase):
    def test_flip_counts_and_positions(self):
        ua = np.asarray([[10, 20, 0], [30, 40, 255], [50, 60, 70]], dtype=np.uint8)
        uv = ua.copy()
        uv[0, 0] = 11      # +1
        uv[1, 1] = 39      # -1
        uv[2, 0] = 53      # +3 (>= 2)
        uv[0, 2] = 1       # +1 in a floor column
        uv[1, 2] = 254     # -1 at saturation (t_a > 255)
        ta = ua.astype(np.float32)
        tv = uv.astype(np.float32)
        ta[1, 2] = np.float32(300.0)
        rows = np.asarray([100, 101, 102])
        cols = [7, 8, 9]
        floor = np.asarray([False, False, True])
        out = sc.cmp_u8(ua, uv, ta, tv, rows, cols, floor, cap=3)
        self.assertEqual(out["flips"], 5)
        self.assertEqual(out["flips_abs1"], 4)
        self.assertEqual(out["flips_ge2"], 1)
        self.assertEqual(out["floor_column_flips"], 2)
        self.assertEqual(out["saturation_flips"], 1)
        self.assertEqual(out["nonfloor_abs1_flips"], 2)
        self.assertEqual(out["delta_hist"], {"-1": 2, "1": 2, "3": 1})
        self.assertEqual(out["max_abs_delta"], 3)
        self.assertEqual(out["bit_identical"], 4)
        self.assertEqual(out["differ_by_column"], {"7": 2, "8": 1, "9": 2})
        self.assertTrue(out["positions_truncated"])
        self.assertEqual([p[:5] for p in out["positions"]], [[100, 7, 10, 11, 1], [100, 9, 0, 1, 1], [101, 8, 40, 39, -1]])
        self.assertEqual(out["positions"][0][5:], [sc._bits(np.float32(10)), sc._bits(np.float32(11))])
        same = sc.cmp_u8(ua, ua.copy(), ta, ta, rows, cols, floor, cap=3)
        self.assertEqual((same["flips"], same["bit_identical"]), (0, 9))
        self.assertNotIn("positions", same)
        with self.assertRaises(sc.ScaleCheckError):
            sc.cmp_u8(ua.astype(np.int16), uv, ta, tv, rows, cols, floor, cap=0)

    def test_floor_column_flips_are_structural_end_to_end(self):
        """A ~99%-tied column makes q01 == q99, so span = 1 keeps variant units: B moves u by far more than 1."""
        n = 400
        rng = np.random.default_rng(5)
        A = rng.random((n, 3))
        T = list(range(0, 40))
        R = list(range(40, n))
        A[:, 1] = 0.5           # column 1: 359 of 360 train rows tie at 0.5, so q01 == q99 == 0.5 (span floor)
        A[R[0], 1] = 0.9        # the one out-of-band train row: u = trunc((0.9 - 0.5) * 255) = 102 under A
        A[0, 1] = 1.0           # the all-rows extremes sit in test rows
        A[1, 1] = 0.0
        AR = A[np.asarray(R)]
        B = sc.phi(AR, AR.min(0), AR.max(0))
        cmp = sc.compare_inputs(sc.codex_inputs(AR), sc.codex_inputs(B), np.asarray(R), 8)
        u = cmp["components"]["hyperion"]["u"]
        self.assertGreater(u["floor_column_flips"], 0)
        self.assertGreater(u["flips_ge2"], 0)
        self.assertEqual(cmp["components"]["hyperion"]["class"], "STRUCTURAL")
        self.assertTrue(all(p[0] in set(R) for p in u["positions"]))


class TestULP(unittest.TestCase):
    def test_ulp_distance(self):
        one = np.float32(1.0)
        a = np.asarray([one, 0.0, -0.0, np.float32(1e-45), one], dtype=np.float32)
        b = np.asarray([np.nextafter(one, np.float32(2)), -0.0, 0.0, -np.float32(1e-45), one], dtype=np.float32)
        self.assertEqual(sc.ulp_distance_f32(a, b).tolist(), [1, 0, 0, 2, 0])
        out = sc.cmp_f32(a, b, np.arange(5), None, cap=10)
        self.assertEqual(out["differ"], 4)
        self.assertEqual(out["signed_zero_flips"], 2)
        self.assertEqual(out["ulp_hist"]["0"], 3)
        self.assertEqual(out["ulp_hist"]["1"], 1)
        self.assertEqual(out["ulp_hist"]["2"], 1)
        self.assertEqual(out["positions"][0][:2], [0, "target"])
        self.assertEqual(out["differ_by_column"], {"target": 4})

    def test_ulp_distance_f64_and_side_scalars(self):
        self.assertEqual(sc.ulp_distance_f64(1.0, float(np.nextafter(1.0, 2.0))), 1)
        self.assertEqual(sc.ulp_distance_f64(0.0, -0.0), 0)
        self.assertEqual(sc.ulp_distance_f64(-5e-324, 5e-324), 2)
        side = sc._side_scalar(0.25, float(np.nextafter(0.25, 1.0)))
        self.assertEqual(side, {"identical": False, "ulp64": 1})       # values are never written
        self.assertEqual(sc._side_scalar(0.0, -0.0), {"identical": False, "ulp64": 0})
        self.assertEqual(sc._side_scalar(0.5, 0.5), {"identical": True, "ulp64": 0})

    def test_structural_threshold_scales_for_yz(self):
        a = np.asarray([100.0, 0.0], dtype=np.float32)
        v = np.asarray([100.0 + 0.0078125, 0.0005], dtype=np.float32)
        plain = sc.cmp_f32(a, v, np.arange(2), None, cap=0)
        scaled = sc.cmp_f32(a, v, np.arange(2), None, cap=0, scale=np.maximum(1.0, np.abs(a.astype(np.float64))))
        self.assertEqual(plain["exceed_2^-12"], 2)
        self.assertEqual(scaled["exceed_2^-12"], 1)

    def test_dtype_and_shape_guards_and_column_map(self):
        with self.assertRaises(sc.ScaleCheckError):
            sc.cmp_f32(np.zeros(3), np.zeros(3, np.float32), np.arange(3), None, cap=0)
        mism = sc.cmp_f32(np.zeros(3, np.float32), np.zeros(4, np.float32), np.arange(4), None, cap=0)
        self.assertTrue(mism["shape_mismatch"])
        a = np.zeros((3, 3), np.float32)
        v = a.copy()
        v[0, 2] = 1.0
        v[2, 2] = 1.0
        v[1, 0] = 2.0 ** -20
        out = sc.cmp_f32(a, v, np.asarray([4, 5, 6]), [10, 11, 13], cap=8)
        self.assertEqual(out["differ_by_column"], {"10": 1, "13": 2})
        self.assertEqual(out["exceed_2^-12"], 2)
        self.assertEqual(sorted((p[0], p[1]) for p in out["positions"]), [(4, 13), (5, 10), (6, 13)])


# ------------------------------------------------------------------------------------------------ variants
class TestVariants(unittest.TestCase):
    def test_c_single_row_columns_and_wrap(self):
        """A column constant on T has its test minimum and maximum in one row: it is counted, and the maximum moves to the
        next test row (wrapping); the train rows are never modified."""
        rng = np.random.default_rng(31)
        A = rng.random((60, 3))
        T = [3, 10, 20, 59]
        R = [i for i in range(60) if i not in set(T)]
        A[T, 1] = 0.5                           # column 1 is constant on T
        At, single = sc.c_perturbation(A, T)
        self.assertEqual(single, 1)
        lo, hi = A[:, 1].min(), A[:, 1].max()
        r = hi - lo
        self.assertEqual(At[3, 1], lo - 0.25 * r)        # t_lo = smallest test index at the test minimum
        self.assertEqual(At[10, 1], hi + 0.5 * r)        # t_hi moved to the next test row
        self.assertTrue(np.array_equal(At[R], A[R]))
        A[T, 1] = np.asarray([0.2, 0.9, 0.9, 0.2])       # ties: the smallest index wins for both extremes
        At, single = sc.c_perturbation(A, T)
        self.assertEqual(single, 0)
        self.assertEqual(At[3, 1], A[:, 1].min() - 0.25 * (A[:, 1].max() - A[:, 1].min()))
        self.assertEqual(At[10, 1], A[:, 1].max() + 0.5 * (A[:, 1].max() - A[:, 1].min()))
        one = sc.c_perturbation(A, [59])[1]                 # a single test row: every column is single-row
        self.assertEqual(one, 3)

    def test_c2_residual_and_b32(self):
        """C2 drops the test rows at a test-only extreme in one pass; a second test-only value below the train minimum
        remains as a residual. B32 rescales fl32(A[R]) and equals B when A[R] already holds float32 values."""
        A = np.zeros((10, 2))
        A[:, 0] = np.linspace(0.2, 0.8, 10)
        A[:, 1] = np.linspace(0.0, 1.0, 10)
        T = [0, 1]
        R = list(range(2, 10))
        A[0, 0], A[1, 0] = 0.0, 0.1                      # both below the train minimum; D = {0} only
        A[1, 1] = 0.5                                    # the target's second test row sits inside the train range
        lo_R, hi_R, lo_all, hi_all = sc.column_extremes(A, R)
        D, lo_S, _hi_S, residual = sc.c2_fit(A, R, T, [0, 1], lo_R, hi_R, lo_all, hi_all)
        self.assertEqual(D, [0])
        self.assertEqual(lo_S[0], 0.1)
        self.assertEqual(residual, 1)
        v, meta = sc.build_variants(A, R, T, [0, 1])
        self.assertEqual(meta["D_rows"], 1)
        self.assertEqual(meta["C2_residual_test_only_extreme_columns"], 1)
        AR = A[R]
        A32 = AR.astype(np.float32).astype(np.float64)
        self.assertTrue(np.array_equal(v["B32"], sc.phi(A32, A32.min(0), A32.max(0))))
        Af = A.astype(np.float32).astype(np.float64)
        vf, _ = sc.build_variants(Af, R, T, [0, 1])
        self.assertTrue(np.array_equal(vf["B32"].view(np.uint64), vf["B"].view(np.uint64)))

    def test_near_zero_fix_route_artifact_is_attributed(self):
        """Spec A P13 in miniature: train values near 1e-14 with a train range near 3.9e-4 and the all-row maximum in a
        test row. C's float64 offset (-0.25 r) quantizes those values, so F_C differs from B there; the difference is
        attributed to the near-zero column, not to the fix."""
        rng = np.random.default_rng(41)
        n = 400
        A = np.zeros((n, 3))
        A[:, 0] = rng.random(n)
        A[:, 2] = rng.random(n)
        T = list(range(0, 80))
        R = list(range(80, n))
        A[:, 1] = rng.random(n) * 3.9e-4
        A[R[:40], 1] = np.arange(1, 41) * 1e-14                # near-zero train values
        A[R[40], 1] = 0.0
        A[T[0], 1] = 1.0                                       # the all-row maximum sits in a test row
        A[T[1], 0], A[T[2], 0] = 0.0, 1.0
        A[T[3], 2], A[T[4], 2] = 0.0, 1.0
        res = sc.analyse(synthetic_ci(A, R, T), None, positions=4, want_a0=False, want_replication=False)
        self.assertEqual(res["near_zero_columns"], [1])
        fa = res["fix_attribution"]["fix_C_vs_B"]
        self.assertGreater(fa["elements_in_near_zero_columns"], 0)
        self.assertIn("1", fa["differing_columns"])
        self.assertEqual(fa["near_zero_columns"], ["1"])
        self.assertIn("1", fa["construction_risk_columns"])
        self.assertEqual(fa["elements_in_construction_risk_columns"], fa["elements_in_near_zero_columns"])
        self.assertTrue(fa["only_artifact"])
        self.assertEqual(res["prescan"]["near_zero_columns"], 1)

    def test_rounding_columns_are_classified(self):
        """B exactly at a float32 midpoint (1 + 2^-24 rounds to even, 1.0) and F_V one float64 ULP above it (rounds up)
        is a straddle; so is any gap below 1/8 float32 ULP; a gap of 1/4 float32 ULP is not. A -0.0 versus +0.0 cast
        (bitwise different, numerically equal) is a signed-zero flip. A column whose casts agree has no entry.
        fix_attribution then explains rounding-column elements."""
        b = 1.0 + 2.0 ** -24
        MB = np.asarray([[0.25, b, 0.5], [0.75, 0.5, 0.25], [0.5, b, 0.75]])
        MV = MB.copy()
        MV[0, 1] = float(np.nextafter(b, 2.0))                    # one float64 ULP: casts 1.0 versus 1 + 2^-23
        self.assertNotEqual(np.float32(MB[0, 1]), np.float32(MV[0, 1]))
        self.assertEqual(sc.rounding_columns(MB, MV, [0, 1, 2], 2), {1: "straddle"})
        MV[2, 1] = b + 2.0 ** -30                                   # still far below a float32 ULP: a straddle
        self.assertEqual(sc.rounding_columns(MB, MV, [0, 1, 2], 2), {1: "straddle"})
        MV[2, 1] = b + 2.0 ** -25                                   # 1/4 float32 ULP: not rounding-level any more
        self.assertEqual(sc.rounding_columns(MB, MV, [0, 1, 2], 2), {})
        MV = MB.copy()
        MV[1, 2] = float(np.nextafter(0.25, 1.0))                   # float64 differs but the float32 casts agree
        self.assertEqual(sc.rounding_columns(MB, MV, [0, 1, 2], 2), {})
        MB[1, 0], MV[1, 0] = -0.0, 0.0                              # IEEE negative zero against positive zero
        self.assertEqual(sc.rounding_columns(MB, MV, [0, 1, 2], 2), {0: "signed_zero"})
        MV[0, 1] = float(np.nextafter(b, 2.0))
        MB[1, 1], MV[1, 1] = -0.0, 0.0
        self.assertEqual(sc.rounding_columns(MB, MV, [0, 1, 2], 2), {0: "signed_zero", 1: "mixed"})
        cmp = {"class": "NUMERICAL", "binding": {"class": "IDENTICAL"},
               "components": {"foundation": {"x_ctx": {"kind": "f32", "differ": 3, "differ_by_column": {"1": 2, "4": 1}}}}}
        fa = sc.fix_attribution(cmp, [], ["4"], {1: "straddle"})
        self.assertEqual((fa["elements_in_rounding_columns"], fa["elements_in_construction_risk_columns"],
                          fa["elements_unexplained"]), (2, 1, 0))
        self.assertEqual(fa["rounding_kinds"], {"straddle": 1})
        self.assertTrue(fa["only_explained"])
        self.assertFalse(fa["only_artifact"])

    def test_severity_amplified_fix_route_artifact_is_attributed(self):
        """A column whose train range is 1e-9 of its all-row range (severity ~1e9, the S-FLOOR regime): C's float64
        offset keeps only ~2^-52 * 1e9 of resolution in train-normalized units, so F_C differs from B in bulk values.
        The near-zero rule misses it; the construction-risk rule explains every differing element."""
        rng = np.random.default_rng(43)
        n = 300
        A = rng.random((n, 3))
        T = list(range(60))
        R = list(range(60, n))
        A[np.asarray(R), 0] = 0.5 + rng.random(len(R)) * 1e-9
        A[T[0], 0], A[T[1], 0] = 0.0, 1.0
        res = sc.analyse(synthetic_ci(A, R, T), None, positions=4, want_a0=False, want_replication=False)
        self.assertGreater(res["prescan"]["severity"], 1e8)
        self.assertEqual(res["near_zero_columns"], [])
        self.assertIn(0, res["construction_risk_columns"]["C"])
        fa = res["fix_attribution"]["fix_C_vs_B"]
        self.assertGreater(fa["differing_elements"], 0)
        self.assertEqual(fa["elements_in_near_zero_columns"], 0)
        self.assertTrue(fa["only_artifact"])
        self.assertEqual(fa["elements_unexplained"], 0)
        self.assertEqual(res["comparisons"]["fix_A1_vs_B"]["class"], "IDENTICAL")   # A1's map is exact here


# ------------------------------------------------------------------------------------------- tabpfn35 (spec A P9)
class TestTabpfnRefined(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(51)
        n_ctx, n_qry = 150, 40
        Xc = np.zeros((n_ctx, 3), np.float32)
        Xq = np.zeros((n_qry, 3), np.float32)
        Xc[:, 0] = 0.3                                                  # constant on ctx
        Xq[:, 0] = rng.random(n_qry).astype(np.float32)
        Xc[:, 1] = rng.choice([0.1, 0.4, 0.7], n_ctx).astype(np.float32)   # 3 values: inferred categorical
        Xq[:, 1] = rng.choice([0.1, 0.4, 0.7, 0.9], n_qry).astype(np.float32)
        Xc[:, 2] = rng.random(n_ctx).astype(np.float32)                 # numerical
        Xq[:, 2] = rng.random(n_qry).astype(np.float32)
        y = rng.random(n_ctx).astype(np.float32)
        self.pa = {"X_ctx": Xc, "X_qry": Xq, "y_ctx": y}

    def variant(self, **changes):
        pv = {k: v.copy() for k, v in self.pa.items()}
        for key, fn in changes.items():
            fn(pv)
        return pv

    def test_classes(self):
        pa, kept = self.pa, [4, 5, 6]
        self.assertEqual(sc.tabpfn_refined(pa, self.variant(), kept, True)["class"], "IDENTICAL")

        def shift_const(pv):
            pv["X_ctx"][:, 0] = 0.0
            pv["X_qry"][:, 0] = pv["X_qry"][:, 0] * 2
        r = sc.tabpfn_refined(pa, self.variant(a=shift_const), kept, True)
        self.assertEqual((r["class"], r["column_kinds"]), ("INERT_ONLY", {"constant_inert": 1}))

        def monotone_cat(pv):
            pv["X_ctx"][:, 1] = pv["X_ctx"][:, 1] * np.float32(2) + np.float32(1)
            pv["X_qry"][:, 1] = pv["X_qry"][:, 1] * np.float32(2) + np.float32(1)
        r = sc.tabpfn_refined(pa, self.variant(a=shift_const, b=monotone_cat), kept, True)
        self.assertEqual(r["class"], "INERT_ONLY")
        self.assertEqual(r["column_kinds"], {"categorical_inert": 1, "constant_inert": 1})

        def reorder_cat(pv):
            pv["X_ctx"][:, 1] = np.float32(1) - pv["X_ctx"][:, 1]
            pv["X_qry"][:, 1] = np.float32(1) - pv["X_qry"][:, 1]
        r = sc.tabpfn_refined(pa, self.variant(a=reorder_cat), kept, True)
        self.assertEqual(r["class"], "STRUCTURAL")
        self.assertEqual(r["structural_columns"], [5])

        def y_only(pv):
            pv["y_ctx"] = (pv["y_ctx"] * np.float32(3)).astype(np.float32)
        self.assertEqual(sc.tabpfn_refined(pa, self.variant(a=y_only), kept, True)["class"], "NUMERICAL")

        def num_change(pv):
            pv["X_qry"][0, 2] = np.nextafter(pv["X_qry"][0, 2], np.float32(2))
        r = sc.tabpfn_refined(pa, self.variant(a=num_change), kept, True)
        self.assertEqual((r["class"], r["fingerprint_rows_changed"]), ("STRUCTURAL", 1))
        self.assertEqual(sc.tabpfn_refined(pa, self.variant(), kept, False)["class"], "STRUCTURAL")

    def test_fingerprint_rounding_and_small_context(self):
        pa = {k: v.copy() for k, v in self.pa.items()}
        pa["X_ctx"][:, 2] = np.float32(1e-14) * np.arange(1, 151, dtype=np.float32)
        pv = {k: v.copy() for k, v in pa.items()}
        pv["X_ctx"][5, 2] = np.nextafter(pv["X_ctx"][5, 2], np.float32(1))   # below np.around(., 12) resolution
        r = sc.tabpfn_refined(pa, pv, [0, 1, 2], True)
        self.assertEqual((r["class"], r["fingerprint_rows_changed"]), ("NUMERICAL", 0))
        small = {k: (v[:90] if k != "X_qry" else v) for k, v in self.pa.items()}   # n_ctx <= 100: no categorical inference
        self.assertTrue(np.array_equal(sc.tabpfn_modalities(small["X_ctx"]), [sc.MOD_CONSTANT, sc.MOD_NUMERICAL,
                                                                              sc.MOD_NUMERICAL]))
        self.assertTrue(np.array_equal(sc.tabpfn_modalities(self.pa["X_ctx"]), [sc.MOD_CONSTANT, sc.MOD_CATEGORICAL,
                                                                               sc.MOD_NUMERICAL]))


# ---------------------------------------------------------------------------- determinism and test-row isolation
class TestDeterminismAndIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        M, cls.R, cls.T = synth_integer_matrix(n=200, F=3, n_test=50, seed=21)
        cls.A = to_f64(exact_scale(M))

    def test_analyse_is_deterministic(self):
        docs = []
        for _ in range(2):
            res = sc.analyse(synthetic_ci(self.A, self.R, self.T), None, positions=16, want_a0=True, want_replication=False)
            docs.append(common.canonical(res))
        self.assertEqual(docs[0], docs[1])
        res = sc.analyse(synthetic_ci(self.A, self.R, self.T), None, positions=16, want_a0=True, want_replication=False)
        self.assertTrue(res["lineage"]["jtt1_rebuild_matches_input_ref"])
        self.assertTrue(res["lineage"]["codex_kept_equals_jtt1_kept"])
        self.assertIn(res["a0"]["status"], ("computed", "skipped", "refused"))
        for k in sc.GATE_KEYS + sc.FIX_GATE_KEYS + sc.DIAG_KEYS + ("A1_vs_A", "B_vs_A1", "C_vs_A1", "C2_vs_A1"):
            self.assertIn(k, res["comparisons"])

    def test_test_rows_never_reach_A_B_or_B32(self):
        """Poisoning every outer-test row (NaN) leaves the A, B and B32 matrices and inputs bit-identical; only A1, C and
        C2, the explicit perturbation variants, read test rows."""
        clean, _ = sc.build_variants(self.A, self.R, self.T, [0, 1, 2, 3])
        poisoned_A = self.A.copy()
        poisoned_A[np.asarray(self.T)] = np.nan
        with np.errstate(invalid="ignore"):
            dirty, _ = sc.build_variants(poisoned_A, self.R, self.T, [0, 1, 2, 3])
        for name in ("A", "B", "B32"):
            self.assertTrue(np.array_equal(clean[name].view(np.uint64), dirty[name].view(np.uint64)), name)
            ia, ib = sc.codex_inputs(clean[name]), sc.codex_inputs(dirty[name])
            cmp = sc.compare_inputs(ia, ib, np.asarray(self.R), 4)
            self.assertEqual(cmp["class"], "IDENTICAL", name)
            self.assertEqual(common.canonical(cmp), common.canonical(sc.compare_inputs(ia, ia, np.asarray(self.R), 4)))
        self.assertTrue(np.isnan(dirty["A1"]).all())

    def test_test_row_values_do_not_reach_A_side_outputs(self):
        """Replacing every outer-test row with other finite values (and so moving every all-row extreme) leaves the
        B_vs_A, B32_vs_A and B_vs_B32 documents byte-identical: no test value reaches them."""
        rng = np.random.default_rng(77)
        other = self.A.copy()
        other[np.asarray(self.T)] = rng.random((len(self.T), self.A.shape[1])) * 3 - 1
        r1 = sc.analyse(synthetic_ci(self.A, self.R, self.T), None, positions=64, want_a0=False, want_replication=False)
        r2 = sc.analyse(synthetic_ci(other, self.R, self.T), None, positions=64, want_a0=False, want_replication=False)
        for k in ("B_vs_A", "B32_vs_A", "B_vs_B32"):
            self.assertEqual(common.canonical(r1["comparisons"][k]), common.canonical(r2["comparisons"][k]), k)
        self.assertNotEqual(common.canonical(r1["comparisons"]["C_vs_A"]), common.canonical(r2["comparisons"]["C_vs_A"]))
        self.assertNotEqual(r1["prescan"], r2["prescan"])          # counts of test-only extremes do move

    def test_random_test_rows_only_move_perturbation_variants(self):
        rng = np.random.default_rng(99)
        other = self.A.copy()
        other[np.asarray(self.T)] = rng.random((len(self.T), self.A.shape[1])) * 3 - 1
        v1, _ = sc.build_variants(self.A, self.R, self.T, [0, 1, 2, 3])
        v2, _ = sc.build_variants(other, self.R, self.T, [0, 1, 2, 3])
        self.assertTrue(np.array_equal(v1["A"], v2["A"]))
        self.assertTrue(np.array_equal(v1["B"], v2["B"]))
        self.assertTrue(np.array_equal(v1["B32"], v2["B32"]))
        self.assertFalse(np.array_equal(v1["A1"], v2["A1"]))
        self.assertFalse(np.array_equal(v1["C"], v2["C"]))

    def test_positions_name_outer_train_rows_only(self):
        res = sc.analyse(synthetic_ci(self.A, self.R, self.T), None, positions=64, want_a0=False, want_replication=False)
        self.assertGreater(walk_positions(self, res["comparisons"], set(self.R)), 0)

    def test_prescan_emits_counts_only(self):
        res = sc.analyse(synthetic_ci(self.A, self.R, self.T), None, positions=0, want_a0=False, want_replication=False)
        test_values = {float(x) for x in self.A[np.asarray(self.T)].ravel()}
        for key, val in res["prescan"].items():
            if key == "severity":
                self.assertIsInstance(val, float)
                continue
            self.assertIsInstance(val, (bool, int, list, type(None)), key)
            if isinstance(val, list):
                self.assertTrue(all(isinstance(x, (int, str)) for x in val), key)
        self.assertNotIn(res["prescan"]["severity"], test_values)

    def test_variant_definitions(self):
        AR = self.A[np.asarray(self.R)]
        v, meta = sc.build_variants(self.A, self.R, self.T, [0, 1, 2, 3])
        self.assertTrue(np.array_equal(v["A"], AR))
        self.assertEqual(float(v["B"].min()), 0.0)
        self.assertEqual(float(v["B"].max()), 1.0)
        # exactly-[0,1] columns over all rows: A1 is bitwise A
        self.assertTrue(np.array_equal(v["A1"], AR))
        # C: every column's new extremes sit in test rows, so the train rows land strictly inside (0, 1)
        self.assertGreater(float(v["C"].min()), 0.0)
        self.assertLess(float(v["C"].max()), 1.0)
        self.assertGreater(meta["D_rows"], 0)
        self.assertEqual(meta["C_single_row_columns"], 0)

    def test_fix_route_comparisons(self):
        """F_V = phi_R(V[R]) against B: every column here is exactly [0,1] over all rows, so A1 == A and the train-only
        fix of A1 is B bit for bit; C's train rows go through two float64 roundings, so it may or may not cancel."""
        res = sc.analyse(synthetic_ci(self.A, self.R, self.T), None, positions=0, want_a0=False, want_replication=False)
        self.assertEqual(res["comparisons"]["fix_A1_vs_B"]["class"], "IDENTICAL")
        self.assertEqual(res["fix_attribution"]["fix_A1_vs_B"]["differing_elements"], 0)
        fix = sc.gate_verdict([res], sc.FIX_GATE_KEYS, sc.FIX_RULE)
        main = sc.gate_verdict([res])
        self.assertEqual(main["result"], "FAIL")                      # tabpfn35 sees raw values under B
        self.assertIn(fix["result"], ("PASS", "FAIL"))
        self.assertEqual(sum(fix["by_variant"]["fix_A1_vs_B"].values()), 1)
        for k in sc.FIX_GATE_KEYS:                                    # never more than rounding on the AI components
            for comp in ("foundation", "target"):
                self.assertNotEqual(res["comparisons"][k]["components"][comp]["class"], "STRUCTURAL", (k, comp))

    def test_c2_equals_a1_without_test_only_extremes(self):
        A = self.A.copy()
        AR = A[np.asarray(self.R)]
        A[np.asarray(self.T)] = AR[: len(self.T)]            # test rows copy train rows: no test-only extreme
        v, meta = sc.build_variants(A, self.R, self.T, [0, 1, 2, 3])
        self.assertEqual(meta["D_rows"], 0)
        self.assertEqual(meta["C2_residual_test_only_extreme_columns"], 0)
        self.assertTrue(np.array_equal(v["C2"], v["A1"]))


# ------------------------------------------------------------------------------------------------------- prescan
class TestPrescan(unittest.TestCase):
    def test_flags_strict_identity_and_floors(self):
        """Column 0 holds a test-only max; column 1 is constant 0.5 everywhere (B maps it to 0, so the strict B-identity
        fails while the loose one holds); column 2 is exactly [0,1] with both extremes in train rows; column 3 is a
        hyperion floor column (q01 == q99) that B remaps; the target has a test-only min."""
        rng = np.random.default_rng(61)
        n = 300
        A = np.zeros((n, 5))
        T = list(range(0, 60))
        R = list(range(60, n))
        Rn, Tn = np.asarray(R), np.asarray(T)
        A[Rn, 0] = rng.permutation(np.linspace(0.0, 0.5, len(R)))   # train covers [0, 0.5]; shuffled (no affine copy)
        A[Tn, 0] = 0.25
        A[T[0], 0] = 1.0                                             # the test-only maximum
        A[:, 1] = 0.5
        A[:, 2] = rng.random(n)
        A[R[0], 2], A[R[1], 2] = 0.0, 1.0
        A[T[5], 2], A[T[6], 2] = 0.25, 0.75
        A[:, 3] = 0.3
        A[R[2], 3] = 0.9
        A[Rn, 4] = np.linspace(0.25, 0.75, len(R))                   # train covers [0.25, 0.75]
        A[Tn, 4] = 0.5
        A[T[1], 4] = 0.0                                             # the test-only minimum of the target
        ci = synthetic_ci(A, R, T)
        ref = sc.codex_inputs(A[np.asarray(R)])
        p = sc.prescan(ci, ref)
        self.assertTrue(p["flagged"] and p["target_flagged"])
        self.assertEqual((p["features_flagged"], p["flagged_columns"]), (1, 2))
        self.assertEqual(p["flagged_cols"], [0, "target"])
        self.assertEqual((p["test_only_min"], p["test_only_max"], p["test_only_both"]), (1, 1, 0))
        self.assertTrue(p["b_identity_all"] is False and p["b_identity_strict"] is False)
        self.assertEqual(p["D_rows"], 2)
        self.assertEqual(p["c2_residual_columns"], 0)
        self.assertEqual(p["floor_columns"], 2)                  # the constant column 1 and the tied column 3
        self.assertEqual(p["floor_varying_columns"], 1)
        self.assertEqual(p["floor_remapped_by_B"], 1)
        self.assertFalse(p["exact01_all_columns"])
        self.assertFalse(p["out01"])
        self.assertEqual(p["k"], 4)
        sev = max((1.0 - A[:, 0].min()) / (A[R, 0].max() - A[R, 0].min()) - 1,
                  (A[:, 4].max() - 0.0) / (A[R, 4].max() - A[R, 4].min()) - 1)
        self.assertAlmostEqual(p["severity"], sev, places=12)
        # loose versus strict B-identity on an unflagged context with a constant non-zero column
        B = A.copy()
        B[:, 0] = np.linspace(0, 1, n)[::-1]
        B[R[0], 0], B[R[1], 0] = 0.0, 1.0
        B[:, 4] = np.linspace(0, 1, n)
        B[R[3], 4], B[R[4], 4] = 0.0, 1.0
        B[:, 3] = np.linspace(0, 1, n)
        B[R[5], 3], B[R[6], 3] = 0.0, 1.0
        q = sc.prescan(synthetic_ci(B, R, T), sc.codex_inputs(B[np.asarray(R)]))
        self.assertFalse(q["flagged"])
        self.assertTrue(q["b_identity_all"])
        self.assertFalse(q["b_identity_strict"])                  # column 1 (0.5) maps to 0 under B
        B[:, 1] = 0.0
        q = sc.prescan(synthetic_ci(B, R, T), sc.codex_inputs(B[np.asarray(R)]))
        self.assertTrue(q["b_identity_all"] and q["b_identity_strict"])
        self.assertTrue(q["exact01_all_columns"])

    def test_summary_and_crosscheck(self):
        recs = [{"prescan": {"flagged": True, "target_flagged": True, "features_flagged": 0, "considered_columns": 3,
                             "flagged_columns": 1, "test_only_min": 1, "test_only_max": 0, "test_only_both": 0,
                             "constant_on_R_columns": 0, "b_nonidentity_columns": 2, "b_identity_all": False,
                             "b_identity_strict": False, "exact01_all_columns": False, "exact01_considered": False,
                             "out01": True, "subsampled": True, "floor_columns": 1, "floor_varying_columns": 1,
                             "floor_remapped_by_B": 1, "floor_flagged": 0, "D_rows": 3, "c2_residual_columns": 0,
                             "foundation_floor_risk": 0, "target_floor_risk": 0, "san": False, "k": 2,
                             "severity": 0.5, "near_zero_columns": 0},
                 "local_complete": True, "cohort": "frozen", "fold": 0}]
        s = sc.prescan_summary(recs)
        self.assertEqual((s["flagged_target_only"], s["floor_remapped_contexts"], s["D_rows_max"]), (1, 1, 3))
        self.assertEqual(s["severity"]["count"], 1)
        self.assertEqual(s["flagged_columns_per_context"], {"1": 1})
        self.assertEqual(sc.strata_summary(recs)["frozen f0"], [1, 1, 1, 1, 0])
        cc = sc.crosscheck({"a": 1, "b": {"x": 2}}, {"a": 1, "b": {"x": 3}})
        self.assertFalse(cc["all_match"])
        self.assertEqual(list(cc["mismatches"]), ["b"])
        self.assertTrue(sc._sig_match(0.00178, "0.0018"))
        self.assertFalse(sc._sig_match(0.00174, "0.0018"))
        self.assertTrue(sc._sig_match(1.7204e13, "1.72e13"))
        self.assertTrue(sc._sig_match(1.0, "1.0"))
        self.assertTrue(sc._sig_match(16.12, "16.1"))
        self.assertFalse(sc._sig_match(16.16, "16.1"))
        self.assertFalse(sc._sig_match(None, "1.0"))


# ------------------------------------------------------------------------------------------------ sample design
def fake_records():
    recs = []
    for d in range(30):
        cohort = "frozen" if d % 2 == 0 else "incoming"
        for fold in range(5):
            sha = common.sha256_bytes(f"{d}/{fold}".encode())
            flagged = (d + fold) % 3 != 0
            recs.append({"context_sha256": sha, "dataset_hash": f"ds{d:02d}", "cohort": cohort, "fold": fold,
                         "rank": sc.rank_of(sha), "view_id": f"ds{d:02d}_f{fold}",
                         "prescan": {"flagged": flagged, "severity": ((d * 7 + fold) % 11) / 10 if flagged else None,
                                     "floor_remapped_by_B": int(d % 5 == 0), "b_identity_all": d % 4 == 1,
                                     "b_identity_strict": d % 4 == 1 and d % 3 != 0,
                                     "target_flagged": flagged and d % 6 == 1, "features_flagged": 0 if d % 6 == 1 else int(flagged),
                                     "san": d == 13, "out01": d == 17, "subsampled": d % 3 == 0,
                                     "b_nonidentity_columns": int(d % 4 != 1), "n_train": 300 + d}})
    return recs


class TestSampleDesign(unittest.TestCase):
    def test_deterministic_and_constrained(self):
        U = fake_records()
        anchor = dict(U[0], context_sha256="a" * 64, dataset_hash="dsAnchor", rank=sc.rank_of("a" * 64),
                      prescan=dict(U[0]["prescan"], n_train=1600))
        a = sc.select_sample(U, 48, anchor)
        b = sc.select_sample(list(reversed(U)), 48, anchor)
        self.assertEqual([(c, r["context_sha256"]) for c, r in a], [(c, r["context_sha256"]) for c, r in b])
        self.assertEqual(len(a), 48)
        self.assertEqual(a[0][0], "ANCHOR")
        per_ds = collections.Counter(r["dataset_hash"] for _c, r in a)
        self.assertLessEqual(max(per_ds.values()), 2)
        strata = collections.Counter((r["cohort"], r["fold"], r["dataset_hash"]) for _c, r in a)
        self.assertEqual(max(strata.values()), 1)
        self.assertEqual(len({r["context_sha256"] for _c, r in a}), 48)
        cats = [c for c, _ in a]
        for cohort, fold in sc.STRATA:      # every stratum gets its F-severe pick (each has flagged contexts)
            self.assertTrue(any(c == "F-severe" and r["cohort"] == cohort and r["fold"] == fold for c, r in a))
        for c, r in a:
            if c == "U-control":
                self.assertTrue(r["prescan"]["b_identity_strict"] and not r["prescan"]["flagged"])
        nmin = min(r["prescan"]["n_train"] for r in U)
        self.assertTrue(any(r["prescan"]["n_train"] == nmin for _c, r in a))
        self.assertTrue(any(r["prescan"]["san"] for _c, r in a))
        self.assertTrue(any(r["prescan"]["out01"] for _c, r in a))
        self.assertEqual(cats[-1], "R-any")
        self.assertEqual([c for c, _r in sc.select_sample(U, 3, anchor)], cats[:3])   # a smaller N is a prefix

    def test_strict_and_loose_u_control_can_differ(self):
        U = fake_records()
        strict = sc.select_sample(U, 48, None, u_control="b_identity_strict")
        loose = sc.select_sample(U, 48, None, u_control="b_identity_all")
        s_u = [r["context_sha256"] for c, r in strict if c == "U-control"]
        l_u = [r["context_sha256"] for c, r in loose if c == "U-control"]
        self.assertNotEqual(s_u, l_u)
        self.assertTrue(all(r["prescan"]["b_identity_all"] for c, r in loose if c == "U-control"))

    def test_rank_domain(self):
        self.assertEqual(sc.rank_of("x"), common.sha256_bytes(b"jope.dl-scale-leak-sample/v1|x"))

    def test_spec_b_pins_are_consistent(self):
        self.assertEqual(len(sc.SPEC_B_SAMPLE_48), 48)
        self.assertEqual(len(set(sc.SPEC_B_SAMPLE_48)), 48)
        self.assertEqual(sc.SPEC_B_SAMPLE_48[0], sc.ANCHOR_CONTEXT)
        self.assertEqual(common.sha256_bytes(common.canonical(list(sc.SPEC_B_SAMPLE_48))), sc.SPEC_B_SAMPLE_SHA256)
        self.assertEqual(sum(v[1] for v in sc.SPEC_B_STRATA_UNSEALED.values()), sc.SPEC_B_PRESCAN["unsealed"]["contexts"])
        self.assertEqual(sum(v[0] for v in sc.SPEC_B_STRATA_UNSEALED.values()), sc.SPEC_B_PRESCAN["unsealed"]["flagged"])
        self.assertEqual(sum(sc.SPEC_B_PRESCAN["all_included"]["flagged_columns_per_context"].values()), 4873)
        self.assertEqual(sum(sc.SPEC_B_PRESCAN["unsealed"]["flagged_columns_per_context"].values()), 4687)


# ------------------------------------------------------------------------------------------------- CLI guards
class TestCliGuards(unittest.TestCase):
    def test_out_and_receipt_paths_are_confined(self):
        with self.assertRaises(sc.ScaleCheckError):
            sc._check_out_dir(Path("/home/ubuntu/scratch/jopedime-dl-training-v1/tmp/not-scale/"))
        with self.assertRaises(sc.ScaleCheckError):
            sc._check_out_dir(Path("/home/ubuntu/scratch/jopedime-dl-training-v1/encodings/scale-check-x"))
        with self.assertRaises(sc.ScaleCheckError):
            sc._check_receipt_path(sc.FINDINGS_DIR / "UPSTREAM-SCALE-01.json")
        with self.assertRaises(sc.ScaleCheckError):
            sc._check_receipt_path(sc.RUN / "receipts" / "P2" / "UPSTREAM-SCALE-01.check-20261005T120000Z.json")
        ok = sc._check_receipt_path(sc.FINDINGS_DIR / "UPSTREAM-SCALE-01.check-20991231T235959Z.json")
        self.assertEqual(ok.name, "UPSTREAM-SCALE-01.check-20991231T235959Z.json")

    def test_no_torch_or_boto3(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("boto3", sys.modules)
        self.assertNotIn("dltrain.s3io", sys.modules)


if __name__ == "__main__":
    unittest.main()
