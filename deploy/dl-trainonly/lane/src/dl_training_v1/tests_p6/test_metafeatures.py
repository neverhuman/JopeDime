"""P6 spec tests T01-T12 (+ the T39 hygiene scan) for the A0 meta-features, dltrain.metafeatures (spec name a0.py).

Synthetic data only: JTT1 bytes are built here in the producer layout (b"JTT1" + <IIIB header, u8 inner labels,
float64 rows with the outer-train rows first and the target last). Every test runs under the P3 write guard.
"""
import ast
import hashlib
import inspect
import json
import math
import os
import random
import struct
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

from dltrain import jtt1, labels
from dltrain import metafeatures as mf
from tests_p3 import fixtures_p3 as fx

PINNED_FEATURE_LIST_SHA256 = "a569865050f6e2a162ac00ee838d58f7e0c8fbf7504fa6802975307013e433c6"
SPEC_FEATURES = (
    "s_log_n_train", "s_log_p", "s_p_over_n", "s_inv_sqrt_n",
    "m_x_nonfinite_cell_frac", "m_x_invalid_col_frac", "m_x_const_col_frac", "m_x_binary_col_frac",
    "m_x_lowcard_col_frac", "m_x_mode_share_mean", "m_x_mode_share_max", "m_x_dominant_col_frac",
    "m_x_at_min_frac_mean", "m_x_dup_row_frac", "m_xy_dup_conflict_frac",
    "y_robust_over_std", "y_mean_med_rz", "y_skew_slog", "y_kurt_ex_slog", "y_rq_p000", "y_rq_p001", "y_rq_p005",
    "y_rq_p010", "y_rq_p025", "y_rq_p075", "y_rq_p090", "y_rq_p095", "y_rq_p099", "y_rq_p100", "y_log1p_range_rz",
    "y_tail3_frac", "y_tail5_frac", "y_unique_frac", "y_mode_share", "y_at_min_frac", "y_at_max_frac",
    "y_bimodality", "y_logshift_skew_slog", "y_lag1_autocorr", "y_iqr_over_range",
    "x_active_col_frac", "x_abs_skew_q50", "x_abs_skew_q90", "x_kurt_ex_q50", "x_kurt_ex_q90", "x_uniq_frac_q10",
    "x_uniq_frac_q50", "x_uniq_frac_q90", "x_out5_frac_mean", "x_out5_frac_max", "x_log1p_range_rz_q50",
    "x_log1p_range_rz_q90", "x_iqr_over_range_q10", "x_iqr_over_range_q50",
    "r_abs_max", "r_abs_q90", "r_abs_q50", "r_abs_mean", "r_top3_abs_mean", "r_sq_sum_log1p", "r_gt01_frac",
    "r_gt03_frac", "r_pos_frac", "rho_abs_max", "rho_abs_q90", "rho_abs_q50", "rho_abs_mean", "rho_minus_r_max",
    "rho_minus_r_mean", "eta2_max", "eta2_q50", "eta2_minus_r2_max", "eta2_minus_r2_mean", "r_best_uniq_frac",
    "c_abs_corr_q50", "c_abs_corr_q90", "c_abs_corr_max", "c_abs_corr_mean", "c_gt09_pair_frac", "c_twin_col_frac",
    "c_pr_frac", "c_top_eig_share",
    "lin_top8_r2_in", "lin_top8_r2_cv", "knn1_top8_r2_cv", "lin_resid_absfit_corr", "lin_resid_skew_slog",
    "lin_resid_kurt_ex_slog",
)
L_GROUP = set(range(82, 88))
THRESHOLD_FEATURES = {30, 31, 48, 49, 60, 61, 78, 79}
F = mf.INDEX
SRC = Path(mf.__file__)
FORBIDDEN_MODULES = {"jope", "gp_worker", "local_gateway", "reuse_cache", "intake_registry"}
FORBIDDEN_STRING = "jopedime-regression-research-2k-" + "tabpfn35-unseen-20261001"
LANE = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------ fixtures
def make_raw(x_train, y_train, inner, x_test, y_test, folds=2) -> bytes:
    """JTT1 bytes in the producer layout (jtt1.py docstring)."""
    x_train, x_test = np.asarray(x_train, dtype=np.float64), np.asarray(x_test, dtype=np.float64)
    n_train, p = x_train.shape
    n_test = x_test.shape[0]
    m = np.empty((n_train + n_test, p + 1), dtype="<f8")
    m[:n_train, :p] = x_train
    m[:n_train, p] = y_train
    m[n_train:, :p] = x_test
    m[n_train:, p] = y_test
    head = b"JTT1" + struct.pack("<IIIB", n_train, n_test, p, folds)
    return head + np.asarray(inner, dtype=np.uint8).tobytes() + m.tobytes()


def continuous(rng, n, p):
    """Continuous columns of mixed shapes and a target that depends on the first columns."""
    gens = (lambda: rng.standard_normal(n), lambda: rng.lognormal(0.0, 0.7, n), lambda: rng.uniform(-2, 3, n),
            lambda: rng.standard_t(3, n), lambda: rng.exponential(1.0, n),
            lambda: np.where(rng.random(n) < 0.3, rng.normal(3, 0.5, n), rng.normal(-1, 1, n)))
    x = np.stack([gens[j % len(gens)]() for j in range(p)], axis=1)
    y = 1.5 * x[:, 0] - 0.7 * x[:, min(1, p - 1)] + 0.3 * np.abs(x[:, min(2, p - 1)]) + rng.standard_normal(n)
    g = (rng.permutation(n) % 2).astype(np.uint8)
    return x, y, g


def split_raw(rng, n_train, n_test, p):
    x, y, _ = continuous(rng, n_train + n_test, p)
    g = (rng.permutation(n_train) % 2).astype(np.uint8)
    return make_raw(x[:n_train], y[:n_train], g, x[n_train:], y[n_train:])


def vec32(raw):
    return np.frombuffer(mf.from_raw(raw)[0], dtype="<f4")


def manual_avg_rank(a):
    order = sorted(range(len(a)), key=lambda i: (a[i], i))
    ranks = [0.0] * len(a)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and a[order[j + 1]] == a[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return np.array(ranks)


def slog(v):
    return math.copysign(math.log1p(abs(v)), v)


class A0Case(fx.P3Case):
    def assert_close_features(self, got, want, *, rtol, atol, skip=(), what=""):
        bad = []
        for i in range(mf.N_FEATURES):
            if i in skip:
                continue
            if not abs(got[i] - want[i]) <= atol + rtol * abs(want[i]):
                bad.append((i, mf.FEATURES[i], float(want[i]), float(got[i])))
        self.assertEqual(bad, [], f"{what}: features outside rtol={rtol} atol={atol}")


# ------------------------------------------------------------------------------------------------ T01
class T01FeatureList(A0Case):
    def test_names_order_count(self):
        self.assertEqual(mf.FEATURES, SPEC_FEATURES)
        self.assertEqual(len(mf.FEATURES), 88)
        self.assertEqual(len(set(mf.FEATURES)), 88)
        self.assertTrue(64 <= len(mf.FEATURES) <= 128)
        self.assertEqual(mf.VERSION, "jope.dl-a0-meta/v1")
        self.assertEqual(mf.VECTOR_BYTES, 352)

    def test_feature_list_sha256_is_pinned_and_recomputed(self):
        doc = {"schema": "jope.dl-a0-feature-list/v1", "version": "jope.dl-a0-meta/v1", "dtype": "<f4",
               "features": list(SPEC_FEATURES)}
        independent = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                                allow_nan=False).encode("utf-8")).hexdigest()
        self.assertEqual(independent, PINNED_FEATURE_LIST_SHA256)
        self.assertEqual(mf.FEATURE_LIST_SHA256, PINNED_FEATURE_LIST_SHA256)

    def test_changing_the_list_or_the_version_alone_breaks_the_pin(self):
        def digest(version, features):
            return hashlib.sha256(json.dumps({"schema": "jope.dl-a0-feature-list/v1", "version": version,
                                              "dtype": "<f4", "features": list(features)}, sort_keys=True,
                                             separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        swapped = list(SPEC_FEATURES)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        renamed = list(SPEC_FEATURES)
        renamed[87] = "lin_resid_kurt_slog"
        for features in (swapped, renamed, SPEC_FEATURES[:-1]):
            self.assertNotEqual(digest(mf.VERSION, features), PINNED_FEATURE_LIST_SHA256)
        self.assertNotEqual(digest("jope.dl-a0-meta/v2", SPEC_FEATURES), PINNED_FEATURE_LIST_SHA256)
        self.assertEqual(digest(mf.VERSION, SPEC_FEATURES), PINNED_FEATURE_LIST_SHA256)


# ------------------------------------------------------------------------------------------------ T02
class T02Output(A0Case):
    def test_from_raw_bytes_and_meta(self):
        rng = np.random.default_rng(2)
        for n_train, p in ((8, 1), (129, 7), (400, 20)):
            raw = split_raw(rng, n_train, 31, p)
            b1, meta = mf.from_raw(raw)
            b2, meta2 = mf.from_raw(raw)
            self.assertEqual(len(b1), 352)
            self.assertEqual(b1, b2)
            self.assertEqual(meta, meta2)
            v = np.frombuffer(b1, dtype="<f4")
            self.assertTrue(np.all(np.isfinite(v)))
            self.assertFalse(np.any((v == 0) & np.signbit(v)), "a -0.0 survived")
            self.assertTrue(np.all(np.abs(v) <= 1e4))
            self.assertEqual(meta, {"n_train": n_train, "p": p, "train_row_sha256": jtt1.train_rows_sha256(raw)})

    def test_to_f32le_maps_negative_zero_and_refuses_nonfinite(self):
        v = np.zeros(88)
        v[3], v[50] = -0.0, -1.5
        out = np.frombuffer(mf.to_f32le(v), dtype="<f4")
        self.assertFalse(np.signbit(out[3]))
        self.assertEqual(out[50], np.float32(-1.5))
        v[7] = np.nan
        with self.assertRaises(mf.A0Error):
            mf.to_f32le(v)
        with self.assertRaises(mf.A0Error):
            mf.to_f32le(np.zeros(87))


# ------------------------------------------------------------------------------------------------ T03
class T03TestBytesCounterfactual(A0Case):
    PATTERNS = (0x7FF8000000000000, 0x7FF0000000000001, 0xFFF8000000000000, 0x7FF0000000000000,
                0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 0xFFF0000000000001)

    def _variants(self, tail: bytes, rnd: random.Random):
        k = len(tail) // 8
        xor = bytes(b ^ 0xFF for b in tail)
        finite = []
        while len(finite) < k:
            (val,) = struct.unpack("<d", rnd.getrandbits(64).to_bytes(8, "little"))
            if math.isfinite(val):
                finite.append(val)
        rand = struct.pack(f"<{k}d", *finite)
        nan = b"".join(struct.pack("<Q", self.PATTERNS[i % len(self.PATTERNS)]) for i in range(k))
        return {"xor_ff": xor, "random_finite_f64": rand, "nan_inf_patterns": nan}

    def test_bytes_after_train_stop_never_change_a0(self):
        rng = np.random.default_rng(3)
        rnd = random.Random(20261005)
        runs = 0
        for p in (1, 2, 7, 64):
            for n_train in (8, 129, 1000):
                raw = split_raw(rng, n_train, 23, p)
                stop = jtt1.train_range(jtt1.parse_header(raw))[1]
                base, meta = mf.from_raw(raw)
                for name, tail in self._variants(raw[stop:], rnd).items():
                    mutated = raw[:stop] + tail
                    self.assertEqual(len(mutated), len(raw))
                    self.assertNotEqual(mutated, raw)
                    got, meta2 = mf.from_raw(mutated)
                    self.assertEqual(got, base, f"p={p} n_train={n_train} variant={name}")
                    self.assertEqual(jtt1.train_rows_sha256(mutated), meta["train_row_sha256"])
                    self.assertEqual(meta2, meta)
                    runs += 1
        self.assertEqual(runs, 36)


# ------------------------------------------------------------------------------------------------ T04
class T04Controls(A0Case):
    def test_train_changes_move_the_vector_and_labels_touch_only_L(self):
        rng = np.random.default_rng(4)
        x, y, g = continuous(rng, 300, 5)
        x[:, 4] = (x[:, 4] > 1.0).astype(float)
        base = mf.features(x, y, g)
        y2 = y.copy()
        y2[17] += 0.5 * y.std()
        self.assertFalse(np.array_equal(mf.features(x, y2, g), base))
        x2 = x.copy()
        x2[23, 2] += 0.25
        self.assertFalse(np.array_equal(mf.features(x2, y, g), base))
        x3 = x.copy()
        x3[40, 4] = 1.0 - x3[40, 4]
        self.assertFalse(np.array_equal(mf.features(x3, y, g), base))
        for row in (0, 5, 299):
            g2 = g.copy()
            g2[row] = 1 - g2[row]
            flipped = mf.features(x, y, g2)
            outside = [i for i in range(88) if i not in L_GROUP and flipped[i] != base[i]]
            self.assertEqual(outside, [])


# ------------------------------------------------------------------------------------------------ T05
class T05Isolation(A0Case):
    def setUp(self):
        super().setUp()
        self.tree = ast.parse(SRC.read_text(), filename=str(SRC))

    def test_no_reference_to_test_rows_header_or_s3(self):
        names = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.update(node.name.split("."))
                if node.asname:
                    names.add(node.asname)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
        self.assertEqual(names & {"n_test", "parse_header", "train_range", "s3io", "boto3", "botocore"}, set())

    def test_raw_bytes_are_never_sliced_and_reach_only_two_jtt1_functions(self):
        parents = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                self.assertNotIn(node.value.id, {"raw", "data"}, "raw bytes are sliced")
        jtt1_attrs = {n.attr for n in ast.walk(self.tree) if isinstance(n, ast.Attribute)
                      and isinstance(n.value, ast.Name) and n.value.id == "jtt1"}
        self.assertEqual(jtt1_attrs, {"train_view", "train_rows_sha256"})
        fn = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "from_raw")
        uses = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "raw" and isinstance(n.ctx, ast.Load)]
        self.assertEqual(len(uses), 2)
        for use in uses:
            call = parents[use]
            self.assertIsInstance(call, ast.Call)
            self.assertIn(use, call.args)
            self.assertIsInstance(call.func, ast.Attribute)
            self.assertEqual((call.func.value.id, call.func.attr in {"train_view", "train_rows_sha256"}), ("jtt1", True))

    def test_signature_and_imports(self):
        self.assertEqual(list(inspect.signature(mf.features).parameters), ["X", "y", "inner_labels"])
        third_party, package = set(), set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                third_party |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "dltrain":
                    package |= {a.name for a in node.names}
                else:
                    third_party.add((node.module or "").split(".")[0])
        self.assertEqual(package, {"common", "jtt1", "labels"})
        self.assertEqual(third_party - {"ast", "functools", "math", "pathlib"}, {"numpy"})


class T39HygieneMetafeatures(A0Case):
    def test_forbidden_imports_top_level_s3_and_string(self):
        text = SRC.read_text()
        tree = ast.parse(text)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported |= {f"{node.module}.{a.name}" for a in node.names}
        self.assertEqual({m.split(".")[0] for m in imported} & FORBIDDEN_MODULES, set())
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                top |= {node.module or ""} | {f"{node.module}.{a.name}" for a in node.names}
        self.assertEqual({t for t in top if t.split(".")[0] in {"boto3", "botocore"} or t == "dltrain.s3io"}, set())
        self.assertNotIn(FORBIDDEN_STRING, text)

    def test_source_tree_sha256_convention(self):
        files = mf.source_tree_files()
        paths = [p for p, _ in files]
        self.assertEqual(paths, sorted(paths))
        self.assertTrue({"dltrain/__init__.py", "dltrain/metafeatures.py", "dltrain/common.py", "dltrain/jtt1.py",
                         "dltrain/labels.py"} <= set(paths))
        for path, digest in files:
            self.assertEqual(hashlib.sha256((LANE / path).read_bytes()).hexdigest(), digest)
        expect = hashlib.sha256("".join(f"{p}\0{d}\n" for p, d in files).encode()).hexdigest()
        self.assertEqual(mf.source_tree_sha256(), expect)


# ------------------------------------------------------------------------------------------------ T06
class T06AffineInvariance(A0Case):
    @staticmethod
    def affine(rng, a):
        """A positive affine map: scale 10**U(-6, 6), shift up to +-1e6 column spreads. (An absolute shift of 1e6 at
        scale 1e-6 would round the data to ~4 significant digits, which no implementation could survive.)"""
        scale = 10.0 ** rng.uniform(-6, 6)
        spread = float(np.max(a) - np.min(a)) or 1.0
        return a * scale + scale * spread * rng.uniform(-1e6, 1e6)

    def check(self, x, y, g, seed, skip=()):
        base = mf.features(x, y, g)
        rng = np.random.default_rng(seed)
        worst = np.zeros(88)
        for _ in range(3):
            x2 = np.stack([self.affine(rng, x[:, j]) for j in range(x.shape[1])], axis=1)
            y2 = self.affine(rng, y)
            got = mf.features(x2, y2, g)
            worst = np.maximum(worst, np.abs(got - base))
            self.assert_close_features(got, base, rtol=1e-6, atol=1e-6, skip=skip, what="affine map")
        return worst

    def test_continuous_data(self):
        rng = np.random.default_rng(6)
        x, y, g = continuous(rng, 1000, 6)
        worst = self.check(x, y, g, 61)
        self.assertLess(float(np.max(worst)), 1e-6, {mf.FEATURES[i]: float(worst[i]) for i in np.argsort(-worst)[:5]})

    def test_discrete_data_except_threshold_features(self):
        rng = np.random.default_rng(7)
        n = 1000
        x = np.stack([rng.integers(0, 3, n), rng.integers(0, 2, n), rng.integers(0, 5, n)], axis=1).astype(float)
        y = x[:, 0] + 2.0 * x[:, 1] - x[:, 2] + rng.integers(0, 3, n)
        g = (rng.permutation(n) % 2).astype(np.uint8)
        self.check(x, y, g, 71, skip=THRESHOLD_FEATURES)

    def test_knn_distance_ties_on_a_grid(self):
        """Feature 84 with exact non-zero 1-NN distance ties (fold 0 on even, fold 1 on odd grid points, so every
        evaluation row has several equidistant fold-train rows with different y). Under the verify maps (per-column
        scale 10**U(-2,2), shift up to 10 spreads) the tie band keeps all 88 features within 1e-6; with the band set
        to zero the same maps move feature 84, so the fixture really exercises rounding-split ties."""
        from unittest import mock
        rng = np.random.default_rng(31)
        n = 600
        g = (np.arange(n) % 2).astype(np.uint8)
        x = 2.0 * rng.integers(0, 5, size=(n, 3)) + g[:, None]
        y = x.sum(axis=1) + rng.standard_normal(n)
        i84 = F["knn1_top8_r2_cv"]

        def deltas(check_all):
            base = mf.features(x, y, g)
            mrng = np.random.default_rng(131)
            out = []
            for _ in range(6):
                sx = 10.0 ** mrng.uniform(-2, 2, size=3)
                bx = sx * mrng.uniform(-10, 10, size=3)
                sy = 10.0 ** mrng.uniform(-2, 2)
                by = sy * mrng.uniform(-10, 10)
                got = mf.features(x * sx + bx, y * sy + by, g)
                if check_all:
                    self.assert_close_features(got, base, rtol=1e-6, atol=1e-6, what="grid ties, tie band on")
                out.append(abs(float(got[i84]) - float(base[i84])))
            return out

        self.assertLess(max(deltas(check_all=True)), 1e-9)
        with mock.patch.multiple(mf, KNN_TIE_RTOL=0.0, KNN_TIE_ATOL=0.0):
            flipped = [d for d in deltas(check_all=False) if d > 1e-6]
        self.assertGreater(len(flipped), 0, "a plain argmin must flip on this fixture, else it tests nothing")


# ------------------------------------------------------------------------------------------------ T07
class T07RawLevelCounterfactual(A0Case):
    @staticmethod
    def minmax_all_rows(a):
        return (a - a.min(axis=0)) / (a.max(axis=0) - a.min(axis=0))

    def test_dataset_level_scaling_cannot_carry_test_information(self):
        rng = np.random.default_rng(8)
        n_train, n_test, p = 600, 150, 5
        x, y, _ = continuous(rng, n_train + n_test, p)
        g = (rng.permutation(n_train) % 2).astype(np.uint8)

        def raw_of(xr, yr):
            xs = self.minmax_all_rows(xr)
            ys = self.minmax_all_rows(yr.reshape(-1, 1))[:, 0]
            return make_raw(xs[:n_train], ys[:n_train], g, xs[n_train:], ys[n_train:])

        base_raw = raw_of(x, y)
        base = vec32(base_raw).astype(np.float64)
        y2 = y.copy()
        y2[n_train + 7] = y.max() + 3.0 * (y.max() - y.min())          # a test row becomes the global y maximum
        x2 = x.copy()
        x2[n_train + 11, 3] = x[:, 3].min() - 2.0 * np.ptp(x[:, 3])    # a test row becomes a column's global minimum
        for xr, yr, what in ((x, y2, "test y max"), (x2, y, "test x min"), (x2, y2, "both")):
            raw = raw_of(xr, yr)
            self.assertNotEqual(jtt1.train_rows_sha256(raw), jtt1.train_rows_sha256(base_raw),
                                "the rescale must change the train bytes")
            self.assert_close_features(vec32(raw).astype(np.float64), base, rtol=1e-6, atol=1e-6, what=what)


# ------------------------------------------------------------------------------------------------ T08
class T08DegenerateInputs(A0Case):
    def test_single_column(self):
        rng = np.random.default_rng(9)
        x, y, g = continuous(rng, 200, 1)
        v = mf.features(x, y, g)
        self.assertTrue(np.all(np.isfinite(v)) and np.all(np.abs(v) <= 1e4))
        self.assertEqual((v[F["c_pr_frac"]], v[F["c_top_eig_share"]]), (1.0, 1.0))
        self.assertEqual(v[F["x_active_col_frac"]], 1.0)

    def test_constant_target(self):
        rng = np.random.default_rng(10)
        x, _, g = continuous(rng, 150, 4)
        v = mf.features(x, np.full(150, 3.7), g)
        expect = {F["y_unique_frac"]: 1 / 150, F["y_mode_share"]: 1.0, F["y_at_min_frac"]: 1.0, F["y_at_max_frac"]: 1.0}
        for i in range(15, 40):
            self.assertEqual(v[i], expect.get(i, 0.0), mf.FEATURES[i])
        self.assertEqual([float(v[i]) for i in range(54, 74)], [0.0] * 20)
        self.assertEqual([float(v[i]) for i in range(82, 88)], [0.0] * 6)

    def test_all_constant_columns(self):
        rng = np.random.default_rng(11)
        n = 120
        x = np.tile(np.array([[1.0, -2.0, 0.5]]), (n, 1))
        y = rng.standard_normal(n)
        v = mf.features(x, y, (np.arange(n) % 2).astype(np.uint8))
        self.assertEqual(v[F["x_active_col_frac"]], 0.0)
        self.assertEqual(v[F["m_x_const_col_frac"]], 1.0)
        for i in list(range(40, 54)) + list(range(54, 74)) + list(range(74, 80)) + list(range(82, 88)):
            self.assertEqual(v[i], 0.0, mf.FEATURES[i])
        self.assertEqual((v[F["c_pr_frac"]], v[F["c_top_eig_share"]]), (1.0, 1.0))

    def test_binary_columns_and_duplicate_rows(self):
        rng = np.random.default_rng(12)
        n = 100
        x = rng.integers(0, 2, (n, 6)).astype(float)
        x[50:] = x[:50]
        y = x @ np.arange(1.0, 7.0) + rng.integers(0, 2, n)
        v = mf.features(x, y, (rng.permutation(n) % 2).astype(np.uint8))
        self.assertTrue(np.all(np.isfinite(v)) and np.all(np.abs(v) <= 1e4))
        self.assertEqual(v[F["m_x_binary_col_frac"]], 1.0)
        self.assertGreaterEqual(v[F["m_x_dup_row_frac"]], 0.5)

    def test_row_count_floor_and_nonfinite_target(self):
        rng = np.random.default_rng(13)
        x, y, g = continuous(rng, 8, 3)
        self.assertTrue(np.all(np.isfinite(mf.features(x, y, g))))
        with self.assertRaises(mf.A0Refused) as cm:
            mf.features(x[:7], y[:7], g[:7])
        self.assertEqual(cm.exception.reason, "too_few_rows")
        for bad in (np.nan, np.inf, -np.inf):
            y2 = y.copy()
            y2[3] = bad
            with self.assertRaises(mf.A0Refused) as cm:
                mf.features(x, y2, g)
            self.assertEqual(cm.exception.reason, "nonfinite_target")
        raw = make_raw(x, np.where(np.arange(8) == 2, np.nan, y), g, x[:2], y[:2])
        with self.assertRaises(mf.A0Refused):
            mf.from_raw(raw)

    def test_nonfinite_cells_are_counted_and_excluded(self):
        rng = np.random.default_rng(14)
        n = 200
        x, y, g = continuous(rng, n, 5)
        bad = x.copy()
        bad[3, 1], bad[10, 1], bad[0, 4] = np.nan, np.inf, -np.inf
        v = mf.features(bad, y, g)
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertAlmostEqual(v[F["m_x_nonfinite_cell_frac"]], 3 / (n * 5), places=15)
        self.assertAlmostEqual(v[F["m_x_invalid_col_frac"]], 2 / 5, places=15)
        reduced = mf.features(x[:, [0, 2, 3]], y, g)
        p_dependent = {1, 2, 4, 5, 6, 7, 8, 11, 40}
        differ = [mf.FEATURES[i] for i in range(88) if i not in p_dependent and v[i] != reduced[i]]
        self.assertEqual(differ, [], "invalid columns leaked into other statistics")


# ------------------------------------------------------------------------------------------------ T09
class T09HandComputed(A0Case):
    def test_target_counts(self):
        y = np.array([1, 1, 1, 2, 3, 3, 4, 5, 5, 5], dtype=float)
        x = np.random.default_rng(15).standard_normal((10, 2))
        v = mf.features(x, y, (np.arange(10) % 2).astype(np.uint8))
        self.assertEqual(v[F["y_unique_frac"]], 0.5)
        self.assertEqual(v[F["y_mode_share"]], 0.3)
        self.assertEqual(v[F["y_at_min_frac"]], 0.3)
        self.assertEqual(v[F["y_at_max_frac"]], 0.3)

    def test_robust_unit_quantiles(self):
        rng = np.random.default_rng(16)
        x, y, g = continuous(rng, 301, 3)
        y = np.exp(y / 2)
        v = mf.features(x, y, g)
        s, med = labels.robust_scale(y), float(np.median(y))
        self.assertAlmostEqual(v[F["y_rq_p100"]], slog((y.max() - med) / s), places=10)
        self.assertAlmostEqual(v[F["y_rq_p000"]], slog((y.min() - med) / s), places=10)
        self.assertAlmostEqual(v[F["y_rq_p090"]], slog((np.quantile(y, 0.9) - med) / s), places=10)
        self.assertAlmostEqual(v[F["y_robust_over_std"]], s / np.std(y), places=10)
        self.assertAlmostEqual(v[F["y_log1p_range_rz"]], math.log1p((y.max() - y.min()) / s), places=10)

    def test_pearson_against_corrcoef(self):
        rng = np.random.default_rng(17)
        x, y, g = continuous(rng, 250, 6)
        v = mf.features(x, y, g)
        r = [np.corrcoef(x[:, j], y)[0, 1] for j in range(6)]
        self.assertAlmostEqual(v[F["r_abs_max"]], max(abs(t) for t in r), places=12)
        self.assertAlmostEqual(v[F["r_pos_frac"]], np.mean(np.array(r) > 0), places=15)

    def test_spearman_with_ties_against_manual_ranks(self):
        rng = np.random.default_rng(18)
        n = 90
        x = rng.integers(0, 6, n).astype(float)
        y = np.round(x + rng.normal(0, 2, n))
        v = mf.features(x.reshape(-1, 1), y, (np.arange(n) % 2).astype(np.uint8))
        rho = np.corrcoef(manual_avg_rank(list(x)), manual_avg_rank(list(y)))[0, 1]
        self.assertAlmostEqual(v[F["rho_abs_max"]], abs(rho), places=12)

    def test_eta2_of_a_binary_column(self):
        rng = np.random.default_rng(19)
        n = 80
        x = rng.integers(0, 2, n).astype(float)
        y = 2 * x + rng.standard_normal(n)
        v = mf.features(x.reshape(-1, 1), y, (np.arange(n) % 2).astype(np.uint8))
        mu = y.mean()
        between = sum((x == b).sum() * (y[x == b].mean() - mu) ** 2 for b in (0.0, 1.0))
        self.assertAlmostEqual(v[F["eta2_max"]], between / ((y - mu) ** 2).sum(), places=12)

    def test_redundancy_orthogonal_and_twin_columns(self):
        base = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1],
                         [-1, -1, -1]], dtype=float)
        x = np.tile(base, (4, 1))
        y = np.random.default_rng(20).standard_normal(32)
        g = (np.arange(32) % 2).astype(np.uint8)
        v = mf.features(x, y, g)
        self.assertAlmostEqual(v[F["c_pr_frac"]], 1.0, places=12)
        self.assertAlmostEqual(v[F["c_abs_corr_max"]], 0.0, places=12)
        self.assertEqual(v[F["c_twin_col_frac"]], 0.0)
        twin = np.random.default_rng(21).standard_normal(32)
        v2 = mf.features(np.stack([twin, twin], axis=1), y, g)
        self.assertAlmostEqual(v2[F["c_pr_frac"]], 0.5, places=12)
        self.assertEqual(v2[F["c_twin_col_frac"]], 1.0)
        self.assertAlmostEqual(v2[F["c_abs_corr_max"]], 1.0, places=12)

    def test_duplicate_rows_and_conflicts(self):
        x = np.array([[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [-0.0, 7], [0.0, 7], [5, 6], [6, 6], [7, 1]], dtype=float)
        y = np.array([5, 5, 5, 1, 2, -0.0, 0.0, 3, 3, 4], dtype=float)
        v = mf.features(x, y, (np.arange(10) % 2).astype(np.uint8))
        self.assertEqual(v[F["m_x_dup_row_frac"]], 0.4)
        self.assertEqual(v[F["m_xy_dup_conflict_frac"]], 0.2)

    def test_linear_landmarker_is_one_for_an_exact_linear_target(self):
        rng = np.random.default_rng(22)
        x = rng.standard_normal((200, 3))
        y = 2 * x[:, 0] - 3 * x[:, 1] + 0.5 * x[:, 2] + 4
        v = mf.features(x, y, (rng.permutation(200) % 2).astype(np.uint8))
        self.assertAlmostEqual(v[F["lin_top8_r2_in"]], 1.0, places=6)
        self.assertAlmostEqual(v[F["lin_top8_r2_cv"]], 1.0, places=6)

    def test_knn_landmarker_is_one_with_exact_twins_across_folds(self):
        rng = np.random.default_rng(23)
        m = 60
        half = rng.standard_normal((m, 3))
        y_half = rng.standard_normal(m)
        x = np.concatenate([half, half])
        y = np.concatenate([y_half, y_half])
        g = np.array([0] * m + [1] * m, dtype=np.uint8)
        v = mf.features(x, y, g)
        self.assertEqual(v[F["knn1_top8_r2_cv"]], 1.0)


# ------------------------------------------------------------------------------------------------ T10
class T10Fuzz(A0Case):
    @staticmethod
    def column(rng, n, cols):
        kind = int(rng.integers(0, 8))
        if kind == 0:
            c = rng.standard_normal(n)
        elif kind == 1:
            c = np.clip(rng.standard_cauchy(n), -1e7, 1e7)
        elif kind == 2:
            c = np.round(rng.standard_normal(n) * 1.5)
        elif kind == 3:
            c = rng.integers(0, 2, n).astype(float)
        elif kind == 4:
            c = np.full(n, rng.standard_normal())
        elif kind == 5:
            c = rng.exponential(1.0, n) ** 3
        elif kind == 6 and cols:
            return cols[int(rng.integers(0, len(cols)))].copy()        # a duplicated column, already scaled
        else:
            c = rng.uniform(-1, 1, n) * (rng.random(n) < 0.1)
        return c * 10.0 ** rng.uniform(-300, 300) if rng.random() < 0.5 else c

    def dataset(self, seed):
        rng = np.random.default_rng([20261005, seed])
        n = int(rng.choice([8, 9, 16, 33, 64, 129, 250]))
        p = int(rng.choice([1, 2, 3, 5, 8, 17, 40, 64, 150, 300], p=[.1, .1, .1, .1, .15, .15, .1, .1, .05, .05]))
        cols = []
        for _ in range(p):
            cols.append(self.column(rng, n, cols))
        x = np.stack(cols, axis=1)
        if rng.random() < 0.3:
            k = int(rng.integers(1, max(2, n // 2)))
            x[rng.integers(0, n, k)] = x[rng.integers(0, n, k)]
        if rng.random() < 0.1:                                          # invalid cells: counted, excluded elsewhere
            x[int(rng.integers(0, n)), int(rng.integers(0, p))] = (np.nan, np.inf, -np.inf)[int(rng.integers(0, 3))]
        y = self.column(rng, n, [])
        if rng.random() < 0.1:
            y = x[:, 0].copy()
        r = rng.random()
        g = (rng.integers(0, 2, n) if r < 0.8 else np.zeros(n) if r < 0.9 else rng.integers(0, 3, n)).astype(np.uint8)
        return x, y, g

    def test_two_hundred_synthetic_datasets(self):
        seen = 0
        cover = {"nonfinite_x": 0, "constant_y": 0, "p_ge_150": 0, "labels_not_01": 0, "huge_scale": 0,
                 "tiny_scale": 0, "dup_rows": 0, "n_8": 0}
        for seed in range(200):
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                x, y, g = self.dataset(seed)
            fin = np.isfinite(x)
            nz = np.abs(x[fin & (x != 0)])
            cover["nonfinite_x"] += int(not fin.all())
            cover["constant_y"] += int(y.max() == y.min())
            cover["p_ge_150"] += int(x.shape[1] >= 150)
            cover["labels_not_01"] += int(set(np.unique(g).tolist()) != {0, 1})
            cover["huge_scale"] += int(nz.size > 0 and nz.max() > 1e200)
            cover["tiny_scale"] += int(nz.size > 0 and nz.min() < 1e-200)
            cover["dup_rows"] += int(np.unique(np.where(fin, x, 0.0), axis=0).shape[0] < x.shape[0])
            cover["n_8"] += int(x.shape[0] == 8)
            with warnings.catch_warnings(), np.errstate(invalid="raise", divide="raise"):
                warnings.simplefilter("error")
                v = mf.features(x, y, g)
                b = mf.to_f32le(v)
            f = np.frombuffer(b, dtype="<f4")
            self.assertTrue(np.all(np.isfinite(v)) and np.all(np.isfinite(f)), f"seed {seed}")
            self.assertTrue(np.all(np.abs(v) <= 1e4), f"seed {seed}: {float(np.max(np.abs(v)))}")
            seen += 1
        self.assertEqual(seen, 200)
        self.assertTrue(all(c > 0 for c in cover.values()), cover)


# ------------------------------------------------------------------------------------------------ T11
CHILD = r"""
import hashlib, os, struct
import numpy as np
from dltrain import metafeatures as mf
rng = np.random.default_rng(20261005)
n, p = 1700, 300
x = rng.standard_normal((n, p))
x[:, ::4] = np.round(x[:, ::4] * 2)
y = x[:, :5].sum(axis=1) + rng.standard_normal(n)
g = (rng.permutation(n) % 2).astype(np.uint8)
xt = rng.standard_normal((40, p))
m = np.concatenate([np.column_stack([x, y]), np.column_stack([xt, rng.standard_normal(40)])]).astype("<f8")
raw = b"JTT1" + struct.pack("<IIIB", n, 40, p, 2) + g.tobytes() + m.tobytes()
print(os.environ["OMP_NUM_THREADS"], os.environ["OPENBLAS_NUM_THREADS"], hashlib.sha256(mf.from_raw(raw)[0]).hexdigest())
"""


class T11ThreadIndependence(A0Case):
    def test_a0_bytes_do_not_depend_on_thread_count(self):
        digests = {}
        for threads in ("1", "4"):
            env = dict(os.environ, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
            out = subprocess.run([sys.executable, "-c", CHILD], env=env, cwd=str(LANE), capture_output=True,
                                 text=True, timeout=900)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            t1, t2, digest = out.stdout.split()
            self.assertEqual((t1, t2), (threads, threads))
            digests[threads] = digest
        self.assertEqual(digests["1"], digests["4"])


# ------------------------------------------------------------------------------------------------ T12
class T12RowPermutation(A0Case):
    def test_permuting_rows_with_their_labels(self):
        rng = np.random.default_rng(24)
        x, y, g = continuous(rng, 500, 6)
        base = mf.features(x, y, g)
        for seed in (1, 2):
            perm = np.random.default_rng(seed).permutation(500)
            got = mf.features(x[perm], y[perm], g[perm])
            self.assert_close_features(got, base, rtol=1e-9, atol=1e-12, skip={F["y_lag1_autocorr"]},
                                       what=f"permutation {seed}")
