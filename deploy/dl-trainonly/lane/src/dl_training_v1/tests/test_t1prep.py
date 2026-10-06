"""T1-PREP-01 leak tests. Every invariance check is paired with a leaky control that must FAIL it.

`upstream_t1` re-creates the acquisition tool's all-row fitting (median imputation, sorted category codes with
__MISSING__ last, min-max of every column, float32 '%.9g' printing) so the tests can change raw outer-test rows and
rebuild T1 exactly as the corpus was built.
"""
import json
import unittest

import numpy as np

from dltrain import t1prep, trainonly

N, N_TRAIN = 220, 160


def _f32_text(a):
    return np.asarray([float("%.9g" % np.float32(v)) for v in np.asarray(a, dtype=np.float64).ravel()]).reshape(np.shape(a))


def upstream_t1(num, cat, y, fit_rows=None):
    """num: (n, a) float with NaN for missing; cat: list of columns, each a list of str keys or None; y: (n,)."""
    n = len(y)
    rows = list(range(n)) if fit_rows is None else list(fit_rows)
    cols, enc, names = [], {}, []
    for j in range(num.shape[1]):
        v = num[:, j].copy()
        med = float(np.median(v[rows][~np.isnan(v[rows])]))
        v[np.isnan(v)] = med
        cols.append(v); names.append(f"n{j}"); enc[f"n{j}"] = {"kind": "numeric", "impute_value": med}
    for j, col in enumerate(cat):
        keys = sorted({col[i] for i in rows if col[i] is not None}, key=float)
        mapping = {k: c for c, k in enumerate(keys)}
        mapping[t1prep.MISSING_KEY] = len(keys)
        unseen = len(keys)
        cols.append(np.asarray([float(mapping.get(k, unseen) if k is not None else mapping[t1prep.MISSING_KEY]) for k in col]))
        names.append(f"c{j}"); enc[f"c{j}"] = {"kind": "categorical_codes", "impute_value": len(keys), "mapping": mapping}
    x = np.column_stack(cols)
    lo, hi = x[rows].min(axis=0), x[rows].max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    ylo, yhi = float(np.min(y[rows])), float(np.max(y[rows]))
    t1 = np.column_stack([_f32_text((x - lo) / span), _f32_text((y - ylo) / (yhi - ylo))])
    meta = {"output_headers": names + ["target"], "feature_columns": names[:1],  # stale, as in 28 real metas
            "feature_scaling": {"method": "minmax over cleaned source rows before sampling", "min": lo.tolist(),
                                "max": hi.tolist(), "range": [0.0, 1.0], "constant_columns": []},
            "target_scaling": {"min": ylo, "max": yhi}, "feature_encodings": enc}
    return t1, meta


def raw_fixture(seed=1):
    rng = np.random.default_rng(seed)
    num = rng.normal(size=(N, 3)) * np.array([1.0, 50.0, 3.0]) + np.array([0.0, 1000.0, -2.0])
    num[rng.random(size=num.shape) < 0.05] = np.nan
    pool = ["3400.0", "4700.0", "5900.0", "6000.0", "6300.0"]
    cat = [[pool[i % 5] if rng.random() > 0.08 else None for i in range(N)]]
    y = 2.0 * np.nan_to_num(num[:, 0]) + 0.01 * np.nan_to_num(num[:, 1]) + rng.normal(scale=0.2, size=N)
    perm = rng.permutation(N)
    tr, te = sorted(perm[:N_TRAIN].tolist()), sorted(perm[N_TRAIN:].tolist())
    cat[0][te[0]] = "5000.0"         # a category only outer-test rows have, between two train keys: a code shift
    cat[0][te[1]] = "5000.0"         # below it is uniform (affine, cancelled by min-max), above it is not
    return num, cat, y, tr, te


def prep(meta):
    return lambda m, tr, te, **kw: t1prep.prepare(m, meta, tr, te, **kw)


def legacy_all_rows(meta):
    """v1-like: the category map is fitted on every row (as upstream did); everything else as v2."""
    def run(m, tr, te, **kw):
        specs = t1prep.layout(meta)
        out = np.asarray(m, dtype=np.float64).copy()
        for s in specs:
            if s["kind"] == "categorical_codes":
                keys = t1prep.decode_categorical(out[:, s["index"]], s)
                out[:, s["index"]] = t1prep.apply_category_codes(keys, t1prep.fit_category_codes(keys))
        return trainonly.prepare_split(out, tr, te, **kw)
    return run


PROGRAMS = {"p_mean": lambda xf, yf, xe: np.full(xe.shape[0], float(np.mean(yf))),
            "p_x0": lambda xf, yf, xe: np.polyval(np.polyfit(xf[:, 0], yf, 1), xe[:, 0])}


class Decode(unittest.TestCase):
    def setUp(self):
        self.num, self.cat, self.y, self.tr, self.te = raw_fixture()
        self.t1, self.meta = upstream_t1(self.num, self.cat, self.y)

    def test_layout_uses_output_headers_not_stale_feature_columns(self):
        specs = t1prep.layout(self.meta)
        self.assertEqual([s["name"] for s in specs], ["n0", "n1", "n2", "c0"])

    def test_narrow_file_uses_feature_columns_and_scaling_by_name(self):
        meta = json.loads(json.dumps(self.meta))
        meta["feature_columns"] = ["n2", "c0"]                  # the file kept two of the four scaled columns
        narrow = self.t1[:, [2, 3, 4]]
        specs = t1prep.layout(meta, narrow.shape[1])
        self.assertEqual([s["name"] for s in specs], ["n2", "c0"])
        self.assertEqual((specs[0]["lo"], specs[1]["hi"]), (self.meta["feature_scaling"]["min"][2], self.meta["feature_scaling"]["max"][3]))
        self.assertEqual(t1prep.decode_categorical(narrow[:, 1], specs[1]), self.cat[0])
        fitted, rep = t1prep.refit(narrow, meta, self.tr)
        self.assertEqual(fitted.shape[1], 3)
        with self.assertRaises(t1prep.T1PrepError):
            t1prep.layout(meta, 4)

    def test_categorical_round_trip_is_exact(self):
        s = t1prep.layout(self.meta)[3]
        self.assertEqual(t1prep.decode_categorical(self.t1[:, 3], s), self.cat[0])

    def test_missing_and_unseen_categories_share_the_other_code(self):
        cat = [list(self.cat[0])]
        cat[0][self.te[2]] = cat[0][self.te[3]] = "6100.0"   # a second unseen category
        t1, meta = upstream_t1(self.num, cat, self.y)
        fitted, rep = t1prep.refit(t1, meta, self.tr)
        j = rep["kept_columns"].index(3)
        other = float(rep["categorical"]["c0"]["train"]["categories"])
        for i in range(N):
            if cat[0][i] in (None, "5000.0", "6100.0"):
                self.assertEqual(float(fitted[i, j]), other, (i, cat[0][i]))
        self.assertEqual(rep["categorical"]["c0"]["test"]["unseen"], 4)
        self.assertTrue(set(fitted[self.tr, j].tolist()) <= set(np.arange(other + 1).tolist()))

    def test_possibly_imputed_covers_every_imputed_cell(self):
        specs = t1prep.layout(self.meta)
        for j in range(3):
            flagged = t1prep.possibly_imputed(self.t1[:, j], specs[j])
            missing = np.isnan(self.num[:, j])
            self.assertTrue(bool(np.all(flagged[missing])), j)
            self.assertLess(int(flagged.sum()), 3 * int(missing.sum()) + 3)

    def test_refusals(self):
        s = t1prep.layout(self.meta)[3]
        bad = self.t1[:, 3].copy()
        bad[0] += 0.3 / (s["hi"] - s["lo"])
        with self.assertRaises(t1prep.T1PrepError):
            t1prep.decode_categorical(bad, s)
        with self.assertRaises(t1prep.T1PrepError):
            t1prep.refit(self.t1[:, 1:], self.meta, self.tr)
        meta = dict(self.meta, feature_scaling=dict(self.meta["feature_scaling"], min=[0.0]))
        with self.assertRaises(t1prep.T1PrepError):
            t1prep.layout(meta)


class CategoriesListFormat(unittest.TestCase):
    """1,405 of 2,496 real categorical columns use {categories: [...], impute_value: <mode code>} instead of a mapping."""

    def setUp(self):
        self.num, self.cat, self.y, self.tr, self.te = raw_fixture(3)
        t1, meta = upstream_t1(self.num, self.cat, self.y)
        e = meta["feature_encodings"]["c0"]
        keys = sorted((k for k in e["mapping"] if k != t1prep.MISSING_KEY), key=float)
        counts = {k: sum(1 for v in self.cat[0] if v == k) for k in keys}
        mode = max(keys, key=lambda k: (counts[k], k))
        col = np.asarray([float(keys.index(v if v is not None else mode)) for v in self.cat[0]])   # mode imputation
        lo, hi = col.min(), col.max()
        t1[:, 3] = _f32_text((col - lo) / (hi - lo))
        meta["feature_encodings"]["c0"] = {"kind": "categorical_codes", "categories": keys, "impute_value": float(keys.index(mode))}
        meta["feature_scaling"]["min"][3], meta["feature_scaling"]["max"][3] = float(lo), float(hi)
        self.t1, self.meta, self.mode = t1, meta, mode

    def test_round_trip_with_mode_imputation(self):
        s = t1prep.layout(self.meta)[3]
        want = [v if v is not None else self.mode for v in self.cat[0]]
        self.assertEqual(t1prep.decode_categorical(self.t1[:, 3], s), want)

    def test_mode_imputed_cells_are_counted(self):
        meta = json.loads(json.dumps(self.meta))
        for name in ("n0", "n1", "n2"):
            meta["feature_encodings"][name]["impute_value"] = None     # isolate the categorical contribution
        _, rep = t1prep.refit(self.t1, meta, self.tr)
        missing_train = sum(1 for i in self.tr if self.cat[0][i] is None)
        mode_train = sum(1 for i in self.tr if self.cat[0][i] in (None, self.mode))
        self.assertGreater(missing_train, 0)
        self.assertEqual(rep["possibly_imputed_cells"]["train"], mode_train)

    def test_train_side_bit_identical_when_test_categories_change(self):
        s = t1prep.layout(self.meta)[3]
        v = self.t1.copy()
        v[self.te, 3] = (0.0 - s["lo"]) / (s["hi"] - s["lo"])
        rep = trainonly.counterfactual_report(self.t1, self.tr, self.te, {"test_codes_zero": v}, PROGRAMS, prepare=prep(self.meta))
        self.assertEqual(rep["differing"], {})

    def test_refuses_both_or_neither_encoding(self):
        meta = json.loads(json.dumps(self.meta))
        meta["feature_encodings"]["c0"]["mapping"] = {"x": 0}
        with self.assertRaises(t1prep.T1PrepError):
            t1prep.layout(meta)
        del meta["feature_encodings"]["c0"]["mapping"], meta["feature_encodings"]["c0"]["categories"]
        with self.assertRaises(t1prep.T1PrepError):
            t1prep.layout(meta)


class UnseenCodeFormat(unittest.TestCase):
    """167 real columns: a mapping fitted on the T1 train split plus {unseen: -1} for categories it never saw."""

    def setUp(self):
        self.num, self.cat, self.y, self.tr, self.te = raw_fixture(4)
        t1, meta = upstream_t1(self.num, self.cat, self.y)
        e = meta["feature_encodings"]["c0"]
        mapping = {k: v for k, v in e["mapping"].items() if k not in (t1prep.MISSING_KEY, "5000.0")}
        mapping = {k: i for i, k in enumerate(sorted(mapping, key=float))}
        col = np.asarray([float(mapping.get(v, -1)) if v is not None else -1.0 for v in self.cat[0]])
        lo, hi = 0.0, float(len(mapping) - 1)
        t1[:, 3] = _f32_text((col - lo) / (hi - lo))
        meta["feature_encodings"]["c0"] = {"kind": "categorical_codes", "mapping": mapping, "unseen": -1}
        meta["feature_scaling"]["min"][3], meta["feature_scaling"]["max"][3] = lo, hi
        self.t1, self.meta, self.mapping = t1, meta, mapping

    def test_unseen_decodes_as_missing_and_is_counted(self):
        s = t1prep.layout(self.meta)[3]
        keys = t1prep.decode_categorical(self.t1[:, 3], s)
        self.assertEqual(keys, [v if v in self.mapping else None for v in self.cat[0]])
        _, rep = t1prep.refit(self.t1, self.meta, self.tr)
        self.assertEqual(rep["categorical"]["c0"]["t1_unseen_cells_all_rows"], sum(1 for v in self.cat[0] if v not in self.mapping))

    def test_other_unknown_codes_are_still_refused(self):
        s = t1prep.layout(self.meta)[3]
        bad = self.t1[:, 3].copy()
        bad[0] = (-2.0 - s["lo"]) / (s["hi"] - s["lo"])
        with self.assertRaises(t1prep.T1PrepError):
            t1prep.decode_categorical(bad, s)


class LeakT1Fixed(unittest.TestCase):
    """The outer-test rows' T1 values change, the outer-train rows' T1 values do not."""

    def setUp(self):
        self.num, self.cat, self.y, self.tr, self.te = raw_fixture()
        self.t1, self.meta = upstream_t1(self.num, self.cat, self.y)
        s = t1prep.layout(self.meta)[3]
        train_code = (s["inverse"] and [c for c, k in s["inverse"].items() if k == "4700.0"][0])
        v = self.t1.copy()
        v[self.te[:2], 3] = (train_code - s["lo"]) / (s["hi"] - s["lo"])   # the test-only category disappears
        w = self.t1.copy()
        rng = np.random.default_rng(9)
        w[self.te, :3] = rng.random(size=(len(self.te), 3)) * 3 - 1
        w[self.te, -1] = rng.random(size=len(self.te)) * 5
        self.variants = {"test_only_category_removed": v, "test_numeric_replaced": w,
                         "test_half_dropped": (self.t1, self.te[: len(self.te) // 2])}

    def test_v2_train_side_bit_identical(self):
        rep = trainonly.counterfactual_report(self.t1, self.tr, self.te, self.variants, PROGRAMS, prepare=prep(self.meta))
        self.assertEqual(rep["differing"], {})

    def test_negative_control_all_rows_category_map_fails(self):
        rep = trainonly.counterfactual_report(self.t1, self.tr, self.te, self.variants, PROGRAMS,
                                              prepare=legacy_all_rows(self.meta))
        self.assertIn("test_only_category_removed", rep["differing"])
        self.assertIn("train_block", rep["differing"]["test_only_category_removed"])


class LeakThroughUpstream(unittest.TestCase):
    """Raw outer-test rows change and T1 is rebuilt the way the corpus was (all-row fits)."""

    def setUp(self):
        self.num, self.cat, self.y, self.tr, self.te = raw_fixture()
        num2, cat2, y2 = self.num.copy(), [list(self.cat[0])], self.y.copy()
        num2[self.te] = num2[self.te] * 7.0 + 300.0
        y2[self.te] = y2[self.te] * 3.0 + 50.0
        for i in self.te[2:6]:
            cat2[0][i] = "6100.0"          # a new test-only category between two train keys
        self.raw2 = (num2, cat2, y2)

    def build(self, raw, fit_rows=None):
        t1, meta = upstream_t1(*raw, fit_rows=fit_rows)
        return t1prep.prepare(t1, meta, self.tr, self.te), t1, meta

    def test_categorical_codes_bit_identical_numeric_within_t1_rounding(self):
        a, t1a, meta_a = self.build((self.num, self.cat, self.y))
        b, t1b, meta_b = self.build(self.raw2)
        cat_j = [i for i, s in enumerate(t1prep.layout(meta_a)) if s["kind"] == "categorical_codes"]
        kept = a["t1prep"]["kept_columns"]
        for j in cat_j:
            k = kept.index(j)
            self.assertEqual(a["train"][:, k].tobytes(), b["train"][:, k].tobytes())
        self.assertEqual({c: r["train"] for c, r in a["t1prep"]["categorical"].items()},
                         {c: r["train"] for c, r in b["t1prep"]["categorical"].items()})
        self.assertNotEqual(a["t1prep"]["categorical"]["c0"]["test"], b["t1prep"]["categorical"]["c0"]["test"])
        sa, sb = t1prep.layout(meta_a), t1prep.layout(meta_b)
        num_j = [j for j in kept if j not in cat_j]
        flag = np.zeros((len(self.tr), len(num_j)), dtype=bool)
        for c, j in enumerate(num_j):
            flag[:, c] = t1prep.possibly_imputed(t1a[self.tr, j], sa[j]) | t1prep.possibly_imputed(t1b[self.tr, j], sb[j])
        num_k = [kept.index(j) for j in num_j]
        dev = np.abs(a["train"][:, num_k] - b["train"][:, num_k])
        clean = dev[~flag]
        self.assertGreater(float(clean.max()), 0.0, "the rounding residual is real and must be reported")
        self.assertLess(float(clean.max()), 1e-5, "outside imputed cells the residual is the float32 printing grid")
        self.assertGreater(float(dev[flag].max()), 1e-3, "imputed cells carry the all-row median: the counted leak")
        ty = np.abs(a["train"][:, -1] - b["train"][:, -1])
        self.assertLess(float(ty.max()), 1e-5)

    def test_upstream_fitted_on_outer_train_only_gives_bit_identical(self):
        a, _, _ = self.build((self.num, self.cat, self.y), fit_rows=self.tr)
        b, _, _ = self.build(self.raw2, fit_rows=self.tr)
        for f in ("train", "inner_labels", "lo", "span"):
            self.assertEqual(np.asarray(a[f]).tobytes(), np.asarray(b[f]).tobytes(), f)

    def test_negative_control_v1_codes_shift(self):
        t1a, meta_a = upstream_t1(self.num, self.cat, self.y)
        t1b, meta_b = upstream_t1(*self.raw2)
        self.assertNotEqual(t1a[self.tr, 3].tobytes(), t1b[self.tr, 3].tobytes())


class TargetCopySelection(unittest.TestCase):
    def setUp(self):
        self.num, self.cat, self.y, self.tr, self.te = raw_fixture()
        num = np.column_stack([self.num, np.zeros(N)])
        num[:, 3] = self.y * 2.0 + 1.0                       # an affine copy on every row
        num[self.te, 3] = np.random.default_rng(4).normal(size=len(self.te))   # but not on the outer-test rows
        self.t1, self.meta = upstream_t1(num, self.cat, self.y)

    def test_selected_on_outer_train_rows_only(self):
        _, rep = t1prep.refit(self.t1, self.meta, self.tr)
        self.assertEqual(rep["target_copy_dropped"], ["n3"])
        v = self.t1.copy()
        v[self.te, 3] = v[self.te, -1]
        _, rep2 = t1prep.refit(v, self.meta, self.tr)
        self.assertEqual(rep2["kept_columns"], rep["kept_columns"])

    def test_negative_control_all_rows_selection_changes(self):
        self.assertEqual(t1prep.target_copy_columns(self.t1), [])
        v = self.t1.copy()
        slope, intercept = np.polyfit(v[self.tr, -1], v[self.tr, 3], 1)
        v[self.te, 3] = slope * v[self.te, -1] + intercept      # the train-row affine relation, now on every row
        self.assertEqual(t1prep.target_copy_columns(v), [3])


if __name__ == "__main__":
    unittest.main()
