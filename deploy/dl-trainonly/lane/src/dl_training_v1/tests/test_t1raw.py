"""L7a content match and train-only refit (dltrain.t1raw). Every invariance or recovery check is paired with a negative
control that MUST FAIL the same check.

The raw source is synthetic; T1 is built from it by the all-row upstream replica (tests.test_t1prep.upstream_t1: median
imputation, category codes with __MISSING__ last, min-max over every cleaned row, float32 '%.9g' printing), so a change to
raw outer-test rows changes T1 everywhere, exactly as the corpus was built.
"""
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import unittest
import zipfile

import numpy as np

from dltrain import jtt1, t1prep, t1raw, trainonly
from tests.test_t1prep import _f32_text, upstream_t1

N_RAW, N_STAGED, N_A, N_TR = 300, 280, 220, 170
POOL = ["3400.0", "4700.0", "5900.0", "6000.0", "6300.0"]
HEADER = ["id", "c0", "n1", "yy", "n0", "n2", "extra"]          # raw order differs from T1 order (n0, n1, n2, c0)
ACQ_DAY = (2026, 10, 2)


def raw_arrays(seed=1):
    rng = np.random.default_rng(seed)
    num = rng.normal(size=(N_RAW, 3)) * np.array([1.0, 50.0, 3.0]) + np.array([0.0, 1000.0, -2.0])
    num[:, 2] = (18000 * 24 + rng.integers(0, 2000 * 24, N_RAW)) / 24.0        # datetime column: days, hour resolution
    num[rng.random(size=num.shape) < 0.06] = np.nan
    cat = [[POOL[i % 5] if rng.random() > 0.08 else None for i in range(N_RAW)]]
    y = 2.0 * np.nan_to_num(num[:, 0]) + 0.01 * np.nan_to_num(num[:, 1]) + rng.normal(scale=0.2, size=N_RAW)
    return num, cat, y


def days_text(v):
    t = dt.datetime(1970, 1, 1) + dt.timedelta(hours=int(round(v * 24)))
    return t.strftime("%Y-%m-%d %H:%M:%S")


def raw_rows(num, cat, y, thousands=False, na="", numeric_key_text=True):
    rows = []
    for i in range(len(y)):
        c = cat[0][i]
        ctext = "" if c is None else (c[:-2] if numeric_key_text and i % 2 == 0 and c.endswith(".0") else c)
        n1 = "" if np.isnan(num[i, 1]) else (format(float(num[i, 1]), ",") if thousands else repr(float(num[i, 1])))
        rows.append([str(i), ctext, n1, repr(float(y[i])), na if np.isnan(num[i, 0]) else repr(float(num[i, 0])),
                     "" if np.isnan(num[i, 2]) else days_text(num[i, 2]), "x%d" % (i % 7)])
    return rows


def csv_text(rows, header=HEADER, delim=",", preamble=(), implicit_index=False):
    buf = io.StringIO()
    for line in preamble:
        buf.write(line + "\n")
    w = csv.writer(buf, delimiter=delim, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(([r[0]] if implicit_index else []) + r)
    return buf.getvalue()


def build(num, cat, y, staged=None):
    """T1 + meta over the staged (cleaned) rows, all-row fitted like the corpus."""
    staged = list(range(N_STAGED)) if staged is None else staged
    t1, meta = upstream_t1(num[staged], [[cat[0][i] for i in staged]], y[staged])
    meta["feature_encodings"]["n2"].update({"derived_from": "datetime", "unit": "days since 1970-01-01"})
    meta["original_headers"] = list(HEADER)
    meta["target_original_name"] = meta["source_target_column_original"] = "yy"
    meta["acquisition"] = {"acquired_at": "2026-10-02T12:00:00+00:00", "tool": "fixture",
                           "download": {"container": "plain;sep=,;enc=utf-8", "encoding": "utf-8"}}
    return t1, meta, staged


def sample(seed=5):
    rng = np.random.default_rng(seed)
    pick = rng.choice(N_STAGED, size=N_A, replace=False)               # research sample: a subset, in its own order
    perm = rng.permutation(N_A)
    return pick, sorted(perm[:N_TR].tolist()), sorted(perm[N_TR:].tolist())


def truth_values(num, cat, y, meta, raw_idx):
    specs = t1raw.columns(meta, 5)
    out = np.empty((len(raw_idx), 5))
    for k, i in enumerate(raw_idx):
        out[k, :3] = num[i]
        c = cat[0][i]
        out[k, 3] = t1raw.MISSING_CODE if c is None else [code for code, key in specs[3]["inverse"].items() if key == c][0]
        out[k, 4] = y[i]
    return out


def same(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def recovery_ok(result, truth):
    """The check a matcher must pass: either it refuses, or every recovered row equals the true raw row."""
    return result["status"] == "unmatched" or same(result["values"], truth)


def train_side(split):
    """sha256 of every train-side field: scaler, train-only-scaled block, refitted unscaled train rows (teacher input),
    inner labels and the JTT1 train rows."""
    h = lambda b: hashlib.sha256(b).hexdigest()
    return {"scaler": h(trainonly.scaler_bytes(split)), "train": h(np.asarray(split["train"], "<f8").tobytes()),
            "fitted_train": h(np.asarray(split["train_raw"], "<f8").tobytes()),
            "inner": h(np.asarray(split["inner_labels"], np.uint8).tobytes()), "jtt1_train_rows": jtt1.train_rows_sha256(split["jtt1"])}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="t1raw-", dir=os.environ.get("TMPDIR"))
        self.num, self.cat, self.y = raw_arrays()
        self.t1, self.meta, self.staged = build(self.num, self.cat, self.y)
        self.pick, self.R, self.T = sample()
        self.A = self.t1[self.pick]
        self.raw_idx = [self.staged[i] for i in self.pick]
        self.truth = truth_values(self.num, self.cat, self.y, self.meta, self.raw_idx)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, text_or_bytes, name="source.csv"):
        p = os.path.join(self.dir, name)
        with open(p, "wb") as f:
            f.write(text_or_bytes.encode("utf-8") if isinstance(text_or_bytes, str) else text_or_bytes)
        return p

    def plain(self, **kw):
        return self.write(csv_text(raw_rows(self.num, self.cat, self.y, **kw)))


class ExactMatch(Fixture):
    def test_every_row_recovers_its_raw_values(self):
        r = t1raw.match_dataset(self.plain(), self.meta, self.A)
        self.assertEqual(r["status"], "matched", r)
        self.assertTrue(same(r["values"], self.truth))
        st = r["stats"]
        self.assertEqual((st["unmatched_rows"], st["ambiguous_rows"], st["oversubscribed_rows"]), (0, 0, 0))
        self.assertLessEqual(st["residual"]["max_dev_over_tol"], 1.0)
        self.assertGreater(st["matched_missing_cells"], 0)
        self.assertTrue(recovery_ok(r, self.truth))

    def test_numeric_text_of_a_float_keyed_category_maps_by_value(self):
        specs = t1raw.columns(self.meta, 5)
        look = t1raw._category_lookup(specs[3])
        code = [c for c, k in specs[3]["inverse"].items() if k == "3400.0"][0]
        self.assertEqual((look("3400"), look("3400.0"), look(" 3400.0"), look("NA"), look(""), look("3401")),
                         (code, code, code, t1raw.MISSING_CODE, t1raw.MISSING_CODE, t1raw.UNMAPPED_CODE))

    def test_missing_raw_cell_must_sit_on_the_impute_value(self):
        rows = raw_rows(self.num, self.cat, self.y)
        imp = self.meta["feature_encodings"]["n0"]["impute_value"]
        i = next(k for k in self.raw_idx if not np.isnan(self.num[k, 0]) and abs(self.num[k, 0] - imp) > 1e-3)
        rows[i][4] = ""                                          # T1 holds a real n0 value here, not the impute value
        r = t1raw.match_dataset(self.write(csv_text(rows)), self.meta, self.A)
        self.assertEqual((r["status"], r["reason"]), ("unmatched", "rows_unmatched"))
        self.assertEqual(r["stats"]["unmatched_rows"], 1)
        self.assertIn("n0", r["stats"]["blame_columns_first_unmatched"])

    def test_missing_raw_category_must_sit_on_the_missing_code(self):
        rows = raw_rows(self.num, self.cat, self.y)
        i = next(k for k in self.raw_idx if self.cat[0][k] is not None)
        rows[i][1] = ""                                          # T1 holds a real category code here
        r = t1raw.match_dataset(self.write(csv_text(rows)), self.meta, self.A)
        self.assertEqual((r["status"], r["reason"]), ("unmatched", "rows_unmatched"))
        self.assertIn("c0", r["stats"]["blame_columns_first_unmatched"])

    def test_missing_vs_impute_valued_raw_rows_are_ambiguous_and_refused(self):
        rows = raw_rows(self.num, self.cat, self.y)
        i = next(k for k in self.raw_idx if np.isnan(self.num[k, 0]))
        twin = list(rows[i])
        twin[0] = "twin"
        twin[4] = repr(float(self.meta["feature_encodings"]["n0"]["impute_value"]))   # a real value equal to the impute
        r = t1raw.match_dataset(self.write(csv_text(rows + [twin])), self.meta, self.A)
        self.assertEqual((r["status"], r["reason"]), ("unmatched", "rows_ambiguous"))

    def test_duplicate_raw_rows_are_one_class_and_one_to_one_is_enforced(self):
        rows = raw_rows(self.num, self.cat, self.y)
        dup = list(rows[self.raw_idx[0]])
        dup[0], dup[6] = "dup", "other"                           # differs only in columns the meta does not use
        r = t1raw.match_dataset(self.write(csv_text(rows + [dup])), self.meta, self.A)
        self.assertEqual(r["status"], "matched")
        self.assertEqual(r["stats"]["duplicate_classes_claimed"], 1)
        A2 = np.vstack([self.A, self.A[:1]])                      # two T1 rows claim a raw row that exists once
        r2 = t1raw.match_dataset(self.plain(), self.meta, A2)
        self.assertEqual((r2["status"], r2["reason"]), ("unmatched", "rows_oversubscribed"))

    def test_raw_rows_absent_from_the_source_are_unmatched(self):
        rows = raw_rows(self.num, self.cat, self.y)
        gone = self.raw_idx[3]
        r = t1raw.match_dataset(self.write(csv_text([x for k, x in enumerate(rows) if k != gone])), self.meta, self.A)
        self.assertEqual((r["status"], r["reason"], r["stats"]["unmatched_rows"]), ("unmatched", "rows_unmatched_target", 1))


class LooseMatcherNegativeControl(Fixture):
    """A raw row is replaced by a near-duplicate decoy that differs by 1e-5 of the column span: beyond the float32
    printing error, inside a loose tolerance. The strict matcher refuses; the loose one accepts the wrong row."""

    def setUp(self):
        super().setUp()
        rows = raw_rows(self.num, self.cat, self.y)
        k = next(i for i in self.raw_idx if not np.isnan(self.num[i, 1]))
        span = self.meta["feature_scaling"]["max"][1] - self.meta["feature_scaling"]["min"][1]
        rows[k][2] = repr(float(self.num[k, 1]) + 1e-5 * span)
        self.path = self.write(csv_text(rows))

    def test_strict_matcher_refuses(self):
        r = t1raw.match_dataset(self.path, self.meta, self.A)
        self.assertEqual(r["status"], "unmatched")
        self.assertTrue(recovery_ok(r, self.truth))

    def test_negative_control_loose_tolerance_accepts_a_wrong_row(self):
        r = t1raw.match_dataset(self.path, self.meta, self.A, tol_rel=1e-3)
        self.assertEqual(r["status"], "matched")
        self.assertFalse(recovery_ok(r, self.truth))


class TrainOnlyRefit(Fixture):
    """Raw outer-test rows change (values, missing pattern, a test-only category, the target) and T1 is rebuilt from
    the changed raw file with the all-row upstream fits. The L7a train side must be bit-identical."""

    def variant(self):
        num, cat, y = self.num.copy(), [list(self.cat[0])], self.y.copy()
        test_raw = [self.raw_idx[i] for i in self.T]
        num[test_raw] = num[test_raw] * 7.0 + 300.0
        num[test_raw, 2] = self.num[test_raw, 2] + 3.0                        # dates move by three days
        rng = np.random.default_rng(13)
        flip = rng.random(size=(len(test_raw), 3)) < 0.3
        for a, i in enumerate(test_raw):
            for j in range(3):
                if flip[a, j]:
                    num[i, j] = np.nan if not np.isnan(num[i, j]) else 17.0 + 24.0 * j + (18000 if j == 2 else 0)
        for i in test_raw[:6]:
            cat[0][i] = "6100.0"                                              # a category only outer-test rows have
        for i in test_raw[6:9]:
            cat[0][i] = None
        y[test_raw] = y[test_raw] * 3.0 + 50.0
        t1, meta, _ = build(num, cat, y)
        path = self.write(csv_text(raw_rows(num, cat, y)), "variant.csv")
        return t1[self.pick], meta, path

    def both(self, prepare):
        base = t1raw.match_dataset(self.plain(), self.meta, self.A)
        A2, meta2, path2 = self.variant()
        var = t1raw.match_dataset(path2, meta2, A2)
        self.assertEqual((base["status"], var["status"]), ("matched", "matched"))
        self.assertNotEqual(self.A[self.R].tobytes(), A2[self.R].tobytes(), "T1 train rows moved: the all-row leak")
        a = prepare(base["values"], base["specs"], self.R, self.T)
        b = prepare(var["values"], var["specs"], self.R, self.T)
        return a, b

    def test_train_side_bit_identical(self):
        a, b = self.both(t1raw.prepare)
        self.assertEqual(train_side(a), train_side(b))
        self.assertNotEqual(a["test"].tobytes(), b["test"].tobytes())
        self.assertNotEqual(a["t1raw"]["imputation"]["imputed_cells"]["test"], b["t1raw"]["imputation"]["imputed_cells"]["test"])
        self.assertEqual(a["t1raw"]["imputation"]["columns"]["n0"]["train_median"], b["t1raw"]["imputation"]["columns"]["n0"]["train_median"])

    def test_test_rows_get_the_train_median(self):
        base = t1raw.match_dataset(self.plain(), self.meta, self.A)
        fitted, rep = t1raw.refit(base["values"], base["specs"], self.R)
        v = base["values"]
        for j, name in enumerate(["n0", "n1", "n2"]):
            med = float(np.median(v[self.R, j][~np.isnan(v[self.R, j])]))
            self.assertEqual(rep["imputation"]["columns"][name]["train_median"], med.hex())
            miss = np.isnan(v[:, j])
            self.assertTrue(bool(np.all(fitted[miss, j] == med)))
            self.assertTrue(bool(np.any(miss[self.T])))
        self.assertEqual(rep["imputation"]["imputed_cells"]["train"], int(np.isnan(v[self.R, :3]).sum()))
        self.assertEqual(rep["imputation"]["imputed_cells"]["test"], int(np.isnan(v[self.T, :3]).sum()))

    def test_negative_control_all_rows_median_fails(self):
        def leaky(values, specs, R, T):
            v = np.array(values, dtype=np.float64)
            for j in range(3):
                col = v[:, j]
                col[np.isnan(col)] = float(np.median(col[~np.isnan(col)]))   # median over outer-train AND outer-test
            return t1raw.prepare(v, specs, R, T)
        a, b = self.both(leaky)
        self.assertNotEqual(train_side(a), train_side(b))

    def test_negative_control_all_rows_category_map_fails(self):
        def leaky(values, specs, R, T):
            fitted, rep = t1raw.refit(values, specs, R)
            keys = [None if c < 0 else specs[3]["inverse"][int(c)] for c in np.asarray(values)[:, 3]]
            fitted[:, rep["kept_columns"].index(3)] = t1prep.apply_category_codes(keys, t1prep.fit_category_codes(keys))
            return trainonly.prepare_split(fitted, R, T)                      # category map fitted on every row
        a, b = self.both(leaky)
        self.assertNotEqual(train_side(a), train_side(b))

    def test_categories_unseen_in_outer_train_share_the_missing_code(self):
        A2, meta2, path2 = self.variant()
        r = t1raw.match_dataset(path2, meta2, A2)
        fitted, rep = t1raw.refit(r["values"], r["specs"], self.R)
        c = rep["categorical"]["c0"]
        self.assertEqual(c["test"]["unseen"], 6)
        other = float(c["train"]["categories"])
        j = rep["kept_columns"].index(3)
        self.assertTrue(bool(np.all(fitted[[self.T[i] for i in range(6)], j] == other)))


class TargetCopyOnTrainRows(Fixture):
    def test_selected_on_outer_train_rows_only(self):
        v = np.array(self.truth)
        v[:, 1] = v[:, 4] * 2.0 + 1.0                                # an affine target copy on outer-train rows
        v[self.T, 1] = np.random.default_rng(3).normal(size=len(self.T))
        specs = t1raw.columns(self.meta, 5)
        _, rep = t1raw.refit(v, specs, self.R)
        self.assertEqual(rep["target_copy_dropped"], ["n1"])
        w = v.copy()
        w[self.T, 1] = w[self.T, 4]
        _, rep2 = t1raw.refit(w, specs, self.R)
        self.assertEqual(rep2["kept_columns"], rep["kept_columns"])
        self.assertEqual(t1prep.target_copy_columns(np.nan_to_num(v[:, [0, 1, 4]])), [])   # all rows: not a copy

    def test_missing_target_and_bad_codes_are_refused(self):
        specs = t1raw.columns(self.meta, 5)
        v = np.array(self.truth)
        v[0, 4] = np.nan
        with self.assertRaises(t1raw.T1RawError):
            t1raw.refit(v, specs, self.R)
        v = np.array(self.truth)
        v[0, 3] = t1raw.UNMAPPED_CODE
        with self.assertRaises(t1raw.T1RawError):
            t1raw.refit(v, specs, self.R)

    def test_column_without_train_values_is_imputed_with_zero_and_counted(self):
        specs = t1raw.columns(self.meta, 5)
        v = np.array(self.truth)
        v[self.R, 0] = np.nan
        fitted, rep = t1raw.refit(v, specs, self.R)
        self.assertEqual(rep["imputation"]["no_train_value_columns"], ["n0"])
        self.assertTrue(bool(np.all(fitted[self.R, 0] == 0.0)))


class CategoriesListFormat(Fixture):
    """String categories, {categories: [...], impute_value: mode code}: a missing raw cell sits on the mode code in T1,
    but L7a knows it is missing and gives it the train-only extra code."""

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(21)
        colours = ["blue", "green", "red"]
        self.keys = [colours[int(k)] if rng.random() > 0.1 else None for k in rng.integers(0, 3, N_RAW)]
        st = [self.keys[i] for i in self.staged]
        counts = {c: st.count(c) for c in colours}
        self.mode = max(colours, key=lambda c: (counts[c], c))
        codes = np.asarray([float(colours.index(k if k is not None else self.mode)) for k in st])
        self.t1[:, 3] = _f32_text(codes / 2.0)
        self.meta["feature_encodings"]["c0"] = {"kind": "categorical_codes", "categories": colours,
                                                "impute_value": float(colours.index(self.mode))}
        self.meta["feature_scaling"]["min"][3], self.meta["feature_scaling"]["max"][3] = 0.0, 2.0
        self.cat = [self.keys]
        self.A = self.t1[self.pick]

    def test_match_and_missing_becomes_its_own_code(self):
        rows = raw_rows(self.num, self.cat, self.y, numeric_key_text=False)
        r = t1raw.match_dataset(self.write(csv_text(rows)), self.meta, self.A)
        self.assertEqual(r["status"], "matched", r)
        miss = [i for i, k in enumerate(self.raw_idx) if self.keys[k] is None]
        self.assertTrue(miss)
        self.assertTrue(bool(np.all(r["values"][miss, 3] == t1raw.MISSING_CODE)))
        fitted, rep = t1raw.refit(r["values"], r["specs"], self.R)
        j = rep["kept_columns"].index(3)
        self.assertTrue(bool(np.all(fitted[miss, j] == float(rep["categorical"]["c0"]["train"]["categories"]))))
        # L7b decoding of the same T1 cells would give the mode key instead
        dec = t1prep.decode_categorical(self.A[:, 3], t1prep.layout(self.meta)[3])
        self.assertTrue(all(dec[i] == self.mode for i in miss))

    def test_unknown_category_text_cannot_match(self):
        rows = raw_rows(self.num, self.cat, self.y, numeric_key_text=False)
        k = next(i for i in self.raw_idx if self.keys[i] is not None)
        rows[k][1] = "purple"
        r = t1raw.match_dataset(self.write(csv_text(rows)), self.meta, self.A)
        self.assertEqual(r["status"], "unmatched")
        self.assertEqual(r["read"]["unmapped_category_cells"], {"c0": 1})


class Reader(Fixture):
    def values_of(self, path, meta=None):
        meta = meta or self.meta
        return t1raw.read_source(path, meta, t1raw.columns(meta, 5))

    def test_containers_delimiters_and_layouts_reconstruct_identically(self):
        rows = raw_rows(self.num, self.cat, self.y)
        ref = self.values_of(self.write(csv_text(rows)))["values"]
        text = csv_text(rows)
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, "w") as z:
            z.writestr("notes.txt", "readme")
            z.writestr("data/table.csv", text)
        meta_zip = json.loads(json.dumps(self.meta))
        meta_zip["acquisition"]["download"]["container"] = "zip:table.csv;sep=,;enc=utf-8"
        meta_nohint = json.loads(json.dumps(self.meta))
        meta_nohint["acquisition"]["download"] = {}
        cases = {
            "gzip": (self.write(gzip.compress(text.encode()), "s.csv.gz"), self.meta),
            "gzip_twice": (self.write(gzip.compress(gzip.compress(text.encode())), "s2.gz"), self.meta),
            "zip_member_hint": (self.write(zb.getvalue(), "s.zip"), meta_zip),
            "semicolon_sniffed": (self.write(csv_text(rows, delim=";"), "semi.csv"), meta_nohint),
            "tab_sniffed": (self.write(csv_text(rows, delim="\t"), "tab.tsv"), meta_nohint),
            "bom_and_preamble": (self.write("\ufeff" + csv_text(rows, preamble=["Source: fixture", "", "units;none"]), "pre.csv"), meta_nohint),
            "implicit_index": (self.write(csv_text(rows, implicit_index=True), "idx.csv"), self.meta),
            "bom_on_a_used_first_column": (self.write("\ufeff" + csv_text([r_[1:] + r_[:1] for r_ in rows], header=HEADER[1:] + HEADER[:1]),
                                                      "bom.csv"), self.meta),
            "thousands": (self.write(csv_text(raw_rows(self.num, self.cat, self.y, thousands=True)), "th.csv"), self.meta),
            "na_text": (self.write(csv_text(raw_rows(self.num, self.cat, self.y, na="NA")), "na.csv"), self.meta),
            "cp1252": (self.write(csv_text(rows).replace("x3", "café").encode("cp1252"), "w.csv"), meta_nohint),
        }
        self.assertTrue(any("," in r[2] for r in raw_rows(self.num, self.cat, self.y, thousands=True)))
        for name, (path, meta) in cases.items():
            got = self.values_of(path, meta)
            self.assertTrue(same(got["values"], ref), name)
        self.assertTrue(self.values_of(cases["implicit_index"][0])["report"]["implicit_index"])
        self.assertEqual(self.values_of(cases["gzip_twice"][0])["report"]["layers"], ["gzip", "gzip"])
        self.assertEqual(self.values_of(cases["cp1252"][0], meta_nohint)["report"]["encoding"], "cp1252")

    def test_replacement_decoded_names_follow_the_tool(self):
        rows = raw_rows(self.num, self.cat, self.y)
        header = [h if h != "n1" else "a\u00f1o" for h in HEADER]           # 'a\xf1o' in latin-1 bytes
        meta = json.loads(json.dumps(self.meta))
        meta["acquisition"]["download"] = {"container": "plain;sep=,"}
        for key in ("output_headers", "original_headers"):
            meta[key] = [h if h != "n1" else "a\ufffdo" for h in meta[key]]   # the tool decoded utf-8 with replacement
        meta["feature_encodings"]["a\ufffdo"] = meta["feature_encodings"].pop("n1")
        path = self.write(csv_text(rows, header=header).encode("latin-1"), "latin.csv")
        r = t1raw.match_dataset(path, meta, self.A)
        self.assertEqual(r["status"], "matched", r)
        self.assertEqual(r["read"]["encoding"], t1raw.REPLACE_DECODE)
        self.assertTrue(same(r["values"], self.truth))

    def test_short_rows_are_padded_missing_and_long_rows_skipped(self):
        rows = raw_rows(self.num, self.cat, self.y)
        text = csv_text(rows) + "999,3400\n" + ",".join(["1"] * 9) + "\n"
        got = self.values_of(self.write(text))
        self.assertEqual(got["report"]["short_rows_padded"], 1)
        self.assertEqual(got["report"]["long_rows_skipped"], 1)
        self.assertEqual(got["values"].shape[0], N_RAW + 1)
        self.assertTrue(bool(np.isnan(got["values"][-1, 4])))

    def test_refusals(self):
        r = t1raw.match_dataset(self.write("<!DOCTYPE html><html></html>", "page.html"), self.meta, self.A)
        self.assertEqual(r["reason"], "markup_not_table")
        r = t1raw.match_dataset(self.write('[{"a": 1}]', "x.json"), self.meta, self.A)
        self.assertEqual(r["reason"], "json_not_supported")
        rows = [r_[:2] + r_[3:] for r_ in raw_rows(self.num, self.cat, self.y)]
        r = t1raw.match_dataset(self.write(csv_text(rows, header=[h for h in HEADER if h != "n1"])), self.meta, self.A)
        self.assertEqual(r["reason"], "header_missing_columns")
        meta = json.loads(json.dumps(self.meta))
        meta["feature_encodings"]["n2"]["unit"] = "seconds"
        r = t1raw.match_dataset(self.plain(), meta, self.A)
        self.assertEqual(r["reason"], "unsupported_derivation")
        bad = self.A.copy()
        bad[0, 3] += 0.3 / (self.meta["feature_scaling"]["max"][3] - self.meta["feature_scaling"]["min"][3])
        r = t1raw.match_dataset(self.plain(), self.meta, bad)
        self.assertEqual(r["reason"], "t1_code_not_integral")

    def test_numeric_cleaning_follows_the_tool(self):
        cases = {"$30,000": 30000.0, "40, 50": 4050.0, "70.1999545%": 70.1999545, "1,234,567.5": 1234567.5, "14,15": 1415.0,
                 " 12 ": 12.0, "1e5": 1e5, "-": None, "": None, "NA": None, "1_000": None, "inf": None, "abc": None, "(5)": None}
        for text, want in cases.items():
            got = t1raw._to_float(text)
            self.assertTrue(math.isnan(got) if want is None else got == want, text)

    def test_bool_text_column_reads_as_one_and_zero(self):
        rows = raw_rows(self.num, self.cat, self.y)
        for i, r_ in enumerate(rows):
            r_[4] = "" if np.isnan(self.num[i, 0]) else ("TRUE" if self.num[i, 0] > 0 else "FALSE")
        got = self.values_of(self.write(csv_text(rows)))
        col = got["values"][:, 0]
        want = np.where(np.isnan(self.num[:, 0]), np.nan, (self.num[:, 0] > 0).astype(float))
        self.assertTrue(same(col, want))
        self.assertEqual(got["report"]["bool_columns"], ["n0"])


def t1_decimal9(num, cat, y, meta, staged):
    """The same T1 printed '%.9g' from float64 (as regression_hunt.ingest did) instead of from float32."""
    names = meta["output_headers"][:-1]
    lo, hi = meta["feature_scaling"]["min"], meta["feature_scaling"]["max"]
    cols = []
    for j, n in enumerate(names):
        e = meta["feature_encodings"][n]
        if e["kind"] == "numeric":
            v = num[staged, j].copy()
            v[np.isnan(v)] = e["impute_value"]
        else:
            v = np.asarray([float(e["mapping"][cat[0][i] if cat[0][i] is not None else t1prep.MISSING_KEY]) for i in staged])
        span = (hi[j] - lo[j]) or 1.0
        cols.append([float("%.9g" % x) for x in (v - lo[j]) / span])
    ts = meta["target_scaling"]
    cols.append([float("%.9g" % x) for x in (y[staged] - ts["min"]) / (ts["max"] - ts["min"])])
    return np.column_stack(cols)


class PrintClass(Fixture):
    """regression_hunt printed T1 as float64 '%.9g': its printing error is ~10x below the float32 bound. The class is
    read from the T1 values; the tighter bound only removes candidates the printing cannot have produced."""

    def setUp(self):
        super().setUp()
        self.A9 = t1_decimal9(self.num, self.cat, self.y, self.meta, self.staged)[self.pick]

    def test_class_detection(self):
        self.assertEqual([t1raw.printed_decimal9(self.A[:, j]) for j in range(5)], [False] * 5)
        self.assertEqual([t1raw.printed_decimal9(self.A9[:, j]) for j in (0, 1, 2, 4)], [True] * 4)
        self.assertFalse(t1raw.printed_decimal9(np.array([0.0, 0.5, 1.0, 0.25])))

    def test_decimal9_t1_matches_exactly_with_the_tighter_bound(self):
        r = t1raw.match_dataset(self.plain(), self.meta, self.A9)
        self.assertEqual(r["status"], "matched", r)
        self.assertTrue(same(r["values"], self.truth))
        self.assertGreaterEqual(r["stats"]["print_class"]["decimal9"], 4)
        self.assertLessEqual(r["stats"]["residual"]["max_dev_over_tol"], 1.0)
        self.assertLess(r["stats"]["residual"]["max_dev_over_span"], 1e-8)

    def decoy_path(self):
        rows = raw_rows(self.num, self.cat, self.y)
        k = next(i for i in self.raw_idx if not np.isnan(self.num[i, 1]))
        span = self.meta["feature_scaling"]["max"][1] - self.meta["feature_scaling"]["min"][1]
        decoy = list(rows[k])
        decoy[0] = "decoy"
        decoy[2] = repr(float(self.num[k, 1]) + 5e-8 * span)      # outside the 9-digit bound, inside the float32 bound
        return self.write(csv_text(rows + [decoy]))

    def test_tighter_bound_resolves_a_near_duplicate(self):
        r = t1raw.match_dataset(self.decoy_path(), self.meta, self.A9)
        self.assertEqual(r["status"], "matched")
        self.assertTrue(same(r["values"], self.truth))

    def test_negative_control_float32_bound_on_decimal9_t1_is_ambiguous(self):
        orig = t1raw.printed_decimal9
        t1raw.printed_decimal9 = lambda col: False
        try:
            r = t1raw.match_dataset(self.decoy_path(), self.meta, self.A9)
        finally:
            t1raw.printed_decimal9 = orig
        self.assertEqual((r["status"], r["reason"]), ("unmatched", "rows_ambiguous"))
        self.assertFalse(r["status"] == "matched" and same(r["values"], self.truth))


class Datetimes(unittest.TestCase):
    CASES = [("2026-12-23", (2026, 12, 23)), ("02/22/2010", (2010, 2, 22)), ("22/02/2010", (2010, 2, 22)),
             ("06/05/2006", (2006, 6, 5)), ("2024-12-31T23:28:55.746Z", (2024, 12, 31, 23, 28, 55, 746000)),
             ("2021-07-17T20:02:29.885357+06:00", (2021, 7, 17, 14, 2, 29, 885357)), ("2010-01", (2010, 1, 1)),
             ("00:00:00", ACQ_DAY), ("13:30", ACQ_DAY + (13, 30)), ("01/02/2015 12:00:00 AM", (2015, 1, 2)),
             ("01/02/2015 12:30:00 PM", (2015, 1, 2, 12, 30)), ("12-Oct-10", (2010, 10, 12)), ("15-02-20", (2020, 2, 15)),
             ("2021 May 10 06:01:09 AM", (2021, 5, 10, 6, 1, 9)), ("05/25/2022 05:12 PM EDT", (2022, 5, 25, 17, 12)),
             ("07/01/2021 12:00:00 AM +0000", (2021, 7, 1)), ("10-02-2024 14:15", (2024, 10, 2, 14, 15)),
             ("February 05, 2013", (2013, 2, 5)), ("20 June 2016", (2016, 6, 20)), ("June, 1998", (1998, 6, 1)),
             ("Oct 2022", (2022, 10, 1)), ("1995-Q1", (1995, 1, 1)), ("2016Q3", (2016, 7, 1)), ("2000", (2000, 1, 1)),
             ("20200101", (2020, 1, 1)), ("2021/11/1 10:00", (2021, 11, 1, 10)), ("1983/10/17 00:00:00+00", (1983, 10, 17)),
             ("01.03.2014 00:00", (2014, 1, 3)), ("1977-07-04 01:00:00-02:30", (1977, 7, 4, 3, 30)),
             ("05-Mar-77", (1977, 3, 5)), ("05-Mar-75", (2075, 3, 5))]   # dateutil: 2-digit years in [ref-50, ref+50)

    def test_cases(self):
        for text, parts in self.CASES:
            want = (dt.datetime(*parts) - dt.datetime(1970, 1, 1)) / dt.timedelta(days=1)
            got = t1raw.datetime_days(text, ACQ_DAY, 2026)
            self.assertEqual(got, want, text)

    def test_missing(self):
        for text in ("", "NA", "5", "End Date", "June", "31/31/2020", "1500-01-01", "2263-01-01"):
            self.assertTrue(math.isnan(t1raw.datetime_days(text, ACQ_DAY, 2026)), text)


if __name__ == "__main__":
    unittest.main()
