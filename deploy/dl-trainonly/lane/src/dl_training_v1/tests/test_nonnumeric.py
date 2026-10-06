"""CORESET-NONNUMERIC text columns (dltrain.nonnumeric). Every recovery or invariance check is paired with a negative
control that MUST FAIL the same check; the all-row encoder (the acquisition's own staging rank used as a value) is the
must-fail control of the train-only checks.

The raw source is synthetic. T1 is built from it the acquisition's way: text columns replaced by their rank among the
sorted distinct non-NA texts of EVERY source row (a row without a target included), a mostly-numeric column read as
numbers, then tests.test_t1prep.upstream_t1 over the cleaned rows (median imputation of the codes, a numerically keyed
category map with __MISSING__ last, min-max, float32 '%.9g' printing). A is a coreset of the cleaned rows.
"""
import csv
import hashlib
import io
import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from dltrain import jtt1, nonnumeric, rowsel_raw, t1prep, t1raw, trainonly
from tests.test_t1prep import upstream_t1

WORDS = ["delta", "alpha", "Gamma", "beta", "epsilon", "zeta"]       # 'Gamma' sorts first (str order, upper case)
CKEYS = ["lo", "mid", "hi"]
HEADER = ["id", "n1", "c0", "n0", "yy", "n2", "n3"]                    # raw order differs from T1 order
N = 240


def _tmpdir():
    return tempfile.mkdtemp(prefix="nonnumeric-", dir=os.environ.get("NONNUMERIC_TEST_TMP") or os.environ.get("TMPDIR"))


def raw_columns(seed=3):
    """Raw text per column for N cleaned rows plus one row without a target (index N) whose text sorts first."""
    rng = np.random.default_rng(seed)
    txt = [WORDS[int(i)] for i in rng.integers(0, len(WORDS), N)]
    for i in (5, 17, 40):
        txt[i] = ""                       # pandas NA
    txt[23], txt[61] = "NA", "None"       # pandas NA text ('None' since pandas 2)
    txt[77] = "NA "                       # not NA: pandas does not strip before the NA test
    mix = [str(int(v)) for v in rng.integers(1, 11, N)]
    mix[9], mix[33] = "5;5", "7;7"        # a mostly-numeric column: the acquisition read it as numbers
    flag = ["TRUE" if v else "FALSE" for v in rng.random(N) < 0.4]
    n3 = rng.integers(0, 10, N).astype(float)
    n3[rng.random(N) < 0.05] = np.nan
    ck = [CKEYS[int(i)] for i in rng.integers(0, 3, N)]
    ck[11] = ""
    y = rng.normal(size=N) * 3.0 + np.array([WORDS.index(t) if t in WORDS else 0 for t in txt]) + np.nan_to_num(n3) * 0.5
    cols = {"n0": txt + ["Aardvark"], "n1": mix + ["3"], "n2": flag + ["FALSE"],
            "n3": ["" if math.isnan(v) else repr(float(v)) for v in n3] + ["4.0"], "c0": ck + ["aaa"],
            "yy": [repr(float(v)) for v in y] + [""], "id": [str(i) for i in range(N + 1)]}
    return cols


def add_twin(cols):
    """Append a copy of a cleaned row whose n3 equals the n3 median, with n3 missing: T1 cannot tell the two apart."""
    vals = np.asarray([float(v) for v in cols["n3"][:N] if v != ""])
    med = float(np.median(vals))
    assert med.is_integer()
    k = next(i for i in range(N) if cols["n3"][i] != "" and float(cols["n3"][i]) == med)
    out = {c: list(v) for c, v in cols.items()}
    for c in out:
        out[c].append(out[c][k])
    out["n3"][-1] = ""
    out["id"][-1] = str(len(out["id"]) - 1)
    return out, k, len(out["id"]) - 1


def csv_bytes(cols):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(HEADER)
    for i in range(len(cols["id"])):
        w.writerow([cols[c][i] for c in HEADER])
    return buf.getvalue().encode("utf-8")


def staging(cols, rank_rows=None):
    """The acquisition: text -> rank over every source row (or rank_rows), the mostly-numeric column as numbers, the bool
    text as 1/0; cleaned rows = rows with a target. Returns T1, meta (n0..n3 numeric, c0 categorical keyed by rank)."""
    n = len(cols["id"])
    rows = list(range(n)) if rank_rows is None else list(rank_rows)
    na = nonnumeric.is_na
    rank0 = {k: i for i, k in enumerate(sorted({cols["n0"][i] for i in rows if not na(cols["n0"][i])}))}
    rankc = {k: i for i, k in enumerate(sorted({cols["c0"][i] for i in rows if not na(cols["c0"][i])}))}
    clean = [i for i in range(n) if cols["yy"][i] != ""]
    num = np.empty((len(clean), 4))
    for r, i in enumerate(clean):
        t = cols["n0"][i]
        num[r, 0] = math.nan if na(t) else rank0[t]
        num[r, 1] = t1raw._to_float(cols["n1"][i])
        num[r, 2] = 1.0 if cols["n2"][i] == "TRUE" else 0.0
        num[r, 3] = math.nan if cols["n3"][i] == "" else float(cols["n3"][i])
    cat = [[None if na(cols["c0"][i]) else repr(float(rankc[cols["c0"][i]])) for i in clean]]
    y = np.asarray([float(cols["yy"][i]) for i in clean])
    t1, meta = upstream_t1(num, cat, y)
    meta["original_headers"] = list(HEADER)
    meta["target_original_name"] = meta["source_target_column_original"] = "yy"
    meta["acquisition"] = {"acquired_at": "2026-06-27T00:00:00+00:00", "tool": "fixture", "source_target_column_original": "yy"}
    return t1, meta, clean


def coreset(clean, exclude=(), include=(), size=170, seed=7):
    rng = np.random.default_rng(seed)
    pool = [r for r in range(len(clean)) if clean[r] not in exclude and clean[r] not in include]
    pick = list(rng.choice(pool, size=size - len(include), replace=False)) + [clean.index(i) for i in include]
    return [int(p) for p in rng.permutation(pick)]


def train_side(split):
    h = lambda b: hashlib.sha256(b).hexdigest()
    rep = split.get("t1raw") or {}
    tr = {k: v for k, v in rep.items() if k != "categorical"}
    tr["categorical_train"] = {k: c["train"] for k, c in sorted((rep.get("categorical") or {}).items())}
    tr["imputation"] = {k: [c["train_missing"], c["train_median"]] for k, c in sorted(rep["imputation"]["columns"].items())} if rep else {}
    return {"scaler": h(trainonly.scaler_bytes(split)), "train": h(np.asarray(split["train"], "<f8").tobytes()),
            "fitted_train": h(np.asarray(split["fitted"][:len(split["train"])], "<f8").tobytes()),
            "inner": h(np.asarray(split["inner_labels"], np.uint8).tobytes()),
            "jtt1_train_rows": jtt1.train_rows_sha256(split["jtt1"]), "refit_train": h(repr(sorted(tr["categorical_train"].items())).encode())}


def all_row_universe(values, specs, text, readings):
    """MUST-FAIL CONTROL: the acquisition's all-row staging rank kept as the column's value (kind numeric)."""
    V = np.array(values, dtype=np.float64, copy=True)
    out = [dict(s) for s in specs]
    for n, cells in text.items():
        if readings[n] != nonnumeric.RANK:
            continue
        j = next(i for i, s in enumerate(out) if s["name"] == n)
        rank = nonnumeric.staging_rank(cells)
        V[:, out[j]["index"]] = [math.nan if nonnumeric.is_na(t) else float(rank[t]) for t in cells]
        out[j] = {"index": out[j]["index"], "name": n, "source_name": out[j]["source_name"], "kind": "numeric",
                  "lo": 0.0, "hi": float(max(len(rank) - 1, 1)), "impute_value": None, "derive": None}
    return V, out


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir = _tmpdir()
        self.cols = raw_columns()
        self.t1, self.meta, self.clean = staging(self.cols)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, cols, name="source.csv"):
        p = os.path.join(self.dir, name)
        with open(p, "wb") as f:
            f.write(csv_bytes(cols))
        return p

    def A(self, pick):
        return self.t1[pick]


class Columns(Fixture):
    def test_text_columns_are_the_coded_text_not_bool_or_string_keyed(self):
        p = self.write(self.cols)
        specs = t1raw.columns(self.meta, self.t1.shape[1])
        rep = t1raw.read_source(p, self.meta, specs)["report"]
        self.assertEqual(rep["bool_columns"], ["n2"])
        self.assertEqual(sorted(nonnumeric.text_columns(specs, rep)), ["c0", "n0", "n1"])
        s = dict(specs[4], inverse={0: "lo", 1: "mid", 2: t1prep.MISSING_KEY})          # string keys: t1raw's own case
        self.assertFalse(nonnumeric._numeric_keyed(s))

    def test_numbers_missing_from_a_value_keyed_map_are_drift_not_text(self):
        spec = {"index": 0, "name": "v", "kind": "categorical_codes", "inverse": {0: "12.0", 1: "20.0", 2: t1prep.MISSING_KEY}}
        rank = {"index": 1, "name": "r", "kind": "categorical_codes", "inverse": {0: "0.0", 1: "1.0", 2: t1prep.MISSING_KEY}}
        got = nonnumeric.confirm_text([spec, rank], {"v": ["12", "18", "", "20"], "r": ["lo", "hi", "NA"]})
        self.assertEqual(sorted(got), ["r"])
        self.assertFalse(nonnumeric.holds_text(["12", "18.5", "-3", "NA", "", "$30,000"]))
        self.assertTrue(nonnumeric.holds_text(["12", "5;5"]))

    def test_read_text_rows_are_read_source_rows(self):
        p = self.write(self.cols)
        specs = t1raw.columns(self.meta, self.t1.shape[1])
        src = t1raw.read_source(p, self.meta, specs)
        text = nonnumeric.read_text(p, self.meta, specs, src["report"], ["n0", "n1", "c0"])
        self.assertEqual(text["n0"], self.cols["n0"])
        nonnumeric.check_alignment(src["values"], specs, text)
        bad = dict(text, n1=text["n1"][1:] + text["n1"][:1])                            # rows shifted by one
        with self.assertRaises(nonnumeric.NonnumericError) as cm:
            nonnumeric.check_alignment(src["values"], specs, bad)
        self.assertEqual(cm.exception.reason, "text_read_drift")

    def test_blank_and_long_records_are_skipped_like_read_source(self):
        lines = csv_bytes(self.cols).decode().split("\n")
        lines.insert(5, "")                                                          # blank record
        lines.insert(7, '""')                                                        # one empty field: blank too
        lines.insert(9, ",".join(["9"] * (len(HEADER) + 2)))                         # over-long record
        p = os.path.join(self.dir, "odd.csv")
        with open(p, "w") as f:
            f.write("\n".join(lines))
        specs = t1raw.columns(self.meta, self.t1.shape[1])
        src = t1raw.read_source(p, self.meta, specs)
        self.assertEqual((src["report"]["blank_rows"], src["report"]["long_rows_skipped"]), (2, 1))
        text = nonnumeric.read_text(p, self.meta, specs, src["report"], ["n0", "n1", "c0"])
        nonnumeric.check_alignment(src["values"], specs, text)
        self.assertEqual(text["n0"], self.cols["n0"])

    def test_pointer_is_pinned_row_wise_text_hash(self):
        self.assertEqual([nonnumeric.pointer(t) for t in ("alpha", "", "Warner Bros.")],
                         [108398974994383, 161479093644081, 47797824527483])
        self.assertLess(max(nonnumeric.pointer(t) for t in WORDS), 2 ** 48)

    def test_na_text_is_exact_pandas_na(self):
        self.assertTrue(nonnumeric.is_na("NA") and nonnumeric.is_na("None") and nonnumeric.is_na(""))
        self.assertFalse(nonnumeric.is_na("NA "))
        self.assertEqual(nonnumeric.staging_rank(["b", "NA", "a", "NA ", "", "b"]), {"NA ": 0, "a": 1, "b": 2})


class Match(Fixture):
    def run_match(self, cols, pick, meta=None):
        p = self.write(cols)
        return nonnumeric.match_dataset(p, meta or self.meta, self.A(pick))

    def test_staging_rank_reproduces_every_t1_row(self):
        pick = coreset(self.clean)
        r = self.run_match(self.cols, pick)
        self.assertEqual(r["status"], "matched", r.get("reason"))
        self.assertEqual(r["stats"]["matched_rows"], len(pick))
        self.assertEqual(r["text_columns"]["n0"]["reading"], nonnumeric.RANK)
        self.assertEqual(r["text_columns"]["c0"]["reading"], nonnumeric.RANK)
        self.assertEqual(r["text_columns"]["n1"]["reading"], nonnumeric.NUMERIC)      # flipped once, by blame and range
        self.assertEqual(len(r["attempts"]), 2)
        self.assertLessEqual(r["stats"]["residual"]["max_dev_over_tol"], 1.0)

    def test_plain_t1raw_refuses_the_same_source(self):
        p = self.write(self.cols)
        m = t1raw.match_dataset(p, self.meta, self.A(coreset(self.clean)))
        self.assertEqual(m["status"], "unmatched")

    def test_negative_control_appearance_order_codes_cannot_match(self):
        cols = raw_columns()
        t1, meta, clean = staging(cols)
        order = {k: i for i, k in enumerate(dict.fromkeys(t for t in cols["n0"] if not nonnumeric.is_na(t)))}
        sorted_rank = nonnumeric.staging_rank(cols["n0"])
        self.assertNotEqual(order, sorted_rank)
        # a T1 whose n0 codes follow first appearance (another encoder) is not reproduced by the staging rule
        relabel = {sorted_rank[k]: order[k] for k in order}
        specs = t1raw.columns(meta, t1.shape[1])
        s = specs[0]
        d, _ = t1raw.t1_cells(t1, specs)
        codes = np.asarray([relabel.get(int(round(v)), v) if abs(v - round(v)) < 1e-6 else v for v in d[:, 0]])
        t2 = t1.copy()
        t2[:, 0] = np.asarray([float("%.9g" % np.float32(x)) for x in (codes - s["lo"]) / (s["hi"] - s["lo"])])
        p = self.write(cols)
        r = nonnumeric.match_dataset(p, meta, t2[coreset(clean)])
        self.assertEqual(r["status"], "unmatched")

    def test_rank_population_is_every_source_row(self):
        # T1 coded over rows WITH a target only: the no-target row's 'Aardvark' no longer shifts the codes
        t1, meta, clean = staging(self.cols, rank_rows=[i for i in range(len(self.cols["id"])) if self.cols["yy"][i] != ""])
        r = nonnumeric.match_dataset(self.write(self.cols), meta, t1[coreset(clean)])
        self.assertEqual(r["status"], "unmatched")
        self.assertEqual(r["reason"], "rows_unmatched_text_column")

    def test_categorical_numeric_keys_compose_with_the_rank(self):
        specs = t1raw.columns(self.meta, self.t1.shape[1])
        self.assertTrue(nonnumeric._numeric_keyed(specs[4]))
        p = self.write(self.cols)
        src = t1raw.read_source(p, self.meta, specs)
        text = nonnumeric.read_text(p, self.meta, specs, src["report"], ["c0"])
        V = nonnumeric.match_matrix(src["values"], specs, text, {"c0": nonnumeric.RANK})
        inv = specs[4]["inverse"]
        rank = nonnumeric.staging_rank(text["c0"])
        for t, v in zip(text["c0"], V[:, 4]):
            if nonnumeric.is_na(t):
                self.assertEqual(v, t1raw.MISSING_CODE)
            elif float(rank[t]).hex() in {float(k).hex() for k in inv.values() if k != t1prep.MISSING_KEY}:
                self.assertEqual(float(inv[int(v)]), float(rank[t]))
            else:
                self.assertEqual(v, t1raw.UNMAPPED_CODE)                            # 'aaa': only the no-target row

    def test_text_target_and_target_name_conflict_are_refused(self):
        meta = dict(self.meta, target_original_name="n0")
        meta["acquisition"] = dict(self.meta["acquisition"])
        r = self.run_match(self.cols, coreset(self.clean), meta)
        self.assertEqual((r["status"], r["reason"]), ("unmatched", "meta_target_name_conflict"))
        meta["acquisition"]["source_target_column_original"] = "n0"
        r = self.run_match(self.cols, coreset(self.clean), meta)
        self.assertEqual((r["status"], r["reason"]), ("unmatched", "target_nonnumeric"))

    def test_numeric_wrong_target_column_with_a_name_conflict_is_named_so(self):
        meta = dict(self.meta, target_original_name="n3")                            # numbers, but not T1's target
        meta["acquisition"] = dict(self.meta["acquisition"])
        r = self.run_match(self.cols, coreset(self.clean), meta)
        self.assertEqual((r["status"], r["reason"]), ("unmatched", "meta_target_name_conflict"))
        meta["acquisition"]["source_target_column_original"] = "n3"
        r = self.run_match(self.cols, coreset(self.clean), meta)
        self.assertEqual((r["status"], r["reason"]), ("unmatched", "rows_unmatched_target"))


class Ambiguity(Fixture):
    def setUp(self):
        super().setUp()
        self.cols2, self.k, self.twin = add_twin(self.cols)
        self.t2, self.meta2, self.clean2 = staging(self.cols2)
        self.p = self.write(self.cols2)

    def test_without_the_twin_rows_the_match_is_strict(self):
        r = nonnumeric.match_dataset(self.p, self.meta2, self.t2[coreset(self.clean2, exclude=(self.k, self.twin))])
        self.assertEqual(r["status"], "matched")

    def test_both_twins_in_t1_is_set_determined(self):
        r = nonnumeric.match_dataset(self.p, self.meta2, self.t2[coreset(self.clean2, include=(self.k, self.twin))])
        self.assertEqual(r["status"], "matched_set_level")
        self.assertEqual(r["stats"]["ambiguous_rows"], 2)
        self.assertTrue(r["ambiguity"]["tight"] and r["ambiguity"]["t1_rows_identical_within_blocks"])

    def test_negative_control_one_twin_in_t1_is_only_covered(self):
        r = nonnumeric.match_dataset(self.p, self.meta2, self.t2[coreset(self.clean2, include=(self.k,), exclude=(self.twin,))])
        self.assertEqual(r["status"], "matched_covered")
        self.assertFalse(r["ambiguity"]["tight"])
        self.assertEqual(r["ambiguity"]["reason"], "spare_capacity")

    def test_negative_control_three_t1_rows_on_two_raw_rows_is_refused(self):
        pick = coreset(self.clean2, include=(self.k, self.twin))
        pick.append(self.clean2.index(self.k))                                       # one T1 row too many
        r = nonnumeric.match_dataset(self.p, self.meta2, self.t2[pick])
        self.assertEqual(r["status"], "unmatched")

    def test_assignment_respects_capacity(self):
        got = nonnumeric._assign({0: [5, 6], 1: [5], 2: [6, 5]}, {5: 1, 6: 2})
        self.assertEqual(sorted(got), [0, 1, 2])
        self.assertEqual(sum(1 for k in got.values() if k == 5), 1)
        self.assertEqual(len(nonnumeric._assign({0: [5], 1: [5]}, {5: 1})), 1)


class Universe(Fixture):
    def setUp(self):
        super().setUp()
        self.p = self.write(self.cols)
        r = nonnumeric.match_dataset(self.p, self.meta, self.A(coreset(self.clean)))
        self.assertEqual(r["status"], "matched")
        self.readings = {n: c["reading"] for n, c in r["text_columns"].items()}
        self.V, self.specs, _ = nonnumeric.read_universe(self.p, self.meta, self.t1.shape[1], self.readings)
        rng = np.random.default_rng(11)
        rows = [int(i) for i in rng.permutation(self.clean)]
        self.R, self.T = rows[:160], rows[160:]

    def perturbed(self):
        """Outer-test raw text changed: new categories sorting between train texts, text -> missing."""
        cols = {c: list(v) for c, v in self.cols.items()}
        for k, i in enumerate(self.T):
            cols["n0"][i] = "beta-test-only" if k % 3 else ""
            cols["c0"][i] = "kk-test-only" if k % 2 else cols["c0"][i]
        return cols

    def split(self, V, specs):
        rows = self.R + self.T
        return t1raw.prepare(V[rows], specs, list(range(len(self.R))), list(range(len(self.R), len(rows))))

    def test_universe_decodes_to_the_raw_text(self):
        s = next(x for x in self.specs if x["name"] == "n0")
        self.assertEqual(s["kind"], "categorical_codes")
        got = [None if c == t1raw.MISSING_CODE else s["inverse"][int(c)] for c in self.V[:, s["index"]]]
        self.assertEqual(got, [None if nonnumeric.is_na(t) else t for t in self.cols["n0"]])
        n1 = next(x for x in self.specs if x["name"] == "n1")
        self.assertEqual(n1["kind"], "numeric")

    def test_codes_are_row_wise(self):
        cols = {c: list(v) for c, v in self.cols.items()}
        cols["n0"][self.T[0]] = "Aaa-new"                                            # sorts before every text
        V2, _, _ = nonnumeric.read_universe(self.write(cols, "p.csv"), self.meta, self.t1.shape[1], self.readings)
        changed = np.flatnonzero(~((V2 == self.V) | (np.isnan(V2) & np.isnan(self.V))).all(axis=1)).tolist()
        self.assertEqual(changed, [self.T[0]])

    def test_negative_control_all_row_rank_codes_move_other_rows(self):
        cols = {c: list(v) for c, v in self.cols.items()}
        cols["n0"][self.T[0]] = "Aaa-new"
        a = nonnumeric.staging_rank(self.cols["n0"])
        b = nonnumeric.staging_rank(cols["n0"])
        moved = [i for i, t in enumerate(self.cols["n0"]) if not nonnumeric.is_na(t) and i != self.T[0] and a[t] != b[t]]
        self.assertGreater(len(moved), 100)

    def test_train_side_bit_identical_when_test_text_changes(self):
        base = train_side(self.split(self.V, self.specs))
        V2, specs2, _ = nonnumeric.read_universe(self.write(self.perturbed(), "p.csv"), self.meta, self.t1.shape[1], self.readings)
        self.assertTrue(np.array_equal(V2[self.R], self.V[self.R], equal_nan=True))
        self.assertEqual(train_side(self.split(V2, specs2)), base)

    def test_must_fail_control_all_row_encoder_changes_the_train_side(self):
        p = self.p
        specs = t1raw.columns(self.meta, self.t1.shape[1])
        src = t1raw.read_source(p, self.meta, specs)
        text = nonnumeric.read_text(p, self.meta, specs, src["report"], sorted(self.readings))
        Va, sa = all_row_universe(src["values"], specs, text, self.readings)
        base = train_side(self.split(Va, sa))
        p2 = self.write(self.perturbed(), "p.csv")
        src2 = t1raw.read_source(p2, self.meta, specs)
        text2 = nonnumeric.read_text(p2, self.meta, specs, src2["report"], sorted(self.readings))
        Vb, sb = all_row_universe(src2["values"], specs, text2, self.readings)
        got = train_side(self.split(Vb, sb))
        self.assertNotEqual(got, base)
        self.assertNotEqual(got["fitted_train"], base["fitted_train"])

    def test_test_only_text_shares_the_missing_code(self):
        V2, specs2, _ = nonnumeric.read_universe(self.write(self.perturbed(), "p.csv"), self.meta, self.t1.shape[1], self.readings)
        sp = self.split(V2, specs2)
        j = next(s["index"] for s in specs2 if s["name"] == "n0")
        kept = sp["t1raw"]["kept_columns"]
        col = sp["fitted"][:, kept.index(j)]
        ntr = len(self.R)
        cats = sp["t1raw"]["categorical"]["n0"]
        self.assertEqual(cats["test"]["unseen"], sum(1 for k in range(len(self.T)) if k % 3))
        self.assertTrue(np.all(col[ntr:][[k % 3 == 1 or k % 3 == 2 for k in range(len(self.T))]] == cats["train"]["categories"]))

    def test_rowsel_raw_accepts_the_universe_and_its_gate_passes(self):
        prep = rowsel_raw.prepare_universe("fixture0000000000", self.V, self.specs)
        self.assertEqual(prep.report["cleanup"]["no_target"], 1)
        self.assertEqual(prep.report["cleanup"]["unmapped_category_with_target"], 0)
        folds = rowsel_raw.outer_split("fixture0000000000", prep.positions, prep.values, self.specs)
        self.assertEqual(rowsel_raw.split_report(prep, folds)["groups_straddling_folds"], 0)
        fit = ~np.isin(prep.positions, folds[0])
        X, maps = rowsel_raw.sampler_matrix(prep, fit)
        j = next(s["index"] for s in self.specs if s["name"] == "n0")
        fit_texts = sorted({self.cols["n0"][p] for p in prep.positions[fit] if not nonnumeric.is_na(self.cols["n0"][p])})
        for r, p in enumerate(prep.positions.tolist()):
            t = self.cols["n0"][p]
            want = math.nan if nonnumeric.is_na(t) else float(fit_texts.index(t) if t in fit_texts else len(fit_texts))
            self.assertTrue((math.isnan(want) and math.isnan(X[r, j])) or X[r, j] == want)
        rows = self.R + self.T
        vcf = rowsel_raw.value_counterfactual(self.V[rows], self.specs, len(self.R))
        self.assertTrue(vcf["train_side_invariant"], vcf["cases"])

    def test_pointer_collision_is_refused(self):
        specs = t1raw.columns(self.meta, self.t1.shape[1])
        src = t1raw.read_source(self.p, self.meta, specs)
        text = nonnumeric.read_text(self.p, self.meta, specs, src["report"], ["n0"])
        real = nonnumeric.pointer
        try:
            nonnumeric.pointer = lambda t: 7                                           # every text on one code
            with self.assertRaises(nonnumeric.NonnumericError) as cm:
                nonnumeric.universe(src["values"], specs, text, {"n0": nonnumeric.RANK})
        finally:
            nonnumeric.pointer = real
        self.assertEqual(cm.exception.reason, "pointer_collision")

    def test_pinned_readings_must_name_the_same_columns(self):
        with self.assertRaises(nonnumeric.NonnumericError) as cm:
            nonnumeric.read_universe(self.p, self.meta, self.t1.shape[1], {"n0": nonnumeric.RANK})
        self.assertEqual(cm.exception.reason, "text_columns_differ_from_pinned")


if __name__ == "__main__":
    unittest.main()
