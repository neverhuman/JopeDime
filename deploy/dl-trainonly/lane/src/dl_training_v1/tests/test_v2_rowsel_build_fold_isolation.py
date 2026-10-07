"""FOLD-1 held-out-target isolation on the T1 routes (owner ENG:watcher 07:22Z; PR #246 review 5437848458): the T1 v2g route
(tools/v2_rowsel_build.build_dataset_v2) and the G3 catalog append (tools/v2_g3_append_build.build_one, T1 split v2).

A context's eligibility, its group target-tie admission and its LINEAR-DETERMINED veto may read y only on ITS OWN outer-train
rows. Perturbing only fold k's outer-test target values (every feature and every other target unchanged) must leave context
k's decision and train side byte-identical. Other folds may legitimately change: fold k's outer-test rows are their training rows.

  EndToEnd   every fold k in turn, with a perturbation that FLIPS another fold's veto (linear and group-tie bases)
  Controls   MUST FAIL the same comparison: an injected cross-fold LINEAR aggregation, an injected all-rows tie guard, and a
             change to one of context k's OWN outer-train targets
"""
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dltrain import jtt1, target_linear as tl
from tests import test_v2_rowsel_build as tb
from tests import test_v2_rowsel_build_split_v2g as tg

rb = tb.rb
H, M, Y = tb.H, tg.M, 5


def _load_g3():
    import importlib.util
    spec = importlib.util.spec_from_file_location("v2_g3_append_build", Path(rb.__file__).with_name("v2_g3_append_build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g3 = _load_g3()


def linear_y(m):
    return 0.4 * m[:, 0] + 0.2 * m[:, 2] - 0.3 * m[:, 3] + 0.1


def group_y(m):
    return np.round(np.sin(7.0 * m[:, tg.GCOL]) ** 2, 6)


def heldout_only(m, rows, values):
    """m with ONLY the target of `rows` (one fold's outer-test rows) replaced; every feature and every other target unchanged."""
    w = np.array(m, copy=True)
    rows = np.asarray(sorted(rows), dtype=np.int64)
    w[rows, Y] = values
    w = tb._r9(w)
    other = np.ones(len(m), dtype=bool); other[rows] = False
    assert np.array_equal(np.delete(w, Y, 1), np.delete(m, Y, 1)) and np.array_equal(w[other, Y], m[other, Y])
    return w


def noise(n, seed):
    return np.random.default_rng(seed).uniform(0.0, 1.0, n)


def side(d, cj):
    enc = gzip.decompress((d / "encoder.jtt1.gz").read_bytes())
    tea = gzip.decompress((d / "teacher.jtt1.gz").read_bytes())
    return {"view_id": cj["view_id"], "n_train": cj["n_train"], "encoder_train": jtt1.train_rows_sha256(enc),
            "teacher_train": jtt1.train_rows_sha256(tea), "a0": (d / "a0.f32").read_bytes(),
            "target_linear": cj.get("target_linear"), "group_target_tie": cj.get("group_target_tie")}


def snapshot(contexts, refused, shard, k):
    """Context k's decision and train side (written, or refused with its reason)."""
    r = {c["fold"]: c for c in refused or []}.get(k)
    if r is not None:
        return {"fold": k, "written": False, "refused": r["refused"], "target_tie": r.get("target_tie"),
                "ols_rel_rmse": (r.get("target_linear") or {}).get("ols_rel_rmse", r.get("ols_rel_rmse"))}
    c = {c["fold"]: c for c in contexts or []}.get(k)
    if c is None:
        return {"fold": k, "evaluated": False}
    d = shard / c["view_id"]
    return {"fold": k, "written": True, **side(d, json.loads((d / "context.json").read_bytes()))}


class T1(unittest.TestCase):
    """tools/v2_rowsel_build.build_dataset_v2 on the T1 v2g route (fixture of test_v2_rowsel_build_split_v2g)."""
    n = 0

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="rowsel-fold-iso-", dir=os.environ.get("TMPDIR")))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def build(self, m):
        T1.n += 1
        shard = self.dir / f"s{T1.n}"; shard.mkdir()
        ds = tb.dataset(m, folds=tg.t2.REGISTRY_FOLDS)
        rec = rb.build_dataset_v2(ds, tb.items(H, "clean", folds=range(4)), shard, tb.CODES, dict(tb.REFS), budget=tb.BUDGET,
                                  profile=tg.profile(),
                                  extra={"registry_grouping": {"registry_grouped_by_feature_column": True, "registry_group_columns": [tg.GCOL]},
                                         "group_annotation": {"group_column": tg.SPEC, "group_source": "fixture"}})
        refused = rec.get("refused_contexts") or ([c for c in rec.get("contexts") or [] if c.get("refused")] if rec["status"] != "done" else [])
        contexts = rec.get("contexts") if rec["status"] == "done" else []
        return rec, (lambda k: snapshot(contexts, refused, shard, k))

    def base(self, kind, j):
        m = np.array(M, copy=True)
        m[:, Y] = linear_y(m) if kind == "linear" else group_y(m)
        m = tb._r9(m)
        folds = tg.folds_of(m)
        return heldout_only(m, folds[j], noise(len(folds[j]), 11)), folds

    def flips(self, snap0, snap1, k):
        return [f for f in range(5) if f != k and snap0(f).get("refused") != snap1(f).get("refused")]

    def test_linear_veto_of_another_fold_never_changes_this_context(self):
        m, folds = self.base("linear", 1)
        rec, snap = self.build(m)
        self.assertEqual([(c["fold"], c["refused"]) for c in rec["refused_contexts"]], [(1, tl.REFUSE)], rec.get("detail"))
        for k in range(5):
            with self.subTest(fold=k):
                rows = folds[k]
                vals = linear_y(m)[rows] if k == 1 else noise(len(rows), 20 + k)
                rec1, snap1 = self.build(heldout_only(m, rows, vals))
                self.assertEqual(rec1["split"]["sha256"], rec["split"]["sha256"])        # the split never sees y
                self.assertEqual(snap1(k), snap(k))
                self.assertTrue(self.flips(snap, snap1, k), "the perturbation must flip another fold's veto")

    def test_group_tie_of_another_fold_never_changes_this_context(self):
        m, folds = self.base("group", 2)
        rec, snap = self.build(m)
        self.assertEqual([(c["fold"], c["refused"]) for c in rec["refused_contexts"]], [(2, "group_column_tied_to_target")])
        for k in (2, 0):
            with self.subTest(fold=k):
                rows = folds[k]
                vals = group_y(m)[rows] if k == 2 else noise(len(rows), 30 + k)
                rec1, snap1 = self.build(heldout_only(m, rows, vals))
                self.assertEqual(snap1(k), snap(k))
                self.assertTrue(self.flips(snap, snap1, k))

    def test_the_tie_guard_reads_only_this_contexts_outer_train_targets(self):
        m, folds = self.base("group", 2)
        prep = tg.prep_of(m)
        seen, real = [], rb.rowsel_split_v2_groups.target_tied_group

        def spy(gv, y):
            seen.append(np.asarray(y, dtype=np.float64).copy())
            return real(gv, y)
        with mock.patch.object(rb.rowsel_split_v2_groups, "target_tied_group", spy):
            self.build(m)
        self.assertEqual(len(seen), 5)
        for k, y in enumerate(seen):
            test = {int(p) for p in folds[k]}
            want = prep.values[prep.rows_of([p for p in prep.positions.tolist() if p not in test]), -1]
            self.assertTrue(np.array_equal(y, want), k)

    def test_control_an_injected_cross_fold_linear_veto_is_detected(self):
        """MUST FAIL the isolation comparison: any context's REFUSE refuses every context (the pre-FOLD-1 aggregation)."""
        m, folds = self.base("linear", 1)
        real = rb.context_refusal

        def injected(mm):
            rec, _ = self.build(mm)
            anyr = any(c.get("refused") == tl.REFUSE for c in (rec.get("refused_contexts") or []) + (rec.get("contexts") or []))
            with mock.patch.object(rb, "context_refusal", lambda tlg: tl.REFUSE if anyr else real(tlg)):
                return self.build(mm)[1]
        self.assertNotEqual(injected(m)(0), injected(heldout_only(m, folds[0], noise(len(folds[0]), 40)))(0))

    def test_control_an_injected_all_rows_tie_guard_is_detected(self):
        """MUST FAIL the isolation comparison: the tie read over every row's y (the pre-FOLD-1 dataset-level guard)."""
        m, folds = self.base("group", 2)

        def all_rows(prep, fo, f):
            return rb.rowsel_split_v2_groups.target_tied_group(rb.t1_group_values(prep.values, tg.GCOL), prep.values[:, -1])
        prof = tg.profile()
        with mock.patch.object(tg, "profile", lambda *a, **k: dict(prof, fold_target_tie_fn=all_rows)):
            s0 = self.build(m)[1](2)
            s1 = self.build(heldout_only(m, folds[2], group_y(m)[folds[2]]))[1](2)
        self.assertNotEqual(s0, s1)

    def test_control_this_contexts_own_train_target_is_detected(self):
        """MUST FAIL: one of context 0's OWN outer-train targets (a fold-1 row) changes context 0."""
        m, folds = self.base("linear", 1)
        rows = folds[1][:1]
        self.assertNotEqual(self.build(m)[1](0), self.build(heldout_only(m, rows, m[rows, Y] + 0.25))[1](0))


class G3(unittest.TestCase):
    """tools/v2_g3_append_build.build_one (catalog-only datasets, T1 split v2): LINEAR-DETERMINED per context."""
    n = 0

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="g3-fold-iso-", dir=os.environ.get("TMPDIR")))
        self.blobs = self.dir / "blobs"; self.blobs.mkdir()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def item(self, m):
        ls = tb.lines_of(m)
        bodies = {"meta": json.dumps(tb.META).encode(), "train": b"\n".join(ls[:tb.NTR]) + b"\n", "test": b"\n".join(ls[tb.NTR:]) + b"\n"}
        files = {}
        for k, b in bodies.items():
            h = hashlib.sha256(b).hexdigest()
            p = self.blobs / h
            if not p.exists():
                p.write_bytes(b)
            files[k] = {"sha256": h}
        return {"dataset_hash": H, "catalog_files": files, "shape": {"columns": m.shape[1]}, "catalog_row_sha256": "c" * 64}

    def build(self, m):
        G3.n += 1
        shard = self.dir / f"s{G3.n}"; shard.mkdir()
        rec = g3.build_one(self.item(m), self.blobs, "train_fit", tb.CODES, shard)
        return rec, (lambda k: snapshot(rec.get("contexts") if rec["status"] == "done" else [], rec.get("refused_contexts"), shard, k))

    def folds(self, m):
        U = rb.t1_universe(H, *[b"\n".join(tb.lines_of(m)[a:b]) + b"\n" for a, b in ((0, tb.NTR), (tb.NTR, len(m)))], m.shape[1])
        prep = rb.t1_prep(U)
        return rb.split_v2_t1(H, prep.positions, prep.values, prep.specs)

    def base(self):
        m = np.array(M, copy=True)
        m[:, Y] = linear_y(m)
        m = tb._r9(m)
        folds = self.folds(m)
        return heldout_only(m, folds[1], noise(len(folds[1]), 51)), folds

    def test_linear_veto_of_another_fold_never_changes_this_context(self):
        m, folds = self.base()
        rec, snap = self.build(m)
        self.assertEqual((rec["status"], [(c["fold"], c["refused"]) for c in rec["refused_contexts"]]), ("done", [(1, tl.REFUSE)]))
        for k in range(5):
            with self.subTest(fold=k):
                rows = folds[k]
                vals = linear_y(m)[rows] if k == 1 else noise(len(rows), 60 + k)
                _, snap1 = self.build(heldout_only(m, rows, vals))
                self.assertEqual(snap1(k), snap(k))
                self.assertTrue([f for f in range(5) if f != k and snap1(f).get("refused") != snap(f).get("refused")])

    def test_control_an_injected_cross_fold_linear_veto_is_detected(self):
        """MUST FAIL the isolation comparison: the pre-FOLD-1 G3 rule (a fold's REFUSE fails the whole dataset)."""
        m, folds = self.base()
        real = g3.B.context_refusal                                     # the builder module the G3 tool imported

        def injected(mm):
            rec, _ = self.build(mm)
            anyr = bool(rec.get("refused_contexts"))
            with mock.patch.object(g3.B, "context_refusal", lambda tlg: tl.REFUSE if anyr else real(tlg)):
                return self.build(mm)[1]
        self.assertNotEqual(injected(m)(0), injected(heldout_only(m, folds[0], noise(len(folds[0]), 70)))(0))

    def test_control_this_contexts_own_train_target_is_detected(self):
        m, folds = self.base()
        rows = folds[1][:1]
        self.assertNotEqual(self.build(m)[1](0), self.build(heldout_only(m, rows, m[rows, Y] + 0.25))[1](0))


if __name__ == "__main__":
    unittest.main()
