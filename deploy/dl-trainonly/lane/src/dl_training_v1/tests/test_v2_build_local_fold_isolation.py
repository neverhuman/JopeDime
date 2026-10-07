"""FOLD-1 held-out-target isolation on the clean non-raw v2g routes (owner ENG:watcher 07:22Z; PR #246 review 5437848458):
tools/v2_build_local.compute_dataset for L7b (T1 table) and L7a (raw rows, tools/v2_build_l7a.py inputs).

Each context's group target-tie admission (fold_target_tie over its own outer-train rows R) and its LINEAR-DETERMINED veto
(context_refusal of its own report) may read y only on ITS OWN outer-train rows. Perturbing only fold k's outer-test target
values must leave context k's decision and train side byte-identical; other folds may change (those rows train them).

  EndToEnd   every fold k in turn with a perturbation that FLIPS another fold's veto: L7b linear and group-tie bases, L7a linear
  Controls   MUST FAIL: an injected cross-fold LINEAR aggregation, an injected all-rows tie guard, and a change to one of
             context k's OWN outer-train targets
"""
import gzip
import json
import unittest
from unittest import mock

import numpy as np

from dltrain import jtt1, t1raw, target_linear as tl
from tests import test_v2_nonraw_split_v2g as tv
from tests import test_v2_nonraw_split_v2 as base

tool, l7a = tv.tool, tv.l7a
H, N, GCOL = tv.H, tv.N, tv.GCOL


REASON_OF = {str: "group_column_target_tied", float: tl.REFUSE}


def snapshot(res, k):
    """Context k's decision and train side from compute_dataset's result, or from its SplitV2Error when every context was
    refused (its detail maps each fold to its own value: the tie reason or the OLS rel-RMSE). The view_id is NOT compared:
    the L7 view identity binds the data object's all-row values_sha256 (provenance), so a changed object has a new view_id;
    the decision, the outer-train indices and every train-side byte must not change."""
    if isinstance(res, Exception) and hasattr(res, "reason"):
        v = json.loads(res.detail).get(str(k))
        return {"fold": k, "written": False, "refused": REASON_OF[type(v)], "value": v}
    r = {c["fold"]: c for c in res.get("refused_contexts") or []}.get(k)
    if r is not None:
        return {"fold": k, "written": False, "refused": r["refused"],
                "value": (r.get("target_linear") or {}).get("ols_rel_rmse") if r["refused"] == tl.REFUSE else r.get("target_tie")}
    hit = [(vid, files, crec) for vid, files, crec in res["results"] if crec["fold"] == k]
    if not hit:
        return {"fold": k, "evaluated": False}
    vid, files, crec = hit[0]
    enc, tea = gzip.decompress(files["encoder.jtt1.gz"]), gzip.decompress(files["teacher.jtt1.gz"])
    return {"fold": k, "written": True, "n_train": crec["n_train"], "outer_train_indices_sha256": crec["outer_train_indices_sha256"],
            "encoder_train": jtt1.train_rows_sha256(enc), "teacher_train": jtt1.train_rows_sha256(tea), "a0": files["a0.f32"],
            "target_linear": crec["target_linear"], "group_target_tie": crec.get("group_target_tie")}


def refusal(s):
    return s.get("refused")


class L7b(unittest.TestCase):
    """The T1-table route: the contexts read A (its target column is the context's y)."""

    def run_(self, M):
        e = tv.g_entry()
        try:
            return tool.compute_dataset(e, tv.t1_inputs(M, e), "L7b", set(), {"code": "x"}, version=tool.SPLIT_V2G)
        except tool.SplitV2Error as err:
            return err

    def folds(self):
        return tool.v2g_split(H, tv.A, tv.SPECS_T1, "t1", tv.GV, tv.SPEC)[0]

    def heldout_only(self, M, rows, values):
        W = np.array(M, copy=True)
        W[np.asarray(rows), -1] = values
        other = np.ones(N, dtype=bool); other[np.asarray(rows)] = False
        assert np.array_equal(W[:, :-1], M[:, :-1]) and np.array_equal(W[other, -1], M[other, -1])
        return W

    def lin(self, M):
        return 0.4 * M[:, 0] + 0.2 * M[:, 1] - 0.3 * M[:, 2] + 0.1

    def grp(self, M):
        return np.round(np.sin(3.0 * M[:, GCOL]) ** 2 + 0.1, 6)

    def base(self, kind, j):
        M = np.array(tv.A, copy=True)
        M[:, -1] = self.lin(M) if kind == "linear" else self.grp(M)
        folds = self.folds()
        return self.heldout_only(M, folds[j], np.random.default_rng(11).normal(size=len(folds[j]))), folds

    def test_linear_veto_of_another_fold_never_changes_this_context(self):
        M, folds = self.base("linear", 1)
        res = self.run_(M)
        self.assertEqual([(c["fold"], c["refused"]) for c in res["refused_contexts"]], [(1, tl.REFUSE)])
        for k in range(5):
            with self.subTest(fold=k):
                vals = self.lin(M)[folds[k]] if k == 1 else np.random.default_rng(20 + k).normal(size=len(folds[k]))
                res1 = self.run_(self.heldout_only(M, folds[k], vals))
                self.assertEqual(snapshot(res1, k), snapshot(res, k))
                self.assertTrue([f for f in range(5) if f != k and refusal(snapshot(res1, f)) != refusal(snapshot(res, f))])

    def test_group_tie_of_another_fold_never_changes_this_context(self):
        M, folds = self.base("group", 2)
        res = self.run_(M)
        self.assertEqual([(c["fold"], c["refused"]) for c in res["refused_contexts"]], [(2, "group_column_target_tied")])
        for k in (2, 0):
            with self.subTest(fold=k):
                vals = self.grp(M)[folds[k]] if k == 2 else np.random.default_rng(30 + k).normal(size=len(folds[k]))
                res1 = self.run_(self.heldout_only(M, folds[k], vals))
                self.assertEqual(snapshot(res1, k), snapshot(res, k))
                self.assertTrue([f for f in range(5) if f != k and refusal(snapshot(res1, f)) != refusal(snapshot(res, f))])

    def test_the_tie_guard_reads_only_this_contexts_outer_train_targets(self):
        M, folds = self.base("group", 2)
        seen, real = [], tool.rowsel_split_v2_groups.target_tied_group

        def spy(gv, y):
            seen.append(np.asarray(y, dtype=np.float64).copy())
            return real(gv, y)
        with mock.patch.object(tool.rowsel_split_v2_groups, "target_tied_group", spy):
            self.run_(M)
        self.assertEqual(len(seen), 5)
        for k, y in enumerate(seen):
            R, _ = tool.fold_rows(folds, k, N)
            self.assertTrue(np.array_equal(y, M[R, -1]), k)

    def test_control_an_injected_cross_fold_linear_veto_is_detected(self):
        """MUST FAIL the isolation comparison: any context's REFUSE refuses every context (the pre-FOLD-1 aggregation)."""
        M, folds = self.base("linear", 1)
        real = tool.context_refusal

        def injected(W):
            res = self.run_(W)
            anyr = not isinstance(res, tool.SplitV2Error) and bool(res.get("refused_contexts"))
            with mock.patch.object(tool, "context_refusal", lambda tlg: tl.REFUSE if anyr else real(tlg)):
                return snapshot(self.run_(W), 0)
        self.assertNotEqual(injected(M), injected(self.heldout_only(M, folds[0], np.random.default_rng(40).normal(size=len(folds[0])))))

    def test_control_an_injected_all_rows_tie_guard_is_detected(self):
        """MUST FAIL the isolation comparison: the tie over every row's y (the pre-FOLD-1 dataset-level guard)."""
        M, folds = self.base("group", 2)
        with mock.patch.object(tool, "fold_target_tie", lambda gv, y, R: tool.rowsel_split_v2_groups.target_tied_group(gv, y)):
            s0 = snapshot(self.run_(M), 2)
            s1 = snapshot(self.run_(self.heldout_only(M, folds[2], self.grp(M)[folds[2]])), 2)
        self.assertNotEqual(s0, s1)

    def test_control_this_contexts_own_train_target_is_detected(self):
        M, folds = self.base("linear", 1)
        rows = folds[1][:1]
        self.assertNotEqual(snapshot(self.run_(M), 0), snapshot(self.run_(self.heldout_only(M, rows, M[rows, -1] + 0.25)), 0))


class L7a(unittest.TestCase):
    """The raw-rows route: contexts read the raw values (their y); group values and the tie's y come from the T1 table A."""

    def data(self, y):
        num = np.nan_to_num(tv.NUM)                                       # no missing cells: imputation stays exact
        A, meta = base.t1_table(num, tv.CAT, y)
        specs = t1raw.columns(meta, A.shape[1])
        return A, meta, specs, base.raw_values(num, tv.CAT, y, specs)

    def run_(self, y):
        A, meta, specs, V = self.data(y)
        e = tv.g_entry()
        inp = l7a.l7a_inputs(e, V, specs, "a" * 64, "m" * 64, base.COV, "incoming", {"sha256": "r" * 64}, {}, base.REMATCH)
        inp = l7a.local.attach_groups(inp, e, A)
        try:
            return l7a.local.compute_dataset(e, inp, "L7a", set(), {"code": "x"}, version=l7a.local.SPLIT_V2G)
        except l7a.local.SplitV2Error as err:
            return err

    def lin(self):
        num = np.nan_to_num(tv.NUM)
        return 2.0 * num[:, 0] + 0.01 * num[:, 1] - 0.3 * num[:, 2] + 5.0

    def folds(self, y):
        A, meta, specs, V = self.data(y)
        gv = l7a.local.t1_group_values(A, GCOL)
        return l7a.local.v2g_split(H, V, specs, l7a.local.ROUTES_V2G["L7a"]["partition"], gv, tv.SPEC)[0]

    def heldout_only(self, y, rows, values):
        y2 = np.array(y, copy=True); y2[np.asarray(rows)] = values
        A0, _, _, V0 = self.data(y)
        A1, _, _, V1 = self.data(y2)
        assert np.array_equal(A0[:, :-1], A1[:, :-1]) and np.array_equal(V0[:, :-1], V1[:, :-1], equal_nan=True)
        return y2

    def snap(self, res, k):
        return snapshot(res, k)

    def base(self):
        y = self.lin()
        folds = self.folds(y)
        return self.heldout_only(y, folds[1], np.random.default_rng(51).normal(size=len(folds[1]))), folds

    def test_linear_veto_of_another_fold_never_changes_this_context(self):
        y, folds = self.base()
        res = self.run_(y)
        self.assertEqual([(c["fold"], c["refused"]) for c in res["refused_contexts"]], [(1, tl.REFUSE)])
        for k in range(5):
            with self.subTest(fold=k):
                vals = self.lin()[folds[k]] if k == 1 else np.random.default_rng(60 + k).normal(size=len(folds[k]))
                res1 = self.run_(self.heldout_only(y, folds[k], vals))
                self.assertEqual(self.snap(res1, k), self.snap(res, k))
                self.assertTrue([f for f in range(5) if f != k and refusal(self.snap(res1, f)) != refusal(self.snap(res, f))])

    def test_control_an_injected_cross_fold_linear_veto_is_detected(self):
        y, folds = self.base()
        real = l7a.local.context_refusal

        def injected(yy):
            res = self.run_(yy)
            anyr = not isinstance(res, l7a.local.SplitV2Error) and bool(res.get("refused_contexts"))
            with mock.patch.object(l7a.local, "context_refusal", lambda tlg: tl.REFUSE if anyr else real(tlg)):
                return self.snap(self.run_(yy), 0)
        self.assertNotEqual(injected(y), injected(self.heldout_only(y, folds[0], np.random.default_rng(70).normal(size=len(folds[0])))))

    def test_control_this_contexts_own_train_target_is_detected(self):
        y, folds = self.base()
        rows = folds[1][:1]
        self.assertNotEqual(self.snap(self.run_(y), 0), self.snap(self.run_(self.heldout_only(y, rows, y[rows] + 0.25)), 0))


if __name__ == "__main__":
    unittest.main()
