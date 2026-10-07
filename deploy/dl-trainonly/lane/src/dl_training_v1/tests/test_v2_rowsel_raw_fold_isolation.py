"""Held-out-target isolation of every raw-route context (PR #246 review 5437848458, owner ENG:watcher 05:20Z).

A context's eligibility, its group target-tie admission and its LINEAR-DETERMINED veto may read y only on ITS OWN outer-train
rows. Perturbing only fold k's held-out targets (its outer-test y; every feature and every other y unchanged) must therefore
leave context k byte-identical: written or refused for the same reason, the same selection/identity, the same train-side
bytes and the same tie/linear reports. Other folds may legitimately change: fold k's held-out rows are their training rows.

  EndToEnd   tools/v2_rowsel_raw.run_dataset on rowsel-split-v2-groups, every fold k in turn, with a perturbation chosen to
             FLIP another fold's veto (the review's counterexample: fold 1's linear veto used to suppress the whole dataset)
  Patterns   the per-context decision path (split, v2g_fold_target_tie, build_fold train side, target_linear) for every fold
             and several held-out-y patterns: noise, constant, exactly linear, group function, scaled/sign-flipped
  Controls   MUST FAIL: the same comparison detects an injected cross-fold linear veto (the pre-fix aggregation) and an
             injected all-rows target-tie guard (the pre-fix dataset-level admission)
"""
import gzip
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dltrain import jtt1, rowsel_raw, t1raw, target_linear as tl
from tests import test_v2_rowsel_raw_v2g as v2g

tool, traw = v2g.tool, v2g.traw
H = v2g.H
CODES = v2g.CODES
GROUP_COL, Y_COL = 3, 4


def folds_of(V):
    prep = rowsel_raw.prepare_universe(H, V, v2g.SPECS)
    split_fn, _ = tool.v2g_functions(v2g.entry())
    return prep, split_fn(H, prep.positions, prep.values, prep.specs)


def group_function(V):
    c = np.asarray(V[:, GROUP_COL], dtype=np.float64)
    return np.where(c == t1raw.MISSING_CODE, -4.0, 1.5 * np.nan_to_num(c, nan=-3.0) + 0.25)


def linear_target(V):
    return 2.0 * V[:, 0] + 0.01 * V[:, 1] - 0.5 * V[:, 2] + 3.0


def base_linear(j=1):
    """y exactly linear in the features except fold j's held-out rows: only context j's own train rows are linear."""
    V = v2g.linear_values()
    _, folds = folds_of(V)
    p = np.asarray(folds[j])
    V[p, Y_COL] = np.random.default_rng(11).normal(0.0, 10.0, len(p))
    return V


def base_tie(j=2):
    """y a function of the group column except fold j's held-out groups: only context j's own train rows are tied."""
    V = np.array(v2g.VALUES, copy=True)
    V[:, Y_COL] = group_function(V)
    _, folds = folds_of(V)
    p = np.asarray(folds[j])
    V[p, Y_COL] = np.random.default_rng(12).normal(0.0, 10.0, len(p))
    return V


PATTERNS = {
    "noise": lambda V, p: np.random.default_rng(13).normal(0.0, 50.0, len(p)),
    "constant": lambda V, p: np.full(len(p), 7.0),
    "linear": lambda V, p: linear_target(np.nan_to_num(V))[p],
    "group_function": lambda V, p: group_function(V)[p],
    "scaled_sign_flip": lambda V, p: -1000.0 * V[p, Y_COL] + 3.0,
}


def heldout_only(V, k, pattern):
    """V with ONLY fold k's outer-test target VALUES replaced (finite: a missing target is a universe-cleanup row, not a
    held-out value); every feature and every other target unchanged."""
    _, folds = folds_of(V)
    p = np.asarray(folds[k])
    W = np.array(V, copy=True)
    W[p, Y_COL] = PATTERNS[pattern](V, p)
    assert np.isfinite(W[p, Y_COL]).all()
    other = np.ones(len(V), dtype=bool); other[p] = False
    assert np.array_equal(np.delete(W, Y_COL, 1), np.delete(V, Y_COL, 1), equal_nan=True)      # no feature changed
    assert np.array_equal(W[other, Y_COL], V[other, Y_COL], equal_nan=True)                  # no other target changed
    return W


def train_side(r, files):
    enc = gzip.decompress(files["encoder.jtt1.gz"])
    tea = gzip.decompress(files["teacher.jtt1.gz"])
    return {"identity": r["identity"], "view_id": r["view_id"], "encoder_train": jtt1.train_rows_sha256(enc),
            "teacher_train": jtt1.train_rows_sha256(tea), "a0": files["a0.f32"], "target_linear": tl.check_jtt1(enc)}


def snapshot(rec, shard, k):
    """Context k's decision and train side from a run_dataset record (written, or refused for a reason)."""
    every = {c["fold"]: c for c in (rec.get("contexts") or []) + (rec.get("refused_contexts") or [])}
    c = every.get(k)
    if c is None:
        return {"fold": k, "evaluated": False, "dataset_reason": rec.get("reason")}
    if c.get("refused") or rec["status"] != "done":
        return {"fold": k, "written": False, "refused": c.get("refused") or rec.get("reason"), "target_tie": c.get("target_tie"),
                "target_linear": c.get("target_linear"), "view_id": c.get("view_id"), "identity": c.get("identity")}
    d = shard / c["view_id"]
    cj = json.loads((d / "context.json").read_bytes())
    enc = gzip.decompress((d / "encoder.jtt1.gz").read_bytes())
    tea = gzip.decompress((d / "teacher.jtt1.gz").read_bytes())
    rs = cj["rowsel"]
    return {"fold": k, "written": True, "view_id": c["view_id"], "identity": rs["identity"], "n_train": cj["n_train"],
            "outer_train_indices_sha256": cj["outer_train_indices_sha256"], "scaler_sha256": cj["scaler_sha256"],
            "selected_positions_sha256": rs["selected_positions_sha256"], "train_fit_identity": rs["train_fit_identity"],
            "target_linear": cj["target_linear"], "encoder_train": jtt1.train_rows_sha256(enc),
            "teacher_train": jtt1.train_rows_sha256(tea), "a0": (d / "a0.f32").read_bytes()}


class Base(v2g.Base):
    n = 0

    def run_v(self, V):
        Base.n += 1
        sd = self.dir / f"run-{Base.n}"; sd.mkdir()
        return self.run_ds(V=V, shard=sd), sd

    def assert_isolated(self, V, k, pattern):
        rec0, sd0 = self.run_v(V)
        rec1, sd1 = self.run_v(heldout_only(V, k, pattern))
        self.assertEqual(rec0["split"]["sha256"], rec1["split"]["sha256"])        # the split never sees y
        return snapshot(rec0, sd0, k), snapshot(rec1, sd1, k), (rec0, rec1)


class EndToEnd(Base):
    def test_linear_veto_of_another_fold_never_changes_this_context(self):
        """Every fold k in turn; the perturbation flips another fold's LINEAR-DETERMINED veto (k == 1 restores exact
        linearity so folds 0, 2-4 become linear; k != 1 makes fold 1's train rows non-linear)."""
        V = base_linear(1)
        rec, sd = self.run_v(V)
        self.assertEqual([(c["fold"], c["refused"]) for c in rec["refused_contexts"]], [(1, tl.REFUSE)])
        self.assertEqual(sorted(c["fold"] for c in rec["contexts"]), [0, 2, 3, 4])
        base = {k: snapshot(rec, sd, k) for k in range(5)}
        for k in range(5):
            with self.subTest(fold=k):
                rec1, sd1 = self.run_v(heldout_only(V, k, "linear" if k == 1 else "noise"))
                self.assertEqual(rec1["split"]["sha256"], rec["split"]["sha256"])
                self.assertEqual(snapshot(rec1, sd1, k), base[k])
                flipped = [f for f in range(5) if f != k and snapshot(rec1, sd1, f).get("refused") != base[f].get("refused")]
                self.assertTrue(flipped, "the perturbation must flip another fold's veto (else the case has no power)")

    def test_group_tie_of_another_fold_never_changes_this_context(self):
        V = base_tie(2)
        rec, sd = self.run_v(V)
        self.assertEqual([(c["fold"], c["refused"]) for c in rec["refused_contexts"]], [(2, "group_column_tied_to_target")])
        base = {k: snapshot(rec, sd, k) for k in range(5)}
        for k in (2, 0):
            with self.subTest(fold=k):
                rec1, sd1 = self.run_v(heldout_only(V, k, "group_function" if k == 2 else "noise"))
                self.assertEqual(snapshot(rec1, sd1, k), base[k])
                self.assertTrue([f for f in range(5) if f != k and snapshot(rec1, sd1, f).get("refused") != base[f].get("refused")])

    def test_the_tie_guard_reads_only_this_contexts_outer_train_targets(self):
        V = base_tie(2)
        prep, folds = folds_of(V)
        seen, real = [], tool.rowsel_split_v2_groups.target_tied_group

        def spy(gv, y):
            seen.append(np.asarray(y, dtype=np.float64).copy())
            return real(gv, y)
        with mock.patch.object(tool.rowsel_split_v2_groups, "target_tied_group", spy):
            self.run_v(V)
        self.assertEqual(len(seen), 5)                                         # one per context (one candidate column)
        for k, y in enumerate(seen):
            test = set(int(p) for p in folds[k])
            want = prep.values[prep.rows_of([p for p in prep.positions.tolist() if p not in test]), -1]
            self.assertTrue(np.array_equal(y, want), k)


class Patterns(unittest.TestCase):
    """The per-context decision path, every fold x every held-out-y pattern, on both bases."""

    def decide(self, V, k):
        prep, folds = folds_of(V)
        tie = tool.v2g_fold_target_tie(v2g.entry(), prep, folds, k)
        srep = tool.v2g_functions(v2g.entry())[1](prep, folds)
        split_ref = {x: srep[x] for x in ("schema", "sha256", "fold_sizes", "groups", "largest_group", "groups_straddling_folds",
                                          "domain", "rule")}
        universe_ref = {"partition": rowsel_raw.PARTITION, "positions_sha256": prep.report["positions_sha256"]}
        r = rowsel_raw.build_fold(prep, folds, k, universe_ref, split_ref, counterfactual=False)
        files = rowsel_raw.context_files(r)
        side = train_side(r, files)
        return {"split": srep["sha256"], "tie": tie, **side, "refused": tool.context_refusal(side["target_linear"])}

    def test_every_fold_every_pattern(self):
        for name, V in (("linear-but-fold-1", base_linear(1)), ("tied-but-fold-2", base_tie(2))):
            base = {k: self.decide(V, k) for k in range(5)}
            for k in range(5):
                for pattern in PATTERNS:
                    with self.subTest(base=name, fold=k, pattern=pattern):
                        self.assertEqual(self.decide(heldout_only(V, k, pattern), k), base[k])


class Controls(Base):
    def test_an_injected_cross_fold_linear_veto_is_detected(self):
        """MUST FAIL the isolation comparison: the pre-fix rule (any fold's REFUSE refuses every context)."""
        V = base_linear(1)
        real = tool.context_refusal

        def run_injected(W):
            probe, _ = self.run_v(W)
            any_refused = any(c.get("refused") == tl.REFUSE for c in probe.get("refused_contexts") or probe.get("contexts") or [])
            with mock.patch.object(tool, "context_refusal", lambda tlg: tl.REFUSE if any_refused else real(tlg)):
                return self.run_v(W)
        rec0, sd0 = run_injected(V)
        rec1, sd1 = run_injected(heldout_only(V, 0, "noise"))
        self.assertNotEqual(snapshot(rec0, sd0, 0), snapshot(rec1, sd1, 0))   # detected: fold 0 changed with ITS held-out y

    def test_an_injected_all_rows_tie_guard_is_detected(self):
        """MUST FAIL the isolation comparison: the pre-fix dataset-level target-tie guard over every row's y."""
        V = base_tie(2)

        def all_rows(e, prep, folds, f):
            cands, names = tool.v2g_candidates(e)
            return {str(c): tool.rowsel_split_v2_groups.target_tied_group(tool.raw_group_values(prep.values, prep.specs, c, n),
                                                                           prep.values[:, -1]) for c, n in zip(cands, names)}
        with mock.patch.object(tool, "v2g_fold_target_tie", all_rows):
            s0, s1, _ = self.assert_isolated(V, 2, "group_function")
        self.assertNotEqual(s0, s1)

    def test_the_comparison_has_power_on_this_contexts_own_train_targets(self):
        """MUST FAIL: changing one of context 0's OWN outer-train targets (a held-out row of fold 1) changes context 0."""
        V = base_linear(1)
        prep, folds = folds_of(V)
        p = int(folds[1][0])
        W = np.array(V, copy=True); W[p, Y_COL] += 1.0
        rec0, sd0 = self.run_v(V)
        rec1, sd1 = self.run_v(W)
        self.assertNotEqual(snapshot(rec0, sd0, 0), snapshot(rec1, sd1, 0))


if __name__ == "__main__":
    unittest.main()
