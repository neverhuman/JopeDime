"""GROUP-UNION-B: tools/v2_rowsel_build.py --split-version rowsel-split-v2-groups (the T1 group datasets on split v2
with annotated groups unioned in). Every invariance check is paired with a control that MUST FAIL it.

  Gates      annotated groups never straddle folds; split v2 (groups ignored) straddles them and the gate refuses it;
             targets never move the split, and a split reading targets (registry port, or target-derived groups) is
             refused; group- and duplicate-preserving outer-test changes leave the split and the train side identical;
             a content-ranked split is refused; a change that merges groups is not a valid group-preserving change
  Cut        cut_id 'trainonly-split-v2g:' + split sha (binding the group spec), replaces_cut = [registry split, the
             as-built v2split cut], comparable_with_replaced false, new '__rowsel2g_' view ids, verify catches tampering
  V2         the v2 route still builds the bytes c0b989b6 (which built the v2split trees) built; None groups == v2
  Evidence   the registry-port evidence: the reproducing column is found, a changed later repeat is not reproduced,
             missing cells are one group as equal empty fields are, a field/value grouping mismatch is refused,
             ambiguous columns are accepted only with identical component structure
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dltrain import rowsel, rowsel_raw
from dltrain import rowsel_raw_split_v2 as v2
from dltrain import rowsel_split_v2_groups as g
from tests import test_v2_rowsel_build as tb
from tests import test_v2_rowsel_build_split_v2 as t2

rb = tb.rb
H = tb.H
GCOL, GNAME = 1, "x1"                                       # fixture column x1: ten levels 0.0 .. 0.9
SPEC = {"index": GCOL, "name": GNAME}
V2_BUILDER = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/code-custody/"
                  "v2_rowsel_build.py.c0b989b638f2037c8b8c498ebc0b8dc329e2502b68f8b7701742a270148c1827")
REPL = {"cut_id": "trainonly-split-v2:" + "9" * 64, "split_sha256": "9" * 64, "tree": "fixture-v2split",
        "view_ids": [f"{H}__r0__f{f}__rowsel2s_fixture" for f in range(5)], "context_sha256s": ["8" * 64] * 5,
        "dataset_record": {"path": "fixture", "sha256": "7" * 64}}
M = t2.M                                                    # full-row and feature-only duplicates included


def profile(index=GCOL, name=GNAME):
    return rb.profile_v2g(index, name, REPL)


def prep_of(m):
    return rb.t1_prep(tb.universe_of(m))


def folds_of(m, index=GCOL):
    p = prep_of(m)
    return profile(index)["split_fn"](H, p.positions, p.values, p.specs)


def groups_of(m, index=GCOL):
    return rb.t1_group_values(prep_of(m).values, index)


def group_preserving_change(m, rows):
    """Outer-test change keeping duplicates and groups: x0, x2, x3 -> 1 - x, the group column shifted by 0.05 (fresh
    values: no original level is k/10 + 0.05), the category code shifted by one."""
    m2 = m.copy()
    rows = np.asarray(sorted(rows))
    for j in (0, 2, 3):
        m2[rows, j] = 1.0 - m2[rows, j]
    m2[rows, GCOL] = m2[rows, GCOL] + 0.05
    m2[rows, 4] = (np.rint(m2[rows, 4] * 3) + 1) % 4 / 3.0
    return tb._r9(m2)


def target_group_split(h, positions, values, specs, folds=5):
    """CONTROL: rows joined by an equal TARGET (a group annotation derived from the target)."""
    tgt = [np.float64(x).tobytes() for x in np.asarray(values)[:, -1]]
    return g.split(h, np.asarray(positions).tolist(), rowsel_raw.feature_bytes(values, specs), tgt, "t1", folds)


def group_ranked_split(h, positions, values, specs, folds=5):
    """CONTROL: groups unioned, components ranked by their group VALUES (content), not by position hashes."""
    pos = np.asarray(positions).tolist()
    gv = rb.t1_group_values(values, GCOL)
    comps = g.components(pos, rowsel_raw.feature_bytes(values, specs), gv)
    of = dict(zip(pos, gv))
    order = sorted(comps, key=lambda c: (hashlib.sha256(b"".join(sorted(of[p] for p in c))).digest(), min(c)))
    sizes, out = [0] * folds, {f: [] for f in range(folds)}
    for c in order:
        f = sizes.index(min(sizes)); out[f] += c; sizes[f] += len(c)
    return {f: sorted(v) for f, v in out.items()}


class Base(unittest.TestCase):
    CODES = tb.CODES

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="rowsel-v2gsplit-", dir=os.environ.get("TMPDIR")))
        self.shard = self.dir / "shard-0"; self.shard.mkdir()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def build(self, m=M, shard=None, prof=None, expect_done=True, **kw):
        ds = tb.dataset(m, folds=t2.REGISTRY_FOLDS)
        its = tb.items(H, "clean", folds=range(4))
        rec = rb.build_dataset_v2(ds, its, shard or self.shard, self.CODES, dict(tb.REFS), budget=tb.BUDGET,
                                  profile=prof or profile(),
                                  extra={"registry_grouping": {"registry_grouped_by_feature_column": True, "registry_group_columns": [GCOL]},
                                         "group_annotation": {"group_column": SPEC, "group_source": "fixture"}}, **kw)
        if expect_done:
            self.assertEqual((rec["status"], len(rec["contexts"])), ("done", 5), rec.get("detail"))
        return ds, rec

    def ctxs(self, rec, shard=None):
        return [json.loads(((shard or self.shard) / c["view_id"] / "context.json").read_bytes()) for c in rec["contexts"]]

    def assert_refused(self, rec, shard, want):
        self.assertEqual((rec["status"], rec["reason"]), ("failed", "split_gate_failed"), rec.get("detail"))
        self.assertIn(want, rec["detail"])
        self.assertEqual([d for d in shard.iterdir() if d.name != "_datasets"], [])


class Gates(Base):
    def test_groups_never_straddle_folds(self):
        _, rec = self.build()
        folds, gv = folds_of(M), groups_of(M)
        pos = prep_of(M).positions.tolist()
        self.assertEqual((rec["split"]["annotation_groups_straddling_folds"], rec["split"]["annotation_groups"]), (0, 10))
        fold_of = {p: f for f, v in folds.items() for p in v}
        for k in set(gv):
            self.assertEqual(len({fold_of[p] for p, x in zip(pos, gv) if x == k}), 1)
        self.assertEqual(rec["split"]["fold_sizes"], [len(folds[f]) for f in range(5)])
        for f, c in enumerate(self.ctxs(rec)):
            self.assertTrue(set(c["rowsel"]["test_positions"]) <= set(folds[f]))
            self.assertFalse(set(c["rowsel"]["train_positions"]) & set(folds[f]))
            self.assertEqual(c["rowsel"]["split"]["annotation_groups_straddling_folds"], 0)

    def test_split_v2_without_groups_straddles_and_is_refused(self):
        """MUST FAIL: the as-built v2 split ignores the annotation."""
        p = prep_of(M)
        self.assertGreater(g.straddling_groups(rb.split_v2_t1(H, p.positions, p.values, p.specs), p.positions.tolist(), groups_of(M)), 0)
        _, rec = self.build(split_fn=rb.split_v2_t1, expect_done=False)
        self.assert_refused(rec, self.shard, "annotation_groups_straddle_folds")

    def test_targets_never_move_the_split(self):
        _, rec = self.build()
        self.assertTrue(rec["split"]["target_check"]["target_invariant"])
        other = self.dir / "shard-y"; other.mkdir()
        _, rec2 = self.build(m=t2.noise_targets(M, range(M.shape[0])), shard=other)
        self.assertEqual(rec2["split"]["sha256"], rec["split"]["sha256"])
        for fn in (t2.registry_split_fn, target_group_split):           # CONTROLS: read the target
            with self.subTest(fn=fn.__name__):
                shard = self.dir / f"shard-{fn.__name__}"; shard.mkdir()
                _, bad = self.build(shard=shard, split_fn=fn, expect_done=False)
                self.assert_refused(bad, shard, "split_reads_target")

    def test_group_and_duplicate_preserving_outer_test_changes_never_move_the_split(self):
        _, rec = self.build()
        self.assertTrue(rec["split"]["group_value_check_fold0"]["invariant"])
        self.assertTrue(rec["split"]["value_check_fold0"]["invariant"])
        folds = folds_of(M)
        pos = prep_of(M).positions.tolist()
        for f in range(5):
            m2 = group_preserving_change(t2.noise_targets(M, folds[f]), folds[f])
            p2 = prep_of(m2)
            self.assertEqual(g.component_partition(pos, rowsel_raw.feature_bytes(p2.values, p2.specs), groups_of(m2)),
                             g.component_partition(pos, rowsel_raw.feature_bytes(prep_of(M).values, prep_of(M).specs), groups_of(M)))
            self.assertEqual(folds_of(m2), folds, f)
        f = 2
        m2 = group_preserving_change(t2.noise_targets(M, folds[f]), folds[f])
        other = self.dir / "shard-x"; other.mkdir()
        _, rec2 = self.build(m=m2, shard=other)
        self.assertEqual(rec2["split"]["sha256"], rec["split"]["sha256"])
        a, b = self.ctxs(rec)[f], self.ctxs(rec2, other)[f]
        for k in ("train_positions", "test_positions"):
            self.assertEqual(a["rowsel"][k], b["rowsel"][k])
        self.assertEqual(a["rowsel"]["view"]["train_fit_identity"], b["rowsel"]["view"]["train_fit_identity"])
        self.assertEqual((a["scaler_sha256"], a["files"]["a0.f32"]), (b["scaler_sha256"], b["files"]["a0.f32"]))
        self.assertNotEqual(a["files"]["encoder.jtt1.gz"], b["files"]["encoder.jtt1.gz"])

    def test_a_content_ranked_split_moves_and_is_refused(self):
        """MUST FAIL: ranking components by group values moves under the same group-preserving change."""
        p = prep_of(M)
        r = rb.group_value_check(group_ranked_split, p, GCOL, 0)
        self.assertTrue(r["valid"]); self.assertFalse(r["invariant"])
        self.assertTrue(rb.group_value_check(profile()["split_fn"], p, GCOL, 0)["invariant"])
        _, rec = self.build(split_fn=group_ranked_split, expect_done=False)
        self.assertEqual((rec["status"], rec["reason"]), ("failed", "split_gate_failed"))
        self.assertIn("split_moves_under_group_preserving_changes", rec["detail"])

    def test_a_group_column_tied_to_the_target_refuses_each_context_on_its_own_train_rows(self):
        """FOLD-1 (owner 07:22Z; supersedes #218 finding 8): a feature column that copies the target (x3 := y) or determines
        it is never refused at dataset level before the split. The split runs (y never reaches it); each context whose
        OWN outer-train rows show the tie is refused (every context here); nothing is written. MUST FAIL on the pre-split
        dataset-level refusal (the split was never computed)."""
        mc = M.copy(); mc[:, 3] = mc[:, 5]
        md = M.copy(); md[:, 5] = tb._r9(np.sin(7.0 * md[:, GCOL]) ** 2)       # y a non-linear function of the group column
        for m, prof, want in ((mc, rb.profile_v2g(3, "x3", REPL), "group_bytes_equal_target_bytes"),
                              (md, profile(), "target_function_of_group")):
            with self.subTest(want=want):
                shard = self.dir / f"shard-{want}"; shard.mkdir()
                calls, real = [], g.split
                with mock.patch.object(g, "split", lambda *a, **k: calls.append(1) or real(*a, **k)):
                    _, rec = self.build(m=m, shard=shard, prof=prof, expect_done=False)
                self.assertTrue(calls)                                          # the split ran
                self.assertEqual((rec["status"], rec["reason"], rec["group_target_tie"]),
                                 ("failed", "group_column_tied_to_target", "per context on its own outer-train rows (fold_target_tie_fn)"), rec.get("detail"))
                prep = prep_of(m)
                folds = prof["split_fn"](H, prep.positions, prep.values, prep.specs)
                built = [f for f in sorted(folds) if folds[f]]
                self.assertEqual(json.loads(rec["detail"]), {str(f): want for f in built})
                for f in built:
                    self.assertEqual(prof["fold_target_tie_fn"](prep, folds, f), want)
                self.assertEqual([d for d in shard.iterdir() if d.name != "_datasets"], [])
        _, ok = self.build()
        self.assertEqual(ok["group_target_tie"], "per context on its own outer-train rows (fold_target_tie_fn)")
        self.assertEqual({c["group_target_tie"]["result"] for c in self.ctxs(ok)}, {"admit"})

    def test_a_change_that_merges_groups_is_not_valid(self):
        """The validity check is the component structure WITH groups: a fold change that moves the fold's group values
        onto a level another fold holds merges groups and must be invalid (so it can never count as invariance)."""
        p = prep_of(M)
        folds = folds_of(M)
        outside = next(x for x in np.unique(p.values[:, GCOL]) if x not in set(p.values[p.rows_of(folds[0]), GCOL].tolist()))
        orig = v2.perturb_fold_values

        def merging(values, specs, rows, kind, salt=v2.CF_SALT):
            V, s = orig(values, specs, rows, kind, salt)
            V[np.asarray(rows), GCOL] = outside
            return V, s
        with mock.patch.object(rb.rowsel_raw_split_v2, "perturb_fold_values", merging):
            r = rb.group_value_check(profile()["split_fn"], p, GCOL, 0)
        self.assertFalse(r["valid"]); self.assertFalse(r["invariant"])


class Cut(Base):
    def test_cut_block_replaces_the_registry_split_and_the_v2split_cut(self):
        ds, rec = self.build()
        sha = g.split_sha256(H, "t1", SPEC, folds_of(M))
        self.assertEqual(rec["split"]["sha256"], sha)
        self.assertNotEqual(sha, g.split_sha256(H, "t1", {"index": GCOL, "name": "renamed"}, folds_of(M)))
        self.assertNotEqual(sha, rb.split_v2_t1_sha256(H, folds_of(M)))
        for c in self.ctxs(rec):
            cut = c["cut"]
            self.assertEqual((cut["cut_id"], cut["schema"], cut["comparable_with_replaced"], cut["group_column"]),
                             ("trainonly-split-v2g:" + sha, "rowsel-split-v2-groups", False, SPEC))
            self.assertEqual(c["identity"]["cut_id"], cut["cut_id"])
            self.assertEqual(c["identity"]["split"], {"schema": "rowsel-split-v2-groups", "sha256": sha, "partition": "t1", "group_column": SPEC})
            reg, old = cut["replaces_cut"]
            self.assertEqual((reg["kind"], reg["split_sha256"], len(reg["epoch0_contexts"])), ("registry-split", ds["split"]["sha256"], 4))
            self.assertEqual((old["cut_id"], old["split_sha256"], old["view_ids"]), (REPL["cut_id"], REPL["split_sha256"], REPL["view_ids"]))
            self.assertTrue(old["kind"].startswith("rowsel-raw-split-v2"))
            self.assertEqual((c["split_version"], c["rowsel"]["split_version"]), ("rowsel-split-v2-groups", "rowsel-split-v2-groups"))
            self.assertEqual(c["group_annotation"]["group_column"], SPEC)
            self.assertEqual(rb.verify_context_dir_v2(self.shard / c["view_id"])["problems"], [])

    def test_new_view_ids(self):
        _, rec = self.build()
        v2dir = self.dir / "shard-v2"; v2dir.mkdir()
        _, rec_v2 = self.build(shard=v2dir, prof=rb.PROFILE_V2)
        for c, c2 in zip(self.ctxs(rec), self.ctxs(rec_v2, v2dir)):
            self.assertIn("__rowsel2g_", c["view_id"]); self.assertIn("__rowsel2s_", c2["view_id"])
            self.assertNotIn(c["view_id"], REPL["view_ids"])
            self.assertEqual(rb.verify_context_dir_v2(v2dir / c2["view_id"])["problems"], [])

    def test_verify_detects_tampering(self):
        _, rec = self.build()
        d = self.shard / rec["contexts"][0]["view_id"]
        ctx = json.loads((d / "context.json").read_bytes())
        bad = {"cut": [dict(ctx, cut=dict(ctx["cut"], replaces_cut=ctx["cut"]["replaces_cut"][0])),
                       dict(ctx, cut=dict(ctx["cut"], replaces_cut=ctx["cut"]["replaces_cut"][:1])),
                       dict(ctx, cut=dict(ctx["cut"], group_column={"index": 0, "name": "x0"})),
                       dict(ctx, cut=dict(ctx["cut"], comparable_with_replaced=True))],
               "group_gates": [dict(ctx, rowsel=dict(ctx["rowsel"], split=dict(ctx["rowsel"]["split"], annotation_groups_straddling_folds=1))),
                               dict(ctx, rowsel=dict(ctx["rowsel"], split=dict(ctx["rowsel"]["split"], group_value_invariant=False)))]}
        for want, cs in bad.items():
            for c2 in cs:
                (d / "context.json").write_text(json.dumps(c2))
                self.assertIn(want, rb.verify_context_dir_v2(d)["problems"])

    def test_dataset_done_is_version_specific_and_resume_never_rewrites(self):
        _, rec = self.build()
        self.assertTrue(rb.dataset_done_v2(self.shard, H, self.CODES, rb.SPLIT_V2G))
        self.assertFalse(rb.dataset_done_v2(self.shard, H, self.CODES, rb.SPLIT_V2))
        stamp = {p: p.stat().st_mtime_ns for d in self.shard.iterdir() if d.name != "_datasets" for p in d.iterdir()}
        _, again = self.build()
        self.assertEqual((again["contexts_written"], again["contexts_already_done"]), (0, 5))
        self.assertEqual(stamp, {p: p.stat().st_mtime_ns for d in self.shard.iterdir() if d.name != "_datasets" for p in d.iterdir()})

    def test_the_group_column_must_be_a_feature(self):
        _, rec = self.build(prof=profile(5, "target"), expect_done=False)
        self.assertEqual((rec["status"], rec["reason"]), ("failed", "group_column_is_not_a_feature"))


M3 = M.copy()
M3[:, GCOL] = (np.rint(M[:, GCOL] * 10) % 3) / 10.0                    # a 3-level group column: 3 components
M3 = tb._r9(M3)


class EmptyFolds(Base):
    """Lane-wide convention: the split always uses 5 folds; with c < 5 components folds c..4 stay empty, are recorded,
    and get no context; the cut binds the 5-fold assignment."""

    def build3(self, m=M3, shard=None, **kw):
        ds = tb.dataset(m, folds=t2.REGISTRY_FOLDS)
        its = tb.items(H, "clean", folds=range(3))
        rec = rb.build_dataset_v2(ds, its, shard or self.shard, self.CODES, dict(tb.REFS), budget=tb.BUDGET,
                                  profile=kw.pop("prof", None) or rb.profile_v2g(GCOL, GNAME, REPL, expected_empty_folds=[3, 4]),
                                  extra={"registry_grouping": {"registry_grouped_by_feature_column": True, "registry_group_columns": [GCOL]}}, **kw)
        return ds, rec

    def test_three_components_build_three_contexts_with_trailing_empty_folds(self):
        _, rec = self.build3()
        self.assertEqual(rec["status"], "done", rec.get("detail"))
        folds = folds_of(M3)
        self.assertEqual([len(folds[f]) == 0 for f in range(5)], [False, False, False, True, True])
        self.assertEqual((rec["split"]["empty_folds"], rec["split"]["effective_folds"], rec["split"]["components"]), ([3, 4], 3, 3))
        self.assertEqual((rec["epoch0_replaced"]["new_contexts"], len(rec["contexts"])), (3, 3))
        self.assertEqual(rec["split"]["annotation_groups_straddling_folds"], 0)
        sha = g.split_sha256(H, "t1", SPEC, folds)                     # the 5-fold assignment, empty folds included
        self.assertEqual(len(folds), 5)
        for f, c in enumerate(self.ctxs(rec)):
            self.assertEqual(c["fold"], f)
            self.assertEqual((c["cut"]["cut_id"], c["cut"]["folds"], c["cut"]["empty_folds"], c["cut"]["effective_folds"]),
                             ("trainonly-split-v2g:" + sha, 5, [3, 4], 3))
            self.assertEqual((c["rowsel"]["split"]["empty_folds"], c["rowsel"]["split"]["effective_folds"]), ([3, 4], 3))
            self.assertEqual(len(c["cut"]["replaces_cut"][1]["view_ids"]), 5)
            self.assertTrue(set(c["rowsel"]["test_positions"]) <= set(folds[f]))
            self.assertEqual(rb.verify_context_dir_v2(self.shard / c["view_id"])["problems"], [])

    def test_deterministic(self):
        _, a = self.build3()
        other = self.dir / "shard-again"; other.mkdir()
        _, b = self.build3(shard=other)
        self.assertEqual([c["view_id"] for c in a["contexts"]], [c["view_id"] for c in b["contexts"]])
        for c in a["contexts"]:
            for f in ("encoder.jtt1.gz", "teacher.jtt1.gz", "a0.f32"):
                self.assertEqual((self.shard / c["view_id"] / f).read_bytes(), (other / c["view_id"] / f).read_bytes())
        self.assertEqual(a["split"]["sha256"], b["split"]["sha256"])

    def test_y_changes_never_change_the_split_or_the_empty_folds(self):
        _, rec = self.build3()
        other = self.dir / "shard-y"; other.mkdir()
        _, rec2 = self.build3(m=t2.noise_targets(M3, range(M3.shape[0])), shard=other)
        self.assertEqual((rec2["split"]["sha256"], rec2["split"]["empty_folds"]), (rec["split"]["sha256"], [3, 4]))
        self.assertTrue(rec["split"]["target_check"]["target_invariant"])
        p = prep_of(M3)
        for kind in ("permuted", "replaced", "outlier", "constant"):
            v = np.array(p.values, copy=True)
            v[:, -1] = {"permuted": v[::-1, -1], "replaced": 1e6, "outlier": -1e12, "constant": 0.0}[kind]
            got = profile()["split_fn"](H, p.positions, v, p.specs)
            self.assertEqual(rb.empty_fold_list([len(got[f]) for f in range(5)]), [3, 4], kind)

    def test_forcing_five_folds_on_three_groups_straddles_or_empties(self):
        """MUST FAIL: 5 non-empty folds over 3 groups need a straddling split (v2 without groups), and v2 without the
        allowance refuses empty folds."""
        p = prep_of(M3)
        v2f = rb.split_v2_t1(H, p.positions, p.values, p.specs)
        self.assertTrue(all(len(v2f[f]) for f in range(5)))
        self.assertGreater(g.straddling_groups(v2f, p.positions.tolist(), groups_of(M3)), 0)
        self.assertEqual(rb.empty_fold_list([len(v) for v in folds_of(M3).values()]), [3, 4])
        _, rec = self.build3(split_fn=rb.split_v2_t1)
        self.assert_refused(rec, self.shard, "annotation_groups_straddle_folds")

    def test_empty_folds_allowed_only_trailing_and_only_below_five_components(self):
        ok = {"groups_straddling_folds": 0, "target_check": {"target_invariant": True}, "value_check_fold0": {"valid": True, "invariant": True}}
        cases = [({"components": 3, "fold_sizes": [5, 4, 3, 0, 0]}, False),
                 ({"components": 4, "fold_sizes": [5, 4, 3, 2, 0]}, False),
                 ({"components": 5, "fold_sizes": [5, 4, 3, 2, 0]}, True),        # 5 components: no fold may be empty
                 ({"components": 7, "fold_sizes": [5, 4, 3, 0, 0]}, True),
                 ({"components": 3, "fold_sizes": [5, 0, 3, 4, 0]}, True),        # not trailing
                 ({"components": 3, "fold_sizes": [5, 4, 0, 0, 0]}, True),        # more empty folds than c leaves
                 ({"components": 1, "fold_sizes": [9, 0, 0, 0, 0]}, True),        # one component: no training rows
                 ({"fold_sizes": [5, 4, 3, 0, 0]}, True)]                         # split v2 (no components): never
        for extra, refused in cases:
            with self.subTest(extra=extra):
                self.assertEqual("empty_outer_fold" in rb.split_gate_failures(dict(ok, **extra)), refused)
        p = prep_of(M3)
        rep5 = {"fold_sizes": [len(v) for v in folds_of(M3).values()], "components": 5}
        self.assertIn("empty_outer_fold", rb.split_gate_failures(dict(ok, **rep5)))

    def test_planned_empty_folds_must_match(self):
        _, rec = self.build3(prof=rb.profile_v2g(GCOL, GNAME, REPL, expected_empty_folds=[4]))
        self.assertEqual((rec["status"], rec["reason"]), ("failed", "empty_folds_differ_from_plan"))
        self.assertEqual([d for d in self.shard.iterdir() if d.name != "_datasets"], [])

    def test_verify_detects_empty_fold_tampering(self):
        _, rec = self.build3()
        d = self.shard / rec["contexts"][0]["view_id"]
        ctx = json.loads((d / "context.json").read_bytes())
        for c2 in (dict(ctx, cut=dict(ctx["cut"], empty_folds=[4])), dict(ctx, cut=dict(ctx["cut"], effective_folds=5)),
                   dict(ctx, rowsel=dict(ctx["rowsel"], split=dict(ctx["rowsel"]["split"], empty_folds=[]))),
                   dict(ctx, rowsel=dict(ctx["rowsel"], split=dict(ctx["rowsel"]["split"], components=5))),
                   dict(ctx, fold=3)):
            (d / "context.json").write_text(json.dumps(c2))
            self.assertIn("empty_folds", rb.verify_context_dir_v2(d)["problems"])


class Summary(Base):
    def test_summary_of_a_v2gsplit_tree(self):
        import argparse
        tree = self.dir / "tree"
        (tree / "shard-0").mkdir(parents=True)
        e0 = [{"view_id": r["view_id"], "context_sha256": r["context_sha256"], "fold": r["fold"]} for r in tb.epoch_rows(H, range(4))]
        ev = {"reproducing": [GCOL], "candidates": [GCOL], "identical_components_across_reproducing": True,
              "registry_folds_compared": 25, "base_seed": 20260930, "per_candidate": {}}
        rg = {"registry_grouped_by_feature_column": True, "registry_group_columns": [GCOL]}
        built = {"dataset_hash": H, "label": "clean", "epoch0": e0, "shard": 0, "est_cpu_s": 1.0, "registry_grouping": rg,
                 "disposition": "build", "group_column": SPEC, "group_source": "fixture", "group_evidence": ev, "replaces_v2split": REPL}
        refused = dict(built, dataset_hash="0" * 16, disposition="not_built", reason="ambiguous_group_columns_with_different_components",
                       group_evidence=dict(ev, reproducing=[GCOL, 3], identical_components_across_reproducing=False))
        refused.pop("shard"); refused.pop("group_column")
        (tree / "plan.json").write_text(json.dumps({"of": 1, "tree": "kmeans-clean-groups", "split_version": rb.SPLIT_V2G,
                                                    "counts": {}, "inputs": {}, "datasets": [built, refused]}))
        _, rec = self.build(shard=tree / "shard-0")
        (tree / "shard-0-of-1.receipt.json").write_text(json.dumps({"expected_datasets": 1, "done": 1, "complete": True, "cpu_s": 1,
                                                                    "elapsed_s": 1, "guard_pauses": 0}))
        log = self.dir / "t.log"; log.write_text("OK\n")
        mut = self.dir / "m.json"; mut.write_text("{\"totals\": {}}")
        a = argparse.Namespace(out=str(tree), tree="kmeans-clean-groups", tests_log=str(log), mutants=str(mut), pilot=None,
                               receipt=str(self.dir / "summary.json"), code_custody=None, code_why=None)
        with mock.patch.object(rb, "code_custody_check", lambda *x, **k: {"ok": True, "files": {}}):
            rc = rb.summarize_v2(a)
        doc = json.loads((self.dir / "summary.json").read_bytes())
        self.assertEqual((rc, doc["split_version"], doc["counts"]["verification_problems"]), (0, rb.SPLIT_V2G, 0), doc["problems"])
        c = doc["counts"]
        self.assertEqual((c["expected_datasets"], c["done_datasets"], c["not_built_datasets"], c["replaced_v2split_contexts"],
                          c["annotation_groups_straddling_folds"], c["group_annotation_dropped_datasets"]), (1, 1, 1, 5, 0, []))
        self.assertEqual(c["not_built_by_reason"], {"ambiguous_group_columns_with_different_components": 1})
        row = next(r for r in doc["datasets"] if r["dataset_hash"] == H)
        self.assertEqual(row["group_column"], SPEC)
        self.assertEqual(row["replaces_cut"][1], REPL["cut_id"])
        self.assertTrue(row["group_value_invariant"] and row["target_invariant"] and row["value_invariant"])
        self.assertEqual(doc["owner_ruling"]["id"], "GROUP-UNION-B")
        self.assertTrue((tree / ".complete").exists())
        # a context whose cut no longer names the planned v2split cut is a problem
        bad = dict(REPL, cut_id="trainonly-split-v2:" + "6" * 64)
        (tree / "plan.json").write_text(json.dumps({"of": 1, "tree": "kmeans-clean-groups", "split_version": rb.SPLIT_V2G, "counts": {},
                                                    "inputs": {}, "datasets": [dict(built, replaces_v2split=bad)]}))
        a.receipt = str(self.dir / "summary2.json")
        with mock.patch.object(rb, "code_custody_check", lambda *x, **k: {"ok": True, "files": {}}):
            self.assertEqual(rb.summarize_v2(a), 1)
        doc2 = json.loads((self.dir / "summary2.json").read_bytes())
        self.assertIn("replaces_cut_or_group_column", {p for x in doc2["problems"] for p in x.get("problems", [])})


class Supplement(Base):
    """plan.json refused a 3-component dataset; the create-only empty-folds supplement plans it into shard-e0 and the
    summary merges it: 3 contexts expected, all receipts present, complete."""

    def test_supplement_plan_build_and_summary(self):
        import argparse
        tree = self.dir / "tree"; tree.mkdir()
        e0 = [{"view_id": r["view_id"], "context_sha256": r["context_sha256"], "fold": r["fold"]} for r in tb.epoch_rows(H, range(3))]
        ev = {"reproducing": [GCOL], "candidates": [GCOL], "identical_components_across_reproducing": True, "registry_folds_compared": 15,
              "base_seed": 20260930, "names": {str(GCOL): GNAME},
              "per_candidate": {str(GCOL): {"reproduces_registry": True, "field_bytes_equals_value_grouping": True, "v2g_components": 3,
                                            "effective_folds": 3}}}
        refused = {"dataset_hash": H, "label": "clean", "epoch0": e0, "est_cpu_s": 1.0, "group_evidence": ev, "replaces_v2split": REPL,
                   "registry_grouping": {"registry_grouped_by_feature_column": True, "registry_group_columns": [GCOL]},
                   "disposition": "not_built", "reason": "fewer_v2g_components_than_outer_folds", "detail": "3 components",
                   "candidate_group_column": SPEC}
        (tree / "plan.json").write_text(json.dumps({"of": 1, "tree": "kmeans-clean-groups", "split_version": rb.SPLIT_V2G, "base_tree": "kmeans-clean",
                                                    "counts": {}, "inputs": {}, "owner_line": "fixture", "budget": tb.BUDGET, "datasets": [refused]}))
        (tree / "shard-0").mkdir()
        (tree / "shard-0-of-1.receipt.json").write_text(json.dumps({"expected_datasets": 0, "done": 0, "complete": True, "cpu_s": 0,
                                                                    "elapsed_s": 0, "guard_pauses": 0}))
        a = argparse.Namespace(out=str(tree), tree="kmeans-clean-groups")
        self.assertEqual(rb.plan_v2g_supplement(a), 0)
        sup = json.loads((tree / rb.SUPPLEMENT_EMPTY_FOLDS).read_bytes())
        e = sup["datasets"][0]
        self.assertEqual((sup["kind"], sup["shard_tag"], sup["expected_new_contexts"], e["disposition"], e["empty_folds"], e["effective_folds"],
                          e["group_column"], e["shard"]), ("supplement", "e", 3, "build", [3, 4], 3, SPEC, 0))
        self.assertEqual(e["supersedes_plan_disposition"]["reason"], "fewer_v2g_components_than_outer_folds")
        self.assertTrue(sup["decision"]["owner_ruling"]["key_present"])
        ds = tb.dataset(M3, folds=t2.REGISTRY_FOLDS)
        rec = rb.build_dataset_v2(ds, tb.items(H, "clean", folds=range(3)), tree / "shard-e0", self.CODES, dict(tb.REFS), budget=tb.BUDGET,
                                  profile=rb.entry_profile(e, rb.SPLIT_V2G), extra={"registry_grouping": e["registry_grouping"]})
        self.assertEqual((rec["status"], len(rec["contexts"])), ("done", 3), rec.get("detail"))
        log = self.dir / "t.log"; log.write_text("OK\n")
        mut = self.dir / "m.json"; mut.write_text("{\"totals\": {}}")
        sa = argparse.Namespace(out=str(tree), tree="kmeans-clean-groups", tests_log=str(log), mutants=str(mut), pilot=None,
                                receipt=str(self.dir / "s1.json"), code_custody=None, code_why=None)
        with mock.patch.object(rb, "code_custody_check", lambda *x, **k: {"ok": True, "files": {}}):
            self.assertEqual(rb.summarize_v2(sa), 1)                       # the supplement's shard receipt is missing
            (tree / "shard-e0-of-1.receipt.json").write_text(json.dumps({"expected_datasets": 1, "done": 1, "complete": True, "cpu_s": 1,
                                                                         "elapsed_s": 1, "guard_pauses": 0}))
            sa.receipt = str(self.dir / "s2.json")
            self.assertEqual(rb.summarize_v2(sa), 0)
        doc = json.loads((self.dir / "s2.json").read_bytes())
        c = doc["counts"]
        self.assertEqual((c["expected_new_contexts"], c["done_new_contexts"], c["not_built_datasets"], c["empty_folds_datasets"]),
                         (3, 3, 0, {H: [3, 4]}))
        self.assertEqual(doc["datasets"][0]["plan"], rb.SUPPLEMENT_EMPTY_FOLDS)
        self.assertEqual(len(doc["plan_supplements"]), 1)
        self.assertTrue(doc["owner_ruling"]["key_present"])
        self.assertTrue((tree / ".complete").exists())
        with self.assertRaises(SystemExit):                                # a complete tree takes no further supplement
            rb.plan_v2g_supplement(a)

    def test_a_context_count_short_of_the_effective_folds_is_a_problem(self):
        import argparse
        tree = self.dir / "tree"
        (tree / "shard-0").mkdir(parents=True)
        e0 = [{"view_id": r["view_id"], "context_sha256": r["context_sha256"], "fold": r["fold"]} for r in tb.epoch_rows(H, range(4))]
        ev = {"reproducing": [GCOL], "candidates": [GCOL], "identical_components_across_reproducing": True, "registry_folds_compared": 25,
              "base_seed": 20260930, "per_candidate": {}}
        e = {"dataset_hash": H, "label": "clean", "epoch0": e0, "shard": 0, "est_cpu_s": 1.0, "disposition": "build", "group_column": SPEC,
             "group_source": "fixture", "group_evidence": ev, "replaces_v2split": REPL, "effective_folds": 4, "empty_folds": [4],
             "registry_grouping": {"registry_grouped_by_feature_column": True, "registry_group_columns": [GCOL]}}
        (tree / "plan.json").write_text(json.dumps({"of": 1, "tree": "kmeans-clean-groups", "split_version": rb.SPLIT_V2G, "counts": {},
                                                    "inputs": {}, "datasets": [e]}))
        self.build(shard=tree / "shard-0")                                 # 10 components: 5 contexts, no empty fold
        (tree / "shard-0-of-1.receipt.json").write_text(json.dumps({"expected_datasets": 1, "done": 1, "complete": True, "cpu_s": 1,
                                                                    "elapsed_s": 1, "guard_pauses": 0}))
        log = self.dir / "t.log"; log.write_text("OK\n")
        mut = self.dir / "m.json"; mut.write_text("{\"totals\": {}}")
        sa = argparse.Namespace(out=str(tree), tree="kmeans-clean-groups", tests_log=str(log), mutants=str(mut), pilot=None,
                                receipt=str(self.dir / "s.json"), code_custody=None, code_why=None)
        with mock.patch.object(rb, "code_custody_check", lambda *x, **k: {"ok": True, "files": {}}):
            self.assertEqual(rb.summarize_v2(sa), 1)
        probs = {p for x in json.loads((self.dir / "s.json").read_bytes())["problems"] for p in x.get("problems", [])}
        self.assertTrue({"context_count", "empty_folds_differ_from_plan"} <= probs, probs)


class V2(Base):
    def test_the_v2_route_builds_the_bytes_c0b989b6_built(self):
        self.assertEqual(hashlib.sha256(V2_BUILDER.read_bytes()).hexdigest(), V2_BUILDER.name.rsplit(".", 1)[1])
        loader = importlib.machinery.SourceFileLoader("v2_rowsel_build_c0b989b6", str(V2_BUILDER))
        old = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader)); loader.exec_module(old)
        ds = tb.dataset(M, folds=t2.REGISTRY_FOLDS)
        its = tb.items(H, "clean", folds=range(4))
        extra = {"registry_grouping": {"registry_grouped_by_feature_column": True, "registry_group_columns": [GCOL]}}
        a_dir, b_dir = self.dir / "new", self.dir / "old"
        a_dir.mkdir(); b_dir.mkdir()
        a = rb.build_dataset_v2(ds, its, a_dir, self.CODES, dict(tb.REFS), budget=tb.BUDGET, extra=dict(extra))
        b = old.build_dataset_v2(ds, its, b_dir, self.CODES, dict(tb.REFS), budget=tb.BUDGET, extra=dict(extra))
        vol = ("utc_us", "cpu_s", "wall_s", "maxrss_mb", "build_cpu_s")
        strip = lambda r: {k: v for k, v in r.items() if k not in vol and k != "contexts"}
        self.assertEqual(a["status"], "done", a.get("detail"))
        self.assertEqual(strip(a), strip(b))
        self.assertEqual([{k: v for k, v in c.items() if k not in ("cpu_s", "target_linear")} for c in a["contexts"]],
                         [{k: v for k, v in c.items() if k != "cpu_s"} for c in b["contexts"]])       # + the LINEAR-DETERMINED result
        for c in a["contexts"]:
            da, db = a_dir / c["view_id"], b_dir / c["view_id"]
            self.assertEqual(sorted(x.name for x in da.iterdir()), sorted(x.name for x in db.iterdir()))
            for f in ("encoder.jtt1.gz", "teacher.jtt1.gz", "a0.f32", ".done"):
                self.assertEqual((da / f).read_bytes(), (db / f).read_bytes(), f)
            ca, cb = (json.loads((x / "context.json").read_bytes()) for x in (da, db))
            self.assertIsNone(ca["target_linear"]["verdict"])
            self.assertEqual({k: v for k, v in ca.items() if k not in vol + ("target_linear",)}, {k: v for k, v in cb.items() if k not in vol})

    def test_no_groups_is_split_v2_on_t1(self):
        p = prep_of(M)
        self.assertEqual(g.split(H, p.positions.tolist(), rowsel_raw.feature_bytes(p.values, p.specs), None, "t1"),
                         rb.split_v2_t1(H, p.positions, p.values, p.specs))


class _PortStub:
    """A deterministic in-test stand-in for the registry split port (rb.load_gsk), so the evidence ORDER can be checked
    without the private port or any private input: lines split on newlines, fields on commas, the duplicate key is the
    joined fields, and the 'registry split' is one fold made of the rows whose group field hashes to an even first byte
    (so a candidate reproduces the stub registry exactly when its fields induce the same rows)."""

    @staticmethod
    def csv_lines(body):
        parts = body.split(b"\n")
        return parts[:-1] if parts and parts[-1] == b"" else parts

    @staticmethod
    def csv_fields(line):
        return line.split(b",")

    @staticmethod
    def dup_key(fields):
        return b"|".join(fields)

    @staticmethod
    def generate_folds(keys, base_seed, groups=None):
        groups = groups if groups is not None else keys
        idx = [i for i, x in enumerate(groups) if hashlib.sha256(bytes(x)).digest()[0] % 2 == 0]
        return 2, [{"repeat": 0, "fold": 0, "seed": int(base_seed), "test_indices": idx}], None


class EvidenceOrderNoPort(unittest.TestCase):
    """PR #218 finding 8 order proof that CI can run (red-team 6016730510): registry_group_evidence with a STUB port
    (no private port, no private input, nothing under RUN read). MUST FAIL on 8b056648, where component_partition ran
    on a tied column's group values before target_tied_group and its structure hash was stored."""
    TIED = 3                                                                # x3 := floor(4y)/4: bins the target

    def setUp(self):
        m = M.copy(); m[:, self.TIED] = np.floor(m[:, 5] * 4) / 4
        self.m = tb._r9(m)
        ls = tb.lines_of(self.m)
        self.tr, self.te = b"\n".join(ls[:tb.NTR]) + b"\n", b"\n".join(ls[tb.NTR:]) + b"\n"
        self.U = rowsel.universe_from_t1(H, self.tr, self.te, 6)
        rows = [_PortStub.csv_fields(l) for b in (self.tr, self.te) for l in _PortStub.csv_lines(b)]
        k, folds, _ = _PortStub.generate_folds(None, 20260930, groups=[r[self.TIED] for r in rows])
        self.split_body = json.dumps({"source": {"dataset_hash": H, "train_sha256": hashlib.sha256(self.tr).hexdigest(),
                                                 "test_sha256": hashlib.sha256(self.te).hexdigest()},
                                      "effective_folds": k, "folds": folds}).encode()
        self.tied_gv = rb.t1_group_values(np.column_stack([self.U.X, self.U.y]), self.TIED)

    def evidence(self, cands):
        calls = []
        real_cp, real_c = g.component_partition, g.components

        def spy_cp(positions, feature_bytes, group_values=None):
            calls.append(None if group_values is None else list(group_values))
            return real_cp(positions, feature_bytes, group_values)

        def spy_c(positions, feature_bytes, group_values=None):
            calls.append(None if group_values is None else list(group_values))
            return real_c(positions, feature_bytes, group_values)
        with mock.patch.object(rb, "load_gsk", lambda: _PortStub), \
                mock.patch.object(rb.rowsel_split_v2_groups, "component_partition", spy_cp), \
                mock.patch.object(rb.rowsel_split_v2_groups, "components", spy_c):
            ev = rb.registry_group_evidence(H, self.U, self.split_body, self.tr, self.te, cands)
        return ev, calls

    def test_the_stub_fixture_is_a_tied_reproducing_column(self):
        self.assertEqual(g.target_tied_group(self.tied_gv, self.m[:, 5]), "group_is_binned_target")
        self.assertIsNone(g.target_tied_group(rb.t1_group_values(np.column_stack([self.U.X, self.U.y]), GCOL), self.m[:, 5]))
        ev, _ = self.evidence([self.TIED, GCOL])
        self.assertEqual(ev["reproducing"], [self.TIED])

    def test_the_census_reads_no_target(self):
        """FOLD-1: the census (registry_group_evidence) reads features only. target_tied_group is never called; the tied
        candidate's component structure is computed like any other; its tie is decided per context at build. MUST FAIL
        on the #218 finding-8 order (target_tied_group first, no structure for a tied column)."""
        with mock.patch.object(rb.rowsel_split_v2_groups, "target_tied_group", side_effect=AssertionError("census read y")):
            ev, calls = self.evidence([self.TIED, GCOL])
        self.assertGreaterEqual(sum(1 for gv in calls if gv == self.tied_gv), 1)       # features only: structure computed
        c = ev["per_candidate"][str(self.TIED)]
        self.assertEqual(c["target_tie"], rb.TIE_PER_CONTEXT)
        self.assertIsNotNone(c["component_structure_sha256"]); self.assertIsNotNone(c["v2g_components"])
        self.assertNotIn("not_computed", c)

    def test_every_candidate_has_a_structure(self):
        """MUST FAIL on the old order, where a tied reproducing candidate left identical_components None."""
        ev, _ = self.evidence([self.TIED, GCOL])
        self.assertIn(ev["identical_components_across_reproducing"], (True, False))
        self.assertNotIn("components_not_computed_for", ev)

    def test_disposition_and_the_stored_plan_entry(self):
        """The tied reproducing column is PLANNED like any other (features only); the plan entry stores its structure and
        the per-context tie rule. MUST FAIL on the old dataset-level 'not_built: group_column_tied_to_target'."""
        ev, _ = self.evidence([self.TIED, GCOL])
        d = rb.group_disposition(ev, tb.COLS)
        self.assertEqual((d["disposition"], d["group_column"]), ("build", {"index": self.TIED, "name": tb.COLS[self.TIED]}), d)
        stored = json.loads(json.dumps({"group_evidence": ev} | d))         # what plan_v2g stores for the dataset
        c = stored["group_evidence"]["per_candidate"][str(self.TIED)]
        self.assertIsNotNone(c["component_structure_sha256"])
        self.assertEqual(c["target_tie"], rb.TIE_PER_CONTEXT)
        self.assertIn("group_column", stored)

    def test_planning_never_decides_on_a_tie(self):
        """group_disposition ignores any tie field (none is computed at planning); the structure rule decides. MUST FAIL on
        the old 'ties FIRST' rule."""
        ev = {"reproducing": [self.TIED, GCOL], "identical_components_across_reproducing": None,
              "per_candidate": {str(self.TIED): {"target_tie": "group_is_binned_target", "component_structure_sha256": None,
                                                 "field_bytes_equals_value_grouping": None, "v2g_components": None},
                                str(GCOL): {"target_tie": None, "component_structure_sha256": "a" * 64,
                                            "field_bytes_equals_value_grouping": True, "v2g_components": 10}}}
        self.assertEqual(rb.group_disposition(ev, tb.COLS)["reason"], "ambiguous_group_columns_with_different_components")
        ev["identical_components_across_reproducing"] = True
        ev["per_candidate"][str(self.TIED)].update(component_structure_sha256="a" * 64, field_bytes_equals_value_grouping=True, v2g_components=10)
        self.assertEqual(rb.group_disposition(ev, tb.COLS)["disposition"], "build")


class Evidence(unittest.TestCase):
    """rb.registry_group_evidence against registry split objects made with the registry port itself."""

    @classmethod
    def setUpClass(cls):
        cls.gsk = rb.load_gsk()

    def bodies(self, lines):
        return b"\n".join(lines[:tb.NTR]) + b"\n", b"\n".join(lines[tb.NTR:]) + b"\n"

    def registry(self, lines, col, tamper=None):
        """A registry split object of these T1 lines with annotation group = the field bytes of column `col`."""
        tr, te = self.bodies(lines)
        rows = [self.gsk.csv_fields(l) for b in (tr, te) for l in self.gsk.csv_lines(b)]
        k, folds, _ = self.gsk.generate_folds([self.gsk.dup_key(r) for r in rows], 20260930, groups=[r[col] for r in rows])
        if tamper:
            tamper(folds)
        body = json.dumps({"source": {"dataset_hash": H, "train_sha256": hashlib.sha256(tr).hexdigest(),
                                      "test_sha256": hashlib.sha256(te).hexdigest()},
                           "effective_folds": k, "folds": folds}).encode()
        return tr, te, body

    def evidence(self, lines, col, cands, tamper=None):
        tr, te, body = self.registry(lines, col, tamper)
        U = rowsel.universe_from_t1(H, tr, te, 6)
        return rb.registry_group_evidence(H, U, body, tr, te, cands)

    def test_the_reproducing_column_is_found(self):
        ev = self.evidence(tb.lines_of(M), GCOL, [GCOL, 2, 4])
        self.assertEqual((ev["reproducing"], ev["registry_folds_compared"], ev["base_seed"]), ([GCOL], 25, 20260930))
        self.assertTrue(ev["per_candidate"][str(GCOL)]["field_bytes_equals_value_grouping"])
        self.assertFalse(ev["per_candidate"]["4"]["reproduces_registry"])
        d = rb.group_disposition(ev, tb.COLS)
        self.assertEqual((d["disposition"], d["group_column"]), ("build", SPEC))

    def test_a_changed_later_repeat_is_not_reproduced(self):
        def swap(folds):
            a = next(x for x in folds if x["repeat"] == 3 and x["fold"] == 0)
            b = next(x for x in folds if x["repeat"] == 3 and x["fold"] == 1)
            a["test_indices"], b["test_indices"] = b["test_indices"], a["test_indices"]
        ev = self.evidence(tb.lines_of(M), GCOL, [GCOL], swap)
        self.assertEqual(ev["reproducing"], [])
        self.assertEqual(rb.group_disposition(ev, tb.COLS)["reason"], "no_group_column_reproduces_registry")

    def test_missing_cells_are_one_group(self):
        m = M.copy()
        m[np.arange(0, M.shape[0], 37), GCOL] = np.nan
        lines = [l.replace(b"nan", b"") for l in tb.lines_of(m)]
        ev = self.evidence(lines, GCOL, [GCOL])
        self.assertEqual(ev["reproducing"], [GCOL])
        self.assertTrue(ev["per_candidate"][str(GCOL)]["field_bytes_equals_value_grouping"])
        self.assertEqual(ev["per_candidate"][str(GCOL)]["missing_cells"], len(range(0, M.shape[0], 37)))
        gv = rb.t1_group_values(np.column_stack([rowsel.universe_from_t1(H, *self.bodies(lines), 6).X, M[:, 5]]), GCOL)
        self.assertEqual({gv[i] for i in range(0, M.shape[0], 37)}, {rb.MISSING_GROUP})
        self.assertEqual(rb.group_disposition(ev, tb.COLS)["disposition"], "build")

    def test_fewer_components_than_outer_folds_plan_trailing_empty_folds(self):
        """A group column with 4 levels gives 4 components: the 5-fold split leaves fold 4 empty (the registry used 4 folds);
        one component is refused (no fold could have training rows)."""
        ev = self.evidence(tb.lines_of(M), 4, [4])
        self.assertEqual((ev["reproducing"], ev["per_candidate"]["4"]["v2g_components"], ev["per_candidate"]["4"]["effective_folds"]), ([4], 4, 4))
        d = rb.group_disposition(ev, tb.COLS)
        self.assertEqual((d["disposition"], d["group_column"], d["empty_folds"], d["effective_folds"]),
                         ("build", {"index": 4, "name": "cat"}, [4], 4))
        one = json.loads(json.dumps(ev)); one["per_candidate"]["4"]["v2g_components"] = 1
        self.assertEqual(rb.group_disposition(one, tb.COLS)["reason"], "fewer_than_two_v2g_components")
        five = rb.group_disposition(self.evidence(tb.lines_of(M), GCOL, [GCOL]), tb.COLS)
        self.assertEqual((five["empty_folds"], five["effective_folds"]), ([], 5))

    def test_a_reproducing_column_tied_to_the_target_is_refused_per_context_at_build(self):
        """FOLD-1: x3 := y reproduces the registry; planning reads no y and builds on x3; at build every context's OWN
        outer-train rows show the tie, so each context is refused there. MUST FAIL on the old dataset-level not_built."""
        m = M.copy(); m[:, 3] = m[:, 5]                                    # x3 copies y
        ev = self.evidence(tb.lines_of(m), 3, [3, GCOL])
        self.assertEqual(ev["reproducing"], [3])
        d = rb.group_disposition(ev, tb.COLS)
        self.assertEqual((d["disposition"], d.get("group_column", {}).get("index")), ("build", 3), d)
        self.assertEqual({ev["per_candidate"][c]["target_tie"] for c in ("3", str(GCOL))}, {rb.TIE_PER_CONTEXT})
        prof = rb.profile_v2g(3, "x3", REPL)
        prep = prep_of(m)
        folds = prof["split_fn"](H, prep.positions, prep.values, prep.specs)
        self.assertEqual({prof["fold_target_tie_fn"](prep, folds, f) for f in folds if folds[f]}, {"group_bytes_equal_target_bytes"})

    def test_the_census_reads_no_target_with_the_registry_port(self):
        """As EvidenceOrderNoPort.test_the_census_reads_no_target, with the registry port: a reproducing candidate whose
        values bin the target (x3 := floor(4y)/4) gets its structure computed like any other, no y is read, and it is
        planned; its tie is decided per context at build. MUST FAIL on the #218 finding-8 order."""
        m = M.copy(); m[:, 3] = np.floor(m[:, 5] * 4) / 4
        m = tb._r9(m)
        with mock.patch.object(rb.rowsel_split_v2_groups, "target_tied_group", side_effect=AssertionError("census read y")):
            ev = self.evidence(tb.lines_of(m), 3, [3, GCOL])
        self.assertEqual(ev["reproducing"], [3])
        c3 = ev["per_candidate"]["3"]
        self.assertEqual(c3["target_tie"], rb.TIE_PER_CONTEXT)
        self.assertIsNotNone(c3["component_structure_sha256"]); self.assertIsNotNone(c3["v2g_components"])
        self.assertIn(ev["identical_components_across_reproducing"], (True, False))
        self.assertEqual(rb.group_disposition(ev, tb.COLS)["disposition"], "build")

    def test_planning_never_decides_on_a_tie_with_the_registry_port_fields(self):
        """The disposition ignores tie fields (planning computes none). MUST FAIL on the old 'ties FIRST' rule."""
        ev = {"reproducing": [3, GCOL], "identical_components_across_reproducing": None,
              "per_candidate": {"3": {"target_tie": "group_is_binned_target", "component_structure_sha256": None,
                                      "field_bytes_equals_value_grouping": None, "v2g_components": None},
                                str(GCOL): {"target_tie": None, "component_structure_sha256": "a" * 64,
                                            "field_bytes_equals_value_grouping": True, "v2g_components": 10}}}
        self.assertEqual(rb.group_disposition(ev, tb.COLS)["reason"], "ambiguous_group_columns_with_different_components")
        ev["identical_components_across_reproducing"] = True
        ev["per_candidate"]["3"].update(component_structure_sha256="a" * 64, field_bytes_equals_value_grouping=True, v2g_components=10)
        self.assertEqual(rb.group_disposition(ev, tb.COLS)["disposition"], "build")

    def test_group_values_are_canonical_float64_values(self):
        gv = rb.t1_group_values(np.array([[-0.0, 1.0], [0.0, 1.0], [0.5, 1.0], [np.nan, 1.0], [np.nan, 2.0]]), 0)
        self.assertEqual(gv[0], gv[1]); self.assertNotEqual(gv[1], gv[2])
        self.assertEqual((gv[3], gv[4]), (rb.MISSING_GROUP, rb.MISSING_GROUP))
        self.assertEqual(gv[2], np.float64(0.5).tobytes())
        self.assertNotEqual(len(rb.MISSING_GROUP), 8)                    # never decoded as a numeric '<f8' group value
        self.assertIsNone(rb.rowsel_split_v2_groups._group_number(rb.MISSING_GROUP))

    def test_a_field_and_value_grouping_mismatch_is_refused(self):
        """'0.1' and '0.10' are one value but two registry groups."""
        lines = tb.lines_of(M)
        hit = [i for i, l in enumerate(lines) if l.split(b",")[GCOL] == b"0.1"][::2]
        for i in hit:
            f = lines[i].split(b","); f[GCOL] = b"0.10"; lines[i] = b",".join(f)
        ev = self.evidence(lines, GCOL, [GCOL])
        self.assertEqual(ev["reproducing"], [GCOL])
        self.assertFalse(ev["per_candidate"][str(GCOL)]["field_bytes_equals_value_grouping"])
        self.assertEqual(rb.group_disposition(ev, tb.COLS)["reason"], "group_value_grouping_differs_from_registry_field_grouping")

    def test_ambiguous_columns_need_identical_component_structure(self):
        m = M.copy()
        m[:, 3] = np.round(0.9 - m[:, GCOL], 1)                    # a relabelling of the group column: same groups
        ev = self.evidence(tb.lines_of(tb._r9(m)), GCOL, [3, GCOL])
        self.assertEqual(ev["reproducing"], [3, GCOL])
        self.assertTrue(ev["identical_components_across_reproducing"])
        d = rb.group_disposition(ev, tb.COLS)
        self.assertEqual((d["disposition"], d["group_column"]["index"]), ("build", GCOL))
        self.assertIn("ambiguous", d["group_source"])
        ev2 = json.loads(json.dumps(ev))
        ev2["per_candidate"]["3"]["component_structure_sha256"] = "0" * 64
        ev2["identical_components_across_reproducing"] = False
        self.assertEqual(rb.group_disposition(ev2, tb.COLS)["reason"], "ambiguous_group_columns_with_different_components")


if __name__ == "__main__":
    unittest.main()
