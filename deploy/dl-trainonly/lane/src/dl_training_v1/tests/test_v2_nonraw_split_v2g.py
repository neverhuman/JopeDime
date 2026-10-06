"""GROUP-UNION-B: the clean non-raw routes re-split with split v2 + annotated groups (tools/v2_build_local.py,
tools/v2_build_l7a.py over dltrain/rowsel_split_v2_groups.py). Every invariance check is paired with a control that
MUST FAIL it.

  Groups      annotated groups never straddle folds; split v2 without groups straddles them (must fail); every group is
              a union with the feature duplicates; with no group values the hook reproduces split v2 exactly
  TargetFree  y never moves the split; duplicate- and group-preserving outer-test feature changes never move it; a split
              whose group key reads y is refused (must fail); the group column can never be the target
  Identity    cut_id 'trainonly-split-v2g:' + split_sha256(h, partition, group spec, folds); replaces_cut = [registry
              split, as-built v2split cut]; comparable_with_replaced false; v2split context schemas; new view tags
  Dataset     outer-test values of fold f never reach context f's train side (end to end); leaky builders refused
  Evidence    the group column rule: one registry-port-reproducing feature column; target/none/disagreeing refused;
              ambiguous columns accepted only with identical components
  Writes      a v2g dataset writes its contexts with the group fields; .done last
"""
import gzip
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dltrain import jtt1, rowsel_raw, rowsel_raw_split_v2 as v2, rowsel_split_v2_groups as g, t1raw
from tests import test_v2_nonraw_split_v2 as base

tool, l7a = base.tool, base.l7a
H, N = base.H, base.N
GCOL, REGIONS = 2, 23


def region_rows():
    """The v2 fixture with feature n2 replaced by a 23-level region code: the annotated group IS a feature column."""
    num, cat, y = base.raw_rows()
    rng = np.random.default_rng(29)
    region = rng.integers(0, REGIONS, size=N).astype(float)
    for i in range(N - base.FULL_DUP - base.FEAT_DUP, N):            # copies keep the region of their source row
        src = next(j for j in range(N) if np.array_equal(num[j], num[i], equal_nan=True) and j != i)
        region[i] = region[src]
    num = np.array(num, copy=True)
    num[:, GCOL] = region
    return num, cat, y


NUM, CAT, Y = region_rows()
A, META = base.t1_table(NUM, CAT, Y)
VALUES = base.raw_values(NUM, CAT, Y, t1raw.columns(META, A.shape[1]))
RAW_SPECS = t1raw.columns(META, A.shape[1])
SPECS_T1 = tool.t1_group_specs(A.shape[1])
SPEC = g.group_spec(GCOL, "n2")
GV = tool.t1_group_values(A, GCOL)
V2CUT = "trainonly-split-v2:" + "c" * 64


def g_entry(h=H):
    e = base.plan_entry(h)
    e["context_extra"] = {"group_annotation_dropped": False, "group_annotation_kept": True, "rights": "licensed",
                          "registry_group_column": {"t1_column_index": GCOL, "t1_column_name": "n2"}}
    e["group_column"] = {"spec": SPEC, "source": "test", "evidence": {}}
    e["v2split"] = {"cut_id": V2CUT, "view_ids": [f"{h}__r0__f{f}__l7bsplit2_{f:012d}" for f in range(5)],
                    "record": {"path": "x", "sha256": "e" * 64}}
    return e


def t1_inputs(M=A, e=None):
    e = e or g_entry()
    return tool.attach_groups(tool.l7b_inputs(e, M, META, "m" * 64, "a" * 64, base.COV, "frozen"), e, M)


def l7a_inputs(V=VALUES, e=None):
    e = e or g_entry()
    inp = l7a.l7a_inputs(e, V, RAW_SPECS, "a" * 64, "m" * 64, base.COV, "incoming", {"sha256": "r" * 64}, {}, base.REMATCH)
    return l7a.local.attach_groups(inp, e, A)                        # group values: the T1 column bytes of A


_CACHE = {}


def computed(mode="L7b"):
    if mode not in _CACHE:
        e = g_entry()
        inp = t1_inputs(e=e) if mode == "L7b" else l7a_inputs(e=e)
        _CACHE[mode] = tool.compute_dataset(e, inp, mode, set(), {"code": "x"}, version=tool.SPLIT_V2G)
    return _CACHE[mode]


def ref_folds(M, specs, partition, gv=GV):
    return g.split(H, list(range(N)), rowsel_raw.feature_bytes(M, specs), gv, partition)


class Groups(unittest.TestCase):
    def test_annotated_groups_never_straddle_and_v2_without_groups_does(self):
        for M, specs, part in ((A, SPECS_T1, "t1"), (VALUES, RAW_SPECS, "l7a")):
            folds, srep = tool.v2g_split(H, M, specs, part, GV, SPEC)
            self.assertEqual(folds, ref_folds(M, specs, part))
            self.assertEqual(srep["annotated_groups_straddling_folds"], 0)
            self.assertEqual(srep["groups_straddling_folds"], 0)
            self.assertEqual(tool.split_gate_v2g(srep), [])
            f2, _ = tool.v2_split(H, M, specs, part)                    # MUST FAIL: v2 ignores the annotation
            self.assertGreater(g.straddling_groups(f2, list(range(N)), GV), 0)

    def test_groups_union_with_feature_duplicates(self):
        comps = g.components(list(range(N)), rowsel_raw.feature_bytes(A, SPECS_T1), GV)
        self.assertLessEqual(len(comps), REGIONS)
        for i in range(N - base.FEAT_DUP, N):                            # feature twins (targets differ): one component
            twins = [j for j in range(N) if j != i and np.array_equal(A[j, :-1], A[i, :-1])]
            ci = next(c for c in comps if i in c)
            self.assertTrue(all(j in ci for j in twins))

    def test_without_group_values_the_hook_is_split_v2(self):
        for M, specs, part in ((A, SPECS_T1, "t1"), (VALUES, RAW_SPECS, "l7a")):
            fn = tool.split_fn_v2g(part, None)
            self.assertEqual(fn(H, np.arange(N), M, specs), tool.v2_split(H, M, specs, part)[0])

    def test_partition_and_group_spec_are_bound(self):
        folds, srep = tool.v2g_split(H, A, SPECS_T1, "t1", GV, SPEC)
        self.assertNotEqual(folds, ref_folds(A, SPECS_T1, "raw"))
        self.assertEqual(srep["sha256"], g.split_sha256(H, "t1", SPEC, folds))
        self.assertNotEqual(srep["sha256"], g.split_sha256(H, "t1", g.group_spec(1, "n1"), folds))
        self.assertNotEqual(srep["sha256"], g.split_sha256(H, "raw", SPEC, folds))


class TargetFree(unittest.TestCase):
    def test_y_and_group_preserving_outer_test_changes_never_move_it(self):
        for M, specs, part in ((A, SPECS_T1, "t1"), (VALUES, RAW_SPECS, "l7a")):
            base_f, srep = tool.v2g_split(H, M, specs, part, GV, SPEC)
            self.assertTrue(srep["target_invariant"] and srep["value_check_invariant"], srep["value_check"])
            self.assertEqual([c["fold"] for c in srep["value_check"]["folds"]], [0, 1, 2, 3, 4])
            for f in (0, 3):
                for kind in v2.CF_KINDS:
                    M2, s2 = v2.perturb_fold_values(M, specs, np.asarray(base_f[f]), kind)
                    got, _ = tool.v2g_split(H, M2, s2, part, GV, SPEC)
                    self.assertEqual(got, base_f, (part, f, kind))
                M3 = np.array(M, copy=True)
                M3[np.asarray(base_f[f]), -1] = np.random.default_rng(5).normal(size=len(base_f[f]))
                self.assertEqual(tool.v2g_split(H, M3, specs, part, GV, SPEC)[0], base_f)

    def test_a_group_key_that_reads_y_is_refused(self):
        """MUST FAIL: the 'group' value carries y; the target check sees it and the dataset is refused."""
        def leaky(partition, group_values):
            def fn(h, positions, values, specs, folds=5):
                yv = [np.float64(v).tobytes() for v in np.asarray(values)[:, -1]]
                return g.split(h, [int(p) for p in positions], rowsel_raw.feature_bytes(values, specs), yv, partition, folds)
            return fn
        with mock.patch.object(tool, "split_fn_v2g", leaky):
            _, srep = tool.v2g_split(H, A, SPECS_T1, "t1", GV, SPEC)
            self.assertIn("split_reads_target", tool.split_gate_v2g(srep))
            with self.assertRaises(tool.SplitV2Error) as cm:
                tool.compute_dataset(g_entry(), t1_inputs(), "L7b", set(), {"code": "x"}, version=tool.SPLIT_V2G)
            self.assertEqual(cm.exception.reason, "split_check_failed")

    def test_a_split_that_ignores_the_groups_is_refused(self):
        with mock.patch.object(tool, "split_fn_v2g", lambda partition, gv: tool.split_fn(partition)):
            with self.assertRaises(tool.SplitV2Error) as cm:
                tool.compute_dataset(g_entry(), t1_inputs(), "L7b", set(), {"code": "x"}, version=tool.SPLIT_V2G)
        self.assertEqual(cm.exception.reason, "split_check_failed")
        self.assertIn("annotated_groups_straddle_folds", cm.exception.detail)

    def test_group_values_are_the_feature_columns_exact_bytes(self):
        gv = tool.t1_group_values(A, GCOL)
        self.assertEqual(gv, [np.float64(float(v) + 0.0).tobytes() for v in A[:, GCOL]])
        M = np.array(A, copy=True)
        M[:, -1] = np.random.default_rng(2).normal(size=N)                 # y never reaches the group key
        self.assertEqual(tool.t1_group_values(M, GCOL), gv)

    def test_the_group_column_is_never_the_target(self):
        with self.assertRaises(tool.SplitV2Error) as cm:
            tool.t1_group_values(A, A.shape[1] - 1)
        self.assertEqual(cm.exception.reason, "group_column_not_a_feature")


class Identity(unittest.TestCase):
    def test_cut_block_split_record_and_contexts(self):
        for mode, part, tag, schema in (("L7b", "t1", "l7bsplit2g_", "jope.dl-v2-local-context/v2split"),
                                        ("L7a", "l7a", "l7asplit2g_", "jope.dl-v2-l7a-context/v2split")):
            res = computed(mode)
            M, specs = (A, SPECS_T1) if mode == "L7b" else (VALUES, RAW_SPECS)
            folds = ref_folds(M, specs, part)
            want = "trainonly-split-v2g:" + g.split_sha256(H, part, SPEC, folds)
            self.assertEqual(len(res["results"]), 5)
            for f, (vid, files, rec) in enumerate(res["results"]):
                cut = rec["cut"]
                self.assertEqual(cut["cut_id"], want)
                self.assertRegex(cut["cut_id"], r"^trainonly-split-v2g:[0-9a-f]{64}$")
                self.assertEqual((cut["schema"], cut["partition"], cut["group_column"], cut["replaces_cut"],
                                  cut["comparable_with_replaced"]),
                                 ("rowsel-split-v2-groups", part, SPEC, ["d" * 64, V2CUT], False))
                sp = rec["split"]
                self.assertEqual((sp["annotated_groups_straddling_folds"], sp["groups_straddling_folds"], sp["target_invariant"],
                                  sp["group_column"]), (0, 0, True, SPEC))
                self.assertEqual(rec["schema"], schema)
                self.assertEqual(rec["split_version"], "rowsel-split-v2-groups")
                ident = hashlib.sha256(tool.canon(rec["identity_body"])).hexdigest()
                self.assertEqual(vid, f"{H}__r0__f{f}__{tag}{ident[:12]}")
                self.assertEqual((rec["identity_body"]["cut_id"], rec["identity_body"]["group_column"]), (want, SPEC))
                self.assertEqual(rec["outer_test_indices_sha256"], hashlib.sha256(json.dumps(folds[f]).encode()).hexdigest())
                self.assertNotIn(vid, g_entry()["v2split"]["view_ids"])
                self.assertEqual((rec["group_union"], rec["group_annotation_dropped"], rec["group_annotation_kept"]),
                                 (True, False, True))
                self.assertEqual(rec["v2split_superseded"]["cut_id"], V2CUT)
                self.assertEqual(rec["universe"]["group_column"], SPEC)
                self.assertTrue(rec["counterfactual"]["train_side_invariant"])

    def test_v2_cut_and_identity_are_unchanged(self):
        res = tool.compute_dataset(base.plan_entry(), base.t1_inputs(), "L7b", set(), {"code": "x"})
        self.assertEqual([r[0] for r in res["results"]], [r[0] for r in base.computed("L7b")["results"]])
        self.assertTrue(res["cut"]["cut_id"].startswith("trainonly-split-v2:"))
        self.assertNotIn("group_column", res["results"][0][2]["identity_body"])


class Dataset(unittest.TestCase):
    def test_outer_test_values_never_reach_the_train_side_end_to_end(self):
        for mode in ("L7b", "L7a"):
            b = computed(mode)
            M, specs, part = (A, SPECS_T1, "t1") if mode == "L7b" else (VALUES, RAW_SPECS, "l7a")
            folds = ref_folds(M, specs, part)
            f = 2
            M2 = base.remap_fold(M, folds[f], (0, 1))                   # non-group numeric features and y of fold f only
            e = g_entry()
            inp = t1_inputs(M2, e) if mode == "L7b" else l7a_inputs(M2, e)
            got = tool.compute_dataset(e, inp, mode, set(), {"code": "x"}, version=tool.SPLIT_V2G)
            self.assertEqual(got["cut"]["cut_id"], b["cut"]["cut_id"])
            self.assertEqual(base.train_side_of(got, f, mode, M2, specs), base.train_side_of(b, f, mode, M, specs))

    def test_leaky_builders_are_refused(self):
        orig = tool.l7b_artifacts

        def leaky(M, meta, R, T):
            k = len(T) // 2
            return orig(M, meta, sorted(list(R)[k:] + list(T)[:k]), sorted(list(R)[:k] + list(T)[k:]))
        with mock.patch.object(tool, "l7b_artifacts", leaky):
            with self.assertRaises(tool.SplitV2Error) as cm:
                tool.compute_dataset(g_entry(), t1_inputs(), "L7b", set(), {"code": "x"}, version=tool.SPLIT_V2G)
        self.assertEqual(cm.exception.reason, "counterfactual_failed")


def census(columns, none=False, target_cols=()):
    return {"reproducing_group_columns": list(columns), "match_group_none": none,
            "group_column_candidates": [{"column": c, "match": True, "is_target": c in target_cols} for c in columns]}


class FewComponents(unittest.TestCase):
    """4 annotated groups: 4 components, so fold 4 is empty (as the registry's effective_folds 4): 4 contexts, the empty
    fold recorded; an empty fold that fewer components do not explain is still refused."""

    def setUp(self):
        self.M = np.array(A, copy=True)
        self.M[:, GCOL] = np.floor(np.arange(N) * 4 / N) / 3.0              # 4 contiguous regions
        self.gv = tool.t1_group_values(self.M, GCOL)

    def test_four_components_build_four_contexts(self):
        e = g_entry()
        res = tool.compute_dataset(e, t1_inputs(self.M, e), "L7b", set(), {"code": "x"}, version=tool.SPLIT_V2G)
        sp = res["split"]
        self.assertEqual((sp["components"], sp["empty_folds"], sp["effective_folds"], sp["empty_folds_allowed"]), (4, [4], 4, True))
        self.assertEqual([r[2]["fold"] for r in res["results"]], [0, 1, 2, 3])
        self.assertEqual(res["epoch0_replaced"]["new_contexts"], 4)
        self.assertEqual([c["fold"] for c in sp["value_check"]["folds"]], [0, 1, 2, 3])
        self.assertTrue(sp["value_check_invariant"])

    def test_an_unexplained_empty_fold_is_refused(self):
        def hole(partition, gv):                                        # MUST FAIL: fold 2 left empty with >= 5 components
            def fn(h, positions, values, specs, folds=5):
                f = g.split(h, [int(p) for p in positions], rowsel_raw.feature_bytes(values, specs), gv, partition, folds)
                f[1] = sorted(f[1] + f[2])
                f[2] = []
                return f
            return fn
        with mock.patch.object(tool, "split_fn_v2g", hole):
            with self.assertRaises(tool.SplitV2Error) as cm:
                tool.compute_dataset(g_entry(), t1_inputs(), "L7b", set(), {"code": "x"}, version=tool.SPLIT_V2G)
        self.assertIn("empty_outer_fold", cm.exception.detail)

    def test_v2_still_refuses_any_empty_fold(self):
        _, srep = tool.v2_split(H, A, SPECS_T1, "t1")
        srep = dict(srep, fold_sizes=[10, 0, 5, 5, 5])
        self.assertIn("empty_outer_fold", tool.split_gate(srep))

    def test_resume_counts_the_expected_contexts(self):
        out = Path(tempfile.mkdtemp(prefix="nonraw-v2gsplit-test-", dir=os.environ.get("TMPDIR")))
        e = g_entry()
        rec = tool.run_dataset_v2(e, out, "L7b", lambda e: t1_inputs(self.M, e), set(), {"code": "x"}, log=lambda s: None,
                                  version=tool.SPLIT_V2G)
        self.assertEqual((rec["status"], rec["expected_contexts"], len(rec["contexts"])), ("done", 4, 4))
        self.assertTrue(tool.dataset_done(out, H, {"code": "x"}))


class Evidence(unittest.TestCase):
    names = ["n0", "n1", "n2", "c0", "target"]

    def test_single_reproducing_feature_column_is_accepted(self):
        r = tool.resolve_group_column(census([2]), {"columns": [2], "n_columns": 5}, self.names)
        self.assertEqual((r["spec"], r["source"]), (SPEC, "inferred_single_column_registry_port_reproduced"))
        self.assertEqual(r["evidence"]["census_reproducing_group_columns"], [2])

    def test_refusals(self):
        cases = ((census([]), {"columns": [2], "n_columns": 5}, "group_column_not_reproduced"),
                 (census([2], none=True), {"columns": [2], "n_columns": 5}, "group_column_not_reproduced"),
                 (census([4], target_cols=(4,)), {"columns": [4], "n_columns": 5}, "group_column_is_target_or_not_a_feature"),
                 (census([1]), {"columns": [2], "n_columns": 5}, "census_and_scope_receipt_disagree"),
                 (None, {"columns": [2], "n_columns": 5}, "no_census_record"))
        for rec, ga, reason in cases:
            self.assertEqual(tool.resolve_group_column(rec, ga, self.names).get("refused"), reason, reason)

    def test_ambiguous_columns_need_identical_components(self):
        same = {1: GV, 2: GV}
        r = tool.resolve_group_column(census([1, 2]), {"columns": [1, 2], "n_columns": 5}, self.names, lambda c: same[c])
        self.assertEqual((r["spec"]["index"], r["source"]), (1, "ambiguous_columns_identical_components"))
        diff = {1: tool.t1_group_values(A, 1), 2: GV}
        r = tool.resolve_group_column(census([1, 2]), {"columns": [1, 2], "n_columns": 5}, self.names, lambda c: diff[c])
        self.assertEqual(r.get("refused"), "ambiguous_group_column_components_differ")
        r = tool.resolve_group_column(census([1, 2]), {"columns": [1, 2], "n_columns": 5}, self.names)
        self.assertEqual(r.get("refused"), "ambiguous_group_column_unresolved")

    def test_ambiguous_plan_entry_is_resolved_on_the_universe_or_refused(self):
        e = g_entry()
        e["group_column"] = {"ambiguous": [1, 2], "names": self.names, "evidence": {"census_candidates": []}}
        with self.assertRaises(tool.SplitV2Error) as cm:                 # columns 1 and 2 give different components
            tool.attach_groups(tool.l7b_inputs(e, A, META, "m" * 64, "a" * 64, base.COV, "frozen"), e, A)
        self.assertEqual(cm.exception.reason, "ambiguous_group_column_components_differ")
        M = np.array(A, copy=True)
        M[:, 1] = M[:, GCOL]                                            # identical columns: accepted, lowest index
        e = g_entry()
        e["group_column"] = {"ambiguous": [1, 2], "names": self.names, "evidence": {"census_candidates": []}}
        inp = tool.attach_groups(tool.l7b_inputs(e, M, META, "m" * 64, "a" * 64, base.COV, "frozen"), e, M)
        self.assertEqual(e["group_column"]["spec"]["index"], 1)
        self.assertEqual(inp["group_values"], tool.t1_group_values(M, 1))


class Writes(unittest.TestCase):
    def test_v2g_dataset_writes_contexts_with_group_fields(self):
        out = Path(tempfile.mkdtemp(prefix="nonraw-v2gsplit-test-", dir=os.environ.get("TMPDIR")))
        e = g_entry()
        order = []
        real = tool.atomic

        def spy(path, data):
            order.append(Path(path).name)
            return real(path, data)
        with mock.patch.object(tool, "atomic", spy):
            rec = tool.run_dataset_v2(e, out, "L7b", lambda e: t1_inputs(e=e), set(), {"code": "x"}, log=lambda s: None,
                                      version=tool.SPLIT_V2G)
        self.assertEqual(rec["status"], "done", rec.get("detail"))
        self.assertEqual((rec["split_version"], rec["group_column"]["spec"], rec["v2split_superseded"]["cut_id"]),
                         ("rowsel-split-v2-groups", SPEC, V2CUT))
        writes = [n for n in order if n != f"{H}.json"]
        self.assertEqual(writes[-5:], [".done"] * 5)
        self.assertNotIn(".done", writes[:-5])
        for c in rec["contexts"]:
            cj = json.loads((out / c["view_id"] / "context.json").read_bytes())
            self.assertEqual((cj["cut"]["cut_id"], cj["group_union"]), (rec["cut"]["cut_id"], True))
            for name, m in cj["files"].items():
                self.assertEqual(hashlib.sha256((out / c["view_id"] / name).read_bytes()).hexdigest(), m["sha256"])


if __name__ == "__main__":
    unittest.main()
