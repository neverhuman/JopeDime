"""RAW-GROUP (a)/(b) and LINEAR-DETERMINED on the raw route: tools/v2_rowsel_raw.py --split-version rowsel-split-v2-groups.
Every invariance check is paired with a control that MUST FAIL it.

  Build       groups never straddle; split sha binds the group spec; cut 'trainonly-split-v2g:' replaces the as-built v2
              contexts; new view ids; the split ignoring the group (plain v2) straddles and is refused
  Admission   features only before the split: ambiguous candidates with different component structures are refused,
              identical ones admitted (lowest index); target_tied_group runs PER CONTEXT on its own outer-train rows
  Gates       targets never move the split; group-preserving outer-test changes never move it and a content-ranked split
              is refused; empty folds only trailing below 5 components and equal to the plan's
  Linear      exactly linear y refused (target_linear_in_features, nothing written); near-linear flagged; v1 replay untouched
  Plan        yes -> v2g entry with candidates, expected empty folds and the as-built contexts it replaces; near -> the
              create-only group-annotation-dropped record; summary + superseded record + .complete
"""
import argparse
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

from dltrain import rowsel_raw, rowsel_split_v2_groups as g, t1raw, target_linear as tl
from tests import test_v2_rowsel_raw as traw

tool = traw.tool
H, SPECS, VALUES = traw.H, traw.SPECS, traw.VALUES
CODES = {"tools/v2_rowsel_raw.py": "1" * 64}
GC = {"index": 3, "name": "c0"}                                           # the categorical fixture column: 6 groups
REPL = {"tree": "fixture-v2", "dataset_record": {"path": "fixture", "sha256": "7" * 64}, "split_sha256": "9" * 64,
        "view_ids": [f"{H}__r0__f{f}__rowselraw1_asbuilt{f}" for f in range(5)]}


def entry(**kw):
    e = dict(traw.plan_entry(), split_version=tool.SPLIT_V2G, group_column=dict(GC), group_candidates=[3], group_candidate_names=["c0"],
             empty_folds=[], effective_folds=5, replaces_v2=REPL)
    e.update(kw)
    return e


def with_values(V):
    return lambda inp: (np.array(V, copy=True), SPECS, {"values_sha256": "0" * 64, "read_report_equal": True})


def three_level_values():
    V = np.array(VALUES, copy=True)
    codes = sorted({c for c in V[:, 3].tolist() if c != t1raw.MISSING_CODE})[:3]
    V[:, 3] = np.asarray(codes)[np.arange(len(V)) % 3]
    return V


def linear_values(noise=0.0):
    V = np.array(VALUES, copy=True)
    V[:, :3] = np.nan_to_num(V[:, :3])
    V[:, 4] = 2.0 * V[:, 0] + 0.01 * V[:, 1] - 0.5 * V[:, 2] + 3.0 + (np.random.default_rng(3).normal(0, noise, len(V)) if noise else 0.0)
    return V


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="rowselraw-v2g-", dir=os.environ.get("TMPDIR")))
        self.shard = self.dir / "shard-0"; self.shard.mkdir()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_ds(self, e=None, V=None, shard=None, version=None):
        e = e or entry()
        with mock.patch.object(tool, "load_inputs", traw.fake_inputs), \
                mock.patch.object(tool, "rematch", traw.fake_rematch if V is None else with_values(V)):
            return tool.run_dataset(e, shard or self.shard, CODES, None, None, None, set(), traw._Guard(), log=lambda s: None,
                                    split_version=version or e.get("split_version", tool.SPLIT_V1))

    def ctxs(self, rec, shard=None):
        return [json.loads(((shard or self.shard) / c["view_id"] / "context.json").read_bytes()) for c in rec["contexts"]]

    def assert_refused(self, rec, reason, shard=None):
        self.assertEqual((rec["status"], rec["reason"]), ("failed", reason), rec.get("detail"))
        self.assertEqual([d for d in (shard or self.shard).iterdir() if d.name != "_datasets"], [])


class Build(Base):
    def test_groups_never_straddle_and_the_cut_binds_the_group_spec(self):
        rec = self.run_ds()
        self.assertEqual((rec["status"], len(rec["contexts"])), ("done", 5), rec.get("detail"))
        prep = rowsel_raw.prepare_universe(H, VALUES, SPECS)
        gv = tool.raw_group_values(prep.values, prep.specs, 3, "c0")
        folds = g.split(H, prep.positions.tolist(), rowsel_raw.feature_bytes(prep.values, prep.specs), gv, "raw")
        self.assertEqual(g.straddling_groups(folds, prep.positions.tolist(), gv), 0)
        sha = g.split_sha256(H, "raw", GC, folds)
        self.assertEqual((rec["split"]["schema"], rec["split"]["sha256"], rec["split"]["annotation_groups_straddling_folds"]),
                         (tool.SPLIT_V2G, sha, 0))
        for c in self.ctxs(rec):
            self.assertEqual((c["split_version"], c["cut"]["cut_id"], c["cut"]["comparable_with_replaced"], c["cut"]["group_column"]),
                             (tool.SPLIT_V2G, "trainonly-split-v2g:" + sha, False, GC))
            self.assertEqual(c["cut"]["replaces_cut"][0]["view_ids"], REPL["view_ids"])
            self.assertNotIn(c["view_id"], REPL["view_ids"])
            self.assertEqual(c["rowsel"]["split"]["schema"], tool.SPLIT_V2G)
            self.assertIn("target_linear", c)
            self.assertEqual(tool.verify_v2g_context(self.shard / c["view_id"], entry(), rec), [])
        v2 = self.run_ds(dict(traw.plan_entry(), split_version=tool.SPLIT_V2), shard=self.dir / "v2")
        self.assertFalse({c["view_id"] for c in v2["contexts"]} & {c["view_id"] for c in rec["contexts"]})

    def test_the_split_ignoring_the_group_straddles_and_is_refused(self):
        """MUST FAIL: split v2 (no group) puts one c0 value in several folds."""
        real = g.split
        with mock.patch.object(tool.rowsel_split_v2_groups, "split", lambda h, p, fb, gv, part, folds=5: real(h, p, fb, None, part, folds)):
            rec = self.run_ds()
        self.assert_refused(rec, "annotation_groups_straddle_folds")

    def test_group_values_missing_is_one_group(self):
        V = np.array([[1.0, 2.0, np.nan, t1raw.MISSING_CODE, 0.5], [1.0, 2.0, 3.0, np.nan, 0.6], [1.0, 2.0, 3.0, 2.0, 0.7],
                      [1.0, 2.0, -0.0, 2.0, 0.8]])
        gv = tool.raw_group_values(V, SPECS, 3, "c0")
        self.assertEqual(gv[0], gv[1]); self.assertEqual(gv[0], tool.RAW_MISSING_GROUP)
        self.assertEqual(gv[2], np.float64(2.0).tobytes())
        self.assertNotEqual(len(tool.RAW_MISSING_GROUP), 8)
        n2 = tool.raw_group_values(V, SPECS, 2, "n2")
        self.assertEqual((n2[0], n2[1], n2[3]), (tool.RAW_MISSING_GROUP, np.float64(3.0).tobytes(), np.float64(0.0).tobytes()))

    def test_the_group_column_must_be_a_named_feature(self):
        self.assert_refused(self.run_ds(entry(group_column={"index": 4, "name": "target"}, group_candidates=[4],
                                              group_candidate_names=["target"])), "group_column_not_a_feature")
        self.assert_refused(self.run_ds(entry(group_column={"index": 3, "name": "n0"}, group_candidate_names=["n0"])),
                            "group_column_name_mismatch")
        self.assert_refused(self.run_ds(entry(group_column=None)), "group_column_not_given")


class Admission(Base):
    def test_a_tied_column_refuses_every_context_on_its_own_train_rows(self):
        """A target-binned group column: every context's OWN outer-train rows show the tie, so every context is refused
        (group_column_tied_to_target) and nothing is written. The guard reads only that context's outer-train targets
        (tests.test_v2_rowsel_raw_fold_isolation; PR #246 review 5437848458); the split never receives y."""
        V = np.array(VALUES, copy=True)
        codes = sorted({c for c in V[:, 3].tolist() if c != t1raw.MISSING_CODE})
        V[:, 3] = np.asarray(codes)[np.clip(np.digitize(V[:, 4], np.quantile(V[:, 4], [0.2, 0.4, 0.6, 0.8])), 0, 4)]
        rec = self.run_ds(V=V)
        self.assert_refused(rec, "group_column_tied_to_target")
        self.assertIn("group_is_binned_target", rec["detail"])
        self.assertEqual([(c["fold"], c["refused"]) for c in rec["contexts"]],
                         [(f, "group_column_tied_to_target") for f in range(5)])
        self.assertIn("per context", rec["group_admission"]["target_tie"])       # no dataset-level, all-y tie decision

    def test_ambiguous_candidates_need_identical_structure(self):
        V = np.array(VALUES, copy=True)
        V[:, 2] = np.where(V[:, 3] == t1raw.MISSING_CODE, np.nan, V[:, 3] * 7.0 + 1.0)    # n2: a relabelling of c0
        ok = self.run_ds(entry(group_column={"index": 2, "name": "n2"}, group_candidates=[2, 3], group_candidate_names=["n2", "c0"]), V=V)
        self.assertEqual(ok["status"], "done", ok.get("detail"))
        self.assertEqual(len(set(ok["group_admission"]["component_structure_sha256"].values())), 1)
        V2 = np.array(VALUES, copy=True)
        V2[:, 2] = np.where(V2[:, 3] == t1raw.MISSING_CODE, np.nan, np.floor(V2[:, 3] / 2))  # coarser: a different structure
        bad = self.run_ds(entry(group_column={"index": 2, "name": "n2"}, group_candidates=[2, 3], group_candidate_names=["n2", "c0"]),
                          V=V2, shard=self.dir / "bad")
        self.assert_refused(bad, "ambiguous_group_columns_with_different_components", self.dir / "bad")
        self.assert_refused(self.run_ds(entry(group_column={"index": 3, "name": "c0"}, group_candidates=[2, 3],
                                              group_candidate_names=["n2", "c0"]), V=V, shard=self.dir / "x"),
                            "group_candidates_inconsistent", self.dir / "x")


class Gates(Base):
    def test_targets_never_move_the_split(self):
        rec = self.run_ds()
        self.assertTrue(rec["split"]["target_check"]["target_invariant"])
        V = np.array(VALUES, copy=True); V[:, 4] = np.random.default_rng(5).normal(size=len(V))
        rec2 = self.run_ds(V=V, shard=self.dir / "y")
        self.assertEqual(rec2["split"]["sha256"], rec["split"]["sha256"])

    def test_group_preserving_changes_never_move_the_split_and_content_ranking_is_refused(self):
        rec = self.run_ds()
        self.assertTrue(rec["split"]["group_value_check_fold0"]["valid"] and rec["split"]["group_value_check_fold0"]["invariant"])

        def content_ranked(h, positions, fb, gv, partition, folds=5):     # CONTROL: components ranked by group VALUE
            comps = g.components(positions, fb, gv)
            of = dict(zip(positions, gv))
            order = sorted(comps, key=lambda c: (hashlib.sha256(b"".join(sorted(of[p] for p in c))).digest(), min(c)))
            sizes, out = [0] * folds, {f: [] for f in range(folds)}
            for c in order:
                f = sizes.index(min(sizes)); out[f] += c; sizes[f] += len(c)
            return {f: sorted(v) for f, v in out.items()}
        with mock.patch.object(tool.rowsel_split_v2_groups, "split", content_ranked):
            bad = self.run_ds(shard=self.dir / "c")
        self.assert_refused(bad, "split_moves_under_group_preserving_changes", self.dir / "c")

    def test_a_change_that_merges_groups_is_not_valid(self):
        """The validity of the group-value check is the component structure WITH groups: moving a fold's group values onto
        a value another fold holds merges groups and must be invalid (never counted as invariance)."""
        prep = rowsel_raw.prepare_universe(H, VALUES, SPECS)
        split_fn, _ = tool.v2g_functions(entry())
        base = split_fn(H, prep.positions, prep.values, prep.specs)
        in0 = set(prep.values[prep.rows_of(base[0]), 3].tolist())
        outside = next(x for x in np.unique(prep.values[:, 3]) if x not in in0)
        orig = tool.rowsel_raw_split_v2.perturb_fold_values

        def merging(values, specs, rows, kind, salt=None):
            V, s = orig(values, specs, rows, kind)
            V[np.asarray(rows), 3] = outside
            return V, s
        self.assertTrue(tool.v2g_group_value_check(split_fn, prep, 3)["valid"])
        with mock.patch.object(tool.rowsel_raw_split_v2, "perturb_fold_values", merging):
            r = tool.v2g_group_value_check(split_fn, prep, 3)
        self.assertFalse(r["valid"]); self.assertFalse(r["invariant"])

    def test_empty_folds_only_trailing_below_five_components_and_as_planned(self):
        V = three_level_values()
        rec = self.run_ds(entry(empty_folds=[3, 4], effective_folds=3), V=V)
        self.assertEqual((rec["status"], len(rec["contexts"]), rec["split"]["empty_folds"], rec["split"]["effective_folds"]),
                         ("done", 3, [3, 4], 3), rec.get("detail"))
        self.assertEqual(sorted(c["fold"] for c in rec["contexts"]), [0, 1, 2])
        for c in self.ctxs(rec):
            self.assertEqual((c["cut"]["folds"], c["cut"]["empty_folds"], c["cut"]["effective_folds"]), (5, [3, 4], 3))
        self.assert_refused(self.run_ds(entry(empty_folds=[4], effective_folds=4), V=V, shard=self.dir / "p"),
                            "empty_folds_differ_from_plan", self.dir / "p")
        ok = {"components": 3, "fold_sizes": [5, 4, 3, 0, 0]}
        self.assertTrue(tool.v2g_empty_folds_allowed(ok))
        for bad in ({"components": 3, "fold_sizes": [5, 0, 3, 4, 0]}, {"components": 5, "fold_sizes": [5, 4, 3, 2, 0]},
                    {"components": 1, "fold_sizes": [9, 0, 0, 0, 0]}, {"fold_sizes": [5, 4, 3, 0, 0]}):
            self.assertFalse(tool.v2g_empty_folds_allowed(bad), bad)


class Linear(Base):
    def test_an_exactly_linear_target_is_refused_on_v2_and_v2g(self):
        for e in (dict(traw.plan_entry(), split_version=tool.SPLIT_V2), entry()):
            with self.subTest(version=e["split_version"]):
                sd = self.dir / f"lin-{e['split_version']}"; sd.mkdir()
                rec = self.run_ds(e, V=linear_values(), shard=sd)
                self.assert_refused(rec, tl.REFUSE, sd)
                self.assertTrue(all(c["target_linear"]["verdict"] == tl.REFUSE for c in rec["contexts"]))

    def test_near_linear_is_built_and_flagged_and_v1_is_untouched(self):
        rec = self.run_ds(dict(traw.plan_entry(), split_version=tool.SPLIT_V2), V=linear_values(noise=1e-4))
        self.assertEqual(rec["status"], "done", rec.get("detail"))
        self.assertEqual({c["target_linear"]["verdict"] for c in rec["contexts"]}, {tl.REVIEW})
        self.assertEqual({c["target_linear"]["verdict"] for c in self.ctxs(rec)}, {tl.REVIEW})
        for c in rec["contexts"]:
            enc = gzip.decompress((self.shard / c["view_id"] / "encoder.jtt1.gz").read_bytes())
            self.assertEqual(json.loads((self.shard / c["view_id"] / "context.json").read_bytes())["target_linear"], tl.check_jtt1(enc))
        v1 = self.run_ds(traw.plan_entry(), V=linear_values(), shard=self.dir / "v1", version=tool.SPLIT_V1)
        self.assertEqual(v1["status"], "done")                             # the v1 replay is byte-identical: no gate
        self.assertNotIn("target_linear", v1["contexts"][0])


class PlanAndSummary(Base):
    def test_plan_yes_near_and_the_dropped_record(self):
        base = self.dir / "ROWSEL-RAW-v2"; base.mkdir()
        rec_p = self.dir / "rec.json"; rec_p.write_text("{}")
        rsha = hashlib.sha256(rec_p.read_bytes()).hexdigest()
        bp = dict(traw.plan_entry(), split_version=tool.SPLIT_V2, est_cpu_s=3.0, shard=0)
        (base / "plan.json").write_text(json.dumps({"of": 1, "split_version": tool.SPLIT_V2, "inputs": {},
                                                    "datasets": [bp, dict(bp, dataset_hash="1" * 16), dict(bp, dataset_hash="2" * 16)]}))

        def ds(h, verdict, groups, cands=(3,), names=("c0",)):
            return {"dataset_hash": h, "tree": "ROWSEL-RAW-v2", "status": "built", "view_ids": REPL["view_ids"],
                    "dataset_record": {"path": str(rec_p), "sha256": rsha, "status": "done"}, "recorded_split": {"sha256": "9" * 64},
                    "carry": {"verdict": f"{verdict}: fixture", "per_column": []},
                    "registry_group_column": {"candidates_reproducing": list(cands), "t1_name": list(names), "ambiguous": len(cands) > 1},
                    "as_built_group_straddling": [{"t1_index": c, "groups": groups} for c in cands]}
        src = self.dir / "src.json"
        src.write_text(json.dumps({"datasets": [ds(H, "yes", 6), ds("1" * 16, "yes", 3, (2, 3), ("n2", "c0")), ds("2" * 16, "near", 4),
                                                {"dataset_hash": "3" * 16, "tree": None, "status": "not_built"}]}))
        out = self.dir / "v2g"
        with mock.patch.object(tool, "RAW_GROUP_SOURCE", src), \
                mock.patch.object(tool, "RAW_GROUP_SOURCE_SHA256", hashlib.sha256(src.read_bytes()).hexdigest()), \
                mock.patch.object(tool, "default_out", lambda route, v: base), mock.patch("builtins.print"):
            self.assertEqual(tool.plan_v2g(argparse.Namespace(out=str(out), route="l7a", of=1)), 0)
        pl = json.loads((out / "plan.json").read_bytes())
        self.assertEqual((pl["split_version"], pl["counts"]["datasets"], pl["counts"]["expected_new_contexts"]), (tool.SPLIT_V2G, 2, 8))
        e1 = {e["dataset_hash"]: e for e in pl["datasets"]}["1" * 16]
        self.assertEqual((e1["group_column"], e1["group_candidates"], e1["empty_folds"], e1["effective_folds"], e1["split_version"]),
                         ({"index": 2, "name": "n2"}, [2, 3], [3, 4], 3, tool.SPLIT_V2G))
        self.assertEqual(e1["replaces_v2"]["view_ids"], REPL["view_ids"])
        drop = json.loads((out / "group-annotation-dropped.json").read_bytes())
        self.assertEqual([(d["dataset_hash"], d["flag"]) for d in drop["datasets"]], [("2" * 16, "group_annotation_dropped")])
        self.assertIn("excluded", drop["datasets"][0]["headline_outer_test"])
        self.assertIn("included", drop["datasets"][0]["training"])
        rec_p.write_text("{\"changed\": true}")                                # an as-built record that drifted is refused
        with mock.patch.object(tool, "RAW_GROUP_SOURCE", src), \
                mock.patch.object(tool, "RAW_GROUP_SOURCE_SHA256", hashlib.sha256(src.read_bytes()).hexdigest()), \
                mock.patch.object(tool, "default_out", lambda route, v: base), self.assertRaises(SystemExit):
            tool.plan_v2g(argparse.Namespace(out=str(self.dir / "v2g-b"), route="l7a", of=1))

    def test_summary_superseded_record_and_complete(self):
        tree = self.dir / "tree"; (tree / "shard-0").mkdir(parents=True)
        e = dict(entry(), shard=0, est_cpu_s=1.0)
        (tree / "plan.json").write_text(json.dumps({"of": 1, "route": "l7a", "split_version": tool.SPLIT_V2G, "counts": {"datasets": 1},
                                                    "decision": {"keys": ["RAW-GROUP"]}, "datasets": [e]}))
        rec = self.run_ds(e, shard=tree / "shard-0")
        self.assertEqual(rec["status"], "done", rec.get("detail"))
        (tree / "shard-0-of-1.receipt.json").write_text("{}")
        log = self.dir / "t.log"; log.write_text("OK\n")
        mut = self.dir / "m.json"; mut.write_text("{\"totals\": {}}")
        a = argparse.Namespace(out=str(tree), tests_log=str(log), mutants=str(mut), receipt=str(self.dir / "s.json"))
        with mock.patch.object(tool, "code_known", lambda r, c: True), mock.patch("builtins.print"):
            self.assertEqual(tool.summarize_v2g(a), 0)
        doc = json.loads((self.dir / "s.json").read_bytes())
        self.assertEqual((doc["counts"]["done_new_contexts"], doc["counts"]["superseded_v2_contexts"], doc["counts"]["verification_problems"]),
                         (5, 5, 0), doc["problems"])
        sup = json.loads((tree / "superseded-v2.json").read_bytes())
        self.assertEqual((sup["datasets"][0]["superseded_view_ids"], len(sup["datasets"][0]["superseded_by"])), (REPL["view_ids"], 5))
        self.assertTrue((tree / ".complete").exists())
        d = tree / "shard-0" / rec["contexts"][0]["view_id"]
        cj = json.loads((d / "context.json").read_bytes())
        for bad in ({"cut": dict(cj["cut"], comparable_with_replaced=True)}, {"cut": dict(cj["cut"], replaces_cut=[])},
                    {"rowsel": dict(cj["rowsel"], split=dict(cj["rowsel"]["split"], annotation_groups_straddling_folds=1))},
                    {"target_linear": dict(cj["target_linear"], verdict=tl.REFUSE)}):
            (d / "context.json").write_text(json.dumps(dict(cj, **bad)))
            self.assertTrue(tool.verify_v2g_context(d, e, rec), bad)


if __name__ == "__main__":
    unittest.main()
