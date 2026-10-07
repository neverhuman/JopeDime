"""NONRAW-SPLIT-A: tools/v2_rowsel_build.py --split-version rowsel-raw-split-v2 (the T1 routes on the raw route's
target-free split). Every invariance check is paired with a control that MUST FAIL it.

  RegistryVsV2  the registry split (grouped-stratified-kfold/v1 port) MOVES under outer-test target noise when full-row
                duplicates exist; split v2 does NOT move under that noise, nor under duplicate-preserving feature changes;
                the same value check flags the registry split and the v1 feature-hash ordering
  BuilderV2     5 folds per dataset on the v2 folds (partition 't1'), quotas from the new fold sizes, the cut block
                (cut_id = 'trainonly-split-v2:' + rowsel.split.sha256, not comparable with the replaced registry cut),
                dataset-level replacement, new view ids, verify; outer-test changes leave the split and the train side
                identical; the split gate refuses a value-reading split; dataset-atomic writes and resume; rights and
                t1_fallback fields
  Registry      the default route still builds the bytes the previous builder (ed00684a) built
  Scope         the three trees: 173/810, 18/75 (none rebuilt on raw), 42/209
"""
import argparse
import hashlib
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
from tests import registry_split_port as reg
from tests import test_v2_rowsel_build as tb

rb = tb.rb
H = tb.H
LEGACY = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/build/"
              "v2_rowsel_build.build-ed00684a320f079ce9016ee6df0a0668bc431b8e37e77566c0f4805157d0fa92.py")
FULL_ROW_DUPS, FEATURE_DUPS = 60, 40


def dup_matrix(seed=11):
    """The builder fixture with FULL_ROW_DUPS rows that copy an earlier row exactly (target included) and FEATURE_DUPS
    rows that copy only its features (target differs)."""
    m = tb.fixture_matrix()
    rs = np.random.RandomState(seed)
    n = m.shape[0]
    src = rs.choice(n - FULL_ROW_DUPS - FEATURE_DUPS, size=FULL_ROW_DUPS + FEATURE_DUPS, replace=False)
    for k, i in enumerate(range(n - FULL_ROW_DUPS - FEATURE_DUPS, n)):
        m[i] = m[src[k]]
        if k >= FULL_ROW_DUPS:
            m[i, 5] = tb._r9(np.array([min(1.0, m[src[k], 5] + 0.137)]))[0]
    return m


M = dup_matrix()
_ORDER = np.random.RandomState(3).permutation(M.shape[0]).tolist()
REGISTRY_FOLDS = {f: sorted(_ORDER[a:b]) for f, (a, b) in enumerate(((0, 100), (100, 250), (250, 420), (420, 600), (600, 800)))}


def registry_folds_of(m):
    """Repeat-0 registry folds of a T1 matrix printed as T1 prints it (positions = line index)."""
    keys = [reg.dup_key(reg.csv_fields(line)) for line in tb.lines_of(m)]
    k, out, _ = reg.generate_folds(keys, reg.BASE_SEED, repeats=1)
    return {r["fold"]: r["test_indices"] for r in out if r["repeat"] == 0}


def registry_split_fn(h, positions, values, specs, folds=5):
    """CONTROL with split_fn's signature: the registry split over (values printed '%.9g')."""
    pos = np.asarray(positions).tolist()
    fo = registry_folds_of(np.asarray(values, dtype=np.float64))
    return {f: sorted(pos[i] for i in v) for f, v in fo.items()}


def position_split_fn(h, positions, values, specs, folds=5):
    """CONTROL: value-free but ignores duplicate groups (straddles)."""
    out = {f: [] for f in range(folds)}
    for p in np.asarray(positions).tolist():
        out[int(hashlib.sha256(f"{h}|{p}".encode()).digest()[0]) % folds].append(p)
    return out


def moved(a, b):
    fa = {p: f for f, v in a.items() for p in v}
    return sum(1 for f, v in b.items() for p in v if fa.get(p) != f)


def noise_targets(m, rows, seed=5):
    m2 = m.copy()
    rows = np.asarray(sorted(rows))
    m2[rows, 5] = tb._r9(np.clip(m2[rows, 5] + np.random.RandomState(seed).normal(0, 0.05, rows.size), 0, 1))
    return m2


def structure_preserving_features(m, rows):
    """Outer-test feature change that keeps the duplicate structure: numeric columns x -> 1 - x, the category code
    shifted by one (both injective)."""
    m2 = m.copy()
    rows = np.asarray(sorted(rows))
    m2[rows, :4] = tb._r9(1.0 - m2[rows, :4])
    m2[rows, 4] = (np.rint(m2[rows, 4] * 3) + 1) % 4 / 3.0
    return tb._r9(m2)


def partition(m):
    U = tb.universe_of(m)
    p = rb.t1_prep(U)
    return v2.duplicate_partition(p.positions, p.values, p.specs)


def v2_folds_of(m):
    U = tb.universe_of(m)
    p = rb.t1_prep(U)
    return rb.split_v2_t1(H, p.positions, p.values, p.specs)


class RegistryVsV2(unittest.TestCase):
    def test_fixture_has_full_row_and_feature_duplicates(self):
        keys = [reg.dup_key(reg.csv_fields(l)) for l in tb.lines_of(M)]
        self.assertGreaterEqual(len(keys) - len(set(keys)), FULL_ROW_DUPS)
        groups = [g for g in partition(M) if len(g) > 1]
        self.assertGreaterEqual(sum(len(g) - 1 for g in groups), FULL_ROW_DUPS + FEATURE_DUPS)

    def test_registry_split_moves_under_outer_test_target_noise(self):
        """MUST FAIL (the registry split): target noise on the rows of one outer-test fold that holds full-row
        duplicates breaks them apart, and rows of every fold move."""
        base = registry_folds_of(M)
        keys = [reg.dup_key(reg.csv_fields(l)) for l in tb.lines_of(M)]
        dup_rows = {i for i, k in enumerate(keys) if keys.count(k) > 1}
        f = max(base, key=lambda g: len(dup_rows & set(base[g])))
        self.assertTrue(dup_rows & set(base[f]))
        got = registry_folds_of(noise_targets(M, base[f]))
        self.assertGreater(moved(base, got), 0)
        other = [p for g in base if g != f for p in base[g]]
        self.assertGreater(sum(1 for p in other if {q: g for g, v in got.items() for q in v}[p] != {q: g for g, v in base.items() for q in v}[p]), 0)

    def test_split_v2_does_not_move_under_that_noise(self):
        base = v2_folds_of(M)
        reg_f = registry_folds_of(M)
        for rows in [base[f] for f in base] + [reg_f[f] for f in reg_f] + [list(range(M.shape[0]))]:
            self.assertEqual(v2_folds_of(noise_targets(M, rows)), base)

    def test_split_v2_does_not_move_under_duplicate_preserving_feature_changes(self):
        base = v2_folds_of(M)
        for f in base:
            m2 = structure_preserving_features(noise_targets(M, base[f]), base[f])
            self.assertEqual(partition(m2), partition(M))                 # the change kept the duplicate structure
            self.assertEqual(v2_folds_of(m2), base)

    def test_the_same_value_check_flags_registry_and_v1_ordering(self):
        prep = rb.t1_prep(tb.universe_of(M))
        for f in range(5):
            r = v2.value_check(rb.split_v2_t1, prep, f)
            self.assertTrue(r["valid"] and r["invariant"], r)
        self.assertFalse(v2.value_check(registry_split_fn, prep, 0)["invariant"])          # CONTROL
        self.assertFalse(v2.value_check(rowsel_raw.outer_split, prep, 0)["invariant"])      # CONTROL (feature-hash order)
        self.assertFalse(rowsel_raw.split_target_check(registry_split_fn, prep)["target_invariant"])
        self.assertTrue(rowsel_raw.split_target_check(rb.split_v2_t1, prep)["target_invariant"])

    def test_t1_partition_and_split_identity(self):
        prep = rb.t1_prep(tb.universe_of(M))
        folds = rb.split_v2_t1(H, prep.positions, prep.values, prep.specs)
        rank = lambda p: hashlib.sha256(f"{H}\0trainonly-rowsel-split-v1\0t1\0{p}".encode()).digest()
        groups = sorted(v2.duplicate_groups(prep.positions, prep.values, prep.specs).values(), key=lambda m: min(rank(p) for p in m))
        sizes, ref = [0] * 5, {f: [] for f in range(5)}
        for g in groups:
            f = sizes.index(min(sizes)); ref[f] += g; sizes[f] += len(g)
        self.assertEqual(folds, {f: sorted(v) for f, v in ref.items()})
        self.assertEqual(rb.split_v2_t1_sha256(H, folds), hashlib.sha256(rowsel._canon(
            {"schema": "rowsel-raw-split-v2", "dataset_hash": H, "partition": "t1", "folds": [folds[f] for f in range(5)]})).hexdigest())
        self.assertNotEqual(rb.split_v2_t1_sha256(H, folds), v2.split_sha256(H, folds))      # partition 't1' is in the id
        self.assertEqual([s["kind"] for s in prep.specs[:-1]], ["numeric"] * 5)
        self.assertTrue(prep.specs[-1].get("target"))


class BuilderV2(unittest.TestCase):
    CODES = tb.CODES

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="rowsel-v2split-", dir=os.environ.get("TMPDIR")))
        self.shard = self.dir / "shard-0"; self.shard.mkdir()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def build(self, m=M, shard=None, epoch_folds=range(4), refs=None, extra=None, expect_done=True, **kw):
        ds = tb.dataset(m, folds=REGISTRY_FOLDS)                         # unequal registry folds: never the v2 sizes
        its = tb.items(H, "clean", folds=epoch_folds)
        rec = rb.build_dataset_v2(ds, its, shard or self.shard, self.CODES, refs or dict(tb.REFS), budget=tb.BUDGET,
                                  extra=extra or {"registry_grouping": {"registry_grouped_by_feature_column": False}}, **kw)
        if expect_done:
            self.assertEqual((rec["status"], len(rec["contexts"])), ("done", 5), rec.get("detail"))
        return ds, rec

    def ctxs(self, rec, shard=None):
        return [json.loads(((shard or self.shard) / c["view_id"] / "context.json").read_bytes()) for c in rec["contexts"]]

    def test_five_folds_on_the_v2_split_with_quotas_from_the_new_sizes(self):
        ds, rec = self.build()
        self.assertEqual((rec["status"], len(rec["contexts"])), ("done", 5), rec.get("detail"))
        folds = v2_folds_of(M)
        self.assertNotEqual(folds, ds["folds"])                           # not the registry folds
        self.assertNotEqual([len(folds[f]) for f in range(5)], [len(ds["folds"][f]) for f in range(5)])
        q = rowsel.fold_quotas({f: len(v) for f, v in folds.items()}, tb.BUDGET)
        self.assertEqual(rec["split"]["fold_sizes"], [len(folds[f]) for f in range(5)])
        for f, c in enumerate(self.ctxs(rec)):
            rs = c["rowsel"]
            self.assertEqual((c["fold"], rs["split"]["schema"], rs["split"]["partition"], c["split_version"]),
                             (f, "rowsel-raw-split-v2", "t1", "rowsel-raw-split-v2"))
            self.assertEqual((rs["quotas"]["n_train"], rs["quotas"]["n_test"]), q[f])
            self.assertEqual(rs["quotas"]["all_folds"], {str(k): list(v) for k, v in q.items()})
            self.assertTrue(set(rs["test_positions"]) <= set(folds[f]))
            self.assertFalse(set(rs["train_positions"]) & set(folds[f]))
            self.assertTrue(rs["outer_split_changed"])

    def test_cut_block_and_dataset_level_replacement(self):
        ds, rec = self.build()
        sha = rb.split_v2_t1_sha256(H, v2_folds_of(M))
        for c in self.ctxs(rec):
            self.assertEqual(c["cut"]["cut_id"], "trainonly-split-v2:" + c["rowsel"]["split"]["sha256"])
            self.assertEqual(c["cut"]["cut_id"], "trainonly-split-v2:" + sha)
            self.assertEqual(c["identity"]["cut_id"], c["cut"]["cut_id"])
            self.assertEqual((c["cut"]["schema"], c["cut"]["comparable_with_replaced"]), ("rowsel-raw-split-v2", False))
            self.assertEqual(c["cut"]["replaces_cut"]["split_sha256"], ds["split"]["sha256"])
            self.assertEqual(len(c["cut"]["replaces_cut"]["epoch0_contexts"]), 4)
            e0 = c["epoch0_replaced"]
            self.assertEqual((e0["level"], e0["count"], e0["new_contexts"], e0["context_count_changed"]), ("dataset", 4, 5, True))
            self.assertEqual(c["identity"]["epoch0_replaced"], [x["view_id"] for x in e0["contexts"]])
            self.assertEqual(rb.verify_context_dir_v2(self.shard / c["view_id"])["problems"], [])

    def test_new_view_ids(self):
        _, rec = self.build()
        for c in self.ctxs(rec):
            self.assertIn("__rowsel2s_", c["view_id"])
            self.assertNotEqual(c["view_id"], rb.new_view_id(H, c["fold"], c["rowsel"]["view"]["identity"]))

    def test_verify_detects_a_comparable_or_wrong_cut(self):
        _, rec = self.build()
        d = self.shard / rec["contexts"][0]["view_id"]
        ctx = json.loads((d / "context.json").read_bytes())
        for bad in ({"comparable_with_replaced": True}, {"cut_id": "trainonly-split-v2:" + "0" * 64}):
            c2 = dict(ctx, cut=dict(ctx["cut"], **bad))
            (d / "context.json").write_text(json.dumps(c2))
            self.assertIn("cut", rb.verify_context_dir_v2(d)["problems"])

    def test_outer_test_changes_leave_the_split_and_the_train_side_identical(self):
        _, rec = self.build()
        folds = v2_folds_of(M)
        f = 2
        m2 = structure_preserving_features(noise_targets(M, folds[f]), folds[f])
        other = self.dir / "shard-x"; other.mkdir()
        _, rec2 = self.build(m=m2, shard=other)
        self.assertEqual(rec2["split"]["sha256"], rec["split"]["sha256"])
        a, b = self.ctxs(rec)[f], self.ctxs(rec2, other)[f]
        for k in ("train_positions", "test_positions"):
            self.assertEqual(a["rowsel"][k], b["rowsel"][k])
        self.assertEqual(a["rowsel"]["view"]["train_fit_identity"], b["rowsel"]["view"]["train_fit_identity"])
        self.assertEqual((a["scaler_sha256"], a["inner_labels_sha256"], a["files"]["a0.f32"]), (b["scaler_sha256"], b["inner_labels_sha256"], b["files"]["a0.f32"]))
        self.assertNotEqual(a["files"]["encoder.jtt1.gz"], b["files"]["encoder.jtt1.gz"])     # the test block did change
        self.assertEqual(a["rowsel"]["downstream_counterfactual"]["train_side_invariant"], True)

    def test_the_split_gate_refuses_value_reading_and_straddling_splits(self):
        for fn, want in ((registry_split_fn, "split_reads_target"), (rowsel_raw.outer_split, "split_moves_under_outer_test_values"),
                         (position_split_fn, "groups_straddle_folds")):
            with self.subTest(fn=fn.__name__):
                shard = self.dir / f"shard-{fn.__name__}"; shard.mkdir()
                _, rec = self.build(shard=shard, split_fn=fn, expect_done=False)
                self.assertEqual((rec["status"], rec["reason"]), ("failed", "split_gate_failed"))
                self.assertIn(want, rec["detail"])
                self.assertEqual([d for d in shard.iterdir() if d.name != "_datasets"], [])

    def test_a_failing_fold_writes_nothing_and_resume_never_rewrites(self):
        _, rec = self.build(select_fn=tb.fit_on_union_view, expect_done=False)                # CONTROL: fitted on train and test rows
        self.assertEqual((rec["status"], rec["reason"]), ("failed", "fold_gate_failed"))
        self.assertEqual([d for d in self.shard.iterdir() if d.name != "_datasets"], [])
        self.assertFalse(rb.dataset_done_v2(self.shard, H, self.CODES))
        _, rec = self.build()
        self.assertTrue(rb.dataset_done_v2(self.shard, H, self.CODES))
        self.assertFalse(rb.dataset_done_v2(self.shard, H, dict(self.CODES, **{"tools/v2_rowsel_build.py": "9" * 64})))
        stamp = {p: p.stat().st_mtime_ns for d in self.shard.iterdir() if d.name != "_datasets" for p in d.iterdir()}
        _, again = self.build()
        self.assertEqual((again["contexts_written"], again["contexts_already_done"]), (0, 5))
        self.assertEqual(stamp, {p: p.stat().st_mtime_ns for d in self.shard.iterdir() if d.name != "_datasets" for p in d.iterdir()})
        d = self.shard / rec["contexts"][0]["view_id"]
        c = json.loads((d / "context.json").read_bytes()); c["context_sha256"] = "0" * 64
        (d / "context.json").write_text(json.dumps(c))
        _, bad = self.build(expect_done=False)
        self.assertEqual((bad["status"], bad["reason"]), ("failed", "done_context_differs"))

    def test_a_downstream_failure_writes_nothing(self):
        fail = lambda U, built, meta, h: {"train_side_invariant": False, "train_side_differs": ["scaler_sha256"]}
        with mock.patch.object(rb, "downstream_counterfactual", fail):
            _, rec = self.build(expect_done=False)
        self.assertEqual((rec["status"], rec["reason"]), ("failed", "fold_gate_failed"))
        self.assertIn("downstream_counterfactual_failed", rec["detail"])
        self.assertEqual([d for d in self.shard.iterdir() if d.name != "_datasets"], [])

    def test_rights_and_t1_fallback_fields(self):
        rights = {"rights": "research-only", "rights_decision": "x#sha256=y"}
        fb = {"reason": "rows_unmatched", "rule": "fixture"}
        _, rec = self.build(refs=dict(tb.REFS, context_extra=rights), extra={"t1_fallback": fb})
        for c in self.ctxs(rec):
            self.assertEqual(({k: c[k] for k in rights}, c["t1_fallback"]), (rights, fb))
            self.assertEqual(rb.verify_context_dir_v2(self.shard / c["view_id"], rights)["problems"], [])
        _, rec2 = self.build(shard=self.dir / "s2")
        self.assertNotIn("t1_fallback", self.ctxs(rec2, self.dir / "s2")[0])


class CodeCustody(unittest.TestCase):
    """The summary verifies recorded code hashes against exact bytes (live file or code custody), never the live tree."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="rowsel-custody-", dir=os.environ.get("TMPDIR")))
        self.custody = self.dir / "custody"; self.custody.mkdir()
        self.live = rb.code_shas()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def put(self, name, body):
        h = hashlib.sha256(body).hexdigest()
        (self.custody / f"{name}.{h}").write_bytes(body)
        return h

    def test_live_and_custody_bytes_are_producible(self):
        r = rb.code_custody_check(dict(self.live), custody=self.custody)
        self.assertTrue(r["ok"]); self.assertEqual(r["differs_from_live"], [])
        old = self.put("v2_build_local.py", b"# the bytes a shard imported before another owner edited the file\n")
        rec = dict(self.live, **{"tools/v2_build_local.py": old})
        r = rb.code_custody_check(rec, custody=self.custody, why={"tools/v2_build_local.py": "edited by its owner"})
        self.assertTrue(r["ok"])
        f = r["files"]["tools/v2_build_local.py"]
        self.assertEqual((f["same_as_live"], f["producible"], f["why"]), (False, True, "edited by its owner"))
        self.assertTrue(f["bytes_from"].startswith(str(self.custody)))
        self.assertEqual(r["differs_from_live"], ["tools/v2_build_local.py"])

    def test_unproducible_hashes_fail(self):
        rec = dict(self.live, **{"tools/v2_build_local.py": "0" * 64})
        self.assertFalse(rb.code_custody_check(rec, custody=self.custody)["ok"])
        (self.custody / f"v2_build_local.py.{'1' * 64}").write_bytes(b"bytes that do not hash to the name")
        rec = dict(self.live, **{"tools/v2_build_local.py": "1" * 64})
        r = rb.code_custody_check(rec, custody=self.custody)
        self.assertFalse(r["ok"]); self.assertFalse(r["files"]["tools/v2_build_local.py"]["producible"])
        self.assertFalse(rb.code_custody_check({k: v for k, v in self.live.items() if k != "dltrain/rowsel.py"}, custody=self.custody)["ok"])

    def summarize(self, codes, label):
        tree = self.dir / label
        (tree / "shard-0").mkdir(parents=True)
        e = {"dataset_hash": H, "label": "clean", "epoch0": [{"view_id": r["view_id"], "context_sha256": r["context_sha256"], "fold": r["fold"]}
                                                            for r in tb.epoch_rows(H, range(4))],
             "shard": 0, "est_cpu_s": 1.0, "registry_grouping": {"registry_grouped_by_feature_column": True, "registry_group_columns": [2]}}
        (tree / "plan.json").write_text(json.dumps({"of": 1, "tree": "kmeans-clean", "split_version": rb.SPLIT_V2,
                                                    "counts": {"datasets": 1, "epoch0_contexts": 4}, "inputs": {}, "datasets": [e]}))
        rec = rb.build_dataset_v2(tb.dataset(M), tb.items(H, "clean", folds=range(4)), tree / "shard-0", codes, dict(tb.REFS),
                                  budget=tb.BUDGET, extra={"registry_grouping": e["registry_grouping"]})
        self.assertEqual(rec["status"], "done", rec.get("detail"))
        (tree / "shard-0-of-1.receipt.json").write_text("{\"expected_datasets\": 1, \"done\": 1, \"complete\": true, \"cpu_s\": 1, "
                                                      "\"elapsed_s\": 1, \"guard_pauses\": 0}")
        log = self.dir / "t.log"; log.write_text("OK\n")
        mut = self.dir / "m.json"; mut.write_text("{\"totals\": {}}")
        a = argparse.Namespace(out=str(tree), tree="kmeans-clean", tests_log=str(log), mutants=str(mut), pilot=None,
                               receipt=str(self.dir / f"summary-{label}.json"), code_custody=str(self.custody),
                               code_why=["tools/v2_build_local.py=edited by its owner"])
        rc = rb.summarize_v2(a)
        return rc, json.loads((self.dir / f"summary-{label}.json").read_bytes()), tree

    def test_summary_completes_when_recorded_bytes_are_in_custody(self):
        old = self.put("v2_build_local.py", b"# imported at build time\n")
        rc, doc, tree = self.summarize(dict(self.live, **{"tools/v2_build_local.py": old}), "ok")
        self.assertEqual((rc, doc["counts"]["verification_problems"]), (0, 0), doc["problems"])
        self.assertTrue((tree / ".complete").exists())
        sets = doc["code_custody"]["recorded_code_sets"]
        self.assertEqual((len(sets), sets[0]["differs_from_live"]), (1, ["tools/v2_build_local.py"]))
        self.assertEqual(sets[0]["files"]["tools/v2_build_local.py"]["why"], "edited by its owner")
        self.assertEqual(doc["counts"]["group_annotation_dropped_datasets"], [H])
        self.assertEqual((doc["datasets"][0]["group_annotation_dropped"], doc["datasets"][0]["registry_group_columns"]), (True, [2]))

    def test_summary_fails_when_a_recorded_hash_cannot_be_produced(self):
        rc, doc, tree = self.summarize(dict(self.live, **{"tools/v2_build_local.py": "2" * 64}), "bad")
        self.assertEqual(rc, 1)
        self.assertEqual({p["problem"] for p in doc["problems"]}, {"code_not_producible"})
        self.assertFalse((tree / ".complete").exists())


class Registry(unittest.TestCase):
    def test_the_default_route_builds_the_previous_builders_bytes(self):
        spec = importlib.util.spec_from_file_location("v2_rowsel_build_ed00684a", LEGACY)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        self.assertEqual(hashlib.sha256(LEGACY.read_bytes()).hexdigest(), LEGACY.name.split("build-")[1][:-3])
        d = Path(tempfile.mkdtemp(prefix="rowsel-legacy-", dir=os.environ.get("TMPDIR")))
        try:
            ds = tb.dataset()
            r, lab, acq = tb.items(H, "clean")[1]
            a = rb.rebuild_context(ds, r, lab, acq, d / "new", tb.CODES, dict(tb.REFS), budget=tb.BUDGET)
            b = old.rebuild_context(ds, r, lab, acq, d / "old", tb.CODES, dict(tb.REFS), budget=tb.BUDGET)
            self.assertEqual(a["view_id"], b["view_id"])
            for f in ("encoder.jtt1.gz", "teacher.jtt1.gz", "a0.f32", ".done"):
                self.assertEqual((a["dir"] / f).read_bytes(), (b["dir"] / f).read_bytes(), f)
            strip = lambda c: {k: v for k, v in c.items() if k not in ("utc_us", "build_cpu_s")}
            self.assertEqual(strip(json.loads((a["dir"] / "context.json").read_bytes())), strip(json.loads((b["dir"] / "context.json").read_bytes())))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Scope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dltrain import scale_check
        cls.rows = scale_check.Sources(load_index=False).included()

    def test_the_three_trees(self):
        trees = [t for t in rb.TREES if not rb.TREES[t].get("opt_in")]        # the drifted T1 fallback is an explicit opt-in
        got = {t: rb.load_tree_scope(self.rows, t)[2] for t in trees}
        self.assertEqual(got, {t: rb.TREES[t]["expected"] for t in trees})
        fb, refs, _ = rb.load_tree_scope(self.rows, "kmeans-fallback")
        raw = {e["dataset_hash"] for e in json.loads(rb.RAW_V2_PLAN.read_bytes())["datasets"]}
        self.assertFalse(set(fb) & raw)
        self.assertTrue(all(it[1] == "acquisition_cap" for its in fb.values() for it in its))
        fr, frefs, _ = rb.load_tree_scope(self.rows, "frozen-kmeans-research")
        self.assertEqual(frefs["context_extra"]["rights"], "research-only")

    def test_a_fallback_dataset_rebuilt_on_raw_is_refused(self):
        d = Path(tempfile.mkdtemp(prefix="rowsel-fb-", dir=os.environ.get("TMPDIR")))
        try:
            nb = json.loads(rb.read_verified(rb.FALLBACK_SOURCE, rb.FALLBACK_SOURCE_SHA256))["buckets"]["incoming_acquisition_capped_kmeans"]["datasets_not_rebuilt"]
            (d / "plan.json").write_text(json.dumps({"datasets": [{"dataset_hash": nb[0]["dataset_hash"]}]}))
            with self.assertRaises(rb.RebuildError) as cm:
                rb.fallback_datasets(raw_plan=d / "plan.json")
            self.assertEqual(cm.exception.reason, "fallback_dataset_rebuilt_on_raw")
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
