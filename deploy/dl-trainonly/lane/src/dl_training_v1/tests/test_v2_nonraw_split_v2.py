"""NONRAW-SPLIT-A: the clean non-raw routes re-split with split v2 (tools/v2_build_local.py, tools/v2_build_l7a.py).
Every invariance check is paired with a control that MUST FAIL it, so a check that cannot fail is not evidence.

  Registry    MUST-FAIL: the registry split (grouped-stratified-kfold/v1, regression path; an in-test port of
              RUN/tmp/split-scope/gsk.py 83183db4, itself a port of virtual_split.rs) moves when only outer-test TARGETS
              get noise, as soon as the universe holds exact full-row duplicates (and does not move without them)
  SplitV2     split v2 over each route's universe (T1 table, partition 't1'; matched L7a rows, partition 'l7a') does not
              move under target changes or duplicate-preserving outer-test feature changes; the partition label is bound;
              duplicate feature rows never straddle folds; the target is never part of the group key
  Dataset     all 5 folds: T = fold f, R = the rest, disjoint, JTT1 train-then-test; cut block and split record; new
              view_ids; epoch0_replaced; deterministic; outer-test values of fold f never reach context f's split,
              scaler, inner labels, encoder/teacher train blocks, A0 or train-side refit (end to end, and through the
              per-context gate); controls (target-reading split, feature-hash ordering, all-row scaler, all-row category
              map, reading test rows into train) are refused and nothing is written
  Writes      .done last, dataset-wide; a done context is never rewritten; resume skips; a collision is refused
  Default     the registry path (default) is the original build, byte for byte (L7b and L7a)
  Epoch0      the epoch-0 universe check (one data object; every context's R u T is the whole table)
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dltrain import jtt1, metafeatures, rowsel_raw, rowsel_raw_split_v2 as v2, t1prep, t1raw, trainonly
from tests.test_t1prep import legacy_all_rows, upstream_t1

SRC = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tool = _load("v2_build_local")
l7a = _load("v2_build_l7a")

H = "5a5b5c5d5e5f6061"
N = 420
POOL = ["3400.0", "4700.0", "5900.0", "6000.0", "6300.0"]
RARE = "5000.0"                     # a meta key no data row holds, between two keys rows hold: a non-affine code shift
FULL_DUP, FEAT_DUP = 36, 24


def raw_rows(n=N, seed=21, full_dup=FULL_DUP, feat_dup=FEAT_DUP):
    """Numeric n0..n2 (missing cells), one categorical, a target; `full_dup` rows copy an earlier row entirely (target
    included: exact full-row duplicates) and `feat_dup` rows copy only the features (different targets)."""
    rng = np.random.default_rng(seed)
    num = rng.normal(size=(n, 3)) * np.array([1.0, 50.0, 3.0]) + np.array([0.0, 1000.0, -2.0])
    num[rng.random(size=num.shape) < 0.04] = np.nan
    cat = [POOL[int(rng.integers(0, 5))] if rng.random() > 0.08 else None for _ in range(n)]
    y = 2.0 * np.nan_to_num(num[:, 0]) + 0.01 * np.nan_to_num(num[:, 1]) + rng.normal(scale=0.2, size=n)
    m = full_dup + feat_dup
    src = rng.choice(n - m, size=m, replace=False)
    for k, i in enumerate(range(n - m, n)):
        num[i], cat[i] = num[src[k]], cat[src[k]]
        y[i] = y[src[k]] if k < full_dup else y[src[k]] + 1.0 + 0.01 * k
    return num, cat, y


def t1_table(num, cat, y):
    """The acquisition's all-row T1 encoding (tests.test_t1prep.upstream_t1) with RARE in the meta mapping."""
    t1, meta = upstream_t1(np.vstack([num, num[:1]]), [cat + [RARE]], np.append(y, y[0]))
    meta["original_headers"] = ["n0", "c0", "n1", "yy", "n2"]
    meta["target_original_name"] = meta["source_target_column_original"] = "yy"
    meta["acquisition"] = {"acquired_at": "2026-10-02T12:00:00+00:00", "tool": "fixture"}
    return np.ascontiguousarray(t1[:-1]), meta


NUM, CAT, Y = raw_rows()
A, META = t1_table(NUM, CAT, Y)
SPECS_T1 = tool.t1_group_specs(A.shape[1])
RAW_SPECS = t1raw.columns(META, A.shape[1])


def raw_values(num, cat, y, specs=RAW_SPECS):
    inv = {k: c for c, k in specs[3]["inverse"].items()}
    codes = np.asarray([t1raw.MISSING_CODE if c is None else inv[c] for c in cat], dtype=np.float64)
    return np.ascontiguousarray(np.column_stack([num, codes, y]))


VALUES = raw_values(NUM, CAT, Y)


def plan_entry(h=H, folds=(0, 1, 2, 3)):
    return {"dataset_hash": h, "values_mode": "L7b", "rights": "licensed", "epoch0_labels": ["clean"],
            "epoch0": [{"view_id": f"{h}__r0__f{f}" if f == 0 else f"{h}__r0__f{f}__test_adaptive_{f:012d}",
                        "context_sha256": f"{f}" * 64, "fold": f} for f in folds],
            "registry_split": {"sha256": "d" * 64, "key": "splits/v1/x.json"}}


COV = {"rows": N, "sample_rows": N, "source_rows": N, "subsampled": False}


def t1_inputs(M=A, meta=META, e=None):
    return tool.l7b_inputs(e or plan_entry(), M, meta, "m" * 64, "a" * 64, COV, "frozen")


REMATCH = {"values_sha256": "v" * 64}


def l7a_inputs(V=VALUES, e=None):
    return l7a.l7a_inputs(e or plan_entry(), V, RAW_SPECS, "a" * 64, "m" * 64, COV, "incoming", {"sha256": "r" * 64}, {}, REMATCH)


_CACHE = {}


def computed(mode="L7b"):
    if mode not in _CACHE:
        inp = t1_inputs() if mode == "L7b" else l7a_inputs()
        _CACHE[mode] = tool.compute_dataset(plan_entry(), inp, mode, set(), {"code": "x"}, version=tool.SPLIT_V2)
    return _CACHE[mode]


def train_side(art):
    return {"scaler": art["scaler_sha256"], "encoder_train_rows": jtt1.train_rows_sha256(art["encoder_raw"]),
            "teacher_train_rows": jtt1.train_rows_sha256(art["teacher_raw"]), "inner_labels": art["inner_labels_sha256"],
            "a0": hashlib.sha256(art["a0"]).hexdigest()}


def dup_preserving_change(M, rows, specs, kind):
    """rowsel_raw_split_v2.perturb_fold_values: injective per column on `rows` only (duplicate partition preserved).
    Returns (values, specs): a categorical change appends new codes to a copy of the spec's inverse."""
    return v2.perturb_fold_values(M, specs, np.asarray(rows, dtype=np.int64), kind)


def remap_fold(M, rows, cols, seed=8):
    """Outer-test rows only: numeric columns `cols` mapped injectively (by rank, to -5 - rank: duplicates among the rows
    stay duplicates, and no row outside can equal them), targets replaced by fresh values."""
    W = np.array(M, dtype=np.float64, copy=True)
    rows = np.asarray(sorted(int(r) for r in rows), dtype=np.int64)
    for j in cols:
        col = W[rows, j]
        u = np.unique(col[np.isfinite(col)])
        rank = {float(v): k for k, v in enumerate(u.tolist())}
        W[rows, j] = [-5.0 - rank[float(v)] if np.isfinite(v) else v for v in col.tolist()]
    W[rows, -1] = np.random.default_rng(seed).normal(1e3, 50.0, size=rows.size)
    return W


# ------------------------------------------------------------------------------------- registry port (must fail)

MASK64 = (1 << 64) - 1


def _fields(row):
    return [("%.9g" % float(v)).encode() for v in row]


def _dup_key(fields):
    h = hashlib.sha256()
    for f in fields:
        h.update(struct.pack("<Q", len(f)))
        h.update(f)
    return h.digest()


def _salted(seed, first):
    return int.from_bytes(hashlib.sha256(struct.pack("<QI", seed & MASK64, first)).digest()[:8], "little")


def registry_folds(M, seed=20260930, folds=5):
    """grouped-stratified-kfold/v1, regression, group=None, repeat 0: components of the exact full-row duplicate key
    (every field, the target included), ordered by (-size, salted(seed, first index)); the first k components open the
    folds, then each goes to the smallest fold (ties by salted(seed ^ fold, first index))."""
    keys = [_dup_key(_fields(r)) for r in np.asarray(M)]
    first, comps = {}, {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
        comps.setdefault(first[k], []).append(i)
    groups = [comps[r] for r in sorted(comps)]
    order = sorted(range(len(groups)), key=lambda g: (-len(groups[g]), _salted(seed, groups[g][0])))
    k = min(len(groups), folds)
    out, sizes = [[] for _ in range(k)], [0] * k
    for pos, gi in enumerate(order):
        if pos < k:
            f = pos
        else:
            m = min(sizes)
            tied = [x for x in range(k) if sizes[x] == m]
            f = tied[0] if len(tied) == 1 else min(tied, key=lambda x: _salted(seed ^ x, groups[gi][0]))
        out[f].extend(groups[gi])
        sizes[f] += len(groups[gi])
    return {f: sorted(v) for f, v in enumerate(out)}


def moved(a, b):
    fa = {p: f for f, v in a.items() for p in v}
    return sum(1 for f, v in b.items() for p in v if fa[p] != f)


def full_row_dup_fold(folds, M):
    keys = [_dup_key(_fields(r)) for r in np.asarray(M)]
    for f, v in folds.items():
        ks = [keys[p] for p in v]
        if len(set(ks)) < len(ks):
            return f
    raise AssertionError("fixture has no full-row duplicate")


class Registry(unittest.TestCase):
    def test_registry_split_moves_under_outer_test_target_noise_with_full_row_duplicates(self):
        """MUST FAIL (the defect NONRAW-SPLIT-A fixes): y reaches the registry fold assignment through the duplicate key."""
        base = registry_folds(A)
        f = full_row_dup_fold(base, A)
        M = np.array(A, copy=True)
        rows = np.asarray(base[f])
        M[rows, -1] = M[rows, -1] + np.random.default_rng(3).normal(scale=1e-3, size=rows.size)   # outer-test y only
        self.assertGreater(moved(base, registry_folds(M)), 0)

    def test_registry_split_does_not_move_without_full_row_duplicates(self):
        """Control: the same target noise on a universe without full-row duplicates leaves it unchanged (the leak is
        the duplicate key, not an ordering on y)."""
        M0 = np.array(A, copy=True)
        M0[:, -1] += np.arange(M0.shape[0]) * 1e-6                           # no two rows share a target any more
        base = registry_folds(M0)
        rows = np.asarray(base[1])
        M = np.array(M0, copy=True)
        M[rows, -1] += np.random.default_rng(4).normal(scale=1e-3, size=rows.size)
        self.assertEqual(moved(base, registry_folds(M)), 0)

    def test_split_v2_does_not_move_under_the_same_target_noise(self):
        base = registry_folds(A)
        f = full_row_dup_fold(base, A)
        M = np.array(A, copy=True)
        rows = np.asarray(base[f])
        M[rows, -1] = M[rows, -1] + np.random.default_rng(3).normal(scale=1e-3, size=rows.size)
        f0, _ = tool.v2_split(H, A, SPECS_T1, "t1")
        f1, _ = tool.v2_split(H, M, SPECS_T1, "t1")
        self.assertEqual(f0, f1)


# ------------------------------------------------------------------------------------- split v2 per universe

def reference_split(h, M, specs, partition, folds=5):
    """Independent restatement: exact-duplicate FEATURE groups (target excluded), group rank = min sha256(h NUL domain NUL
    partition NUL index), groups in ascending rank into the smallest fold, ties to the lower fold."""
    groups = {}
    for p, key in enumerate(rowsel_raw.feature_bytes(M, specs)):
        groups.setdefault(key, []).append(p)

    def rank(p):
        return hashlib.sha256(f"{h}\0trainonly-rowsel-split-v1\0{partition}\0{p}".encode()).digest()
    sizes, out = [0] * folds, {f: [] for f in range(folds)}
    for m in sorted(groups.values(), key=lambda m: min(rank(p) for p in m)):
        f = sizes.index(min(sizes))
        out[f] += m
        sizes[f] += len(m)
    return {f: sorted(v) for f, v in out.items()}


class SplitV2(unittest.TestCase):
    def test_t1_route_matches_the_rule_with_its_partition_label(self):
        folds, srep = tool.v2_split(H, A, SPECS_T1, "t1")
        self.assertEqual(folds, reference_split(H, A, SPECS_T1, "t1"))
        self.assertNotEqual(folds, reference_split(H, A, SPECS_T1, "raw"))         # the label is bound
        self.assertEqual(srep["partition"], "t1")
        self.assertEqual([c["fold"] for c in srep["value_check"]["folds"]], [0, 1, 2, 3, 4])   # every fold is gated
        self.assertEqual(sorted(srep["target_check"]["variants_differing"]), [])
        self.assertEqual(srep["sha256"], v2.split_sha256(H, folds))                # exactly as the module returns it
        self.assertEqual(srep["schema"], "rowsel-raw-split-v2")

    def test_l7a_route_matches_the_rule_with_its_partition_label(self):
        inp = l7a_inputs()
        folds, srep = tool.v2_split(H, inp["values"], inp["specs"], tool.ROUTES["L7a"]["partition"])
        self.assertEqual(tool.ROUTES["L7a"]["partition"], "l7a")
        self.assertEqual(folds, reference_split(H, VALUES, RAW_SPECS, "l7a"))
        self.assertNotEqual(folds, reference_split(H, VALUES, RAW_SPECS, "t1"))

    def test_group_key_excludes_the_target(self):
        specs = tool.t1_group_specs(A.shape[1])
        self.assertEqual([s["index"] for s in specs[:-1]], list(range(A.shape[1] - 1)))
        M = np.array(A[:2], copy=True)
        M[1, :-1] = M[0, :-1]
        M[1, -1] = M[0, -1] + 5.0
        k = rowsel_raw.group_keys(M, specs)
        self.assertEqual(k[0], k[1])

    def test_feature_only_duplicates_share_a_fold(self):
        for M, specs, part in ((A, SPECS_T1, "t1"), (VALUES, RAW_SPECS, "l7a")):
            folds, srep = tool.v2_split(H, M, specs, part)
            self.assertEqual(srep["groups_straddling_folds"], 0)
            self.assertGreaterEqual(srep["duplicate_rows"], FULL_DUP + FEAT_DUP)
            fold_of = {p: f for f, v in folds.items() for p in v}
            for i in range(N - FEAT_DUP, N):          # feature-only copies (targets differ) of an earlier row
                twins = [j for j in range(N) if j != i and np.array_equal(M[j, :-1], M[i, :-1], equal_nan=True)]
                self.assertTrue(twins)
                self.assertTrue(all(fold_of[j] == fold_of[i] for j in twins))

    def test_no_move_under_target_changes_or_duplicate_preserving_outer_test_feature_changes(self):
        for M, specs, part in ((A, SPECS_T1, "t1"), (VALUES, RAW_SPECS, "l7a")):
            base, srep = tool.v2_split(H, M, specs, part)
            self.assertTrue(srep["target_invariant"] and srep["value_check_invariant"], srep["value_check"])
            for f in range(5):
                for kind in v2.CF_KINDS:
                    M2, s2 = dup_preserving_change(M, base[f], specs, kind)
                    self.assertGreater(int(np.sum(~((M2 == M) | (np.isnan(M2) & np.isnan(M))))), 0)
                    self.assertEqual(v2.duplicate_partition(np.arange(N), M2, s2),
                                     v2.duplicate_partition(np.arange(N), M, specs))
                    got, _ = tool.v2_split(H, M2, s2, part)
                    self.assertEqual(got, base, (part, f, kind))

    def test_controls_fail_the_split_gates(self):
        """MUST FAIL: the v1 feature-hash ordering moves under outer-test feature changes, a y-sorted split reads y, and a
        position split straddles duplicate groups; split_gate refuses each."""
        with mock.patch.object(tool, "split_fn", lambda partition: rowsel_raw.outer_split):
            _, srep = tool.v2_split(H, A, SPECS_T1, "t1")
        self.assertIn("split_moves_under_outer_test_values", tool.split_gate(srep))

        def by_target(partition):
            def fn(h, positions, values, specs, folds=5):
                order = np.argsort(np.asarray(values)[:, -1], kind="mergesort")
                return {f: sorted(int(np.asarray(positions)[i]) for i in order[f::folds]) for f in range(folds)}
            return fn
        with mock.patch.object(tool, "split_fn", by_target):
            _, srep = tool.v2_split(H, A, SPECS_T1, "t1")
        self.assertIn("split_reads_target", tool.split_gate(srep))
        self.assertIn("groups_straddle_folds", tool.split_gate(srep))


# ------------------------------------------------------------------------------------- one dataset, all folds

class Dataset(unittest.TestCase):
    def test_every_fold_is_built_with_t_equal_the_fold(self):
        for mode in ("L7b", "L7a"):
            res = computed(mode)
            part = tool.ROUTES[mode]["partition"]
            M, specs = (A, SPECS_T1) if mode == "L7b" else (VALUES, RAW_SPECS)
            folds = reference_split(H, M, specs, part)
            self.assertEqual(len(res["results"]), 5)
            for f, (vid, files, rec) in enumerate(res["results"]):
                T = folds[f]
                R = [p for p in range(N) if p not in set(T)]
                self.assertEqual(rec["fold"], f)
                self.assertEqual(rec["outer_test_indices_sha256"], hashlib.sha256(json.dumps(T).encode()).hexdigest())
                self.assertEqual(rec["outer_train_indices_sha256"], hashlib.sha256(json.dumps(R).encode()).hexdigest())
                self.assertEqual((rec["n_train"], rec["n_test"]), (len(R), len(T)))
                raw = gzip.decompress(files["encoder.jtt1.gz"])
                hdr = jtt1.parse_header(raw)
                self.assertEqual((hdr["n_train"], hdr["n_test"], hdr["inner_folds"]), (len(R), len(T), 2))
                self.assertEqual(rec["values_mode"], mode)
                self.assertTrue(rec["counterfactual"]["train_side_invariant"])

    def test_cut_block_and_split_record(self):
        for mode in ("L7b", "L7a"):
            res = computed(mode)
            part = tool.ROUTES[mode]["partition"]
            M, specs = (A, SPECS_T1) if mode == "L7b" else (VALUES, RAW_SPECS)
            folds = v2.outer_split(H, np.arange(N), M, specs, partition=part)
            want = "trainonly-split-v2:" + v2.split_sha256(H, folds)
            for _, _, rec in res["results"]:
                cut = rec["cut"]
                self.assertEqual(cut["cut_id"], want)
                self.assertEqual((cut["schema"], cut["partition"], cut["replaces_cut"], cut["comparable_with_replaced"]),
                                 ("rowsel-raw-split-v2", part, "d" * 64, False))
                sp = rec["split"]
                self.assertEqual((sp["groups_straddling_folds"], sp["target_invariant"], sp["schema"], sp["partition"]),
                                 (0, True, "rowsel-raw-split-v2", part))
                self.assertEqual(sp["fold_sizes"], [len(folds[f]) for f in range(5)])
                self.assertEqual(sp["sha256"], v2.split_sha256(H, folds))
                self.assertEqual(rec["split_version"], "rowsel-raw-split-v2")

    def test_new_view_ids_and_epoch0_replaced(self):
        e = plan_entry()
        for mode, tag in (("L7b", "l7bsplit2_"), ("L7a", "l7asplit2_")):
            res = computed(mode)
            old = {c["view_id"] for c in e["epoch0"]}
            vids = [r[0] for r in res["results"]]
            self.assertEqual(len(set(vids)), 5)
            for f, (vid, _, rec) in enumerate(res["results"]):
                ident = hashlib.sha256(tool.canon(rec["identity_body"])).hexdigest()
                self.assertEqual(rec["identity"], ident)
                body = rec["identity_body"]                                    # the identity binds the new cut and rows
                self.assertEqual((body["cut_id"], body["partition"], body["fold"], body["values_mode"]),
                                 (rec["cut"]["cut_id"], rec["cut"]["partition"], f, mode))
                self.assertEqual((body["outer_train_indices_sha256"], body["outer_test_indices_sha256"]),
                                 (rec["outer_train_indices_sha256"], rec["outer_test_indices_sha256"]))
                self.assertEqual(vid, f"{H}__r0__f{f}__{tag}{ident[:12]}")
                self.assertNotIn(vid, old)
                e0 = rec["epoch0_replaced"]
                self.assertEqual((e0["level"], e0["count"], e0["new_contexts"], e0["context_count_changed"]),
                                 ("dataset", 4, 5, True))
                self.assertEqual([c["view_id"] for c in e0["contexts"]], [c["view_id"] for c in e["epoch0"]])
                self.assertEqual(rec["row_selection_leak"], ["clean"])

    def test_group_annotation_fields_reach_every_context(self):
        ga = {"dataset_hash": H, "columns": [3], "components": 6, "distinct_values": 6, "n_columns": 5, "subroute": "clean_frozen"}
        names = tool.t1_column_names(META, A.shape[1])
        x = tool.group_annotation_fields(ga, names)
        self.assertEqual((x["group_annotation_dropped"], x["registry_group_column"]["t1_column_index"],
                          x["registry_group_column"]["t1_column_name"]), (True, 3, "c0"))
        amb = tool.group_annotation_fields(dict(ga, columns=[1, 3]), names)
        self.assertEqual(amb["registry_group_column"], "ambiguous")
        self.assertEqual([c["t1_column_index"] for c in amb["registry_group_column_candidates"]], [1, 3])
        self.assertEqual(tool.group_annotation_fields(None), {"group_annotation_dropped": False, "registry_group_column": None})
        for mode, inp in (("L7b", t1_inputs()), ("L7a", l7a_inputs())):
            e = dict(plan_entry(), context_extra=dict(x, rights="licensed"))
            res = tool.compute_dataset(e, inp, mode, set(), {"code": "x"}, version=tool.SPLIT_V2)
            for _, _, rec in res["results"]:
                self.assertIs(rec["group_annotation_dropped"], True)
                self.assertEqual(rec["registry_group_column"], x["registry_group_column"])
            self.assertEqual(res["cut"]["cut_id"], computed(mode)["cut"]["cut_id"])   # the split ignores the annotation

    def test_deterministic(self):
        a = computed("L7b")
        b = tool.compute_dataset(plan_entry(), t1_inputs(), "L7b", set(), {"code": "x"}, version=tool.SPLIT_V2)
        self.assertEqual([(v, f) for v, f, _ in a["results"]], [(v, f) for v, f, _ in b["results"]])

    def test_outer_test_values_never_reach_the_train_side_end_to_end(self):
        """Fold f's rows get new targets and duplicate-preserving feature values; the whole dataset is rebuilt: the
        split and context f's split, scaler, inner labels, encoder/teacher train blocks and A0 are bit-identical."""
        for mode in ("L7b", "L7a"):
            base = computed(mode)
            part = tool.ROUTES[mode]["partition"]
            M, specs = (A, SPECS_T1) if mode == "L7b" else (VALUES, RAW_SPECS)
            folds = v2.outer_split(H, np.arange(N), M, specs, partition=part)
            for f in (1, 3):
                M2 = remap_fold(M, folds[f], (0, 1, 2))                         # numeric features and y of fold f only
                self.assertEqual(v2.duplicate_partition(np.arange(N), M2, specs), v2.duplicate_partition(np.arange(N), M, specs))
                self.assertFalse(np.array_equal(M2[np.asarray(folds[f])], np.asarray(M)[np.asarray(folds[f])], equal_nan=True))
                inp = t1_inputs(M2) if mode == "L7b" else l7a_inputs(M2)
                got = tool.compute_dataset(plan_entry(), inp, mode, set(), {"code": "x"}, version=tool.SPLIT_V2)
                self.assertEqual(got["cut"]["cut_id"], base["cut"]["cut_id"])
                want = train_side_of(base, f, mode, M, specs)
                have = train_side_of(got, f, mode, M2, specs)
                self.assertEqual(have, want, (mode, f))

    def test_control_registry_split_moves_context_train_side(self):
        """MUST FAIL: with the registry split, the same outer-test target noise changes another context's train rows."""
        base = registry_folds(A)
        f = full_row_dup_fold(base, A)
        M = np.array(A, copy=True)
        rows = np.asarray(base[f])
        M[rows, -1] += np.random.default_rng(3).normal(scale=1e-3, size=rows.size)
        new = registry_folds(M)
        g = next(g for g in range(5) if new[g] != base[g])
        R0 = [p for p in range(N) if p not in set(base[g])]
        R1 = [p for p in range(N) if p not in set(new[g])]
        a0 = train_side(tool.l7b_artifacts(A, META, R0, base[g]))
        a1 = train_side(tool.l7b_artifacts(M, META, R1, new[g]))
        self.assertNotEqual(a0["encoder_train_rows"], a1["encoder_train_rows"])


def train_side_of(res, f, mode, M, specs):
    rec = res["results"][f][2]
    files = res["results"][f][1]
    return {"outer_train": rec["outer_train_indices_sha256"], "outer_test": rec["outer_test_indices_sha256"],
            "scaler": rec["scaler_sha256"], "inner_labels": rec["inner_labels_sha256"],
            "encoder_train_rows": jtt1.train_rows_sha256(gzip.decompress(files["encoder.jtt1.gz"])),
            "teacher_train_rows": jtt1.train_rows_sha256(gzip.decompress(files["teacher.jtt1.gz"])),
            "a0": hashlib.sha256(files["a0.f32"]).hexdigest(),
            "refit_train": rec["counterfactual"]["train_side"]["refit_train"]}


class Gates(unittest.TestCase):
    """Leaky builders are refused by the per-context gate; nothing is written for the dataset."""

    def _refused(self, mode, inp, reason="counterfactual_failed"):
        with self.assertRaises(tool.SplitV2Error) as cm:
            tool.compute_dataset(plan_entry(), inp, mode, set(), {"code": "x"}, version=tool.SPLIT_V2)
        self.assertEqual(cm.exception.reason, reason, cm.exception.detail[:300])

    def test_all_row_category_map_is_caught(self):
        orig = tool.l7b_artifacts

        def leaky(M, meta, R, T):
            art = orig(M, meta, R, T)
            split = legacy_all_rows(meta)(M, R, T)
            art["encoder_raw"] = split["jtt1"]
            return art
        with mock.patch.object(tool, "l7b_artifacts", leaky):
            self._refused("L7b", t1_inputs())

    def test_all_row_scaler_is_caught(self):
        orig = tool.l7b_artifacts

        def leaky(M, meta, R, T):
            art = orig(M, meta, R, T)
            allr = np.asarray(M, dtype=np.float64)[sorted(list(R) + list(T))]
            art["scaler_sha256"] = hashlib.sha256(allr.min(axis=0).tobytes() + allr.max(axis=0).tobytes()).hexdigest()
            return art
        with mock.patch.object(tool, "l7b_artifacts", leaky):
            self._refused("L7b", t1_inputs())

    def test_test_rows_in_the_train_block_are_caught(self):
        orig = tool.l7b_artifacts

        def leaky(M, meta, R, T):
            k = len(T) // 2
            return orig(M, meta, sorted(list(R)[k:] + list(T)[:k]), sorted(list(R)[:k] + list(T)[k:]))
        with mock.patch.object(tool, "l7b_artifacts", leaky):
            self._refused("L7b", t1_inputs())

    def test_a_build_that_differs_from_the_gate_base_is_caught(self):
        inp = t1_inputs()
        good = inp["build"]

        def other(R, T):                                                      # writes something the gate never tested
            art = dict(good(R, T))
            art["a0"] = bytes(len(art["a0"]))
            return art
        inp["build"] = other
        self._refused("L7b", inp)

    def test_l7a_build_that_differs_from_the_gate_base_is_caught(self):
        inp = l7a_inputs()
        good = inp["build"]

        def other(R, T):
            art = dict(good(R, T))
            art["a0"] = bytes(len(art["a0"]))
            return art
        inp["build"] = other
        self._refused("L7a", inp)

    def test_jtt1_layout_mismatch_is_refused(self):
        inp = t1_inputs()
        good = inp["build"]
        inp["build"] = lambda R, T: good(R[: len(R) - 3], T)
        self._refused("L7b", inp, "jtt1_layout")

    def test_l7a_all_row_refit_is_caught(self):
        orig = t1raw.prepare

        def leaky(values, specs, tr, te, **kw):                               # category codes fitted on every row
            v = np.asarray(values, dtype=np.float64)
            fitted, rep = t1raw.refit(v, specs, list(range(v.shape[0])))
            split = trainonly.prepare_split(fitted, tr, te, **kw)
            split["t1raw"], split["fitted"] = rep, fitted
            return split
        with mock.patch.object(t1raw, "prepare", leaky):
            self._refused("L7a", l7a_inputs())
        self.assertIs(t1raw.prepare, orig)

    def test_split_gates_refuse_a_target_reading_split(self):
        def by_target(partition):
            def fn(h, positions, values, specs, folds=5):
                order = np.argsort(np.asarray(values)[:, -1], kind="mergesort")
                return {f: sorted(int(np.asarray(positions)[i]) for i in order[f::folds]) for f in range(folds)}
            return fn
        with mock.patch.object(tool, "split_fn", by_target):
            self._refused("L7b", t1_inputs(), "split_check_failed")

    def test_view_id_collision_is_refused(self):
        vid = computed("L7b")["results"][3][0]
        with self.assertRaises(tool.SplitV2Error) as cm:
            tool.compute_dataset(plan_entry(), t1_inputs(), "L7b", {vid}, {"code": "x"}, version=tool.SPLIT_V2)
        self.assertEqual(cm.exception.reason, "view_id_collision")

    def test_counterfactual_perturbations_change_test_rows_only(self):
        res = computed("L7b")
        folds = v2.outer_split(H, np.arange(N), A, SPECS_T1, partition="t1")
        R = [p for p in range(N) if p not in set(folds[0])]
        for kind in tool.T1_PERTURBATIONS:
            W, T2 = tool.perturb_t1_test_rows(A, META, R, folds[0], kind)
            self.assertEqual(W[R].tobytes(), A[R].tobytes())
            self.assertTrue(kind == "drop" or not np.array_equal(W, A), kind)
        self.assertEqual(sorted(res["results"][0][2]["counterfactual"]["cases"]), sorted(tool.T1_PERTURBATIONS))


# ------------------------------------------------------------------------------------- stored refit report (PR #218 #7)

def _folds_t1():
    return v2.outer_split(H, np.arange(N), A, SPECS_T1, partition="t1")


def _rt(folds, f):
    T = list(folds[f])
    return [p for p in range(N) if p not in set(T)], T


def _report_test_side(rep):
    """The test-side fields of a stored refit report (t1prep for L7b, t1raw for L7a)."""
    out = {"categorical_test": {k: v["test"] for k, v in sorted(rep["categorical"].items())}}
    if "possibly_imputed_cells" in rep:
        out["possibly_imputed_test"] = rep["possibly_imputed_cells"]["test"]
    if "imputation" in rep:
        out["imputed_test"] = rep["imputation"]["imputed_cells"]["test"]
        out["test_missing"] = {k: c["test_missing"] for k, c in sorted(rep["imputation"]["columns"].items())}
    return out


class StoredReport(unittest.TestCase):
    """PR #218 finding 7: context.json stores the whole refit report (t1prep for L7b, t1raw for L7a), including the
    test-side categorical missing/unseen counts and possibly_imputed_cells.test (t1raw: imputed_cells.test and per-column
    test_missing). They are counts over outer-test FEATURE cells: the stored report is bit-identical when outer-test
    TARGETS change, and a change of outer-test features that moves those counts leaves the scaler, the inner labels, the
    encoder/teacher train rows and A0 bit-identical (the counts are inputs to none of them). Control: the same feature
    change does move the test-side counts, so the target invariance is not vacuous."""

    def _target_variants(self, M, T):
        rng = np.random.default_rng(31)
        Ta = np.asarray(T)
        out = []
        for kind in ("noise", "permute", "outlier", "constant"):
            W = np.array(M, dtype=np.float64, copy=True)
            if kind == "noise":
                W[Ta, -1] = rng.normal(1e3, 50.0, size=Ta.size)
            elif kind == "permute":
                W[Ta, -1] = W[Ta[rng.permutation(Ta.size)], -1]
            elif kind == "outlier":
                W[Ta[0], -1] = -1e12
            else:
                W[Ta, -1] = 0.0
            out.append((kind, W))
        return out

    def test_l7b_stored_report_is_invariant_under_outer_test_targets(self):
        folds = _folds_t1()
        for f in range(5):
            R, T = _rt(folds, f)
            want = tool.canon(tool.l7b_artifacts(A, META, R, T)["report"])
            for kind, W in self._target_variants(A, T):
                self.assertEqual(tool.canon(tool.l7b_artifacts(W, META, R, T)["report"]), want, (f, kind))

    def test_l7a_stored_report_is_invariant_under_outer_test_targets(self):
        folds = v2.outer_split(H, np.arange(N), VALUES, RAW_SPECS, partition="l7a")
        for f in (0, 3):
            R, T = _rt(folds, f)
            want = tool.canon(l7a.l7a_artifacts(VALUES, RAW_SPECS, R, T)["report"])
            for kind, W in self._target_variants(VALUES, T):
                self.assertEqual(tool.canon(l7a.l7a_artifacts(W, RAW_SPECS, R, T)["report"]), want, (f, kind))

    def test_the_record_written_to_context_json_is_invariant_under_outer_test_targets(self):
        base_res = computed("L7b")
        folds = _folds_t1()
        f = 2
        W = np.array(A, copy=True)
        W[np.asarray(folds[f]), -1] = np.random.default_rng(7).normal(1e3, 50.0, size=len(folds[f]))
        got = tool.compute_dataset(plan_entry(), t1_inputs(W), "L7b", set(), {"code": "x"}, version=tool.SPLIT_V2)
        self.assertEqual(tool.canon(got["results"][f][2]["t1prep"]), tool.canon(base_res["results"][f][2]["t1prep"]))
        self.assertEqual(got["results"][f][2]["counterfactual"]["train_side"], base_res["results"][f][2]["counterfactual"]["train_side"])

    def test_test_side_counts_are_feature_counts_that_feed_nothing_on_the_train_side(self):
        specs = t1prep.layout(META, A.shape[1])
        cat = next(s for s in specs if s["kind"] == "categorical_codes")
        num = next(s for s in specs if s["kind"] == "numeric" and s.get("impute_value") is not None)
        miss = next(c for c, k in cat["inverse"].items() if k == t1prep.MISSING_KEY)
        rare = next(c for c, k in cat["inverse"].items() if k == RARE)
        folds = _folds_t1()
        R, T = _rt(folds, 1)
        Ta = np.asarray(T)
        W = np.array(A, copy=True)
        W[Ta[0::3], cat["index"]] = tool._code_value(cat, miss)                # test cells made missing
        W[Ta[1::3], cat["index"]] = tool._code_value(cat, rare)                # a category no train row holds
        W[Ta[2::3], num["index"]] = (float(num["impute_value"]) - num["lo"]) / (num["hi"] - num["lo"])   # = impute value
        a0, a1 = tool.l7b_artifacts(A, META, R, T), tool.l7b_artifacts(W, META, R, T)
        self.assertNotEqual(_report_test_side(a1["report"]), _report_test_side(a0["report"]))   # control: they move
        self.assertEqual(tool.t1_train_side(a1), tool.t1_train_side(a0))                       # nothing on the train side
        self.assertEqual(train_side(a1), train_side(a0))
        self.assertEqual(a1["inner_loads"], a0["inner_loads"])


# ------------------------------------------------------------------------------------- writes and resume

def scratch():
    return Path(tempfile.mkdtemp(prefix="nonraw-v2split-test-", dir=os.environ.get("TMPDIR")))


class Writes(unittest.TestCase):
    def _run(self, out, inp=None, existing=()):
        e = dict(plan_entry(), context_extra=tool.group_annotation_fields(None))
        return tool.run_dataset_v2(e, out, "L7b", lambda e: inp or t1_inputs(), set(existing), {"code": "x"},
                                   log=lambda s: None, version=tool.SPLIT_V2)

    def test_done_last_resume_and_no_rewrite(self):
        out = scratch()
        order = []
        real = tool.atomic

        def spy(path, data):
            order.append(Path(path).name)
            return real(path, data)
        with mock.patch.object(tool, "atomic", spy):
            rec = self._run(out)
        self.assertEqual(rec["status"], "done", rec.get("detail"))
        writes = [n for n in order if n != f"{H}.json"]
        self.assertEqual(writes[-5:], [".done"] * 5)
        self.assertNotIn(".done", writes[:-5])
        self.assertTrue(tool.dataset_done(out, H, {"code": "x"}))
        self.assertFalse(tool.dataset_done(out, H, {"code": "y"}))
        vids = [c["view_id"] for c in rec["contexts"]]
        stamp = {v: (out / v / "context.json").read_bytes() for v in vids}
        rec2 = self._run(out)
        self.assertEqual((rec2["status"], rec2["contexts_written"], rec2["contexts_already_done"]), ("done", 0, 5))
        self.assertEqual({v: (out / v / "context.json").read_bytes() for v in vids}, stamp)

    def test_done_context_that_differs_is_never_rewritten(self):
        out = scratch()
        rec = self._run(out)
        v = rec["contexts"][2]["view_id"]
        cj = json.loads((out / v / "context.json").read_bytes())
        cj["identity"] = "0" * 64
        tampered = (json.dumps(cj, sort_keys=True) + "\n").encode()
        (out / v / "context.json").write_bytes(tampered)
        rec2 = self._run(out)
        self.assertEqual((rec2["status"], rec2["reason"]), ("failed", "done_context_differs"))
        self.assertEqual((out / v / "context.json").read_bytes(), tampered)

    def test_failed_dataset_writes_no_context(self):
        out = scratch()
        vid = computed("L7b")["results"][4][0]
        rec = self._run(out, existing={vid})
        self.assertEqual((rec["status"], rec["reason"]), ("failed", "view_id_collision"))
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["_datasets"])

    def test_record_fields(self):
        out = scratch()
        rec = self._run(out)
        cj = json.loads((out / rec["contexts"][0]["view_id"] / "context.json").read_bytes())
        for k in ("cut", "split", "epoch0_replaced", "identity", "identity_body", "universe", "counterfactual", "t1prep",
                  "group_annotation_dropped", "registry_group_column",
                  "code_sha256", "context_sha256", "files", "utc_us"):
            self.assertIn(k, cj)
        self.assertEqual(cj["universe"]["partition"], "t1")
        self.assertFalse(cj["published"])
        for name, m in cj["files"].items():
            self.assertEqual(hashlib.sha256((out / rec["contexts"][0]["view_id"] / name).read_bytes()).hexdigest(), m["sha256"])


# ------------------------------------------------------------------------------------- default path unchanged

def original_l7b(M, meta, R, T):
    """The pre-hook tools/v2_build_local.build_one computation (sha256 0a702649...), inline."""
    split = t1prep.prepare(M, meta, R, T)
    fitted, rep = t1prep.refit(M, meta, R)
    teacher = trainonly.jtt1_bytes(fitted[R], fitted[T].reshape(-1, fitted.shape[1]), split["inner_labels"])
    return (gzip.compress(split["jtt1"], compresslevel=3, mtime=0), gzip.compress(teacher, compresslevel=3, mtime=0),
            metafeatures.from_raw(split["jtt1"])[0], rep)


def original_l7a(values, specs, R, T):
    """The pre-hook tools/v2_build_l7a.build_l7a computation (sha256 914489bb...), inline."""
    split = t1raw.prepare(values, specs, R, T)
    fitted = split["fitted"]
    teacher = trainonly.jtt1_bytes(fitted[R], fitted[T].reshape(-1, fitted.shape[1]), split["inner_labels"])
    return (gzip.compress(split["jtt1"], compresslevel=3, mtime=0), gzip.compress(teacher, compresslevel=3, mtime=0),
            metafeatures.from_raw(split["jtt1"])[0], split["t1raw"])


class Default(unittest.TestCase):
    T = list(range(3, N, 5))
    R = [p for p in range(N) if p % 5 != 3]

    def test_there_is_no_default_split(self):
        """PR #218 finding 10: without --split-version both tools refuse (argparse error, exit 2) before any build runs."""
        base_args = ["--out", "x", "--shard", "0", "--of", "3"]
        for mod, builders in ((tool, ("build_registry", "build_v2", "make_plan", "summarize_v2")),
                              (l7a, ("build", "summarize"))):
            patches = [mock.patch.object(mod, b) for b in builders]
            mocks = [pt.start() for pt in patches]
            try:
                with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as cm:
                    mod.main(base_args)
                self.assertEqual(cm.exception.code, 2)
                for m in mocks:
                    m.assert_not_called()
            finally:
                for pt in patches:
                    pt.stop()
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            tool.arg_parser().parse_args(base_args)

    def test_registry_runs_only_when_explicit(self):
        args = ["--split-version", "registry-epoch0", "--out", "x", "--shard", "0", "--of", "3"]
        self.assertEqual(tool.arg_parser().parse_args(args).split_version, "registry-epoch0")
        with mock.patch.object(tool, "build_registry", return_value=7) as reg, mock.patch.object(tool, "build_v2") as b2:
            self.assertEqual(tool.main(args), 7)
        reg.assert_called_once()
        b2.assert_not_called()
        with mock.patch.object(l7a, "build", return_value=5) as reg, mock.patch.object(l7a.local, "build_v2") as b2:
            self.assertEqual(l7a.main(args), 5)
        reg.assert_called_once()
        b2.assert_not_called()
        v2 = ["--split-version", "rowsel-raw-split-v2", "--out", "x", "--shard", "0", "--of", "3"]
        with mock.patch.object(tool, "build_registry") as reg, mock.patch.object(tool, "build_v2", return_value=3) as b2:
            self.assertEqual(tool.main(v2), 3)
        reg.assert_not_called()
        self.assertEqual(b2.call_args[0][0].split_version, "rowsel-raw-split-v2")

    def test_functions_refuse_a_missing_split_version(self):
        """Function level: no default and None refused, before anything is read or written."""
        def refused(fn, *a, **kw):
            with self.assertRaises(tool.SplitV2Error) as cm:
                fn(*a, **kw)
            self.assertEqual(cm.exception.reason, "split_version_required")
        refused(tool.compute_dataset, plan_entry(), t1_inputs(), "L7b", set(), {"code": "x"})
        refused(tool.compute_dataset, plan_entry(), t1_inputs(), "L7b", set(), {"code": "x"}, version=None)
        refused(tool.compute_dataset, plan_entry(), t1_inputs(), "L7b", set(), {"code": "x"}, version="registry-epoch0")
        out = scratch()
        loader = mock.Mock(side_effect=AssertionError("loader must not run"))
        refused(tool.run_dataset_v2, plan_entry(), out, "L7b", loader, set(), {"code": "x"}, log=lambda s: None)
        loader.assert_not_called()
        self.assertEqual(list(out.iterdir()), [])
        refused(tool.code_shas_v2)
        refused(tool.routes, None)
        ns = argparse.Namespace(plan="clean-l7b", out=str(out / "p"), of=1, allow_other_out=True, shard=0, datasets=None)
        with mock.patch.object(tool, "check_pins", side_effect=AssertionError("must refuse first")), \
                mock.patch.object(tool, "load_plan", side_effect=AssertionError("must refuse first")):
            refused(tool.make_plan, ns)
            refused(tool.build_v2, ns)
        self.assertFalse((out / "p").exists())
        with self.assertRaises(TypeError):
            l7a.code_shas_v2()                                           # no default at all
        with self.assertRaises(l7a.local.SplitV2Error):
            l7a.code_shas_v2(None)
        self.assertEqual(tool.code_shas_v2(version=tool.SPLIT_V2), tool.code_shas_v2((), tool.SPLIT_V2))

    def test_l7b_default_build_is_byte_identical_to_the_original(self):
        out = scratch()
        c = {"A": A, "R": self.R, "T": self.T, "row": {"view_id": f"{H}__r0__f3", "dataset_hash": H, "fold": 3,
                                                       "context_sha256": "e" * 64}}
        ctx = tool.build_one(c, META, "m" * 64, out, {"code": "x"})
        enc, tea, a0, rep = original_l7b(A, META, self.R, self.T)
        self.assertEqual([(out / n).read_bytes() for n in ("encoder.jtt1.gz", "teacher.jtt1.gz", "a0.f32")], [enc, tea, a0])
        self.assertEqual(ctx["t1prep"], rep)
        self.assertEqual(sorted(ctx), sorted(["schema", "view_id", "dataset_hash", "fold", "context_sha256", "outer_train_indices_sha256",
                                              "outer_test_indices_sha256", "n_train", "n_test", "files", "scaler_sha256",
                                              "inner_labels_sha256", "inner_loads", "t1prep", "t1_meta_sha256", "values_mode",
                                              "published", "code_sha256", "utc_us"]))
        self.assertEqual((ctx["schema"], ctx["view_id"], ctx["context_sha256"]), ("jope.dl-v2-local-context/v1", f"{H}__r0__f3", "e" * 64))
        self.assertEqual(ctx["outer_test_indices_sha256"], hashlib.sha256(json.dumps(self.T).encode()).hexdigest())

    def test_l7a_default_build_is_byte_identical_to_the_original(self):
        out = scratch()
        built = l7a.build_l7a({"R": self.R, "T": self.T}, VALUES, RAW_SPECS, out)
        enc, tea, a0, rep = original_l7a(VALUES, RAW_SPECS, self.R, self.T)
        self.assertEqual([(out / n).read_bytes() for n in ("encoder.jtt1.gz", "teacher.jtt1.gz", "a0.f32")], [enc, tea, a0])
        self.assertEqual(built["t1raw"], rep)
        self.assertEqual(built["files"]["encoder.jtt1.gz"]["sha256"], hashlib.sha256(enc).hexdigest())
        self.assertEqual(sorted(built), sorted(["outer_train_indices_sha256", "outer_test_indices_sha256", "n_train", "n_test",
                                                "files", "scaler_sha256", "inner_labels_sha256", "inner_loads", "t1raw"]))


# ------------------------------------------------------------------------------------- epoch-0 universe

def fake_ctx(view, fold, n, T, data="a" * 64, shape=(N, N)):
    tset = set(T)
    return {"row": {"view_id": view}, "fold": fold, "A": np.zeros((n, 3)), "R": [i for i in range(n) if i not in tset], "T": T,
            "refs": {"data_sha256": data}, "shape": {"sample_rows": shape[0], "source_rows": shape[1]}}


class Epoch0(unittest.TestCase):
    def test_universe_is_the_whole_shared_data_object(self):
        ctxs = [fake_ctx(f"v{f}", f, N, list(range(f, N, 5))) for f in (0, 2, 4)]
        M, data, cov = tool.epoch0_universe(ctxs)
        self.assertEqual((M.shape[0], data, cov["rows"], cov["epoch0_folds"], cov["subsampled"]), (N, "a" * 64, N, [0, 2, 4], False))
        self.assertEqual(cov["epoch0_test_rows_union"], len(set(range(0, N, 5)) | set(range(2, N, 5)) | set(range(4, N, 5))))

    def test_refusals(self):
        two = [fake_ctx("v0", 0, N, list(range(0, N, 5))), fake_ctx("v1", 1, N, list(range(1, N, 5)), data="b" * 64)]
        with self.assertRaises(tool.SplitV2Error) as cm:
            tool.epoch0_universe(two)
        self.assertEqual(cm.exception.reason, "epoch0_contexts_on_different_data_objects")
        bad = fake_ctx("v0", 0, N, list(range(0, N, 5)))
        bad["R"] = bad["R"][:-1]                                               # a row in neither R nor T
        with self.assertRaises(tool.SplitV2Error) as cm:
            tool.epoch0_universe([bad])
        self.assertEqual(cm.exception.reason, "epoch0_context_not_full_universe")

    def test_l7a_rematch_must_reproduce_the_record(self):
        m = {"status": "matched", "values": VALUES, "specs": RAW_SPECS, "read": {"r": 1}, "stats": {"s": 1}}
        rec = {"values_sha256": hashlib.sha256(np.ascontiguousarray(VALUES, "<f8").tobytes()).hexdigest(), "read": {"r": 1},
               "match": {"s": 1}}
        vals, specs, rep = l7a.check_rematch(m, rec)
        self.assertTrue(rep["read_report_equal"] and rep["match_stats_equal"])
        for k, v, reason in (("values_sha256", "0" * 64, "rematch_values_differ"), ("read", {"r": 2}, "rematch_read_report_differs")):
            with self.assertRaises(l7a.local.SplitV2Error) as cm:          # the L7a tool's own copy of the machinery
                l7a.check_rematch(m, dict(rec, **{k: v}))
            self.assertEqual(cm.exception.reason, reason)

    def test_new_trees_are_not_epoch0_trees(self):
        for t in tool.NEW_TREES:
            self.assertNotIn(t, tool.OLD_TREES)
            self.assertTrue(t.name.endswith(("-v2split", "-v2gsplit")))


if __name__ == "__main__":
    unittest.main()
