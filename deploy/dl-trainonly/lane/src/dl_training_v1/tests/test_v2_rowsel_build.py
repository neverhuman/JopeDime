"""Offline proofs for tools/v2_rowsel_build.py (ROWSEL-TRAINONLY T1 rebuild). No S3, no network, temp trees only.

(i)    train-side invariance: changing outer-test row values changes nothing in the train selection, the encoder JTT1
       train block, the teacher train block, the scaler, the inner labels or A0;
(ii)   draw invariance: the test rows are unchanged when any row's values change;
(iii)  must-fail controls: a selector that fits on train and test rows together (fit = U), and a target-keyed draw, are
       caught by the same checks and blocked by the counterfactual gate. A downstream min-max over all rows is
       blocked by the downstream gate;
(iv)   re-indexing: R and T in A map back to the right universe positions, also read back from the written files;
(v)    the files equal tools/v2_build_local.build_one's for the same (A, R, T, meta);
(vi)   resume skips .done, and a counterfactual failure blocks .done;
(vii)  scope and labels, identity, and the append-only epoch-1 overlay.
"""
import dataclasses
import gzip
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

from dltrain import jtt1, rowsel, t1prep, trainonly

SRC = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rb = _load("v2_rowsel_build")
bl = _load("v2_build_local")

H = "fa11ba5e00c0ffee"
H2 = "0dd5a1e5c0ffee00"
N, NTR, BUDGET, K = 800, 640, 360, 5
COLS = ["x0", "x1", "x2", "x3", "cat", "target"]
META = {"output_headers": COLS, "feature_columns": COLS[:-1],
        "feature_scaling": {"method": "minmax fixture", "min": [0.0] * 5, "max": [1.0, 1.0, 1.0, 1.0, 3.0],
                            "range": [0.0, 1.0]},
        "feature_encodings": {"cat": {"kind": "categorical_codes", "impute_value": 0,
                                      "mapping": {"a": 0, "b": 1, "c": 2, "__MISSING__": 3}}}}
CODES = {f: "0" * 64 for f in rb.CODE_FILES}
REFS = {"row_selection_leak": {"path": "fixture", "sha256": "1" * 64}, "acq_selection": {"path": "fixture", "sha256": "2" * 64}}


def _r9(a):
    return np.vectorize(lambda v: float("%.9g" % v))(a)


def fixture_matrix(seed=20261005, n=N):
    """T1-like values: every feature in [0, 0.9] (min-max scaled at acquisition), a categorical code column
    (code / 3), a target in [0, 1]; all printed '%.9g' as the acquisition tool does."""
    rs = np.random.RandomState(seed)
    g = rs.randint(0, 4, n)
    m = np.empty((n, 6))
    m[:, 0] = np.clip(0.2 * g + rs.normal(0, 0.05, n), 0, 0.9)
    m[:, 1] = rs.randint(0, 10, n) / 10.0
    m[:, 2] = rs.beta(2, 5, n) * 0.9
    m[:, 3] = np.where(rs.random_sample(n) < 0.8, 0.1, rs.uniform(0, 0.9, n))
    m[:, 4] = rs.randint(0, 4, n) / 3.0
    m[:, 5] = np.clip(0.5 * m[:, 0] + 0.3 * m[:, 2] + 0.05 * m[:, 4] + rs.normal(0, 0.05, n), 0, 1)
    return _r9(m)


def lines_of(m):
    return [",".join("%.9g" % v for v in row).encode() for row in m]


def universe_of(m, h=H, ntr=NTR):
    ls = lines_of(m)
    return rowsel.universe_from_t1(h, b"\n".join(ls[:ntr]) + b"\n", b"\n".join(ls[ntr:]) + b"\n", m.shape[1])


def matrix_of(U):
    return np.column_stack([U.X, U.y])


def fixture_folds(n=N, k=K, seed=7):
    order = np.random.RandomState(seed).permutation(n)
    return {f: sorted(int(i) for i in order[f::k]) for f in range(k)}


def perturb(m, positions, seed=99):
    """Valid T1 values at `positions`: new numeric features and target in [0, 1], categorical code shifted by one."""
    rs = np.random.RandomState(seed)
    out = m.copy()
    idx = np.asarray(sorted(positions), dtype=np.int64)
    out[idx, :4] = rs.uniform(0, 0.9, (idx.size, 4))
    out[idx, 4] = (np.rint(out[idx, 4] * 3) + 1) % 4 / 3.0
    out[idx, 5] = rs.uniform(0, 1, idx.size)
    return _r9(out)


def dataset(m=None, h=H, folds=None):
    m = fixture_matrix() if m is None else m
    U = universe_of(m, h)
    folds = fixture_folds(U.n) if folds is None else folds
    return {"h": h, "U": U, "folds": folds, "meta": META, "meta_sha256": hashlib.sha256(json.dumps(META).encode()).hexdigest(),
            "t1": {"train": {"sha256": "a" * 64, "bytes": 1, "lines": NTR}, "test": {"sha256": "b" * 64, "bytes": 1, "lines": U.n - NTR}},
            "split": {"kind": "registry", "sha256": "c" * 64, "version_id": "fixture", "key": "fixture", "bytes": 1,
                      "schema": rb.SPLIT_SCHEMA, "method": rb.SPLIT_METHOD, "track": "fixture", "effective_folds": K, "repeat": 0},
            "lineage": {"manifest_sha256": "d" * 64, "membership_sha256": "e" * 64, "data_sha256": "f" * 64,
                        "membership_selected_sha256_matches_manifest": True, "port_reproduces_v1_selection": True,
                        "membership_remap_matches_epoch0_folds": {str(f): True for f in folds},
                        "universe_rows": U.n, "features": U.features},
            "load_cpu_s": 0.0}


def epoch_rows(h=H, folds=range(K), cohort="incoming"):
    return [{"view_id": f"{h}__r0__f{f}__test_adaptive_{hashlib.sha256(f'{h}{f}'.encode()).hexdigest()[:12]}",
             "dataset_hash": h, "fold": f, "cohort": cohort,
             "context_sha256": hashlib.sha256(f"epoch0|{h}|{f}".encode()).hexdigest()} for f in folds]


def items(h=H, label="clean", folds=range(K), cohort="incoming"):
    acq = {"class": "kmeans", "cohort": cohort, "reduced": label == "acquisition_cap",
           "selection_markers": ["original_size.downsampled"] if label == "acquisition_cap" else []}
    return [(r, label, acq) for r in epoch_rows(h, folds, cohort)]


def no_guard():
    return 0


# ------------------------------------------------------------------------------------------------- leaky controls

def fit_on_union_view(universe, parent_folds, fold, budget=rowsel.DEFAULT_BUDGET, seed=None, quotas=None):
    """MUST-FAIL control: the sampler fitted on U (train and outer-test rows together), outer-test rows then removed."""
    folds = rowsel.validate_parent_folds(universe, parent_folds)
    seed = rowsel.derive_seed(universe.dataset_hash) if seed is None else seed
    if quotas is None:
        quotas = rowsel.fold_quotas({f: v.size for f, v in folds.items()}, budget)[int(fold)]
    n_train, n_test = (int(q) for q in quotas)
    held = set(folds[int(fold)].tolist())
    fit = rowsel.sampler_select(universe, universe.positions.tolist(), n_train + n_test, seed)
    train = [p for p in fit.positions if p not in held][:n_train]
    ids = [universe.row_ids[i] for i in universe.index_of(folds[int(fold)])]
    drawn = set(rowsel.draw_test_rows(universe.dataset_hash, ids, n_test))
    test = [int(p) for p, rid in zip(folds[int(fold)], ids) if rid in drawn]
    return rowsel.make_view(universe, fold, budget, (n_train, n_test), seed, dataclasses.replace(fit, positions=tuple(train)), test)


def target_keyed_draw_view(universe, parent_folds, fold, budget=rowsel.DEFAULT_BUDGET, seed=None, quotas=None):
    """MUST-FAIL control for the draw: the test rows are the smallest targets of the fold."""
    v = rowsel.select_view(universe, parent_folds, fold, budget, seed, quotas)
    fold_pos = sorted(int(p) for p in parent_folds[fold])
    ys = universe.y[universe.index_of(fold_pos)]
    chosen = [p for _, p in sorted(zip(ys.tolist(), fold_pos))[:v.quotas[1]]]
    return rowsel.make_view(universe, fold, budget, v.quotas, v.seed, v.train_fit, chosen)


def train_side(select_fn, ds, f):
    view = select_fn(ds["U"], ds["folds"], f, BUDGET, None, quotas=None)
    built = rb.assemble(ds["U"], view, ds["meta"])
    return view, built, {"train_positions": view.train_positions, "train_fit_identity": view.train_fit.identity,
                         **rb.train_side_fields(built["art"])}


def train_side_diff(select_fn, ds, ds2, f):
    _, b1, a = train_side(select_fn, ds, f)
    _, b2, b = train_side(select_fn, ds2, f)
    return sorted(k for k in a if a[k] != b[k]), b1, b2


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="rowsel-build-"))
        cls.m = fixture_matrix()
        cls.ds = dataset(cls.m)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------- (i) train side

class TrainSideInvariance(Fixture):
    def test_outer_test_value_change_leaves_train_side_identical(self):
        for f in range(K):
            ds2 = dataset(perturb(self.m, self.ds["folds"][f], seed=100 + f))
            differs, b1, b2 = train_side_diff(rowsel.select_view, self.ds, ds2, f)
            self.assertEqual(differs, [], f"fold {f}")
            # the perturbation really reached the context: its outer-test rows and encoder bytes changed
            self.assertFalse(np.array_equal(b1["A"][b1["T"]], b2["A"][b2["T"]]))
            self.assertNotEqual(hashlib.sha256(b1["art"]["split"]["jtt1"]).hexdigest(),
                                hashlib.sha256(b2["art"]["split"]["jtt1"]).hexdigest())

    def test_written_context_train_fields_identical_after_test_change(self):
        f = 2
        ds2 = dataset(perturb(self.m, self.ds["folds"][f], seed=7))
        out = self.tmp / "ts"
        c1 = rb.rebuild_context(self.ds, epoch_rows()[f], "clean", items()[0][2], out / "a", CODES, REFS, budget=BUDGET)["context"]
        c2 = rb.rebuild_context(ds2, epoch_rows()[f], "clean", items()[0][2], out / "b", CODES, REFS, budget=BUDGET)["context"]
        for k in ("scaler_sha256", "inner_labels_sha256", "outer_train_indices_sha256", "n_train"):
            self.assertEqual(c1[k], c2[k], k)
        self.assertEqual(c1["files"]["a0.f32"], c2["files"]["a0.f32"])
        self.assertEqual(c1["rowsel"]["train_positions"], c2["rowsel"]["train_positions"])
        self.assertEqual(c1["rowsel"]["view"]["train_fit_identity"], c2["rowsel"]["view"]["train_fit_identity"])
        self.assertNotEqual(c1["files"]["encoder.jtt1.gz"]["raw_sha256"], c2["files"]["encoder.jtt1.gz"]["raw_sha256"])
        r1 = gzip.decompress((out / "a" / c1["view_id"] / "encoder.jtt1.gz").read_bytes())
        r2 = gzip.decompress((out / "b" / c2["view_id"] / "encoder.jtt1.gz").read_bytes())
        self.assertEqual(jtt1.train_rows_sha256(r1), jtt1.train_rows_sha256(r2))

    def test_small_change_to_one_train_row_is_detected(self):
        """The check can fail: one selected train row's target moved by one float32 ulp changes the train side."""
        f = 1
        view, _, _ = train_side(rowsel.select_view, self.ds, f)
        p = view.train_positions[0]
        m2 = self.m.copy()
        m2[p, 5] = float(np.nextafter(np.float32(m2[p, 5]), np.float32(2)))
        differs, _, _ = train_side_diff(rowsel.select_view, self.ds, dataset(m2), f)
        self.assertIn("encoder_train_rows_sha256", differs)


# ------------------------------------------------------------------------------------------------- (ii) draw

class DrawInvariance(Fixture):
    def test_test_rows_unchanged_when_any_row_values_change(self):
        everything = dataset(perturb(self.m, range(N), seed=5))
        for f in range(K):
            v1 = rowsel.select_view(self.ds["U"], self.ds["folds"], f, BUDGET)
            v2 = rowsel.select_view(everything["U"], everything["folds"], f, BUDGET)
            self.assertEqual(v1.test_positions, v2.test_positions)
            self.assertEqual(v1.test_draw_sha256, v2.test_draw_sha256)
            self.assertNotEqual(v1.train_values_sha256, v2.train_values_sha256)

    def test_target_keyed_draw_is_caught(self):
        f = 0
        ds2 = dataset(perturb(self.m, self.ds["folds"][f], seed=11))
        v1 = target_keyed_draw_view(self.ds["U"], self.ds["folds"], f, BUDGET)
        v2 = target_keyed_draw_view(ds2["U"], ds2["folds"], f, BUDGET)
        self.assertNotEqual(v1.test_positions, v2.test_positions)
        rep = rowsel.counterfactual_report(target_keyed_draw_view, self.ds["U"], self.ds["folds"], f, BUDGET)
        self.assertFalse(rep["test_draw_invariant"])


# ------------------------------------------------------------------------------------------------- (iii) must fail

class MustFailControls(Fixture):
    def test_train_union_test_selector_is_caught_by_the_same_check(self):
        caught = 0
        for f in range(K):
            ds2 = dataset(perturb(self.m, self.ds["folds"][f], seed=200 + f))
            differs, _, _ = train_side_diff(fit_on_union_view, self.ds, ds2, f)
            self.assertIn("train_fit_identity", differs)
            caught += "train_positions" in differs or "encoder_train_rows_sha256" in differs
        self.assertGreater(caught, 0)

    def test_v1_selector_is_caught_by_the_same_check(self):
        f = 3
        ds2 = dataset(perturb(self.m, self.ds["folds"][f], seed=17))
        differs, _, _ = train_side_diff(rowsel.v1_view, self.ds, ds2, f)
        self.assertIn("train_fit_identity", differs)

    def test_counterfactual_gate_blocks_the_union_selector(self):
        out = self.tmp / "gate"
        res = rb.rebuild_context(self.ds, epoch_rows()[0], "clean", items()[0][2], out, CODES, REFS, budget=BUDGET,
                                 select_fn=fit_on_union_view)
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["reason"], "counterfactual_failed")
        self.assertFalse(out.exists() and any(out.iterdir()))

    def test_counterfactual_gate_blocks_a_target_keyed_draw(self):
        res = rb.rebuild_context(self.ds, epoch_rows()[0], "clean", items()[0][2], self.tmp / "gate2", CODES, REFS,
                                 budget=BUDGET, select_fn=target_keyed_draw_view)
        self.assertEqual((res["status"], res["reason"]), ("failed", "counterfactual_failed"))

    def test_parent_fold_guard_refuses_test_rows_outside_the_fold(self):
        """A value-free selector that moves 5 complement rows into the test set passes the counterfactual check (its
        train side and draw never read a value). The parent-fold guard is the check that refuses it."""
        def outside(universe, parent_folds, fold, budget=rowsel.DEFAULT_BUDGET, seed=None, quotas=None):
            folds = rowsel.validate_parent_folds(universe, parent_folds)
            seed = rowsel.derive_seed(universe.dataset_hash) if seed is None else seed
            if quotas is None:
                quotas = rowsel.fold_quotas({f: v.size for f, v in folds.items()}, budget)[int(fold)]
            n_train, n_test = (int(q) for q in quotas)
            held = set(folds[int(fold)].tolist())
            comp = [int(p) for p in universe.positions if int(p) not in held]
            cids = [universe.row_ids[i] for i in universe.index_of(comp)]
            moved = set(rowsel.draw_test_rows(universe.dataset_hash, cids, 5))
            extra = [p for p, rid in zip(comp, cids) if rid in moved]
            fit = rowsel.sampler_select(universe, [p for p in comp if p not in set(extra)], n_train, seed)
            fids = [universe.row_ids[i] for i in universe.index_of(folds[int(fold)])]
            drawn = set(rowsel.draw_test_rows(universe.dataset_hash, fids, n_test - 5))
            test = [int(p) for p, rid in zip(folds[int(fold)], fids) if rid in drawn] + extra
            return rowsel.make_view(universe, fold, budget, (n_train, n_test), seed, fit, test)

        rep = rowsel.counterfactual_report(outside, self.ds["U"], self.ds["folds"], 1, BUDGET)
        self.assertTrue(rep["train_side_invariant"] and rep["test_draw_invariant"])
        out = self.tmp / "guard"
        res = rb.rebuild_context(self.ds, epoch_rows()[1], "clean", items()[0][2], out, CODES, REFS, budget=BUDGET,
                                 select_fn=outside)
        self.assertEqual((res["status"], res["reason"]), ("failed", "view_outside_parent_fold"))
        self.assertFalse(out.exists() and any(out.iterdir()))

    def test_counterfactual_gate_refuses_a_view_it_did_not_check(self):
        """The written view must be the view the counterfactual check rebuilt: a selector whose first answer differs
        from its later ones (the check only sees the later ones) is refused."""
        calls = []

        def flaky(universe, parent_folds, fold, budget=rowsel.DEFAULT_BUDGET, seed=None, quotas=None):
            calls.append(1)
            return rowsel.select_view(universe, parent_folds, fold, budget, "another-seed" if len(calls) == 1 else seed, quotas)

        res = rb.rebuild_context(self.ds, epoch_rows()[2], "clean", items()[0][2], self.tmp / "flaky", CODES, REFS,
                                 budget=BUDGET, select_fn=flaky)
        self.assertEqual((res["status"], res["reason"]), ("failed", "counterfactual_failed"))
        self.assertGreater(len(calls), 10)

    def test_downstream_gate_blocks_an_all_row_scaler(self):
        f = 4
        view = rowsel.select_view(self.ds["U"], self.ds["folds"], f, BUDGET)
        m2 = self.m.copy()
        m2[view.test_positions[0], 0] = 1.0          # an outer-test row holds the all-row maximum of x0
        ds2 = dataset(m2)
        orig = t1prep.prepare

        def all_row_minmax(A, meta, R, T, **kw):     # UPSTREAM-SCALE-01: min-max over every row of A
            split = orig(A, meta, R, T, **kw)
            fitted, _ = t1prep.refit(A, meta, R)
            lo = fitted.min(axis=0)
            span = fitted.max(axis=0) - lo
            span = np.where(span == 0, 1.0, span)
            split["lo"], split["span"] = lo, span
            split["jtt1"] = trainonly.jtt1_bytes((fitted[R] - lo) / span, (fitted[T] - lo) / span, split["inner_labels"])
            return split

        clean = rb.rebuild_context(ds2, epoch_rows()[f], "clean", items()[0][2], self.tmp / "dcf-ok", CODES, REFS, budget=BUDGET)
        self.assertEqual(clean["status"], "done")
        with mock.patch.object(rb.t1prep, "prepare", all_row_minmax):
            res = rb.rebuild_context(ds2, epoch_rows()[f], "clean", items()[0][2], self.tmp / "dcf", CODES, REFS, budget=BUDGET)
        self.assertEqual((res["status"], res["reason"]), ("failed", "downstream_counterfactual_failed"))
        self.assertIn("scaler_sha256", json.loads(res["detail"]))
        self.assertFalse((self.tmp / "dcf").exists())

    def test_downstream_counterfactual_really_changes_test_rows(self):
        view = rowsel.select_view(self.ds["U"], self.ds["folds"], 0, BUDGET)
        built = rb.assemble(self.ds["U"], view, META)
        dcf = rb.downstream_counterfactual(self.ds["U"], built, META, H)
        self.assertTrue(dcf["train_side_invariant"])
        self.assertGreater(dcf["test_rows_changed"], len(built["T"]) // 2)
        self.assertTrue(dcf["encoder_raw_changed"])


# ------------------------------------------------------------------------------------------------- (iv)(v) build

class BuiltContexts(Fixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.out = cls.tmp / "built"
        cls.rec = rb.build_shard(cls.out, 0, 1, items(H, "clean") + items(H2, "acquisition_cap", folds=(0, 2)),
                                 lambda h, rows: cls.ds if h == H else dataset(fixture_matrix(seed=3), H2),
                                 CODES, REFS, budget=BUDGET, guard_fn=no_guard)
        cls.ctx = {}
        for d in sorted((cls.out / "shard-0").iterdir()):
            c = json.loads((d / "context.json").read_bytes())
            cls.ctx[(c["dataset_hash"], c["fold"])] = (d, c)

    def test_shard_receipt(self):
        r = self.rec
        self.assertEqual((r["expected"], r["built_now"], r["done"], r["failed"], r["complete"]), (7, 7, 7, [], True))
        self.assertEqual(r["done_by_label"], {"clean": 5, "acquisition_cap": 2})
        self.assertEqual(r["counterfactual_failures"], 0)
        self.assertTrue((self.out / "shard-0-of-1.receipt.json").exists())
        self.assertEqual(sorted(p.name for p in (self.out / "_datasets").iterdir()), [f"{H2}.json", f"{H}.json"])

    def test_reindex_maps_back_to_universe_positions(self):
        U = self.ds["U"]
        for f in range(K):
            d, c = self.ctx[(H, f)]
            view = rowsel.select_view(U, self.ds["folds"], f, BUDGET)
            built = rb.assemble(U, view, META)
            apos, A, R, T = built["apos"], built["A"], built["R"], built["T"]
            self.assertEqual([apos[i] for i in R], list(view.train_positions))
            self.assertEqual([apos[i] for i in T], list(view.test_positions))
            self.assertEqual(sorted(R + T), list(range(len(apos))))
            self.assertEqual(A.tobytes(), matrix_of(U)[apos].tobytes())
            self.assertEqual(c["rowsel"]["train_positions"], list(view.train_positions))
            self.assertEqual(c["rowsel"]["test_positions"], list(view.test_positions))
            self.assertEqual(c["outer_train_indices_sha256"], hashlib.sha256(json.dumps(R).encode()).hexdigest())
            self.assertEqual(c["outer_test_indices_sha256"], hashlib.sha256(json.dumps(T).encode()).hexdigest())
            self.assertEqual(c["rowsel"]["a_positions_sha256"], hashlib.sha256(json.dumps(apos).encode()).hexdigest())
            # read back from disk: the teacher JTT1 keeps T1 values for the numeric columns and the target, so its
            # rows must equal the universe rows at the train positions, then at the test positions, in that order
            raw = gzip.decompress((d / "teacher.jtt1.gz").read_bytes())
            h = jtt1.parse_header(raw)
            self.assertEqual((h["n_train"], h["n_test"]), (len(R), len(T)))
            block = np.frombuffer(raw[17 + h["n_train"]:], dtype="<f8").reshape(-1, h["p"] + 1)
            want = matrix_of(U)[list(view.train_positions) + list(view.test_positions)]
            num = [0, 1, 2, 3]
            self.assertEqual(block[:, num].tobytes(), np.ascontiguousarray(want[:, num]).tobytes())
            self.assertEqual(block[:, -1].tobytes(), np.ascontiguousarray(want[:, -1]).tobytes())

    def test_view_quotas_and_partition(self):
        for f in range(K):
            _, c = self.ctx[(H, f)]
            held = set(self.ds["folds"][f])
            tr, te = c["rowsel"]["train_positions"], c["rowsel"]["test_positions"]
            self.assertFalse(set(tr) & held)
            self.assertTrue(set(te) <= held)
            q = rowsel.fold_quotas({k: len(v) for k, v in self.ds["folds"].items()}, BUDGET)[f]
            self.assertEqual((c["n_train"], c["n_test"]), q)
            self.assertEqual(c["n_train"] + c["n_test"], BUDGET)
            self.assertEqual([c["rowsel"]["quotas"]["n_train"], c["rowsel"]["quotas"]["n_test"]], list(q))
            self.assertTrue(c["rowsel"]["quotas"]["quota_met"])

    def test_files_equal_build_one(self):
        U = self.ds["U"]
        for f in (0, 3):
            d, c = self.ctx[(H, f)]
            view = rowsel.select_view(U, self.ds["folds"], f, BUDGET)
            built = rb.assemble(U, view, META)
            ref = self.tmp / f"build_one-{f}"
            bl.build_one({"A": built["A"], "R": built["R"], "T": built["T"], "row": epoch_rows()[f]}, META, "m", ref, {})
            for name in ("encoder.jtt1.gz", "teacher.jtt1.gz", "a0.f32"):
                self.assertEqual((d / name).read_bytes(), (ref / name).read_bytes(), name)
            lc = json.loads((ref / "context.json").read_bytes())
            for k in ("scaler_sha256", "inner_labels_sha256", "inner_loads", "outer_train_indices_sha256",
                      "outer_test_indices_sha256", "t1prep", "n_train", "n_test"):
                self.assertEqual(lc[k], c[k], k)

    def test_context_fields_and_identity(self):
        for (h, f), (d, c) in self.ctx.items():
            capped = h == H2
            self.assertEqual(c["schema"], "jope.dl-v2-rowsel-context/v1")
            self.assertEqual((c["values_mode"], c["published"]), ("L7b", False))
            self.assertEqual(c["row_selection_leak"], ["acquisition_cap"] if capped else ["clean"])
            self.assertEqual(c["row_selection_leak_epoch0"], ["kmeans"])
            rs = c["rowsel"]
            self.assertEqual(rs["route"], "t1")
            self.assertIs(rs.get("raw_route_superseded_if_matched"), True if capped else None)
            self.assertIs(rs["outer_split_changed"], False)
            self.assertEqual(rs["universe"]["kind"], "t1")
            self.assertEqual(rs["parent_split"]["kind"], "registry")
            self.assertEqual(rs["seed"], rowsel.derive_seed(h))
            e0 = [r for r in epoch_rows(h) if r["fold"] == f][0]
            self.assertEqual((c["epoch0"]["view_id"], c["epoch0"]["context_sha256"]), (e0["view_id"], e0["context_sha256"]))
            self.assertEqual(c["identity"]["epoch0"], {"view_id": e0["view_id"], "context_sha256": e0["context_sha256"]})
            view = rowsel.select_view(dataset(fixture_matrix(seed=3), H2)["U"] if capped else self.ds["U"],
                                      fixture_folds(), f, BUDGET)
            self.assertEqual(rs["view"]["identity"], view.identity)
            self.assertEqual(c["view_id"], f"{h}__r0__f{f}__rowsel1_{view.identity[:12]}")
            self.assertEqual(d.name, c["view_id"])
            self.assertEqual(c["context_sha256"], hashlib.sha256(json.dumps(c["identity"], sort_keys=True,
                                                                             separators=(",", ":")).encode()).hexdigest())
            self.assertNotEqual(c["context_sha256"], e0["context_sha256"])
            self.assertEqual(c["identity"]["files_raw_sha256"]["encoder.jtt1"], c["files"]["encoder.jtt1.gz"]["raw_sha256"])
            self.assertTrue(rs["counterfactual"]["train_side_invariant"] and rs["counterfactual"]["test_draw_invariant"])
            self.assertTrue(rs["downstream_counterfactual"]["train_side_invariant"])
            self.assertEqual(sorted(p.name for p in d.iterdir()),
                             [".done", "a0.f32", "context.json", "encoder.jtt1.gz", "teacher.jtt1.gz"])
            self.assertEqual(rb.verify_context_dir(d)["problems"], [])

    def test_verify_detects_a_tampered_file(self):
        d, c = self.ctx[(H, 1)]
        copy = self.tmp / "tamper" / d.name
        shutil.copytree(d, copy)
        with open(copy / "a0.f32", "r+b") as fh:
            fh.write(b"\0")
        self.assertIn("file:a0.f32", rb.verify_context_dir(copy)["problems"])


# ------------------------------------------------------------------------------------------------- (vi) resume, gate

class ResumeAndGate(Fixture):
    def test_resume_skips_done(self):
        out = self.tmp / "resume"
        calls = []

        def loader(h, rows):
            calls.append(h)
            return self.ds

        r1 = rb.build_shard(out, 0, 1, items(), loader, CODES, REFS, budget=BUDGET, guard_fn=no_guard)
        self.assertEqual((r1["built_now"], r1["already_done"], r1["done"]), (5, 0, 5))
        before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in out.glob("shard-0/*/*")}
        r2 = rb.build_shard(out, 0, 1, items(), loader, CODES, REFS, budget=BUDGET, guard_fn=no_guard)
        self.assertEqual((r2["built_now"], r2["already_done"], r2["done"], r2["complete"]), (0, 5, 5, True))
        self.assertEqual(calls, [H])
        self.assertEqual(before, {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in out.glob("shard-0/*/*")})
        victim = sorted((out / "shard-0").iterdir())[2]
        (victim / ".done").unlink()
        r3 = rb.build_shard(out, 0, 1, items(), loader, CODES, REFS, budget=BUDGET, guard_fn=no_guard)
        self.assertEqual((r3["built_now"], r3["already_done"], r3["done"]), (1, 4, 5))
        self.assertEqual(calls, [H, H])

    def test_counterfactual_failure_blocks_done_and_is_recorded(self):
        out = self.tmp / "cf-fail"
        r = rb.build_shard(out, 0, 1, items(), lambda h, rows: self.ds, CODES, REFS, budget=BUDGET,
                           select_fn=fit_on_union_view, guard_fn=no_guard)
        self.assertEqual((r["built_now"], r["done"], r["complete"], r["counterfactual_failures"]), (0, 0, False, 5))
        self.assertEqual({x["reason"] for x in r["failed"]}, {"counterfactual_failed"})
        self.assertEqual({x["view_id"] for x in r["failed"]}, {e["view_id"] for e in epoch_rows()})
        self.assertEqual(list((out / "shard-0").iterdir()), [])
        r2 = rb.build_shard(out, 0, 1, items(), lambda h, rows: self.ds, CODES, REFS, budget=BUDGET, guard_fn=no_guard)
        self.assertEqual((r2["built_now"], r2["done"], r2["complete"]), (5, 5, True))

    def test_dataset_failure_fails_every_context_with_its_reason(self):
        def loader(h, rows):
            raise rb.RebuildError("membership_remap", "fixture")
        r = rb.build_shard(self.tmp / "ds-fail", 0, 1, items(), loader, CODES, REFS, budget=BUDGET, guard_fn=no_guard)
        self.assertEqual([x["reason"] for x in r["failed"]], ["membership_remap"] * 5)
        self.assertEqual(r["datasets"][H], {"status": "failed", "reason": "membership_remap"})

    def test_sharding_rule(self):
        out = self.tmp / "shards"
        want = int(H[:8], 16) % 2
        r = rb.build_shard(out, 1 - want, 2, items(), lambda h, rows: self.ds, CODES, REFS, budget=BUDGET, guard_fn=no_guard)
        self.assertEqual((r["expected"], r["done"]), (0, 0))


# ------------------------------------------------------------------------------------------------- (vii) scope

class Scope(unittest.TestCase):
    def rows_and_labels(self):
        rows, leak, acq = [], {}, {}
        spec = [("aa", "incoming", ["kmeans"], True), ("bb", "incoming", ["kmeans"], False), ("cc", "frozen", ["kmeans"], False),
                ("dd", "frozen", ["all_rows_coreset"], True), ("ee", "incoming", ["clean"], False),
                ("ff", "frozen", ["clean"], True)]
        for h, cohort, label, capped in spec:
            h = h * 8
            for f in range(2):
                r = {"view_id": f"{h}__r0__f{f}__x", "dataset_hash": h, "fold": f, "cohort": cohort, "context_sha256": "0" * 64}
                rows.append(r)
                leak[r["view_id"]] = {"view_id": r["view_id"], "dataset_hash": h, "fold": f, "cohort": cohort,
                                      "row_selection_leak": label}
            acq[h] = {"cohort": cohort, "class": label[0], "reduced": capped,
                      "selection_markers": ["original_size.downsampled"] if capped else []}
        return rows, leak, acq

    def test_admits_only_incoming_kmeans_and_labels_by_acquisition_cap(self):
        rows, leak, acq = self.rows_and_labels()
        got = [(r["dataset_hash"][:2], r["fold"], lab) for r, lab, _ in rb.select_scope(rows, leak, acq)]
        self.assertEqual(got, [("aa", 0, "acquisition_cap"), ("aa", 1, "acquisition_cap"), ("bb", 0, "clean"), ("bb", 1, "clean")])

    def test_refuses_unlabelled_or_inconsistent_inputs(self):
        rows, leak, acq = self.rows_and_labels()
        del leak[rows[0]["view_id"]]
        with self.assertRaises(rb.RebuildError):
            rb.select_scope(rows, leak, acq)
        rows, leak, acq = self.rows_and_labels()
        acq["aa" * 8]["selection_markers"] = []
        with self.assertRaises(rb.RebuildError):
            rb.select_scope(rows, leak, acq)

    @unittest.skipUnless(rb.LEAK.exists() and rb.ACQ.exists() and rb.EPOCH0.exists(), "lane inputs not present")
    def test_real_scope_is_the_admitted_class(self):
        rows = [c for c in json.loads(rb.EPOCH0.read_bytes())["contexts"] if c.get("status") == "included"]
        scope, refs, counts = rb.load_scope(rows)
        self.assertEqual(counts, {"datasets": 375, "contexts": 1749, "clean": 810, "acquisition_cap": 939})
        self.assertEqual({r["cohort"] for r, _, _ in scope}, {"incoming"})


class UniverseChecks(unittest.TestCase):
    def test_line_split_must_equal_materialize(self):
        good = b"0.1,0.2,0.3\n0.4,0.5,0.6\n"
        self.assertEqual(rb.t1_universe(H, good, good, 3).n, 4)
        with self.assertRaises(rb.RebuildError) as cm:
            rb.t1_universe(H, b"0.1,0.2,0.3\n\r\n0.4,0.5,0.6\n", good, 3)
        self.assertEqual(cm.exception.reason, "line_split_differs")

    def test_a_matrix_refuses_bytes_that_disagree_with_values(self):
        U = universe_of(fixture_matrix(n=40), ntr=30)
        rb.a_matrix(U, [0, 5, 31])
        bad = list(U.row_bytes)
        bad[5] = lines_of(fixture_matrix(seed=1, n=40))[5]
        U2 = rowsel.Universe(U.dataset_hash, U.positions, U.row_ids, U.X, U.y, tuple(bad))
        with self.assertRaises(rb.RebuildError) as cm:
            rb.a_matrix(U2, [0, 5, 31])
        self.assertEqual(cm.exception.reason, "a_matrix_mismatch")

    def test_parent_folds_must_partition_and_be_contiguous(self):
        split = {"effective_folds": 2, "folds": [{"repeat": 0, "fold": 0, "test_indices": [0, 2]},
                                                 {"repeat": 0, "fold": 1, "test_indices": [1, 3]},
                                                 {"repeat": 1, "fold": 0, "test_indices": [0, 1]}]}
        self.assertEqual(rb.parent_folds_from_split(split, 4), {0: [0, 2], 1: [1, 3]})
        with self.assertRaises(rb.RebuildError):
            rb.parent_folds_from_split(split, 5)
        split["folds"][1]["fold"] = 2
        with self.assertRaises(rb.RebuildError):
            rb.parent_folds_from_split(split, 4)


# ------------------------------------------------------------------------------------------------- (vii) overlay

class Overlay(Fixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.out = cls.tmp / "ov-build"
        rb.build_shard(cls.out, 0, 1, items(H, "clean", folds=(0, 1, 2)), lambda h, rows: cls.ds, CODES, REFS,
                       budget=BUDGET, guard_fn=no_guard)
        cls.epoch = cls.tmp / "epoch-000.json"
        cls.epoch.write_bytes(b'{"epoch": 0, "fixture": true}\n')

    def test_header_pins_epoch_and_records_are_keyed_on_epoch0(self):
        p = self.tmp / "ov1.jsonl"
        res = rb.write_overlay(self.out, p, self.epoch, None)
        lines = [json.loads(l) for l in p.read_text().splitlines()]
        self.assertEqual(lines[0]["kind"], "header")
        self.assertEqual(lines[0]["overlay_of"]["sha256"], hashlib.sha256(self.epoch.read_bytes()).hexdigest())
        self.assertEqual((res["appended"], res["records"]), (3, 3))
        e0 = {r["view_id"]: r for r in epoch_rows()}
        for r in lines[1:]:
            k = r["key"]
            self.assertEqual(k["context_sha256"], e0[k["view_id"]]["context_sha256"])
            self.assertEqual(k["fold"], e0[k["view_id"]]["fold"])
            d = self.out / "shard-0" / r["replacement"]["view_id"]
            c = json.loads((d / "context.json").read_bytes())
            self.assertEqual(r["replacement"]["context_sha256"], c["context_sha256"])
            self.assertEqual(c["epoch0"]["view_id"], k["view_id"])
            self.assertEqual(r["row_selection_leak"], ["clean"])
            self.assertIs(r["rowsel"]["outer_split_changed"], False)

    def test_append_only(self):
        p = self.tmp / "ov2.jsonl"
        rb.write_overlay(self.out, p, self.epoch, None)
        first = p.read_bytes()
        self.assertEqual(rb.write_overlay(self.out, p, self.epoch, None)["appended"], 0)
        self.assertEqual(p.read_bytes(), first)
        out2 = self.tmp / "ov-build2"
        shutil.copytree(self.out, out2)
        rb.build_shard(out2, 0, 1, items(H, "clean"), lambda h, rows: self.ds, CODES, REFS, budget=BUDGET, guard_fn=no_guard)
        res = rb.write_overlay(out2, p, self.epoch, None)
        self.assertEqual((res["appended"], res["records"]), (2, 5))
        self.assertTrue(p.read_bytes().startswith(first))

    def test_refuses_another_epoch_and_epoch_drift(self):
        p = self.tmp / "ov3.jsonl"
        rb.write_overlay(self.out, p, self.epoch, None)
        other = self.tmp / "other-epoch.json"
        other.write_bytes(b"{}\n")
        before = p.read_bytes()
        with self.assertRaises(rb.RebuildError) as cm:
            rb.write_overlay(self.out, p, other, None)
        self.assertEqual(cm.exception.reason, "overlay_header_mismatch")
        with self.assertRaises(rb.RebuildError) as cm:
            rb.write_overlay(self.out, self.tmp / "ov3b.jsonl", self.epoch, "0" * 64)
        self.assertEqual(cm.exception.reason, "epoch_drift")
        self.assertEqual(p.read_bytes(), before)
        self.assertFalse((self.tmp / "ov3b.jsonl").exists())

    def test_refuses_a_conflicting_replacement(self):
        p = self.tmp / "ov4.jsonl"
        rb.write_overlay(self.out, p, self.epoch, None)
        lines = p.read_text().splitlines()
        rec = json.loads(lines[1])
        rec["replacement"]["context_sha256"] = "9" * 64
        p.write_text("\n".join([lines[0], json.dumps(rec)] + lines[2:]) + "\n")
        tampered = p.read_bytes()
        with self.assertRaises(rb.RebuildError) as cm:
            rb.write_overlay(self.out, p, self.epoch, None)
        self.assertEqual(cm.exception.reason, "overlay_conflict")
        self.assertEqual(p.read_bytes(), tampered)

    def test_refuses_unverified_contexts(self):
        out = self.tmp / "ov-bad"
        shutil.copytree(self.out, out)
        d = sorted((out / "shard-0").iterdir())[0]
        (d / "extra.bin").write_bytes(b"x")
        with self.assertRaises(rb.RebuildError) as cm:
            rb.write_overlay(out, self.tmp / "ov5.jsonl", self.epoch, None)
        self.assertEqual(cm.exception.reason, "overlay_unverified_contexts")
        self.assertFalse((self.tmp / "ov5.jsonl").exists())


# ------------------------------------------------------------------------------------------------- frozen, rights

RIGHTS = {"rights": "research-only", "rights_decision": "/fixture/owner-decisions.json#sha256=" + "7" * 64}


class FrozenScope(unittest.TestCase):
    def fixture(self):
        rows, leak, acq = [], {}, {}
        spec = [("aa", "incoming", ["kmeans"]), ("cc", "frozen", ["kmeans"]), ("gg", "frozen", ["kmeans"]),
                ("dd", "frozen", ["all_rows_coreset"]), ("ff", "frozen", ["clean"])]
        for h, cohort, label in spec:
            h = h * 8
            for f in range(2):
                r = {"view_id": f"{h}__r0__f{f}", "dataset_hash": h, "fold": f, "cohort": cohort, "context_sha256": "0" * 64}
                rows.append(r)
                leak[r["view_id"]] = {"view_id": r["view_id"], "dataset_hash": h, "fold": f, "cohort": cohort,
                                      "row_selection_leak": label}
            acq[h] = {"cohort": cohort, "class": label[0], "reduced": False, "selection_markers": []}
        return rows, leak, acq

    def test_frozen_scope_is_frozen_kmeans_in_the_list_only(self):
        rows, leak, acq = self.fixture()
        listed = {"cc" * 8, "dd" * 8, "ff" * 8, "aa" * 8}
        got = [(r["dataset_hash"][:2], r["fold"], lab) for r, lab, _ in rb.select_scope(rows, leak, acq, "frozen", listed)]
        self.assertEqual(got, [("cc", 0, "clean"), ("cc", 1, "clean")])
        self.assertEqual({r["dataset_hash"][:2] for r, _, _ in rb.select_scope(rows, leak, acq, "frozen")}, {"cc", "gg"})
        self.assertEqual({r["dataset_hash"][:2] for r, _, _ in rb.select_scope(rows, leak, acq)}, {"aa"})

    def test_rights_hosts(self):
        gate = [{"dataset_hash": "a" * 16, "host": "vincentarelbundock.github.io"},
                {"dataset_hash": "b" * 16, "host": "archive.ics.uci.edu"},
                {"dataset_hash": "c" * 16, "host": "data.cityofchicago.org"}]
        self.assertEqual(rb.check_rights_hosts({"a" * 16, "b" * 16}, gate),
                         {"vincentarelbundock.github.io": 1, "archive.ics.uci.edu": 1})
        for bad in ({"a" * 16, "c" * 16}, {"d" * 16}):
            with self.assertRaises(rb.RebuildError) as cm:
                rb.check_rights_hosts(bad, gate)
            self.assertEqual(cm.exception.reason, "rights_host_not_admitted")

    def test_rights_decision_is_pinned(self):
        tmp = Path(tempfile.mkdtemp(prefix="rights-"))
        try:
            good = tmp / "d.json"
            good.write_bytes(json.dumps({"decisions": {"RIGHTS-RESEARCH": "lift"}}).encode())
            digest = hashlib.sha256(good.read_bytes()).hexdigest()
            self.assertEqual(rb.rights_decision_ref(good, digest), f"{good}#sha256={digest}")
            with self.assertRaises(rb.RebuildError):
                rb.rights_decision_ref(good, "0" * 64)
            other = tmp / "o.json"
            other.write_bytes(json.dumps({"decisions": {"ROWSEL-SPLIT": "x"}}).encode())
            with self.assertRaises(rb.RebuildError):
                rb.rights_decision_ref(other, hashlib.sha256(other.read_bytes()).hexdigest())
        finally:
            shutil.rmtree(tmp)

    @unittest.skipUnless(rb.RIGHTS_DATASETS.exists() and rb.RIGHTS_GATE.exists() and rb.EPOCH0.exists(), "lane inputs not present")
    def test_real_frozen_scope(self):
        rows = [c for c in json.loads(rb.EPOCH0.read_bytes())["contexts"] if c.get("status") == "included"]
        scope, refs, counts = rb.load_scope(rows, scope="frozen-kmeans-research")
        self.assertEqual(counts, {"datasets": 42, "contexts": 209, "clean": 209, "acquisition_cap": 0})
        self.assertEqual({r["cohort"] for r, _, _ in scope}, {"frozen"})
        self.assertTrue(set(refs["rights_hosts"]) <= rb.RIGHTS_HOSTS)
        self.assertEqual(refs["context_extra"]["rights"], "research-only")
        self.assertTrue(refs["context_extra"]["rights_decision"].endswith("#sha256=" + rb.RIGHTS_DECISION_SHA256))
        listed = set(json.loads(rb.RIGHTS_DATASETS.read_bytes()))
        self.assertTrue({r["dataset_hash"] for r, _, _ in scope} <= listed)


class FrozenInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="frozen-in-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def split_fixture(self):
        body = json.dumps({"schema": rb.SPLIT_SCHEMA, "folds": []}).encode()
        digest = hashlib.sha256(body).hexdigest()
        (self.tmp / f"{digest}.json").write_bytes(body)
        sref = {"sha256": digest, "bytes": len(body), "key": f"splits/v1/{H}/x/{digest}.json", "version_id": "v1"}
        index = {digest: {"sha256": digest, "bytes": len(body), "key": sref["key"], "version_id": "v1", "dataset_hash": H}}
        return body, sref, index

    def test_frozen_split_read_from_fetch_dir_and_receipt_checked(self):
        body, sref, index = self.split_fixture()
        got, custody = rb.load_split("frozen", H, sref, self.tmp, index)
        self.assertEqual(got, body)
        self.assertEqual(custody["path"], str(self.tmp / f"{sref['sha256']}.json"))
        for field, value in (("version_id", "other"), ("dataset_hash", H2), ("key", "splits/v1/x"), ("bytes", 1)):
            bad = {k: dict(v, **{field: value}) for k, v in index.items()}
            with self.assertRaises(rb.RebuildError) as cm:
                rb.load_split("frozen", H, sref, self.tmp, bad)
            self.assertEqual(cm.exception.reason, "split_fetch_receipt", field)
        with self.assertRaises(rb.RebuildError):
            rb.load_split("frozen", H, sref, self.tmp, {})
        (self.tmp / f"{sref['sha256']}.json").write_bytes(body[:-1] + b" ")
        with self.assertRaises(rb.RebuildError) as cm:
            rb.load_split("frozen", H, sref, self.tmp, index)
        self.assertEqual(cm.exception.reason, "lineage_sha_mismatch")

    def test_meta_cap_markers(self):
        self.assertEqual(rb.meta_cap_markers({}), [])
        self.assertEqual(rb.meta_cap_markers({"acquisition": {"row_policy": "all"}, "downsampling": None}), [])
        self.assertEqual(rb.meta_cap_markers({"acquisition": {"row_policy": "stratified_cap_to_final_max"}}),
                         ["row_policy:stratified_cap_to_final_max"])
        self.assertEqual(rb.meta_cap_markers({"original_size": {"downsampled": True}, "downsampling": {"method": "hybrid_coreset"}}),
                         ["downsampling:hybrid_coreset", "original_size.downsampled"])

    def test_v1_work_check(self):
        train, test = b"1,2,3\n4,5,6\n", b"7,8,9\n"
        work = self.tmp / "work" / H / "sampler"
        work.mkdir(parents=True)
        inp = self.tmp / "work" / H / "input.csv"
        inp.write_bytes(b"1,2,3\n4,5,6\n7,8,9\n")
        man = {"selected_source_indices_sha256": "s" * 64}
        res = {"input_csv_sha256": hashlib.sha256(inp.read_bytes()).hexdigest(), "input_rows": 3,
               "sampler": {"seed": rowsel.v1_seed(H)}, "selected_source_indices_sha256": "s" * 64}
        (work / "result.json").write_text(json.dumps(res))
        got = rb.v1_work_check(H, {"train": train, "test": test}, man, 3, self.tmp)
        self.assertTrue(all(v for k, v in got.items() if isinstance(v, bool)), got)
        inp.write_bytes(b"1,2,3\n4,5,6\n7,8,0\n")
        self.assertFalse(rb.v1_work_check(H, {"train": train, "test": test}, man, 3, self.tmp)["input_csv_equals_t1_concatenation"])
        self.assertEqual(rb.v1_work_check(H2, {"train": train, "test": test}, man, 3, self.tmp), {"present": False})


class FrozenBuild(Fixture):
    def test_rights_fields_in_contexts_and_overlay(self):
        out = self.tmp / "frozen-build"
        refs = dict(REFS, context_extra=RIGHTS, owner_line="ROWSEL-TRAINONLY+RIGHTS-RESEARCH")
        its = items(H, "clean", folds=(0, 1), cohort="frozen") + items(H2, "acquisition_cap", folds=(2,), cohort="frozen")
        r = rb.build_shard(out, 0, 1, its, lambda h, rows: self.ds if h == H else dataset(fixture_matrix(seed=3), H2),
                           CODES, refs, budget=BUDGET, guard_fn=no_guard)
        self.assertEqual((r["done"], r["complete"]), (3, True))
        for d in sorted((out / "shard-0").iterdir()):
            c = json.loads((d / "context.json").read_bytes())
            self.assertEqual((c["rights"], c["rights_decision"]), (RIGHTS["rights"], RIGHTS["rights_decision"]))
            self.assertEqual((c["cohort"], c["owner_line"]), ("frozen", "ROWSEL-TRAINONLY+RIGHTS-RESEARCH"))
            self.assertNotIn("raw_route_superseded_if_matched", c["rowsel"])      # ROWSEL-SPLIT covers incoming only
            self.assertEqual(rb.verify_context_dir(d)["problems"], [])
        epoch = self.tmp / "frozen-epoch.json"
        epoch.write_bytes(b"{}\n")
        ov = self.tmp / "frozen-ov.jsonl"
        rb.write_overlay(out, ov, epoch, None)
        recs = [json.loads(l) for l in ov.read_text().splitlines()[1:]]
        self.assertEqual(len(recs), 3)
        self.assertEqual({(x["rights"], x["rights_decision"]) for x in recs}, {(RIGHTS["rights"], RIGHTS["rights_decision"])})
        d = sorted((out / "shard-0").iterdir())[0]
        c = json.loads((d / "context.json").read_bytes())
        c["rights"] = "production"
        (d / "context.json").write_text(json.dumps(c))
        self.assertIn("rights", rb.verify_context_dir(d)["problems"])

    def test_incoming_contexts_carry_no_rights(self):
        out = self.tmp / "incoming-norights"
        rb.build_shard(out, 0, 1, items(H, "clean", folds=(0,)), lambda h, rows: self.ds, CODES, REFS, budget=BUDGET,
                       guard_fn=no_guard)
        c = json.loads(next((out / "shard-0").iterdir()).joinpath("context.json").read_bytes())
        self.assertNotIn("rights", c)
        self.assertEqual(c["owner_line"], "ROWSEL-TRAINONLY")

    def test_cap_label_must_agree_with_meta(self):
        ds = dict(self.ds, lineage=dict(self.ds["lineage"], meta_cap_markers=["row_policy:stratified_cap_to_final_max"]))
        r = rb.build_shard(self.tmp / "cap-disagree", 0, 1, items(H, "clean", folds=(0, 1), cohort="frozen"),
                           lambda h, rows: ds, CODES, REFS, budget=BUDGET, guard_fn=no_guard)
        self.assertEqual([x["reason"] for x in r["failed"]], ["acquisition_cap_disagrees"] * 2)
        self.assertEqual(r["done"], 0)
        ok = dict(self.ds, lineage=dict(self.ds["lineage"], meta_cap_markers=[]))
        r2 = rb.build_shard(self.tmp / "cap-agree", 0, 1, items(H, "clean", folds=(0,), cohort="frozen"),
                            lambda h, rows: ok, CODES, REFS, budget=BUDGET, guard_fn=no_guard)
        self.assertEqual(r2["done"], 1)


if __name__ == "__main__":
    unittest.main()
