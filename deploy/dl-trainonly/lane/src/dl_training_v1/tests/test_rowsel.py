"""ROWSEL-TRAINONLY prototype tests: no outer-test row may affect which training rows a view holds.

(i)   invariance: perturbing, replacing or dropping parent-fold-f rows leaves the train side bit-identical (selected
      train positions and ids, every selection-fit fingerprint, the train rows' values);
(ii)  the test draw is a function of row ids only (changing test or train row values does not change it);
(iii) negative controls that MUST FAIL those checks: the v1 selector fitted on all rows, a test draw keyed on the
      target, and a draw keyed on v1's byte-bound source_row_id;
(iv)  budget and per-fold proportions;
(v)   determinism, plus a golden pin and a Rust-reference pin (the port reproduces research-sampler's selection).

A check that cannot fail is not evidence, so every invariance check is paired with a control that fails it.
"""
import dataclasses
import hashlib
import math
import struct
import unittest

import numpy as np

from dltrain import rowsel

FIXTURE_HASH = "f1x7ure0rowsel01"
N, F, BUDGET = 1500, 8, 1000
FOLD_SIZES = (280, 300, 310, 305, 305)

# Rust reference (research_2k/target/release/jope-research-sampler, sha256 RUST_BINARY_SHA256, built from main.rs
# sha256 RUST_MAIN_SHA256) run once on fixture_lines() as input.csv with max_rows=1000, target_bins=16,
# feature_clusters=16, projection_dims=8 and seed rowsel.derive_seed(FIXTURE_HASH). Recorded in the prototype receipt.
RUST_BINARY_SHA256 = "9a00f7bf2c0dad3c2dd006ae1dd7364de66d950b04118d8b4c959f81a28bee05"
RUST_MAIN_SHA256 = "d9a6dc9a20b9c1fd918980fab58884d40a9a065e738e0c430d17092166cd6619"
RUST_INPUT_CSV_SHA256 = "76bbc67456c2ccb833a19d42129f77ad4c1219db25f7d43c59d156475aad3a7b"
RUST_SELECTED_SOURCE_INDICES_SHA256 = "794d465e1e7b7f51a830a1b5ae249f761f4d6ce2c585d5fe6aa260136efc67ce"
RUST_OUTPUT_ROWS = 1000
GOLDEN_VIEW_IDENTITIES = "f7907ac060ef86ee,3a7c08b16991ce95,1f803aea6051497e,879c530a77cc457e,3063cac89c967683"


def _fmt(v: float) -> str:
    return "" if not math.isfinite(v) else "%.6g" % v


def fixture_lines(n=N, features=F, seed=20261005):
    """Deterministic CSV lines (legacy RandomState stream, stable across numpy versions): clustered, heavy-tailed,
    coarse (ties) and constant-ish columns, rows with 1, 2, 3 and 5 missing cells (all four missing bins), tied targets."""
    rs = np.random.RandomState(seed)
    group = rs.randint(0, 5, size=n)
    x = np.empty((n, features))
    x[:, 0] = np.array([-4.0, -1.0, 0.5, 3.0, 8.0])[group] + rs.normal(0, 0.6, n)
    x[:, 1] = np.exp(rs.normal(0, 1.5, n))
    x[:, 2] = rs.randint(0, 10, size=n).astype(float)
    x[:, 3] = rs.normal(100, 30, n)
    x[:, 4] = rs.uniform(-1, 1, n) * 1e4
    x[:, 5] = np.where(rs.random_sample(n) < 0.9, 0.0, rs.normal(0, 1, n))
    for j in range(6, features):
        x[:, j] = rs.normal(0, 1, n) + 0.3 * x[:, 0]
    y = np.round(2.0 * x[:, 0] + np.log1p(x[:, 1]) + 0.4 * x[:, 2] + rs.normal(0, 0.5, n), 2)
    pattern = rs.random_sample(n)
    miss = np.zeros((n, features), dtype=bool)
    miss[pattern < 0.10, 3] = True                                         # 1 missing  -> bin 1
    miss[(pattern >= 0.10) & (pattern < 0.16), 1] = True                   # 2 missing  -> bin 1 at F=8 (2*4 == 8)
    miss[(pattern >= 0.10) & (pattern < 0.16), 4] = True
    two_three = (pattern >= 0.16) & (pattern < 0.21)                       # 3 missing  -> bin 2
    miss[two_three, 1] = miss[two_three, 3] = miss[two_three, 6] = True
    five = (pattern >= 0.21) & (pattern < 0.24)                            # 5 missing  -> bin 3
    for j in (1, 3, 4, 6, 7):
        miss[five, j] = True
    x[miss] = np.nan
    return [(",".join([_fmt(v) for v in x[i]] + [_fmt(y[i])])).encode() for i in range(n)]


def fixture_universe():
    body = fixture_lines()
    n_train = 1200
    return rowsel.universe_from_t1(FIXTURE_HASH, b"\n".join(body[:n_train]) + b"\n",
                                   b"\n".join(body[n_train:]) + b"\n", F + 1)


def fixture_folds(universe=None, sizes=FOLD_SIZES):
    universe = universe or fixture_universe()
    order = np.random.RandomState(7).permutation(universe.positions)
    out, start = {}, 0
    for f, s in enumerate(sizes):
        out[f] = sorted(int(p) for p in order[start:start + s])
        start += s
    return out


_U = fixture_universe()
_FOLDS = fixture_folds(_U)


def target_keyed_view(universe, parent_folds, fold, budget=rowsel.DEFAULT_BUDGET, seed=None, quotas=None):
    """NEGATIVE CONTROL: the train side is the clean one, but the test draw takes the fold rows with the lowest target."""
    v = rowsel.select_view(universe, parent_folds, fold, budget, seed, quotas)
    fold_pos = sorted(int(p) for p in parent_folds[fold])
    yv = universe.y[universe.index_of(fold_pos)]
    chosen = [p for _, p in sorted(zip(yv.tolist(), fold_pos))[:v.quotas[1]]]
    return rowsel.make_view(universe, fold, budget, v.quotas, v.seed, v.train_fit, chosen)


def byte_bound_id_view(universe, parent_folds, fold, budget=rowsel.DEFAULT_BUDGET, seed=None, quotas=None):
    """NEGATIVE CONTROL (D9): the draw keyed on v1's source_row_id, which hashes the row bytes (target included)."""
    ids = tuple(rowsel.v1_source_row_id(universe.dataset_hash, "u", int(p), b)
                for p, b in zip(universe.positions, universe.row_bytes))
    u2 = rowsel.Universe(universe.dataset_hash, universe.positions, ids, universe.X, universe.y, universe.row_bytes)
    return rowsel.select_view(u2, parent_folds, fold, budget, seed, quotas)


class PortFaithfulness(unittest.TestCase):
    def test_rank_bins_matches_rust_unit_tests(self):
        bins = rowsel.rank_bins(np.arange(100, dtype=float), 10)
        self.assertEqual((int(bins.min()), int(bins.max())), (0, 9))
        self.assertEqual(rowsel.rank_bins(np.ones(20), 16).tolist(), [0] * 20)
        b = rowsel.rank_bins(np.array([3.0, 1.0, 2.0, 1.0, 5.0, 4.0]), 3)       # ranks 3,0,2,1,5,4 -> rank*3//6
        self.assertEqual(b.tolist(), [1, 0, 1, 0, 2, 2])

    def test_total_order_key_is_rust_total_cmp(self):
        v = np.array([0.0, -0.0, -1.0, 1.0, -np.inf, np.inf, 5e-324, -5e-324])
        order = np.argsort(rowsel._total_order_key(v), kind="stable")
        self.assertEqual(v[order].tolist(), [-np.inf, -1.0, -5e-324, -0.0, 0.0, 5e-324, 1.0, np.inf])
        self.assertTrue(math.copysign(1.0, v[order][3]) < 0)

    def test_allocate_quotas_matches_rust_unit_test(self):
        sizes = {(0, 0, 0): 100, (1, 0, 0): 10, (2, 0, 0): 1}
        q = rowsel.allocate_quotas(sizes, 50)
        self.assertEqual(sum(q.values()), 50)
        self.assertTrue(all(q[k] <= sizes[k] for k in sizes))
        self.assertEqual(q, {(0, 0, 0): 44, (1, 0, 0): 5, (2, 0, 0): 1})
        many = {(i, 0, 0): 3 for i in range(10)}                      # more strata than the total: no minimum of one
        self.assertEqual(sum(rowsel.allocate_quotas(many, 4).values()), 4)
        with self.assertRaises(rowsel.RowselError):
            rowsel.allocate_quotas({(0, 0, 0): 2}, 3)

    def test_priority_is_domain_separated_and_matches_main_rs_layout(self):
        self.assertNotEqual(rowsel.priority("seed", "a", 1, b"row"), rowsel.priority("seed", "b", 1, b"row"))
        want = hashlib.sha256(b"seed\0a\0" + struct.pack("<Q", 1) + b"\0row").digest()
        self.assertEqual(rowsel.priority("seed", "a", 1, b"row"), want)

    def test_column_sums_are_sequential_not_pairwise(self):
        a = np.array([[1.0]] + [[1e-16]] * 199)                    # sequential: every 1e-16 is lost against 1.0
        s = 0.0
        for v in a[:, 0]:
            s += v
        self.assertEqual(rowsel._seq_colsum(a)[0], s)
        self.assertNotEqual(float(np.sum(a[:, 0])), s)                # numpy's pairwise sum differs here

    def test_parse_mirrors_main_rs(self):
        X, y = rowsel.parse_csv_lines([b"1, ,3.5,1e2", b"-0,,inf,7"], 4)
        self.assertTrue(math.isnan(X[0, 1]) and math.isnan(X[1, 1]) and math.isinf(X[1, 2]))
        self.assertEqual(y.tolist(), [100.0, 7.0])
        with self.assertRaises(rowsel.RowselError):
            rowsel.parse_csv_lines([b"1_0,2"], 2)
        with self.assertRaises(rowsel.RowselError):
            rowsel.parse_csv_lines([b"1,nan"], 2)
        with self.assertRaises(rowsel.RowselError):
            rowsel.parse_csv_lines([b"1,2,3"], 2)
        self.assertEqual(rowsel.lines(b"a\r\n\r\nb\n\n"), [b"a", b"b"])

    def test_fixture_covers_every_missing_bin(self):
        self.assertEqual(sorted(set(rowsel.missing_bins(_U.X).tolist())), [0, 1, 2, 3])

    def test_port_reproduces_rust_reference_selection(self):
        body = b"\n".join(fixture_lines()) + b"\n"
        self.assertEqual(hashlib.sha256(body).hexdigest(), RUST_INPUT_CSV_SHA256)
        fit = rowsel.sampler_select(_U, _U.positions.tolist(), RUST_OUTPUT_ROWS, rowsel.derive_seed(FIXTURE_HASH))
        self.assertEqual(len(fit.output_order), RUST_OUTPUT_ROWS)
        self.assertEqual(rowsel.selected_source_indices_sha256(fit.output_order), RUST_SELECTED_SOURCE_INDICES_SHA256)


class TrainSideInvariance(unittest.TestCase):
    """(i) perturb / replace / drop parent-fold-f rows: the train side must be bit-identical, for every fold."""

    def test_train_side_invariant_for_every_fold_and_perturbation(self):
        for fold in sorted(_FOLDS):
            with self.subTest(fold=fold):
                r = rowsel.counterfactual_report(rowsel.select_view, _U, _FOLDS, fold, BUDGET)
                bad = {k: c["train_side_differs"] for k, c in r["cases"].items()
                       if k.startswith("fold:") and c["train_side_differs"]}
                self.assertEqual(bad, {})
                self.assertTrue(r["train_side_invariant"])
                self.assertEqual(sorted(k for k in r["cases"] if k.startswith("fold:")),
                                 sorted("fold:" + k for k in rowsel.FOLD_PERTURBATIONS))

    def test_train_rows_are_bit_identical_after_test_rows_change(self):
        base = rowsel.select_view(_U, _FOLDS, 2, BUDGET)
        for kind in rowsel.FOLD_PERTURBATIONS:
            u2, f2 = rowsel.perturb_rows(_U, _FOLDS, _FOLDS[2], kind)
            v2 = rowsel.select_view(u2, f2, 2, BUDGET, quotas=base.quotas)
            a, b = _U.index_of(base.train_positions), u2.index_of(v2.train_positions)
            self.assertEqual(base.train_positions, v2.train_positions, kind)
            self.assertEqual(_U.X[a].tobytes(), u2.X[b].tobytes(), kind)
            self.assertEqual(_U.y[a].tobytes(), u2.y[b].tobytes(), kind)
            self.assertEqual(base.train_fit.fingerprints, v2.train_fit.fingerprints, kind)

    def test_perturbations_really_change_the_fold_rows(self):
        fold_idx = _U.index_of(_FOLDS[1])
        for kind in ("features", "targets", "missingness"):
            u2, _ = rowsel.perturb_rows(_U, _FOLDS, _FOLDS[1], kind)
            changed = (u2.X[fold_idx].tobytes() != _U.X[fold_idx].tobytes()
                       or u2.y[fold_idx].tobytes() != _U.y[fold_idx].tobytes())
            self.assertTrue(changed, kind)
        u2, f2 = rowsel.perturb_rows(_U, _FOLDS, _FOLDS[1], "drop")
        self.assertEqual(u2.n, _U.n - (len(_FOLDS[1]) + 1) // 2)
        u3, f3 = rowsel.perturb_rows(_U, _FOLDS, _FOLDS[1], "replace")
        self.assertFalse(set(f3[1]) & set(_FOLDS[1]))

    def test_small_change_to_one_train_row_is_detected(self):
        """Power check: the train-side comparison sees a one-ULP change to a single training value (row identity) and
        a 1e-6 relative change (fitted state: the projection mean)."""
        base = rowsel.select_view(_U, _FOLDS, 0, BUDGET)
        i = int(_U.index_of([base.train_positions[0]])[0])
        j = int(np.flatnonzero(np.isfinite(_U.X[i]))[0])
        for change, expect in ((lambda v: np.nextafter(v, np.inf), ("train_values_sha256", "train_fit_identity")),
                               (lambda v: v * (1 + 1e-6) + 1e-6, ("train_fit_fingerprints.feature_means_sha256",))):
            X = _U.X.copy()
            X[i, j] = change(X[i, j])
            u2 = rowsel.Universe(_U.dataset_hash, _U.positions, _U.row_ids, X, _U.y, _U.row_bytes)
            diff = rowsel._differs(rowsel.train_side(base), rowsel.train_side(rowsel.select_view(u2, _FOLDS, 0, BUDGET)))
            for field in expect:
                self.assertIn(field, diff)


class TestDrawIdsOnly(unittest.TestCase):
    """(ii) the test draw depends on row ids only."""

    def test_draw_invariant_to_test_and_train_values(self):
        for fold in (0, 3):
            r = rowsel.counterfactual_report(rowsel.select_view, _U, _FOLDS, fold, BUDGET)
            self.assertTrue(r["test_draw_invariant"], r["cases"])
            for kind in ("fold:features", "fold:targets", "fold:missingness", "fold:outlier",
                         "train:features", "train:targets", "train:missingness"):
                self.assertEqual(r["cases"][kind]["test_draw_differs"], [], kind)

    def test_draw_is_the_sha256_rank_of_ids(self):
        ids = [_U.row_ids[i] for i in _U.index_of(_FOLDS[4])]
        ranked = sorted(ids, key=lambda rid: hashlib.sha256(
            FIXTURE_HASH.encode() + b"\0trainonly-rowsel-v1\0" + rid.encode()).digest())
        self.assertEqual(rowsel.draw_test_rows(FIXTURE_HASH, ids, 120), ranked[:120])
        self.assertEqual(rowsel.draw_test_rows(FIXTURE_HASH, list(reversed(ids)), 120), ranked[:120])
        v = rowsel.select_view(_U, _FOLDS, 4, BUDGET)
        self.assertEqual(sorted(v.test_row_ids), sorted(ranked[:v.quotas[1]]))

    def test_t1_ids_are_value_free(self):
        self.assertEqual(_U.row_ids[0], rowsel.positional_row_id(FIXTURE_HASH, "train", 0))
        self.assertEqual(_U.row_ids[1200], rowsel.positional_row_id(FIXTURE_HASH, "test", 0))
        self.assertNotEqual(_U.row_ids[0], rowsel.v1_source_row_id(FIXTURE_HASH, "train", 0, _U.row_bytes[0]))


class NegativeControls(unittest.TestCase):
    """(iii) leaky selectors that MUST FAIL the checks above."""

    def test_v1_all_rows_selector_fails_train_invariance(self):
        for fold in (0, 2):
            r = rowsel.counterfactual_report(rowsel.v1_view, _U, _FOLDS, fold, BUDGET)
            self.assertFalse(r["train_side_invariant"])
            for kind in rowsel.FOLD_PERTURBATIONS:
                self.assertNotEqual(r["cases"]["fold:" + kind]["train_side_differs"], [], (fold, kind))
            self.assertIn("train_fit_fingerprints.target_bins_sha256", r["cases"]["fold:targets"]["train_side_differs"])
            self.assertIn("train_fit_fingerprints.feature_means_sha256",
                          r["cases"]["fold:features"]["train_side_differs"])

    def test_v1_selector_changes_which_train_rows_exist(self):
        changed = 0
        for kind in rowsel.FOLD_PERTURBATIONS:
            base = rowsel.v1_view(_U, _FOLDS, 1, BUDGET)
            u2, f2 = rowsel.perturb_rows(_U, _FOLDS, _FOLDS[1], kind)
            changed += rowsel.v1_view(u2, f2, 1, BUDGET).train_positions != base.train_positions
        self.assertGreaterEqual(changed, 4)

    def test_target_keyed_draw_fails_draw_invariance(self):
        r = rowsel.counterfactual_report(target_keyed_view, _U, _FOLDS, 0, BUDGET)
        self.assertTrue(r["train_side_invariant"])                # its train side is the clean one
        self.assertFalse(r["test_draw_invariant"])
        self.assertIn("test_row_ids", r["cases"]["fold:targets"]["test_draw_differs"])

    def test_byte_bound_v1_source_row_id_fails_draw_invariance(self):
        r = rowsel.counterfactual_report(byte_bound_id_view, _U, _FOLDS, 0, BUDGET)
        self.assertFalse(r["test_draw_invariant"])
        self.assertIn("test_row_ids", r["cases"]["fold:targets"]["test_draw_differs"])


class BudgetAndProportions(unittest.TestCase):
    """(iv)"""

    def test_v1_parameters(self):
        self.assertEqual((rowsel.DEFAULT_BUDGET, rowsel.TARGET_BINS, rowsel.FEATURE_CLUSTERS, rowsel.PROJECTION_DIMS,
                          rowsel.KMEANS_ITERATIONS), (2000, 16, 16, 8, 12))

    def test_fold_quotas_sum_to_budget_in_proportion(self):
        for sizes, budget in ((FOLD_SIZES, BUDGET), ((1701, 1701, 1702, 1701, 1701), 2000), ((5, 9, 2), 7),
                              ((3001, 2999, 3000, 3000, 3001), 2000)):
            q = rowsel.fold_quotas(dict(enumerate(sizes)), budget)
            total = sum(sizes)
            self.assertEqual(sum(t for _, t in q.values()), budget)
            for f, (n_train, n_test) in q.items():
                self.assertEqual(n_train + n_test, budget)
                self.assertLess(abs(n_test - budget * sizes[f] / total), 1.0)
                self.assertLessEqual(n_test, sizes[f])
                self.assertLessEqual(n_train, total - sizes[f])

    def test_views_respect_quotas_and_partition(self):
        q = rowsel.fold_quotas({f: len(v) for f, v in _FOLDS.items()}, BUDGET)
        for fold, members in _FOLDS.items():
            v = rowsel.select_view(_U, _FOLDS, fold, BUDGET)
            held = set(members)
            self.assertEqual(v.quotas, q[fold])
            self.assertEqual((len(v.train_positions), len(v.test_positions)), q[fold])
            self.assertEqual(len(v.train_positions) + len(v.test_positions), BUDGET)
            self.assertTrue(set(v.test_positions) <= held)
            self.assertFalse(set(v.train_positions) & held)
            self.assertFalse(v.train_fit.passthrough)
            self.assertEqual(v.train_fit.fingerprints["fit_rows"], _U.n - len(members))

    def test_universe_within_budget_keeps_every_row(self):
        v = rowsel.select_view(_U, _FOLDS, 3, budget=_U.n)
        self.assertTrue(v.train_fit.passthrough)
        self.assertEqual(set(v.test_positions), set(_FOLDS[3]))
        self.assertEqual(set(v.train_positions), set(_U.positions.tolist()) - set(_FOLDS[3]))

    def test_explicit_quotas_and_short_fold(self):
        v = rowsel.select_view(_U, _FOLDS, 0, BUDGET, quotas=(600, 10_000))
        self.assertEqual(len(v.train_positions), 600)
        self.assertEqual(set(v.test_positions), set(_FOLDS[0]))
        with self.assertRaises(rowsel.RowselError):
            rowsel.select_view(_U, _FOLDS, 0, BUDGET, quotas=(-1, 5))

    def test_parent_folds_must_partition(self):
        bad = dict(_FOLDS)
        bad[0] = bad[0][1:]
        with self.assertRaises(rowsel.RowselError):
            rowsel.select_view(_U, bad, 1, BUDGET)
        dup = dict(_FOLDS)
        dup[1] = dup[1] + dup[0][:1]
        with self.assertRaises(rowsel.RowselError):
            rowsel.select_view(_U, dup, 1, BUDGET)


class Determinism(unittest.TestCase):
    """(v)"""

    def test_repeat_is_identical(self):
        a = rowsel.select_view(_U, _FOLDS, 1, BUDGET)
        b = rowsel.select_view(fixture_universe(), fixture_folds(), 1, BUDGET)
        self.assertEqual(a, b)

    def test_row_order_of_the_universe_does_not_matter(self):
        perm = np.random.RandomState(3).permutation(_U.n)
        u2 = rowsel.Universe(_U.dataset_hash, _U.positions[perm], tuple(_U.row_ids[i] for i in perm), _U.X[perm],
                             _U.y[perm], tuple(_U.row_bytes[i] for i in perm))
        self.assertEqual(rowsel.select_view(u2, _FOLDS, 4, BUDGET).identity,
                         rowsel.select_view(_U, _FOLDS, 4, BUDGET).identity)

    def test_seed_matters(self):
        a = rowsel.select_view(_U, _FOLDS, 1, BUDGET)
        b = rowsel.select_view(_U, _FOLDS, 1, BUDGET, seed=rowsel.v1_seed(FIXTURE_HASH))
        self.assertNotEqual(a.train_positions, b.train_positions)
        self.assertEqual(a.test_positions, b.test_positions)         # the draw does not use the sampler seed

    def test_golden_view_identities(self):
        got = ",".join(rowsel.select_view(_U, _FOLDS, f, BUDGET).identity[:16] for f in sorted(_FOLDS))
        self.assertEqual(got, GOLDEN_VIEW_IDENTITIES)


if __name__ == "__main__":
    unittest.main()
