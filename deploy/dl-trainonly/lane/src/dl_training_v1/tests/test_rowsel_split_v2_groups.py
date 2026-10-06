"""dltrain/rowsel_split_v2_groups.py (GROUP-UNION-B). Every invariance check is paired with a control that MUST FAIL it.

  Equivalence  group_values=None reproduces rowsel_raw_split_v2 exactly (partitions 'raw' and 't1')
  Groups       annotated groups never straddle folds; split v2 without groups straddles them (must fail); unions chain
               through feature duplicates and groups; rows without a group value are never joined by group
  TargetFree   y changes never move the split; duplicate- and group-preserving outer-test feature/group changes never
               move it; the v1 feature-hash ordering and a group-value-ranked split move (must fail)
  Identity     the split sha binds dataset, partition (not hard-coded), group-column spec and folds; cut_id prefix
"""
import hashlib
import json
import unittest

import numpy as np

from dltrain import rowsel_raw
from dltrain import rowsel_raw_split_v2 as v2
from dltrain import rowsel_split_v2_groups as g
from tests import test_v2_rowsel_raw as base

H, SPECS = base.H, base.SPECS
REGIONS, GCOL = 23, 2


def _with_regions():
    """The builder fixture with feature n2 replaced by a coarse region code (23 levels, some missing): as in the data,
    the annotated group IS a feature column, and its value is that column's exact bytes."""
    V = np.array(base.VALUES, copy=True)
    rng = np.random.default_rng(17)
    V[:, GCOL] = rng.integers(0, REGIONS, size=len(V)).astype(float)
    V[rng.random(len(V)) < 0.03, GCOL] = np.nan
    return rowsel_raw.prepare_universe(H, V, SPECS)


PREP = _with_regions()
POS = PREP.positions.tolist()
FB = rowsel_raw.feature_bytes(PREP.values, SPECS)


def region_of(values):
    """Group value = the exact float64 bytes of feature column GCOL (missing -> no group)."""
    v = np.asarray(values, dtype=np.float64)[:, GCOL]
    return [None if np.isnan(x) else np.float64(x).tobytes() for x in v]


GV = region_of(PREP.values)
F_G = g.split(H, POS, FB, GV, "raw")


def values_split(group_fn, partition="raw"):
    """split_fn(h, positions, values, specs) for rowsel_raw.split_target_check / value checks."""
    def fn(h, positions, values, specs, folds=5):
        return g.split(h, positions, rowsel_raw.feature_bytes(values, specs), group_fn(values) if group_fn else None, partition)
    return fn


def group_ranked_split(h, positions, values, specs, folds=5):
    """CONTROL: groups unioned, but components ranked by their group value (content), not by position hashes."""
    comps = g.components(positions, rowsel_raw.feature_bytes(values, specs), region_of(values))
    gv = dict(zip(np.asarray(positions).tolist(), region_of(values)))
    order = sorted(comps, key=lambda c: (hashlib.sha256(repr(sorted(str(gv[p]) for p in c)).encode()).digest(), min(c)))
    sizes, out = [0] * folds, {f: [] for f in range(folds)}
    for c in order:
        f = sizes.index(min(sizes)); out[f] += c; sizes[f] += len(c)
    return {f: sorted(v) for f, v in out.items()}


def perturb_fold(values, specs, fold_positions, seed=3):
    """Outer-test change that keeps duplicates and groups: numeric features remapped injectively by rank to fresh
    values (column n1, the group source, included: the region of a remapped row moves to fresh regions consistently),
    the category re-keyed, the target noised."""
    rows = PREP.rows_of(sorted(fold_positions))
    V, sp = v2.perturb_fold_values(values, specs, rows, "features_text_and_targets")
    return V, sp


class Equivalence(unittest.TestCase):
    def test_none_reproduces_v2_on_raw_and_t1(self):
        self.assertEqual(g.split(H, POS, FB, None, "raw"), v2.outer_split(H, PREP.positions, PREP.values, SPECS))
        self.assertEqual(g.split(H, POS, FB, None, "t1"), v2.outer_split(H, PREP.positions, PREP.values, SPECS, partition="t1"))
        self.assertNotEqual(g.split(H, POS, FB, None, "raw"), g.split(H, POS, FB, None, "t1"))

    def test_components_without_groups_are_v2_duplicate_groups(self):
        self.assertEqual(g.component_partition(POS, FB), v2.duplicate_partition(PREP.positions, PREP.values, SPECS))


class Groups(unittest.TestCase):
    def test_groups_never_straddle(self):
        self.assertEqual(g.straddling_groups(F_G, POS, GV), 0)
        fold_of = {p: f for f, v in F_G.items() for p in v}
        for k in set(x for x in GV if x is not None):
            self.assertEqual(len({fold_of[p] for p, x in zip(POS, GV) if x == k}), 1)
        sizes = [len(F_G[f]) for f in range(5)]
        self.assertLessEqual(max(sizes) - min(sizes), max(len(c) for c in g.components(POS, FB, GV)))

    def test_split_without_groups_straddles(self):
        """MUST FAIL: v2 ignores the annotation, so rows of one region land in several folds."""
        self.assertGreater(g.straddling_groups(v2.outer_split(H, PREP.positions, PREP.values, SPECS), POS, GV), 0)

    def test_unions_chain_and_none_is_no_group(self):
        pos = [10, 11, 12, 13, 14]
        fb = [b"a", b"a", b"b", b"c", b"d"]
        gv = [None, b"G", b"G", None, None]
        self.assertEqual(sorted(g.components(pos, fb, gv)), [[10, 11, 12], [13], [14]])
        self.assertEqual(sorted(g.components(pos, fb, [None] * 5)), [[10, 11], [12], [13], [14]])
        self.assertEqual(sorted(g.components(pos, [b"x", b"y", b"z", b"w", b"v"], [None, None, None, None, None])),
                         [[10], [11], [12], [13], [14]])

    def test_input_order_does_not_matter(self):
        perm = np.random.default_rng(2).permutation(len(POS))
        self.assertEqual(g.split(H, [POS[i] for i in perm], [FB[i] for i in perm], [GV[i] for i in perm], "raw"), F_G)


class TargetFree(unittest.TestCase):
    def test_targets_never_move_the_split(self):
        r = rowsel_raw.split_target_check(values_split(region_of), PREP)
        self.assertTrue(r["target_invariant"], r)
        self.assertFalse(rowsel_raw.split_target_check(base.stratified_split, PREP)["target_invariant"])      # control

    def test_duplicate_and_group_preserving_outer_test_changes_never_move_the_split(self):
        fn = values_split(region_of)
        for f in range(5):
            V2, s2 = perturb_fold(PREP.values, SPECS, F_G[f])
            fb2 = rowsel_raw.feature_bytes(V2, s2)
            self.assertEqual(g.component_partition(POS, fb2, region_of(V2)), g.component_partition(POS, FB, GV))   # valid change
            self.assertEqual(fn(H, PREP.positions, V2, s2), F_G, f)

    def test_content_ranked_splits_move_under_the_same_change(self):
        """MUST FAIL: v1's feature-hash ordering and a group-value-ranked ordering both move."""
        for ctl in (rowsel_raw.outer_split, group_ranked_split):
            base_f = ctl(H, PREP.positions, PREP.values, SPECS)
            f = max(base_f, key=lambda k: len(base_f[k]))
            V2, s2 = perturb_fold(PREP.values, SPECS, base_f[f])
            self.assertNotEqual(ctl(H, PREP.positions, V2, s2), base_f, ctl.__name__)

    def test_component_partition_includes_groups(self):
        self.assertNotEqual(g.component_partition(POS, FB, GV), g.component_partition(POS, FB, None))
        gv2 = list(GV)
        a = next(k for k, x in enumerate(GV) if x is not None)
        b = next(k for k, x in enumerate(GV) if x is not None and x != GV[a])
        gv2[b] = GV[a]                                                    # a group-only change: two groups now share a row
        self.assertNotEqual(g.component_partition(POS, FB, gv2), g.component_partition(POS, FB, GV))

    def test_component_partition_sees_a_merge(self):
        fb2 = list(FB)
        i, j = 0, next(k for k in range(1, len(FB)) if GV[k] != GV[0] and FB[k] != FB[0])
        fb2[j] = fb2[i]
        self.assertNotEqual(g.component_partition(POS, fb2, GV), g.component_partition(POS, FB, GV))


class Identity(unittest.TestCase):
    def test_split_sha_binds_dataset_partition_group_spec_and_folds(self):
        spec = g.group_spec(GCOL, "n2")
        s = g.split_sha256(H, "raw", spec, F_G)
        body = {"schema": "rowsel-split-v2-groups", "dataset_hash": H, "partition": "raw", "group_column": {"index": GCOL, "name": "n2"},
                "folds": [F_G[f] for f in range(5)]}
        self.assertEqual(s, hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        self.assertNotEqual(s, g.split_sha256(H, "t1", spec, F_G))
        self.assertNotEqual(s, g.split_sha256(H, "raw", g.group_spec(1, "n2"), F_G))
        self.assertNotEqual(s, g.split_sha256(H, "raw", g.group_spec(GCOL, "n1"), F_G))
        self.assertNotEqual(s, g.split_sha256(H, "raw", None, F_G))
        self.assertNotEqual(s, g.split_sha256("f" * 16, "raw", spec, F_G))
        self.assertNotEqual(g.split_sha256(H, "raw", None, F_G), v2.split_sha256(H, F_G))
        self.assertEqual(g.cut_id(s), "trainonly-split-v2g:" + s)
        self.assertIsNone(g.group_spec(None, None))



def f8(values):
    """The builders' group encoding: canonical little-endian float64 bytes (NaN -> None here)."""
    return [None if np.isnan(v) else np.asarray(v + 0.0, dtype="<f8").tobytes() for v in np.asarray(values, dtype=np.float64)]


class TargetTie(unittest.TestCase):
    """g.target_tied_group: dataset-level refusal of a group column that copies, determines or bins the target
    (finding 8, owner pre-review F4)."""
    Y = np.asarray(PREP.values[:, -1], dtype=np.float64)

    def test_a_non_last_column_copying_the_target_is_refused(self):
        V = np.array(PREP.values, copy=True)
        V[:, 1] = V[:, -1]                                               # feature n1 := y (not the last column)
        self.assertEqual(g.target_tied_group(f8(V[:, 1]), V[:, -1]), "group_bytes_equal_target_bytes")
        y = np.array(V[:, -1], copy=True); y[:7] = -0.0                 # a -0.0 target matches the canonical +0.0 group bytes
        self.assertEqual(g.target_tied_group(f8(y), y), "group_bytes_equal_target_bytes")

    def test_an_affine_copy_is_refused(self):
        y = np.random.default_rng(5).normal(size=400)                     # distinct targets: every group a singleton
        self.assertEqual(g.target_tied_group(f8(3.0 * y - 2.0), y), "group_affine_copy_of_target")
        self.assertEqual(g.target_tied_group(f8(-0.25 * y + 7.0), y), "group_affine_copy_of_target")
        gv = f8(3.0 * y - 2.0); gv[::10] = [None] * len(gv[::10])        # missing cells do not hide the copy
        self.assertEqual(g.target_tied_group(gv, y), "group_affine_copy_of_target")
        self.assertIsNone(g.target_tied_group(f8(3.0 * y - 2.0 + np.random.default_rng(6).normal(0, 1e-3, y.size)), y))

    def test_a_column_determining_the_target_is_refused(self):
        rng = np.random.default_rng(7)
        region = rng.integers(0, 23, 600)
        effect = rng.normal(size=23)                                      # a non-linear map: no affine copy
        y = effect[region]
        gv = f8(region.astype(float))
        y_none = y.copy(); gv_none = list(gv)
        for i in range(0, 600, 50):                                       # rows without a group value: varying targets
            gv_none[i] = None; y_none[i] = rng.normal()
        self.assertEqual(g.target_tied_group(gv, y), "target_function_of_group")
        self.assertEqual(g.target_tied_group(gv_none, y_none), "target_function_of_group")
        self.assertEqual(g.target_tied_group([("r", int(r)) for r in region], y), "target_function_of_group")   # any hashable

    def test_a_legitimate_group_column_is_admitted(self):
        gv = region_of(PREP.values)
        self.assertIsNone(g.target_tied_group(gv, self.Y))
        y = np.array(self.Y, copy=True)
        k = next(i for i, x in enumerate(gv) if x is not None)
        y[k] = np.frombuffer(gv[k], dtype="<f8")[0]                      # one row whose group bytes equal its target
        self.assertIsNone(g.target_tied_group(gv, y))

    def test_an_all_singleton_column_is_admitted(self):
        y = np.round(np.random.default_rng(9).normal(size=300), 1)        # repeated targets
        ids = np.random.default_rng(10).permutation(300).astype(float)    # unique group per row, unrelated to y
        self.assertIsNone(g.target_tied_group(f8(ids), y))
        self.assertIsNone(g.target_tied_group([None] * 300, y))
        self.assertIsNone(g.target_tied_group([], []))

    def test_a_binned_target_is_refused(self):
        """F4: floor(y/k) and quantile bins: y varies inside each group, yet the groups are a step function of y."""
        y = np.random.default_rng(11).normal(size=600)
        self.assertEqual(g.target_tied_group([int(v) for v in np.floor(y / 0.5)], y), "group_is_binned_target")
        self.assertEqual(g.target_tied_group(f8(np.floor(y / 0.5)), y), "group_is_binned_target")
        q = np.searchsorted(np.quantile(y, np.linspace(0, 1, 11)[1:-1]), y, side="right")       # 10 quantile bins
        self.assertEqual(len(set(q.tolist())), 10)
        self.assertEqual(g.target_tied_group([f"bin{b}" for b in q], y), "group_is_binned_target")
        two = (y > np.median(y)).astype(int)                                # exactly 2 bins
        self.assertEqual(g.target_tied_group(two.tolist(), y), "group_is_binned_target")
        gv = [int(v) for v in np.floor(y / 0.5)]; gv[::7] = [None] * len(gv[::7])          # rows without a group
        self.assertEqual(g.target_tied_group(gv, y), "group_is_binned_target")

    def test_disjoint_ranges_without_a_function_of_y_are_admitted(self):
        """A y value shared by two groups: not a function of y, so disjoint multi-row ranges alone do not refuse."""
        self.assertIsNone(g.target_tied_group(["A", "A", "B", "B", "C"], [1.0, 2.0, 5.0, 6.0, 1.0]))
        self.assertEqual(g.target_tied_group(["A", "A", "B", "B", "C"], [1.0, 2.0, 5.0, 6.0, 9.0]), "group_is_binned_target")

    def test_near_total_determination(self):
        """F4: >= 99% of all rows in single-target multi-row groups is refused; 98% is admitted."""
        rng = np.random.default_rng(12)

        def case(constant_groups, varying_groups):
            gv, y = [], []
            for k in range(constant_groups):
                gv += [f"c{k}"] * 10; y += [float(k) + 0.5] * 10
            for k in range(varying_groups):
                gv += [f"v{k}"] * 10; y += rng.uniform(-5, constant_groups + 5, 10).tolist()
            return gv, np.asarray(y)
        self.assertEqual(g.target_tied_group(*case(99, 1)), "target_nearly_determined_by_group")    # 990 / 1000
        self.assertIsNone(g.target_tied_group(*case(98, 2)))                                        # 980 / 1000
        gv, y = case(99, 1)
        self.assertIsNone(g.target_tied_group(gv + [None] * 10, np.concatenate([y, np.zeros(10)])))   # 990 / 1010 < 99%

    def test_a_legitimate_group_overlapping_in_y_is_admitted(self):
        gv = region_of(PREP.values)
        rng_by = {}
        for x, t in zip(gv, self.Y):
            if x is not None:
                lo, hi = rng_by.get(x, (t, t)); rng_by[x] = (min(lo, t), max(hi, t))
        r = sorted(rng_by.values())
        self.assertTrue(any(b[0] <= a[1] for a, b in zip(r, r[1:])))         # the groups overlap in y
        self.assertIsNone(g.target_tied_group(gv, self.Y))

    def test_the_split_never_takes_the_target(self):
        import inspect
        for fn in (g.split, g.components):
            self.assertFalse({"y", "target", "targets"} & set(inspect.signature(fn).parameters))
        params = inspect.signature(g.split).parameters                     # F5: no default selects a universe
        self.assertIs(params["partition"].default, inspect.Parameter.empty)
        self.assertIs(params["group_values"].default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            g.split(H, POS, FB, GV)
        self.assertEqual(g.split(H, POS, FB, GV, "raw"), F_G)
        with self.assertRaises(ValueError):
            g.target_tied_group([b"a"], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
