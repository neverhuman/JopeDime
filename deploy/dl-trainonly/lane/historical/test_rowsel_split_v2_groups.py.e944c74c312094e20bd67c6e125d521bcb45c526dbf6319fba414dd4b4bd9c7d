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


if __name__ == "__main__":
    unittest.main()
