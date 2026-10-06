"""ROWSEL-SPLIT v2 (dltrain/rowsel_raw_split_v2.py): the fold assignment is a function of the dataset hash, the positions
and the duplicate partition only. Every invariance check is paired with a control that MUST FAIL it.

  Rule        an independent restatement of the owner text gives the same folds; v1 is untouched (golden sha)
  Invariance  outer-test feature, text and target changes that preserve the duplicate partition leave every fold
              identical (every fold, numeric and text universes); the v1 feature-hash ordering FAILS the same check;
              the check's own validity test sees a partition change
  Structure   duplicates never straddle folds (position-hash control straddles); fold sizes stay within the largest
              duplicate group, as v1
  Pipeline    v2 folds go through the unchanged per-fold build and pass both counterfactual gates
"""
import hashlib
import shutil
import unittest

import numpy as np

from dltrain import nonnumeric, rowsel_raw, t1raw
from dltrain import rowsel_raw_split_v2 as v2
from tests import test_nonnumeric as tnn
from tests import test_v2_rowsel_raw as base

H, PREP, SPECS = base.H, base.PREP, base.SPECS
V1_GOLDEN_SPLIT_SHA256 = "55766811a417d5a951c59db53e22723ae67b4919cb337a7d4881b2afe388ee43"
FOLDS2 = v2.outer_split(H, PREP.positions, PREP.values, SPECS)


def reference_split(h, positions, values, specs, folds=5):
    """Independent restatement of ROWSEL-SPLIT: 'sha256 of dataset hash, partition, index; exact-duplicate rows kept
    together', greedy into the smallest fold (ties to the lower fold)."""
    groups = {}
    for p, row in zip(np.asarray(positions).tolist(), np.asarray(values)):
        key = []
        for s in specs[:-1]:
            v = row[s["index"]]
            if s["kind"] == "categorical_codes":
                key.append(None if v == t1raw.MISSING_CODE else ("c", s["inverse"][int(v)]))
            else:
                key.append("nan" if np.isnan(v) else float(v) + 0.0)
        groups.setdefault(tuple(key), []).append(p)

    def rank(p):
        return hashlib.sha256(f"{h}\0trainonly-rowsel-split-v1\0raw\0{p}".encode()).digest()
    order = sorted(groups.values(), key=lambda m: min(rank(p) for p in m))
    sizes, out = [0] * folds, {f: [] for f in range(folds)}
    for m in order:
        f = sizes.index(min(sizes))
        out[f] += m
        sizes[f] += len(m)
    return {f: sorted(v) for f, v in out.items()}


class Rule(unittest.TestCase):
    def test_v2_is_the_owner_text(self):
        self.assertEqual(FOLDS2, reference_split(H, PREP.positions, PREP.values, SPECS))
        self.assertEqual(sorted(p for v in FOLDS2.values() for p in v), PREP.positions.tolist())

    def test_v1_is_untouched_and_v2_has_its_own_identity(self):
        v1 = rowsel_raw.outer_split(H, PREP.positions, PREP.values, SPECS)
        self.assertEqual(rowsel_raw.split_sha256(H, v1), V1_GOLDEN_SPLIT_SHA256)
        self.assertEqual((rowsel_raw.SPLIT_SCHEMA, rowsel_raw.SPLIT_DOMAIN), ("rowsel-raw-split-v1", "trainonly-rowsel-split-v1"))
        self.assertNotEqual(FOLDS2, v1)
        self.assertNotEqual(v2.split_sha256(H, v1), rowsel_raw.split_sha256(H, v1))      # schema is part of the identity
        rep = v2.split_report(PREP, FOLDS2)
        self.assertEqual((rep["schema"], rep["sha256"]), ("rowsel-raw-split-v2", v2.split_sha256(H, FOLDS2)))

    def test_row_rank_layout(self):
        self.assertEqual(v2.row_rank(H, 17), hashlib.sha256(f"{H}\0trainonly-rowsel-split-v1\0raw\0" "17".encode()).digest())
        self.assertNotEqual(v2.row_rank(H, 17), v2.row_rank(H, 17, partition="other"))

    def test_input_order_does_not_matter(self):
        perm = np.random.default_rng(4).permutation(PREP.positions.size)
        self.assertEqual(v2.outer_split(H, PREP.positions[perm], PREP.values[perm], SPECS), FOLDS2)


class Invariance(unittest.TestCase):
    def test_every_outer_test_fold_change_leaves_the_split_identical(self):
        for f in range(5):
            with self.subTest(fold=f):
                r = v2.value_check(v2.outer_split, PREP, f)
                self.assertTrue(r["valid"], r)
                self.assertTrue(r["invariant"], r)
                self.assertEqual(sorted(r["cases"]), sorted(v2.CF_KINDS))
                self.assertTrue(all(c["cells_changed"] > 0 for c in r["cases"].values()))

    def test_v1_feature_hash_ordering_fails_the_same_check(self):
        r = v2.value_check(rowsel_raw.outer_split, PREP, 0)
        self.assertTrue(r["valid"], r)                                   # the same, valid perturbations
        self.assertFalse(r["invariant"])
        self.assertGreater(r["cases"]["features_and_text"]["rows_moved"], PREP.positions.size // 2)
        self.assertEqual(r["cases"]["targets"]["rows_moved"], 0)         # v1 never read targets either

    def test_numeric_only_universe(self):
        """No categorical column: the numeric remap alone must move v1 and leave v2 identical."""
        specs = [dict(SPECS[j], index=k) for k, j in enumerate((0, 1, 2))] + [dict(SPECS[4], index=3)]
        prep = rowsel_raw.prepare_universe(H, np.ascontiguousarray(base.VALUES[:, [0, 1, 2, 4]]), specs)
        self.assertTrue(v2.value_check(v2.outer_split, prep, 4)["invariant"])
        r1 = v2.value_check(rowsel_raw.outer_split, prep, 4)
        self.assertTrue(r1["valid"], r1)
        self.assertGreater(r1["cases"]["features_and_text"]["rows_moved"], 0)

    def test_coarse_universe_where_rows_differ_only_by_category(self):
        """Few distinct feature rows (a 4-level numeric x the categories): many groups differ only in the category, so a
        non-injective category remap would merge them; the perturbation must stay valid and v2 identical."""
        V = np.column_stack([np.floor(np.abs(base.VALUES[:, 0])) % 4, base.VALUES[:, 3], base.VALUES[:, 4]])
        specs = [dict(SPECS[0], index=0), dict(SPECS[3], index=1), dict(SPECS[4], index=2)]
        prep = rowsel_raw.prepare_universe(H, V, specs)
        groups = v2.duplicate_groups(prep.positions, prep.values, specs)
        self.assertLess(len(groups), 40)
        for f in range(5):
            r = v2.value_check(v2.outer_split, prep, f)
            self.assertTrue(r["valid"] and r["invariant"], r)

    def test_value_check_reports_a_partition_changing_perturbation_as_invalid(self):
        def merging(values, specs, rows, kind, salt=v2.CF_SALT):
            V = np.array(values, copy=True)
            V[rows, :-1] = V[rows[0], :-1]                              # CONTROL: collapses the fold into one group
            V[rows, -1] += 1.0
            return V, specs
        orig = v2.perturb_fold_values
        try:
            v2.perturb_fold_values = merging
            r = v2.value_check(v2.outer_split, PREP, 0)
        finally:
            v2.perturb_fold_values = orig
        self.assertFalse(r["valid"])
        self.assertFalse(r["invariant"])

    def test_targets_never_move_the_split(self):
        r = rowsel_raw.split_target_check(v2.outer_split, PREP)
        self.assertTrue(r["target_invariant"], r)
        self.assertFalse(rowsel_raw.split_target_check(base.stratified_split, PREP)["target_invariant"])   # control

    def test_perturbation_preserves_partition_and_validity_check_has_power(self):
        rows = PREP.rows_of(FOLDS2[1])
        V2, s2 = v2.perturb_fold_values(PREP.values, SPECS, rows, "features_and_text")
        self.assertEqual(v2.duplicate_partition(PREP.positions, V2, s2), v2.duplicate_partition(PREP.positions, PREP.values, SPECS))
        self.assertTrue(np.array_equal(np.delete(V2, rows, axis=0), np.delete(PREP.values, rows, axis=0), equal_nan=True))
        merged = np.array(PREP.values, copy=True)                        # CONTROL: two different rows made equal
        a, b = (int(i) for i in rows[:2])
        merged[b, :-1] = merged[a, :-1]
        self.assertNotEqual(v2.duplicate_partition(PREP.positions, merged, SPECS),
                            v2.duplicate_partition(PREP.positions, PREP.values, SPECS))

    def test_text_level_change_on_a_nonnumeric_universe(self):
        """Raw TEXT of outer-test rows changed injectively (t -> t + NUL 'x'): v2 identical, v1 moves."""
        d = tnn._tmpdir()
        try:
            cols = tnn.raw_columns()
            t1, meta, clean = tnn.staging(cols)
            A = t1[tnn.coreset(clean)]
            p = f"{d}/a.csv"
            with open(p, "wb") as f:
                f.write(tnn.csv_bytes(cols))
            m = nonnumeric.match_dataset(p, meta, A)
            self.assertIn(m["status"], ("matched", "matched_set_level", "matched_covered"))
            readings = {n: c["reading"] for n, c in m["text_columns"].items()}
            prep = rowsel_raw.prepare_universe(H, m["values"], m["universe_specs"])
            for split_fn, want_same in ((v2.outer_split, True), (rowsel_raw.outer_split, False)):
                folds = split_fn(H, prep.positions, prep.values, prep.specs)
                cols2 = {c: list(v) for c, v in cols.items()}
                for q in folds[2]:
                    for n, rd in readings.items():
                        if rd == nonnumeric.RANK and not nonnumeric.is_na(cols2[n][q]):
                            cols2[n][q] = cols2[n][q] + "\0x"
                p2 = f"{d}/b.csv"
                with open(p2, "wb") as f:
                    f.write(tnn.csv_bytes(cols2))
                V2, s2, _ = nonnumeric.read_universe(p2, meta, A.shape[1], readings)
                prep2 = rowsel_raw.prepare_universe(H, V2, s2)
                self.assertEqual(prep2.positions.tolist(), prep.positions.tolist())
                self.assertEqual(v2.duplicate_partition(prep2.positions, prep2.values, s2),
                                 v2.duplicate_partition(prep.positions, prep.values, prep.specs))
                self.assertNotEqual(prep2.row_bytes, prep.row_bytes)
                got = split_fn(H, prep2.positions, prep2.values, s2)
                self.assertEqual(got == folds, want_same, split_fn.__module__)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Structure(unittest.TestCase):
    def test_duplicates_never_straddle_folds(self):
        rep = v2.split_report(PREP, FOLDS2)
        self.assertGreaterEqual(rep["duplicate_rows"], base.DUP)
        self.assertEqual(rep["groups_straddling_folds"], 0)
        self.assertGreater(rowsel_raw.split_report(PREP, base.position_split(H, PREP.positions, PREP.values, SPECS))
                           ["groups_straddling_folds"], 0)                     # control

    def test_fold_sizes_stay_balanced_as_v1(self):
        for folds in (FOLDS2, rowsel_raw.outer_split(H, PREP.positions, PREP.values, SPECS)):
            sizes = [len(folds[f]) for f in range(5)]
            self.assertLessEqual(max(sizes) - min(sizes), max(len(g) for g in v2.duplicate_partition(PREP.positions, PREP.values, SPECS)))
        V = np.array(PREP.values, copy=True)                             # one giant duplicate group (46 % of rows)
        V[: int(0.46 * len(V)), :-1] = V[0, :-1]
        prep = rowsel_raw.prepare_universe(H, V, SPECS)
        for fn in (v2.outer_split, rowsel_raw.outer_split):
            sizes = [len(v) for v in fn(H, prep.positions, prep.values, SPECS).values()]
            big = max(len(g) for g in v2.duplicate_partition(prep.positions, prep.values, SPECS))
            self.assertGreaterEqual(max(sizes), big)                     # a group is never split
            self.assertLessEqual(max(sizes) - min(sizes), big)           # greedy fill bound, whatever the group order


class Pipeline(unittest.TestCase):
    def test_v2_folds_build_and_pass_both_gates(self):
        rep = v2.split_report(PREP, FOLDS2)
        ref = {k: rep[k] for k in ("schema", "sha256", "fold_sizes")}
        r = rowsel_raw.build_fold(PREP, FOLDS2, 3, base.UREF, ref)
        self.assertTrue(r["passed"], r["failures"])
        self.assertEqual(r["identity_body"]["split"]["schema"], "rowsel-raw-split-v2")
        self.assertFalse(set(r["train_order"]) & set(FOLDS2[3]))


if __name__ == "__main__":
    unittest.main()
