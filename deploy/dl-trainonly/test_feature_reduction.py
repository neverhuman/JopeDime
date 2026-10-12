"""feature_reduction: train-only selection for fixed-width consumers (Jepson ruling, owner 2026-10-07 07:17 MT).

LeakGate is the ruling's gate: shuffling, replacing or zeroing the outer-test targets, and replacing or permuting the
outer-test rows, leave the selection byte-identical, while a deliberately leaky selector that fits on every row is caught by
the same perturbation (must-fail control)."""
import json
import struct
import unittest

import numpy as np

import feature_reduction as FR
from dltrain import jtt1


def pack(train, test, labels):
    """JTT1 bytes from float64 blocks (features then target as the last column) and u8 inner-fold labels."""
    n_train, n_test, p = train.shape[0], test.shape[0], train.shape[1] - 1
    body = np.ascontiguousarray(np.vstack([train, test]), dtype="<f8").tobytes()
    return jtt1.MAGIC + struct.pack("<IIIB", n_train, n_test, p, 2) + np.asarray(labels, dtype=np.uint8).tobytes() + body


def context(seed=0, n_train=120, n_test=60, p=40):
    """Targets depend on columns 3, 11, 19, 27 and 35 (those below p); everything else is noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_test, p))
    y = 0.1 * rng.normal(size=n_train + n_test)
    for c, w in ((3, 2.0), (11, -1.5), (19, 1.2), (27, 1.0), (35, -0.8)):
        if c < p:
            y = y + w * X[:, c]
    block = np.column_stack([X, y])
    labels = np.arange(n_train) % 2
    return block[:n_train], block[n_train:], labels


def selection_bytes(raw, k=5, **kw):
    idx, rec = FR.select_for_jtt1(raw, k, **kw)
    return np.asarray(idx, dtype="<i8").tobytes() + json.dumps(rec, sort_keys=True).encode()


class LeakGate(unittest.TestCase):
    def setUp(self):
        self.train, self.test, self.labels = context()
        self.base = selection_bytes(pack(self.train, self.test, self.labels))

    def variants(self):
        rng = np.random.default_rng(99)
        t = self.test.copy(); t[:, -1] = rng.permutation(t[:, -1]); yield "shuffled outer-test targets", t
        t = self.test.copy(); t[:, -1] = 1e6 * t[:, 7]; yield "outer-test targets replaced by a feature copy", t
        t = self.test.copy(); t[:, -1] = 0.0; yield "zeroed outer-test targets", t
        t = self.test.copy(); t[:, :-1] = rng.normal(size=t[:, :-1].shape); yield "replaced outer-test features", t
        yield "permuted outer-test rows", self.test[rng.permutation(len(self.test))]
        yield "fewer outer-test rows", self.test[:7]

    def test_outer_test_perturbations_leave_the_selection_byte_identical(self):
        for name, test in self.variants():
            with self.subTest(name):
                self.assertEqual(selection_bytes(pack(self.train, test, self.labels)), self.base)

    def test_every_seed_holds_the_gate(self):
        for seed in (1, 2, 3):
            with self.subTest(seed=seed):
                tr, te, lab = context(seed=seed)
                base = selection_bytes(pack(tr, te, lab), seed=seed)
                te2 = te.copy(); te2[:, -1] = np.random.default_rng(seed).permutation(te2[:, -1]) * 7
                self.assertEqual(selection_bytes(pack(tr, te2, lab), seed=seed), base)

    def test_a_leaky_selector_is_caught_by_the_same_perturbation(self):
        """Must-fail control: fitting on train AND test rows lets an outer-test target perturbation change the kept set."""
        def leaky(raw, k=5):
            h = jtt1.parse_header(raw)
            rows = np.frombuffer(memoryview(raw)[jtt1.HEADER_BYTES + h["n_train"]:], dtype="<f8").reshape(-1, h["p"] + 1)
            inner = np.arange(rows.shape[0]) % 2
            idx, _ = FR.select(rows[:, :-1], rows[:, -1], inner, k)
            return list(idx)
        tr, te, lab = context(n_train=40, n_test=400)
        te2 = te.copy(); te2[:, -1] = 50 * te2[:, 7] + 50 * te2[:, 8] + 50 * te2[:, 9] + 50 * te2[:, 10] + 50 * te2[:, 12]
        self.assertNotEqual(leaky(pack(tr, te, lab)), leaky(pack(tr, te2, lab)))
        self.assertEqual(selection_bytes(pack(tr, te, lab)), selection_bytes(pack(tr, te2, lab)))


class Behaviour(unittest.TestCase):
    def test_two_runs_are_byte_identical(self):
        raw = pack(*context(seed=4))
        self.assertEqual(selection_bytes(raw), selection_bytes(raw))

    def test_informative_columns_are_kept(self):
        idx, _ = FR.select_for_jtt1(pack(*context(n_train=240)), 5)
        self.assertLessEqual({3, 11, 19}, set(int(i) for i in idx))

    def test_p_at_or_below_k_is_a_no_op(self):
        raw = pack(*context(p=12))
        idx, rec = FR.select_for_jtt1(raw, 12)
        self.assertEqual(list(idx), list(range(12)))
        self.assertEqual(rec["method"], "identity (p <= k)")
        self.assertIsNone(rec["model"])
        out, rec2 = FR.teacher_view(raw, 12)
        self.assertIs(out, raw)

    def test_the_record_carries_the_ruling_fields(self):
        idx, rec = FR.select_for_jtt1(pack(*context()), 5)
        for key in ("method", "model", "k", "seed", "selected_feature_list_sha256", "n_features_original"):
            self.assertIn(key, rec)
        self.assertEqual(rec["schema"], FR.SCHEMA)
        self.assertEqual((rec["k"], rec["seed"], rec["n_features_original"]), (5, FR.SEED, 40))
        self.assertEqual(rec["selected_feature_list_sha256"], FR.list_sha256(idx))
        self.assertEqual(rec["fit_rows"], "outer_train_only")
        self.assertEqual(rec["inner_split"], "jtt1-inner-fold-labels")
        self.assertIn(rec["model"]["name"], ("catboost", "xgboost"))

    def test_the_xgboost_fallback_selects_deterministically(self):
        raw = pack(*context(seed=5))
        a = selection_bytes(raw, prefer="xgboost")
        self.assertEqual(a, selection_bytes(raw, prefer="xgboost"))
        self.assertIn(b'"name": "xgboost"', a)

    def test_reduce_keeps_both_blocks_positionally(self):
        tr, te, lab = context()
        raw = pack(tr, te, lab)
        out, rec = FR.teacher_view(raw, 5)
        h = jtt1.parse_header(out)
        self.assertEqual((h["n_train"], h["n_test"], h["p"], h["inner_folds"]), (120, 60, 5, 2))
        cols = rec["selected_feature_indices"] + [40]
        _, labels, X, y = jtt1.train_view(out)
        np.testing.assert_array_equal(labels, lab)
        np.testing.assert_array_equal(np.column_stack([X, y]), tr[:, cols])
        rows = np.frombuffer(memoryview(out)[jtt1.HEADER_BYTES + 120:], dtype="<f8").reshape(180, 6)
        np.testing.assert_array_equal(rows[120:], te[:, cols])

    def test_bad_kept_columns_are_refused(self):
        raw = pack(*context())
        for idx in ([], [3, 3], [5, 2], [0, 40]):
            with self.subTest(idx=idx), self.assertRaises(ValueError):
                FR.reduce_jtt1(raw, idx)


def constant_fold_context(y_train, n_test=5, p=513, seed=7, scales=None):
    """Lane C's #257 reproducer shape: n_train=20 in two inner folds of 10, p=513, k=512."""
    rng = np.random.default_rng(seed)
    n_train = len(y_train)
    X = rng.normal(size=(n_train + n_test, p))
    if scales is not None:
        X = X * scales
    y = np.concatenate([np.asarray(y_train, dtype=np.float64), rng.normal(size=n_test)])
    block = np.column_stack([X, y])
    return block[:n_train], block[n_train:], np.array([0] * (n_train // 2) + [1] * (n_train - n_train // 2))


class ConstantFolds(unittest.TestCase):
    """#257 review 5443900942: a constant inner training target is a valid fold. It is supported, never excluded."""

    def test_one_constant_training_fold_selects_deterministically(self):
        tr, te, lab = constant_fold_context([0.0] * 10 + list(range(10)))
        raw = pack(tr, te, lab)
        for prefer in ("catboost", "xgboost"):
            with self.subTest(prefer=prefer):
                idx, rec = FR.select_for_jtt1(raw, 512, prefer=prefer)
                self.assertEqual(len(idx), 512)
                self.assertEqual(rec["constant_inner_folds"], [1])   # held-out fold 1 trains on fold 0: all zeros
                self.assertIsNone(rec["fallback"])
                self.assertEqual((rec["method"], rec["model"]["name"]), (FR.METHOD, prefer))
                self.assertEqual(rec["selected_feature_list_sha256"], FR.list_sha256(idx))
                self.assertEqual(selection_bytes(raw, 512, prefer=prefer), selection_bytes(raw, 512, prefer=prefer))

    def test_wholly_constant_targets_fall_back_to_train_row_variance(self):
        scales = np.ones(513)
        scales[[5, 200, 400]] = 0.01   # three clearly lowest-variance columns are the ones dropped
        tr, te, lab = constant_fold_context([3.5] * 20, scales=scales)
        raw = pack(tr, te, lab)
        idx, rec = FR.select_for_jtt1(raw, 510)
        self.assertEqual(rec["constant_inner_folds"], [0, 1])
        self.assertEqual(rec["fallback"], FR.FALLBACK_ALL_CONSTANT)
        expect = sorted(sorted(range(513), key=lambda j: (-tr[:, j].var(), j))[:510])
        self.assertEqual([int(i) for i in idx], expect)
        self.assertNotIn(5, idx)
        self.assertNotIn(200, idx)
        self.assertNotIn(400, idx)
        self.assertEqual(selection_bytes(raw, 510), selection_bytes(raw, 510))

    def test_the_leak_gate_holds_with_constant_folds(self):
        for y_train in ([0.0] * 10 + list(range(10)), [3.5] * 20):
            tr, te, lab = constant_fold_context(y_train)
            base = selection_bytes(pack(tr, te, lab), 512)
            rng = np.random.default_rng(11)
            for name, t in (("shuffled", np.column_stack([te[:, :-1], rng.permutation(te[:, -1])])),
                            ("replaced", np.column_stack([te[:, :-1], 1e6 * te[:, 3]])),
                            ("rows", np.column_stack([rng.normal(size=te[:, :-1].shape), te[:, -1]]))):
                with self.subTest(constant=y_train[0] == y_train[-1], perturbation=name):
                    self.assertEqual(selection_bytes(pack(tr, t, lab), 512), base)


def constant_feature_context():
    """Reviewer #257 repro: outer columns vary, but inner fold 0 trains on all ones."""
    X = np.ones((25, 513))
    X[:10] = np.random.default_rng(91).normal(size=(10, 513))
    block = np.column_stack([X, np.arange(25, dtype=np.float64)])
    return block[:20], block[20:], np.array([0] * 10 + [1] * 10)


class ConstantFeatureFolds(unittest.TestCase):
    def test_one_constant_feature_fold_selects_with_both_models(self):
        tr, te, lab = constant_feature_context()
        self.assertTrue(np.all(tr[:, :-1].var(axis=0) > 0))
        self.assertTrue(np.all(tr[lab == 1, :-1].var(axis=0) == 0))
        raw = pack(tr, te, lab)
        for prefer in ("catboost", "xgboost"):
            with self.subTest(prefer=prefer):
                idx, rec = FR.select_for_jtt1(raw, 512, prefer=prefer)
                self.assertEqual(len(idx), 512)
                self.assertEqual(rec["constant_feature_inner_folds"], [0])
                self.assertEqual(rec["constant_inner_folds"], [])
                self.assertIsNone(rec["fallback"])
                self.assertEqual(rec["model"]["name"], prefer)
                self.assertEqual(selection_bytes(raw, 512, prefer=prefer), selection_bytes(raw, 512, prefer=prefer))

    def test_all_uninformative_folds_use_train_variance_and_column_ties(self):
        for shape in ("features", "mixed", "globally constant"):
            tr, te, lab = constant_feature_context()
            tr[lab == 0, :-1] = np.arange(513)
            if shape == "mixed":
                tr[lab == 0, -1] = 7.0
            elif shape == "globally constant":
                tr[:, :-1] = 1.0
            expect = sorted(sorted(range(513), key=lambda j: (-tr[:, j].var(), j))[:512])
            for prefer in ("catboost", "xgboost"):
                with self.subTest(shape=shape, prefer=prefer):
                    idx, rec = FR.select_for_jtt1(pack(tr, te, lab), 512, prefer=prefer)
                    self.assertEqual(list(idx), expect)
                    self.assertEqual(rec["constant_inner_folds"], [1] if shape == "mixed" else [])
                    self.assertEqual(rec["constant_feature_inner_folds"], [0] if shape == "mixed" else [0, 1])
                    self.assertEqual(rec["fallback"], FR.FALLBACK_ALL_UNINFORMATIVE)
                    if shape == "globally constant":
                        self.assertEqual(list(idx), list(range(512)))

    def test_constant_feature_fold_preserves_outer_test_invariance(self):
        tr, te, lab = constant_feature_context()
        rng = np.random.default_rng(92)
        variants = [te.copy() for _ in range(4)]
        variants[0][:, -1] = rng.permutation(te[:, -1])
        variants[1][:, -1] = 1e6 * te[:, 0]
        variants[2][:, -1] = 0.0
        variants[3][:, :-1] = rng.normal(size=te[:, :-1].shape)
        variants.extend([te[::-1], te[:2]])
        for prefer in ("catboost", "xgboost"):
            base = selection_bytes(pack(tr, te, lab), 512, prefer=prefer)
            for i, test in enumerate(variants):
                with self.subTest(prefer=prefer, perturbation=i):
                    self.assertEqual(selection_bytes(pack(tr, test, lab), 512, prefer=prefer), base)

    def test_a_varying_informative_column_still_fits(self):
        tr, te, lab = constant_feature_context()
        tr[:, :-1] = 1.0
        tr[:, 512] = np.arange(20)
        tr[:, -1] = 2 * tr[:, 512]
        for prefer in ("catboost", "xgboost"):
            with self.subTest(prefer=prefer):
                idx, rec = FR.select_for_jtt1(pack(tr, te, lab), 1, prefer=prefer)
                self.assertEqual(list(idx), [512])
                self.assertNotIn("constant_feature_inner_folds", rec)
                self.assertEqual(rec["constant_inner_folds"], [])
                self.assertIsNone(rec["fallback"])


if __name__ == "__main__":
    unittest.main()
