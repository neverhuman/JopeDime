"""Owner regression test (relayed directive 2026-10-05 15:47Z, UPSTREAM-SCALE-01 / S-STAT-09): no outer-test row may
affect any training encoding, scaler, inner-fold assignment, teacher label or best-GP selection.

Each positive case is paired with a negative control that must FAIL the same check: the upstream all-rows scaler
(research_2k), a selector that reads the outer-test score, and a one-ULP change to one training value. A check that
cannot fail would not be evidence (S-STAT-01).
"""
import unittest

import numpy as np

from dltrain import jtt1, metafeatures, trainonly

N, P, N_TRAIN = 240, 5, 170


def ridge(cols, lam=1e-6):
    def fit_predict(x_fit, y_fit, x_eval):
        a = np.column_stack([np.ones(x_fit.shape[0]), x_fit[:, cols]])
        coef = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ y_fit)
        return np.column_stack([np.ones(x_eval.shape[0]), x_eval[:, cols]]) @ coef
    return fit_predict


def mean_model(x_fit, y_fit, x_eval):
    return np.full(x_eval.shape[0], float(np.mean(y_fit)))


PROGRAMS = {"p_all": ridge(list(range(P))), "p_x0": ridge([0]), "p_x34": ridge([3, 4]), "p_mean": mean_model}


def a0_encoder(raw):
    return metafeatures.from_raw(raw)[0]


def fixture(seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, P)) * np.array([1.0, 30.0, 0.01, 5.0, 1e3]) + np.array([0.0, -4.0, 2.0, 100.0, -7.0])
    x[:, 2] = np.round(x[:, 2], 3)                      # a coarse column, so exact duplicates and ties exist
    x[10:16] = x[3]                                      # an exact-feature component of 7 rows
    y = 1.5 * x[:, 0] - 0.02 * x[:, 1] + 0.001 * x[:, 4] + rng.normal(scale=0.3, size=N)
    m = np.column_stack([x, y])
    perm = rng.permutation(N)
    return m, sorted(perm[:N_TRAIN].tolist()), sorted(perm[N_TRAIN:].tolist())


def variants(m, tr, te, seed=11):
    """Perturb or replace only the outer-test rows (features and target)."""
    rng = np.random.default_rng(seed)
    out = {}
    v = m.copy(); v[te] *= 1000.0; out["test_x1000"] = v
    v = m.copy(); v[te] = rng.normal(size=(len(te), P + 1)) * 1e4; out["test_replaced_random"] = v
    v = m.copy(); v[te] = m[np.asarray(te)[rng.permutation(len(te))]]; out["test_permuted"] = v
    v = m.copy(); v[te, -1] = np.where(np.arange(len(te)) % 2, 1e6, -1e6); out["test_target_extreme"] = v
    v = m.copy(); v[te, :-1] = m[tr].min(axis=0)[:-1] - 50.0; out["test_features_below_train_min"] = v
    out["test_half_dropped"] = (m, te[: len(te) // 2])
    out["test_none"] = (m, [])
    big = np.vstack([m, rng.normal(size=(60, P + 1)) * 1e5])
    out["test_plus_60_new_rows"] = big
    split = trainonly.prepare_split(m, tr, te)
    v = m.copy(); v[te, -1] = PROGRAMS["p_x0"](split["train"][:, :-1], split["train"][:, -1], split["test"][:, :-1]) \
        * split["span"][-1] + split["lo"][-1]
    out["test_target_equals_p_x0_prediction"] = v
    return out


def legacy_all_rows_prepare(matrix, outer_train, outer_test, min_inner_rows=trainonly.MIN_INNER_ROWS):
    """The upstream research_2k behaviour: min-max fitted over every row of the view, outer-test rows included."""
    m = np.asarray(matrix, dtype="<f8")
    rows = list(outer_train) + list(outer_test)
    lo, span = trainonly.fit_minmax(m[rows])
    tr = trainonly.apply_minmax(m[list(outer_train)], lo, span)
    te = trainonly.apply_minmax(m[list(outer_test)], lo, span) if outer_test else np.empty((0, m.shape[1]))
    inner, ev = trainonly.inner_assignments(tr, min_inner_rows)
    return {"lo": lo, "span": span, "train": tr, "test": te, "inner_labels": inner, "inner_cv": ev,
            "train_raw": m[list(outer_train)], "test_raw": m[list(outer_test)].reshape(-1, m.shape[1]),
            "jtt1": trainonly.jtt1_bytes(tr, te, inner)}


def test_score_selector(records):
    """A forbidden selector: the lowest outer-test RMSE."""
    scored = [r for r in records if r["outer_test"]["rmse"] is not None]
    return min(scored, key=lambda r: (r["outer_test"]["rmse"], r["program_id"]))["program_id"] if scored else None


class TrainOnlyInvariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.tr, cls.te = fixture()
        cls.var = variants(cls.m, cls.tr, cls.te)

    def test_every_train_field_is_bit_identical_under_test_row_changes(self):
        rep = trainonly.counterfactual_report(self.m, self.tr, self.te, self.var, PROGRAMS, encoder=a0_encoder)
        self.assertEqual(rep["differing"], {})
        self.assertTrue(rep["bit_identical"])
        self.assertEqual(len(rep["variants"]), len(self.var))
        for name, v in rep["variants"].items():
            self.assertEqual({f: v["fields"][f] for f in trainonly.TRAIN_FIELDS},
                             {f: rep["base"][f] for f in trainonly.TRAIN_FIELDS}, name)
        # the perturbations are real: the separately stored outer-test score moves
        self.assertTrue(rep["variants"]["test_x1000"]["outer_test_changed"])
        self.assertTrue(rep["variants"]["test_replaced_random"]["outer_test_changed"])

    def test_negative_control_all_rows_scaler_fails(self):
        rep = trainonly.counterfactual_report(self.m, self.tr, self.te, self.var, PROGRAMS, encoder=a0_encoder,
                                              prepare=legacy_all_rows_prepare)
        self.assertFalse(rep["bit_identical"])
        for name in ("test_x1000", "test_replaced_random", "test_features_below_train_min", "test_plus_60_new_rows"):
            self.assertIn(name, rep["differing"])
            for field in ("scaler", "train_block", "jtt1_train_rows", "encoding", "teacher_inner"):
                self.assertIn(field, rep["differing"][name], (name, field))
        self.assertTrue(any("inner_labels" in d for d in rep["differing"].values()),
                        "the all-rows scaler should move at least one inner-fold assignment (digest tie-break)")
        # permuting the test rows keeps the all-rows min/max, so even the legacy path is unchanged there
        self.assertNotIn("test_permuted", rep["differing"])

    def test_negative_control_test_score_selector_fails(self):
        rep = trainonly.counterfactual_report(self.m, self.tr, self.te, self.var, PROGRAMS, selector=test_score_selector)
        self.assertEqual(rep["differing"].get("test_target_equals_p_x0_prediction"), ["selection"])
        ok = trainonly.counterfactual_report(self.m, self.tr, self.te,
                                             {"t": self.var["test_target_equals_p_x0_prediction"]}, PROGRAMS)
        self.assertTrue(ok["bit_identical"])

    def test_one_ulp_change_to_one_train_value_is_detected(self):
        base = trainonly.train_side(trainonly.prepare_split(self.m, self.tr, self.te), PROGRAMS, a0_encoder)
        v = self.m.copy()
        r, c = self.tr[5], 1
        v[r, c] = np.nextafter(v[r, c], np.inf)
        got = trainonly.train_side(trainonly.prepare_split(v, self.tr, self.te), PROGRAMS, a0_encoder)
        self.assertIn("train_block", [f for f in trainonly.TRAIN_FIELDS if got[f] != base[f]])
        with self.assertRaises(ValueError):
            trainonly.counterfactual_report(self.m, self.tr, self.te, {"train_touched": v}, PROGRAMS)


class InnerValidationInvariance(unittest.TestCase):
    """C5 scope: with the fold assignment held fixed, fold j's validation rows reach neither fold j's refitted scaler nor
    its fitted program. (The outer-train scaler, the encoder inputs and the assignment itself are functions of every
    outer-train row, inner-validation rows included, so they are not expected to stay fixed here.)"""

    def setUp(self):
        self.m, self.tr, self.te = fixture(5)
        self.split = trainonly.prepare_split(self.m, self.tr, self.te)
        self.g = self.split["inner_labels"]

    def perturbed(self, j, how):
        v = self.m.copy()
        rows = np.asarray(self.tr)[self.g == j]
        rng = np.random.default_rng(j + 100)
        if how == "scale":
            v[rows] = v[rows] * 1000.0 + 3.0
        else:
            v[rows] = rng.normal(size=(rows.size, P + 1)) * 1e4
        return trainonly.prepare_split(v, self.tr, self.te, inner_labels=self.g)

    def test_fold_fit_bit_identical_and_only_its_validation_score_moves(self):
        for j in (0, 1):
            for how in ("scale", "replace"):
                alt = self.perturbed(j, how)
                for name, fn in PROGRAMS.items():
                    a = trainonly.inner_fold_fit(self.split, j, fn, fingerprint=True)
                    b = trainonly.inner_fold_fit(alt, j, fn, fingerprint=True)
                    self.assertEqual((a["scaler_sha256"], a["fit_sha256"]), (b["scaler_sha256"], b["fit_sha256"]),
                                     (j, how, name))
                    self.assertNotEqual(a["rmse"], b["rmse"], (j, how, name))
                    o_a = trainonly.inner_fold_fit(self.split, 1 - j, fn, fingerprint=True)
                    o_b = trainonly.inner_fold_fit(alt, 1 - j, fn, fingerprint=True)
                    self.assertNotEqual(o_a["scaler_sha256"], o_b["scaler_sha256"], "the other fold trains on these rows")

    def test_negative_control_outer_scaled_inner_cv_fails(self):
        """The pre-C5 behaviour (inner folds fitted on the outer-train-scaled block) leaks fold j's validation rows."""
        alt = self.perturbed(0, "scale")
        fn = PROGRAMS["p_all"]

        def outer_scaled_fit(split, j):
            tr, g = split["train"], split["inner_labels"]
            return np.asarray(fn(tr[g != j, :-1], tr[g != j, -1], tr[g != j, :-1]), "<f8").tobytes()
        self.assertNotEqual(outer_scaled_fit(self.split, 0), outer_scaled_fit(alt, 0))

    def test_fixed_labels_are_checked(self):
        with self.assertRaises(ValueError):
            trainonly.prepare_split(self.m, self.tr, self.te, inner_labels=self.g[:-1])
        bad = self.g.copy()
        dup = [i for i, r in enumerate(self.tr) if r in (3, 10)]
        if len(dup) == 2:
            bad[dup[0]], bad[dup[1]] = 0, 1
            with self.assertRaises(trainonly.IneligibleSplit):
                trainonly.prepare_split(self.m, self.tr, self.te, inner_labels=bad)
        with self.assertRaises(trainonly.IneligibleSplit):
            trainonly.prepare_split(self.m, self.tr, self.te, inner_labels=np.zeros_like(self.g))

    def test_min_inner_rows_is_not_a_silent_no_op(self):
        loads = np.bincount(self.g, minlength=2)
        trainonly.prepare_split(self.m, self.tr, self.te, min_inner_rows=int(loads.min()))
        with self.assertRaises(trainonly.IneligibleSplit):
            trainonly.prepare_split(self.m, self.tr, self.te, min_inner_rows=int(loads.min()) + 1)


class Design(unittest.TestCase):
    def setUp(self):
        self.m, self.tr, self.te = fixture(3)
        self.split = trainonly.prepare_split(self.m, self.tr, self.te)

    def test_scaler_fitted_on_train_rows_only(self):
        s = self.split
        np.testing.assert_array_equal(s["lo"], self.m[self.tr].min(axis=0))
        self.assertEqual(s["train"].min(axis=0).tolist(), [0.0] * (P + 1))
        self.assertEqual(s["train"].max(axis=0).tolist(), [1.0] * (P + 1))
        self.assertTrue(bool(np.any((s["test"] < 0) | (s["test"] > 1))), "test rows use train constants, not their own")

    def test_jtt1_layout_and_train_view(self):
        h, lab, x, y = jtt1.train_view(self.split["jtt1"])
        self.assertEqual((h["n_train"], h["n_test"], h["p"], h["inner_folds"]), (N_TRAIN, N - N_TRAIN, P, 2))
        np.testing.assert_array_equal(lab, self.split["inner_labels"])
        self.assertEqual(np.column_stack([x, y]).tobytes(), self.split["train"].tobytes())

    def test_inner_cv_sees_train_rows_only_and_test_score_sees_test_rows(self):
        test_rows = {r.tobytes() for r in self.split["test"][:, :-1]}
        seen = []

        def spy(x_fit, y_fit, x_eval):
            seen.append((x_fit.copy(), y_fit.copy(), x_eval.copy()))
            return np.zeros(x_eval.shape[0])

        trainonly.inner_cv(self.split, spy)
        self.assertEqual(len(seen), 2)
        g = self.split["inner_labels"]
        raw = self.m[self.tr]
        for j, (x_fit, y_fit, x_eval) in enumerate(seen):
            lo, span = trainonly.fit_minmax(raw[g != j])          # fold j's own inner-train rows, nothing else
            want = trainonly.apply_minmax(raw, lo, span)
            self.assertEqual(x_fit.tobytes(), np.ascontiguousarray(want[g != j, :-1]).tobytes())
            self.assertEqual(y_fit.tobytes(), np.ascontiguousarray(want[g != j, -1]).tobytes())
            self.assertEqual(x_eval.tobytes(), np.ascontiguousarray(want[g == j, :-1]).tobytes())
            # outer-test rows never reach the fold, except exact raw duplicates of a training row (the fixture's 3/10-15)
            raw_train = {r.tobytes() for r in raw[:, :-1]}
            own = [i for i in self.te if self.m[i, :-1].tobytes() not in raw_train]
            self.assertGreater(len(own), 0)
            outer_test_j = trainonly.apply_minmax(self.m[own], lo, span)[:, :-1]
            self.assertFalse({r.tobytes() for r in outer_test_j} & {r.tobytes() for r in np.vstack([x_fit, x_eval])})
        seen.clear()
        trainonly.outer_test(self.split, spy)
        x_fit, _y, x_eval = seen[0]
        self.assertEqual(x_fit.tobytes(), self.split["train"][:, :-1].tobytes())
        self.assertTrue(all(r.tobytes() in test_rows for r in x_eval))

    def test_scores_are_in_raw_target_units(self):
        y_tr, y_te = self.m[self.tr, -1], self.m[self.te, -1]
        want_test = float(np.sqrt(np.mean((y_te - np.mean(y_tr)) ** 2)))
        got = trainonly.outer_test(self.split, mean_model)["rmse"]
        self.assertLess(abs(got - want_test), 1e-9 * want_test)   # mean of scaled y mapped back: a few ULP of rounding
        g = self.split["inner_labels"]
        for j in (0, 1):
            r = trainonly.inner_fold_fit(self.split, j, mean_model)
            want = float(np.sqrt(np.mean((y_tr[g == j] - np.mean(y_tr[g != j])) ** 2)))
            self.assertEqual(r["baseline_rmse"], want)
            self.assertLess(abs(r["rmse"] - want), 1e-9 * want)

    def test_inner_folds_are_group_safe_and_balanced(self):
        g = self.split["inner_labels"]
        by_row = {}
        for row, lab in zip(self.split["train"][:, :-1], g):
            self.assertEqual(by_row.setdefault(row.tobytes(), int(lab)), int(lab))
        self.assertGreaterEqual(int(np.bincount(g, minlength=2).min()), trainonly.MIN_INNER_ROWS)

    def test_scores_stored_separately_and_selection_ignores_test(self):
        recs = [trainonly.teacher_record(self.split, pid, fn) for pid, fn in sorted(PROGRAMS.items())]
        for r in recs:
            self.assertEqual(set(r), {"program_id", "inner_cv", "outer_test"})
            self.assertNotIn("rmse", r["inner_cv"])
        inner = [(r["program_id"], r["inner_cv"]) for r in recs]
        best = trainonly.select_best(inner)
        self.assertEqual(best, "p_all")
        with self.assertRaises(ValueError):
            trainonly.select_best([(r["program_id"], r["outer_test"]) for r in recs])
        for extra in ("outer_test", "rmse", "q_test_grid", "test_rmse"):
            with self.assertRaises(ValueError):
                trainonly.select_best([(r["program_id"], {**r["inner_cv"], extra: 0.0}) for r in recs])
        with self.assertRaises(ValueError):
            trainonly.select_best([(recs[0]["program_id"], {k: v for k, v in recs[0]["inner_cv"].items() if k != "q"})])
        self.assertEqual(trainonly.select_best(list(reversed(inner))), best)

    def test_gate_failures_rank_last(self):
        a = {"gate_pass": False, "q_grid": 10_000_000, "q": 10.0, "rmse_j": [0.0, 0.0], "baseline_rmse_j": [1.0, 1.0]}
        b = {"gate_pass": True, "q_grid": -5, "q": -5e-6, "rmse_j": [1.0, 1.0], "baseline_rmse_j": [1.0, 1.0]}
        self.assertEqual(trainonly.select_best([("a", a), ("b", b)]), "b")
        self.assertIsNone(trainonly.select_best([]))

    def test_refusals(self):
        with self.assertRaises(ValueError):
            trainonly.prepare_split(self.m, self.tr, self.tr[:3])
        with self.assertRaises(ValueError):
            trainonly.prepare_split(self.m, self.tr + self.tr[:1], self.te)
        bad = self.m.copy()
        bad[self.tr[0], 0] = np.nan
        with self.assertRaises(ValueError):
            trainonly.prepare_split(bad, self.tr, self.te)
        with self.assertRaises(trainonly.IneligibleSplit):
            trainonly.prepare_split(self.m, self.tr[:40], self.te)
        const = self.m.copy()
        const[:, :-1] = 1.0
        with self.assertRaises(trainonly.IneligibleSplit):
            trainonly.prepare_split(const, self.tr, self.te)

    def test_constant_column_divides_by_one(self):
        m = self.m.copy()
        m[:, 2] = 4.25
        s = trainonly.prepare_split(m, self.tr, self.te)
        self.assertEqual(float(s["span"][2]), 1.0)
        self.assertEqual(set(s["train"][:, 2].tolist()), {0.0})


if __name__ == "__main__":
    unittest.main()
