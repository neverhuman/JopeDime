"""Boundary and negative tests for independent red-team semantics."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import io
import numpy as np
import independent_replay as replay
import statistics as stats


class AuditTests(unittest.TestCase):
    def test_ties_are_never_wins(self):
        self.assertEqual(
            stats.win_tie_loss([-1e-4, 0, 1e-4, -1.01e-4, 1.01e-4]), [1, 3, 1]
        )

    def test_elo_direction_and_closed_form(self):
        for wins, losses in [(0, 0), (10, 2), (2, 10), (0, 7), (7, 0)]:
            expected = 1500 + 400 * np.log10((wins + 0.01) / (losses + 0.01))
            self.assertAlmostEqual(stats.elo_mm(wins, losses), expected, places=10)
        self.assertGreater(stats.elo_mm(10, 2), 1500)
        self.assertLess(stats.elo_mm(2, 10), 1500)

    def test_strict_margin_infinite_and_tie_break(self):
        scores = np.array([[1, 2, 2], [1, 1, 1], [1, 1.5, 0]], dtype=float)
        np.testing.assert_array_equal(replay.select(scores, 0.5), [1, 0, 0])
        np.testing.assert_array_equal(replay.select(scores, np.inf), [0, 0, 0])

    def test_consensus_requires_same_nonzero_slot(self):
        m = np.array([[1, 1, 0, 2], [1, 2, 0, 2], [0, 3, 1, 2]])
        np.testing.assert_array_equal(replay.consensus(m, 2), [1, 0, 0, 2])
        np.testing.assert_array_equal(replay.consensus(m, 3), [0, 0, 0, 2])

    def test_family_equal_not_context_weighted(self):
        u = np.array([[0, 4], [0, 0], [0, 0], [0, 0]], dtype=float)
        p = np.ones((3, 4), dtype=int)
        d = stats.family_deltas(u, p, np.array(["A", "B", "B", "B"]))
        np.testing.assert_array_equal(d, np.array([[4, 0]] * 3))
        self.assertEqual(d.mean(), 2)

    def test_permutation_degenerate_singletons(self):
        u = np.array([[0, 1], [0, 2]], dtype=float)
        p = np.ones((3, 2), dtype=int)
        result = stats.permutation_null(u, p, np.array(["A", "B"]), draws=20)
        self.assertEqual(result["p_upper_plus_one"], 1)
        self.assertEqual(result["unique_statistics"], 1)
        self.assertEqual(result["families_with_nonconstant_pick_vectors"], 0)

    def test_permutation_constant_vectors_with_roundoff(self):
        utility = np.random.default_rng(5).normal(size=(10, 8))
        picks = np.full((3, 10), 2)
        families = np.array(["A"] * 3 + ["B"] * 5 + ["C"] * 2)
        result = stats.permutation_null(utility, picks, families, draws=20)
        self.assertEqual(result["p_upper_plus_one"], 1.0)
        self.assertEqual(result["observed"], result["null_min"])

    def test_identical_seeds_effective_one(self):
        p = np.array([[0, 1, 2]] * 3)
        d = np.array([[0.0, 1.0, -2.0]] * 3)
        self.assertAlmostEqual(
            stats.seed_dependence(p, d)["effective_seed_count_spectral"], 1
        )

    def test_prefix_does_not_request_validation_rows(self):
        # Record reads after header parsing; forbidden suffix contains a NaN sentinel.
        a = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]])
        with tempfile.TemporaryDirectory(
            prefix="h305-fixture-", dir=Path(__file__).parent
        ) as td:
            path = Path(td) / "shared.npz"
            np.savez(path, values=a)
            with zipfile.ZipFile(path) as z:
                raw = z.read("values.npy")
            stream = io.BytesIO(raw)

            class Wrapper:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def read(self, n=-1):
                    return stream.read(n)

            with patch.object(zipfile.ZipFile, "open", return_value=Wrapper()):
                actual = replay.prefix_array(path, "values", 2)
            np.testing.assert_array_equal(actual, a[:2])
            self.assertEqual(len(raw) - stream.tell(), a[2:].nbytes)

    def test_tampered_model_rejected_before_prediction(self):
        with tempfile.TemporaryDirectory(
            prefix="h305-fixture-", dir=Path(__file__).parent
        ) as td:
            directory = Path(td)
            (directory / "ranker.txt").write_text("tampered")
            config = {
                "ranker": {"file": "ranker.txt", "sha256": "0" * 64},
                "selected_n_estimators": 50,
            }
            with self.assertRaises(AssertionError):
                replay.member_pick(
                    directory, config, np.zeros((1, 8, 568), dtype=np.float32)
                )

    def test_nested_selection_tie_conservative_and_wrong_choice(self):
        from audit_structure import check_nested_lock

        table = [
            dict(tau=t, k=k, a=a, inner_gain=0.0, inner_switches=100)
            for t in (0.05, 0.1, 0.15)
            for k in (5, 6, 7)
            for a in (2, 3)
        ]
        chosen = next(x for x in table if (x["tau"], x["k"], x["a"]) == (0.05, 7, 3))
        lock = dict(
            inner_table=table,
            chosen=chosen,
            n_tied=18,
            outer_fold=0,
            root_seed=30001,
            members=[dict(selected_tau=0.05)] * 3,
        )
        self.assertTrue(check_nested_lock(lock)["selection_rule_matches"])
        lock["chosen"] = table[0]
        with self.assertRaises(AssertionError):
            check_nested_lock(lock)

    def test_outer_family_partition(self):
        families = ["same-family", "same-family", "another-family"]
        folds = [replay.outer_fold(f) for f in families]
        self.assertEqual(folds[0], folds[1])
        self.assertTrue(all(0 <= f < 4 for f in folds))


if __name__ == "__main__":
    unittest.main()
