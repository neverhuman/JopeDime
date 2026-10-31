import math
import unittest

from runner import family_folds, mean_regret_by_family, select_tied_candidate


class H123HelpersTest(unittest.TestCase):
    def test_mean_regret_weights_families_equally(self):
        # Family A has two contexts; family B has one. Family means are 2 and 1.5.
        self.assertEqual(
            mean_regret_by_family(
                [[3, 2], [4, 1], [5, 0]],
                [[2, 1], [2, 1], [4, 3]],
                ["A", "A", "B"],
            ),
            1.75,
        )

    def test_selection_uses_only_max_utility_ties(self):
        candidates = [
            {"round": 10, "utility": 1.0, "artifact_sha256": "b" * 64, "train_mean_regret": 0.2},
            {"round": 11, "utility": 1.0, "artifact_sha256": "c" * 64, "train_mean_regret": 0.1},
            {"round": 12, "utility": 1.0, "artifact_sha256": "a" * 64, "train_mean_regret": 0.1},
            {"round": 13, "utility": 0.9, "artifact_sha256": "d" * 64, "train_mean_regret": 0.0},
        ]
        self.assertEqual(select_tied_candidate(candidates)["round"], 11)

    def test_selection_requires_three_verified_top_artifacts(self):
        candidates = [
            {"round": n, "utility": 1.0, "artifact_sha256": f"{n:064x}", "train_mean_regret": 0.2}
            for n in (1, 2)
        ]
        with self.assertRaisesRegex(ValueError, "fewer than three"):
            select_tied_candidate(candidates)

    def test_nonfinite_utilities_refused(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            mean_regret_by_family([[1.0, math.inf]], [[1.0, 0.0]], ["A"])

    def test_family_folds_are_deterministic_and_reject_non_train_rows(self):
        rows = [
            {"family_id": f"f{i}", "dl_split": "DL-Train", "task": "regression", "origin": "real"}
            for i in range(12)
        ]
        first = family_folds(rows)
        self.assertEqual(first, family_folds(rows))
        self.assertEqual(len(first["score_families"]), 2)
        self.assertFalse(set(first["train_families"]) & set(first["score_families"]))
        rows[0]["dl_split"] = "DL-Val"
        with self.assertRaisesRegex(ValueError, "real DL-Train"):
            family_folds(rows)


if __name__ == "__main__":
    unittest.main()
