from __future__ import annotations

import unittest

import numpy as np

from research.flywheel_selfplay.h75_closed_loop_evolution.scoring.score_panel import (
    load_reference,
)
from research.flywheel_selfplay.h75_w4_cached_baseline_shard import run_shard as h75_w4
from research.flywheel_selfplay.h79_w4_prior_2plus2_cached.run_shard import (
    ARMS,
    ELO_SOURCE,
    Oracle,
    check_family_exclusion,
    contrast,
    lengthscale,
    select,
    summarize,
)

IDS = [f"{i:064x}" for i in range(6)]
# c0/c1 lead the prior; c2 sits next to c0, c3/c4 are far from both revealed programs.
MU = np.array([1.0, 0.875, 0.5, 0.625, 0.5625, 0.0])
X = np.array(
    [[0.0, 0.0], [10.0, 0.0], [0.25, 0.0], [0.0, 10.0], [10.0, 10.0], [5.0, 5.0]]
)


def prior(mu: np.ndarray = MU) -> dict:
    return {"mu": mu, "prior10_cost": -mu[::-1].copy(), "proj16": X, "scale": 1.0}


def oracle(utilities: list[float], gate: list[bool] | None = None) -> Oracle:
    return Oracle(
        -np.asarray(utilities),
        np.asarray(gate if gate is not None else [True] * len(utilities)),
    )


class OracleTest(unittest.TestCase):
    def test_reveals_each_candidate_once_within_four(self) -> None:
        o = oracle(list(MU))
        self.assertEqual(o.reveal([0, 1]), [1.0, 0.875])
        with self.assertRaises(ValueError):
            o.reveal([1])
        with self.assertRaises(ValueError):
            o.reveal([2, 2])
        o.reveal([2, 3])
        with self.assertRaises(ValueError):
            o.reveal([4])

    def test_gate_failure_is_minus_two(self) -> None:
        self.assertEqual(oracle([0.5, 0.5], [False, True]).reveal([0, 1]), [-2.0, 0.5])


class PolicyTest(unittest.TestCase):
    def test_top4_ranks_by_prior_and_breaks_ties_by_program_sha(self) -> None:
        self.assertEqual(
            select("gbm_top4", 701, "ctx", IDS, prior(), oracle(list(MU))), [0, 1, 3, 4]
        )
        tied = np.array([1.0, 1.0, 0.5, 0.5, 0.5, 0.0])
        self.assertEqual(
            select("gbm_top4", 701, "ctx", IDS, prior(tied), oracle(list(MU))),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            select("prior10_top4", 701, "ctx", IDS, prior(), oracle(list(MU))),
            [5, 4, 2, 1],
        )

    def test_zero_residuals_reduce_two_plus_two_to_top4(self) -> None:
        for arm in ("gbm_2plus2", "gbm_2plus2_swap"):
            self.assertEqual(
                select(arm, 701, "ctx", IDS, prior(), oracle(list(MU))), [0, 1, 3, 4]
            )

    def test_positive_residual_pulls_in_its_neighbour(self) -> None:
        outcomes = list(MU)
        outcomes[0] = 3.0
        picks = select("gbm_2plus2", 701, "ctx", IDS, prior(), oracle(outcomes))
        self.assertEqual(picks[:2], [0, 1])
        self.assertIn(2, picks[2:])

    def test_gate_failure_pushes_away_from_its_neighbour(self) -> None:
        mu = np.array([1.0, 0.875, 0.75, 0.625, 0.5625, 0.0])
        picks = select(
            "gbm_2plus2",
            701,
            "ctx",
            IDS,
            prior(mu),
            oracle(list(mu), [False] + [True] * 5),
        )
        self.assertEqual(picks, [0, 1, 3, 4])
        self.assertEqual(
            select("gbm_top4", 701, "ctx", IDS, prior(mu), oracle(list(mu))),
            [0, 1, 2, 3],
        )

    def test_swap_placebo_detaches_the_residual_from_its_program(self) -> None:
        outcomes = list(MU)
        outcomes[0] = 3.0
        picks = select("gbm_2plus2_swap", 701, "ctx", IDS, prior(), oracle(outcomes))
        self.assertNotIn(2, picks[2:])
        self.assertIn(4, picks[2:])

    def test_random4_is_the_published_h75_selector(self) -> None:
        o = oracle(list(MU))
        picks = select("random4", 702, "ctx", IDS, prior(), o)
        self.assertEqual(
            [IDS[i] for i in picks], h75_w4.select_ids("random4", IDS, "ctx", 702)
        )
        self.assertEqual(o.revealed, picks)

    def test_degenerate_embedding_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            lengthscale(np.zeros((4, 16)))
        self.assertAlmostEqual(lengthscale(np.array([[0.0], [1.0], [3.0]])), 2.0)


class ScoringTest(unittest.TestCase):
    def test_family_exclusion_requires_whole_families_per_w1_fold(self) -> None:
        folds = {
            "folds": [
                {"fold_index": 0, "held_contexts": ["a", "b"]},
                {"fold_index": 1, "held_contexts": ["c"]},
            ]
        }
        rows = [{"context_sha256": "a", "fold": 1}]
        family = {"a": "f1", "b": "f1", "c": "f2"}
        result = check_family_exclusion(folds, family, rows)
        self.assertEqual(result["acquisition_families_single_w1_fold"], 1)
        self.assertEqual(
            result["acquisition_contexts_whose_h67_fold_label_differs_from_w1"], 1
        )
        with self.assertRaises(ValueError):
            check_family_exclusion(folds, {"a": "f1", "b": "f2", "c": "f1"}, rows)

    def test_contrast_reports_family_harm_and_paired_bootstrap(self) -> None:
        deltas = {"f1": 0.5, "f2": -0.25, "f3": 0.0, "f4": 0.00005}
        draws = np.array([[0, 1, 2, 3], [0, 0, 0, 0], [1, 1, 1, 1]])
        result = contrast(deltas, draws)
        self.assertAlmostEqual(result["family_mean"], 0.0625125)
        self.assertEqual(result["families_losing_more_than_0_01_fraction"], 0.25)
        self.assertEqual(result["family_win_tie_loss"], [1, 2, 1])
        self.assertAlmostEqual(
            result["bootstrap_p2_5"], -0.25 + 0.05 * (0.0625125 + 0.25)
        )

    def test_summary_anchors_family_elo_on_prior_top4(self) -> None:
        family = {"c1": "f1", "c2": "f1", "c3": "f2"}
        base = {
            "random4": 0.0,
            "prior10_top4": 0.25,
            "gbm_top4": 0.5,
            "gbm_2plus2_swap": 0.5,
        }
        ctx = {
            c: {**base, "gbm_2plus2": 0.5 + bonus}
            for c, bonus in (("c1", 0.25), ("c2", 0.0), ("c3", 0.25))
        }
        result = summarize(ctx, family, load_reference(ELO_SOURCE))
        self.assertEqual(set(result["family_elo"]["ratings"]), set(ARMS))
        self.assertEqual(result["family_elo"]["ratings"]["gbm_top4"], 1500.0)
        self.assertGreater(result["family_elo"]["ratings"]["gbm_2plus2"], 1500.0)
        self.assertEqual(result["primary"]["families"], 2)
        self.assertAlmostEqual(result["primary"]["family_mean"], 0.1875)
        self.assertAlmostEqual(
            result["family_mean_retained_best_utility"]["gbm_2plus2"], 0.6875
        )


if __name__ == "__main__":
    unittest.main()
