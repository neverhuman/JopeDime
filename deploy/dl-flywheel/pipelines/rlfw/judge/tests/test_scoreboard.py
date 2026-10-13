"""J21 scoreboards keep proposal, search, and delivered claims apart from L6."""

import unittest

from rlfw.judge.scoreboard import (
    automated_report,
    delivered_generalization,
    l6_vs_frozen_v1,
    proposal_quality,
    search_quality,
)


def _proposal_rows():
    rows = []
    for context_id, first, programs, incumbent in (
        ("c1", 1.0, [0.2, 0.4, 1.0, 4.0, 0.1, 0.0, 0.3, 0.5], 3.0),
        ("c2", 3.0, [6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 10.0),
    ):
        rows.append({"context_id": context_id, "budget": 1, "programs": [first]})
        rows.append(
            {
                "context_id": context_id,
                "budget": 8,
                "programs": programs,
                "hwm": incumbent,
            }
        )
    return rows


def _search_rows():
    scores = {
        "independent": {"c1": 1.0, "c2": 1.0},
        "policy_seeded": {"c1": 3.0, "c2": 1.0},
        "frozen_seeded": {"c1": 1.0, "c2": 1.0},
        "retrieval": {"c1": 0.0, "c2": 2.0},
    }
    return [
        {"context_id": context_id, "arm": arm, "q": score}
        for arm, contexts in scores.items()
        for context_id, score in contexts.items()
    ]


def _delivered_rows():
    return [
        {
            "context_id": "c1",
            "candidates": [
                {"program_id": "pA", "train_q": 1.0, "outer_q": 0.0},
                {"program_id": "pB", "train_q": 0.0, "outer_q": 5.0},
            ],
        },
        {
            "context_id": "c2",
            "candidates": [
                {"program_id": "pD", "train_q": 2.0, "outer_q": 9.0},
                {"program_id": "pC", "train_q": 2.0, "outer_q": 2.0},
            ],
        },
    ]


class ScoreboardTest(unittest.TestCase):
    def test_proposal_reports_first_program_budget_and_unlowered_hwm(self):
        report = proposal_quality(_proposal_rows())
        self.assertEqual(report["budgets"], [1, 8])
        self.assertAlmostEqual(report["J_1"], 2.0)
        self.assertAlmostEqual(report["M_8"], 5.0)
        self.assertAlmostEqual(report["J_8_hwm"], 0.5)
        self.assertFalse(report["hwm_lowered"])
        self.assertFalse(report["promotion"])

    def test_proposal_refuses_a_budget_that_drops_a_context(self):
        rows = _proposal_rows()
        rows = [row for row in rows if not (row["budget"] == 8 and row["context_id"] == "c2")]
        with self.assertRaises(ValueError):
            proposal_quality(rows)

    def test_search_pairs_the_four_arms_against_independent(self):
        report = search_quality(_search_rows())
        self.assertEqual(report["n"], 2)
        self.assertAlmostEqual(report["mean_q"]["policy_seeded"], 2.0)
        self.assertAlmostEqual(report["delta_vs_independent"]["policy_seeded"], 1.0)
        self.assertAlmostEqual(report["delta_vs_independent"]["frozen_seeded"], 0.0)
        self.assertTrue(report["not_an_l6_verdict"])
        self.assertFalse(report["promotion"])

    def test_search_refuses_a_missing_arm(self):
        rows = [row for row in _search_rows() if row["arm"] != "retrieval"]
        with self.assertRaises(ValueError):
            search_quality(rows)

    def test_delivered_uses_training_selection_not_the_outer_maximum(self):
        report = delivered_generalization(_delivered_rows())
        self.assertAlmostEqual(report["mean_delivered_q"], 1.0)
        self.assertAlmostEqual(report["mean_oracle_outer_q"], 7.0)
        self.assertFalse(report["oracle_deployable"])
        self.assertFalse(report["oracle_affects_selection"])
        self.assertNotAlmostEqual(
            report["mean_delivered_q"], report["mean_oracle_outer_q"]
        )

    def test_l6_below_gate_stays_inconclusive_when_the_interval_excludes_zero(self):
        frozen = {str(i): 0.90 for i in range(3)}
        arm = {str(i): 0.89 for i in range(3)}
        report = l6_vs_frozen_v1(arm, frozen)
        self.assertEqual(report["label"], "INCONCLUSIVE")
        self.assertLess(report["ci_high"], 0)
        self.assertFalse(report["improvement_claim"])
        self.assertFalse(report["l6_is_program_quality"])
        self.assertFalse(report["qualifying"])

    def test_l6_gate_names_lower_and_higher_without_a_claim(self):
        frozen = {str(i): 0.90 for i in range(881)}
        lower = l6_vs_frozen_v1({str(i): 0.89 for i in range(881)}, frozen)
        higher = l6_vs_frozen_v1({str(i): 0.91 for i in range(881)}, frozen)
        flat = l6_vs_frozen_v1(dict(frozen), frozen)
        self.assertEqual(lower["label"], "LOWER")
        self.assertEqual(higher["label"], "HIGHER")
        self.assertEqual(flat["label"], "INCONCLUSIVE")
        self.assertTrue(lower["above_v1_is_worse"])
        for report in (lower, higher, flat):
            self.assertFalse(report["improvement_claim"])
            self.assertFalse(report["promotion"])
            self.assertFalse(report["seal_spent"])

    def test_cluster_interval_does_not_decide_the_l6_label(self):
        frozen = {"a": 2.0, "b": 2.0, "c": 2.0, "d": 2.0}
        arm = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
        groups = [[-1.0, -1.0], [-1.0, -1.0]]
        report = l6_vs_frozen_v1(arm, frozen, groups)
        self.assertEqual(report["label"], "INCONCLUSIVE")
        self.assertLess(report["cluster_high"], 0)
        self.assertFalse(report["cluster_decides"])

    def test_l6_refuses_unmatched_contexts(self):
        with self.assertRaises(ValueError):
            l6_vs_frozen_v1({"a": 0.1, "b": 0.1}, {"a": 0.1, "c": 0.1})

    def test_automated_report_keeps_l6_out_of_the_delivered_board(self):
        frozen = {str(i): 0.90 for i in range(4)}
        arm = {str(i): 0.89 for i in range(4)}
        report = automated_report(
            _proposal_rows(), _search_rows(), _delivered_rows(), arm, frozen
        )
        self.assertEqual(report["delivered"]["selection"], "training_side_max_train_q")
        self.assertTrue(report["l6_does_not_substitute_for_delivered"])
        self.assertFalse(report["l6_vs_frozen_v1"]["l6_is_program_quality"])
        self.assertFalse(report["improvement_claim"])
        self.assertFalse(report["seal_spent"])
