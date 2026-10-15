import hashlib
import itertools
import math
import unittest
from dataclasses import replace
from statistics import fmean, pvariance

from rlfw.multiseed.analysis import (
    Candidate,
    Context,
    Observation,
    Selection,
    Study,
    _replicate,
    multiseed_report,
    optimism_bound,
    score_selected_teacher,
)


def sha(name):
    return hashlib.sha256(name.encode()).hexdigest()


def study(contexts=None):
    return Study(
        (11, 13, 17),
        contexts or (Context("a", "parent-a", sha("a")), Context("b", "parent-b", sha("b"))),
        ("parent-train",),
        sha("frozen-v1"),
        sha("v1-training-recipe"),
        100,
    )


def panel(plan):
    return [
        Observation(
            s,
            c.context_id,
            c.comparison_sha256,
            sha(f"model-{s}"),
            sha(f"train-{s}"),
            sha(f"measure-{s}-{c.context_id}"),
            plan.updates,
            1 + i / 10 - 0.2,
            1.0,
        )
        for i, s in enumerate(plan.seeds)
        for c in plan.contexts
    ]


class MultiSeedTests(unittest.TestCase):
    def test_all_seeds_are_primary_and_lower_loss_is_better(self):
        plan = study()
        report = multiseed_report(plan, panel(plan), draws=1000)
        self.assertAlmostEqual(report["mean_delta_v1"], -0.1)
        self.assertAlmostEqual(report["between_seed_sample_sd"], 0.1)
        self.assertAlmostEqual(report["mean_l6_cost"], 0.9)
        self.assertEqual(report["best_seed_descriptive"], 11)
        self.assertEqual((report["seeds"], report["parents"], report["contexts"]), (3, 2, 2))
        self.assertFalse(report["qualified"])
        self.assertIsNone(report["claim"])
        self.assertFalse(report["training_authorized"])

    def test_crossed_replication_preserves_seed_effect_across_parents(self):
        # Exact independent enumeration: variance of a mean of 3 seed draws is
        # population seed variance /3, not /6 pseudoreplicated cells.
        sums = ((-0.2, -0.2), (-0.1, -0.1), (0.0, 0.0))
        values = [
            _replicate(sums, (1, 1), ss, pp)
            for ss in itertools.product(range(3), repeat=3)
            for pp in itertools.product(range(2), repeat=2)
        ]
        self.assertAlmostEqual(fmean(values), -0.1)
        self.assertAlmostEqual(pvariance(values), pvariance((-0.2, -0.1, 0.0)) / 3)

    def test_context_macro_and_parent_sensitivity_are_separate(self):
        plan = study(
            (Context("a1", "a", sha("a1")), Context("a2", "a", sha("a2")), Context("b", "b", sha("b")))
        )
        rows = [replace(r, l6_cost=0.9 if r.context_id.startswith("a") else 1.2) for r in panel(plan)]
        report = multiseed_report(plan, rows, draws=100)
        self.assertAlmostEqual(report["mean_delta_v1"], 0.0)
        self.assertAlmostEqual(report["parent_balanced_delta_sensitivity"], 0.05)
        self.assertEqual(report["parents"], 2)

    def test_bootstrap_is_deterministic_and_input_order_does_not_change_results(self):
        plan = study()
        rows = panel(plan)
        self.assertEqual(
            multiseed_report(plan, rows, draws=200), multiseed_report(plan, rows[::-1], draws=200)
        )

    def test_incomplete_duplicate_unknown_and_extra_rows_are_refused(self):
        plan = study()
        rows = panel(plan)
        for bad in (
            [],
            rows[:-1],
            rows + [rows[0]],
            rows[:-1] + [replace(rows[-1], seed=99)],
            rows[:-1] + [replace(rows[-1], context_id="unknown")],
        ):
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                multiseed_report(plan, bad, draws=100)

    def test_comparison_update_model_and_reference_drift_are_refused(self):
        plan = study()
        rows = panel(plan)
        for change in (
            {"comparison_sha256": sha("changed")},
            {"updates": 99},
            {"checkpoint_sha256": sha("other-model")},
            {"training_receipt_sha256": sha("other-run")},
            {"v1_l6_cost": 1.1},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                multiseed_report(plan, [replace(rows[0], **change)] + rows[1:], draws=100)

    def test_relabelling_reference_or_reusing_one_seed_does_not_make_multiple_seeds(self):
        plan = study()
        for checkpoints, receipts in (
            (sha("model-11"), sha("train-11")),
            (plan.v1_checkpoint_sha256, sha("train-13")),
        ):
            rows = [
                replace(r, checkpoint_sha256=checkpoints, training_receipt_sha256=receipts)
                if r.seed == 13
                else r
                for r in panel(plan)
            ]
            with self.assertRaises(ValueError):
                multiseed_report(plan, rows, draws=100)

    def test_seed_cardinality_roles_and_roster_are_checked_at_public_boundary(self):
        plan = study()
        for change in (
            {"seeds": (11,)},
            {"seeds": (11, 11, 13)},
            {"seeds": (True, 13, 17)},
            {"contexts": plan.contexts * 2},
            {"contexts": (plan.contexts[0],)},
            {"train_parent_ids": ("parent-a",)},
            {"updates": 0},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(plan, **change)

    def test_nonfinite_bool_negative_cost_and_bad_hashes_are_refused(self):
        row = panel(study())[0]
        for value in (math.nan, math.inf, -1.0, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(row, l6_cost=value)
        for value in ("ea078456", "g" * 64, "A" * 64):
            with self.assertRaises(ValueError):
                replace(row, checkpoint_sha256=value)

    def test_bootstrap_controls_refuse_invalid_or_unbounded_work(self):
        plan = study()
        for draws in (True, 0, 99, 100_001):
            with self.assertRaises(ValueError):
                multiseed_report(plan, panel(plan), draws=draws)
        with self.assertRaises(ValueError):
            multiseed_report(plan, panel(plan), draws=100, bootstrap_seed=-1)

    def test_planning_never_authorizes_training_or_fills_unmeasured_power(self):
        plan = study().plan()
        self.assertFalse(plan["training_authorized"])
        self.assertEqual(plan["power"], "UNMEASURED")
        self.assertFalse(plan["qualified"])
        self.assertNotIn("command", plan)


class TeacherTests(unittest.TestCase):
    def selection(self, values=(0.1, 0.2)):
        return Selection(
            sha("comparison"),
            tuple(Candidate(sha(f"program-{i}"), v, sha(f"inner-{i}")) for i, v in enumerate(values)),
            ("fit-0",),
            ("fit-0", "select-1"),
        )

    def score(self, selection, **changes):
        args = dict(
            commitment_sha256=selection.sha256,
            comparison_sha256=selection.comparison_sha256,
            program_sha256=selection.winner.program_sha256,
            fit_row_ids=selection.fit_row_ids,
            judge_row_ids=("judge-2", "judge-3"),
            judge_q=-0.1,
            incumbent_judge_q=0.0,
            measurement_sha256=sha("judge"),
            incumbent_measurement_sha256=sha("incumbent-judge"),
        )
        args.update(changes)
        return score_selected_teacher(selection, **args)

    def test_exact_null_toy_shows_winner_optimism_and_heldout_correction(self):
        # Independent +/-0.1 score noise, true utility zero for both programs.
        # E[max(inner1,inner2)]=0.05; independent heldout score stays at zero.
        reports = [
            self.score(self.selection(inner), judge_q=outer)
            for inner in itertools.product((-0.1, 0.1), repeat=2)
            for outer in (-0.1, 0.1)
        ]
        self.assertAlmostEqual(fmean(r["inner_q"] for r in reports), 0.05)
        self.assertAlmostEqual(fmean(r["heldout_q"] for r in reports), 0.0)
        self.assertAlmostEqual(fmean(r["observed_optimism_gap"] for r in reports), 0.05)
        self.assertTrue(all(not r["qualified_reward"] and not r["inner_q_is_reward"] for r in reports))

    def test_large_inner_winner_can_lose_on_independent_judge_rows(self):
        result = self.score(self.selection((1.0, 2.0)), judge_q=-0.7)
        self.assertEqual(result["heldout_advantage"], -0.7)
        self.assertEqual(result["observed_optimism_gap"], 2.7)
        self.assertIsNone(result["claim"])

    def test_judge_rows_cannot_overlap_any_fit_or_selection_exposure(self):
        selection = self.selection()
        for rows in (("fit-0",), ("select-1",), ("judge-2", "fit-0")):
            with self.assertRaises(ValueError):
                self.score(selection, judge_row_ids=rows)

    def test_post_selection_program_comparison_commitment_or_fit_change_is_refused(self):
        selection = self.selection()
        for change in (
            {"program_sha256": sha("other")},
            {"comparison_sha256": sha("other")},
            {"commitment_sha256": sha("other")},
            {"fit_row_ids": ("judge-2",)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.score(selection, **change)

    def test_fitting_exposure_duplicate_programs_mutable_rows_and_bad_q_are_refused(self):
        selection = self.selection()
        for change in (
            {"fit_row_ids": ("unregistered-fit",)},
            {"candidates": selection.candidates * 2},
            {"selection_row_ids": ["fit-0"]},
            {"selection_row_ids": ("fit-0", "fit-0")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(selection, **change)
        with self.assertRaises(ValueError):
            Candidate(sha("program"), math.nan, sha("score"))
        with self.assertRaises(ValueError):
            self.score(selection, judge_q=math.inf)

    def test_ties_choose_stable_program_hash_without_judge_feedback(self):
        selection = self.selection((0.1, 0.1))
        self.assertEqual(selection.winner.program_sha256, min(c.program_sha256 for c in selection.candidates))
        self.assertEqual(selection.winner, replace(selection, candidates=selection.candidates[::-1]).winner)

    def test_subgaussian_bound_is_an_assumption_not_a_numeric_reward_correction(self):
        self.assertEqual(optimism_bound(100)["status"], "UNMEASURED")
        self.assertIsNone(optimism_bound(100)["bound"])
        self.assertEqual(optimism_bound(1, sigma=0.1)["bound"], 0.0)
        self.assertGreater(optimism_bound(100, sigma=0.1)["bound"], optimism_bound(2, sigma=0.1)["bound"])
        self.assertIsNone(optimism_bound(100, sigma=0.1)["corrected_reward"])
        for count, noise in ((0, 0.1), (True, 0.1), (2, -1.0), (2, math.nan)):
            with self.assertRaises(ValueError):
                optimism_bound(count, sigma=noise)


if __name__ == "__main__":
    unittest.main()
