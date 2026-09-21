import tempfile
import unittest
from pathlib import Path

import flywheel_plan as f
from resume_expired_cycle import atomic


class FlywheelPlanTests(unittest.TestCase):
    def test_validation_best_is_not_latest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "train"
            output.mkdir()
            atomic(output / "latest.json", dict(checkpoint="checkpoints/step-3000", state_sha256="late"))
            (output / "metrics.jsonl").write_text(
                '{"validation":{"sequence_nll":1.40},"checkpoint":"checkpoints/step-1000"}\n'
                '{"validation":{"sequence_nll":1.38},"checkpoint":"checkpoints/step-2000"}\n'
                '{"validation":{"sequence_nll":1.39},"checkpoint":"checkpoints/step-3000"}\n'
            )
            (root / "checkpoints/step-2000").mkdir(parents=True)
            atomic(root / "checkpoints/step-2000/state.json", {})
            best = f.validation_best(root, output)
            self.assertEqual(best["checkpoint"], "checkpoints/step-2000")

    def test_teachers_improved_needs_new_first_label_programs(self):
        self.assertFalse(f.teachers_improved({"rebuilt_groups": 0}, 0))
        self.assertFalse(f.teachers_improved({"rebuilt_groups": 0}, None))
        self.assertFalse(f.teachers_improved({"rebuilt_groups": 7}, 0))
        self.assertTrue(f.teachers_improved({"rebuilt_groups": 0}, 3))

    def test_consecutive_diagnostic_hold_trips_after_three_rejects(self):
        self.assertFalse(f.consecutive_diagnostic_hold(["diagnostic_rejected"] * 2))
        self.assertTrue(f.consecutive_diagnostic_hold(["diagnostic_rejected"] * 3))
        self.assertFalse(
            f.consecutive_diagnostic_hold(
                ["diagnostic_rejected"] * 5 + ["diagnostic_accepted"] + ["diagnostic_rejected"] * 2
            )
        )
        self.assertTrue(f.consecutive_diagnostic_hold([{"event": "diagnostic_rejected"}] * 3))

    def test_search_beating_and_serve_gates(self):
        self.assertFalse(f.search_beating_training_allowed(0.55))
        self.assertTrue(f.search_beating_training_allowed(f.RECOVERY_FROZEN["ranking_bt"]))
        self.assertFalse(f.may_serve(True, True, True, False))
        self.assertTrue(f.may_serve(True, True, True, True))
        self.assertFalse(f.may_serve(True, True, False, True))
        self.assertEqual(
            f.neural_seed_win_rate(
                [
                    dict(best_retained_teacher=1.0, best_initial_neural_seed=1.1),
                    dict(best_retained_teacher=1.0, best_initial_neural_seed=0.9),
                ]
            ),
            0.5,
        )

    def test_enrich_does_not_treat_ranking_score_as_logprob(self):
        population = {
            "candidates": [
                dict(genome="(pipe incumbent)", origin="incumbent_cut_teacher"),
                dict(genome="(pipe neural)", origin="neural_policy"),
                dict(genome="(pipe rand1)", origin="random_exploration"),
            ]
        }
        out = f.enrich_seed_population(
            population, ["(pipe (add x y))"], [dict(program="(pipe neural)", score=-0.2)]
        )
        self.assertNotIn("policy_logprob", out["candidates"][1])
        self.assertIn("same_family_cross_cut_teacher", [row["origin"] for row in out["candidates"]])


if __name__ == "__main__":
    unittest.main()
