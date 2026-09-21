import json
import tempfile
import unittest
from pathlib import Path

import expert_iteration_setup as s


class ExpertIterationSetupTests(unittest.TestCase):
    def test_prefix_search_is_unqualified_not_impossible(self):
        self.assertTrue(s.refuse_alphago_zero())
        self.assertFalse(s.AGZ["applicable"])
        self.assertTrue(s.EI["applicable"])
        self.assertEqual(s.EI["value_domain"], "complete_program_rmse_only")
        self.assertEqual(s.AGZ["improvement"], "mcts_visit_counts")
        self.assertEqual(s.EI["expert"], "fixed_budget_gp")
        self.assertEqual(s.EI["legacy_expert_budget"]["evaluations"], 1000)
        self.assertFalse(s.AGZ["mcts_intrinsically_invalid"])

    def test_pairwise_bt_does_not_claim_outcome_value_equivalence(self):
        self.assertFalse(s.pairwise_is_the_value_target())
        self.assertFalse(s.pairwise_is_the_value_head())
        self.assertFalse(s.strategy()["critic_shares_decoder_trunk"])
        self.assertEqual(s.LOSS["ranking_backward"], "dense_pairwise_v2")
        self.assertEqual(s.LOSS["ranking_bt"], 0.25)
        self.assertEqual(s.LOSS["ranking_listwise"], 0.1)
        self.assertEqual(s.LOSS["sequence_nll"], 1.0)
        self.assertEqual(s.LOSS["stability"], 0.0)
        self.assertTrue(s.LOSS["teacher_in_each_comparison"])
        self.assertFalse(s.EI["optimality_proof"])
        self.assertFalse(s.AGZ["mcts_intrinsically_invalid"])
        self.assertFalse(s.AGZ["mcts_now"])

    def test_measured_rates_are_not_a_flywheel(self):
        self.assertTrue(s.current_rates_are_not_flywheel())
        self.assertEqual(s.MEASURED["neural_seed_beats"], 0)
        self.assertEqual(s.MEASURED["final_gp_beats"], 2)
        self.assertFalse(s.strategy()["flywheel_effect"])
        self.assertFalse(s.strategy()["flywheel_getting_smarter"])

    def test_fake_upgrades_are_refused(self):
        for name in (
            "unqualified_mcts_gain_claim",
            "unqualified_width_gain_claim",
            "unqualified_loss_gain_claim",
            "occupancy_as_intelligence",
            "2_of_140_as_world_class",
        ):
            self.assertTrue(s.refuse_fake_upgrade(name))
        self.assertFalse(s.refuse_fake_upgrade("new_identity_funded_train"))

    def history(self):
        return [
            dict(
                neural_seed_rate=seed,
                gp_improves_on_seed=repair,
                parent=str(i) * 64,
                corpus=str(i + 3) * 64,
                gp=str(i + 6) * 64,
            )
            for i, seed, repair in ((1, 0.02, 0.9), (2, 0.11, 0.5))
        ]

    def test_matching_rate_trends_are_diagnostics_never_certification(self):
        history = self.history()
        for row in history:
            row.update(independent_evaluation=True, p1_accepted=True, p2_accepted=True)
        diagnostic = s.rate_series_diagnostic(history)
        self.assertEqual(diagnostic["status"], "diagnostic_only")
        self.assertTrue(diagnostic["seed_rate_strictly_increasing"])
        self.assertTrue(diagnostic["repair_rate_strictly_decreasing"])
        self.assertFalse(diagnostic["identity_labels_verified"])
        self.assertFalse(diagnostic["qualified_flywheel"])
        self.assertFalse(s.flywheel_getting_smarter(history))

    def test_malformed_rates_and_missing_lineage_are_unavailable(self):
        for key in ("neural_seed_rate", "gp_improves_on_seed"):
            for value in (
                None,
                True,
                False,
                float("nan"),
                float("inf"),
                -float("inf"),
                -0.1,
                1.1,
                10**400,
                "0.9",
            ):
                with self.subTest(key=key, value=str(value)):
                    history = self.history()
                    history[0][key] = value
                    self.assertEqual(s.rate_series_diagnostic(history)["status"], "unavailable")
                    self.assertFalse(s.flywheel_getting_smarter(history))
        for key in ("parent", "corpus", "gp"):
            history = self.history()
            del history[0][key]
            self.assertEqual(s.rate_series_diagnostic(history)["status"], "unavailable")
        for history in (None, {}, [], [None, None], [self.history()[0]]):
            self.assertEqual(s.rate_series_diagnostic(history)["status"], "unavailable")
            self.assertFalse(s.flywheel_getting_smarter(history))

    def test_recovery_parent_and_equal_rates_are_reported_without_learning_claim(self):
        history = self.history()
        history[1] = dict(history[0], parent=s.RECOVERY_PARENT)
        diagnostic = s.rate_series_diagnostic(history)
        self.assertEqual(diagnostic["status"], "diagnostic_only")
        self.assertFalse(diagnostic["seed_rate_strictly_increasing"])
        self.assertFalse(s.flywheel_getting_smarter(history))

    def test_original_new_step_window_is_not_legacy_block_length(self):
        self.assertEqual(s.strategy()["arch"]["new_steps"], 1000)
        self.assertEqual(s.strategy()["arch"]["task_cap_seconds"], 300)
        self.assertEqual(s.strategy()["arch"]["optimizer"], "carry")

    def test_claude_example_ratio_is_formula_only(self):
        row = s.claude_example_pair_weight_ratio()
        self.assertEqual(row["scope"], "formula_only")
        self.assertFalse(row["science_agreed_scan"])
        self.assertGreater(row["ratio"], 200)
        self.assertLess(row["ratio"], 230)

    def test_emit_does_not_claim_launch_or_serve(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = s.emit(Path(folder))
            self.assertFalse(summary["flywheel_effect"])
            self.assertFalse(summary["launch_train_allowed"])
            self.assertTrue(summary["refuse_alphago_zero"])
            self.assertFalse(summary["pairwise_is_the_value_head"])
            body = json.loads((Path(folder) / "STRATEGY.json").read_text())
            self.assertIn("Anthony, Tian, Barber 2017", json.dumps(body))
            self.assertFalse(body["science_agreed_218x_scan"])
            self.assertFalse(body["one_hundred_eight_blocks_one_parent"])
            self.assertFalse(body["flywheel_effect"])

    def test_emit_preserves_existing_diagnostic_histories(self):
        with tempfile.TemporaryDirectory() as folder:
            s.emit(folder)
            paths = [Path(folder) / name for name in ("STRATEGY.json", "SUMMARY.json")]
            original = [path.read_bytes() for path in paths]
            with self.assertRaises(FileExistsError):
                s.emit(folder)
            self.assertEqual([path.read_bytes() for path in paths], original)
        with tempfile.TemporaryDirectory() as folder:
            summary = Path(folder) / "SUMMARY.json"
            summary.write_text("retained-original\n")
            with self.assertRaises(FileExistsError):
                s.emit(folder)
            self.assertEqual(summary.read_text(), "retained-original\n")
            self.assertFalse((Path(folder) / "STRATEGY.json").exists())


if __name__ == "__main__":
    unittest.main()
