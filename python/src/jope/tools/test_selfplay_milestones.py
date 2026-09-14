import unittest

import selfplay_milestones as m


class SelfPlayMilestoneTests(unittest.TestCase):
    def test_search_beating_waits_for_stable_bt(self):
        self.assertFalse(m.search_beating_allowed(0.55))
        self.assertTrue(m.search_beating_allowed(m.RECOVERY_BT))
        self.assertTrue(m.search_beating_allowed(m.RECOVERY_BT * 1.01))

    def test_may_serve_requires_all_gates(self):
        ok, _ = m.may_serve(True, True, True, True)
        self.assertFalse(ok)
        self.assertIn("original promotion", " ".join(m.may_serve(True, True, True, True)[1]))
        blocked, reasons = m.may_serve(True, True, True, False)
        self.assertFalse(blocked)
        self.assertTrue(any("quality-report" in r for r in reasons))
        self.assertFalse(m.may_serve(True, True, False, True)[0])
        self.assertFalse(m.may_serve(False, True, True, True)[0])

    def test_diagnostic_matches_frozen_guards(self):
        ok, _ = m.frozen_diagnostic_pass(m.RECOVERY_NLL, m.RECOVERY_BT)
        self.assertTrue(ok)
        failed, reasons = m.frozen_diagnostic_pass(1.4086, 0.542)
        self.assertFalse(failed)
        self.assertIn("diagnostic regression: sequence_nll", reasons)
        self.assertIn("diagnostic regression: ranking_bt", reasons)

    def test_existence_proof_and_world_class_bars(self):
        zero = m.win_rates([{"best_retained_teacher": 1.0, "best_initial_neural_seed": 1.14,
                             "best_final_gp": 1.0}] * 20)
        self.assertEqual(zero["neural_seed_rate"], 0.0)
        self.assertFalse(m.existence_proof(zero))
        self.assertFalse(m.world_class(zero))
        winning = m.win_rates(
            [{"best_retained_teacher": 1.0, "best_initial_neural_seed": 0.9, "best_final_gp": 0.8}] * 20)
        self.assertFalse(m.existence_proof(winning))
        self.assertFalse(m.world_class(winning))
        self.assertTrue(m.descriptive_bars(winning))
        self.assertEqual(winning["neural_seed_n"], 1)
        self.assertEqual(winning["duplicate_rows"], 19)

    def test_nonfinite_boolean_negative_metrics_never_pass(self):
        for value in (float("nan"), float("inf"), -1, True, False, "1"):
            self.assertFalse(m.frozen_diagnostic_pass(value, m.RECOVERY_BT)[0])
            self.assertFalse(m.frozen_diagnostic_pass(m.RECOVERY_NLL, value)[0])
            self.assertFalse(m.ranking_bt_stable(value))
            rates = m.win_rates([dict(best_retained_teacher=1., best_initial_neural_seed=value,
                                     best_final_gp=value)])
            self.assertEqual(rates["neural_seed_n"], 0)
            self.assertFalse(m.world_class(rates))

    def test_specialist_loop_rejects_mixed_corpus_and_refits(self):
        self.assertFalse(m.specialist_loop("binary", True, 0, mixed_corpus=True)[0])
        self.assertFalse(m.specialist_loop("binary", True, 0.1, mixed_corpus=False)[0])
        self.assertTrue(m.specialist_loop("binary", True, 0, mixed_corpus=False)[0])


if __name__ == "__main__":
    unittest.main()
