import unittest

import closed_loop as c


class ClosedLoopTests(unittest.TestCase):
    def test_hold_blocks_recovery_train(self):
        ok, reason = c.launch_train_allowed("held", True, True, 3, corpus_is_recovery=True)
        self.assertFalse(ok)
        self.assertIn("native funded task", reason)
        ok, reason = c.launch_train_allowed("open", True, True, 3, corpus_is_recovery=True)
        self.assertFalse(ok)
        self.assertIn("native funded task", reason)

    def test_two_panels_required(self):
        ok, _ = c.two_panels(c.RECOVERY_NLL, c.RECOVERY_BT, True)
        self.assertTrue(ok)
        self.assertFalse(c.two_panels(1.41, 0.55, True)[0])
        self.assertFalse(c.two_panels(c.RECOVERY_NLL, c.RECOVERY_BT, False)[0])

    def test_existence_diagnostic_cannot_authorize_serving(self):
        ok, reason = c.serve_decision(True, True, True, True, 0.0)
        self.assertFalse(ok)
        self.assertIn("original promotion evidence", reason)
        self.assertFalse(c.serve_decision(True, True, True, True, 1.0)[0])

    def test_binary_cannot_bind_recovery(self):
        with self.assertRaisesRegex(ValueError, "recovery"):
            c.binary_campaign_bind("inv", c.RECOVERY_SHA)
        bind = c.binary_campaign_bind("inv", "ab" * 32)
        self.assertFalse(bind["wired_to_live_gp"])
        self.assertEqual(bind["campaign"]["task"], "binary")

    def test_loop_rejects_refits(self):
        ok, _ = c.loop_status(0.0, False, True)
        self.assertTrue(ok)
        self.assertFalse(c.loop_status(0.2, False, True)[0])

    def test_stage_d_does_not_invent_existence(self):
        snap = c.from_stage_d({
            "predicted_first_neural_seed_vs_teacher": {"n": 140, "beats_teacher": 0},
            "source": "generation-000048",
        })
        self.assertEqual(snap["neural_seed_rate"], 0.0)
        self.assertFalse(snap["existence_proof"])
        self.assertFalse(snap["may_serve"])
        self.assertIn("Claude new-index-only canary", snap["awaiting_agent_acks"][1])

    def test_open_recovery_still_refuses_without_new_identity(self):
        ok, reason = c.launch_train_allowed("open", True, True, 3, corpus_is_recovery=False)
        self.assertFalse(ok)
        self.assertIn("native funded task", reason)

    def test_grown_family_train_never_launches(self):
        gate = c.grown_family_train(
            c.RECOVERY_NLL, c.RECOVERY_BT, True, True, True, True, True, 0.05,
            hold="open", automatic_training=True, teachers_growth=True, new_teacher_beats=1,
            corpus_is_recovery=False, new_gp_identity=True, admission_v2=True,
            reuse_excluded_from_growth=True, historical_import_complete=True,
        )
        self.assertFalse(gate["train_launched"])
        self.assertTrue(gate["two_panels"])
        self.assertFalse(gate["launch_allowed"])
        self.assertFalse(gate["may_serve"])
        blocked = c.grown_family_train(
            1.41, 0.55, True, False, False, False, False, 0.0,
            hold="held", automatic_training=False, teachers_growth=False, new_teacher_beats=0,
            corpus_is_recovery=True,
        )
        self.assertFalse(blocked["train_launched"])
        self.assertFalse(blocked["launch_allowed"])
        self.assertFalse(blocked["may_serve"])

    def test_az2017_cycle_refuses_and_lists_missing_conjuncts(self):
        cycle = c.az2017_cycle(
            c.RECOVERY_NLL, c.RECOVERY_BT, True, False, False, False, False, 0.0,
            hold="held", automatic_training=False, teachers_growth=False, new_teacher_beats=0,
            corpus_is_recovery=True,
        )
        self.assertFalse(cycle["train_launched"])
        self.assertFalse(cycle["may_serve"])
        self.assertFalse(cycle["claimed_savings_745"])
        self.assertFalse(cycle["world_class"])
        self.assertIn("new_gp_identity", cycle["missing_conjuncts"])
        self.assertIn("existence_5pct", cycle["missing_conjuncts"])


if __name__ == "__main__":
    unittest.main()
