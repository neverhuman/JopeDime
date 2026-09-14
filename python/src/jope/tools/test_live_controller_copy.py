"""Plan helpers against a copy of the live hold. Does not restart 779dacc4."""
import unittest

import flywheel_plan as f
import selfplay_milestones as m


LIVE_COPY = dict(
    campaign="automatic-training-hold-v1",
    manifest_prefix="779dacc4",
    hold="held",
    serving="c2f229c9",
    gp_pid=774452,
    events=[{"event": "diagnostic_rejected"}] * 3,
    packing={"rebuilt_groups": 8},
    first_label_growth=0,
    frozen_bt=0.542,
)


class LiveControllerCopyTests(unittest.TestCase):
    def test_helpers_agree_with_held_recovery_copy(self):
        copy = dict(LIVE_COPY)
        self.assertTrue(f.consecutive_diagnostic_hold(copy["events"]))
        self.assertFalse(f.teachers_improved(copy["packing"], copy["first_label_growth"]))
        self.assertFalse(f.search_beating_training_allowed(copy["frozen_bt"]))
        self.assertFalse(f.may_serve(False, False, False, False))
        self.assertFalse(m.existence_proof({"neural_seed_rate": 0.0}))
        self.assertEqual(copy["hold"], "held")
        self.assertEqual(copy["gp_pid"], 774452)
        self.assertEqual(copy["manifest_prefix"], "779dacc4")


if __name__ == "__main__":
    unittest.main()
