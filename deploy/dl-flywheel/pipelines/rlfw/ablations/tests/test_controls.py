"""J8 control fixtures. No checkpoints, receipts, or program execution."""

import unittest
from copy import deepcopy

from rlfw.ablations.controls import (
    ARM_NAMES,
    CLOSED_SCRATCH,
    FROZEN_V1_FULL_VAL_L6,
    classify_arm,
    direction,
    load_arm,
    require_closed_scratch,
    require_readout,
)


def _arm(name, **extra):
    spec = load_arm(name)
    spec.update(extra)
    return spec


class J8ControlTests(unittest.TestCase):
    def test_each_declared_config_is_its_own_intervention(self):
        self.assertEqual(ARM_NAMES, (
            "frozen", "restart_only", "offline_replay", "data_volume", "fresh_feedback",
        ))
        for name in ARM_NAMES:
            self.assertEqual(classify_arm(load_arm(name)), name)
            self.assertIs(load_arm(name)["launch_allowed"], False)
            self.assertIs(load_arm(name)["trie_arm"], False)

    def test_restart_only_refuses_any_feedback_weight(self):
        for key, value in (("lambda_pi", 0.01), ("lambda_kl", 0.2), ("lambda_knn", 0.25)):
            with self.assertRaises(ValueError):
                classify_arm(_arm("restart_only", **{key: value}))
        with self.assertRaises(ValueError):
            classify_arm(_arm("restart_only", lambda_pi=[0.0, 0.01]))
        with self.assertRaises(ValueError):
            classify_arm(_arm("restart_only", max_updates=2621))
        with self.assertRaises(ValueError):
            classify_arm(_arm("restart_only", new_feedback=True))

    def test_offline_replay_is_not_a_trie_or_volume_arm(self):
        with self.assertRaises(ValueError):
            classify_arm(_arm("offline_replay", lambda_kl=0.2))
        with self.assertRaises(ValueError):
            classify_arm(_arm("offline_replay", lambda_knn=0.25))
        with self.assertRaises(ValueError):
            classify_arm(_arm("offline_replay", trie_arm=True))
        with self.assertRaises(ValueError):
            classify_arm(_arm(
                "offline_replay",
                observation_count_matched=True,
                intervention="offline_replay",
            ))
        self.assertEqual(classify_arm(load_arm("data_volume")), "data_volume")
        self.assertNotEqual(classify_arm(load_arm("data_volume")), "offline_replay")

    def test_fresh_feedback_stays_unlaunched(self):
        with self.assertRaises(ValueError):
            classify_arm(_arm("fresh_feedback", launch_allowed=True))
        mixed = _arm("fresh_feedback", observation_count_matched=True, preexisting_observations=True)
        self.assertEqual(classify_arm(mixed), "fresh_feedback")

    def test_frozen_reference_tie_and_near_boundary(self):
        pinned = 0.8973272357035962
        self.assertEqual(FROZEN_V1_FULL_VAL_L6, pinned)
        self.assertEqual(direction(pinned), "tied_not_a_claim")
        below_the_pin = 0.8973271178517981
        self.assertLess(below_the_pin, pinned)
        self.assertGreater(below_the_pin, 0.897327)
        self.assertEqual(direction(below_the_pin), "below_v1_not_a_claim")
        self.assertEqual(direction(0.9022044512849824), "worse")
        self.assertEqual(direction(0.9003862184454733), "worse")

    def test_closed_scratch_is_above_v1_and_not_an_improvement(self):
        for name in ("restart_only", "offline_replay"):
            row = require_closed_scratch(name)
            self.assertEqual(direction(row["full_val_l6"]), "worse")
            self.assertGreater(row["full_val_l6"], FROZEN_V1_FULL_VAL_L6)
            self.assertIs(row["non_qualifying"], True)
            self.assertIs(row["benchmark_valid"], False)
            claimed = deepcopy(row)
            claimed["improvement"] = True
            with self.assertRaises(ValueError):
                require_readout(claimed)
        earlier = CLOSED_SCRATCH["restart_only"]["earlier_epoch_full_val_l6"]
        self.assertEqual(direction(earlier), "below_v1_not_a_claim")
        self.assertNotEqual(earlier, CLOSED_SCRATCH["restart_only"]["full_val_l6"])
        self.assertEqual(CLOSED_SCRATCH["restart_only"]["updates"], 704)
        self.assertEqual(CLOSED_SCRATCH["restart_only"]["stop_reason"], "max_epochs")
        self.assertEqual(CLOSED_SCRATCH["restart_only"]["harness_arm"], "control")
        self.assertEqual(CLOSED_SCRATCH["offline_replay"]["updates"], 352)
        self.assertEqual(
            CLOSED_SCRATCH["offline_replay"]["stop_reason"],
            "kl_disqualify_all_treatment_arms",
        )
        self.assertEqual(load_arm("restart_only")["harness_arm"], "control")
        self.assertEqual(load_arm("offline_replay")["lambda_pi"], 0.01)
        self.assertEqual(load_arm("offline_replay")["lambda_kl"], 0.0)


if __name__ == "__main__":
    unittest.main()
