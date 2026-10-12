"""Lane B power table. Hypothetical rows are not a measured judge sigma."""

import json
import math
import unittest
from pathlib import Path

from rlfw.judge.power import (
    SPEC_POWER_FACTOR,
    UNMEASURED_JUDGE_NOISE,
    comparison_power,
    families_needed,
    minimum_detectable_effect,
    publish_document,
    tier_for_counts,
)

TABLE = Path(__file__).resolve().parents[1] / "POWER_TABLE.json"


class PowerFormulaTest(unittest.TestCase):
    def test_spec_factor_matches_the_published_examples(self):
        self.assertEqual(SPEC_POWER_FACTOR, 7.84)
        self.assertTrue(math.isclose(families_needed(1.0, 0.2), 196.0))
        self.assertTrue(math.isclose(families_needed(1.0, 0.1), 784.0))
        detectable = minimum_detectable_effect(54)
        self.assertTrue(math.isclose(detectable, math.sqrt(7.84 / 54)))
        self.assertAlmostEqual(detectable, 0.381, places=3)
        power = comparison_power(196, 1.0, 0.2)
        self.assertGreater(power, 0.79)
        self.assertLess(power, 0.81)

    def test_published_table_matches_the_generator_and_leaves_noise_unmeasured(self):
        published = json.loads(TABLE.read_text())
        self.assertEqual(published, json.loads(json.dumps(publish_document())))
        unmeasured = published["rows"][-1]
        self.assertIsNone(unmeasured["sigma_D"])
        self.assertEqual(unmeasured["status"], "unmeasured")
        self.assertEqual(unmeasured["name"], UNMEASURED_JUDGE_NOISE["name"])
        self.assertFalse(published["measured_judge_sigma"])
        for row in published["rows"][:-1]:
            self.assertFalse(row["measured"])
            self.assertEqual(row["sigma_source"], "hypothetical_unit_sigma")

    def test_tier_counts_are_the_spec_floors(self):
        self.assertEqual(tier_for_counts(9, 20), "below_smoke")
        self.assertEqual(tier_for_counts(10, 20), "smoke_mechanics_only")
        self.assertEqual(tier_for_counts(50, 100), "week1_directional")
        self.assertEqual(tier_for_counts(100, 200), "internal_promotion")
        self.assertEqual(tier_for_counts(200, 399), "internal_promotion")
        self.assertEqual(tier_for_counts(200, 400), "sealed_count_floor")


if __name__ == "__main__":
    unittest.main()
