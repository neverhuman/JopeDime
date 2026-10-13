"""Lane B power table. Hypothetical rows are not a measured judge sigma."""

import json
import math
import unittest
from pathlib import Path

from rlfw.judge.power import (
    OWNER_ROUNDED_SE_PAIR,
    SPEC_POWER_FACTOR,
    UNMEASURED_JUDGE_NOISE,
    comparison_power,
    families_needed,
    interpret_noise,
    minimum_detectable_effect,
    n_at_power,
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

    def test_se_pair_is_not_a_paired_sd_and_n_882_is_the_misread(self):
        as_sd = interpret_noise(OWNER_ROUNDED_SE_PAIR, 0.005, "paired_sd")
        as_se = interpret_noise(OWNER_ROUNDED_SE_PAIR, 0.005, "standard_error")
        self.assertEqual(as_sd["n80"], 882)
        self.assertEqual(n_at_power(0.05296631946624709, 0.005), 881)
        self.assertFalse(as_se["applicable"])
        self.assertIsNone(as_se["n80"])
        self.assertEqual(as_se["n80_if_misread_as_paired_sd"], 882)
        check = publish_document()["se_pair_unit_check"]
        self.assertEqual(check["unit"], "standard_error")
        self.assertFalse(check["paired_sd"])
        self.assertIsNone(check["n80"])
        self.assertIsNone(check["families_needed_for_power_80"])
        self.assertEqual(check["n80_if_misread_as_paired_sd_owner_round"], 882)
        self.assertEqual(check["n80_if_misread_as_paired_sd_exact"], 881)
        self.assertEqual(check["status"], "not_a_sample_size")


if __name__ == "__main__":
    unittest.main()
