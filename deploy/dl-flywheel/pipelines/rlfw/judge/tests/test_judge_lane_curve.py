"""Claim and winner-curve contract tests for the judge lane."""

import unittest

from .test_judge_lane import _disclosures, cluster_stat, g_curve, g_judge, judge_claim

class ClaimAndCurveTest(unittest.TestCase):
    def test_winners_curse_fixture_falls_as_b_grows(self):
        contexts = []
        for lineage in range(3):
            disclosures = _disclosures(f"curve-{lineage}")
            contexts.append(
                {
                    "disclosures": disclosures,
                    "judge_q_by_program": {
                        item["program_sha256"]: 1 - item["inner_q_grid"] for item in disclosures
                    },
                    "lineage_id": f"L{lineage}",
                }
            )
        curve = g_curve(contexts, draws=30, seed=1)
        self.assertGreater(curve[1]["mean"], curve[8]["mean"])

    def test_siblings_do_not_narrow_the_lineage_interval(self):
        one = cluster_stat({"L1": [1.0], "L2": [3.0]}, draws=40, seed=3)
        many = cluster_stat({"L1": [1.0, 1.0, 1.0], "L2": [3.0, 3.0]}, draws=40, seed=3)
        self.assertEqual(one["ci95"], many["ci95"])
        self.assertEqual(one["lineages"], many["lineages"])
        self.assertGreater(many["units"], one["units"])

    def test_g4_claim_text_requires_every_gate_on_a_fixture(self):
        units = [
            {"censored": False, "judge_q": 1.0, "lineage_id": f"L{index}"} for index in range(12)
        ]
        report = g_judge(units, draws=20, seed=1)
        self.assertGreater(report["ci95"][0], 0.0)
        allowed = judge_claim(
            report,
            n_families=200,
            n_contexts=400,
            canaries_a2_a4=True,
            overlap_status="no_overlap",
            sigma_source="fixture",
            observed_sigma=1.0,
            registered_delta=0.2,
        )
        self.assertTrue(allowed["allowed"])
        self.assertEqual(allowed["claim"], "Policy-training-unseen-family lift")
        self.assertEqual(allowed["sigma_source"], "fixture")
        self.assertEqual(allowed["multiplicity"], 1)
        blocked = judge_claim(
            report,
            n_families=200,
            n_contexts=400,
            canaries_a2_a4=True,
            overlap_status="no_overlap",
            sigma_source="unmeasured",
            observed_sigma=None,
            registered_delta=0.2,
        )
        self.assertFalse(blocked["allowed"])
        self.assertIsNone(blocked["claim"])
        self.assertIn("observed noise unmeasured", blocked["reasons"])
        with self.assertRaises(ValueError):
            judge_claim(
                report,
                n_families=200,
                n_contexts=400,
                canaries_a2_a4=True,
                overlap_status="no_overlap",
                sigma_source="measured",
                observed_sigma=1.0,
                registered_delta=0.2,
            )
        underpowered = judge_claim(
            report,
            n_families=200,
            n_contexts=400,
            canaries_a2_a4=True,
            overlap_status="no_overlap",
            sigma_source="fixture",
            observed_sigma=1.0,
            registered_delta=0.1,
        )
        self.assertFalse(underpowered["allowed"])
        self.assertGreater(underpowered["power_required_families"], 200)
        censored = g_judge([{"censored": True, "judge_q": None, "lineage_id": "L0"}], draws=10, seed=1)
        refused = judge_claim(
            censored,
            n_families=200,
            n_contexts=400,
            canaries_a2_a4=True,
            overlap_status="no_overlap",
            sigma_source="fixture",
            observed_sigma=1.0,
            registered_delta=0.2,
        )
        self.assertIn("censoring can reverse the sign", refused["reasons"])
        self.assertIsNone(refused["claim"])


