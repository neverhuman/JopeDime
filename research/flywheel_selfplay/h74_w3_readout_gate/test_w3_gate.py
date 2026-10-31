from __future__ import annotations

import unittest

from research.flywheel_selfplay.h74_w3_readout_gate.w3_gate import (
    EXPECTED_CONTEXTS,
    EXPECTED_FAMILIES,
    EXPECTED_DOPE_SHA256,
    PRIMARY_CONTRASTS,
    Refusal,
    evaluate,
    validate,
)


def evidence() -> dict:
    counts = [6] * 194 + [5] * 11
    assert sum(counts) == EXPECTED_CONTEXTS
    closure = {
        "training_role": "DL-Train",
        "context_count": EXPECTED_CONTEXTS,
        "family_count": EXPECTED_FAMILIES,
        "fold_seed": 4900,
        "fold_count": 5,
        "seeds": [701, 702, 703],
        "dl_val_rows": 0,
        "outer_rows": 0,
        "sealed_rows": 0,
        "dope_sha256": EXPECTED_DOPE_SHA256,
        "admitted_export_sha256": "a" * 64,
        "fold_manifest_sha256": "b" * 64,
        "winner_config_sha256": "c" * 64,
        "h59_disposition_sha256": "d" * 64,
    }
    contrasts = {}
    for contrast in PRIMARY_CONTRASTS:
        contrasts[contrast] = [
            {
                "family_id": f"family-{index:03d}",
                "context_count": count,
                "seed_deltas": [0.02, 0.02, 0.02],
                "gate_failures_arm": [0, 0, 0],
                "gate_failures_control": [0, 0, 0],
            }
            for index, count in enumerate(counts)
        ]
    return {
        "schema": "jope.h74.w3-family-evidence/v1",
        "closure": closure,
        "bootstrap": {
            "unit": "family",
            "draws": 10000,
            "seed": 20261031,
            "seed_average_within_family": True,
            "paired_contrasts": True,
            "adjustment": "bonferroni_two_primary_contrasts",
        },
        "kill_checks": {
            "auxiliary_loss_only_gain": False,
            "shuffled_auxiliary_matches_gain": False,
            "near_pass_semantics_changed": False,
            "zero_update_choices_preserved": True,
        },
        "contrasts": contrasts,
    }


class W3GateTests(unittest.TestCase):
    def test_synthetic_positive_requires_both_primary_contrasts(self) -> None:
        result = evaluate(evidence())
        self.assertEqual(result["decision"], "ADVANCE_BOTH_PRIMARY_CONTRASTS")
        self.assertFalse(result["launch_authorized"])
        self.assertFalse(result["source_provenance_verified"])
        self.assertEqual(set(result["contrasts"]), set(PRIMARY_CONTRASTS))

    def test_strict_lower_bound_boundary_does_not_pass(self) -> None:
        item = evidence()
        for contrast in PRIMARY_CONTRASTS:
            for row in item["contrasts"][contrast]:
                row["seed_deltas"] = [0.001, 0.001, 0.001]
        result = evaluate(item)
        self.assertEqual(result["decision"], "NO_ADVANCE")
        self.assertEqual(
            result["contrasts"][PRIMARY_CONTRASTS[0]]["adjusted_family_mean_lcb"], 0.001
        )

    def test_harm_tail_above_ten_percent_does_not_pass(self) -> None:
        item = evidence()
        for contrast in PRIMARY_CONTRASTS:
            for row in item["contrasts"][contrast][:21]:
                row["seed_deltas"] = [-0.02, -0.02, -0.02]
        result = evaluate(item)
        self.assertEqual(result["decision"], "NO_ADVANCE")
        self.assertEqual(
            result["contrasts"][PRIMARY_CONTRASTS[0]]["families_loss_gt_0_01"], 21
        )

    def test_excess_gate_failure_rate_blocks_contrast(self) -> None:
        item = evidence()
        item["contrasts"][PRIMARY_CONTRASTS[0]][0]["gate_failures_arm"] = [1, 0, 0]
        result = evaluate(item)
        self.assertFalse(result["contrasts"][PRIMARY_CONTRASTS[0]]["primary_gate_pass"])
        self.assertEqual(result["decision"], "NO_ADVANCE")

    def test_registered_kill_condition_blocks_advancement(self) -> None:
        item = evidence()
        item["kill_checks"]["shuffled_auxiliary_matches_gain"] = True
        self.assertFalse(evaluate(item)["kill_checks_pass"])

    def test_changed_or_leaky_training_closure_refuses(self) -> None:
        for key, value in (("fold_seed", 4901), ("dl_val_rows", 1), ("sealed_rows", 1)):
            item = evidence()
            item["closure"][key] = value
            with self.subTest(key=key), self.assertRaises(Refusal):
                validate(item)

    def test_contrasts_must_have_same_roster_and_context_counts(self) -> None:
        item = evidence()
        item["contrasts"][PRIMARY_CONTRASTS[1]][0]["family_id"] = "different-family"
        with self.assertRaises(Refusal):
            validate(item)

    def test_nonfinite_or_malformed_seed_values_refuse(self) -> None:
        item = evidence()
        item["contrasts"][PRIMARY_CONTRASTS[0]][0]["seed_deltas"] = [
            0.1,
            float("inf"),
            0.1,
        ]
        with self.assertRaises(Refusal):
            validate(item)

    def test_exact_roster_size_is_required(self) -> None:
        item = evidence()
        item["contrasts"][PRIMARY_CONTRASTS[0]].pop()
        with self.assertRaises(Refusal):
            validate(item)


if __name__ == "__main__":
    unittest.main()
