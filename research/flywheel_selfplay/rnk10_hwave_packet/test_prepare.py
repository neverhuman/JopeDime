import copy
import math
import unittest

from research.flywheel_selfplay.rnk10_hwave_packet.prepare import (
    DOPE_C51,
    PacketError,
    build_packet,
    validate_candidate,
)


def sha(character):
    return character * 64


def candidate_result():
    return {
        "schema": "jope.rnk10.hwave-candidate-result/v1",
        "wave": "H157",
        "status": "COMPLETE",
        "audit_status": "PASS",
        "one_shot": True,
        "retry_count": 0,
        "result_sha256": sha("1"),
        "audit_sha256": sha("2"),
        "access_audit": {
            "sealed_reads": 0,
            "holdout_reads": 0,
            "outer_test_reads": 0,
            "meta_test_reads": 0,
        },
        "frozen_selection": {
            "status": "FROZEN_WINNER",
            "rank": 1,
            "rule": "highest_elo_then_wave_then_candidate_id",
            "candidate_count": 2,
            "audit_status": "PASS",
            "considered_results_sha256": sha("a"),
            "rule_sha256": sha("b"),
        },
        "protocol": {
            "sha256": sha("3"),
            "selection_split": "DL-Train-inner",
            "selection_metric": "family_grouped_elo_vs_r882",
            "speed_gate": False,
            "dope_sha256": DOPE_C51,
            "r882_id": "r882",
            "r882_checkpoint_sha256": sha("4"),
        },
        "candidate": {
            "id": "h157-seed-2",
            "checkpoint_sha256": sha("5"),
            "source_sha256": sha("6"),
            "optimizer_updates": 120,
            "dope_sha256": DOPE_C51,
        },
        "selection_metrics": {
            "family_grouped_elo_vs_r882": 1501.25,
            "speed_used_for_selection": False,
        },
    }


class CandidateValidationTests(unittest.TestCase):
    def test_accepts_final_candidate_that_beats_r882_by_elo(self):
        self.assertEqual(
            validate_candidate(candidate_result())["family_grouped_elo_vs_r882"],
            1501.25,
        )

    def test_rejects_partial_or_unaudited_result(self):
        for field, value in (("status", "INCOMPLETE"), ("audit_status", "FAIL")):
            result = candidate_result()
            result[field] = value
            with self.subTest(field=field), self.assertRaises(PacketError):
                validate_candidate(result)

    def test_rejects_non_beating_candidate_and_exact_tie(self):
        for elo in (1500.0, 1499.99, math.nan, math.inf):
            result = candidate_result()
            result["selection_metrics"]["family_grouped_elo_vs_r882"] = elo
            with self.subTest(elo=elo), self.assertRaises(PacketError):
                validate_candidate(result)

    def test_rejects_nonwinner_or_unaudited_candidate_inventory(self):
        for key, value in (("rank", 2), ("audit_status", "FAIL")):
            result = candidate_result()
            result["frozen_selection"][key] = value
            with self.subTest(key=key), self.assertRaises(PacketError):
                validate_candidate(result)

    def test_rejects_protected_data_reads_or_missing_access_audit(self):
        result = candidate_result()
        result["access_audit"]["sealed_reads"] = 1
        with self.assertRaises(PacketError):
            validate_candidate(result)
        result = candidate_result()
        del result["access_audit"]
        with self.assertRaises(PacketError):
            validate_candidate(result)

    def test_rejects_retries_wrong_fold_wrong_reference_and_speed_gate(self):
        mutations = (
            ("retry_count", 1, None),
            ("protocol", None, "selection_split", "DL-Val"),
            ("protocol", None, "r882_id", "r783"),
            ("protocol", None, "speed_gate", True),
            ("candidate", None, "dope_sha256", sha("a")),
            ("selection_metrics", None, "speed_used_for_selection", True),
        )
        for mutation in mutations:
            result = candidate_result()
            if mutation[0] == "retry_count":
                result[mutation[0]] = mutation[1]
            else:
                result[mutation[0]][mutation[2]] = mutation[3]
            with self.subTest(mutation=mutation), self.assertRaises(PacketError):
                validate_candidate(result)

    def test_rejects_missing_candidate_identity_or_optimizer_updates(self):
        for key, value in (("checkpoint_sha256", "unknown"), ("optimizer_updates", 0)):
            result = candidate_result()
            result["candidate"][key] = value
            with self.subTest(key=key), self.assertRaises(PacketError):
                validate_candidate(result)


class PacketShellTests(unittest.TestCase):
    def test_packet_is_pinned_unused_and_never_authorizes_launch(self):
        packet = build_packet(
            candidate_result(),
            evaluator_source_sha256=sha("7"),
            roster_sha256=sha("8"),
            evaluation_identity_sha256=sha("9"),
            max_unique_outcome_reads=800,
        )
        self.assertEqual(packet["status"], "READY_FOR_INDEPENDENT_REVIEW")
        self.assertFalse(packet["launch_authorized"])
        self.assertEqual(packet["evaluation"]["ledger_state"], "UNUSED")
        self.assertEqual(packet["evaluation"]["consumed_reads"], 0)
        self.assertEqual(packet["evaluation"]["retry_count"], 0)
        self.assertEqual(packet["protected_data"]["sealed"], "EXCLUDED")
        self.assertEqual(packet["candidate_selection"]["wave"], "H157")

    def test_bad_evaluator_or_roster_identity_refuses_packet(self):
        args = dict(
            result=candidate_result(),
            evaluator_source_sha256=sha("7"),
            roster_sha256=sha("8"),
            evaluation_identity_sha256=sha("9"),
            max_unique_outcome_reads=800,
        )
        for key in (
            "evaluator_source_sha256",
            "roster_sha256",
            "evaluation_identity_sha256",
        ):
            broken = copy.copy(args)
            broken[key] = "not-a-hash"
            with self.subTest(key=key), self.assertRaises(PacketError):
                build_packet(**broken)

    def test_invalid_or_zero_outcome_budget_refuses_packet(self):
        with self.assertRaises(PacketError):
            build_packet(
                candidate_result(),
                evaluator_source_sha256=sha("7"),
                roster_sha256=sha("8"),
                evaluation_identity_sha256=sha("9"),
                max_unique_outcome_reads=0,
            )


if __name__ == "__main__":
    unittest.main()
