"""Synthetic metadata controls; none admits data or qualifies W5."""

import copy
import io
import unittest
from unittest.mock import Mock

import verify


def rows():
    result = {
        "context_sha256": "target", "table_sha256": "table", "cut_id": "cut",
        "family_alias_component": "family", "source_context_sha256": "target",
        "role": "DL-Train", "current_role_alias_manifest_sha256": "role-proof",
        "original_training_row_ids": ["r1", "r2", "r3"], "screen_row_ids": ["r1", "r3"],
    }
    for field in ("context_sha256", "table_sha256", "cut_id", "family_alias_component"):
        result["screen_" + field] = result[field]
    return result


def pair():
    parent = dict(
        program_sha256="parent", context_sha256="target", table_sha256="table",
        cut_id="cut", evaluator="eval", metric="metric", gate="gate", vocabulary="vocab",
        status="completed", evaluation_kind="full", outcome_origin="measured", gate_passed=True,
        source_version="version", source_sha256="source", collection_policy="selected",
        original_outcome_sha256="lossless", outcome_sha256="lossless",
    )
    child = dict(parent, program_sha256="child", parent_ids=["parent"], gate_passed=False)
    return {"parent": parent, "child": child, "operator": "mutation"}


class AdmissionTests(unittest.TestCase):
    def test_rejects_body_reads_before_calling_s3(self):
        client = Mock()
        bad = [
            (verify.PREFIX + "round-01/scores/retained_scores_round_01.jsonl", "any"),
            ("other/round-01/persistence_receipt.json", next(iter(verify.PINS.values()))[0]),
            (verify.PREFIX + "round-01/persistence_receipt.json", "latest"),
            (verify.PREFIX + "corpus_v3_wider_strict_train_256/corpus_v3_manifest.json", "any"),
        ]
        for key, version in bad:
            with self.subTest(key=key, version=version), self.assertRaises(ValueError):
                verify.read_counter(client, key, version)
        client.get_object.assert_not_called()

    def test_counter_hash_is_over_bytes_not_object_metadata(self):
        relative, (version, _) = next(iter(verify.PINS.items()))
        client = Mock()
        client.get_object.return_value = {"Body": io.BytesIO(b"x" * 806), "VersionId": version}
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            verify.read_counter(client, verify.PREFIX + relative, version)

    def test_empty_inventory_refuses_with_unknown_counts(self):
        result = verify.verify({})
        self.assertEqual((result["verdict"], result["w5"]), ("FAIL", "HELD"))
        self.assertIsNone(result["eligible_pair_count"])
        self.assertIsNone(result["pair_coverage"])
        self.assertEqual(result["row_closure"], "UNKNOWN")

    def test_claimed_counter_hashes_without_bytes_do_not_admit_pairs(self):
        inventory = dict(bucket=verify.BUCKET, prefix=verify.PREFIX, listing_complete=True,
                         counters=[dict(key=verify.PREFIX + k, version=v, sha256=h)
                                   for k, (v, h) in verify.PINS.items()])
        result = verify.verify(inventory)
        self.assertIsNone(result["retention_receipts_verified"])
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("admitted_parent_operator_child_export_missing", result["reasons"])
        self.assertIn("original_training_and_strat300_row_ledger_missing", result["reasons"])

    def test_target_subset_positive_control(self):
        self.assertEqual(verify.row_errors(rows()), [])

    def test_strat300_nq_zero_does_not_establish_row_closure(self):
        self.assertTrue(verify.row_errors({"nq": 0, "nrows": 300}))

    def test_excluded_row_sibling_wrong_cut_and_alias_each_refuse(self):
        for change in (
            {"screen_row_ids": ["r1", "excluded"]},
            {"source_context_sha256": "sibling"},
            {"screen_cut_id": "other-cut"},
            {"screen_family_alias_component": "other-alias"},
            {"current_role_alias_manifest_sha256": None},
            {"role": "DL-Val"},
            {"screen_row_ids": ["r1", "r1"]},
        ):
            with self.subTest(change=change):
                self.assertTrue(verify.row_errors(dict(rows(), **change)))

    def test_completed_gate_failure_can_be_measured(self):
        self.assertEqual(verify.pair_errors(pair()), [])

    def test_missing_screened_execution_failure_and_finite_sentinel_refuse(self):
        for change in (
            {"status": "missing"}, {"status": "failed"}, {"evaluation_kind": "screen"},
            {"outcome_origin": "sentinel", "fitness": 100, "logratio": 0},
            {"gate_passed": None}, {"outcome_sha256": "rounded"},
        ):
            candidate = pair()
            candidate["child"].update(change)
            with self.subTest(change=change):
                self.assertTrue(verify.pair_errors(candidate))

    def test_ast_similarity_cannot_replace_explicit_join(self):
        candidate = pair()
        candidate["child"]["parent_ids"] = []
        candidate["child"]["similar_ast"] = "parent"
        self.assertIn("exact_parent_operator_join_missing", verify.pair_errors(candidate))

    def test_evaluator_table_cut_and_source_provenance_are_required(self):
        for field in ("context_sha256", "table_sha256", "cut_id", "evaluator", "metric", "gate", "vocabulary"):
            candidate = pair()
            candidate["child"][field] = "other"
            with self.subTest(field=field):
                self.assertTrue(verify.pair_errors(candidate))
        for field in ("source_sha256", "source_version", "collection_policy", "original_outcome_sha256"):
            candidate = copy.deepcopy(pair())
            del candidate["parent"][field]
            with self.subTest(field=field):
                self.assertTrue(verify.pair_errors(candidate))


if __name__ == "__main__":
    unittest.main(verbosity=2)
