import json
import tempfile
import unittest
from pathlib import Path

import unpublished_cut_census as u


class UnpublishedCutCensusTests(unittest.TestCase):
    def fixture(self, rows, protected_values_opened=False, **extra):
        body = dict(
            schema="parallel.redteam_metadata_audit.v1",
            protected_values_opened=protected_values_opened,
            expansion_rows=rows,
            component_groups_outside_retained_indexes=1930,
            component_complete_groups=12,
            **extra,
        )
        return body

    def test_metadata_complete_does_not_fill_the_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "census.json"
            path.write_text(
                json.dumps(
                    self.fixture(
                        [
                            dict(
                                component_metadata_complete=True,
                                registered_eligible=False,
                                metadata_eligible_addition=False,
                                task="regression",
                            ),
                            dict(
                                component_metadata_complete=False,
                                registered_eligible=False,
                                metadata_eligible_addition=False,
                                task="binary",
                            ),
                        ]
                    )
                )
            )
            receipt = u.recount(path)
            self.assertEqual(receipt["component_metadata_complete"], 1)
            self.assertEqual(receipt["registered_eligible"], 0)
            self.assertTrue(receipt["empty_queue"])
            self.assertEqual(receipt["execution_queue"], 0)
            self.assertFalse(receipt["claimed_4048_certified"])
            self.assertFalse(receipt["live_index_written"])
            self.assertIsNone(receipt["actual_eligible_additions"])
            self.assertIsNone(receipt["native_membership_verified"])

    def test_refuses_protected_open_census(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "census.json"
            path.write_text(json.dumps(self.fixture([], protected_values_opened=True)))
            with self.assertRaises(ValueError):
                u.recount(path)

    def test_missing_protected_access_declaration_is_not_false(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "census.json"
            body = self.fixture([])
            del body["protected_values_opened"]
            path.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, "explicitly"):
                u.recount(path)

    def test_retained_and_metadata_candidates_do_not_claim_native_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "census.json"
            path.write_text(
                json.dumps(
                    self.fixture(
                        [
                            dict(
                                component_metadata_complete=True,
                                registered_eligible=True,
                                metadata_eligible_addition=True,
                                task="regression",
                                retained_index=False,
                                historical_registry_dispositions=[dict(partition="meta_train")],
                            ),
                            dict(
                                component_metadata_complete=False,
                                registered_eligible=False,
                                retained_index=True,
                                missing_components=["target"],
                            ),
                            dict(task="regression"),
                        ],
                        active=dict(rows=8),
                        errors=[dict(path="unreadable.json")],
                    )
                )
            )
            receipt = u.recount(path)
            self.assertEqual(receipt["registered_eligible"], 1)
            self.assertIn("1 metadata-declared registered", receipt["reason"])
            self.assertEqual(receipt["retained_index_rows"], 1)
            self.assertEqual(receipt["component_complete_without_retained_assembly"], 1)
            self.assertEqual(receipt["component_metadata_unknown"], 1)
            self.assertEqual(receipt["retained_membership_unknown"], 1)
            self.assertEqual(receipt["active_membership"], dict(rows=8))
            self.assertEqual(receipt["missing_or_incompatible_components"], dict(target=1))
            self.assertEqual(receipt["historical_registry_dispositions"], dict(meta_train=1))
            self.assertEqual(len(receipt["source_errors"]), 1)
            self.assertIsNone(receipt["actual_eligible_additions"])
            self.assertEqual(receipt["execution_queue"], 0)


if __name__ == "__main__":
    unittest.main()
