import unittest

import component_ledger as L


class ComponentLedgerTests(unittest.TestCase):
    def test_already_in_index_is_not_new(self):
        row = L.join_row("aa", already_in_index=True, stage="discovered")
        self.assertEqual(row["stage"], "context-published")
        self.assertFalse(row["counted_as_new"])
        ledger = L.build_ledger([row, L.join_row("bb")])
        self.assertEqual(ledger["already_in_index"], 1)
        self.assertEqual(ledger["counted_as_new"], 0)
        self.assertEqual(ledger["unindexed_candidates"], 1)

    def test_retained_arrival_is_not_original_discovery(self):
        row = L.join_row("copied-component", stage="retained")
        self.assertFalse(row["counted_as_new"])
        self.assertTrue(row["unindexed_candidate"])
        self.assertIn("unavailable", row["novelty_evidence"])

    def test_growth_subtracts_live_and_sixty_four(self):
        growth = L.growth_candidates(
            ["aa", "bb", "cc"],
            already_in_index=["aa"],
            live_ids=["bb"],
            sealed_ids=[],
            evidenced_unresolved=["dd"],
        )
        self.assertEqual(growth["n_already_in_index"], 1)
        self.assertEqual(growth["novel_ids"], ["cc"])
        self.assertEqual(growth["auto_admitted"], 0)

    def test_source_path_is_not_resolution(self):
        receipt = L.provenance_receipt("aa", {}, source_path="/quant/aa")
        self.assertFalse(receipt["resolved"])
        self.assertTrue(receipt["source_path_is_lead_only"])


if __name__ == "__main__":
    unittest.main()
