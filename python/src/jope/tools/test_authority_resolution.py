import unittest

import authority_resolution as a


class AuthorityResolutionTests(unittest.TestCase):
    def test_merge_is_two_cycle_execution(self):
        receipt = a.resolve("explicit_merge")
        self.assertEqual(receipt["live_execution_authority"], a.CODEX_TWO_CYCLE)
        self.assertEqual(receipt["diagnostic_authority"], a.CLOSE_FLYWHEEL)
        self.assertFalse(receipt["frozen_z_live_pointer"])
        self.assertFalse(receipt["blanket_v6_eligibility"])
        self.assertFalse(receipt["gp_unfence"])
        self.assertFalse(receipt["writes_live_index"])

    def test_close_flywheel_cannot_win_execution(self):
        with self.assertRaisesRegex(ValueError, "not live execution"):
            a.resolve("close_flywheel")

    def test_two_cycle_alone_still_forbids_v6_and_unfence(self):
        receipt = a.resolve("codex_two_cycle")
        self.assertEqual(receipt["live_execution_authority"], a.CODEX_TWO_CYCLE)
        self.assertFalse(receipt["gp_unfence"])
        self.assertIn("parallel/src/eval_registry.rs", receipt["grok_stops_editing"])


if __name__ == "__main__":
    unittest.main()
