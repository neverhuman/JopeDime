import unittest

import registry_namespace as r


class RegistryNamespaceTests(unittest.TestCase):
    def test_current_evidence_refuses_enable(self):
        ok, missing = r.enable_namespace()
        self.assertFalse(ok)
        self.assertIn("historical_import_complete", missing)
        self.assertIn("new_gp_identity", missing)
        snap = r.snapshot()
        self.assertFalse(snap["enabled"])
        self.assertFalse(snap["wired_to_live_gp"])
        self.assertFalse(snap["claimed_savings"])
        self.assertEqual(snap["reusable_historical_completions"], 0)

    def test_forbids_745_claim(self):
        with self.assertRaisesRegex(ValueError, "74.5"):
            r.enable_namespace(claimed_savings_745=True)
        claim = r.savings_claim(10, 100, modeled_wall=80, measured_core=12)
        self.assertFalse(claim["claimed_745"])
        self.assertFalse(claim["achieved"])
        with self.assertRaisesRegex(ValueError, "overload"):
            r.savings_claim(50, 10)

    def test_all_gates_still_refuse_enable_from_this_helper(self):
        evidence = {name: True for name in r.GATES}
        with self.assertRaisesRegex(ValueError, "not authorized"):
            r.enable_namespace(evidence)


if __name__ == "__main__":
    unittest.main()
