"""The sealed roster proposal fails closed when a family meets the training closure."""

import unittest

from rlfw.judge.roster import proof_disjoint, proposal_document


def family(contexts):
    return {
        "context_ids": contexts,
        "view_ids": ["view-" + contexts[0]],
        "dataset_hashes": ["dataset-" + contexts[0]],
        "member_roles": ["sealed_final"],
    }


class RosterProofTest(unittest.TestCase):
    def test_disjoint_families_pass_and_do_not_spend_a_seal(self):
        sealed = {"fam-a": family(["c1"])}
        closure = {
            "family_ids": {"fam-train"},
            "context_ids": {"c-train"},
            "view_ids": {"view-train"},
            "dataset_hashes": {"dataset-train"},
        }
        result = proof_disjoint(sealed, closure)
        self.assertTrue(result["passed"])
        self.assertFalse(result["seal_spent"])
        self.assertFalse(result["seal_created"])
        doc = proposal_document(
            {"meta_training": ["fam-train"], "development": [], "confirmation": [], "sealed_final": ["fam-a"]},
            result,
            {"checked": ["synthetic"]},
        )
        self.assertFalse(doc["adopted"])
        self.assertEqual(doc["alpha_budget"], "unspent_jepson_decision")
        self.assertEqual(doc["encoder_pretraining_closure"], "unmeasured")

    def test_a_family_in_the_training_closure_fails(self):
        sealed = {"fam-a": family(["c1"])}
        closure = {
            "family_ids": set(),
            "context_ids": {"c1"},
            "view_ids": set(),
            "dataset_hashes": set(),
        }
        result = proof_disjoint(sealed, closure)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["reasons"], ["context_ids"])

    def test_a_split_role_fails(self):
        record = family(["c1"])
        record["member_roles"] = ["sealed_final", "meta_training"]
        result = proof_disjoint(
            {"fam-a": record},
            {"family_ids": set(), "context_ids": set(), "view_ids": set(), "dataset_hashes": set()},
        )
        self.assertFalse(result["passed"])
        self.assertIn("role_split", result["failures"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
