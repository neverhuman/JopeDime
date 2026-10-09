"""Paired measured margin, fresh bitwise replay, refit and train-only refusals."""

from copy import deepcopy
import hashlib
import math
import unittest

from rlfw.arena.game_spec import digest
from rlfw.teacher_dry_run import TeacherConfig, dry_run
from rlfw.tests.test_exit_targets import row, proof

SHA = "a" * 64


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.config = TeacherConfig(SHA, SHA, SHA)
        self.pair = {
            "context_id": "train-a",
            "incumbent": row(0, 0.0),
            "challenger": row(1, math.log(2)),
        }
        self.saved = {}
        self.fresh_calls = []
        for index, r in enumerate((self.pair["incumbent"], self.pair["challenger"])):
            folds = [
                {
                    "fold_id": f,
                    "row_ids": [f"fold{f}-row{i}" for i in range(4)],
                    "targets": [0.0] * 4,
                    "predictions": [2.0 if index == 0 else 1.0] * 4,
                    "rmse": 2.0 if index == 0 else 1.0,
                }
                for f in (0, 1)
            ]
            self.saved[r["receipt_sha256"]] = {
                **proof(r),
                "epsilon": 1e-8,
                "folds": folds,
            }

    def verify(self, r):
        return deepcopy(self.saved[r["receipt_sha256"]])

    def fresh(self, r):
        self.fresh_calls.append(r["program"]["program_sha256"])
        fresh = deepcopy(r)
        fresh["receipt_sha256"] = hashlib.sha256(
            ("fresh-" + r["receipt_sha256"]).encode()
        ).hexdigest()
        verified = self.verify(r)
        verified["receipt_sha256"] = fresh["receipt_sha256"]
        return {
            "verification_kind": "fresh-native-rust-recompile",
            "measurement": fresh,
            "proof": verified,
        }

    def propose(self, pairs=None, role=None, verify=None, fresh=None, refits=None):
        return dry_run(
            pairs or [self.pair],
            self.config,
            role_check=role or (lambda c: {"role": "train", "role_map_sha256": SHA}),
            verify_receipt=verify or self.verify,
            verify_fresh=fresh or self.fresh,
            verify_refits=refits
            or (lambda p: self.fail("small contexts must not request refits")),
        )

    def test_positive_margin_proposes_small_context_within_protocol_only_no_writes(
        self,
    ):
        result = self.propose()
        proposal = result["manifest"]["proposals"][0]
        self.assertTrue(proposal["would_replace_shadow_teacher"])
        self.assertEqual(
            proposal["qualification_scope"], "promoted_within_protocol_only"
        )
        self.assertAlmostEqual(proposal["paired_d"], math.log(2))
        self.assertAlmostEqual(proposal["row_bootstrap_se"], 0.0)
        self.assertEqual(proposal["margin"], 0.01)
        self.assertEqual(proposal["statistical_observations"], 2)
        self.assertEqual(len(self.fresh_calls), 2)
        self.assertEqual(result["manifest"]["writes"], 0)
        self.assertEqual(result["sha256"], digest(result["manifest"]))

    def test_dev_refused_before_artifact_access_even_unknown(self):
        self.pair["incumbent"].update(role="development", outcome_class="infra_unknown")
        with self.assertRaisesRegex(ValueError, "train-role"):
            self.propose(
                role=lambda c: {"role": "development", "role_map_sha256": SHA},
                verify=lambda r: self.fail("no protected artifact reads"),
            )

    def test_historical_prediction_parity_and_unqualified_teacher_are_refused(self):
        for changes in (
            {"fitness": 1.0},
            {"q_hat": 1.0},
            {"source": "predicted"},
            {"outcome_class": "parity_custody"},
            {"outcome_class": "infra_unknown"},
        ):
            pairs = [deepcopy(self.pair)]
            pairs[0]["incumbent"].update(changes)
            with self.assertRaisesRegex(ValueError, "qualified measured"):
                self.propose(pairs)
        with self.assertRaisesRegex(ValueError, "closure"):
            self.propose(verify=lambda r: {**self.verify(r), "parity_tolerance": 1e-8})

    def test_exact_comparison_and_program_receipt_identity_are_required(self):
        wrong = deepcopy(self.pair)
        wrong["challenger"]["hwm_key_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            self.propose([wrong])
        with self.assertRaisesRegex(ValueError, "closure"):
            self.propose(
                verify=lambda r: {**self.verify(r), "program_sha256": "c" * 64}
            )
        with self.assertRaisesRegex(ValueError, "one registered"):
            self.propose([self.pair, self.pair])

    def test_fold_disagreement_sets_conservative_margin_and_does_not_replay(self):
        incumbent = self.saved[self.pair["incumbent"]["receipt_sha256"]]
        challenger = self.saved[self.pair["challenger"]["receipt_sha256"]]
        incumbent["folds"][1].update(predictions=[1.0] * 4, rmse=1.0)
        challenger["folds"][1].update(predictions=[2.0] * 4, rmse=2.0)
        p = self.propose()["manifest"]["proposals"][0]
        self.assertFalse(p["would_replace_shadow_teacher"])
        self.assertAlmostEqual(p["se_pair"], math.log(2))
        self.assertAlmostEqual(p["margin"], 1.96 * math.log(2))
        self.assertEqual(self.fresh_calls, [])

    def test_paired_row_membership_and_targets_must_match(self):
        new = self.saved[self.pair["challenger"]["receipt_sha256"]]
        new["folds"][0]["row_ids"][0] = "different-row"
        with self.assertRaisesRegex(ValueError, "membership"):
            self.propose()

    def test_fresh_replay_is_distinct_evidence_and_bitwise_not_another_sample(self):
        def same(r):
            return {
                "verification_kind": "fresh-native-rust-recompile",
                "measurement": r,
                "proof": self.verify(r),
            }

        with self.assertRaisesRegex(ValueError, "distinct evidence"):
            self.propose(fresh=same)

        def altered(r):
            result = self.fresh(r)
            result["proof"]["folds"][0]["predictions"][0] += 1e-15
            return result

        with self.assertRaisesRegex(ValueError, "bitwise"):
            self.propose(fresh=altered)

        def cached(r):
            return {**self.fresh(r), "verification_kind": "cache-hit"}

        with self.assertRaisesRegex(ValueError, "fresh native"):
            self.propose(fresh=cached)

    def registered_refits(self, lift=0.1):
        for r in (self.pair["incumbent"], self.pair["challenger"]):
            for fold in self.saved[r["receipt_sha256"]]["folds"]:
                fold["row_ids"] = [f"fold{fold['fold_id']}-row{i}" for i in range(32)]
                fold["targets"] = [0.0] * 32
                fold["predictions"] = [fold["rmse"]] * 32
        reg = {
            "context_id": "train-a",
            "hwm_key_sha256": SHA,
            "scope": "train-fitting-only",
            "program_sha256s": [
                self.pair[k]["program"]["program_sha256"]
                for k in ("incumbent", "challenger")
            ],
            "split_ids": [f"split-{i}" for i in range(5)],
        }
        self.pair.update(refit_registration=reg, refit_registration_sha256=digest(reg))
        return [
            {
                "split_id": split,
                "registration_sha256": digest(reg),
                "receipt_sha256": hashlib.sha256(split.encode()).hexdigest(),
                "closure_verified": True,
                "parity_verified": True,
                "parity_tolerance": 1e-9,
                "scope": "train-fitting-only",
                "context_id": "train-a",
                "program_sha256s": reg["program_sha256s"],
                "fold_d": [lift, lift],
            }
            for split in reg["split_ids"]
        ]

    def test_large_context_requires_five_registered_qualified_positive_median_refits(
        self,
    ):
        refits = self.registered_refits()
        result = self.propose(refits=lambda p: refits)["manifest"]["proposals"][0]
        self.assertTrue(result["would_replace_shadow_teacher"])
        self.assertEqual(result["qualification_scope"], "robust_cross_split")
        with self.assertRaisesRegex(ValueError, "five preregistered"):
            self.propose(refits=lambda p: refits[:4])
        refits[0]["scope"] = "development"
        with self.assertRaisesRegex(ValueError, "qualified paired"):
            self.propose(refits=lambda p: refits)

    def test_negative_refit_median_refuses_despite_original_margin(self):
        refits = self.registered_refits(-0.1)
        result = self.propose(refits=lambda p: refits)["manifest"]["proposals"][0]
        self.assertFalse(result["would_replace_shadow_teacher"])

    def test_role_change_before_final_report_is_a_hard_refusal(self):
        calls = []

        def role(c):
            calls.append(c)
            return {
                "role": "development" if len(calls) >= 3 else "train",
                "role_map_sha256": SHA,
            }

        with self.assertRaisesRegex(ValueError, "train-role"):
            self.propose(role=role)


if __name__ == "__main__":
    unittest.main()
