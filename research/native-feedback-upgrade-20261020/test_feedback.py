"""Behavioral proofs for leakage, measured targets and actual learning."""
import copy
import unittest

import numpy as np
import torch

from feedback import (FeedbackLedger, ValueAdapter, choose, cluster_split,
                      measured_loss, parameter_hash, train_adapter, trie_log_probs,
                      paired_interval, selection_metric)


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.split = {"train": {"partition": "train", "group": "a"},
                      "assessment": {"partition": "assessment", "group": "b"}}
        self.row = {"context_id": "train", "parent_id": "a", "protocol_sha256": "p", "incumbent_q": 0.,
                    "members": [{"program_sha256": "x", "status": "complete", "q": .1},
                                {"program_sha256": "y", "status": "complete", "q": .9}]}

    def test_assessment_outcomes_cannot_enter_learning(self):
        ledger = FeedbackLedger(self.split, [self.row])
        with self.assertRaisesRegex(ValueError, "only TRAIN"):
            ledger.acquire("assessment", 0, wave=1, propensity=1.)
        self.assertEqual(ledger.events, [])

    def test_duplicate_is_idempotent_and_failure_never_synthetic_target(self):
        ledger = FeedbackLedger(self.split, [self.row])
        first = ledger.acquire("train", 0, wave=1, propensity=.5)
        self.assertEqual(ledger.acquire("train", 0, wave=2, propensity=.5), first)
        self.assertEqual(len(ledger.events), 1)
        failed = copy.deepcopy(self.row)
        failed["members"][1].update(status="timeout", q=None)
        second = FeedbackLedger(self.split, [failed])
        self.assertIsNone(second.acquire("train", 1, wave=1, propensity=.5))
        self.assertEqual(len(second.targets), 0)
        self.assertEqual(len(second.events), 1)

    def test_exact_trie_probabilities_normalize_at_shared_prefix_and_eos(self):
        seqs = [[1, 5, 2], [1, 5, 7, 2], [1, 8, 2]]
        logits = torch.zeros(3, 3, 10)
        values = trie_log_probs(logits, seqs).exp()
        torch.testing.assert_close(values, torch.tensor([.25, .25, .5]))
        with self.assertRaisesRegex(ValueError, "unique"):
            trie_log_probs(logits[:2], [seqs[0], seqs[0]])

    def test_acquisition_depends_on_predictions_not_outcomes(self):
        selected, propensity = choose([0, 1, 2], [0., 10., 0.], [0., 0., 0.],
                                      mode="targeted", rng=np.random.default_rng(1))
        self.assertEqual(selected, 1)
        self.assertAlmostEqual(propensity, .75 + .25/3)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            choose([], None, None, mode="uniform", rng=np.random.default_rng(1))

    def test_targets_update_weights_and_improve_measured_order(self):
        torch.manual_seed(4)
        model = ValueAdapter(4)
        ledger = FeedbackLedger(self.split, [self.row])
        for i in range(2):
            ledger.acquire("train", i, wave=0, propensity=.5)
        features = {"train": torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])}
        before = parameter_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        result = train_adapter(model, optimizer, features, ledger, steps=40, seed=2)
        self.assertNotEqual(before, parameter_hash(model))
        self.assertEqual(result["snapshot_sha256"], ledger.identity())
        self.assertGreater(model(features["train"])[1].item(), model(features["train"])[0].item())
        self.assertTrue(all(step["target_hashes"] for step in result["steps"]))

    def test_ties_have_no_pairwise_direction_and_nonfinite_refused(self):
        pred = torch.zeros(3, requires_grad=True)
        loss = measured_loss(pred, torch.ones(3))
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.zeros(3), atol=1e-7, rtol=0)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            measured_loss(pred, torch.tensor([1., float("nan"), 2.]))

    def test_retained_parent_equivalence_cannot_cross_partitions(self):
        rows = [{"context_id": str(i), "parent_id": str(i)} for i in range(40)]
        meta = {"cids": list(map(str, range(40))), "family": list(map(str, range(40))),
                "dataset": list(map(str, range(40)))}
        gids = np.arange(40)
        gids[1] = gids[0]
        split = cluster_split(rows, meta, gids, 5)
        self.assertEqual(split["0"], split["1"])

    def test_paired_interval_uses_parents_not_context_pseudoreplication(self):
        candidate = {str(i): 1. for i in range(20)}
        reference = {str(i): .5 for i in range(20)}
        groups = {str(i): str(i // 10) for i in range(20)}
        result = paired_interval(candidate, reference, groups, resamples=1000)
        self.assertEqual(result["parents"], 2)
        self.assertEqual(result["effect"], .5)
        self.assertEqual(result["lower"], .5)

    def test_budget_selects_before_reading_quality_and_preserves_baseline(self):
        result = selection_metric(self.row, np.array([10., 0.]), 1)
        self.assertEqual(result["selected"], [0])
        self.assertEqual(result["quality"], .1)
        self.assertEqual(result["regret"], 1.)


if __name__ == "__main__":
    unittest.main()
