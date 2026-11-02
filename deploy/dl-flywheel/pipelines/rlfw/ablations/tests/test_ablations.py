"""C11 / PR-18 fixtures. No BENCH rows, checkpoints, or program execution."""

import inspect
import math
import unittest
from copy import deepcopy

from rlfw.ablations.arena_row import measurement_from_slot, uniform_mu
from rlfw.ablations.catalog import CatalogTrie
from rlfw.ablations.compare import ci_verdict, lineage_deltas, score_context
from rlfw.ablations.exact import effective_support, j_b, kl_divergence
from rlfw.ablations.gumbel import gumbel_topk
from rlfw.ablations.l6_receipt import L6_ROW_FIELDS, require_g03_receipt, require_l6_cost_row
from rlfw.ablations.tests.test_controls import J8ControlTests as J8ControlTests
from rlfw.ablations.losses import dpo_loss, ipo_loss
from rlfw.ablations.measured import build_pairs, checked_measurement
from rlfw.ablations.ope import self_normalized_is, uniform_trie_ope
from rlfw.ablations.puct import puct_select
from rlfw.trie_logprob import child_logprob

SHA = "a" * 64
SHA_B = "b" * 64
CTX = "c" * 64
LIN = "d" * 64
EVAL = "e" * 64


def _row(program, q_grid, role="train", subrole=None, **extra):
    row = {
        "reward_kind": "verified_comparable_q",
        "q": q_grid / 1_000_000,
        "q_grid": q_grid,
        "outcome": "qualified_complete",
        "qualified": True,
        "parity_verified": True,
        "closure_verified": True,
        "parity_max_normalized_error": 0.0,
        "program_sha256": program,
        "context_id": CTX,
        "lineage_id": LIN,
        "role": role,
        "subrole": subrole,
        "evaluator_sha256": EVAL,
        "protocol": "jope.flywheel-s1-inner-rmse/v1",
        "tokens": (1, 3, 2),
    }
    row.update(extra)
    return row


from rlfw.ablations.tests.ablations_test_core import (
    ArenaRowTests,
    CompareTests,
    ExactTests,
    LossTests,
    MeasuredTests,
    OpeTests,
    SearchTests,
)

# Keep unittest IDs stable: the offline runner consumes this registered module name.
for _test_class in (
    ArenaRowTests,
    CompareTests,
    ExactTests,
    LossTests,
    MeasuredTests,
    OpeTests,
    SearchTests,
):
    _test_class.__module__ = __name__
del _test_class


class TorchTests(unittest.TestCase):
    def _model(self):
        import torch

        from rlfw.ablations.policy_io import _import_dltrain, load_weights

        torch.manual_seed(7)
        PolicyConfig, PolicyLM, _, parameter_sha256 = _import_dltrain()
        model = PolicyLM(
            PolicyConfig(
                vocab_size=16,
                ctx_dim=8,
                d_model=16,
                n_layers=1,
                n_heads=4,
                d_ff=32,
                max_seq=16,
                dropout=0.1,
                pad_id=0,
            )
        )
        return torch, model, parameter_sha256, load_weights

    def test_torch_logprob_matches_trie_logprob(self):
        import torch

        from rlfw.ablations.torch_loss import child_logprob_torch, sequence_logprob

        trie = CatalogTrie([(1, 3, 2), (1, 4, 2)])
        logits = torch.tensor(
            [0.0, 0.2, -0.4, 0.5, 1.5, -0.2] + [0.0] * 10, dtype=torch.float64
        )
        for chosen, prefix in ((3, (1,)), (4, (1,)), (2, (1, 3))):
            got = float(child_logprob_torch(logits, trie.get(prefix), chosen))
            expected = child_logprob(logits.tolist(), trie.get(prefix), chosen, 1.0)
            self.assertAlmostEqual(got, expected, places=12)
        row = torch.stack([logits, logits])
        total = float(sequence_logprob(row, (1, 3, 2), trie))
        manual = child_logprob(logits.tolist(), trie.get((1,)), 3, 1.0) + child_logprob(
            logits.tolist(), trie.get((1, 3)), 2, 1.0
        )
        self.assertAlmostEqual(total, manual, places=12)

    def test_ipo_step_changes_policy_not_q_head(self):
        import torch

        from rlfw.ablations.torch_loss import model_sequence_logprob, preference_loss, preference_step

        torch_mod, model, parameter_sha256, _load = self._model()
        reference = deepcopy(model)
        trie = CatalogTrie([(1, 3, 2), (1, 4, 2)])
        ctx = torch_mod.zeros(1, 8)
        self.assertNotIn("q", inspect.signature(preference_loss).parameters)
        self.assertNotIn("q_hat", inspect.signature(preference_loss).parameters)
        pure = preference_loss(
            model, reference, (1, 3, 2), (1, 4, 2), trie, ctx, "ipo", tau=2.0, lambda_kl=0.0
        )
        self.assertAlmostEqual(float(pure["loss"]), 0.0625, places=5)
        dpo = preference_loss(
            model, reference, (1, 3, 2), (1, 4, 2), trie, ctx, "dpo", lambda_kl=0.0
        )
        self.assertAlmostEqual(float(dpo["loss"]), math.log(2), places=5)
        shifted = deepcopy(reference)
        with torch_mod.no_grad():
            shifted.tok_emb.weight[:, 0].add_(0.5)
        tilted = preference_loss(
            model, shifted, (1, 3, 2), (1, 4, 2), trie, ctx, "ipo", tau=2.0, lambda_kl=0.02
        )
        plain = preference_loss(
            model, shifted, (1, 3, 2), (1, 4, 2), trie, ctx, "ipo", tau=2.0, lambda_kl=0.0
        )
        self.assertGreater(float(tilted["loss"]), float(plain["loss"]))
        before = parameter_sha256(model)
        reference_before = parameter_sha256(reference)
        heads = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name.startswith(("q_head.", "feas_head.", "rt_head."))
        }
        first = model_sequence_logprob(model, (1, 3, 2), trie, ctx)
        second = model_sequence_logprob(model, (1, 3, 2), trie, ctx)
        self.assertTrue(torch_mod.equal(first, second))
        optimizer = torch_mod.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
        step = preference_step(
            model,
            reference,
            (1, 3, 2),
            (1, 4, 2),
            trie,
            ctx,
            optimizer,
            "ipo",
            tau=2.0,
            lambda_kl=0.0,
        )
        self.assertTrue(math.isfinite(step["loss"]))
        self.assertNotEqual(parameter_sha256(model), before)
        self.assertEqual(parameter_sha256(reference), reference_before)
        for name, parameter in model.named_parameters():
            if name in heads:
                self.assertTrue(torch_mod.equal(parameter, heads[name]))

    def test_right_padding_does_not_change_prefix_logits(self):
        torch_mod, model, _, _load = self._model()
        from rlfw.ablations.torch_loss import sequence_logprob

        trie = CatalogTrie([(1, 3, 2), (1, 3, 5, 2)])
        ctx = torch_mod.zeros(1, 8)
        model.eval()
        short = torch_mod.tensor([[1, 3, 2]])
        padded = torch_mod.tensor([[1, 3, 2, 0, 0]])
        length_short = torch_mod.tensor([3])
        length_padded = torch_mod.tensor([3])
        seq = torch_mod.zeros(1, dtype=torch_mod.long)
        with torch_mod.no_grad():
            short_logits = model(short, length_short, ctx, seq).logits[0]
            padded_logits = model(padded, length_padded, ctx, seq).logits[0]
        self.assertTrue(torch_mod.allclose(short_logits[:2], padded_logits[:2]))
        self.assertTrue(
            torch_mod.allclose(
                sequence_logprob(short_logits, (1, 3, 2), trie),
                sequence_logprob(padded_logits, (1, 3, 2), trie),
            )
        )

    def test_load_drops_optimizer_and_reties_head(self):
        torch_mod, model, parameter_sha256, load_weights = self._model()
        expected = parameter_sha256(model)
        raw = {
            f"model.{name}": value.detach().clone()
            for name, value in model.state_dict().items()
            if name != "lm_head.weight"
        }
        raw["optimizer.tok_emb.weight.exp_avg"] = torch_mod.zeros_like(model.tok_emb.weight)
        raw["rng.torch_cpu"] = torch_mod.zeros(4)
        fresh_torch, fresh, _, _ = self._model()
        del fresh_torch
        with torch_mod.no_grad():
            fresh.tok_emb.weight.add_(1.0)
        digest = load_weights(fresh, raw, expected)
        self.assertEqual(digest, expected)
        self.assertIs(fresh.lm_head.weight, fresh.tok_emb.weight)
        with self.assertRaises(ValueError):
            load_weights(fresh, {"model.lm_head.weight": fresh.tok_emb.weight.detach()}, expected)


class L6ReceiptTests(unittest.TestCase):
    def _row(self, **extra):
        row = {
            "context_id": CTX,
            "arm": "ipo_tau_2",
            "epoch": 8,
            "updates": 2272,
            "seed": 7,
            "l6_cost": 0.9,
        }
        row.update(extra)
        return row

    def test_row_keeps_the_six_g03_fields(self):
        row = require_l6_cost_row(self._row(split="full_val"))
        self.assertEqual(tuple(field for field in L6_ROW_FIELDS), L6_ROW_FIELDS)
        for field in L6_ROW_FIELDS:
            self.assertIn(field, row)
        self.assertEqual(row["epoch"], 8)
        self.assertEqual(row["updates"], 2272)
        self.assertEqual(row["seed"], 7)

    def test_each_missing_field_is_refused(self):
        for field in L6_ROW_FIELDS:
            row = self._row()
            del row[field]
            with self.assertRaises(ValueError):
                require_l6_cost_row(row)

    def test_nonfinite_cost_and_bool_counts_are_refused(self):
        for bad in (
            self._row(l6_cost=float("nan")),
            self._row(l6_cost=True),
            self._row(epoch=True),
            self._row(updates=-1),
            self._row(seed=True),
            self._row(context_id=""),
            self._row(arm=""),
        ):
            with self.assertRaises(ValueError):
                require_l6_cost_row(bad)

    def test_receipt_without_rows_is_refused(self):
        with self.assertRaises(ValueError):
            require_g03_receipt({"l6_row_fields": list(L6_ROW_FIELDS), "l6_cost_mean": 0.9})
        with self.assertRaises(ValueError):
            require_g03_receipt({"l6_row_fields": ["l6_cost"], "l6_rows": [self._row()]})

    def test_receipt_carries_every_context_row(self):
        frozen = self._row(arm="frozen_baseline", epoch=0, updates=0, l6_cost=0.5)
        trained = self._row()
        receipt = require_g03_receipt(
            {"l6_row_fields": list(L6_ROW_FIELDS), "l6_rows": [frozen, trained]}
        )
        self.assertEqual(len(receipt["l6_rows"]), 2)
        self.assertEqual(
            [(row["context_id"], row["arm"], row["epoch"], row["updates"], row["seed"]) for row in receipt["l6_rows"]],
            [(CTX, "frozen_baseline", 0, 0, 7), (CTX, "ipo_tau_2", 8, 2272, 7)],
        )
        self.assertTrue(all(math.isfinite(row["l6_cost"]) for row in receipt["l6_rows"]))

    def test_duplicate_identity_is_refused(self):
        first = self._row(l6_cost=0.5)
        same = self._row(l6_cost=0.5)
        conflict = self._row(l6_cost=0.9)
        with self.assertRaises(ValueError):
            require_g03_receipt(
                {"l6_row_fields": list(L6_ROW_FIELDS), "l6_rows": [first, same]}
            )
        with self.assertRaises(ValueError):
            require_g03_receipt(
                {"l6_row_fields": list(L6_ROW_FIELDS), "l6_rows": [first, conflict]}
            )


if __name__ == "__main__":
    unittest.main()
