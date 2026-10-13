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


class MeasuredTests(unittest.TestCase):
    def test_pair_uses_measured_margin_only(self):
        pairs, refused = build_pairs(
            [_row(SHA, 2_000_000), _row(SHA_B, 0, tokens=(1, 4, 2))], 0.5
        )
        self.assertEqual(refused, {})
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["margin"], 2.0)
        self.assertEqual(pairs[0]["winner"], SHA)

    def test_development_and_predicted_never_become_targets(self):
        rows = [
            _row(SHA, 2_000_000, role="development", subrole="D-tune"),
            _row(SHA_B, 0, role="development", subrole="D-confirm", tokens=(1, 4, 2)),
            _row("f" * 64, 100, role="sealed"),
            _row("1" * 64, 100, q_hat=0.2),
            _row("2" * 64, 100, fitness=1.0),
            _row("3" * 64, 100, outcome="unknown"),
            _row("4" * 64, 100, parity_verified=False),
        ]
        pairs, refused = build_pairs(rows, 0.0)
        self.assertEqual(pairs, [])
        self.assertGreater(refused["role"], 0)
        self.assertGreater(refused["invalid"], 0)

    def test_conflicting_q_grid_drops_the_context(self):
        pairs, refused = build_pairs(
            [_row(SHA, 10), _row(SHA, 11), _row(SHA_B, 0, tokens=(1, 4, 2))], 0.0
        )
        self.assertEqual(pairs, [])
        self.assertEqual(refused["conflicting_q"], 1)

    def test_q_must_match_integer_grid(self):
        with self.assertRaises(ValueError):
            checked_measurement(_row(SHA, 335805, q=0.9))

    def test_normalized_row_round_trips(self):
        normalized = checked_measurement(_row(SHA, 335805))
        again = checked_measurement(normalized)
        self.assertEqual(again["q_grid"], 335805)
        self.assertEqual(again["role"], "train")

    def test_nonfinite_threshold_refuses_pairs(self):
        rows = [_row(SHA, 2_000_000), _row(SHA_B, 0, tokens=(1, 4, 2))]
        for bad in (float("nan"), float("inf"), float("-inf"), None):
            with self.assertRaises(ValueError):
                build_pairs(rows, bad)
        pairs, refused = build_pairs(rows, 0.0)
        self.assertEqual(refused, {})
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["winner"], SHA)
        self.assertAlmostEqual(pairs[0]["margin"], 2.0)


class LossTests(unittest.TestCase):
    def test_ipo_and_dpo_at_zero_margin(self):
        self.assertAlmostEqual(ipo_loss(0.0, 2), 0.0625)
        self.assertAlmostEqual(dpo_loss(0.0), math.log(2))
        with self.assertRaises(ValueError):
            ipo_loss(0.0, 3)
        with self.assertRaises(ValueError):
            dpo_loss(0.0, beta=0.2)


class ExactTests(unittest.TestCase):
    def test_j_b_two_atoms(self):
        self.assertAlmostEqual(j_b([0.0, 1.0], [0.5, 0.5], 0.0, 1), 0.5)
        self.assertAlmostEqual(j_b([0.0, 1.0], [0.5, 0.5], 0.0, 2), 0.75)

    def test_support_and_kl(self):
        self.assertAlmostEqual(effective_support([0.25, 0.25, 0.25, 0.25]), 4.0)
        self.assertAlmostEqual(kl_divergence([0.5, 0.5], [0.5, 0.5]), 0.0)
        with self.assertRaises(ValueError):
            kl_divergence([1.0, 0.0], [0.0, 1.0])


class SearchTests(unittest.TestCase):
    def test_puct_reveals_only_the_selected_leaf(self):
        programs = [(1, 3, 2), (1, 4, 2)]
        priors = [0.2, 0.8]
        seen = []

        def q_of(index):
            seen.append(index)
            return float(index)

        one = puct_select(programs, priors, q_of, 1)
        self.assertEqual(one["selected"], [1])
        self.assertEqual(seen, [1])
        self.assertTrue(one["backs_up_selected_measured_q"])
        self.assertFalse(one["reads_unselected_q"])
        seen.clear()
        both = puct_select(programs, priors, q_of, 2)
        self.assertEqual(sorted(both["selected"]), [0, 1])
        self.assertEqual(seen, both["selected"])
        with self.assertRaises(ValueError):
            puct_select(programs, priors, q_of, 1, c=2.0)

    def test_selected_q_backup_changes_the_next_program(self):
        programs = [(1, 10, 2), (1, 11, 2), (1, 11, 3, 2)]
        priors = [1 / 3, 1 / 3, 1 / 3]

        def run(first_q):
            seen = []

            def q_of(index):
                seen.append(index)
                return first_q if index == 1 else 0.0

            chosen = puct_select(programs, priors, q_of, 2)
            return chosen["selected"], seen

        low, seen_low = run(-1.0)
        high, seen_high = run(1.0)
        self.assertEqual(low[0], 1)
        self.assertEqual(high[0], 1)
        self.assertEqual(low[1], 0)
        self.assertEqual(high[1], 2)
        self.assertEqual(seen_low, low)
        self.assertEqual(seen_high, high)

    def test_gumbel_is_seeded_and_does_not_take_q(self):
        self.assertNotIn("q", inspect.signature(gumbel_topk).parameters)
        priors = [0.8, 0.2]
        self.assertEqual(gumbel_topk(priors, 2, seed=7), gumbel_topk(priors, 2, seed=7))
        self.assertEqual(len(set(gumbel_topk(priors, 2, seed=7))), 2)

    def test_single_child_logprob_is_zero(self):
        trie = CatalogTrie([(1, 3, 2)])
        self.assertEqual(child_logprob([0.0, 0.0, 0.0, 4.0], trie.get((1,)), 3, 1.0), 0.0)


class OpeTests(unittest.TestCase):
    def test_ess_gate(self):
        reported = self_normalized_is([1.0] * 100, [1.0] * 100)
        self.assertEqual(reported["ess"], 100)
        self.assertEqual(reported["value"], 1.0)
        self.assertFalse(reported["gate"])
        self.assertEqual(reported["verdict"], "REPORTED")
        small = self_normalized_is([1.0] * 50, [1.0] * 50)
        self.assertIsNone(small["value"])
        self.assertEqual(small["verdict"], "INCONCLUSIVE")
        self.assertFalse(small["gate"])

    def test_non_randomized_and_missing_mu_are_inconclusive(self):
        draws = [
            {"arm": "rerank32:current_latest", "reward": 1.0, "pi": 0.1, "mu": None},
            {
                "arm": "uniform_trie",
                "reward": 1.0,
                "pi": None,
                "mu": None,
                "propensity_source": "arm_definition_uniform_trie",
            },
        ]
        report = uniform_trie_ope(draws, 6640)
        self.assertIsNone(report["value"])
        self.assertEqual(report["verdict"], "INCONCLUSIVE")
        self.assertFalse(report["gate"])
        self.assertGreater(report["dropped"]["non_randomized"], 0)
        self.assertGreater(report["dropped"]["missing_mu"], 0)

    def test_underflowing_log_pi_still_has_ess(self):
        draw = {
            "arm": "uniform_trie",
            "reward": 1.0,
            "log_pi": -800.0,
            "mu": 1 / 6640,
            "propensity_source": "arm_definition_uniform_trie",
            "in_catalog": True,
        }
        report = uniform_trie_ope([dict(draw) for _ in range(100)], 6640)
        self.assertEqual(report["ess"], 100)
        self.assertEqual(report["value"], 1.0)
        self.assertFalse(report["gate"])


class CompareTests(unittest.TestCase):
    def _programs(self):
        return [
            {"program_sha256": SHA, "tokens": (1, i, 2), "q": float(i) / 10, "logprob": None}
            for i in range(3, 11)
        ]

    def test_ci_touching_zero_is_inconclusive(self):
        self.assertEqual(ci_verdict((0.0, 1.0)), "INCONCLUSIVE")
        self.assertEqual(ci_verdict((-1.0, 0.0)), "INCONCLUSIVE")
        self.assertEqual(ci_verdict((0.1, 1.0)), "GREATER")
        self.assertEqual(ci_verdict((-2.0, -0.1)), "LESS")
        self.assertEqual(ci_verdict((None, None)), "INCONCLUSIVE")

    def test_lineage_cluster_and_no_promotion(self):
        mixed = score_context(
            {
                "context_id": CTX,
                "lineage_id": LIN,
                "subrole": "D-tune",
                "incumbent_q": 0.0,
                "programs": [
                    {"program_sha256": SHA, "tokens": (1, 3, 2), "q": 1.0, "logprob": None},
                    {"program_sha256": SHA_B, "tokens": (1, 4, 2), "q": -1.0, "logprob": None},
                ],
                "charged": {},
            },
            budget=2,
        )
        other = dict(mixed)
        other["context_id"] = "9" * 64
        other["lineage_id"] = "8" * 64
        other["arms"] = {
            "puct": {"delta_vs_incumbent": -1.0},
            "incumbent": {"delta_vs_incumbent": 0.0},
        }
        # The first context's PUCT delta is positive; the second lineage is -1.
        mixed["arms"]["puct"]["delta_vs_incumbent"] = 1.0
        report = lineage_deltas([mixed, other], "puct", subrole="D-tune")
        self.assertEqual(report["verdict"], "INCONCLUSIVE")
        self.assertFalse(report["promoted"])
        both = lineage_deltas(
            [
                {"excluded": None, "subrole": "D-tune", "lineage_id": LIN, "arms": {"puct": {"delta_vs_incumbent": 1.0}}},
                {"excluded": None, "subrole": "D-tune", "lineage_id": "8" * 64, "arms": {"puct": {"delta_vs_incumbent": 1.0}}},
            ],
            "puct",
            subrole="D-tune",
        )
        self.assertEqual(both["verdict"], "GREATER")
        self.assertFalse(both["promoted"])
        self.assertEqual(both["ci95"], [1.0, 1.0])

    def test_short_support_is_excluded(self):
        row = score_context(
            {
                "context_id": CTX,
                "lineage_id": LIN,
                "subrole": "D-confirm",
                "incumbent_q": 0.0,
                "programs": self._programs()[:7],
                "charged": {"uniform_trie": [0.1] * 8},
            }
        )
        self.assertEqual(row["excluded"], "measured_support_below_budget")


class ArenaRowTests(unittest.TestCase):
    def test_uniform_propensity_comes_from_the_arm_definition(self):
        mu = uniform_mu(1 / 6640)
        self.assertEqual(mu["propensity_source"], "arm_definition_uniform_trie")
        self.assertAlmostEqual(mu["mu"], 1 / 6640)
        with self.assertRaises(ValueError):
            uniform_mu(None)

    def test_slot_evidence_mismatch_is_refused(self):
        payload = {
            "q_grid": 10,
            "program_sha256": SHA,
            "outcome": "qualified_complete",
            "qualified": True,
            "context_id": CTX,
            "lineage_id": LIN,
            "arm": "uniform_trie",
            "seed": 1,
            "slot": 1,
        }
        evidence = {
            "q_grid": 11,
            "program_sha256": SHA,
            "outcome_class": "qualified_complete",
            "q": 1e-5,
            "parity_verified": True,
            "closure_verified": True,
            "parity_max_normalized_error": 0.0,
            "evaluator_sha256": EVAL,
            "protocol": "jope.flywheel-s1-inner-rmse/v1",
        }
        with self.assertRaises(ValueError):
            measurement_from_slot(payload, evidence, (1, 3, 2), "development", "D-tune")


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
