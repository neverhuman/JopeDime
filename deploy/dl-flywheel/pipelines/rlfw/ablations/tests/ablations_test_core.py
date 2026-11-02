"""Core deterministic C11 ablation fixtures, collected through test_ablations."""

import inspect
import math
import unittest

from rlfw.ablations.tests.test_ablations import CTX, EVAL, LIN, SHA, SHA_B, _row
from rlfw.ablations.arena_row import measurement_from_slot, uniform_mu
from rlfw.ablations.catalog import CatalogTrie
from rlfw.ablations.compare import ci_verdict, lineage_deltas, score_context
from rlfw.ablations.exact import effective_support, j_b, kl_divergence
from rlfw.ablations.gumbel import gumbel_topk
from rlfw.ablations.measured import build_pairs, checked_measurement
from rlfw.ablations.ope import self_normalized_is, uniform_trie_ope
from rlfw.ablations.puct import puct_select
from rlfw.trie_logprob import child_logprob

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
