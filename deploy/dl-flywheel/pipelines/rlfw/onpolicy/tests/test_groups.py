"""Acceptance and rejection of reward closure, frozen rounds and IS."""

import math
import unittest
from dataclasses import FrozenInstanceError, replace

from rlfw.onpolicy.collector import QUARANTINE, RoundCollector
from rlfw.onpolicy.groups import Draw, ResolvedGroup, UtilityRule, VerifiedQ, verify_q
from rlfw.onpolicy.objective import (
    objective,
    parent_macro_mean,
    pkpo_weights,
    rloo_weights,
    staleness_gate,
)
from rlfw.onpolicy.tests.fixtures import (
    SHA,
    fixture_publication,
    group_for,
    logs,
    q_for,
)
from rlfw.trie_logprob import TrieLogProb


class GroupTests(unittest.TestCase):
    def test_N1_refuses_unmeasured_fitness_parity_unknown_and_failures(self):
        key, publication, verify = fixture_publication(0.3)
        for outcome in (*QUARANTINE, "incompatible", "none"):
            with self.subTest(outcome=outcome), self.assertRaises(ValueError):
                verify_q(
                    key,
                    replace(publication, outcome=outcome),
                    parent="p",
                    role_check=lambda k, p: True,
                    verify_closure=verify,
                )
        for source in ("predicted", "fitness_only", "historical", "g0_not_ready"):
            with self.assertRaises(ValueError):
                verify_q(
                    key,
                    replace(publication, source=source),
                    parent="p",
                    role_check=lambda k, p: True,
                    verify_closure=verify,
                )

    def test_closure_must_verify_bodies_parity_comparison_and_raw_grid(self):
        key, p, verify = fixture_publication(0.3)
        for bad in (
            replace(p, q_grid=0.3),
            replace(p, q_grid=True),
            replace(p, q_grid=999),
            replace(p, hwm_key="0" * 64),
            replace(p, artifacts={}),
        ):
            with self.assertRaises(ValueError):
                verify_q(
                    key,
                    bad,
                    parent="p",
                    role_check=lambda k, p: True,
                    verify_closure=verify,
                )
        p.artifacts["rust_parity"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            verify_q(
                key, p, parent="p", role_check=lambda k, p: True, verify_closure=verify
            )

    def test_train_role_is_fresh_and_required_and_verification_is_immutable(self):
        key, p, verify = fixture_publication(0.3)
        for role in (False, None, "train"):
            with self.assertRaises(ValueError):
                verify_q(
                    key,
                    p,
                    parent="p",
                    role_check=lambda k, p: role,
                    verify_closure=verify,
                )
        with self.assertRaisesRegex(ValueError, "verify_q"):
            VerifiedQ(key.digest, "c", "p", SHA, "exec", 1, SHA, object())

        def mutating(k, publication):
            publication.artifacts["input"]["bytes"] += 1
            return True

        with self.assertRaisesRegex(ValueError, "changed"):
            verify_q(
                key,
                p,
                parent="p",
                role_check=lambda k, p: True,
                verify_closure=mutating,
            )
        q = q_for()
        with self.assertRaises(FrozenInstanceError):
            q.q_grid = 0

    def test_raw_clipped_mismatch_refused_and_raw_q_retained(self):
        g = group_for((8.0, 10.0), h=8.5)
        with self.assertRaisesRegex(ValueError, "mixed"):
            replace(g.spec, incumbent_rule=UtilityRule("raw"))
        self.assertEqual(g.outcomes[0].raw, 8.0)
        self.assertEqual(g.utilities, (math.log(1000),) * 2)
        self.assertEqual(g.h, math.log(1000))
        t = objective(
            g,
            g.spec.policy,
            tuple(d.behavior for d in g.spec.draws),
            update=10,
            seconds=101,
            probe_kl=0,
            role_check=lambda spec: True,
        )
        self.assertEqual(t.hwm_status, "solved")
        raw = ResolvedGroup(
            replace(
                g.spec,
                sample_rule=UtilityRule("raw"),
                incumbent_rule=UtilityRule("raw"),
            ),
            g.outcomes,
        )
        t = objective(
            raw,
            raw.spec.policy,
            tuple(d.behavior for d in raw.spec.draws),
            update=10,
            seconds=101,
            probe_kl=0,
            role_check=lambda spec: True,
        )
        self.assertGreater(max(t.hwm), 0)

    def test_actor_role_change_after_result_verification_is_refused(self):
        g = group_for()
        with self.assertRaisesRegex(ValueError, "train-role"):
            objective(
                g,
                g.spec.policy,
                tuple(d.behavior for d in g.spec.draws),
                update=10,
                seconds=101,
                probe_kl=0,
                role_check=lambda spec: False,
            )

    def test_no_support_or_unknown_incumbent_is_dense_only(self):
        g = group_for(h=None)
        t = objective(
            g,
            g.spec.policy,
            tuple(d.behavior for d in g.spec.draws),
            update=10,
            seconds=101,
            probe_kl=0,
            role_check=lambda spec: True,
        )
        self.assertEqual(t.hwm_status, "no_comparable_incumbent")
        self.assertEqual(t.hwm, (0.0, 0.0))
        self.assertNotEqual(t.dense, (0.0, 0.0))

    def test_incomplete_group_and_incompatible_comparison_are_refused(self):
        g = group_for()
        with self.assertRaisesRegex(ValueError, "shrunk"):
            ResolvedGroup(g.spec, g.outcomes[:1])
        for bad in (
            q_for(context="other"),
            q_for(parent="other"),
            q_for(program="0" * 64),
        ):
            with self.assertRaises(ValueError):
                ResolvedGroup(g.spec, (bad, g.outcomes[1]))
        for sampling in ("dedup", "rerank", "mutation", "archive"):
            with self.assertRaises(ValueError):
                replace(g.spec, sampling=sampling)

    def test_repeated_program_draws_keep_multiplicity_and_exact_delivery(self):
        g = group_for()
        draw = Draw("d2", g.spec.draws[0].program, g.spec.draws[0].behavior)
        spec = replace(g.spec, draws=(g.spec.draws[0], draw))
        c = RoundCollector("r", (spec,))
        for d in spec.draws:
            c.resolve("g", d.draw_id, g.outcomes[0])
            c.resolve("g", d.draw_id, g.outcomes[0])
        result = c.close()
        self.assertEqual(len(result.groups[0].outcomes), 2)
        self.assertEqual(result.groups[0].outcomes, (g.outcomes[0],) * 2)

    def test_incomplete_and_unqualified_groups_quarantined_whole(self):
        g = group_for()
        for reason in (None, *QUARANTINE):
            c = RoundCollector("r", (g.spec,))
            c.resolve("g", "0", g.outcomes[0])
            if reason:
                c.resolve("g", "1", reason)
            result = c.close()
            self.assertEqual(result.groups, ())
            self.assertEqual(result.quarantined, (("g", (reason or "incomplete",)),))

    def test_one_step_per_round_frozen_checkpoint_and_incumbent(self):
        g = group_for()
        c = RoundCollector("r", (g.spec,))
        for d, q in zip(g.spec.draws, g.outcomes):
            c.resolve("g", d.draw_id, q)
        result = c.close()
        self.assertEqual(result.groups[0].h, g.h)
        self.assertEqual(result.quarantined, ())
        with self.assertRaises(ValueError):
            c.close()
        with self.assertRaises(ValueError):
            c.resolve("g", "1", "infra_unknown")
        with self.assertRaises(ValueError):
            RoundCollector(
                "r",
                (
                    g.spec,
                    replace(
                        g.spec,
                        group_id="g2",
                        policy=replace(g.spec.policy, checkpoint="0" * 64),
                    ),
                ),
            )

    def test_terminal_failure_cannot_be_replaced_or_retried(self):
        g = group_for()
        c = RoundCollector("r", (g.spec,))
        c.resolve("g", "0", "infra_unknown")
        with self.assertRaises(ValueError):
            c.resolve("g", "0", g.outcomes[0])
        with self.assertRaises(ValueError):
            c.resolve("g", "absent", g.outcomes[0])

    def test_parent_macro_reduction_does_not_weight_sibling_or_group_volume(self):
        rows = [("p0", "c0", 1.0), ("p0", "c0", 3.0), ("p1", "c1", 6.0)]
        self.assertEqual(parent_macro_mean(rows), 4.0)
        self.assertEqual(parent_macro_mean(rows + [("p0", "sibling", 2.0)] * 100), 4.0)


class StalenessTests(unittest.TestCase):
    def gate(self, g=None, policy=None, readouts=None, **kwargs):
        g = g or group_for()
        return staleness_gate(
            g,
            policy or g.spec.policy,
            readouts or tuple(d.behavior for d in g.spec.draws),
            **({"update": 10, "seconds": 101, "probe_kl": 0.0} | kwargs),
        )

    def test_lag_tags_fresh_boundaries_and_exact_IS_outside_window(self):
        g = group_for()
        current = replace(g.spec.policy, checkpoint="0" * 64)
        boundary = self.gate(g, current, update=74, seconds=280)
        self.assertTrue(boundary.accepted)
        self.assertEqual(
            (boundary.lag_updates, boundary.lag_seconds, boundary.correction),
            (64, 180.0, "IS_in_window"),
        )
        old = self.gate(g, current, update=75, seconds=281)
        self.assertTrue(old.accepted)
        self.assertEqual(old.correction, "IS_outside_window")
        self.assertEqual(old.ess, 2.0)

    def test_all_support_identity_fields_must_match(self):
        g = group_for()
        for field in ("trie", "vocabulary", "normalizer", "protocol", "temperature"):
            current = replace(
                g.spec.policy, **{field: 0.7 if field == "temperature" else "0" * 64}
            )
            self.assertEqual(self.gate(g, current).reason, "support_mismatch")

    def test_ratio_boundaries_are_inclusive_unclipped_and_full_group(self):
        g = group_for(probabilities=(0.5, 0.5))
        current = replace(g.spec.policy, checkpoint="0" * 64)
        boundary = self.gate(g, current, logs((0.25, 1.0)), update=11)
        self.assertTrue(boundary.accepted)
        self.assertEqual(boundary.ratios, (0.5, 2.0))
        self.assertEqual(boundary.joint_ratio, 1.0)
        for p in ((0.249, 0.751), (1e-300, 0.5)):
            bad = self.gate(g, current, logs(p), update=11)
            self.assertFalse(bad.accepted)
            self.assertEqual(bad.reason, "ratio_outside_[0.5,2]")

    def test_KL_stop_and_same_checkpoint_trace_drift_refused(self):
        self.assertFalse(self.gate(probe_kl=0.05).accepted)
        self.assertTrue(self.gate(probe_kl=0.03).accepted)
        self.assertEqual(
            self.gate(readouts=logs((0.5, 0.5))).reason,
            "same_checkpoint_trace_mismatch",
        )
        self.assertEqual(self.gate(update=11).reason, "same_checkpoint_trace_mismatch")
        g = group_for()
        self.assertEqual(
            self.gate(g, replace(g.spec.policy, checkpoint="0" * 64)).reason,
            "checkpoint_changed_without_update",
        )

    def test_negative_or_nonfinite_lag_KL_missing_trace_and_grammar_refused(self):
        for kwargs in (
            {"update": 9},
            {"seconds": 99},
            {"probe_kl": math.nan},
            {"probe_kl": -0.1},
        ):
            with self.assertRaises(ValueError):
                self.gate(**kwargs)
        with self.assertRaises(ValueError):
            self.gate(readouts=logs((0.5,)))
        with self.assertRaises(ValueError):
            TrieLogProb(-1, (-1,), aggregation="mean")

    def test_objective_uses_item_IS_dense_and_joint_IS_HWM(self):
        g = group_for()
        current = replace(g.spec.policy, checkpoint="0" * 64)
        readouts = logs((0.5, 0.5))
        terms = objective(
            g,
            current,
            readouts,
            update=11,
            seconds=110,
            probe_kl=0.01,
            role_check=lambda spec: True,
        )
        item = (0.5 / 0.4, 0.5 / 0.6)
        self.assertEqual(
            terms.dense, tuple(w * r for w, r in zip(rloo_weights(g.utilities), item))
        )
        self.assertEqual(
            terms.hwm,
            tuple(w * math.prod(item) for w in pkpo_weights(g.utilities, g.h, 2)),
        )
        dense, hwm = terms.losses([x.total for x in readouts])
        self.assertAlmostEqual(
            dense, -sum(w * x.total for w, x in zip(terms.dense, readouts))
        )
        self.assertAlmostEqual(
            hwm, -sum(w * x.total for w, x in zip(terms.hwm, readouts))
        )
        with self.assertRaises(ValueError):
            terms.losses((-1.0, -1.0))
        with self.assertRaises(ValueError):
            objective(
                g,
                current,
                logs((0.05, 0.95)),
                update=11,
                seconds=110,
                probe_kl=0.01,
                role_check=lambda spec: True,
            )


if __name__ == "__main__":
    unittest.main()
