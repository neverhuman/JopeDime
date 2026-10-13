"""C4 exact gradients and GB-03 archive/frozen-round CPU gates; synthetic q closure only."""

import hashlib
import importlib.util
import itertools
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

import rlfw
from rlfw.onpolicy.archive import archive_target
from rlfw.onpolicy.frozen_round import ActorState, FrozenRoundStep
from rlfw.onpolicy.groups import Draw, GroupSpec, PolicyIdentity, ResolvedGroup
from rlfw.onpolicy.objective import objective
from rlfw.onpolicy.tests.fixtures import SHA, group_for, logs, q_for
from rlfw.onpolicy.tests.test_math import H, Q, Z, analytic, softmax
from rlfw.trie_logprob import TrieLogProb


class AutogradTests(unittest.TestCase):
    def test_actual_tensor_loss_gradient_matches_exact_mean_and_best_of_B(self):
        torch.set_num_threads(1)
        p = softmax(Z)
        policy = PolicyIdentity(SHA, SHA, SHA, SHA, SHA)
        rewards = tuple(q_for(q, program=str(i) * 64) for i, q in enumerate(Q))
        incumbent = q_for(H)
        mean = math.fsum(x * y for x, y in zip(p, Q))
        expected_dense = [-p[i] * (Q[i] - mean) for i in range(3)]
        for b, g in ((1, 1), (1, 2), (2, 4), (4, 4)):
            z = torch.tensor(Z, dtype=torch.float64, requires_grad=True)
            current_lp = torch.log_softmax(z, dim=0)
            dense_sum = current_lp.sum() * 0
            hwm_sum = current_lp.sum() * 0
            for actions in itertools.product(range(3), repeat=g):
                outcomes = tuple(rewards[i] for i in actions)
                readouts = logs([p[i] for i in actions])
                draws = tuple(Draw(str(j), rewards[i].program, readouts[j]) for j, i in enumerate(actions))
                spec = GroupSpec(
                    "g",
                    "p",
                    "c",
                    incumbent.comparison,
                    policy,
                    0,
                    0,
                    draws,
                    b,
                    incumbent,
                )
                terms = objective(
                    ResolvedGroup(spec, outcomes),
                    policy,
                    readouts,
                    update=0,
                    seconds=1,
                    probe_kl=0,
                    role_check=lambda spec: True,
                )
                dense, hwm = terms.losses(tuple(current_lp[i] for i in actions))
                mass = math.prod(p[i] for i in actions)
                dense_sum = dense_sum + mass * dense
                hwm_sum = hwm_sum + mass * hwm
            dense_grad = torch.autograd.grad(dense_sum, z, retain_graph=True)[0]
            hwm_grad = torch.autograd.grad(hwm_sum, z)[0]
            self.assertLess(
                max(abs(float(x) - y) for x, y in zip(dense_grad, expected_dense)),
                1e-12,
            )
            self.assertLess(
                max(abs(float(x) + y) for x, y in zip(hwm_grad, analytic(Z, Q, H, b)[1])),
                1e-12,
            )

    def test_joint_IS_coefficients_are_detached_and_empty_HWM_is_connected_zero(self):
        from rlfw.onpolicy.tests.fixtures import group_for

        g = group_for(h=None)
        current = replace(g.spec.policy, checkpoint="0" * 64)
        z = torch.tensor((0.0, 0.0), dtype=torch.float64, requires_grad=True)
        lp = torch.log_softmax(z, dim=0)
        terms = objective(
            g,
            current,
            logs((0.5, 0.5)),
            update=11,
            seconds=101,
            probe_kl=0.01,
            role_check=lambda spec: True,
        )
        dense, hwm = terms.losses(lp)
        self.assertTrue(dense.requires_grad and hwm.requires_grad)
        self.assertEqual(torch.autograd.grad(hwm, z, retain_graph=True)[0].tolist(), [0.0, 0.0])
        gradient = torch.autograd.grad(dense, z)[0]
        expected = [-terms.dense[i] + 0.5 * math.fsum(terms.dense) for i in range(2)]
        self.assertLess(max(abs(float(x) - y) for x, y in zip(gradient, expected)), 1e-12)


MEASURED_GB03_PROBE = {}


def slate():
    return tuple(q_for(q, program=str(i) * 64) for i, q in enumerate((-1.0, 0.0, 1.0)))


class ArchiveTests(unittest.TestCase):
    def target(self, rows=None, **kwargs):
        return archive_target(rows or slate(), tau=0.2, role_check=lambda rows: True, **kwargs)

    def test_high_q_has_high_teacher_mass_and_duplicate_delivery_cannot_bias_it(self):
        rows = slate()
        target = self.target(rows)
        self.assertGreater(target.weights[2], 0.99)
        self.assertAlmostEqual(sum(target.weights), 1.0)
        repeated = self.target((rows[0],) * 100 + rows[1:] + (rows[2],) * 7)
        self.assertEqual(target, repeated)

    def test_archive_order_is_deterministic_and_common_quality_shift_cancels(self):
        rows = slate()
        target = self.target(rows)
        self.assertEqual(target, self.target(rows[::-1]))
        shifted = tuple(q_for(q.raw + 10, program=q.program) for q in rows)
        self.assertEqual(target.weights, self.target(shifted).weights)

    def test_adaptive_archive_reference_and_frozen_pi_reference_are_distinct(self):
        rows = slate()
        policy = PolicyIdentity(SHA, SHA, SHA, SHA, SHA)
        refs = dict(zip((q.program for q in rows), logs((0.7, 0.2, 0.1))))
        weighted = self.target(rows, reference_logprobs=refs, reference_policy=policy)
        self.assertAlmostEqual(weighted.weights[2] / weighted.weights[1], 0.5 * math.exp(5))
        self.assertNotEqual(weighted.weights, self.target(rows).weights)
        self.assertIn("alpha1", weighted.reference)
        self.assertIn("alpha0", self.target(rows).reference)

    def test_numeric_q_predicted_q_and_mixed_comparison_never_form_targets(self):
        for rows in [
            (0.1, 0.2),
            ({"q_hat": 0.1},),
            (slate()[0], q_for(1, context="heldout")),
        ]:
            with self.assertRaises((TypeError, ValueError)):
                self.target(rows)

    def test_conflicting_rewards_for_one_program_are_not_selected_by_maximum(self):
        row = slate()[0]
        with self.assertRaises(ValueError):
            self.target((row, q_for(4, program=row.program)))

    def test_direct_target_construction_cannot_bypass_reward_or_normalization_checks(
        self,
    ):
        target = self.target()
        with self.assertRaises(TypeError):
            replace(target, outcomes=(0.1, 0.2, 0.3))
        with self.assertRaises(TypeError):
            replace(target, weights=list(target.weights))
        for weights in ((-1.0, 1.0, 1.0), (0.2, 0.2, 0.2), (math.nan, 0.0, 1.0)):
            with self.assertRaises(ValueError):
                replace(target, weights=weights)

    def test_nonfinite_temperature_reference_and_changed_role_are_refused(self):
        for tau in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                archive_target(slate(), tau=tau, role_check=lambda rows: True)
        with self.assertRaises(ValueError):
            archive_target(slate(), tau=0.2, role_check=lambda rows: False)
        with self.assertRaises(ValueError):
            self.target(reference_policy=PolicyIdentity(SHA, SHA, SHA, SHA, SHA))

    def test_constructor_freezes_scalar_weights_and_refuses_tensor_teachers(self):
        target = self.target()
        teacher = torch.tensor(
            [-5.0, 0.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        masses = torch.softmax(teacher, 0)
        for weights in (
            tuple(masses),
            tuple(masses.detach()),
            tuple(torch.tensor(w, requires_grad=True) for w in target.weights),
        ):
            with self.assertRaisesRegex(TypeError, "Python-float"):
                replace(target, weights=weights)
        integers = replace(target, weights=(0, 0, 1))
        self.assertEqual(integers.weights, (0.0, 0.0, 1.0))
        self.assertTrue(all(type(w) is float for w in integers.weights))
        frozen = replace(target, weights=tuple(float(w.detach()) for w in masses))
        snapshot = frozen.weights
        masses.detach().fill_(1 / 3)
        self.assertEqual(frozen.weights, snapshot)
        z = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        lp = torch.log_softmax(z, 0)
        traces = {
            q.program: TrieLogProb(float(v.detach()), (float(v.detach()), 0.0))
            for q, v in zip(frozen.outcomes, lp)
        }
        policy = PolicyIdentity(SHA, SHA, SHA, SHA, SHA)
        frozen.loss(
            policy, traces, dict(zip(traces, lp)), role_check=lambda rows: True
        ).backward()
        self.assertIsNone(teacher.grad)
        expected = 1 / 3 - torch.tensor(snapshot, dtype=torch.float64)
        self.assertTrue(torch.allclose(z.grad, expected, atol=1e-15, rtol=0))
        self.assertGreater(float(torch.linalg.vector_norm(z.grad)), 0)
        self.assertEqual(frozen.weights, snapshot)

    def test_sequence_sum_loss_reaches_actor_and_small_step_improves_exact_quality(
        self,
    ):
        target = self.target()
        policy = PolicyIdentity(SHA, SHA, SHA, SHA, SHA)
        z = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        before = torch.softmax(z, 0).detach()
        lp = torch.log_softmax(z, 0)
        traces = {
            q.program: TrieLogProb(float(v.detach()), (float(v.detach()), 0.0))
            for q, v in zip(target.outcomes, lp)
        }
        loss = target.loss(policy, traces, dict(zip(traces, lp)), role_check=lambda rows: True)
        grad = torch.autograd.grad(loss, z)[0]
        self.assertGreater(float(torch.linalg.vector_norm(grad)), 0)
        after = torch.softmax(z.detach() - 0.01 * grad, 0)
        self.assertGreater(float(after[2]), float(before[2]))
        q = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        self.assertGreater(float(after @ q), float(before @ q))

    def test_missing_mean_token_likelihood_or_fresh_role_failure_is_refused(self):
        target = self.target()
        p = PolicyIdentity(SHA, SHA, SHA, SHA, SHA)
        traces = {q.program: TrieLogProb(-2.0, (-1.0, -1.0)) for q in target.outcomes}
        for graph, role in [
            ({k: -1.0 for k in traces}, True),
            ({k: -2.0 for k in traces}, False),
            ({}, True),
        ]:
            with self.assertRaises(ValueError):
                target.loss(p, traces, graph, role_check=lambda rows: role)


class FrozenRoundTests(unittest.TestCase):
    def ready(self, *, scope="measured_leaf_pi_M", group=None):
        group = group or group_for()
        step = FrozenRoundStep("round", (group.spec,), scope=scope)
        for d, q in zip(group.spec.draws, group.outcomes):
            step.resolve(group.spec.group_id, d.draw_id, q)
        return step, group

    def args(self, group, **changes):
        z = torch.log(torch.tensor((0.4, 0.6), dtype=torch.float64)).requires_grad_()
        args = dict(
            state_probe=lambda: ActorState(group.spec.policy, group.spec.generated_update, False),
            recompute=lambda group: (logs((0.4, 0.6)), torch.log_softmax(z, 0)),
            apply_update=lambda loss: None,
            seconds=101.0,
            probe_kl=0.0,
            role_check=lambda group: True,
        )
        args.update(changes)
        return args, z

    def test_real_optimizer_callback_is_once_and_result_keeps_pi_M_label(self):
        step, group = self.ready()
        args, z = self.args(group)
        optimizer = torch.optim.SGD((z,), lr=0.01)
        calls = []
        before = z.detach().clone()

        def update(loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            calls.append(1)

        args["apply_update"] = update
        result = step.step(**args)
        self.assertEqual(calls, [1])
        self.assertFalse(torch.equal(before, z.detach()))
        self.assertEqual(result.scope, "measured_leaf_pi_M")
        self.assertEqual(result.update_callback_calls, 1)
        with self.assertRaises(ValueError):
            step.step(**args)
        with self.assertRaises(ValueError):
            step.resolve(group.spec.group_id, group.spec.draws[0].draw_id, group.outcomes[0])

    def test_incomplete_or_unknown_draw_blocks_whole_wave_without_renormalizing(self):
        group = group_for()
        for failure in (None, "infra_unknown", "parity_unqualified"):
            complete = replace(group.spec, group_id="complete")
            step = FrozenRoundStep("round", (complete, group.spec), scope="full_support_iid")
            for draw, outcome in zip(complete.draws, group.outcomes):
                step.resolve("complete", draw.draw_id, outcome)
            step.resolve("g", "0", group.outcomes[0])
            if failure is not None:
                step.resolve("g", "1", failure)
            calls = []
            args, _ = self.args(group, apply_update=lambda loss: calls.append(1))
            result = step.step(**args)
            self.assertEqual(result.state, "quarantined_no_update")
            self.assertEqual(result.update_callback_calls, 0)
            self.assertEqual(calls, [])

    def test_checkpoint_update_support_and_dropout_drift_block_actor_update(self):
        for state in [
            ActorState(replace(group_for().spec.policy, checkpoint="0" * 64), 10, False),
            ActorState(group_for().spec.policy, 11, False),
            ActorState(replace(group_for().spec.policy, temperature=0.5), 10, False),
            ActorState(group_for().spec.policy, 10, True),
        ]:
            step, group = self.ready()
            calls = []
            args, _ = self.args(
                group,
                state_probe=lambda: state,
                apply_update=lambda loss: calls.append(1),
            )
            with self.assertRaises(ValueError):
                step.step(**args)
            self.assertEqual(calls, [])

    def test_drift_during_recomputation_is_not_corrected_into_frozen_round(self):
        step, group = self.ready()
        state = [ActorState(group.spec.policy, 10, False)]
        args, z = self.args(group, state_probe=lambda: state[0])

        def readout(group):
            state[0] = replace(state[0], update=11)
            return logs((0.4, 0.6)), torch.log_softmax(z, 0)

        args["recompute"] = readout
        with self.assertRaises(ValueError):
            step.step(**args)

    def test_role_and_KL_guard_failures_consume_round_without_callback(self):
        for options in ({"role_check": lambda group: False}, {"probe_kl": 0.05}):
            step, group = self.ready()
            calls = []
            args, _ = self.args(group, apply_update=lambda loss: calls.append(1), **options)
            with self.assertRaises(ValueError):
                step.step(**args)
            self.assertEqual(calls, [])
            with self.assertRaises(ValueError):
                step.step(**args)

    def test_no_implicit_onpolicy_scope_or_zero_objective(self):
        with self.assertRaises(ValueError):
            FrozenRoundStep("r", (group_for().spec,), scope="onpolicy")
        step, group = self.ready()
        args, _ = self.args(group, lambda_rloo=0, lambda_hwm=0)
        with self.assertRaises(ValueError):
            step.step(**args)

    def test_partial_optimizer_callback_failure_is_never_retried(self):
        step, group = self.ready()
        calls = []

        def failure(loss):
            calls.append(1)
            raise RuntimeError("optimizer outcome unknown")

        args, _ = self.args(group, apply_update=failure)
        with self.assertRaises(RuntimeError):
            step.step(**args)
        with self.assertRaises(ValueError):
            step.step(**args)
        self.assertEqual(calls, [1])


class ActualActorTests(unittest.TestCase):
    def test_fresh_actual_actor_draws_receive_one_frozen_RLOO_update(self):
        root = Path(rlfw.__file__).resolve().parents[1] / "continuous/consumer/dltrain"
        name = "gb03_fresh_fixture_dltrain"
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
        module = __import__(name + ".model", fromlist=["PolicyLM"])
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = module.PolicyLM(module.PolicyConfig(6, 4, 12, 1, 3, 24, 3)).double().eval()
        tokens = torch.tensor([[1, 3, 2], [1, 4, 2], [1, 5, 2]])

        def readout():
            output = model(
                tokens,
                torch.tensor([3, 3, 3]),
                torch.tensor([[0.2, 0.1, 0.3, 0.5]], dtype=torch.float64),
                torch.tensor([0, 0, 0]),
            )
            return torch.log_softmax(output.logits[0, 0, 3:6], 0)

        def digest_parameters():
            digest = hashlib.sha256()
            for name, p in sorted(model.named_parameters()):
                digest.update(name.encode() + b"\0")
                digest.update(p.detach().numpy().astype("<f8").tobytes())
            return digest.hexdigest()

        rows = slate()
        generated_lp = readout()
        before = generated_lp.detach().exp()
        generated_hash = digest_parameters()
        policy = PolicyIdentity(generated_hash, SHA, SHA, SHA, SHA)
        selected = torch.multinomial(
            before, 16, replacement=True, generator=torch.Generator().manual_seed(13)
        ).tolist()
        traces = tuple(
            TrieLogProb(float(generated_lp[i].detach()), (float(generated_lp[i].detach()), 0.0))
            for i in selected
        )
        draws = tuple(Draw(str(j), rows[i].program, traces[j]) for j, i in enumerate(selected))
        group = GroupSpec("fresh", "p", "c", rows[0].comparison, policy, 0, 0.0, draws, 8, None)
        step = FrozenRoundStep("fresh-actor-round", (group,), scope="full_support_iid")
        for d, i in zip(draws, selected):
            step.resolve("fresh", d.draw_id, rows[i])
        counts = [0]
        gradient = {}

        def probe():
            return ActorState(
                replace(policy, checkpoint=digest_parameters()),
                counts[0],
                model.training,
            )

        def recompute(group):
            lp = readout()
            linked = tuple(lp[i] for i in selected)
            return tuple(TrieLogProb(float(x.detach()), (float(x.detach()), 0.0)) for x in linked), linked

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        def update(loss):
            named = list(model.named_parameters())
            gs = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
            gradient["actor_applied_0.01"] = math.sqrt(
                sum(
                    float((g * g).sum())
                    for (n, _), g in zip(named, gs)
                    if n == "tok_emb.weight" and g is not None
                )
            )
            gradient["shared_trunk_applied_0.01"] = math.sqrt(
                sum(
                    float((g * g).sum())
                    for (n, _), g in zip(named, gs)
                    if n.startswith(("ctx_mlp.", "blocks.", "ln_f.", "pos_emb.")) and g is not None
                )
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counts[0] += 1

        result = step.step(
            state_probe=probe,
            recompute=recompute,
            apply_update=update,
            seconds=1.0,
            probe_kl=0.0,
            role_check=lambda group: True,
            lambda_rloo=0.01,
        )
        with torch.no_grad():
            after = readout().exp()
        quality = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        self.assertEqual(counts, [1])
        self.assertEqual(len(draws), 16)
        self.assertGreater(float(after @ quality), float(before @ quality))
        self.assertGreater(gradient["actor_applied_0.01"], 0)
        self.assertGreater(gradient["shared_trunk_applied_0.01"], 0)
        self.assertNotEqual(generated_hash, digest_parameters())
        self.assertEqual(result.scope, "full_support_iid")
        with self.assertRaises(ValueError):
            step.step(
                state_probe=probe,
                recompute=recompute,
                apply_update=update,
                seconds=2.0,
                probe_kl=0.0,
                role_check=lambda group: True,
            )
        MEASURED_GB03_PROBE["fresh_frozen_RLOO"] = {
            "fixture_only": True,
            "scope": "full-support iid on synthetic three-program trie; not production full support",
            "raw_draws": 16,
            "unique_programs": len(set(selected)),
            "optimizer_updates": counts[0],
            "parameter_before": generated_hash,
            "parameter_after": digest_parameters(),
            "gradient_norms": gradient,
            "top_q_probability_before": float(before[2]),
            "top_q_probability_after": float(after[2]),
            "exact_expected_q_before": float(before @ quality),
            "exact_expected_q_after": float(after @ quality),
            "production_qualification": False,
        }

    def test_actual_PolicyLM_archive_gradient_and_top_q_probability_movement(self):
        root = Path(rlfw.__file__).resolve().parents[1] / "continuous/consumer/dltrain"
        name = "gb03_fixture_dltrain"
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
        module = __import__(name + ".model", fromlist=["PolicyLM"])
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = module.PolicyLM(module.PolicyConfig(6, 4, 12, 1, 3, 24, 3)).double().eval()
        self.assertIs(model.lm_head.weight, model.tok_emb.weight)
        tokens = torch.tensor([[1, 3, 2], [1, 4, 2], [1, 5, 2]])

        def readout():
            output = model(
                tokens,
                torch.tensor([3, 3, 3]),
                torch.tensor([[0.2, 0.1, 0.3, 0.5]], dtype=torch.float64),
                torch.tensor([0, 0, 0]),
            )
            return torch.log_softmax(output.logits[0, 0, 3:6], 0)

        target = archive_target(slate(), tau=0.2, role_check=lambda rows: True)
        policy = PolicyIdentity(SHA, SHA, SHA, SHA, SHA)
        lp = readout()
        before = lp.detach().exp()
        traces = {
            q.program: TrieLogProb(float(v.detach()), (float(v.detach()), 0.0))
            for q, v in zip(target.outcomes, lp)
        }
        archive = target.loss(policy, traces, dict(zip(traces, lp)), role_check=lambda rows: True)
        anchor = lp.detach().clone()
        kl = (lp.exp() * (lp - anchor)).sum()
        named = list(model.named_parameters())
        params = [p for _, p in named]
        norms = {}
        for label, loss in [
            ("archive_raw", archive),
            ("archive_applied_0.01", 0.01 * archive),
            ("KL_raw", kl),
            ("KL_applied_0.2", 0.2 * kl),
        ]:
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            actor = [g for (n, _), g in zip(named, grads) if n == "tok_emb.weight" and g is not None]
            trunk = [
                g
                for (n, _), g in zip(named, grads)
                if g is not None and n.startswith(("blocks.", "ctx_mlp.", "ln_f.", "pos_emb."))
            ]
            norms[label] = {
                "actor": math.sqrt(sum(float((g * g).sum()) for g in actor)),
                "shared_trunk": math.sqrt(sum(float((g * g).sum()) for g in trunk)),
            }
        self.assertGreater(norms["archive_raw"]["actor"], 0)
        self.assertGreater(norms["archive_raw"]["shared_trunk"], 0)
        self.assertAlmostEqual(
            norms["archive_applied_0.01"]["actor"],
            0.01 * norms["archive_raw"]["actor"],
            places=12,
        )

        def fixture_parameter_hash():
            digest = hashlib.sha256()
            for name, parameter in sorted(model.named_parameters()):
                digest.update(name.encode() + b"\0")
                digest.update(str(parameter.dtype).encode() + b"\0")
                digest.update(str(tuple(parameter.shape)).encode() + b"\0")
                digest.update(parameter.detach().cpu().contiguous().numpy().astype("<f8").tobytes())
            return digest.hexdigest()

        before_hash = fixture_parameter_hash()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        (0.01 * archive).backward()
        optimizer.step()
        with torch.no_grad():
            after = readout().exp()
        quality = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        self.assertGreater(float(after[2]), float(before[2]))
        self.assertGreater(float(after @ quality), float(before @ quality))
        self.assertNotEqual(before_hash, fixture_parameter_hash())
        MEASURED_GB03_PROBE.update(
            {
                "fixture_only": True,
                "actor_class": "actual PolicyLM; small synthetic configuration",
                "model_source_sha256": hashlib.sha256((root / "model.py").read_bytes()).hexdigest(),
                "parameter_before": before_hash,
                "parameter_after": fixture_parameter_hash(),
                "parameter_hash_algorithm": "fixture-named-f64-bytes-sha256-v1; not canonical v1 identity",
                "gradient_norms_by_loss_term": norms,
                "top_q_probability_before": float(before[2]),
                "top_q_probability_after": float(after[2]),
                "exact_expected_q_before": float(before @ quality),
                "exact_expected_q_after": float(after @ quality),
                "optimizer_updates": 1,
                "production_qualification": False,
            }
        )


if __name__ == "__main__":
    unittest.main()
