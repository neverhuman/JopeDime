"""Actual CPU float64 autodiff through the library's scalar loss interface."""

import itertools
import math
import unittest
from dataclasses import replace

import torch

from rlfw.onpolicy.groups import Draw, GroupSpec, PolicyIdentity, ResolvedGroup
from rlfw.onpolicy.objective import objective
from rlfw.onpolicy.tests.fixtures import SHA, logs, q_for
from rlfw.onpolicy.tests.test_math import H, Q, Z, analytic, softmax


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
                draws = tuple(
                    Draw(str(j), rewards[i].program, readouts[j])
                    for j, i in enumerate(actions)
                )
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
                max(
                    abs(float(x) + y) for x, y in zip(hwm_grad, analytic(Z, Q, H, b)[1])
                ),
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
        self.assertEqual(
            torch.autograd.grad(hwm, z, retain_graph=True)[0].tolist(), [0.0, 0.0]
        )
        gradient = torch.autograd.grad(dense, z)[0]
        expected = [-terms.dense[i] + 0.5 * math.fsum(terms.dense) for i in range(2)]
        self.assertLess(
            max(abs(float(x) - y) for x, y in zip(gradient, expected)), 1e-12
        )


if __name__ == "__main__":
    unittest.main()
