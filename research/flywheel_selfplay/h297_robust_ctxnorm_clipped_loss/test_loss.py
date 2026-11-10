"""Toy-context checks only; no scientific outcomes are used by these tests."""

import unittest
import torch
import torch.nn.functional as F
from loss import normalize_targets, training_loss

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class LossTests(unittest.TestCase):
    def test_quantiles_and_iqr(self):
        x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        torch.testing.assert_close(normalize_targets(x, 0.0), (x - 1.5) / 1.501)

    def test_failed_cost_saturates_and_smoothing_is_after_clipping(self):
        x = torch.tensor([[-1.0, -1.0, -1.0, -1.0, 2.0]])
        torch.testing.assert_close(
            normalize_targets(x, 0.0), torch.tensor([[0.0, 0.0, 0.0, 0.0, 3.0]])
        )
        torch.testing.assert_close(
            normalize_targets(x, 0.05), torch.tensor([[0.03, 0.03, 0.03, 0.03, 2.88]])
        )

    def test_constant_context_finite_zero(self):
        torch.testing.assert_close(
            normalize_targets(torch.ones(3, 8)), torch.zeros(3, 8)
        )

    def test_context_independent_and_shift_invariant(self):
        x = torch.tensor([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]])
        y = normalize_targets(x)
        torch.testing.assert_close(y[0], y[1])
        torch.testing.assert_close(y[:1], normalize_targets(x[:1]))

    def test_control_exact_original_loss(self):
        p = torch.tensor([[0.1, 0.4, -0.3]], requires_grad=True)
        t = torch.tensor([[0.0, 2.0, -1.0]])
        expected = F.smooth_l1_loss(p, t) + 0.1 * F.kl_div(
            F.log_softmax(-p / 0.2, dim=-1),
            F.softmax(-t / 0.2, dim=-1),
            reduction="batchmean",
        )
        torch.testing.assert_close(
            training_loss(p, t, "CONTROL"), expected, rtol=0, atol=0
        )

    def test_new_losses_have_finite_nonzero_prediction_gradients(self):
        for arm in ["CTXNORM_CLIP", "CTXNORM_CLIP_NOSMOOTH"]:
            p = torch.tensor([[0.1, 0.4, -0.3, 3.0]], requires_grad=True)
            t = torch.tensor([[0.0, 2.0, -1.0, 100.0]])
            loss = training_loss(p, t, arm)
            loss.backward()
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
