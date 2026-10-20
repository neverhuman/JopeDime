"""Small CPU tensor/autograd controls; no models, optimizer, fit, or GPU access."""
import unittest
import torch
from losses import context_loss


class LossTests(unittest.TestCase):
    def test_correct_ranking_has_lower_loss(self):
        q = torch.tensor([[1., 2., 3.]])
        mask = torch.ones_like(q, dtype=torch.bool)
        for name in ('composite_control', 'top_listwise', 'bootstrap_thompson'):
            good = context_loss(torch.tensor([[-2., 0., 2.]]), q, mask, name)
            bad = context_loss(torch.tensor([[2., 0., -2.]]), q, mask, name)
            self.assertLess(float(good), float(bad))

    def test_tied_members_permutation_invariant(self):
        p = torch.tensor([[.3, -.4, 2.]])
        q = torch.tensor([[2., 2., 1.]])
        mask = torch.ones_like(q, dtype=torch.bool)
        a = context_loss(p, q, mask, 'top_listwise')
        b = context_loss(p[:, [1, 0, 2]], q[:, [1, 0, 2]], mask, 'top_listwise')
        torch.testing.assert_close(a, b)

    def test_padding_does_not_change_value_or_gradients(self):
        for name in ('composite_control', 'top_listwise'):
            p = torch.tensor([[.2, -.3]], requires_grad=True)
            a = context_loss(p, torch.tensor([[1., 2.]]), torch.tensor([[True, True]]), name)
            a.sum().backward()
            padded = torch.tensor([[.2, -.3, 400.]], requires_grad=True)
            b = context_loss(padded, torch.tensor([[1., 2., float('nan')]]), torch.tensor([[True, True, False]]), name)
            b.sum().backward()
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(p.grad, padded.grad[:, :2])
            self.assertEqual(float(padded.grad[0, 2]), 0.)

    def test_all_ties_have_finite_gradients(self):
        p = torch.zeros(2, 4, requires_grad=True)
        loss = context_loss(p, torch.ones_like(p), torch.ones_like(p, dtype=torch.bool), 'top_listwise')
        loss.sum().backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_accumulated_context_gradients_match_full_batch(self):
        full = torch.tensor([[.1, .2], [.3, .4], [.2, -.5]], requires_grad=True)
        q = torch.tensor([[1., 2.], [4., 3.], [1., 1.]])
        mask = torch.ones_like(q, dtype=torch.bool)
        context_loss(full, q, mask, 'top_listwise').mean().backward()
        micro = full.detach().clone().requires_grad_()
        for i in range(3):
            (context_loss(micro[i:i+1], q[i:i+1], mask[i:i+1], 'top_listwise').sum() / 3).backward()
        torch.testing.assert_close(full.grad, micro.grad)

    def test_nonfinite_observed_target_refused(self):
        with self.assertRaises(ValueError):
            context_loss(torch.ones(1, 2), torch.tensor([[1., float('nan')]]), torch.ones(1, 2, dtype=torch.bool), 'top_listwise')

    def test_single_member_not_ranking_support(self):
        with self.assertRaises(ValueError):
            context_loss(torch.ones(1, 2), torch.ones(1, 2), torch.tensor([[True, False]]), 'top_listwise')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
