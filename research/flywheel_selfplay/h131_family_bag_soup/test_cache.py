import unittest
import torch
from anchor_cache import AnchorCache


class Fake:
    def __init__(self):
        self.calls = 0
        self.weight = torch.tensor([2.0])

    def state_dict(self):
        return {"weight": self.weight}

    def predict_costs(self, *, x, flag=True):
        self.calls += 1
        y = x * self.weight + (1 if flag else 0)
        return dict(total_cost=y, eff_prior=y + 1, fail_logit=y + 2)


class Tests(unittest.TestCase):
    def test_live_validation_then_reuse(self):
        a = Fake()
        c = AnchorCache()
        c.attach(a)
        c.enabled = True
        x = torch.tensor([3.0])
        p = a.predict_costs(x=x)
        a.predict_costs(x=x)
        a.predict_costs(x=x)
        self.assertEqual(a.calls, 2)
        self.assertEqual(c.receipt()["verified_keys"], 1)
        self.assertTrue(torch.equal(p["total_cost"], torch.tensor([7.0])))

    def test_every_input_and_state_bound(self):
        a = Fake()
        c = AnchorCache()
        c.attach(a)
        c.enabled = True
        x = torch.tensor([3.0])
        a.predict_costs(x=x)
        a.predict_costs(x=x, flag=False)
        x.add_(1)
        a.predict_costs(x=x)
        self.assertEqual(c.receipt()["entries"], 3)
        b = Fake()
        b.weight.add_(1)
        with self.assertRaises(RuntimeError):
            c.attach(b)

    def test_uncached_training(self):
        a = Fake()
        c = AnchorCache()
        c.attach(a)
        x = torch.tensor([3.0])
        a.predict_costs(x=x)
        a.predict_costs(x=x)
        self.assertEqual(a.calls, 2)
        self.assertFalse(c.entries)

    def test_mismatch_refused(self):
        a = Fake()
        c = AnchorCache()
        c.attach(a)
        c.enabled = True
        x = torch.tensor([3.0])
        a.predict_costs(x=x)
        a.weight.add_(1)
        with self.assertRaises(RuntimeError):
            a.predict_costs(x=x)


if __name__ == "__main__":
    unittest.main()
