"""The §6.3.2 calibration readout (PLAN §6.3.2 "Calibration readout", §6.3 backend rule, §5 P10 feasibility calibration):
``logit += log(pi_nat / pi_group)`` with pi_nat per backend from train_fit and pi_group from the consumption journal's
L_feas class counts.

- the correction against a hand-built float64 reference;
- pi_group pools ``feas_class_counts`` only, so remote and unknown backend members never enter it;
- the dev/diag readout over the fixture equals a manual per-program recomputation, keeps the RNG and the model mode,
  and refuses every other split before reading anything.

CPU only, deterministic mode, nothing written to disk.
"""
from __future__ import annotations

import dataclasses
import math
import random
import unittest

import numpy as np
import torch

from dltrain import common, evaluate, model, sampler
from tests import fixtures_p7 as fx

LOG = math.log


def setUpModule() -> None:
    model.configure_determinism(4)


def _reference(logits, pi_nat, pi_group) -> list[float]:
    """Float64 by hand: z_k = l_k + log(pi_nat_k / sum pi_nat) - log(pi_group_k / sum pi_group); softmax over z."""
    sn, sg = math.fsum(pi_nat), math.fsum(pi_group)
    z = [x + LOG(n / sn) - LOG(g / sg) if n > 0 else -math.inf for x, n, g in zip(logits, pi_nat, pi_group)]
    top = max(z)
    e = [math.exp(x - top) if math.isfinite(x) else 0.0 for x in z]
    return [x / math.fsum(e) for x in e]


def _slot(ctx: str, class_counts, feas_class_counts) -> dict:
    return {"slot": 0, "context_sha256": ctx, "class_counts": list(class_counts),
            "feas_class_counts": list(feas_class_counts)}


class PriorShiftTests(unittest.TestCase):
    def test_correction_matches_the_float64_reference(self):
        cases = [([0.3, -1.2, 0.5], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1]),
                 ([2.0, 0.0, -3.0], [0.9, 0.05, 0.05], [0.5, 0.25, 0.25]),
                 ([-0.4, 1.7, 0.9], [0.2, 0.3, 0.5], [0.75, 0.125, 0.125]),
                 ([0.0, 0.0, 0.0], [6.0, 1.0, 1.0], [2.0, 1.0, 1.0])]          # counts: the scale cancels
        for logits, nat, group in cases:
            with self.subTest(logits=logits, nat=nat, group=group):
                got = evaluate.prior_shift_probs(logits, nat, group)
                ref = _reference(logits, nat, group)
                for g, r in zip(got, ref):
                    self.assertAlmostEqual(g, r, delta=1e-15)
                self.assertAlmostEqual(math.fsum(got), 1.0, delta=1e-15)
        logits = [0.3, -1.2, 0.5]
        e = [math.exp(x) for x in logits]
        softmax = [x / math.fsum(e) for x in e]
        for g, r in zip(evaluate.prior_shift_probs(logits, [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]), softmax):
            self.assertAlmostEqual(g, r, delta=1e-15)                        # equal priors: no correction
        upweighted = evaluate.prior_shift_probs(logits, [0.2, 0.6, 0.2], [0.2, 0.3, 0.5])
        self.assertGreater(upweighted[1], softmax[1])                       # a class more frequent at natural rates
        self.assertEqual(evaluate.prior_shift_probs(logits, [0.8, 0.2, 0.0], [0.5, 0.25, 0.25])[2], 0.0)

    def test_refusals(self):
        good = ([0.1, 0.2, 0.3], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25])
        bad = [([0.1, 0.2], good[1], good[2]), ([0.1, math.nan, 0.3], good[1], good[2]),
               ([0.1, 0.2, math.inf], good[1], good[2]), (good[0], [0.5, -0.1, 0.6], good[2]),
               (good[0], [0.0, 0.0, 0.0], good[2]), (good[0], good[1], [0.5, 0.5, 0.0]),
               (good[0], good[1], [0.5, math.nan, 0.5]), (good[0], [1.0, 2.0], good[2])]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                evaluate.prior_shift_probs(*args)


class PiGroupTests(unittest.TestCase):
    def test_pi_group_pools_only_the_L_feas_members(self):
        records = [{"step": 1, "slots": [_slot("c-a", (6, 1, 1), (5, 0, 1)), _slot("c-b", (3, 3, 2), (2, 3, 2))]},
                   {"step": 2, "slots": [_slot("c-a", (7, 1, 0), (7, 1, 0))]}]
        self.assertEqual(evaluate.pi_group_from_journal(records), (14 / 21, 4 / 21, 3 / 21))
        self.assertEqual(evaluate.pi_group_from_journal(records, context_sha256="c-a"), (12 / 14, 1 / 14, 1 / 14))
        self.assertNotEqual(evaluate.pi_group_from_journal(records, context_sha256="c-a"),
                            (13 / 16, 2 / 16, 1 / 16), "the remote member's class entered pi_group")
        with self.assertRaises(ValueError):
            evaluate.pi_group_from_journal(records, context_sha256="absent")
        with self.assertRaises(ValueError):
            evaluate.pi_group_from_journal([{"step": 1, "slots": [_slot("c", (1, 0, 0), (0, 0, 0))]}])
        for bad in ((1, 0), (1, -1, 0), (1.0, 0, 0), (True, 0, 0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evaluate.pi_group_from_journal([{"step": 1, "slots": [_slot("c", (1, 0, 0), bad)]}])

    def test_pi_group_of_a_fixture_journal_excludes_remote_members(self):
        data = fx.train_data()
        draws = sampler.MacroSampler(data.fit, fx.RUN_SEED, data.config)
        records = [draws.draw_next().journal_record() for _ in range(300)]
        local, every = [0, 0, 0], [0, 0, 0]
        for rec in records:
            for s in rec["slots"]:
                for m in s["members"]:
                    lab = data.corpus.label(s["context_sha256"], m)
                    every[lab.feas_class] += 1
                    local[lab.feas_class] += lab.feas_valid
        self.assertNotEqual(local, every)
        self.assertEqual(evaluate.pi_group_from_journal(records), tuple(c / sum(local) for c in local))
        self.assertTrue(all(c > 0 for c in local), "the fixture journal must sample every class at local")


class FeasibilityReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fx.train_data()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(77)
            cls.net = model.PolicyLM(fx.model_config())
        draws = sampler.MacroSampler(cls.data.fit, fx.RUN_SEED, cls.data.config)
        cls.pi_group = evaluate.pi_group_from_journal([draws.draw_next().journal_record() for _ in range(200)])

    def _manual_logits(self, ctx: str, prog: str) -> list[float]:
        toks = list(self.data.corpus.tokens(ctx, prog))
        z = torch.from_numpy(np.array(self.data.corpus.encodings[ctx], dtype=np.float32)).reshape(1, -1)
        with torch.no_grad():
            out = self.net(torch.tensor([toks]), torch.tensor([len(toks)]), z, torch.tensor([0]))
        return [float(x) for x in out.feas_logits[0].double()]

    def test_dev_readout_equals_a_manual_recomputation(self):
        corpus, prior_nat = self.data.corpus, self.data.fit.class_prior_nat
        self.assertEqual(sorted(prior_nat), ["local", "remote"])
        self.net.train()
        random.seed(5)
        np.random.seed(6)
        before = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
        report = evaluate.feasibility_readout(self.net, self.data, split="dev", pi_group=self.pi_group)
        self.assertTrue(self.net.training, "the readout must restore the training mode")
        self.assertTrue(torch.equal(before[0], torch.get_rng_state()))
        self.assertEqual(before[1], random.getstate())
        self.assertTrue(np.array_equal(before[2][1], np.random.get_state()[1]))
        common.canonical(report)                                            # JSON, no NaN
        dev = sorted(c for c, row in corpus.contexts.items()
                     if row["split_role"] == "dev" and corpus.assignment[row["family_id"]] == "dev")
        expected = [(c, lab.program_sha256) for c in dev for lab in corpus.labels.records[c] if lab.feas_class is not None]
        self.assertEqual([(r["context_sha256"], r["program_sha256"]) for r in report["programs"]], expected)
        self.assertEqual((report["schema"], report["split"], report["role"], report["contexts"]),
                         ("jope.dl-eval-feasibility/v1", "dev", "training_validation", len(dev)))
        self.assertEqual(report["pi_group"], list(self.pi_group))
        backends = {r["backend"] for r in report["programs"]}
        self.assertEqual(backends, {"local", "remote"})
        self.net.eval()
        try:
            for r in report["programs"][::7]:
                logits = self._manual_logits(r["context_sha256"], r["program_sha256"])
                for got, want in zip(r["logits"], logits):
                    self.assertAlmostEqual(got, want, delta=1e-5)
                e = [math.exp(x - max(r["logits"])) for x in r["logits"]]
                for got, want in zip(r["probs_raw"], [x / math.fsum(e) for x in e]):
                    self.assertAlmostEqual(got, want, delta=1e-15)
                ref = _reference(r["logits"], prior_nat[r["backend"]], self.pi_group)
                for got, want in zip(r["probs_corrected"], ref):
                    self.assertAlmostEqual(got, want, delta=1e-15)
                self.assertEqual(r["feas_valid"], r["backend"] == "local")
        finally:
            self.net.train()

    def test_backend_without_a_natural_prior_is_unknown(self):
        fit = dataclasses.replace(self.data.fit, class_prior_nat={"local": self.data.fit.class_prior_nat["local"]})
        data = dataclasses.replace(self.data, fit=fit)
        report = evaluate.feasibility_readout(self.net, data, split="diag", pi_group=self.pi_group)
        self.assertEqual((report["split"], report["role"]), ("diag", "diagnostic_only"))
        rows = report["programs"]
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["probs_corrected"] is None, r["backend"] != "local")
        self.assertTrue(any(r["probs_corrected"] is None for r in rows))

    def test_refuses_other_splits_before_reading_and_a_zero_pi_group(self):
        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError(f"data or model accessed ({name}) before the split refusal")

        for split in ("heldout", "held", "sealed_excluded", "train_fit", None):
            with self.subTest(split=split), self.assertRaises(evaluate.EvaluateRefused):
                evaluate.feasibility_readout(Untouchable(), Untouchable(), split=split, pi_group=self.pi_group)
        with self.assertRaises(ValueError):
            evaluate.feasibility_readout(self.net, self.data, split="dev", pi_group=(0.9, 0.1, 0.0))


if __name__ == "__main__":
    unittest.main()
