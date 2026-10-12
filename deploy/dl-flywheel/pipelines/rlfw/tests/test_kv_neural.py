"""Small real PolicyLM MATH-SDPA fixtures; no corpus or checkpoint inputs."""

import hashlib
import sys
import time
import unittest
from pathlib import Path

import torch

from rlfw.kv_decode import KVPolicy, kv_sample, strict_decode
from rlfw.policy_binding import PinnedPolicy
from rlfw.tests.test_kv_trie import ToyTrie
from rlfw.trie_enumerate import trie_enumerate, verify_sample

ROOT = Path(__file__).resolve().parents[2] / "continuous"
sys.path[:0] = [str(ROOT), str(ROOT / "consumer"), str(ROOT / "runtime")]
from dltrain.model import PolicyConfig, PolicyLM, configure_determinism
from neural_decode import BatchedTrie

OBSERVATIONS = {}


def fixture_policy():
    configure_determinism(1)
    torch.manual_seed(17)
    p = object.__new__(PinnedPolicy)
    p.torch, p.device = torch, "cpu"
    p.net = PolicyLM(
        PolicyConfig(
            vocab_size=7,
            ctx_dim=3,
            d_model=16,
            n_layers=2,
            n_heads=2,
            d_ff=32,
            max_seq=8,
        )
    ).eval()
    p.trie = object.__new__(BatchedTrie)
    p.trie.nodes, p.trie.programs = ToyTrie.nodes, ToyTrie.programs
    p.trie.identity, p.trie.max_seq = "a" * 64, 8
    return p


class NeuralTests(unittest.TestCase):
    def test_analysis_seed_replay_complete_leaves_and_bound(self):
        p = fixture_policy()
        kwargs = {
            "seeds": list(range(100)),
            "deadline": time.time() + 60,
            "cache_bytes": 5000,
        }
        first, stats = kv_sample(p, [0, 0, 0], **kwargs)
        second, _ = kv_sample(p, [0, 0, 0], **kwargs)
        self.assertEqual(first, second)
        self.assertTrue(all(tuple(r["tokens"]) in p.trie.programs for r in first))
        self.assertLessEqual(stats["cache_peak_bytes"], 5000)
        self.assertGreater(stats["valid_distinct_per_second"], 0)

    def test_exact_enumeration_real_model(self):
        p = fixture_policy()
        rows, stats = trie_enumerate(p, [0.2, -0.3, 0.7])
        self.assertAlmostEqual(stats["mass"], 1, delta=1e-12)
        self.assertTrue(verify_sample(p, [0.2, -0.3, 0.7], rows, fraction=1)["passed"])

    def test_incremental_logits_and_eos_heads_declared_tolerance(self):
        p = fixture_policy()
        ctx = [0.2, -0.3, 0.7]
        k = KVPolicy(p, ctx)
        maximum = 0.0
        for tokens in p.trie.programs:
            for i in range(1, len(tokens)):
                got = torch.tensor(k.next_logits(ctx, tokens[:i]), dtype=torch.float64)
                want = torch.tensor(p.next_logits(ctx, tokens[:i]), dtype=torch.float64)
                maximum = max(maximum, float((got - want).abs().max()))
                torch.testing.assert_close(got, want, rtol=0, atol=1e-6)
            self.assertAlmostEqual(
                k.heads(tokens)["q"], p.score_heads([ctx], [tokens])[0]["q"], delta=1e-6
            )
        # Reduction-shape drift may be zero on a backend; closeness above is
        # required everywhere. Record the observation rather than assume drift.
        OBSERVATIONS["incremental_logits"] = {
            "max_delta": maximum,
            "tolerance": 1e-6,
            "device": "cpu",
            "torch": torch.__version__,
        }

    def test_kv_context_model_changes_and_cap(self):
        p = fixture_policy()
        k = KVPolicy(p, [0, 0, 0], cache_bytes=5000)
        k.next_logits([0, 0, 0], (1, 4))
        self.assertLessEqual(k.cache.peak_bytes, 5000)
        with self.assertRaisesRegex(ValueError, "cross-context"):
            k.next_logits([0, 0, 1], (1, 4))
        with torch.no_grad():
            next(p.net.parameters()).add_(0.1)
        with self.assertRaisesRegex(ValueError, "unchanged"):
            k.state((1,))

    def test_kv_enumeration_matches_small_support(self):
        p = fixture_policy()
        rows, stats = trie_enumerate(p, [0, 0, 0], mode="kv_analysis")
        self.assertAlmostEqual(stats["mass"], 1, delta=1e-12)
        self.assertTrue(
            verify_sample(p, [0, 0, 0], rows, fraction=1, tolerance=1e-6)["passed"]
        )

    def test_1000_seeded_traces_and_teacher_forced_logits_bitwise(self):
        p = fixture_policy()
        total_steps, digest = 0, hashlib.sha256()
        for start in range(0, 1000, 100):
            seeds = list(range(start, start + 100))
            # Mixed contexts and termination lengths exercise active-row shrink.
            ctx = torch.tensor([[i % 3 / 10, i % 7 / 10, -0.2] for i in seeds])
            want, _ = p.sample_many(ctx, seeds=seeds, deadline=time.time() + 60)

            def observe(tokens, lengths, contexts, indices, output):
                nonlocal total_steps
                fresh = p.net(tokens, lengths, contexts, indices)
                self.assertTrue(torch.equal(output.logits, fresh.logits))
                digest.update(output.logits.detach().numpy().tobytes())
                total_steps += len(tokens)

            got, stats = strict_decode(
                p,
                ctx,
                seeds=seeds,
                deadline=time.time() + 60,
                observer=observe,
                cache_bytes=100000,
            )
            self.assertEqual(got, want)
            self.assertFalse(stats["incremental_kv"])
            self.assertLessEqual(stats["cache_peak_bytes"], 100000)
        print(
            f"strict fixtures: traces=1000 teacher_forced_rows={total_steps} logits_sha256={digest.hexdigest()}"
        )

    def test_strict_invalid_seeds_use_deployed_refusal(self):
        p = fixture_policy()
        with self.assertRaises(ValueError):
            strict_decode(p, [[0, 0, 0]], seeds=[-1], deadline=time.time() + 60)
