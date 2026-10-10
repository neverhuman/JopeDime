import math
import unittest
from typing import ClassVar

from rlfw.kv_decode import ByteLRU
from rlfw.trie_enumerate import ranking_report, trie_enumerate, verify_sample
from rlfw.trie_logprob import trie_program_logprob


class ToyTrie:
    programs: ClassVar = {
        (1, 3, 2): {"program_sha256": "a" * 64},
        (1, 4, 5, 2): {"program_sha256": "b" * 64},
        (1, 4, 6, 2): {"program_sha256": "c" * 64},
    }
    nodes: ClassVar = [{3: 1, 4: 3}, {2: 2}, {}, {5: 4, 6: 6}, {2: 5}, {}, {2: 7}, {}]

    def get(self, prefix):
        n = 0
        for token in prefix[1:]:
            n = self.nodes[n][token]
        return self.nodes[n]


class ToyPolicy:
    trie = ToyTrie()
    device = "cpu"

    def eval(self):
        pass

    def next_logits(self, context, prefix):
        return [0.0, -1.0, 3.0, 2.0, -1.0, 4.0, 1.0]

    def logprobs_many(self, contexts, rows, **kwargs):
        return [
            trie_program_logprob(self, c, p, self.trie, **kwargs)
            for c, p in zip(contexts, rows)
        ]

    def score_heads(self, contexts, rows):
        return [{"q": 0.2, "p_complete": 0.8, "runtime_seconds": 0.01} for p in rows]


class EnumerationTests(unittest.TestCase):
    def test_mass_and_exact_leaf_parity_at_two_temperatures(self):
        for t in (1.0, 0.7):
            rows, stats = trie_enumerate(ToyPolicy(), None, temperature=t)
            self.assertAlmostEqual(stats["mass"], 1.0, delta=1e-12)
            self.assertEqual(stats["branch_forwards"], 2)
            self.assertTrue(
                verify_sample(ToyPolicy(), None, rows, fraction=1.0, temperature=t)[
                    "passed"
                ]
            )

    def test_invalid_mode_temperature_and_device_refused(self):
        for kwargs in (
            {"mode": "cuda_exact"},
            {"temperature": 0},
            {"temperature": math.nan},
        ):
            with self.assertRaises(ValueError):
                trie_enumerate(ToyPolicy(), None, **kwargs)
        p = ToyPolicy()
        p.device = "cuda"
        with self.assertRaises(ValueError):
            trie_enumerate(p, None)

    def test_lru_payload_bound_eviction_and_replacement(self):
        c = ByteLRU(10)
        c.put("a", 1, 6)
        c.put("b", 2, 4)
        self.assertEqual(c.get("a"), 1)
        c.put("c", 3, 4)
        self.assertIsNone(c.get("b"))
        c.put("a", 4, 8)
        self.assertEqual(c.bytes, 8)
        c.put("large", 5, 11)
        self.assertEqual(c.bytes, 8)
        self.assertLessEqual(c.peak_bytes, 10)

    def test_changed_probability_fails_sample_check(self):
        rows, _ = trie_enumerate(ToyPolicy(), None)
        rows[0]["logprob"] += 0.01
        self.assertFalse(verify_sample(ToyPolicy(), None, rows, fraction=1.0)["passed"])


class RankingTests(unittest.TestCase):
    def rows(self, count=12):
        return [
            {
                "context_id": "ctx",
                "program_sha256": str(i),
                "q": float(i),
                "q_hat": float(i),
                "length": 100 + i * 20,
                "qualified": True,
            }
            for i in range(count)
        ]

    def test_perfect_and_inverted_rank_and_top8(self):
        rows = self.rows()
        result = ranking_report(rows)
        self.assertAlmostEqual(result["contexts"][0]["spearman"], 1)
        self.assertEqual(result["contexts"][0]["top8_recall_measured_support"], 1)
        for r in rows:
            r["q_hat"] = -r["q"]
        result = ranking_report(rows)
        self.assertAlmostEqual(result["contexts"][0]["spearman"], -1)
        self.assertEqual(result["contexts"][0]["top8_recall_measured_support"], 0.5)

    def test_null_rank_constant_q_and_uninformative_small_support(self):
        rows = self.rows(2)
        for r in rows:
            r["q"] = 0
        c = ranking_report(rows)["contexts"][0]
        self.assertIsNone(c["spearman"])
        self.assertFalse(c["top8_informative"])

    def test_singleton_unknown_excluded_and_duplicate_refused(self):
        rows = self.rows(2)
        rows[0]["qualified"] = False
        self.assertEqual(ranking_report(rows)["multi_label_contexts"], 0)
        rows = self.rows(2)
        with self.assertRaises(ValueError):
            ranking_report(rows + rows[:1])

    def test_calibration_counts_and_finite_values(self):
        r = ranking_report(self.rows())
        self.assertEqual(
            sum(v["n"] for k, v in r["calibration"].items() if k.startswith("decile:")),
            12,
        )
        self.assertEqual(
            sum(v["n"] for k, v in r["calibration"].items() if k.startswith("length:")),
            12,
        )
        rows = self.rows()
        rows[0]["q"] = math.nan
        with self.assertRaises(ValueError):
            ranking_report(rows)
