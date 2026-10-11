"""Gate A adapters against the repository's real PolicyLM and real decoder."""

import hashlib
import math
import time
import unittest

import torch
from dltrain import dsl_tokenizer as T
from dltrain import model as M
from dltrain import objective as O
from neural_decode import BatchedTrie

from rlfw.compact_support import CompactSupportTrie, encode
from rlfw.probability_contract import actor_logprob, weighted_archive_ce


class ActualPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        torch.manual_seed(31)
        self.vocab = T.Vocab.from_names({})
        self.net = M.PolicyLM(
            M.PolicyConfig(self.vocab.size, 4, 16, 1, 2, 32, 128, dropout=0.2)
        )
        texts = [
            "(pipe X (ridge))",
            "(pipe X (ridge lambda=1))",
            "(pipe X (standardize) (ridge lambda=0.1))",
        ]
        self.trie = BatchedTrie(
            [
                {
                    "dsl": text,
                    "program_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
                for text in texts
            ],
            self.vocab,
            128,
        )
        self.tokens = list(self.trie.programs)
        self.z = torch.randn(1, 4)

    def likelihood(self, rows=None):
        rows = self.tokens if rows is None else rows
        return actor_logprob(self.net, self.z.expand(len(rows), -1), rows, self.trie)

    def test_sampler_equals_training_recomputation_eos_and_dropout_off(self):
        self.net.train()
        rows, _ = self.trie.decode_many(
            self.net,
            self.z.expand(7, -1),
            seeds=list(range(7)),
            temperature=0.7,
            deadline=time.time() + 30,
        )
        rng = torch.get_rng_state().clone()
        ll = actor_logprob(
            self.net,
            self.z.expand(7, -1),
            [row["tokens"] for row in rows],
            self.trie,
            temperature=0.7,
        )
        self.assertTrue(self.net.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for index, row in enumerate(rows):
            expected = math.fsum(
                math.log(x["chosen_probability"]) for x in row["trace"]
            )
            self.assertAlmostEqual(float(ll.total[index].detach()), expected, places=10)
            self.assertEqual(ll.per_token[index].dtype, torch.float64)
            self.assertEqual(len(ll.per_token[index]), len(row["tokens"]) - 1)
            self.assertEqual(row["trace"][-1]["chosen"], T.EOS)

    def test_sum_one_raw_action_order_duplicates_and_sequence_sum(self):
        ll = self.likelihood()
        self.assertAlmostEqual(float(ll.total.exp().sum().detach()), 1.0, places=6)
        for total, part in zip(ll.total, ll.per_token):
            self.assertAlmostEqual(
                float(total.detach()), float(part.sum().detach()), places=12
            )
            self.assertNotAlmostEqual(
                float(total.detach()), float(part.mean().detach()), places=4
            )
        ordered = [self.tokens[2], self.tokens[0], self.tokens[2]]
        duplicate = self.likelihood(ordered).total
        self.assertAlmostEqual(
            float(duplicate[0].detach()), float(duplicate[2].detach()), places=12
        )
        self.assertAlmostEqual(
            float(duplicate[1].detach()), float(ll.total[0].detach()), places=6
        )

    def test_actual_weighted_ce_gradient_raises_higher_q_probability(self):
        before = self.likelihood().total.detach()
        higher_q = int(before.argmin())
        q = torch.zeros(3, dtype=torch.float64)
        q[higher_q] = 2.0
        weights = torch.softmax(q / 0.25, dim=0)
        original = [p.detach().clone() for p in self.net.parameters()]
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.002)
        optimizer.zero_grad()
        loss = weighted_archive_ce(self.likelihood(), weights)
        loss.backward()
        self.assertGreater(
            sum(
                float(p.grad.square().sum())
                for p in self.net.parameters()
                if p.grad is not None
            ),
            0,
        )
        optimizer.step()
        after = self.likelihood().total.detach()
        self.assertGreater(float(after[higher_q].exp()), float(before[higher_q].exp()))
        self.assertTrue(
            any(
                not torch.equal(p.detach(), old)
                for p, old in zip(self.net.parameters(), original)
            )
        )

    def test_actual_objective_mean_is_not_actor_sequence_logprob(self):
        self.net.eval()
        row = self.tokens[0]
        tokens = torch.tensor([row])
        lengths = torch.tensor([len(row)])
        logits = self.net(tokens, lengths, self.z, torch.tensor([0])).logits
        masks = torch.zeros_like(logits, dtype=torch.bool)
        for pos in range(1, len(row)):
            masks[0, pos - 1, list(self.trie.get(row[:pos]))] = True
        valid = O.valid_positions(lengths, len(row))
        ce = O.sequence_ce(logits, tokens[:, 1:], masks, valid)
        ll = self.likelihood([row]).total[0]
        self.assertAlmostEqual(
            float(ce.detach()) * (len(row) - 1), -float(ll.detach()), places=5
        )
        self.assertNotAlmostEqual(float(ce.detach()), -float(ll.detach()), places=4)

    def test_incomplete_foreign_actions_and_bad_weights_refused(self):
        for rows in [[self.tokens[0][:-1]], [(1, 2)]]:
            with self.assertRaisesRegex(ValueError, "including EOS"):
                self.likelihood(rows)
        for weights in [[0, 0, 0], [-1, 1, 1], [float("nan"), 0, 1]]:
            with self.assertRaises(ValueError):
                weighted_archive_ce(self.likelihood(), weights)

    def test_compact_index_actual_sampler_matches_eager_trie(self):
        programs = list(self.trie.programs.values())
        compact = CompactSupportTrie(
            programs,
            encoded_by_sha={
                p["program_sha256"]: encode(t) for t, p in self.trie.programs.items()
            },
            vocabulary_sha256=self.vocab.sha256,
            max_seq=128,
        )
        z = self.z.expand(8, -1)
        args = {
            "seeds": list(range(8)),
            "temperature": 0.7,
            "deadline": time.time() + 30,
        }
        eager, _ = self.trie.decode_many(self.net, z, **args)
        indexed, _ = compact.decode_many(self.net, z, **args)
        self.assertEqual([x["tokens"] for x in eager], [x["tokens"] for x in indexed])
        self.assertEqual([x["trace"] for x in eager], [x["trace"] for x in indexed])
        a = actor_logprob(self.net, z, [x["tokens"] for x in eager], self.trie).total
        b = actor_logprob(self.net, z, [x["tokens"] for x in indexed], compact).total
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
