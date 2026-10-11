#!/usr/bin/env python3
"""Run actual-PolicyLM Gate A fixtures, requiring CPU torch (no skip fallback)."""

import argparse
import hashlib
import json
import os
import sys
import time
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--onpolicy-source",
        type=Path,
        help="Read-only onpolicy package for the cross-lane integration fixture",
    )
    args = parser.parse_args()
    pipelines = Path(__file__).resolve().parents[1]
    continuous = pipelines / "continuous"
    for path in [
        pipelines,
        continuous,
        continuous / "consumer",
        continuous / "runtime",
    ]:
        sys.path.insert(0, str(path))
    from rlfw.run_rlfw_offline import flatten, source_identity

    started = time.time()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName("rlfw.tests.test_actual_policy_probability")
    external = {}
    if args.onpolicy_source:
        import importlib
        from types import ModuleType

        import torch
        from dltrain import dsl_tokenizer as T
        from dltrain import model as M
        from neural_decode import BatchedTrie

        from rlfw.probability_contract import actor_logprob

        package = ModuleType("gate_a_onpolicy")
        package.__path__ = [str(args.onpolicy_source.resolve())]
        sys.modules[package.__name__] = package
        sampling = importlib.import_module("gate_a_onpolicy.sampling")
        external = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.onpolicy_source.glob("*.py"))
        }

        class ActualOnpolicyAdapter(unittest.TestCase):
            def runTest(self):
                context = sampling.ScheduledContext(
                    "parent",
                    "context",
                    "no_winner_original",
                    "train",
                    True,
                    original=True,
                )
                chosen = sampling._parent_context_choice(
                    [context], __import__("random").Random(7)
                )
                self.assertTrue(chosen.optimizer_allowed)
                with self.assertRaises(ValueError):
                    sampling.ScheduledContext(
                        "parent",
                        "context",
                        "no_winner_original",
                        "development",
                        True,
                        original=True,
                    )
                torch.manual_seed(77)
                vocab = T.Vocab.from_names({})
                texts = ["(pipe X (ridge))", "(pipe X (ridge lambda=1))"]
                trie = BatchedTrie(
                    [
                        {
                            "dsl": text,
                            "program_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        }
                        for text in texts
                    ],
                    vocab,
                    128,
                )
                net = M.PolicyLM(
                    M.PolicyConfig(vocab.size, 4, 16, 1, 2, 32, 128, dropout=0.2)
                )
                z = torch.zeros(1, 4)
                rows, _ = trie.decode_many(
                    net, z, seeds=[7], temperature=1, deadline=time.time() + 30
                )
                ll = actor_logprob(net, z, [rows[0]["tokens"]], trie)
                trace = math.fsum(
                    math.log(t["chosen_probability"]) for t in rows[0]["trace"]
                )
                self.assertAlmostEqual(float(ll.total[0].detach()), trace, places=10)

        import math

        suite.addTest(ActualOnpolicyAdapter())
    collected = list(flatten(suite))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    source, checkout = source_identity()
    passed = bool(collected) and not loader.errors and result.wasSuccessful()
    passed = passed and result.testsRun == len(collected) and not result.skipped
    receipt = {
        "schema": "jope.rlfw-gate-a-actual-policy/v1",
        "label": "SCRATCH NON-QUALIFYING",
        "source_commit": source,
        "checkout_sha": checkout,
        "started_unix": started,
        "finished_unix": time.time(),
        "command": sys.argv,
        "collected": collected,
        "executed": result.testsRun,
        "failed": [t.id() for t, _ in result.failures],
        "errors": [t.id() for t, _ in result.errors],
        "skipped": result.skipped,
        "passed": passed,
        "python": sys.version,
        "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "external_onpolicy_source_sha256": external,
        "onpolicy_sampling_scope": "Context/replay scheduling; actor token sampling is neural_decode.BatchedTrie",
        "source_sha256": {
            str(p.relative_to(pipelines)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                pipelines / "rlfw/probability_contract.py",
                pipelines / "rlfw/tests/test_actual_policy_probability.py",
                continuous / "consumer/dltrain/model.py",
                continuous / "consumer/dltrain/objective.py",
                continuous / "neural_decode.py",
            ]
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
