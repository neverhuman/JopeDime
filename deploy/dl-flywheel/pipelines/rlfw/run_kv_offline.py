"""Isolated additive KV/trie fixtures; retains exact source and test identities."""

import argparse
import hashlib
import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlfw.run_rlfw_offline import flatten, source_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    started = time.time()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(
        ("rlfw.tests.test_kv_trie", "rlfw.tests.test_kv_neural")
    )
    identities = list(flatten(suite))
    if loader.errors or not identities:
        raise RuntimeError("KV test collection failed")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    from rlfw.tests.test_kv_neural import OBSERVATIONS

    source, checkout = source_identity()
    receipt = {
        "schema": "jope.kv-trie-fixtures/v1",
        "source_commit": source,
        "checkout_sha": checkout,
        "started_unix": started,
        "finished_unix": time.time(),
        "command": sys.argv,
        "collected": identities,
        "observations": OBSERVATIONS,
        "executed": result.testsRun,
        "failed": [t.id() for t, _ in result.failures],
        "errors": [t.id() for t, _ in result.errors],
        "skipped": [(t.id(), reason) for t, reason in result.skipped],
        "passed": result.wasSuccessful()
        and result.testsRun == len(identities)
        and not result.skipped,
        "source_sha256": {
            str(p.relative_to(Path(__file__).parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in [
                Path(__file__).parent / n
                for n in (
                    "kv_decode.py",
                    "trie_enumerate.py",
                    "run_kv_offline.py",
                    "tests/test_kv_trie.py",
                    "tests/test_kv_neural.py",
                )
            ]
        },
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
