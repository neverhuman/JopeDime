#!/usr/bin/env python3
"""Run the R25-kNN-CE CPU fixtures and bind the exact collected identities and source hashes into a receipt."""

import argparse
import hashlib
import json
import os
import sys
import time
import unittest
from pathlib import Path

MODULES = (
    "rlfw.r25_knnce.tests.test_knn_table",
    "rlfw.r25_knnce.tests.test_harness_gate",
)
SOURCES = (
    "knn_table.py",
    "lpi_scratch_train_v7.py",
    "lpi_scratch_data_v4.py",
    "launch_v7.sh",
    "configs/r25.json",
    "tests/test_knn_table.py",
    "tests/test_harness_gate.py",
)


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item.id()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.parents[1]))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.time()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(MODULES)
    identities = list(flatten(suite))
    if loader.errors or not identities:
        raise RuntimeError("collection failed or zero tests")
    expected = json.loads((root / "test-identities.json").read_text())
    if sorted(identities) != expected:
        raise RuntimeError("collection differs from registered exact identities")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {
        "schema": "jope.r25-knnce-offline/v1",
        "scope": "CPU fixtures only: neighbour-table exclusions (held-out lineages, validation, same parent lineage, "
                 "non-candidates), determinism and tie order; harness v7 config refusal before any output; pinned "
                 "R25 config equals the ENG:watcher sign-off. No snapshot, checkpoint, GPU or training run.",
        "collected": len(identities),
        "executed": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "identities": sorted(identities),
        "sources": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES},
        "seconds": time.time() - started,
    }
    ok = result.wasSuccessful() and not result.skipped and result.testsRun == len(identities)
    receipt["passed"] = ok
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: receipt[k] for k in ("collected", "executed", "failures", "errors", "skipped", "passed")}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
