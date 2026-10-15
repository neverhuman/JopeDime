#!/usr/bin/env python3
"""Run J22 synthetic CPU fixtures, checking the registered test roster."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path


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
    started = time.time()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName("rlfw.multiseed.tests.test_multiseed")
    identities = sorted(flatten(suite))
    expected = json.loads((root / "test-identities.json").read_text())
    if loader.errors or not identities or identities != expected:
        raise RuntimeError("collection failed or differs from exact registered identities")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    receipt = {
        "schema": "jope.j22-offline/v1",
        "source_commit": os.environ.get("JOPE_SOURCE_SHA", head),
        "checkout_sha": os.environ.get("GITHUB_SHA", head),
        "started_unix": started,
        "finished_unix": time.time(),
        "command": sys.argv,
        "python": sys.version,
        "collected": identities,
        "executed": result.testsRun,
        "failed": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "skipped": [(test.id(), reason) for test, reason in result.skipped],
        "passed": result.wasSuccessful() and result.testsRun == len(identities) and not result.skipped,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".json", ".md")
        },
        "input_scope": "synthetic only; no dataset, checkpoint, measurement body or cloud access",
        "GPU_work": 0,
        "optimizer_updates": 0,
        "GP_executions": 0,
        "training_authorized": False,
        "qualified": False,
        "power": "UNMEASURED",
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x") as out:
            json.dump(receipt, out, indent=2, allow_nan=False)
            out.write("\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
