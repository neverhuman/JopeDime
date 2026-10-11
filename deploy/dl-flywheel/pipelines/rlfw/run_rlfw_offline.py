#!/usr/bin/env python3
"""Run the registered, data-free research fixtures and retain exact test identities."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

MODULES = (
    "rlfw.tests.test_pr1",
    "rlfw.tests.test_binding",
    "rlfw.tests.test_arena_kpi",
    "rlfw.tests.test_arena_scheduler",
    "rlfw.tests.test_slate",
    "rlfw.tests.test_no_winner_bootstrap",
    "rlfw.tests.test_successor_support",
)


def source_identity():
    local = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
    ).strip()
    return (
        os.environ.get("JOPE_SOURCE_SHA") or local,
        os.environ.get("GITHUB_SHA") or local,
    )


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item.id()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    started = time.time()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(MODULES)
    identities = list(flatten(suite))
    if loader.errors or not identities:
        raise RuntimeError("test collection failed or collected zero tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    source_commit, checkout_sha = source_identity()
    receipt = {
        "schema": "jope.rlfw-offline/v1",
        "started_unix": started,
        "finished_unix": time.time(),
        "collected": identities,
        "executed": result.testsRun,
        "failed": [t.id() for t, _ in result.failures],
        "errors": [t.id() for t, _ in result.errors],
        "skipped": [(t.id(), reason) for t, reason in result.skipped],
        "passed": result.wasSuccessful()
        and result.testsRun == len(identities)
        and not result.skipped,
        "source_commit": source_commit,
        "checkout_sha": checkout_sha,
        "command": sys.argv,
        "python": sys.version,
        "environment": {
            key: os.environ.get(key) for key in ("TMPDIR", "PYTHONDONTWRITEBYTECODE")
        },
        "source_sha256": {
            str(p.relative_to(Path(__file__).parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(Path(__file__).parent.rglob("*.py"))
        },
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
