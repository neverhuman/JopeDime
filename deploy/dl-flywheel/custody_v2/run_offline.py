#!/usr/bin/env python3
"""Run every custody-v2 offline fixture and retain the exact roster and source hashes."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item.id()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName("custody_v2.tests.test_retention")
    ids = list(flatten(suite))
    if loader.errors or not ids:
        raise RuntimeError("nonempty complete fixture collection required")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sha = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "/usr/bin/git",
            "--no-optional-locks",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            ".",
        ],
        cwd=Path(__file__).parent,
        text=True,
    ).splitlines()
    receipt = {
        "schema": "jope.custody-v2-offline/v1",
        "source_sha": os.environ.get("JOPE_SOURCE_SHA", sha),
        "checkout_sha": os.environ.get("GITHUB_SHA", sha),
        "command": sys.argv,
        "collected": ids,
        "executed": result.testsRun,
        "failed": [t.id() for t, _ in result.failures],
        "errors": [t.id() for t, _ in result.errors],
        "skipped": [(t.id(), why) for t, why in result.skipped],
        "passed": result.wasSuccessful()
        and result.testsRun == len(ids)
        and not result.skipped,
        "source_sha256": {
            str(p.relative_to(Path(__file__).parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "source_dirty_paths": dirty,
        "source_commit_contains_executed_inputs": not dirty,
        "remote_calls": 0,
        "qualified": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
