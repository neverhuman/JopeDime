"""Run all registered ingest fixtures without S3 access or evaluator imports."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item.id()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()

    def source_hashes():
        return {
            str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.rglob("*.py")
        }

    hashes_before = source_hashes()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(
        ("ingest_joined_roster.tests.test_roster", "ingest_joined_roster.tests.test_s3")
    )
    collected = list(flatten(suite))
    if loader.errors or not collected:
        raise RuntimeError("complete nonzero fixture collection required")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    try:
        checkout = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent
        ).strip()
    except subprocess.CalledProcessError:
        checkout = None
    hashes_after = source_hashes()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(Path(__file__).parent)],
        text=True,
        cwd=Path(__file__).parent,
    ).strip()
    receipt = {
        "schema": "jope.ingest-offline/v1",
        "source_commit": os.environ.get("JOPE_SOURCE_SHA") or checkout,
        "checkout_sha": checkout,
        "collected": collected,
        "executed": result.testsRun,
        "failed": [t.id() for t, _ in result.failures],
        "errors": [t.id() for t, _ in result.errors],
        "skipped": [(t.id(), why) for t, why in result.skipped],
        "passed": result.wasSuccessful()
        and result.testsRun == len(collected)
        and not result.skipped
        and hashes_before == hashes_after,
        "started_unix": started,
        "finished_unix": time.time(),
        "command": sys.argv,
        "source_hashes": hashes_after,
        "source_hashes_at_start": hashes_before,
        "source_unchanged_during_run": hashes_before == hashes_after,
        "source_dirty": bool(dirty),
        "s3_access": False,
        "gpu_model_learner_or_evaluator_execution": False,
        "non_qualifying": True,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
