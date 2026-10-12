"""Run isolated analytics fixtures with a source-bound receipt (numpy/pyarrow)."""

import argparse
import json
import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rlfw.run_rlfw_offline import flatten, source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(
        (
            "rlfw.analytics.tests.test_metrics",
            "rlfw.analytics.tests.test_io",
            "rlfw.analytics.tests.test_scratch",
            "rlfw.analytics.tests.test_context_costs",
        )
    )
    collected = list(flatten(suite))
    if loader.errors or not collected:
        raise RuntimeError("nonzero complete collection required")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    source, checkout = source_identity()
    receipt = {
        "schema": "jope.rlfw-analytics-offline/v1",
        "source_commit": source,
        "checkout_sha": checkout,
        "started_unix": started,
        "finished_unix": time.time(),
        "collected": collected,
        "executed": result.testsRun,
        "failed": [t.id() for t, _ in result.failures],
        "errors": [t.id() for t, _ in result.errors],
        "skipped": [(t.id(), why) for t, why in result.skipped],
        "passed": result.wasSuccessful()
        and result.testsRun == len(collected)
        and not result.skipped,
        "command": sys.argv,
        "environment": {"TMPDIR": os.environ.get("TMPDIR")},
        "cgroup": Path("/proc/self/cgroup").read_text().strip(),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
