#!/usr/bin/env python3
"""Run C4/S7/S8 CPU fixtures and bind exact identities/numerics to source."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

MODULES = (
    "rlfw.onpolicy.tests.test_math",
    "rlfw.onpolicy.tests.test_groups",
    "rlfw.onpolicy.tests.test_sampling",
    "rlfw.onpolicy.tests.test_autograd",
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
    from rlfw.onpolicy.tests.test_math import (
        H,
        Q,
        Z,
        analytic,
        exhaustive,
        finite,
        softmax,
    )

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
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    numerics = []
    for b in (1, 2, 4, 8):
        for g in (b, 2 * b):
            exact = exhaustive(softmax(Z), Q, H, b, g)
            truth, fd = analytic(Z, Q, H, b)[1], finite(Z, Q, H, b)
            numerics.append(
                {
                    "B": b,
                    "G": g,
                    "ordered_groups": 3**g,
                    "multinomial_count_vectors": math.comb(g + 2, 2),
                    "exhaustive": exact,
                    "analytic": truth,
                    "finite_difference": fd,
                    "analytic_max_error": max(abs(x - y) for x, y in zip(exact, truth)),
                    "finite_difference_max_error": max(abs(x - y) for x, y in zip(exact, fd)),
                }
            )
    receipt = {
        "schema": "jope.onpolicy-offline/v1",
        "source_commit": os.environ.get("JOPE_SOURCE_SHA", head),
        "checkout_sha": os.environ.get("GITHUB_SHA", head),
        "started_unix": started,
        "finished_unix": time.time(),
        "command": sys.argv,
        "python": sys.version,
        "collected": identities,
        "executed": result.testsRun,
        "failed": [t.id() for t, _ in result.failures],
        "errors": [t.id() for t, _ in result.errors],
        "skipped": [(t.id(), reason) for t, reason in result.skipped],
        "passed": result.wasSuccessful() and result.testsRun == len(identities) and not result.skipped,
        "input": {
            "logits": Z,
            "q": Q,
            "h": H,
            "origin": "declared synthetic toy; same inputs as merged audit_math.py",
        },
        "exact_gradient_checks": numerics,
        "reference_vector_status": "pending original logits/q/h/B; not fabricated from rounded output",
        "environment": {
            k: os.environ.get(k) for k in ("TMPDIR", "OMP_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")
        },
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".json", ".md")
        },
        "GP_executions": 0,
        "cloud_requests": 0,
        "GPU_work": 0,
        "runtime_optimizer_updates": 0,  # no production/runtime updates
        "GB03_CPU_synthetic_actor_diagnostics": __import__(
            "rlfw.onpolicy.tests.test_autograd", fromlist=["MEASURED_GB03_PROBE"]
        ).MEASURED_GB03_PROBE,
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x") as out:
            json.dump(receipt, out, indent=2, allow_nan=False)
            out.write("\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
