#!/usr/bin/env python3
"""REL-07 T3: reasoning and telemetry observer neutrality."""

from __future__ import annotations

import json
import subprocess
import sys

from common import (
    ROOT,
    Context,
    ProviderError,
    context,
    exact_case_passed,
    rust_counts,
    sha256,
    verify_candidate,
)


TESTS = (
    "telemetry::tests::reasoning_off_or_detailed_and_telemetry_modes_preserve_search",
    "telemetry::tests::telemetry_on_or_off_leaves_the_search_identical",
    "telemetry::tests::real_sensor_tick_cost_measurement",
    "telemetry::tests::fixed_work_overhead_measurement",
)
SINGLE_PASS = {
    "collected": 1,
    "executed": 1,
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}


def command(test: str) -> list[str]:
    args = ["cargo", "test", "--locked", "-p", "jope-service", "--lib", test, "--"]
    if test.endswith("fixed_work_overhead_measurement"):
        args.extend(("--ignored", "--nocapture"))
    args.extend(("--exact", "--test-threads=1"))
    return args


def check_binding(ctx: Context, run) -> None:
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError("T3 requires the clean, candidate-pinned source checkout")
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during T3")


def qualify(ctx: Context, run=None) -> int:
    run = subprocess.run if run is None else run
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("T3 result must be a new regular file")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("T3 evidence must be outside the source checkout")
    check_binding(ctx, run)
    verify_candidate(ctx, run=run)
    for test in TESTS:
        check_binding(ctx, run)
        args = command(test)
        print(json.dumps({"test": test, "argv": args}), flush=True)
        result = run(
            args, cwd=ROOT, capture_output=True, text=True, check=False, timeout=3600
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        check_binding(ctx, run)
        if result.returncode != 0:
            return result.returncode
        counts = rust_counts(result.stdout)
        if counts != SINGLE_PASS or not exact_case_passed(result.stdout, test):
            raise ProviderError(f"T3 requires exactly one passing case for {test}")
    check_binding(ctx, run)
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": ctx.gate_id,
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": {key: value * len(TESTS) for key, value in SINGLE_PASS.items()},
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return qualify(context("T3"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"T3 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
