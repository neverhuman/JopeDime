#!/usr/bin/env python3
"""REL-07 E1: candidate-bound language capability intersection checks."""

from __future__ import annotations

import json
import subprocess
import sys

from common import (
    ROOT,
    Context,
    ProviderError,
    context,
    rust_counts,
    sha256,
    verify_candidate,
)


TESTS = (
    ("feat-math", "evolve::api::capability::tests::three_language_intersection_and_exclusions_are_exact"),
    ("feat-math", "evolve::api::capability::tests::impossible_or_empty_requests_have_stable_error"),
    ("feat-math", "evolve::api::capability::tests::coverage_agrees_with_actual_recipe_emitters"),
    ("jope-service", "discover::propose::tests::actual_language_intersection_filters_every_generated_operation"),
    ("jope-service", "http::runs::tests::requested_export_languages_filter_the_actual_native_seed_and_mutations"),
    ("jope-service", "http::runs::tests::python_and_rust_plan_uses_the_declared_capability_intersection"),
)
SINGLE_PASS = {
    "collected": 1,
    "executed": 1,
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}


def command(crate: str, test: str) -> list[str]:
    return [
        "cargo",
        "test",
        "--locked",
        "-p",
        crate,
        "--lib",
        test,
        "--",
        "--exact",
        "--test-threads=1",
    ]


def check_binding(ctx: Context, run) -> None:
    """Require exact clean source and unchanged artifact before and after each test."""
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError("E1 requires the clean, candidate-pinned source checkout")
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during E1")


def qualify(ctx: Context, run=None) -> int:
    run = subprocess.run if run is None else run
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("E1 result must be a new regular file")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("E1 evidence must be outside the source checkout")
    check_binding(ctx, run)
    verify_candidate(ctx, run=run)
    for crate, test in TESTS:
        check_binding(ctx, run)
        args = command(crate, test)
        print(json.dumps({"test": test, "argv": args}), flush=True)
        result = run(
            args, cwd=ROOT, capture_output=True, text=True, check=False, timeout=3600
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        check_binding(ctx, run)
        if result.returncode != 0:
            return result.returncode
        if rust_counts(result.stdout) != SINGLE_PASS or f"test {test} ... ok" not in result.stdout:
            raise ProviderError(f"E1 requires exactly one passing case for {test}")
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
        return qualify(context("E1"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"E1 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
