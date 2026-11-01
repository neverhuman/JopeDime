#!/usr/bin/env python3
"""REL-07 T1: candidate-bound CPU accounting, workers and missing sensors."""

from __future__ import annotations

import json
import re
import subprocess
import sys

from common import (
    ARTIFACT_SHA256,
    ROOT,
    SOURCE_SHA,
    Context,
    ProviderError,
    context,
    exact_case_passed,
    rust_counts,
    sha256,
    verify_candidate,
)


TESTS = (
    "telemetry::tests::cpu_math_follows_f06_without_clamping",
    "telemetry::tests::rss_sums_engine_and_live_workers_and_new_workers_baseline",
    "telemetry::tests::missing_sensors_produce_null_with_a_reason_never_zero",
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
    return [
        "cargo",
        "test",
        "--locked",
        "-p",
        "jope-service",
        "--lib",
        test,
        "--",
        "--exact",
        "--test-threads=1",
    ]


def check_paths(ctx: Context) -> None:
    if (
        ctx.gate_id != "T1"
        or not SOURCE_SHA.fullmatch(ctx.source_sha)
        or not ARTIFACT_SHA256.fullmatch(ctx.artifact_sha256)
    ):
        raise ProviderError("T1 requires its own gate and full source/artifact hashes")
    for path in (ctx.artifact, ctx.result):
        if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ProviderError("T1 paths must be absolute and contain no symlinks")
    if not ctx.artifact.is_file():
        raise ProviderError("T1 candidate must be a regular file")
    if ctx.result.exists():
        raise ProviderError("T1 result must be a new regular file")
    if any(p.exists() and not p.is_dir() for p in ctx.result.parents):
        raise ProviderError("T1 result parents must be directories")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("T1 evidence must be outside the source checkout")


def check_binding(ctx: Context, run) -> None:
    """Standalone providers must enforce the coordinator's source binding too."""
    check_paths(ctx)
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError(
                "T1 requires the clean, candidate-pinned source checkout"
            )
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during T1")


def qualify(ctx: Context, run=None) -> int:
    run = subprocess.run if run is None else run
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
        summaries = re.findall(r"^test result: .+$", result.stdout, re.MULTILINE)
        cases = re.findall(r"^test (\S+) \.\.\. ", result.stdout, re.MULTILINE)
        if (
            len(summaries) != 1
            or not summaries[0].startswith("test result: ok.")
            or rust_counts(summaries[0]) != SINGLE_PASS
            or cases != [test]
            or not exact_case_passed(result.stdout, test)
        ):
            raise ProviderError(f"T1 requires exactly one passing case for {test}")
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
    check_binding(ctx, run)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return qualify(context("T1"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"T1 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
