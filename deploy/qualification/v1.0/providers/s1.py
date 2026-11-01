#!/usr/bin/env python3
"""Candidate-bound S1 surface-equivalence acceptance provider."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

from common import (
    Context,
    ProviderError,
    context,
    rust_counts,
    sha256,
    verify_candidate,
)


ROOT = Path(__file__).resolve().parents[4]
TEST_NAME = "all_product_surfaces_share_the_bounded_run_projection"
SINGLE_PASS = {
    "collected": 1,
    "executed": 1,
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}
UNTRUSTED_TEST_INPUTS = (
    "JOPE_INT01_QUALIFY_HOME",
    "JOPE_INT01_CSV",
    "JOPE_INT01_CSV_SHA256",
    "JOPE_INT01_RECEIPTS",
    "JOPE_INT01_EXECUTABLE_PIN",
    "JOPE_INT01_CONTEXT_SHA256",
    "JOPE_GOLDEN_DIR",
    "CARGO_TARGET_DIR",
)


def command() -> list[str]:
    return [
        "cargo",
        "test",
        "--locked",
        "-p",
        "jope-appliance",
        "--test",
        "e2e",
        "--features",
        "engine,mcp",
        TEST_NAME,
        "--",
        "--exact",
        "--test-threads=1",
    ]


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in UNTRUSTED_TEST_INPUTS:
        env.pop(name, None)
    env.update(
        {
            "CARGO_BUILD_JOBS": "4",
            "CARGO_INCREMENTAL": "0",
            "CARGO_PROFILE_DEV_DEBUG": "line-tables-only",
        }
    )
    return env


def check_binding(
    ctx: Context, run: Callable[..., subprocess.CompletedProcess]
) -> None:
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError(
                "S1 requires the clean, candidate-pinned source checkout"
            )


def ensure_external_result(ctx: Context) -> None:
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("S1 result must be a new regular file")
    try:
        ctx.result.resolve(strict=False).relative_to(ROOT.resolve())
    except ValueError:
        return
    raise ProviderError("S1 result must be outside the source checkout")


def write_result(ctx: Context, status: str, counts: dict[str, int]) -> None:
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "S1",
        "status": status,
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": counts,
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    with ctx.result.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")


def qualify(
    ctx: Context,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    if ctx.gate_id != "S1":
        raise ProviderError("S1 provider requires gate S1")
    ensure_external_result(ctx)
    verify_candidate(ctx, run=run)
    check_binding(ctx, run)
    before = sha256(ctx.artifact)
    if before != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed before S1 assertions")

    completed = run(
        command(),
        cwd=ROOT,
        env=child_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        return completed.returncode

    counts = rust_counts(completed.stdout)
    check_binding(ctx, run)
    if sha256(ctx.artifact) != before:
        raise ProviderError("candidate bytes changed during S1 assertions")
    selected_test_passed = f"test {TEST_NAME} ... ok" in completed.stdout.splitlines()
    status = "PASS" if counts == SINGLE_PASS and selected_test_passed else "FAIL"
    write_result(ctx, status, counts)
    return 0 if status == "PASS" else 1


def main() -> int:
    try:
        return qualify(context("S1"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"S1 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
