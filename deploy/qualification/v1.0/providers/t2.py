#!/usr/bin/env python3
"""REL-07 T2: candidate-bound real-run no-GP-leak assertion."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
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
REAL_RUN_TEST = "bounded_real_run_http_status_and_sse_do_not_leak_internal_sentinels"
POSITIVE_TESTS = (
    "synthetic_internal_fields_stay_out_of_current_public_wire_surfaces",
    "scanner_reports_a_deliberately_leaking_surface",
    "scanner_rejects_each_sentinel_on_every_public_surface",
    REAL_RUN_TEST,
)
SINGLE_PASS = {
    "collected": 1,
    "executed": 1,
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}
SINGLE_EXPECTED_FAILURE = {
    "collected": 1,
    "executed": 1,
    "passed": 0,
    "failed": 1,
    "skipped": 0,
    "excluded": 0,
}
LEAK_CONTROL = 'Leak { surface: Http, sentinel: "T2_DSL_SENTINEL_8f27" }'


def build_command() -> list[str]:
    return ["cargo", "build", "--locked", "-p", "jope-appliance", "--bin", "jopedime"]


def test_command(test: str) -> list[str]:
    return [
        "cargo",
        "test",
        "--locked",
        "-p",
        "jope-service",
        "--features",
        "failpoints",
        "--test",
        "no_gp_leak",
        "--",
        test,
        "--exact",
        "--test-threads=1",
    ]


def command_environment(leak_control: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CARGO_TARGET_DIR", None)
    env.pop("JOPE_PUB06_TEST_LEAK", None)
    env["CARGO_BUILD_JOBS"] = "8"
    env["CARGO_INCREMENTAL"] = "0"
    env["CARGO_PROFILE_DEV_DEBUG"] = "line-tables-only"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    if leak_control:
        env["JOPE_PUB06_TEST_LEAK"] = "1"
    return env


def check_binding(
    ctx: Context, run: Callable[..., subprocess.CompletedProcess]
) -> None:
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(
            args, cwd=ROOT, capture_output=True, text=True, check=False, timeout=30
        )
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError(
                "T2 requires the clean checkout pinned to candidate source"
            )
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during T2")


def _run(
    ctx: Context,
    args: list[str],
    run: Callable[..., subprocess.CompletedProcess],
    *,
    leak_control: bool = False,
) -> subprocess.CompletedProcess:
    check_binding(ctx, run)
    print(
        json.dumps(
            {"argv": args, "expected": "detected-leak" if leak_control else "pass"}
        )
    )
    result = run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
        env=command_environment(leak_control),
    )
    sys.stdout.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    check_binding(ctx, run)
    return result


def _require_one_pass(test: str, result: subprocess.CompletedProcess) -> None:
    output = (result.stdout or "") + (result.stderr or "")
    counts = rust_counts(output)
    if (
        result.returncode != 0
        or counts != SINGLE_PASS
        or f"test {test} ... ok" not in output
    ):
        raise ProviderError(f"T2 requires exactly one passing case for {test}")


def _require_leak_control(result: subprocess.CompletedProcess) -> None:
    output = (result.stdout or "") + (result.stderr or "")
    counts = rust_counts(output)
    if (
        result.returncode == 0
        or counts != SINGLE_EXPECTED_FAILURE
        or REAL_RUN_TEST not in output
        or LEAK_CONTROL not in output
    ):
        raise ProviderError(
            "T2 product-path negative control did not detect the expected leak"
        )


def qualify(
    ctx: Context, run: Callable[..., subprocess.CompletedProcess] | None = None
) -> int:
    run = subprocess.run if run is None else run
    if ctx.gate_id != "T2":
        raise ProviderError("T2 provider requires gate T2")
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("T2 result must be a new regular file")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("T2 evidence must be outside the source checkout")

    check_binding(ctx, run)
    verify_candidate(ctx, run=run)
    build = _run(ctx, build_command(), run)
    if build.returncode != 0:
        return build.returncode

    for test in POSITIVE_TESTS:
        result = _run(ctx, test_command(test), run)
        if result.returncode != 0:
            return result.returncode
        _require_one_pass(test, result)

    negative = _run(ctx, test_command(REAL_RUN_TEST), run, leak_control=True)
    _require_leak_control(negative)
    print(
        json.dumps(
            {
                "negative_control": "PASS",
                "test": REAL_RUN_TEST,
                "detected": LEAK_CONTROL,
            }
        )
    )
    check_binding(ctx, run)

    # Counts cover positive Rust test cases only. The intentional failing control
    # is separately logged and accepted only for the exact expected sentinel.
    assertions = len(POSITIVE_TESTS)
    counts = {
        "collected": assertions,
        "executed": assertions,
        "passed": assertions,
        "failed": 0,
        "skipped": 0,
        "excluded": 0,
    }
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "T2",
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": counts,
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return qualify(context("T2"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"T2 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
