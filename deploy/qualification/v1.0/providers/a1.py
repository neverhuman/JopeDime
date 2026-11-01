#!/usr/bin/env python3
"""REL-07 A1: in-process/HTTP contracts and the no-engine client against release bytes."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

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
    "golden_operations_match_http_and_in_process_including_idempotent_replays",
    "resumable_sse_uses_http_and_in_process_transports",
    "real_router_response_body_is_byte_identical_across_transports",
    "a1_lite_registry_and_candidate_background_contract",
)
LITE_TEST = TESTS[-1]

SINGLE_PASS = {
    "collected": 1,
    "executed": 1,
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}


def command(test: str) -> list[str]:
    args = ["cargo", "test", "--locked", "-p", "jope-client"]
    if test == LITE_TEST:
        args += ["--no-default-features", "--test", "a1_lite"]
    else:
        args += ["--features", "engine", "--test", "contract"]
    args += [test, "--", "--exact", "--test-threads=1"]
    if test == LITE_TEST:
        args += ["--ignored"]
    return args


def child_environment(ctx: Context) -> dict[str, str]:
    import os

    env = os.environ.copy()
    for key in [
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_RUSTFLAGS",
        "JOPEDIME_ENDPOINT",
        "JOPE_STORE_FAILPOINT",
        "JOPE_STORE_FAILPOINT_OCCURRENCE",
        "JOPE_STORE_FAILPOINT_PARK",
        "JOPE_STORE_TRACE_BOUNDARIES",
    ]:
        env.pop(key, None)
    env.update(
        JOPE_REL07_ARTIFACT=str(ctx.artifact),
        JOPE_REL07_SOURCE_SHA=ctx.source_sha,
        JOPE_QUALIFICATION_TEST_TMP=str(Path.home() / ".local/state/jope-a1"),
        CARGO_BUILD_JOBS="4",
        CARGO_INCREMENTAL="0",
        CARGO_PROFILE_DEV_DEBUG="line-tables-only",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        TOKIO_WORKER_THREADS="4",
    )
    return env


def check_binding(ctx: Context, run) -> None:
    """Standalone providers must enforce the coordinator's source binding too."""
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError(
                "A1 requires the clean, candidate-pinned source checkout"
            )
    if ctx.artifact.is_symlink() or not ctx.artifact.is_file():
        raise ProviderError("A1 candidate must remain a regular non-symlink file")
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during A1")


def qualify(ctx: Context, run=None) -> int:
    run = subprocess.run if run is None else run
    if ctx.gate_id != "A1":
        raise ProviderError("A1 requires gate A1")
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("A1 result must be a new regular file")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("A1 evidence must be outside the source checkout")
    check_binding(ctx, run)
    verify_candidate(ctx, run=run)
    for test in TESTS:
        check_binding(ctx, run)
        args = command(test)
        print(json.dumps({"test": test, "argv": args}), flush=True)
        result = run(
            args,
            cwd=ROOT,
            env=child_environment(ctx),
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        check_binding(ctx, run)
        if result.returncode != 0:
            return result.returncode
        counts = rust_counts(result.stdout)
        summaries = re.findall(r"^test result: .*", result.stdout, re.M)
        if (
            counts != SINGLE_PASS
            or len(summaries) != 1
            or result.stdout.splitlines().count(f"test {test} ... ok") != 1
        ):
            raise ProviderError(f"A1 requires exactly one passing case for {test}")
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
        return qualify(context("A1"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"A1 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
