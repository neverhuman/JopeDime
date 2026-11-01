#!/usr/bin/env python3
"""REL-07 R3: candidate-bound idempotency and conflict behavior."""

from __future__ import annotations

import json
import os
import re
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
    ("same_request_replays_original_outcome_and_operation_after_reopen", False),
    ("principals_do_not_share_keys_and_concurrent_retries_execute_once", False),
    ("abort_rolls_back_both_effect_and_command_and_retry_can_succeed", False),
    ("malformed_saved_receipt_fails_closed_without_running_effect", False),
    ("conflict_is_409_problem_and_successful_http_replay_is_identical", False),
    ("request_boundary_replays_and_conflicts_without_reapplying_metadata", False),
    ("actual_crash_retry_is_once_after_commit_and_clean_before_commit", True),
)
SINGLE_PASS = {
    "collected": 1,
    "executed": 1,
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}


def command(test: str, failpoints: bool) -> list[str]:
    args = ["cargo", "test", "--locked", "-p", "jope-service", "--lib"]
    if failpoints:
        args.extend(["--features", "failpoints"])
    args.extend(
        [
            f"store::commands::tests::{test}",
            "--",
            "--exact",
            "--test-threads=1",
            "--nocapture",
        ]
    )
    return args


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    try:
        requested_jobs = int(env.get("CARGO_BUILD_JOBS", "8"))
    except ValueError:
        requested_jobs = 8
    env["CARGO_BUILD_JOBS"] = str(max(1, min(requested_jobs, 8)))
    env["CARGO_INCREMENTAL"] = "0"
    env["CARGO_PROFILE_DEV_DEBUG"] = "line-tables-only"
    return env


def check_result_path(ctx: Context) -> None:
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("R3 result must be a new regular file")
    if not ctx.result.is_absolute():
        raise ProviderError("R3 result path must be absolute")
    for part in (ctx.result, *ctx.result.parents):
        if part.is_symlink():
            raise ProviderError("R3 result path cannot contain symlinks")
    if not ctx.result.parent.is_dir():
        raise ProviderError("R3 result parent must be an existing directory")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("R3 evidence must be outside the source checkout")


def check_binding(ctx: Context, run, env: dict[str, str]) -> None:
    if (
        not ctx.artifact.is_absolute()
        or any(part.is_symlink() for part in (ctx.artifact, *ctx.artifact.parents))
        or not ctx.artifact.is_file()
    ):
        raise ProviderError(
            "R3 candidate must remain an absolute regular non-symlink file"
        )
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(
            args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError(
                "R3 requires the clean candidate-pinned source checkout"
            )
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during R3")


def validate_output(test: str, output: str) -> None:
    summaries = re.findall(r"^test result: .+$", output, re.MULTILINE)
    if (
        output.count("test result:") != 1
        or len(summaries) != 1
        or not summaries[0].startswith("test result: ok.")
        or rust_counts(output) != SINGLE_PASS
    ):
        raise ProviderError(f"R3 requires exactly one passing case for {test}")
    expected = f"store::commands::tests::{test}"
    cases = re.findall(r"^test (\S+) \.\.\. ", output, re.MULTILINE)
    if cases != [expected] or not exact_case_passed(output, expected):
        raise ProviderError(f"R3 exact selector did not pass: {test}")


def qualify(ctx: Context, run=None) -> int:
    run = subprocess.run if run is None else run
    if ctx.gate_id != "R3":
        raise ProviderError("R3 provider requires gate R3")
    check_result_path(ctx)
    env = child_environment()

    def child(args, **kwargs):
        kwargs.setdefault("env", env)
        return run(args, **kwargs)

    check_binding(ctx, child, env)
    verify_candidate(ctx, run=child)
    check_binding(ctx, child, env)

    for test, failpoints in TESTS:
        args = command(test, failpoints)
        print(json.dumps({"test": test, "argv": args}), flush=True)
        completed = child(
            args,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        check_binding(ctx, child, env)
        if completed.returncode != 0:
            return completed.returncode
        validate_output(test, completed.stdout)

    check_binding(ctx, child, env)
    check_result_path(ctx)
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": ctx.gate_id,
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": {key: value * len(TESTS) for key, value in SINGLE_PASS.items()},
    }
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return qualify(context("R3"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"R3 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
