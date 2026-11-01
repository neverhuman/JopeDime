#!/usr/bin/env python3
"""REL-07 R1: candidate-bound crash-consistency SIGKILL matrix."""

from __future__ import annotations

import json
import os
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


TEST = "r1_sigkill_at_public_blob_and_journal_boundaries_recovers_atomic_records_and_sequences"
EXPECTED_INTERNAL_CUTS = {
    f"INT02 CAS round={round_number} committed={round_number - 1} hook={hook} signal=9"
    for round_number in range(1, 4)
    for hook in (
        "after_write",
        "after_file_sync",
        "after_hash_verify",
        "after_rename",
        "after_directory_sync",
    )
}
SINGLE_PASS = {
    "collected": 1,
    "executed": 1,
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}


def command() -> list[str]:
    return [
        "cargo",
        "test",
        "--locked",
        "-p",
        "jope-appliance",
        "--test",
        "system",
        "--features",
        "jope-service/failpoints",
        TEST,
        "--",
        "--exact",
        "--test-threads=1",
        "--nocapture",
    ]


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    try:
        requested_jobs = int(env.get("CARGO_BUILD_JOBS", "8"))
    except ValueError:
        requested_jobs = 8
    env["CARGO_BUILD_JOBS"] = str(max(1, min(requested_jobs, 8)))
    env["CARGO_INCREMENTAL"] = "0"
    env["CARGO_PROFILE_DEV_DEBUG"] = "line-tables-only"
    env["INT02_INTERNAL_CAS_CUTS"] = "1"
    return env


def check_binding(ctx: Context, run, env: dict[str, str]) -> None:
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
                "R1 requires the clean candidate-pinned source checkout"
            )
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during R1")


def check_result_path(ctx: Context) -> None:
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("R1 result must be a new regular file")
    if not ctx.result.is_absolute():
        raise ProviderError("R1 result path must be absolute")
    for part in (ctx.result, *ctx.result.parents):
        if part.is_symlink():
            raise ProviderError("R1 result path cannot contain symlinks")
    if not ctx.result.parent.is_dir():
        raise ProviderError("R1 result parent must be an existing directory")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("R1 evidence must be outside the source checkout")


def validate_output(output: str, errors: str = "") -> None:
    """libtest reports on stdout; the system harness reports CAS cuts on stderr."""
    if rust_counts(output) != SINGLE_PASS:
        raise ProviderError("R1 requires exactly one passing exact system-test case")
    if f"test {TEST} ... ok" not in output:
        raise ProviderError("R1 exact crash-consistency test did not report success")
    streams = output + "\n" + errors
    missing = sorted(line for line in EXPECTED_INTERNAL_CUTS if line not in streams)
    if missing:
        raise ProviderError("R1 did not report every internal CAS SIGKILL cut")
    if streams.count("INT02 CAS round=") != len(EXPECTED_INTERNAL_CUTS):
        raise ProviderError(
            "R1 internal CAS cut count differs from the required matrix"
        )


def qualify(ctx: Context, run=None) -> int:
    run = subprocess.run if run is None else run
    if ctx.gate_id != "R1":
        raise ProviderError("R1 provider requires gate R1")
    check_result_path(ctx)
    env = child_environment()

    def child(args, **kwargs):
        kwargs.setdefault("env", env)
        return run(args, **kwargs)

    check_binding(ctx, child, env)
    verify_candidate(ctx, run=child)
    check_binding(ctx, child, env)
    args = command()
    print(json.dumps({"test": TEST, "argv": args}), flush=True)
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
    validate_output(completed.stdout, completed.stderr)
    check_result_path(ctx)
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": ctx.gate_id,
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": SINGLE_PASS,
    }
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
            raise ProviderError("R1 SIGKILL provider requires native Linux or macOS")
        if os.name != "posix":
            raise ProviderError("R1 SIGKILL provider requires native Unix signals")
        return qualify(context("R1"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"R1 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
