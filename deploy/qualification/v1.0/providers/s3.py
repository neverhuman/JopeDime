#!/usr/bin/env python3
"""REL-07 S3: candidate-bound FIT-row, finalization and CV leakage checks."""

from __future__ import annotations

import json
import os
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
    rust_counts,
    sha256,
    verify_candidate,
)


TESTS = (
    "discover::audit::tests::dataset_namespace_and_every_phase_permission_are_enforced",
    "discover::audit::tests::invalid_rosters_empty_access_and_reader_failures_are_not_passing_evidence",
    "discover::audit::tests::search_log_contains_only_fit_rows_and_refusal_is_atomic",
    "discover::folds::tests::grouped_constraints_hold_in_every_phase",
    "discover::folds::tests::time_and_gap_are_strict_for_all_phases",
    "discover::folds::tests::combined_time_ties_groups_and_gap_are_never_relaxed",
    "discover::folds::tests::overlapping_group_intervals_remain_whole_in_a_feasible_time_split",
    "discover::folds::tests::nested_calibration_inherits_constraints_and_excludes_outer_validation",
    "discover::folds::tests::caps_cannot_introduce_outside_rows_and_bad_inputs_do_not_fallback",
    "discover::finalize::tests::finalization_audits_each_real_partition_and_declares_the_refit_and_score_scopes",
    "discover::finalize::tests::completed_replay_and_recovery_do_not_read_or_score_protected_partitions_again",
    "discover::finalize::tests::failed_bounded_fit_preserves_audit_and_never_retries_or_reads_holdout_or_test",
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
        "-j",
        "8",
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
        ctx.gate_id != "S3"
        or not SOURCE_SHA.fullmatch(ctx.source_sha)
        or not ARTIFACT_SHA256.fullmatch(ctx.artifact_sha256)
    ):
        raise ProviderError("S3 requires its own gate and full source/artifact hashes")
    for path in (ctx.artifact, ctx.result):
        if not path.is_absolute() or any(
            part.is_symlink() for part in (path, *path.parents)
        ):
            raise ProviderError("S3 paths must be absolute and contain no symlinks")
    if not ctx.artifact.is_file():
        raise ProviderError("S3 candidate must be a regular file")
    if ctx.result.exists():
        raise ProviderError("S3 result must be a new regular file")
    if any(p.exists() and not p.is_dir() for p in ctx.result.parents):
        raise ProviderError("S3 result parents must be directories")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("S3 evidence must be outside the source checkout")


def check_binding(ctx: Context, run) -> None:
    check_paths(ctx)
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError("S3 requires the clean, candidate-pinned checkout")
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during S3")


def qualify(ctx: Context, run=None) -> int:
    run = subprocess.run if run is None else run
    check_binding(ctx, run)
    verify_candidate(ctx, run=run)
    env = os.environ.copy()
    env.update(CARGO_INCREMENTAL="0", CARGO_PROFILE_DEV_DEBUG="line-tables-only")
    for test in TESTS:
        check_binding(ctx, run)
        args = command(test)
        print(json.dumps({"test": test, "argv": args}), flush=True)
        result = run(
            args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,
            env=env,
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        check_binding(ctx, run)
        if result.returncode != 0:
            return result.returncode
        summaries = re.findall(r"^test result: .+$", result.stdout, re.MULTILINE)
        cases = re.findall(r"^test (\S+) \.\.\. (\S+)$", result.stdout, re.MULTILINE)
        if (
            len(summaries) != 1
            or not summaries[0].startswith("test result: ok.")
            or rust_counts(result.stdout) != SINGLE_PASS
            or cases != [(test, "ok")]
        ):
            raise ProviderError(f"S3 requires exactly one passing case for {test}")
    check_binding(ctx, run)
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "S3",
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
        return qualify(context("S3"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"S3 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
