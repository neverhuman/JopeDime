#!/usr/bin/env python3
"""REL-07 S4: candidate-bound user objective authority and unchanged ranker inputs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from functools import partial

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
    "discover::r#loop::tests::custom_objective_controls_every_promotion_despite_opposing_surrogate",
    "ranker::admission::tests::custom_objective_changes_elite_order_without_changing_ranker_inputs",
    "discover::elites::tests::capacity_evicts_by_user_objective_even_for_the_best_surrogate",
    "discover::elites::tests::directions_are_independent_and_maximized_objective_evicts_the_lowest",
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
        ctx.gate_id != "S4"
        or not SOURCE_SHA.fullmatch(ctx.source_sha)
        or not ARTIFACT_SHA256.fullmatch(ctx.artifact_sha256)
    ):
        raise ProviderError("S4 requires its own gate and full source/artifact hashes")
    for path in (ctx.artifact, ctx.result):
        if not path.is_absolute() or any(
            part.is_symlink() for part in (path, *path.parents)
        ):
            raise ProviderError("S4 paths must be absolute and contain no symlinks")
    if not ctx.artifact.is_file():
        raise ProviderError("S4 candidate must be a regular file")
    if ctx.result.exists():
        raise ProviderError("S4 result must be a new regular file")
    if any(p.exists() and not p.is_dir() for p in ctx.result.parents):
        raise ProviderError("S4 result parents must be directories")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("S4 evidence must be outside the source checkout")


def check_binding(ctx: Context, run) -> None:
    check_paths(ctx)
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError("S4 requires the clean, candidate-pinned checkout")
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during S4")


def qualify(ctx: Context, run=None) -> int:
    env = os.environ.copy()
    env.update(CARGO_INCREMENTAL="0", CARGO_PROFILE_DEV_DEBUG="line-tables-only")
    run = partial(subprocess.run if run is None else run, env=env)
    check_binding(ctx, run)
    verify_candidate(ctx, run=run)
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
            raise ProviderError(f"S4 requires exactly one passing case for {test}")
    check_binding(ctx, run)
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "S4",
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
        return qualify(context("S4"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"S4 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
