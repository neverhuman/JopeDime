#!/usr/bin/env python3
"""REL-07 A2: candidate-bound generated schemas, typed errors and u64 wire values."""

from __future__ import annotations

from functools import partial
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


CHECKS = (
    (
        "jope-api",
        ("--bin", "jope-api-openapi"),
        "tests::generated_document_matches_committed_openapi",
    ),
    (
        "jope-api",
        ("--bin", "jope-api-openapi"),
        "tests::generated_document_covers_registry_without_pending_exchanges",
    ),
    (
        "jope-api",
        ("--test", "contract"),
        "every_top_level_schema_matches_the_committed_snapshot",
    ),
    (
        "jope-api",
        ("--lib",),
        "error::tests::every_appendix_d_row_matches_exactly_and_round_trips",
    ),
    (
        "jope-api",
        ("--lib",),
        "error::tests::problem_serializes_standard_members_extensions_and_media_type",
    ),
    (
        "jope-api",
        ("--lib",),
        "error::tests::problem_rejects_unknown_code_and_missing_required_extensions",
    ),
    (
        "jope-api",
        ("--lib",),
        "error::tests::error_schema_is_a_closed_string_catalog_and_problem_exposes_the_profile",
    ),
    (
        "jope-api",
        ("--lib",),
        "decimal::tests::serializes_and_deserializes_boundary_values_as_strings",
    ),
    ("jope-api", ("--lib",), "decimal::tests::rejects_json_numbers"),
    ("jope-api", ("--lib",), "decimal::tests::json_schema_describes_a_string"),
    (
        "jope-api",
        ("--test", "contract"),
        "event_kind_selects_its_body_and_cursors_stay_strings",
    ),
    (
        "jope-service",
        ("--lib",),
        "http::auth::tests::every_http_catalog_error_has_a_problem_json_response",
    ),
    (
        "jope-service",
        ("--lib",),
        "http::router::tests::all_registry_method_path_pairs_reach_explicit_capability_stubs",
    ),
    (
        "jope-service",
        ("--lib",),
        "http::router::tests::fallbacks_have_typed_errors_and_remain_authenticated",
    ),
)
SINGLE_PASS = dict(collected=1, executed=1, passed=1, failed=0, skipped=0, excluded=0)


def command(check: tuple) -> list[str]:
    crate, target, test = check
    return [
        "cargo",
        "test",
        "--locked",
        "-j",
        "4",
        "-p",
        crate,
        *target,
        test,
        "--",
        "--exact",
        "--test-threads=1",
    ]


def check_paths(ctx: Context) -> None:
    if (
        ctx.gate_id != "A2"
        or not SOURCE_SHA.fullmatch(ctx.source_sha)
        or not ARTIFACT_SHA256.fullmatch(ctx.artifact_sha256)
    ):
        raise ProviderError("A2 requires its own gate and full source/artifact hashes")
    for path in (ctx.artifact, ctx.result):
        if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ProviderError("A2 paths must be absolute and contain no symlinks")
    if not ctx.artifact.is_file():
        raise ProviderError("A2 candidate must be a regular file")
    if ctx.result.exists():
        raise ProviderError("A2 result must be a new regular file")
    if any(p.exists() and not p.is_dir() for p in ctx.result.parents):
        raise ProviderError("A2 result parents must be directories")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("A2 evidence must be outside the source checkout")


def check_binding(ctx: Context, run) -> None:
    check_paths(ctx)
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError("A2 requires the clean, candidate-pinned checkout")
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during A2")


def qualify(ctx: Context, run=None) -> int:
    env = os.environ.copy()
    env.update(CARGO_INCREMENTAL="0", CARGO_PROFILE_DEV_DEBUG="line-tables-only")
    run = partial(subprocess.run if run is None else run, env=env)
    check_binding(ctx, run)
    verify_candidate(ctx, run=run)
    for check in CHECKS:
        test = check[2]
        check_binding(ctx, run)
        args = command(check)
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
        cases = re.findall(r"^test (\S+) \.\.\. (\S+)$", result.stdout, re.MULTILINE)
        if (
            len(summaries) != 1
            or not summaries[0].startswith("test result: ok.")
            or rust_counts(result.stdout) != SINGLE_PASS
            or cases != [(test, "ok")]
        ):
            raise ProviderError(f"A2 requires exactly one passing case for {test}")
    check_binding(ctx, run)
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "A2",
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": {key: value * len(CHECKS) for key, value in SINGLE_PASS.items()},
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    check_binding(ctx, run)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return qualify(context("A2"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"A2 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
