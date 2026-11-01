#!/usr/bin/env python3
"""REL-07 A4: candidate-bound MCP protocol conformance, without speed verdicts."""

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


# Each selector must execute exactly once. The first lifecycle test itself covers
# all four shipped versions; the remaining cases cover schemas, aliases and guards.
# The separate native-fitting p95 test is outside the owner's no-speed scope.
TESTS = (
    "mcp::aliases::tests::alias_schemas_preserve_legacy_arguments",
    "mcp::aliases::tests::fitness_legacy_handler_keeps_its_result_shape",
    "mcp::aliases::tests::lists_every_legacy_name_as_deprecated_with_replacement",
    "mcp::aliases::tests::removed_narration_tools_return_structured_removal_error",
    "mcp::dispatch::export_destination_tests::empty_export_destination_is_rejected_before_any_transport_call",
    "mcp::tests::canonical_schema_gaps_are_explicit_and_cached_resources_are_bounded",
    "mcp::tests::discovery_does_not_allow_other_premature_notifications",
    "mcp::tests::discovery_with_early_roots_change_keeps_modern_bootstrap_open",
    "mcp::tests::four_versions_lifecycle_twelve_tools_resources_and_stdout_purity",
    "mcp::tests::generated_input_and_success_error_output_schemas_validate_actual_wire_values",
    "mcp::tests::invalid_arguments_side_effect_guards_notifications_and_version_exclusion",
    "mcp::tests::invalid_discovery_metadata_does_not_allow_a_roots_hint",
    "mcp::tests::legacy_aliases_are_listed_and_dispatched_through_existing_handlers",
    "mcp::tests::local_model_path_is_rejected_before_dataset_upload",
    "mcp::tests::long_poll_uses_bounded_http_slices_and_returns_within_the_requested_deadline",
    "mcp::tests::malformed_frame_recovers_and_notification_cancellation_suppresses_reply",
    "mcp::tests::model_path_accepts_only_an_explicit_artifact_id",
    "mcp::tests::model_predict_uploads_local_csv_and_atomically_writes_requested_output",
    "mcp::tests::modern_request_metadata_and_string_ids_are_enforced",
    "mcp::tests::prediction_file_retry_verifies_bytes_and_preserves_changed_output",
    "mcp::tests::roots_change_without_discovery_does_not_open_a_session",
    "mcp::tests::stdout_relay_refuses_non_rpc_bytes_without_forwarding_them",
    "mcp::tests::uncached_manifest_is_reported_as_explicitly_unavailable",
)
SINGLE_PASS = dict(collected=1, executed=1, passed=1, failed=0, skipped=0, excluded=0)


def command(test: str) -> list[str]:
    return [
        "cargo",
        "test",
        "--locked",
        "-p",
        "jope-appliance",
        "--bin",
        "jopedime",
        test,
        "--",
        "--exact",
        "--test-threads=1",
        "--color=never",
    ]


def controlled_runner(run):
    environment = os.environ.copy()
    environment.update(
        CARGO_INCREMENTAL="0",
        CARGO_PROFILE_DEV_DEBUG="line-tables-only",
        CARGO_BUILD_JOBS="4",
    )

    def invoke(args, **kwargs):
        return run(args, **{**kwargs, "env": environment})

    return invoke


def check_binding(ctx: Context, run) -> None:
    if (
        not ctx.artifact.is_absolute()
        or ctx.artifact.is_symlink()
        or not ctx.artifact.is_file()
    ):
        raise ProviderError("candidate must remain a regular non-symlink file")
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError("provider requires the clean candidate-pinned checkout")
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during gate assertions")


def new_result(ctx: Context) -> None:
    if not ctx.result.is_absolute() or ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("result must be a new absolute non-symlink path")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("result must be outside the source checkout")


def publish(ctx: Context, count: int, run, **details) -> int:
    check_binding(ctx, run)
    new_result(ctx)
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": ctx.gate_id,
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": {key: count * value for key, value in SINGLE_PASS.items()},
        **details,
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def qualify(ctx: Context, run=None) -> int:
    run = controlled_runner(subprocess.run if run is None else run)
    if ctx.gate_id != "A4":
        raise ProviderError("provider requires gate A4")
    new_result(ctx)
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
        new_result(ctx)
        if result.returncode != 0:
            return result.returncode
        if (
            rust_counts(result.stdout) != SINGLE_PASS
            or result.stdout.splitlines().count(f"test {test} ... ok") != 1
            or result.stdout.count("test result:") != 1
        ):
            raise ProviderError(f"A4 requires exactly one passing case for {test}")
    return publish(
        ctx,
        len(TESTS),
        run,
        protocols=["2024-11-05", "2025-06-18", "2025-11-25", "2026-07-28"],
        timing="not measured: owner no-speed-gates policy",
    )


def main() -> int:
    try:
        return qualify(context("A4"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"A4 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
