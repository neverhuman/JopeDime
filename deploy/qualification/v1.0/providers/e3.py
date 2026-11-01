#!/usr/bin/env python3
"""REL-07 E3: candidate-bound export publication and filesystem safety."""

from __future__ import annotations

import json
import re
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


TESTS = (
    (
        "jope-service",
        "export::bundle::tests::unsupported_required_language_fails_before_writer_publication",
    ),
    (
        "jope-service",
        "export::bundle::tests::manifest_rejects_traversal_and_bad_hashes",
    ),
    ("jope-service", "export::bundle::tests::retention_flag_tracks_training_row_heads"),
    (
        "jope-client",
        "files::tests::verified_zip_is_extracted_and_force_backs_up_existing_output",
    ),
    (
        "jope-client",
        "files::tests::mismatch_existing_output_and_interrupted_archive_leave_output_untouched",
    ),
    ("jope-client", "files::tests::refuses_parent_traversal_and_symlink_entries"),
    (
        "jope-client",
        "files::tests::export_retry_checks_every_existing_byte_and_refuses_tampering",
    ),
)
SINGLE_PASS = {
    "collected": 1,
    "executed": 1,
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}


def command(crate: str, test: str) -> list[str]:
    return [
        "cargo",
        "test",
        "--locked",
        "-p",
        crate,
        "--lib",
        test,
        "--",
        "--exact",
        "--test-threads=1",
    ]


def check_binding(ctx: Context, run) -> None:
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError(
                "E3 requires the clean, candidate-pinned source checkout"
            )
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during E3")


def qualify(ctx: Context, run=None) -> int:
    run = subprocess.run if run is None else run
    if ctx.gate_id != "E3":
        raise ProviderError("provider requires gate E3")
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("E3 result must be a new regular file")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("E3 evidence must be outside the source checkout")
    check_binding(ctx, run)
    verify_candidate(ctx, run=run)
    for crate, test in TESTS:
        check_binding(ctx, run)
        args = command(crate, test)
        print(json.dumps({"crate": crate, "test": test, "argv": args}), flush=True)
        result = run(
            args, cwd=ROOT, capture_output=True, text=True, check=False, timeout=3600
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        check_binding(ctx, run)
        if result.returncode != 0:
            return result.returncode
        if (
            rust_counts(result.stdout) != SINGLE_PASS
            or len(re.findall(r"^test result:", result.stdout, re.MULTILINE)) != 1
            or len(re.findall(r"^running 1 test$", result.stdout, re.MULTILINE)) != 1
            or re.findall(r"^test (\S+) \.\.\. ok$", result.stdout, re.MULTILINE)
            != [test]
        ):
            raise ProviderError(
                f"E3 requires exactly one passing case for {crate}::{test}"
            )
    check_binding(ctx, run)
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "E3",
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
        return qualify(context("E3"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"E3 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
