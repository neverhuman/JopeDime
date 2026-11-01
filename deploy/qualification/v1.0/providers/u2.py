#!/usr/bin/env python3
"""Candidate-bound Linux assertions for REL-07 U2 terminal restoration."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path

from common import (
    Context,
    ProviderError,
    context,
    rust_counts,
    sha256,
    verify_candidate,
)


ROOT = Path(__file__).resolve().parents[4]
TESTS = (
    "terminal::app::tests::linux_u2_restores_normal_error_panic_and_three_real_signals",
    "terminal::app::tests::linux_fallbacks_never_change_raw_mode_and_resize_preserves_draft",
    "terminal::app::tests::no_color_replays_all_five_real_recorded_fixtures_without_escape_bytes",
    "terminal::restore::tests::restoration_commands_show_cursor_reset_colors_and_disable_paste",
    "terminal::tests::detects_each_color_level_and_disables_color_when_requested",
    "terminal::tests::strips_escape_sequences_and_controls_without_losing_plain_text",
    "terminal::tests::uses_unicode_symbols_only_for_utf8_locales",
)
SINGLE_PASS = {
    "collected": 1,
    "executed": 1,
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}
SUMMARY = re.compile(
    r"test result: (?:ok|FAILED)\. \d+ passed; \d+ failed; "
    r"\d+ ignored; \d+ measured; \d+ filtered out"
)


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
        "--nocapture",
        "--color=never",
        "--test-threads=1",
    ]


def check_binding(ctx: Context, run) -> None:
    if (
        not ctx.artifact.is_absolute()
        or ctx.artifact.is_symlink()
        or not ctx.artifact.is_file()
        or sha256(ctx.artifact) != ctx.artifact_sha256
    ):
        raise ProviderError("U2 candidate must remain the declared regular file")
    if (
        not ctx.result.is_absolute()
        or ctx.result.exists()
        or ctx.result.is_symlink()
        or ctx.result.resolve().is_relative_to(ROOT.resolve())
    ):
        raise ProviderError("U2 result must be a new path outside the checkout")
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError(
                "U2 requires the clean candidate-pinned source checkout"
            )


def case_counts(output: str, test: str) -> dict[str, int]:
    if len(SUMMARY.findall(output)) != 1 or output.count(f"test {test} ... ok") != 1:
        raise ProviderError(f"U2 did not execute exactly the required test {test}")
    counts = rust_counts(output)
    if counts != SINGLE_PASS:
        raise ProviderError(f"U2 requires exactly one passing case for {test}")
    return counts


def qualify(ctx: Context, run=None, system: str | None = None) -> int:
    run = subprocess.run if run is None else run
    system = platform.system() if system is None else system
    if system != "Linux":
        raise ProviderError(
            "U2 currently has exact terminal restoration acceptance tests only on Linux"
        )

    check_binding(ctx, run)
    verify_candidate(ctx, run=run)
    totals = {key: 0 for key in SINGLE_PASS}
    for test in TESTS:
        check_binding(ctx, run)
        completed = run(
            command(test),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        check_binding(ctx, run)
        if completed.returncode != 0:
            return completed.returncode
        counts = case_counts(completed.stdout, test)
        for key, value in counts.items():
            totals[key] += value

    check_binding(ctx, run)
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "U2",
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": totals,
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return qualify(context("U2"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"U2 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
