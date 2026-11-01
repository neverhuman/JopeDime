#!/usr/bin/env python3
"""CPU provider for REL-07 M1 fused-bundle reproducibility and freeze checks."""

from __future__ import annotations

import sys
from pathlib import Path

from common import ProviderError, context, junit_counts, run_suite, verify_candidate


ROOT = Path(__file__).resolve().parents[4]


def command(result: Path) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        str(ROOT / "python"),
        "--locked",
        "--no-sync",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--junitxml",
        str(result.with_suffix(".junit.xml")),
        str(ROOT / "research/flywheel_selfplay/test_export_fused_bundle.py"),
    ]


def main() -> int:
    try:
        ctx = context("M1")
        verify_candidate(ctx)
        return run_suite(
            ctx,
            command(ctx.result),
            junit_counts,
            ctx.result.with_suffix(".junit.xml"),
        )
    except (OSError, ProviderError) as error:
        print(f"M1 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
