#!/usr/bin/env python3
"""CPU provider for REL-07 M4 signed bundle-integrity refusal controls."""

from __future__ import annotations

import sys
from pathlib import Path

from common import ProviderError, context, run_suite, rust_counts, verify_candidate


ROOT = Path(__file__).resolve().parents[4]


def command() -> list[str]:
    return [
        "cargo",
        "test",
        "--locked",
        "-p",
        "jope-service",
        "--lib",
        "ranker::bundle::integrity_tests::m4_",
    ]


def main() -> int:
    try:
        ctx = context("M4")
        verify_candidate(ctx)
        return run_suite(ctx, command(), rust_counts)
    except (OSError, ProviderError) as error:
        print(f"M4 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
