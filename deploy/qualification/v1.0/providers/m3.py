#!/usr/bin/env python3
"""REL-07 M3 source checks; refuses PASS while full parity evidence is incomplete."""

from __future__ import annotations

import json
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
    "ranker::featurizer::structural::tests::all_1309_train_program_records_match_all_30_python_golden_coordinates_exactly",
    "ranker::featurizer::parameters::tests::all_1309_row_free_programs_match_the_pinned_python_oracle",
    "ranker::featurizer::positions::tests::all_1309_row_free_programs_match_the_pinned_python_position_oracle",
    "ranker::featurizer::tests::full_composition_covers_all_1309_reference_records_in_frozen_order",
    "ranker::proj16::tests::matrix_matches_custody_python_fixture_exactly",
    "ranker::proj16::tests::matches_python_projection_on_one_thousand_seeded_feat412_vectors",
    "discover::propose::tests::real_train_parents_preserve_round_determinism_and_language_exclusions",
    "discover::propose::tests::ten_thousand_train_parent_outputs_report_m3_frequency_comparison",
)
FREQUENCY_DIAGNOSTIC = TESTS[-1]
DIAGNOSTIC_PREFIX = "DSC06_M3_FREQUENCY_REPORT="
SINGLE_PASS = {
    "collected": 1,
    "executed": 1,
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}
INCOMPLETE = (
    "full historical feat412 corpus",
    "full-recipe landmark extraction compared with training capture",
    "rung-0 full_train_cv evaluator values and identified R0 evaluator",
    "R0-comparable proposer frequencies across exploration modes",
)


def command(test: str) -> list[str]:
    args = [
        "cargo",
        "test",
        "--locked",
        "-p",
        "jope-service",
        "--lib",
        test,
        "--",
        "--exact",
        "--test-threads=1",
    ]
    if test == FREQUENCY_DIAGNOSTIC:
        args.append("--nocapture")
    return args


def check_binding(ctx: Context, run) -> None:
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError(
                "M3 requires the clean, candidate-pinned source checkout"
            )
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during M3")


def _frequency_report(output: str) -> dict:
    reports = [
        line[len(DIAGNOSTIC_PREFIX) :]
        for line in output.splitlines()
        if line.startswith(DIAGNOSTIC_PREFIX)
    ]
    if len(reports) != 1:
        raise ProviderError(
            "M3 proposer test must emit exactly one frequency diagnostic"
        )
    try:
        report = json.loads(reports[0])
    except json.JSONDecodeError as error:
        raise ProviderError(
            "M3 proposer frequency diagnostic is invalid JSON"
        ) from error
    if (
        report.get("schema") != "jopedime.dsc06-generated-frequency-diagnostic/v1"
        or report.get("population_comparable_to_reference") is not False
        or report.get("m3_qualified") is not False
    ):
        raise ProviderError(
            "M3 proposer diagnostic no longer states its limited population"
        )
    return report


def qualify(ctx: Context, run=None) -> int:
    run = subprocess.run if run is None else run
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("M3 result must be a new regular file")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("M3 evidence must be outside the source checkout")
    check_binding(ctx, run)
    verify_candidate(ctx, run=run)
    diagnostic = None
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
        if result.returncode != 0:
            return result.returncode
        if (
            rust_counts(result.stdout) != SINGLE_PASS
            or f"test {test} ... ok" not in result.stdout
        ):
            raise ProviderError(f"M3 requires exactly one passing case for {test}")
        if test == FREQUENCY_DIAGNOSTIC:
            diagnostic = _frequency_report(result.stdout)
    if diagnostic is None:
        raise ProviderError("M3 proposer frequency diagnostic was not collected")

    # The source assertions above are genuine bounded controls, but they do not
    # establish every release criterion in §14. Never create a PASS receipt from
    # this subset; wait for full-historical and training-capture comparisons.
    check_binding(ctx, run)
    missing = "; ".join(INCOMPLETE)
    raise ProviderError(f"M3 remains not qualified; missing evidence: {missing}")


def main() -> int:
    try:
        return qualify(context("M3"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"M3 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
