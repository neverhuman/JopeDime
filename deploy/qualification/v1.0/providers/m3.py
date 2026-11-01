#!/usr/bin/env python3
"""REL-07 M3 source checks; refuses PASS while full parity evidence is incomplete."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys

from common import (
    exact_case_passed,
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
    reports = []
    for line in output.splitlines():
        # Under --nocapture the diagnostic follows libtest's "test NAME ... ".
        head, separator, report = line.partition(DIAGNOSTIC_PREFIX)
        if separator and (not head or head.endswith(" ... ")):
            reports.append(report)
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
        if rust_counts(result.stdout) != SINGLE_PASS or not exact_case_passed(
            result.stdout, test
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


def input_preflight(store: Path, catalog_sha256: str) -> dict:
    """Verify the pinned fixture inventory without claiming numerical parity."""

    def require(condition, message):
        if not condition:
            raise ProviderError(f"M3 input preflight: {message}")

    def digest(value):
        require(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "lowercase full SHA256 required",
        )
        return value

    require(
        store.is_absolute() and store.is_dir(), "absolute custody directory required"
    )
    require(
        store.resolve() == store
        and not any(p.is_symlink() for p in (store, *store.parents)),
        "custody directory must not traverse symlinks",
    )

    def object_file(pin, size=None):
        path = store / digest(pin)
        before = path.lstat()
        require(
            stat.S_ISREG(before.st_mode),
            "custody object must be a regular non-symlink file",
        )
        require(0 < before.st_size <= 64 * 1024 * 1024, "custody object byte limit")
        require(size is None or before.st_size == size, "custody object length differs")
        require(sha256(path) == pin, "custody object hash differs")
        after = path.lstat()
        require(
            stat.S_ISREG(after.st_mode)
            and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "custody object changed during verification",
        )
        return path, before.st_size

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate catalog JSON key")
            result[key] = value
        return result

    catalog_file, catalog_bytes = object_file(catalog_sha256)
    catalog_raw = catalog_file.read_bytes()
    require(
        hashlib.sha256(catalog_raw).hexdigest() == catalog_sha256,
        "catalog bytes changed before parsing",
    )
    try:
        catalog = json.loads(catalog_raw, object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProviderError("M3 input preflight: malformed catalog JSON") from error
    # Recheck the exact bytes after parsing; a catalog swap cannot redirect inventory reads.
    object_file(catalog_sha256, catalog_bytes)
    require(isinstance(catalog, dict), "catalog object required")
    require(
        catalog.get("schema") == "jope.golden-custody/v1"
        and catalog.get("storage") == "sha256-custody",
        "hash-custody catalog required",
    )
    files = catalog.get("files")
    require(
        isinstance(files, dict) and 0 < len(files) <= 10000,
        "nonempty bounded file inventory required",
    )
    sizes = {}
    payload_bytes = 0
    for name, spec in files.items():
        require(
            isinstance(name, str)
            and name
            and "\\" not in name
            and "\x00" not in name
            and not PurePosixPath(name).is_absolute()
            and all(part not in (".", "..") for part in name.split("/"))
            and all(name.split("/")),
            "safe relative logical fixture path required",
        )
        require(isinstance(spec, dict), "file specification required")
        size = spec.get("bytes")
        require(
            type(size) is int and 0 < size <= 64 * 1024 * 1024,
            "positive bounded file byte count required",
        )
        pin = digest(spec.get("sha256"))
        if pin in sizes:
            require(sizes[pin] == size, "one object has contradictory lengths")
        else:
            object_file(pin, size)
            sizes[pin] = size
        payload_bytes += size
    totals = dict(
        file_count=len(files) + 1,
        payload_bytes=payload_bytes,
        total_bytes=payload_bytes + catalog_bytes,
        unique_payload_bytes=sum(sizes.values()),
        unique_custody_bytes=sum(sizes.values()) + catalog_bytes,
    )
    for name, observed in totals.items():
        require(
            type(catalog.get(name)) is int and catalog[name] == observed,
            f"catalog {name} differs from verified inventory",
        )
    coverage = catalog.get("dsl_coverage")
    require(isinstance(coverage, dict), "declared DSL coverage required")
    for name in (
        "delivered_unique_programs",
        "evaluated_source_records",
        "parent_source_records",
        "covered_historical_programs",
        "missing_historical_programs",
        "excluded_unlogged",
    ):
        require(
            type(coverage.get(name)) is int and coverage[name] >= 0,
            f"invalid declared DSL count: {name}",
        )
    for name in ("historical_corpus_complete", "original_historical_state_recovered"):
        require(type(catalog.get(name)) is bool, f"strict boolean required: {name}")
    object_file(catalog_sha256, catalog_bytes)
    return dict(
        schema="jope.rel07.m3-input-preflight/v1",
        id="M3",
        status="INPUTS_VERIFIED_NOT_QUALIFIED",
        catalog_sha256=catalog_sha256,
        catalog_bytes=catalog_bytes,
        logical_files_verified=len(files),
        unique_objects_verified=len(sizes) + 1,
        inventory_totals=totals,
        declared_scope=catalog.get("historical_scope"),
        declared_authority=catalog.get("authority"),
        declared_dsl_coverage=coverage,
        reconstructed_corpus_complete=catalog["historical_corpus_complete"],
        original_historical_state_recovered=catalog[
            "original_historical_state_recovered"
        ],
        tests_executed=0,
        release_qualified=False,
        missing_scientific_comparisons=list(INCOMPLETE),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-inputs", action="store_true")
    parser.add_argument("--store", type=Path)
    parser.add_argument("--catalog-sha256")
    args = parser.parse_args(argv)
    try:
        if args.preflight_inputs:
            if args.store is None or args.catalog_sha256 is None:
                raise ProviderError(
                    "M3 input preflight requires --store and --catalog-sha256"
                )
            print(
                json.dumps(
                    input_preflight(args.store, args.catalog_sha256), sort_keys=True
                )
            )
            return 0
        if args.store is not None or args.catalog_sha256 is not None:
            raise ProviderError("M3 custody arguments require --preflight-inputs")
        return qualify(context("M3"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"M3 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
