#!/usr/bin/env python3
"""Candidate-bound CPU ranker parity provider for REL-07 M2."""

from __future__ import annotations

import argparse
import json
import math
import os
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
INDEX_SHA256 = "ddaa6cfc558f0a7e56fe73f657266d00d6c821739a602afc701f87fa1c4f5af1"
TEST = "ranker::cuda::fixture::real_full_forward_cpu_fixture_replay"
REQUIRED_STAGES = {"v256", "col", "dope629", "lm48", "tokens", "mask", "costs"}
SINGLE_PASS = {
    "collected": 1,
    "executed": 1,
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}


def command() -> list[str]:
    return [
        "cargo",
        "test",
        "--locked",
        "-p",
        "jope-service",
        "--lib",
        TEST,
        "--",
        "--ignored",
        "--exact",
        "--color=never",
        "--test-threads=1",
    ]


def check_binding(
    ctx: Context, run, *, require_fresh_report: bool = True
) -> tuple[Path, Path, Path]:
    if (
        not ctx.artifact.is_absolute()
        or ctx.artifact.is_symlink()
        or not ctx.artifact.is_file()
        or sha256(ctx.artifact) != ctx.artifact_sha256
    ):
        raise ProviderError("M2 candidate must remain the declared regular file")
    if (
        not ctx.result.is_absolute()
        or ctx.result.exists()
        or ctx.result.is_symlink()
        or ctx.result.resolve().is_relative_to(ROOT.resolve())
    ):
        raise ProviderError("M2 result must be a new path outside the checkout")
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError(
                "M2 requires the clean candidate-pinned source checkout"
            )

    golden_value = os.environ.get("JOPE_GOLDEN_DIR")
    bundle_value = os.environ.get("JOPE_PARITY_BUNDLE_DIR")
    report_value = os.environ.get("JOPE_PARITY_REPORT_DIR")
    if not golden_value or not bundle_value or not report_value:
        raise ProviderError(
            "M2 requires the admitted RES-07 packet, frozen RES-06 bundle, and new report directory"
        )
    golden = Path(golden_value)
    bundle = Path(bundle_value)
    report = Path(report_value)
    if (
        not golden.is_absolute()
        or golden.is_symlink()
        or not golden.is_dir()
        or not bundle.is_absolute()
        or bundle.is_symlink()
        or not bundle.is_dir()
        or not report.is_absolute()
        or report.is_symlink()
        or report.resolve().is_relative_to(ROOT.resolve())
    ):
        raise ProviderError(
            "M2 inputs must be absolute directories; report must be outside the checkout"
        )
    index_sha = os.environ.get("JOPE_GOLDEN_INDEX_SHA256")
    if index_sha != INDEX_SHA256:
        raise ProviderError("M2 requires the owner-accepted RES-07 R5 packet index")
    index = golden / "index.json"
    if index.is_symlink() or not index.is_file() or sha256(index) != INDEX_SHA256:
        raise ProviderError("M2 RES-07 index does not match the accepted packet")
    for name in ("dope.safetensors", "jope.safetensors"):
        tensor_file = bundle / name
        if tensor_file.is_symlink() or not tensor_file.is_file():
            raise ProviderError(f"M2 frozen bundle requires a regular {name} file")
    if require_fresh_report and report.exists():
        raise ProviderError("M2 report directory must not already exist")
    if not require_fresh_report and not report.is_dir():
        raise ProviderError("M2 report directory is missing after creation")
    return golden, bundle, report


def validate_parity_report(path: Path, source_sha: str, index_sha: str) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProviderError("M2 CPU replay omitted a valid RNK-12 report") from error
    if (
        report.get("source") != source_sha
        or report.get("consumer") != "RNK-12"
        or report.get("fixture_index_sha256") != index_sha
        or report.get("contexts") != 50
        or report.get("slates") != 100
        or report.get("passed") is not True
        or report.get("failed_values") != 0
        or report.get("trace_errors") != []
        or report.get("admitted_set_mismatches_outside_tie_band") != 0
        or type(report.get("decisive_admission_slates")) is not int
        or type(report.get("tie_band_slates")) is not int
        or report["decisive_admission_slates"] + report["tie_band_slates"] != 100
    ):
        raise ProviderError(
            "M2 CPU replay report failed its source, coverage, or parity checks"
        )
    stages = report.get("stages")
    if not isinstance(stages, dict) or not REQUIRED_STAGES.issubset(stages):
        raise ProviderError("M2 CPU replay report omitted a required numerical stage")
    for stage in REQUIRED_STAGES:
        values = stages[stage]
        if (
            not isinstance(values, dict)
            or not isinstance(values.get("values"), int)
            or values["values"] <= 0
            or values.get("failed_values") != 0
            or not isinstance(values.get("max_tolerance_ratio"), (int, float))
            or not math.isfinite(values["max_tolerance_ratio"])
            or values["max_tolerance_ratio"] > 1.0
        ):
            raise ProviderError(
                f"M2 CPU replay report has invalid {stage} parity evidence"
            )
    return report


def qualify(ctx: Context, run=None) -> int:
    run = subprocess.run if run is None else run
    raw_jobs = os.environ.get("CARGO_BUILD_JOBS", "4")
    if raw_jobs not in {str(value) for value in range(1, 5)}:
        raise ProviderError("M2 CPU replay requires 1..4 inherited Cargo build jobs")
    golden, bundle, report_dir = check_binding(ctx, run)
    verify_candidate(ctx, run=run)
    check_binding(ctx, run)
    report_dir.mkdir(mode=0o700)
    child_env = os.environ.copy()
    child_env.update(
        {
            "JOPE_SOURCE_SHA": ctx.source_sha,
            "JOPE_GOLDEN_DIR": str(golden),
            "JOPE_GOLDEN_INDEX_SHA256": INDEX_SHA256,
            "JOPE_PARITY_BUNDLE_DIR": str(bundle),
            "JOPE_PARITY_REPORT_DIR": str(report_dir),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "CARGO_BUILD_JOBS": raw_jobs,
            "CARGO_INCREMENTAL": "0",
            "CARGO_PROFILE_DEV_DEBUG": "line-tables-only",
        }
    )
    completed = run(
        command(),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
        env=child_env,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    check_binding(ctx, run, require_fresh_report=False)
    if completed.returncode != 0:
        return completed.returncode
    counts = rust_counts(completed.stdout)
    if counts != SINGLE_PASS or completed.stdout.count(f"test {TEST} ... ok") != 1:
        raise ProviderError(
            "M2 must execute exactly its one ignored real CPU replay test"
        )
    validate_parity_report(report_dir / "RNK-12.json", ctx.source_sha, INDEX_SHA256)
    check_binding(ctx, run, require_fresh_report=False)
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("M2 candidate changed during parity assertions")
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "M2",
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": counts,
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir")
    parser.add_argument("--golden-index-sha256")
    parser.add_argument("--parity-bundle-dir")
    parser.add_argument("--parity-report-dir")
    args = parser.parse_args(argv)
    for option, variable in (
        (args.golden_dir, "JOPE_GOLDEN_DIR"),
        (args.golden_index_sha256, "JOPE_GOLDEN_INDEX_SHA256"),
        (args.parity_bundle_dir, "JOPE_PARITY_BUNDLE_DIR"),
        (args.parity_report_dir, "JOPE_PARITY_REPORT_DIR"),
    ):
        if option is not None:
            os.environ[variable] = option
    try:
        return qualify(context("M2"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"M2 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
