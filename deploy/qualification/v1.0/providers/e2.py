#!/usr/bin/env python3
"""REL-07 E2: bind the real-toolchain export parity harness to a candidate."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from common import ROOT, Context, ProviderError, context, sha256, verify_candidate


HARNESS = "deploy/ci/export-parity.sh"
LANGUAGES = {"python", "rust", "javascript"}


def regular(path: Path) -> None:
    if (
        not path.is_absolute()
        or any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
    ):
        raise ProviderError("E2 inputs must be absolute regular non-symlink files")


def check_binding(ctx: Context, run) -> None:
    regular(ctx.artifact)
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during E2")
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode or result.stdout.strip() != expected:
            raise ProviderError("E2 requires the clean, candidate-pinned checkout")


def load(path: Path) -> dict:
    regular(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ProviderError("E2 report objects must be JSON objects")
    return value


def validate_report(folder: Path, ctx: Context) -> tuple[dict, int]:
    """Check the harness's complete census, rather than treating exit zero as proof."""
    report = load(folder / "report.json")
    if (
        report.get("schema") != "jope.export-parity/v1"
        or report.get("status") != "pass"
        or report.get("passed") is not True
        or report.get("source_sha") != ctx.source_sha
        or report.get("source_end") != ctx.source_sha
        or report.get("source_stable") is not True
        or report.get("source_dirty") != ""
        or report.get("source_dirty_end") != ""
        or report.get("fault_injected") is not False
    ):
        raise ProviderError("E2 requires a fresh, clean, complete parity PASS")
    tools = report["toolchains"]
    if (
        tools.get("python") != "Python 3.12.3"
        or tools.get("node") != "v26.1.0"
        or not tools.get("rustc", "").startswith("rustc 1.97.1 ")
    ):
        raise ProviderError("E2 requires the harness's pinned real toolchains")
    for key, path in (
        ("harness_sha256", ROOT / HARNESS),
        ("cargo_lock_sha256", ROOT / "Cargo.lock"),
        ("registry_sha256", folder / "registry.json"),
        ("fixture_sha256", folder / "fixtures.json"),
    ):
        regular(path)
        if report.get(key) != sha256(path):
            raise ProviderError(f"E2 {key} identity mismatch")

    files = report["files"]
    if (
        not files
        or not {"fixtures.json", "registry.json", "harness.sh"} <= files.keys()
    ):
        raise ProviderError("E2 omitted its retained evidence inventory")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ProviderError("E2 evidence paths cannot escape their report")
        path = folder / relative
        regular(path)
        if (
            path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise ProviderError("E2 retained evidence changed")

    fixtures = load(folder / "fixtures.json")
    cells, observed = fixtures["cells"], report["cells"]

    def identity(row):
        return row["key"], row["language"], row["supported"]

    if not cells or Counter(map(identity, cells)) != Counter(map(identity, observed)):
        raise ProviderError("E2 registry cell inventory differs from the fixture")
    supported = sum(row["supported"] is True for row in cells)
    if (
        supported <= 0
        or report["expected_supported_cells"] != supported
        or report["executed_supported_cells"] != supported
        or report["unsupported_cells"] != len(cells) - supported
        or any(
            row["status"] != ("pass" if row["supported"] else "unsupported_registry")
            for row in observed
        )
    ):
        raise ProviderError("E2 supported cells must all execute and pass")

    def modules(expected, actual):
        slots = [
            (case["key"], e["language"]) for case in expected for e in case["emissions"]
        ]
        seen = [(case["key"], case["language"]) for case in actual]
        if (
            not slots
            or len(set(slots)) != len(slots)
            or Counter(slots) != Counter(seen)
            or any(
                case["status"] != "pass"
                or type(case["rows"]) is not int
                or case["rows"] <= 0
                or type(case["self_check_rows"]) is not int
                or case["self_check_rows"] <= 0
                for case in actual
            )
        ):
            raise ProviderError(
                "E2 requires every emitted module and populated self-check"
            )
        return len(slots)

    count = modules(fixtures["cases"], report["cases"])
    if (
        report["expected_emitted_modules"] != count
        or report["executed_emitted_modules"] != count
        or not any(case["key"].startswith("routing/") for case in report["cases"])
    ):
        raise ProviderError("E2 omitted emitted modules or routing probes")
    calibration = report["calibration"]
    calibrated = modules(fixtures["calibration"], calibration["cases"])
    if (
        calibration["strict"] is not True
        or calibration["expected_modules"] != calibrated
        or calibration["executed_modules"] != calibrated
        or calibration["passed_modules"] != calibrated
        or calibration["failed"] != []
        or {case["capability"] for case in fixtures["calibration"]}
        != {"affine_floor", "platt"}
    ):
        raise ProviderError("E2 requires strict affine/floor and Platt goldens")
    raw = report["raw_numeric_contract"]
    if (
        len(raw) != 3
        or {case["language"] for case in raw} != LANGUAGES
        or any(case["status"] != "pass" or case["rows"] <= 0 for case in raw)
        or {case["language"] for case in report["cases"]} != LANGUAGES
        or any(command["exit"] != 0 for command in report["commands"])
        or not report["commands"]
    ):
        raise ProviderError(
            "E2 requires all three real-language and raw-contract executions"
        )
    return report, count + calibrated + len(raw)


def qualify(ctx: Context, run=None) -> int:
    run = subprocess.run if run is None else run
    if (
        not ctx.result.is_absolute()
        or ctx.result.exists()
        or any(p.is_symlink() for p in (ctx.result, *ctx.result.parents))
        or ctx.result.resolve().is_relative_to(ROOT.resolve())
    ):
        raise ProviderError(
            "E2 result must be new, outside the checkout and non-symlink"
        )
    jobs = os.environ.get("CARGO_BUILD_JOBS", "8")
    if jobs not in {str(i) for i in range(1, 9)}:
        raise ProviderError("E2 build jobs must be 1..8")
    environment = os.environ.copy()
    environment.update(
        CARGO_BUILD_JOBS=jobs,
        CARGO_INCREMENTAL="0",
        CARGO_PROFILE_DEV_DEBUG="line-tables-only",
        RAYON_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
    )
    check_binding(ctx, run)

    def candidate_command(args, **kwargs):
        return run(["nice", "-n", "10", *args], env=environment, **kwargs)

    verify_candidate(ctx, run=candidate_command)
    check_binding(ctx, run)
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    retained = Path(
        tempfile.mkdtemp(prefix=ctx.result.name + ".parity-", dir=ctx.result.parent)
    )
    folder = retained / "report"
    args = ["nice", "-n", "10", "bash", HARNESS, "--require-calibration", str(folder)]
    print(json.dumps({"argv": args, "report": str(folder)}), flush=True)
    completed = run(
        args,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    check_binding(ctx, run)
    if completed.returncode:
        return completed.returncode
    report, count = validate_report(folder, ctx)
    check_binding(ctx, run)
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": ctx.gate_id,
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": dict(
            collected=count,
            executed=count,
            passed=count,
            failed=0,
            skipped=0,
            excluded=0,
        ),
        "parity": {
            "report": str(folder / "report.json"),
            "sha256": sha256(folder / "report.json"),
            "supported_cells": report["executed_supported_cells"],
            "unsupported_registry_cells": report["unsupported_cells"],
            "toolchains": report["toolchains"],
        },
    }
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return qualify(context("E2"))
    except (
        OSError,
        subprocess.TimeoutExpired,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(f"E2 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
