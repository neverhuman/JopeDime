#!/usr/bin/env python3
"""Candidate-bound M5 causal admission replay with real frozen CPU fixtures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

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

TEST = "ranker::adapter::real_cpu_parity::m5_serving_causal_measurements"
INDEX_SHA256 = "27905f29703431bedacac90eed1bd54257fd0c59bdf24b56605d0b9416ae1393"
R882_CATALOG_SHA256 = "3af56a817bfae39b70f0b2f60668fe88fae178650ee596e1ba7a180acba1e0c6"
R882_DIGEST = "sha256:057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc"
R782_DIGEST = "sha256:94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c"
R782_JOPE_SHA256 = "ce39c75f37624e64dfbf89e6493adbf74811a5248f9355997161491e93a132ef"
R882_JOPE_SHA256 = "ff3f9ea5d76fa6c13cd74d40a9effbf7e9c891d98746dc542d851726b9e3138c"
SINGLE_PASS = dict(collected=1, executed=1, passed=1, failed=0, skipped=0, excluded=0)
DEFAULT_OFF_PASS = dict(
    collected=3, executed=3, passed=3, failed=0, skipped=0, excluded=0
)


@dataclass(frozen=True)
class Inputs:
    golden: Path
    bundle: Path


@dataclass(frozen=True)
class DefaultOffInput:
    receipt: Path
    receipt_sha256: str


def default_off_report(ctx: Context, inputs: DefaultOffInput) -> dict:
    """Validate the R21 waiver evidence without running the waived causal replay."""
    regular(inputs.receipt)
    if inputs.receipt.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("M5 default-off receipt must be external to the checkout")
    if not re.fullmatch(r"[0-9a-f]{64}", inputs.receipt_sha256):
        raise ProviderError("M5 default-off receipt requires a full SHA-256 pin")
    if sha256(inputs.receipt) != inputs.receipt_sha256:
        raise ProviderError("M5 default-off receipt bytes differ from the pinned hash")
    try:
        report = json.loads(inputs.receipt.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProviderError(
            "M5 default-off receipt must be valid UTF-8 JSON"
        ) from error
    if (
        not isinstance(report, dict)
        or set(report)
        != {
            "schema",
            "source_sha",
            "status",
            "default_config",
            "serving_admission",
            "s2_determinism",
        }
        or report["schema"] != "jope.rnk10.default-off/v1"
        or report["source_sha"] != ctx.source_sha
        or report["status"] != "PASS"
    ):
        raise ProviderError("M5 requires a source-bound R21 default-off PASS receipt")
    default = report["default_config"]
    serving = report["serving_admission"]
    determinism = report["s2_determinism"]
    if (
        not isinstance(default, dict)
        or set(default)
        != {"track", "learned_guidance_enabled", "config_sha256", "bundle_sha256"}
        or default["track"] != "heuristic"
        or default["learned_guidance_enabled"] is not False
        or any(
            not isinstance(default[key], str)
            or not re.fullmatch(r"[0-9a-f]{64}", default[key])
            for key in ("config_sha256", "bundle_sha256")
        )
    ):
        raise ProviderError("M5 receipt does not prove learned guidance defaults off")
    serving_keys = {"ranker_absent_sha256", "ranker_present_disabled_sha256"}
    if (
        not isinstance(serving, dict)
        or set(serving) != serving_keys
        or any(
            not isinstance(serving[key], str)
            or not re.fullmatch(r"[0-9a-f]{64}", serving[key])
            for key in serving_keys
        )
        or serving["ranker_absent_sha256"] != serving["ranker_present_disabled_sha256"]
    ):
        raise ProviderError(
            "M5 receipt does not prove byte-identical serving admission"
        )
    determinism_keys = {"run_1_sha256", "run_2_sha256"}
    if (
        not isinstance(determinism, dict)
        or set(determinism) != determinism_keys
        or any(
            not isinstance(determinism[key], str)
            or not re.fullmatch(r"[0-9a-f]{64}", determinism[key])
            for key in determinism_keys
        )
        or determinism["run_1_sha256"] != determinism["run_2_sha256"]
    ):
        raise ProviderError("M5 receipt does not prove default-path S2 determinism")
    return report


def qualify_default_off(ctx: Context, inputs: DefaultOffInput) -> int:
    report = default_off_report(ctx, inputs)
    if any(
        path.is_symlink() for path in (ctx.result, *ctx.result.parents)
    ) or ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("M5 result must be external and non-symlink")
    jobs = os.environ.get("CARGO_BUILD_JOBS", "8")
    if jobs not in {str(i) for i in range(1, 9)}:
        raise ProviderError("M5 build jobs must be 1..8")
    env = os.environ.copy()
    env.update(
        CARGO_BUILD_JOBS=jobs,
        CARGO_INCREMENTAL="0",
        CARGO_PROFILE_DEV_DEBUG="line-tables-only",
        CUDA_VISIBLE_DEVICES="",
    )

    def run(args, **kwargs):
        return subprocess.run(["nice", "-n", "10", *args], env=env, **kwargs)

    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode or result.stdout.strip() != expected:
            raise ProviderError("M5 requires a clean candidate-pinned source checkout")
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed before M5 default-off validation")
    verify_candidate(ctx, run=run)
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode or result.stdout.strip() != expected:
            raise ProviderError("M5 source changed during default-off validation")
    if (
        sha256(ctx.artifact) != ctx.artifact_sha256
        or sha256(inputs.receipt) != inputs.receipt_sha256
    ):
        raise ProviderError(
            "M5 candidate or default-off receipt changed during validation"
        )
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "M5",
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": DEFAULT_OFF_PASS,
        "default_off": {
            "receipt": str(inputs.receipt),
            "sha256": inputs.receipt_sha256,
            "schema": report["schema"],
        },
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def regular(path: Path) -> None:
    if (
        not path.is_absolute()
        or any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
    ):
        raise ProviderError("M5 requires absolute regular non-symlink input files")


def check_paths(ctx: Context, inputs: Inputs) -> None:
    if (
        ctx.gate_id != "M5"
        or not SOURCE_SHA.fullmatch(ctx.source_sha)
        or not ARTIFACT_SHA256.fullmatch(ctx.artifact_sha256)
    ):
        raise ProviderError("M5 requires its own gate and full source/artifact hashes")
    regular(ctx.artifact)
    if (
        not ctx.result.is_absolute()
        or any(p.is_symlink() for p in (ctx.result, *ctx.result.parents))
        or ctx.result.exists()
        or any(p.exists() and not p.is_dir() for p in ctx.result.parents)
        or ctx.result.resolve().is_relative_to(ROOT.resolve())
    ):
        raise ProviderError("M5 result must be new, external and non-symlink")
    for path, expected in (
        (inputs.golden / "index.json", INDEX_SHA256),
        (inputs.bundle / "release-catalog.json", R882_CATALOG_SHA256),
    ):
        regular(path)
        if sha256(path) != expected:
            raise ProviderError(
                "M5 requires the pinned TRAIN fixture and signed r882 catalog"
            )


def check_binding(ctx: Context, inputs: Inputs, run) -> None:
    check_paths(ctx, inputs)
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode or result.stdout.strip() != expected:
            raise ProviderError(
                "M5 requires the clean candidate-pinned source checkout"
            )
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during M5")


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
        "--test-threads=1",
        "--show-output",
    ]


def causal_report(path: Path) -> dict:
    regular(path)
    report = json.loads(path.read_text())
    if (
        report["schema"] != "jope.m5-causal/v3"
        or report["consumer"] != "M5-causal"
        or report["device"] != "cpu"
    ):
        raise ProviderError("M5 native report must contain the v2 CPU replay")
    output_checks = report["output_checks"]
    oracle = report["reference_oracle"]
    if (
        output_checks["finite_cost_values"] != 1600
        or output_checks["invalid_cost_values"] != 0
        or oracle["status"] != "not_comparable_model_weights"
        or oracle["comparison_performed"] is not False
        or oracle["reference_bundle_digest"] != R782_DIGEST
        or oracle["reference_jope_weights_sha256"] != R782_JOPE_SHA256
        or oracle["actual_bundle_digest"] != R882_DIGEST
        or oracle["actual_jope_weights_sha256"] != R882_JOPE_SHA256
    ):
        raise ProviderError("M5 report has invalid output checks or oracle identity")
    details = report["details"]
    serving = details["serving_4_plus_8"]
    if (
        details["contexts"] != 50
        or details["slates"] != 100
        or details["admitted_order_comparison"] != "not_comparable_model_weights"
        or details["admitted_order_mismatches"] is not None
        or details["quality_gate_granted"] is not False
        or details["Q1_ready"] is not False
        or serving["slates"] != 50
        or serving["landmark_slots"] != [0, 1, 7, 9]
        or serving["interventions"] != [-6.0, 0.0, 6.0]
        or serving["masks_and_dope_only_slots_preserved"] is not True
        or any(
            type(serving[key]) is not int or not 0 <= serving[key] <= 50
            for key in (
                "landmark_only_changed_admission",
                "fitness_changed_admission",
                "disconnected_heuristic_mismatches",
            )
        )
    ):
        raise ProviderError(
            "M5 requires 50 serving trials and bounded causal control counts"
        )
    expected_pass = (
        output_checks["invalid_cost_values"] == 0
        and serving["landmark_only_changed_admission"] > 0
        and serving["fitness_changed_admission"] > 0
        and serving["disconnected_heuristic_mismatches"] > 0
    )
    if report["passed"] is not expected_pass:
        raise ProviderError("M5 passed flag disagrees with its measured controls")
    return report


def qualify(ctx: Context, inputs: Inputs, run=None) -> int:
    execute = subprocess.run if run is None else run
    jobs = os.environ.get("CARGO_BUILD_JOBS", "8")
    if jobs not in {str(i) for i in range(1, 9)}:
        raise ProviderError("M5 build jobs must be 1..8")
    env = os.environ.copy()
    env.pop("JOPE_PARITY_BUNDLE_DIR", None)
    env.update(
        CARGO_BUILD_JOBS=jobs,
        CARGO_INCREMENTAL="0",
        CARGO_PROFILE_DEV_DEBUG="line-tables-only",
        RAYON_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
        JOPE_GOLDEN_DIR=str(inputs.golden),
        JOPE_GOLDEN_INDEX_SHA256=INDEX_SHA256,
        JOPE_RNK10_SIGNED_BUNDLE_DIR=str(inputs.bundle),
    )

    def run(args, **kwargs):
        return execute(["nice", "-n", "10", *args], env=env, **kwargs)

    check_binding(ctx, inputs, run)
    verify_candidate(ctx, run=run)
    check_binding(ctx, inputs, run)
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(
        tempfile.mkdtemp(prefix=ctx.result.name + ".m5-", dir=ctx.result.parent)
    )
    env["JOPE_PARITY_REPORT_DIR"] = str(folder)
    args = command()
    print(
        json.dumps({"argv": args, "native_report_directory": str(folder)}), flush=True
    )
    completed = run(
        args, cwd=ROOT, capture_output=True, text=True, check=False, timeout=3600
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    check_binding(ctx, inputs, run)
    if completed.returncode:
        return completed.returncode
    summaries = re.findall(r"^test result: .+$", completed.stdout, re.MULTILINE)
    cases = re.findall(r"^test (\S+) \.\.\. (\S+)$", completed.stdout, re.MULTILINE)
    if (
        len(summaries) != 1
        or not summaries[0].startswith("test result: ok.")
        or rust_counts(completed.stdout) != SINGLE_PASS
        or cases != [(TEST, "ok")]
    ):
        raise ProviderError(
            "M5 requires exactly one executed, passing real causal fixture test"
        )
    native_path = folder / "M5-causal.json"
    report = causal_report(native_path)
    check_binding(ctx, inputs, run)
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "M5",
        "status": "PASS" if report["passed"] is True else "FAIL",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": SINGLE_PASS,
        "causal": {
            "report": str(native_path),
            "sha256": sha256(native_path),
            "fixture_index_sha256": INDEX_SHA256,
            "adapter_bundle_digest": R882_DIGEST,
            "serving_4_plus_8": report["details"]["serving_4_plus_8"],
        },
    }
    check_binding(ctx, inputs, run)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    if report["passed"] is not True:
        print("M5 native replay completed with a genuine causal FAIL", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=os.environ.get("JOPE_GOLDEN_DIR") or None,
        help="Pinned TRAIN fixtures (default: JOPE_GOLDEN_DIR)",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=os.environ.get("JOPE_RNK10_SIGNED_BUNDLE_DIR") or None,
        help="Signed r882 release bundle (default: JOPE_RNK10_SIGNED_BUNDLE_DIR)",
    )
    parser.add_argument("--default-off-receipt", type=Path)
    parser.add_argument("--default-off-receipt-sha256")
    args = parser.parse_args()
    try:
        if (
            args.default_off_receipt is not None
            or args.default_off_receipt_sha256 is not None
        ):
            if (
                args.default_off_receipt is None
                or args.default_off_receipt_sha256 is None
                or args.golden_dir is not None
                or args.bundle_dir is not None
            ):
                raise ProviderError("M5 default-off receipt inputs cannot be mixed")
            return qualify_default_off(
                context("M5"),
                DefaultOffInput(
                    args.default_off_receipt, args.default_off_receipt_sha256
                ),
            )
        if args.golden_dir is None or args.bundle_dir is None:
            raise ProviderError(
                "M5 requires --golden-dir/JOPE_GOLDEN_DIR and "
                "--bundle-dir/JOPE_RNK10_SIGNED_BUNDLE_DIR"
            )
        return qualify(context("M5"), Inputs(args.golden_dir, args.bundle_dir))
    except (
        OSError,
        subprocess.TimeoutExpired,
        ProviderError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(f"M5 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
