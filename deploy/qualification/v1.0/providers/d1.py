#!/usr/bin/env python3
"""REL-07 D1: candidate-bound Linux clean-host run plus macOS arm64 receipt."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from common import (
    ROOT,
    Context,
    ProviderError,
    context,
    sha256,
    verify_candidate,
)


MAC_RECEIPT_SCHEMA = "jope.rel07.d1-platform-leg/v1"
LINUX_DRIVER = ROOT / "deploy/appliance/test-clean-host-prestage.sh"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
RUN_PASS_COUNTS = {
    "collected": 1,
    "executed": 1,
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}


def read_regular_json(path: Path, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ProviderError(f"D1 {label} must be a regular, non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProviderError(f"D1 {label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise ProviderError(f"D1 {label} must contain a JSON object")
    return value


def verify_macos_receipt(
    ctx: Context, path: Path, expected_sha256: str
) -> tuple[dict, str]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ProviderError(
            "D1 BLOCKED: macOS arm64 receipt must be an absolute regular non-symlink file"
        )
    if not SHA256.fullmatch(expected_sha256):
        raise ProviderError("D1 macOS receipt SHA-256 must be lowercase full hex")
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise ProviderError("D1 macOS receipt SHA-256 differs from its binding")
    receipt = read_regular_json(path, "macOS arm64 receipt")
    if sha256(path) != actual_sha256:
        raise ProviderError("D1 macOS receipt changed while being verified")
    if receipt.get("schema") != MAC_RECEIPT_SCHEMA:
        raise ProviderError("D1 macOS arm64 receipt has an unsupported schema")
    if receipt.get("gate_id") != "D1" or receipt.get("platform") != "macos-arm64":
        raise ProviderError("D1 macOS receipt is not a macOS arm64 D1 leg")
    if receipt.get("status") != "PASS":
        raise ProviderError("D1 macOS arm64 leg did not PASS")
    if receipt.get("source_sha") != ctx.source_sha:
        raise ProviderError(
            "D1 macOS receipt source SHA differs from the candidate source"
        )
    if receipt.get("artifact_sha256") != ctx.artifact_sha256:
        raise ProviderError(
            "D1 macOS receipt candidate SHA-256 differs from this candidate"
        )

    checks = receipt.get("checks")
    exports = checks.get("exports") if isinstance(checks, dict) else None
    required = {
        "network_disabled": True,
        "toolchain_absent": True,
        "discovery": True,
        "prediction": True,
        "link_audit_os_libraries_only": True,
    }
    if not isinstance(checks, dict) or any(
        checks.get(key) is not True for key in required
    ):
        raise ProviderError("D1 macOS receipt omits a required clean-host assertion")
    if not isinstance(exports, dict) or exports != {
        "python": True,
        "rust": True,
        "javascript": True,
    }:
        raise ProviderError("D1 macOS receipt must pass all three run exports")
    return receipt, actual_sha256


def verify_linux_receipt(ctx: Context, path: Path) -> dict:
    receipt = read_regular_json(path, "Linux clean-host receipt")
    if receipt.get("schema") != "jope.appliance-clean-host/v1":
        raise ProviderError("D1 Linux clean-host receipt has an unsupported schema")
    if receipt.get("qualification") != "pre-stage" or receipt.get("status") != "passed":
        raise ProviderError("D1 Linux clean-host driver did not pass its golden path")
    if receipt.get("exit_status") != 0 or receipt.get("source_sha") != ctx.source_sha:
        raise ProviderError("D1 Linux clean-host receipt is not bound to this source")
    if receipt.get("binary_sha256") != ctx.artifact_sha256:
        raise ProviderError(
            "D1 Linux clean-host receipt is not bound to this candidate"
        )
    if receipt.get("isolation") != "network namespace; loopback only; no default route":
        raise ProviderError("D1 Linux clean-host run did not prove network isolation")
    if not str(receipt.get("toolchain_path", "")).startswith("empty ("):
        raise ProviderError("D1 Linux clean-host run did not hide the toolchain")
    if receipt.get("method_discovery") != "passed":
        raise ProviderError("D1 Linux method discovery did not pass")
    discovery = receipt.get("run_discovery")
    if not isinstance(discovery, dict) or discovery.get("status") != "passed":
        raise ProviderError("D1 Linux run discovery did not pass")
    exports = receipt.get("export_status")
    if not isinstance(exports, dict) or exports != {
        "python": "passed",
        "rust": "passed",
        "javascript": "passed",
    }:
        raise ProviderError("D1 Linux run must pass all three exports")
    prediction_rows = receipt.get("prediction_rows")
    if type(prediction_rows) is not int or prediction_rows <= 0:
        raise ProviderError("D1 Linux prediction produced no rows")
    if receipt.get("link_audit") != "passed":
        raise ProviderError("D1 Linux OS-library link audit did not pass")
    return receipt


def run_linux_leg(ctx: Context, run) -> tuple[dict, Path]:
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise ProviderError("D1 Linux leg requires a Linux x86-64 host")
    evidence_value = os.environ.get("JOPE_LANE_EVIDENCE")
    if not evidence_value:
        raise ProviderError("D1 requires the release runner evidence directory")
    evidence = Path(evidence_value)
    if evidence.is_symlink() or not evidence.is_absolute():
        raise ProviderError("D1 evidence directory must be absolute and non-symlink")
    evidence.mkdir(parents=True, exist_ok=True)
    if evidence.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("D1 evidence directory must be outside the source checkout")
    work = evidence / "linux-clean-host"
    if work.exists() or work.is_symlink():
        raise ProviderError("D1 Linux clean-host work directory must be new")
    receipt_path = work / "clean-host.json"
    command = [
        "bash",
        str(LINUX_DRIVER),
        str(ctx.artifact),
        str(work),
        "--require-golden-path",
    ]
    print(json.dumps({"platform": "linux-x86_64", "argv": command}), flush=True)
    completed = run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
        env=os.environ.copy(),
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise ProviderError(
            f"D1 Linux clean-host leg failed with exit {completed.returncode}"
        )
    return verify_linux_receipt(ctx, receipt_path), receipt_path


def check_binding(ctx: Context, run) -> None:
    for argv, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError(
                "D1 requires the clean, candidate-pinned source checkout"
            )
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("D1 candidate bytes changed during qualification")


def qualify(
    ctx: Context,
    macos_receipt_path: Path,
    macos_receipt_sha256: str,
    run=None,
) -> int:
    runner = subprocess.run if run is None else run
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("D1 result must be a new regular file")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("D1 result must be outside the source checkout")

    macos_receipt, macos_sha = verify_macos_receipt(
        ctx, macos_receipt_path, macos_receipt_sha256
    )
    check_binding(ctx, runner)
    verify_candidate(ctx, run=runner)
    check_binding(ctx, runner)
    linux_receipt, linux_receipt_path = run_linux_leg(ctx, runner)
    check_binding(ctx, runner)

    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "D1",
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": {key: 2 * value for key, value in RUN_PASS_COUNTS.items()},
        "platforms": {
            "linux-x86_64": {
                "receipt": str(linux_receipt_path),
                "checks": [
                    "network-disabled",
                    "no-python-rustc-node",
                    "method-and-run-discovery",
                    "python-rust-javascript-exports",
                    "prediction",
                    "os-library-link-audit",
                ],
                "prediction_rows": linux_receipt["prediction_rows"],
            },
            "macos-arm64": {
                "receipt_sha256": macos_sha,
                "source_sha": macos_receipt["source_sha"],
                "artifact_sha256": macos_receipt["artifact_sha256"],
            },
        },
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--macos-receipt",
        type=Path,
        default=os.environ.get("JOPE_REL07_D1_MACOS_RECEIPT"),
    )
    parser.add_argument(
        "--macos-receipt-sha256",
        default=os.environ.get("JOPE_REL07_D1_MACOS_RECEIPT_SHA256"),
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = arguments()
        if args.macos_receipt is None or args.macos_receipt_sha256 is None:
            raise ProviderError(
                "D1 BLOCKED: macOS arm64 receipt path and SHA-256 binding are required"
            )
        return qualify(context("D1"), args.macos_receipt, args.macos_receipt_sha256)
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"D1 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
