#!/usr/bin/env python3
"""REL-07 D2 size budget provider; latency is deliberately outside its verdict."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from common import ROOT, ProviderError, context, sha256


MAX_FINAL_BYTES = 40_000_000
CHECK_NAMES = {"base_executable_bytes", "final_executable_bytes"}


def cargo_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CARGO_TARGET_DIR", None)
    env.update(
        {
            "CARGO_BUILD_JOBS": "4",
            "CARGO_INCREMENTAL": "0",
            "CARGO_PROFILE_DEV_DEBUG": "line-tables-only",
        }
    )
    return env


def command(verb: str, artifact: Path) -> list[str]:
    return [
        "cargo",
        "run",
        "--locked",
        "-p",
        "jope-dist",
        "--",
        verb,
        "--file",
        str(artifact),
    ] + (
        ["--budget", "deploy/appliance/size-budgets.toml"]
        if verb == "size-report"
        else []
    )


def check_binding(ctx, run) -> None:
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError(
                "D2 requires the clean, candidate-pinned source checkout"
            )
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during D2")


def verify_candidate(ctx, run) -> None:
    before = sha256(ctx.artifact)
    for verb in ("inspect", "verify"):
        result = run(
            command(verb, ctx.artifact),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,
            env=cargo_env(),
        )
        if result.returncode != 0:
            raise ProviderError(f"jope-dist {verb} rejected the D2 candidate")
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ProviderError(f"jope-dist {verb} did not return JSON") from error
        if not isinstance(report, dict):
            raise ProviderError(f"jope-dist {verb} returned a non-object value")
        if verb == "inspect":
            try:
                manifest = report["manifest"]
                profile = str(manifest["profile"])
                source_sha = manifest["source"]["sha40"]
                scope = manifest["scope"]
            except (KeyError, TypeError) as error:
                raise ProviderError(
                    "candidate inspection omitted release identity"
                ) from error
            if Path(profile).stem != "fused-cpu" or scope == "development":
                raise ProviderError("D2 requires a non-development fused-cpu candidate")
            if source_sha != ctx.source_sha or not re.fullmatch(
                r"[0-9a-f]{40}", source_sha
            ):
                raise ProviderError(
                    "candidate source pin differs from the tested checkout"
                )
        elif (
            report.get("schema") != "jope.appliance-verify/v1"
            or report.get("ok") is not True
        ):
            raise ProviderError("candidate verification did not report full-byte PASS")
        if sha256(ctx.artifact) != before:
            raise ProviderError("candidate changed during D2 candidate verification")


def parse_size_report(output: str) -> tuple[dict, bool, int]:
    try:
        report = json.loads(output)
    except json.JSONDecodeError as error:
        raise ProviderError("jope-dist size-report did not return JSON") from error
    if not isinstance(report, dict):
        raise ProviderError("jope-dist size-report returned a non-object value")
    if report.get("schema") != "jope.appliance-size/v1":
        raise ProviderError("jope-dist size-report returned an unknown schema")
    if report.get("profile") != "fused-cpu":
        raise ProviderError("D2 size report is not for fused-cpu")
    if not isinstance(report.get("ok"), bool):
        raise ProviderError("size report has no boolean budget result")
    final_bytes = report.get("final_executable_bytes")
    if (
        isinstance(final_bytes, bool)
        or not isinstance(final_bytes, int)
        or final_bytes <= 0
    ):
        raise ProviderError("size report has no valid final executable byte count")
    base_bytes = report.get("base_executable_bytes")
    if (
        isinstance(base_bytes, bool)
        or not isinstance(base_bytes, int)
        or base_bytes <= 0
    ):
        raise ProviderError("size report has no valid base executable byte count")
    checks = report.get("budget_checks")
    if not isinstance(checks, list) or len(checks) != len(CHECK_NAMES):
        raise ProviderError("size report must contain both executable budget checks")
    by_name = {}
    for row in checks:
        if not isinstance(row, dict) or row.get("what") not in CHECK_NAMES:
            raise ProviderError("size report contains an unexpected budget check")
        if row["what"] in by_name or not isinstance(row.get("pass"), bool):
            raise ProviderError("size report has duplicate or malformed budget checks")
        by_name[row["what"]] = row["pass"]
    if set(by_name) != CHECK_NAMES:
        raise ProviderError("size report omitted an executable budget check")
    passed_count = sum(by_name.values()) + int(final_bytes <= MAX_FINAL_BYTES)
    verdict = passed_count == 3 and report["ok"]
    return report, verdict, passed_count


def qualify(ctx, run=None) -> int:
    run = subprocess.run if run is None else run
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("D2 result must be a new regular file")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("D2 evidence must be outside the source checkout")
    check_binding(ctx, run)
    verify_candidate(ctx, run)
    check_binding(ctx, run)
    args = command("size-report", ctx.artifact)
    print(json.dumps({"assertion": "size-budget-only", "argv": args}), flush=True)
    child = run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
        env=cargo_env(),
    )
    sys.stdout.write(child.stdout)
    sys.stderr.write(child.stderr)
    report, passed, passed_count = parse_size_report(child.stdout)
    check_binding(ctx, run)
    if child.returncode not in (0, 1) or (child.returncode == 0) != report.get("ok"):
        raise ProviderError("size-report exit status disagrees with its report")
    passed = passed and child.returncode == 0
    counts = {
        "collected": 3,
        "executed": 3,
        "passed": passed_count,
        "failed": 3 - passed_count,
        "skipped": 0,
        "excluded": 0,
    }
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "D2",
        "status": "PASS" if passed else "FAIL",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": counts,
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0 if passed else 1


def main() -> int:
    try:
        return qualify(context("D2"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"D2 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
