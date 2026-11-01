#!/usr/bin/env python3
"""REL-07 S2: native discovery determinism across workers and recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from common import (
    ROOT,
    Context,
    ProviderError,
    context,
    rust_counts,
    sha256,
    verify_candidate,
)


TEST = (
    "discover::r#loop::tests::"
    "real_native_workers_preserve_projection_across_capacity_pause_and_sigkill"
)
BUILD_COMMAND = [
    "cargo",
    "build",
    "--locked",
    "-p",
    "jope-service",
    "--all-targets",
]
EXPECTED_COUNTS = {
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
        "--nocapture",
        "--test-threads=1",
    ]


def check_binding(ctx: Context, run) -> None:
    """Require the test source and candidate bytes to stay at their pinned IDs."""
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError(
                "S2 requires the clean, candidate-pinned source checkout"
            )
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during S2")


def temporary_root() -> Path:
    target = ROOT / "target"
    if target.is_symlink():
        raise ProviderError("S2 target directory must not be a symlink")
    target.mkdir(exist_ok=True)
    root = target / "s2-provider-tmp"
    if root.is_symlink():
        raise ProviderError("S2 temporary directory must not be a symlink")
    root.mkdir(exist_ok=True)
    if not root.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("S2 temporary directory must remain inside the checkout")
    return root.resolve()


def qualify(ctx: Context, run=None) -> int:
    runner = subprocess.run if run is None else run
    if not sys.platform.startswith(("linux", "darwin")):
        raise ProviderError("S2 SIGKILL recovery assertion requires Linux or macOS")
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("S2 result must be a new regular file")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("S2 evidence must be outside the source checkout")

    child_env = os.environ.copy()
    child_env["TMPDIR"] = str(temporary_root())

    def invoke(args, **kwargs):
        kwargs.setdefault("env", child_env)
        return runner(args, **kwargs)

    check_binding(ctx, invoke)
    verify_candidate(ctx, run=invoke)
    check_binding(ctx, invoke)

    print(
        json.dumps({"phase": "native-worker-build", "argv": BUILD_COMMAND}), flush=True
    )
    build = invoke(
        BUILD_COMMAND,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    sys.stdout.write(build.stdout)
    sys.stderr.write(build.stderr)
    check_binding(ctx, invoke)
    if build.returncode != 0:
        return build.returncode

    args = command()
    print(json.dumps({"test": TEST, "argv": args}), flush=True)
    result = invoke(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    check_binding(ctx, invoke)
    if result.returncode != 0:
        return result.returncode

    counts = rust_counts(result.stdout)
    if counts != EXPECTED_COUNTS or f"test {TEST} ... ok" not in result.stdout:
        raise ProviderError("S2 requires exactly one passing native determinism case")

    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "S2",
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": counts,
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return qualify(context("S2"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"S2 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
