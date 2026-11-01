#!/usr/bin/env python3
"""REL-07 R2: native worker faults, engine RSS, and successful search recovery."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from common import ROOT, Context, ProviderError, context, sha256, verify_candidate


TEST = "discover::r#loop::tests::rel07_r2_worker_faults_preserve_engine_memory_and_complete_search"
TRAIN_SHA = "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931"
COUNTS = dict(collected=1, executed=1, passed=1, failed=0, skipped=0, excluded=0)
ENV = dict(
    CARGO_BUILD_JOBS="4",
    CARGO_INCREMENTAL="0",
    CARGO_PROFILE_DEV_DEBUG="line-tables-only",
    CUDA_VISIBLE_DEVICES="",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    RAYON_NUM_THREADS="1",
)


def regular(path: Path) -> None:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ProviderError("R2 requires an absolute regular non-symlink file")
    if any(parent.is_symlink() for parent in path.parents):
        raise ProviderError("R2 file parent must not be a symlink")


def result_path(ctx: Context) -> None:
    if (
        not ctx.result.is_absolute()
        or ctx.result.exists()
        or ctx.result.is_symlink()
        or any(parent.is_symlink() for parent in ctx.result.parents)
        or ctx.result.resolve().is_relative_to(ROOT.resolve())
    ):
        raise ProviderError(
            "R2 result must be new and outside source, without symlinks"
        )


def binding(ctx: Context, run) -> None:
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        completed = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if completed.returncode or completed.stdout.strip() != expected:
            raise ProviderError("R2 requires the clean candidate-pinned checkout")
    regular(ctx.artifact)
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("R2 candidate bytes changed")
    result_path(ctx)


def execute(args, *, cwd=ROOT, env=None, timeout=1800, **_kwargs):
    """Bound this invocation and terminate only its own process group on failure."""

    def interrupted(signum, _frame):
        raise InterruptedError(f"R2 interrupted by signal {signum}")

    with subprocess.Popen(
        ["nice", "-n", "10", *args],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as child:
        previous = {
            sig: signal.signal(sig, interrupted)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }

        def stop():
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                return child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                return child.communicate()

        try:
            stdout, stderr = child.communicate(timeout=timeout)
            code = child.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr = stop()
            code = 124
        except BaseException:
            stop()
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return subprocess.CompletedProcess(args, code, stdout, stderr)


def library(output: str) -> Path:
    paths = []
    for line in output.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(row, dict)
            and row.get("reason") == "compiler-artifact"
            and row.get("target", {}).get("name") == "jope_service"
            and "lib" in row.get("target", {}).get("kind", [])
        ):
            paths.extend(
                Path(name)
                for name in row.get("filenames", [])
                if name.endswith(".rlib")
            )
    if len(paths) != 1:
        raise ProviderError("R2 requires exactly one current service library artifact")
    regular(paths[0])
    return paths[0]


def snapshot_dependencies(
    output: str, destination: Path
) -> tuple[Path, dict[str, str]]:
    service = library(output)
    destination.mkdir(mode=0o700)
    digests = {}
    for line in output.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("reason") != "compiler-artifact":
            continue
        for name in row.get("filenames", []):
            source = Path(name)
            if source.suffix not in (".rlib", ".rmeta", ".so"):
                continue
            regular(source)
            digest = sha256(source)
            target = destination / source.name
            if target.exists():
                if digests.get(source.name) != digest:
                    raise ProviderError("R2 ambiguous dependency artifact name")
                continue
            with source.open("rb") as stream, target.open("xb") as copy:
                shutil.copyfileobj(stream, copy)
            if sha256(target) != digest or sha256(source) != digest:
                raise ProviderError("R2 dependency changed during snapshot")
            digests[source.name] = digest
    return destination / service.name, digests


def measurements(output: str, source: str) -> list[dict]:
    summaries = re.findall(
        r"(?m)^test result: (\w+)\. (\d+) passed; (\d+) failed; "
        r"(\d+) ignored; (\d+) measured; (\d+) filtered out\s*$",
        output,
    )
    if len(summaries) != 1 or summaries[0][:5] != ("ok", "1", "0", "0", "0"):
        raise ProviderError("R2 requires exactly one passing non-skipped native test")
    if f"test {TEST} ... " not in output or "running 1 test\n" not in output:
        raise ProviderError("R2 exact native selector did not execute")
    try:
        rows = [
            json.loads(text) for text in re.findall(r"REL07_R2 (\{[^\n]*\})", output)
        ]
    except json.JSONDecodeError as error:
        raise ProviderError("R2 malformed native measurements") from error
    if len(rows) != 2 or {row.get("fault") for row in rows} != {"timeout", "memory"}:
        raise ProviderError("R2 needs one timeout and one memory fault measurement")
    for row in rows:
        if (
            row.get("schema") != "jope.rel07.worker-isolation/v1"
            or row.get("source_sha") != source
            or row.get("train_sha256") != TRAIN_SHA
            or row.get("failure")
            != {"timeout": "Timeout", "memory": "MemoryLimit"}[row["fault"]]
            or row.get("completed") is not True
            or row.get("champion_present") is not True
        ):
            raise ProviderError("R2 fault identity or healthy search completion failed")
        for key in (
            "wall_limit_ms",
            "elapsed_ms",
            "baseline_rss_bytes",
            "peak_rss_bytes",
            "rss_samples",
            "workers_started",
            "workers_remaining",
        ):
            if type(row.get(key)) is not int or row[key] < (
                0 if key == "workers_remaining" else 1
            ):
                raise ProviderError(f"R2 missing valid native measurement: {key}")
        if (
            row["wall_limit_ms"] != 1000
            or row["elapsed_ms"] > row["wall_limit_ms"] + 2000
            or row["peak_rss_bytes"] < row["baseline_rss_bytes"]
            or row["peak_rss_bytes"] * 10 > row["baseline_rss_bytes"] * 11
            or row["workers_started"] < 2
            or row["workers_remaining"] != 0
        ):
            raise ProviderError(
                "R2 worker bound, engine RSS, replacement or cleanup failed"
            )
    return rows


def qualify(ctx: Context, run=None) -> int:
    if ctx.gate_id != "R2" or not sys.platform.startswith("linux"):
        raise ProviderError("R2 requires gate R2 and native Linux RSS isolation")
    run = execute if run is None else run
    result_path(ctx)
    env = dict(os.environ, **ENV)
    evidence = ctx.result.with_name(ctx.result.name + ".r2")
    evidence.mkdir(parents=True, mode=0o700)
    records = []

    def recorded(args, **kwargs):
        result = run(args, env=env, **kwargs)
        number = len(records)
        (evidence / f"{number:02}.stdout").write_text(result.stdout)
        (evidence / f"{number:02}.stderr").write_text(result.stderr)
        records.append(dict(command=args, exit=result.returncode))
        (evidence / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        return result

    binding(ctx, recorded)
    manifest_target = None

    def verified(args, **kwargs):
        nonlocal manifest_target
        result = recorded(args, **kwargs)
        if "inspect" in args and result.returncode == 0:
            manifest_target = (
                json.loads(result.stdout).get("manifest", {}).get("target")
            )
        return result

    verify_candidate(ctx, run=verified)
    host = recorded(
        ["rustc", "-vV"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    match = re.search(r"(?m)^host: (\S+)$", host.stdout)
    if (
        host.returncode
        or match is None
        or manifest_target != match[1]
        or "linux" not in match[1]
    ):
        raise ProviderError("R2 candidate must match the native Linux compiler target")
    binding(ctx, recorded)
    build = recorded(
        [
            "cargo",
            "build",
            "--locked",
            "-p",
            "jope-service",
            "--lib",
            "--message-format=json",
        ],
        cwd=ROOT,
    )
    if build.returncode:
        raise ProviderError("R2 current service library build failed")
    dependencies = evidence / "dependencies"
    snapshot, digests = snapshot_dependencies(build.stdout, dependencies)
    digest = digests[snapshot.name]
    (evidence / "dependencies.json").write_text(
        json.dumps(digests, sort_keys=True) + "\n"
    )
    runtime = evidence / "runtime"
    runtime.mkdir(mode=0o700)
    env.update(
        JOPE_BUILD_SOURCE_SHA=ctx.source_sha,
        JOPE_DSC_SERVICE_RLIB=str(snapshot),
        JOPE_DSC_DEPENDENCY_DIR=str(dependencies),
        JOPE_DSC_RECEIPT_DIR=str(evidence),
        TMPDIR=str(runtime),
    )
    binding(ctx, recorded)
    test = recorded(
        [
            "cargo",
            "test",
            "--locked",
            "-p",
            "jope-service",
            "--lib",
            TEST,
            "--",
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ],
        cwd=ROOT,
    )
    if test.returncode:
        raise ProviderError("R2 native worker isolation test failed")
    rows = measurements(test.stdout, ctx.source_sha)
    binding(ctx, recorded)
    for name, expected in digests.items():
        regular(dependencies / name)
        if sha256(dependencies / name) != expected:
            raise ProviderError("R2 tested dependency snapshot changed")
    payload = dict(
        schema="jope.rel07.gate-execution/v1",
        id="R2",
        status="PASS",
        source_sha=ctx.source_sha,
        artifact_sha256=ctx.artifact_sha256,
        counts=COUNTS,
        measurements=rows,
        service_rlib_sha256=digest,
        evidence=str(evidence),
    )
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return qualify(context("R2"))
    except (OSError, ValueError, KeyError, TypeError, InterruptedError) as error:
        print(f"R2 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
