#!/usr/bin/env python3
"""REL-07 R4: run the maintained terminal/MCP reconnect probe on release bytes."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from common import ROOT, Context, ProviderError, context, sha256, verify_candidate


RUNTIME_BASE = Path.home() / ".local/state/jope-r4"
COUNTS = {
    "collected": 3,
    "executed": 3,
    "passed": 3,
    "failed": 0,
    "skipped": 0,
    "excluded": 0,
}


def binding(ctx: Context, run) -> None:
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        result = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ProviderError("R4 requires the clean candidate-pinned checkout")
    if ctx.artifact.is_symlink() or not ctx.artifact.is_file():
        raise ProviderError("R4 candidate must remain a regular non-symlink file")
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("R4 candidate bytes changed")


def read_json(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ProviderError(f"R4 missing regular evidence file: {path.name}")
    try:
        return json.loads(path.read_text())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProviderError(f"R4 malformed evidence: {path.name}") from error


def execute(args: list[str], env: dict[str, str], timeout: float = 300):
    def interrupted(signum, _frame):
        raise InterruptedError(f"R4 probe interrupted by signal {signum}")

    with subprocess.Popen(
        args,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as child:
        previous = {
            sig: signal.signal(sig, interrupted)
            for sig in (signal.SIGTERM, signal.SIGINT)
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
            # This group contains only this invocation's probe and its children.
            stdout, stderr = stop()
            code = 124
        except BaseException:
            stop()
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return subprocess.CompletedProcess(args, code, stdout, stderr)


def assertions(ctx: Context, evidence: Path) -> None:
    report = read_json(evidence / "report.json")
    if (
        not isinstance(report, dict)
        or report.get("source") != ctx.source_sha
        or report.get("binary_sha256") != ctx.artifact_sha256
        or not isinstance(report.get("version"), dict)
        or report.get("version", {}).get("source_sha40") != ctx.source_sha
        or "error" in report
        or report.get("cleanup_errors") != []
    ):
        raise ProviderError("R4 report identity, execution or cleanup failed")
    for name in (
        "projection_oracle_controls",
        "r4_qualified",
        "mcp_sequences_contiguous",
        "mcp_equals_cli_events",
        "terminal_event_projection_observed",
        "terminal_champion_render_matches_mcp_snapshot",
    ):
        if report.get(name) is not True:
            raise ProviderError(f"R4 missing actual assertion: {name}")
    if any(
        not isinstance(report.get(key), dict)
        for key in ["terminal_first", "terminal_reattach", "r4_final"]
    ):
        raise ProviderError("R4 terminal observations are missing")
    if (
        report.get("state_at_close") != "running"
        or report.get("terminal_first", {}).get("exit") != 0
        or report.get("terminal_reattach", {}).get("exit") != 0
        or report.get("r4_final", {}).get("state") != "completed"
    ):
        raise ProviderError("R4 terminal closure or completed reattach failed")

    events = read_json(evidence / "live-mcp-events.json")
    cli = read_json(evidence / "live-cli-events.json")
    final = report["r4_final"]
    if not isinstance(events, list) or not events or events != cli:
        raise ProviderError("R4 MCP and durable CLI events differ or are empty")
    if (
        [int(event["sequence"]) for event in events] != list(range(1, len(events) + 1))
        or int(final["event_watermark"]) != len(events)
        or any(event["run_id"] != final["run_id"] for event in events)
    ):
        raise ProviderError("R4 cursor replay has a gap, duplicate or wrong run")
    streams = read_json(evidence / "terminal-sse-streams.json")
    if not isinstance(streams, list) or len(streams) < 2:
        raise ProviderError("R4 requires both original and reattached terminal streams")
    if not any(
        isinstance(stream, dict) and stream.get("events") == events
        for stream in streams[1:]
    ):
        raise ProviderError("R4 reattached terminal differs from MCP events")
    screen = evidence / "terminal-after-reattach.pty"
    if screen.is_symlink() or not screen.is_file():
        raise ProviderError("R4 missing regular terminal screen evidence")
    best = final["best"]
    score = best["cv_value"]
    if (
        not isinstance(best["algorithm_id"], str)
        or not best["algorithm_id"]
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
    ):
        raise ProviderError("R4 final champion is missing")
    text = screen.read_text(errors="replace")
    if best["algorithm_id"] not in text or f"{score:.3f}" not in text:
        raise ProviderError("R4 terminal champion differs from the final snapshot")


def qualify(ctx: Context, run=None) -> int:
    run = subprocess.run if run is None else run
    if ctx.gate_id != "R4":
        raise ProviderError("R4 requires gate R4")
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("R4 result must be new")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("R4 evidence must be outside the source checkout")
    binding(ctx, run)
    verify_candidate(ctx, run=run)
    binding(ctx, run)
    evidence = ctx.result.with_name(ctx.result.name + ".r4")
    evidence.mkdir(mode=0o700, parents=True)
    source = (ROOT / "crates/jope-appliance/tests/system/live_driver.rs").read_text()
    for name, filename in (
        ("DRIVER", "live_probe.py"),
        ("OBSERVER", "observe_http.py"),
    ):
        match = re.search(rf'const {name}: &str = r####"(.*?)"####;', source, re.S)
        if match is None:
            raise ProviderError(f"R4 maintained {name} probe unavailable")
        (evidence / filename).write_text(match[1])
    RUNTIME_BASE.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix="r4-", dir=RUNTIME_BASE))
    (evidence / "runtime-location.txt").write_text(str(runtime) + "\n")
    args = [
        sys.executable,
        "-B",
        str(evidence / "live_probe.py"),
        "--binary",
        str(ctx.artifact),
        "--source",
        ctx.source_sha,
        "--source-root",
        str(ROOT),
        "--out",
        str(evidence),
        "--runtime-root",
        str(runtime),
        "--r4-only",
        "--capture-sse",
        "--train-rows",
        "96",
        "--wall-seconds",
        "300",
    ]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
    )
    completed = execute(args, env)
    (evidence / "stdout.log").write_text(completed.stdout)
    (evidence / "stderr.log").write_text(completed.stderr)
    binding(ctx, run)
    if completed.returncode != 0:
        return completed.returncode
    assertions(ctx, evidence)
    binding(ctx, run)
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "R4",
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": COUNTS,
    }
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return qualify(context("R4"))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"R4 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
