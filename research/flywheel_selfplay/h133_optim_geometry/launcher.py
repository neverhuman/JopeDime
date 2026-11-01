#!/usr/bin/env python3
"""Run one H133 A6 worker with a parent-owned pause/resume monitor."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
WALL_SECONDS = 4 * 60 * 60
POLL_SECONDS = 0.25
LOAD_PAUSE = 86.7
MEM_PAUSE = 30 * 1024**3
LOAD_RESUME = 80.0
MEM_RESUME = 34 * 1024**3
RESUME_STABLE_SECONDS = 60.0
RSS_HARD = 6 * 1024**3
SCRATCH_HARD = 2 * 1024**3
DISK_HARD = 93.8
CPU_HARD = 85.0
THREADS_HARD = 4


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def cpu_snapshot() -> tuple[int, int]:
    vals = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
    return sum(vals), vals[3] + vals[4]


def host_cpu_pct(before: tuple[int, int]) -> tuple[float, tuple[int, int]]:
    after = cpu_snapshot()
    total = after[0] - before[0]
    idle = after[1] - before[1]
    return 100.0 * (total - idle) / max(total, 1), after


def scratch_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            continue
    return total


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def snapshot(proc: psutil.Process, run_root: Path, cpu_before: tuple[int, int]) -> dict:
    fs = os.statvfs("/")
    used = fs.f_blocks - fs.f_bfree
    available_blocks = fs.f_bavail
    used_pct = 100.0 * used / max(used + available_blocks, 1)
    try:
        descendants = [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    rss = threads = 0
    for item in descendants:
        try:
            rss += item.memory_info().rss
            threads += item.num_threads()
        except psutil.NoSuchProcess:
            pass
    cpu, next_cpu = host_cpu_pct(cpu_before)
    return {
        "utc": utc(),
        "host": os.uname().nodename,
        "load1": os.getloadavg()[0],
        "mem_available_bytes": mem_available(),
        "host_cpu_pct": cpu,
        "rss_bytes": rss,
        "process_threads": threads,
        "disk_used_pct": used_pct,
        "scratch_bytes": scratch_bytes(run_root),
        "_cpu_after": next_cpu,
    }


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def stop_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def hard_failure(s: dict) -> str | None:
    checks = [
        (s["rss_bytes"] > RSS_HARD, "RSS>6GiB"),
        (s["process_threads"] > THREADS_HARD, "threads>4"),
        (s["disk_used_pct"] >= DISK_HARD, "disk>=93.8%"),
        (s["scratch_bytes"] >= SCRATCH_HARD, "scratch>=2GiB"),
        (s["host_cpu_pct"] > CPU_HARD, "host CPU>85%"),
    ]
    return next((label for bad, label in checks if bad), None)


def main() -> int:
    attempt_started = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--r882", type=Path, required=True)
    ap.add_argument("--merge-receipt", type=Path, required=True)
    ap.add_argument("--run-root", type=Path)
    args = ap.parse_args()
    source = args.source_root.resolve()
    input_root = args.input_root.resolve()
    r882 = args.r882.resolve()
    run_root = (args.run_root or (source / "attempt_A6_013")).resolve()
    if run_root.exists():
        raise RuntimeError(f"run root exists; refusing implicit resume/retry: {run_root}")
    if os.geteuid() == 0:
        raise RuntimeError("H133 requires an unprivileged CPU process")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", None):
        raise RuntimeError("CUDA must be hidden")
    if sha(r882) != "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64":
        raise RuntimeError("r882 SHA mismatch")
    manifest = json.loads((HERE / "manifest.json").read_text())
    for name, digest in manifest.get("a6_executor_sha256", {}).items():
        if sha(HERE / name) != digest:
            raise RuntimeError(f"A6 source SHA mismatch: {name}")
    merge_receipt = json.loads(args.merge_receipt.read_text())
    if merge_receipt.get("schema") != "jope.h133.a6.merge_receipt/v1":
        raise RuntimeError("A6 merge receipt schema missing")
    if (
        merge_receipt.get("pr_state") != "MERGED"
        or not merge_receipt.get("merge_commit_sha")
        or not merge_receipt.get("main_readback_sha")
        or not merge_receipt.get("merged_at")
        or merge_receipt.get("main_contains_merge_commit") is not True
    ):
        raise RuntimeError("merged A6 PR readback required")
    for name in ("runner.py", "launcher.py", "manifest.json"):
        expected = merge_receipt.get("source_file_sha256", {}).get(name)
        if not expected or sha(HERE / name) != expected:
            raise RuntimeError(f"merged A6 receipt source pin mismatch: {name}")
    runner = HERE / "runner.py"
    if not runner.is_file():
        raise RuntimeError("runner.py missing")
    helper_root = source / "runtime" / "research" / "flywheel_selfplay"
    calibration = manifest["calibration"]
    helper_pins = dict(calibration["helper_module_sha256"])
    helper_pins["h132_ewc_bitfit/run_h132_cpu.py"] = calibration["h132_runner_sha256"]
    for rel, digest in helper_pins.items():
        path = helper_root / rel
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"pinned helper source mismatch: {rel}")
    closure_bytes = tree_bytes(input_root) + r882.stat().st_size
    closure_bytes += sum((source / rel).stat().st_size for rel in [
        "train_only_export.json", "fold_manifest_133100.json"
    ])
    closure_bytes += tree_bytes(source / "runtime")
    closure_bytes += runner.stat().st_size + Path(__file__).stat().st_size
    if os.uname().nodename.startswith("xbabe1") and closure_bytes >= 2 * 1024**3:
        raise RuntimeError(f"xbabe1 input closure is not below 2 GiB: {closure_bytes}")
    if time.monotonic() - attempt_started >= WALL_SECONDS:
        raise TimeoutError("four-hour wall budget expired during preflight")
    run_root.mkdir(parents=True)
    log_path = run_root / "worker.log"
    events_path = run_root / "pause_events.jsonl"
    receipt_path = run_root / "launcher_receipt.json"
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "H133_MANAGED_PAUSE": "1",
        "H133_LAUNCHER_PID": str(os.getpid()),
        "H133_SOURCE_ROOT": str(source),
        "H133_RUNTIME_ROOT": str(source / "runtime"),
        "H133_RUN_ROOT": str(run_root),
        "H133_INPUT_ROOT": str(input_root),
        "H133_R882_PATH": str(r882),
        "JE_ARROW_MALLOC_CONF": "background_thread:false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "NUMEXPR_MAX_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(run_root / "bytecode"),
    })
    started = attempt_started
    record = {
        "schema": "jope.h133.a6_launcher/v1",
        "status": "RUNNING",
        "attempt": 13,
        "started_utc": utc(),
        "host": os.uname().nodename,
        "wall_budget_seconds": WALL_SECONDS,
        "source_root": str(source),
        "input_root": str(input_root),
        "r882_sha256": sha(r882),
        "merge_receipt": merge_receipt,
        "input_closure_bytes": closure_bytes,
        "input_closure_below_2GiB": closure_bytes < 2 * 1024**3,
        "runner_sha256": sha(runner),
        "launcher_sha256": sha(Path(__file__)),
        "pause_policy": {
            "pause_if_load1_gt": LOAD_PAUSE,
            "pause_if_mem_available_lt_bytes": MEM_PAUSE,
            "resume_if_load1_lt": LOAD_RESUME,
            "resume_if_mem_available_gte_bytes": MEM_RESUME,
            "resume_stable_seconds": RESUME_STABLE_SECONDS,
        },
        "hard_guard_policy": {
            "rss_bytes_max": RSS_HARD,
            "threads_max": THREADS_HARD,
            "disk_percent_strictly_below": DISK_HARD,
            "scratch_bytes_strictly_below": SCRATCH_HARD,
            "host_cpu_percent_max": CPU_HARD,
        },
        "pause_events": [],
    }
    proc = None
    try:
        with log_path.open("wb") as log:
            cmd = ["nice", "-n", "19", sys.executable, str(runner)]
            proc = subprocess.Popen(
                cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            child = psutil.Process(proc.pid)
            record["worker_pid"] = proc.pid
            if child.nice() != 19:
                raise RuntimeError(f"worker nice is {child.nice()}, expected 19")
            pause_started = None
            pause_started_monotonic = None
            stable_since = None
            pause_reason = None
            cpu_before = cpu_snapshot()
            with events_path.open("w") as out:
                out.flush()
            while proc.poll() is None:
                if time.monotonic() - started >= WALL_SECONDS:
                    raise TimeoutError("four-hour wall budget exceeded")
                time.sleep(POLL_SECONDS)
                s = snapshot(child, run_root, cpu_before)
                cpu_before = s.pop("_cpu_after")
                append_jsonl(run_root / "resources.jsonl", s)
                reason = hard_failure(s)
                if reason:
                    raise RuntimeError("hard resource guard: " + reason)
                bad_load = s["load1"] > LOAD_PAUSE
                bad_mem = s["mem_available_bytes"] < MEM_PAUSE
                if pause_started is None and (bad_load or bad_mem):
                    pause_started = utc()
                    pause_started_monotonic = time.monotonic()
                    pause_reason = {
                        "load1": s["load1"],
                        "mem_available_bytes": s["mem_available_bytes"],
                    }
                    stable_since = None
                    record["pause_events"].append({
                        "event": "SIGSTOP",
                        "utc": pause_started,
                        "reason_snapshot": pause_reason,
                    })
                    append_jsonl(events_path, record["pause_events"][-1])
                    stop_group(proc.pid, signal.SIGSTOP)
                elif pause_started is not None:
                    recovered = s["load1"] < LOAD_RESUME and s["mem_available_bytes"] >= MEM_RESUME
                    if recovered:
                        stable_since = stable_since or time.monotonic()
                        if time.monotonic() - stable_since >= RESUME_STABLE_SECONDS:
                            resumed = utc()
                            stop_group(proc.pid, signal.SIGCONT)
                            event = {
                                "event": "SIGCONT",
                                "utc": resumed,
                                "pause_started_utc": pause_started,
                                "pause_seconds": round(time.monotonic() - pause_started_monotonic, 3),
                                "recovery_stable_seconds": RESUME_STABLE_SECONDS,
                                "recovery_snapshot": {
                                    "load1": s["load1"],
                                    "mem_available_bytes": s["mem_available_bytes"],
                                },
                            }
                            record["pause_events"].append(event)
                            append_jsonl(events_path, event)
                            pause_started = pause_started_monotonic = stable_since = pause_reason = None
                    else:
                        stable_since = None
                record["last_snapshot"] = s
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                receipt_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
            code = proc.wait()
        metrics = run_root / "metrics.json"
        record["worker_exit_status"] = code
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        record["metrics_sha256"] = sha(metrics) if metrics.exists() else None
        record["status"] = "COMPLETE" if code == 0 and metrics.exists() else "INCOMPLETE"
        record["finished_utc"] = utc()
        receipt_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        return 0 if record["status"] == "COMPLETE" else (code or 1)
    except BaseException as exc:
        record["status"] = "INCOMPLETE"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["finished_utc"] = utc()
        if proc is not None and proc.poll() is None:
            stop_group(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stop_group(proc.pid, signal.SIGKILL)
                proc.wait()
        receipt_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
