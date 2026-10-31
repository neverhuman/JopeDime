#!/usr/bin/env python3
"""Serialize H55 CUDA cells under qualification.lock; never use CPU fallback."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "research/flywheel_selfplay/h55/runner.py"
LOCK = Path.home() / "jope_plan/qualification.lock"
RUN_ROOT = Path(os.environ.get("H55_RUN_ROOT", str(Path.home() / "jope-data/h55/h55-wave")))
MIN_GPU_FREE_MIB = 6144
MAX_LOCK_WAIT_SECONDS = 3 * 60 * 60
RSS_LIMIT_KIB = 6 * 1024 * 1024
MIN_MEM_AVAILABLE_KIB = 20 * 1024 * 1024
CELL_ORDER = [("swa8", seed) for seed in (12, 13, 14)] + [("ema25", seed) for seed in (12, 13, 14)]
TIMING_CELL = ("ema25", 12)


def terminal_arm_by_preregistered_order(admission: dict) -> str | None:
    """Return the first admitted arm; red-team policy permits one read per seed."""
    return next((arm for arm in ("swa8", "ema25")
                 if admission.get("arms", {}).get(arm, {}).get("admitted") is True), None)


class LockWaitTimeout(TimeoutError):
    def __init__(self, wait_seconds: float, cumulative_wait_seconds: float):
        self.wait_seconds = wait_seconds
        self.cumulative_wait_seconds = cumulative_wait_seconds
        super().__init__(f"qualification.lock wait exceeded 3h; cumulative={cumulative_wait_seconds:.1f}s")


class RunnerExitError(RuntimeError):
    def __init__(self, key: str, code: int, log_path: Path, wait_seconds: float,
                 wall_seconds: float, preflight: dict):
        self.wait_seconds = wait_seconds
        self.wall_seconds = wall_seconds
        self.preflight = preflight
        self.log_path = log_path
        super().__init__(f"{key} runner exited {code}; log={log_path}")


class CellPreflightError(RuntimeError):
    def __init__(self, message: str, wait_seconds: float):
        self.wait_seconds = wait_seconds
        super().__init__(message)


def remaining_lock_wait_seconds(cumulative_wait_seconds: float) -> float:
    return max(0.0, MAX_LOCK_WAIT_SECONDS - cumulative_wait_seconds)


def disk_guard_error(usage=None) -> str | None:
    usage = usage or shutil.disk_usage("/")
    used_percent = 100.0 * (usage.total - usage.free) / usage.total
    if usage.free < 1024 * 1024 * 1024:
        return f"less than 1 GiB free on /: {usage.free} bytes"
    if used_percent >= 95.0:
        return f"filesystem / is at or above 95% full: {used_percent:.2f}%"
    return None


def is_disk_guard_failure(exc: Exception) -> bool:
    message = str(exc)
    return "filesystem / is at or above 95% full" in message or "less than 1 GiB free on /" in message


def exception_wait_seconds(exc: Exception) -> float:
    return float(getattr(exc, "wait_seconds", 0.0))


def note(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} — {text}\n"
    with path.open("a") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    print(line.rstrip(), flush=True)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".new")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def gpu_free_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    )
    return int(out.splitlines()[0].strip())


def memory_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is unavailable")


def swap_kib() -> tuple[int, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("SwapTotal:", "SwapFree:")):
            key, value, _ = line.split()
            values[key.rstrip(":")] = int(value)
    return values.get("SwapTotal", 0), values.get("SwapFree", 0)


def cpu_percent(sample_seconds: float = 0.5) -> float:
    def read():
        fields = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
        idle = fields[3] + fields[4]
        return sum(fields), idle
    total_a, idle_a = read()
    time.sleep(sample_seconds)
    total_b, idle_b = read()
    total = total_b - total_a
    return 100.0 * (total - (idle_b - idle_a)) / total if total else 0.0


def process_rss_kib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None


def safe_preflight() -> dict:
    cpu = cpu_percent()
    available = memory_available_kib()
    total_swap, free_swap = swap_kib()
    usage = shutil.disk_usage("/")
    disk_percent = 100.0 * (usage.total - usage.free) / usage.total
    if available < MIN_MEM_AVAILABLE_KIB:
        raise RuntimeError(f"MemAvailable below 20 GiB: {available} KiB")
    if cpu > 85.0:
        raise RuntimeError(f"host CPU utilization above 85%: {cpu:.1f}%")
    if usage.free < 1024 * 1024 * 1024:
        raise RuntimeError(f"less than 1 GiB free on /: {usage.free} bytes")
    if disk_percent >= 95.0:
        raise RuntimeError(f"filesystem / is at or above 95% full: {disk_percent:.2f}%")
    return {
        "host_cpu_percent_before": cpu,
        "mem_available_kib_before": available,
        "swap_total_kib": total_swap,
        "swap_free_kib": free_swap,
        "disk_free_bytes": usage.free,
        "disk_used_percent": disk_percent,
    }


def acquire_lock(status_path: Path, wait_total: float) -> tuple[object, float, int]:
    handle = LOCK.open("a+")
    started = time.monotonic()
    next_note = started
    next_full_preflight = started
    resource_ready = False
    disk_blocked = False
    preflight_reason = "initial resource preflight pending"
    while True:
        now = time.monotonic()
        disk_error = disk_guard_error()
        if disk_error:
            if not disk_blocked:
                note(status_path, f"WAIT disk guard before lock acquisition: {disk_error}")
            disk_blocked = True
            resource_ready = False
            preflight_reason = disk_error
            if now >= next_full_preflight:
                try:
                    safe_preflight()
                except RuntimeError as exc:
                    preflight_reason = str(exc)
                    note(status_path, f"WAIT resource preflight before lock acquisition: {preflight_reason}")
                next_full_preflight = now + 300
        elif disk_blocked:
            try:
                safe_preflight()
                resource_ready = True
                disk_blocked = False
                preflight_reason = ""
            except RuntimeError as exc:
                resource_ready = False
                disk_blocked = is_disk_guard_failure(exc)
                preflight_reason = str(exc)
                next_full_preflight = now + 300
        elif now >= next_full_preflight:
            try:
                safe_preflight()
                resource_ready = True
                preflight_reason = ""
            except RuntimeError as exc:
                resource_ready = False
                preflight_reason = str(exc)
                disk_blocked = is_disk_guard_failure(exc)
                note(status_path, f"WAIT resource preflight before lock acquisition: {preflight_reason}")
            next_full_preflight = now + 300
        if resource_ready:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                free = gpu_free_mib()
                if free >= MIN_GPU_FREE_MIB:
                    return handle, time.monotonic() - started, free
                fcntl.flock(handle, fcntl.LOCK_UN)
            except BlockingIOError:
                pass
        waited = time.monotonic() - started
        if waited >= remaining_lock_wait_seconds(wait_total):
            handle.close()
            raise LockWaitTimeout(waited, wait_total + waited)
        if time.monotonic() >= next_note:
            detail = f"; resource_block={preflight_reason}" if preflight_reason else ""
            note(status_path, f"WAIT qualification.lock / GPU >=6GiB; cumulative_wait={wait_total + waited:.0f}s{detail}")
            next_note = time.monotonic() + 30
        time.sleep(5)


def run_child(cmd: list[str], log_path: Path, timeout: float, status_path: Path) -> tuple[int, float, int, float]:
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CARGO_INCREMENTAL": "0",
        "CARGO_PROFILE_DEV_DEBUG": "line-tables-only",
    })
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    max_rss = 0
    max_cpu = 0.0
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.nice(19),
        )
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                rss = process_rss_kib(proc.pid)
                if rss is not None:
                    max_rss = max(max_rss, rss)
                if rss is not None and rss >= RSS_LIMIT_KIB:
                    proc.terminate()
                    raise RuntimeError(f"child RSS reached 6 GiB: {rss} KiB")
                if elapsed >= timeout:
                    proc.terminate()
                    raise TimeoutError(f"fixed resource watchdog expired after {timeout:.1f}s")
                cpu_now = cpu_percent(0.2)
                max_cpu = max(max_cpu, cpu_now)
                if cpu_now > 85.0:
                    proc.terminate()
                    raise RuntimeError(f"host CPU utilization exceeded 85%: {cpu_now:.1f}%")
                time.sleep(1.8)
            code = proc.wait()
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise
    return code, time.monotonic() - started, max_rss, max_cpu


def run_locked_cell(
    *, key: str, cmd: list[str], output_dir: Path, timeout: float,
    status_path: Path, cumulative_wait: float,
) -> tuple[dict, float]:
    preflight = safe_preflight()
    handle, wait_seconds, free_mib = acquire_lock(status_path, cumulative_wait)
    try:
        try:
            post_lock = safe_preflight()
        except RuntimeError as exc:
            raise CellPreflightError(str(exc), wait_seconds) from exc
        preflight.update({f"post_lock_{key}": value for key, value in post_lock.items()})
        if free_mib < MIN_GPU_FREE_MIB:
            raise CellPreflightError("GPU memory guard failed after lock acquisition", wait_seconds)
        preflight["gpu_free_mib_after_lock"] = free_mib
        preflight["gpu_min_free_mib"] = MIN_GPU_FREE_MIB
        preflight["qualification_lock_wait_seconds"] = wait_seconds
        note(status_path, f"START {key}; CUDA free={free_mib} MiB; lock_wait={wait_seconds:.1f}s; preflight={json.dumps(preflight, sort_keys=True)}")
        log_path = RUN_ROOT / "logs" / f"{key.replace('/', '-')}.log"
        code, wall, max_rss, max_cpu = run_child(cmd, log_path, timeout, status_path)
        preflight["child_max_rss_kib"] = max_rss
        preflight["child_max_host_cpu_observed_percent"] = max_cpu
        preflight["cell_wall_seconds"] = wall
        if code != 0:
            raise RunnerExitError(key, code, log_path, wait_seconds, wall, preflight)
        summary_path = (output_dir if key.startswith("val/") else
                        output_dir / ("timing.json" if key == "timing" else "summary.json"))
        if summary_path.is_file():
            payload = json.loads(summary_path.read_text())
            payload["coordinator_preflight"] = preflight
            atomic_json(summary_path, payload)
        note(status_path, f"DONE {key}; wall={wall:.1f}s; RSSmax={max_rss} KiB")
        return {"key": key, "status": "complete", "preflight": preflight, "log": str(log_path)}, wait_seconds
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def runner_cmd(mode: str, out: Path, arm: str | None = None, seed: int | None = None, extra: list[str] | None = None) -> list[str]:
    cmd = [sys.executable, str(RUNNER), "--mode", mode, "--out", str(out), "--device", "cuda:0", "--lock-held"]
    if arm is not None:
        cmd += ["--arm", arm]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if extra:
        cmd += extra
    return cmd


def run_timing() -> dict:
    status_path = RUN_ROOT / "H55_STATUS.log"
    timing_dir = RUN_ROOT / "timing" / "ema25-seed12"
    if timing_dir.exists():
        raise RuntimeError(f"timing receipt already exists: {timing_dir}")
    result, wait = run_locked_cell(
        key="timing", cmd=runner_cmd("timing", timing_dir, "ema25", 12),
        output_dir=timing_dir, timeout=15 * 60, status_path=status_path, cumulative_wait=0.0,
    )
    wave = {
        "schema": "jope.h55.timing-wave/v1", "status": "timing_complete",
        "prereg_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "timing_cell": result, "lock_wait_seconds": wait,
        "outcome_cells_started": 0, "fold_b_scored": False, "dl_val_reads": 0,
        "start_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(RUN_ROOT / "timing_wave.json", wave)
    return wave


def amendment_cap(path: Path) -> int:
    relpath = path
    if not relpath.is_absolute():
        from_root = ROOT / relpath
        relpath = from_root if from_root.exists() else ROOT / "research/flywheel_selfplay/h55" / relpath
    try:
        relpath = relpath.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("execution amendment must be committed inside this repository") from exc
    committed = subprocess.check_output(["git", "show", f"HEAD:{relpath.as_posix()}"], cwd=ROOT, text=True)
    if path.is_absolute() and path.read_text() != committed:
        raise RuntimeError("execution amendment working copy differs from committed HEAD")
    if "amendment-" not in relpath.name or not any(ch.isdigit() for ch in relpath.name):
        raise RuntimeError("execution amendment filename must be dated")
    timing_receipt = RUN_ROOT / "timing/ema25-seed12/timing.json"
    audit_receipt = RUN_ROOT / "audit-timing.json"
    if not timing_receipt.is_file() or not audit_receipt.is_file():
        raise RuntimeError("timing cell and passing timing audit are required before outcomes")
    timing = json.loads(timing_receipt.read_text())
    audit = json.loads(audit_receipt.read_text())
    if timing.get("status") != "complete" or timing.get("fold_b_scored") is not False or audit.get("timing_cell") != "pass":
        raise RuntimeError("timing receipt or independent audit is not passing")
    text = committed
    marker = "total_gpu_wall_cap_seconds:"
    for line in text.splitlines():
        if marker in line:
            value = line.split(marker, 1)[1].strip().split()[0]
            cap = int(value)
            if not 1 <= cap <= 1800:
                raise RuntimeError("amended H55 total GPU wall cap must be <=1800 seconds")
            projection = float(timing["wall_seconds"]) * 4.0 * 1.20
            if projection > cap:
                raise RuntimeError(f"timing projection {projection:.1f}s exceeds amended total GPU cap {cap}s")
            return cap
    raise RuntimeError("dated amendment is missing total_gpu_wall_cap_seconds")


def run_outcomes(amendment: Path, prior_lock_wait_seconds: float = 0.0,
                 prior_gpu_wall_seconds: float = 0.0) -> dict:
    cap = amendment_cap(amendment)
    status_path = RUN_ROOT / "H55_STATUS.log"
    outcome_root = RUN_ROOT / "outcomes"
    wave_path = outcome_root / "wave.json"
    if wave_path.exists():
        raise RuntimeError("H55 outcome wave receipt already exists")
    if outcome_root.exists() and any(outcome_root.iterdir()):
        raise RuntimeError("H55 outcome directory contains a prior or partial wave")
    outcome_root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    cells = []
    wait_total = prior_lock_wait_seconds
    gpu_wall_total = prior_gpu_wall_seconds
    reason = None
    for index, (arm, seed) in enumerate(CELL_ORDER, 1):
        remaining = cap - gpu_wall_total
        if remaining <= 10:
            reason = "preregistered total locked-cell GPU wall cap reached between cells"
            break
        out = outcome_root / "cells" / arm / f"seed{seed}"
        try:
            cell, waited = run_locked_cell(
                key=f"{arm}/seed{seed}", cmd=runner_cmd("outcome", out, arm, seed),
                output_dir=out, timeout=remaining, status_path=status_path,
                cumulative_wait=wait_total,
            )
            wait_total += waited
            gpu_wall_total += float(cell["preflight"]["cell_wall_seconds"])
            cells.append(cell)
        except Exception as exc:
            reason = f"infra-void/stopped at {arm}/seed{seed}: {type(exc).__name__}: {exc}"
            if isinstance(exc, LockWaitTimeout):
                wait_total = exc.cumulative_wait_seconds
                cells.append({"key": f"{arm}/seed{seed}", "status": "infra_void", "error": reason,
                              "lock_wait_seconds": exc.wait_seconds})
            elif isinstance(exc, CellPreflightError):
                wait_total += exc.wait_seconds
                cells.append({"key": f"{arm}/seed{seed}", "status": "infra_void", "error": reason,
                              "lock_wait_seconds": exc.wait_seconds})
            elif isinstance(exc, RunnerExitError):
                wait_total += exc.wait_seconds
                gpu_wall_total += exc.wall_seconds
                cells.append({"key": f"{arm}/seed{seed}", "status": "infra_void", "error": reason,
                              "lock_wait_seconds": exc.wait_seconds, "preflight": exc.preflight,
                              "log": str(exc.log_path)})
            else:
                cells.append({"key": f"{arm}/seed{seed}", "status": "infra_void", "error": reason,
                              "lock_wait_seconds": exception_wait_seconds(exc)})
            break
    completed = len(cells) == len(CELL_ORDER) and all(c["status"] == "complete" for c in cells)
    wave = {
        "schema": "jope.h55.outcome-wave/v1", "complete": completed,
        "prereg_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "execution_amendment": str(amendment), "total_gpu_wall_cap_seconds": cap,
        "planned_order": [{"arm": a, "seed": s} for a, s in CELL_ORDER],
        "cells": cells, "stop_reason": reason, "wall_seconds": time.monotonic() - start,
        "total_gpu_cell_wall_seconds": gpu_wall_total,
        "prior_lock_wait_seconds": prior_lock_wait_seconds,
        "prior_gpu_wall_seconds": prior_gpu_wall_seconds,
        "lock_wait_seconds": wait_total, "terminal_val_reads": 0,
    }
    atomic_json(wave_path, wave)
    return wave


def run_terminal_val() -> dict:
    outcome_root = RUN_ROOT / "outcomes"
    admission_path = outcome_root / "admission.json"
    if not admission_path.is_file():
        raise RuntimeError("build_metrics.py must create admission.json before terminal DL-Val")
    admission = json.loads(admission_path.read_text())
    val_root = outcome_root / "terminal-val"
    reads = []
    wait_total = 0.0
    status_path = RUN_ROOT / "H55_STATUS.log"
    # Red-team cap: at most one terminal read for each seed, even if both arms admit.
    selected_arm = terminal_arm_by_preregistered_order(admission)
    for arm in ([selected_arm] if selected_arm else []):
        for seed in (12, 13, 14):
            cell_dir = outcome_root / "cells" / arm / f"seed{seed}"
            adapter = cell_dir / "candidate_adapter.pt"
            out = val_root / arm / f"seed{seed}.json"
            cmd = runner_cmd(
                "terminal-val", out, arm, seed,
                ["--adapter", str(adapter), "--admission", str(admission_path)],
            )
            result_row, waited = run_locked_cell(
                key=f"val/{arm}/seed{seed}", cmd=cmd, output_dir=out,
                timeout=5 * 60, status_path=status_path, cumulative_wait=wait_total,
            )
            wait_total += waited
            reads.append(result_row)
    summary = {
        "schema": "jope.h55.terminal-val-wave/v1",
        "reads": reads,
        "read_count": len(reads),
        "admitted_arms": [a for a in ("swa8", "ema25")
                          if admission.get("arms", {}).get(a, {}).get("admitted") is True],
        "terminal_arm_selected_by_preregistered_order": selected_arm,
        "no_admission_means_no_dl_val": True,
        "lock_wait_seconds": wait_total,
    }
    atomic_json(outcome_root / "terminal_val_wave.json", summary)
    return summary


def run_dry() -> dict:
    dry_path = RUN_ROOT / "dry-run.json"
    cmd = [sys.executable, str(RUNNER), "--mode", "dry-run", "--out", str(dry_path), "--device", "cpu"]
    preflight = safe_preflight()
    code, wall, max_rss, max_cpu = run_child(
        cmd, RUN_ROOT / "logs/dry-run.log", 15 * 60, RUN_ROOT / "H55_STATUS.log"
    )
    if code != 0:
        raise RuntimeError(f"H55 two-context dry run exited {code}")
    result = json.loads(dry_path.read_text())
    result["wall_seconds"] = wall
    result["max_rss_kib"] = max_rss
    result["max_host_cpu_percent"] = max_cpu
    result["preflight"] = preflight
    atomic_json(dry_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--timing", action="store_true")
    modes.add_argument("--outcomes", action="store_true")
    modes.add_argument("--terminal-val", action="store_true")
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--prior-lock-wait-seconds", type=float, default=0.0)
    parser.add_argument("--prior-gpu-wall-seconds", type=float, default=0.0)
    args = parser.parse_args()
    usage = shutil.disk_usage("/")
    disk_percent = 100.0 * (usage.total - usage.free) / usage.total
    if usage.free < 1024 * 1024 * 1024 or disk_percent >= 95.0:
        raise RuntimeError(f"unsafe disk before run-wave writes: free={usage.free} bytes used={disk_percent:.2f}%")
    if args.dry_run:
        result = run_dry()
    elif args.timing:
        result = run_timing()
    elif args.outcomes:
        if args.amendment is None:
            raise RuntimeError("outcome cells require the committed dated execution amendment")
        if not 0.0 <= args.prior_lock_wait_seconds < MAX_LOCK_WAIT_SECONDS:
            raise RuntimeError("prior lock wait must be nonnegative and below the three-hour limit")
        if not 0.0 <= args.prior_gpu_wall_seconds < 1800.0:
            raise RuntimeError("prior locked-cell wall must be nonnegative and below the 1,800-second cap")
        result = run_outcomes(args.amendment, args.prior_lock_wait_seconds,
                              args.prior_gpu_wall_seconds)
    else:
        result = run_terminal_val()
    mode = "dry-run" if args.dry_run else "timing" if args.timing else "outcomes" if args.outcomes else "terminal-val"
    print(json.dumps({"mode": mode, "result": result}, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
