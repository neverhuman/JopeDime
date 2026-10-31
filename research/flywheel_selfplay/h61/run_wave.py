#!/usr/bin/env python3
"""Serialize H61 timing, 12 paired outcome cells, fold-B admission scoring, and eligible terminal DL-Val reads."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "research/flywheel_selfplay/h61/runner.py"
DEFAULT_AMENDMENT = ROOT / "research/flywheel_selfplay/h61/amendment-2026-10-31.md"
LOCK = Path.home() / "jope_plan/qualification.lock"
RUN_ROOT = Path(os.environ.get("H61_RUN_ROOT", str(Path.home() / "jope-data/h61/h61-wave")))
MIN_GPU_FREE_MIB = 6144
MIN_MEM_AVAILABLE_KIB = 16 * 1024 * 1024
MAX_LOCK_WAIT_SECONDS = 3 * 60 * 60
MAX_WAVE_GPU_WALL_SECONDS = 60 * 60
RSS_LIMIT_KIB = 6 * 1024 * 1024
MAX_LAUNCH_DISK_USED_PERCENT = 94.5
WRITER_MAX_DISK_USED_PERCENT = 95.0
MAX_DISK_GUARD_RETRIES = 2
LOCK_POLL_SECONDS = 5.0
LOCK_STATUS_NOTE_INTERVAL_SECONDS = 30.0
ARMS = ("S0", "S1", "S2", "S3")
SEEDS = (12, 13, 14)
CELL_ORDER = [(arm, seed) for seed in SEEDS for arm in ARMS]


class InfraVoid(RuntimeError):
    def __init__(self, message: str, wait_seconds: float = 0.0, cell_wall_seconds: float = 0.0):
        self.wait_seconds = wait_seconds
        self.cell_wall_seconds = cell_wall_seconds
        super().__init__(message)


class DiskGuardInfraVoid(InfraVoid):
    """A completed or writing child whose artifact write was blocked by disk pressure; retryable on the same cell."""


def atomic_json(path: Path, value: dict) -> None:
    err = disk_guard_error()
    if err:
        raise RuntimeError(err)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def note(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(f"{datetime.now(timezone.utc).isoformat()} — {message}\n")
        stream.flush()
        os.fsync(stream.fileno())


def memory_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable unavailable")


def cpu_percent(interval: float = 0.5) -> float:
    def read() -> tuple[int, int]:
        fields = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
        return sum(fields), fields[3] + fields[4]

    total_a, idle_a = read()
    time.sleep(interval)
    total_b, idle_b = read()
    total = total_b - total_a
    return 100 * (total - (idle_b - idle_a)) / total if total else 0.0


def gpu_free_mib() -> int:
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
        text=True,
    )
    return int(result.strip().splitlines()[0])


def disk_usage_stats() -> dict[str, float | int]:
    stats = os.statvfs("/")
    total = stats.f_blocks * stats.f_frsize
    available = stats.f_bavail * stats.f_frsize
    used_percent = 100 * (stats.f_blocks - stats.f_bavail) / stats.f_blocks
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_percent": used_percent,
    }


def disk_guard_error() -> str | None:
    usage = disk_usage_stats()
    if usage["available_bytes"] < 1024**3:
        return f"less than 1 GiB available on / ({usage['available_bytes']} bytes)"
    if usage["used_percent"] >= WRITER_MAX_DISK_USED_PERCENT:
        return f"filesystem / is at or above 95% full ({usage['used_percent']:.2f}%)"
    return None


def launch_disk_guard_error() -> str | None:
    usage = disk_usage_stats()
    if usage["available_bytes"] < 1024**3:
        return f"less than 1 GiB available on / ({usage['available_bytes']} bytes)"
    if usage["used_percent"] >= MAX_LAUNCH_DISK_USED_PERCENT:
        return (
            f"filesystem / launch gate requires below {MAX_LAUNCH_DISK_USED_PERCENT:.1f}% "
            f"used ({usage['used_percent']:.2f}%)"
        )
    return None


def safe_preflight() -> dict:
    disk_error = launch_disk_guard_error()
    if disk_error:
        raise RuntimeError(disk_error)
    cpu = cpu_percent()
    available = memory_available_kib()
    if cpu > 85.0:
        raise RuntimeError(f"host CPU above 85%: {cpu:.1f}%")
    if available < MIN_MEM_AVAILABLE_KIB:
        raise RuntimeError(f"MemAvailable below 16 GiB: {available} KiB")
    disk = disk_usage_stats()
    return {
        "host_cpu_percent": cpu,
        "mem_available_kib": available,
        "disk_free_bytes": disk["available_bytes"],
        "disk_used_percent": disk["used_percent"],
    }


def process_rss_kib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None


def remaining_lock_wait(cumulative: float) -> float:
    return max(0.0, MAX_LOCK_WAIT_SECONDS - cumulative)


def preflight_wait_reason(error: RuntimeError) -> str | None:
    message = str(error)
    if "launch gate requires below" in message:
        usage = disk_usage_stats()
        return f"disk {usage['used_percent']:.2f}% vs {MAX_LAUNCH_DISK_USED_PERCENT:.1f}% launch gate"
    if "less than 1 GiB available on /" in message:
        usage = disk_usage_stats()
        return f"disk {usage['used_percent']:.2f}% used; available below 1 GiB"
    if "filesystem /" in message:
        usage = disk_usage_stats()
        return f"disk {usage['used_percent']:.2f}% vs 95% guard"
    if "MemAvailable" in message or "CPU" in message:
        return message
    return None


def lock_holder_pid_from_proc_locks(contents: str, lock_key: str) -> int | None:
    for line in contents.splitlines():
        fields = line.split()
        if (
            len(fields) >= 6
            and fields[1] == "FLOCK"
            and fields[3] in ("READ", "WRITE")
            and fields[5] == lock_key
            and fields[4].isdigit()
        ):
            return int(fields[4])
    return None


def lock_holder_pid(handle) -> int | None:
    stat = os.fstat(handle.fileno())
    lock_key = f"{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    return lock_holder_pid_from_proc_locks(Path("/proc/locks").read_text(), lock_key)


def format_lock_wait_reason(holder: int | None) -> str:
    return f"qualification.lock busy; holder PID={holder if holder is not None else 'unknown'}"


def lock_busy_wait_reason(handle) -> str:
    return format_lock_wait_reason(lock_holder_pid(handle))


def gpu_wait_reason(free_mib: int) -> str:
    return f"GPU free {free_mib} MiB below {MIN_GPU_FREE_MIB} MiB"


def preflight_snapshot(*, immediate_first_retry: bool = True) -> dict:
    """Return pre-lock snapshot; when immediate_first_retry=True, avoid pre-lock cpu_percent sleep so cells do not yield."""
    if immediate_first_retry:
        disk = disk_usage_stats()
        return {
            "disk_free_bytes": disk["available_bytes"],
            "disk_used_percent": disk["used_percent"],
            "mode": "fast_statvfs_only",
        }
    try:
        return safe_preflight()
    except RuntimeError as exc:
        reason = preflight_wait_reason(exc)
        if reason is None:
            raise
        return {"blocked": reason}


def acquire_lock(
    status_path: Path,
    wait_total: float,
    *,
    immediate_first_retry: bool = True,
    lock_path: Path = LOCK,
    flock_fn: Callable = fcntl.flock,
    sleep_fn: Callable[[float], None] = time.sleep,
    gpu_free_fn: Callable[[], int] = gpu_free_mib,
    preflight_fn: Callable[[], dict] = safe_preflight,
) -> tuple[object, float, int, dict]:
    """Acquire qualification.lock; when immediate_first_retry=True, retry immediately on first contention without sleeping."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    started = time.monotonic()
    last_note = 0.0
    # When immediate_first_retry=True, do not run pre-flock safe_preflight() on the initial attempt
    # (safe_preflight() is always verified under the lock right after flock succeeds).
    last_full_check = started if immediate_first_retry else 0.0
    immediate_retry_used = False
    while True:
        now = time.monotonic()
        acquired = False
        keep_locked = False
        wait_reason = "preflight pending"
        try:
            disk_error = launch_disk_guard_error()
            if disk_error:
                raise RuntimeError(disk_error)
            status = None
            if now - last_full_check >= 300:
                try:
                    status = preflight_fn()
                except RuntimeError:
                    last_full_check = now
                    raise
                last_full_check = now
            flock_fn(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            free = gpu_free_fn()
            if free < MIN_GPU_FREE_MIB:
                wait_reason = gpu_wait_reason(free)
            else:
                try:
                    status = preflight_fn()
                except RuntimeError as exc:
                    wait_reason = preflight_wait_reason(exc) or str(exc)
                    raise
                keep_locked = True
                return handle, time.monotonic() - started, free, status
        except BlockingIOError:
            wait_reason = lock_busy_wait_reason(handle)
        except RuntimeError as exc:
            reason = preflight_wait_reason(exc)
            if reason is None:
                raise
            wait_reason = reason
        except BaseException:
            if acquired:
                flock_fn(handle, fcntl.LOCK_UN)
                acquired = False
            handle.close()
            raise
        finally:
            if acquired and not keep_locked:
                flock_fn(handle, fcntl.LOCK_UN)
        waited = time.monotonic() - started
        if waited >= remaining_lock_wait(wait_total):
            handle.close()
            raise InfraVoid(
                f"qualification.lock wait exceeded separate 3h limit; total={wait_total + waited:.1f}s",
                wait_total + waited,
            )
        if immediate_first_retry and not immediate_retry_used:
            immediate_retry_used = True
            continue
        if now - last_note >= LOCK_STATUS_NOTE_INTERVAL_SECONDS:
            note(status_path, f"WAIT {wait_reason}; cumulative_wait={wait_total + waited:.1f}s")
            last_note = now
        sleep_fn(LOCK_POLL_SECONDS)


def child_command(
    mode: str,
    out: Path,
    *,
    arm: str | None = None,
    seed: int | None = None,
    run_root: Path = RUN_ROOT,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--mode",
        mode,
        "--device",
        "cuda:0",
        "--lock-held",
        "--out",
        str(out),
        "--run-root",
        str(run_root),
    ]
    if arm is not None:
        command += ["--arm", arm]
    if seed is not None:
        command += ["--seed", str(seed)]
    return command


def run_child(command: list[str], log_path: Path, timeout: float) -> tuple[int, float, int, float]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CARGO_INCREMENTAL": "0",
            "CARGO_PROFILE_DEV_DEBUG": "line-tables-only",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    max_rss = 0
    max_cpu = 0.0
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.nice(10),
        )
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                rss = process_rss_kib(proc.pid)
                if rss is not None:
                    max_rss = max(max_rss, rss)
                if rss is not None and rss >= RSS_LIMIT_KIB:
                    proc.terminate()
                    raise RuntimeError(f"H61 child RSS reached 6 GiB: {rss} KiB")
                if elapsed >= timeout:
                    proc.terminate()
                    raise TimeoutError(f"H61 fixed wall watchdog expired after {timeout:.1f}s")
                cpu = cpu_percent(0.2)
                max_cpu = max(max_cpu, cpu)
                if cpu > 85.0 or memory_available_kib() < MIN_MEM_AVAILABLE_KIB:
                    proc.terminate()
                    raise RuntimeError("H61 CPU or MemAvailable safety guard tripped")
                time.sleep(1.0)
            code = proc.wait()
        except BaseException as exc:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if isinstance(exc, (RuntimeError, TimeoutError)):
                raise InfraVoid(str(exc), cell_wall_seconds=time.monotonic() - started) from exc
            raise
    return code, time.monotonic() - started, max_rss, max_cpu


def run_locked_cell(
    key: str,
    command: list[str],
    log: Path,
    timeout: float,
    status_path: Path,
    wait_total: float,
    *,
    immediate_first_retry: bool = True,
) -> tuple[dict, float, float]:
    before = preflight_snapshot(immediate_first_retry=immediate_first_retry)
    handle, waited, free_mib, lock_preflight = acquire_lock(
        status_path,
        wait_total,
        immediate_first_retry=immediate_first_retry,
    )
    try:
        post_lock = lock_preflight
        if free_mib < MIN_GPU_FREE_MIB:
            raise InfraVoid("GPU free memory below 6 GiB after lock", waited)
        note(
            status_path,
            f"START {key}; wait={waited:.2f}s; CUDAfree={free_mib}MiB; "
            f"preflight={json.dumps(post_lock, sort_keys=True)}",
        )
        try:
            code, wall, max_rss, max_cpu = run_child(command, log, timeout)
        except InfraVoid as exc:
            note(status_path, f"VOID {key}; child guard; wall={exc.cell_wall_seconds:.2f}s; reason={exc}")
            raise
        if code != 0:
            detail = f"{key} child exit={code}; log={log}; RSSmax={max_rss}; CPUmax={max_cpu:.1f}%"
            try:
                child_log = log.read_text(errors="replace")[-16_384:]
            except OSError:
                child_log = ""
            if (
                "unsafe filesystem / before H61 write" in child_log
                or "unsafe filesystem / before H55 write" in child_log
                or "RetryableDiskInfraVoid" in child_log
            ):
                note(
                    status_path,
                    f"VOID {key}; reason=post-training disk writer guard; wall={wall:.2f}s; log={log}",
                )
                raise DiskGuardInfraVoid(detail, waited, wall)
            raise InfraVoid(detail, waited, wall)
        note(status_path, f"DONE {key}; wall={wall:.2f}s; RSSmax={max_rss}KiB; CPUmax={max_cpu:.1f}%")
        return (
            {
                "key": key,
                "status": "complete",
                "lock_wait_seconds": waited,
                "cell_wall_seconds": wall,
                "gpu_free_mib_after_lock": free_mib,
                "preflight_before_lock": before,
                "preflight_after_lock": lock_preflight,
                "child_max_rss_kib": max_rss,
                "child_max_cpu_percent": max_cpu,
                "log": str(log),
            },
            waited,
            wall,
        )
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def run_cell_with_disk_guard_retries(
    key: str,
    command: list[str],
    log: Path,
    timeout: float,
    status_path: Path,
    wait_total: float,
    output_path: Path,
    *,
    immediate_first_retry: bool = True,
) -> tuple[dict, float, float, float]:
    retries = 0
    void_wall = 0.0
    local_wait = 0.0
    while True:
        attempt_log = log if retries == 0 else log.with_name(f"{log.stem}.retry{retries}{log.suffix}")
        try:
            cell, waited, wall = run_locked_cell(
                key,
                command,
                attempt_log,
                timeout,
                status_path,
                wait_total + local_wait,
                immediate_first_retry=immediate_first_retry,
            )
            cell["disk_guard_retry_count"] = retries
            cell["prior_void_cell_wall_seconds"] = void_wall
            cell["attempt_logs"] = [
                str(log if index == 0 else log.with_name(f"{log.stem}.retry{index}{log.suffix}"))
                for index in range(retries + 1)
            ]
            return cell, local_wait + waited, wall, void_wall
        except DiskGuardInfraVoid as exc:
            local_wait += exc.wait_seconds
            void_wall += exc.cell_wall_seconds
            if retries >= MAX_DISK_GUARD_RETRIES:
                raise DiskGuardInfraVoid(
                    f"{key} exhausted {MAX_DISK_GUARD_RETRIES} disk-guard retries; {exc}",
                    local_wait,
                    void_wall,
                ) from exc
            if output_path.exists():
                if (output_path / "summary.json").exists():
                    raise InfraVoid(
                        f"{key} disk-guard retry refused because summary.json already exists at {output_path}; {exc}",
                        local_wait,
                        void_wall,
                    ) from exc
                shutil.rmtree(output_path, ignore_errors=True)
            retries += 1
            note(
                status_path,
                f"RETRY {key}; disk guard; retry={retries}/{MAX_DISK_GUARD_RETRIES}; "
                f"void_wall={void_wall:.2f}s; wait for disk below {MAX_LAUNCH_DISK_USED_PERCENT:.1f}%",
            )
        except InfraVoid as exc:
            raise InfraVoid(
                str(exc),
                local_wait + exc.wait_seconds,
                void_wall + exc.cell_wall_seconds,
            ) from exc


def parse_amendment(path: Path) -> tuple[int, int, int, float, float]:
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("H61 amendment must be inside the repository") from exc
    if not rel.name.startswith("amendment-") or not any(ch.isdigit() for ch in rel.name):
        raise RuntimeError("H61 requires a dated pre-outcome amendment")
    committed = subprocess.check_output(["git", "show", f"HEAD:{rel.as_posix()}"], cwd=ROOT, text=True)
    if path.read_text() != committed:
        raise RuntimeError("H61 amendment working file differs from committed HEAD")
    fields = {}
    for line in committed.splitlines():
        if ":" in line and not line.lstrip().startswith("#") and not line.lstrip().startswith("-"):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    cap = int(fields["total_gpu_cell_wall_cap_seconds"])
    rounds = int(fields["fixed_round_horizon"])
    steps = int(fields["fixed_steps_per_round"])
    h57_real_cell = float(fields["measured_h57_real_cell_seconds"])
    projected = float(fields["projected_total_seconds"])
    if not 0 < cap <= MAX_WAVE_GPU_WALL_SECONDS or rounds != 20 or steps != 2 or projected > cap:
        raise RuntimeError("H61 amendment cap/horizon/projection violates the <=60-minute preregistration")
    if abs(h57_real_cell - 607.1792591279373) > 0.1:
        raise RuntimeError("H61 amendment must record the measured 607.2s H57 real cell time")
    return cap, rounds, steps, h57_real_cell, projected


def timing_mode(args) -> dict:
    status = RUN_ROOT / "H61_STATUS.log"
    out = RUN_ROOT / "timing"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("H61 timing evidence already exists")
    result, waited, wall = run_locked_cell(
        "timing/S0/seed12",
        child_command("timing", out, arm="S0", seed=12, run_root=RUN_ROOT),
        RUN_ROOT / "logs/timing.log",
        timeout=MAX_WAVE_GPU_WALL_SECONDS,
        status_path=status,
        wait_total=args.prior_lock_wait_seconds,
        immediate_first_retry=args.immediate_lock_retry,
    )
    receipt = json.loads((out / "timing.json").read_text())
    wave = {
        "schema": "jope.h61.timing-wave/v1",
        "status": "complete",
        "timing_cell": receipt,
        "cell_record": result,
        "prior_lock_wait_seconds": args.prior_lock_wait_seconds,
        "attempt_lock_wait_seconds": waited,
        "lock_wait_seconds": args.prior_lock_wait_seconds + waited,
        "coordinator_wall_seconds": wall,
        "outcome_cells_started": 0,
        "fold_b_scored": False,
        "dl_val_reads": 0,
    }
    atomic_json(RUN_ROOT / "timing/timing_wave.json", wave)
    return wave


def outcome_mode(args) -> dict:
    cap, rounds, steps, h57_real_cell, projected = parse_amendment(args.amendment)
    if (RUN_ROOT / "outcomes/wave.json").exists():
        raise RuntimeError("H61 outcome receipt already exists; preserve it")
    if (RUN_ROOT / "outcomes").exists() and any((RUN_ROOT / "outcomes").iterdir()):
        raise RuntimeError("H61 outcome directory is not empty")
    wave_path = RUN_ROOT / "outcomes/wave.json"
    status = RUN_ROOT / "H61_STATUS.log"
    cells = []
    wait_total = float(args.prior_lock_wait_seconds)
    gpu_wall = 0.0
    void_gpu_wall = 0.0
    reason = None
    for arm, seed in CELL_ORDER:
        out = RUN_ROOT / f"outcomes/cells/{arm}/seed{seed}"
        remaining = cap - gpu_wall
        if remaining <= 0:
            reason = "fixed H61 total GPU cell-wall cap reached before next cell"
            break
        command = child_command("outcome", out, arm=arm, seed=seed, run_root=RUN_ROOT)
        key = f"{arm}/seed{seed}"
        try:
            cell, waited, wall, void_wall = run_cell_with_disk_guard_retries(
                key,
                command,
                RUN_ROOT / f"logs/{arm}-seed{seed}.log",
                timeout=remaining,
                status_path=status,
                wait_total=wait_total,
                output_path=out,
                immediate_first_retry=args.immediate_lock_retry,
            )
            wait_total += waited
            gpu_wall += wall
            void_gpu_wall += void_wall
            cells.append(cell)
            atomic_json(
                RUN_ROOT / "outcomes/wave.partial.json",
                {
                    "schema": "jope.h61.outcome-wave/v1",
                    "complete": False,
                    "cells": cells,
                    "lock_wait_seconds": wait_total,
                    "total_gpu_cell_wall_seconds": gpu_wall,
                    "void_cell_wall_seconds": void_gpu_wall,
                    "total_gpu_cell_wall_cap_seconds": cap,
                    "immediate_lock_retry": args.immediate_lock_retry,
                },
            )
        except Exception as exc:
            wait_total += float(getattr(exc, "wait_seconds", 0.0))
            failed_void_wall = float(getattr(exc, "cell_wall_seconds", 0.0))
            void_gpu_wall += failed_void_wall
            reason = f"infra-void/stopped at {key}: {type(exc).__name__}: {exc}"
            cells.append(
                {
                    "key": key,
                    "status": "infra_void",
                    "error": reason,
                    "lock_wait_seconds": float(getattr(exc, "wait_seconds", 0.0)),
                    "void_cell_wall_seconds": failed_void_wall,
                    "disk_guard_retry_count": MAX_DISK_GUARD_RETRIES
                    if isinstance(exc, DiskGuardInfraVoid)
                    else 0,
                }
            )
            atomic_json(
                RUN_ROOT / "outcomes/wave.partial.json",
                {
                    "schema": "jope.h61.outcome-wave/v1",
                    "complete": False,
                    "cells": cells,
                    "lock_wait_seconds": wait_total,
                    "total_gpu_cell_wall_seconds": gpu_wall,
                    "void_cell_wall_seconds": void_gpu_wall,
                    "total_gpu_cell_wall_cap_seconds": cap,
                    "immediate_lock_retry": args.immediate_lock_retry,
                },
            )
            break

    complete = len(cells) == len(CELL_ORDER) and all(cell["status"] == "complete" for cell in cells)
    if complete:
        score_out = RUN_ROOT / "outcomes/fold-b-score"
        try:
            score_cell, waited, wall, void_wall = run_cell_with_disk_guard_retries(
                "fold-b/admission",
                child_command("fold-b", score_out, run_root=RUN_ROOT),
                RUN_ROOT / "logs/fold-b.log",
                timeout=max(1.0, cap - gpu_wall),
                status_path=status,
                wait_total=wait_total,
                output_path=score_out,
                immediate_first_retry=args.immediate_lock_retry,
            )
            wait_total += waited
            gpu_wall += wall
            void_gpu_wall += void_wall
            cells.append(score_cell)
        except Exception as exc:
            wait_total += float(getattr(exc, "wait_seconds", 0.0))
            complete = False
            reason = f"infra-void/stopped at fold-B scoring: {type(exc).__name__}: {exc}"
            cells.append(
                {
                    "key": "fold-b/admission",
                    "status": "infra_void",
                    "error": reason,
                    "lock_wait_seconds": float(getattr(exc, "wait_seconds", 0.0)),
                }
            )

    wave = {
        "schema": "jope.h61.outcome-wave/v1",
        "complete": complete,
        "preregistration_commit": subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "HEAD", "--", "research/flywheel_selfplay/h61/PREREG.md"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "execution_amendment": str(args.amendment),
        "fixed_round_horizon": rounds,
        "fixed_steps_per_round": steps,
        "measured_h57_real_cell_seconds": h57_real_cell,
        "projected_total_seconds": projected,
        "immediate_lock_retry": args.immediate_lock_retry,
        "primary_ranking_metric": "bradley_terry_elo_family_grouped",
        "auc_reported": False,
        "cell_order": [{"arm": arm, "seed": seed} for arm, seed in CELL_ORDER],
        "cells": cells,
        "stop_reason": reason,
        "lock_wait_seconds": wait_total,
        "total_gpu_cell_wall_seconds": gpu_wall,
        "total_gpu_cell_wall_cap_seconds": cap,
        "void_cell_wall_seconds": void_gpu_wall,
        "void_wall_excluded_from_successful_outcome_cap": True,
        "candidate_cells": len(CELL_ORDER),
        "fold_b_candidate_reads": len(CELL_ORDER) if complete else 0,
        "terminal_val_reads": 0,
    }
    atomic_json(wave_path, wave)
    if complete:
        admission = json.loads((RUN_ROOT / "outcomes/admission.json").read_text())
        if any(admission.get("arms", {}).get(arm, {}).get("admitted") is True for arm in ARMS):
            terminal_mode(immediate_first_retry=args.immediate_lock_retry)
    return wave


def terminal_mode(*, immediate_first_retry: bool = True) -> dict:
    wave_path = RUN_ROOT / "outcomes/wave.json"
    wave = json.loads(wave_path.read_text())
    if wave.get("complete") is not True:
        raise RuntimeError("H61 terminal validation requires a complete candidate/fold-B wave")
    admission = json.loads((RUN_ROOT / "outcomes/admission.json").read_text())
    arms = [
        arm for arm in ARMS if admission.get("arms", {}).get(arm, {}).get("admitted") is True
    ]
    planned = [(arm, seed) for arm in arms for seed in SEEDS]
    already = {
        (json.loads(path.read_text())["arm"], int(json.loads(path.read_text())["seed"]))
        for path in (RUN_ROOT / "outcomes/terminal-val").glob("*/*.json")
        if path.exists()
    }
    reads = []
    wait_total = float(wave.get("lock_wait_seconds", 0.0))
    gpu_wall = float(wave.get("total_gpu_cell_wall_seconds", 0.0))
    cap = float(wave.get("total_gpu_cell_wall_cap_seconds", 0.0))
    for arm, seed in planned:
        if (arm, seed) in already:
            raise RuntimeError(f"refusing duplicate H61 terminal read {arm}/seed{seed}")
        remaining = cap - gpu_wall
        if remaining <= 0:
            raise RuntimeError("H61 terminal DL-Val budget exhausted under the frozen 60-minute cap")
        out = RUN_ROOT / f"outcomes/terminal-val/{arm}/seed{seed}.json"
        command = child_command("terminal-val", out, arm=arm, seed=seed, run_root=RUN_ROOT)
        try:
            cell, waited, wall, _ = run_cell_with_disk_guard_retries(
                f"terminal-val/{arm}/seed{seed}",
                command,
                RUN_ROOT / f"logs/terminal-val-{arm}-seed{seed}.log",
                remaining,
                RUN_ROOT / "H61_STATUS.log",
                wait_total,
                out.parent,
                immediate_first_retry=immediate_first_retry,
            )
        except Exception as exc:
            wave["terminal_stop_reason"] = f"infra-void at {arm}/seed{seed}: {type(exc).__name__}: {exc}"
            wave["terminal_reads_completed"] = len(reads)
            wave["lock_wait_seconds"] = wait_total + float(getattr(exc, "wait_seconds", 0.0))
            wave["total_gpu_cell_wall_seconds"] = gpu_wall
            atomic_json(wave_path, wave)
            raise
        wait_total += waited
        gpu_wall += wall
        read = json.loads(out.read_text())
        reads.append(read)
        wave["cells"].append(cell)
    wave["terminal_reads_completed"] = len(reads)
    wave["lock_wait_seconds"] = wait_total
    wave["total_gpu_cell_wall_seconds"] = gpu_wall
    wave["terminal_val_reads"] = len(reads)
    atomic_json(wave_path, wave)
    summary = {
        "schema": "jope.h61.terminal-val-wave/v1",
        "reads": reads,
        "read_count": len(reads),
        "admitted_arms": arms,
        "lock_wait_seconds": wait_total,
    }
    atomic_json(RUN_ROOT / "outcomes/terminal_val_wave.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--timing", action="store_true")
    modes.add_argument("--outcomes", action="store_true")
    modes.add_argument("--terminal-val", action="store_true")
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--prior-lock-wait-seconds", type=float, default=0.0)
    parser.add_argument(
        "--immediate-lock-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-try qualification.lock immediately (0s sleep before first retry) after lock release/contention",
    )
    args = parser.parse_args()
    if args.prior_lock_wait_seconds < 0:
        raise RuntimeError("prior lock wait cannot be negative")
    if args.amendment is None and args.outcomes:
        raise RuntimeError("H61 outcomes require the committed dated resource amendment")
    if args.timing:
        mode, result = "timing", timing_mode(args)
    elif args.outcomes:
        mode, result = "outcomes", outcome_mode(args)
    else:
        mode, result = "terminal-val", terminal_mode(immediate_first_retry=args.immediate_lock_retry)
    print(json.dumps({"mode": mode, "result": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
