#!/usr/bin/env python3
"""Serialize H57 timing, candidate/null cells, fold-B scoring, and terminal reads."""
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
RUNNER = ROOT / "research/flywheel_selfplay/h57/runner.py"
LOCK = Path.home() / "jope_plan/qualification.lock"
RUN_ROOT = Path(os.environ.get("H57_RUN_ROOT", str(Path.home() / "jope-data/h57/h57-wave")))
MIN_GPU_FREE_MIB = 6144
MIN_MEM_AVAILABLE_KIB = 20 * 1024 * 1024
MAX_LOCK_WAIT_SECONDS = 3 * 60 * 60
MAX_WAVE_GPU_WALL_SECONDS = 90 * 60
RSS_LIMIT_KIB = 6 * 1024 * 1024
CELL_ORDER = [(arm, seed) for arm in ("sigma001", "sigma003") for seed in (15, 16, 17)]
CELL_ORDER.append(("null_sigma001", 15))


class InfraVoid(RuntimeError):
    def __init__(self, message: str, wait_seconds: float = 0.0):
        self.wait_seconds = wait_seconds
        super().__init__(message)


def atomic_json(path: Path, value: dict) -> None:
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


def disk_guard_error() -> str | None:
    usage = shutil.disk_usage("/")
    used = 100 * (usage.total - usage.free) / usage.total
    if usage.free < 1024**3:
        return f"less than 1 GiB free on / ({usage.free} bytes)"
    if used >= 95.0:
        return f"filesystem / is at or above 95% full ({used:.2f}%)"
    return None


def safe_preflight() -> dict:
    disk_error = disk_guard_error()
    if disk_error:
        raise RuntimeError(disk_error)
    cpu = cpu_percent()
    available = memory_available_kib()
    if cpu > 85.0:
        raise RuntimeError(f"host CPU above 85%: {cpu:.1f}%")
    if available < MIN_MEM_AVAILABLE_KIB:
        raise RuntimeError(f"MemAvailable below 20 GiB: {available} KiB")
    return {"host_cpu_percent": cpu, "mem_available_kib": available,
            "disk_free_bytes": shutil.disk_usage("/").free,
            "disk_used_percent": 100 * (shutil.disk_usage("/").total - shutil.disk_usage("/").free)
            / shutil.disk_usage("/").total}


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


def acquire_lock(status_path: Path, wait_total: float) -> tuple[object, float, int, dict]:
    handle = LOCK.open("a+")
    started = time.monotonic()
    last_note = 0.0
    last_full_check = 0.0
    while True:
        now = time.monotonic()
        acquired = False
        keep_locked = False
        try:
            status = safe_preflight() if now - last_full_check >= 300 else None
            if status is not None:
                last_full_check = now
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            free = gpu_free_mib()
            if free >= MIN_GPU_FREE_MIB:
                # Recheck under the lock so pressure changes during acquisition
                # cannot authorize a launch. The exception path below releases it.
                status = safe_preflight()
                keep_locked = True
                return handle, time.monotonic() - started, free, status
        except BlockingIOError:
            pass
        except RuntimeError as exc:
            if "filesystem /" not in str(exc) and "MemAvailable" not in str(exc) and "CPU" not in str(exc):
                raise
        except BaseException:
            if acquired:
                fcntl.flock(handle, fcntl.LOCK_UN)
                acquired = False
            handle.close()
            raise
        finally:
            if acquired and not keep_locked:
                fcntl.flock(handle, fcntl.LOCK_UN)
        waited = time.monotonic() - started
        if waited >= remaining_lock_wait(wait_total):
            handle.close()
            raise InfraVoid(f"qualification.lock wait exceeded separate 3h limit; total={wait_total+waited:.1f}s",
                             wait_total + waited)
        if now - last_note >= 30:
            note(status_path, f"WAIT qualification.lock / GPU >=6GiB; cumulative_wait={wait_total+waited:.1f}s")
            last_note = now
        time.sleep(5)


def child_command(mode: str, out: Path, *, arm: str | None = None, seed: int | None = None,
                  generations: int = 8, null_arm: bool = False, run_root: Path = RUN_ROOT) -> list[str]:
    command = [sys.executable, str(RUNNER), "--mode", mode, "--device", "cuda:0", "--lock-held",
               "--out", str(out), "--run-root", str(run_root), "--generations", str(generations)]
    if arm is not None:
        command += ["--arm", arm]
    if seed is not None:
        command += ["--seed", str(seed)]
    if null_arm:
        command.append("--null-arm")
    return command


def run_child(command: list[str], log_path: Path, timeout: float) -> tuple[int, float, int, float]:
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "CARGO_INCREMENTAL": "0",
                "CARGO_PROFILE_DEV_DEBUG": "line-tables-only"})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    max_rss = 0
    max_cpu = 0.0
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                preexec_fn=lambda: os.nice(10))
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                rss = process_rss_kib(proc.pid)
                if rss is not None:
                    max_rss = max(max_rss, rss)
                if rss is not None and rss >= RSS_LIMIT_KIB:
                    proc.terminate()
                    raise RuntimeError(f"H57 child RSS reached 6 GiB: {rss} KiB")
                if elapsed >= timeout:
                    proc.terminate()
                    raise TimeoutError(f"H57 fixed wall watchdog expired after {timeout:.1f}s")
                cpu = cpu_percent(0.2)
                max_cpu = max(max_cpu, cpu)
                if cpu > 85.0 or memory_available_kib() < MIN_MEM_AVAILABLE_KIB:
                    proc.terminate()
                    raise RuntimeError("H57 CPU or MemAvailable safety guard tripped")
                time.sleep(1.0)
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


def run_locked_cell(key: str, command: list[str], log: Path, timeout: float,
                    status_path: Path, wait_total: float) -> tuple[dict, float, float]:
    before = safe_preflight()
    handle, waited, free_mib, lock_preflight = acquire_lock(status_path, wait_total)
    try:
        post_lock = safe_preflight()
        if free_mib < MIN_GPU_FREE_MIB:
            raise InfraVoid("GPU free memory below 6 GiB after lock", waited)
        note(status_path, f"START {key}; wait={waited:.2f}s; CUDAfree={free_mib}MiB; preflight={json.dumps(post_lock, sort_keys=True)}")
        code, wall, max_rss, max_cpu = run_child(command, log, timeout)
        if code != 0:
            raise InfraVoid(f"{key} child exit={code}; log={log}; RSSmax={max_rss}; CPUmax={max_cpu:.1f}%", waited)
        note(status_path, f"DONE {key}; wall={wall:.2f}s; RSSmax={max_rss}KiB; CPUmax={max_cpu:.1f}%")
        return {"key": key, "status": "complete", "lock_wait_seconds": waited,
                "cell_wall_seconds": wall, "gpu_free_mib_after_lock": free_mib,
                "preflight_before_lock": before, "preflight_after_lock": lock_preflight,
                "child_max_rss_kib": max_rss, "child_max_cpu_percent": max_cpu,
                "log": str(log)}, waited, wall
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def timing_mode(args) -> dict:
    status = RUN_ROOT / "H57_STATUS.log"
    out = RUN_ROOT / "timing"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("H57 timing evidence already exists")
    result, waited, wall = run_locked_cell(
        "timing/sigma001/seed15",
        child_command("timing", out, arm="sigma001", seed=15, generations=1),
        RUN_ROOT / "logs/timing.log", timeout=MAX_WAVE_GPU_WALL_SECONDS,
        status_path=status, wait_total=0.0,
    )
    receipt = json.loads((out / "timing.json").read_text())
    wave = {"schema": "jope.h57.timing-wave/v1", "status": "complete",
            "timing_cell": receipt, "lock_wait_seconds": waited,
            "coordinator_wall_seconds": wall, "outcome_cells_started": 0,
            "fold_b_scored": False, "dl_val_reads": 0}
    atomic_json(RUN_ROOT / "timing/timing_wave.json", wave)
    return wave


def parse_amendment(path: Path) -> tuple[int, int, float]:
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("H57 amendment must be inside the repository") from exc
    if not rel.name.startswith("amendment-") or not any(ch.isdigit() for ch in rel.name):
        raise RuntimeError("H57 requires a dated pre-outcome amendment")
    committed = subprocess.check_output(["git", "show", f"HEAD:{rel.as_posix()}"], cwd=ROOT, text=True)
    if path.read_text() != committed:
        raise RuntimeError("H57 amendment working file differs from committed HEAD")
    fields = {}
    for line in committed.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    cap = int(fields["total_gpu_cell_wall_cap_seconds"])
    horizon = int(fields["fixed_generation_horizon"])
    projected = float(fields["projected_total_seconds"])
    if not 0 < cap <= MAX_WAVE_GPU_WALL_SECONDS or horizon not in (8, 12) or projected > cap:
        raise RuntimeError("H57 amendment cap/horizon/projection violates the <=90-minute preregistration")
    timing = json.loads((RUN_ROOT / "timing/timing.json").read_text())
    timing_gen = float(timing["generations"][0]["generation_wall_seconds"])
    if abs(float(fields["measured_generation_seconds"]) - timing_gen) > 0.01:
        raise RuntimeError("H57 amendment measured generation time differs from timing receipt")
    return cap, horizon, projected


def outcome_mode(args) -> dict:
    cap, horizon, projected = parse_amendment(args.amendment)
    if (RUN_ROOT / "outcomes/wave.json").exists():
        raise RuntimeError("H57 outcome receipt already exists; preserve it")
    if (RUN_ROOT / "outcomes").exists() and any((RUN_ROOT / "outcomes").iterdir()):
        raise RuntimeError("H57 outcome directory is not empty")
    wave_path = RUN_ROOT / "outcomes/wave.json"
    status = RUN_ROOT / "H57_STATUS.log"
    cells = []
    wait_total = 0.0
    gpu_wall = 0.0
    reason = None
    for arm_name, seed in CELL_ORDER:
        is_null = arm_name == "null_sigma001"
        arm = "sigma001" if is_null else arm_name
        out = RUN_ROOT / ("outcomes/null/seed15" if is_null else f"outcomes/cells/{arm}/seed{seed}")
        remaining = cap - gpu_wall
        if remaining <= 0:
            reason = "fixed H57 total GPU cell-wall cap reached before next cell"
            break
        command = child_command("outcome", out, arm=arm, seed=seed, generations=horizon,
                                null_arm=is_null)
        key = f"{arm_name}/seed{seed}"
        try:
            cell, waited, wall = run_locked_cell(
                key, command, RUN_ROOT / f"logs/{arm_name}-seed{seed}.log",
                timeout=remaining, status_path=status, wait_total=wait_total,
            )
            wait_total += waited
            gpu_wall += wall
            cells.append(cell)
            atomic_json(RUN_ROOT / "outcomes/wave.partial.json", {
                "schema": "jope.h57.outcome-wave/v1", "complete": False,
                "cells": cells, "lock_wait_seconds": wait_total,
                "total_gpu_cell_wall_seconds": gpu_wall,
                "total_gpu_cell_wall_cap_seconds": cap})
        except Exception as exc:
            wait_total += float(getattr(exc, "wait_seconds", 0.0))
            reason = f"infra-void/stopped at {key}: {type(exc).__name__}: {exc}"
            cells.append({"key": key, "status": "infra_void", "error": reason,
                          "lock_wait_seconds": float(getattr(exc, "wait_seconds", 0.0))})
            break
    complete = len(cells) == len(CELL_ORDER) and all(cell["status"] == "complete" for cell in cells)
    if complete:
        score_out = RUN_ROOT / "outcomes/fold-b-score"
        try:
            score_cell, waited, wall = run_locked_cell(
                "fold-b/admission", child_command("fold-b", score_out, run_root=RUN_ROOT),
                RUN_ROOT / "logs/fold-b.log", timeout=max(1.0, cap - gpu_wall),
                status_path=status, wait_total=wait_total,
            )
            wait_total += waited
            gpu_wall += wall
            cells.append(score_cell)
        except Exception as exc:
            wait_total += float(getattr(exc, "wait_seconds", 0.0))
            complete = False
            reason = f"infra-void/stopped at fold-B scoring: {type(exc).__name__}: {exc}"
            cells.append({"key": "fold-b/admission", "status": "infra_void", "error": reason,
                          "lock_wait_seconds": float(getattr(exc, "wait_seconds", 0.0))})
    wave = {"schema": "jope.h57.outcome-wave/v1", "complete": complete,
            "preregistration_commit": subprocess.check_output(
                ["git", "log", "-1", "--format=%H", "origin/main", "--",
                 "research/flywheel_selfplay/h57/PREREGISTRATION.md"],
                cwd=ROOT, text=True).strip(),
            "execution_amendment": str(args.amendment), "fixed_generation_horizon": horizon,
            "projected_total_seconds": projected, "cell_order": [{"arm": arm, "seed": seed} for arm, seed in CELL_ORDER],
            "cells": cells, "stop_reason": reason, "lock_wait_seconds": wait_total,
            "total_gpu_cell_wall_seconds": gpu_wall, "total_gpu_cell_wall_cap_seconds": cap,
            "candidate_and_null_cells": 7, "fold_b_candidate_reads": 6, "terminal_val_reads": 0}
    atomic_json(wave_path, wave)
    return wave


def terminal_mode() -> dict:
    wave_path = RUN_ROOT / "outcomes/wave.json"
    wave = json.loads(wave_path.read_text())
    if wave.get("complete") is not True:
        raise RuntimeError("H57 terminal validation requires a complete candidate/null/fold-B wave")
    admission = json.loads((RUN_ROOT / "outcomes/admission.json").read_text())
    arms = [arm for arm in ("sigma001", "sigma003")
            if admission.get("arms", {}).get(arm, {}).get("admitted") is True]
    planned = [(arm, seed) for arm in arms for seed in (15, 16, 17)]
    already = {(json.loads(path.read_text())["arm"], json.loads(path.read_text())["seed"])
               for path in (RUN_ROOT / "outcomes/terminal-val").glob("*/*.json")}
    reads = []
    wait_total = float(wave.get("lock_wait_seconds", 0.0))
    gpu_wall = float(wave.get("total_gpu_cell_wall_seconds", 0.0))
    cap = float(wave.get("total_gpu_cell_wall_cap_seconds", 0.0))
    for arm, seed in planned:
        if (arm, seed) in already:
            raise RuntimeError(f"refusing duplicate H57 terminal read {arm}/seed{seed}")
        remaining = cap - gpu_wall
        if remaining <= 0:
            raise RuntimeError("H57 terminal DL-Val budget exhausted under the frozen 90-minute cap")
        out = RUN_ROOT / f"outcomes/terminal-val/{arm}/seed{seed}.json"
        command = child_command("terminal-val", out, arm=arm, seed=seed, run_root=RUN_ROOT)
        try:
            cell, waited, wall = run_locked_cell(
                f"terminal-val/{arm}/seed{seed}", command,
                RUN_ROOT / f"logs/terminal-val-{arm}-seed{seed}.log", remaining,
                RUN_ROOT / "H57_STATUS.log", wait_total)
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
    summary = {"schema": "jope.h57.terminal-val-wave/v1", "reads": reads,
               "read_count": len(reads), "admitted_arms": arms,
               "lock_wait_seconds": wait_total}
    atomic_json(RUN_ROOT / "outcomes/terminal_val_wave.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--timing", action="store_true")
    modes.add_argument("--outcomes", action="store_true")
    modes.add_argument("--terminal-val", action="store_true")
    parser.add_argument("--amendment", type=Path)
    args = parser.parse_args()
    if args.amendment is None and args.outcomes:
        raise RuntimeError("H57 outcomes require the committed dated resource amendment")
    if disk_guard_error():
        raise RuntimeError(disk_guard_error())
    if args.timing:
        mode, result = "timing", timing_mode(args)
    elif args.outcomes:
        mode, result = "outcomes", outcome_mode(args)
    else:
        mode, result = "terminal-val", terminal_mode()
    print(json.dumps({"mode": mode, "result": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
