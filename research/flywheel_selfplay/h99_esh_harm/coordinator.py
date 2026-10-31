"""H99 FIFO/resource and per-attempt staging wrapper, tested without launching a worker."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from runner import H80_RUNNER, execution_plan

CAP_SECONDS = 9_000.0
MIN_MEM_AVAILABLE_KIB = 16 * 1024 * 1024
MAX_LOAD = 85.0
MAX_DISK_USED_PERCENT = 94.5
POLL_SECONDS = 30.0


def load_h80_wave():
    path = H80_RUNNER.parent / "run_wave.py"
    spec = importlib.util.spec_from_file_location("h99_shared_h80_wave", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared H80 FIFO coordinator at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_resource_snapshot(*, load1: float, mem_available_kib: int, disk_used_percent: float) -> None:
    if load1 > MAX_LOAD:
        raise RuntimeError(f"host load {load1:.2f}/102 exceeds {MAX_LOAD:.0f}/102")
    if mem_available_kib < MIN_MEM_AVAILABLE_KIB:
        raise RuntimeError("MemAvailable is below 16 GiB")
    if disk_used_percent >= MAX_DISK_USED_PERCENT:
        raise RuntimeError("disk usage is at or above the 94.5% launch ceiling")


def wait_for_cpu_gate(*, load_fn=os.getloadavg, sleep_fn=time.sleep, max_wait_seconds=10_800) -> float:
    """Require two load checks <=85/102, 30 seconds apart, before creating a FIFO ticket."""
    started = time.monotonic()
    stable = 0
    while stable < 2:
        load1 = float(load_fn()[0])
        stable = stable + 1 if load1 <= MAX_LOAD else 0
        if stable < 2:
            if time.monotonic() - started + POLL_SECONDS > max_wait_seconds:
                raise RuntimeError("H99 CPU gate did not stabilize within three hours")
            sleep_fn(POLL_SECONDS)
    return time.monotonic() - started


def launch_preflight(*, wave=None) -> dict:
    """Apply the shared H80 disk/CPU/memory/GPU preflight plus the H99 load ceiling."""
    helper = wave or load_h80_wave()
    snapshot = helper.safe_preflight()
    load1 = float(os.getloadavg()[0])
    check_resource_snapshot(
        load1=load1,
        mem_available_kib=int(snapshot["mem_available_kib"]),
        disk_used_percent=float(snapshot["disk_used_percent"]),
    )
    return {**snapshot, "load1": load1, "wave": "H99"}


def acquire_h99_fifo(status_path: Path, *, wait_total: float = 0.0, wave=None):
    """Queue at the shared FIFO tail and acquire the qualification lock as H99."""
    helper = wave or load_h80_wave()
    return helper.acquire_lock(
        status_path,
        wait_total,
        after_release=True,
        wave_name="H99",
        preflight_fn=launch_preflight,
    )


def release_lock(handle, *, flock_fn=fcntl.flock) -> None:
    try:
        flock_fn(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def apply_worker_limits() -> None:
    """Set the authorized worker thread caps and raise niceness to 19."""
    os.environ.update(
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    current_nice = os.nice(0)
    if current_nice < 19:
        os.nice(19 - current_nice)


def write_budget(path: Path, budget: dict) -> None:
    """Atomically persist cumulative wall charges before another attempt can start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".budget-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(budget, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def attempt_scratch_path(run_root: Path, arm: str, seed: int, attempt: int) -> Path:
    plan = execution_plan()
    allowed = {cell["arm"] for cell in plan["candidate_cells"]} | {"NULL"}
    if arm not in allowed or seed not in (*plan["paired_seeds"], plan["null_cell"]["seed"]):
        raise ValueError("unregistered H99 cell")
    if attempt < 1:
        raise ValueError("H99 attempt numbers start at one")
    return Path(run_root) / "scratch" / f"{arm}-seed{seed}-attempt{attempt}"


def stage_r882_copy(source: Path, scratch_dir: Path, *, expected_sha256: str | None = None) -> Path:
    """Copy a pinned read-only source into a fresh writable per-attempt scratch directory."""
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("r882 source must be a regular, non-symlink file")
    h80 = load_h80_wave().runner_mod
    expected = expected_sha256 or h80.R882_SHA
    if h80.sha256(source) != expected:
        raise RuntimeError("r882 source SHA256 does not match the H80 pin")
    scratch_dir = Path(scratch_dir)
    if scratch_dir.exists() or scratch_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse H99 attempt scratch: {scratch_dir}")
    scratch_dir.mkdir(parents=True, exist_ok=False)
    destination = scratch_dir / "r882-copy.pt"
    try:
        # copyfile opens a new destination with the process umask; it does not inherit
        # a read-only source mode or alter the source inode/mode.
        shutil.copyfile(source, destination)
        if h80.sha256(destination) != expected:
            raise RuntimeError("staged r882 copy failed its SHA256 check")
        return destination
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def run_fifo_cells(
    cells: list[dict],
    *,
    run_root: Path,
    checkpoint: Path,
    worker,
    acquire_fn=acquire_h99_fifo,
    release_fn=release_lock,
    cpu_gate_fn=wait_for_cpu_gate,
    setup_fn=apply_worker_limits,
    clock_fn=time.monotonic,
) -> dict:
    """Run a caller-supplied H99 cell worker one cell per FIFO lease, charging all wall time."""
    expected = execution_plan()
    expected_cells = [*expected["candidate_cells"], expected["null_cell"]]
    if cells != expected_cells:
        raise ValueError("H99 wave cell list differs from the frozen seven-cell plan")
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    budget_path = run_root / "budget.json"
    if budget_path.exists():
        budget = json.loads(budget_path.read_text())
        if budget.get("in_progress") is not None:
            raise RuntimeError("H99 has an unaccounted in-progress cell; refusing budget reset")
    else:
        budget = {"schema": "jope.h99.gpu-budget/v1", "cap_seconds": CAP_SECONDS, "attempts": [], "in_progress": None}
    if budget.get("cap_seconds") != CAP_SECONDS:
        raise RuntimeError("H99 cumulative GPU cap changed")
    if budget["attempts"]:
        raise RuntimeError("H99 budget already has attempts; refusing to repeat or reset cells")
    total_wait = 0.0
    setup_fn()

    for cell in cells:
        spent = sum(float(item["wall_seconds"]) for item in budget["attempts"])
        remaining = CAP_SECONDS - spent
        if remaining <= 10.0:
            raise RuntimeError("H99 cumulative 9,000-second GPU cell-wall cap exhausted")
        cpu_gate_fn()
        handle, waited, gpu_free, preflight = acquire_fn(run_root / "H99_STATUS.log", wait_total=total_wait)
        total_wait += float(waited)
        scratch = attempt_scratch_path(run_root, cell["arm"], cell["seed"], 1)
        started = clock_fn()
        try:
            staged = stage_r882_copy(checkpoint, scratch)
            budget["in_progress"] = {**cell, "scratch": str(scratch), "started_utc": datetime.now(timezone.utc).isoformat()}
            write_budget(budget_path, budget)
            result = worker(cell, staged, scratch, remaining)
            wall = clock_fn() - started
            if wall > remaining:
                raise RuntimeError("H99 cell exceeded the remaining cumulative GPU budget")
            budget["attempts"].append(
                {**cell, "scratch": str(scratch), "wait_seconds": float(waited), "gpu_free_mib": int(gpu_free),
                 "preflight": preflight, "wall_seconds": wall, "status": "complete", "result": result}
            )
            budget["in_progress"] = None
            write_budget(budget_path, budget)
        except BaseException as exc:
            wall = clock_fn() - started
            budget["attempts"].append(
                {**cell, "scratch": str(scratch), "wait_seconds": float(waited), "wall_seconds": wall,
                 "status": "void-or-unknown", "error": f"{type(exc).__name__}: {exc}"}
            )
            budget["in_progress"] = None
            write_budget(budget_path, budget)
            raise
        finally:
            release_fn(handle)
    return budget
