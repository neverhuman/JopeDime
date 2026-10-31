"""H99 FIFO/resource and per-attempt staging wrapper, tested without launching a worker."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile
import time

from runner import H80_RUNNER, execution_plan

CAP_SECONDS = 9_000.0
MIN_MEM_AVAILABLE_KIB = 16 * 1024 * 1024
MAX_LOAD = 85.0
# Resume launches require extra disk headroom; executor.py retains its 94.5% in-cell abort.
MAX_LAUNCH_DISK_USED_PERCENT = 94.0
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


class H99Lease:
    """Keep the owner/ETA queue marker visible for the lifetime of one lock lease."""

    def __init__(self, handle, marker: Path):
        self.handle = handle
        self.marker = marker


def check_resource_snapshot(*, load1: float, mem_available_kib: int, disk_used_percent: float) -> None:
    if load1 > MAX_LOAD:
        raise RuntimeError(f"host load {load1:.2f}/102 exceeds {MAX_LOAD:.0f}/102")
    if mem_available_kib < MIN_MEM_AVAILABLE_KIB:
        raise RuntimeError("MemAvailable is below 16 GiB")
    if disk_used_percent >= MAX_LAUNCH_DISK_USED_PERCENT:
        raise RuntimeError("disk usage is not below the strict 94.0% launch ceiling")


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


def acquire_h99_fifo(
    status_path: Path,
    *,
    wait_total: float = 0.0,
    wave=None,
    owner: str = "luna_v1_res03",
    eta_utc: str = "unknown",
    queue_dir: Path | None = None,
):
    """Queue at the shared FIFO tail and acquire the qualification lock as H99."""
    helper = wave or load_h80_wave()
    queue_dir = Path(queue_dir or helper.fifo_lock.QUEUE_DIR)
    queue_dir.mkdir(parents=True, exist_ok=True)
    pid_is_alive = getattr(getattr(helper, "fifo_lock", None), "pid_is_alive", lambda _pid: True)
    for stale in queue_dir.glob("H99-owner-*.json"):
        try:
            prior = json.loads(stale.read_text())
            if isinstance(prior.get("pid"), int) and not pid_is_alive(prior["pid"]):
                stale.unlink(missing_ok=True)
        except (OSError, ValueError, TypeError):
            continue
    marker = queue_dir / f"H99-owner-{os.getpid()}.json"
    payload = {
        "schema": "jope.h99.fifo-owner/v1", "wave": "H99", "owner": owner,
        "eta_utc": eta_utc, "pid": os.getpid(), "status_path": str(status_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    acquired = False
    try:
        result = helper.acquire_lock(
            status_path,
            wait_total,
            after_release=True,
            wave_name="H99",
            preflight_fn=launch_preflight,
        )
        acquired = True
        return H99Lease(result[0], marker), *result[1:]
    finally:
        if not acquired:
            marker.unlink(missing_ok=True)


def release_lock(handle, *, flock_fn=fcntl.flock) -> None:
    marker = getattr(handle, "marker", None)
    raw_handle = getattr(handle, "handle", handle)
    try:
        flock_fn(raw_handle, fcntl.LOCK_UN)
    finally:
        raw_handle.close()
        if marker is not None:
            marker.unlink(missing_ok=True)


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


@contextmanager
def _wall_deadline(seconds: float):
    """Interrupt a cell/scoring action at the remaining cumulative wall budget."""
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        raise RuntimeError("H99 requires a positive remaining wall budget and interval timer support")
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise RuntimeError("refusing to replace an existing process wall timer")
    previous = signal.getsignal(signal.SIGALRM)

    def timeout(_signum, _frame):
        raise TimeoutError("H99 cumulative 9,000-second GPU cell-wall cap reached")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def attempt_scratch_path(run_root: Path, arm: str, seed: int, attempt: int) -> Path:
    plan = execution_plan()
    allowed = {cell["arm"] for cell in plan["candidate_cells"]} | {"NULL"}
    if arm not in allowed or seed not in (*plan["paired_seeds"], plan["null_cell"]["seed"]):
        raise ValueError("unregistered H99 cell")
    if attempt < 1:
        raise ValueError("H99 attempt numbers start at one")
    return Path(run_root) / "scratch" / f"{arm}-seed{seed}-attempt{attempt}"


def _attempt_from_record(record: dict) -> int:
    attempt = record.get("attempt")
    if attempt is None:
        match = re.search(r"-attempt([1-9][0-9]*)$", Path(record.get("scratch", "")).name)
        if match is None:
            raise RuntimeError("H99 budget attempt has no valid attempt number")
        attempt = int(match.group(1))
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise RuntimeError("H99 budget attempt number is invalid")
    return attempt


def _cell_histories(budget: dict, cells: list[dict]) -> tuple[dict[tuple[str, int], list[dict]], list[dict]]:
    """Validate and index append-only cell/scoring attempts from a cumulative budget."""
    if budget.get("schema") != "jope.h99.gpu-budget/v1" or budget.get("cap_seconds") != CAP_SECONDS:
        raise RuntimeError("H99 cumulative budget schema or cap changed")
    if budget.get("in_progress") is not None:
        raise RuntimeError("H99 has an unaccounted in-progress cell; refusing retry selection")
    expected = {(cell["arm"], int(cell["seed"])): cell for cell in cells}
    histories: dict[tuple[str, int], list[dict]] = {key: [] for key in expected}
    scoring = []
    attempts = budget.get("attempts")
    if not isinstance(attempts, list):
        raise RuntimeError("H99 cumulative budget attempts must be a list")
    for record in attempts:
        if not isinstance(record, dict):
            raise RuntimeError("H99 cumulative budget contains a malformed attempt")
        wall = record.get("wall_seconds")
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(float(wall)) or wall < 0:
            raise RuntimeError("H99 cumulative budget contains an invalid wall charge")
        wait = record.get("wait_seconds", 0.0)
        if isinstance(wait, bool) or not isinstance(wait, (int, float)) or not math.isfinite(float(wait)) or wait < 0:
            raise RuntimeError("H99 cumulative budget contains an invalid FIFO wait charge")
        if record.get("action") == "fold_b_scoring":
            if record.get("status") not in {"complete", "void-or-unknown"}:
                raise RuntimeError("H99 Fold-B scoring ledger status is invalid")
            scoring.append(record)
            continue
        key = (record.get("arm"), record.get("seed"))
        if key not in expected:
            raise RuntimeError(f"H99 budget contains an unregistered cell: {key}")
        cell = expected[key]
        if any(record.get(name) != value for name, value in cell.items()):
            raise RuntimeError(f"H99 budget cell differs from the frozen plan: {key}")
        if record.get("status") not in {"complete", "void-or-unknown"}:
            raise RuntimeError(f"H99 budget cell status is invalid: {key}")
        attempt = _attempt_from_record(record)
        if Path(record.get("scratch", "")).name != attempt_scratch_path(
            Path("scratch"), cell["arm"], cell["seed"], attempt
        ).name:
            raise RuntimeError(f"H99 budget scratch path does not match its attempt: {key}")
        history = histories[key]
        if history and attempt != _attempt_from_record(history[-1]) + 1:
            raise RuntimeError(f"H99 attempts are not consecutive for {key}")
        if history and history[-1]["status"] == "complete":
            raise RuntimeError(f"H99 cell was retried after completion: {key}")
        if record.get("attempt") is not None and record["attempt"] != attempt:
            raise RuntimeError(f"H99 attempt number changed while validating {key}")
        history.append(record)
    if len(scoring) > 1:
        raise RuntimeError("H99 Fold-B scoring is already recorded; refusing duplicate scoring")
    if scoring and any(not history or history[-1]["status"] != "complete" for history in histories.values()):
        raise RuntimeError("H99 Fold-B scoring was recorded before every fixed cell completed")
    spent = sum(float(item["wall_seconds"]) for item in attempts)
    if spent > CAP_SECONDS:
        raise RuntimeError("H99 cumulative 9,000-second GPU cell-wall cap is already exhausted")
    return histories, scoring


def completed_attempt(run_root: Path, arm: str, seed: int) -> int:
    """Return the sole successful attempt for a cell from its durable cumulative ledger."""
    plan = execution_plan()
    cells = [*plan["candidate_cells"], plan["null_cell"]]
    budget = json.loads((Path(run_root) / "budget.json").read_text())
    histories, _ = _cell_histories(budget, cells)
    history = histories.get((arm, int(seed)))
    if not history or history[-1]["status"] != "complete":
        raise RuntimeError(f"H99 cell has no completed attempt: {arm}/seed{seed}")
    return _attempt_from_record(history[-1])


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
    finalize_worker=None,
) -> dict:
    """Resume incomplete cells in plan order, charging each attempt under its own FIFO lease."""
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
    histories, scoring_attempts = _cell_histories(budget, expected_cells)
    if scoring_attempts and scoring_attempts[0]["status"] == "void-or-unknown":
        raise RuntimeError("H99 Fold-B scoring is void-or-unknown; refusing a duplicate protected-fold read")
    total_wait = sum(float(item.get("wait_seconds", 0.0)) for item in budget["attempts"])
    setup_fn()

    for cell in cells:
        key = (cell["arm"], int(cell["seed"]))
        history = histories[key]
        if history and history[-1]["status"] == "complete":
            continue
        attempt = _attempt_from_record(history[-1]) + 1 if history else 1
        spent = sum(float(item["wall_seconds"]) for item in budget["attempts"])
        remaining = CAP_SECONDS - spent
        if remaining <= 10.0:
            raise RuntimeError("H99 cumulative 9,000-second GPU cell-wall cap exhausted")
        cpu_gate_fn()
        handle, waited, gpu_free, preflight = acquire_fn(run_root / "H99_STATUS.log", wait_total=total_wait)
        total_wait += float(waited)
        scratch = attempt_scratch_path(run_root, cell["arm"], cell["seed"], attempt)
        started = clock_fn()
        try:
            with _wall_deadline(remaining):
                staged = stage_r882_copy(checkpoint, scratch)
                budget["in_progress"] = {**cell, "attempt": attempt, "scratch": str(scratch), "started_utc": datetime.now(timezone.utc).isoformat()}
                write_budget(budget_path, budget)
                result = worker(cell, staged, scratch, remaining, attempt)
            wall = clock_fn() - started
            if wall > remaining:
                raise RuntimeError("H99 cell exceeded the remaining cumulative GPU budget")
            budget["attempts"].append(
                {**cell, "attempt": attempt, "scratch": str(scratch), "wait_seconds": float(waited), "gpu_free_mib": int(gpu_free),
                 "preflight": preflight, "wall_seconds": wall, "status": "complete", "result": result}
            )
            budget["in_progress"] = None
            write_budget(budget_path, budget)
            history.append(budget["attempts"][-1])
        except BaseException as exc:
            wall = clock_fn() - started
            budget["attempts"].append(
                {**cell, "attempt": attempt, "scratch": str(scratch), "wait_seconds": float(waited), "wall_seconds": wall,
                 "status": "void-or-unknown", "error": f"{type(exc).__name__}: {exc}"}
            )
            budget["in_progress"] = None
            write_budget(budget_path, budget)
            history.append(budget["attempts"][-1])
            raise
        finally:
            release_fn(handle)
    if finalize_worker is not None:
        histories, scoring_attempts = _cell_histories(budget, expected_cells)
        if any(not history or history[-1]["status"] != "complete" for history in histories.values()):
            raise RuntimeError("H99 cannot score Fold B until all seven fixed cells are complete")
        if scoring_attempts:
            if scoring_attempts[0]["status"] == "complete":
                return budget
            raise RuntimeError("H99 Fold-B scoring is void-or-unknown; refusing a duplicate protected-fold read")
        spent = sum(float(item["wall_seconds"]) for item in budget["attempts"])
        remaining = CAP_SECONDS - spent
        if remaining <= 10.0:
            raise RuntimeError("H99 cumulative 9,000-second GPU cell-wall cap exhausted before Fold-B scoring")
        cpu_gate_fn()
        handle, waited, gpu_free, preflight = acquire_fn(run_root / "H99_STATUS.log", wait_total=total_wait)
        total_wait += float(waited)
        score_attempt = 1
        score_scratch = run_root / "scratch" / f"fold-b-scoring-attempt{score_attempt}"
        started = clock_fn()
        action = {"action": "fold_b_scoring", "attempt": score_attempt, "scratch": str(score_scratch), "started_utc": datetime.now(timezone.utc).isoformat()}
        try:
            with _wall_deadline(remaining):
                staged = stage_r882_copy(checkpoint, score_scratch)
                budget["in_progress"] = action
                write_budget(budget_path, budget)
                result = finalize_worker(staged, score_scratch, remaining, score_attempt)
            wall = clock_fn() - started
            if wall > remaining:
                raise RuntimeError("H99 Fold-B scoring exceeded remaining cumulative GPU budget")
            budget["attempts"].append({**action, "wait_seconds": float(waited), "gpu_free_mib": int(gpu_free), "preflight": preflight, "wall_seconds": wall, "status": "complete", "result": result})
            budget["in_progress"] = None
            write_budget(budget_path, budget)
        except BaseException as exc:
            wall = clock_fn() - started
            budget["attempts"].append({**action, "wait_seconds": float(waited), "wall_seconds": wall, "status": "void-or-unknown", "error": f"{type(exc).__name__}: {exc}"})
            budget["in_progress"] = None
            write_budget(budget_path, budget)
            raise
        finally:
            release_fn(handle)
    return budget
