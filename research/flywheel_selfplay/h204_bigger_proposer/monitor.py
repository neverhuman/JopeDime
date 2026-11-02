"""H204 Bigger Proposer: Resource Monitor & Stage-1 Launcher on `xbabe2`.

Enforces:
  - Host == `xbabe2` (never `xbabe0`, never live flywheel PID on `xbabe1`)
  - `nice == 19`
  - Thread rule (b) via `#1730` `capture_numeric_threads` + `admit_numeric_threads`:
    `OMP`/`MKL`/`OpenBLAS`/`torch` intra-op `<= 4`; CUDA runtime/driver threads diagnostic
  - Host CPU load1 `<= 85%` of cores
  - Campaign budgets: `<= 30` GPU-h, `<= 200` CPU core-h, `<= 12` wall-h
  - Process RSS `<= 6 GiB`
  - Local cache `<= 20 GiB`
  - Disk used ratio `<= 0.938` and free space `>= 40 GiB`
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch

from research.flywheel_selfplay.h204_bigger_proposer.analysis import (
    AUTHORITY_STRING,
    EVALUATOR_CONTRACT_ID,
    GPU_DEVICE,
    HOST,
    MAX_CPU_CORE_HOURS,
    MAX_CPU_UTIL_PCT,
    MAX_DISK_USED_RATIO,
    MAX_GPU_HOURS,
    MAX_LOCAL_CACHE_BYTES,
    MAX_NUMERIC_THREADS,
    MAX_RSS_BYTES,
    MAX_WALL_HOURS,
    MIN_DISK_FREE_BYTES,
    OWNER,
    REQUIRED_NICE,
    S3_EVIDENCE_PREFIX,
    SEEDS,
    STAGE1_ARMS,
    STAGE1_WIDTH,
    WAVE,
    WIDTHS,
)
from research.flywheel_selfplay.h204_bigger_proposer.labeler import check_disk_budget
from research.flywheel_selfplay.h204_bigger_proposer.trainer import (
    enforce_numeric_thread_limits,
    read_proc_os_threads,
    run_h204_stage1,
)
from research.flywheel_selfplay.hwave_numeric_threads import (
    admit_numeric_threads,
    capture_numeric_threads,
)

NUMERIC_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
)

_CACHED_WORKER_REPORT: dict[str, Any] | None = None


def verify_numeric_thread_rule_b(
    max_threads: int = MAX_NUMERIC_THREADS,
    *,
    refresh_native_pools: bool | None = None,
) -> dict[str, Any]:
    """Verify thread rule (b) via `#1730` `capture_numeric_threads` + `admit_numeric_threads`."""
    global _CACHED_WORKER_REPORT
    env_vals: dict[str, int] = {}
    violations: list[str] = []
    for key in NUMERIC_ENV_KEYS:
        raw = os.environ.get(key, "").strip()
        if not raw:
            violations.append(f"{key}=unset")
            continue
        try:
            val = int(raw)
        except ValueError:
            violations.append(f"{key}={raw!r}")
            continue
        env_vals[key] = val
        if val < 1 or val > max_threads:
            violations.append(f"{key}={val}>{max_threads}")

    torch_threads = int(torch.get_num_threads())
    if torch_threads < 1 or torch_threads > max_threads:
        violations.append(f"torch.get_num_threads()={torch_threads}>{max_threads}")

    if refresh_native_pools is None:
        refresh_native_pools = threading.current_thread() is threading.main_thread()

    admission: dict[str, Any] | None = None
    try:
        if refresh_native_pools or _CACHED_WORKER_REPORT is None:
            worker_report = capture_numeric_threads(torch)
            _CACHED_WORKER_REPORT = dict(worker_report)
        else:
            worker_report = {
                "pid": os.getpid(),
                "environment": {
                    k: os.environ.get(k)
                    for k in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                    )
                },
                "torch_intraop": torch_threads,
                "native_pools": list(_CACHED_WORKER_REPORT.get("native_pools", [])),
            }
        os_threads = read_proc_os_threads()
        admission = admit_numeric_threads([worker_report], os_threads=os_threads)
        if int(admission["numeric_threads_upper_bound"]) > max_threads:
            violations.append(
                f"numeric_threads_upper_bound={admission['numeric_threads_upper_bound']}>{max_threads}"
            )
    except Exception as exc:
        violations.append(f"admit_numeric_threads:{exc}")

    return {
        "ok": len(violations) == 0,
        "max_numeric_threads": int(max_threads),
        "torch_num_threads": torch_threads,
        "env_threads": env_vals,
        "admission": admission,
        "cuda_runtime_threads_exempt": True,
        "violations": violations,
    }


def ensure_nice_19(required_nice: int = REQUIRED_NICE) -> int:
    """Ensure current process runs at nice >= 19."""
    current = os.nice(0)
    if current < required_nice:
        current = os.nice(required_nice - current)
    return int(current)


def read_proc_rss_bytes(pid: int | None = None) -> int:
    """Read resident set size (VmRSS) in bytes from `/proc/<pid>/status`."""
    target_pid = pid if pid is not None else os.getpid()
    status_path = Path(f"/proc/{target_pid}/status")
    if not status_path.is_file():
        return 0
    for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return 0


def query_nvidia_smi_line() -> str:
    """Capture a one-line `nvidia-smi` summary for `cuda:0` on `xbabe2`."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5.0,
        ).strip()
        return out.splitlines()[0] if out else "nvidia-smi: empty"
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def collect_resource_snapshot(
    run_dir: Path,
    *,
    started_monotonic: float | None = None,
) -> dict[str, Any]:
    """Collect a single resource & guardrail snapshot for `resource_monitor.jsonl`."""
    rss_bytes = read_proc_rss_bytes()
    disk = check_disk_budget(run_dir)
    cache_dir = run_dir / "cache"
    cache_bytes = 0
    if cache_dir.is_dir():
        for p in cache_dir.rglob("*"):
            if p.is_file():
                try:
                    cache_bytes += p.stat().st_size
                except OSError:
                    continue

    thread_check = verify_numeric_thread_rule_b(MAX_NUMERIC_THREADS)
    load1, load5, load15 = os.getloadavg()
    n_cpus = os.cpu_count() or 128
    max_load1 = (MAX_CPU_UTIL_PCT / 100.0) * float(n_cpus)
    cpu_ok = bool(min(load1, load5, load15) <= max_load1)

    elapsed_hours = (
        max(0.0, (time.monotonic() - started_monotonic) / 3600.0)
        if started_monotonic is not None
        else 0.0
    )
    gpu_hours = elapsed_hours
    cpu_core_hours = elapsed_hours * float(MAX_NUMERIC_THREADS)
    budget_ok = bool(
        elapsed_hours <= MAX_WALL_HOURS
        and gpu_hours <= MAX_GPU_HOURS
        and cpu_core_hours <= MAX_CPU_CORE_HOURS
    )

    ok = bool(
        rss_bytes <= MAX_RSS_BYTES
        and cache_bytes <= MAX_LOCAL_CACHE_BYTES
        and disk["ok"]
        and thread_check["ok"]
        and cpu_ok
        and budget_ok
    )
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "nice": os.nice(0),
        "rss_bytes": int(rss_bytes),
        "max_rss_bytes": int(MAX_RSS_BYTES),
        "cache_bytes": int(cache_bytes),
        "max_cache_bytes": int(MAX_LOCAL_CACHE_BYTES),
        "disk": disk,
        "thread_rule_b": thread_check,
        "loadavg": {
            "load1": load1,
            "load5": load5,
            "load15": load15,
            "n_cpus": n_cpus,
            "max_load1": max_load1,
            "ok": cpu_ok,
        },
        "budgets": {
            "elapsed_wall_hours": elapsed_hours,
            "max_wall_hours": MAX_WALL_HOURS,
            "gpu_hours": gpu_hours,
            "max_gpu_hours": MAX_GPU_HOURS,
            "cpu_core_hours": cpu_core_hours,
            "max_cpu_core_hours": MAX_CPU_CORE_HOURS,
            "ok": budget_ok,
        },
        "nvidia_smi": query_nvidia_smi_line(),
        "ok": ok,
    }


def build_launch_manifest(
    *,
    run_dir: Path,
    prereg_pr: int | str,
    source_commit: str,
    device_str: str = GPU_DEVICE,
    k_candidates: int = 16,
    epochs: int = 30,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Create and write `launch.json` containing the verbatim owner authority string."""
    pr_str = str(prereg_pr).strip()
    if not pr_str or pr_str.lower() == "pending" or not pr_str.isdigit():
        raise ValueError(
            f"prereg_pr must be a numeric PR id (not 'pending'), got {prereg_pr!r}"
        )
    commit_str = str(source_commit).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit_str):
        raise ValueError(
            f"source_commit must be a concrete git SHA (not 'HEAD'), got {source_commit!r}"
        )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        os.environ.get("JOPE_RESEARCH_RUN_ID", "").strip() or f"h204-{run_dir.name}"
    )
    os.environ["JOPE_RESEARCH_RUN_ID"] = run_id
    os.environ["JOPE_RESEARCH_DOPE_UNFREEZE"] = "0"

    enforce_numeric_thread_limits(MAX_NUMERIC_THREADS)
    nice_val = ensure_nice_19(REQUIRED_NICE)
    snap = collect_resource_snapshot(run_dir)

    manifest = {
        "wave": WAVE,
        "title": "H204 Bigger Proposer (widths 128 -> 256 -> 512) with Research-Only DOPE Unfreeze",
        "owner": OWNER,
        "host": socket.gethostname(),
        "expected_host": HOST,
        "authority": AUTHORITY_STRING,
        "prereg_pr": int(pr_str),
        "source_commit": commit_str,
        "launched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "nice": nice_val,
        "device": device_str,
        "nvidia_smi_at_launch": snap["nvidia_smi"],
        "evaluator_contract_id": EVALUATOR_CONTRACT_ID,
        "stage": "Stage 1 (labeling + width 128 only)",
        "stage1_width": STAGE1_WIDTH,
        "planned_widths": list(WIDTHS),
        "stage1_arms": list(STAGE1_ARMS),
        "seeds": list(SEEDS),
        "k_candidates_per_context": int(k_candidates),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "research_dope_unfreeze": True,
        "research_run_id": run_id,
        "dope_policy": {
            "research_dope_unfreeze": True,
            "research_run_id": run_id,
            "entrypoint": "research",
            "dope_eval_mode": True,
            "serving_eligible": False,
        },
        "dope_unfreeze_env": {
            "JOPE_RESEARCH_DOPE_UNFREEZE": "1 (scoped to PROPOSER_W* configure_research_dope)",
            "JOPE_RESEARCH_RUN_ID": run_id,
        },
        "resource_guards": {
            "max_numeric_threads": MAX_NUMERIC_THREADS,
            "cuda_driver_threads_exempt": True,
            "max_cpu_util_pct": MAX_CPU_UTIL_PCT,
            "max_rss_bytes": MAX_RSS_BYTES,
            "max_local_cache_bytes": MAX_LOCAL_CACHE_BYTES,
            "max_disk_used_ratio": MAX_DISK_USED_RATIO,
            "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
            "max_gpu_hours": MAX_GPU_HOURS,
            "max_cpu_core_hours": MAX_CPU_CORE_HOURS,
            "max_wall_hours": MAX_WALL_HOURS,
        },
        "initial_resource_snapshot": snap,
        "s3_evidence_prefix": S3_EVIDENCE_PREFIX,
    }
    launch_path = run_dir / "launch.json"
    launch_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="H204 Stage-1 Monitored GPU Runner")
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Output run directory"
    )
    parser.add_argument(
        "--prereg-pr",
        type=str,
        required=True,
        help="Preregistration PR number (numeric)",
    )
    parser.add_argument(
        "--source-commit", type=str, required=True, help="Concrete git commit SHA"
    )
    parser.add_argument(
        "--device", type=str, default=GPU_DEVICE, help="CUDA device (default cuda:0)"
    )
    parser.add_argument(
        "--k-candidates", type=int, default=16, help="Candidates per context (8..16)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Training epochs per fold"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--allow-stage2",
        action="store_true",
        help="Run widths 256/512 if W128 passes gate",
    )
    args = parser.parse_args()

    hostname = socket.gethostname()
    if hostname == "xbabe0":
        raise RuntimeError("Forbidden host xbabe0!")
    if hostname != HOST:
        raise RuntimeError(f"Expected host {HOST!r}, got {hostname!r}")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested CUDA device {args.device!r} on {hostname}, but CUDA is unavailable!"
            )
        torch.cuda.init()
        _ = torch.zeros(1, device=torch.device(args.device))

    enforce_numeric_thread_limits(MAX_NUMERIC_THREADS)
    ensure_nice_19(REQUIRED_NICE)

    launch_manifest = build_launch_manifest(
        run_dir=args.run_dir,
        prereg_pr=args.prereg_pr,
        source_commit=args.source_commit,
        device_str=args.device,
        k_candidates=args.k_candidates,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    started_monotonic = time.monotonic()
    stop_event = threading.Event()
    monitor_log_path = args.run_dir / "resource_monitor.jsonl"
    init_snap = launch_manifest["initial_resource_snapshot"]
    with monitor_log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(init_snap, sort_keys=True) + "\n")

    def _monitor_loop() -> None:
        while not stop_event.wait(15.0):
            snap = collect_resource_snapshot(
                args.run_dir, started_monotonic=started_monotonic
            )
            with monitor_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(snap, sort_keys=True) + "\n")
            if not snap["ok"]:
                (args.run_dir / "GUARD_STOP.json").write_text(
                    json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8"
                )
                os._exit(2)

    mon_thread = threading.Thread(target=_monitor_loop, daemon=True)
    mon_thread.start()
    try:
        run_h204_stage1(
            run_dir=args.run_dir,
            device_str=args.device,
            k_candidates=args.k_candidates,
            epochs=args.epochs,
            batch_size=args.batch_size,
            allow_stage2=bool(args.allow_stage2),
        )
    finally:
        stop_event.set()
        mon_thread.join(timeout=5.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
