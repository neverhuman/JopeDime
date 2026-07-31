#!/usr/bin/env python3
"""Pressure-aware node supervisor for isolated Golden46 registered predictions."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil


CAMPAIGN_ID = "golden46-full42-20260723-v1"
QUANT_ROOT = Path(os.environ.get("GOLDEN46_QUANT_ROOT", "/home/ubuntu/remote_super/quant"))
DEFAULT_CAMPAIGN_ROOT = Path(
    os.environ.get(
        "GOLDEN46_CAMPAIGN_ROOT",
        str(QUANT_ROOT / "campaigns" / CAMPAIGN_ID),
    )
)
DEFAULT_COHORT = DEFAULT_CAMPAIGN_ROOT / "manifests" / "cohort.json"
EXECUTION_SCRIPT = Path("/home/ubuntu/jope-dime/scripts/golden46_execution.py")
PYTHON = QUANT_ROOT / ".venv-model-queue" / "bin" / "python"
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "LIGHTGBM_NUM_THREADS": "1",
    "XGBOOST_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
INITIAL_TIMEOUT_SECONDS = 2 * 60 * 60
RETRY_TIMEOUT_SECONDS = 8 * 60 * 60


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def import_specs() -> dict[str, Any]:
    if str(QUANT_ROOT) not in sys.path:
        sys.path.insert(0, str(QUANT_ROOT))
    from models.tools.model_zoo import specs_by_name

    return specs_by_name("regression")


def pressure_snapshot(reserve_gib: float, prior_swap: tuple[int, int]) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_io_counters()
    swap_counts = (
        int(getattr(swap, "sin", 0)),
        int(getattr(swap, "sout", 0)),
    )
    swap_active = swap_counts[0] > prior_swap[0] or swap_counts[1] > prior_swap[1]
    io_wait = 0.0
    pressure_path = Path("/proc/pressure/io")
    if pressure_path.is_file():
        try:
            first = pressure_path.read_text().splitlines()[0].split()
            io_wait = float(dict(item.split("=") for item in first[1:])["avg10"])
        except Exception:
            io_wait = 0.0
    available_gib = memory.available / 1024**3
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_available_gib": available_gib,
        "memory_percent": memory.percent,
        "swap_used_gib": swap.used / 1024**3,
        "swap_in": swap_counts[0],
        "swap_out": swap_counts[1],
        "swap_active": swap_active,
        "io_pressure_avg10": io_wait,
        "disk_read_bytes": int(getattr(disk, "read_bytes", 0)) if disk else 0,
        "disk_write_bytes": int(getattr(disk, "write_bytes", 0)) if disk else 0,
        "launch_paused": available_gib < reserve_gib or swap_active or io_wait >= 25.0,
    }


@dataclass
class Running:
    method_id: str
    process: subprocess.Popen[str]
    started: float
    log_path: Path


def process_rss(process: subprocess.Popen[str]) -> int:
    try:
        parent = psutil.Process(process.pid)
        return parent.memory_info().rss + sum(
            child.memory_info().rss
            for child in parent.children(recursive=True)
            if child.is_running()
        )
    except (psutil.Error, OSError):
        return 0


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()
        process.wait()


def valid_prediction_receipt(artifact_root: Path, dataset_id: str, method_id: str) -> bool:
    path = artifact_root / dataset_id / method_id / "worker-receipt.json"
    if not path.is_file():
        return False
    try:
        receipt = read_json(path)
        prediction = path.parent / Path(receipt["predictions"]["path"]).name
    except Exception:
        return False
    return (
        receipt.get("dataset_hash") == dataset_id
        and receipt.get("algorithm") == method_id
        and receipt.get("status") == "predicted-unscored"
        and prediction.is_file()
    )


def launch(
    method_id: str,
    dataset_id: str,
    *,
    worker_root: Path,
    artifact_root: Path,
    log_root: Path,
) -> Running:
    command = [
        str(PYTHON),
        str(EXECUTION_SCRIPT),
        "--worker-root",
        str(worker_root),
        "--artifact-root",
        str(artifact_root),
        "run-registered",
        "--dataset",
        dataset_id,
        "--method",
        method_id,
    ]
    log_path = log_root / f"{method_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    log.write("$ " + " ".join(command) + "\n")
    log.flush()
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, **THREAD_ENV},
        preexec_fn=os.setsid,
    )
    log.close()
    return Running(method_id, process, time.monotonic(), log_path)


def run_node(args: argparse.Namespace) -> int:
    cohort = read_json(args.cohort)
    registered = set(cohort["registered_method_ids"])
    specs = import_specs()
    worker_manifest = read_json(args.worker_root / args.dataset / "worker-manifest.json")
    from models.tools.model_zoo import DatasetInfo

    dataset_info = DatasetInfo(
        args.dataset,
        int(worker_manifest["train_rows"]),
        int(worker_manifest["test_rows"]),
        int(worker_manifest["n_features"]),
    )
    frozen_ineligible = {
        item["method_id"]: item["reason"]
        for item in cohort["full_zts_terminal_ineligibilities"]
    }
    requested_resources = set(args.resources.split(","))
    requested_methods = (
        None
        if args.methods is None
        else set(part for part in args.methods.split(",") if part)
    )
    methods: list[str] = []
    ineligible: list[dict[str, str]] = []
    for method_id in cohort["registered_method_ids"]:
        if (
            method_id not in registered
            or specs[method_id].resource not in requested_resources
            or (requested_methods is not None and method_id not in requested_methods)
        ):
            continue
        eligible, reason = specs[method_id].check_eligibility(dataset_info)
        if not eligible:
            if args.dataset != "ZTS" or frozen_ineligible.get(method_id) != reason:
                raise ValueError(f"unexpected ineligibility for {method_id}: {reason}")
            ineligible.append({"method_id": method_id, "reason": reason})
            continue
        if args.dataset == "ZTS" and method_id in frozen_ineligible:
            raise ValueError(f"frozen ZTS ineligibility disappeared for {method_id}")
        methods.append(method_id)
    artifact_root = args.artifact_root
    pending = [
        method_id
        for method_id in methods
        if args.force or not valid_prediction_receipt(artifact_root, args.dataset, method_id)
    ]
    phase = "-".join(sorted(requested_resources))
    campaign_node_root = args.campaign_root / "nodes" / args.node / args.dataset / phase
    telemetry_path = campaign_node_root / "telemetry.jsonl"
    events_path = campaign_node_root / "events.jsonl"
    log_root = campaign_node_root / "logs"
    running: dict[int, Running] = {}
    completed: list[str] = []
    failed: list[dict[str, Any]] = []
    timed_out: list[str] = []
    worker_limit = min(args.workers, args.maximum_workers)
    last_adjust = time.monotonic()
    last_telemetry = 0.0
    prior_swap = (0, 0)
    cpu_samples: list[float] = []
    peak_rss = 0
    start = time.monotonic()
    timeout_seconds = (
        max(args.timeout_seconds, RETRY_TIMEOUT_SECONDS)
        if args.force
        else args.timeout_seconds
    )
    psutil.cpu_percent(interval=None)

    append_jsonl(
        events_path,
        {
            "event": "node_start",
            "node": args.node,
            "dataset": args.dataset,
            "resources": sorted(requested_resources),
            "methods": methods,
            "ineligible": ineligible,
            "pending": len(pending),
            "initial_worker_limit": worker_limit,
            "ts_unix": time.time(),
        },
    )
    while pending or running:
        now = time.monotonic()
        pressure = pressure_snapshot(args.reserve_gib, prior_swap)
        prior_swap = (pressure["swap_in"], pressure["swap_out"])
        if now - last_adjust >= args.adjust_interval:
            cpu = float(pressure["cpu_percent"])
            if cpu < args.target_cpu_low and not pressure["launch_paused"]:
                worker_limit = min(args.maximum_workers, worker_limit + args.adjustment)
            elif cpu > args.target_cpu_high:
                worker_limit = max(args.adjustment, worker_limit - args.adjustment)
            last_adjust = now
            append_jsonl(
                events_path,
                {
                    "event": "worker_limit_adjusted",
                    "worker_limit": worker_limit,
                    "cpu_percent": cpu,
                    "ts_unix": time.time(),
                },
            )

        if not pressure["launch_paused"]:
            while pending and len(running) < worker_limit:
                method_id = pending.pop(0)
                item = launch(
                    method_id,
                    args.dataset,
                    worker_root=args.worker_root,
                    artifact_root=artifact_root,
                    log_root=log_root,
                )
                running[item.process.pid] = item
                append_jsonl(
                    events_path,
                    {
                        "event": "launched",
                        "method_id": method_id,
                        "pid": item.process.pid,
                        "worker_limit": worker_limit,
                        "ts_unix": time.time(),
                    },
                )

        for pid, item in list(running.items()):
            return_code = item.process.poll()
            elapsed = now - item.started
            if return_code is None and elapsed >= timeout_seconds:
                stop_process(item.process)
                timed_out.append(item.method_id)
                return_code = 124
            if return_code is None:
                continue
            running.pop(pid)
            if return_code == 0 and valid_prediction_receipt(
                artifact_root,
                args.dataset,
                item.method_id,
            ):
                completed.append(item.method_id)
            else:
                failed.append(
                    {
                        "method_id": item.method_id,
                        "return_code": return_code,
                        "seconds": elapsed,
                        "log": str(item.log_path),
                    }
                )
            append_jsonl(
                events_path,
                {
                    "event": "finished",
                    "method_id": item.method_id,
                    "return_code": return_code,
                    "seconds": elapsed,
                    "ts_unix": time.time(),
                },
            )

        rss = sum(process_rss(item.process) for item in running.values())
        peak_rss = max(peak_rss, rss)
        if now - last_telemetry >= args.telemetry_interval:
            cpu_samples.append(float(pressure["cpu_percent"]))
            append_jsonl(
                telemetry_path,
                {
                    **pressure,
                    "running": len(running),
                    "pending": len(pending),
                    "completed": len(completed),
                    "failed": len(failed),
                    "worker_limit": worker_limit,
                    "children_rss_bytes": rss,
                    "ts_unix": time.time(),
                },
            )
            last_telemetry = now
        if pending or running:
            time.sleep(args.poll_interval)

    summary = {
        "schema": "golden46.node-run-summary.v1",
        "campaign_id": CAMPAIGN_ID,
        "node": args.node,
        "dataset": args.dataset,
        "phase": phase,
        "resources": sorted(requested_resources),
        "target_methods": methods,
        "ineligible": ineligible,
        "completed": completed,
        "failed": failed,
        "timed_out": timed_out,
        "wall_time_seconds": time.monotonic() - start,
        "mean_cpu_percent": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0,
        "peak_children_rss_bytes": peak_rss,
        "maximum_worker_limit": args.maximum_workers,
        "timeout_seconds": timeout_seconds,
        "memory_reserve_gib": args.reserve_gib,
        "telemetry": str(telemetry_path),
    }
    atomic_json(campaign_node_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--resources", default="cpu,cpu_light")
    parser.add_argument("--methods", default=None)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--maximum-workers", type=int, default=112)
    parser.add_argument("--adjustment", type=int, default=8)
    parser.add_argument("--adjust-interval", type=float, default=120.0)
    parser.add_argument("--target-cpu-low", type=float, default=70.0)
    parser.add_argument("--target-cpu-high", type=float, default=85.0)
    parser.add_argument("--reserve-gib", type=float, required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(INITIAL_TIMEOUT_SECONDS),
    )
    parser.add_argument("--telemetry-interval", type=float, default=10.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_node(parse_args()))
