#!/usr/bin/env python3
"""Independent perpetual supervisors for queue, reserve, ingest, and evolution."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

from test_exact_common import atomic_json, dynamic_queue_targets, sha256_file

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
SCRIPTS = Path(__file__).resolve().parent
SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
SCQD = Path("/home/ubuntu/smart_cluster/scripts/scqd.sh")
SERVICES = ("queue", "reserve", "ingestion", "controller", "watchdog")
INGEST_BATCH_LIMIT = 100

from test_exact_service_runtime import (
    burn_targets,
    loaded_bundle,
    mark_micro_wave_request,
    parse_result,
    pending_micro_wave_request,
    refresh_status_async,
    run,
    start_wave_generation,
    wave_generation_state,
)
from test_exact_service_cycles import (
    controller_cycle,
    ensure_current_proposals,
    watchdog_cycle,
)



def queue_cycle() -> dict:
    result = parse_result(
        run(
            [
                sys.executable,
                str(SCRIPTS / "test_exact_backlog.py"),
                "dynamic-refill",
            ],
            timeout=3_600,
        )
    )
    if result.get("needed") and not result.get("submitted"):
        result["alert"] = "queue_below_floor_without_launch_ready_reserve"
    return result


def reserve_cycle() -> dict:
    from test_exact_backlog import (
        bounded_submission_count,
        prepare,
        ready_manifest_count,
        unclaimed_manifest_count,
    )
    from test_exact_wave import checkpoint, plan_micro_cohort

    queued_target, queued_hard_cap, p95 = burn_targets()
    required = queued_target
    ready_before = ready_manifest_count()
    unclaimed_before = unclaimed_manifest_count()
    prepared = 0
    if ready_before < required and unclaimed_before:
        prepared = len(prepare(bounded_submission_count(required - ready_before)))
    ready_mid = ready_manifest_count()
    unclaimed_mid = unclaimed_manifest_count()
    generation = wave_generation_state()
    request = pending_micro_wave_request()
    if request is not None:
        _, active_checkpoint = checkpoint()
        if request.get("checkpoint_sha256") != active_checkpoint:
            mark_micro_wave_request(request, "superseded_checkpoint")
            request = None
    can_generate = ready_mid + unclaimed_mid < required
    if generation is None and can_generate:
        micro = plan_micro_cohort()
        proposal = ensure_current_proposals(Path(micro["cohort"]))
        if proposal.get("ready"):
            _, checkpoint_hash = checkpoint()
            if proposal["checkpoint_sha256"] != checkpoint_hash:
                raise RuntimeError("proposal checkpoint changed before wave generation")
            generation = start_wave_generation(
                proposal["proposals"], micro["cohort"]
            )
            if request is not None:
                mark_micro_wave_request(request, "accepted")
        else:
            generation = proposal
    result = {
        "queued_target": queued_target,
        "queued_hard_cap": queued_hard_cap,
        "p95_jobs_per_minute": p95,
        "required_launch_ready": required,
        "ready_before": ready_before,
        "unclaimed_before": unclaimed_before,
        "generated": generation,
        "micro_wave_request": (
            None
            if request is None
            else {
                "request_sha256": request["request_sha256"],
                "disposition": (
                    "accepted"
                    if generation
                    and generation.get("phase") == "generating_wave"
                    else "deferred_launch_ready_horizon"
                ),
            }
        ),
        "prepared": prepared,
        "ready_after": ready_manifest_count(),
        "unclaimed_after": unclaimed_manifest_count(),
    }
    if result["ready_after"] < queued_target:
        result["alert"] = "launch_ready_reserve_below_one_horizon"
    return result


def ingestion_cycle() -> dict:
    result = parse_result(
        run(
            [
                sys.executable,
                str(SCRIPTS / "test_exact_ingest.py"),
                "reconcile",
                "--limit",
                str(INGEST_BATCH_LIMIT),
            ],
            timeout=3_600,
        )
    )
    status = ROOT / "ops" / "last-ingest.json"
    if status.is_file():
        age = time.time() - status.stat().st_mtime
        if age > 300:
            result["alert"] = "no_labels_for_five_minutes"
    return result


def status_path(service: str, root: Path = ROOT) -> Path:
    return root / "ops" / "services" / f"{service}.json"


def service_state(service: str, root: Path = ROOT) -> dict:
    path = status_path(service, root)
    return json.loads(path.read_text()) if path.is_file() else {}


def record_starting(service: str, root: Path = ROOT) -> dict:
    prior = service_state(service, root)
    value = {
        **prior,
        "schema": "JopeSelfplayServiceState.v1",
        "service": service,
        "pid": os.getpid(),
        "loaded_bundle_sha256": loaded_bundle(root),
        "process_started_at_ms": int(time.time() * 1000),
        "last_success_at_ms": prior.get("last_success_at_ms"),
        "lifecycle": "starting",
        "error": None,
    }
    atomic_json(status_path(service, root), value)
    return value


def execute_cycle(service: str, root: Path = ROOT) -> dict:
    prior = service_state(service, root)
    if service == "queue":
        result = queue_cycle()
    elif service == "reserve":
        result = reserve_cycle()
    elif service == "ingestion":
        result = ingestion_cycle()
    elif service == "controller":
        result = controller_cycle()
    else:
        result = watchdog_cycle(prior.get("result") or {})
    timestamp = int(time.time() * 1000)
    value = {
        "schema": "JopeSelfplayServiceState.v1",
        "service": service,
        "pid": os.getpid(),
        "loaded_bundle_sha256": loaded_bundle(root),
        "last_attempt_at_ms": timestamp,
        "last_success_at_ms": timestamp,
        "lifecycle": "running",
        "error": None,
        "result": result,
    }
    atomic_json(status_path(service, root), value)
    error_path = root / "ops" / "services" / f"{service}.error.json"
    error_path.unlink(missing_ok=True)
    return value


def record_error(service: str, error: BaseException, root: Path = ROOT) -> dict:
    prior = service_state(service, root)
    value = {
        **prior,
        "schema": "JopeSelfplayServiceState.v1",
        "service": service,
        "pid": os.getpid(),
        "loaded_bundle_sha256": loaded_bundle(root),
        "last_attempt_at_ms": int(time.time() * 1000),
        "lifecycle": "error",
        "error": str(error),
    }
    atomic_json(status_path(service, root), value)
    atomic_json(
        root / "ops" / "services" / f"{service}.error.json",
        {
            "schema": "JopeSelfplayServiceError.v1",
            "service": service,
            "at_ms": int(time.time() * 1000),
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=SERVICES)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    intervals = {
        "queue": 5.0,
        "reserve": 5.0,
        "ingestion": 5.0,
        "controller": 30.0,
        "watchdog": 5.0,
    }
    interval = max(1.0, args.interval or intervals[args.service])
    lock_path = args.root / "ops" / "services" / f"{args.service}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            # jankurai:allow HLT-008-FALSE-GREEN-RISK reason=verified-exit-behavior expires=2026-12-31
            raise SystemExit(f"{args.service} service already owns its lock") from error
        print(
            json.dumps(record_starting(args.service, args.root), sort_keys=True),
            flush=True,
        )
        while True:
            try:
                value = execute_cycle(args.service, args.root)
            except Exception as error:
                value = record_error(args.service, error, args.root)
            print(json.dumps(value, sort_keys=True), flush=True)
            if args.service == "queue":
                refresh_status_async(args.root)
            if args.once:
                break
            time.sleep(interval)


if __name__ == "__main__":
    main()
