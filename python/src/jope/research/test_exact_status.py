#!/usr/bin/env python3
"""Aggregate independent self-play supervisors into operational status v3."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from test_exact_common import atomic_json, validate_catalog_manifest
from test_exact_evaluator import load_pointer
from test_exact_pair_registry import PairRegistry
from test_exact_sealed_audit import SealedAuditRegistry
from test_exact_service import SERVICES, loaded_bundle, service_state
from test_exact_compile_cache import cache_status

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")


def pointer(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def service_is_stale(state: dict, now: int) -> bool:
    success = state.get("last_success_at_ms")
    reference = int(success) if success is not None else 0
    if state.get("lifecycle") == "starting":
        reference = max(reference, int(state.get("process_started_at_ms") or 0))
    return reference == 0 or now - reference > 5 * 60_000


def build_status(root: Path = ROOT) -> dict:
    now = int(time.time() * 1000)
    services = {name: service_state(name, root) for name in SERVICES}
    registry = PairRegistry(root / "ops" / "pair-registry.sqlite")
    pair_status = registry.stats()
    registry.close()
    sealed_registry = SealedAuditRegistry(
        root / "sealed-audit" / "registry.sqlite"
    )
    sealed = sealed_registry.status()
    sealed_registry.close()
    catalog = validate_catalog_manifest(
        root / "catalog" / "current.json", verify_contexts=False
    )
    flywheel = pointer(root / "ops" / "flywheel-state.json") or {}
    try:
        from test_exact_monitor import trainer_inventory

        trainer = trainer_inventory()
    except Exception as error:
        trainer = {"error": str(error)}
    learning = pointer(root / "training" / "learning.json")
    milestone = pointer(root / "training" / "milestone.json")
    loaded = loaded_bundle(root)
    reported_bundles = {
        value.get("loaded_bundle_sha256")
        for value in services.values()
        if value
    }
    alerts = []
    for name, state in services.items():
        if not state:
            alerts.append(f"{name}:no_status")
            continue
        if state.get("error"):
            alerts.append(f"{name}:error")
        if service_is_stale(state, now):
            alerts.append(f"{name}:stale")
    queue = (services.get("queue") or {}).get("result") or {}
    reserve = (services.get("reserve") or {}).get("result") or {}
    queue_depth = int(
        queue.get("useful_depth_after", queue.get("useful_depth", 0))
    )
    queued_target = int(queue.get("queued_target", 256))
    hard_cap = int(queue.get("queued_hard_cap", 512))
    if queue_depth < queued_target:
        alerts.append("queue:below_fresh_horizon")
    if queue_depth > hard_cap:
        alerts.append("queue:above_hard_cap")
    if float(queue.get("maximum_queue_age_seconds") or 0.0) > 30 * 60:
        alerts.append("queue:older_than_30_minutes")
    if reserve.get("ready_after", 0) > queued_target:
        alerts.append("reserve:above_one_launch_ready_horizon")
    if reported_bundles and reported_bundles != {loaded}:
        alerts.append("deployment:loaded_bundle_mismatch")
    freshness = pointer(root / "ops" / "freshness-status.json") or {}
    if not freshness:
        alerts.append("freshness:no_status")
    elif not freshness.get("healthy", False):
        alerts.append("freshness:unhealthy")
    micro_cohort = None
    pending_cohorts = sorted((root / "waves" / "pending").glob("*.cohort.json"))
    if pending_cohorts:
        cohort_path = pending_cohorts[-1]
        micro_cohort = {
            **json.loads(cohort_path.read_text()),
            "path": str(cohort_path),
        }
    cache_progress = {}
    active_counts = catalog.get("task_counts") or {}
    for task in ("regression", "binary"):
        cache_root = Path(
            f"/home/ubuntu/jope-flywheel/encodings/{task}-expanded"
        )
        completed = (
            sum(
                (path / "cache-manifest.v1.json").is_file()
                for path in cache_root.iterdir()
            )
            if cache_root.is_dir()
            else 0
        )
        cache_progress[task] = {
            "completed": completed,
            "active_catalog": int(active_counts.get(task, 0)),
        }
    corpus_path = root / "corpus" / "strict-test-exact.sqlite"
    adaptive_fitness_watermark = 0
    if corpus_path.is_file():
        corpus = sqlite3.connect(f"file:{corpus_path}?mode=ro", uri=True)
        try:
            adaptive_fitness_watermark = int(
                corpus.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            )
        finally:
            corpus.close()
    p95 = float(queue.get("p95_jobs_per_minute") or 0.0)
    queue_horizon_minutes = None if p95 <= 0 else queue_depth / p95
    result = {
        "schema": "JopeSelfplayOperationalStatus.v3",
        "updated_at_ms": now,
        "queue": {
            **queue,
            "queued_target": queued_target,
            "queued_hard_cap": hard_cap,
            "useful_depth": queue_depth,
            "horizon_minutes": queue_horizon_minutes,
            "running": queue.get("running_after", queue.get("running", 0)),
            "dispatching": queue.get(
                "dispatching_after", queue.get("dispatching", 0)
            ),
            "held": queue.get("held_after", queue.get("held", 0)),
        },
        "reserve": reserve,
        "ingestion": (services.get("ingestion") or {}).get("result"),
        "trainer": trainer,
        "evolution": {
            "tournament": flywheel.get("tournament"),
            "last_terminal": flywheel.get("last_terminal"),
            "last_tournament": flywheel.get("last_tournament"),
            "consecutive_rejections": flywheel.get(
                "consecutive_rejections", 0
            ),
        },
        "promotion": {
            "internal_learning": learning,
            "milestone": milestone,
        },
        "scq_watchdog": (services.get("watchdog") or {}).get("result"),
        "evaluator": {
            "active": load_pointer("active", root),
            "candidate": load_pointer("candidate", root),
        },
        "catalog": catalog,
        "pair_registry": pair_status,
        "loaded_bundle_sha256": loaded,
        "service_loaded_bundle_sha256s": sorted(
            value for value in reported_bundles if value is not None
        ),
        "sealed_audit": sealed,
        "freshness": freshness,
        "queue_epochs": freshness.get("queue_epochs", []),
        "stale_holds": freshness.get("stale_held", 0),
        "stale_retirements": pair_status.get("superseded_unstarted", 0),
        "active_micro_cohort": micro_cohort,
        "expanded_cache_progress": cache_progress,
        "adaptive_fitness_watermark": adaptive_fitness_watermark,
        "mcp": {
            "socket": str(root / "ops" / "freshness.sock"),
            "healthy": (root / "ops" / "freshness.sock").exists()
            and bool(freshness),
            "protocol_version": "2025-06-18",
        },
        "node_slot_policy": queue.get("nodes", []),
        "compile_cache": cache_status(root / "compiled-cache"),
        "checkpoint_to_first_job_latency_ms": freshness.get(
            "checkpoint_to_first_job_latency_ms"
        ),
        "services": services,
        "alerts": sorted(set(alerts)),
    }
    atomic_json(root / "ops" / "status-v3.json", result)
    atomic_json(root / "ops" / "status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build_status(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
