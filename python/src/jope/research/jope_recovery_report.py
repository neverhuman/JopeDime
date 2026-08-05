#!/usr/bin/env python3
"""Build the native, fail-closed JOPE-Dime recovery and continuity views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple


TASKS = ("regression", "binary", "multiclass", "timeseries")
REPORT_SCHEMA = "JopeRecoveryReport.v2"
CONTINUITY_SCHEMA = "JopeRecoveryContinuity.v1"
REQUIRED_GREEN_SECONDS = 86_400
MAXIMUM_OBSERVATION_GAP_SECONDS = 120


class Component(NamedTuple):
    path: str
    schema: str
    freshness_budget_seconds: int
    check: Callable[[dict[str, Any]], tuple[bool, str]]


def ok(_: dict[str, Any]) -> tuple[bool, str]:
    return True, "native document is current"


def queue_check(value: dict[str, Any]) -> tuple[bool, str]:
    if value.get("error") or value.get("fail_closed_reason"):
        return False, str(value.get("error") or value.get("fail_closed_reason"))
    if value.get("admission_allowed") is not True or value.get("degraded") is True:
        return False, "queue admission is closed or degraded"
    if useful_queue_depth(value) <= 0:
        return False, "fresh useful queue depth is zero"
    return True, "fresh useful queue depth is nonzero"


def supply_check(value: dict[str, Any]) -> tuple[bool, str]:
    if value.get("error") or value.get("hold_reason"):
        return False, str(value.get("error") or value.get("hold_reason"))
    if value.get("admission_allowed") is not True:
        return False, "supply admission is closed"
    useful_work = int(value.get("useful_work") or 0)
    target = int(value.get("useful_work_target") or 0)
    if (
        value.get("performance_fresh") is not True
        or value.get("target_feasible") is not True
        or int(value.get("complete_reserve_waves") or 0) < 2
        or useful_work < target
    ):
        return False, "useful-work runway or complete-wave reserve is not proven"
    return True, f"useful work {useful_work} meets target {target}"


def ingestion_check(value: dict[str, Any]) -> tuple[bool, str]:
    if int(value.get("failures") or 0) or value.get("errors"):
        return False, "ingestion reports failures"
    return True, "ingestion reconcile has no failures"


def freezer_check(value: dict[str, Any]) -> tuple[bool, str]:
    if value.get("degraded") is True or value.get("error") or value.get("timed_out") is True:
        return False, str(value.get("error") or "freezer is degraded or timed out")
    return True, f"freezer action {value.get('action') or value.get('lifecycle') or 'unknown'}"


def controller_check(value: dict[str, Any]) -> tuple[bool, str]:
    if value.get("error") or value.get("detail"):
        return False, str(value.get("error") or value.get("detail"))
    return True, f"controller action {value.get('action') or 'idle'}"


def trainer_check(value: dict[str, Any]) -> tuple[bool, str]:
    action = value.get("action")
    allowed = {
        "active_trainer_preserved",
        "trainer_submitted",
        "gpu_admission_requested",
        "trainer_result_harvested",
    }
    if action not in allowed:
        return False, f"trainer state is not proven active or recoverable: {action}"
    return True, str(action)


def promotion_check(value: dict[str, Any]) -> tuple[bool, str]:
    result = value.get("result") or {}
    if value.get("error") or result.get("error"):
        return False, str(value.get("error") or result.get("error"))
    return True, str(result.get("action") or "promotion executor idle")


def repair_check(value: dict[str, Any]) -> tuple[bool, str]:
    if value.get("permanent_failures_are_imputed") is not False:
        return False, "repair evidence may be imputed"
    if value.get("error"):
        return False, str(value["error"])
    return True, f"{int(value.get('queue_depth') or 0)} repairs remain promotion-ineligible"


def causal_evidence_check(value: dict[str, Any]) -> tuple[bool, str]:
    if value.get("authoritative") is not True or value.get("error"):
        return False, str(value.get("error") or "causal evidence is non-authoritative")
    return True, "causal evidence ledger is authoritative"


def freshness_check(value: dict[str, Any]) -> tuple[bool, str]:
    if value.get("healthy") is not True or int(value.get("stale_epoch_count") or 0):
        return False, "freshness authority is unhealthy or has stale epochs"
    if value.get("admission_paused") is True:
        return False, "freshness authority paused admission"
    return True, "freshness authority is healthy"


def liveness_check(value: dict[str, Any]) -> tuple[bool, str]:
    alarms = value.get("alarms") or []
    if alarms:
        return False, "; ".join(str(alarm) for alarm in alarms)
    return True, "liveness watchdog has no active alarms"


def unit_audit_check(value: dict[str, Any]) -> tuple[bool, str]:
    if value.get("healthy") is not True or value.get("violations"):
        return False, "effective unit inventory has drift or violations"
    return True, "effective unit inventory matches the active bundle"


COMPONENTS = {
    "controller": Component(
        "ops/task-learning-controller.v2.json", "TaskLearningControllerStatus.v2", 120, controller_check
    ),
    "ingestion": Component("ops/last-ingest.json", "JopeSelfplayIngestReceipt.v1", 300, ingestion_check),
    "queue": Component("ops/queue-keeper-v3.json", "JopeSelfplayQueueKeeper.v3", 120, queue_check),
    "supply": Component("ops/micro-wave-supply-v2.json", "JopeMicroWaveSupplyStatus.v2", 300, supply_check),
    "freezer": Component("ops/corpus-freezer-v1.json", "JopeSelfplayCorpusFreezer.v1", 120, freezer_check),
    # The task controller owns the active trainer intent.  The obsolete standalone
    # watchdog document is intentionally not authoritative for this gate.
    "trainer": Component(
        "ops/task-learning-controller.v2.json", "TaskLearningControllerStatus.v2", 120, trainer_check
    ),
    "promotion_executor": Component("ops/canary-executor.v2.json", "CanaryExecutorStatus.v2", 120, promotion_check),
    "repair": Component(
        "ops/causal-pair-repair-executor.v1.json", "CausalPairRepairExecutor.v1", 120, repair_check
    ),
    "causal_evidence": Component(
        "ops/causal-evidence-audit.v2.json", "CausalEvidenceAudit.v3", 900, causal_evidence_check
    ),
    # This is activation state rather than a heartbeat; liveness below proves the
    # current process.  A 31-day budget still prevents indefinitely stale policy.
    "freshness": Component(
        "ops/freshness-status.json", "JopeSelfplayFreshnessStatus.v1", 31 * 86_400, freshness_check
    ),
    "liveness_watchdog": Component("ops/flywheel-liveness.v1.json", "FlywheelLiveness.v1", 120, liveness_check),
    "effective_unit_audit": Component(
        "ops/effective-unit-audit.v1.json", "JopeEffectiveUnitAudit.v1", 86_400, unit_audit_check
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_timestamp(value: dict[str, Any]) -> int | None:
    for key in (
        "heartbeat_at_ms",
        "last_heartbeat_at_ms",
        "generated_at_ms",
        "updated_at_ms",
        "activated_at_ms",
    ):
        timestamp = value.get(key)
        if isinstance(timestamp, int) and not isinstance(timestamp, bool):
            return timestamp
    return None


def useful_queue_depth(value: dict[str, Any]) -> int:
    explicit = value.get("useful_work")
    if isinstance(explicit, int) and not isinstance(explicit, bool):
        return max(0, explicit)
    return sum(
        max(0, int(value.get(key) or 0))
        for key in ("queued", "starting", "dispatching", "validated_ready")
    )


def component_rows(root: Path, now_ms: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, spec in COMPONENTS.items():
        path = root / spec.path
        value = read_json(path)
        timestamp = native_timestamp(value)
        age_seconds = None if timestamp is None else max(0, now_ms - timestamp) // 1000
        present = bool(value)
        schema_valid = value.get("schema") == spec.schema
        fresh = age_seconds is not None and age_seconds <= spec.freshness_budget_seconds
        check_healthy, check_reason = spec.check(value) if present and schema_valid else (False, "")
        if not present:
            reason = "native component document is missing"
        elif not schema_valid:
            reason = f"schema mismatch: expected {spec.schema}, got {value.get('schema')}"
        elif timestamp is None:
            reason = "native component document has no timestamp"
        elif not fresh:
            reason = f"native component document is stale by {age_seconds}s"
        else:
            reason = check_reason
        rows[name] = {
            "path": spec.path,
            "schema": value.get("schema"),
            "sha256": sha256_file(path) if path.is_file() else None,
            "timestamp_ms": timestamp,
            "age_seconds": age_seconds,
            "freshness_budget_seconds": spec.freshness_budget_seconds,
            "fresh": fresh,
            "healthy": present and schema_valid and fresh and check_healthy,
            "reason": reason,
        }
    return rows


def current_catalog(root: Path) -> tuple[dict[str, set[str]], str | None]:
    pointer = read_json(root / "catalog/current.json")
    path_value = pointer.get("catalog")
    if not isinstance(path_value, str):
        return {task: set() for task in TASKS}, None
    path = Path(path_value)
    if not path.is_absolute():
        path = root / "catalog" / path
    if not path.is_file():
        return {task: set() for task in TASKS}, None
    digest = sha256_file(path)
    if pointer.get("catalog_sha256") != digest:
        return {task: set() for task in TASKS}, None
    entries = read_json(path).get("entries") or []
    families = {task: set() for task in TASKS}
    for row in entries:
        if not isinstance(row, dict):
            continue
        task, family = row.get("task"), row.get("source_family")
        if task in families and isinstance(family, str) and family:
            families[task].add(family)
    return families, digest


def task_eligibility(root: Path, task: str, catalog_digest: str | None) -> dict[str, Any]:
    path = root / "training/task-training-eligibility" / f"{task}.v1.json"
    value = read_json(path)
    valid = (
        value.get("schema") == "JopeTaskTrainingEligibility.v1"
        and value.get("task") == task
        and catalog_digest is not None
        and value.get("catalog_sha256") == catalog_digest
    )
    return value if valid else {}


def latest_learning(root: Path, task: str) -> tuple[dict[str, Any], str | None]:
    directory = root / "ops/learning-reports" / task
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    if directory.is_dir():
        for path in directory.glob("*.v2.json"):
            value = read_json(path)
            if value.get("schema") == "JopeLearningReport.v2" and value.get("task") == task:
                candidates.append((int(value.get("generated_at_ms") or 0), path, value))
    if not candidates:
        return {}, None
    _, path, value = max(candidates, key=lambda row: (row[0], str(row[1])))
    return value, sha256_file(path)


def task_rows(root: Path, queue_fresh: bool, queue_depth: int) -> dict[str, Any]:
    accepted = read_json(root / "training/accepted-by-task.json").get("tasks") or {}
    controller = read_json(root / "ops/task-learning-controller.v2.json")
    shadows = controller.get("shadow_models") or {}
    corpus = read_json(root / "training/current-corpus.json")
    families, catalog_digest = current_catalog(root)
    promotion_ledger = read_json(root / "training/causal-evidence-ledger.v3.json")
    promotion_epochs = promotion_ledger.get("promotion_epoch") or {}
    rows: dict[str, Any] = {}
    for task in TASKS:
        eligibility = task_eligibility(root, task, catalog_digest)
        optional = task in {"multiclass", "timeseries"}
        evidence_enabled = eligibility.get("evidence_lane_enabled") if eligibility else (not optional)
        training_eligible = eligibility.get("training_eligible") if eligibility else (not optional)
        promotion_epoch = promotion_epochs.get(task) or {}
        promotion_eligible = bool(
            training_eligible
            and promotion_epoch.get("production_gate_open") is True
            and queue_fresh
            and queue_depth > 0
        )
        learning, learning_digest = latest_learning(root, task)
        metrics = learning.get("metrics") or {}
        blocker = None
        if optional and not eligibility:
            blocker = "missing_catalog_bound_training_eligibility_receipt"
        elif optional and len(families[task]) < int(eligibility.get("minimum_required_count") or 500):
            blocker = "independent_source_family_floor_not_met"
        elif training_eligible is not True:
            blocker = eligibility.get("advisory_reason") or "training_ineligible"
        elif not queue_fresh or queue_depth <= 0:
            blocker = "fresh_useful_queue_unavailable"
        elif promotion_epoch.get("production_gate_open") is not True:
            blocker = "promotion_epoch_gate_closed"
        rows[task] = {
            "accepted_checkpoint": (accepted.get(task) or {}).get("checkpoint_sha256"),
            "latest_shadow": (shadows.get(task) or {}).get("checkpoint_sha256"),
            "corpus_watermark": corpus.get("watermark"),
            "independent_source_family_count": len(families[task]),
            "evidence_eligibility": evidence_enabled,
            "training_eligibility": training_eligible,
            "promotion_eligibility": promotion_eligible,
            "fresh_queue_state": {"fresh": queue_fresh, "useful_depth": queue_depth},
            "blocking_gate": blocker,
            "last_verified_learning_delta": metrics.get("sealed_panel_lift"),
            "learning_disposition": learning.get("disposition"),
            "learning_report_sha256": learning_digest,
        }
    return rows


def learning_curve(root: Path) -> list[dict[str, Any]]:
    rows = []
    for task in TASKS:
        directory = root / "ops/learning-reports" / task
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.v2.json")):
            value = read_json(path)
            if value.get("schema") != "JopeLearningReport.v2":
                continue
            metrics = value.get("metrics") or {}
            rows.append(
                {
                    "model_checkpoint_sha256": value.get("checkpoint_sha256"),
                    "task": task,
                    "proposal_validity": metrics.get("proposal_validity"),
                    "matched_strict_win_rate": metrics.get("matched_strict_win_rate"),
                    "champion_rate": metrics.get("champion_rate"),
                    "gp_relative_lift": metrics.get("gp_relative_lift"),
                    "p95_latency_ms": metrics.get("p95_latency_ms"),
                    "failure_rate": metrics.get("failure_rate"),
                    "replay_coverage": metrics.get("replay_coverage"),
                    "sealed_panel_lift": metrics.get("sealed_panel_lift"),
                    "promotion_disposition": value.get("disposition"),
                    "report_sha256": sha256_file(path),
                }
            )
    return rows


def build_report(root: Path, now_ms: int | None = None) -> dict[str, Any]:
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    components = component_rows(root, now_ms)
    gates = {name: row["healthy"] for name, row in components.items()}
    queue = read_json(root / COMPONENTS["queue"].path)
    queue_depth = useful_queue_depth(queue)
    queue_fresh = components["queue"]["fresh"]
    healthy = all(gates.values()) and queue_fresh and queue_depth > 0
    continuity = read_json(root / "ops/recovery-continuity.v1.json")
    continuity_ready = (
        continuity.get("schema") == CONTINUITY_SCHEMA
        and continuity.get("release_ready") is True
        and continuity.get("last_observation_green") is True
    )
    return {
        "schema": REPORT_SCHEMA,
        "generated_at_ms": now_ms,
        "healthy": healthy,
        "release_gate": "ready" if healthy and continuity_ready else ("warming" if healthy else "blocked"),
        "gates": gates,
        "components": components,
        "queue": {
            "useful_depth": queue_depth,
            "queued": int(queue.get("queued") or 0),
            "dispatching": int(queue.get("dispatching") or 0),
            "running": int(queue.get("running") or 0),
            "launch_ready": int(queue.get("launch_ready") or 0),
            "fresh": queue_fresh,
        },
        "tasks": task_rows(root, queue_fresh, queue_depth),
        "learning_curve": learning_curve(root),
        "continuity": continuity or None,
        "claims_policy": {
            "corpus_growth_is_not_improvement": True,
            "completed_training_is_not_improvement": True,
            "test_adaptive_evidence_is_benchmark_invalid": True,
            "accepted_improvement_requires_three_family_disjoint_single_use_panels": True,
        },
    }


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update_continuity(root: Path, report: dict[str, Any], now_ms: int) -> dict[str, Any]:
    path = root / "ops/recovery-continuity.v1.json"
    previous = read_json(path)
    green = report.get("healthy") is True and int((report.get("queue") or {}).get("useful_depth") or 0) > 0
    previous_observed = previous.get("last_observed_at_ms")
    gap_ok = (
        isinstance(previous_observed, int)
        and 0 <= now_ms - previous_observed <= MAXIMUM_OBSERVATION_GAP_SECONDS * 1000
    )
    continuing = (
        previous.get("schema") == CONTINUITY_SCHEMA
        and previous.get("last_observation_green") is True
        and isinstance(previous.get("green_since_ms"), int)
        and gap_ok
        and green
    )
    green_since = previous["green_since_ms"] if continuing else (now_ms if green else None)
    green_seconds = 0 if green_since is None else max(0, now_ms - green_since) // 1000
    continuity = {
        "schema": CONTINUITY_SCHEMA,
        "green_since_ms": green_since,
        "last_observed_at_ms": now_ms,
        "last_observation_green": green,
        "last_useful_queue_nonzero_at_ms": now_ms if green else previous.get("last_useful_queue_nonzero_at_ms"),
        "maximum_observation_gap_seconds": MAXIMUM_OBSERVATION_GAP_SECONDS,
        "required_green_seconds": REQUIRED_GREEN_SECONDS,
        "continuously_green_seconds": green_seconds,
        "release_ready": green and green_seconds >= REQUIRED_GREEN_SECONDS,
        "reset_reason": None if green else "one or more native gates failed or useful queue depth was zero",
    }
    atomic_write(path, continuity)
    return continuity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--now-ms", type=int)
    args = parser.parse_args()
    now_ms = int(time.time() * 1000) if args.now_ms is None else args.now_ms
    report = build_report(args.root, now_ms)
    continuity = update_continuity(args.root, report, now_ms)
    report["continuity"] = continuity
    report["release_gate"] = (
        "ready" if report["healthy"] and continuity["release_ready"] else ("warming" if report["healthy"] else "blocked")
    )
    output = args.output or args.root / "ops/recovery-report.v2.json"
    atomic_write(output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if not args.strict or report["release_gate"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
