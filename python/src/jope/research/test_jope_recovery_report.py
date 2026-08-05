from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("jope_recovery_report.py")
SPEC = importlib.util.spec_from_file_location("jope_recovery_report", MODULE_PATH)
assert SPEC and SPEC.loader
report_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report_module)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def native_documents(root: Path, timestamp: int = 1_000) -> None:
    values = {
        "controller": {"action": "active_trainer_preserved"},
        "ingestion": {"failures": 0, "errors": []},
        "queue": {
            "admission_allowed": True,
            "degraded": False,
            "useful_work": 4,
            "queued": 4,
            "dispatching": 0,
        },
        "supply": {
            "admission_allowed": True,
            "below_floor": False,
            "performance_fresh": True,
            "target_feasible": True,
            "complete_reserve_waves": 2,
            "useful_work": 6_144,
            "useful_work_target": 6_144,
        },
        "freezer": {"action": "collecting", "degraded": False},
        "trainer": {"action": "active_trainer_preserved"},
        "promotion_executor": {"pending_intents": 0, "result": {"action": "idle"}},
        "repair": {"queue_depth": 0, "permanent_failures_are_imputed": False},
        "causal_evidence": {"authoritative": True},
        "freshness": {
            "healthy": True,
            "stale_epoch_count": 0,
            "admission_paused": False,
        },
        "liveness_watchdog": {"alarms": []},
        "effective_unit_audit": {"healthy": True, "violations": []},
    }
    written: set[str] = set()
    for name, spec in report_module.COMPONENTS.items():
        if spec.path in written:
            continue
        value = {"schema": spec.schema, "generated_at_ms": timestamp, **values[name]}
        if name == "controller":
            value.update(values["trainer"])
        write(root / spec.path, value)
        written.add(spec.path)


def test_report_uses_native_documents_and_fails_closed_on_empty_queue(tmp_path):
    native_documents(tmp_path)
    report = report_module.build_report(tmp_path, now_ms=1_001)
    assert report["schema"] == "JopeRecoveryReport.v2"
    assert report["healthy"] is True
    assert "services" not in report
    assert report["components"]["queue"]["path"] == "ops/queue-keeper-v3.json"
    queue = json.loads((tmp_path / "ops/queue-keeper-v3.json").read_text())
    queue["useful_work"] = 0
    queue["queued"] = 0
    write(tmp_path / "ops/queue-keeper-v3.json", queue)
    report = report_module.build_report(tmp_path, now_ms=1_001)
    assert report["healthy"] is False
    assert report["gates"]["queue"] is False


def test_stale_native_document_cannot_be_masked_by_status_aggregate(tmp_path):
    native_documents(tmp_path)
    write(
        tmp_path / "ops/status-v3.json",
        {"schema": "JopeSelfplayOperationalStatus.v3", "updated_at_ms": 10_000, "healthy": True},
    )
    supply = json.loads((tmp_path / "ops/micro-wave-supply-v2.json").read_text())
    supply["generated_at_ms"] = 1
    write(tmp_path / "ops/micro-wave-supply-v2.json", supply)
    report = report_module.build_report(tmp_path, now_ms=1_000_000)
    assert report["gates"]["supply"] is False
    assert report["components"]["supply"]["fresh"] is False


def test_continuity_requires_frequent_green_observations_and_resets(tmp_path):
    native_documents(tmp_path)
    first = report_module.build_report(tmp_path, now_ms=1_000)
    continuity = report_module.update_continuity(tmp_path, first, 1_000)
    assert continuity["green_since_ms"] == 1_000
    native_documents(tmp_path, timestamp=1_000 + 86_400_000)
    late = report_module.build_report(tmp_path, now_ms=1_000 + 86_400_000)
    continuity = report_module.update_continuity(tmp_path, late, 1_000 + 86_400_000)
    assert continuity["release_ready"] is False, "a day-long monitoring gap must reset continuity"

    # Seed a genuinely continuous interval and observe it within the allowed gap.
    write(
        tmp_path / "ops/recovery-continuity.v1.json",
        {
            "schema": "JopeRecoveryContinuity.v1",
            "green_since_ms": 1_000,
            "last_observed_at_ms": 1_000 + 86_399_000,
            "last_observation_green": True,
        },
    )
    continuity = report_module.update_continuity(tmp_path, late, 1_000 + 86_400_000)
    assert continuity["release_ready"] is True
    queue = json.loads((tmp_path / "ops/queue-keeper-v3.json").read_text())
    queue["useful_work"] = 0
    queue["queued"] = 0
    write(tmp_path / "ops/queue-keeper-v3.json", queue)
    failed = report_module.build_report(tmp_path, now_ms=1_000 + 86_400_001)
    continuity = report_module.update_continuity(tmp_path, failed, 1_000 + 86_400_001)
    assert continuity["green_since_ms"] is None
    assert continuity["release_ready"] is False


def test_report_exposes_all_task_fields_and_missing_optional_eligibility(tmp_path):
    native_documents(tmp_path)
    write(
        tmp_path / "training/accepted-by-task.json",
        {"tasks": {"regression": {"checkpoint_sha256": "a" * 64}}},
    )
    report = report_module.build_report(tmp_path, now_ms=1_001)
    assert set(report["tasks"]) == set(report_module.TASKS)
    assert report["tasks"]["regression"]["accepted_checkpoint"] == "a" * 64
    assert report["tasks"]["multiclass"]["training_eligibility"] is False
    assert report["tasks"]["multiclass"]["blocking_gate"] == "missing_catalog_bound_training_eligibility_receipt"
