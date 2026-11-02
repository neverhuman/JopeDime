from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_exact_controller import (
    PHASES,
    ActionJournal,
    FlywheelController,
    PendingAction,
    action_key,
    default_handlers,
)

def catalog_entries():
    return [
        {"task": task, "dataset_hash": f"{task}-{index}"}
        for task, count in (("regression", 2_322), ("binary", 542))
        for index in range(count)
    ]

def snapshot(labels=50_000):
    return {
        "valid_labels": labels,
        "accepted_ancestor": "a" * 64,
        "catalog_sha256": "b" * 64,
        "catalog_entries": catalog_entries(),
        "catalog_counts": {"regression": 2_322, "binary": 542},
        "evaluator_sha256": "c" * 64,
    }

def test_production_controller_owns_deterministic_post_training_phases():
    handlers = default_handlers()
    expected = {
        "build_acceptance_evidence": "_build_acceptance_evidence",
        "run_checkpoint_gate": "_run_checkpoint_gate",
        "advance_accepted_pointer": "_advance_accepted_pointer",
        "seed_next_strict_wave": "_seed_next_strict_wave",
        "freeze_winner_scan": "_freeze_winner_scan",
    }
    assert {phase: handlers[phase].__name__ for phase in expected} == expected
    assert handlers["generate_gate_proposals"].__name__ == "_generate_gate_proposals"
    assert handlers["evaluate_frozen_gate"].__name__ == "_evaluate_frozen_gate"
    assert handlers["run_sotarena_qualification"].__name__ == "_run_sotarena_qualification"

def test_action_key_binds_every_control_input():
    base = {
        "action": "freeze_corpus",
        "corpus_watermark": 50_000,
        "accepted_ancestor": "a" * 64,
        "recipe": "a+b+c",
        "seed": 17,
        "catalog_sha256": "b" * 64,
        "evaluator_sha256": "c" * 64,
    }
    assert action_key(**base) == action_key(**dict(reversed(list(base.items()))))
    changed = {**base, "seed": 18}
    assert action_key(**base) != action_key(**changed)

def test_journal_retries_pending_action_and_never_duplicates_success(tmp_path):
    journal = ActionJournal(tmp_path)
    calls = []

    def callback(request):
        calls.append(request)
        if len(calls) == 1:
            raise PendingAction("still running")
        return {"ok": True}

    with pytest.raises(PendingAction):
        journal.perform("key", "action", {"x": 1}, callback)
    assert journal.perform("key", "action", {"x": 1}, callback) == {"ok": True}
    assert journal.perform("key", "action", {"x": 1}, callback) == {"ok": True}
    assert len(calls) == 2
    events = [
        json.loads(line) for line in (tmp_path / "ops/controller-events.jsonl").read_text().splitlines()
    ]
    assert [row["status"] for row in events] == [
        "started",
        "pending",
        "started",
        "succeeded",
    ]
    journal.close()

@pytest.mark.parametrize("gate_pass,expected", [(True, "accepted"), (False, "rejected")])
def test_controller_resumes_each_phase_and_gate_is_fail_closed(tmp_path, gate_pass, expected):
    calls = []

    def handler(name):
        def run(request):
            calls.append(name)
            if name == "run_checkpoint_gate":
                return {"pass": gate_pass}
            if name == "advance_accepted_pointer":
                return {"advanced": True, "checkpoint_sha256": "d" * 64}
            return {"ok": True}

        return run

    handlers = {phase: handler(phase) for phase in PHASES}
    for _ in PHASES:
        controller = FlywheelController(tmp_path, handlers)
        controller.cycle(snapshot())
        controller.close()
        state = json.loads((tmp_path / "ops/flywheel-state.json").read_text())
        if state["tournament"] is None:
            break
    state = json.loads((tmp_path / "ops/flywheel-state.json").read_text())
    assert state["last_terminal"] == expected
    assert calls.count("freeze_corpus") == 1
    if gate_pass:
        assert "advance_accepted_pointer" in calls
    else:
        assert "advance_accepted_pointer" not in calls
    completed = json.loads(Path(state["last_tournament"]).read_text())
    pointer = completed["artifacts"]["advance_accepted_pointer"]
    assert pointer["advanced"] is gate_pass
    if not gate_pass:
        assert pointer["checkpoint_sha256"] == "a" * 64

def test_controller_coalesces_threshold_and_rejects_catalog_drift(tmp_path):
    controller = FlywheelController(tmp_path, {})
    result = controller.cycle(snapshot(labels=49_999))
    assert result["phase"] == "collecting_labels"
    assert result["next_watermark"] == 50_000
    broken = snapshot()
    broken["catalog_entries"].append({"task": "regression", "dataset_hash": "unexpected"})
    with pytest.raises(ValueError, match="catalog count mismatch"):
        controller.cycle(broken)
    controller.close()
