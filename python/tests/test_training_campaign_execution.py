"""Execution, checkpoint continuation, and recovery contracts for training campaigns."""

import copy
from pathlib import Path

import pytest

from test_training_campaign import campaign, plan, state


@pytest.mark.parametrize("task", campaign.TASKS)
def test_real_cpu_optimizer_reload_continuation_and_export(state, monkeypatch, task):
    import torch

    from jope.policy import train_v3

    before_threads = torch.get_num_threads()
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        train_v3,
        "policy_v3_validation_metrics",
        lambda *args, **kwargs: {"pairwise_win_accuracy": 0.5, "seq_nll": 1.0},
    )
    value = plan(state, task=task)

    # Same controller and real optimizer path; no subprocess is needed in this CPU fixture.
    def in_process(arguments, env, attempt, deadline):
        started_ms = int(campaign.time.time() * 1000)
        started = campaign.time.monotonic()
        campaign.worker(state, task, attempt)
        campaign.write(
            attempt / "outcome.json",
            {
                "status": "completed",
                "exit_code": 0,
                "started_unix_ms": started_ms,
                "ended_unix_ms": int(campaign.time.time() * 1000),
                "deadline_unix_ms": deadline,
                "wall_seconds": campaign.time.monotonic() - started,
            },
            immutable=True,
        )
        return "completed"

    monkeypatch.setattr(campaign, "launch_worker", in_process)
    try:
        result = campaign.run(state)
        assert result["success"] is True
        first_window = campaign.read(state / "window.json")
        assert campaign.run(state, resume=True)["success"] is True
        assert campaign.read(state / "window.json") == first_window
        proof = campaign.verify(state)
        head = campaign.read_head(state / task / "head.json")
        assert head["committed_passes"] == 2
        assert head["training_proof"]["optimizer_state_resumed"] is True
        assert proof["tasks"][0]["optimizer_updates"] > 0
        assert (
            head["training_proof"]["parameter_sha256"] != head["training_proof"]["initial_parameter_sha256"]
        )
        exported = campaign.export(state, state / "export")
        assert exported["verification"] == proof
        assert campaign.read(state / "export" / "manifest.json")["serving_eligible"] is False
        # Every execution receipt in the committed chain is byte-bound.
        outcome = Path(head["execution"]["outcome"]["path"])
        campaign.write(outcome, {"status": "failed", "exit_code": 1})
        with pytest.raises(ValueError, match="artifact changed"):
            campaign.verify(state)
        assert value["data_kind"] == "synthetic_fixture"
    finally:
        torch.set_num_threads(before_threads)


@pytest.mark.parametrize("missing_outcome", [False, True])
def test_campaign_recovers_committed_update_in_original_window(state, monkeypatch, missing_outcome):
    import torch

    from jope.policy import train_v3

    threads = torch.get_num_threads()
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        train_v3,
        "policy_v3_validation_metrics",
        lambda *args, **kwargs: {"pairwise_win_accuracy": 0.5, "seq_nll": 1.0},
    )
    plan(state, task="regression")
    native_request = state / "native-invocations/fixture/request.json"
    if missing_outcome:
        campaign.write(state / "native-plan.json", {"fixture": True})
        campaign.write(
            native_request,
            {
                "schema": "jope.training-child/v1",
                "state": str(state),
                "native_plan_sha256": campaign.sha(state / "native-plan.json"),
                "nonce": "fixture",
            },
        )
        monkeypatch.setenv("JOPE_NATIVE_INVOCATION", str(native_request))
    original_save = train_v3._atomic_torch_save
    interrupted = False

    def interrupt_once(payload, path):
        nonlocal interrupted
        original_save(payload, path)
        if payload.get("strict_progress") and not interrupted:
            interrupted = True
            raise InterruptedError("fixture committed update interruption")

    def in_process(arguments, env, attempt, deadline):
        started_ms = int(campaign.time.time() * 1000)
        status = "completed"
        try:
            campaign.worker(state, "regression", attempt)
        except InterruptedError:
            status = "cancelled"
        if missing_outcome and status == "cancelled":
            campaign.write(
                native_request.parent / "drained.json",
                {
                    "schema": "jope.training-drained/v1",
                    "request_sha256": campaign.sha(native_request),
                    "populated": False,
                    "ended_unix_ms": int(campaign.time.time() * 1000),
                    "fixture": "controller died before outcome",
                },
            )
            return status
        campaign.write(
            attempt / "outcome.json",
            {
                "status": status,
                "exit_code": 0 if status == "completed" else 1,
                "started_unix_ms": started_ms,
                "ended_unix_ms": int(campaign.time.time() * 1000),
                "deadline_unix_ms": deadline,
                "wall_seconds": 0.0,
            },
            immutable=True,
        )
        return status

    monkeypatch.setattr(train_v3, "_atomic_torch_save", interrupt_once)
    monkeypatch.setattr(campaign, "launch_worker", in_process)
    try:
        assert campaign.run(state)["success"] is False
        window = campaign.read(state / "window.json")
        category = campaign.read(state / "regression" / "window.json")
        assert campaign.run(state, resume=True)["success"] is True
        assert campaign.read(state / "window.json") == window
        assert campaign.read(state / "regression" / "window.json") == category
        recovered = campaign.read(state / "regression" / "attempt-000002" / "complete.json")["recovered_from"]
        assert recovered and Path(recovered["checkpoint"]["path"]).parent.name == "attempt-000001"
        assert campaign.verify(state)["tasks"][0]["optimizer_updates"] > 0
        Path(recovered["checkpoint"]["path"]).write_bytes(b"changed")
        with pytest.raises(ValueError, match="artifact changed"):
            campaign.verify(state)
    finally:
        torch.set_num_threads(threads)


def test_real_f32_context_identity_refuses_lossy_numbers_and_changed_bytes(state):
    import hashlib
    import struct

    values = [0.25] * 2048
    training = {"selected_training_rows_sha256": "f" * 64}
    context = {
        "lineage": "asm.experimental-regression-context/v1",
        "policy_compatible": False,
        "training": training,
        "values": values,
        "context_sha256": hashlib.sha256(struct.pack("<2048f", *values)).hexdigest(),
    }
    context_path = state / "context.json"
    campaign.write(context_path, context)
    reference = campaign.artifact(context_path)
    producer = {
        "output_sha256": reference["sha256"],
        "fitting_rows_sha256": "f" * 64,
        "encoder_sha256": "e" * 64,
        "inputs": [reference],
        "training": training,
    }
    producer_path = state / "producer.json"
    campaign.write(producer_path, producer)
    row = {
        "dataset_id": "d" * 64,
        "embedding": values[:1536],
        "target_embedding": values[1536:],
        "metadata": {"task_type": "regression"},
    }
    record = {
        "task": "regression",
        "dataset_id": "d" * 64,
        "encoder_sha256": "e" * 64,
        "partition": "fit",
        "fitting_rows_sha256": "f" * 64,
        "lineage": context["lineage"],
        "context_encoding": "f32le/v1",
        "embedding_f32le_sha256": hashlib.sha256(struct.pack("<1536f", *row["embedding"])).hexdigest(),
        "target_embedding_f32le_sha256": hashlib.sha256(
            struct.pack("<512f", *row["target_embedding"])
        ).hexdigest(),
        "context_artifact": reference,
        "producer_receipt": campaign.artifact(producer_path),
    }
    campaign.validate_context(row, record, "e" * 64, "admitted")
    changed = copy.deepcopy(row)
    changed["embedding"][0] = 1.1
    with pytest.raises(ValueError, match="exact finite f32"):
        campaign.validate_context(changed, record, "e" * 64, "admitted")
    changed = copy.deepcopy(row)
    changed["embedding"][0] = 0.5
    with pytest.raises(ValueError, match="f32 bytes changed"):
        campaign.validate_context(changed, record, "e" * 64, "admitted")


def test_stopped_recovery_refuses_unproven_and_rejected_attempts(state):
    native = state / "native-invocations/fixture/request.json"
    campaign.write(state / "native-plan.json", {"fixture": True})
    campaign.write(
        native,
        {
            "schema": "jope.training-child/v1",
            "state": str(state),
            "nonce": "fixture",
            "native_plan_sha256": campaign.sha(state / "native-plan.json"),
        },
    )
    attempt = state / "regression/attempt-000001"
    attempt.mkdir(parents=True)
    started = {"started_unix_ms": 1000, "native_invocation": campaign.artifact(native)}
    with pytest.raises(FileNotFoundError):
        campaign.stopped_attempt(state, attempt, started)
    observed = {
        "schema": "jope.training-drained/v1",
        "populated": False,
        "request_sha256": campaign.sha(native),
        "ended_unix_ms": 1100,
    }
    for change in ({"populated": True}, {"request_sha256": "b" * 64}, {"ended_unix_ms": 999}):
        campaign.write(native.parent / "drained.json", {**observed, **change})
        with pytest.raises(ValueError, match="native stopped proof differs"):
            campaign.stopped_attempt(state, attempt, started)
    campaign.write(native.parent / "drained.json", observed)
    stop = campaign.read(campaign.checked(campaign.stopped_attempt(state, attempt, started)))
    assert stop["wall_seconds"] is None and stop["gpu_seconds"] is None
    campaign.write(attempt / "outcome.json", {"status": "proof_rejected"})
    with pytest.raises(ValueError, match="recoverable stopped outcome"):
        campaign.stopped_attempt(state, attempt, started)
    campaign.write(attempt / "rejection.json", {"status": "proof_rejected"})
    assert campaign.stopped_attempt(state, attempt, started) is None
