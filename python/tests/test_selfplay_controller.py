from __future__ import annotations

import json
import hashlib
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest


from test_exact_controller import (  # noqa: E402
    PHASES,
    ActionJournal,
    FlywheelController,
    PendingAction,
    action_key,
    default_handlers,
)
from test_exact_packed_corpus import materialize, verify  # noqa: E402
import test_exact_packed_corpus as packed_corpus  # noqa: E402
import test_exact_train as exact_train  # noqa: E402
from migrate_jain_web_cpu import cpu_payload  # noqa: E402


def catalog_entries():
    return [
        {"task": task, "dataset_hash": f"{task}-{index}"}
        for task, count in (("regression", 2_322), ("binary", 542))
        for index in range(count)
    ]


def test_production_controller_owns_deterministic_post_training_phases():
    handlers = default_handlers()
    expected = {
        "build_acceptance_evidence": "_build_acceptance_evidence",
        "run_checkpoint_gate": "_run_checkpoint_gate",
        "advance_accepted_pointer": "_advance_accepted_pointer",
        "seed_next_strict_wave": "_seed_next_strict_wave",
        "freeze_winner_scan": "_freeze_winner_scan",
    }
    assert {
        phase: handlers[phase].__name__ for phase in expected
    } == expected
    assert handlers["generate_gate_proposals"].__name__ == "_generate_gate_proposals"
    assert handlers["evaluate_frozen_gate"].__name__ == "_evaluate_frozen_gate"
    assert (
        handlers["run_sotarena_qualification"].__name__
        == "_run_sotarena_qualification"
    )


def snapshot(labels=50_000):
    return {
        "valid_labels": labels,
        "accepted_ancestor": "a" * 64,
        "catalog_sha256": "b" * 64,
        "catalog_entries": catalog_entries(),
        "catalog_counts": {"regression": 2_322, "binary": 542},
        "evaluator_sha256": "c" * 64,
    }


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
        json.loads(line)
        for line in (tmp_path / "ops/controller-events.jsonl").read_text().splitlines()
    ]
    assert [row["status"] for row in events] == [
        "started",
        "pending",
        "started",
        "succeeded",
    ]
    journal.close()


@pytest.mark.parametrize("gate_pass,expected", [(True, "accepted"), (False, "rejected")])
def test_controller_resumes_each_phase_and_gate_is_fail_closed(
    tmp_path, gate_pass, expected
):
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
    broken["catalog_entries"].append(
        {"task": "regression", "dataset_hash": "unexpected"}
    )
    with pytest.raises(ValueError, match="catalog count mismatch"):
        controller.cycle(broken)
    controller.close()


def _training_row(dataset, task, score, split):
    return {
        "dataset_id": dataset,
        "source": "test",
        "canonical_genome": "(pipe X (ridge))",
        "engine": "test",
        "valid": True,
        "metric": "r2" if task == "regression" else "auc",
        "score": score,
        "cost_ms": 12,
        "metadata": {
            "task_type": task,
            "dataset_hash": dataset,
            "semantic_plan_hash": "d" * 64,
            "receipt_sha256": "e" * 64,
            "model_checkpoint_sha256": "f" * 64,
            "generation": "strict-wave-000001",
            "test_adaptive": True,
            "benchmark_valid": False,
            "selection_bias": "test-exact-reused",
            "split": split,
        },
    }


def test_packed_corpus_is_content_bound_reversible_and_parity_checked(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text(json.dumps(_training_row("r", "regression", 0.2, "train")) + "\n")
    validation.write_text(
        json.dumps(_training_row("b", "binary", 0.8, "validation")) + "\n"
    )
    data = tmp_path / "contexts.f32"
    np.arange(2 * 2_048, dtype="<f4").tofile(data)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "entries": [
                    {"task": "regression", "dataset_hash": "r", "catalog_index": 0},
                    {"task": "binary", "dataset_hash": "b", "catalog_index": 1},
                ]
            }
        )
    )
    import hashlib

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    catalog_pointer = tmp_path / "current.json"
    catalog_pointer.write_text(
        json.dumps(
            {
                "catalog": str(catalog),
                "catalog_sha256": digest(catalog),
                "data": str(data),
                "data_sha256": digest(data),
            }
        )
    )
    source = tmp_path / "corpus-manifest.json"
    source.write_text(
        json.dumps(
            {
                "train": str(train),
                "train_sha256": digest(train),
                "validation": str(validation),
                "validation_sha256": digest(validation),
                "catalog_sha256": digest(catalog),
                "source_label_watermark": 2,
            }
        )
    )
    packed = materialize(
        source,
        tmp_path / "packed",
        tokenizer=lambda row: [1, 2] if row["dataset_id"] == "r" else [3],
        catalog_pointer_path=catalog_pointer,
    )
    root = tmp_path / "packed" / packed["binding_sha256"]
    verify(packed, root)
    assert packed["row_count"] == 2
    assert packed["task_counts"] == {"regression": 1, "binary": 1}
    assert np.load(root / "token_offsets.npy").tolist() == [0, 2, 3]
    assert not (root / "observations.jsonl").exists()
    assert (root / packed["selected_row_index"]["path"]).stat().st_size == 2 * 32
    assert sum(
        len((root / shard["path"]).read_text().splitlines())
        for shard in packed["label_shards"]
    ) == 2
    staged_source = tmp_path / "staged-corpus-manifest.json"
    staged_train = tmp_path / "staged-train.jsonl"
    staged_validation = tmp_path / "staged-validation.jsonl"
    staged_source.write_bytes(source.read_bytes())
    staged_train.write_bytes(train.read_bytes())
    staged_validation.write_bytes(validation.read_bytes())
    train.write_text("{}\n")
    # The packed artifact is a self-contained runtime input: ordinary training
    # verifies the immutable source-hash binding without reopening/staging the
    # multi-gigabyte JSONL files.
    verify(packed, root)
    with pytest.raises(ValueError, match="source train hash mismatch"):
        verify(
            packed,
            root,
            source_paths={"train": train, "validation": validation},
        )
    verify(
        packed,
        root,
        source_manifest_path=staged_source,
        source_paths={
            "train": staged_train,
            "validation": staged_validation,
        },
    )


def test_packing_uses_frozen_catalog_binding_after_live_pointer_advances(tmp_path):
    import hashlib

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text(json.dumps(_training_row("r", "regression", 0.2, "train")) + "\n")
    validation.write_text(
        json.dumps(_training_row("b", "binary", 0.8, "validation")) + "\n"
    )
    catalog = tmp_path / "catalog-a.json"
    catalog.write_text(
        json.dumps(
            {
                "entries": [
                    {"task": "regression", "dataset_hash": "r", "catalog_index": 0},
                    {"task": "binary", "dataset_hash": "b", "catalog_index": 1},
                ]
            }
        )
    )
    data = tmp_path / "contexts-a.f32"
    np.arange(2 * 2_048, dtype="<f4").tofile(data)
    pointer_snapshot = tmp_path / "pointer-a.json"
    pointer_snapshot.write_text("{}")
    binding = tmp_path / "catalog-binding.json"
    binding.write_text(
        json.dumps(
            {
                "schema": "CatalogBinding.v1",
                "catalog_path": str(catalog),
                "catalog_sha256": digest(catalog),
                "data_path": str(data),
                "data_sha256": digest(data),
                "pointer_snapshot_path": str(pointer_snapshot),
                "pointer_snapshot_sha256": digest(pointer_snapshot),
                "created_at_ms": 1,
            },
            sort_keys=True,
        )
    )
    source = tmp_path / "corpus-manifest.json"
    source.write_text(
        json.dumps(
            {
                "train": str(train),
                "train_sha256": digest(train),
                "validation": str(validation),
                "validation_sha256": digest(validation),
                "catalog_sha256": digest(catalog),
                "catalog_binding": str(binding),
                "catalog_binding_sha256": digest(binding),
                "source_label_watermark": 2,
            }
        )
    )
    live_pointer = tmp_path / "current.json"
    live_pointer.write_text(
        json.dumps(
            {
                "catalog": "/advanced/catalog.json",
                "catalog_sha256": "f" * 64,
                "data": "/advanced/contexts.f32",
                "data_sha256": "e" * 64,
            }
        )
    )

    packed = materialize(
        source,
        tmp_path / "packed",
        tokenizer=lambda row: [1],
        catalog_pointer_path=live_pointer,
        catalog_binding_path=binding,
    )
    assert packed["catalog_sha256"] == digest(catalog)
    assert packed["catalog_binding_sha256"] == digest(binding)


def test_packed_v3_reuses_content_addressed_label_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(packed_corpus, "LABEL_SHARD_ROWS", 1)
    train = tmp_path / "train.jsonl"
    validation_a = tmp_path / "validation-a.jsonl"
    validation_b = tmp_path / "validation-b.jsonl"
    train.write_text(json.dumps(_training_row("r", "regression", 0.2, "train")) + "\n")
    validation_a.write_text(
        json.dumps(_training_row("b-a", "binary", 0.8, "validation")) + "\n"
    )
    validation_b.write_text(
        json.dumps(_training_row("b-b", "binary", 0.7, "validation")) + "\n"
    )
    data = tmp_path / "contexts.f32"
    np.arange(3 * 2_048, dtype="<f4").tofile(data)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "entries": [
                    {"task": "regression", "dataset_hash": "r", "catalog_index": 0},
                    {"task": "binary", "dataset_hash": "b-a", "catalog_index": 1},
                    {"task": "binary", "dataset_hash": "b-b", "catalog_index": 2},
                ]
            }
        )
    )

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    pointer = tmp_path / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "catalog": str(catalog),
                "catalog_sha256": digest(catalog),
                "data": str(data),
                "data_sha256": digest(data),
            }
        )
    )
    manifests = []
    for index, validation in enumerate((validation_a, validation_b), 1):
        manifest = tmp_path / f"corpus-{index}.json"
        manifest.write_text(
            json.dumps(
                {
                    "train": str(train),
                    "train_sha256": digest(train),
                    "validation": str(validation),
                    "validation_sha256": digest(validation),
                    "catalog_sha256": digest(catalog),
                    "source_label_watermark": 2,
                }
            )
        )
        manifests.append(manifest)
    first = materialize(
        manifests[0],
        tmp_path / "packed",
        tokenizer=lambda row: [1],
        catalog_pointer_path=pointer,
    )
    second = materialize(
        manifests[1],
        tmp_path / "packed",
        tokenizer=lambda row: [1],
        catalog_pointer_path=pointer,
    )
    assert first["label_shards"][0]["sha256"] == second["label_shards"][0]["sha256"]
    assert second["label_shards"][0]["reused"] is True


def test_packed_corpus_keeps_only_complete_receipt_exact_training_pairs(tmp_path):
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    pair_id = "a" * 64
    parent = "b" * 64
    direct = _training_row("r", "regression", 0.55, "train")
    comparator = _training_row("r", "regression", 0.50, "train")
    for row, lane, semantic, receipt in (
        (direct, "native_dl_direct", "1" * 64, "3" * 64),
        (comparator, "matched_gp_comparator", "2" * 64, "4" * 64),
    ):
        row["metadata"].update(
            {
                "semantic_plan_hash": semantic,
                "receipt_sha256": receipt,
                "causal_pair_id": pair_id,
                "causal_lane": lane,
                "causal_pair_index": 0,
                "causal_accepted_parent_checkpoint_sha256": parent,
                "causal_wave": 205,
            }
        )
    direct["metadata"].update(
        {
            "descendant_credit_receipt_sha256": "5" * 64,
            "descendant_horizon": 4,
            "descendant_censored": False,
        }
    )
    comparator["metadata"]["validity_hard_negative"] = True
    validation_row = _training_row("b", "binary", 0.8, "validation")
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text(
        "\n".join(json.dumps(row) for row in (direct, comparator)) + "\n"
    )
    validation.write_text(json.dumps(validation_row) + "\n")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "task": "regression",
                        "dataset_hash": "r",
                        "catalog_index": 0,
                    },
                    {"task": "binary", "dataset_hash": "b", "catalog_index": 1},
                ]
            }
        )
    )
    data = tmp_path / "contexts.f32"
    np.arange(2 * 2_048, dtype="<f4").tofile(data)
    pointer = tmp_path / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "catalog": str(catalog),
                "catalog_sha256": digest(catalog),
                "data": str(data),
                "data_sha256": digest(data),
            }
        )
    )
    source = tmp_path / "corpus-manifest.json"
    source.write_text(
        json.dumps(
            {
                "train": str(train),
                "train_sha256": digest(train),
                "validation": str(validation),
                "validation_sha256": digest(validation),
                "catalog_sha256": digest(catalog),
                "source_label_watermark": 3,
            }
        )
    )
    packed = materialize(
        source,
        tmp_path / "packed",
        tokenizer=lambda row: [1],
        catalog_pointer_path=pointer,
    )
    root = tmp_path / "packed" / packed["binding_sha256"]
    verify(packed, root)
    assert packed["causal_pair_count"] == 1
    pair = json.loads((root / "causal_pairs.jsonl").read_text())
    assert pair["pair_id"] == pair_id
    assert (pair["direct_ordinal"], pair["comparator_ordinal"]) == (0, 1)
    assert pair["hard"] is True
    assert pair["novel"] is True
    recipe_e_rows = [
        json.loads(line)
        for line in (root / "recipe_e_evidence.jsonl").read_text().splitlines()
    ]
    assert [row["lane"] for row in recipe_e_rows] == [
        "completed_descendant_trajectories",
        "validity_instability_hard_negatives",
    ]
    assert packed["recipe_e_evidence_stats"]["task_counts"]["regression"] == {
        "completed_descendant_trajectories": 1,
        "validity_instability_hard_negatives": 1,
        "completed_descendant_receipts": 1,
    }


def test_causal_training_members_require_successful_receipt_exact_pair(
    tmp_path, monkeypatch
):
    root = tmp_path / "runtime"
    wave = root / "waves/strict-wave-000205"
    wave.mkdir(parents=True)
    parent = "b" * 64
    (wave / "summary.json").write_text(
        json.dumps(
            {
                "immutable": True,
                "checkpoint_sha256": parent,
                "wave_summary": {
                    "evidence_eligibility": {
                        "promotion": True,
                        "recovery_accounting_only": False,
                    }
                },
            }
        )
    )
    monkeypatch.setattr(exact_train, "ROOT", root)
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE causal_evaluation_receipts_v2(
            receipt_sha256 TEXT,
            candidate_sha256 TEXT,
            task TEXT,
            context_identity TEXT,
            accepted_parent_checkpoint_sha256 TEXT,
            wave INTEGER,
            lane TEXT,
            pair_index INTEGER,
            pair_id TEXT,
            score REAL,
            success INTEGER,
            proposal_valid INTEGER,
            ancestry TEXT,
            cross_context_backfill INTEGER,
            valid_lineage INTEGER,
            canary INTEGER
        )
        """
    )
    pair_id = "a" * 64
    common = ("regression", "context", parent, 205, 0, pair_id)
    connection.executemany(
        """
        INSERT INTO causal_evaluation_receipts_v2 VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        [
            (
                "1" * 64,
                "4" * 64,
                *common[:4],
                "native_dl_direct",
                *common[4:],
                None,
                0,
                1,
                "accepted_dl",
                0,
                1,
            ),
            (
                "2" * 64,
                "4" * 64,
                *common[:4],
                "native_dl_direct",
                *common[4:],
                0.55,
                1,
                1,
                "accepted_dl",
                0,
                1,
            ),
            (
                "3" * 64,
                "5" * 64,
                *common[:4],
                "matched_gp_comparator",
                *common[4:],
                0.50,
                1,
                1,
                "none",
                0,
                1,
            ),
        ],
    )

    members, stats = exact_train.complete_causal_members(connection)
    assert ("1" * 64, "4" * 64) not in members
    assert set(members) == {
        ("2" * 64, "4" * 64),
        ("3" * 64, "5" * 64),
    }
    assert stats["complete_pairs"] == 1
    assert stats["task_counts"] == {"regression": 1}


def test_corrected_ingest_requires_hash_bound_prediction_evidence(
    tmp_path, monkeypatch
):
    import test_exact_ingest as ingest
    from test_exact_common import selfplay_authority, sha256_text

    monkeypatch.setattr(ingest, "CORPUS", tmp_path / "corpus.sqlite")
    output = tmp_path / "receipt"
    predictions = output / "predictions/prediction.f64le"
    predictions.parent.mkdir(parents=True)
    prediction_bytes = struct.pack("<dd", 0.25, 0.75)
    predictions.write_bytes(prediction_bytes)
    genome = "(pipe X (ridge))"
    semantic = sha256_text(genome)
    authority = selfplay_authority(
        task="regression",
        wave="strict-wave-000001",
        frozen_seed_sha256="1" * 64,
        model_checkpoint_sha256="2" * 64,
        hall_of_fame_sha256="3" * 64,
    )
    manifest = {
        "task": "regression",
        "dataset_hash": "dataset",
        "train_sha256": "4" * 64,
        "test_sha256": "5" * 64,
        "candidates": [
            {
                "semantic_sha256": semantic,
                "canonical_genome": genome,
                "source": "accepted_dl_checkpoint",
            }
        ],
        "label_authority": authority,
    }
    (output / ".portable-manifest.json").write_text(json.dumps(manifest))
    (output / "SHARD_OK").write_text("ok\n")
    receipt = {
        "task": "regression",
        "dataset_hash": "dataset",
        "semantic_sha256": semantic,
        "canonical_genome": genome,
        "tier": "test_exact",
        "metric": "r2",
        "score": 0.4,
        "secondary_metric": "rmse",
        "secondary_score": 0.5,
        "valid": True,
        "runtime_seconds": 1.0,
        "train_sha256": "4" * 64,
        "test_sha256": "5" * 64,
        "manifest_sha256": "6" * 64,
        "created_at_ms": 7,
        "predictions_path": "predictions/prediction.f64le",
        "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "prediction_count": 2,
        "balanced_accuracy": None,
        **{
            "test_adaptive": authority["test_adaptive"],
            "benchmark_valid": authority["benchmark_valid"],
            "selection_bias": authority["selection_bias"],
            "generation": authority["generation"],
            "frozen_seed_sha256": authority["frozen_seed_sha256"],
            "model_checkpoint_sha256": authority["model_checkpoint_sha256"],
            "hall_of_fame_sha256": authority["hall_of_fame_sha256"],
        },
    }
    receipt_path = output / "dataset.receipts.jsonl"
    receipt_path.write_text(json.dumps(receipt) + "\n")
    database = ingest.corpus_db()
    predictions.write_bytes(prediction_bytes[:-1])
    with pytest.raises(RuntimeError, match="row alignment drift"):
        ingest.validate_and_ingest("job", output, database)
    predictions.write_bytes(prediction_bytes)
    result = ingest.validate_and_ingest("job", output, database)
    database.commit()
    stored = database.execute(
        "SELECT predictions_sha256,prediction_count,strict_wave FROM observations"
    ).fetchone()
    database.close()
    assert result == {"rows": 1, "valid": 1, "failed": 0}
    assert tuple(stored) == (
        hashlib.sha256(prediction_bytes).hexdigest(),
        2,
        "strict-wave-000001",
    )


def test_jain_cpu_clone_removes_every_gpu_admission_path():
    source = {
        "Config": {
            "Image": "jain",
            "Env": [
                "SAFE=value",
                "NVIDIA_VISIBLE_DEVICES=all",
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            ],
        },
        "Image": "sha256:" + "a" * 64,
        "HostConfig": {
            "DeviceRequests": [{"Capabilities": [["gpu"]]}],
            "Devices": [
                {"PathOnHost": "/dev/nvidia0"},
                {"PathOnHost": "/dev/safe-device"},
            ],
            "Runtime": "nvidia",
        },
    }
    payload = cpu_payload(source)
    assert payload["HostConfig"]["DeviceRequests"] == []
    assert payload["HostConfig"]["Devices"] == [
        {"PathOnHost": "/dev/safe-device"}
    ]
    assert payload["HostConfig"]["Runtime"] == "runc"
    assert payload["Env"] == [
        "SAFE=value",
        "NVIDIA_VISIBLE_DEVICES=void",
        "JAIN_DISABLE_GPU_EXEC=1",
    ]
