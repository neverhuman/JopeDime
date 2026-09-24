"""Explicit synthetic full-table training; never accepted checkpoint promotion."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

import pytest

from jope.grammars import all_grammars
from jope.policy import train_v3
from jope.policy.model_v3 import load_policy_v3
from jope.policy.tokens import build_action_vocab
from jope.policy.training_corpus_contract import load_frozen_action_vocab
from jope.schema import TrainingRow


def table():
    value = asdict(build_action_vocab(all_grammars()))
    value.pop("op_id_by_name")
    value["schema"] = "parallel.action_vocab.v1"
    return json.loads(json.dumps(value))


def rows():
    return [
        TrainingRow(
            dataset_id=f"explicit-synthetic-{split}-{index}",
            source="asm-synthetic-training-fixture",
            embedding=[0.25] * 1536,
            target_embedding=[1.0, 0.5] + [0.0] * 510,
            genome="(pipe X (ridge))",
            score=float(index),
            metadata={"split": split, "task_type": "regression", "provenance": "corpus"},
        )
        for split in ("train", "development")
        for index in range(2)
    ]


def test_frozen_table_never_rebuilds_token_ids_from_parsing_views(tmp_path, monkeypatch):
    data = table()
    (tmp_path / "active_vocab.json").write_text(json.dumps(data))

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen token IDs were rebuilt")

    monkeypatch.setattr(train_v3, "build_action_vocab", forbidden)
    monkeypatch.setattr(train_v3, "_vocab_for_training_rows", forbidden)
    specs, vocab, retained = train_v3.resolve_training_vocabulary(tmp_path / "rows.jsonl", rows())
    expected_specs, expected_vocab = load_frozen_action_vocab(data)
    assert retained == data
    assert vocab.to_json() == expected_vocab.to_json()
    assert [spec.to_json() for spec in specs] == [spec.to_json() for spec in expected_specs]
    assert vocab.engine_ids == expected_vocab.engine_ids
    assert vocab.task_ids == expected_vocab.task_ids
    assert vocab.head_tasks == expected_vocab.head_tasks
    with pytest.raises(ValueError, match="Prime cannot be appended"):
        train_v3.resolve_training_vocabulary(tmp_path / "rows.jsonl", rows(), force_prime=True)


def test_actual_full_table_optimizer_reload_and_continuation(tmp_path, monkeypatch):
    import torch

    before = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        train_v3,
        "policy_v3_validation_metrics",
        lambda *args, **kwargs: {"pairwise_win_accuracy": 0.5, "seq_nll": 1.0},
    )
    data = table()
    vocab_path = tmp_path / "active_vocab.json"
    vocab_path.write_text(json.dumps(data))
    checkpoint, resume = tmp_path / "policy.pt", tmp_path / "resume.pt"
    options = {
        "epochs": 1,
        "batch_size": 2,
        "device": "cpu",
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "ffn_dim": 16,
        "greedy_eval_datasets": 0,
        "checkpoint_state": "current",
        "phase_a_fraction": 0.0,
        "training_scope": "category:regression",
        "training_rows": rows(),
        "resume_checkpoint": resume,
    }
    try:
        first = train_v3.train_policy_v3_from_jsonl(tmp_path / "rows.jsonl", checkpoint, **options)
        original = torch.load(checkpoint, map_location="cpu", weights_only=True)
        loaded = load_policy_v3(original)
        train_v3.verify_training_proof(loaded, original["training_proof"])
        assert first["training_proof"]["optimizer_updates_this_run"] > 0
        assert loaded.frozen_action_vocab == data
        assert loaded.checkpoint_payload()["frozen_action_vocab"] == data
        assert original["training_contract"]["model_metadata"]["frozen_action_vocab"] == data
        assert original["action_vocab"] == {"tokens": data["tokens"], "version": data["version"]}

        second = train_v3.train_policy_v3_from_jsonl(
            tmp_path / "rows.jsonl", checkpoint, resume_from=resume, **options
        )
        current = torch.load(checkpoint, map_location="cpu", weights_only=True)
        retained = torch.load(resume, map_location="cpu", weights_only=False)
        train_v3.verify_training_proof(load_policy_v3(retained), retained["training_proof"])
        assert second["training_proof"]["optimizer_state_resumed"] is True
        assert (
            second["training_proof"]["initial_parameter_sha256"]
            == original["training_proof"]["parameter_sha256"]
        )
        assert current["training_proof"]["parameter_sha256"] != original["training_proof"]["parameter_sha256"]
        assert current["frozen_action_vocab"] == retained["frozen_action_vocab"] == data

        changed = copy.deepcopy(current)
        changed["action_vocab"]["version"] = "changed-version"
        with pytest.raises(ValueError, match="token IDs/version changed"):
            load_policy_v3(changed)
        changed = copy.deepcopy(current)
        changed["grammar_specs"][0]["max_transforms"] += 1
        with pytest.raises(ValueError, match="parsing views changed"):
            load_policy_v3(changed)
        changed_table = copy.deepcopy(data)
        changed_table["version"] = "changed-table-version"
        vocab_path.write_text(json.dumps(changed_table))
        with pytest.raises(ValueError, match="full action table differs"):
            train_v3.train_policy_v3_from_jsonl(
                tmp_path / "rows.jsonl", checkpoint, resume_from=resume, **options
            )
    finally:
        torch.set_num_threads(before)
