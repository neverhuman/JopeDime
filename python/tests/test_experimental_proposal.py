"""Pure boundaries plus separately collected actual CPU checkpoint proposal proof."""

from __future__ import annotations

import copy
import hashlib
import json
import struct

import pytest

from jope.policy import experimental_proposal as proposal


def artifact(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")))
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def training():
    return {
        "prepared_sha256": "a" * 64,
        "training_sha256": "b" * 64,
        "selected_training_rows_sha256": "c" * 64,
        "preprocessing_sha256": "d" * 64,
        "cut": 0,
    }


def request():
    ref = {"path": "/private/absent", "sha256": "e" * 64}
    return {
        "schema": proposal.REQUEST,
        "task": "binary",
        **{
            name: ref
            for name in (
                "export_manifest",
                "checkpoint",
                "vocabulary",
                "context",
                "generic_bank",
                "canonicalizer",
            )
        },
        "canonicalizer_source_sha40": "f" * 40,
        "runtime_sha256": "0" * 64,
        "training": training(),
        "data_schema": {},
        "neural_seed": 7,
        "sampling": {"count": 256, "temperature": 0.7, "top_p": 0.9, "max_len": 120},
    }


def test_request_refuses_labels_unknown_task_and_unbounded_sampling():
    proposal.validate_request(request())
    for key, value in (("heldout_labels", [0, 1]), ("task", "timeseries")):
        changed = {**request(), key: value}
        with pytest.raises(ValueError):
            proposal.validate_request(changed)
    for key, value in (
        ("count", 4097),
        ("temperature", float("nan")),
        ("top_p", 0),
        ("max_len", 121),
    ):
        changed = copy.deepcopy(request())
        changed["sampling"][key] = value
        with pytest.raises(ValueError):
            proposal.validate_request(changed)


def test_context_requires_exact_train_identity_lineage_width_and_f32_bytes(tmp_path):
    values = [0.25] * 2048
    context = {
        "lineage": proposal.campaign.CLASS_CONTEXT,
        "training": training(),
        "values": values,
        "context_sha256": hashlib.sha256(struct.pack("<2048f", *values)).hexdigest(),
        "classes": 2,
        "policy_compatible": False,
    }
    req = request()
    req["context"] = artifact(tmp_path / "context.json", context)
    assert proposal.load_context(req)["values"] == values
    for mutate in (
        lambda c: c.update(policy_compatible=True),
        lambda c: c.update(lineage="accepted-regression"),
        lambda c: c["values"].pop(),
        lambda c: c["training"].update(cut=1),
        lambda c: c.update(classes=3),
        lambda c: c["values"].__setitem__(0, 0.5),
        lambda c: c.update(labels=["renamed-a", "renamed-b"]),
    ):
        changed = copy.deepcopy(context)
        mutate(changed)
        req["context"] = artifact(tmp_path / "context.json", changed)
        with pytest.raises(ValueError):
            proposal.load_context(req)


def test_export_binds_exact_checkpoint_vocabulary_and_continuation_proof(tmp_path):
    req = request()
    (tmp_path / "binary").mkdir()
    req["checkpoint"] = artifact(tmp_path / "binary" / "policy.pt", {"fixture": True})
    req["vocabulary"] = artifact(tmp_path / "binary" / "vocabulary.json", {"fixture": True})
    plan = {
        "schema": proposal.campaign.PLAN,
        "lineage": proposal.campaign.LINEAGE,
        "tasks": ["binary"],
        "source_sha": req["canonicalizer_source_sha40"],
        "category_inputs": {"binary": {"inputs": {"vocabulary": req["vocabulary"]}}},
    }
    plan["plan_sha256"] = proposal.campaign.digest(plan)
    plan_ref = artifact(tmp_path / "plan.json", plan)
    verification = {
        "schema": "jope.training-verification/v1",
        "lineage": proposal.campaign.LINEAGE,
        "plan_sha256": plan["plan_sha256"],
        "completed_plan": True,
        "tasks": [{"task": "binary", "completed_passes": 2, "planned_passes": 2}],
    }
    proof_ref = artifact(tmp_path / "verification.json", verification)
    manifest = {
        "schema": "jope.experimental-training-export/v1",
        "lineage": proposal.campaign.LINEAGE,
        "serving_eligible": False,
        "artifacts": [
            {"path": "binary/policy.pt", "sha256": req["checkpoint"]["sha256"]},
            {"path": "binary/vocabulary.json", "sha256": req["vocabulary"]["sha256"]},
            {"path": "plan.json", "sha256": plan_ref["sha256"]},
            {"path": "verification.json", "sha256": proof_ref["sha256"]},
        ],
    }
    req["export_manifest"] = artifact(tmp_path / "manifest.json", manifest)
    actual_plan, record, proof_sha = proposal.export_inputs(req)
    assert actual_plan == plan and record["completed_passes"] == 2
    assert proof_sha == proof_ref["sha256"]
    changed = copy.deepcopy(req)
    changed["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint differs"):
        proposal.export_inputs(changed)
    changed = copy.deepcopy(req)
    changed["canonicalizer_source_sha40"] = "0" * 40
    with pytest.raises(ValueError, match="export plan task/lineage changed"):
        proposal.export_inputs(changed)
    verification["tasks"][0]["completed_passes"] = 1
    # Updating only the proof bytes cannot change the manifest's pinned evidence.
    proof_ref = artifact(tmp_path / "verification.json", verification)
    with pytest.raises(ValueError, match="artifact bytes changed"):
        proposal.export_inputs(req)
    manifest["artifacts"][-1]["sha256"] = proof_ref["sha256"]
    req["export_manifest"] = artifact(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="continuation proof incomplete"):
        proposal.export_inputs(req)


def results(count):
    return [{"recipe": {"executable_sha256": f"{i:064x}"}} for i in range(count)]


def test_origins_slots_dedup_and_bank_draw_identity_remain_honest():
    bank = {"candidates": [{"seed": i + 1000, "genome": str(i)} for i in range(100)]}
    valid = [
        {"program_index": 0, "tokens": [1], "reward_mean": 0.0, "reward_std": 0.1},
        {"program_index": 1, "tokens": [2], "reward_mean": 1.0, "reward_std": 0.2},
    ]
    records = results(102)
    records[1] = copy.deepcopy(records[0])
    candidates, count = proposal.assemble(valid, bank, records)
    assert count == 1 and len(candidates) == 80
    assert [row["slot"] for row in candidates] == list(range(80))
    assert candidates[0]["tokens"] == [2]
    assert candidates[0]["origin"] == "neural_policy"
    assert {row["origin"] for row in candidates[1:64]} == {"neural_rejection_fallback"}
    assert {row["origin"] for row in candidates[64:]} == {"random_exploration"}
    for row in candidates[1:]:
        assert row["candidate_seed"] == bank["candidates"][row["bank_index"]]["seed"]
        assert row["rank_log_weight"] is None
    assert len({row["recipe"]["executable_sha256"] for row in candidates}) == 80
    with pytest.raises(ValueError, match="cannot fill80"):
        proposal.assemble([], {"candidates": bank["candidates"][:79]}, results(79))


def test_all_rejected_samples_never_gain_neural_origin():
    bank = {"candidates": [{"seed": i, "genome": str(i)} for i in range(80)]}
    rows, count = proposal.assemble([], bank, results(80))
    assert count == 0 and all(row["origin"] != "neural_policy" for row in rows)


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_actual_trained_classifier_reload_sampling_uses_task_head(tmp_path, monkeypatch, task):
    """Tiny real optimizer/load/decoder proof; CPU-only, no fit or serving claim."""
    import torch

    from jope.grammars import all_grammars
    from jope.grammars.vocab import build_active_vocab_v3
    from jope.policy import train_v3
    from jope.policy.decode import sample_sequences
    from jope.policy.tokens import decode_tokens
    from jope.schema import TrainingRow

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        train_v3,
        "policy_v3_validation_metrics",
        lambda *a, **k: {"pairwise_win_accuracy": 0.5, "seq_nll": 1.0},
    )
    head = {"regression": "ridge", "binary": "logit_ridge", "multiclass": "multiclass_ridge"}[task]
    rows = [
        TrainingRow(
            dataset_id=f"synthetic-{split}-{index}",
            source="asm-synthetic-training-fixture",
            embedding=[0.25] * 1536,
            target_embedding=[0.125] * 512,
            genome=f"(pipe X ({head}))",
            score=float(index),
            metadata={"split": split, "task_type": task, "provenance": "corpus"},
        )
        for split in ("train", "development")
        for index in range(2)
    ]
    corpus = tmp_path / "corpus.jsonl"
    vocab_path = tmp_path / "active_vocab.json"
    vocab_path.write_text(json.dumps(build_active_vocab_v3([], all_grammars())))
    checkpoint, resume = tmp_path / "policy.pt", tmp_path / "resume.pt"
    options = dict(
        training_rows=rows,
        epochs=1,
        batch_size=2,
        device="cpu",
        d_model=8,
        n_layers=1,
        n_heads=2,
        ffn_dim=16,
        phase_a_fraction=0.0,
        greedy_eval_datasets=0,
        checkpoint_state="current",
        resume_checkpoint=resume,
        training_scope=f"category:{task}",
    )
    try:
        train_v3.train_policy_v3_from_jsonl(corpus, checkpoint, **options)
        train_v3.train_policy_v3_from_jsonl(corpus, checkpoint, resume_from=resume, **options)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        binding_plan = {
            "plan_sha256": "a" * 64,
            "category_inputs": {task: {"data_kind": "synthetic_fixture"}},
        }
        payload["asm_training"] = {
            "plan_sha256": "a" * 64,
            "lineage": proposal.campaign.LINEAGE,
            "task": task,
            "data_kind": "synthetic_fixture",
        }
        torch.save(payload, checkpoint)
        req = request()
        req["task"] = task
        req["checkpoint"] = {
            "path": str(checkpoint.resolve()),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
        req["vocabulary"] = {
            "path": str(vocab_path.resolve()),
            "sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        }
        context = {"values": [0.25] * 1536 + [0.125] * 512}
        record = {
            "parameter_sha256": payload["training_proof"]["parameter_sha256"],
            "optimizer_updates": payload["training_proof"]["optimizer_updates_total"],
        }
        model, _ = proposal.strict_model(req, context, binding_plan, record)
        train_v3.verify_training_proof(model, payload["training_proof"])
        assert payload["training_proof"]["optimizer_state_resumed"]
        assert model.target_keep_dim == 512
        embedding, target = torch.tensor([0.25] * 1536), torch.tensor([0.125] * 512)
        sequences = sample_sequences(
            model,
            embedding,
            target,
            n=4,
            engine="featmath",
            task=task,
            seed=123,
            max_len=120,
            device="cpu",
        )
        for sequence in sequences:
            engine, actions = decode_tokens(sequence, model.vocab)
            assert engine == "featmath" and actions["task"] == task
        # Every task, including regression, must route its own reward lane.
        seen = []
        original = model.reward_stats

        def capture_reward_stats(z, token_ids, pad, task_types):
            seen.append(task_types)
            return original(z, token_ids, pad, task_types)

        monkeypatch.setattr(model, "reward_stats", capture_reward_stats)
        mean, deviation = model.score_token_sequences(embedding, target, sequences, task=task)
        assert seen == [[task] * len(sequences)]
        assert bool(torch.isfinite(mean).all() and torch.isfinite(deviation).all())
        # The general loader allows old missing heads; this experimental consumer may not.
        damaged = copy.deepcopy(payload)
        del damaged["model"][
            next(
                key
                for key in damaged["model"]
                if key.startswith("reward_heads." if task == "regression" else "binary_reward_heads.")
            )
        ]
        torch.save(damaged, checkpoint)
        req["checkpoint"]["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="missing or extra state"):
            proposal.strict_model(req, context, binding_plan, record)
    finally:
        torch.set_num_threads(previous_threads)


def test_regression_context_refuses_classifier_lineage_or_classes(tmp_path):
    req = request()
    req["task"] = "regression"
    values = [0.25] * 2048
    context = {
        "lineage": proposal.CONTEXTS["regression"],
        "training": training(),
        "values": values,
        "context_sha256": proposal.sha_bytes(struct.pack("<2048f", *values)),
        "classes": 0,
        "policy_compatible": False,
    }
    req["context"] = artifact(tmp_path / "regression-context.json", context)
    assert proposal.load_context(req) == context
    for changed in (
        {**context, "classes": 2},
        {**context, "lineage": proposal.CONTEXTS["binary"]},
        {**context, "policy_compatible": True},
    ):
        req["context"] = artifact(tmp_path / "regression-context.json", changed)
        with pytest.raises(ValueError):
            proposal.load_context(req)


def test_experimental_architecture_refuses_implicit_defaults_and_reinterpretation():
    payload = {
        "model_version": "fixture-only",
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "ffn_dim": 16,
        "dataset_dim": 1536,
        "target_dim": 512,
        "reward_ensemble_k": 4,
        "embedding_rows": 100,
        "routing_classes": 0,
        "fused_task_heads": True,
        "target_keep_dim": 512,
        "split_reward_context": False,
        "input_block_norm": False,
        "dataset_block_rms": 1.0,
        "target_block_rms": 1.0,
        "context_schema": {},
        "task_head_schema": {},
    }
    assert proposal.architecture(payload) == payload
    for key in proposal.ARCHITECTURE_FIELDS:
        changed = dict(payload)
        del changed[key]
        with pytest.raises(ValueError, match="incomplete experimental architecture"):
            proposal.architecture(changed)
    for key, value in (
        ("n_heads", True),
        ("routing_classes", -1),
        ("split_reward_context", 0),
        ("target_block_rms", float("nan")),
    ):
        with pytest.raises(ValueError):
            proposal.architecture({**payload, key: value})
