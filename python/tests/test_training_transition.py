# ruff: noqa: F811
# Pytest fixtures imported from the existing contract suite.
"""Actual tiny optimizer transitions; these are CPU mechanics, not quality evidence."""

import copy
import json

import pytest
import torch
from test_policy_training_contract import small_trainer, synthetic_rows  # noqa: F401

from jope.policy import train_v3
from jope.policy import training_transition as transition


def corpus(path, rows):
    path.write_text("".join(json.dumps(row.to_json()) + "\n" for row in rows))
    return path


def test_explicit_transition_preserves_moments_and_rejects_invariant_drift(
    tmp_path, small_trainer, monkeypatch
):
    options = {**small_trainer, "strict_resume": True, "amp": False}
    old = corpus(tmp_path / "old.jsonl", synthetic_rows())
    parent = tmp_path / "parent.pt"
    train_v3.train_policy_v3_from_jsonl(old, tmp_path / "policy.pt", resume_checkpoint=parent, **options)
    before = parent.read_bytes()
    saved = torch.load(parent, weights_only=False)
    rows = synthetic_rows()
    rows[0].score = 17.0
    new = corpus(tmp_path / "new.jsonl", rows)
    batches = [[0, 1], [0, 1]]
    manifest = transition.prepare(parent, new, batches)
    # Ordinary continuation continues to reject an undeclared corpus change.
    with pytest.raises(ValueError, match="contract differs"):
        train_v3.train_policy_v3_from_jsonl(
            new,
            tmp_path / "bad.pt",
            resume_from=parent,
            resume_checkpoint=tmp_path / "bad-resume.pt",
            batch_schedule=batches,
            **options,
        )
    with pytest.raises(ValueError, match="invariant contract"):
        train_v3.train_policy_v3_from_jsonl(
            new,
            tmp_path / "bad.pt",
            resume_from=parent,
            resume_checkpoint=tmp_path / "bad-resume.pt",
            batch_schedule=batches,
            corpus_transition=manifest,
            **{**options, "lr": 0.123},
        )
    result = train_v3.train_policy_v3_from_jsonl(
        new,
        tmp_path / "next.pt",
        resume_from=parent,
        resume_checkpoint=tmp_path / "next-resume.pt",
        batch_schedule=batches,
        corpus_transition=manifest,
        **options,
    )
    assert result["training_proof"]["optimizer_state_resumed"]
    assert result["training_proof"]["optimizer_updates_this_run"] == 2
    after = torch.load(tmp_path / "next-resume.pt", weights_only=False)
    assert after["consumed_batches"][0]["row_indices"] == [0, 1]
    assert sum(b["nonpadding_tokens"] for b in after["consumed_batches"]) > 0
    assert parent.read_bytes() == before
    for key, state in saved["optimizer"]["state"].items():
        assert after["optimizer"]["state"][key]["step"] == state["step"] + 2
    bad = copy.deepcopy(manifest)
    bad["parent_optimizer_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed parent"):
        transition.validate(bad, saved, saved["training_contract"], parent, new, batches)

    # Interrupt after the last update but before epoch/controller completion.
    # The new attempt must retain that boundary even when no batches remain.
    original_save = train_v3._atomic_torch_save

    def interrupt_last(payload, path):
        original_save(payload, path)
        progress = payload.get("strict_progress")
        if progress and progress["sampler"]["next_batch"] == len(batches):
            raise KeyboardInterrupt("fixture: last update committed")

    with monkeypatch.context() as stopped:
        stopped.setattr(train_v3, "_atomic_torch_save", interrupt_last)
        with pytest.raises(KeyboardInterrupt):
            train_v3.train_policy_v3_from_jsonl(
                new,
                tmp_path / "stopped.pt",
                resume_from=parent,
                resume_checkpoint=tmp_path / "stopped-resume.pt",
                batch_schedule=batches,
                corpus_transition=manifest,
                **options,
            )
    train_v3.train_policy_v3_from_jsonl(
        new,
        tmp_path / "recovered.pt",
        resume_from=tmp_path / "stopped-resume.pt",
        resume_checkpoint=tmp_path / "recovered-resume.pt",
        batch_schedule=batches,
        **options,
    )
    recovered = torch.load(tmp_path / "recovered-resume.pt", weights_only=False)
    assert transition.state_digest(recovered["model"]) == transition.state_digest(after["model"])
    assert transition.state_digest(recovered["optimizer"]) == transition.state_digest(after["optimizer"])
    assert recovered["consumed_batches"] == after["consumed_batches"]


def test_fresh_teacher_does_not_relabel_replay_or_routing():
    reference = synthetic_rows()
    mixed = copy.deepcopy(reference)
    fresh = copy.deepcopy(reference[0])
    fresh.genome = "(pipe X (elasticnet))"
    fresh.score = 100.0
    mixed.insert(2, fresh)
    control = copy.deepcopy(reference)
    parent = {"routing_families": ["ridge", "elasticnet"], "routing_classes": 3}
    transition.freeze_supervision(mixed, copy.deepcopy(reference), parent)
    transition.freeze_supervision(control, copy.deepcopy(reference), parent)
    assert transition.supervision_snapshot(mixed[:2]) == transition.supervision_snapshot(control[:2])
    assert [row.metadata["oof_delta_to_incumbent"] for row in mixed[:3]] == [-1.0, 0.0, 99.0]
    assert [row.metadata["routing_class"] for row in mixed[:3]] == [0, 0, 0]
    assert [row.score for row in mixed[:3]] == [0.0, 1.0, 100.0]
