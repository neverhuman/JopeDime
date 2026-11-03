"""Shared policy, decoding, and scoring helpers for exact proposal generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path

import torch

from test_exact_common import sha256_file, sha256_text

DEFAULT_CATALOG = Path("/home/ubuntu/jope-flywheel/test_exact_gen0/catalog/current.json")
DEFAULT_CHECKPOINT = Path(
    "/home/ubuntu/jope-flywheel/training/round1/checkpoints/"
    "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592.model.pt"
)
DEFAULT_OUTPUT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0/proposals/dl-proposals.jsonl")
STRIDE_FLOATS = 1536 + 512
PROPOSAL_SELECTION_POLICY_SCHEMA = "JopeProposalSelectionPolicy.v1"
# Terminal-head allowlist mirroring the Rust reference task_valid
# (jope_selfplay_wave.rs): binary requires a classifier-capable head;
# regression accepts any head except the two classifier-only ones.
BINARY_HEADS = frozenset(
    ("logit_ridge", "logistic", "tab_boost", "extra_trees", "gam_boost", "portfolio")
)
CLASSIFIER_ONLY_HEADS = frozenset(("logit_ridge", "logistic"))
MULTICLASS_HEADS = frozenset(
    (
        "multiclass_ridge",
        "multiclass_ovr_logistic",
        "multiclass_diagonal_lda",
        "multiclass_nearest_centroid",
    )
)


def default_proposal_selection_policy() -> Path:
    bundled = Path(__file__).with_name("proposal-selection-policy.v1.json")
    if bundled.is_file():
        return bundled
    return (
        Path(__file__).resolve().parents[4]
        / "crates/jope-campaigns/manifests/selfplay/proposal-selection-policy.v1.json"
    )


def load_proposal_selection_policy(path: Path | None = None) -> tuple[dict, str]:
    path = path or default_proposal_selection_policy()
    value = json.loads(path.read_text())
    if value.get("schema") != PROPOSAL_SELECTION_POLICY_SCHEMA:
        raise RuntimeError("proposal selection policy schema is not v1")
    if value.get("proposal_budget") != 8 or value.get("invalid_slot_utility") != -1.0:
        raise RuntimeError("proposal selection policy budget/invalid-slot drift")
    routing = value.get("task_head_routing")
    if routing != {
        "binary": "binary",
        "multiclass": "binary",
        "regression": "regression",
        "timeseries": "regression",
    }:
        raise RuntimeError("proposal selection policy task routing drift")
    learned = value.get("learned_risk") or {}
    if learned.get("enabled") is not False or learned.get("calibration_sha256") is not None:
        raise RuntimeError("learned risk must remain disabled without calibrated evidence")
    required_coefficients = {
        "reward_dispersion",
        "instability",
        "static_risk",
        "instability_hard_threshold",
        "instability_hard_penalty",
        "complexity",
        "novelty",
        "latency",
        "calibrated_win",
        "calibrated_delta",
    }
    coefficients = value.get("coefficients")
    if not isinstance(coefficients, dict) or set(coefficients) != required_coefficients:
        raise RuntimeError("proposal selection policy coefficients are incomplete")
    if any(
        not isinstance(coefficient, (int, float))
        or isinstance(coefficient, bool)
        or not math.isfinite(float(coefficient))
        for coefficient in coefficients.values()
    ):
        raise RuntimeError("proposal selection policy coefficient is not finite")
    return value, sha256_file(path)


def head_valid_for_task(head, task):
    if task == "binary":
        return head in BINARY_HEADS
    if task == "multiclass":
        return head in MULTICLASS_HEADS
    if task in {"regression", "timeseries"}:
        return head not in CLASSIFIER_ONLY_HEADS and head not in MULTICLASS_HEADS
    return False


def rank_scored_proposals(proposals, requested):
    """Keep the global reward best at slot zero, then add conservative head diversity."""
    ranked = sorted(
        proposals,
        key=lambda row: (
            -float(row["selection_score"]),
            row["semantic_sha256"],
        ),
    )
    if not ranked:
        return []
    selected = [ranked[0]]
    selected_semantics = {ranked[0]["semantic_sha256"]}
    represented_heads = {ranked[0]["_head_op"]}
    for row in ranked[1:]:
        if len(selected) >= requested:
            break
        if row["_head_op"] in represented_heads:
            continue
        selected.append(row)
        selected_semantics.add(row["semantic_sha256"])
        represented_heads.add(row["_head_op"])
    for row in ranked[1:]:
        if len(selected) >= requested:
            break
        if row["semantic_sha256"] in selected_semantics:
            continue
        selected.append(row)
        selected_semantics.add(row["semantic_sha256"])
    return selected


def deterministic_normal(seed, dataset_hash, coordinate):
    """Batch-order-independent N(0,1) draw from a sealed SHA-256 stream."""

    pair = coordinate // 2
    digest = hashlib.sha256(
        f"train-only-redteam-jitter-v1\0{seed}\0{dataset_hash}\0{pair}".encode()
    ).digest()
    first = (int.from_bytes(digest[:8], "little") + 0.5) / 2**64
    second = (int.from_bytes(digest[8:16], "little") + 0.5) / 2**64
    radius = math.sqrt(-2.0 * math.log(first))
    angle = 2.0 * math.pi * second
    return radius * (math.cos(angle) if coordinate % 2 == 0 else math.sin(angle))


def filter_exact_dataset_manifest(entries, dataset_manifest, *, catalog_sha256):
    """Return exactly the train-only study contexts named by a sealed manifest."""

    value = json.loads(dataset_manifest.read_text())
    if value.get("schema") != "TrainOnlyProposalDatasetManifest.v1":
        raise RuntimeError("unsupported train-only proposal dataset manifest")
    if value.get("catalog_sha256") != catalog_sha256:
        raise RuntimeError("train-only proposal manifest catalog drift")
    requested = value.get("entries") or []
    identities = [
        (row.get("task"), row.get("dataset_hash"), row.get("context_sha256"))
        for row in requested
    ]
    if (
        not identities
        or len(identities) != len(set(identities))
        or any(task != "regression" for task, _, _ in identities)
    ):
        raise RuntimeError(
            "train-only proposal manifest needs distinct regression contexts"
        )
    by_identity = {
        (entry["task"], entry["dataset_hash"]): entry for entry in entries
    }
    selected = []
    for task, dataset_hash, context_sha256 in identities:
        entry = by_identity.get((task, dataset_hash))
        if entry is None or entry.get("context_sha256") != context_sha256:
            raise RuntimeError(
                f"train-only proposal context is absent or changed: {dataset_hash}"
            )
        selected.append(entry)
    return selected


def decode_batch(model, sequences, decoder, seed, task, validity):
    from jope.pipeline_text import format_pipeline_actions
    from jope.policy.tokens import decode_tokens

    rows = []
    for sequence in sequences:
        engine, actions = decode_tokens(sequence, model.vocab)
        if engine != "featmath":
            continue
        validity["decoded"] += 1
        validity[f"{task}:decoded"] = validity.get(f"{task}:decoded", 0) + 1
        if not head_valid_for_task(actions["head"]["op"], task):
            continue
        validity["kept"] += 1
        validity[f"{task}:kept"] = validity.get(f"{task}:kept", 0) + 1
        genome = format_pipeline_actions(actions)
        rows.append(
            {
                "task": task,
                "canonical_genome": genome,
                "semantic_sha256": sha256_text(genome),
                "source": "accepted_dl_checkpoint",
                "decoder": decoder,
                "decoder_seed": seed,
                "tokens": sequence,
                # Carried in-process only (popped before the row is written) so
                # the cross-dataset backfill pool can decide task-compatibility
                # from the decoder's own head op instead of re-parsing the
                # canonical text. Keeps the emitted row schema unchanged.
                "_head_op": actions["head"]["op"],
                "_actions": actions,
            }
        )
    return rows


def score_context_candidates(
    model,
    embedding,
    target,
    proposals,
    *,
    task,
    selection_policy,
    device,
    selector_calibration=None,
):
    from jope.policy.decode import candidate_risk_reason_v3
    from jope.policy.model_v3 import pad_sequences
    from jope.selector_calibration import score_candidate_with_calibration

    if not proposals:
        return []
    sequences = [row["tokens"] for row in proposals]
    if task not in {"regression", "binary", "multiclass", "timeseries"}:
        raise ValueError(f"unknown proposal task {task!r}")
    if any(row.get("task") != task for row in proposals):
        raise ValueError("proposal scoring batch mixes tasks")
    coefficients = selection_policy["coefficients"]
    with torch.no_grad():
        mean, std = model.score_token_sequences(
            embedding, target, sequences, device=device, task=task
        )
        token_ids, pad = pad_sequences(sequences, device=device)
        reward_context = model.reward_context(
            embedding.float().reshape(1, -1).to(device),
            target.float().reshape(1, -1).to(device),
        )
        p_bad = torch.sigmoid(
            model.instability_logit(
                token_ids,
                pad,
                reward_context.expand(token_ids.shape[0], -1),
                [task] * token_ids.shape[0],
            )
        ).cpu()
    for index, row in enumerate(proposals):
        reward_mean = float(mean[index])
        uncertainty = float(std[index])
        instability = float(p_bad[index])
        risk_reason = candidate_risk_reason_v3("featmath", row["_actions"])
        reward_lcb = reward_mean - float(coefficients["reward_dispersion"]) * uncertainty
        row.update(
            {
                "reward_mean": reward_mean,
                "reward_uncertainty": uncertainty,
                "reward_lcb": reward_lcb,
                "instability_probability": instability,
                "candidate_risk_reason": risk_reason,
            }
        )
        if selector_calibration is not None:
            calibration_row = {
                "engine": "featmath",
                "task_type": row.get("task"),
                "reward": reward_mean,
                "reward_mean": reward_mean,
                "reward_std": uncertainty,
                "p_bad": instability,
                "head_type": row["_head_op"],
            }
            row.update(
                score_candidate_with_calibration(
                    calibration_row, selector_calibration
                )
            )
        calibrated_win = row.get("calibrated_win_rate")
        calibrated_delta = row.get("calibrated_expected_delta")
        calibration_signal = 0.0
        if isinstance(calibrated_win, (int, float)):
            calibration_signal += float(coefficients["calibrated_win"]) * (
                float(calibrated_win) - 0.5
            )
        if isinstance(calibrated_delta, (int, float)) and math.isfinite(
            float(calibrated_delta)
        ):
            calibration_signal += float(coefficients["calibrated_delta"]) * math.tanh(
                float(calibrated_delta)
            )
        row["selection_score"] = (
            row["reward_lcb"]
            - float(coefficients["instability"]) * instability
            - (float(coefficients["static_risk"]) if risk_reason else 0.0)
            - (
                float(coefficients["instability_hard_penalty"])
                if instability > float(coefficients["instability_hard_threshold"])
                else 0.0
            )
            + calibration_signal
        )
        row["proposal_score"] = row["selection_score"]
    return proposals


