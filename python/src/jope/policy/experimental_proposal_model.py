"""Pinned architecture validation for experimental policy checkpoints."""

import math

from .experimental_proposal_inputs import require

ARCHITECTURE_FIELDS = (
    "model_version",
    "d_model",
    "n_layers",
    "n_heads",
    "ffn_dim",
    "dataset_dim",
    "target_dim",
    "reward_ensemble_k",
    "embedding_rows",
    "routing_classes",
    "fused_task_heads",
    "target_keep_dim",
    "split_reward_context",
    "input_block_norm",
    "dataset_block_rms",
    "target_block_rms",
    "context_schema",
    "task_head_schema",
)


def architecture(payload):
    # The exact checkpoint SHA pins these fields; refuse the general loader's
    # historical defaults or shape-compatible reinterpretation on this route.
    require(all(key in payload for key in ARCHITECTURE_FIELDS), "incomplete experimental architecture")
    for key in (
        "d_model",
        "n_layers",
        "n_heads",
        "ffn_dim",
        "dataset_dim",
        "target_dim",
        "reward_ensemble_k",
        "embedding_rows",
        "target_keep_dim",
    ):
        require(type(payload[key]) is int and payload[key] > 0, "invalid experimental architecture")
    require(
        type(payload["routing_classes"]) is int and payload["routing_classes"] >= 0,
        "invalid routing architecture",
    )
    for key in ("fused_task_heads", "split_reward_context", "input_block_norm"):
        require(type(payload[key]) is bool, "invalid experimental architecture switch")
    for key in ("dataset_block_rms", "target_block_rms"):
        require(
            type(payload[key]) in (int, float) and math.isfinite(payload[key]) and payload[key] > 0,
            "invalid experimental normalization architecture",
        )
    return {key: payload[key] for key in ARCHITECTURE_FIELDS}
