"""Checkpoint compatibility and reconstruction for policy_v3."""

from __future__ import annotations

import copy
from typing import Any

from .model_v3_arch import DATASET_BLOCK_RMS, TARGET_BLOCK_RMS, JopePolicyModelV3

def _validate_action_vocab_compatibility(
    saved_tokens: list[str] | None,
    current_tokens: list[str],
    embedding_rows: int,
) -> None:
    """Accept only an exact append-only extension that fits the frozen table."""

    if not saved_tokens or saved_tokens == current_tokens:
        return
    if current_tokens[: len(saved_tokens)] != saved_tokens:
        raise ValueError("action vocab drift: checkpoint tokens are not an exact current prefix")
    if embedding_rows < len(current_tokens):
        raise ValueError("action vocab extension exceeds the checkpoint embedding table")


def load_policy_v3(state: dict[str, Any]) -> JopePolicyModelV3:
    """Rebuild a policy_v3 model from a checkpoint payload."""
    from ..grammars.spec import GrammarSpec
    from .tokens import build_action_vocab, build_prime_action_vocab

    specs_json = state.get("grammar_specs")
    frozen_action_vocab = state.get("frozen_action_vocab")
    if frozen_action_vocab is not None:
        from .training_corpus_contract import load_frozen_action_vocab

        specs, vocab = load_frozen_action_vocab(frozen_action_vocab)
        if specs_json != [spec.to_json() for spec in specs]:
            raise ValueError("frozen action table parsing views changed")
        if state.get("action_vocab") != vocab.to_json():
            raise ValueError("frozen action table token IDs/version changed")
    elif specs_json:
        specs = [GrammarSpec.from_json(s) for s in specs_json]
        vocab = (
            build_prime_action_vocab()
            if any(spec.engine == "prime" for spec in specs)
            else build_action_vocab(specs)
        )
    else:
        raise ValueError("policy_v3 checkpoint missing grammar_specs")
    saved_tokens = (state.get("action_vocab") or {}).get("tokens")
    _validate_action_vocab_compatibility(
        saved_tokens,
        vocab.tokens,
        int(state.get("embedding_rows") or 0),
    )
    # Task-head schema: v1 (2 tasks, identity lanes) and v2 (4 tasks, TASK_LANE
    # aliases) rebuild the SAME physical architecture — lanes are runtime routing,
    # not tensors — so both load; anything else is from a future writer and must
    # not be silently reinterpreted.
    task_head_schema = state.get("task_head_schema") or {}
    schema_name = str(task_head_schema.get("schema") or "PrimeTaskHeads.v1")
    if schema_name not in {"PrimeTaskHeads.v1", "PrimeTaskHeads.v2"}:
        raise ValueError(
            f"unsupported task_head_schema {schema_name!r}; "
            "this build understands PrimeTaskHeads.v1 (identity lanes) and "
            "PrimeTaskHeads.v2 (timeseries/multiclass lane aliases)"
        )
    model = JopePolicyModelV3(
        vocab,
        dataset_dim=int(state.get("dataset_dim", 512)),
        target_dim=int(state.get("target_dim", 512)),
        d_model=int(state.get("d_model", 256)),
        n_layers=int(state.get("n_layers", 2)),
        n_heads=int(state.get("n_heads", 4)),
        ffn_dim=int(state.get("ffn_dim", 1024)),
        reward_heads=int(state.get("reward_ensemble_k", 4)),
        embedding_rows=int(state.get("embedding_rows") or 0) or None,
        routing_classes=int(state.get("routing_classes", 0)),
        fused_task_heads=bool(state.get("fused_task_heads", False)),
        # v3.4 keys default to the v3.3 architecture so historical checkpoints rebuild
        # exactly as they were written and load 152/152 tensors.
        target_keep_dim=state.get("target_keep_dim"),
        split_reward_context=bool(state.get("split_reward_context", False)),
        input_block_norm=bool(state.get("input_block_norm", False)),
        dataset_block_rms=float(state.get("dataset_block_rms", DATASET_BLOCK_RMS)),
        target_block_rms=float(state.get("target_block_rms", TARGET_BLOCK_RMS)),
    )
    model.frozen_action_vocab = copy.deepcopy(frozen_action_vocab)
    # strict=False: checkpoints predating the stability head still load (its params
    # stay at init and retrain on the next fit). v3.1 changed reward/stability head
    # SHAPES (conditional 3d inputs) — shape-mismatched tensors from older checkpoints
    # are skipped (those heads re-init and retrain) instead of crashing the load.
    current = model.state_dict()
    filtered = {k: v for k, v in state["model"].items() if k in current and current[k].shape == v.shape}
    # A silently-skipped CONTEXT tensor is the "lobotomy" failure class: the network still
    # loads, PyTorch reports nothing, and the dataset conditioning is random for the rest of
    # the run. Heads may re-init; the two context encoders may not.
    context_missing = sorted(
        key
        for key in current
        if key.split(".")[0] in {"encoder", "reward_encoder", "ctx_proj"} and key not in filtered
    )
    if context_missing:
        raise ValueError(
            "policy_v3 checkpoint cannot supply the context encoder(s) at the rebuilt "
            f"shape; would re-randomise {context_missing}. Migrate the checkpoint instead "
            "of loading it (see train_v3.migrate_context_state)."
        )
    model.load_state_dict(filtered, strict=False)
    model._grammar_specs = specs  # for checkpoint round-trips (loop refits)
    model.eval()
    return model
