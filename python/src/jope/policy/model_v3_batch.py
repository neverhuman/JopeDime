"""Batch assembly and row-token preparation for policy_v3."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .automaton import replay_masks
from .losses import bootstrap_mask, rwr_weights
from .model_v3_arch import TASK_LANE, PolicyV3Batch
from .tokens import PAD, ActionVocab, encode_actions


def pad_sequences(
    sequences: Sequence[Sequence[int]], device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(len(s) for s in sequences)
    token_ids = torch.full((len(sequences), T), PAD, dtype=torch.long, device=device)
    pad_mask = torch.zeros((len(sequences), T), dtype=torch.bool, device=device)
    for i, seq in enumerate(sequences):
        token_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        pad_mask[i, : len(seq)] = True
    return token_ids, pad_mask


def encode_row_artifacts(row, vocab: ActionVocab):
    """Encode one row to (token seq, per-position legal-id lists). Pure fn of the row —
    cache it once and reuse across epochs (this is the dominant per-epoch CPU cost)."""
    actions = (row.metadata or {}).get("actions")
    if not isinstance(actions, dict):
        from ..grammars import grammar_for
        from ..pipeline_text import parse_pipeline_actions

        actions = parse_pipeline_actions(row.canonical_genome or row.genome, grammar_for(row.engine))
    task = str((row.metadata or {}).get("task_type") or actions.get("task") or "regression")
    seq = encode_actions(row.engine, actions, vocab, task=task)
    masks = replay_masks(seq, vocab)
    return seq, masks


def row_artifact_key(row) -> str:
    """Stable identity for the pure genome->token/mask preprocessing result."""
    import hashlib
    import json

    metadata = row.metadata or {}
    payload = [
        row.engine,
        str(metadata.get("task_type") or "regression"),
        row.canonical_genome or row.genome,
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def precompute_row_artifacts(
    rows: list,
    vocab: ActionVocab,
    persistent_cache: dict[str, tuple[list[int], list[list[int]]]] | None = None,
) -> dict[int, tuple[list[int], list[list[int]]]]:
    """Encode every row ONCE (token ids + legal masks), keyed by id(row). Rows that fall
    outside the policy action space are omitted (skip-not-crash, counted)."""
    from .automaton import IllegalToken

    cache: dict[int, tuple[list[int], list[list[int]]]] = {}
    for row in rows:
        key = row_artifact_key(row)
        if persistent_cache is not None and key in persistent_cache:
            cache[id(row)] = persistent_cache[key]
            continue
        try:
            artifact = encode_row_artifacts(row, vocab)
            cache[id(row)] = artifact
            if persistent_cache is not None:
                persistent_cache[key] = artifact
        except (IllegalToken, ValueError):
            vocab.counters["rows_skipped_encode"] = vocab.counters.get("rows_skipped_encode", 0) + 1
    return cache


def rows_to_policy_v3_batch(
    rows: list,
    vocab: ActionVocab,
    *,
    device: str | torch.device = "cpu",
    reward_k: int = 4,
    precomputed_rwr: dict[int, float] | None = None,
    artifact_cache: dict[int, tuple[list[int], list[list[int]]]] | None = None,
) -> PolicyV3Batch:
    """Collate TrainingRows into a policy_v3 batch (tokens, masks, labels).

    ``artifact_cache`` (from ``precompute_row_artifacts``) skips per-epoch re-encoding —
    the biggest CPU cost that otherwise starves the GPU.
    """
    from .automaton import IllegalToken

    sequences: list[list[int]] = []
    masks_per_row: list[list[list[int]]] = []
    canonicals: list[str] = []
    kept_rows: list = []
    for row in rows:
        cached = artifact_cache.get(id(row)) if artifact_cache is not None else None
        if cached is not None:
            seq, masks = cached
        else:
            try:
                seq, masks = encode_row_artifacts(row, vocab)
            except (IllegalToken, ValueError):
                vocab.counters["rows_skipped_encode"] = vocab.counters.get("rows_skipped_encode", 0) + 1
                continue
        sequences.append(seq)
        masks_per_row.append(masks)
        canonicals.append(row.canonical_genome or row.genome)
        kept_rows.append(row)
    if not sequences:
        raise ValueError("no encodable rows in batch")
    rows = kept_rows
    task_types = [str((row.metadata or {}).get("task_type") or "regression") for row in rows]
    if any(task not in TASK_LANE for task in task_types):
        raise ValueError(
            "policy rows must use a known task_type "
            f"({'/'.join(TASK_LANE)}); unknown values are never coerced"
        )
    token_ids, pad_mask = pad_sequences(sequences, device=device)
    B, T = token_ids.shape

    # Vectorized legal-mask scatter: gather all (batch, time, token) triples, one index op,
    # built on CPU then moved — replaces the per-position Python double loop.
    bi: list[int] = []
    ti: list[int] = []
    vi: list[int] = []
    for i, masks in enumerate(masks_per_row):
        for t, ids in enumerate(masks):
            n = len(ids)
            if n:
                bi.extend([i] * n)
                ti.extend([t] * n)
                vi.extend(ids)
    legal = torch.zeros((B, T - 1, vocab.size), dtype=torch.bool)
    if vi:
        legal[torch.tensor(bi), torch.tensor(ti), torch.tensor(vi)] = True
    legal = legal.to(device, non_blocking=True)

    def _finite(value) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return out if math.isfinite(out) else float("nan")

    robust_delta = torch.tensor(
        [_finite((row.metadata or {}).get("oof_delta_to_incumbent")) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    runtime = torch.tensor([_finite(row.cost_ms) for row in rows], dtype=torch.float32, device=device)

    def _bad(row, task: str) -> float:
        """1.0 when a candidate is worse than its task's trivial baseline.

        Every branch must be explicit about which metric it is reading, because the
        threshold that means "worse than baseline" is different for each. Previously
        only binary was special-cased and everything else fell through to the r2 test
        ``value < 0.0`` against ``row.score``:

          * timeseries scores are ``negative_mase`` (= -MASE), which is negative for
            essentially every candidate, so the whole task was labelled bad;
          * multiclass scores are ``macro_f1`` in [0, 1], which is never negative, so
            the whole task was labelled good.

        Either way the model got a constant label and could learn nothing from the
        task while the pipeline looked healthy.
        """
        meta = row.metadata or {}
        if meta.get("eval_error"):
            return 1.0
        if task == "binary":
            value = _finite(meta.get("oof_auc", meta.get("score_robust", row.score)))
            return float("nan") if math.isnan(value) else 0.0
        if task == "multiclass":
            # macro_f1 in [0, 1]. Like AUC it has no negative region, so a finite
            # score is not evidence of failure on its own.
            value = _finite(meta.get("oof_macro_f1", row.score))
            return float("nan") if math.isnan(value) else 0.0
        if task == "timeseries":
            # negative_mase = -MASE, so parity with the seasonal-naive baseline is
            # -1.0 and anything below it is genuinely worse than doing nothing.
            value = _finite(meta.get("oof_negative_mase", row.score))
            if math.isnan(value):
                return float("nan")
            return 1.0 if value < -1.0 else 0.0
        raw = meta.get("oof_r2_raw", row.score)
        value = _finite(raw)
        if math.isnan(value):
            return float("nan")
        return 1.0 if value < 0.0 else 0.0

    bad_label = torch.tensor(
        [_bad(row, task) for row, task in zip(rows, task_types)],
        dtype=torch.float32,
        device=device,
    )
    reward_pair_ids = [
        str((row.metadata or {}).get("reward_pair_identity") or row.dataset_id) for row in rows
    ]
    if precomputed_rwr is not None:
        weight_list = [precomputed_rwr.get(id(row), 1.0) for row in rows]
    elif any((row.metadata or {}).get("rwr_weight_global") is not None for row in rows):
        # global RWR weights were precomputed corpus-wide (prep_v31) but never used —
        # batch-local recomputation is noisier; prefer the stamped weights.
        weight_list = [float((row.metadata or {}).get("rwr_weight_global") or 1.0) for row in rows]
    else:
        weight_list = rwr_weights(
            [f"{task}:{identity}" for task, identity in zip(task_types, reward_pair_ids)],
            [
                None
                if not math.isfinite(_finite((row.metadata or {}).get("score_robust", row.score)))
                else _finite((row.metadata or {}).get("score_robust", row.score))
                for row in rows
            ],
            [str((row.metadata or {}).get("provenance") or "gp_report") for row in rows],
        )
    rwr_weight = torch.tensor(weight_list, dtype=torch.float32, device=device)
    bootstrap = torch.tensor(
        [
            [bootstrap_mask(f"{task}:{identity}", row.engine, canonical, k) for k in range(reward_k)]
            for row, canonical, task, identity in zip(rows, canonicals, task_types, reward_pair_ids)
        ],
        dtype=torch.bool,
        device=device,
    )
    # v3.1 self-play labels (absent → None; losses fall back to legacy targets)
    rank_pct_vals = [_finite((row.metadata or {}).get("rank_pct")) for row in rows]
    has_rank = any(math.isfinite(v) for v in rank_pct_vals)
    is_self_vals = [
        str((row.metadata or {}).get("provenance", "")).startswith("self")
        or str((row.metadata or {}).get("provenance", "")) == "self"
        for row in rows
    ]
    former_best_vals = [_finite((row.metadata or {}).get("former_best_score")) for row in rows]

    return PolicyV3Batch(
        embedding=torch.tensor([row.embedding for row in rows], dtype=torch.float32, device=device),
        target_summary=torch.tensor(
            [
                (row.target_embedding or [0.0] * 512)[:512]
                + [0.0] * max(0, 512 - len(row.target_embedding or []))
                for row in rows
            ],
            dtype=torch.float32,
            device=device,
        ),
        dataset_ids=[str(row.dataset_id) for row in rows],
        reward_pair_ids=reward_pair_ids,
        engines=[row.engine for row in rows],
        task_types=task_types,
        token_ids=token_ids,
        pad_mask=pad_mask,
        legal=legal,
        robust_delta=robust_delta,
        rwr_weight=rwr_weight,
        runtime=runtime,
        bad_label=bad_label,
        bootstrap=bootstrap,
        canonicals=canonicals,
        rank_pct=(torch.tensor(rank_pct_vals, dtype=torch.float32, device=device) if has_rank else None),
        is_self=(torch.tensor(is_self_vals, dtype=torch.bool, device=device) if any(is_self_vals) else None),
        former_best=(
            torch.tensor(former_best_vals, dtype=torch.float32, device=device)
            if any(math.isfinite(v) for v in former_best_vals)
            else None
        ),
        routing_label=torch.tensor(
            [int((row.metadata or {}).get("routing_class", -1)) for row in rows],
            dtype=torch.int64,
            device=device,
        ),
        seq_weight=torch.tensor(
            [float((row.metadata or {}).get("seq_weight", 1.0)) for row in rows],
            dtype=torch.float32,
            device=device,
        ),
    )
