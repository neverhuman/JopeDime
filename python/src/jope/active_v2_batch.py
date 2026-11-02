from __future__ import annotations

from typing import Any

import torch

from .active_v2_losses import (
    _decode_params,
    _finite_or_nan,
    _labels_to_params,
    _mae,
    _pairwise_accuracy,
    _pearson,
    _row_actions,
    _row_target_summary,
    _select_index,
    _top1_regret,
)
from .active_v2_model import JopeActiveModelV2
from .active_v2_vocab import (
    ActiveV2Batch,
    active_v2_param_specs,
    genome_features_from_actions,
    normalize_active_v2_vocab,
)
from .data.active import ACTIVE_MAX_TRANSFORMS, format_pipeline_actions
from .model import FAMILY_LIST, HEAD_LIST, NONE_TOKEN
from .schema import EMBEDDING_DIM

def rows_to_active_v2_batch(
    rows: list,
    vocab: dict[str, Any],
    device: str | torch.device = "cpu",
) -> ActiveV2Batch:
    spec_vocab = normalize_active_v2_vocab(vocab)
    n = len(rows)
    embedding = torch.tensor(
        [row.embedding[:EMBEDDING_DIM] for row in rows], dtype=torch.float32, device=device
    )
    target_summary = torch.tensor(
        [_row_target_summary(row) for row in rows], dtype=torch.float32, device=device
    )
    transform_ops = {op: i for i, op in enumerate(spec_vocab["transform_ops"])}
    head_ops = {op: i for i, op in enumerate(spec_vocab["head_ops"])}
    transform_count = torch.zeros(n, dtype=torch.long, device=device)
    slot_ops = torch.zeros((n, ACTIVE_MAX_TRANSFORMS), dtype=torch.long, device=device)
    slot_append = torch.full((n, ACTIVE_MAX_TRANSFORMS), -1.0, dtype=torch.float32, device=device)
    head = torch.zeros(n, dtype=torch.long, device=device)
    param_specs = active_v2_param_specs(spec_vocab)
    param_labels = {
        spec["key"]: torch.full((n,), -100, dtype=torch.long, device=device) for spec in param_specs
    }
    value = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    rank_percentile = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    top_candidate = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    complexity = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    runtime = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    family = torch.zeros((n, len(FAMILY_LIST)), dtype=torch.float32, device=device)
    genome_feature_rows: list[list[float]] = []
    specs_by_lookup = {(spec["kind"], spec["slot"], spec["op"], spec["param"]): spec for spec in param_specs}

    for i, row in enumerate(rows):
        actions = _row_actions(row)
        genome_feature_rows.append(genome_features_from_actions(actions, row.family_bitmap, spec_vocab))
        transforms = list(actions.get("transforms") or [])[:ACTIVE_MAX_TRANSFORMS]
        transform_count[i] = len(transforms)
        for slot, stage in enumerate(transforms):
            op = str(stage.get("op") or NONE_TOKEN)
            slot_ops[i, slot] = transform_ops.get(op, 0)
            params = stage.get("params") or {}
            slot_append[i, slot] = 1.0 if bool(params.get("append")) else 0.0
            for param, raw_value in params.items():
                lookup = ("transform", slot, op, param)
                spec = specs_by_lookup.get(lookup)
                if spec is None:
                    continue
                token = format_param_token(raw_value)
                if token in spec["tokens"]:
                    param_labels[spec["key"]][i] = spec["tokens"].index(token)
        head_stage = actions.get("head") or {}
        head_op = str(head_stage.get("op") or "ridge")
        head[i] = head_ops.get(head_op, 0)
        for param, raw_value in (head_stage.get("params") or {}).items():
            spec = specs_by_lookup.get(("head", 0, head_op, param))
            if spec is None:
                continue
            token = format_param_token(raw_value)
            if token in spec["tokens"]:
                param_labels[spec["key"]][i] = spec["tokens"].index(token)
        delta = (row.metadata or {}).get("oof_delta_to_incumbent")
        value[i] = _finite_or_nan(delta if delta is not None else row.score)
        rank_percentile[i] = _finite_or_nan((row.metadata or {}).get("rank_percentile"))
        if (row.metadata or {}).get("candidate_rank") is not None:
            top_candidate[i] = 1.0 if int(row.metadata["candidate_rank"]) == 1 else 0.0
        complexity[i] = _finite_or_nan(row.complexity)
        runtime[i] = _finite_or_nan(row.cost_ms)
        for j, name in enumerate(FAMILY_LIST):
            if row.family_bitmap.get(name):
                family[i, j] = 1.0

    genome_features = torch.tensor(genome_feature_rows, dtype=torch.float32, device=device)
    return ActiveV2Batch(
        embedding=embedding,
        target_summary=target_summary,
        dataset_ids=[str(row.dataset_id) for row in rows],
        transform_count=transform_count,
        slot_ops=slot_ops,
        slot_append=slot_append,
        head=head,
        param_labels=param_labels,
        value=value,
        rank_percentile=rank_percentile,
        top_candidate=top_candidate,
        complexity=complexity,
        runtime=runtime,
        family=family,
        genome_features=genome_features,
    )


def active_v2_labels_to_genome(batch: ActiveV2Batch, vocab: dict[str, Any], index: int = 0) -> str:
    spec_vocab = normalize_active_v2_vocab(vocab)
    transforms = []
    for slot in range(ACTIVE_MAX_TRANSFORMS):
        op_idx = int(batch.slot_ops[index, slot].detach().cpu())
        op = spec_vocab["transform_ops"][max(0, min(op_idx, len(spec_vocab["transform_ops"]) - 1))]
        if op == NONE_TOKEN:
            continue
        params: dict[str, Any] = {}
        if float(batch.slot_append[index, slot].detach().cpu()) > 0.5:
            params["append"] = True
        _labels_to_params(params, batch, spec_vocab, kind="transform", slot=slot, op=op, index=index)
        transforms.append({"op": op, "params": params})
    head_idx = int(batch.head[index].detach().cpu())
    head_op = spec_vocab["head_ops"][max(0, min(head_idx, len(spec_vocab["head_ops"]) - 1))]
    head_params: dict[str, Any] = {}
    _labels_to_params(head_params, batch, spec_vocab, kind="head", slot=0, op=head_op, index=index)
    return format_pipeline_actions(
        {
            "transform_count": len(transforms),
            "transforms": transforms,
            "head": {"op": head_op, "params": head_params},
        }
    )


def decode_active_v2_output(
    output: dict[str, Any],
    vocab: dict[str, Any],
    *,
    row: int = 0,
    temperature: float = 0.0,
    seed: int = 0,
    head_index: int | None = None,
    count_index: int | None = None,
) -> tuple[str, dict[str, Any]]:
    spec_vocab = normalize_active_v2_vocab(vocab)
    generator = torch.Generator(device=output["head_logits"].device)
    generator.manual_seed(seed)
    count_logits = output["transform_count_logits"][row]
    count = count_index if count_index is not None else _select_index(count_logits, temperature, generator)
    count = max(0, min(int(count), ACTIVE_MAX_TRANSFORMS))
    head_logits = output["head_logits"][row]
    head_idx = head_index if head_index is not None else _select_index(head_logits, temperature, generator)
    head_op = spec_vocab["head_ops"][max(0, min(int(head_idx), len(spec_vocab["head_ops"]) - 1))]
    transforms = []
    for slot in range(ACTIVE_MAX_TRANSFORMS):
        if slot >= count:
            continue
        op_idx = _select_index(output["slot_op_logits"][row, slot], temperature, generator)
        op = spec_vocab["transform_ops"][max(0, min(int(op_idx), len(spec_vocab["transform_ops"]) - 1))]
        if op == NONE_TOKEN:
            continue
        params: dict[str, Any] = {}
        if float(output["slot_append_logits"][row, slot].detach().cpu()) > 0.0:
            params["append"] = True
        _decode_params(
            params,
            output,
            spec_vocab,
            kind="transform",
            slot=slot,
            op=op,
            row=row,
            temperature=temperature,
            generator=generator,
        )
        transforms.append({"op": op, "params": params})
    head_params: dict[str, Any] = {}
    _decode_params(
        head_params,
        output,
        spec_vocab,
        kind="head",
        slot=0,
        op=head_op,
        row=row,
        temperature=temperature,
        generator=generator,
    )
    actions = {
        "transform_count": len(transforms),
        "transforms": transforms,
        "head": {"op": head_op, "params": head_params},
    }
    return format_pipeline_actions(actions), actions


def active_v2_validation_metrics(
    model: JopeActiveModelV2,
    rows: list,
    vocab: dict[str, Any],
    *,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    if not rows:
        return {}
    model.eval()
    exact = 0
    head_ok = 0
    transform_ok = 0
    total = 0
    values: list[float] = []
    reward_preds: list[float] = []
    value_preds: list[float] = []
    dataset_ids: list[str] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = rows_to_active_v2_batch(batch_rows, vocab, device=device)
            out = model(batch.embedding, batch.target_summary, batch.genome_features)
            pred_heads = out["head_logits"].argmax(dim=-1)
            pred_slots = out["slot_op_logits"].argmax(dim=-1)
            for i, row in enumerate(batch_rows):
                genome, _ = decode_active_v2_output(out, vocab, row=i)
                if genome == (row.canonical_genome or row.genome):
                    exact += 1
                head_ok += int(pred_heads[i].item() == batch.head[i].item())
                transform_ok += int(torch.equal(pred_slots[i], batch.slot_ops[i]))
                total += 1
            mask = torch.isfinite(batch.value)
            if mask.any():
                keep = mask.detach().cpu().tolist()
                values.extend(batch.value[mask].detach().cpu().tolist())
                value_preds.extend(out["value"][mask].detach().cpu().tolist())
                reward = out.get("reward")
                reward_preds.extend(
                    (reward if reward is not None else out["value"])[mask].detach().cpu().tolist()
                )
                dataset_ids.extend([ds for ds, k in zip(batch.dataset_ids, keep, strict=True) if k])
    return {
        "grammar_exact_match": exact / max(1, total),
        "head_accuracy": head_ok / max(1, total),
        "ordered_transform_accuracy": transform_ok / max(1, total),
        "pairwise_win_accuracy": _pairwise_accuracy(reward_preds, values, dataset_ids),
        "top1_historical_regret": _top1_regret(reward_preds, values, dataset_ids),
        "value_mae": _mae(value_preds, values),
        "reward_value_correlation": _pearson(reward_preds, values),
        "value_correlation": _pearson(value_preds, values),
    }


