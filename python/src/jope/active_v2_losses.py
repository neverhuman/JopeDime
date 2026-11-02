from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import torch
from torch.nn import functional as F

from .active_v2_vocab import ActiveV2Batch, _param_key
from .data.active import TARGET_SUMMARY_DIM, parse_param_token, parse_pipeline_actions

def _row_target_summary(row) -> list[float]:
    values = row.target_embedding[:TARGET_SUMMARY_DIM] if row.target_embedding is not None else []
    out = [float(value) for value in values]
    if len(out) < TARGET_SUMMARY_DIM:
        out.extend([0.0] * (TARGET_SUMMARY_DIM - len(out)))
    return out[:TARGET_SUMMARY_DIM]


def _row_actions(row) -> dict[str, Any]:
    actions = (row.metadata or {}).get("actions")
    if isinstance(actions, dict):
        return actions
    return parse_pipeline_actions(row.canonical_genome or row.genome)


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _balanced_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, *, ignore_index: int = -100
) -> torch.Tensor:
    target = target.long()
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    classes, counts = torch.unique(target[valid], return_counts=True)
    weights = torch.ones(logits.shape[-1], dtype=logits.dtype, device=logits.device)
    inv = counts.float().rsqrt()
    inv = inv / inv.mean().clamp_min(1e-6)
    weights[classes] = inv.to(logits.device)
    return F.cross_entropy(logits, target, weight=weights, ignore_index=ignore_index)


def _ranknet_loss(pred: torch.Tensor, label: torch.Tensor, dataset_ids: list[str]) -> torch.Tensor | None:
    pairs: list[torch.Tensor] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    finite = torch.isfinite(label).detach().cpu().tolist()
    for idx, dataset_id in enumerate(dataset_ids):
        if finite[idx]:
            grouped[dataset_id].append(idx)
    for indices in grouped.values():
        for a_pos, a in enumerate(indices):
            for b in indices[a_pos + 1 :]:
                if label[a] == label[b]:
                    continue
                sign = 1.0 if label[a] > label[b] else -1.0
                pairs.append(F.softplus(-sign * (pred[a] - pred[b])))
                if len(pairs) >= 256:
                    break
            if len(pairs) >= 256:
                break
        if len(pairs) >= 256:
            break
    if not pairs:
        return None
    return torch.stack(pairs).mean()


def _listwise_top_loss(
    pred_top: torch.Tensor, label_top: torch.Tensor, dataset_ids: list[str]
) -> torch.Tensor | None:
    losses: list[torch.Tensor] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    finite = torch.isfinite(label_top).detach().cpu().tolist()
    for idx, dataset_id in enumerate(dataset_ids):
        if finite[idx]:
            grouped[dataset_id].append(idx)
    for indices in grouped.values():
        positives = [idx for idx in indices if label_top[idx] > 0.5]
        if not positives or len(indices) < 2:
            continue
        logits = pred_top[indices].unsqueeze(0)
        target = torch.tensor([indices.index(positives[0])], dtype=torch.long, device=pred_top.device)
        losses.append(F.cross_entropy(logits, target))
    if not losses:
        return None
    return torch.stack(losses).mean()


def _group_by_dataset(label: torch.Tensor, dataset_ids: list[str]) -> dict[str, list[int]]:
    finite = torch.isfinite(label).detach().cpu().tolist()
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, dataset_id in enumerate(dataset_ids):
        if finite[idx]:
            grouped[dataset_id].append(idx)
    return grouped


def _reward_bradley_terry_loss(
    reward: torch.Tensor,
    value: torch.Tensor,
    dataset_ids: list[str],
    *,
    max_pairs: int = 4096,
) -> torch.Tensor | None:
    """Within-dataset, score-gap-weighted logistic ranking loss on the genome reward.

    P(a beats b) = sigmoid(r_a - r_b); this is the Bradley-Terry / Elo model. Pairs are
    weighted by the true score gap so decisive orderings matter more than ties. Within a
    dataset the incumbent is constant, so ``value`` (score - incumbent) preserves the
    score order and its gaps.
    """
    grouped = _group_by_dataset(value, dataset_ids)
    winners: list[int] = []
    losers: list[int] = []
    gaps: list[float] = []
    values = value.detach().cpu().tolist()
    for indices in grouped.values():
        for pos, a in enumerate(indices):
            for b in indices[pos + 1 :]:
                va, vb = values[a], values[b]
                if va == vb:
                    continue
                if va > vb:
                    winners.append(a)
                    losers.append(b)
                else:
                    winners.append(b)
                    losers.append(a)
                gaps.append(abs(va - vb))
    if not winners:
        return None
    win_t = torch.tensor(winners, device=reward.device)
    lose_t = torch.tensor(losers, device=reward.device)
    weights = torch.tensor(gaps, dtype=reward.dtype, device=reward.device)
    if win_t.numel() > max_pairs:
        sel = torch.linspace(0, win_t.numel() - 1, max_pairs, device=reward.device).round().long()
        win_t, lose_t, weights = win_t[sel], lose_t[sel], weights[sel]
    weights = weights / weights.mean().clamp_min(1e-6)
    diff = reward[win_t] - reward[lose_t]
    return (F.softplus(-diff) * weights).mean()


def _reward_listwise_loss(
    reward: torch.Tensor,
    value: torch.Tensor,
    dataset_ids: list[str],
) -> torch.Tensor | None:
    """ListNet-style listwise loss: match softmax(reward) to softmax of centered scores."""
    grouped = _group_by_dataset(value, dataset_ids)
    losses: list[torch.Tensor] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        idx_t = torch.tensor(indices, device=reward.device)
        centered = value[idx_t].float()
        centered = centered - centered.mean()
        target = torch.softmax(centered, dim=0)
        logp = torch.log_softmax(reward[idx_t], dim=0)
        losses.append(-(target * logp).sum())
    if not losses:
        return None
    return torch.stack(losses).mean()


def _select_index(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


def _decode_params(
    params: dict[str, Any],
    output: dict[str, Any],
    vocab: dict[str, Any],
    *,
    kind: str,
    slot: int,
    op: str,
    row: int,
    temperature: float,
    generator: torch.Generator,
) -> None:
    param_space = (vocab.get("transform_params") if kind == "transform" else vocab.get("head_params")) or {}
    for param, tokens in sorted((param_space.get(op) or {}).items()):
        key = _param_key(kind, slot, op, param)
        logits = output["param_logits"].get(key)
        if logits is None or not tokens:
            continue
        idx = _select_index(logits[row], temperature, generator)
        params[param] = parse_param_token(tokens[max(0, min(idx, len(tokens) - 1))])


def _labels_to_params(
    params: dict[str, Any],
    batch: ActiveV2Batch,
    vocab: dict[str, Any],
    *,
    kind: str,
    slot: int,
    op: str,
    index: int,
) -> None:
    param_space = (vocab.get("transform_params") if kind == "transform" else vocab.get("head_params")) or {}
    for param, tokens in sorted((param_space.get(op) or {}).items()):
        key = _param_key(kind, slot, op, param)
        labels = batch.param_labels.get(key)
        if labels is None:
            continue
        label = int(labels[index].detach().cpu())
        if label < 0 or label >= len(tokens):
            continue
        params[param] = parse_param_token(tokens[label])


def _pairwise_accuracy(preds: list[float], values: list[float], dataset_ids: list[str]) -> float:
    wins = 0
    total = 0
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, dataset_id in enumerate(dataset_ids):
        grouped[dataset_id].append(idx)
    for indices in grouped.values():
        for a_pos, a in enumerate(indices):
            for b in indices[a_pos + 1 :]:
                if values[a] == values[b]:
                    continue
                total += 1
                wins += int((preds[a] > preds[b]) == (values[a] > values[b]))
    return wins / total if total else 0.0


def _top1_regret(preds: list[float], values: list[float], dataset_ids: list[str]) -> float:
    regrets: list[float] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, dataset_id in enumerate(dataset_ids):
        grouped[dataset_id].append(idx)
    for indices in grouped.values():
        best = max(values[idx] for idx in indices)
        chosen = max(indices, key=lambda idx: preds[idx])
        regrets.append(best - values[chosen])
    return float(sum(regrets) / len(regrets)) if regrets else 0.0


def _mae(preds: list[float], values: list[float]) -> float:
    if not preds:
        return 0.0
    return float(sum(abs(pred - value) for pred, value in zip(preds, values, strict=True)) / len(preds))


def _pearson(preds: list[float], values: list[float]) -> float:
    if len(preds) < 2 or len(values) < 2:
        return 0.0
    pred_t = torch.tensor(preds, dtype=torch.float32)
    value_t = torch.tensor(values, dtype=torch.float32)
    pred_c = pred_t - pred_t.mean()
    value_c = value_t - value_t.mean()
    denom = pred_c.norm() * value_c.norm()
    if float(denom) <= 1e-12:
        return 0.0
    return float((pred_c * value_c).sum() / denom)
