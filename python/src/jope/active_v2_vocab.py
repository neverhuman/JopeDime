from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .data.active import ACTIVE_MAX_TRANSFORMS
from .grammar import HEADS, TRANSFORMS
from .model import FAMILY_LIST, NONE_TOKEN

@dataclass
class ActiveV2Batch:
    embedding: torch.Tensor
    target_summary: torch.Tensor
    dataset_ids: list[str]
    transform_count: torch.Tensor
    slot_ops: torch.Tensor
    slot_append: torch.Tensor
    head: torch.Tensor
    param_labels: dict[str, torch.Tensor]
    value: torch.Tensor
    rank_percentile: torch.Tensor
    top_candidate: torch.Tensor
    complexity: torch.Tensor
    runtime: torch.Tensor
    family: torch.Tensor
    genome_features: torch.Tensor | None = None


def normalize_active_v2_vocab(vocab: dict[str, Any] | None) -> dict[str, Any]:
    vocab = vocab or {}
    if "transform_ops" in vocab and "head_ops" in vocab:
        return {
            "syntax": vocab.get("syntax") or "active_jain_regressor_v2",
            "max_transforms": int(vocab.get("max_transforms") or ACTIVE_MAX_TRANSFORMS),
            "transform_ops": list(vocab.get("transform_ops") or [NONE_TOKEN] + sorted(TRANSFORMS)),
            "head_ops": list(vocab.get("head_ops") or sorted(HEADS)),
            "transform_params": vocab.get("transform_params") or {},
            "head_params": vocab.get("head_params") or {},
        }
    raw_transforms = vocab.get("transforms") if isinstance(vocab.get("transforms"), dict) else {}
    raw_heads = vocab.get("heads") if isinstance(vocab.get("heads"), dict) else {}
    transform_ops = sorted(op for op in raw_transforms if op in TRANSFORMS) or sorted(TRANSFORMS)
    head_ops = sorted(op for op in raw_heads if op in HEADS) or sorted(HEADS)
    return {
        "syntax": vocab.get("syntax") or "active_jain_regressor_v2",
        "max_transforms": int(vocab.get("max_transforms") or ACTIVE_MAX_TRANSFORMS),
        "transform_ops": [NONE_TOKEN] + transform_ops,
        "head_ops": head_ops,
        "transform_params": _normalize_param_vocab(raw_transforms, transform_ops),
        "head_params": _normalize_param_vocab(raw_heads, head_ops),
    }


def active_v2_param_specs(vocab: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    transform_params = vocab.get("transform_params") or {}
    for slot in range(ACTIVE_MAX_TRANSFORMS):
        for op, params in sorted(transform_params.items()):
            for param, tokens in sorted(params.items()):
                if not tokens:
                    continue
                specs.append(
                    {
                        "key": _param_key("transform", slot, op, param),
                        "kind": "transform",
                        "slot": slot,
                        "op": op,
                        "param": param,
                        "tokens": list(tokens),
                    }
                )
    head_params = vocab.get("head_params") or {}
    for op, params in sorted(head_params.items()):
        for param, tokens in sorted(params.items()):
            if not tokens:
                continue
            specs.append(
                {
                    "key": _param_key("head", 0, op, param),
                    "kind": "head",
                    "slot": 0,
                    "op": op,
                    "param": param,
                    "tokens": list(tokens),
                }
            )
    return specs


# Extra scalar features appended to the per-op counts in a genome feature vector.
_GENOME_SCALAR_FEATURES = 2


def genome_feature_dim(vocab: dict[str, Any] | None) -> int:
    """Fixed width of the genome feature vector consumed by the reward head."""
    spec = normalize_active_v2_vocab(vocab)
    return len(spec["transform_ops"]) + len(spec["head_ops"]) + len(FAMILY_LIST) + _GENOME_SCALAR_FEATURES


def genome_features_from_actions(
    actions: dict[str, Any] | None,
    family_bitmap: dict[str, bool] | None,
    vocab: dict[str, Any] | None,
) -> list[float]:
    """Encode a genome (its parsed actions) as a fixed-size vector for the reward head.

    Uses per-op *counts* (not binary) so repeated transforms are represented, a head
    one-hot, the family bitmap, and two shape scalars (transform count, append fraction).
    Order matches ``genome_feature_dim``.
    """
    spec = normalize_active_v2_vocab(vocab)
    transform_ops = spec["transform_ops"]
    head_ops = spec["head_ops"]
    t_index = {op: i for i, op in enumerate(transform_ops)}
    h_index = {op: i for i, op in enumerate(head_ops)}
    actions = actions or {}
    transforms = list(actions.get("transforms") or [])
    tvec = [0.0] * len(transform_ops)
    append_count = 0
    for stage in transforms:
        if not isinstance(stage, dict):
            continue
        op = str(stage.get("op") or NONE_TOKEN)
        if op in t_index:
            tvec[t_index[op]] += 1.0
        if (stage.get("params") or {}).get("append"):
            append_count += 1
    scale = float(max(1, ACTIVE_MAX_TRANSFORMS))
    tvec = [value / scale for value in tvec]
    hvec = [0.0] * len(head_ops)
    head_op = str((actions.get("head") or {}).get("op") or "")
    if head_op in h_index:
        hvec[h_index[head_op]] = 1.0
    fam = [1.0 if (family_bitmap or {}).get(name) else 0.0 for name in FAMILY_LIST]
    scalars = [len(transforms) / scale, append_count / scale]
    return tvec + hvec + fam + scalars



def _normalize_param_vocab(raw: dict[str, Any], ops: list[str]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for op in ops:
        params = raw.get(op) if isinstance(raw.get(op), dict) else {}
        out[op] = {}
        for param, values in sorted(params.items()):
            if isinstance(values, (list, tuple, set)):
                tokens = sorted({str(value) for value in values})
            else:
                tokens = [str(values)]
            out[op][str(param)] = tokens
    return out


def _param_key(kind: str, slot: int, op: str, param: str) -> str:
    return _safe_key(f"{kind}_{slot}_{op}_{param}")


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)

