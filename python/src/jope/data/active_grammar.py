"""Pipeline genome parsing and formatting helpers for active data.

Stage splitting, tokenization, and the parameter codec live in
``jope.pipeline_text``. This module keeps the featmath head and transform gate.
"""

from __future__ import annotations

from typing import Any

import jope.pipeline_text as pipeline_text
from jope.grammar import HEADS, TRANSFORMS

ACTIVE_MAX_TRANSFORMS = 6


def parse_pipeline_actions(genome: str) -> dict[str, Any]:
    stages = _split_pipeline_stages(genome)
    if not stages:
        raise ValueError("pipeline contains no stages")
    parsed = [_parse_stage(stage) for stage in stages]
    head = parsed[-1]
    transforms = parsed[:-1]
    if head["op"] not in HEADS:
        raise ValueError(f"{head['op']}: inactive head")
    inactive = [stage["op"] for stage in transforms if stage["op"] not in TRANSFORMS]
    if inactive:
        raise ValueError(f"inactive transforms: {inactive}")
    if len(transforms) > ACTIVE_MAX_TRANSFORMS:
        raise ValueError(f"too many transforms: {len(transforms)}")
    return {
        "transform_count": len(transforms),
        "transforms": transforms,
        "head": head,
    }


def format_pipeline_actions(actions: dict[str, Any]) -> str:
    stages = list(actions.get("transforms") or []) + [actions["head"]]
    return "(pipe X " + " ".join(_format_stage(stage) for stage in stages) + ")"


def parse_param_token(token: str) -> Any:
    return _parse_param(token)


def format_param_token(value: Any) -> str:
    return _format_param(value)


def _split_pipeline_stages(genome: str) -> list[str]:
    return pipeline_text.split_pipeline_stages(genome)


def _parse_stage(stage: str) -> dict[str, Any]:
    return pipeline_text.parse_stage(stage)


def _tokenize_stage(text: str) -> list[str]:
    return pipeline_text.tokenize_stage(text)


def _parse_param(raw: str) -> Any:
    return pipeline_text.parse_param_token(raw)


def _format_stage(stage: dict[str, Any]) -> str:
    return pipeline_text.format_stage(stage)


def _format_param(value: Any) -> str:
    return pipeline_text.format_param_token(value)


def _freeze_vocab(vocab: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, list[str]]]:
    return {
        op: {key: sorted(values) for key, values in sorted(params.items())}
        for op, params in sorted(vocab.items())
    }


def _param_token(value: Any) -> str:
    return _format_param(value)
