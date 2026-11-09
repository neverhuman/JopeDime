"""Pipeline genome parsing and formatting helpers for active data."""

from __future__ import annotations

import math
import re
from typing import Any

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
    text = genome.strip()
    if not text.startswith("(pipe X") or not text.endswith(")"):
        raise ValueError("expected active pipeline genome")
    body = text[len("(pipe X") : -1].strip()
    stages: list[str] = []
    depth = 0
    start: int | None = None
    in_quote = False
    escaped = False
    for i, ch in enumerate(body):
        if in_quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_quote = False
            continue
        if ch == '"':
            in_quote = True
        elif ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                stages.append(body[start : i + 1])
                start = None
            if depth < 0:
                raise ValueError("unbalanced pipeline")
    if depth != 0 or in_quote:
        raise ValueError("unbalanced pipeline")
    return stages


def _parse_stage(stage: str) -> dict[str, Any]:
    if not stage.startswith("(") or not stage.endswith(")"):
        raise ValueError(f"bad stage: {stage}")
    tokens = _tokenize_stage(stage[1:-1].strip())
    if not tokens:
        raise ValueError("empty stage")
    op = tokens[0]
    params: dict[str, Any] = {}
    for token in tokens[1:]:
        if "=" not in token:
            raise ValueError(f"bad param token: {token}")
        key, raw = token.split("=", 1)
        params[key] = _parse_param(raw)
    return {"op": op, "params": params}


def _tokenize_stage(text: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []
    in_quote = False
    escaped = False
    for ch in text:
        if in_quote:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_quote = False
            continue
        if ch == '"':
            in_quote = True
            buf.append(ch)
        elif ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _parse_param(raw: str) -> Any:
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if re.fullmatch(r"[-+]?\d+", raw):
        return int(raw)
    try:
        value = float(raw)
    except ValueError:
        return raw
    if not math.isfinite(value):
        raise ValueError(f"non-finite param: {raw}")
    return value


def _format_stage(stage: dict[str, Any]) -> str:
    parts = [stage["op"]]
    for key, value in sorted((stage.get("params") or {}).items()):
        parts.append(f"{key}={_format_param(value)}")
    return "(" + " ".join(parts) + ")"


def _format_param(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-12:
            return f"{value:.1f}"
        return repr(value)
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_+*.-]+", text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _freeze_vocab(vocab: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, list[str]]]:
    return {
        op: {key: sorted(values) for key, values in sorted(params.items())}
        for op, params in sorted(vocab.items())
    }


def _param_token(value: Any) -> str:
    return _format_param(value)
