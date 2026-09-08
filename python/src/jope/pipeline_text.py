"""Engine-aware genome text handling: tokenize, parse, and format pipeline genomes.

This is the low-level text layer shared by all engines. ``parse_pipeline_actions``
gates against a GrammarSpec (any engine); the legacy featmath-only wrapper lives in
``data/active.py``. Canonical-parity rule: strings produced here are only ever used to
FEED a bridge — the bridge's ``canonical_genome`` is the sole stored identity.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .grammars.spec import GrammarSpec


def split_pipeline_stages(genome: str) -> list[str]:
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


def tokenize_stage(text: str) -> list[str]:
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


def parse_stage(stage: str) -> dict[str, Any]:
    if not stage.startswith("(") or not stage.endswith(")"):
        raise ValueError(f"bad stage: {stage}")
    tokens = tokenize_stage(stage[1:-1].strip())
    if not tokens:
        raise ValueError("empty stage")
    op = tokens[0]
    params: dict[str, Any] = {}
    for token in tokens[1:]:
        if "=" not in token:
            raise ValueError(f"bad param token: {token}")
        key, raw = token.split("=", 1)
        params[key] = parse_param_token(raw)
    return {"op": op, "params": params}


def parse_param_token(raw: str) -> Any:
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


def format_param_token(value: Any) -> str:
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


def format_stage(stage: dict[str, Any]) -> str:
    parts = [stage["op"]]
    for key, value in sorted((stage.get("params") or {}).items()):
        parts.append(f"{key}={format_param_token(value)}")
    return "(" + " ".join(parts) + ")"


def parse_pipeline_actions(genome: str, spec: "GrammarSpec") -> dict[str, Any]:
    """Parse a genome into actions, gated against the given engine grammar."""
    stages = split_pipeline_stages(genome)
    if not stages:
        raise ValueError("pipeline contains no stages")
    parsed = [parse_stage(stage) for stage in stages]
    head = parsed[-1]
    transforms = parsed[:-1]
    if head["op"] not in spec.heads:
        raise ValueError(f"{head['op']}: not a {spec.engine} head")
    unknown = [stage["op"] for stage in transforms if stage["op"] not in spec.transforms]
    if unknown:
        raise ValueError(f"not {spec.engine} transforms: {unknown}")
    if len(transforms) > spec.max_transforms:
        raise ValueError(f"too many transforms: {len(transforms)}")
    return {
        "engine": spec.engine,
        "transform_count": len(transforms),
        "transforms": transforms,
        "head": head,
    }


def format_pipeline_actions(actions: dict[str, Any]) -> str:
    stages = list(actions.get("transforms") or []) + [actions["head"]]
    return "(pipe X " + " ".join(format_stage(stage) for stage in stages) + ")"


def parse_any(genome: str, specs: Sequence["GrammarSpec"]) -> tuple[str, dict[str, Any]] | None:
    """Try engines in order (featmath first for legacy ambiguity); None if none parse."""
    for spec in specs:
        try:
            return spec.engine, parse_pipeline_actions(genome, spec)
        except ValueError:
            continue
    return None
