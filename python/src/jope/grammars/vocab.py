"""v3 vocab: per-engine grammar snapshots + learned-bin merging from corpora."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..pipeline_text import format_param_token, parse_pipeline_actions
from .families import FAMILY_LIST_V3
from .spec import GrammarSpec, OpSpec, ParamSpec

if TYPE_CHECKING:  # pragma: no cover
    from ..schema import TrainingRow


def merge_learned_bins(
    spec: GrammarSpec,
    observed: dict[str, dict[str, list[str]]],
    *,
    cap: int = 24,
) -> GrammarSpec:
    """Union observed param tokens into the static bins (frequency-capped upstream)."""

    def merge_ops(ops: dict[str, OpSpec]) -> dict[str, OpSpec]:
        out: dict[str, OpSpec] = {}
        for name, op in ops.items():
            seen = observed.get(name) or {}
            new_params: list[ParamSpec] = []
            for param in op.params:
                if param.kind == "expr":
                    new_params.append(param)
                    continue
                tokens = seen.get(param.name) or []
                if not tokens:
                    new_params.append(param)
                    continue
                existing = [format_param_token(b) for b in param.bins]
                merged = list(param.bins)
                for token in tokens:
                    if token not in existing and len(merged) < cap:
                        try:
                            from ..pipeline_text import parse_param_token

                            merged.append(parse_param_token(token))
                            existing.append(token)
                        except ValueError:
                            continue
                new_params.append(replace(param, bins=tuple(merged)))
            out[name] = replace(op, params=tuple(new_params))
        return out

    return replace(spec, transforms=merge_ops(spec.transforms), heads=merge_ops(spec.heads))


def observed_param_tokens(
    rows: Iterable[TrainingRow],
    spec: GrammarSpec,
    *,
    cap: int = 24,
) -> dict[str, dict[str, list[str]]]:
    """Collect observed param tokens per (op, param) for rows of this spec's engine."""
    counts: dict[str, dict[str, Counter]] = {}
    for row in rows:
        if getattr(row, "engine", "featmath") != spec.engine:
            continue
        genome = row.canonical_genome or row.genome
        try:
            actions = parse_pipeline_actions(genome, spec)
        except ValueError:
            continue
        stages = list(actions["transforms"]) + [actions["head"]]
        for stage in stages:
            op = stage["op"]
            op_spec = spec.op(op)
            if op_spec is None:
                continue
            for key, value in (stage.get("params") or {}).items():
                if key == "append":
                    continue
                param = op_spec.param(key)
                if param is None or param.kind == "expr":
                    continue
                counts.setdefault(op, {}).setdefault(key, Counter())[format_param_token(value)] += 1
    return {
        op: {param: [token for token, _ in counter.most_common(cap)] for param, counter in params.items()}
        for op, params in counts.items()
    }


def build_active_vocab_v3(rows: Iterable[TrainingRow], specs: list[GrammarSpec]) -> dict[str, Any]:
    """Multi-engine vocab artifact with embedded grammar snapshots (learned bins merged)."""
    rows = list(rows)
    engines: dict[str, Any] = {}
    for spec in specs:
        observed = observed_param_tokens(rows, spec)
        merged = merge_learned_bins(spec, observed)
        engines[spec.engine] = merged.to_json()
    return {
        "syntax": "active_multi_engine_v3",
        "families": list(FAMILY_LIST_V3),
        "engines": engines,
    }


def grammars_from_vocab_v3(vocab: dict[str, Any]) -> dict[str, GrammarSpec]:
    """Rehydrate engine grammars from a v3/v4 vocab artifact (frozen snapshot).

    Old snapshots (schema < 4) predate ``tasks``/``generate`` serialization; naive
    ``OpSpec.from_json`` defaults would make every op task-universal and sampleable
    (e.g. regression-only ``gpr`` becomes binary-legal). For any op whose frozen JSON
    lacks those fields, inherit them from the SOURCE grammar instead.
    """
    import dataclasses

    from . import all_grammars

    engines = vocab.get("engines") or {}
    out: dict[str, GrammarSpec] = {}
    source = {spec.engine: spec for spec in all_grammars()}
    for engine, data in engines.items():
        spec = GrammarSpec.from_json(data)
        src = source.get(engine)
        if src is not None:
            raw_ops = {**(data.get("transforms") or {}), **(data.get("heads") or {})}
            patched: dict[str, dict] = {"transforms": {}, "heads": {}}
            for field, group, src_group in (
                ("transforms", spec.transforms, src.transforms),
                ("heads", spec.heads, src.heads),
            ):
                for name, op in group.items():
                    raw = raw_ops.get(name) or {}
                    src_op = src_group.get(name)
                    if src_op is not None:
                        changes = {}
                        if "tasks" not in raw:
                            changes["tasks"] = tuple(src_op.tasks)
                        if "generate" not in raw:
                            changes["generate"] = bool(src_op.generate)
                        if changes:
                            op = dataclasses.replace(op, **changes)
                    patched[field][name] = op
            spec = dataclasses.replace(spec, transforms=patched["transforms"], heads=patched["heads"])
        out[engine] = spec
    return out
