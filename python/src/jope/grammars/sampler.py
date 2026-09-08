"""Samplers over GrammarSpec: random genomes and bounded symbolic expressions.

Used for seed-corpus generation and property tests. Sampled genomes are candidates —
callers validate through the engine bridge and keep the bridge canonical.
"""

from __future__ import annotations

import random
from typing import Any

from ..pipeline_text import format_stage
from .spec import ExprSpec, GrammarSpec, OpSpec


def sample_expr(
    spec: ExprSpec, rng: random.Random, *, elementwise: bool, allow_col: bool = True
) -> str:
    """Grow a bounded expression tree and render it infix (parseable by both engines).

    Always contains at least one variable: a constant-only expression is both useless
    and hazardous (a bare numeric expr param would parse as a number, not a string).
    ``allow_col=False`` omits col(i) refs — required for newmath, whose fit crashes when
    an earlier stage shrinks the matrix below the referenced column (RF4).
    """
    leaves: list[str] = list(spec.vars_common)
    if allow_col:
        leaves += [f"col({i})" for i in range(spec.max_col)]
    if elementwise:
        leaves = list(spec.vars_map) + leaves

    # Depth is 1-based (root node at depth 1) to match the decoding automaton exactly:
    # a non-leaf at depth d puts children at d+1, so non-leaves need depth < max_depth.
    for _ in range(12):
        budget = {"nodes": 0}
        has_var = {"yes": False}

        def leaf() -> str:
            budget["nodes"] += 1
            if rng.random() < 0.25:
                return rng.choice(spec.consts)
            has_var["yes"] = True
            return rng.choice(leaves)

        def grow(depth: int) -> str:
            remaining = spec.max_nodes - budget["nodes"]
            if depth >= spec.max_depth or remaining <= 2 or rng.random() < 0.35:
                return leaf()
            choice = rng.random()
            if choice < 0.45 and remaining >= 2:
                budget["nodes"] += 1
                func = rng.choice(spec.funcs1)
                return f"{func}({grow(depth + 1)})"
            if choice < 0.6 and remaining >= 3 and spec.funcs2:
                budget["nodes"] += 1
                func = rng.choice(spec.funcs2)
                return f"{func}({grow(depth + 1)}, {grow(depth + 1)})"
            if choice < 0.65 and remaining >= 4 and spec.funcs3:
                budget["nodes"] += 3  # the func node itself + its two const-leaf args
                func = rng.choice(spec.funcs3)
                return f"{func}({grow(depth + 1)}, {rng.choice(spec.consts)}, {rng.choice(spec.consts)})"
            budget["nodes"] += 1
            op = rng.choice(spec.binops)
            return f"({grow(depth + 1)} {op} {grow(depth + 1)})"

        expr = grow(1)
        if has_var["yes"] and budget["nodes"] <= spec.max_nodes:
            return expr
    return rng.choice(leaves)


def _sample_params(op: OpSpec, rng: random.Random, spec: GrammarSpec) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for param in op.params:
        if param.kind == "expr":
            if spec.expr is not None and (not param.bins or rng.random() < 0.7):
                params[param.name] = sample_expr(
                    spec.expr,
                    rng,
                    elementwise=(op.name == "map"),
                    allow_col=(spec.engine != "newmath"),
                )
            elif param.bins:
                params[param.name] = rng.choice(list(param.bins))
            elif param.default is not None:
                params[param.name] = param.default
            continue
        if not param.bins:
            continue
        if not param.required and rng.random() < 0.15:
            continue  # exercise engine defaults
        params[param.name] = rng.choice(list(param.bins))
    if op.allow_append and rng.random() < 0.35:
        params["append"] = True
    return params


def sample_genome_from_spec(
    spec: GrammarSpec,
    rng: random.Random,
    *,
    max_transforms: int | None = None,
    expr_fraction: float = 0.2,
    max_cost_class: str = "extreme",
    task: str = "regression",
) -> str:
    """Sample one candidate genome string from the grammar (validate downstream).

    ``task`` filters heads by ``OpSpec.tasks`` (e.g. featmath gpr is regression-only).
    """
    cost_order = {"cheap": 0, "medium": 1, "expensive": 2, "extreme": 3}
    cap = cost_order.get(max_cost_class, 3)
    heads = [
        op
        for op in spec.heads.values()
        if op.generate and cost_order.get(op.cost_class, 0) <= cap and op.supports_task(task)
    ]
    transforms = [op for op in spec.transforms.values() if op.generate and not op.takes_expr]
    expr_ops = [op for op in spec.transforms.values() if op.generate and op.takes_expr]

    limit = min(max_transforms if max_transforms is not None else spec.max_transforms, spec.max_transforms)
    n = rng.randint(0, max(0, limit))
    chosen: list[OpSpec] = [rng.choice(transforms) for _ in range(n)]
    if expr_ops and spec.expr is not None and rng.random() < expr_fraction:
        slot = rng.randrange(len(chosen) + 1) if chosen else 0
        chosen.insert(slot, rng.choice(expr_ops))
        chosen = chosen[: spec.max_transforms]

    stages = [format_stage({"op": op.name, "params": _sample_params(op, rng, spec)}) for op in chosen]
    head = rng.choice(heads)
    stages.append(format_stage({"op": head.name, "params": _sample_params(head, rng, spec)}))
    return "(pipe X " + " ".join(stages) + ")"


def sample_genomes_from_spec(
    spec: GrammarSpec,
    n: int,
    *,
    seed: int = 0,
    expr_fraction: float = 0.2,
    max_cost_class: str = "extreme",
    task: str = "regression",
) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    attempts = 0
    while len(out) < n and attempts < n * 60:
        attempts += 1
        genome = sample_genome_from_spec(
            spec, rng, expr_fraction=expr_fraction, max_cost_class=max_cost_class, task=task
        )
        if genome not in seen:
            seen.add(genome)
            out.append(genome)
    return out
