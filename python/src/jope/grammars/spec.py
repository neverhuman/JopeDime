"""GrammarSpec: the single source of truth for each engine's genome grammar.

The Python modules (featmath.py / newmath.py) are authoritative; a frozen JSON snapshot
is embedded into every corpus artifact so training and inference never depend on repo
state. Parity with the Rust bridges is enforced by the grammar handshake + probe tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

CostClass = Literal["cheap", "medium", "expensive", "extreme", "portfolio"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: Literal["int", "float", "str", "bool", "expr"]
    bins: tuple[Any, ...] = ()
    default: Any | None = None
    required: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "bins": list(self.bins),
            "default": self.default,
            "required": self.required,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ParamSpec":
        return cls(
            name=str(data["name"]),
            kind=data.get("kind", "str"),
            bins=tuple(data.get("bins") or ()),
            default=data.get("default"),
            required=bool(data.get("required")),
        )


@dataclass(frozen=True)
class OpSpec:
    name: str
    role: Literal["transform", "head"]
    params: tuple[ParamSpec, ...] = ()
    families: tuple[str, ...] = ()
    cost_class: CostClass = "cheap"
    allow_append: bool = False  # transforms only; newmath never uses append
    generate: bool = True  # False => parse/encode-only (never sampled)
    takes_expr: bool = False  # op carries a symbolic expression param
    # Tasks this op is worth sampling for. Transforms are task-agnostic; some heads are
    # regression-only (e.g. featmath gpr — a GP-likelihood regressor poor at $/AUC).
    # ORDER IS LOAD-BEARING: tokens.py derives TASK token ids in first-seen order, so
    # "regression" and "binary" MUST stay first — appending new tasks after them keeps
    # every existing token id stable (gen-31 boundary invariant).
    tasks: tuple[str, ...] = ("regression", "binary", "timeseries", "multiclass")

    def param(self, name: str) -> ParamSpec | None:
        for spec in self.params:
            if spec.name == name:
                return spec
        return None

    def supports_task(self, task: str) -> bool:
        return task in self.tasks

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "params": [p.to_json() for p in self.params],
            "families": list(self.families),
            "cost_class": self.cost_class,
            "allow_append": self.allow_append,
            "generate": self.generate,
            "takes_expr": self.takes_expr,
            "tasks": list(self.tasks),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "OpSpec":
        return cls(
            name=str(data["name"]),
            role=data.get("role", "transform"),
            params=tuple(ParamSpec.from_json(p) for p in data.get("params") or ()),
            families=tuple(data.get("families") or ()),
            cost_class=data.get("cost_class", "cheap"),
            allow_append=bool(data.get("allow_append")),
            generate=bool(data.get("generate", True)),
            takes_expr=bool(data.get("takes_expr")),
            tasks=tuple(data.get("tasks") or ("regression", "binary", "timeseries", "multiclass")),
        )


@dataclass(frozen=True)
class ExprSpec:
    """Policy-side bounded symbolic-expression sub-grammar.

    The whitelist is the intersection of both engines' MathExpr grammars (verified
    empirically by tests/test_grammar_parity.py). Policy caps are far tighter than the
    engine caps (64 nodes / depth 16) so decoded expressions always terminate small.
    """

    vars_map: tuple[str, ...] = ("x",)  # elementwise var, legal only inside `map`
    vars_common: tuple[str, ...] = ("row_mean", "row_std", "row_min", "row_max", "row_l2")
    funcs1: tuple[str, ...] = ("abs", "sqrt", "log1p", "sin", "cos", "tanh", "sigmoid")
    funcs2: tuple[str, ...] = ("min", "max", "step")
    funcs3: tuple[str, ...] = ("clip",)
    binops: tuple[str, ...] = ("+", "-", "*", "/")
    # Canonical const strings (kept as strings so render/parse is exact).
    consts: tuple[str, ...] = (
        "-2", "-1", "-0.5", "-0.25", "-0.1", "0", "0.05", "0.1",
        "0.25", "0.5", "1", "2", "3", "5", "10", "20",
    )
    max_col: int = 8
    max_depth: int = 4
    max_nodes: int = 15

    def to_json(self) -> dict[str, Any]:
        return {
            "vars_map": list(self.vars_map),
            "vars_common": list(self.vars_common),
            "funcs1": list(self.funcs1),
            "funcs2": list(self.funcs2),
            "funcs3": list(self.funcs3),
            "binops": list(self.binops),
            "consts": list(self.consts),
            "max_col": self.max_col,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ExprSpec":
        return cls(
            vars_map=tuple(data.get("vars_map") or ("x",)),
            vars_common=tuple(data.get("vars_common") or ()),
            funcs1=tuple(data.get("funcs1") or ()),
            funcs2=tuple(data.get("funcs2") or ()),
            funcs3=tuple(data.get("funcs3") or ()),
            binops=tuple(data.get("binops") or ()),
            consts=tuple(data.get("consts") or ()),
            max_col=int(data.get("max_col", 8)),
            max_depth=int(data.get("max_depth", 4)),
            max_nodes=int(data.get("max_nodes", 15)),
        )


@dataclass(frozen=True)
class GrammarSpec:
    engine: str
    version: str
    evaluator_version: str
    max_transforms: int
    transforms: dict[str, OpSpec]
    heads: dict[str, OpSpec]
    expr: ExprSpec | None
    fallback_genome: str
    max_expr_stages: int = 3

    def op(self, name: str) -> OpSpec | None:
        return self.transforms.get(name) or self.heads.get(name)

    def to_json(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "version": self.version,
            "evaluator_version": self.evaluator_version,
            "max_transforms": self.max_transforms,
            "transforms": {name: op.to_json() for name, op in self.transforms.items()},
            "heads": {name: op.to_json() for name, op in self.heads.items()},
            "expr": self.expr.to_json() if self.expr else None,
            "fallback_genome": self.fallback_genome,
            "max_expr_stages": self.max_expr_stages,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "GrammarSpec":
        return cls(
            engine=str(data["engine"]),
            version=str(data.get("version") or "unknown"),
            evaluator_version=str(data.get("evaluator_version") or ""),
            max_transforms=int(data.get("max_transforms", 6)),
            transforms={k: OpSpec.from_json(v) for k, v in (data.get("transforms") or {}).items()},
            heads={k: OpSpec.from_json(v) for k, v in (data.get("heads") or {}).items()},
            expr=ExprSpec.from_json(data["expr"]) if data.get("expr") else None,
            fallback_genome=str(data.get("fallback_genome") or ""),
            max_expr_stages=int(data.get("max_expr_stages", 3)),
        )

    def with_learned_bins(self, observed: dict[str, dict[str, list[str]]], *, cap: int = 24) -> "GrammarSpec":
        """Union observed param tokens (from corpora) into the static bins, capped."""
        from .vocab import merge_learned_bins  # local import to avoid a cycle

        return merge_learned_bins(self, observed, cap=cap)

    def iter_probe_genomes(self) -> Iterator[tuple[str, str]]:
        """Yield (label, genome) probes covering every op x param x bin.

        Used by the parity test: every probe must validate through the engine bridge and
        canonicalize idempotently.
        """
        cheap_head = self._cheapest_head()
        for op in self.transforms.values():
            yield f"{op.name}:defaults", self._probe([self._stage(op, {})], cheap_head)
            for param in op.params:
                if param.kind == "expr":
                    for bin_value in param.bins:
                        yield (
                            f"{op.name}:{param.name}={bin_value}",
                            self._probe([self._stage(op, {param.name: bin_value})], cheap_head),
                        )
                    continue
                for bin_value in param.bins:
                    yield (
                        f"{op.name}:{param.name}={bin_value}",
                        self._probe([self._stage(op, {param.name: bin_value})], cheap_head),
                    )
        for head in self.heads.values():
            yield f"{head.name}:defaults", self._probe([], head)
            for param in head.params:
                for bin_value in param.bins:
                    yield (
                        f"{head.name}:{param.name}={bin_value}",
                        self._probe([], head, {param.name: bin_value}),
                    )

    def _cheapest_head(self) -> OpSpec:
        for name in ("ridge", "elasticnet", "logistic"):
            if name in self.heads:
                return self.heads[name]
        return next(iter(self.heads.values()))

    def _stage(self, op: OpSpec, params: dict[str, Any]) -> str:
        from ..pipeline_text import format_stage

        merged = dict(params)
        for spec in op.params:
            if spec.required and spec.name not in merged and spec.default is not None:
                merged[spec.name] = spec.default
        return format_stage({"op": op.name, "params": merged})

    def _probe(self, transform_stages: list[str], head: OpSpec, head_params: dict[str, Any] | None = None) -> str:
        from ..pipeline_text import format_stage

        head_stage = format_stage({"op": head.name, "params": head_params or {}})
        stages = transform_stages + [head_stage]
        return "(pipe X " + " ".join(stages) + ")"
