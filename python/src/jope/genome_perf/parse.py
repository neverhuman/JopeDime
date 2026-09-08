"""Shared enriched parse of a genome. Every feature module consumes `ParsedGenome`, nothing else.

Built on `fitness_corrector.genome_features.split_stages_quoted` / `parse_stage` -- the quote-aware
tokenizer, NOT the older regex parser in `full_cv_meta_selector_v2`, which drops 13.4% of labels
whose params contain parens inside quoted expressions.

Grammar facts (families, cost_class, param bins, head role) come from `jope.grammars`, which is the
handshake-tested source of truth shared with Rust. This is deliberately NOT the 9-bucket `FAMILY`
map inside `genome_features.py`: that map is stale and silently collapses 14+ live featmath ops
(quantile_bins, poly_interact, row_stats, row_spectral, rff_sin, hinge, landmark_dist, bin_onehot,
norm_row, step, target_bin_encode, oblique_hinge, winsor_moments) into `fam_other`.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field

from jope.fitness_corrector.genome_features import parse_stage, split_stages_quoted
from jope.grammars import FAMILY_LIST_V3, grammar_for

COST_CLASSES = ("cheap", "medium", "expensive", "extreme", "portfolio")
_COL_RE = re.compile(r"col\((\d+)\)")


@dataclass
class Stage:
    op: str
    params: dict[str, str]
    index: int
    n_stages: int
    role: str                      # "transform" | "head" | "unknown"
    families: tuple[str, ...]
    cost_class: str
    expr: str | None = None

    @property
    def position_norm(self) -> float:
        """Position in [0,1]; 0.0 for a single-stage pipeline."""
        return self.index / (self.n_stages - 1) if self.n_stages > 1 else 0.0


@dataclass
class ParsedGenome:
    text: str
    engine: str
    stages: list[Stage]
    ok: bool
    skeleton: str = ""
    exprs: list[str] = field(default_factory=list)

    @property
    def head(self) -> Stage | None:
        return self.stages[-1] if self.stages else None

    @property
    def transforms(self) -> list[Stage]:
        return self.stages[:-1]

    def ops(self) -> list[str]:
        return [s.op for s in self.stages]


@functools.lru_cache(maxsize=4)
def _grammar_index(engine: str):
    """-> (op -> OpSpec, max_transforms, expr_spec, head_ops, param_bins)."""
    g = grammar_for(engine)
    # GrammarSpec.transforms / .heads are dicts keyed by op name -> OpSpec.
    ops = {**dict(g.transforms), **dict(g.heads)}
    heads = set(g.heads)
    return ops, int(g.max_transforms), g.expr, heads, g


@functools.lru_cache(maxsize=200_000)
def parse(text: str, engine: str = "featmath") -> ParsedGenome:
    """Parse one genome. Cached on (text, engine) -- 34,419 distinct genomes parse in ~5 s once."""
    ops_index, _max_t, _expr, head_ops, _g = _grammar_index(engine)
    raw = split_stages_quoted(text)
    if raw is None:
        # (sym ...) symbolic genomes and portfolios are not pipelines; callers must check .ok
        return ParsedGenome(text=text, engine=engine, stages=[], ok=False)

    n = len(raw)
    stages: list[Stage] = []
    exprs: list[str] = []
    for i, chunk in enumerate(raw):
        op, params = parse_stage(chunk)
        spec = ops_index.get(op)
        expr = params.get("expr")
        if expr:
            exprs.append(expr)
        stages.append(
            Stage(
                op=op,
                params=params,
                index=i,
                n_stages=n,
                role=("head" if op in head_ops else "transform") if spec else "unknown",
                families=tuple(getattr(spec, "families", ()) or ()) if spec else (),
                cost_class=getattr(spec, "cost_class", "medium") if spec else "medium",
                expr=expr,
            )
        )
    return ParsedGenome(
        text=text,
        engine=engine,
        stages=stages,
        ok=True,
        skeleton=">".join(s.op for s in stages),
        exprs=exprs,
    )


def max_transforms(engine: str) -> int:
    return _grammar_index(engine)[1]


def expr_spec(engine: str):
    return _grammar_index(engine)[2]


def param_bins(engine: str, op: str, param: str):
    """-> the grammar's declared bin list for a param, or None."""
    spec = _grammar_index(engine)[0].get(op)
    if spec is None:
        return None
    for p in getattr(spec, "params", ()) or ():
        if p.name == param:
            return getattr(p, "bins", None)
    return None


def col_refs(expr: str) -> list[int]:
    """Column indices referenced by an expression, e.g. 'sin(col(6))' -> [6]."""
    return [int(m) for m in _COL_RE.findall(expr or "")]


def dominant_family(pg: ParsedGenome) -> str:
    """Most common family across transform stages; '' when none is declared."""
    counts: dict[str, int] = {}
    for s in pg.transforms:
        for fam in s.families:
            counts[fam] = counts.get(fam, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def family_vector(pg: ParsedGenome) -> dict[str, int]:
    """Multi-label family counts over the canonical 21 keys (ops carry a TUPLE of families)."""
    out = {f: 0 for f in FAMILY_LIST_V3}
    for s in pg.stages:
        for fam in s.families:
            if fam in out:
                out[fam] += 1
    return out
