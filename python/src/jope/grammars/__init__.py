"""Per-engine grammar specifications (single source of truth for the genome language)."""

from __future__ import annotations

from .families import FAMILY_LIST_V3
from .spec import ExprSpec, GrammarSpec, OpSpec, ParamSpec

__all__ = [
    "FAMILY_LIST_V3",
    "ExprSpec",
    "GrammarSpec",
    "OpSpec",
    "ParamSpec",
    "grammar_for",
    "all_grammars",
]


def grammar_for(engine: str) -> GrammarSpec:
    if engine == "featmath":
        from .featmath import GRAMMAR

        return GRAMMAR
    if engine == "newmath":
        from .newmath import GRAMMAR

        return GRAMMAR
    if engine == "prime":
        from .prime import GRAMMAR

        return GRAMMAR
    raise ValueError(f"unknown engine {engine!r}")


def all_grammars(*, include_prime: bool = False) -> list[GrammarSpec]:
    specs = [grammar_for("featmath"), grammar_for("newmath")]
    if include_prime:
        specs.append(grammar_for("prime"))
    return specs


def prime_grammar() -> GrammarSpec:
    return grammar_for("prime")
