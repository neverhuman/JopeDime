"""Action-sequence tokenization for policy_v3.

A genome is encoded as: ``BOS ENGINE {OP VAL... [APPEND] [EXPR...END]}*k OP_head VAL... EOS``.

Design points (see the plan):
- Engine-qualified op tokens (``OP=featmath:rff_cos`` != ``OP=newmath:rff_cos``): same-name
  ops have different grids and semantics per engine.
- No PARAM tokens: param identity is positional — given the op, exactly one VAL token per
  declared param, in spec order. Optional params use an ``<ABSENT>`` VAL.
- EXPR sub-tokens are shared across engines (the whitelist is the verified intersection).
  ``EXPR_END`` is forced when the arity stack empties (0 NLL; explicit boundary).
- Vocab ids are append-only across versions; the table is stored in checkpoints.
"""

from __future__ import annotations

import math
import os
import re
import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..grammars.spec import ExprSpec, GrammarSpec
from ..pipeline_text import format_param_token

PAD, BOS, EOS = 0, 1, 2
SPECIALS = ("PAD", "BOS", "EOS")
ABSENT = "<ABSENT>"
GENERATION31_VOCAB_SIZE = 1_103
GENERATION31_TOKEN_SHA256 = "64854f2f795424ed05f1094ad5bf3a1d8ab72aedf6e0694f42d4a5212006b491"
GENERATION31_SNAPSHOT_SHA256 = "f221b02503dc3060475967b2597de3f71f661c1ad2e0ec7b15d1a4fcb41d9c72"


@dataclass
class Slot:
    param: str
    kind: str
    val_ids: list[int]
    bin_texts: list[str]  # formatted bin values, aligned with val_ids
    absent_id: int | None
    is_expr: bool
    required: bool
    default: Any


@dataclass
class OpInfo:
    token_id: int
    engine: str
    name: str
    role: str  # "transform" | "head"
    slots: list[Slot]
    allow_append: bool
    takes_expr: bool
    # False => parse/encode-only: historical rows still replay, but SAMPLING must never
    # legalize this op (quarantined/unstable/parse-compat ops leak into proposals otherwise).
    generate: bool = True


@dataclass
class ExprTable:
    var_map_ids: list[int]  # elementwise vars (only legal inside `map`)
    var_col_ids: list[int]  # col(i) refs — masked for engines where runtime width shrinks
    var_common_ids: list[int]
    const_ids: list[int]
    const_values: list[float]
    func_arity: dict[int, int]  # func/binop token id -> arity
    end_id: int
    symbol: dict[int, str]  # token id -> rendered symbol
    is_binop: dict[int, bool]
    max_depth: int
    max_nodes: int

    def leaf_ids(self, elementwise: bool, *, allow_col: bool = True) -> list[int]:
        base = list(self.var_common_ids) + list(self.const_ids)
        if allow_col:
            base = list(self.var_col_ids) + base
        return (list(self.var_map_ids) + base) if elementwise else base

    def nonleaf_ids(self) -> list[int]:
        return list(self.func_arity)


@dataclass
class ActionVocab:
    tokens: list[str]
    index: dict[str, int]
    engines: list[str]
    engine_ids: dict[str, int]
    ops: dict[int, OpInfo]
    op_id_by_name: dict[tuple[str, str], int]  # (engine, op) -> token id
    engine_transforms: dict[str, list[int]]
    engine_heads: dict[str, list[int]]
    append_ids: tuple[int, int, int]  # (APPEND=<ABSENT>, APPEND=0, APPEND=1)
    expr: ExprTable
    max_transforms: dict[str, int]
    max_expr_stages: dict[str, int]
    version: str
    # Fused regression+binary support: a TASK token follows ENGINE, and heads are masked
    # by task at decode. Legacy checkpoints without these fall back to regression-only.
    tasks: list[str] = field(default_factory=lambda: ["regression"])
    task_ids: dict[str, int] = field(default_factory=dict)
    head_tasks: dict[int, tuple[str, ...]] = field(default_factory=dict)  # head op_id -> tasks
    counters: dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.tokens)

    def heads_for(self, engine: str, task: str, *, generate_only: bool = False) -> list[int]:
        """Head op ids for (engine, task) — filtered by each head's OpSpec.tasks.

        ``generate_only=True`` additionally drops parse-only heads (``generate=False``) —
        used by SAMPLING; replay stays permissive so historical rows still encode.
        """
        return [
            hid
            for hid in self.engine_heads[engine]
            if task in self.head_tasks.get(hid, ("regression", "binary", "timeseries", "multiclass"))
            and (not generate_only or self.ops[hid].generate)
        ]

    def to_json(self) -> dict[str, Any]:
        return {"tokens": self.tokens, "version": self.version}


class EncodeError(ValueError):
    pass


from .tokens_build import (
    append_missing_task_tokens, build_action_vocab, extend_action_vocab_append_only,
    extend_engine_ops_append_only, reconcile_head_tasks_from_grammar,
)
from .tokens_generation import (
    build_generation31_action_vocab, build_prime_action_vocab, generation31_vocab_snapshot_path,
)
from .tokens_expr import (
    _expr_to_prefix, _parse_expr, _snap_const, _tokenize_expr, encode_expr, render_expr,
)
from .tokens_actions import _snap_bin, decode_tokens, encode_actions
