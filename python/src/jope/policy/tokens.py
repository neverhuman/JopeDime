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

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def build_action_vocab(specs: Sequence[GrammarSpec]) -> ActionVocab:
    tokens: list[str] = list(SPECIALS)

    def add(token: str) -> int:
        tokens.append(token)
        return len(tokens) - 1

    engines = [spec.engine for spec in specs]
    engine_ids = {engine: add(f"ENGINE={engine}") for engine in engines}
    head_tasks: dict[int, tuple[str, ...]] = {}

    expr_spec = next((s.expr for s in specs if s.expr is not None), None) or ExprSpec()
    # EXPR block (shared) — built before ops so op expr slots can reference macro bins only.
    var_map_ids = [add(f"EXPR_VAR={name}") for name in expr_spec.vars_map]
    var_common_ids = [add(f"EXPR_VAR={name}") for name in expr_spec.vars_common]
    var_col_ids = [add(f"EXPR_VAR=col{i}") for i in range(expr_spec.max_col)]
    const_ids = [add(f"EXPR_CONST={c}") for c in expr_spec.consts]
    const_values = [float(c) for c in expr_spec.consts]
    func_arity: dict[int, int] = {}
    is_binop: dict[int, bool] = {}
    symbol: dict[int, str] = {}
    for name in expr_spec.funcs1:
        tid = add(f"EXPR_FUNC={name}")
        func_arity[tid] = 1
        is_binop[tid] = False
        symbol[tid] = name
    for name in expr_spec.funcs2:
        tid = add(f"EXPR_FUNC={name}")
        func_arity[tid] = 2
        is_binop[tid] = False
        symbol[tid] = name
    for name in expr_spec.funcs3:
        tid = add(f"EXPR_FUNC={name}")
        func_arity[tid] = 3
        is_binop[tid] = False
        symbol[tid] = name
    for name in expr_spec.binops:
        tid = add(f"EXPR_BINOP={name}")
        func_arity[tid] = 2
        is_binop[tid] = True
        symbol[tid] = name
    end_id = add("EXPR_END")
    for tid, name in zip(var_map_ids, expr_spec.vars_map):
        symbol[tid] = name
    for tid, name in zip(var_common_ids, list(expr_spec.vars_common)):
        symbol[tid] = name
    for tid, name in zip(var_col_ids, [f"col({i})" for i in range(expr_spec.max_col)]):
        symbol[tid] = name
    for tid, const in zip(const_ids, expr_spec.consts):
        symbol[tid] = const

    append_ids = (add(f"APPEND={ABSENT}"), add("APPEND=0"), add("APPEND=1"))

    ops: dict[int, OpInfo] = {}
    op_id_by_name: dict[tuple[str, str], int] = {}
    engine_transforms: dict[str, list[int]] = {}
    engine_heads: dict[str, list[int]] = {}
    for spec in specs:
        transforms: list[int] = []
        heads: list[int] = []
        for group, bucket in ((spec.transforms, transforms), (spec.heads, heads)):
            for name in sorted(group):
                op = group[name]
                op_id = add(f"OP={spec.engine}:{name}")
                slots: list[Slot] = []
                for param in op.params:
                    bin_texts = [format_param_token(b) for b in param.bins]
                    val_ids = [
                        add(f"VAL={spec.engine}:{name}:{param.name}={text}") for text in bin_texts
                    ]
                    absent_id = None
                    if not param.required:
                        absent_id = add(f"VAL={spec.engine}:{name}:{param.name}={ABSENT}")
                    slots.append(
                        Slot(
                            param=param.name,
                            kind=param.kind,
                            val_ids=val_ids,
                            bin_texts=bin_texts,
                            absent_id=absent_id,
                            is_expr=(param.kind == "expr"),
                            required=param.required,
                            default=param.default,
                        )
                    )
                ops[op_id] = OpInfo(
                    token_id=op_id,
                    engine=spec.engine,
                    name=name,
                    role=op.role,
                    slots=slots,
                    allow_append=op.allow_append and op.role == "transform",
                    takes_expr=op.takes_expr,
                    generate=bool(getattr(op, "generate", True)),
                )
                op_id_by_name[(spec.engine, name)] = op_id
                bucket.append(op_id)
                if op.role == "head":
                    head_tasks[op_id] = tuple(op.tasks)
        engine_transforms[spec.engine] = transforms
        engine_heads[spec.engine] = heads

    # TASK token (fused regression+binary), appended so existing OP/VAL/EXPR ids stay
    # stable across the non-fused -> fused vocab bump. Union of tasks any head declares.
    task_set: list[str] = []
    for spec in specs:
        for op in spec.heads.values():
            for t in op.tasks:
                if t not in task_set:
                    task_set.append(t)
    if not task_set:
        task_set = ["regression"]
    task_ids = {task: add(f"TASK={task}") for task in task_set}

    expr_table = ExprTable(
        var_map_ids=var_map_ids,
        var_col_ids=var_col_ids,
        var_common_ids=var_common_ids,
        const_ids=const_ids,
        const_values=const_values,
        func_arity=func_arity,
        end_id=end_id,
        symbol=symbol,
        is_binop=is_binop,
        max_depth=expr_spec.max_depth,
        max_nodes=expr_spec.max_nodes,
    )
    version = "+".join(f"{spec.engine}:{spec.version}" for spec in specs)
    return ActionVocab(
        tokens=tokens,
        index={token: i for i, token in enumerate(tokens)},
        engines=engines,
        engine_ids=engine_ids,
        ops=ops,
        op_id_by_name=op_id_by_name,
        engine_transforms=engine_transforms,
        engine_heads=engine_heads,
        append_ids=append_ids,
        expr=expr_table,
        max_transforms={spec.engine: spec.max_transforms for spec in specs},
        max_expr_stages={spec.engine: spec.max_expr_stages for spec in specs},
        version=version,
        tasks=task_set,
        task_ids=task_ids,
        head_tasks=head_tasks,
    )


def extend_action_vocab_append_only(base: ActionVocab, spec: GrammarSpec) -> ActionVocab:
    """Append one engine without changing any existing token id.

    Prime uses this instead of rebuilding the legacy table: generation-31
    checkpoints and corpora keep their original 1,103-token prefix exactly.
    """
    if spec.engine in base.engines:
        raise ValueError(f"engine already present: {spec.engine}")
    vocab = copy.deepcopy(base)

    def add(token: str) -> int:
        if token in vocab.index:
            raise ValueError(f"token collision while extending vocab: {token}")
        vocab.tokens.append(token)
        token_id = len(vocab.tokens) - 1
        vocab.index[token] = token_id
        return token_id

    engine_id = add(f"ENGINE={spec.engine}")
    vocab.engines.append(spec.engine)
    vocab.engine_ids[spec.engine] = engine_id
    transforms: list[int] = []
    heads: list[int] = []
    for group, bucket in ((spec.transforms, transforms), (spec.heads, heads)):
        # Grammar insertion order is canonical for append-only extensions. Prime's
        # loader inserts transforms in Rust wire-ID order, so policy IDs follow the
        # registry rather than an independent alphabetical ordering.
        for name, op in group.items():
            op_id = add(f"OP={spec.engine}:{name}")
            slots: list[Slot] = []
            for param in op.params:
                bin_texts = [format_param_token(value) for value in param.bins]
                val_ids = [
                    add(f"VAL={spec.engine}:{name}:{param.name}={text}")
                    for text in bin_texts
                ]
                absent_id = None
                if not param.required:
                    absent_id = add(f"VAL={spec.engine}:{name}:{param.name}={ABSENT}")
                slots.append(Slot(
                    param=param.name,
                    kind=param.kind,
                    val_ids=val_ids,
                    bin_texts=bin_texts,
                    absent_id=absent_id,
                    is_expr=(param.kind == "expr"),
                    required=param.required,
                    default=param.default,
                ))
            vocab.ops[op_id] = OpInfo(
                token_id=op_id,
                engine=spec.engine,
                name=name,
                role=op.role,
                slots=slots,
                allow_append=op.allow_append and op.role == "transform",
                takes_expr=op.takes_expr,
                generate=op.generate,
            )
            vocab.op_id_by_name[(spec.engine, name)] = op_id
            bucket.append(op_id)
            if op.role == "head":
                vocab.head_tasks[op_id] = tuple(op.tasks)
    vocab.engine_transforms[spec.engine] = transforms
    vocab.engine_heads[spec.engine] = heads
    vocab.max_transforms[spec.engine] = spec.max_transforms
    vocab.max_expr_stages[spec.engine] = spec.max_expr_stages
    vocab.version = f"{vocab.version}+{spec.engine}:{spec.version}"
    return vocab


# Canonical task universe, in wire order (jope-dime TASK_CODE: regression=0,
# binary=1, timeseries=2, multiclass=3). Append-only: never reorder, only extend.
CANONICAL_TASKS = ("regression", "binary", "timeseries", "multiclass")


def append_missing_task_tokens(
    base: ActionVocab, tasks: Sequence[str] = CANONICAL_TASKS
) -> ActionVocab:
    """Append TASK tokens for canonical tasks absent from the vocab, id-stable.

    The append-only engine/op extenders never create TASK tokens (the gen-31 base
    pins its own TASK=regression/TASK=binary tail behind a sha256 fixture), so the
    four-task program's timeseries and multiclass conditioning tokens need this
    final pass. TASK tokens are conditioning tokens: they must exist for every
    canonical task even while no engine head declares it yet (multiclass decode
    heads are the four ``multiclass_*`` engine heads; see
    ``reconcile_head_tasks_from_grammar`` for how task CAPABILITY is granted).
    New tokens land strictly at the END of the table in the canonical order
    above, so every existing token id is untouched.
    """
    vocab = copy.deepcopy(base)
    for task in tasks:
        if task in vocab.task_ids:
            continue
        token = f"TASK={task}"
        if token in vocab.index:
            raise ValueError(f"token collision while appending task token: {token}")
        vocab.tokens.append(token)
        token_id = len(vocab.tokens) - 1
        vocab.index[token] = token_id
        vocab.task_ids[task] = token_id
        vocab.tasks.append(task)
    return vocab


def reconcile_head_tasks_from_grammar(
    base: ActionVocab,
    spec: GrammarSpec,
    *,
    grant: Sequence[str],
    exclude: Sequence[str] = (),
) -> ActionVocab:
    """Grant live-grammar task capability to heads the append-only extenders skipped.

    Token IDENTITY is the frozen gen-31 snapshot's; task CAPABILITY is the live
    grammar's. The extenders are append-only, so a head already present in the
    snapshot keeps the snapshot's narrow ``("regression", "binary")`` tuple forever
    even after the live grammar widens it — which is why ``featmath`` had zero
    sampleable timeseries heads while ``featmath.py`` declared eleven of them.

    ``heads_for`` gates on two dimensions: the task must be in ``head_tasks[hid]``
    AND (for sampling) ``ops[hid].generate`` must be true. This pass moves only the
    first. It appends NO tokens and shifts NO ids, so the gen-31 prefix, every
    identity fixture, and every existing checkpoint are untouched — ``head_tasks``
    is rebuilt from grammar specs on every load and is never serialized.

    ``exclude`` is mandatory rather than cosmetic: the live tuples are not fully
    curated. ``logistic`` claims all four tasks, but ``corpus_eval`` rejects
    classifier heads for timeseries, so granting it would emit genomes that decode
    cleanly and then fail at evaluation. Grant only what the engine will execute.
    """

    vocab = copy.deepcopy(base)
    granted = tuple(dict.fromkeys(grant))
    unknown = [task for task in granted if task not in CANONICAL_TASKS]
    if unknown:
        raise ValueError(f"cannot grant non-canonical task(s): {sorted(unknown)}")
    excluded = set(exclude)
    order = {task: index for index, task in enumerate(CANONICAL_TASKS)}
    for name in spec.heads:
        if name in excluded:
            continue
        op = spec.op(name)
        op_id = vocab.op_id_by_name.get((spec.engine, name))
        if op is None or op_id is None:
            continue
        current = tuple(vocab.head_tasks.get(op_id, ()))
        additions = [
            task for task in granted if task in op.tasks and task not in current
        ]
        if additions:
            vocab.head_tasks[op_id] = tuple(
                sorted({*current, *additions}, key=lambda task: order[task])
            )
    return vocab


def extend_engine_ops_append_only(
    base: ActionVocab,
    spec: GrammarSpec,
    names: Sequence[str],
) -> ActionVocab:
    """Append parse-compatibility ops to an existing engine without ID drift."""
    if spec.engine not in base.engines:
        raise ValueError(f"engine is not present: {spec.engine}")
    vocab = copy.deepcopy(base)

    def add(token: str) -> int:
        if token in vocab.index:
            raise ValueError(f"token collision while extending vocab: {token}")
        vocab.tokens.append(token)
        token_id = len(vocab.tokens) - 1
        vocab.index[token] = token_id
        return token_id

    for name in names:
        op = spec.op(name)
        if op is None:
            raise ValueError(f"unknown compatibility op: {spec.engine}:{name}")
        if (spec.engine, name) in vocab.op_id_by_name:
            raise ValueError(f"operator already present: {spec.engine}:{name}")
        op_id = add(f"OP={spec.engine}:{name}")
        slots: list[Slot] = []
        for param in op.params:
            bin_texts = [format_param_token(value) for value in param.bins]
            val_ids = [
                add(f"VAL={spec.engine}:{name}:{param.name}={text}")
                for text in bin_texts
            ]
            absent_id = None
            if not param.required:
                absent_id = add(
                    f"VAL={spec.engine}:{name}:{param.name}={ABSENT}"
                )
            slots.append(
                Slot(
                    param=param.name,
                    kind=param.kind,
                    val_ids=val_ids,
                    bin_texts=bin_texts,
                    absent_id=absent_id,
                    is_expr=(param.kind == "expr"),
                    required=param.required,
                    default=param.default,
                )
            )
        vocab.ops[op_id] = OpInfo(
            token_id=op_id,
            engine=spec.engine,
            name=name,
            role=op.role,
            slots=slots,
            allow_append=op.allow_append and op.role == "transform",
            takes_expr=op.takes_expr,
            generate=op.generate,
        )
        vocab.op_id_by_name[(spec.engine, name)] = op_id
        if op.role == "head":
            vocab.engine_heads[spec.engine].append(op_id)
            vocab.head_tasks[op_id] = tuple(op.tasks)
        else:
            vocab.engine_transforms[spec.engine].append(op_id)
    vocab.version += f"+{spec.engine}:compat"
    return vocab


def generation31_vocab_snapshot_path() -> Path:
    """Return the Prime-owned frozen grammar snapshot for generation 31."""
    if configured := os.environ.get("JOPE_GENERATION31_VOCAB"):
        return Path(configured).expanduser().resolve(strict=False)
    from ..prime_config import load_config

    return load_config().generation31_vocab_snapshot


def build_generation31_action_vocab(path: str | Path | None = None) -> ActionVocab:
    """Rehydrate and verify generation 31's immutable 1,103-token vocabulary."""
    from ..grammars.spec import GrammarSpec

    snapshot_path = Path(path) if path is not None else generation31_vocab_snapshot_path()
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot_hash = hashlib.sha256(snapshot_bytes).hexdigest()
    if snapshot_hash != GENERATION31_SNAPSHOT_SHA256:
        raise ValueError(
            "generation-31 grammar snapshot drift: "
            f"sha256={snapshot_hash}"
        )
    snapshot = json.loads(snapshot_bytes)
    if snapshot.get("schema_version") != 4 or snapshot.get("syntax") != "active_multi_engine_v3":
        raise ValueError("generation-31 vocabulary snapshot has an unsupported schema")
    engines = snapshot.get("engines")
    if not isinstance(engines, dict) or tuple(engines) != ("featmath", "newmath"):
        raise ValueError("generation-31 vocabulary snapshot must contain featmath then newmath")
    specs = [GrammarSpec.from_json(engines[name]) for name in ("featmath", "newmath")]
    vocab = build_action_vocab(specs)
    payload = json.dumps(
        vocab.tokens, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    token_hash = hashlib.sha256(payload).hexdigest()
    if vocab.size != GENERATION31_VOCAB_SIZE or token_hash != GENERATION31_TOKEN_SHA256:
        raise ValueError(
            "generation-31 vocabulary snapshot drift: "
            f"size={vocab.size}, token_sha256={token_hash}"
        )
    if (
        vocab.tokens[59] != "VAL=featmath:bin_onehot:bins=15"
        or vocab.tokens[-2:] != ["TASK=regression", "TASK=binary"]
    ):
        raise ValueError("generation-31 vocabulary boundary fixture drift")
    return vocab


def build_prime_action_vocab() -> ActionVocab:
    """Append Prime after generation 31's exact checkpoint vocabulary."""
    from ..grammars import grammar_for, prime_grammar
    from ..grammars.prime import registry_snapshot

    vocab = extend_action_vocab_append_only(
        build_generation31_action_vocab(), prime_grammar()
    )
    compatibility_names = [
        str(entry["name"])
        for entry in registry_snapshot()
        if entry.get("role") == "head"
        and "compatibility" in entry.get("tags", [])
    ]
    compatibility_names.extend(
        ("logit_ridge", "tab_boost", "gam_boost", "extra_trees")
    )
    # Four-task program: the multiclass decode heads. These are genuinely NEW
    # tokens (the snapshot has no multiclass head at all), appended here — after
    # every pre-existing op and before the TASK tokens — so every id below the
    # previous vocab size is unchanged and existing checkpoints keep loading via
    # `_validate_action_vocab_compatibility`'s append-only prefix rule. A vocab
    # token can never be renamed once appended, so these names are pinned to what
    # feat-math main actually executes (`corpus_eval` + `fit_v3_head`).
    compatibility_names.extend(
        (
            "multiclass_ridge",
            "multiclass_ovr_logistic",
            "multiclass_diagonal_lda",
            "multiclass_nearest_centroid",
        )
    )
    vocab = extend_engine_ops_append_only(
        vocab,
        grammar_for("featmath"),
        list(dict.fromkeys(compatibility_names)),
    )
    # Four-task program: grant TIMESERIES capability to the featmath heads the
    # snapshot froze at ("regression", "binary"). Appends no tokens — see
    # reconcile_head_tasks_from_grammar. `logistic`/`logit_ridge` are excluded
    # because corpus_eval rejects classifier heads for timeseries, so granting them
    # would emit genomes that decode and then fail evaluation. Multiclass is NOT
    # granted here: only the four multiclass_* engine heads implement
    # `multiclass_probabilities`, so every other head fails closed with "bound model
    # does not carry a multiclass head".
    vocab = reconcile_head_tasks_from_grammar(
        vocab,
        grammar_for("featmath"),
        grant=("timeseries",),
        exclude=("logistic", "logit_ridge"),
    )
    # TASK=timeseries / TASK=multiclass are appended AFTER every engine/op/compat
    # token so the gen-31 prefix (and every later id) is stable.
    return append_missing_task_tokens(vocab)


# ---------------------------------------------------------------------------
# Expression parsing (string -> AST -> prefix token ids) and rendering back.
# Grammar: the verified intersection of both engines' MathExpr grammars.
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize_expr(text: str) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "(),+*/":
            out.append(ch)
            i += 1
            continue
        if ch == "-":
            prev = out[-1] if out else None
            if (prev is None or prev in "(,+-*/") and i + 1 < n and (text[i + 1].isdigit() or text[i + 1] == "."):
                match = _NUM_RE.match(text, i + 1)
                assert match is not None
                out.append("-" + match.group(0))
                i = match.end()
                continue
            out.append("-")
            i += 1
            continue
        match = _NUM_RE.match(text, i)
        if match and match.start() == i:
            out.append(match.group(0))
            i = match.end()
            continue
        match = _IDENT_RE.match(text, i)
        if match and match.start() == i:
            out.append(match.group(0))
            i = match.end()
            continue
        raise EncodeError(f"cannot tokenize expr at {text[i:i+8]!r}")
    return out


def _parse_expr(text: str) -> Any:
    """Parse the shared expr grammar into an AST of tuples."""
    tokens = _tokenize_expr(text)
    pos = {"i": 0}

    def peek() -> str | None:
        return tokens[pos["i"]] if pos["i"] < len(tokens) else None

    def take(expected: str | None = None) -> str:
        tok = peek()
        if tok is None:
            raise EncodeError("unexpected end of expr")
        if expected is not None and tok != expected:
            raise EncodeError(f"expected {expected!r}, got {tok!r}")
        pos["i"] += 1
        return tok

    def parse_primary() -> Any:
        tok = take()
        if tok == "(":
            node = parse_binary(1)
            take(")")
            return node
        if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", tok):
            return ("const", tok)
        if _IDENT_RE.fullmatch(tok):
            if peek() == "(":
                take("(")
                args = [parse_binary(1)]
                while peek() == ",":
                    take(",")
                    args.append(parse_binary(1))
                take(")")
                if tok == "col":
                    if len(args) != 1 or args[0][0] != "const":
                        raise EncodeError("col() expects one integer literal")
                    return ("var", f"col({int(float(args[0][1]))})")
                return ("func", tok, args)
            return ("var", tok)
        raise EncodeError(f"unexpected expr token {tok!r}")

    def parse_binary(min_prec: int) -> Any:
        prec = {"+": 1, "-": 1, "*": 2, "/": 2}
        left = parse_primary()
        while True:
            tok = peek()
            if tok not in prec or prec[tok] < min_prec:
                return left
            take()
            right = parse_binary(prec[tok] + 1)
            left = ("binop", tok, left, right)

    node = parse_binary(1)
    if pos["i"] != len(tokens):
        raise EncodeError(f"trailing expr tokens: {tokens[pos['i']:]}")
    return node


def _snap_const(value: float, table: ExprTable) -> int:
    best = 0
    best_cost = float("inf")
    for i, bin_value in enumerate(table.const_values):
        if value == bin_value:
            return table.const_ids[i]
        if bin_value != 0 and value != 0 and (bin_value > 0) == (value > 0):
            cost = abs(math.log(abs(value) / abs(bin_value)))
        else:
            cost = abs(value - bin_value) + 1.0
        if cost < best_cost:
            best_cost = cost
            best = i
    return table.const_ids[best]


def _expr_to_prefix(node: Any, table: ExprTable, out: list[int], counters: dict[str, int]) -> None:
    kind = node[0]
    if kind == "const":
        value = float(node[1])
        tid = _snap_const(value, table)
        if abs(table.const_values[table.const_ids.index(tid)] - value) > 1e-12:
            counters["const_snapped"] = counters.get("const_snapped", 0) + 1
        out.append(tid)
        return
    if kind == "var":
        name = node[1]
        for tid, sym in table.symbol.items():
            if sym == name and (
                tid in table.var_map_ids or tid in table.var_common_ids or tid in table.var_col_ids
            ):
                out.append(tid)
                return
        raise EncodeError(f"unknown expr var {name!r}")
    if kind == "func":
        name, args = node[1], node[2]
        for tid, arity in table.func_arity.items():
            if not table.is_binop.get(tid) and table.symbol[tid] == name:
                if arity != len(args):
                    raise EncodeError(f"{name} arity {len(args)} != {arity}")
                out.append(tid)
                for arg in args:
                    _expr_to_prefix(arg, table, out, counters)
                return
        raise EncodeError(f"unknown expr func {name!r}")
    if kind == "binop":
        op, left, right = node[1], node[2], node[3]
        for tid, arity in table.func_arity.items():
            if table.is_binop.get(tid) and table.symbol[tid] == op:
                out.append(tid)
                _expr_to_prefix(left, table, out, counters)
                _expr_to_prefix(right, table, out, counters)
                return
        raise EncodeError(f"unknown expr binop {op!r}")
    raise EncodeError(f"bad expr node {node!r}")


def encode_expr(text: str, table: ExprTable, counters: dict[str, int] | None = None) -> list[int]:
    """Expr string -> prefix token ids (without EXPR_END)."""
    counters = counters if counters is not None else {}
    out: list[int] = []
    _expr_to_prefix(_parse_expr(text), table, out, counters)
    return out


def render_expr(ids: Sequence[int], table: ExprTable) -> str:
    """Prefix token ids -> deterministic infix string (parseable by both engines)."""
    pos = {"i": 0}

    def read() -> str:
        tid = ids[pos["i"]]
        pos["i"] += 1
        arity = table.func_arity.get(tid)
        if arity is None:
            return table.symbol[tid]
        args = [read() for _ in range(arity)]
        if table.is_binop.get(tid):
            return f"({args[0]} {table.symbol[tid]} {args[1]})"
        return f"{table.symbol[tid]}({', '.join(args)})"

    text = read()
    if pos["i"] != len(ids):
        raise ValueError("trailing expr tokens in render")
    return text


# ---------------------------------------------------------------------------
# Genome (actions dict) <-> token ids
# ---------------------------------------------------------------------------


def encode_actions(
    engine: str, actions: dict[str, Any], vocab: ActionVocab, *, task: str = "regression"
) -> list[int]:
    """Encode parsed actions into a token id sequence (BOS ENGINE TASK ...EOS)."""
    if task not in vocab.task_ids:
        task = vocab.tasks[0]
    out = [BOS, vocab.engine_ids[engine], vocab.task_ids[task]]
    counters = vocab.counters
    stages = list(actions.get("transforms") or []) + [actions["head"]]
    for stage in stages:
        name = stage["op"]
        op_id = vocab.op_id_by_name.get((engine, name))
        if op_id is None:
            raise EncodeError(f"unknown op {engine}:{name}")
        op = vocab.ops[op_id]
        out.append(op_id)
        params = dict(stage.get("params") or {})
        append_flag = params.pop("append", None)
        for slot in op.slots:
            value = params.get(slot.param)
            if slot.is_expr:
                if value is None:
                    value = slot.default
                text = str(value)
                if text in slot.bin_texts:  # macro bin
                    out.append(slot.val_ids[slot.bin_texts.index(text)])
                else:
                    out.extend(encode_expr(text, vocab.expr, counters))
                    out.append(vocab.expr.end_id)
                continue
            if value is None:
                if slot.absent_id is not None:
                    out.append(slot.absent_id)
                elif slot.val_ids:
                    counters["required_defaulted"] = counters.get("required_defaulted", 0) + 1
                    out.append(slot.val_ids[0])
                else:
                    raise EncodeError(f"{engine}:{name}.{slot.param}: no bins and no absent")
                continue
            text = format_param_token(value)
            if text in slot.bin_texts:
                out.append(slot.val_ids[slot.bin_texts.index(text)])
            else:
                snapped = _snap_bin(value, slot)
                if snapped is None:
                    if slot.absent_id is not None:
                        counters["param_dropped"] = counters.get("param_dropped", 0) + 1
                        out.append(slot.absent_id)
                    else:
                        raise EncodeError(f"{engine}:{name}.{slot.param}={value!r} not encodable")
                else:
                    counters["param_snapped"] = counters.get("param_snapped", 0) + 1
                    out.append(snapped)
        if op.allow_append:
            if append_flag is None:
                out.append(vocab.append_ids[0])
            else:
                out.append(vocab.append_ids[2 if append_flag else 1])
    out.append(EOS)
    return out


def _snap_bin(value: Any, slot: Slot) -> int | None:
    if not slot.val_ids:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric: list[tuple[int, float]] = []
        for tid, text in zip(slot.val_ids, slot.bin_texts):
            try:
                numeric.append((tid, float(text)))
            except ValueError:
                continue
        if numeric:
            v = float(value)
            return min(
                numeric,
                key=lambda pair: abs(math.log(abs(v) / abs(pair[1])))
                if v != 0 and pair[1] != 0 and (v > 0) == (pair[1] > 0)
                else abs(v - pair[1]) + 1.0,
            )[0]
    return None


def decode_tokens(ids: Sequence[int], vocab: ActionVocab) -> tuple[str, dict[str, Any]]:
    """Token ids -> (engine, actions dict). Inverse of encode_actions."""
    from ..pipeline_text import parse_param_token

    if not ids or ids[0] != BOS:
        raise ValueError("sequence must start with BOS")
    i = 1
    engine = None
    for name, tid in vocab.engine_ids.items():
        if ids[i] == tid:
            engine = name
    if engine is None:
        raise ValueError("missing ENGINE token")
    i += 1
    # TASK token (fused vocab). Consume it if present; older sequences omit it.
    task = "regression"
    if i < len(ids):
        for name, tid in vocab.task_ids.items():
            if ids[i] == tid:
                task = name
                i += 1
                break
    transforms: list[dict[str, Any]] = []
    head: dict[str, Any] | None = None
    while i < len(ids) and ids[i] != EOS:
        op = vocab.ops.get(ids[i])
        if op is None:
            raise ValueError(f"expected OP token at {i}, got {vocab.tokens[ids[i]]}")
        i += 1
        params: dict[str, Any] = {}
        for slot in op.slots:
            if slot.is_expr:
                if i < len(ids) and ids[i] in slot.val_ids:  # macro bin
                    params[slot.param] = slot.bin_texts[slot.val_ids.index(ids[i])]
                    i += 1
                    continue
                expr_ids: list[int] = []
                while i < len(ids) and ids[i] != vocab.expr.end_id:
                    expr_ids.append(ids[i])
                    i += 1
                if i >= len(ids):
                    raise ValueError("unterminated EXPR block")
                i += 1  # consume EXPR_END
                params[slot.param] = render_expr(expr_ids, vocab.expr)
                continue
            tid = ids[i]
            i += 1
            if slot.absent_id is not None and tid == slot.absent_id:
                continue
            if tid not in slot.val_ids:
                raise ValueError(f"bad VAL token for {op.name}.{slot.param}: {vocab.tokens[tid]}")
            params[slot.param] = parse_param_token(slot.bin_texts[slot.val_ids.index(tid)])
        if op.allow_append:
            tid = ids[i]
            i += 1
            if tid == vocab.append_ids[2]:
                params["append"] = True
            elif tid == vocab.append_ids[1]:
                params["append"] = False
            elif tid != vocab.append_ids[0]:
                raise ValueError("expected APPEND token")
        stage = {"op": op.name, "params": params}
        if op.role == "head":
            head = stage
            break
        transforms.append(stage)
    if head is None:
        raise ValueError("sequence has no head stage")
    return engine, {
        "engine": engine,
        "task": task,
        "transform_count": len(transforms),
        "transforms": transforms,
        "head": head,
    }
