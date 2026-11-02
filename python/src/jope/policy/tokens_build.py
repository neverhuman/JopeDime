from __future__ import annotations

import copy
import math
from typing import Any, Sequence

from ..grammars.spec import ExprSpec, GrammarSpec
from ..pipeline_text import format_param_token
from .tokens import (
    ABSENT, SPECIALS, ActionVocab, EncodeError, ExprTable, OpInfo, Slot,
)

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
