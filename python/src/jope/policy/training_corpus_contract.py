"""Lossless measured-proposal admission. Pure metadata; never imports a model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..grammars.spec import GrammarSpec
from ..pipeline_text import format_pipeline_actions, parse_pipeline_actions
from .tokens import build_action_vocab, decode_tokens, encode_actions


def require(value, message):
    if not value:
        raise ValueError(message)


def checked(reference):
    path = Path(reference["path"])
    require(
        path.is_absolute() and path.resolve(strict=True) == path,
        "recipe path must be physical",
    )
    require(
        path.is_file() and path.stat().st_size <= 32 << 20,
        "recipe must be a bounded file",
    )
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == reference["sha256"], "recipe bytes changed")
    return json.loads(data)


def load_frozen_action_vocab(data):
    """Load an original parallel.action_vocab.v1 without assigning any new IDs.

    GrammarSpec objects are parsing views only. Training and proposal callers must
    retain the returned ActionVocab and full source table in checkpoint metadata;
    rebuilding a vocabulary from these views changes the historical token IDs.
    """
    from ..grammars.spec import ExprSpec, OpSpec, ParamSpec
    from ..pipeline_text import parse_param_token
    from .tokens import ActionVocab, ExprTable, OpInfo, Slot

    require(data.get("schema") == "parallel.action_vocab.v1", "unknown frozen action table")
    tokens = list(data["tokens"])
    require(
        tokens[:3] == ["PAD", "BOS", "EOS"] and len(set(tokens)) == len(tokens),
        "invalid reserved/duplicate tokens",
    )
    index = {name: i for i, name in enumerate(tokens)}
    require(data.get("index", index) == index, "frozen token index differs")
    engines = list(data["engines"])
    tasks = list(data["tasks"])
    require(
        engines and tasks and len(set(engines)) == len(engines) and len(set(tasks)) == len(tasks),
        "invalid engine/task inventory",
    )

    def token(tid, expected=None):
        require(type(tid) is int and 0 <= tid < len(tokens), "token ID outside frozen table")
        require(
            expected is None or tokens[tid] == expected,
            "token ID/string binding differs",
        )
        return tid

    engine_ids = dict(data["engine_ids"])
    task_ids = dict(data["task_ids"])
    require(
        set(engine_ids) == set(engines) and set(task_ids) == set(tasks),
        "engine/task maps incomplete",
    )
    for engine, tid in engine_ids.items():
        token(tid, f"ENGINE={engine}")
    for task, tid in task_ids.items():
        token(tid, f"TASK={task}")
    expr_data = dict(data["expr"])
    for name in ("func_arity", "symbol", "is_binop"):
        expr_data[name] = {int(k): v for k, v in expr_data[name].items()}
    expr = ExprTable(**expr_data)
    for tid in [
        *expr.var_map_ids,
        *expr.var_col_ids,
        *expr.var_common_ids,
        *expr.const_ids,
        *expr.func_arity,
        expr.end_id,
    ]:
        token(tid)
    token(expr.end_id, "EXPR_END")
    require(
        len(expr.const_ids) == len(expr.const_values),
        "expression constant inventory differs",
    )
    for tid in [*expr.var_map_ids, *expr.var_col_ids, *expr.var_common_ids]:
        require(
            tid in expr.symbol and tokens[tid].startswith("EXPR_VAR="),
            "expression variable differs",
        )
    for tid, value in zip(expr.const_ids, expr.const_values):
        token(tid, f"EXPR_CONST={expr.symbol[tid]}")
        require(float(expr.symbol[tid]) == value, "expression constant value differs")
    for tid, arity in expr.func_arity.items():
        require(
            type(arity) is int and 1 <= arity <= 3 and tid in expr.is_binop,
            "expression arity differs",
        )
        token(tid, f"EXPR_{'BINOP' if expr.is_binop[tid] else 'FUNC'}={expr.symbol[tid]}")
    append_ids = tuple(data["append_ids"])
    require(len(append_ids) == 3, "append table incomplete")
    for tid, suffix in zip(append_ids, ("<ABSENT>", "0", "1")):
        token(tid, f"APPEND={suffix}")
    ops = {}
    for key, raw in data["ops"].items():
        op_data = dict(raw)
        slots = [Slot(**slot) for slot in op_data.pop("slots")]
        op = OpInfo(**op_data, slots=slots)
        require(
            str(op.token_id) == str(key) and op.engine in engines and op.role in ("head", "transform"),
            "operator identity differs",
        )
        token(op.token_id, f"OP={op.engine}:{op.name}")
        require(
            len({slot.param for slot in slots}) == len(slots),
            "duplicate parameter slots",
        )
        for slot in slots:
            require(
                len(slot.val_ids) == len(slot.bin_texts) and len(set(slot.val_ids)) == len(slot.val_ids),
                "value table differs",
            )
            require(slot.is_expr == (slot.kind == "expr"), "expression slot kind differs")
            for tid, text in zip(slot.val_ids, slot.bin_texts):
                token(tid, f"VAL={op.engine}:{op.name}:{slot.param}={text}")
            if slot.absent_id is not None:
                token(slot.absent_id, f"VAL={op.engine}:{op.name}:{slot.param}=<ABSENT>")
        ops[op.token_id] = op
    names = {(op.engine, op.name): tid for tid, op in ops.items()}
    require(len(names) == len(ops), "duplicate engine/operator names")
    transforms = {e: list(data["engine_transforms"][e]) for e in engines}
    heads = {e: list(data["engine_heads"][e]) for e in engines}
    head_tasks = {int(tid): tuple(ts) for tid, ts in data["head_tasks"].items()}
    admitted_ops = []
    for engine in engines:
        for ids, role in ((transforms[engine], "transform"), (heads[engine], "head")):
            for tid in ids:
                require(
                    tid in ops and ops[tid].engine == engine and ops[tid].role == role,
                    "engine operator index differs",
                )
                admitted_ops.append(tid)
        for tid in heads[engine]:
            require(
                tid in head_tasks and head_tasks[tid] and set(head_tasks[tid]) <= set(tasks),
                "head task policy incomplete",
            )
    require(
        len(set(admitted_ops)) == len(admitted_ops) and set(admitted_ops) == set(ops),
        "operator inventory differs",
    )
    require(
        set(head_tasks) == {tid for ids in heads.values() for tid in ids},
        "head task inventory differs",
    )
    version = data["version"]
    require(isinstance(version, str) and version, "frozen version absent")
    vocab = ActionVocab(
        tokens=tokens,
        index=index,
        engines=engines,
        engine_ids=engine_ids,
        ops=ops,
        op_id_by_name=names,
        engine_transforms=transforms,
        engine_heads=heads,
        append_ids=append_ids,
        expr=expr,
        max_transforms=dict(data["max_transforms"]),
        max_expr_stages=dict(data["max_expr_stages"]),
        version=version,
        tasks=tasks,
        task_ids=task_ids,
        head_tasks=head_tasks,
    )
    expr_spec = ExprSpec(
        vars_map=tuple(expr.symbol[i] for i in expr.var_map_ids),
        vars_common=tuple(expr.symbol[i] for i in expr.var_common_ids),
        funcs1=tuple(expr.symbol[i] for i, n in expr.func_arity.items() if n == 1 and not expr.is_binop[i]),
        funcs2=tuple(expr.symbol[i] for i, n in expr.func_arity.items() if n == 2 and not expr.is_binop[i]),
        funcs3=tuple(expr.symbol[i] for i, n in expr.func_arity.items() if n == 3 and not expr.is_binop[i]),
        binops=tuple(expr.symbol[i] for i in expr.func_arity if expr.is_binop[i]),
        consts=tuple(expr.symbol[i] for i in expr.const_ids),
        max_col=len(expr.var_col_ids),
        max_depth=expr.max_depth,
        max_nodes=expr.max_nodes,
    )
    specs = []
    for engine in engines:

        def view(tid):
            op = ops[tid]
            return OpSpec(
                name=op.name,
                role=op.role,
                params=tuple(
                    ParamSpec(
                        name=s.param,
                        kind=s.kind,
                        bins=tuple(text if s.is_expr else parse_param_token(text) for text in s.bin_texts),
                        default=s.default,
                        required=s.required,
                    )
                    for s in op.slots
                ),
                allow_append=op.allow_append,
                generate=op.generate,
                takes_expr=op.takes_expr,
                tasks=head_tasks[tid] if op.role == "head" else tuple(tasks),
            )

        specs.append(
            GrammarSpec(
                engine=engine,
                version=version,
                evaluator_version="frozen-action-table/v1",
                max_transforms=vocab.max_transforms[engine],
                transforms={ops[i].name: view(i) for i in transforms[engine]},
                heads={ops[i].name: view(i) for i in heads[engine]},
                expr=expr_spec,
                fallback_genome="",
                max_expr_stages=vocab.max_expr_stages[engine],
            )
        )
    return specs, vocab


def validate_teaching_recipe(row, qualification, vocabulary):
    """Bind original bytes to measured derived recipe and exact frozen policy tokens.

    The caller independently finds the qualification; the recipe cannot nominate
    its own admission. A nearest numeric bin, dropped parameter, source-grammar
    upgrade or canonical override is an error, not a new training example.
    """
    metadata = row["metadata"]
    reference = metadata["teaching_recipe"]
    require(reference["sha256"] == qualification["record_sha256"], "unmeasured recipe blob")
    frozen = checked(reference)
    candidate = frozen["candidate"]
    recipe = candidate["recipe"]
    original = row["genome"]
    executable = recipe["executable_canonical"]
    require(
        original == candidate["genome"] == recipe["original_proposal"]
        and hashlib.sha256(original.encode()).hexdigest()
        == recipe["original_bytes_sha256"]
        == metadata["original_proposal_sha256"],
        "original proposal bytes differ",
    )
    require(
        hashlib.sha256(executable.encode()).hexdigest()
        == recipe["executable_sha256"]
        == candidate["program_sha256"]
        == qualification["program_sha256"]
        == metadata["executable_program_sha256"],
        "derived executable differs from measured program",
    )
    require(
        row.get("canonical_genome") in (None, original),
        "canonical override is forbidden",
    )
    engine = row.get("engine", "featmath")
    task = metadata["task_type"]
    require(task == qualification["comparison"]["task"], "teaching task differs")
    require(
        row["dataset_id"] == qualification["comparison"]["dataset_hash"],
        "teaching dataset differs",
    )
    if vocabulary.get("schema") == "parallel.action_vocab.v1":
        specs, vocab = load_frozen_action_vocab(vocabulary)
    else:
        specs = []
        for name, data in vocabulary["engines"].items():
            require(name == data["engine"], "frozen engine key differs")
            for op in [*data["heads"].values(), *data["transforms"].values()]:
                require(
                    "tasks" in op and "generate" in op,
                    "frozen vocabulary omits task policy",
                )
            specs.append(GrammarSpec.from_json(data))
        vocab = build_action_vocab(specs)
    by_engine = {spec.engine: spec for spec in specs}
    require(engine in by_engine, "engine absent from frozen vocabulary")
    spec = by_engine[engine]
    actions = parse_pipeline_actions(original, spec)
    stages = actions["transforms"] + [actions["head"]]
    for i, stage in enumerate(stages):
        op = (spec.heads if i == len(stages) - 1 else spec.transforms)[stage["op"]]
        require(op.supports_task(task), "task is absent from frozen operator policy")
        allowed = {p.name for p in op.params} | ({"append"} if op.allow_append else set())
        require(set(stage["params"]) <= allowed, "unencoded parameter in original proposal")
    require(task in vocab.task_ids, "unknown task cannot fall back to regression")
    tokens = encode_actions(engine, actions, vocab, task=task)
    require(
        not any(vocab.counters.values()),
        "vocabulary would snap/drop/default original proposal",
    )
    decoded_engine, decoded = decode_tokens(tokens, vocab)
    require(
        decoded_engine == engine
        and decoded["task"] == task
        and format_pipeline_actions(decoded) == format_pipeline_actions(actions),
        "frozen token roundtrip changes original proposal",
    )
    require(
        encode_actions(engine, decoded, vocab, task=task) == tokens,
        "token roundtrip is unstable",
    )
