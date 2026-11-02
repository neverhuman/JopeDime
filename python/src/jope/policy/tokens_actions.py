from __future__ import annotations

from typing import Any, Sequence

from ..pipeline_text import format_param_token, parse_param_token
from .tokens import ActionVocab, BOS, EOS, EncodeError, Slot, _snap_bin, encode_expr, render_expr

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
