from __future__ import annotations

import math
import re
from typing import Any, Sequence

from .tokens import EncodeError, ExprTable

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
