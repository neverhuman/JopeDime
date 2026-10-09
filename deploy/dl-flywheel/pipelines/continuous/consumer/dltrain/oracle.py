"""PLAN §5 P4 canonicalizer oracle (also used by P10 and P11).

Tier A (`canonical`, `parse_genome`, `render`) is an in-process Python replica of the pinned
proposal engine's `canonical()` (test_primary_multifidelity proposal-engine main.rs:50-57):

    GenomeIr::parse(src) -> canonical() -> GenomeIr::parse(text).canonical() == text

over the vendored feat-math 0.2.0 sources pinned in FEAT_MATH_PINS (dsl.rs, genome.rs, math_expr.rs).
Python's defaults differ from Rust in several places; the replica follows Rust:

- whitespace is Rust `char::is_whitespace` (the 25 Unicode White_Space code points, RUST_WHITESPACE),
  never `str.isspace` (which also accepts U+001C..U+001F);
- dispatch order `(pipe`, `(portfolio`, `(sym ` (ASCII space), on the trimmed text;
- the dsl.rs tokenizer (quoted atoms, in-atom quote toggling, backslash escapes) and the second
  unescape in `unquote`;
- value typing order: quoted -> Str; ASCII case-insensitive true/false -> Bool; [+-]?[0-9]+ within
  i64 -> Int (overflow falls through); the Rust f64 grammar (re.ASCII) -> Float; else Str.
  `int()`/`float()` are only called after an ASCII regex matched;
- Display: Int decimal; Float NaN/inf/-inf, '%.1f' when |v - round(v)| < 1e-12, else the shortest
  round-trip digits printed positionally; Bool lowercase; Str quoted only when empty or containing
  White_Space, ( ) " or =, with backslash and quote escaped;
- params: last duplicate wins, keys sorted by code point (= UTF-8 byte order = BTreeMap order);
- blend: as_usize semantics (Int >= 0; Float >= 0 truncating and saturating), defaults for missing keys;
- sym: the body is kept VERBATIM and only validated by a MathExpr lex + parse (identifiers lowercased
  for semantics only, c<N>/col<N>/col(N), aliases, exact arity, nodes <= 64, depth <= 16).

Parsing is iterative. The math parser mirrors the Rust recursive-descent frames and raises
`Undecidable` (not a CanonicalError) when the Rust parser would recurse deeper than
RUST_FRAME_LIMIT frames or walk an AST deeper than RUST_AST_DEPTH_LIMIT; such inputs could overflow
the engine's stack, so they are sent to Tier B instead of being decided here.

Tier B (`PinnedEngine`) runs the pinned engine copy RUN/bin/<PINNED_ENGINE_SHA256> with `--seed`:
its sha256 is verified before every batch; stdin is one JSON request with N distinct prior seeds and
count = N; an input is accepted unchanged only if an output with operator == "historical_seed" has
dsl == input (and program_sha256 == sha256(input)); outputs are matched by DSL, never by position;
any other operator is a counted fill-in; a non-zero exit fails the whole batch, which is bisected to
localize the cause and is never read as a per-input rejection.

CLI (inside the lane slice):
  replica-check  [--engine BIN]                     Rust #[test] vectors through Tier A (and Tier B)
  tier-a --corpus PATH [--partitions ...]           Tier A + Tier B corpus mode over P4a dictionaries
  tier-b --feature-count P --train-rows N (--dsl TEXT | --input FILE)   generated mode
  differential [--n N] [--seed S] [--bases FILE]    fuzzed differential Tier A vs Tier B
"""
from __future__ import annotations

import argparse
import json
import math
import operator
import random
import re
import struct
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Union

from dltrain import common

# ---------------------------------------------------------------------------------------------
# Pins and constants
# ---------------------------------------------------------------------------------------------
PINNED_ENGINE_SHA256 = "155417e8b52729918e33bc8ceabe20c26d56e6d01a4a926d7273028bf31b07d6"
RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
DEFAULT_ENGINE = RUN / "bin" / PINNED_ENGINE_SHA256
LOCAL_LAMBDA = Path("/home/ubuntu/JopeDime/.agent/experiments/lambda-training/local_lambda")
FEAT_MATH_SRC = LOCAL_LAMBDA / "all_regression/vendor/feat-math/src"
# The sources this replica ports, by sha256 (read-only; checked by tests, never required at import).
FEAT_MATH_PINS = {
    "dsl.rs": "2e16adf644411aa49860baf07d182887196068bdbf71a2c56512623053cb9bf1",
    "genome.rs": "f6db05df29345f508d7ef8270a81dc64e09fd2951e648906f31da91bf99473b7",
    "math_expr.rs": "47bc9ba5a7e7627d998bca5429f1c593fe364a132e84375bc20840e66baf1e57",
    "heads.rs": "724a885313da9c2e222b27becd0e426be8ef28680faae344abfd2ad71279ee13",
    "advanced_v3/pipeline.rs": "d25e9de2c99f8714e805d855d1c4ab4a453bf38a5cfa4b3c41ba57225b3f6dfc",
    "target_transform.rs": "f0eca01c7d1bbe9c1b51518ccc0eb8096e204dbd6e7ce88806be82441a25c16a",
}
ENGINE_SOURCE_PINS = {
    # canonical(), generate(), admitted_width(), standard_lane_model_safe() and the #[test] vectors
    "full_regression/test_primary_multifidelity/rust/proposal-engine/src/main.rs":
        "1c4d2d3b04cdc500765ead12246afeaaa262b58a5b47aa83e9fc750315e3f64a",
    # evaluator caps: genome() at main.rs:191-215
    "full_regression/test_primary_multifidelity/rust/lambda-worker/src/main.rs":
        "0e2f29a0d83addb17b7b256b6fda58c8f52dd5747f727f84975f250726beef0f",
    # island caps: admit_dsl() at main.rs:958-974
    "full_regression/research_2k/rust/gp-island-engine/src/main.rs":
        "4259c067730f876e35235f369b31c0571110391c39aca7e85248e686aeade225",
}
PROPOSALS_SCHEMA = "jope.gp-proposals/v7"
MAX_SEED_BATCH = 4096          # proposal main.rs:1215 (count 1..=4096)
MAX_FEATURE_COUNT = 2048       # proposal main.rs:1215 (feature_count 1..=2048)
U64_MAX = (1 << 64) - 1        # usize/u64 on the engine's x86_64 target
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
MATH_MAX_NODES = 64            # math_expr.rs:11
MATH_MAX_DEPTH = 16            # math_expr.rs:12
# Tier A decides an input only while the Rust recursion it implies stays far below the engine's
# 8 MiB main-thread stack; beyond either limit the input is Undecidable and goes to Tier B.
RUST_FRAME_LIMIT = 4096        # recursive-descent frames (parse_expr/add/mul/pow/unary/primary/args)
RUST_AST_DEPTH_LIMIT = 4096    # count_nodes/depth/Drop recursion over the AST
TIMEOUT_EXIT = 124
STDERR_HEAD_CHARS = 2048
EVALUATOR_MAX_BYTES = 32768    # lambda-worker main.rs:192
EVALUATOR_MAX_PIPE_STAGES = 10  # lambda-worker main.rs:203
ISLAND_MAX_WINSOR = 3          # gp-island-engine main.rs:959
ISLAND_MAX_LPAREN = {"(pipe ": 11, "(sym ": 18, "(portfolio ": 32}   # gp-island-engine main.rs:962-971

# Rust char::is_whitespace == Unicode White_Space (25 code points). Python's str.isspace() also
# accepts U+001C..U+001F, which Rust does not.
RUST_WHITESPACE = ("\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007"
                   "\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000")
_WS_SET = frozenset(RUST_WHITESPACE)
_WS_CLASS = "".join("\\u%04x" % ord(c) for c in RUST_WHITESPACE)
_FAST_TOKEN_RE = re.compile("[()]|[^()" + _WS_CLASS + "]+")
_NEEDS_QUOTES_RE = re.compile("[" + _WS_CLASS + '()"=]')
_I64_RE = re.compile(r"[+-]?[0-9]+", re.ASCII)
# core::num::dec2flt grammar: Sign? ('inf' | 'infinity' | 'nan' | Number), case-insensitive (ASCII only).
_F64_RE = re.compile(r"[+-]?(?:inf|infinity|nan|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+-]?[0-9]+)?)",
                     re.ASCII | re.IGNORECASE)
_INF = math.inf
_KEY0 = operator.itemgetter(0)


def is_rust_whitespace(ch: str) -> bool:
    return ch in _WS_SET


def rust_trim(s: str) -> str:
    """`str::trim()`: strip leading and trailing Unicode White_Space."""
    return s.strip(RUST_WHITESPACE)


# ---------------------------------------------------------------------------------------------
# Errors and the genome model
# ---------------------------------------------------------------------------------------------
class CanonicalError(ValueError):
    """Tier A rejects the input. kind: parse | blend | sym | math | idempotence."""

    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind


class Undecidable(Exception):
    """Tier A cannot decide this input (the engine's recursion could exceed its stack): ask Tier B.

    Deliberately not a CanonicalError, so `except CanonicalError` never reads it as a rejection."""

    kind = "undecidable"


class EngineIntegrityError(RuntimeError):
    """The engine copy's sha256 differs from the pin; nothing may be run."""


class EngineProtocolError(RuntimeError):
    """The engine exited 0 but its stdout broke the proposals contract (a STOP-level anomaly)."""


ParamValue = tuple  # ("Int", int) | ("Float", float) | ("Bool", bool) | ("Str", str)
StageT = tuple      # (op, ((key, ParamValue), ...)) with keys in code-point order


@dataclass(frozen=True)
class Pipeline:
    stages: tuple  # tuple[tuple[str, tuple[tuple[str, ParamValue], ...]], ...]


@dataclass(frozen=True)
class Symbolic:
    source: str    # the body between the quotes, verbatim
    affine: bool


@dataclass(frozen=True)
class Portfolio:
    members: tuple  # tuple[Pipeline, ...], order-preserving
    repeats_auto: bool


Genome = Union[Pipeline, Symbolic, Portfolio]


# ---------------------------------------------------------------------------------------------
# Rust numeric primitives
# ---------------------------------------------------------------------------------------------
def parse_i64(text: str) -> int | None:
    """`text.parse::<i64>()`: ASCII [+-]?[0-9]+ in range, else None (overflow falls through)."""
    if not _I64_RE.fullmatch(text):
        return None
    neg = text[0] == "-"
    digits = text[1:] if text[0] in "+-" else text
    digits = digits.lstrip("0") or "0"
    if len(digits) > 19:          # also keeps int() under CPython's 4300-digit conversion limit
        return None
    v = -int(digits) if neg else int(digits)
    return v if I64_MIN <= v <= I64_MAX else None


def parse_f64(text: str) -> float | None:
    """`text.parse::<f64>()`: the Rust grammar first (no '_', no whitespace, ASCII digits), then float()."""
    if not _F64_RE.fullmatch(text):
        return None
    return float(text)


def format_f64(v: float) -> str:
    """dsl.rs:86-92: `{:.1}` when (v - v.round()).abs() < 1e-12, else Rust Display (shortest, positional)."""
    if v != v:
        return "NaN"
    if v == _INF:
        return "inf"
    if v == -_INF:
        return "-inf"
    # Half-away (Rust) and half-even (Python) rounding differ only at exact halves, where both
    # differences are 0.5; elsewhere v - round(v) is exact.
    if abs(v - round(v)) < 1e-12:
        return "%.1f" % v
    return _shortest_positional(v)


_TIE_MAX_DENOMINATOR = 1 << 18   # a tie needs an exact expansion of <= 18 significant digits


def _shortest_positional(v: float) -> str:
    """Rust `{}` for a finite, non-integral f64: the shortest round-trip digits, printed positionally.

    CPython's repr finds the same shortest length and, like Rust, the candidate closest to the exact
    value. They differ on an exact tie between the two shortest candidates (the exact binary value has
    one more significant digit, a final 5): repr rounds to even, Rust's flt2dec rounds up in magnitude
    when the upper candidate round-trips (e.g. 1000000000000000.25 prints as 1000000000000000.3)."""
    r = repr(v)
    if v.as_integer_ratio()[1] > _TIE_MAX_DENOMINATOR:
        return format(Decimal(r), "f") if "e" in r else r     # > 18 fractional digits: no tie possible
    exact = Decimal(v)                      # exact binary value
    te = exact.as_tuple()
    if len(te.digits) <= 18 and te.digits[-1] == 5:
        td = Decimal(r).as_tuple()
        if len(te.digits) == len(td.digits) + 1:
            q = te.exponent + 1
            low = Decimal((te.sign, te.digits[:-1], q))            # truncated toward zero
            high = low + Decimal((te.sign, (1,), q))               # one unit away from zero (exact: <= 18 digits)
            return format(high if float(high) == v else low, "f")
    if "e" in r:
        return format(Decimal(r), "f")
    return r


def as_usize(pv: ParamValue) -> int | None:
    """dsl.rs:27-33: Int >= 0, or Float >= 0 cast with `as usize` (truncating, saturating)."""
    tag, v = pv
    if tag == "Int":
        return v if v >= 0 else None
    if tag == "Float":
        if v >= 0.0:              # False for NaN
            if v == _INF:
                return U64_MAX
            iv = int(v)
            return iv if iv <= U64_MAX else U64_MAX
        return None
    return None


def parse_param_value(value: str) -> ParamValue:
    """dsl.rs:355-372."""
    unq = _unquote(value)
    if unq is not None:
        return ("Str", unq)
    low = value.lower() if value.isascii() else None
    if low == "true":
        return ("Bool", True)
    if low == "false":
        return ("Bool", False)
    iv = parse_i64(value)
    if iv is not None:
        return ("Int", iv)
    fv = parse_f64(value)
    if fv is not None:
        return ("Float", fv)
    return ("Str", value)


def _unquote(value: str) -> str | None:
    """dsl.rs:393-409: a "..." value is unescaped a second time ('\\x' -> 'x'; a final lone '\\' is dropped)."""
    if not (len(value) >= 2 and value[0] == '"' and value[-1] == '"'):
        return None
    inner = value[1:-1]
    if "\\" not in inner:
        return inner
    out = []
    i, n = 0, len(inner)
    while i < n:
        c = inner[i]
        i += 1
        if c == "\\":
            if i < n:
                out.append(inner[i])
                i += 1
        else:
            out.append(c)
    return "".join(out)


def format_param_value(pv: ParamValue) -> str:
    """dsl.rs:82-103 `impl Display for ParamValue`."""
    tag, v = pv
    if tag == "Int":
        return str(v)
    if tag == "Float":
        return format_f64(v)
    if tag == "Bool":
        return "true" if v else "false"
    if not v or _NEEDS_QUOTES_RE.search(v):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return v


# ---------------------------------------------------------------------------------------------
# dsl.rs tokenizer and parsers
# ---------------------------------------------------------------------------------------------
TOK_L, TOK_R, TOK_ATOM = "(", ")", "atom"


def dsl_tokenize(src: str) -> list:
    """dsl.rs:212-272 `tokenize`. Returns (kind, text, start, end) with kind in '(' | ')' | 'atom';
    text is the processed atom (escapes applied), [start, end) the raw char span in `src`."""
    toks = []
    if '"' not in src:
        # Without a quote no escape or toggle can occur: atoms are maximal runs of non-White_Space,
        # non-paren chars (a backslash outside quotes is literal).
        for m in _FAST_TOKEN_RE.finditer(src):
            t = m.group()
            toks.append((t if (t == "(" or t == ")") else TOK_ATOM, t, m.start(), m.end()))
    else:
        i, n = 0, len(src)
        while i < n:
            ch = src[i]
            if ch == "(":
                toks.append((TOK_L, "(", i, i + 1))
                i += 1
            elif ch == ")":
                toks.append((TOK_R, ")", i, i + 1))
                i += 1
            elif ch in _WS_SET:
                i += 1
            elif ch == '"':
                start = i
                i += 1
                buf = []
                closed = False
                while i < n:
                    c = src[i]
                    i += 1
                    if c == "\\":
                        if i >= n:
                            raise CanonicalError("parse", "unterminated escape in quoted atom")
                        buf.append(src[i])
                        i += 1
                    elif c == '"':
                        closed = True
                        break
                    else:
                        buf.append(c)
                if not closed:
                    raise CanonicalError("parse", "unterminated quoted atom")
                toks.append((TOK_ATOM, "".join(buf), start, i))
            else:
                start = i
                buf = [ch]
                i += 1
                in_quote = False
                while i < n:
                    c = src[i]
                    if not in_quote and (c in _WS_SET or c == "(" or c == ")"):
                        break
                    if c == '"':
                        in_quote = not in_quote
                    if c == "\\" and in_quote:
                        i += 1
                        if i >= n:
                            raise CanonicalError("parse", "unterminated escape in quoted atom")
                        buf.append(src[i])
                        i += 1
                        continue
                    buf.append(c)
                    i += 1
                if in_quote:
                    raise CanonicalError("parse", "unterminated quoted parameter")
                toks.append((TOK_ATOM, "".join(buf), start, i))
    if not toks:
        raise CanonicalError("parse", "empty DSL expression")
    return toks


class _DslParser:
    """dsl.rs:274-353 `Parser` over dsl_tokenize output."""

    __slots__ = ("toks", "pos", "n")

    def __init__(self, toks):
        self.toks = toks
        self.pos = 0
        self.n = len(toks)

    def peek_r(self) -> bool:
        return self.pos < self.n and self.toks[self.pos][0] == TOK_R

    def expect_l(self) -> None:
        if self.pos < self.n and self.toks[self.pos][0] == TOK_L:
            self.pos += 1
            return
        raise CanonicalError("parse", f"expected '(' at token {self.pos}")

    def expect_r(self) -> None:
        if self.pos < self.n and self.toks[self.pos][0] == TOK_R:
            self.pos += 1
            return
        raise CanonicalError("parse", f"expected ')' at token {self.pos}")

    def next_atom(self) -> str | None:
        if self.pos < self.n and self.toks[self.pos][0] == TOK_ATOM:
            text = self.toks[self.pos][1]
            self.pos += 1
            return text
        return None

    def expect_atom(self, want: str) -> None:
        got = self.next_atom()
        if got is None:
            raise CanonicalError("parse", "expected atom")
        if got != want:
            raise CanonicalError("parse", f"expected atom '{want}'")

    def params(self, what: str) -> dict:
        params = {}
        while not self.peek_r():
            atom = self.next_atom()
            if atom is None:
                raise CanonicalError("parse", f"{what} parameter must be an atom")
            key, eq, value = atom.partition("=")
            if not eq:
                raise CanonicalError("parse", "parameter must be key=value")
            params[key] = parse_param_value(value)     # BTreeMap::insert: the last duplicate wins
        return params

    def parse_pipe(self) -> tuple:
        self.expect_l()
        self.expect_atom("pipe")
        self.expect_atom("X")
        stages = []
        while not self.peek_r():
            stages.append(self.parse_stage())
        self.expect_r()
        if not stages:
            raise CanonicalError("parse", "pipe must contain at least one stage")
        return tuple(stages)

    def parse_stage(self) -> StageT:
        self.expect_l()
        op = self.next_atom()
        if op is None:
            raise CanonicalError("parse", "stage missing operator")
        params = self.params("stage")
        self.expect_r()
        return (op, tuple(sorted(params.items(), key=_KEY0)))


def parse_dsl(src: str) -> Pipeline:
    """dsl.rs:132-143 `parse_dsl`."""
    p = _DslParser(dsl_tokenize(src))
    stages = p.parse_pipe()
    if p.pos != p.n:
        raise CanonicalError("parse", "trailing tokens after DSL expression")
    return Pipeline(stages)


def parse_portfolio(src: str) -> tuple:
    """dsl.rs:150-210 `parse_portfolio`: (members, blend params dict)."""
    p = _DslParser(dsl_tokenize(src))
    p.expect_l()
    p.expect_atom("portfolio")
    members = []
    blend = None
    while not p.peek_r():
        p.expect_l()
        tag = p.next_atom()
        if tag is None:
            raise CanonicalError("parse", "portfolio: expected member or blend")
        if tag == "member":
            if blend is not None:
                raise CanonicalError("parse", "portfolio: members must precede the blend stage")
            stages = p.parse_pipe()
            p.expect_r()
            members.append(Pipeline(stages))
        elif tag == "blend":
            if blend is not None:
                raise CanonicalError("parse", "portfolio: duplicate blend stage")
            blend = p.params("blend")
            p.expect_r()
        else:
            raise CanonicalError("parse", "portfolio: unexpected element (expected member or blend)")
    p.expect_r()
    if p.pos != p.n:
        raise CanonicalError("parse", "trailing tokens after portfolio expression")
    if not members or len(members) > 12:
        raise CanonicalError("parse", f"portfolio: member count must be 1..12, got {len(members)}")
    for m in members:
        if any(op == "portfolio" for op, _ in m.stages):
            raise CanonicalError("parse", "portfolio: members must not nest portfolios")
    return tuple(members), (blend if blend is not None else {})


def parse_blend(params: dict) -> tuple:
    """genome.rs:38-69 `parse_blend`: (folds, repeats_auto)."""
    for key in params:
        if key not in ("kind", "folds", "repeats", "guard"):
            raise CanonicalError("blend", "unknown parameter")
    kind = params.get("kind")
    if kind is not None and kind != ("Str", "nnls"):
        raise CanonicalError("blend", "kind must be nnls")
    folds = params.get("folds")
    if folds is not None and as_usize(folds) != 5:
        raise CanonicalError("blend", "folds must be 5")
    rep = params.get("repeats")
    if rep is None or rep == ("Str", "auto"):
        repeats_auto = True
    elif as_usize(rep) == 1:
        repeats_auto = False
    else:
        raise CanonicalError("blend", "repeats must be auto or 1")
    guard = params.get("guard")
    if guard is not None and guard != ("Str", "clip"):
        raise CanonicalError("blend", "guard must be clip")
    return 5, repeats_auto


# ---------------------------------------------------------------------------------------------
# math_expr.rs lexer and parser (iterative)
# ---------------------------------------------------------------------------------------------
MATH_FUNCS = {   # math_expr.rs:611-663 (name -> arity)
    "abs": 1, "sqrt": 1, "log1p": 1, "sin": 1, "cos": 1, "tanh": 1, "sigmoid": 1, "exp": 1, "log": 1,
    "asinh": 1, "atan": 1, "erf": 1, "sign": 1, "floor": 1, "round": 1, "min": 2, "max": 2, "step": 2,
    "clip": 3,
}
MATH_VARS = {    # math_expr.rs:591-600 (alias -> variable)
    "x": "x", "row_mean": "row_mean", "mean": "row_mean", "row_sum": "row_sum", "row_l1": "row_l1",
    "row_l2": "row_l2", "row_norm": "row_l2", "row_min": "row_min", "row_max": "row_max", "row_std": "row_std",
}
_MATH_PUNCT = frozenset("+-*/^(),")


def _ascii_digit(c: str) -> bool:
    return "0" <= c <= "9"


def _ascii_alpha(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z")


def math_lex(src: str) -> list:
    """math_expr.rs:339-424 `lex`. Tokens (kind, value, start, end); kinds: num id + - * / ^ ( ) , end."""
    toks = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch in _WS_SET:
            i += 1
        elif ch in _MATH_PUNCT:
            toks.append((ch, None, i, i + 1))
            i += 1
        elif _ascii_digit(ch) or ch == ".":
            start = i
            i += 1
            while i < n:
                c = src[i]
                if _ascii_digit(c) or c == "." or c == "e" or c == "E" or (
                        (c == "+" or c == "-") and src[i - 1] in "eE"):
                    i += 1
                else:
                    break
            raw = src[start:i]
            value = parse_f64(raw)
            if value is None:
                raise CanonicalError("math", f"invalid numeric literal {raw!r}")
            if not math.isfinite(value):
                raise CanonicalError("math", f"non-finite numeric literal {raw!r}")
            toks.append(("num", value, start, i))
        elif _ascii_alpha(ch) or ch == "_":
            start = i
            i += 1
            while i < n:
                c = src[i]
                if _ascii_alpha(c) or _ascii_digit(c) or c == "_":
                    i += 1
                else:
                    break
            toks.append(("id", src[start:i].lower(), start, i))
        else:
            raise CanonicalError("math", f"unexpected character {ch!r} in math expression")
    toks.append(("end", None, n, n))
    return toks


def _parse_usize(digits: str) -> int:
    if not digits:
        raise CanonicalError("math", "cannot parse integer from empty string")
    d = digits.lstrip("0") or "0"
    if len(d) > 20 or int(d) > U64_MAX:
        raise CanonicalError("math", "number too large to fit in target type")
    return int(d)


def _parse_var(name: str):
    v = MATH_VARS.get(name)
    if v is not None:
        return v
    if name.startswith("c") and all(_ascii_digit(c) for c in name[1:]):
        return ("col", _parse_usize(name[1:]))
    if name.startswith("col") and all(_ascii_digit(c) for c in name[3:]):
        return ("col", _parse_usize(name[3:]))
    raise CanonicalError("math", f"unknown math variable {name!r}")


def _parse_col_call(args: tuple):
    if len(args) != 1:
        raise CanonicalError("math", "col() expects exactly one argument")
    a = args[0]
    if a[0] == "c" and a[1] >= 0.0 and abs(a[1] - round(a[1])) < 1e-12:
        v = a[1]
        return ("v", ("col", U64_MAX if v >= 18446744073709551615.0 else int(v)))
    raise CanonicalError("math", "col() expects a non-negative integer literal")


_F_EXPR, _F_ADD, _F_MUL, _F_POW, _F_UNARY, _F_PRIMARY, _F_ARGS = range(7)


@dataclass(frozen=True)
class MathInfo:
    ast: tuple       # ('c', v) | ('v', var) | ('n', a) | ('b', op, a, b) | ('f', name, args)
    nodes: int
    depth: int
    frames: int      # deepest Rust recursive-descent frame stack the parse implies


def parse_math(src: str) -> MathInfo:
    """math_expr.rs:196-213 `MathExpr::parse`: lex, parse, expect_end, then nodes <= 64 and depth <= 16."""
    ast, frames = _parse_math_ast(src)
    nodes, depth = _ast_stats(ast)
    if nodes > MATH_MAX_NODES:
        raise CanonicalError("math", f"math expression has {nodes} nodes, max is {MATH_MAX_NODES}")
    if depth > MATH_MAX_DEPTH:
        raise CanonicalError("math", f"math expression depth {depth} exceeds max {MATH_MAX_DEPTH}")
    return MathInfo(ast, nodes, depth, frames)


def math_shape(src: str) -> tuple:
    """(nodes, depth) of a body that lexes and parses, before the 64/16 limits are applied."""
    ast, _frames = _parse_math_ast(src)
    return _ast_stats(ast)


def _parse_math_ast(src: str) -> tuple:
    """Iterative recursive descent: the explicit stack mirrors the Rust frames (parse_expr, parse_add,
    parse_mul, parse_pow, parse_unary, parse_primary, parse_args) in the same order, so errors, the
    AST and the frame depth are those of the Rust parser. Returns (ast, deepest frame stack)."""
    toks = math_lex(src)
    pos = 0
    stack = [[_F_EXPR, 0, None, None]]
    frames = 1
    ret = None
    while stack:
        depth_now = len(stack)
        if depth_now > frames:
            frames = depth_now
            if frames > RUST_FRAME_LIMIT:
                raise Undecidable(f"math parse recursion exceeds {RUST_FRAME_LIMIT} frames")
        fr = stack[-1]
        kind = fr[0]
        st = fr[1]
        if kind == _F_EXPR:
            if st == 0:
                fr[1] = 1
                stack.append([_F_ADD, 0, None, None])
            else:
                stack.pop()
        elif kind == _F_ADD or kind == _F_MUL:
            child = _F_MUL if kind == _F_ADD else _F_POW
            if st == 0:
                fr[1] = 1
                stack.append([child, 0, None, None])
                continue
            fr[2] = ret if st == 1 else ("b", fr[3], fr[2], ret)
            t = toks[pos][0]
            if (kind == _F_ADD and (t == "+" or t == "-")) or (kind == _F_MUL and (t == "*" or t == "/")):
                pos += 1
                fr[3] = t
                fr[1] = 2
                stack.append([child, 0, None, None])
            else:
                ret = fr[2]
                stack.pop()
        elif kind == _F_POW:
            if st == 0:
                fr[1] = 1
                stack.append([_F_UNARY, 0, None, None])
            elif st == 1:
                fr[2] = ret
                if toks[pos][0] == "^":
                    pos += 1
                    fr[1] = 2
                    stack.append([_F_POW, 0, None, None])
                else:
                    stack.pop()
            else:
                ret = ("b", "^", fr[2], ret)
                stack.pop()
        elif kind == _F_UNARY:
            if st == 0:
                t = toks[pos][0]
                if t == "+":
                    pos += 1
                    fr[1] = 1
                    stack.append([_F_UNARY, 0, None, None])
                elif t == "-":
                    pos += 1
                    fr[1] = 2
                    stack.append([_F_UNARY, 0, None, None])
                else:
                    fr[1] = 1
                    stack.append([_F_PRIMARY, 0, None, None])
            elif st == 1:
                stack.pop()
            else:
                ret = ("n", ret)
                stack.pop()
        elif kind == _F_PRIMARY:
            if st == 0:
                t, v = toks[pos][0], toks[pos][1]
                if t == "num":
                    pos += 1
                    ret = ("c", v)
                    stack.pop()
                elif t == "id":
                    pos += 1
                    if toks[pos][0] == "(":
                        pos += 1
                        fr[1] = 1
                        fr[2] = v
                        stack.append([_F_ARGS, 0, None, None])
                    else:
                        ret = ("v", _parse_var(v))
                        stack.pop()
                elif t == "(":
                    pos += 1
                    fr[1] = 2
                    stack.append([_F_EXPR, 0, None, None])
                else:
                    raise CanonicalError("math", f"expected expression, got {t}")
            elif st == 1:
                name, args = fr[2], ret
                if name == "col":
                    ret = _parse_col_call(args)
                else:
                    arity = MATH_FUNCS.get(name)
                    if arity is None:
                        raise CanonicalError("math", f"unsupported math function {name!r}")
                    if len(args) != arity:
                        raise CanonicalError("math", f"{name} received {len(args)} arguments")
                    ret = ("f", name, args)
                stack.pop()
            else:
                if toks[pos][0] != ")":
                    raise CanonicalError("math", "expected ')'")
                pos += 1
                stack.pop()
        else:  # _F_ARGS
            if st == 0:
                if toks[pos][0] == ")":
                    pos += 1
                    ret = ()
                    stack.pop()
                else:
                    fr[1] = 1
                    fr[2] = []
                    stack.append([_F_EXPR, 0, None, None])
            else:
                fr[2].append(ret)
                t = toks[pos][0]
                if t == ",":
                    pos += 1
                    stack.append([_F_EXPR, 0, None, None])
                elif t == ")":
                    pos += 1
                    ret = tuple(fr[2])
                    stack.pop()
                else:
                    raise CanonicalError("math", f"expected ',' or ')', got {t}")
    if toks[pos][0] != "end":
        raise CanonicalError("math", "trailing tokens in math expression")
    return ret, frames


def _ast_stats(root) -> tuple:
    """math_expr.rs:955-971 count_nodes and depth, iteratively (Rust recurses over the AST)."""
    nodes = 0
    depth = 0
    stack = [(root, 1)]
    while stack:
        node, d = stack.pop()
        nodes += 1
        if d > depth:
            depth = d
            if depth > RUST_AST_DEPTH_LIMIT:
                raise Undecidable(f"math AST deeper than {RUST_AST_DEPTH_LIMIT}")
        tag = node[0]
        if tag == "n":
            stack.append((node[1], d + 1))
        elif tag == "b":
            stack.append((node[3], d + 1))
            stack.append((node[2], d + 1))
        elif tag == "f":
            stack.extend((a, d + 1) for a in node[2])
    return nodes, depth


# ---------------------------------------------------------------------------------------------
# GenomeIr::parse, canonical() and the printer
# ---------------------------------------------------------------------------------------------
def _parse_sym(trimmed: str) -> Symbolic:
    """genome.rs:89-114."""
    rest = trimmed[5:]                     # strip_prefix("(sym ")
    if not rest.endswith(")"):
        raise CanonicalError("sym", "closed-form program requires a closing delimiter")
    rest = rust_trim(rest[:-1])
    cut = rest.find(" ")                    # split_once(' '): ASCII space only
    if cut < 0:
        raise CanonicalError("sym", 'closed-form program must be (sym <affine|raw> "expr")')
    mode, expr_src = rest[:cut], rust_trim(rest[cut + 1:])
    if mode != "affine" and mode != "raw":
        raise CanonicalError("sym", "symbolic mode must be affine or raw")
    if not expr_src.startswith('"'):
        raise CanonicalError("sym", "symbolic expression must have exactly one pair of quotes")
    body = expr_src[1:]
    if not body.endswith('"'):
        raise CanonicalError("sym", "symbolic expression must have exactly one pair of quotes")
    body = body[:-1]
    if '"' in body:
        raise CanonicalError("sym", "unexpected quote in symbolic expression")
    parse_math(body)
    return Symbolic(body, mode == "affine")


def parse_genome(src: str) -> Genome:
    """Tier A port of genome.rs:75-116 `GenomeIr::parse`."""
    if not isinstance(src, str):
        raise TypeError("program text must be str")
    try:
        src.encode("utf-8")
    except UnicodeEncodeError:
        raise CanonicalError("parse", "program text is not a sequence of Unicode scalar values") from None
    trimmed = rust_trim(src)
    if trimmed.startswith("(pipe"):
        return parse_dsl(trimmed)
    if trimmed.startswith("(portfolio"):
        members, blend = parse_portfolio(trimmed)
        _folds, repeats_auto = parse_blend(blend)
        return Portfolio(members, repeats_auto)
    if trimmed.startswith("(sym "):
        return _parse_sym(trimmed)
    raise CanonicalError("parse", "unrecognized program form")


def _render_stage(stage: StageT) -> str:
    op, params = stage
    return "(" + op + "".join(" " + k + "=" + format_param_value(v) for k, v in params) + ")"


def _render_pipe(pipe: Pipeline) -> str:
    return "(pipe X" + "".join(" " + _render_stage(s) for s in pipe.stages) + ")"


def render(g: Genome) -> str:
    """genome.rs:118-143 `canonical()` (DslExpr/Stage/ParamValue Display for pipelines)."""
    if isinstance(g, Pipeline):
        return _render_pipe(g)
    if isinstance(g, Symbolic):
        return '(sym ' + ("affine" if g.affine else "raw") + ' "' + g.source + '")'
    if isinstance(g, Portfolio):
        return ("(portfolio" + "".join(" (member " + _render_pipe(m) + ")" for m in g.members)
                + " (blend folds=5 guard=clip kind=nnls repeats=" + ("auto" if g.repeats_auto else "1") + "))")
    raise TypeError(f"not a genome: {type(g).__name__}")


def canonical(src: str) -> str:
    """Proposal main.rs:50-57: parse -> canonical text -> re-parse; the re-render must equal the text.

    Raises CanonicalError when the engine would refuse the input, Undecidable when Tier A cannot tell."""
    text = render(parse_genome(src))
    try:
        again = render(parse_genome(text))
    except CanonicalError as exc:
        raise CanonicalError("idempotence", f"canonical text does not re-parse ({exc})") from None
    if again != text:
        raise CanonicalError("idempotence", "canonicalization not idempotent")
    return text


def is_canonical(dsl: str) -> bool:
    """True iff canonical(dsl) == dsl. Undecidable propagates (send the input to Tier B)."""
    try:
        return canonical(dsl) == dsl
    except CanonicalError:
        return False


def program_sha256(dsl: str) -> str:
    return common.sha256_bytes(dsl.encode("utf-8"))


def genome_form(g: Genome) -> str:
    return "pipe" if isinstance(g, Pipeline) else "sym" if isinstance(g, Symbolic) else "portfolio"


# ---------------------------------------------------------------------------------------------
# Caps, width/lane ports and static semantics
# ---------------------------------------------------------------------------------------------
def form_caps(dsl: str) -> list:
    """Violated evaluator caps (lambda-worker main.rs:191-215) and island caps (gp-island-engine
    main.rs:958-974); [] when the program passes both."""
    out = []
    nbytes = len(dsl.encode("utf-8"))
    try:
        g = parse_genome(dsl)
    except (CanonicalError, Undecidable):
        g = None
    if nbytes > EVALUATOR_MAX_BYTES:
        out.append("evaluator:bytes")
    if g is None:
        out.append("evaluator:parse")
    elif isinstance(g, Pipeline) and len(g.stages) > EVALUATOR_MAX_PIPE_STAGES:
        out.append("evaluator:pipe_stages")
    elif isinstance(g, Portfolio):
        if not 2 <= len(g.members) <= 3:
            out.append("evaluator:portfolio_members")
        if sum(len(m.stages) for m in g.members) > 24:
            out.append("evaluator:portfolio_stages")
    if nbytes > EVALUATOR_MAX_BYTES:
        out.append("island:bytes")
    if dsl.count("(winsor_clip") > ISLAND_MAX_WINSOR:
        out.append("island:winsor_clip")
    for prefix, cap in ISLAND_MAX_LPAREN.items():
        if dsl.startswith(prefix):
            if dsl.count("(") > cap:
                out.append("island:lparen")
            break
    else:
        out.append("island:form")
    return out


def _sat_mul(a: int, b: int) -> int:
    r = a * b
    return r if r <= U64_MAX else U64_MAX


def _sat_add(a: int, b: int) -> int:
    r = a + b
    return r if r <= U64_MAX else U64_MAX


def _sat_sub(a: int, b: int) -> int:
    return a - b if a > b else 0


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _get_usize(params: dict, key: str, default: int) -> int:
    pv = params.get(key)
    if pv is None:
        return default
    v = as_usize(pv)
    return default if v is None else v


def _append(params: dict) -> bool:
    """proposal main.rs:925-928."""
    v = params.get("append")
    return v == ("Bool", True) or (v is not None and v[0] == "Int" and v[1] == 1)


def _generated_width(op: str, params: dict, width: int) -> int:
    """proposal main.rs:930-955."""
    if op == "piecewise_linear":
        return _sat_mul(width, _get_usize(params, "bins", 8))
    if op == "corr_screen" or op == "select_corr":
        return min(width, _get_usize(params, "k", 32))
    if op == "pls_project":
        return min(_get_usize(params, "k", 8), width)
    if op == "pair_prod":
        topk = min(_clamp(_get_usize(params, "topk", 8), 2, 24), width)
        return _sat_mul(topk, _sat_sub(topk, 1)) // 2
    if op == "cheb":
        return _sat_mul(width, _clamp(_get_usize(params, "degree", 4), 1, 8))
    if op == "bin_onehot":
        return _sat_mul(width, _clamp(_get_usize(params, "bins", 6), 2, 32))
    if op == "hinge":
        return _sat_mul(width, _clamp(_get_usize(params, "knots", 4), 1, 16))
    if op == "poly_interact":
        return min(_sat_add(width, _sat_mul(width, _sat_sub(width, 1)) // 2), 512)
    if op in ("rff_cos", "rff_sin", "oblique_hinge"):
        return _clamp(_get_usize(params, "n", 32), 1, 256)
    if op == "rand_proj":
        return _clamp(_get_usize(params, "k", 16), 1, 256)
    if op == "landmark_dist":
        return _clamp(_get_usize(params, "k", 16), 1, 128)
    if op == "row_stats":
        return 10
    if op == "row_spectral":
        return 5
    if op == "row_diff_stats":
        return 8
    if op == "derive":
        return 1
    if op == "fft_mag":
        return _clamp(_get_usize(params, "k", 8), 1, max(width // 2, 1))
    return width          # includes `target`: pass-through, never stripped


def _check_width_args(p: int, train_rows: int = 0) -> None:
    if type(p) is not int or not 0 <= p <= U64_MAX or type(train_rows) is not int or not 0 <= train_rows <= U64_MAX:
        raise ValueError("p and train_rows must be usize integers")


def _width_walk(stages: tuple, p: int, train_rows: int | None):
    """The stage loop shared by main.rs:957-981 and 983-1026; returns (ok, estimated_state_bytes)."""
    width = p
    state = 0
    for op, params in stages[:max(len(stages) - 1, 0)]:
        pd = dict(params)
        if train_rows is not None and (op == "rank" or op == "blom"):
            state = _sat_add(state, _sat_mul(_sat_mul(train_rows, width), 8))
        generated = _generated_width(op, pd, width)
        if generated == 0 or generated > 512:
            return False, state
        merged = _sat_add(width, generated) if _append(pd) else generated
        if merged > 512:
            return False, state
        width = merged
    return True, state


def _pipeline_or_none(dsl: str) -> Pipeline | None:
    try:
        g = parse_genome(dsl)
    except (CanonicalError, Undecidable):
        return None
    return g if isinstance(g, Pipeline) else None


def admitted_width(dsl: str, p: int) -> bool:
    """Exact port of proposal main.rs:957-981 (a non-pipeline or unparseable source is admitted)."""
    _check_width_args(p)
    pipe = _pipeline_or_none(dsl)
    return True if pipe is None else _width_walk(pipe.stages, p, None)[0]


def _lane_safe(stages: tuple, p: int, train_rows: int) -> bool:
    ok, state = _width_walk(stages, p, train_rows)
    if not ok:
        return False
    if not stages:
        return False
    head_op, head_params = stages[-1]
    hp = dict(head_params)
    if train_rows > 5_000:
        if head_op in ("histgbm2", "tab_boost", "gpr", "ksmooth"):
            return False
        if head_op == "krr" and (train_rows > 15_000 or _get_usize(hp, "m", 800) > 512):
            return False
        if head_op == "histgbm" and (_get_usize(hp, "depth", 4) > 4 or _get_usize(hp, "rounds", 160) > 160):
            return False
        if head_op == "extra_trees" and (_get_usize(hp, "trees", 128) > 128 or _get_usize(hp, "depth", 16) > 16):
            return False
    return state <= 24 * 1024 * 1024


def standard_lane_model_safe(dsl: str, p: int, train_rows: int) -> bool:
    """Exact port of proposal main.rs:983-1026 (a non-pipeline or unparseable source is safe)."""
    _check_width_args(p, train_rows)
    pipe = _pipeline_or_none(dsl)
    return True if pipe is None else _lane_safe(pipe.stages, p, train_rows)


def width_preflight(dsl: str, p: int, train_rows: int) -> str:
    """admitted | refused:width | refused:lane | n/a (sym, portfolio or unparseable). Informational
    for corpus programs; the engine applies these checks only to its own fill-ins."""
    _check_width_args(p, train_rows)
    pipe = _pipeline_or_none(dsl)
    if pipe is None:
        return "n/a"
    if not _width_walk(pipe.stages, p, None)[0]:
        return "refused:width"
    if not _lane_safe(pipe.stages, p, train_rows):
        return "refused:lane"
    return "admitted"


# heads.rs:36-55 is_head_op plus advanced_v3/pipeline.rs:453-467 is_v3_head.
V3_HEADS = frozenset({
    "ridge", "logistic", "ridge_gcv", "elasticnet", "histgbm", "gbm", "ensemble", "ensemble_ridge_gbm",
    "histgbm2", "krr", "gpr", "ksmooth", "hurdle2", "stack", "portfolio",
    "logit_ridge", "multiclass_ridge", "multiclass_ovr_logistic", "multiclass_diagonal_lda",
    "multiclass_nearest_centroid", "tab_boost", "extra_trees", "gam_boost", "pls",
})


def _pipe_semantics(stages: tuple, prefix: str) -> list:
    out = []
    for op, params in stages:      # advanced_v3/pipeline.rs:514-520
        for key, (tag, v) in params:
            if tag == "Float" and not math.isfinite(v):
                out.append(f"{prefix}nonfinite_param:{op}.{key}")
    cursor = 0                     # advanced_v3/pipeline.rs:469-497 (target_transform.rs:37)
    if cursor < len(stages) and stages[cursor][0] == "target":
        cursor += 1
    if cursor < len(stages) and stages[cursor][0] == "input":
        cursor += 1
    if any(op == "input" or op == "target" for op, _ in stages[cursor:]):
        out.append(f"{prefix}target_input_not_leading")
    end = len(stages) - 1 if stages and stages[-1][0] in V3_HEADS else len(stages)
    for op, _ in stages[cursor:end]:  # advanced_v3/pipeline.rs:539-545
        if op in V3_HEADS:
            out.append(f"{prefix}head_not_last:{op}")
    return out


def static_semantics(dsl: str) -> list:
    """Generation-only static rules, never an acceptance gate: finite params, target/input lead in
    target-then-input order, model heads only as the final stage. Applied to pipelines and to every
    portfolio member; sym programs have none."""
    g = parse_genome(dsl)
    if isinstance(g, Pipeline):
        return _pipe_semantics(g.stages, "")
    if isinstance(g, Portfolio):
        out = []
        for i, m in enumerate(g.members):
            out.extend(_pipe_semantics(m.stages, f"member{i}:"))
        return out
    return []


# ---------------------------------------------------------------------------------------------
# Tier B: the pinned engine
# ---------------------------------------------------------------------------------------------
_PROPOSAL_KEYS = frozenset({"draw", "dsl", "grammar_family", "operator", "parents", "program_sha256"})


@dataclass(frozen=True)
class SeedBatch:
    accepted: dict       # input -> dsl, the input came back byte-identical as historical_seed
    changed: dict        # input -> canonical dsl, attributable by DSL matching only
    missing: list        # inputs with no attributable historical_seed output (rejected or deduplicated)
    fill_ins: int        # outputs with any other operator
    exit_code: int       # exit code of the first invocation (non-zero means it was bisected)
    stderr_head: str
    failed: list = field(default_factory=list)        # inputs whose singleton batch exited non-zero
    unattributed: list = field(default_factory=list)  # historical_seed outputs not matched to an input
    batches: int = 1                                  # engine invocations, bisection included


def _merge_batches(a: SeedBatch, b: SeedBatch, exit_code: int, stderr_head: str) -> SeedBatch:
    return SeedBatch(accepted={**a.accepted, **b.accepted}, changed={**a.changed, **b.changed},
                     missing=a.missing + b.missing, fill_ins=a.fill_ins + b.fill_ins, exit_code=exit_code,
                     stderr_head=stderr_head, failed=a.failed + b.failed,
                     unattributed=a.unattributed + b.unattributed, batches=1 + a.batches + b.batches)


class PinnedEngine:
    """Tier B: the pinned proposal engine copy, run as `<path> --seed` with one JSON request on stdin."""

    def __init__(self, path: Path | str = DEFAULT_ENGINE, sha256: str = PINNED_ENGINE_SHA256,
                 timeout_s: float = 300.0) -> None:
        self.path = Path(path)
        self.sha256 = sha256
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self.invocations = 0
        self.verifications = 0

    def verify(self) -> None:
        """sha256 of the copy must equal the pin; called before every batch, bisection included."""
        digest = common.sha256_file(self.path)
        with self._lock:
            self.verifications += 1
        if digest != self.sha256:
            raise EngineIntegrityError(f"{self.path}: sha256 {digest} != pinned {self.sha256}")

    def _invoke(self, payload: bytes) -> tuple:
        """Run the engine once: (exit code, stdout bytes, stderr bytes). Tests override this."""
        try:
            proc = subprocess.run([str(self.path), "--seed"], input=payload, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=self.timeout_s, env={}, close_fds=True,
                                  check=False)
        except subprocess.TimeoutExpired as exc:
            return TIMEOUT_EXIT, b"", (exc.stderr or b"") + b"\n[oracle] engine timeout"
        return proc.returncode, proc.stdout, proc.stderr

    def seed_batch(self, dsls: Sequence[str], *, feature_count: int = 8, train_rows: int = 0,
                   seed: int = 0) -> SeedBatch:
        """Tier B over N distinct inputs (1 <= N <= 4096, count = N). Bisects on a non-zero exit."""
        inputs = list(dsls)
        if not 1 <= len(inputs) <= MAX_SEED_BATCH:
            raise ValueError(f"a seed batch holds 1..{MAX_SEED_BATCH} programs, got {len(inputs)}")
        if len(set(inputs)) != len(inputs):
            raise ValueError("prior_seeds must be distinct")
        for x in inputs:
            if not isinstance(x, str):
                raise TypeError("prior seeds must be str")
            x.encode("utf-8")     # a lone surrogate cannot be sent (UnicodeEncodeError is a ValueError)
        if type(feature_count) is not int or not 1 <= feature_count <= MAX_FEATURE_COUNT:
            raise ValueError(f"feature_count must be 1..{MAX_FEATURE_COUNT}")
        if type(train_rows) is not int or not 0 <= train_rows <= U64_MAX:
            raise ValueError("train_rows must be a usize")
        if type(seed) is not int or not 0 <= seed <= U64_MAX:
            raise ValueError("seed must be a u64")
        return self._batch(inputs, feature_count, train_rows, seed)

    def _batch(self, inputs: list, feature_count: int, train_rows: int, seed: int) -> SeedBatch:
        self.verify()
        request = {"seed": seed, "count": len(inputs), "feature_count": feature_count,
                   "train_rows": train_rows, "prior_seeds": inputs}
        payload = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        with self._lock:
            self.invocations += 1
        code, out, err = self._invoke(payload)
        head = err.decode("utf-8", "replace")[:STDERR_HEAD_CHARS]
        if code == 0:
            return self._interpret(inputs, out, head)
        if len(inputs) == 1:
            return SeedBatch({}, {}, [], 0, code, head, failed=[inputs[0]])
        mid = len(inputs) // 2
        left = self._batch(inputs[:mid], feature_count, train_rows, seed)
        right = self._batch(inputs[mid:], feature_count, train_rows, seed)
        return _merge_batches(left, right, code, head)

    def _interpret(self, inputs: list, out: bytes, head: str) -> SeedBatch:
        try:
            text = out.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EngineProtocolError(f"stdout is not UTF-8: {exc}") from None
        if not text.endswith("\n") or text.count("\n") != 1:
            raise EngineProtocolError("stdout must be exactly one JSON line")
        try:
            doc = json.loads(text)
        except ValueError as exc:
            raise EngineProtocolError(f"stdout is not JSON: {exc}") from None
        if not isinstance(doc, dict) or set(doc) != {"proposals", "schema"} or doc["schema"] != PROPOSALS_SCHEMA:
            raise EngineProtocolError("stdout is not a jope.gp-proposals/v7 document")
        proposals = doc["proposals"]
        if not isinstance(proposals, list) or len(proposals) != len(inputs):
            raise EngineProtocolError(f"expected {len(inputs)} proposals")
        hist = {}
        fill_ins = 0
        for p in proposals:
            if not isinstance(p, dict) or set(p) != _PROPOSAL_KEYS:
                raise EngineProtocolError("proposal keys differ from the v7 contract")
            dsl = p["dsl"]
            if not isinstance(dsl, str) or p["program_sha256"] != program_sha256(dsl):
                raise EngineProtocolError("program_sha256 != sha256(dsl)")
            if p["operator"] == "historical_seed":
                if dsl in hist:
                    raise EngineProtocolError("duplicate historical_seed dsl")
                hist[dsl] = p
            else:
                fill_ins += 1
        accepted = {x: x for x in inputs if x in hist}
        unmatched_in = [x for x in inputs if x not in hist]
        input_set = set(inputs)
        unmatched_out = [d for d in hist if d not in input_set]
        changed = {}
        if len(unmatched_in) == 1 and len(unmatched_out) == 1:
            # Each historical_seed output comes from exactly one input, and an input whose own text is
            # an output is canonical, so the only unmatched output belongs to the only unmatched input.
            changed[unmatched_in[0]] = unmatched_out[0]
            unmatched_in, unmatched_out = [], []
        return SeedBatch(accepted, changed, unmatched_in, fill_ins, 0, head, unattributed=unmatched_out)


def tier_b_verdict(engine: PinnedEngine, dsl: str, *, feature_count: int = 8, train_rows: int = 0) -> tuple:
    """One input with count=1: ('accept', canonical) | ('reject', None) | ('engine_failure', stderr_head)."""
    sb = engine.seed_batch([dsl], feature_count=feature_count, train_rows=train_rows)
    if dsl in sb.accepted:
        return ("accept", dsl)
    if dsl in sb.changed:
        return ("accept", sb.changed[dsl])
    if dsl in sb.failed:
        return ("engine_failure", sb.stderr_head)
    return ("reject", None)


def tier_a_verdict(dsl: str) -> tuple:
    """('accept', canonical) | ('reject', kind) | ('undecidable', None)."""
    try:
        return ("accept", canonical(dsl))
    except CanonicalError as exc:
        return ("reject", exc.kind)
    except Undecidable:
        return ("undecidable", None)


# ---------------------------------------------------------------------------------------------
# Drivers: corpus mode, generated mode, differential
# ---------------------------------------------------------------------------------------------
def _example(dsl: str) -> dict:
    return {"program_sha256": program_sha256(dsl), "head": dsl[:160]}


def corpus_check(engine: PinnedEngine, programs: Iterable, *, batch: int = 4096) -> dict:
    """Tier A then Tier B corpus mode over (program_sha256, dsl) pairs (distinct programs expected,
    e.g. programs.iter_distinct). Every program must hash to its sha, be canonical under Tier A, and
    come back unchanged from Tier B; anything else is drift (STOP)."""
    if not 1 <= batch <= MAX_SEED_BATCH:
        raise ValueError(f"batch must be 1..{MAX_SEED_BATCH}")
    c = Counter()
    examples: dict = {}
    pending: list = []
    pending_set: set = set()

    def note(kind: str, dsl: str) -> None:
        bucket = examples.setdefault(kind, [])
        if len(bucket) < 20:
            bucket.append(_example(dsl))

    def flush() -> None:
        if not pending:
            return
        sb = engine.seed_batch(pending)
        c["tier_b_batches"] += 1
        c["tier_b_invocations"] += sb.batches
        c["tier_b_fill_ins"] += sb.fill_ins
        c["tier_b_unattributed"] += len(sb.unattributed)
        for x in pending:
            if x in sb.accepted:
                c["tier_b_accepted_unchanged"] += 1
            elif x in sb.changed:
                c["tier_b_changed"] += 1
                note("tier_b_changed", x)
            elif x in sb.failed:
                c["tier_b_failed"] += 1
                note("tier_b_failed", x)
            else:
                c["tier_b_missing"] += 1
                note("tier_b_missing", x)
        pending.clear()
        pending_set.clear()

    for sha, dsl in programs:
        c["programs"] += 1
        if program_sha256(dsl) != sha:
            c["sha_mismatch"] += 1
            note("sha_mismatch", dsl)
        verdict, value = tier_a_verdict(dsl)
        if verdict == "accept":
            if value == dsl:
                c["tier_a_canonical"] += 1
            else:
                c["tier_a_changed"] += 1
                note("tier_a_changed", dsl)
        else:
            key = "tier_a_undecidable" if verdict == "undecidable" else "tier_a_rejected"
            c[key] += 1
            note(key, dsl)
        if dsl in pending_set:
            c["duplicate_in_batch"] += 1
            continue
        pending.append(dsl)
        pending_set.add(dsl)
        if len(pending) >= batch:
            flush()
    flush()
    bad = ("sha_mismatch", "tier_a_changed", "tier_a_rejected", "tier_a_undecidable", "tier_b_changed",
           "tier_b_missing", "tier_b_failed", "tier_b_unattributed", "tier_b_fill_ins")
    ok = c["programs"] > 0 and all(c[k] == 0 for k in bad)
    return {"schema": "jope.dl-oracle-corpus/v1", "engine_sha256": engine.sha256,
            "counts": {k: c[k] for k in sorted(set(c) | set(bad) | {"programs", "tier_a_canonical",
                                                                      "tier_b_accepted_unchanged"})},
            "examples": examples, "pass": ok}


def generated_check(engine: PinnedEngine, dsl: str, *, feature_count: int, train_rows: int,
                    noncanonical: str = "repair", semantics: bool = True, confirm_rejection: bool = False) -> dict:
    """Tier B generated mode (P10/P11) for one decoded program, in the design's order: Tier A
    canonical (repair or reject a parseable non-canonical text), one Tier B call that must return the
    canonical text unchanged (else drift: STOP), evaluator and island caps, width and lane ports,
    optional static semantics. The cache key is (context_sha256, program_sha256) for the caller.
    confirm_rejection replays a Tier A rejection through Tier B (callers do this for a sample); an
    engine acceptance then is drift."""
    if noncanonical not in ("repair", "reject"):
        raise ValueError("noncanonical must be repair or reject")
    out = {"input_sha256": program_sha256(dsl), "status": None, "canonical": None, "program_sha256": None,
           "noncanonical_repaired": False, "tier_a": None, "tier_b": None, "caps": [], "width_preflight": None,
           "semantics": []}
    verdict, value = tier_a_verdict(dsl)
    out["tier_a"] = verdict
    if verdict == "reject":
        out.update(status="invalid_parse", reason=value)
        if confirm_rejection:
            b_verdict, b_value = tier_b_verdict(engine, dsl, feature_count=feature_count, train_rows=train_rows)
            out["tier_b"] = b_verdict
            if b_verdict == "accept":
                out.update(status="drift", reason="tier_a rejected, tier_b accepted")
        return out
    candidate = dsl if verdict == "undecidable" else value
    if verdict == "accept" and value != dsl:
        if noncanonical == "reject":
            out.update(status="invalid_static", reason="noncanonical")
            return out
        out["noncanonical_repaired"] = True
    b_verdict, b_value = tier_b_verdict(engine, candidate, feature_count=feature_count, train_rows=train_rows)
    out["tier_b"] = b_verdict
    if verdict == "undecidable":
        if b_verdict == "engine_failure":
            out.update(status="invalid_engine_failure", reason=b_value)
            return out
        if b_verdict == "reject":
            out.update(status="invalid_parse", reason="tier_b_reject")
            return out
        if b_value != candidate:
            if noncanonical == "reject":
                out.update(status="invalid_static", reason="noncanonical")
                return out
            out["noncanonical_repaired"] = True
            candidate = b_value
            if tier_b_verdict(engine, candidate, feature_count=feature_count, train_rows=train_rows) != ("accept", candidate):
                out.update(status="drift", reason="tier_b canonical text is not stable")
                return out
    elif b_verdict != "accept" or b_value != candidate:
        out.update(status="drift", reason=f"tier_a accepted, tier_b {b_verdict}")
        return out
    out["canonical"] = candidate
    out["program_sha256"] = program_sha256(candidate)
    out["caps"] = form_caps(candidate)
    out["width_preflight"] = width_preflight(candidate, feature_count, train_rows)
    if semantics:
        out["semantics"] = static_semantics(candidate)
    if out["caps"]:
        out["status"] = "refused:caps"
    elif out["width_preflight"].startswith("refused"):
        out["status"] = out["width_preflight"]
    elif out["semantics"]:
        out["status"] = "invalid_semantics"
    else:
        out["status"] = "valid"
    return out


def _sampled(text: str, percent: int) -> bool:
    return int(program_sha256(text)[:8], 16) % 100 < percent


def differential(engine: PinnedEngine, inputs: Iterable, *, workers: int = 1, feature_count: int = 8,
                 max_examples: int = 50, resubmit_percent: int = 1) -> dict:
    """Tier A vs Tier B (count=1 per input) on every input; inputs are str or (category, str).

    They must agree on accept/reject and, when accepting, on the canonical text. Tier A undecidable
    inputs are resolved by Tier B and counted apart; an engine failure is never read as a rejection.
    For a deterministic resubmit_percent sample of the inputs Tier B changed, its canonical text is
    submitted again and must come back unchanged (the PLAN's idempotence re-submission)."""
    items = [(x if isinstance(x, tuple) else ("all", x)) for x in inputs]
    texts = [t for _, t in items]
    if len(set(texts)) != len(texts):
        raise ValueError("differential inputs must be distinct")
    tier_a = [tier_a_verdict(t) for t in texts]
    batches_before, verif_before = engine.invocations, engine.verifications

    def run_b(t: str) -> tuple:
        return tier_b_verdict(engine, t, feature_count=feature_count)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tier_b = list(pool.map(run_b, texts))
    else:
        tier_b = [run_b(t) for t in texts]
    resubmit = sorted({b[1] for t, b in zip(texts, tier_b)
                       if b[0] == "accept" and b[1] != t and _sampled(t, resubmit_percent)})
    resubmitted = [run_b(t) for t in resubmit]
    total, per_cat = Counter(), {}
    total["resubmitted"] = len(resubmit)
    total["resubmit_unchanged"] = sum(r == ("accept", t) for t, r in zip(resubmit, resubmitted))
    disagreements = []
    for (cat, text), a, b in zip(items, tier_a, tier_b):
        cc = per_cat.setdefault(cat, Counter())
        keys = ["inputs"]
        keys.append("a_" + ("accept_unchanged" if a == ("accept", text) else a[0] if a[0] != "accept" else "accept_changed"))
        keys.append("b_" + ("accept_unchanged" if b == ("accept", text) else b[0] if b[0] != "accept" else "accept_changed"))
        if b != ("accept", text):
            keys.append("noncanonical")
        if a[0] == "undecidable":
            keys.append("undecidable_resolved_by_b" if b[0] != "engine_failure" else "undecidable_engine_failure")
        else:
            same = (a[0] == "reject" and b[0] == "reject") or (a[0] == "accept" and b == a)
            keys.append("agree" if same else "disagree")
            if not same and len(disagreements) < max_examples:
                disagreements.append({"category": cat, "input": text, "tier_a": list(a), "tier_b": list(b)})
        for k in keys:
            cc[k] += 1
            total[k] += 1
    ok = (total["disagree"] == 0 and total["undecidable_engine_failure"] == 0 and total["inputs"] > 0
          and total["resubmit_unchanged"] == total["resubmitted"])
    return {"schema": "jope.dl-oracle-differential/v1", "engine_sha256": engine.sha256,
            "counts": dict(sorted(total.items())),
            "per_category": {k: dict(sorted(v.items())) for k, v in sorted(per_cat.items())},
            "engine_invocations": engine.invocations - batches_before,
            "engine_verifications": engine.verifications - verif_before,
            "disagreements": disagreements, "pass": ok}


# ---------------------------------------------------------------------------------------------
# Differential fuzz generator
# ---------------------------------------------------------------------------------------------
FUZZ_BASES = (
    "(pipe X (ridge_gcv))",
    "(pipe X (standardize) (histgbm depth=4 lr=0.1 min_leaf=8 rounds=160))",
    "(pipe X (robust_scale) (fft_mag axis=features) (polar) (rff_cos n=32 seed=7) (cheb degree=4) (ridge alpha=1.0))",
    "(pipe X (map expr=\"sigmoid(col(0))\") (derive expr=\"row_mean + c0\"))",
    "(pipe X (target kind=log1p q=0.05) (winsor_clip q=0.01804076952840708) (norm_row) (ridge lambda_frac=0.000001))",
    "(pipe X (power_transform kind=yeo_johnson) (select_corr by=corr k=24) (extra_trees depth=9 max_features=1.0 min_leaf=3 seed=999761 trees=128))",
    "(pipe X (impute add_indicator=true strategy=median value=-1.0) (step t=-0.0) (histgbm2 delta=0.000001 loss=huber lr=0.09988591634790045 rounds=300 val_shuffle=true))",
    "(pipe X (quantile_bins bins=16 append=true) (pair_prod topk=8) (gpr kernel=rbf+linear m=400 steps=40))",
    "(sym affine \"col(0)\")",
    "(sym raw \"col(0)\")",
    "(sym affine \"sigmoid(((c5^1.2705454457961025)/-1.1374340345718594))\")",
    "(sym affine \"((-0.87)--1.44)\")",
    "(sym affine \"(c1*-0.43)\")",
    "(sym affine \"clip(max(c3,row_mean),-8.727099312855202e-5,min(c0,c395))\")",
    "(portfolio (member (pipe X (ridge_gcv))) (member (pipe X (blom) (gpr kernel=rbf+linear steps=40))) (blend folds=5 guard=clip kind=nnls repeats=auto))",
    "(portfolio (member (pipe X (norm_row) (krr))) (member (pipe X (winsor_clip q=0.01804076952840708) (norm_row) (ridge_gcv))) (blend folds=5 guard=clip kind=nnls repeats=1))",
)

_FUZZ_WS = tuple(RUST_WHITESPACE)
_FUZZ_NOT_WS = ("\x1c", "\x1d", "\x1e", "\x1f", "\u200b", "\u180e", "\ufeff", "\u2060", "\x00", "\x7f")
_FUZZ_UNICODE = ("\u00e9", "\u00df", "\u03a9", "\u4e2d\u6587", "\u0131", "\u0130", "\u212a", "\u017f", "\u0665", "\uff15", "\U0001f642", "e\u0301", "\u05d0",
                 "\u00b2", "\u01c5", "\u2160")
_FUZZ_EDGE = (
    "", " ", "\xa0", "\x1c", "\t\n\r", "()", "(", ")", "pipe", "(pipe", "(pipe X", "(pipe X)", "(pipe X ())",
    "(pipe X (ridge))\x1c", "\x1c(pipe X (ridge))", "\x1f(pipe X (ridge))", "(pipe X (ridge))\x1d",
    "\xa0(pipe X (ridge))\u3000", "\u2028(pipe X (ridge))\u2029", "\x85(pipe X (ridge))\x85",
    "(pipe\xa0X\u2003(ridge))", "(pipe\x1cX (ridge))", "(pipe X\x1c(ridge))", "(pipe X (ridge)\x1c)",
    "(pipe X (ridge\x1c))", "( pipe X (ridge))", "(pipeX (ridge))", "(pipe X(ridge))", "(pipe X (ridge) )",
    "(pipe x (ridge))", "(PIPE X (ridge))", "(pipe X (ridge)) (ridge)", "(pipe X (ridge)))", "(pipe X ((ridge)))",
    "(pipe X (ridge lambda_frac))", "(pipe X (ridge =5))", "(pipe X (ridge k=))", "(pipe X (ridge k==5))",
    "(pipe X (\"\"))", "(pipe X (\"a b\"))", "(pipe X (\"a=b\"))", "(pipe \"X\" (ridge))", "(\"pipe\" X (ridge))",
    "(pipe X (ridge \"k=v\"))", "(pipe X (ridge \"k\"=v))", "(pipe X (ridge k\"x y\"=v))",
    "(pipe X (ridge k=\"a\\\\ b\"))", "(pipe X (ridge k=\"a\\\"b\"))", "(pipe X (ridge k=\"1.50\"))",
    "(pipe X (ridge k=\"5\"))", "(pipe X (ridge k=\"true\"))", "(pipe X (ridge k=\"TRUE\"))",
    "(pipe X (ridge k=\"nan\"))", "(pipe X (ridge k=\"NaN\"))", "(pipe X (ridge k=\"inf\"))", "(pipe X (ridge k=\"+inf\"))",
    "(pipe X (ridge k=1e-5))", "(pipe X (ridge k=1.50))", "(pipe X (ridge k=+5))", "(pipe X (ridge k=05))",
    "(pipe X (ridge k=TRUE))", "(pipe X (ridge k=nan))", "(pipe X (ridge k=-nan))", "(pipe X (ridge k=inf))",
    "(pipe X (ridge k=-INFINITY))", "(pipe X (ridge k=1_000))", "(pipe X (ridge k=9223372036854775807))",
    "(pipe X (ridge k=9223372036854775808))", "(pipe X (ridge k=-9223372036854775808))",
    "(pipe X (ridge k=-9223372036854775809))", "(pipe X (ridge k=1e400))", "(pipe X (ridge k=-1e-400))",
    "(pipe X (ridge k=\u0665))", "(pipe X (ridge k=\uff15))", "(pipe X (ridge k=\u0131nf))",
    "(pipe X (ridge k=\"\u0131nf\"))", "(pipe X (ridge k=1.))", "(pipe X (ridge k=.5))", "(pipe X (ridge k=.))",
    "(pipe X (ridge k=-0))", "(pipe X (ridge k=-0.0e0))", "(pipe X (ridge k=0x10))", "(pipe X (ridge k=1e))",
    "(sym affine \"c0\")\x1c", "\x1c(sym affine \"c0\")", "(sym affine \"c0\"\xa0)", "(sym  affine \"c0\")",
    "(sym affine  \"c0\")", "(sym affine \"c0\" )", "(sym\taffine \"c0\")", "(sym affine\t\"c0\")",
    "(sym affine\xa0\"c0\")", "(sym affine \xa0\"c0\")", "(sym AFFINE \"c0\")", "(sym Raw \"c0\")",
    "(sym affine \"c0\x1c\")", "(sym affine \"c0\xa0\")", "(sym affine \"\")", "(sym affine \"\"\"\")",
    "(sym affine \"c0\") )", "(sym affine \"c0\"))", "(sym affine c0)", "(sym affine \"c0)",
    "(portfolio (member (pipe X (ridge))))", "(portfolio (member (pipe X (ridge))) (blend))",
    "(portfolio (blend))", "(portfolio (member (pipe X (ridge))) (blend folds=5.9 repeats=1.9))",
    "(portfolio (member (pipe X (ridge))) (blend folds=\"5\"))", "(portfolio (member (pipe X (ridge))) (blend repeats=\"auto\"))",
)


_SPECIAL_NUMERALS = ("nan", "NaN", "-nan", "+nan", "NAN", "inf", "-inf", "+inf", "INF", "Infinity", "-infinity",
                     "+INFINITY", "infinit", "nan1", "1e999", "-1e999", "1e-999", "-0", "-0.0", "+0.0", "0e0", "-0e-0")


def _r_ws(rng: random.Random) -> str:
    k = rng.random()
    n = 1 if k < 0.6 else 2 if k < 0.85 else 3
    return "".join(rng.choice(_FUZZ_WS) if rng.random() < 0.7 else " " for _ in range(n))


def _editable(pipe: Pipeline) -> list:
    return [[op, [[k, format_param_value(v)] for k, v in params]] for op, params in pipe.stages]


def _render_edit(stages: list) -> str:
    return "(pipe X" + "".join(" (" + op + "".join(" " + k + "=" + v for k, v in params) + ")"
                               for op, params in stages) + ")"


def _int_spellings(v: int) -> list:
    s = str(v)
    neg = s.startswith("-")
    mag = s[1:] if neg else s
    sign = "-" if neg else ""
    return [("+" + s) if not neg else s.replace("-", "-0", 1), sign + "0" + mag, sign + "000" + mag,
            s + ".0", s + ".", s + "e0", s + "E+00", sign + mag + "0e-1", '"' + s + '"', s[:1] + "_" + s[1:],
            sign + "0x" + mag, s + ".000", ("+" + s + ".5") if not neg else s + ".5", sign + mag.zfill(25)]


def _float_spellings(v: float) -> list:
    r = repr(v)
    out = [r, f"{v:.17e}", f"{v:.25e}", f"{v:.20E}", format(Decimal(v), "f"), f"{v:.30f}", "+" + r,
           r + "0", r + "000", f"{v:e}", f"{v:.16g}", f"{v:.3f}"]
    if r.startswith("0."):
        out.append(r[1:])
    if r.startswith("-0."):
        out.append("-" + r[2:])
    if "e" not in r and "." in r:
        out.append(r.replace(".", "") + "e-" + str(len(r.split(".")[1])))
    return out


def is_shortest_tie(v: float) -> bool:
    """True when v prints via the shortest path and its exact value sits halfway between the two
    shortest round-trip candidates (where Rust rounds up in magnitude and CPython's repr to even)."""
    if not math.isfinite(v) or abs(v - round(v)) < 1e-12:
        return False
    te = Decimal(v).as_tuple()
    return (len(te.digits) <= 18 and te.digits[-1] == 5
            and len(te.digits) == len(Decimal(repr(v)).as_tuple().digits) + 1)


def _tie_double(rng: random.Random) -> float:
    """Rejection-sample a shortest-repr tie: an integer part of d digits plus an odd dyadic fraction
    whose exact expansion makes the total one digit longer than the shortest round-trip form."""
    for _ in range(400):
        d = rng.randint(1, 16)
        m = max(1, 18 - d + rng.randint(-2, 1))
        v = (rng.randrange(10 ** (d - 1), 10 ** d) + rng.randrange(1, 2 ** m, 2) / 2 ** m) * rng.choice((1.0, -1.0))
        if is_shortest_tie(v):
            return v
    return 1000000000000000.25


def _rand_double(rng: random.Random) -> float:
    k = rng.random()
    if k < 0.3:
        return rng.uniform(-1.0, 1.0) * 10.0 ** rng.uniform(-14.0, 17.0)
    if k < 0.45:
        n = rng.randint(-10 ** 6, 10 ** 6)
        return n + rng.choice((1e-13, -1e-13, 5e-13, -5e-13, 9.99e-13, -9.99e-13, 1e-12, -1e-12, 1.0001e-12,
                               -1.0001e-12, 2e-12, -2e-12, 0.5, -0.5, 0.25, 1e-9))
    if k < 0.6:
        while True:
            v = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0]
            if math.isfinite(v):
                return v
    if k < 0.75:  # shortest-repr ties (the one place CPython's repr and Rust's Display disagree)
        return _tie_double(rng)
    if k < 0.8:
        return rng.choice((5e-324, -5e-324, 2.2250738585072014e-308, 1e-12, 9.999999999999999e-13,
                           1.0000000000000002e-12, 1e-4, 9.999999999999999e-5, 1e16, 9007199254740993.0,
                           4503599627370495.5, 0.1, 0.2, 0.30000000000000004, 1 / 3, 2 / 3, math.pi, math.e,
                           1e15 + 0.3, 123456789.12345678, 1e-5, 1.5e-5, 0.000020741850001116463,
                           0.5408000000000001, 1.7976931348623157e308, -0.0, 0.0))
    return round(rng.uniform(-1.0, 1.0) * 10 ** rng.randint(0, 6), rng.randint(1, 17))


def _balanced_body(rng: random.Random, leaves: int) -> str:
    """2*leaves - 1 nodes in a balanced binary tree (depth ceil(log2(leaves)) + 1)."""
    if leaves == 1:
        return rng.choice(("c0", "c3", "x", "1.5", "row_mean", "C2"))
    half = leaves // 2
    return "(" + _balanced_body(rng, half) + rng.choice("+-*/^") + _balanced_body(rng, leaves - half) + ")"


def _rand_body(rng: random.Random, target_nodes: int) -> str:
    """A random MathExpr source with about `target_nodes` nodes (for the node/depth boundaries)."""
    leaves = ("c0", "c1", "c12", "row_mean", "x", "0.5", "1e-5", "2", "col(3)", "C7", "ROW_STD", "mean")
    if target_nodes <= 1:
        return rng.choice(leaves)
    k = rng.random()
    if k < 0.2:
        return "-" + _rand_body(rng, target_nodes - 1)
    if k < 0.35:
        f = rng.choice(("sin", "abs", "SQRT", "tanh", "erf", "log1p"))
        return f + "(" + _rand_body(rng, target_nodes - 1) + ")"
    left = rng.randint(1, max(1, target_nodes - 2))
    right = max(1, target_nodes - 1 - left)
    op = rng.choice("+-*/^")
    a, b = _rand_body(rng, left), _rand_body(rng, right)
    return rng.choice(("({}{}{})", "{} {} {}", "({} {} {})")).format(a, op, b)


class _Fuzzer:
    def __init__(self, bases: Sequence[str], seed: int) -> None:
        self.rng = random.Random(seed)
        self.pipes, self.syms, self.ports = [], [], []
        for b in bases:
            try:
                g = parse_genome(b)
                if canonical(b) != b:
                    continue
            except (CanonicalError, Undecidable):
                continue
            (self.pipes if isinstance(g, Pipeline) else self.syms if isinstance(g, Symbolic) else self.ports).append(g)
        if not self.pipes or not self.syms:
            raise ValueError("fuzz bases need at least one canonical pipe and one canonical sym program")
        rng = self.rng
        for _ in range(64):  # portfolios made from base pipes: canonical by construction
            ms = tuple(rng.choice(self.pipes) for _ in range(rng.choice((1, 2, 2, 3))))
            self.ports.append(Portfolio(ms, rng.random() < 0.5))

    # -- pipe mutations ------------------------------------------------------------------------
    def pipe(self) -> list:
        return _editable(self.rng.choice(self.pipes))

    def m_ws(self) -> str:
        rng = self.rng
        g = rng.choice(self.pipes + self.ports)
        base = render(g)
        if '"' in base:
            return None
        out = [_r_ws(rng) if rng.random() < 0.3 else ""]
        for i, ch in enumerate(base):
            if ch == " ":
                out.append(_r_ws(rng))
                continue
            out.append(ch)
            if ch == "(" and i > 0 and rng.random() < 0.15:
                out.append(_r_ws(rng))
            elif ch == ")" and i + 1 < len(base) and base[i + 1] == ")" and rng.random() < 0.15:
                out.append(_r_ws(rng))
        if rng.random() < 0.3:
            out.append(_r_ws(rng))
        return "".join(out)

    def m_lookalike(self) -> str:
        rng = self.rng
        base = render(rng.choice(self.pipes + self.syms))
        bad = rng.choice(_FUZZ_NOT_WS)
        k = rng.random()
        if k < 0.25:
            return bad + base
        if k < 0.5:
            return base + bad
        spaces = [i for i, ch in enumerate(base) if ch == " "]
        if spaces and k < 0.85:
            i = rng.choice(spaces)
            return base[:i] + (bad if rng.random() < 0.5 else " " + bad) + base[i + 1:]
        i = rng.randrange(len(base) + 1)
        return base[:i] + bad + base[i:]

    def m_order(self) -> str:
        rng = self.rng
        st = self.pipe()
        multi = [s for s in st if len(s[1]) >= 2]
        if not multi:
            return None
        s = rng.choice(multi)
        before = list(s[1])
        while s[1] == before:
            rng.shuffle(s[1])
        return _render_edit(st)

    def m_dup(self) -> str:
        rng = self.rng
        st = self.pipe()
        with_params = [s for s in st if s[1]]
        if not with_params:
            s = rng.choice(st)
            s[1] = [["append", "true"], ["append", rng.choice(("false", "1", "TRUE", "0"))]]
            return _render_edit(st)
        s = rng.choice(with_params)
        i = rng.randrange(len(s[1]))
        key = s[1][i][0]
        alt = rng.choice(("0", "1", "2.5", "-1.0", "true", "x", '"y z"', s[1][i][1]))
        j = rng.randrange(len(s[1]) + 1)
        s[1].insert(j, [key, alt])
        if rng.random() < 0.3:
            rng.shuffle(s[1])
        return _render_edit(st)

    def _numeric_slots(self, st: list) -> list:
        slots = []
        for si, (op, params) in enumerate(st):
            for pi, (k, v) in enumerate(params):
                pv = parse_param_value(v)
                if pv[0] in ("Int", "Float"):
                    slots.append((si, pi, pv))
        return slots

    def m_num(self) -> str:
        rng = self.rng
        st = self.pipe()
        slots = self._numeric_slots(st)
        if not slots:
            return None
        si, pi, (tag, v) = rng.choice(slots)
        if rng.random() < 0.1:
            spell = rng.choice(_SPECIAL_NUMERALS)
        else:
            spell = rng.choice(_int_spellings(v) if tag == "Int" else _float_spellings(v))
        st[si][1][pi][1] = spell
        return _render_edit(st)

    def m_float(self) -> str:
        rng = self.rng
        st = self.pipe()
        v = _rand_double(rng)
        spell = rng.choice(_float_spellings(v))
        slots = self._numeric_slots(st)
        if slots and rng.random() < 0.7:
            si, pi, _ = rng.choice(slots)
            st[si][1][pi][1] = spell
        else:
            st[-1][1].append([rng.choice(("lambda_frac", "lr", "q", "zz")), spell])
        return _render_edit(st)

    def m_i64(self) -> str:
        rng = self.rng
        st = self.pipe()
        lit = rng.choice(("9223372036854775807", "9223372036854775808", "-9223372036854775808",
                          "-9223372036854775809", "+9223372036854775807", "09223372036854775807",
                          "-09223372036854775808", "18446744073709551615", "18446744073709551616",
                          "99999999999999999999", "1" + "0" * 30, "0" * 30 + "1", "-" + "0" * 40,
                          "+0", "-0", "00", "4294967296", "-2147483649", "9007199254740993",
                          "9223372036854775807.0", "-9223372036854775808.5", "1e19", "-1e19"))
        slots = self._numeric_slots(st)
        if slots and rng.random() < 0.6:
            si, pi, _ = rng.choice(slots)
            st[si][1][pi][1] = lit
        else:
            st[-1][1].append([rng.choice(("seed", "rounds", "k")), lit])
        return _render_edit(st)

    def m_bool(self) -> str:
        rng = self.rng
        st = self.pipe()
        slots = [(si, pi) for si, (_, ps) in enumerate(st) for pi, (_, v) in enumerate(ps) if v in ("true", "false")]
        spell = rng.choice(("TRUE", "True", "tRuE", "FALSE", "False", "fAlSe", '"true"', '"false"', "1", "0",
                            "yes", "t", "truE", "\"TRUE\"", "true\u200b"))
        if slots:
            si, pi = rng.choice(slots)
            st[si][1][pi][1] = spell
        else:
            st[rng.randrange(len(st))][1].append(["append", spell])
        return _render_edit(st)

    def m_quote(self) -> str:
        rng = self.rng
        st = self.pipe()
        q = rng.choice(('"a b"', '"a\\"b"', '"a\\\\b"', '"a\\\\ b"', '"x=y"', '"(x)"', '""', '"\\""', '"1.50"',
                        '"5"', '"true"', '"TRUE"', '"nan"', '"NaN"', '"inf"', '"-inf"', '"+inf"', '"abc',
                        'a"b', 'a"b"c', '"a"b"', '"\\\\"', '"a\\b"', '"tab\there"', '"nb\xa0sp"', '"fs\x1cx"',
                        '"\u3000"', '"\u00e9"', '"rbf+linear"', '"huber"', 'x\\y', '"x\\', '"\\"x\\""', '"0.1"', '"-0.0"'))
        slots = [(si, pi) for si, (_, ps) in enumerate(st) for pi in range(len(ps))]
        k = rng.random()
        if slots and k < 0.5:
            si, pi = rng.choice(slots)
            st[si][1][pi][1] = q
        elif slots and k < 0.65:
            si, pi = rng.choice(slots)
            kk, vv = st[si][1][pi]
            st[si][1][pi] = ['"' + kk + "=" + vv + '"', None]      # the whole param as a quoted atom
            return "(pipe X" + "".join(" (" + op + "".join(" " + a + ("" if b is None else "=" + b) for a, b in ps) + ")"
                                       for op, ps in st) + ")"
        elif k < 0.8:
            si = rng.randrange(len(st))
            st[si][0] = '"' + st[si][0] + '"' if rng.random() < 0.7 else '"' + st[si][0] + ' x"'
        else:
            st[rng.randrange(len(st))][1].append([rng.choice(("expr", "kind", "k\"q\"")), q])
        return _render_edit(st)

    def m_unicode(self) -> str:
        rng = self.rng
        st = self.pipe()
        u = rng.choice(_FUZZ_UNICODE)
        k = rng.random()
        si = rng.randrange(len(st))
        if k < 0.3:
            st[si][0] = st[si][0] + u
        elif k < 0.55:
            st[si][1].append([u + "key", rng.choice(("1", "x", '"' + u + '"', u))])
        elif k < 0.8:
            st[si][1].append(["kind", rng.choice(('"' + u + '"', u + " ", '"' + u + ' x"', u))])
        else:
            text = _render_edit(st)
            i = rng.randrange(len(text) + 1)
            return text[:i] + u + text[i:]
        text = _render_edit(st)
        return text.replace(" ", rng.choice(_FUZZ_WS), 1) if rng.random() < 0.5 else text

    def m_struct(self) -> str:
        rng = self.rng
        base = render(rng.choice(self.pipes + self.syms + self.ports))
        k = rng.randrange(12)
        if k == 0:
            i = rng.randrange(len(base))
            return base[:i] + base[i + 1:]
        if k == 1:
            i = rng.randrange(len(base) + 1)
            return base[:i] + rng.choice(("(", ")", '"', "=", " ", "\\", "X", "\x1c", "\xa0", ",", "e")) + base[i:]
        if k == 2:
            i = rng.randrange(len(base) - 1)
            return base[:i] + base[i + 1] + base[i] + base[i + 2:]
        if k == 3:
            return base.replace(" X", "", 1)
        if k == 4:
            return base.replace("pipe", rng.choice(("PIPE", "Pipe", "pipes", "pip")), 1)
        if k == 5:
            return base[:rng.randrange(1, len(base))]
        if k == 6:
            return base + rng.choice((")", " junk", " (ridge)", "(", '"', " x=1"))
        if k == 7:
            return base.replace("(pipe X", "(pipe X (" + rng.choice(("", "=", "a=b", "x")) + ")", 1)
        if k == 8:
            return base.replace(" (", " ((", 1).replace(")", "))", 1)
        if k == 9:
            return base.replace("=", rng.choice(("", "==", " = ", "=\xa0")), 1)
        if k == 10:
            return base.replace(")", " k=)", 1)
        return base.replace("(", "( ", 1)

    # -- sym ----------------------------------------------------------------------------------
    def m_sym_wrap(self) -> str:
        rng = self.rng
        g = rng.choice(self.syms)
        mode = "affine" if g.affine else "raw"
        body = g.source
        k = rng.randrange(14)
        if k == 0:
            return "(sym" + " " * rng.randint(2, 4) + mode + ' "' + body + '")'
        if k == 1:
            return "(sym " + mode + _r_ws(rng) + ' "' + body + '")'
        if k == 2:
            return "(sym " + mode + ' "' + body + '"' + _r_ws(rng) + ")"
        if k == 3:
            return _r_ws(rng) + "(sym " + mode + ' "' + body + '")' + _r_ws(rng)
        if k == 4:
            return "(sym" + rng.choice(_FUZZ_WS[:5] + ("\xa0", "\u3000")) + mode + ' "' + body + '")'
        if k == 5:
            return "(sym " + mode + rng.choice(("\t", "\xa0", "\u2003", "\n")) + '"' + body + '")'
        if k == 6:
            return "(sym " + rng.choice((mode.upper(), mode.capitalize(), mode + "s", mode[:-1], "bogus")) + ' "' + body + '")'
        if k == 7:
            return "(sym " + mode + " " + body + ")"
        if k == 8:
            return "(sym " + mode + ' "' + body + ")"
        if k == 9:
            return "(sym " + mode + ' ""' + body + '"")'
        if k == 10:
            return "(sym " + mode + ' "' + body + '")' + rng.choice((" trailing", ")", " )", "\x1c", "\x1f"))
        if k == 11:
            return "(SYM " + mode + ' "' + body + '")'
        if k == 12:
            return "(sym " + mode + ' "' + body + '"' + rng.choice(("\x1c", " \x1c", "\x1e ")) + ")"
        return "(sym " + mode + "  " + rng.choice(("\xa0", "\u3000", "")) + '"' + body + '")'

    def m_sym_body(self) -> str:
        rng = self.rng
        g = rng.choice(self.syms)
        body = g.source
        k = rng.randrange(16)
        if k == 0:
            body = body.upper()
        elif k == 1:
            body = re.sub(r"\bc(\d+)", lambda m: rng.choice(("col", "C", "c0", "COL")) + m.group(1), body)
        elif k == 2:
            body = body.replace("row_mean", "mean").replace("row_l2", "row_norm") if "row_" in body else "mean+row_norm"
        elif k == 3:
            i = rng.randrange(len(body) + 1)
            body = body[:i] + rng.choice((" ", "\xa0", "\t", "\x1c", "\u3000", "\u200b")) + body[i:]
        elif k == 4:
            body = rng.choice(("1e400", "1e", "1.2.3", ".", "..5", "1e+", "1.e5", ".5", "5.", "1E-5", "1e-400",
                               "0x10", "1_0", "inf", "nan", "+1", "--1", "-+-c0", "1ee5", "2e3e4", "\u0661"))
        elif k == 5:
            body = rng.choice(("c", "col", "col()", "col(1,2)", "col(-1)", "col(+2)", "col(2.0000000000001)",
                               "col(2.5)", "col(c1)", "col((3))", "col(1e30)", "c18446744073709551615",
                               "c18446744073709551616", "col007", "c007", "made_up(x)", "sin", "sin()",
                               "sin(c0,c1)", "min(c0)", "max(c0,c1,c2)", "clip(c0,c1)", "clip(c0,1,2,3)",
                               "step(c0)", "row_l1+row_sum+x", "ROW_STD^-2", "_x", "c0c1", "c0 c1", "(c0", "c0)",
                               "()", "", ",", "c0,c1", "sin (c0)", "SIN(C0)", "Mean", "e", "E5", "c1\u00e9", "c1\"x"))
        elif k == 6:
            i = rng.randrange(len(body))
            body = body[:i] + body[i + 1:]
        elif k == 7:
            body = "(" * rng.randint(1, 3) + body + ")" * rng.randint(1, 3)
        elif k == 8:
            body = "-" * rng.randint(1, 20) + "(" + body + ")"
        elif k == 9:
            body = body.replace("(", "( ").replace(",", " , ")
        else:
            body = _rand_body(rng, rng.randint(1, 30))
        wrap = rng.random()
        mode = "affine" if g.affine else "raw"
        if wrap < 0.5:
            return "(sym " + mode + ' "' + body + '")'
        return "(sym " + mode + _r_ws(rng) + '"' + body + '")' if wrap < 0.8 else "  (sym " + mode + ' "' + body + '") '

    def m_nodes_depth(self) -> str:
        rng = self.rng
        k = rng.randrange(8)
        if k == 0:   # left-deep chain: 2n+1 nodes, depth n+1
            n = rng.choice((31, 32, 33))
            body = "c0" + "".join(rng.choice("+-*/") + rng.choice(("c1", "1", "x")) for _ in range(n))
        elif k == 1:  # nested calls: depth d+1, nodes d+1
            d = rng.choice((14, 15, 16, 17))
            f = rng.choice(("sigmoid", "sin", "abs", "LOG1P"))
            body = (f + "(") * d + "c0" + ")" * d
        elif k == 2:  # unary chain
            d = rng.choice((14, 15, 16, 17))
            body = "-" * d + "c0"
        elif k == 3:  # pow chain (right associative)
            d = rng.choice((7, 8, 9))
            body = "^".join(["c0"] * d)
        elif k == 4:  # random body with a target size
            body = _rand_body(rng, rng.choice((62, 63, 64, 65, 66)))
        elif k == 5:  # balanced trees: 63/64/65/66 nodes at depth <= 8 (the node cap on its own)
            leaves = rng.choice((32, 33))
            body = rng.choice(("", "-")) + _balanced_body(rng, leaves)
        elif k == 6:  # redundant parens add no node (frame-safe depth)
            d = rng.choice((16, 17, 100, 300, 600))
            body = "(" * d + rng.choice(("c0", "-c0", "c0+c1")) + ")" * d
        else:         # unary plus adds no node
            d = rng.choice((15, 16, 17, 40))
            body = "+" * d + "-" * rng.choice((14, 15, 16)) + "c0"
        mode = rng.choice(("affine", "raw"))
        return rng.choice(("(sym  {} \"{}\")", "(sym {}  \"{}\")", " (sym {} \"{}\")", "(sym {} \"{}\" )",
                           "(sym {} \"{}\")\n")).format(mode, body)

    # -- portfolio ----------------------------------------------------------------------------
    def m_blend(self) -> str:
        rng = self.rng
        members = list(rng.choice(self.ports).members)
        parts = {"kind": "kind=nnls", "folds": "folds=5", "repeats": rng.choice(("repeats=auto", "repeats=1")),
                 "guard": "guard=clip"}
        k = rng.randrange(10)
        if k == 0:
            keys = list(parts)
            rng.shuffle(keys)
            blend = "(blend " + " ".join(parts[x] for x in keys) + ")"
        elif k == 1:
            keys = [x for x in parts if rng.random() < 0.5]
            blend = "(blend" + "".join(" " + parts[x] for x in keys) + ")"
        elif k == 2:
            blend = rng.choice(("", "(blend)"))
        elif k == 3:
            key = rng.choice(("folds", "repeats"))
            val = rng.choice(("5", "5.0", "5.9", "6", "4.99", "05", "+5", '"5"', "5e0", "1", "1.0", "1.9", "0.9",
                              "2", "auto", "AUTO", '"auto"', "true", "-0.0", "nan", "inf", "0", "1e0", "+1"))
            parts[key] = key + "=" + val
            blend = "(blend " + " ".join(parts.values()) + ")"
        elif k == 4:
            parts[rng.choice(("kind", "guard"))] = rng.choice(("kind=NNLS", "kind=\"nnls\"", "kind=caruana",
                                                               "guard=CLIP", "guard=\"clip\"", "guard=none", "kind=1"))
            blend = "(blend " + " ".join(parts.values()) + ")"
        elif k == 5:
            blend = "(blend " + " ".join(parts.values()) + " " + rng.choice(("bogus=1", "x=", "folds=5", "repeats=1")) + ")"
        elif k == 6:
            blend = "(blend) (blend)" if rng.random() < 0.5 else "(blend repeats=1) (member " + render(members[0]) + ")"
        elif k == 7:
            n = rng.choice((0, 1, 12, 13))
            members = [rng.choice(self.pipes) for _ in range(n)]
            blend = "(blend " + rng.choice(("repeats=1", "repeats=auto", "")) + ")"
        elif k == 8:
            members = members + [Pipeline((("portfolio", (("preset", ("Str", "core")),)),))]
            blend = "(blend)"
        else:
            blend = "(blend " + " ".join(parts.values()) + ")"
            text = "(portfolio" + "".join(" (member " + render(m) + ")" for m in members) + " " + blend + ")"
            return text.replace(" ", _r_ws(self.rng), rng.randint(1, 6))
        return "(portfolio" + "".join(" (member " + render(m) + ")" for m in members) + (" " + blend if blend else "") + ")"


FUZZ_CATEGORIES = (  # (name, method, share of n)
    ("ws", "m_ws", 0.11), ("lookalike", "m_lookalike", 0.05), ("order", "m_order", 0.08), ("dup", "m_dup", 0.08),
    ("num", "m_num", 0.12), ("float", "m_float", 0.10), ("i64", "m_i64", 0.03), ("bool", "m_bool", 0.03),
    ("quote", "m_quote", 0.08), ("unicode", "m_unicode", 0.05), ("struct", "m_struct", 0.08),
    ("sym_wrap", "m_sym_wrap", 0.06), ("sym_body", "m_sym_body", 0.05), ("nodes_depth", "m_nodes_depth", 0.03),
    ("blend", "m_blend", 0.05),
)


def fuzz_inputs(bases: Sequence[str] = FUZZ_BASES, *, n: int = 24000, seed: int = 0) -> list:
    """Deterministic, distinct (category, text) inputs for the differential: the fixed edge list, then
    mutations of canonical bases by category share. Never a lone surrogate, and the sym nesting stays
    under RUST_FRAME_LIMIT so no input can overflow the engine's stack."""
    fz = _Fuzzer(list(bases), seed)
    seen = set(bases)
    out = []
    for text in _FUZZ_EDGE:
        if text not in seen:
            seen.add(text)
            out.append(("edge", text))
    quotas = [(name, getattr(fz, meth), max(1, int(round(share * n)))) for name, meth, share in FUZZ_CATEGORIES]
    for name, fn, quota in quotas:
        made = 0
        tries = 0
        while made < quota and tries < quota * 40:
            tries += 1
            text = fn()
            if text is None or text in seen:
                continue
            seen.add(text)
            out.append((name, text))
            made += 1
    return out


# ---------------------------------------------------------------------------------------------
# Built-in Rust #[test] vectors (replica-check)
# ---------------------------------------------------------------------------------------------
# (source, expected canonical text or None for an error); transcribed from dsl.rs, genome.rs,
# math_expr.rs (bodies wrapped in `(sym affine "...")`) and proposal main.rs canonical() calls.
RUST_VECTORS = (
    ("(pipe X (robust_scale) (fft_mag axis=features) (polar) (rff_cos n=32 seed=7) (cheb degree=4) (ridge alpha=1.0))",
     "(pipe X (robust_scale) (fft_mag axis=features) (polar) (rff_cos n=32 seed=7) (cheb degree=4) (ridge alpha=1.0))"),
    ("(robust_scale X)", None),
    ("(pipe X)", None),
    ("(pipe X (map expr=\"sigmoid(col(0))\") (derive expr=\"row_mean + c0\"))",
     "(pipe X (map expr=\"sigmoid(col(0))\") (derive expr=\"row_mean + c0\"))"),
    ("(sym bogus \"col(0)\")", None),
    ("(sym affine \"col(0)\"", None),
    ("(sym raw col(0))", None),
    ("(sym raw \"col(0))", None),
    ("(sym raw col(0)\")", None),
    ("(sym raw \"\"col(0)\"\")", None),
    ("(sym raw \"col(0)\") trailing", None),
    ("(sym raw \"col(0)\")", "(sym raw \"col(0)\")"),
    ("(sym affine \"col(0)\")", "(sym affine \"col(0)\")"),
    ("(portfolio (member (pipe X (ridge_gcv))) (member (pipe X (blom) (gpr kernel=rbf+linear steps=40))) "
     "(blend kind=nnls folds=5 repeats=auto guard=clip))",
     "(portfolio (member (pipe X (ridge_gcv))) (member (pipe X (blom) (gpr kernel=rbf+linear steps=40))) "
     "(blend folds=5 guard=clip kind=nnls repeats=auto))"),
    ("(portfolio (member (pipe X (blom) (gpr kernel=rbf+linear steps=40))) (member (pipe X (ridge_gcv))) (blend))",
     "(portfolio (member (pipe X (blom) (gpr kernel=rbf+linear steps=40))) (member (pipe X (ridge_gcv))) "
     "(blend folds=5 guard=clip kind=nnls repeats=auto))"),
    ("(portfolio (blend))", None),
    ("(portfolio" + " (member (pipe X (ridge)))" * 13 + " (blend))", None),
    ("(portfolio (member (pipe X (portfolio preset=core))) (blend))", None),
    ("(portfolio (member (pipe X (ridge))) (blend kind=caruana))", None),
    ("(portfolio (member (pipe X (ridge))) (blend folds=3))", None),
    ("(portfolio (member (pipe X (ridge))) (blend repeats=2))", None),
    ("(portfolio (member (pipe X (ridge))) (blend guard=none))", None),
    ("(portfolio (member (pipe X (ridge))) (blend bogus=1))", None),
    ("(portfolio (blend) (member (pipe X (ridge))))", None),
    ("(portfolio (member (pipe X (ridge))) (blend repeats=1))",
     "(portfolio (member (pipe X (ridge))) (blend folds=5 guard=clip kind=nnls repeats=1))"),
    ("(sym affine \"clip(1,(-1)^0.5,(-1)^0.5)\")", "(sym affine \"clip(1,(-1)^0.5,(-1)^0.5)\")"),
    ("(sym affine \"min(clip(1,(-1)^0.5,(-1)^0.5),7)+1\")", "(sym affine \"min(clip(1,(-1)^0.5,(-1)^0.5),7)+1\")"),
    ("(sym affine \"clip(1,(-1)^0.5,3)\")", "(sym affine \"clip(1,(-1)^0.5,3)\")"),
    ("(sym affine \"sigmoid(x) + c0\")", "(sym affine \"sigmoid(x) + c0\")"),
    ("(sym affine \"clip(row_l2 + row_mean, 0, 10)\")", "(sym affine \"clip(row_l2 + row_mean, 0, 10)\")"),
    ("(sym affine \"step(c0, c1)\")", "(sym affine \"step(c0, c1)\")"),
    ("(sym affine \"step(c0)\")", None),
    ("(sym affine \"made_up(x)\")", None),
    ("(sym affine \"" + "sigmoid(" * 17 + "x" + ")" * 17 + "\")", None),
    ("(pipe X (standardize) (histgbm depth=4 lr=0.1 min_leaf=8 rounds=160))",
     "(pipe X (standardize) (histgbm depth=4 lr=0.1 min_leaf=8 rounds=160))"),
    ("(pipe X (histgbm2 bags=3 col_subsample=0.8 delta=1.0 leaves=63 loss=huber lr=0.07 min_leaf=5 rounds=300 "
     "row_subsample=0.85 val_pct=20))",
     "(pipe X (histgbm2 bags=3 col_subsample=0.8 delta=1.0 leaves=63 loss=huber lr=0.07 min_leaf=5 rounds=300 "
     "row_subsample=0.85 val_pct=20))"),
    ("(pipe X (robust_scale) (histgbm bins=64 depth=7 lambda=1.02 lr=0.202 max_features=294 min_leaf=9 rounds=400))",
     "(pipe X (robust_scale) (histgbm bins=64 depth=7 lambda=1.02 lr=0.202 max_features=294 min_leaf=9 rounds=400))"),
)


def replica_check(engine: PinnedEngine | None = None) -> dict:
    results = []
    for src, want in RUST_VECTORS:
        verdict, value = tier_a_verdict(src)
        got = value if verdict == "accept" else None
        row = {"source": src, "expected": want, "tier_a": got, "ok": got == want and verdict != "undecidable"}
        if engine is not None:
            bv, bval = tier_b_verdict(engine, src)
            row["tier_b"] = bval if bv == "accept" else None
            row["ok"] = row["ok"] and row["tier_b"] == want
        results.append(row)
    return {"schema": "jope.dl-oracle-replica-check/v1", "vectors": len(results),
            "passed": sum(r["ok"] for r in results), "failures": [r for r in results if not r["ok"]],
            "pass": all(r["ok"] for r in results), "tier_b": engine is not None}


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def _iter_corpus(path: Path, partitions: Sequence[str]) -> Iterator:
    """(program_sha256, dsl) from a JSONL file or a P4a program-dict directory. Every partition, heldout included,
    is read through the projection role 'roundtrip' ((program_sha256, dsl) only): Tier A corpus mode is a P4 read,
    not P10 reporting."""
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    yield rec["program_sha256"], rec["dsl"]
        return
    try:
        from dltrain import programs as _programs
        iter_distinct = getattr(_programs, "iter_distinct", None)
    except ImportError:
        iter_distinct = None
    for part in partitions:
        if iter_distinct is not None:
            yield from iter_distinct(path, part, role="roundtrip")
            continue
        import pyarrow.parquet as pq
        for f in sorted((path / part).glob("*.parquet")):
            table = pq.read_table(f, columns=["program_sha256", "dsl"])
            for sha, dsl in zip(table.column("program_sha256").to_pylist(), table.column("dsl").to_pylist()):
                yield (sha.hex() if isinstance(sha, (bytes, bytearray)) else sha), dsl


def context_shapes(epoch: Path) -> dict:
    """context_sha256 -> (encoded_features, train_rows) from each included context's pinned prepared.json
    (sha256-verified; only those two fields are read)."""
    doc = json.loads(Path(epoch).read_bytes())
    shapes = {}
    for c in doc["contexts"]:
        if c.get("status") != "included":
            continue
        ref = c["prepared_ref"]
        data = Path(ref["path"]).read_bytes()
        if common.sha256_bytes(data) != ref["sha256"]:
            raise RuntimeError(f"prepared.json sha256 mismatch for {c['context_sha256']}")
        ctx = json.loads(data)["context"]
        shapes[c["context_sha256"]] = (int(ctx["encoded_features"]), int(ctx["train_rows"]))
    return shapes


def _iter_context_rows(path: Path, partition: str) -> Iterator:
    """(context_sha256, dsl) rows of one P4a partition, through the projection role 'preflight' (those two columns
    only), for the informational width preflight."""
    try:
        from dltrain import programs as _programs
        open_partition = getattr(_programs, "open_partition", None)
    except ImportError:
        open_partition = None
    if open_partition is not None:
        batches = open_partition(path, partition, role="preflight").to_batches(columns=["context_sha256", "dsl"],
                                                                              use_threads=False)
    else:
        import pyarrow.parquet as pq
        batches = (b for f in sorted((path / partition).glob("*.parquet"))
                   for b in pq.ParquetFile(f).iter_batches(columns=["context_sha256", "dsl"]))
    for batch in batches:
        for ctx, dsl in zip(batch.column("context_sha256").to_pylist(), batch.column("dsl").to_pylist()):
            yield (ctx.hex() if isinstance(ctx, (bytes, bytearray)) else ctx), dsl


def preflight_histogram(rows: Iterable, shapes: dict) -> dict:
    """Informational width_preflight counts over (context_sha256, dsl) rows; never a gate."""
    hist = Counter()
    for ctx, dsl in rows:
        shape = shapes.get(ctx)
        hist["context_unknown" if shape is None else width_preflight(dsl, shape[0], shape[1])] += 1
    return dict(sorted(hist.items()))


def _emit(doc: dict, out: str | None) -> None:
    doc = dict(doc, utc_us=common.utc_us(), feat_math_pins=FEAT_MATH_PINS, engine_source_pins=ENGINE_SOURCE_PINS)
    if out:
        res = common.write_receipt(out, doc)
        print(json.dumps({k: res[k] for k in ("path", "sha256", "bytes")}))
    else:
        print(json.dumps(doc, sort_keys=True, ensure_ascii=False))


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dltrain.oracle", description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", default=str(DEFAULT_ENGINE), help="pinned engine copy (sha256-verified)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rc = sub.add_parser("replica-check")
    rc.add_argument("--tier-b", action="store_true", help="also run every vector through the engine")
    rc.add_argument("--out")
    ta = sub.add_parser("tier-a")
    ta.add_argument("--corpus", required=True, help="program-dict dir (P4a) or a JSONL of {program_sha256,dsl}")
    ta.add_argument("--partitions", default="train_fit,dev,diag,heldout")
    ta.add_argument("--batch", type=int, default=MAX_SEED_BATCH)
    ta.add_argument("--epoch", help="epoch manifest: adds the informational width_preflight histogram per partition")
    ta.add_argument("--out")
    tb = sub.add_parser("tier-b")
    tb.add_argument("--feature-count", type=int, required=True)
    tb.add_argument("--train-rows", type=int, required=True)
    tb.add_argument("--noncanonical", choices=("repair", "reject"), default="repair")
    tb.add_argument("--no-semantics", action="store_true")
    g = tb.add_mutually_exclusive_group(required=True)
    g.add_argument("--dsl")
    g.add_argument("--input", help="JSONL with a 'dsl' field per line")
    df = sub.add_parser("differential")
    df.add_argument("--n", type=int, default=24000)
    df.add_argument("--seed", type=int, default=0)
    df.add_argument("--bases", help="JSONL with a 'dsl' field per line (canonical programs)")
    df.add_argument("--workers", type=int, default=3)
    df.add_argument("--out")
    args = ap.parse_args(argv)
    engine = PinnedEngine(args.engine)
    if args.cmd == "replica-check":
        doc = replica_check(engine if args.tier_b else None)
        _emit(doc, args.out)
        return 0 if doc["pass"] else 1
    if args.cmd == "tier-a":
        parts = [p for p in args.partitions.split(",") if p]
        doc = corpus_check(engine, _iter_corpus(Path(args.corpus), parts), batch=args.batch)
        doc["partitions"] = parts
        if args.epoch and Path(args.corpus).is_dir():
            shapes = context_shapes(Path(args.epoch))
            doc["width_preflight"] = {p: preflight_histogram(_iter_context_rows(Path(args.corpus), p), shapes)
                                      for p in parts}
        _emit(doc, args.out)
        return 0 if doc["pass"] else 1
    if args.cmd == "tier-b":
        if args.dsl is not None:
            texts = [args.dsl]
        else:
            with open(args.input, encoding="utf-8") as f:
                texts = [json.loads(line)["dsl"] for line in f if line.strip()]
        for t in texts:
            print(json.dumps(generated_check(engine, t, feature_count=args.feature_count, train_rows=args.train_rows,
                                             noncanonical=args.noncanonical, semantics=not args.no_semantics),
                             sort_keys=True, ensure_ascii=False))
        return 0
    bases = list(FUZZ_BASES)
    if args.bases:
        with open(args.bases, encoding="utf-8") as f:
            bases += [json.loads(line)["dsl"] for line in f if line.strip()]
    inputs = fuzz_inputs(bases, n=args.n, seed=args.seed)
    doc = differential(engine, inputs, workers=args.workers)
    doc.update(n_requested=args.n, seed=args.seed, bases=len(bases))
    _emit(doc, args.out)
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
