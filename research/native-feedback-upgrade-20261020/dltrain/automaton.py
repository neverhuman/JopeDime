"""PLAN §5 P4 grammar automaton over the lossless DSL tokenizer (design: P4 "automaton" section).

TWO MODES over one vocabulary (dltrain.dsl_tokenizer):

SYNTAX  a data-independent superset of the canonical GenomeIr::canonical() image, used for teacher forcing of
        dev, diag and held programs that POLICY rejects. It allows every parser form (pipes; sym affine and raw;
        portfolios with 1..12 members and either blend), strictly increasing keys, canonical numeric lexical
        forms, spelled names (OP/KEY/STR/ID *_BEGIN bytes *_END, with the tokenizer's uniqueness rule), quoted
        strings, whitespace inside sym bodies, DSL <= 32,768 bytes and <= max_seq tokens. Gate: it accepts 100%
        of every partition's tokenization (within max_seq). It is also inside the parser's language: every
        sequence it admits detokenizes to text Tier A (dltrain.oracle.parse_genome) parses.
POLICY  SYNTAX intersected with the train_fit signature tables (signatures.json), the evaluator and island
        caps and max_seq:
        observed forms, op roles (final / non-final), per-op tries of observed sorted keysets, per-(op,key)
        value tables (kinds, CAT sets, Bool values, numeric digit maxima and signs, observed STR literals),
        observed sym features (numeral shapes, column digits, identifiers, unary/binary operators, groups,
        whitespace, node cap), portfolio member counts and blend. No spelled names and no RAW. Gate: it accepts
        100% of train_fit by construction (a miss is an incomplete table: STOP).

Generation-only rules (Rules; never part of an acceptance gate): target_lead (target only as stage 0, input only
as stage 0 or right after a leading target; advanced_v3/pipeline.rs:469-497) and col_lt_p (c<N> with
N < feature_count; spelled column identifiers and col(N) calls are refused under it).

STATE. An immutable, hashable NamedTuple (mode, ctx, used, form, sub): `ctx` interns (mode, rules) per Grammar;
`used` is the resource vector (tokens, bytes, '(' count, "(winsor_clip" count, pipe stages, sym nodes) consumed so
far; `sub` is the form's structural sub-state (pipe frame: phase, member flag, target-lead state, op role, trie
node or last key, value table, numeric DFA or spelled bytes; portfolio: phase, members; sym: phase, a bounded
frame stack of GROUP | CALL(arity, args done, max arg depth) | COL frames, an operator-precedence depth record per
level, the numeral / column-digit DFA or spelled identifier bytes, pending whitespace bytes).

LIVENESS. Every cap is a resource: a token t is allowed in s iff advance(s, t) is valid and some vector v of the
completion FRONTIER C(s') satisfies used(s') + v <= caps(s') component-wise. C(s) is the set of Pareto-minimal
resource vectors (tokens, bytes, '(' count, "(winsor_clip" count, stages, sym nodes) over every completion of s
(SYNTAX caps only tokens, bytes and nodes, so it projects the other three out). One lexicographic minimum is not
enough: the token-minimal completion can need more '(' (a short STR literal full of parentheses), stages or bytes
than a longer one, and POLICY would then refuse programs of its own train_fit. Every frontier vector is realized
by a token whose successor frontier holds the rest of it, so a masked walk never dead-ends or overruns a cap, and
acceptance is exact for all six caps. The spelled-name completion searches try every byte that can continue the
previous key's path (0x00-0xFF) and a representative byte of every other transition class; their memo keys are
exact equivalence classes (a quoted STR keeps whether its content is still empty), and a search that stops at its
limit is reported as such, never as a dead end. The sym depth limit (<= 16) is exact: each level keeps the
operator-precedence depths (A add, M mul, pow chain count/max, pending unary minus, last operand).

NOT REGULAR (Tier A enforces them at decode, design risks): Rust's shortest round-trip float rule and the 1e-12
integral rule (the automaton admits some non-canonical floats) and i64 overflow of 19-digit Ints. SYNTAX also
refuses a few absurd canonical texts outside its finite-state view (documented refusals, never in the corpus):
sym numerals >= 1e308 (finiteness is proven by sig_int_digits + exponent <= 308), column indices with more than
19 significant digits, col(N) whose argument is not an integer literal [0-9]+(.0*)? (optionally wrapped in unary
'+' and parentheses), quoted values, op or key names with a backslash inside quotes other than an escaped quote
(never canonical), and keys holding an odd number of '"' (Rust's key is the atom text before the first '=', so
such a key ends inside a quote region that the value closes; the tokenizer still round-trips them losslessly and
the vocabulary builder STOPs on one in train_fit).

MASKS. allowed(s) evaluates a small candidate set per phase (advance + completion frontier) once per structural
key (the state minus `used`) and caches one cost row per frontier vector; a token is in the mask when any of its
rows satisfies `cost <= caps - used`, and masks are bucketed by min(remaining, max finite cost) per resource
(mask_key). compile_masks() turns teacher sequences into per-position mask ids over a deduplicated bool[M, V]
table; mask i is the one token i was chosen under (mask 0 = {BOS}).

ARTIFACTS. build_signatures(train_fit dsls) -> signatures.json (jope.dl-dsl-signatures/v1, vocabulary
independent); build_grammar(signatures, vocab, max_seq) -> grammar.json (jope.dl-dsl-grammar/v1, bound to
vocab_sha256 and signatures_sha256). dltrain.dsl_tokenizer.build_artifacts writes both from train_fit only.

CLI (inside the lane slice):
  gate  --dict DIR --tok TOK [--partitions P ...]     POLICY on train_fit, SYNTAX on every partition
                                                       (writes automaton-gate.json next to the artifacts)
  walks --tok TOK [--n N] [--seed S] [--mode policy|syntax] [--bias class|uniform]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np

from dltrain import common
from dltrain import dsl_tokenizer as T
from dltrain import oracle as _O

SIGNATURES_SCHEMA = "jope.dl-dsl-signatures/v1"
GRAMMAR_SCHEMA = "jope.dl-dsl-grammar/v1"
GATE_SCHEMA = "jope.dl-dsl-automaton-gate/v1"
INFINITE = 1 << 30                 # min_to_complete() of a state with no completion
_BIG = 1 << 40                     # an unbounded resource cap
_NRES = 6                          # tokens, bytes, lparen, winsor, stages, nodes
_ZERO = (0,) * _NRES

MATH_FUNCS = dict(_O.MATH_FUNCS)          # name -> arity (math_expr.rs:611-663)
MATH_VARS = frozenset(_O.MATH_VARS)       # variable names and aliases (math_expr.rs:591-600)
_WS_CHARS = tuple(_O.RUST_WHITESPACE)     # Rust char::is_whitespace (25 code points)
_WS_SET = frozenset(_WS_CHARS)
_WS_BYTES = tuple(sorted(c.encode("utf-8") for c in _WS_CHARS))
_PUNCT_BYTES = frozenset(b"+-*/^(),")
_DIGITS = frozenset(b"0123456789")
_ID_START = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")
_ID_CONT = _ID_START | _DIGITS
_COL_RE = re.compile(r"c[0-9]+")                     # tokenizer COL lexeme (raw, lowercase c)
_COLVAR_RE = re.compile(r"c(?:ol)?([0-9]+)")         # math_expr c<N> / col<N> variables (lowercased)
_WINSOR = "(winsor_clip"
SYM_MAX_DEPTH = _O.MATH_MAX_DEPTH         # 16
SYM_MAX_NODES = _O.MATH_MAX_NODES         # 64
COL_MAX_SIG_DIGITS = 19                   # < 1e19 < u64::MAX, so math_expr _parse_usize never overflows
NUM_FINITE_MAG = 308                      # sig_int_digits + exponent <= 308 proves a sym numeral finite

# Form codes (State.form)
F_INIT, F_START, F_PIPE, F_SYM, F_PORT, F_DONE = range(6)
FORM_NAMES = ("INIT", "START", "PIPE", "SYM", "PORT", "DONE")
# Pipe-frame phases
P_OPEN, P_OP, P_OPSP, P_KEYS, P_KEYSP, P_VAL, P_NUM, P_STR, P_SEND, P_END = range(10)
PIPE_PHASE_NAMES = ("PIPE_OPENED", "OP", "OP_SPELLING", "KEYS", "KEY_SPELLING", "VALUE", "NUMBER", "STRING",
                    "STAGE_CLOSED", "PIPE_CLOSED")
# Portfolio phases
Q_SEP, Q_MEMBER, Q_PIPE, Q_MCLOSE, Q_END = range(5)
# Sym phases
S_OPND, S_OPTR, S_NUM, S_COL, S_IDSP, S_FUNC, S_END = range(7)
# Op roles (POLICY): bit 1 final-capable, bit 2 non-final-capable
ROLE_FINAL, ROLE_NONFINAL = 1, 2


class Mode(str, Enum):
    SYNTAX = "syntax"
    POLICY = "policy"


@dataclass(frozen=True)
class Caps:
    """Source caps: evaluator (lambda-worker main.rs:191-216) and island admit_dsl (gp-island-engine
    main.rs:958-974); max_bytes applies to both modes, the others to POLICY only."""
    max_seq: int
    max_bytes: int = 32768
    pipe_stages: int = 10
    lparen_pipe: int = 11
    lparen_sym: int = 18
    lparen_portfolio: int = 32
    members: tuple = (2, 3)
    member_stages: int = 24
    winsor: int = 3
    sym_nodes: int = 64

    def to_json(self) -> dict:
        return {"max_seq": self.max_seq, "max_bytes": self.max_bytes, "pipe_stages": self.pipe_stages,
                "lparen_pipe": self.lparen_pipe, "lparen_sym": self.lparen_sym,
                "lparen_portfolio": self.lparen_portfolio, "members": list(self.members),
                "member_stages": self.member_stages, "winsor": self.winsor, "sym_nodes": self.sym_nodes}

    @classmethod
    def from_json(cls, doc: dict) -> "Caps":
        d = dict(doc)
        d["members"] = tuple(d["members"])
        return cls(**d)


@dataclass(frozen=True)
class Rules:
    """Generation-only decode-protocol flags (never part of an acceptance gate)."""
    feature_count: int | None = None
    target_lead: bool = False
    col_lt_p: bool = False

    def key(self) -> tuple:
        if self.col_lt_p and (type(self.feature_count) is not int or self.feature_count < 1):
            raise ValueError("col_lt_p needs feature_count >= 1")
        return (self.feature_count if self.col_lt_p else None, bool(self.target_lead), bool(self.col_lt_p))


class State(NamedTuple):
    """Immutable, hashable automaton state. `ctx` is the Grammar-interned (mode, rules); `used` the resource
    vector (tokens, bytes, lparen, winsor, stages, nodes); `form` one of F_*; `sub` the form's structure."""
    mode: str
    ctx: int
    used: tuple
    form: int
    sub: tuple


@dataclass(frozen=True)
class Verdict:
    ok: bool
    position: int | None
    reason: str | None


class GrammarError(ValueError):
    """The token is not allowed in this state (structure, uniqueness, caps or budget)."""


class AutomatonStop(T.TokenizerStop):
    """A P4 STOP: an input the signature builder cannot table (RAW, non-canonical structure) or inconsistent
    artifacts (vocabulary or signatures that do not belong together). A TokenizerStop, so the tokenizer's build
    CLI reports it as a STOP (exit 2) when it builds signatures.json and grammar.json."""


# ---------------------------------------------------------------------------------------------
# Small helpers. A completion cost is a FRONTIER: a sorted tuple of the Pareto-minimal resource vectors
# (tokens, bytes, lparen, winsor, stages, nodes) over every completion, or None when no completion exists.
# ---------------------------------------------------------------------------------------------
def _vadd(a, b) -> tuple:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4], a[5] + b[5])


def _leq(a, b) -> bool:
    """a <= b in every component."""
    return a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2] and a[3] <= b[3] and a[4] <= b[4] and a[5] <= b[5]


def _prune(vectors, cap=None):
    """The Pareto frontier of `vectors` (sorted, duplicate-free); vectors above `cap` in any component can never fit
    and are dropped. None when nothing remains."""
    if cap is not None:
        vectors = [v for v in vectors if _leq(v, cap)]
    if not vectors:
        return None
    if len(vectors) == 1:
        return (tuple(vectors[0]),)
    out = []
    for v in sorted(set(vectors)):          # a dominating vector sorts before the vectors it dominates
        if not any(_leq(w, v) for w in out):
            out.append(v)
    return tuple(out)


def _cadd(a, b):
    """Frontier of the concatenated completions (Minkowski sum); None absorbs."""
    if a is None or b is None:
        return None
    if len(a) == 1 and len(b) == 1:
        return (_vadd(a[0], b[0]),)
    return _prune([_vadd(x, y) for x in a for y in b])


def _cmin2(a, b):
    """Frontier of the union of two completion sets; None is the empty set."""
    if a is None:
        return b
    if b is None:
        return a
    return _prune(list(a) + list(b))


def _fits(used, cost, caps) -> bool:
    """One cost vector fits every cap."""
    return (used[0] + cost[0] <= caps[0] and used[1] + cost[1] <= caps[1] and used[2] + cost[2] <= caps[2]
            and used[3] + cost[3] <= caps[3] and used[4] + cost[4] <= caps[4] and used[5] + cost[5] <= caps[5])


def _ffits(used, frontier, caps) -> bool:
    """Some completion of the frontier fits every cap."""
    return frontier is not None and any(_fits(used, v, caps) for v in frontier)


def _utf8_status(buf: bytes) -> tuple:
    """(ok, complete): ok iff `buf` is a valid UTF-8 prefix (strict: no surrogates, no overlongs)."""
    try:
        buf.decode("utf-8")
        return True, True
    except UnicodeDecodeError as exc:
        if exc.reason == "unexpected end of data" and exc.end == len(buf):
            return True, False
        return False, False


def _winsor_count(text: str) -> int:
    return text.count(_WINSOR)


def _id_role(lower: str):
    """math_expr role of a lowercased identifier: ('F', arity) | ('K',) col(N) call | ('V', col index or -1) |
    None (the parser would refuse it)."""
    ar = MATH_FUNCS.get(lower)
    if ar is not None:
        return ("F", ar)
    if lower == "col":
        return ("K",)
    if lower in MATH_VARS:
        return ("V", -1)
    m = _COLVAR_RE.fullmatch(lower)
    if m is not None:
        sig = m.group(1).lstrip("0")
        if len(sig) <= COL_MAX_SIG_DIGITS:
            return ("V", int(m.group(1)))
    return None


_FIXED_ID_NAMES = tuple(sorted(set(MATH_FUNCS) | {"col"} | set(MATH_VARS)))


# ---------------------------------------------------------------------------------------------
# Signatures: tables from the DISTINCT train_fit programs only (vocabulary independent)
# ---------------------------------------------------------------------------------------------
_NUMERAL_RE = re.compile(r"([0-9]*)(\.?)([0-9]*)(?:([eE])([+-]?)([0-9]+))?")


def _numeral_shape(text: str) -> dict:
    """Shape features of a math_expr numeral lexeme (no sign)."""
    m = _NUMERAL_RE.fullmatch(text)
    if m is None or not (m.group(1) or m.group(3)):
        raise AutomatonStop(f"sym numeral {text!r} has no Rust f64 shape")
    ip, dot, fp, emark, esign, ed = m.groups()
    mant = (ip + fp).lstrip("0")
    return {"int_digits": len(ip), "dot": bool(dot), "frac_digits": len(fp),
            "int_leading_zero": len(ip) > 1 and ip[0] == "0", "frac_last_zero": bool(fp) and fp[-1] == "0",
            "exp": emark is not None, "exp_mark": emark or "", "exp_sign": esign or "",
            "exp_digits": len(ed or ""), "exp_leading_zero": bool(ed) and len(ed) > 1 and ed[0] == "0",
            "sig": len(mant)}


def _pipe_lexemes(lx: list, i: int) -> tuple:
    """Stages of the pipe whose PIPE_OPEN is lx[i]: ([(op, [(key, kind, text)])], index after its CLOSE)."""
    if lx[i].kind != "PIPE_OPEN":
        raise AutomatonStop("expected PIPE_OPEN")
    i += 1
    stages = []
    while lx[i].kind == "STAGE_OPEN":
        op = lx[i + 1].text
        i += 2
        params = []
        while lx[i].kind == "KEY":
            params.append((lx[i].name, lx[i + 1].kind, lx[i + 1].text))
            i += 2
        if lx[i].kind != "CLOSE":
            raise AutomatonStop("malformed stage")
        stages.append((op, params))
        i += 1
    if not stages or lx[i].kind != "CLOSE":
        raise AutomatonStop("malformed pipe")
    return stages, i + 1


class _SigBuilder:
    def __init__(self) -> None:
        self.programs = 0
        self.forms = Counter()
        self.ops = {}
        self.keysets = {}
        self.values = {}
        self.stages_max = 0
        self.port = {"members": Counter(), "blend": Counter(), "member_stages_max": 0, "sum_stages_max": 0}
        self.sym = {"programs": 0, "nodes_max": 0, "depth_max": 0, "unary_plus": 0, "unary_minus": 0,
                    "binary": Counter(), "groups": 0, "calls": Counter(), "col_calls": 0, "ws": Counter(),
                    "ids": Counter()}
        self.num = {"count": 0, "int_only": 0, "dot_frac": 0, "trailing_dot": 0, "leading_dot": 0,
                    "max_int_digits": 0, "int_leading_zero": 0, "max_frac_digits": 0, "frac_last_zero": 0,
                    "exp": 0, "exp_marks": Counter(), "exp_signs": Counter(), "max_exp_digits": 0,
                    "exp_leading_zero": 0, "max_sig": 0}
        self.col = {"count": 0, "max_digits": 0, "leading_zero": 0}
        self.lparen = {"pipe": 0, "sym": 0, "portfolio": 0}
        self.winsor_max = 0
        self.bytes_max = 0

    # ---- pipes -------------------------------------------------------------------------------
    def _value(self, op: str, key: str, kind: str, text: str) -> None:
        v = self.values.setdefault(op, {}).setdefault(key, {"int": None, "float": None, "true": 0, "false": 0,
                                                             "cat": Counter(), "str": Counter()})
        if kind == "TRUE":
            v["true"] += 1
        elif kind == "FALSE":
            v["false"] += 1
        elif kind == "NUM":
            neg = text.startswith("-")
            body = text[1:] if neg else text
            if "." in body:
                ip, fp = body.split(".")
                f = v["float"] or {"count": 0, "neg": 0, "max_int_digits": 0, "max_frac_digits": 0}
                f["count"] += 1
                f["neg"] += neg
                f["max_int_digits"] = max(f["max_int_digits"], len(ip))
                f["max_frac_digits"] = max(f["max_frac_digits"], len(fp))
                v["float"] = f
            else:
                n = v["int"] or {"count": 0, "neg": 0, "max_digits": 0}
                n["count"] += 1
                n["neg"] += neg
                n["max_digits"] = max(n["max_digits"], len(body))
                v["int"] = n
        elif kind == "VALUE":
            (v["cat"] if T.is_cat_name(text) else v["str"])[text] += 1
        else:
            raise AutomatonStop(f"unexpected value lexeme {kind}")

    def _pipe(self, stages: list, member: bool) -> None:
        self.stages_max = max(self.stages_max, len(stages)) if not member else self.stages_max
        last = len(stages) - 1
        for j, (op, params) in enumerate(stages):
            rec = self.ops.setdefault(op, {"final": 0, "nonfinal": 0, "member": 0, "occurrences": 0})
            rec["occurrences"] += 1
            rec["final" if j == last else "nonfinal"] += 1
            rec["member"] += member
            keys = tuple(k for k, _kind, _t in params)
            enc = [k.encode("utf-8") for k in keys]
            if any(a >= b for a, b in zip(enc, enc[1:])):
                raise AutomatonStop(f"stage {op!r}: keys are not strictly increasing (non-canonical text)")
            self.keysets.setdefault(op, set()).add(keys)
            for key, kind, text in params:
                self._value(op, key, kind, text)

    # ---- sym ---------------------------------------------------------------------------------
    def _numeral(self, text: str) -> None:
        f = _numeral_shape(text)
        n = self.num
        n["count"] += 1
        if not f["dot"]:
            n["int_only"] += 1
        elif f["int_digits"] and f["frac_digits"]:
            n["dot_frac"] += 1
        elif f["int_digits"]:
            n["trailing_dot"] += 1
        else:
            n["leading_dot"] += 1
        n["max_int_digits"] = max(n["max_int_digits"], f["int_digits"])
        n["int_leading_zero"] += f["int_leading_zero"]
        n["max_frac_digits"] = max(n["max_frac_digits"], f["frac_digits"])
        n["frac_last_zero"] += f["frac_last_zero"]
        if f["exp"]:
            n["exp"] += 1
            n["exp_marks"][f["exp_mark"]] += 1
            n["exp_signs"][f["exp_sign"] or "none"] += 1
            n["max_exp_digits"] = max(n["max_exp_digits"], f["exp_digits"])
            n["exp_leading_zero"] += f["exp_leading_zero"]
        n["max_sig"] = max(n["max_sig"], f["sig"])

    def _sym(self, dsl: str, lx: list) -> None:
        s = self.sym
        s["programs"] += 1
        body_lx = [x for x in lx[1:-1]]
        info = _O.parse_math(dsl[lx[0].end:lx[-1].start])
        s["nodes_max"] = max(s["nodes_max"], info.nodes)
        s["depth_max"] = max(s["depth_max"], info.depth)
        operand = True          # an operand is expected
        prev = None             # previous non-WS lexeme
        stack = []              # True for a col() call frame
        for x in body_lx:
            k, t = x.kind, x.text
            if k == "WS":
                s["ws"]["U+%04X" % ord(t)] += 1
                continue
            if k == "NUMERAL":
                if not (stack and stack[-1]):
                    self._numeral(t)
                operand = False
            elif k == "COL":
                d = t[1:]
                self.col["count"] += 1
                self.col["max_digits"] = max(self.col["max_digits"], len(d))
                self.col["leading_zero"] += len(d) > 1 and d[0] == "0"
                operand = False
            elif k == "IDENT":
                s["ids"][t] += 1
                operand = False      # a function name is followed by '(' (handled there)
            elif t == "(":
                if prev is not None and prev.kind == "IDENT":
                    low = prev.text.lower()
                    if low == "col":
                        s["col_calls"] += 1
                        stack.append(True)
                    else:
                        s["calls"][low] += 1
                        stack.append(False)
                else:
                    s["groups"] += 1
                    stack.append(stack[-1] if stack else False)
                operand = True
            elif t == ")":
                if stack:
                    stack.pop()
                operand = False
            elif t == ",":
                operand = True
            elif t in "+-":
                if operand:
                    s["unary_plus" if t == "+" else "unary_minus"] += 1
                else:
                    s["binary"][t] += 1
                    operand = True
            else:                    # * / ^
                s["binary"][t] += 1
                operand = True
            prev = x

    # ---- one program -------------------------------------------------------------------------
    def add(self, dsl: str) -> None:
        lx = T.lex(dsl)
        head = lx[0].kind
        if head == "RAW":
            raise AutomatonStop(f"train_fit program sha256={_O.program_sha256(dsl)} matches no form (RAW)")
        self.programs += 1
        self.bytes_max = max(self.bytes_max, len(dsl.encode("utf-8")))
        self.winsor_max = max(self.winsor_max, _winsor_count(dsl))
        lp = dsl.count("(")
        if head == "PIPE_OPEN":
            self.forms["pipe"] += 1
            self.lparen["pipe"] = max(self.lparen["pipe"], lp)
            stages, end = _pipe_lexemes(lx, 0)
            if end != len(lx):
                raise AutomatonStop("trailing lexemes after a pipe")
            self._pipe(stages, False)
        elif head == "PORTFOLIO_OPEN":
            self.forms["portfolio"] += 1
            self.lparen["portfolio"] = max(self.lparen["portfolio"], lp)
            i = 1
            members = []
            while lx[i].kind == "MEMBER_OPEN":
                stages, i = _pipe_lexemes(lx, i + 1)
                if lx[i].kind != "CLOSE":
                    raise AutomatonStop("malformed member")
                i += 1
                members.append(stages)
            blend = lx[i].kind
            if blend not in ("BLEND_AUTO", "BLEND_ONE") or i + 1 != len(lx) or not members:
                raise AutomatonStop("malformed portfolio")
            for stages in members:
                self._pipe(stages, True)
            p = self.port
            p["members"][str(len(members))] += 1
            p["blend"][blend] += 1
            p["member_stages_max"] = max(p["member_stages_max"], max(len(m) for m in members))
            p["sum_stages_max"] = max(p["sum_stages_max"], sum(len(m) for m in members))
        else:
            self.forms["sym_affine" if head == "SYM_AFFINE_OPEN" else "sym_raw"] += 1
            self.lparen["sym"] = max(self.lparen["sym"], lp)
            self._sym(dsl, lx)

    def doc(self) -> dict:
        def counts(c):
            return {k: c[k] for k in sorted(c)}

        values = {}
        for op in sorted(self.values):
            values[op] = {}
            for key in sorted(self.values[op]):
                v = self.values[op][key]
                values[op][key] = {"int": v["int"], "float": v["float"], "true": v["true"], "false": v["false"],
                                   "cat": counts(v["cat"]), "str": counts(v["str"])}
        sym = dict(self.sym)
        for k in ("binary", "calls", "ws", "ids"):
            sym[k] = counts(sym[k])
        num = dict(self.num)
        num["exp_marks"] = counts(num["exp_marks"])
        num["exp_signs"] = counts(num["exp_signs"])
        sym["numeral"] = num
        sym["col"] = dict(self.col)
        port = dict(self.port)
        port["members"] = counts(port["members"])
        port["blend"] = counts(port["blend"])
        return {"schema": SIGNATURES_SCHEMA, "programs": self.programs,
                "source": "distinct train_fit programs (program-dict/train_fit) only",
                "forms": {f: self.forms.get(f, 0) for f in ("pipe", "sym_affine", "sym_raw", "portfolio")},
                "ops": {op: dict(self.ops[op]) for op in sorted(self.ops)},
                "keysets": {op: [list(ks) for ks in sorted(self.keysets[op], key=lambda ks: [k.encode("utf-8") for k in ks])]
                            for op in sorted(self.keysets)},
                "values": values, "pipe": {"stages_max": self.stages_max}, "portfolio": port, "sym": sym,
                "observed": {"lparen_max": dict(self.lparen), "winsor_max": self.winsor_max,
                             "bytes_max": self.bytes_max},
                "counts_unit": "occurrences over the distinct train_fit programs"}


def build_signatures(train_fit_dsls: Iterable[str]) -> dict:
    """Signature tables (signatures.json) from the DISTINCT train_fit programs only: forms, op roles, per-op
    keyset sets, per-(op,key) value tables, sym features and portfolio shapes. Order never matters: every
    table is sorted. A RAW or structurally non-canonical program is a STOP (AutomatonStop)."""
    b = _SigBuilder()
    for dsl in train_fit_dsls:
        b.add(dsl)
    return b.doc()


# ---------------------------------------------------------------------------------------------
# Grammar artifact: the signature tables bound to vocabulary ids (grammar.json)
# ---------------------------------------------------------------------------------------------
SYNTAX_NUM = {"int_digits": 19, "float_int_digits": 309, "frac_digits": 340, "sig_digits": 17}


def signatures_sha256(signatures: dict) -> str:
    return common.sha256_bytes(common.canonical(signatures))


def _need_id(vocab, cls: str, name: str) -> int:
    i = vocab.id(cls, name)
    if i is None:
        raise AutomatonStop(f"signatures name {cls}:{name!r} is not in the vocabulary (artifacts do not belong together)")
    return i


def build_grammar(signatures: dict, vocab, max_seq: int, caps: Caps | None = None) -> dict:
    """grammar.json: the POLICY tables of `signatures` bound to `vocab` ids, the caps and max_seq. Pure function of
    its inputs (deterministic bytes); refuses signatures and a vocabulary that do not belong together."""
    if not isinstance(signatures, dict) or signatures.get("schema") != SIGNATURES_SCHEMA:
        raise AutomatonStop(f"not a {SIGNATURES_SCHEMA} document")
    if type(max_seq) is not int or not 4 <= max_seq <= T.MAX_SEQ_CAP:
        raise AutomatonStop(f"max_seq must be an int in 4..{T.MAX_SEQ_CAP}, got {max_seq!r}")
    caps = caps or Caps(max_seq=max_seq)
    if caps.max_seq != max_seq:
        raise ValueError("caps.max_seq != max_seq")
    forms = signatures["forms"]
    opener = []
    if forms.get("pipe"):
        opener.append("PIPE_OPEN")
    if forms.get("sym_affine"):
        opener.append("SYM_AFFINE_OPEN")
    if forms.get("sym_raw"):
        opener.append("SYM_RAW_OPEN")
    if forms.get("portfolio"):
        opener.append("PORTFOLIO_OPEN")
    ops = []
    for name, rec in signatures["ops"].items():
        role = (ROLE_FINAL if rec["final"] else 0) | (ROLE_NONFINAL if rec["nonfinal"] else 0)
        ops.append({"id": _need_id(vocab, "OP", name), "name": name, "role": role})
    ops.sort(key=lambda r: r["id"])
    profiles, prof_index = [], {}

    def profile(v: dict) -> int:
        i, f = v["int"], v["float"]
        doc = {"int_ok": i is not None, "int_neg": bool(i and i["neg"]), "a_int": i["max_digits"] if i else 0,
               "float_ok": f is not None, "float_neg": bool(f and f["neg"]),
               "a_float": f["max_int_digits"] if f else 0, "b": f["max_frac_digits"] if f else 0,
               "true": v["true"] > 0, "false": v["false"] > 0,
               "cat": sorted(_need_id(vocab, "CAT", c) for c in v["cat"]),
               "str": sorted(v["str"], key=lambda s: s.encode("utf-8"))}
        for s in doc["str"]:
            if not T.spelling_allowed("STR", s, vocab):
                raise AutomatonStop(f"observed STR literal {s!r} would never be spelled by the tokenizer")
        key = common.canonical(doc)
        if key not in prof_index:
            prof_index[key] = len(profiles)
            profiles.append(doc)
        return prof_index[key]

    nodes, roots = [], {}
    for op in sorted(signatures["keysets"], key=lambda n: n.encode("utf-8")):
        op_id = _need_id(vocab, "OP", op)
        vals = signatures["values"].get(op, {})
        root = len(nodes)
        roots[str(op_id)] = root
        nodes.append({"op": op_id, "terminal": False, "edges": {}})
        for ks in signatures["keysets"][op]:
            n = root
            for k in ks:
                kid = _need_id(vocab, "KEY", k)
                e = nodes[n]["edges"].get(kid)
                if e is None:
                    e = [len(nodes), profile(vals[k])]
                    nodes[n]["edges"][kid] = e
                    nodes.append({"op": op_id, "terminal": False, "edges": {}})
                n = e[0]
            nodes[n]["terminal"] = True
    trie = [{"op": nd["op"], "terminal": nd["terminal"],
             "edges": [[kid, e[0], e[1]] for kid, e in sorted(nd["edges"].items())]} for nd in nodes]
    sym = signatures["sym"]
    num = sym["numeral"]
    ws = []
    for code in sym["ws"]:
        ch = chr(int(code[2:], 16))
        if ch not in _WS_SET:
            raise AutomatonStop(f"observed sym whitespace {code} is not Rust whitespace")
        ws.append(ch.encode("utf-8").hex())
    ids = []
    for name in sym["ids"]:
        tid = _need_id(vocab, "ID", name)
        if _id_role(name.lower()) is None:
            raise AutomatonStop(f"observed sym identifier {name!r} is neither a variable nor a function")
        ids.append(tid)
    port = signatures["portfolio"]
    members = sorted(int(k) for k in port["members"] if caps.members[0] <= int(k) <= caps.members[1])
    policy = {
        "forms": opener,
        "ops": ops,
        "op_roots": {k: roots[k] for k in sorted(roots, key=int)},
        "trie": trie,
        "value_profiles": profiles,
        "sym": {
            "nodes_cap": min(caps.sym_nodes, SYM_MAX_NODES, int(sym["nodes_max"])),
            "unary_plus": sym["unary_plus"] > 0, "unary_minus": sym["unary_minus"] > 0,
            "binary": sorted(sym["binary"]), "groups": sym["groups"] > 0, "col_calls": sym["col_calls"] > 0,
            "ws_utf8_hex": sorted(ws), "ids": sorted(ids),
            "col": {"ok": sym["col"]["count"] > 0, "max_digits": int(sym["col"]["max_digits"]),
                    "leading_zero": sym["col"]["leading_zero"] > 0},
            "numeral": {"ok": num["count"] > 0, "int_only": num["int_only"] > 0, "dot_frac": num["dot_frac"] > 0,
                        "trailing_dot": num["trailing_dot"] > 0, "leading_dot": num["leading_dot"] > 0,
                        "max_int_digits": int(num["max_int_digits"]), "int_leading_zero": num["int_leading_zero"] > 0,
                        "max_frac_digits": int(num["max_frac_digits"]), "frac_last_zero": num["frac_last_zero"] > 0,
                        "exp": num["exp"] > 0, "exp_marks": "".join(sorted(num["exp_marks"])),
                        "exp_signs": "".join(sorted("n" if s == "none" else s for s in num["exp_signs"])),
                        "max_exp_digits": int(num["max_exp_digits"]), "exp_leading_zero": num["exp_leading_zero"] > 0,
                        "max_sig": int(num["max_sig"])},
        },
        "portfolio": {"members": members, "blend": sorted(port["blend"])},
    }
    return {"schema": GRAMMAR_SCHEMA, "vocab_sha256": vocab.sha256, "vocab_size": vocab.size,
            "signatures_sha256": signatures_sha256(signatures), "max_seq": max_seq, "caps": caps.to_json(),
            "policy": policy,
            "syntax": {"numeric": dict(SYNTAX_NUM), "portfolio_members": [1, 12], "sym_nodes": SYM_MAX_NODES,
                       "sym_depth": SYM_MAX_DEPTH, "col_max_sig_digits": COL_MAX_SIG_DIGITS,
                       "sym_numeral_finite_magnitude": NUM_FINITE_MAG},
            "rules": {"target_lead": "generation-only", "col_lt_p": "generation-only"},
            "mask_alignment": "masks[i] is the mask token i was chosen under; masks[0] = {BOS}"}


# ---------------------------------------------------------------------------------------------
# Runtime grammar
# ---------------------------------------------------------------------------------------------
_FX = T.FIXED_IDS
BOS, EOS = _FX["BOS"], _FX["EOS"]
OP_BEGIN, OP_END, KEY_BEGIN, KEY_END = _FX["OP_BEGIN"], _FX["OP_END"], _FX["KEY_BEGIN"], _FX["KEY_END"]
STR_BEGIN, STR_END, ID_BEGIN, ID_END = _FX["STR_BEGIN"], _FX["STR_END"], _FX["ID_BEGIN"], _FX["ID_END"]
PIPE_OPEN, STAGE_OPEN, CLOSE = _FX["PIPE_OPEN"], _FX["STAGE_OPEN"], _FX["CLOSE"]
SYM_AFFINE_OPEN, SYM_RAW_OPEN, SYM_CLOSE = _FX["SYM_AFFINE_OPEN"], _FX["SYM_RAW_OPEN"], _FX["SYM_CLOSE"]
PORTFOLIO_OPEN, MEMBER_OPEN, BLEND_AUTO, BLEND_ONE = (_FX["PORTFOLIO_OPEN"], _FX["MEMBER_OPEN"], _FX["BLEND_AUTO"],
                                                     _FX["BLEND_ONE"])
TRUE, FALSE, COL = _FX["TRUE"], _FX["FALSE"], _FX["COL"]
BYTE_BASE, DATA_BASE = T.BYTE_BASE, T.DATA_BASE
_OPENERS = (PIPE_OPEN, SYM_AFFINE_OPEN, SYM_RAW_OPEN, PORTFOLIO_OPEN)
# BFS alphabet for spelled-name completions (representatives of every transition type).
_REP_BYTES = (0x78, 0x71, 0x7A, 0x21, 0x20, 0x22, 0x5C, 0x80, 0x90, 0xA0, 0xC2)
_SPELL_BFS_LIMIT = 40000
_ASCII_CHARS = tuple(chr(i) for i in range(128))
_BYTE1 = tuple(bytes((i,)) for i in range(256))
_CMEMO_MAX, _L1_MAX, _L2_MAX, _SMALL_MEMO_MAX = 1_000_000, 150_000, 1_000_000, 250_000
_BINOP_BYTES = frozenset(b"+-*/^")
RESOURCE_NAMES = ("max_seq", "max_bytes", "lparen", "winsor", "stages", "sym_nodes")
_SYM_BYTE_SET = sorted(set(b for w in _WS_BYTES for b in w) | _PUNCT_BYTES | _DIGITS | _ID_CONT | set(b".eE+-"))


class Grammar:
    """The two-mode automaton over one vocabulary (see the module docstring)."""

    def __init__(self, grammar: dict, signatures: dict, vocab) -> None:
        if not isinstance(grammar, dict) or grammar.get("schema") != GRAMMAR_SCHEMA:
            raise AutomatonStop(f"not a {GRAMMAR_SCHEMA} document")
        if grammar.get("vocab_sha256") != vocab.sha256:
            raise AutomatonStop("grammar.json is bound to another vocabulary")
        if grammar.get("signatures_sha256") != signatures_sha256(signatures):
            raise AutomatonStop("grammar.json is bound to other signatures")
        caps = Caps.from_json(grammar["caps"])
        rebuilt = build_grammar(signatures, vocab, int(grammar["max_seq"]), caps)
        if common.canonical(rebuilt) != common.canonical(grammar):
            raise AutomatonStop("grammar.json differs from build_grammar(signatures, vocab, max_seq)")
        self.doc = grammar
        self.vocab = vocab
        self.V = vocab.size
        self.max_seq = int(grammar["max_seq"])
        self.caps = caps
        V = self.V
        pieces = [vocab.piece(i) for i in range(V)]
        self.plen = [len(p) for p in pieces]
        self.plpar = [p.count(b"(") for p in pieces]
        self.r_op = vocab.class_range("OP")
        self.r_key = vocab.class_range("KEY")
        self.r_cat = vocab.class_range("CAT")
        self.r_id = vocab.class_range("ID")
        self.op_name = {i: vocab.name(i) for i in range(*self.r_op)}
        self.pwin = [0] * V
        for i in range(V):
            n = pieces[i].count(_WINSOR.encode())
            if self.r_op[0] <= i < self.r_op[1] and pieces[i].startswith(b"winsor_clip"):
                n += 1
            self.pwin[i] = n
        self.key_bytes = {i: vocab.name(i).encode("utf-8") for i in range(*self.r_key)}
        self.id_role = {i: _id_role(vocab.name(i).lower()) for i in range(*self.r_id)}
        self._names = {c: frozenset(vocab.names(c)) for c in T.DATA_CLASSES}
        pol = grammar["policy"]
        self.p_forms = frozenset(_FX[n] for n in pol["forms"])
        self.p_role = {r["id"]: r["role"] for r in pol["ops"]}
        self.p_root = {int(k): v for k, v in pol["op_roots"].items()}
        self.p_nodes = [(nd["op"], bool(nd["terminal"]), {e[0]: (e[1], e[2]) for e in nd["edges"]})
                        for nd in pol["trie"]]
        self.p_prof = [self._value_profile(p) for p in pol["value_profiles"]]
        ps = pol["sym"]
        self.ps = ps
        self.ps_ws = frozenset(bytes.fromhex(h) for h in ps["ws_utf8_hex"])
        self._ws_tab = {}
        for pol_flag, full in ((False, frozenset(_WS_BYTES)), (True, self.ps_ws)):
            rest = {}
            for w in full:
                for i in range(1, len(w)):
                    rest[w[:i]] = min(rest.get(w[:i], 99), len(w) - i)
            self._ws_tab[pol_flag] = (full, frozenset(rest), rest)
        self.ps_ids = frozenset(ps["ids"])
        self.ps_binary = frozenset(ps["binary"])
        self.p_members = tuple(pol["portfolio"]["members"])
        self.p_blend = frozenset(_FX[n] for n in pol["portfolio"]["blend"])
        nm = ps["numeral"]
        self.num_policy = ("P", nm["ok"], nm["int_only"], nm["dot_frac"], nm["trailing_dot"], nm["leading_dot"],
                           nm["max_int_digits"], nm["int_leading_zero"], nm["max_frac_digits"], nm["frac_last_zero"],
                           nm["exp"], nm["exp_marks"], nm["exp_signs"], nm["max_exp_digits"], nm["exp_leading_zero"],
                           nm["max_sig"])
        self.num_syntax = ("S", True, True, True, True, True, 999, True, 999, True, True, "Ee", "+-n", 999, True, 999)
        self.num_intlit = ("I", True, True, True, True, False, 999, True, 999, True, False, "", "", 0, False, 999)
        self.col_policy = (ps["col"]["ok"], ps["col"]["max_digits"], ps["col"]["leading_zero"])
        self.col_syntax = (True, 999, True)
        self.syn_prof = (True, True, SYNTAX_NUM["int_digits"], True, True, SYNTAX_NUM["float_int_digits"],
                         SYNTAX_NUM["frac_digits"], True, True, None, None)
        # candidate token sets per phase class
        rng = np.arange
        bytes_all = rng(BYTE_BASE, DATA_BASE)
        self.c_op = np.concatenate([rng(*self.r_op), [OP_BEGIN]]).astype(np.int64)
        self.c_keys = np.concatenate([rng(*self.r_key), [KEY_BEGIN, CLOSE]]).astype(np.int64)
        numb = [BYTE_BASE + b for b in b"-.0123456789"]
        self.c_val = np.concatenate([numb, [TRUE, FALSE], rng(*self.r_cat), [STR_BEGIN]]).astype(np.int64)
        self.c_num = np.concatenate([numb, rng(*self.r_key), [KEY_BEGIN, CLOSE]]).astype(np.int64)
        self.c_spell = {OP_END: np.concatenate([bytes_all, [OP_END]]).astype(np.int64),
                        KEY_END: np.concatenate([bytes_all, [KEY_END]]).astype(np.int64),
                        STR_END: np.concatenate([bytes_all, [STR_END]]).astype(np.int64)}
        self.c_sym = np.concatenate([[BYTE_BASE + b for b in _SYM_BYTE_SET], rng(*self.r_id),
                                     [ID_BEGIN, ID_END, COL, SYM_CLOSE]]).astype(np.int64)
        self.c_fixed = {k: np.array(v, dtype=np.int64) for k, v in {
            "init": [BOS], "start": list(_OPENERS), "open": [STAGE_OPEN], "send": [STAGE_OPEN, CLOSE],
            "eos": [EOS], "close": [CLOSE], "sep": [MEMBER_OPEN, BLEND_AUTO, BLEND_ONE], "member": [PIPE_OPEN],
            "none": []}.items()}
        # caches
        self._ctx_index = {}
        self._ctxs = []
        self._cmemo = {}
        self._l1 = {}
        self._l2 = {}
        self._masks = []
        self._mask_index = {}
        self._spell_memo = {}
        self._special_prefixes = {}
        for cls, extra in (("OP", ("portfolio", "target", "input")), ("KEY", ())):
            pre = set()
            for name in tuple(vocab.names(cls)) + extra:
                nb = name.encode("utf-8")
                pre.update(nb[:i] for i in range(len(nb) + 1))
            self._special_prefixes[cls] = frozenset(pre)
        self._num_memo = {}
        self._stage_memo = {}
        self._idc_memo = {}
        self._spell_limited = set()        # spelled-name memo keys whose search hit its limit (not proven dead)
        c = caps
        # Per mode, the largest cap any form applies: a frontier vector above it can never fit and is dropped.
        # SYNTAX leaves '(' / winsor / stages uncapped, so those components are projected out of its frontiers.
        self._maxcap = {True: (self.max_seq, c.max_bytes, max(c.lparen_pipe, c.lparen_sym, c.lparen_portfolio),
                               c.winsor, max(c.pipe_stages, c.member_stages), int(ps["nodes_cap"])),
                        False: (self.max_seq, c.max_bytes, 0, 0, 0, SYM_MAX_NODES)}
        self._kmin_tab = self._trie_minima()

    @classmethod
    def load(cls, grammar, signatures, vocab) -> "Grammar":
        """grammar / signatures: a path to the JSON artifact or the parsed dict; vocab: dsl_tokenizer.Vocab."""
        if not isinstance(grammar, dict):
            grammar = json.loads(Path(grammar).read_bytes())
        if not isinstance(signatures, dict):
            signatures = json.loads(Path(signatures).read_bytes())
        return cls(grammar, signatures, vocab)

    # ---- context ------------------------------------------------------------------------------
    def _ctx(self, mode, rules: Rules) -> int:
        mode = Mode(mode)
        key = (mode is Mode.POLICY, rules.key())
        i = self._ctx_index.get(key)
        if i is None:
            i = self._ctx_index[key] = len(self._ctxs)
            self._ctxs.append(key)
        return i

    def start(self, mode, rules: Rules = Rules()) -> State:
        """The state before BOS (its only allowed token is BOS)."""
        mode = Mode(mode)
        return State(mode.value, self._ctx(mode, rules), _ZERO, F_INIT, ())

    def _caps(self, ctx: int, form: int) -> tuple:
        pol = self._ctxs[ctx][0]
        c = self.caps
        if not pol:
            return (self.max_seq, c.max_bytes, _BIG, _BIG, _BIG, SYM_MAX_NODES)
        if form == F_PIPE:
            return (self.max_seq, c.max_bytes, c.lparen_pipe, c.winsor, c.pipe_stages, _BIG)
        if form == F_SYM:
            return (self.max_seq, c.max_bytes, c.lparen_sym, c.winsor, _BIG, self.ps["nodes_cap"])
        if form == F_PORT:
            return (self.max_seq, c.max_bytes, c.lparen_portfolio, c.winsor, c.member_stages, _BIG)
        return (self.max_seq, c.max_bytes, _BIG, c.winsor, _BIG, _BIG)

    def _lead0(self, ctx: int) -> int:
        return 0 if self._ctxs[ctx][1][1] else 2

    @staticmethod
    def _value_profile(p: dict) -> tuple:
        return (p["int_ok"], p["int_neg"], p["a_int"], p["float_ok"], p["float_neg"], p["a_float"], p["b"],
                p["true"], p["false"], frozenset(p["cat"]), tuple(s.encode("utf-8") for s in p["str"]))

    def _tokcost(self, t: int) -> tuple:
        return (1, self.plen[t], self.plpar[t], self.pwin[t], 0, 0)

    def _tokf(self, t: int) -> tuple:
        """The one-token frontier of t."""
        return ((1, self.plen[t], self.plpar[t], self.pwin[t], 0, 0),)

    def _clip(self, ctx: int, f):
        """A frontier for `ctx`: SYNTAX projects out the resources it leaves uncapped ('(', winsor, stages), and
        vectors above every cap of the mode are dropped (they can never fit)."""
        if f is None:
            return None
        pol = self._ctxs[ctx][0]
        if len(f) == 1:                          # the common case: one vector, nothing to prune
            v = f[0] if pol else (f[0][0], f[0][1], 0, 0, 0, f[0][5])
            return (v,) if _leq(v, self._maxcap[pol]) else None
        if not pol:
            f = [(v[0], v[1], 0, 0, 0, v[5]) for v in f]
        return _prune(list(f), self._maxcap[pol])

    # ---- spelled names (SYNTAX only) ----------------------------------------------------------
    # A spelled state is (buf, scan): buf the bytes so far, scan an incremental scanner state.
    #   OP/KEY: (in_quote, pending utf-8 bytes, nonempty, order) -- order (KEY only): 0 on the last key's path,
    #           1 already greater (or no last key); a byte below the path is refused.
    #   STR:    (mode 0 empty | 1 bare | 2 quoted, closed, escape pending, special char seen, pending utf-8 bytes)
    #   ID:     ASCII [A-Za-z_][A-Za-z0-9_]*, scan = 0
    def _op_ok_name(self, ctx: int, text: str, member: bool, lead: int) -> bool:
        if member and text == "portfolio":
            return False
        if self._ctxs[ctx][1][1]:
            if text == "target" and lead != 0:
                return False
            if text == "input" and lead not in (0, 1):
                return False
        return True

    @staticmethod
    def _spell_init(kind: str, arg) -> tuple:
        if kind == "OP":
            return (b"", (False, b"", 0, 1))
        if kind == "KEY":
            return (b"", (False, b"", 0, 1 if arg is None else 0))
        if kind == "STR":
            return (b"", (0, False, False, False, b""))
        return (b"", 0)

    @staticmethod
    def _utf8_char(pend: bytes, b: int):
        """Feed one byte: ('char', str) | ('pend', bytes) | None (invalid UTF-8)."""
        if not pend:
            if b < 0x80:
                return ("char", _ASCII_CHARS[b])
            if 0xC2 <= b <= 0xF4:
                return ("pend", bytes((b,)))
            return None
        nb = pend + bytes((b,))
        ok, complete = _utf8_status(nb)
        if not ok:
            return None
        return ("char", nb.decode("utf-8")) if complete else ("pend", nb)

    def _spell_step(self, kind: str, st: tuple, b: int, arg):
        """The spelled state after one more byte, or None when no extension could be valid."""
        buf, sc = st
        nbuf = buf + bytes((b,))
        if kind == "ID":
            if b in (_ID_CONT if buf else _ID_START):
                return (nbuf, 0)
            return None
        if kind == "STR":
            mode, closed, esc, special, pend = sc
            if mode == 2 and closed:
                return None
            if mode == 0 and b == 0x22:
                return (nbuf, (2, False, False, False, b""))
            u = self._utf8_char(pend, b)
            if u is None:
                return None
            nmode = 1 if mode == 0 else mode
            if u[0] == "pend":
                return (nbuf, (nmode, closed, esc, special, u[1]))
            ch = u[1]
            if nmode == 1:
                if ch in _WS_SET or ch in '()"=':
                    return None
                return (nbuf, (1, False, False, False, b""))
            if esc:
                if ch != '"':
                    return None
                return (nbuf, (2, False, False, True, b""))
            if ch == "\\":
                return (nbuf, (2, False, True, special, b""))
            if ch == '"':
                return (nbuf, (2, True, False, special, b""))
            return (nbuf, (2, False, False, special or ch in _WS_SET or ch in "()=", b""))
        # OP / KEY
        inq, pend, nonempty, order = sc
        if kind == "KEY" and order == 0:
            i = len(buf)
            if i < len(arg):
                if b < arg[i]:
                    return None
                order = 1 if b > arg[i] else 0
            else:
                order = 1
        u = self._utf8_char(pend, b)
        if u is None:
            return None
        if u[0] == "pend":
            return (nbuf, (inq, u[1], nonempty, order))
        ch = u[1]
        if kind == "KEY" and ch == "=":
            return None
        if not nonempty and (ch in '()"' or ch in _WS_SET):
            return None
        if ch == '"':
            inq = not inq
        elif inq:
            if ch == "\\":
                return None
        elif ch == "(" or ch == ")" or ch in _WS_SET:
            return None
        return (nbuf, (inq, b"", 1, order))

    def _spell_end(self, ctx: int, kind: str, st: tuple, arg) -> bool:
        buf, sc = st
        if kind == "ID":
            return bool(buf) and self._id_end_role(ctx, buf.decode("ascii")) is not None
        if kind == "STR":
            mode, closed, esc, special, pend = sc
            if pend or mode == 0:
                return False
            if mode == 2:
                return closed and (special or len(buf) == 2)
            text = buf.decode("utf-8")
            if text in ("NaN", "inf", "-inf"):
                return True
            return T.is_cat_name(text) and text not in self._names["CAT"]
        inq, pend, nonempty, order = sc
        if inq or pend:
            return False
        text = buf.decode("utf-8")
        if kind == "OP":
            member, lead = arg
            return bool(nonempty) and text not in self._names["OP"] and self._op_ok_name(ctx, text, member, lead)
        return order == 1 and (arg is None or buf > arg) and text not in self._names["KEY"]

    def _id_end_role(self, ctx: int, text: str):
        if not text or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text) or _COL_RE.fullmatch(text):
            return None
        if text in self._names["ID"]:
            return None
        return self._id_role_ok(ctx, _id_role(text.lower()))

    def _id_role_ok(self, ctx: int, role):
        if role is None:
            return None
        rk = self._ctxs[ctx][1]
        if rk[2]:                     # col_lt_p
            if role[0] == "K":
                return None
            if role[0] == "V" and role[1] >= 0 and role[1] >= rk[0]:
                return None
        return role

    def _spell_memo_key(self, ctx: int, kind: str, st: tuple, arg) -> tuple:
        """An exact equivalence class of spelled states: two states with one key have the same completions."""
        buf0, sc0 = st
        if kind == "OP" and buf0 not in self._special_prefixes["OP"]:
            return ("OPa", sc0)                  # no extension can be a vocabulary or forbidden op name
        if kind == "KEY" and sc0[3] == 1 and buf0 not in self._special_prefixes["KEY"]:
            return ("KEYa", sc0)                 # already above the last key, no extension is a vocabulary key
        if kind == "STR" and sc0[0] == 2:
            # quoted strings never collide with names; what remains is whether the content is still empty
            # (`""` ends a canonical value, a closed `"a"` without a special char has no completion)
            return ("STRq", sc0, len(buf0) == (2 if sc0[1] else 1))
        return (ctx, kind, buf0, arg)

    def _spell_cands(self, kind: str, p: tuple, arg, reps: list) -> list:
        """Bytes to try after spelled state p. A key still on the previous key's path may only continue with that
        key's byte or a byte above it (any byte up to 0xFF: 0x7F and non-ASCII leads and continuations too)."""
        if kind == "KEY" and arg is not None and p[1][3] == 0 and len(p[0]) < len(arg):
            a = arg[len(p[0])]
            return list(range(a, 0x100))
        return reps

    def _spell_extra(self, ctx: int, kind: str, st: tuple, arg):
        """Minimal bytes to append before the class's *_END can be emitted (None: no completion found)."""
        mk = self._spell_memo_key(ctx, kind, st, arg)
        hit = self._spell_memo.get(mk, -1)
        if hit != -1:
            return hit
        res = None
        if self._spell_end(ctx, kind, st, arg):
            res = 0
        else:
            reps = list(_REP_BYTES) if kind != "ID" else list(b"xqXQ_0")
            # a key on the previous key's path may have to follow it to its end before going above it
            path = (len(arg) - len(st[0])) if (kind == "KEY" and arg is not None and st[1][3] == 0) else 0
            frontier = [st]
            seen = 0
            depth = 0
            while frontier and res is None and depth < 8 + max(path, 0) and seen <= _SPELL_BFS_LIMIT:
                depth += 1
                nxt = []
                for p in frontier:
                    for b in self._spell_cands(kind, p, arg, reps):
                        q = self._spell_step(kind, p, b, arg)
                        seen += 1
                        if q is None:
                            continue
                        if self._spell_end(ctx, kind, q, arg):
                            res = depth
                            break
                        nxt.append(q)
                    if res is not None:
                        break
                frontier = nxt
            if res is None and frontier:
                self._spell_limited.add(mk)      # the search stopped at its limit: not a proven dead end
        if len(self._spell_memo) >= _SMALL_MEMO_MAX:
            self._spell_memo.clear()
            self._spell_limited.clear()
        self._spell_memo[mk] = res
        return res

    # ---- pipe numeric values: -?(0|[1-9][0-9]*)(.[0-9]+)? with Rust Display rules -----------------
    # state (neg, int_digits, first_int_digit_zero, in_frac, frac_digits, last_frac_zero, frac_all_zero, sig)
    _PNUM0 = (0, 0, 0, 0, 0, 0, 1, 0)

    def _vprof(self, prof: int) -> tuple:
        return self.syn_prof if prof < 0 else self.p_prof[prof]

    @staticmethod
    def _pnum_step(vp: tuple, st: tuple, ch: int):
        neg, idig, izero, frac, fdig, flast0, fall0, sig = st
        int_ok, int_neg, a_int, float_ok, float_neg, a_float, b = vp[:7]
        if ch == 0x2D:                                   # '-'
            if idig or neg or frac or not ((int_ok and int_neg) or (float_ok and float_neg)):
                return None
            return (1, 0, 0, 0, 0, 0, 1, 0)
        if 0x30 <= ch <= 0x39:
            d = ch - 0x30
            nsig = sig + 1 if sig else (1 if d else 0)
            if nsig > 18:
                nsig = 18
            if not frac:
                if idig and izero:
                    return None                          # leading zero
                cap = max(a_int if (int_ok and (not neg or int_neg)) else 0,
                          a_float if (float_ok and (not neg or float_neg)) else 0)
                if idig + 1 > cap:
                    return None
                return (neg, idig + 1, izero if idig else (1 if d == 0 else 0), 0, 0, 0, 1, nsig)
            if fdig + 1 > b:
                return None
            return (neg, idig, izero, 1, fdig + 1, 1 if d == 0 else 0, fall0 if d == 0 else 0, nsig)
        if ch == 0x2E:                                   # '.'
            if frac or not idig or not float_ok or (neg and not float_neg) or idig > a_float:
                return None
            return (neg, idig, izero, 1, 0, 0, 1, sig)
        return None

    @staticmethod
    def _pnum_accept(vp: tuple, st: tuple) -> bool:
        neg, idig, izero, frac, fdig, flast0, fall0, sig = st
        int_ok, int_neg, a_int, float_ok, float_neg, a_float, b = vp[:7]
        if not frac:
            return bool(idig >= 1 and int_ok and idig <= a_int and (not neg or (int_neg and not izero)))
        return bool(fdig >= 1 and float_ok and (not neg or float_neg) and idig <= a_float and fdig <= b
                    and ((fdig == 1 and fall0) or not flast0) and (fall0 or sig <= 17))

    def _pnum_extra(self, prof: int, st: tuple):
        mk = ("p", prof, st)
        hit = self._num_memo.get(mk, -1)
        if hit != -1:
            return hit
        vp = self._vprof(prof)
        res = self._bfs(lambda x, c: self._pnum_step(vp, x, c), lambda x: self._pnum_accept(vp, x), st, b"01.-", 6)
        if len(self._num_memo) >= _SMALL_MEMO_MAX:
            self._num_memo.clear()
        self._num_memo[mk] = res
        return res

    @staticmethod
    def _bfs(step, accept, st, alphabet: bytes, limit: int):
        if accept(st):
            return 0
        frontier, seen = [st], {st}
        for depth in range(1, limit + 1):
            nxt = []
            for x in frontier:
                for ch in alphabet:
                    y = step(x, ch)
                    if y is None or y in seen:
                        continue
                    if accept(y):
                        return depth
                    seen.add(y)
                    nxt.append(y)
            if not nxt:
                return None
            frontier = nxt
        return None

    # ---- sym numerals: (D+(.D*)?|.D+)([eE][+-]?D+)? (math_expr lex), finite -------------------------
    # profile ("S"|"P"|"I", ok, int_only, dot_frac, trailing_dot, leading_dot, max_int, int_lead0, max_frac,
    #          frac_last0, exp, exp_marks, exp_signs, max_exp_digits, exp_lead0, max_sig)
    # state (part, dot, idig, izero, sint, fdig, flast0, nz, sig, emark, esign, edig, eval, ezero)
    def _numprof(self, ctx: int, const_only: bool) -> tuple:
        if const_only:
            return self.num_intlit
        return self.num_policy if self._ctxs[ctx][0] else self.num_syntax

    @staticmethod
    def _snum_start(npf: tuple, ch: int):
        if not npf[1]:
            return None
        if 0x30 <= ch <= 0x39:
            st = (0, 0, 0, 0, 0, 0, 0, 0, 0, "", "", 0, 0, 0)
            return Grammar._snum_step(npf, st, ch)
        if ch == 0x2E and npf[5]:                        # leading '.'
            return (1, 1, 0, 0, 0, 0, 0, 0, 0, "", "", 0, 0, 0)
        return None

    @staticmethod
    def _snum_step(npf: tuple, st: tuple, ch: int):
        (kind, _ok, int_only, dot_frac, trailing_dot, leading_dot, max_int, int_lead0, max_frac, frac_last0,
         exp_ok, exp_marks, exp_signs, max_exp, exp_lead0, max_sig) = npf
        part, dot, idig, izero, sint, fdig, flast0, nz, sig, emark, esign, edig, ev, ezero = st
        if 0x30 <= ch <= 0x39:
            d = ch - 0x30
            if part == 0:
                if idig and izero and not int_lead0:
                    return None
                if idig + 1 > max_int:
                    return None
                nsig = sig + 1 if sig else (1 if d else 0)
                if nsig > max_sig:
                    return None
                nsint = sint + 1 if (sint or d) else 0
                return (0, 0, idig + 1, izero if idig else (1 if d == 0 else 0), min(nsint, 999), 0, 0,
                        nz or (1 if d else 0), min(nsig, 999), "", "", 0, 0, 0)
            if part == 1:
                if kind == "I" and d:
                    return None
                if fdig + 1 > max_frac:
                    return None
                nsig = sig + 1 if sig else (1 if d else 0)
                if nsig > max_sig:
                    return None
                return (1, 1, idig, izero, sint, fdig + 1, 1 if d == 0 else 0, nz or (1 if d else 0),
                        min(nsig, 999), "", "", 0, 0, 0)
            # exponent digits
            if edig + 1 > max_exp:
                return None
            if edig and ezero and not exp_lead0:
                return None
            nz_e = ezero if edig else (1 if d == 0 else 0)
            return (4, dot, idig, izero, sint, fdig, flast0, nz, sig, emark, esign, edig + 1, min(ev * 10 + d, 999),
                    nz_e)
        if ch == 0x2E:                                   # '.'
            if part != 0 or not (trailing_dot or dot_frac):
                return None
            return (1, 1, idig, izero, sint, 0, 0, nz, sig, "", "", 0, 0, 0)
        if ch == 0x65 or ch == 0x45:                     # e E
            c = chr(ch)
            if part > 1 or not exp_ok or c not in exp_marks or idig + fdig == 0:
                return None
            return (2, dot, idig, izero, sint, fdig, flast0, nz, sig, c, "", 0, 0, 0)
        if ch == 0x2B or ch == 0x2D:                     # + - (exponent sign only)
            c = chr(ch)
            if part != 2 or c not in exp_signs:
                return None
            return (3, dot, idig, izero, sint, fdig, flast0, nz, sig, emark, c, 0, 0, 0)
        return None

    @staticmethod
    def _snum_accept(npf: tuple, st: tuple) -> bool:
        (kind, _ok, int_only, dot_frac, trailing_dot, leading_dot, _mi, _il, _mf, frac_last0, _e, _m, exp_signs,
         _me, _el, _ms) = npf
        part, dot, idig, izero, sint, fdig, flast0, nz, sig, emark, esign, edig, ev, ezero = st
        if part in (2, 3) or idig + fdig == 0:
            return False
        if not dot:
            if not int_only:
                return False
        elif idig and fdig:
            if not dot_frac:
                return False
        elif idig:
            if not trailing_dot:
                return False
        elif not leading_dot:
            return False
        if fdig and flast0 and not frac_last0:
            return False
        if part == 4 and not esign and "n" not in exp_signs:
            return False
        if nz:
            mag = sint - ev if esign == "-" else sint + ev
            if mag > NUM_FINITE_MAG:
                return False
        return True

    def _snum_extra(self, npf: tuple, st: tuple):
        mk = ("s", npf[0], st)
        hit = self._num_memo.get(mk, -1)
        if hit != -1:
            return hit
        res = self._bfs(lambda x, c: self._snum_step(npf, x, c), lambda x: self._snum_accept(npf, x), st,
                        b"019.eE+-", 8)
        if len(self._num_memo) >= _SMALL_MEMO_MAX:
            self._num_memo.clear()
        self._num_memo[mk] = res
        return res

    def _snum_min_leaf(self, npf: tuple):
        """Fewest chars of a complete numeral under the profile (None if none)."""
        mk = ("leaf", npf[0])
        hit = self._num_memo.get(mk, -1)
        if hit != -1:
            return hit
        best = None
        for ch in b"019.":
            st = self._snum_start(npf, ch)
            if st is None:
                continue
            e = self._snum_extra(npf, st)
            if e is not None and (best is None or 1 + e < best):
                best = 1 + e
        self._num_memo[mk] = best
        return best

    # ---- sym column digits after COL: [0-9]+ -------------------------------------------------------
    # state (ndig, first_digit_zero, sig_digits, value)
    def _colprof(self, ctx: int) -> tuple:
        pol, rk = self._ctxs[ctx]
        lim = rk[0] if rk[2] else None
        return (self.col_policy if pol else self.col_syntax) + (lim,)

    @staticmethod
    def _col_step(cp: tuple, st: tuple, ch: int):
        ok, max_dig, lead0_ok, lim = cp
        if not ok or not 0x30 <= ch <= 0x39:
            return None
        d = ch - 0x30
        ndig, z, sigd, val = st
        if ndig and z and not lead0_ok:
            return None
        if ndig + 1 > max_dig:
            return None
        nsig = sigd + 1 if (sigd or d) else 0
        if nsig > COL_MAX_SIG_DIGITS:
            return None
        nval = 0
        if lim is not None:
            nval = val * 10 + d
            if nval >= lim:
                return None
        return (ndig + 1, z if ndig else (1 if d == 0 else 0), nsig, nval)

    def _col_extra(self, cp: tuple, st: tuple):
        if st[0] >= 1:
            return 0
        return 1 if self._col_step(cp, st, 0x30) is not None else None

    # ---- pipe frames ----------------------------------------------------------------------------
    # pf = (phase, member, lead, role, node, prof, vst); node = POLICY trie node | SYNTAX last key bytes or None;
    # prof = value-profile index (-1: SYNTAX); vst = numeric state | spelled / literal bytes | None.
    def _tok_desc(self, t: int) -> str:
        cls = self.vocab.cls(t)
        if cls == "BYTE":
            b = t - BYTE_BASE
            return f"byte {chr(b)!r}" if 0x20 <= b < 0x7F else f"byte 0x{b:02x}"
        if cls in T.DATA_CLASSES:
            return f"{cls} token {self.vocab.name(t)!r}"
        return self.vocab.name(t)

    def _next_lead(self, ctx: int, lead: int, name: str) -> int:
        if not self._ctxs[ctx][1][1]:
            return 2
        return 1 if (lead == 0 and name == "target") else 2

    def _pf_adv(self, ctx: int, pf: tuple, t: int) -> tuple:
        """(pf', winsor delta, stage delta)."""
        phase, member, lead, role, node, prof, vst = pf
        pol = self._ctxs[ctx][0]
        if phase == P_OPEN:
            if t == STAGE_OPEN:
                return (P_OP, member, lead, 0, None, 0, None), 0, 1
        elif phase == P_OP:
            if self.r_op[0] <= t < self.r_op[1]:
                name = self.op_name[t]
                if not self._op_ok_name(ctx, name, member, lead):
                    raise GrammarError(f"op {name!r} is not allowed here")
                nlead = self._next_lead(ctx, lead, name)
                if pol:
                    rl = self.p_role.get(t)
                    if rl is None:
                        raise GrammarError(f"op {name!r} has no train_fit signature")
                    return (P_KEYS, member, nlead, rl, self.p_root[t], 0, None), self.pwin[t], 0
                return (P_KEYS, member, nlead, ROLE_FINAL | ROLE_NONFINAL, None, 0, None), self.pwin[t], 0
            if t == OP_BEGIN and not pol:
                return (P_OPSP, member, lead, 0, None, 0, self._spell_init("OP", None)), 0, 0
        elif phase == P_OPSP:
            if BYTE_BASE <= t < DATA_BASE:
                q = self._spell_step("OP", vst, t - BYTE_BASE, None)
                if q is not None:
                    return (P_OPSP, member, lead, 0, None, 0, q), 0, 0
            elif t == OP_END and self._spell_end(ctx, "OP", vst, (member, lead)):
                text = vst[0].decode("utf-8")
                w = (1 if text.startswith("winsor_clip") else 0) + text.count(_WINSOR)
                return (P_KEYS, member, self._next_lead(ctx, lead, text), ROLE_FINAL | ROLE_NONFINAL, None, 0,
                        None), w, 0
        elif phase == P_KEYS:
            return self._pf_keys(ctx, pf, t)
        elif phase == P_KEYSP:
            if BYTE_BASE <= t < DATA_BASE:
                q = self._spell_step("KEY", vst, t - BYTE_BASE, node)
                if q is not None:
                    return (P_KEYSP, member, lead, role, node, -1, q), 0, 0
            elif t == KEY_END:
                if self._spell_end(ctx, "KEY", vst, node):
                    return (P_VAL, member, lead, role, vst[0], -1, None), vst[0].count(_WINSOR.encode()), 0
                if vst[1][0]:
                    raise GrammarError("a key that ends inside a quote (an odd number of '\"') is a documented "
                                       "SYNTAX refusal")
        elif phase == P_VAL:
            vp = self._vprof(prof)
            if t == TRUE or t == FALSE:
                if vp[7 if t == TRUE else 8]:
                    return (P_KEYS, member, lead, role, node, 0, None), 0, 0
            elif self.r_cat[0] <= t < self.r_cat[1]:
                if vp[9] is None or t in vp[9]:
                    return (P_KEYS, member, lead, role, node, 0, None), self.pwin[t], 0
            elif t == STR_BEGIN:
                if vp[10] is None:
                    return (P_STR, member, lead, role, node, prof, self._spell_init("STR", None)), 0, 0
                if vp[10]:
                    return (P_STR, member, lead, role, node, prof, b""), 0, 0
            elif BYTE_BASE <= t < DATA_BASE:
                st = self._pnum_step(vp, self._PNUM0, t - BYTE_BASE)
                if st is not None:
                    return (P_NUM, member, lead, role, node, prof, st), 0, 0
            if pol:
                raise GrammarError(f"value {self._tok_desc(t)} was never observed for this (op,key)")
        elif phase == P_NUM:
            vp = self._vprof(prof)
            if BYTE_BASE <= t < DATA_BASE:
                st = self._pnum_step(vp, vst, t - BYTE_BASE)
                if st is not None:
                    return (P_NUM, member, lead, role, node, prof, st), 0, 0
                raise GrammarError(f"numeric value cannot continue with {self._tok_desc(t)} (outside its table)")
            if not self._pnum_accept(vp, vst):
                raise GrammarError("numeric value is incomplete or outside its table")
            return self._pf_keys(ctx, (P_KEYS, member, lead, role, node, 0, None), t)
        elif phase == P_STR:
            vp = self._vprof(prof)
            if BYTE_BASE <= t < DATA_BASE:
                if vp[10] is not None:
                    buf = vst + bytes((t - BYTE_BASE,))
                    if any(lit.startswith(buf) for lit in vp[10]):
                        return (P_STR, member, lead, role, node, prof, buf), 0, 0
                else:
                    q = self._spell_step("STR", vst, t - BYTE_BASE, None)
                    if q is not None:
                        return (P_STR, member, lead, role, node, prof, q), 0, 0
            elif t == STR_END:
                if vp[10] is not None:
                    if vst in vp[10]:
                        return (P_KEYS, member, lead, role, node, 0, None), vst.count(_WINSOR.encode()), 0
                elif self._spell_end(ctx, "STR", vst, None):
                    return (P_KEYS, member, lead, role, node, 0, None), vst[0].count(_WINSOR.encode()), 0
        elif phase == P_SEND:
            if t == STAGE_OPEN and (not pol or role & ROLE_NONFINAL):
                return (P_OP, member, lead, 0, None, 0, None), 0, 1
            if t == CLOSE and (not pol or role & ROLE_FINAL):
                return (P_END, member, lead, role, None, 0, None), 0, 0
        raise GrammarError(f"{self._tok_desc(t)} is not allowed in pipe phase {PIPE_PHASE_NAMES[phase]}")

    def _pf_keys(self, ctx: int, pf: tuple, t: int) -> tuple:
        _phase, member, lead, role, node, _prof, _vst = pf
        pol = self._ctxs[ctx][0]
        if t == CLOSE:
            if pol and not self.p_nodes[node][1]:
                raise GrammarError("this keyset was never observed for the op (trie node is not terminal)")
            return (P_SEND, member, lead, role, None, 0, None), 0, 0
        if self.r_key[0] <= t < self.r_key[1]:
            if pol:
                e = self.p_nodes[node][2].get(t)
                if e is None:
                    raise GrammarError("unseen keyset: the key is not a child of the op's keyset trie node")
                return (P_VAL, member, lead, role, e[0], e[1], None), self.pwin[t], 0
            kb = self.key_bytes[t]
            if node is not None and kb <= node:
                raise GrammarError("keys must be strictly increasing (UTF-8 byte order)")
            return (P_VAL, member, lead, role, kb, -1, None), self.pwin[t], 0
        if t == KEY_BEGIN and not pol:
            return (P_KEYSP, member, lead, role, node, -1, self._spell_init("KEY", node)), 0, 0
        raise GrammarError(f"token {t} is not allowed after an op or a value")

    # ---- minimal completion costs of pipe frames ---------------------------------------------------
    # one-vector frontiers of the forced structural tokens
    _C_CLOSE = ((1, 1, 0, 0, 0, 0),)
    _C_EOS = ((1, 0, 0, 0, 0, 0),)
    _C_STAGE = ((1, 2, 1, 0, 1, 0),)
    _C_PIPE = ((1, 7, 1, 0, 0, 0),)
    _C_MEMBER = ((1, 9, 1, 0, 0, 0),)
    _F_ZERO = (_ZERO,)

    @staticmethod
    def _scalar(k: int, extra_tokens: int = 0) -> tuple:
        """Frontier of k spelled / numeric bytes (one token and one byte each) plus marker tokens (empty pieces)."""
        return ((k + extra_tokens, k, 0, 0, 0, 0),)

    def _vmin(self, prof: int):
        """Frontier of the complete values of a profile."""
        mk = ("v", prof)
        hit = self._num_memo.get(mk, -1)
        if hit != -1:
            return hit
        vp = self._vprof(prof)
        cands = []
        e = self._pnum_extra(prof, self._PNUM0)
        if e is not None:
            cands.append((e, e, 0, 0, 0, 0))
        if vp[7]:
            cands.append(self._tokcost(TRUE))
        if vp[8]:
            cands.append(self._tokcost(FALSE))
        for c in (vp[9] if vp[9] is not None else range(*self.r_cat)):
            cands.append(self._tokcost(c))
        if vp[10] is not None:
            for lit in vp[10]:
                cands.append((len(lit) + 2, len(lit), lit.count(b"("), lit.count(_WINSOR.encode()), 0, 0))
        best = _prune(cands)
        self._num_memo[mk] = best
        return best

    def _trie_minima(self) -> list:
        """POLICY: frontier of the key/value completions from each trie node to a terminal node."""
        n = len(self.p_nodes)
        out = [None] * n
        cap = self._maxcap[True]
        for i in range(n - 1, -1, -1):
            _op, terminal, edges = self.p_nodes[i]
            best = self._F_ZERO if terminal else None
            for kid, (child, prof) in edges.items():
                best = _cmin2(best, _cadd(_cadd(self._tokf(kid), self._vmin(prof)), out[child]))
            out[i] = None if best is None else _prune(list(best), cap)
        return out

    def _kmin(self, ctx: int, node):
        return self._kmin_tab[node] if self._ctxs[ctx][0] else self._F_ZERO

    def _tail(self, ctx: int, member: int, lead: int, role: int):
        """From P_SEND (after a stage CLOSE) through the pipe CLOSE. Closing dominates opening another stage in
        every resource, so a final-capable op needs only the CLOSE."""
        pol = self._ctxs[ctx][0]
        if not pol or role & ROLE_FINAL:
            return self._C_CLOSE
        if role & ROLE_NONFINAL:
            return self._G(ctx, member, lead)
        return None

    def _H(self, ctx: int, member: int, lead: int):
        """From P_OP (stage opened, op to choose) through the pipe CLOSE."""
        mk = ("H", ctx, member, lead)
        if mk in self._stage_memo:
            return self._stage_memo[mk]
        # cycle guard: a chain through one more non-final stage is dominated in every resource by the same
        # completion without it (more tokens, bytes, '(' and stages), so the guarded value is exact
        self._stage_memo[mk] = None
        pol = self._ctxs[ctx][0]
        best = None
        for t in range(*self.r_op):
            name = self.op_name[t]
            if not self._op_ok_name(ctx, name, member, lead):
                continue
            if pol:
                rl = self.p_role.get(t)
                if rl is None:
                    continue
                k = self._kmin_tab[self.p_root[t]]
            else:
                rl, k = ROLE_FINAL | ROLE_NONFINAL, self._F_ZERO
            tail = self._tail(ctx, member, self._next_lead(ctx, lead, name), rl)
            best = _cmin2(best, _cadd(_cadd(_cadd(self._tokf(t), k), self._C_CLOSE), tail))
        if not pol:
            ks = self._spell_extra(ctx, "OP", self._spell_init("OP", None), (member, lead))
            if ks is not None:
                best = _cmin2(best, _cadd(self._scalar(ks, 2), _cadd(self._C_CLOSE, self._C_CLOSE)))
        best = self._clip(ctx, best)
        self._stage_memo[mk] = best
        return best

    def _G(self, ctx: int, member: int, lead: int):
        """From P_OPEN (a stage must be opened) through the pipe CLOSE."""
        return _cadd(self._C_STAGE, self._H(ctx, member, lead))

    def _pf_cost(self, ctx: int, pf: tuple):
        """Frontier of the completions of a pipe frame through its pipe CLOSE."""
        phase, member, lead, role, node, prof, vst = pf
        if phase == P_END:
            return self._F_ZERO
        if phase == P_OPEN:
            return self._G(ctx, member, lead)
        if phase == P_OP:
            return self._H(ctx, member, lead)
        if phase == P_SEND:
            return self._tail(ctx, member, lead, role)
        if phase == P_OPSP:
            k = self._spell_extra(ctx, "OP", vst, (member, lead))
            if k is None:
                return None
            return _cadd(self._scalar(k, 1), _cadd(self._C_CLOSE, self._C_CLOSE))
        rest = _cadd(_cadd(self._kmin(ctx, node), self._C_CLOSE), self._tail(ctx, member, lead, role))
        if phase == P_KEYS:
            return rest
        if phase == P_KEYSP:
            k = self._spell_extra(ctx, "KEY", vst, node)
            if k is None:
                return None
            return _cadd(_cadd(((k + 1, k + 1, 0, 0, 0, 0),), self._vmin(-1)), rest)   # KEY_END piece "="
        if phase == P_VAL:
            return _cadd(self._vmin(prof), rest)
        if phase == P_NUM:
            e = self._pnum_extra(prof, vst)
            return None if e is None else _cadd(self._scalar(e), rest)
        if phase == P_STR:
            vp = self._vprof(prof)
            if vp[10] is not None:
                cands = []
                for lit in vp[10]:
                    if lit.startswith(vst):
                        r = lit[len(vst):]
                        # the literal's "(winsor_clip" count is charged at STR_END, its '(' byte by byte
                        cands.append((len(r) + 1, len(r), r.count(b"("), lit.count(_WINSOR.encode()), 0, 0))
                return _cadd(_prune(cands), rest)
            k = self._spell_extra(ctx, "STR", vst, None)
            return None if k is None else _cadd(self._scalar(k, 1), rest)
        raise AssertionError(phase)

    # ---- portfolios ------------------------------------------------------------------------------
    def _port_counts(self, ctx: int) -> tuple:
        if self._ctxs[ctx][0]:
            return frozenset(self.p_members), (max(self.p_members) if self.p_members else 0), self.p_blend
        return frozenset(range(1, 13)), 12, frozenset((BLEND_AUTO, BLEND_ONE))

    def _port_after(self, ctx: int, m: int):
        """From Q_SEP with m members closed: more members and/or the blend, then EOS."""
        mk = ("P", ctx, m)
        if mk in self._stage_memo:
            return self._stage_memo[mk]
        ok_counts, maxm, blends = self._port_counts(ctx)
        best = None
        if m in ok_counts:
            for b in blends:
                best = _cmin2(best, _cadd(self._tokf(b), self._C_EOS))
        if m + 1 <= maxm:
            mem = _cadd(_cadd(self._C_MEMBER, self._C_PIPE), self._G(ctx, 1, self._lead0(ctx)))
            best = _cmin2(best, _cadd(_cadd(mem, self._C_CLOSE), self._port_after(ctx, m + 1)))
        best = self._clip(ctx, best)
        self._stage_memo[mk] = best
        return best

    # ---- sym bodies (math_expr.rs lex + MathExpr::parse as a bounded pushdown automaton) -----------
    # sub = (sphase, frames, lvl, aux, wsp); lvl = (A, M, PC, PMAX, U, D, const_only): the operator-precedence
    # depth record of one expression level (A: add-level depth so far, M: mul-level depth so far, PC/PMAX: pow
    # chain length and max(k + depth(a_k)), U: pending unary minus, D: depth of the completed operand).
    # frames: ("G", parent lvl) | ("C", arity, args done, max arg depth, parent lvl) | ("K", parent lvl) col(N).
    _LVL0 = (0, 0, 0, 0, 0, 0, 0)
    _LVL0_CO = (0, 0, 0, 0, 0, 0, 1)

    @staticmethod
    def _total(lvl: tuple) -> int:
        A, M, PC, PMAX, _U, D, _co = lvl
        p = D if PC == 0 else max(PMAX, PC + D)
        m = p if M == 0 else 1 + max(M, p)
        return m if A == 0 else 1 + max(A, m)

    @staticmethod
    def _operand(lvl: tuple, dp: int) -> tuple:
        """Complete the level's pending operand with a primary of depth dp (pending unary minus applied)."""
        D = dp + lvl[4]
        if D > SYM_MAX_DEPTH:
            raise GrammarError("sym AST depth would exceed 16")
        return (lvl[0], lvl[1], lvl[2], lvl[3], 0, D, lvl[6])

    def _ws_step(self, ctx: int, wsp: bytes, b: int):
        """Pending whitespace bytes after one more byte: b"" when a whitespace char completed, None if invalid."""
        full, pre, _rest = self._ws_tab[self._ctxs[ctx][0]]
        nw = wsp + bytes((b,)) if wsp else _BYTE1[b]
        if nw in full:
            return b""
        if nw in pre:
            return nw
        return None

    def _after_id(self, frames: tuple, lvl: tuple, role: tuple) -> tuple:
        if role[0] == "V":
            return (S_OPTR, frames, self._operand(lvl, 1), None, b"")
        if role[0] == "F":
            return (S_FUNC, frames, lvl, ("C", role[1]), b"")
        return (S_FUNC, frames, lvl, ("K",), b"")

    def _push_call(self, frames: tuple, lvl: tuple, aux: tuple) -> tuple:
        if aux[0] == "C":
            return (S_OPND, frames + (("C", aux[1], 0, 0, lvl),), self._LVL0, None, b"")
        return (S_OPND, frames + (("K", lvl),), self._LVL0_CO, None, b"")

    def _sym_adv(self, ctx: int, sub: tuple, t: int) -> tuple:
        """(sub', nodes delta)."""
        sphase, frames, lvl, aux, wsp = sub
        isbyte = BYTE_BASE <= t < DATA_BASE
        b = t - BYTE_BASE
        if wsp:
            if isbyte:
                n = self._ws_step(ctx, wsp, b)
                if n is not None:
                    return (sphase, frames, lvl, aux, n), 0
            raise GrammarError("incomplete whitespace character")
        if sphase == S_NUM:
            npf = self._numprof(ctx, lvl[6])
            if isbyte:
                st = self._snum_step(npf, aux, b)
                if st is not None:
                    return (S_NUM, frames, lvl, st, b""), 0
            if not self._snum_accept(npf, aux):
                raise GrammarError("sym numeral is incomplete or outside its table")
            return self._sym_optr(ctx, (S_OPTR, frames, self._operand(lvl, 1), None, b""), t)
        if sphase == S_COL:
            if isbyte and 0x30 <= b <= 0x39:
                st = self._col_step(self._colprof(ctx), aux, b)
                if st is None:
                    raise GrammarError("column digits outside the table (or >= feature_count)")
                return (S_COL, frames, lvl, st, b""), 0
            if aux[0] < 1:
                raise GrammarError("COL needs at least one digit")
            return self._sym_optr(ctx, (S_OPTR, frames, self._operand(lvl, 1), None, b""), t)
        if sphase == S_IDSP:
            if isbyte:
                q = self._spell_step("ID", aux, b, None)
                if q is not None:
                    return (S_IDSP, frames, lvl, q, b""), 0
            elif t == ID_END and aux[0]:
                role = self._id_end_role(ctx, aux[0].decode("ascii"))
                if role is not None:
                    return self._after_id(frames, lvl, role), 0
            raise GrammarError("not a valid spelled identifier")
        if sphase == S_FUNC:
            if isbyte:
                if b == 0x28:
                    return self._push_call(frames, lvl, aux), 0
                n = self._ws_step(ctx, b"", b)
                if n is not None:
                    return (S_FUNC, frames, lvl, aux, n), 0
            raise GrammarError("a function name must be followed by '('")
        if sphase == S_OPND:
            return self._sym_opnd(ctx, sub, t)
        if sphase == S_OPTR:
            return self._sym_optr(ctx, sub, t)
        raise GrammarError("the sym body is closed")

    def _sym_opnd(self, ctx: int, sub: tuple, t: int) -> tuple:
        _sphase, frames, lvl, _aux, _wsp = sub
        pol = self._ctxs[ctx][0]
        co = lvl[6]
        if BYTE_BASE <= t < DATA_BASE:
            b = t - BYTE_BASE
            if b == 0x28:                                  # '(' group
                if pol and not self.ps["groups"]:
                    raise GrammarError("parenthesized groups were never observed")
                return (S_OPND, frames + (("G", lvl),), self._LVL0_CO if co else self._LVL0, None, b""), 0
            if b == 0x2D:                                  # unary '-'
                if co or (pol and not self.ps["unary_minus"]):
                    raise GrammarError("unary minus is not allowed here")
                if lvl[4] + 1 > SYM_MAX_DEPTH:
                    raise GrammarError("sym AST depth would exceed 16")
                return (S_OPND, frames, lvl[:4] + (lvl[4] + 1,) + lvl[5:], None, b""), 1
            if b == 0x2B:                                  # unary '+' (no node)
                if pol and not self.ps["unary_plus"]:
                    raise GrammarError("unary plus was never observed")
                return sub, 0
            if b == 0x2E or 0x30 <= b <= 0x39:
                st = self._snum_start(self._numprof(ctx, co), b)
                if st is None:
                    raise GrammarError("numeral start outside the table")
                return (S_NUM, frames, lvl, st, b""), (0 if co else 1)
            n = self._ws_step(ctx, b"", b)
            if n is not None:
                return (S_OPND, frames, lvl, None, n), 0
            raise GrammarError("an operand is expected")
        if t == COL:
            if co or not self._colprof(ctx)[0]:
                raise GrammarError("COL is not allowed here")
            return (S_COL, frames, lvl, (0, 0, 0, 0), b""), 1
        if self.r_id[0] <= t < self.r_id[1]:
            if co or (pol and t not in self.ps_ids):
                raise GrammarError("identifier is not allowed here")
            role = self._id_role_ok(ctx, self.id_role[t])
            if role is None or (role[0] == "K" and pol and not self.ps["col_calls"]):
                raise GrammarError("identifier is neither a usable variable nor a function")
            return self._after_id(frames, lvl, role), 1
        if t == ID_BEGIN and not pol and not co:
            return (S_IDSP, frames, lvl, self._spell_init("ID", None), b""), 1
        raise GrammarError("an operand is expected")

    def _sym_optr(self, ctx: int, sub: tuple, t: int) -> tuple:
        _sphase, frames, lvl, _aux, _wsp = sub
        co = lvl[6]
        if BYTE_BASE <= t < DATA_BASE:
            b = t - BYTE_BASE
            if b in _BINOP_BYTES:
                if co:
                    raise GrammarError("col() takes an integer literal")
                ch = chr(b)
                if self._ctxs[ctx][0] and ch not in self.ps_binary:
                    raise GrammarError(f"binary {ch!r} was never observed")
                A, M, PC, PMAX, _U, D, _co = lvl
                if ch == "^":
                    nl = (A, M, PC + 1, max(PMAX, PC + 1 + D), 0, 0, co)
                else:
                    p = D if PC == 0 else max(PMAX, PC + D)
                    if ch in "*/":
                        nl = (A, p if M == 0 else 1 + max(M, p), 0, 0, 0, 0, co)
                    else:
                        m = p if M == 0 else 1 + max(M, p)
                        nl = (m if A == 0 else 1 + max(A, m), 0, 0, 0, 0, 0, co)
                if max(nl[0], nl[1], nl[3]) > SYM_MAX_DEPTH:
                    raise GrammarError("sym AST depth would exceed 16")
                return (S_OPND, frames, nl, None, b""), 1
            if b == 0x2C:                                  # ','
                if not frames or frames[-1][0] != "C" or frames[-1][2] + 1 >= frames[-1][1]:
                    raise GrammarError("',' outside a call that needs more arguments")
                _k, ar, k, m, plvl = frames[-1]
                return (S_OPND, frames[:-1] + (("C", ar, k + 1, max(m, self._total(lvl)), plvl),), self._LVL0,
                        None, b""), 0
            if b == 0x29:                                  # ')'
                if not frames:
                    raise GrammarError("unbalanced ')'")
                fr = frames[-1]
                if fr[0] == "G":
                    nl = self._operand(fr[1], self._total(lvl))
                elif fr[0] == "C":
                    _k, ar, k, m, plvl = fr
                    if k + 1 != ar:
                        raise GrammarError("function arity mismatch")
                    nl = self._operand(plvl, 1 + max(m, self._total(lvl)))
                else:
                    nl = self._operand(fr[1], 1)
                return (S_OPTR, frames[:-1], nl, None, b""), 0
            n = self._ws_step(ctx, b"", b)
            if n is not None:
                return (S_OPTR, frames, lvl, None, n), 0
            raise GrammarError("an operator is expected")
        if t == SYM_CLOSE:
            if frames:
                raise GrammarError("SYM_CLOSE with open frames")
            if self._total(lvl) > SYM_MAX_DEPTH:
                raise GrammarError("sym AST depth would exceed 16")
            return (S_END, (), self._LVL0, None, b""), 0
        raise GrammarError("an operator is expected")

    # ---- minimal completion of sym bodies --------------------------------------------------------
    def _leaf(self, ctx: int, co: int):
        mk = ("leaf", ctx, co)
        if mk in self._stage_memo:
            return self._stage_memo[mk]
        pol = self._ctxs[ctx][0]
        cands = []
        n = self._snum_min_leaf(self._numprof(ctx, co))
        if n is not None:
            cands.append((n, n, 0, 0, 0, 0 if co else 1))
        if not co:
            for t in range(*self.r_id):
                role = self._id_role_ok(ctx, self.id_role[t])
                if role is None or role[0] != "V" or (pol and t not in self.ps_ids):
                    continue
                cands.append((1, self.plen[t], self.plpar[t], 0, 0, 1))
            cp = self._colprof(ctx)
            if cp[0] and self._col_step(cp, (0, 0, 0, 0), 0x30) is not None:
                cands.append((2, 2, 0, 0, 0, 1))
            if not pol:
                for k, role in self._id_completions(ctx, b""):
                    if role[0] == "V":
                        cands.append((k + 2, k, 0, 0, 0, 1))
        best = self._clip(ctx, cands)
        self._stage_memo[mk] = best
        return best

    def _id_completions(self, ctx: int, buf: bytes) -> list:
        """[(bytes to append, role)] minimal per role class for a spelled identifier prefix (SYNTAX)."""
        mk = (ctx, buf)
        hit = self._idc_memo.get(mk)
        if hit is not None:
            return hit
        text = buf.decode("ascii")
        low = text.lower()
        best = {}

        def consider(ext: str) -> None:
            for v in (ext, ext.upper(), ext[:1].upper() + ext[1:]):
                role = self._id_end_role(ctx, text + v)
                if role is not None:
                    key = ("F", role[1]) if role[0] == "F" else (role[0],)
                    if key not in best or len(v) < best[key][0]:
                        best[key] = (len(v), ("V", -1) if role[0] == "V" else role)
                    return

        for name in _FIXED_ID_NAMES:
            if name.startswith(low):
                consider(name[len(low):])
        for pat in ("c", "col"):
            if pat.startswith(low):
                consider(pat[len(low):] + "0")
            elif low.startswith(pat) and low[len(pat):].isdigit():
                consider("")
        out = sorted(best.values())
        if len(self._idc_memo) >= _SMALL_MEMO_MAX:
            self._idc_memo.clear()
        self._idc_memo[mk] = out
        return out

    def _sym_close(self, ctx: int, frames: tuple, lvl: tuple):
        """From operator position (lvl has a completed operand): close every frame, then SYM_CLOSE and EOS."""
        cost = self._F_ZERO
        d = self._total(lvl)
        if d > SYM_MAX_DEPTH:
            return None
        rp = self._tokf(BYTE_BASE + 0x29)
        for fr in reversed(frames):
            if fr[0] == "G":
                cost = _cadd(cost, rp)
                plvl, dp = fr[1], d
            elif fr[0] == "C":
                _k, ar, k, m, plvl = fr
                mx = max(m, d)
                missing = ar - (k + 1)
                if missing:
                    leaf = self._leaf(ctx, 0)
                    if leaf is None:
                        return None
                    one = _cadd(self._tokf(BYTE_BASE + 0x2C), leaf)
                    for _ in range(missing):
                        cost = _cadd(cost, one)
                    mx = max(mx, 1)
                cost = _cadd(cost, rp)
                dp = 1 + mx
            else:
                cost = _cadd(cost, rp)
                plvl, dp = fr[1], 1
            D = dp + plvl[4]
            if D > SYM_MAX_DEPTH:
                return None
            d = self._total((plvl[0], plvl[1], plvl[2], plvl[3], 0, D, plvl[6]))
            if d > SYM_MAX_DEPTH:
                return None
        return _cadd(cost, ((2, 2, 0, 0, 0, 0),))       # SYM_CLOSE '")' + EOS

    def _sym_cost(self, ctx: int, sub: tuple):
        sphase, frames, lvl, aux, wsp = sub
        c = self._F_ZERO
        if wsp:
            k = self._ws_tab[self._ctxs[ctx][0]][2].get(wsp)
            if k is None:
                return None
            c = self._scalar(k)
        if sphase == S_END:
            return _cadd(c, self._C_EOS)
        if sphase == S_OPTR:
            return _cadd(c, self._sym_close(ctx, frames, lvl))
        try:
            if sphase == S_OPND:
                leaf = self._leaf(ctx, lvl[6])
                if leaf is None:
                    return None
                return _cadd(_cadd(c, leaf), self._sym_close(ctx, frames, self._operand(lvl, 1)))
            if sphase == S_NUM:
                e = self._snum_extra(self._numprof(ctx, lvl[6]), aux)
                if e is None:
                    return None
                return _cadd(_cadd(c, self._scalar(e)), self._sym_close(ctx, frames, self._operand(lvl, 1)))
            if sphase == S_COL:
                e = self._col_extra(self._colprof(ctx), aux)
                if e is None:
                    return None
                return _cadd(_cadd(c, self._scalar(e)), self._sym_close(ctx, frames, self._operand(lvl, 1)))
        except GrammarError:
            return None
        if sphase == S_FUNC:
            return _cadd(_cadd(c, self._tokf(BYTE_BASE + 0x28)),
                         self._C(ctx, F_SYM, self._push_call(frames, lvl, aux)))
        if sphase == S_IDSP:
            best = None
            for k, role in self._id_completions(ctx, aux[0]):
                try:
                    sub2 = self._after_id(frames, lvl, role)
                except GrammarError:
                    continue
                best = _cmin2(best, _cadd(self._scalar(k, 1), self._C(ctx, F_SYM, sub2)))
            return _cadd(c, best)
        raise AssertionError(sphase)

    # ---- the automaton API ---------------------------------------------------------------------
    def _pf_new(self, ctx: int, member: int) -> tuple:
        return (P_OPEN, member, self._lead0(ctx), 0, None, 0, None)

    def advance(self, s: State, tok: int) -> State:
        """The successor state; raises GrammarError when the token is structurally invalid (budget and
        liveness are checked by allowed() / accepts())."""
        t = int(tok)
        if not 0 <= t < self.V:
            raise GrammarError(f"token id {t} outside 0..{self.V - 1}")
        ctx, used, form, sub = s.ctx, s.used, s.form, s.sub
        w = st = nd = 0
        if form == F_INIT:
            if t != BOS:
                raise GrammarError("a sequence starts with BOS")
            nform, nsub = F_START, ()
        elif form == F_START:
            if t not in _OPENERS:
                raise GrammarError("a program starts with PIPE_OPEN, SYM_*_OPEN or PORTFOLIO_OPEN")
            if self._ctxs[ctx][0] and t not in self.p_forms:
                raise GrammarError("form never observed in train_fit")
            if t == PIPE_OPEN:
                nform, nsub = F_PIPE, self._pf_new(ctx, 0)
            elif t == SYM_AFFINE_OPEN or t == SYM_RAW_OPEN:
                nform, nsub = F_SYM, (S_OPND, (), self._LVL0, None, b"")
            elif t == PORTFOLIO_OPEN:
                nform, nsub = F_PORT, (Q_SEP, 0, None)
            else:
                raise GrammarError("a program starts with PIPE_OPEN, SYM_*_OPEN or PORTFOLIO_OPEN")
        elif form == F_PIPE:
            if sub[0] == P_END:
                if t != EOS:
                    raise GrammarError("EOS expected")
                nform, nsub = F_DONE, ()
            else:
                nsub, w, st = self._pf_adv(ctx, sub, t)
                nform = F_PIPE
        elif form == F_SYM:
            if sub[0] == S_END:
                if t != EOS:
                    raise GrammarError("EOS expected")
                nform, nsub = F_DONE, ()
            else:
                nsub, nd = self._sym_adv(ctx, sub, t)
                nform = F_SYM
        elif form == F_PORT:
            nform = F_PORT
            q, m, pf = sub
            if q == Q_SEP:
                counts, maxm, blends = self._port_counts(ctx)
                if t == MEMBER_OPEN and m + 1 <= maxm:
                    nsub = (Q_MEMBER, m + 1, None)
                elif (t == BLEND_AUTO or t == BLEND_ONE) and t in blends and m in counts:
                    nsub = (Q_END, m, None)
                else:
                    raise GrammarError("MEMBER_OPEN or an observed blend with an allowed member count expected")
            elif q == Q_MEMBER:
                if t != PIPE_OPEN:
                    raise GrammarError("a member holds a pipe")
                nsub = (Q_PIPE, m, self._pf_new(ctx, 1))
            elif q == Q_PIPE:
                npf, w, st = self._pf_adv(ctx, pf, t)
                nsub = (Q_MCLOSE, m, None) if npf[0] == P_END else (Q_PIPE, m, npf)
            elif q == Q_MCLOSE:
                if t != CLOSE:
                    raise GrammarError("member CLOSE expected")
                nsub = (Q_SEP, m, None)
            else:
                if t != EOS:
                    raise GrammarError("EOS expected")
                nform, nsub = F_DONE, ()
        else:
            raise GrammarError("the sequence already ended")
        nu = (used[0] + 1, used[1] + self.plen[t], used[2] + self.plpar[t], used[3] + w, used[4] + st, used[5] + nd)
        return State(s.mode, ctx, nu, nform, nsub)

    def _C(self, ctx: int, form: int, sub: tuple):
        """The completion frontier: the Pareto-minimal resource vectors (tokens, bytes, lparen, winsor, stages,
        nodes) over every completion of the state (SYNTAX: lparen, winsor and stages projected out), or None when
        no completion exists. Every vector is realized by a token whose successor frontier holds the rest."""
        mk = (ctx, form, sub)
        try:
            return self._cmemo[mk]
        except KeyError:
            pass
        if form == F_DONE:
            c = self._F_ZERO
        elif form == F_INIT:
            c = _cadd(self._tokf(BOS), self._C(ctx, F_START, ()))
        elif form == F_START:
            c = None
            pol = self._ctxs[ctx][0]
            for t in _OPENERS:
                if pol and t not in self.p_forms:
                    continue
                s2 = self.advance(State("", ctx, (1, 0, 0, 0, 0, 0), F_START, ()), t)
                c2 = self._C(ctx, s2.form, s2.sub)
                cap = self._caps(ctx, s2.form)
                fit = [v for v in (c2 or ()) if _fits(s2.used, v, cap)]
                if fit:
                    c = _cmin2(c, _cadd(self._tokf(t), tuple(fit)))
        elif form == F_PIPE:
            c = _cadd(self._pf_cost(ctx, sub), self._C_EOS) if sub[0] != P_END else self._C_EOS
        elif form == F_SYM:
            c = self._sym_cost(ctx, sub)
        else:
            q, m, pf = sub
            if q == Q_SEP:
                c = self._port_after(ctx, m)
            elif q == Q_MEMBER:
                c = _cadd(_cadd(_cadd(self._C_PIPE, self._G(ctx, 1, self._lead0(ctx))), self._C_CLOSE),
                          self._port_after(ctx, m))
            elif q == Q_PIPE:
                c = _cadd(_cadd(self._pf_cost(ctx, pf), self._C_CLOSE), self._port_after(ctx, m))
            elif q == Q_MCLOSE:
                c = _cadd(self._C_CLOSE, self._port_after(ctx, m))
            else:
                c = self._C_EOS
        c = self._clip(ctx, c)
        if len(self._cmemo) >= _CMEMO_MAX:
            self._cmemo.clear()                  # a pure cache: completion costs are recomputed on demand
        self._cmemo[mk] = c
        return c

    def is_final(self, s: State) -> bool:
        return s.form == F_DONE

    def min_to_complete(self, s: State) -> int:
        """Fewest further tokens (EOS included) of a completion that fits every remaining budget; INFINITE when
        none exists."""
        c = self._C(s.ctx, s.form, s.sub)
        cap = self._caps(s.ctx, s.form)
        fit = [v[0] for v in (c or ()) if _fits(s.used, v, cap)]
        return min(fit) if fit else INFINITE

    def completion_fits(self, s: State) -> bool:
        """Some completion of s fits every remaining budget (the condition allowed() applies to successors)."""
        return _ffits(s.used, self._C(s.ctx, s.form, s.sub), self._caps(s.ctx, s.form))

    def completion_frontier(self, s: State):
        """The completion frontier of s (see _C); None when s has no completion."""
        return self._C(s.ctx, s.form, s.sub)

    def _candidates(self, s: State) -> np.ndarray:
        form = s.form
        fx = self.c_fixed
        if form == F_INIT:
            return fx["init"]
        if form == F_START:
            return fx["start"]
        if form == F_SYM:
            return fx["eos"] if s.sub[0] == S_END else self.c_sym
        if form == F_PIPE:
            return self._pf_cands(s.sub)
        if form == F_PORT:
            q = s.sub[0]
            if q == Q_PIPE:
                return self._pf_cands(s.sub[2])
            return fx[("sep", "member", None, "close", "eos")[q]]
        return fx["none"]

    def _pf_cands(self, pf: tuple) -> np.ndarray:
        ph = pf[0]
        if ph == P_OP:
            return self.c_op
        if ph == P_KEYS:
            return self.c_keys
        if ph == P_VAL:
            return self.c_val
        if ph == P_NUM:
            return self.c_num
        if ph == P_OPSP:
            return self.c_spell[OP_END]
        if ph == P_KEYSP:
            return self.c_spell[KEY_END]
        if ph == P_STR:
            return self.c_spell[STR_END]
        return self.c_fixed[{P_OPEN: "open", P_SEND: "send", P_END: "eos"}[ph]]

    def _l1_entry(self, s: State) -> tuple:
        sk = (s.ctx, s.form, s.sub)
        e = self._l1.get(sk)
        if e is not None:
            return e
        ids, owner, costs, caps = [], [], [], []
        base = State(s.mode, s.ctx, _ZERO, s.form, s.sub)
        for t in self._candidates(s).tolist():
            try:
                s2 = self.advance(base, t)
            except GrammarError:
                continue
            c = self._C(s2.ctx, s2.form, s2.sub)
            if c is None:
                continue
            k = len(ids)
            ids.append(t)
            cap = self._caps(s2.ctx, s2.form)
            for v in c:                          # one row per frontier vector: t is allowed if any row fits
                owner.append(k)
                costs.append(_vadd(s2.used, v))
                caps.append(cap)
        ids_a = np.array(ids, dtype=np.int64)
        owner_a = np.array(owner, dtype=np.int64)
        costs_a = np.array(costs, dtype=np.int64).reshape(-1, _NRES)
        caps_a = np.array(caps, dtype=np.int64).reshape(-1, _NRES)
        maxcost = tuple(int(x) for x in costs_a.max(axis=0)) if len(costs) else _ZERO
        e = (ids_a, owner_a, costs_a, caps_a, maxcost)
        if len(self._l1) >= _L1_MAX:
            self._l1.clear()
        self._l1[sk] = e
        return e

    def _bucket(self, s: State, maxcost: tuple) -> tuple:
        cap = self._caps(s.ctx, s.form)
        u = s.used
        return tuple(min(cap[i] - u[i], maxcost[i]) for i in range(_NRES))

    def _mask_id(self, s: State) -> int:
        sk = (s.ctx, s.form, s.sub)
        e = self._l1.get(sk)
        if e is None:
            e = self._l1_entry(s)
        ids, owner, costs, caps, maxcost = e
        cap = self._caps(s.ctx, s.form)
        u = s.used
        k2 = (sk, (min(cap[0] - u[0], maxcost[0]), min(cap[1] - u[1], maxcost[1]), min(cap[2] - u[2], maxcost[2]),
                   min(cap[3] - u[3], maxcost[3]), min(cap[4] - u[4], maxcost[4]), min(cap[5] - u[5], maxcost[5])))
        mid = self._l2.get(k2)
        if mid is None:
            mask = np.zeros(self.V, dtype=bool)
            if len(ids):
                ok = ((costs + np.asarray(u, dtype=np.int64)) <= caps).all(axis=1)
                mask[ids[owner[ok]]] = True
            key = mask.tobytes()
            mid = self._mask_index.get(key)
            if mid is None:
                mid = self._mask_index[key] = len(self._masks)
                mask.setflags(write=False)
                self._masks.append(mask)
            if len(self._l2) >= _L2_MAX:
                self._l2.clear()
            self._l2[k2] = mid
        return mid

    def allowed(self, s: State) -> np.ndarray:
        """bool[V]: the tokens that keep a completion within every budget (read-only array)."""
        return self._masks[self._mask_id(s)]

    def mask_key(self, s: State) -> bytes:
        """sha256 of the fields that determine allowed(s): the interned mode and rules, the form, the structural
        sub-state and min(remaining, max finite cost) per resource. Stable across processes."""
        e = self._l1_entry(s)
        return common.sha256_bytes(repr((self._ctxs[s.ctx], s.form, s.sub, self._bucket(s, e[4]))).encode()).encode()

    def _spell_component(self, s: State):
        """(kind, spelled state, arg) when s is inside a spelled OP, KEY or STR (SYNTAX), else None."""
        pf = s.sub if s.form == F_PIPE else (s.sub[2] if s.form == F_PORT and s.sub[0] == Q_PIPE else None)
        if pf is None:
            return None
        phase, member, lead, _role, node, prof, vst = pf
        if phase == P_OPSP:
            return ("OP", vst, (member, lead))
        if phase == P_KEYSP:
            return ("KEY", vst, node)
        if phase == P_STR and self._vprof(prof)[10] is None:
            return ("STR", vst, None)
        return None

    def _fail_reason(self, s2: State, c) -> str:
        if c is None:
            comp = self._spell_component(s2)
            if comp is not None and self._spell_memo_key(s2.ctx, comp[0], comp[1], comp[2]) in self._spell_limited:
                return "no completion found within the spelled-name search limit (not proven a dead end)"
            return "no completion from this state (dead end)"
        cap = self._caps(s2.ctx, s2.form)
        sets = [frozenset(i for i in range(_NRES) if s2.used[i] + v[i] > cap[i]) for v in c]
        common_bad = frozenset.intersection(*sets)
        if common_bad:
            return "budget exceeded: " + ",".join(RESOURCE_NAMES[i] for i in sorted(common_bad))
        alts = sorted({",".join(RESOURCE_NAMES[i] for i in sorted(x)) for x in sets})
        return "budget exceeded: every completion breaks a cap (" + " | ".join(alts[:4]) + ")"

    def accepts(self, ids: Sequence[int], mode, rules: Rules = Rules()) -> Verdict:
        """Run the full sequence (BOS ... EOS); position is the index of the first refused token."""
        s = self.start(mode, rules)
        for i, t in enumerate(ids):
            try:
                s2 = self.advance(s, t)
            except GrammarError as exc:
                return Verdict(False, i, str(exc))
            c = self._C(s2.ctx, s2.form, s2.sub)
            if not _ffits(s2.used, c, self._caps(s2.ctx, s2.form)):
                return Verdict(False, i, self._fail_reason(s2, c))
            s = s2
        if s.form != F_DONE:
            return Verdict(False, len(ids), "the sequence does not end with EOS")
        return Verdict(True, None, None)

    def compile_masks(self, seqs: Iterable[Sequence[int]], mode, rules: Rules = Rules()) -> tuple:
        """(table bool[M, V], [int32 mask ids per sequence]): masks[i] is the mask token i was chosen under
        (masks[0] = {BOS}); the table holds each distinct mask once, in first-use order. A sequence the mode
        refuses raises GrammarError."""
        local, rows, out = {}, [], []
        for j, ids in enumerate(seqs):
            s = self.start(mode, rules)
            arr = np.empty(len(ids), dtype=np.int32)
            for i, t in enumerate(ids):
                g = self._mask_id(s)
                if not self._masks[g][t]:
                    v = self.accepts(ids, mode, rules)
                    raise GrammarError(f"sequence {j} position {i}: token {t} refused ({v.reason})")
                li = local.get(g)
                if li is None:
                    li = local[g] = len(rows)
                    rows.append(g)
                arr[i] = li
                s = self.advance(s, t)
            if s.form != F_DONE:
                raise GrammarError(f"sequence {j} does not end with EOS")
            out.append(arr)
        table = np.stack([self._masks[g] for g in rows]) if rows else np.zeros((0, self.V), dtype=bool)
        return table, out

    def sample(self, rng: random.Random, mode, rules: Rules = Rules(), *, bias: str = "class") -> list:
        """One random masked walk BOS ... EOS. bias='uniform' draws uniformly over the allowed tokens;
        bias='class' first draws a token class (fixed, byte, OP, KEY, CAT, ID) uniformly, then a token."""
        s = self.start(mode, rules)
        ids = []
        cls = self._tok_class
        while s.form != F_DONE:
            cand = np.flatnonzero(self.allowed(s))
            if not len(cand):
                raise AssertionError(f"dead end after {ids}")
            if bias == "uniform":
                t = int(cand[rng.randrange(len(cand))])
            else:
                groups = {}
                for x in cand.tolist():
                    groups.setdefault(cls[x], []).append(x)
                g = groups[sorted(groups)[rng.randrange(len(groups))]]
                t = g[rng.randrange(len(g))]
            ids.append(t)
            s = self.advance(s, t)
            if len(ids) > self.max_seq:
                raise AssertionError("walk exceeded max_seq")
        return ids

    @property
    def _tok_class(self) -> list:
        tc = getattr(self, "_tok_class_cache", None)
        if tc is None:
            order = {"SPECIAL": 0, "MARKER": 0, "STRUCT": 0, "BYTE": 1, "OP": 2, "KEY": 3, "CAT": 4, "ID": 5}
            tc = [order[self.vocab.cls(i)] for i in range(self.V)]
            self._tok_class_cache = tc
        return tc

    def cache_stats(self) -> dict:
        return {"structural_keys": len(self._l1), "mask_keys": len(self._l2), "distinct_masks": len(self._masks),
                "completion_memo": len(self._cmemo), "spell_memo": len(self._spell_memo),
                "numeral_memo": len(self._num_memo)}

    def clear_caches(self) -> None:
        """Drop every cache except the interned mask table (mask ids stay valid)."""
        for c in (self._cmemo, self._l1, self._l2, self._spell_memo, self._spell_limited, self._num_memo,
                  self._idc_memo):
            c.clear()


# ---------------------------------------------------------------------------------------------
# Walks, gate and CLI
# ---------------------------------------------------------------------------------------------
def walk_check(grammar: Grammar, n: int, *, seed: int = 0, mode=Mode.POLICY, rules: Rules = Rules(),
               bias: str = "class") -> dict:
    """n random masked walks: each must end in EOS within max_seq, detokenize to text Tier A parses, and
    re-tokenize to the same ids; POLICY walks must also pass the evaluator/island caps (oracle.form_caps)."""
    rng = random.Random(seed)
    vocab = grammar.vocab
    stats = Counter()
    lengths = Counter()
    failures = []
    for _ in range(n):
        ids = grammar.sample(rng, mode, rules, bias=bias)
        stats["walks"] += 1
        lengths[len(ids)] += 1
        ok_len = ids[0] == BOS and ids[-1] == EOS and len(ids) <= grammar.max_seq
        text = T.detokenize(ids, vocab)
        try:
            g = _O.parse_genome(text)
            parsed = True
            form = _O.genome_form(g)
        except (_O.CanonicalError, _O.Undecidable) as exc:
            parsed, form = False, "error:" + type(exc).__name__
        stats["forms:" + form] += 1
        retok = T.tokenize(text, vocab) == ids
        caps_ok = (not _O.form_caps(text)) if Mode(mode) is Mode.POLICY else True
        canon = parsed and _O.is_canonical(text)
        stats["eos_within_max_seq"] += ok_len
        stats["tier_a_parsed"] += parsed
        stats["retokenize_identical"] += retok
        stats["caps_ok"] += caps_ok
        stats["tier_a_canonical"] += canon
        stats["accepted_by_mode"] += grammar.accepts(ids, mode, rules).ok
        if Mode(mode) is Mode.POLICY:
            stats["policy_within_syntax"] += grammar.accepts(ids, Mode.SYNTAX, rules).ok
        if not (ok_len and parsed and retok and caps_ok) and len(failures) < 10:
            failures.append({"text_head": text[:200], "parsed": parsed, "retokenize": retok, "caps_ok": caps_ok})
    n_ok = stats["walks"]
    return {"mode": Mode(mode).value, "bias": bias, "seed": seed, **{k: stats[k] for k in sorted(stats)},
            "length_max": max(lengths) if lengths else 0, "length_min": min(lengths) if lengths else 0,
            "failures": failures,
            "pass": bool(n_ok and stats["eos_within_max_seq"] == n_ok and stats["tier_a_parsed"] == n_ok
                         and stats["retokenize_identical"] == n_ok and stats["caps_ok"] == n_ok
                         and stats["accepted_by_mode"] == n_ok
                         and (Mode(mode) is not Mode.POLICY or stats["policy_within_syntax"] == n_ok))}


def load_tok_dir(tok_dir) -> tuple:
    """(vocab, Grammar) from a tokenizer artifact directory (vocab.json, signatures.json, grammar.json,
    lengths.json); refuses artifacts that do not belong together."""
    tok_dir = Path(tok_dir)
    vocab = T.Vocab.from_json((tok_dir / "vocab.json").read_bytes())
    grammar = Grammar.load(tok_dir / "grammar.json", tok_dir / "signatures.json", vocab)
    lengths = json.loads((tok_dir / "lengths.json").read_bytes())
    if lengths.get("vocab_sha256") != vocab.sha256 or int(lengths.get("max_seq", -1)) != grammar.max_seq:
        raise AutomatonStop("lengths.json, vocab.json and grammar.json disagree on vocab_sha256 or max_seq")
    return vocab, grammar


def gate(dict_dir, tok_dir, partitions: Sequence[str] = T.PARTITIONS, *, write: bool = True) -> dict:
    """P4 automaton gate over a P4a program dictionary: POLICY must accept 100% of train_fit (a miss is an
    incomplete table: STOP) and SYNTAX 100% of every partition's programs within max_seq; POLICY acceptance of
    dev/diag/heldout is reported. Writes automaton-gate.json next to the artifacts."""
    dict_dir, tok_dir = Path(dict_dir), Path(tok_dir)
    vocab, grammar = load_tok_dir(tok_dir)
    parts = {}
    for part in partitions:
        if part not in T.PARTITIONS:
            raise ValueError(f"unknown partition {part!r}")
        st = {"programs": 0, "over_max_seq": 0, "policy_accepted": 0, "syntax_accepted": 0,
              "policy_reasons": Counter(), "syntax_reasons": Counter(), "policy_examples": [], "syntax_examples": []}
        for sha, dsl in T._iter_distinct(dict_dir, part):
            ids = T.tokenize(dsl, vocab)
            st["programs"] += 1
            over = len(ids) > grammar.max_seq
            st["over_max_seq"] += over
            for m in (Mode.POLICY, Mode.SYNTAX):
                v = grammar.accepts(ids, m)
                key = m.value
                if v.ok:
                    st[key + "_accepted"] += 1
                else:
                    st[key + "_reasons"][v.reason] += 1
                    if len(st[key + "_examples"]) < 20:
                        st[key + "_examples"].append({"program_sha256": sha, "position": v.position,
                                                      "reason": v.reason})
        n = st["programs"]
        st["policy_reasons"] = dict(st["policy_reasons"].most_common(30))
        st["syntax_reasons"] = dict(st["syntax_reasons"].most_common(30))
        st["policy_acceptance"] = st["policy_accepted"] / n if n else None
        st["syntax_ok"] = st["syntax_accepted"] == n - st["over_max_seq"]
        if part == "train_fit":
            st["pass"] = bool(n and st["policy_accepted"] == n and st["syntax_accepted"] == n)
        else:
            st["pass"] = st["syntax_ok"]
        parts[part] = st
    doc = {"schema": GATE_SCHEMA, "vocab_sha256": vocab.sha256, "max_seq": grammar.max_seq,
           "grammar_sha256": common.sha256_file(tok_dir / "grammar.json"),
           "signatures_sha256": common.sha256_file(tok_dir / "signatures.json"),
           "automaton_sha256": common.sha256_file(__file__), "partitions": parts,
           "pass": all(p["pass"] for p in parts.values()), "utc_us": common.utc_us()}
    if write:
        common.atomic_write(tok_dir / "automaton-gate.json",
                            (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    return doc


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dltrain.automaton", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="POLICY on train_fit and SYNTAX on every partition (automaton-gate.json)")
    g.add_argument("--dict", default=str(T.DEFAULT_DICT))
    g.add_argument("--tok", default=str(T.DEFAULT_TOK))
    g.add_argument("--partitions", nargs="+", choices=T.PARTITIONS, default=list(T.PARTITIONS))
    w = sub.add_parser("walks", help="random masked walks: EOS within max_seq, Tier A parse, unique tokenization")
    w.add_argument("--tok", default=str(T.DEFAULT_TOK))
    w.add_argument("--n", type=int, default=2000)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.POLICY.value)
    w.add_argument("--bias", choices=("class", "uniform"), default="class")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "gate":
            doc = gate(args.dict, args.tok, args.partitions)
            print(json.dumps({"pass": doc["pass"], "partitions": {
                p: {k: v for k, v in d.items() if not k.endswith("_examples")} for p, d in doc["partitions"].items()}},
                sort_keys=True))
            return 0 if doc["pass"] else 1
        _vocab, grammar = load_tok_dir(args.tok)
        doc = walk_check(grammar, args.n, seed=args.seed, mode=args.mode, bias=args.bias)
        print(json.dumps(doc, sort_keys=True))
        return 0 if doc["pass"] else 1
    except AutomatonStop as exc:
        print(json.dumps({"stop": "AutomatonStop", "reason": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
