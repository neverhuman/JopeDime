"""PLAN §5 P4 lossless tokenizer over canonical GenomeIr text (design: P4 "tokenizer" section).

PRINCIPLE. A token is a byte piece and detokenize(ids) is the concatenation of the pieces (BOS, EOS, PAD and
most markers have empty pieces), so losslessness is a property of the segmentation, never of a printer. The
tokenizer never parses or re-prints a number. tokenize() is an offset-preserving port of feat-math's own lexers
run on canonical text: the dsl.rs:212-272 atom rule for pipe and portfolio atoms, and math_expr.rs:339-424 for
sym bodies (through dltrain.oracle.math_lex, the Tier A replica's lexer, so both share one implementation).
Whitespace is Rust `char::is_whitespace` (oracle.RUST_WHITESPACE), never str.isspace (which adds U+001C..U+001F).

ID LAYOUT (vocab.json, schema jope.dl-dsl-vocab/v1), pinned to the vendored feat-math 0.2.0 sources FEAT_MATH_PINS:
  0 PAD, 1 BOS, 2 EOS                         no UNK: the tokenizer is total
  3-12  markers                               OP_BEGIN/OP_END "", KEY_BEGIN " "/KEY_END "=", STR_BEGIN/STR_END "",
                                              ID_BEGIN/ID_END "", RAW_BEGIN/RAW_END "" (RAW is diagnostic only)
  13-25 structural pieces of the printer      PIPE_OPEN "(pipe X", STAGE_OPEN " (", CLOSE ")",
        (dsl.rs:93,105-123; genome.rs:118-143; SYM_AFFINE_OPEN '(sym affine "', SYM_RAW_OPEN '(sym raw "',
         math_expr.rs:131,591-609)           SYM_CLOSE '")', PORTFOLIO_OPEN "(portfolio", MEMBER_OPEN " (member ",
                                              BLEND_AUTO / BLEND_ONE (the whole blend stage plus the final "))"),
                                              TRUE "true", FALSE "false", COL "c"
  26-281 one token per byte 0x00-0xFF         printable ASCII is the spelling, numeral and punctuation alphabet
  282+  data-derived names, from train_fit only, each class sorted by UTF-8 bytes, in class order
        OP (stage op names, portfolio members included), KEY (piece " key="), CAT (bare values that
        parse_param_value types as Str: never numeric, never bool), ID (sym identifiers other than c<N>).

LEXING of canonical text (`lex`, vocabulary independent and total):
- pipe:      PIPE_OPEN {STAGE_OPEN op {KEY value}* CLOSE}+ CLOSE, with the printer's exact spacing. An atom follows
             dsl.rs: it ends at an unquoted White_Space or paren, '"' toggles quoting inside the atom, a backslash
             inside quotes skips one char; it never starts with '"' (a quoted-start atom is never canonical). The
             key is the atom text before the first '=' (the first raw '=' is the first processed '='), and every
             parameter atom must hold one.
- value:     true/false -> TRUE/FALSE; ^-?[0-9]+(\\.[0-9]+)?$ -> one byte token per char (NUM); otherwise a VALUE,
             emitted as CAT when the text is a CAT name, else spelled STR_BEGIN <bytes> STR_END (quoted strings,
             unseen categoricals, inf, -inf, NaN, empty and every other spelling).
- sym:       SYM_*_OPEN body SYM_CLOSE; the body is lexed exactly as math_expr lex does (a lex error is no form):
             White_Space and + - * / ^ ( ) , are byte tokens; numerals [0-9.][0-9.eE]* (+/- only right after e/E)
             are byte tokens; identifiers [A-Za-z_][A-Za-z0-9_]*: c[0-9]+ is COL plus its digit bytes (leading
             zeros verbatim, no column OOV), an ID name is its ID token, anything else is ID_BEGIN <bytes> ID_END.
- portfolio: PORTFOLIO_OPEN {MEMBER_OPEN pipe CLOSE}+ (BLEND_AUTO | BLEND_ONE).
- anything else (leading or trailing whitespace included) matches no form: RAW_BEGIN <bytes> RAW_END, counted.
  Any RAW in the corpus is a STOP.

UNIQUENESS. A spelled form is emitted only for a name absent from its class vocabulary; STR is never spelled for
true/false or numeric text, and ID never for c[0-9]+ (`spelling_allowed` is that rule, for the automaton's *_END
masks). Each text therefore has exactly one token sequence, and tokenize(detokenize(ids)) == ids.

skeleton(dsl) replaces each numeric literal (a pipe value that parse_param_value types Int or Float, or a sym
numeral) with '#', keeping c<N>, categoricals and structure; skeleton_sha256 feeds P5.

CLI (inside the lane slice; reads the P4a program dictionary through dltrain.programs: train_fit under role
fit, the other partitions under the projection role roundtrip):
  build     --dict DIR [--epoch E] --out TOK [--all-partitions] [--walks N] [--no-automaton | --require-automaton]
            (--all-partitions: then the round trip, the automaton gate and N masked walks per mode; all must pass)
  build     --poison-test --dict DIR --out TOK [--seed S]
  roundtrip --dict DIR --tok TOK (--all-partitions | --partitions P ...) [--walks N] [--require-automaton]
            (the round trip plus, when grammar.json is present, the automaton gate and walks)
  show      --vocab vocab.json --dsl TEXT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from dltrain import common
from dltrain import oracle as _O

# ---------------------------------------------------------------------------------------------
# Pins, schemas and paths
# ---------------------------------------------------------------------------------------------
VOCAB_SCHEMA = "jope.dl-dsl-vocab/v1"
LENGTHS_SCHEMA = "jope.dl-dsl-lengths/v1"
ROUNDTRIP_SCHEMA = "jope.dl-dsl-roundtrip/v1"
BUILD_SCHEMA = "jope.dl-dsl-build/v1"
POISON_SCHEMA = "jope.dl-dsl-poison/v1"
# The lexers this module ports (sha256 of the vendored feat-math 0.2.0 sources; dsl.rs and genome.rs are
# byte-identical to crates/feat-math). Vocab.from_json refuses a vocabulary pinned to any other version.
FEAT_MATH_SRC = _O.FEAT_MATH_SRC
FEAT_MATH_PINS = {
    "dsl.rs": "2e16adf644411aa49860baf07d182887196068bdbf71a2c56512623053cb9bf1",
    "genome.rs": "f6db05df29345f508d7ef8270a81dc64e09fd2951e648906f31da91bf99473b7",
    "math_expr.rs": "47bc9ba5a7e7627d998bca5429f1c593fe364a132e84375bc20840e66baf1e57",
}
PARTITIONS = ("train_fit", "dev", "diag", "heldout")
RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
DEFAULT_DICT = RUN / "corpus" / "epoch-000" / "program-dict"
DEFAULT_TOK = RUN / "tokenizer" / "v1"
DEFAULT_EPOCH = RUN / "epochs" / "epoch-000.json"
ARTIFACTS = ("vocab.json", "signatures.json", "grammar.json", "lengths.json")
MAX_SEQ_CAP = 640                 # Full-corpus v2: all qualified teachers; observed maximum 533.
PRODUCT_MAX_PROGRAM_TOKENS = 128  # featmath.json MAX_PROGRAM_TOKENS (reported only)

# ---------------------------------------------------------------------------------------------
# Fixed id block (data independent)
# ---------------------------------------------------------------------------------------------
SPECIALS = ("PAD", "BOS", "EOS")
MARKERS = ("OP_BEGIN", "OP_END", "KEY_BEGIN", "KEY_END", "STR_BEGIN", "STR_END", "ID_BEGIN", "ID_END",
           "RAW_BEGIN", "RAW_END")
BLEND_AUTO_TEXT = " (blend folds=5 guard=clip kind=nnls repeats=auto))"
BLEND_ONE_TEXT = " (blend folds=5 guard=clip kind=nnls repeats=1))"
STRUCTURAL = {
    "PIPE_OPEN": "(pipe X",                 # dsl.rs:117
    "STAGE_OPEN": " (",                     # dsl.rs:119 + dsl.rs:107
    "CLOSE": ")",                           # dsl.rs:111,121; genome.rs:133
    "SYM_AFFINE_OPEN": '(sym affine "',     # genome.rs:122-126
    "SYM_RAW_OPEN": '(sym raw "',
    "SYM_CLOSE": '")',
    "PORTFOLIO_OPEN": "(portfolio",         # genome.rs:129
    "MEMBER_OPEN": " (member ",             # genome.rs:131
    "BLEND_AUTO": BLEND_AUTO_TEXT,          # genome.rs:135-139 with folds=5 (the only value parse_blend admits)
    "BLEND_ONE": BLEND_ONE_TEXT,
    "TRUE": "true",                         # dsl.rs:93 Bool Display
    "FALSE": "false",
    "COL": "c",                             # math_expr.rs:131 Var::Col renders c{i}
}
FIXED: dict = {**{n: b"" for n in SPECIALS},
               "OP_BEGIN": b"", "OP_END": b"", "KEY_BEGIN": b" ", "KEY_END": b"=", "STR_BEGIN": b"",
               "STR_END": b"", "ID_BEGIN": b"", "ID_END": b"", "RAW_BEGIN": b"", "RAW_END": b"",
               **{n: t.encode("utf-8") for n, t in STRUCTURAL.items()}}
FIXED_IDS = {name: i for i, name in enumerate(FIXED)}
PAD, BOS, EOS = FIXED_IDS["PAD"], FIXED_IDS["BOS"], FIXED_IDS["EOS"]
OP_BEGIN, OP_END = FIXED_IDS["OP_BEGIN"], FIXED_IDS["OP_END"]
KEY_BEGIN, KEY_END = FIXED_IDS["KEY_BEGIN"], FIXED_IDS["KEY_END"]
STR_BEGIN, STR_END = FIXED_IDS["STR_BEGIN"], FIXED_IDS["STR_END"]
ID_BEGIN, ID_END = FIXED_IDS["ID_BEGIN"], FIXED_IDS["ID_END"]
RAW_BEGIN, RAW_END = FIXED_IDS["RAW_BEGIN"], FIXED_IDS["RAW_END"]
COL = FIXED_IDS["COL"]
BYTE_BASE = len(FIXED)            # 26
DATA_BASE = BYTE_BASE + 256       # 282
DATA_CLASSES = ("OP", "KEY", "CAT", "ID")
SPELL_CLASSES = {"OP": (OP_BEGIN, OP_END), "KEY": (KEY_BEGIN, KEY_END), "STR": (STR_BEGIN, STR_END),
                 "ID": (ID_BEGIN, ID_END), "RAW": (RAW_BEGIN, RAW_END)}
_BEGIN_CLASS = {b: (c, e) for c, (b, e) in SPELL_CLASSES.items()}
_END_IDS = frozenset(e for _b, e in SPELL_CLASSES.values())


def _fixed_class(name: str) -> str:
    return "SPECIAL" if name in SPECIALS else "MARKER" if name in MARKERS else "STRUCT"


_FIXED_ENTRIES = tuple((_fixed_class(n), n, p) for n, p in FIXED.items())
_BYTE_ENTRIES = tuple(("BYTE", "0x%02x" % b, bytes([b])) for b in range(256))
assert len(FIXED) == 26 and len(_FIXED_ENTRIES) + len(_BYTE_ENTRIES) == DATA_BASE == 282


def byte_id(b: int) -> int:
    return BYTE_BASE + b


# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------
class TokenizerStop(RuntimeError):
    """A P4 STOP condition (RAW or a fallback in train_fit, a round-trip failure, max_seq > 512, ...)."""


class SourcePinError(RuntimeError):
    """The vendored feat-math sources differ from FEAT_MATH_PINS: refuse to build."""


# ---------------------------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------------------------
_WS = frozenset(_O.RUST_WHITESPACE)
_WS_CLASS = "".join("\\u%04x" % ord(c) for c in _O.RUST_WHITESPACE)
_ATOM_FAST_RE = re.compile('[^()"' + _WS_CLASS + "]+")       # an atom in a text without any '"'
_NUM_RE = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")               # pipe numeric value (fullmatch)
_COL_RE = re.compile(r"c[0-9]+")                              # sym column (fullmatch)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")             # math_expr identifier (fullmatch)
_NEEDS_QUOTES_RE = re.compile("[" + _WS_CLASS + '()"=]')      # dsl.rs:374-379 needs_quotes (non-empty case)

_P_PIPE = STRUCTURAL["PIPE_OPEN"]
_P_PORT = STRUCTURAL["PORTFOLIO_OPEN"]
_P_MEMBER = STRUCTURAL["MEMBER_OPEN"]
_P_AFFINE = STRUCTURAL["SYM_AFFINE_OPEN"]
_P_RAW = STRUCTURAL["SYM_RAW_OPEN"]
_N_PIPE, _N_PORT, _N_MEMBER = len(_P_PIPE), len(_P_PORT), len(_P_MEMBER)

# Lexeme kinds. Structural kinds map to one fixed id; byte kinds to one byte token per UTF-8 byte; a COL lexeme
# (c<N>) is the COL token followed by its digit bytes.
STRUCT_KINDS = tuple(k for k in STRUCTURAL if k != "COL")
BYTE_KINDS = ("NUM", "NUMERAL", "WS", "PUNCT")
NAME_KINDS = ("OP", "KEY", "VALUE", "IDENT", "COL")
LEXEME_KINDS = STRUCT_KINDS + BYTE_KINDS + NAME_KINDS + ("RAW",)
_KIND_FIXED = {k: FIXED_IDS[k] for k in STRUCT_KINDS}
_BYTE_KIND_SET = frozenset(BYTE_KINDS)


@dataclass(frozen=True)
class Lexeme:
    """One lexeme of `lex(dsl)`; `text == dsl[start:end]` always (offsets are str indices), and the lexemes of a
    text tile it exactly. A KEY lexeme spans its whole piece " key=" (`name` is the key)."""

    kind: str
    text: str
    start: int
    end: int

    @property
    def name(self) -> str:
        return self.text[1:-1] if self.kind == "KEY" else self.text


class _NoForm(Exception):
    pass


def _scan_atom(s: str, pos: int, quotes: bool) -> int:
    """End of the dsl.rs:242-265 atom starting at `pos` (the `_` branch: the first char is not '(' ')' '"' or
    White_Space). In-atom '"' toggles quoting, a backslash inside quotes skips one char; an unterminated quote or
    escape is a lex error. Raises _NoForm for an empty atom, a quoted-start atom or a lex error."""
    if not quotes:
        m = _ATOM_FAST_RE.match(s, pos)
        if m is None:
            raise _NoForm
        return m.end()
    n = len(s)
    if pos >= n:
        raise _NoForm
    ch = s[pos]
    if ch == "(" or ch == ")" or ch == '"' or ch in _WS:
        raise _NoForm
    i = pos + 1
    in_quote = False
    while i < n:
        c = s[i]
        if not in_quote and (c == "(" or c == ")" or c in _WS):
            break
        if c == '"':
            in_quote = not in_quote
        elif c == "\\" and in_quote:
            i += 2                      # skip the backslash and push the next char verbatim
            if i > n:
                raise _NoForm           # "unterminated escape in quoted atom"
            continue
        i += 1
    if in_quote:
        raise _NoForm                   # "unterminated quoted parameter"
    return i


def _value_kind(s: str, a: int, b: int) -> str:
    if b - a == 4 and s.startswith("true", a):
        return "TRUE"
    if b - a == 5 and s.startswith("false", a):
        return "FALSE"
    if _NUM_RE.fullmatch(s, a, b):
        return "NUM"
    return "VALUE"


def _lex_pipe(s: str, pos: int, out: list, quotes: bool) -> int:
    """PIPE_OPEN {STAGE_OPEN op {KEY value}* CLOSE}+ CLOSE starting at `pos` (which holds "(pipe X")."""
    out.append(("PIPE_OPEN", pos, pos + _N_PIPE))
    pos += _N_PIPE
    n = len(s)
    stages = 0
    while True:
        if s.startswith(" (", pos):
            out.append(("STAGE_OPEN", pos, pos + 2))
            pos += 2
            end = _scan_atom(s, pos, quotes)
            out.append(("OP", pos, end))
            pos = end
            while True:
                c = s[pos] if pos < n else ""
                if c == " ":
                    a = pos + 1
                    end = _scan_atom(s, a, quotes)
                    eq = s.find("=", a, end)
                    if eq < 0:
                        raise _NoForm   # "parameter must be key=value"
                    out.append(("KEY", pos, eq + 1))
                    out.append((_value_kind(s, eq + 1, end), eq + 1, end))
                    pos = end
                elif c == ")":
                    out.append(("CLOSE", pos, pos + 1))
                    pos += 1
                    break
                else:
                    raise _NoForm
            stages += 1
        elif stages and pos < n and s[pos] == ")":
            out.append(("CLOSE", pos, pos + 1))
            return pos + 1
        else:
            raise _NoForm


def _lex_portfolio(s: str, out: list, quotes: bool) -> None:
    out.append(("PORTFOLIO_OPEN", 0, _N_PORT))
    pos = _N_PORT
    members = 0
    while s.startswith(_P_MEMBER, pos):
        out.append(("MEMBER_OPEN", pos, pos + _N_MEMBER))
        pos += _N_MEMBER
        if not s.startswith(_P_PIPE, pos):
            raise _NoForm
        pos = _lex_pipe(s, pos, out, quotes)
        if not s.startswith(")", pos):
            raise _NoForm
        out.append(("CLOSE", pos, pos + 1))
        pos += 1
        members += 1
    rest = len(s) - pos
    if members and rest == len(BLEND_AUTO_TEXT) and s.startswith(BLEND_AUTO_TEXT, pos):
        out.append(("BLEND_AUTO", pos, len(s)))
    elif members and rest == len(BLEND_ONE_TEXT) and s.startswith(BLEND_ONE_TEXT, pos):
        out.append(("BLEND_ONE", pos, len(s)))
    else:
        raise _NoForm


def _lex_sym(s: str, out: list, kind: str, open_len: int) -> None:
    n = len(s)
    if n < open_len + 2 or not s.endswith('")'):
        raise _NoForm
    body_end = n - 2
    body = s[open_len:body_end]
    try:
        toks = _O.math_lex(body)       # math_expr.rs:339-424 (shared with the Tier A replica)
    except _O.CanonicalError:
        raise _NoForm from None         # a '"' in the body is an "unexpected character" too
    out.append((kind, 0, open_len))
    p = 0
    for tkind, _value, a, b in toks:
        if tkind == "end":
            break
        while p < a:                    # the lexer skipped only White_Space here
            out.append(("WS", open_len + p, open_len + p + 1))
            p += 1
        if tkind == "num":
            out.append(("NUMERAL", open_len + a, open_len + b))
        elif tkind == "id":
            out.append(("COL" if _COL_RE.fullmatch(body, a, b) else "IDENT", open_len + a, open_len + b))
        else:
            out.append(("PUNCT", open_len + a, open_len + b))
        p = b
    while p < len(body):
        out.append(("WS", open_len + p, open_len + p + 1))
        p += 1
    out.append(("SYM_CLOSE", body_end, n))


def _lex_spans(dsl: str) -> list:
    """(kind, start, end) triples tiling `dsl`; [("RAW", 0, len)] when the text matches no form."""
    if not isinstance(dsl, str):
        raise TypeError("program text must be str")
    if not dsl.isascii():
        dsl.encode("utf-8")   # a lone surrogate is not a Unicode scalar value: UnicodeEncodeError (a ValueError)
    out = []
    try:
        if dsl.startswith(_P_PIPE):
            if _lex_pipe(dsl, 0, out, '"' in dsl) != len(dsl):
                raise _NoForm
        elif dsl.startswith(_P_PORT):
            _lex_portfolio(dsl, out, '"' in dsl)
        elif dsl.startswith(_P_AFFINE):
            _lex_sym(dsl, out, "SYM_AFFINE_OPEN", len(_P_AFFINE))
        elif dsl.startswith(_P_RAW):
            _lex_sym(dsl, out, "SYM_RAW_OPEN", len(_P_RAW))
        else:
            raise _NoForm
    except _NoForm:
        return [("RAW", 0, len(dsl))]
    return out


def lex(dsl: str) -> list:
    """Vocabulary-independent, total lexing of canonical text; one RAW lexeme only when no form matches."""
    return [Lexeme(k, dsl[a:b], a, b) for k, a, b in _lex_spans(dsl)]


def program_form(dsl: str) -> str:
    """pipe | sym | portfolio | raw, as the lexer sees the text."""
    kind = _lex_spans(dsl)[0][0]
    return {"PIPE_OPEN": "pipe", "SYM_AFFINE_OPEN": "sym", "SYM_RAW_OPEN": "sym",
            "PORTFOLIO_OPEN": "portfolio"}.get(kind, "raw")


def is_cat_name(value: str) -> bool:
    """A CAT name: a bare value that parse_param_value types as Str and that the printer emits unquoted, so never
    empty, quoted, numeric (inf/NaN included) or bool, and free of White_Space ( ) " and =."""
    return (isinstance(value, str) and value != "" and not _NEEDS_QUOTES_RE.search(value)
            and _O.parse_param_value(value)[0] == "Str")


def _is_op_name(name: str) -> bool:
    spans = _lex_spans("(pipe X (" + name + "))") if isinstance(name, str) else []
    return len(spans) == 5 and spans[2][0] == "OP" and spans[2][1:] == (9, 9 + len(name))


def _is_key_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    spans = _lex_spans("(pipe X (o " + name + "=0))")
    return len(spans) == 7 and spans[3][0] == "KEY" and spans[3][1:] == (10, 12 + len(name))


def _is_id_name(name: str) -> bool:
    return isinstance(name, str) and bool(_IDENT_RE.fullmatch(name)) and not _COL_RE.fullmatch(name)


_NAME_CHECKS = {"OP": _is_op_name, "KEY": _is_key_name, "CAT": is_cat_name, "ID": _is_id_name}


def name_piece(cls: str, name: str) -> bytes:
    """The byte piece of a data-derived name token."""
    if cls == "KEY":
        return (" " + name + "=").encode("utf-8")
    return name.encode("utf-8")


# ---------------------------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------------------------
def _table(entries: Sequence) -> list:
    return [{"class": c, "name": n, "piece_hex": p.hex()} for c, n, p in entries]


def _table_sha256(entries: Sequence) -> str:
    return common.sha256_bytes(common.canonical(_table(entries)))


def _utf8_key(name: str) -> bytes:
    return name.encode("utf-8")


@dataclass(frozen=True)
class Vocab:
    """(class, name, piece) per id. The fixed block (0-281) is this module's; the data block holds the OP, KEY,
    CAT and ID names, each class sorted by UTF-8 bytes and validated (it must lex back as that class).
    `sha256` is vocab_sha256: the sha256 of the canonical JSON table [{class, name, piece_hex}] in id order."""

    entries: tuple
    sha256: str
    _index: dict = field(init=False, repr=False, compare=False)
    _pieces: tuple = field(init=False, repr=False, compare=False)
    _ranges: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        entries = tuple((str(c), str(n), bytes(p)) for c, n, p in self.entries)
        if entries[:DATA_BASE] != _FIXED_ENTRIES + _BYTE_ENTRIES:
            raise ValueError("vocab fixed block (ids 0-281) differs from this tokenizer's layout")
        ranges = {"SPECIAL": (0, 3), "MARKER": (3, 13), "STRUCT": (13, BYTE_BASE), "BYTE": (BYTE_BASE, DATA_BASE)}
        pos = DATA_BASE
        for cls in DATA_CLASSES:
            start = pos
            prev = None
            while pos < len(entries) and entries[pos][0] == cls:
                _c, name, piece = entries[pos]
                if not _NAME_CHECKS[cls](name):
                    raise ValueError(f"vocab id {pos}: {name!r} is not a valid {cls} name")
                if piece != name_piece(cls, name):
                    raise ValueError(f"vocab id {pos}: piece does not match {cls} name {name!r}")
                key = _utf8_key(name)
                if prev is not None and key <= prev:
                    raise ValueError(f"vocab class {cls} is not strictly sorted by UTF-8 bytes at id {pos}")
                prev = key
                pos += 1
            ranges[cls] = (start, pos)
        if pos != len(entries):
            raise ValueError(f"vocab id {pos}: unexpected class {entries[pos][0]!r} (expected OP, KEY, CAT, ID order)")
        got = _table_sha256(entries)
        if got != self.sha256:
            raise ValueError(f"vocab_sha256 mismatch: table hashes to {got}, recorded {self.sha256}")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_index", {(c, n): i for i, (c, n, _p) in enumerate(entries)})
        object.__setattr__(self, "_pieces", tuple(p for _c, _n, p in entries))
        object.__setattr__(self, "_ranges", ranges)

    @classmethod
    def from_names(cls, names: Mapping[str, Iterable[str]]) -> "Vocab":
        """A vocabulary over the given data-derived names ({"OP": [...], "KEY": [...], "CAT": [...], "ID": [...]})."""
        unknown = set(names) - set(DATA_CLASSES)
        if unknown:
            raise ValueError(f"unknown name classes {sorted(unknown)}")
        entries = list(_FIXED_ENTRIES + _BYTE_ENTRIES)
        for c in DATA_CLASSES:
            for n in sorted(set(names.get(c, ())), key=_utf8_key):
                entries.append((c, n, name_piece(c, n)))
        entries = tuple(entries)
        return cls(entries, _table_sha256(entries))

    @property
    def size(self) -> int:
        return len(self.entries)

    def id(self, cls: str, name: str) -> int | None:
        return self._index.get((cls, name))

    def piece(self, i: int) -> bytes:
        i = operator.index(i)
        if not 0 <= i < len(self._pieces):
            raise ValueError(f"token id {i} outside 0..{len(self._pieces) - 1}")
        return self._pieces[i]

    def cls(self, i: int) -> str:
        self.piece(i)
        return self.entries[i][0]

    def name(self, i: int) -> str:
        self.piece(i)
        return self.entries[i][1]

    def class_range(self, cls: str) -> tuple:
        """[start, end) of a class: SPECIAL, MARKER, STRUCT, BYTE, OP, KEY, CAT or ID."""
        return self._ranges[cls]

    def names(self, cls: str) -> tuple:
        a, b = self._ranges[cls]
        return tuple(n for _c, n, _p in self.entries[a:b])

    def to_json(self, extra: Mapping | None = None) -> bytes:
        """vocab.json: the schema, the feat-math pins, the class ranges, the table and vocab_sha256, plus any
        `extra` sections (counts, input closure, build parameters). Deterministic bytes."""
        doc = {"schema": VOCAB_SCHEMA, "feat_math_pins": dict(FEAT_MATH_PINS), "size": self.size,
               "class_ranges": {c: list(r) for c, r in self._ranges.items()},
               "fixed_layout": {"specials": list(SPECIALS), "markers": list(MARKERS),
                                "structural": list(STRUCTURAL), "byte_base": BYTE_BASE, "data_base": DATA_BASE},
               "tokens": _table(self.entries), "vocab_sha256": self.sha256}
        for k, v in (extra or {}).items():
            if k in doc:
                raise ValueError(f"extra section {k!r} would overwrite a reserved vocab.json key")
            doc[k] = v
        return _json_bytes(doc)

    @classmethod
    def from_json(cls, raw: bytes) -> "Vocab":
        """Parse and fully re-validate vocab.json; refuses another schema, other feat-math pins, a different fixed
        block, an unsorted or invalid name, inconsistent class ranges or a vocab_sha256 mismatch."""
        doc = json.loads(raw)
        if not isinstance(doc, dict) or doc.get("schema") != VOCAB_SCHEMA:
            raise ValueError(f"not a {VOCAB_SCHEMA} document")
        if doc.get("feat_math_pins") != FEAT_MATH_PINS:
            raise ValueError("vocab.json is pinned to other feat-math sources; this tokenizer refuses it")
        toks = doc.get("tokens")
        if not isinstance(toks, list):
            raise ValueError("vocab.json has no token table")
        entries = []
        for i, t in enumerate(toks):
            if (not isinstance(t, dict) or set(t) != {"class", "name", "piece_hex"}
                    or not all(isinstance(t[k], str) for k in t)):
                raise ValueError(f"vocab.json token {i} is malformed")
            entries.append((t["class"], t["name"], bytes.fromhex(t["piece_hex"])))
        v = cls(tuple(entries), doc.get("vocab_sha256"))
        if doc.get("size") != v.size or doc.get("class_ranges") != {c: list(r) for c, r in v._ranges.items()}:
            raise ValueError("vocab.json size or class ranges disagree with its table")
        return v


def _json_bytes(doc) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _name_units(dsl: str, spans: list):
    """(class, name) for each data-name occurrence: OP, KEY, CAT (VALUEs that are CAT names) and ID."""
    for kind, a, b in spans:
        if kind == "OP":
            yield "OP", dsl[a:b]
        elif kind == "KEY":
            yield "KEY", dsl[a + 1:b - 1]
        elif kind == "VALUE":
            v = dsl[a:b]
            if is_cat_name(v):
                yield "CAT", v
        elif kind == "IDENT":
            yield "ID", dsl[a:b]


def build_vocab(train_fit_dsls: Iterable[str], *, min_count: int = 1) -> tuple:
    """Build the vocabulary from train_fit programs only (pass DISTINCT programs; order never matters).

    Returns (Vocab, counts) with counts[class][name] = occurrences in the input (min_count filters on it).
    A program that matches no form (RAW) is a STOP: TokenizerStop."""
    if type(min_count) is not int or min_count < 1:
        raise ValueError("min_count must be an int >= 1")
    counts = {c: Counter() for c in DATA_CLASSES}
    programs = 0
    for dsl in train_fit_dsls:
        spans = _lex_spans(dsl)
        if spans[0][0] == "RAW":
            raise TokenizerStop(f"train_fit program sha256={_O.program_sha256(dsl)} matches no form (RAW)")
        programs += 1
        for c, name in _name_units(dsl, spans):
            counts[c][name] += 1
    names = {c: [n for n, k in counts[c].items() if k >= min_count] for c in DATA_CLASSES}
    for c in DATA_CLASSES:
        for n in names[c]:
            if not _NAME_CHECKS[c](n):
                # e.g. a key holding an odd number of '"' (canonical: Rust's key is the atom text before the first
                # '='), which lexes back as a key only together with the value that closes its quote
                raise TokenizerStop(f"train_fit {c} name {n!r} cannot be a vocabulary token (it does not lex back "
                                    f"as a {c} on its own)")
    vocab = Vocab.from_names(names)
    return vocab, {c: {n: counts[c][n] for n in sorted(counts[c], key=_utf8_key)} for c in DATA_CLASSES}


# ---------------------------------------------------------------------------------------------
# Tokenize / detokenize
# ---------------------------------------------------------------------------------------------
def _byte_ids(text: str) -> list:
    return [BYTE_BASE + b for b in text.encode("utf-8")]


def tokenize(dsl: str, vocab: Vocab) -> list:
    """BOS, the unique token sequence of `dsl`, EOS. Total over str without lone surrogates."""
    index = vocab._index
    ids = [BOS]
    for kind, a, b in _lex_spans(dsl):
        fid = _KIND_FIXED.get(kind)
        if fid is not None:
            ids.append(fid)
        elif kind in _BYTE_KIND_SET:
            ids.extend(_byte_ids(dsl[a:b]))
        elif kind == "OP":
            name = dsl[a:b]
            i = index.get(("OP", name))
            if i is None:
                ids.append(OP_BEGIN)
                ids.extend(_byte_ids(name))
                ids.append(OP_END)
            else:
                ids.append(i)
        elif kind == "KEY":
            name = dsl[a + 1:b - 1]
            i = index.get(("KEY", name))
            if i is None:
                ids.append(KEY_BEGIN)
                ids.extend(_byte_ids(name))
                ids.append(KEY_END)
            else:
                ids.append(i)
        elif kind == "VALUE":
            text = dsl[a:b]
            i = index.get(("CAT", text))
            if i is None:
                ids.append(STR_BEGIN)
                ids.extend(_byte_ids(text))
                ids.append(STR_END)
            else:
                ids.append(i)
        elif kind == "COL":
            ids.append(COL)
            ids.extend(_byte_ids(dsl[a + 1:b]))
        elif kind == "IDENT":
            name = dsl[a:b]
            i = index.get(("ID", name))
            if i is None:
                ids.append(ID_BEGIN)
                ids.extend(_byte_ids(name))
                ids.append(ID_END)
            else:
                ids.append(i)
        else:  # RAW: the whole text
            ids.append(RAW_BEGIN)
            ids.extend(_byte_ids(dsl))
            ids.append(RAW_END)
    ids.append(EOS)
    return ids


def detokenize(ids: Sequence[int], vocab: Vocab) -> str:
    """Concatenate the pieces (PAD, BOS, EOS and the empty markers contribute nothing) and decode strict UTF-8."""
    pieces = vocab._pieces
    n = len(pieces)
    buf = []
    for i in ids:
        i = operator.index(i)
        if not 0 <= i < n:
            raise ValueError(f"token id {i} outside 0..{n - 1}")
        buf.append(pieces[i])
    return b"".join(buf).decode("utf-8")


def fallback_spans(ids: Sequence[int], vocab: Vocab) -> list:
    """[(class, start, end)] for every spelled span: class in OP | KEY | STR | ID | RAW, [start, end) over token
    positions from *_BEGIN through *_END. Raises ValueError on a malformed spelling."""
    out = []
    n = len(ids)
    size = vocab.size
    i = 0
    while i < n:
        t = operator.index(ids[i])
        if not 0 <= t < size:
            raise ValueError(f"token id {t} outside 0..{size - 1}")
        hit = _BEGIN_CLASS.get(t)
        if hit is None:
            if t in _END_IDS:
                raise ValueError(f"position {i}: *_END without its *_BEGIN")
            i += 1
            continue
        cls, end_tok = hit
        j = i + 1
        while j < n and ids[j] != end_tok:
            if not BYTE_BASE <= ids[j] < DATA_BASE:
                raise ValueError(f"position {j}: a {cls} spelling may hold byte tokens only")
            j += 1
        if j >= n:
            raise ValueError(f"position {i}: unterminated {cls} spelling")
        out.append((cls, i, j + 1))
        i = j + 1
    return out


def spelled_text(ids: Sequence[int], span: tuple, vocab: Vocab) -> str:
    """The text a spelled span stands for (a KEY span's key, without the ' ' and '=' marker pieces)."""
    _cls, a, b = span
    return bytes(operator.index(i) - BYTE_BASE for i in ids[a + 1:b - 1]).decode("utf-8")


def is_oov_span(cls: str, text: str) -> bool:
    """True for an out-of-vocabulary fallback: a spelled OP, KEY or ID name, a spelled STR whose text is a CAT name
    (it would be a CAT token had train_fit held it), or RAW. A STR spelling of a value that can never be a CAT name
    (quoted strings, NaN, inf, -inf, other non-CAT spellings) is a literal value, not a fallback."""
    return cls != "STR" or is_cat_name(text)


def oov_spans(ids: Sequence[int], vocab: Vocab) -> list:
    """The fallback_spans that are out-of-vocabulary fallbacks (PLAN: OOV is measured as the fallback rate)."""
    return [s for s in fallback_spans(ids, vocab) if is_oov_span(s[0], spelled_text(ids, s, vocab))]


def spelling_allowed(cls: str, text: str, vocab: Vocab) -> bool:
    """The uniqueness rule behind the automaton's *_END masks: True iff the tokenizer would spell `text` in class
    `cls` (OP, KEY, STR or ID) instead of emitting a name, TRUE/FALSE, numeral bytes or COL."""
    if cls == "OP":
        return vocab.id("OP", text) is None
    if cls == "KEY":
        return vocab.id("KEY", text) is None
    if cls == "STR":
        return text not in ("true", "false") and not _NUM_RE.fullmatch(text) and vocab.id("CAT", text) is None
    if cls == "ID":
        return not _COL_RE.fullmatch(text) and vocab.id("ID", text) is None
    raise ValueError(f"unknown spelling class {cls!r}")


def token_names(ids: Sequence[int], vocab: Vocab) -> list:
    """Readable names (debugging): fixed names, 'b:x' for printable bytes, 'b:0x..' otherwise, CLASS:name."""
    out = []
    for i in ids:
        c, n, p = vocab.entries[operator.index(i)]
        if c == "BYTE":
            out.append("b:" + (p.decode("ascii") if 0x20 <= p[0] < 0x7F else n))
        elif c in DATA_CLASSES:
            out.append(c + ":" + n)
        else:
            out.append(n)
    return out


# ---------------------------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------------------------
def skeleton(dsl: str) -> str:
    """Each numeric literal (a pipe value typed Int or Float by parse_param_value, or a sym numeral) becomes '#';
    c<N>, categoricals, bools and structure are kept. Collapsing Int and Float spellings happens only here."""
    parts = []
    for kind, a, b in _lex_spans(dsl):
        if kind == "NUM" or kind == "NUMERAL" or (kind == "VALUE" and _O.parse_param_value(dsl[a:b])[0] in ("Int", "Float")):
            parts.append("#")
        else:
            parts.append(dsl[a:b])
    return "".join(parts)


def skeleton_sha256(dsl: str) -> str:
    return common.sha256_bytes(skeleton(dsl).encode("utf-8"))


# ---------------------------------------------------------------------------------------------
# Corpus statistics
# ---------------------------------------------------------------------------------------------
_QUANTILES = (("p50", "0.5"), ("p90", "0.9"), ("p99", "0.99"), ("p99_9", "0.999"))


def _hist_summary(hist: Counter) -> dict:
    """Nearest-rank quantiles, max and mean of an integer histogram (exact, deterministic)."""
    n = sum(hist.values())
    if n == 0:
        return {"programs": 0}
    items = sorted(hist.items())
    out = {"programs": n}
    for name, q in _QUANTILES:
        rank = max(1, math.ceil(Fraction(q) * n))
        acc = 0
        for v, c in items:
            acc += c
            if acc >= rank:
                out[name] = v
                break
    out["max"] = items[-1][0]
    out["min"] = items[0][0]
    out["mean"] = round(sum(v * c for v, c in items) / n, 4)
    out["histogram"] = [[v, c] for v, c in items]
    return out


def ceil32(n: int) -> int:
    return -(-n // 32) * 32


class TokenStats:
    """Streaming tokenization statistics: per-id counts, lengths by form, RAW, OOV fallbacks (spelled names) and
    STR literals (spelled values that are never CAT names), and the round-trip checks: detokenize(tokenize(dsl)) ==
    dsl and its sha256 (check=True), plus Tier A canonical(text) == text (canonical=True, the design's round-trip
    step 3). There is no tokenize(detokenize(ids)) == ids counter: for ids that came from tokenize it cannot fail
    once the text round-trips; the automaton walks check retokenization of generated sequences instead."""

    def __init__(self, vocab: Vocab, *, check: bool = True, max_seq: int | None = None, canonical: bool = False) -> None:
        self.vocab = vocab
        self.check = check
        self.canonical = canonical and check
        self.max_seq = max_seq
        self.programs = 0
        self.tokens = 0
        self.per_id = [0] * vocab.size
        self.lengths = {"pipe": Counter(), "sym": Counter(), "portfolio": Counter(), "raw": Counter()}
        self.dsl_bytes_max = 0
        self.raw_programs = 0
        self.fallback_programs = 0           # programs with >= 1 OOV fallback span
        self.fallback_spans = Counter()      # OOV fallback spans by class
        self.str_literal_programs = 0
        self.str_literal_spans = 0
        self.roundtrip_ok = self.sha_ok = self.canonical_ok = self.canonical_undecidable = 0
        self.over_max_seq = 0
        self.over_product = 0
        self.failures = []

    def _fail(self, kind: str, sha: str) -> None:
        if len(self.failures) < 20:
            self.failures.append({"kind": kind, "program_sha256": sha})

    def add(self, dsl: str, sha: str | None = None) -> list:
        ids = tokenize(dsl, self.vocab)
        self.programs += 1
        self.tokens += len(ids)
        per = self.per_id
        for i in ids:
            per[i] += 1
        form = _FORM_OF.get(ids[1], "raw")
        self.lengths[form][len(ids)] += 1
        nbytes = len(dsl.encode("utf-8"))
        if nbytes > self.dsl_bytes_max:
            self.dsl_bytes_max = nbytes
        if len(ids) > PRODUCT_MAX_PROGRAM_TOKENS:
            self.over_product += 1
        if self.max_seq is not None and len(ids) > self.max_seq:
            self.over_max_seq += 1
        if ids[1] == RAW_BEGIN:
            self.raw_programs += 1
        oov = literal = 0
        for span in fallback_spans(ids, self.vocab):
            if is_oov_span(span[0], spelled_text(ids, span, self.vocab)):
                self.fallback_spans[span[0]] += 1
                oov += 1
            else:
                literal += 1
        if oov:
            self.fallback_programs += 1
        if literal:
            self.str_literal_programs += 1
            self.str_literal_spans += literal
        if self.check:
            sha = sha if sha is not None else _O.program_sha256(dsl)
            text = detokenize(ids, self.vocab)
            if text == dsl:
                self.roundtrip_ok += 1
            else:
                self._fail("roundtrip", sha)
            if _O.program_sha256(text) == sha:
                self.sha_ok += 1
            else:
                self._fail("sha256", sha)
            if self.canonical:
                try:
                    canon = _O.canonical(text) == text
                except _O.Undecidable:
                    canon = None                  # Tier B decides it in corpus mode (oracle tier-a --corpus)
                except _O.CanonicalError:
                    canon = False
                if canon:
                    self.canonical_ok += 1
                elif canon is None:
                    self.canonical_undecidable += 1
                else:
                    self._fail("tier_a_canonical", sha)
        return ids

    def max_length(self) -> int:
        return max((max(h) for h in self.lengths.values() if h), default=0)

    def lengths_doc(self) -> dict:
        allh = Counter()
        for h in self.lengths.values():
            allh.update(h)
        by_form = {"all": _hist_summary(allh)}
        for f in ("pipe", "sym", "portfolio", "raw"):
            by_form[f] = _hist_summary(self.lengths[f])
        return by_form

    def ok(self) -> bool:
        n = self.programs
        return (self.raw_programs == 0 and (not self.check or (self.roundtrip_ok == n and self.sha_ok == n
                                                                   and (not self.canonical
                                                                        or self.canonical_ok + self.canonical_undecidable == n))))

    def summary(self) -> dict:
        n = self.programs
        doc = {"programs": n, "tokens": self.tokens, "raw_programs": self.raw_programs,
               "fallback_programs": self.fallback_programs,
               "fallback_rate": (self.fallback_programs / n) if n else 0.0,
               "fallback_spans": {c: self.fallback_spans.get(c, 0) for c in SPELL_CLASSES},
               "str_literal_programs": self.str_literal_programs, "str_literal_spans": self.str_literal_spans,
               "over_product_max_program_tokens": self.over_product, "dsl_bytes_max": self.dsl_bytes_max,
               "length": self.lengths_doc()}
        if self.max_seq is not None:
            doc["over_max_seq"] = self.over_max_seq
        if self.check:
            doc.update(roundtrip_ok=self.roundtrip_ok, sha_ok=self.sha_ok, failures=list(self.failures))
        if self.canonical:
            doc["tier_a_canonical_ok"] = self.canonical_ok
            doc["tier_a_undecidable"] = self.canonical_undecidable
        doc["pass"] = self.ok()
        return doc


_FORM_OF = {FIXED_IDS["PIPE_OPEN"]: "pipe", FIXED_IDS["SYM_AFFINE_OPEN"]: "sym", FIXED_IDS["SYM_RAW_OPEN"]: "sym",
            FIXED_IDS["PORTFOLIO_OPEN"]: "portfolio"}


# ---------------------------------------------------------------------------------------------
# Artifacts over the P4a program dictionary
# ---------------------------------------------------------------------------------------------
def verify_feat_math_sources(src: Path | str = FEAT_MATH_SRC) -> dict:
    """sha256 of the vendored dsl.rs, genome.rs and math_expr.rs; raises SourcePinError unless they equal the pins."""
    got = {name: common.sha256_file(Path(src) / name) for name in FEAT_MATH_PINS}
    if got != FEAT_MATH_PINS:
        bad = sorted(n for n in FEAT_MATH_PINS if got[n] != FEAT_MATH_PINS[n])
        raise SourcePinError(f"feat-math sources differ from the tokenizer pins: {bad}")
    return got


def _programs_module():
    from dltrain import programs as P   # pyarrow: imported only when a dictionary is read
    return P


def _iter_distinct(dict_dir: Path, partition: str, role: str | None = None):
    """(program_sha256, dsl) of one partition: fit for train_fit (the vocabulary's only input), else the projection
    role 'roundtrip' (programs: never the heldout_report role, which belongs to P10)."""
    P = _programs_module()
    return P.iter_distinct(dict_dir, partition, role=role or ("fit" if partition == "train_fit" else "roundtrip"))


def _automaton_api(mode):
    """(build_signatures, build_grammar) from dltrain.automaton when present; mode None=auto, False=skip, True=require."""
    if mode is False:
        return None
    try:
        from dltrain import automaton as A
    except ImportError:
        A = None
    api = (getattr(A, "build_signatures", None), getattr(A, "build_grammar", None)) if A is not None else (None, None)
    if callable(api[0]) and callable(api[1]):
        return api
    if mode is True:
        raise TokenizerStop("dltrain.automaton.build_signatures/build_grammar are not available")
    return None


def _manifest_closure(dict_dir: Path, partition: str) -> dict:
    """File-level input closure of one partition (paths relative to the dictionary): chunk bytes and sha256."""
    doc = json.loads((dict_dir / "manifest.json").read_bytes())
    files = []
    for c in doc["partitions"][partition]["chunks"]:
        p = dict_dir / c["path"]
        files.append({"path": c["path"], "bytes": p.stat().st_size, "sha256": common.sha256_file(p)})
    return {"manifest_sha256": common.sha256_file(dict_dir / "manifest.json"), "files": files}


def build_artifacts(dict_dir: Path | str, epoch: Path | str | None, out_dir: Path | str, *,
                    automaton: bool | None = None, min_count: int = 1) -> dict:
    """vocab.json and lengths.json (plus signatures.json and grammar.json through dltrain.automaton when present)
    from the DISTINCT train_fit programs of a P4a dictionary only. No other partition is opened.

    The artifacts are pure functions of the train_fit program set: row order and chunking never change their bytes.
    vocab.json therefore records a content-defined input closure (distinct programs and the sha256 of their sorted
    program_sha256 list); the file-level closure (each train_fit chunk's sha256), the epoch pin and the verified
    source pins go to build-receipt.json. STOPs (TokenizerStop): RAW, a fallback in train_fit with min_count=1,
    sha256(dsl) != program_sha256, an empty train_fit, max_seq > 512."""
    dict_dir, out_dir = Path(dict_dir), Path(out_dir)
    sources = verify_feat_math_sources()
    vocab, name_counts = build_vocab((dsl for _sha, dsl in _iter_distinct(dict_dir, "train_fit")), min_count=min_count)
    stats = TokenStats(vocab, check=True)
    closure = hashlib.sha256()
    prev = None
    for sha, dsl in _iter_distinct(dict_dir, "train_fit"):
        if prev is not None and sha <= prev:
            raise TokenizerStop("train_fit programs are not distinct and sha-sorted")
        prev = sha
        if _O.program_sha256(dsl) != sha:
            raise TokenizerStop(f"train_fit sha256(dsl) != program_sha256 for {sha}")
        closure.update(sha.encode("ascii") + b"\n")
        stats.add(dsl, sha)
    if stats.programs == 0:
        raise TokenizerStop("train_fit holds no programs")
    if not stats.ok():
        raise TokenizerStop(f"train_fit tokenization failed: {stats.summary()['failures'][:3]} raw={stats.raw_programs}")
    if min_count == 1 and stats.fallback_programs:
        raise TokenizerStop(f"{stats.fallback_programs} train_fit programs need an OOV fallback with min_count=1")
    max_len = stats.max_length()
    max_seq = ceil32(max_len)
    if max_seq > MAX_SEQ_CAP:
        raise TokenizerStop(f"max_seq ceil32({max_len}) = {max_seq} exceeds {MAX_SEQ_CAP}")
    input_closure = {"partition": "train_fit", "distinct_programs": stats.programs,
                     "program_set_sha256": closure.hexdigest(),
                     "program_set_rule": "sha256 of the sorted distinct program_sha256 hex values, one per line"}
    vocab_bytes = vocab.to_json(extra={
        "build": {"min_count": min_count, "source": "program-dict/train_fit (distinct programs only)"},
        "input_closure": input_closure,
        "counts": {"train_fit_programs": stats.programs, "train_fit_tokens": stats.tokens,
                   "per_id": stats.per_id, "unit": "token occurrences over the distinct train_fit programs",
                   "names": name_counts,
                   "by_class": {c: len(vocab.names(c)) for c in DATA_CLASSES}}})
    lengths_doc = {"schema": LENGTHS_SCHEMA, "vocab_sha256": vocab.sha256, "input_closure": input_closure,
                   "unit": "tokens per program, BOS and EOS included",
                   "train_fit": stats.lengths_doc(), "max_length": max_len, "max_seq": max_seq,
                   "max_seq_rule": f"ceil32(train_fit max length), refused above {MAX_SEQ_CAP}",
                   "over_product_max_program_tokens": stats.over_product,
                   "product_max_program_tokens": PRODUCT_MAX_PROGRAM_TOKENS,
                   "dsl_bytes_max": stats.dsl_bytes_max}
    written = {"vocab.json": vocab_bytes, "lengths.json": _json_bytes(lengths_doc)}
    api = _automaton_api(automaton)
    if api is not None:
        build_signatures, build_grammar = api
        signatures = build_signatures(dsl for _sha, dsl in _iter_distinct(dict_dir, "train_fit"))
        grammar = build_grammar(signatures, vocab, max_seq)
        written["signatures.json"] = _json_bytes(signatures)
        written["grammar.json"] = _json_bytes(grammar)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {name: common.atomic_write(out_dir / name, data) for name, data in written.items()}
    receipt = {"schema": BUILD_SCHEMA, "utc_us": common.utc_us(), "dict_dir": str(dict_dir),
               "train_fit_closure": _manifest_closure(dict_dir, "train_fit"),
               "epoch": ({"path": str(epoch), "sha256": common.sha256_file(epoch)} if epoch else None),
               "feat_math_sources": sources, "tokenizer_sha256": common.sha256_file(__file__),
               "vocab_sha256": vocab.sha256, "vocab_size": vocab.size, "max_seq": max_seq,
               "automaton": "built" if api is not None else ("skipped" if automaton is False else "unavailable"),
               "artifacts": {n: {"sha256": r["sha256"], "bytes": r["bytes"]} for n, r in sorted(results.items())},
               "train_fit": {k: v for k, v in stats.summary().items() if k != "length"}}
    common.atomic_write(out_dir / "build-receipt.json", common.receipt_bytes(receipt))
    return receipt


def _load_tok(tok_dir: Path) -> tuple:
    raw = (tok_dir / "vocab.json").read_bytes()
    vocab = Vocab.from_json(raw)
    lengths = json.loads((tok_dir / "lengths.json").read_bytes())
    if lengths.get("schema") != LENGTHS_SCHEMA or lengths.get("vocab_sha256") != vocab.sha256:
        raise TokenizerStop("lengths.json does not belong to vocab.json")
    return vocab, int(lengths["max_seq"]), json.loads(raw).get("input_closure") or {}


def roundtrip(dict_dir: Path | str, tok_dir: Path | str, partitions: Sequence[str] = PARTITIONS) -> dict:
    """Byte-exact round trip, sha256, Tier A canonical(text) == text, RAW and fallback rates per partition
    (distinct programs, read through the projection role 'roundtrip'; PLAN §5 P4 lists all four partitions).
    For train_fit it also requires the program set to equal vocab.json's input closure and no OOV fallback.
    Writes roundtrip.json; `pass` is False on any failure or RAW. The automaton gate (SYNTAX on every partition,
    POLICY 100% on train_fit) and the masked walks run next to it in the CLI (p4_gates)."""
    dict_dir, tok_dir = Path(dict_dir), Path(tok_dir)
    vocab, max_seq, closure = _load_tok(tok_dir)
    parts = {}
    for part in partitions:
        if part not in PARTITIONS:
            raise ValueError(f"unknown partition {part!r}")
        stats = TokenStats(vocab, check=True, max_seq=max_seq, canonical=True)
        digest = hashlib.sha256()
        for sha, dsl in _iter_distinct(dict_dir, part):
            digest.update(sha.encode("ascii") + b"\n")
            stats.add(dsl, sha)
        parts[part] = stats.summary()
        if part == "train_fit":
            ok = (digest.hexdigest() == closure.get("program_set_sha256")
                  and stats.programs == closure.get("distinct_programs"))
            parts[part]["closure_ok"] = ok
            parts[part]["pass"] = parts[part]["pass"] and ok and stats.fallback_programs == 0
    doc = {"schema": ROUNDTRIP_SCHEMA, "vocab_sha256": vocab.sha256, "max_seq": max_seq,
           "partitions": parts, "pass": all(p["pass"] for p in parts.values()), "utc_us": common.utc_us()}
    common.atomic_write(tok_dir / "roundtrip.json", _json_bytes(doc))
    return doc


def p4_gates(dict_dir: Path | str, tok_dir: Path | str, partitions: Sequence[str] = PARTITIONS, *, walks: int = 2000,
             seed: int = 0, require: bool = False) -> dict:
    """The automaton half of the P4 round trip (design steps 3-4), run next to roundtrip(): automaton.gate (SYNTAX
    accepts every partition's programs within max_seq, POLICY 100% of train_fit and reported elsewhere; writes
    automaton-gate.json) and `walks` masked walks per mode (EOS within max_seq, Tier A parse, identical
    retokenization of the generated ids, caps). Without grammar.json it is skipped (pass None) unless require."""
    tok_dir = Path(tok_dir)
    if not (tok_dir / "grammar.json").exists():
        if require:
            raise TokenizerStop("grammar.json is absent: the automaton gate cannot run")
        return {"automaton": "absent", "pass": None}
    from dltrain import automaton as A
    gate = A.gate(dict_dir, tok_dir, partitions)
    _vocab, grammar = A.load_tok_dir(tok_dir)
    walk_docs = {m.value: A.walk_check(grammar, walks, seed=seed, mode=m) for m in (A.Mode.POLICY, A.Mode.SYNTAX)} if walks else {}
    return {"automaton": "present", "gate_pass": gate["pass"],
            "gate": {p: {k: v for k, v in d.items() if not k.endswith("_examples")} for p, d in gate["partitions"].items()},
            "walks": {m: {k: v for k, v in w.items() if k != "failures"} for m, w in walk_docs.items()},
            "walk_failures": {m: w["failures"][:3] for m, w in walk_docs.items() if not w["pass"]},
            "pass": bool(gate["pass"] and all(w["pass"] for w in walk_docs.values()))}


# ---------------------------------------------------------------------------------------------
# Poison test (vocabulary closure)
# ---------------------------------------------------------------------------------------------
def _poison_programs(rng: random.Random, tag: str, *, long_stages: int) -> list:
    """Programs with names no corpus holds (op, key, categorical, random-cased sym identifiers) and one long pipe.
    All are canonical per Tier A and lex to a form."""
    h = "%08x" % rng.getrandbits(32)
    op, key, cat = f"zz_poison_{tag}_{h}", f"zz_poison_key_{h}", f"zzpoisoncat{h}"
    fn = "".join(ch.upper() if rng.random() < 0.5 else ch for ch in "sigmoid")
    if fn == "sigmoid":
        fn = "SiGmOiD"
    var = "".join(ch.upper() if rng.random() < 0.5 else ch for ch in "row_mean")
    if var == "row_mean":
        var = "ROW_mean"
    progs = [f"(pipe X ({op}) (ridge {key}={cat}))",
             f"(pipe X (standardize {key}=1.0) ({op} alpha=0.000001 {key}=true))",
             f'(sym affine "{fn}(c{rng.randrange(1000)})*{var}")',
             f"(portfolio (member (pipe X ({op}))) (member (pipe X (ridge {key}=2))){BLEND_ONE_TEXT}",
             "(pipe X" + "".join(f" ({op} {key}={i})" for i in range(long_stages)) + ")"]
    for p in progs:
        if _lex_spans(p)[0][0] == "RAW" or _O.canonical(p) != p:
            raise AssertionError(f"poison program is not canonical: {p[:80]}")
    return progs


def _rows_table(dsls: Sequence[str], partition: str, context: str = ""):
    """A ROW_SCHEMA table holding `dsls` in one synthetic context of `partition` (shape-correct, label-free)."""
    import pyarrow as pa
    P = _programs_module()
    ctx = hashlib.sha256(b"dl-poison-context-" + partition.encode() + b"/" + context.encode()).digest()
    n = len(dsls)
    values = {"context_sha256": [ctx] * n,
              "program_sha256": [hashlib.sha256(d.encode("utf-8")).digest() for d in dsls],
              "dsl": list(dsls), "family_id": ["poison"] * n, "split_role": [partition] * n,
              "record_status": ["completed"] * n, "shared_seed": [False] * n,
              "cohort": ["frozen"] * n, "form": [program_form(d) for d in dsls],
              "dsl_bytes": [len(d.encode("utf-8")) for d in dsls]}
    arrays = []
    for f in P.ROW_SCHEMA:
        if f.name not in values:
            raise RuntimeError(f"ROW_SCHEMA field {f.name} has no poison value")
        if pa.types.is_dictionary(f.type):
            dictionary = sorted(set(values[f.name]))
            idx = [dictionary.index(v) for v in values[f.name]]
            arrays.append(pa.DictionaryArray.from_arrays(pa.array(idx, type=f.type.index_type),
                                                         pa.array(dictionary, type=f.type.value_type)))
        else:
            arrays.append(pa.array(values[f.name], type=f.type))
    return pa.Table.from_arrays(arrays, schema=P.ROW_SCHEMA)


def write_dictionary_chunk(path: Path, dsls: Sequence[str], partition: str, *, context: str = "") -> dict:
    """Write one synthetic program-dictionary chunk, one synthetic context (the poison test and the tests)."""
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_rows_table(dsls, partition, context), str(path), compression="zstd")
    return {"path": str(path), "rows": len(dsls), "bytes": path.stat().st_size, "sha256": common.sha256_file(path)}


def _perturbed_dict(src: Path, manifest: dict, dst: Path, poison: Mapping[str, Sequence[str]], *, replace: bool) -> Path:
    """A copy of the dictionary at `dst`: every chunk is a symlink to the original except that each partition in
    `poison` gets a new chunk of poison programs, replacing its chunks (replace=True, so those originals are never
    opened) or appended to them."""
    doc = json.loads(json.dumps(manifest))
    doc.pop("sealed", None)                  # the copy never carries the sealed side table (nothing here reads it)
    for part, entry in doc["partitions"].items():
        chunks = [] if (replace and part in poison) else list(entry["chunks"])
        for c in chunks:
            link = dst / c["path"]
            link.parent.mkdir(parents=True, exist_ok=True)
            os.symlink((src / c["path"]).resolve(), link)
        if part in poison:
            rel = f"{part}/poison-00000.parquet"
            rec = write_dictionary_chunk(dst / rel, poison[part], part)
            chunks.append({"path": rel, "rows": rec["rows"], "bytes": rec["bytes"], "sha256": rec["sha256"]})
        entry["chunks"] = chunks
    common.atomic_write(dst / "manifest.json", _json_bytes(doc))
    return dst


def poison_test(dict_dir: Path | str, tok_dir: Path | str, *, seed: int = 0, automaton: bool | None = None) -> dict:
    """PLAN §5 P4 vocabulary-closure test. Rebuild with dev, diag and heldout replaced by poison programs (new ops,
    keys, values, identifiers and programs longer than any train_fit program): vocab.json, signatures.json,
    grammar.json and lengths.json (max_seq) must stay byte-identical. Positive control: poison appended to
    train_fit must change vocab.json. Work files live in a temporary directory under TMPDIR, removed afterwards."""
    dict_dir, tok_dir = Path(dict_dir), Path(tok_dir)
    manifest = json.loads((dict_dir / "manifest.json").read_bytes())
    rng = random.Random(seed)
    work = Path(tempfile.mkdtemp(prefix="dl-tok-poison-"))
    try:
        ref = tok_dir
        if not (ref / "vocab.json").exists():
            ref = work / "tok-ref"
            build_artifacts(dict_dir, None, ref, automaton=automaton)
        ref_bytes = {n: (ref / n).read_bytes() for n in ARTIFACTS if (ref / n).exists()}
        ref_max_seq = json.loads(ref_bytes["lengths.json"])["max_seq"]
        long_stages = max(64, ref_max_seq // 4 + 8)
        other = {p: _poison_programs(rng, p, long_stages=long_stages) for p in ("dev", "diag", "heldout")}
        d_other = _perturbed_dict(dict_dir, manifest, work / "dict-other", other, replace=True)
        t_other = work / "tok-other"
        build_artifacts(d_other, None, t_other, automaton=automaton)
        got_other = {n: (t_other / n).read_bytes() for n in ARTIFACTS if (t_other / n).exists()}
        train = {"train_fit": _poison_programs(rng, "train_fit", long_stages=4)}
        d_train = _perturbed_dict(dict_dir, manifest, work / "dict-train", train, replace=False)
        t_train = work / "tok-train"
        build_artifacts(d_train, None, t_train, automaton=automaton)
        got_train = {n: (t_train / n).read_bytes() for n in ARTIFACTS if (t_train / n).exists()}
        identical = {n: got_other.get(n) == b for n, b in ref_bytes.items()}
        changed = {n: got_train.get(n) != b for n, b in ref_bytes.items()}
        doc = {"schema": POISON_SCHEMA, "seed": seed, "reference": str(ref) if ref == tok_dir else "rebuilt",
               "artifacts": sorted(ref_bytes), "identical_after_dev_diag_heldout_poison": identical,
               "same_artifact_set": sorted(got_other) == sorted(ref_bytes),
               "changed_after_train_fit_poison": changed,
               "poison_programs": {p: len(v) for p, v in {**other, **train}.items()},
               "poison_max_dsl_bytes": max(len(d.encode()) for v in other.values() for d in v),
               "vocab_sha256": {"reference": Vocab.from_json(ref_bytes["vocab.json"]).sha256,
                                "train_fit_poisoned": Vocab.from_json(got_train["vocab.json"]).sha256}}
        doc["pass"] = bool(all(identical.values()) and doc["same_artifact_set"] and changed.get("vocab.json"))
        return doc
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def _print(doc: dict) -> None:
    print(json.dumps(doc, sort_keys=True, ensure_ascii=False))


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dltrain.dsl_tokenizer", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="vocab/lengths (and automaton) artifacts from train_fit, or --poison-test")
    b.add_argument("--dict", default=str(DEFAULT_DICT))
    b.add_argument("--epoch", default=str(DEFAULT_EPOCH), help="pinned in build-receipt.json ('' to omit)")
    b.add_argument("--out", default=str(DEFAULT_TOK))
    b.add_argument("--all-partitions", action="store_true",
                   help="then run the round trip and the automaton gate and walks over all four partitions")
    b.add_argument("--poison-test", action="store_true")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--min-count", type=int, default=1)
    b.add_argument("--walks", type=int, default=2000, help="masked walks per automaton mode (0 skips them)")
    g = b.add_mutually_exclusive_group()
    g.add_argument("--no-automaton", action="store_true")
    g.add_argument("--require-automaton", action="store_true")
    r = sub.add_parser("roundtrip", help="round trip, sha, Tier A canonical, RAW and fallback rates per partition, "
                                         "then the automaton gate and walks when grammar.json is present")
    r.add_argument("--dict", default=str(DEFAULT_DICT))
    r.add_argument("--tok", default=str(DEFAULT_TOK))
    r.add_argument("--walks", type=int, default=2000, help="masked walks per automaton mode (0 skips them)")
    r.add_argument("--require-automaton", action="store_true", help="STOP when grammar.json is absent")
    rg = r.add_mutually_exclusive_group(required=True)
    rg.add_argument("--all-partitions", action="store_true")
    rg.add_argument("--partitions", nargs="+", choices=PARTITIONS)
    s = sub.add_parser("show", help="print the tokens of one program")
    s.add_argument("--vocab", required=True)
    s.add_argument("--dsl", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "build":
            mode = False if args.no_automaton else (True if args.require_automaton else None)
            if args.poison_test:
                doc = poison_test(args.dict, args.out, seed=args.seed, automaton=mode)
                _print(doc)
                return 0 if doc["pass"] else 1
            rec = build_artifacts(args.dict, args.epoch or None, args.out, automaton=mode, min_count=args.min_count)
            out = {"build": {k: rec[k] for k in ("vocab_sha256", "vocab_size", "max_seq", "automaton", "artifacts")}}
            ok = True
            if args.all_partitions:
                rt = roundtrip(args.dict, args.out, PARTITIONS)
                out["roundtrip"] = {p: {k: v for k, v in d.items() if k not in ("length", "failures")}
                                    for p, d in rt["partitions"].items()}
                ok = rt["pass"]
                if mode is not False:
                    gates = p4_gates(args.dict, args.out, PARTITIONS, walks=args.walks, seed=args.seed,
                                     require=bool(mode))
                    out["automaton_gates"] = gates
                    ok = ok and gates["pass"] is not False
            _print(out)
            return 0 if ok else 1
        if args.cmd == "roundtrip":
            parts = PARTITIONS if args.all_partitions else tuple(args.partitions)
            rt = roundtrip(args.dict, args.tok, parts)
            gates = p4_gates(args.dict, args.tok, parts, walks=args.walks, require=args.require_automaton)
            ok = rt["pass"] and gates["pass"] is not False
            _print({"pass": ok, "partitions": {p: {k: v for k, v in d.items() if k not in ("length",)}
                                               for p, d in rt["partitions"].items()},
                    "automaton_gates": gates})
            return 0 if ok else 1
        vocab = Vocab.from_json(Path(args.vocab).read_bytes())
        ids = tokenize(args.dsl, vocab)
        _print({"ids": ids, "tokens": token_names(ids, vocab), "fallback_spans": fallback_spans(ids, vocab),
                "roundtrip": detokenize(ids, vocab) == args.dsl, "skeleton": skeleton(args.dsl)})
        return 0
    except (TokenizerStop, SourcePinError) as exc:
        _print({"stop": type(exc).__name__, "reason": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
