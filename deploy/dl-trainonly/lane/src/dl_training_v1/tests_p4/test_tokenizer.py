"""PLAN §5 P4 tokenizer tests (dltrain.dsl_tokenizer, design "tokenizer" section).

- the fixed id layout and the feat-math source pins;
- an independent reference recognizer (oracle.dsl_tokenize + the printer's exact spacing) against `lex`: RAW is
  used exactly when a text matches no form, and the atom spans agree;
- totality: random Unicode texts (control chars, U+001C, non-ASCII) and mutations of canonical programs round-trip
  byte-exactly;
- canonical generator: random Tier A-canonical programs (unseen ops, keys, categoricals and identifiers, quoted
  values with spaces, parens, escaped quotes and '=', an empty key, non-ASCII) never lex to RAW, round-trip and
  tokenize uniquely;
- uniqueness: spelled forms only for names absent from their class, never STR for true/false/numeric/CAT text and
  never ID for c[0-9]+ or ID names;
- exact floats: 1 vs 1.0, -0.0, 0.000001, 24-char floats, i64 bounds, exponents, inline negatives; the hand-picked
  fixtures are confirmed canonical by Tier A and by the pinned engine (Tier B, sha256-verified first);
- skeleton, vocab.json (schema, pins, counts, vocab_sha256; refusals), artifacts over synthetic program
  dictionaries (determinism under shuffling and re-chunking, round trip, poison test, CLI);
- vocabulary STOPs: a train_fit name that cannot be a vocabulary token (a key with an odd number of '"');
- the CLI wiring of the automaton gate and masked walks next to the round trip (build --all-partitions, roundtrip);
- a <= 40-context corpus sample: vocabulary from half of the sample's datasets, round trip, sha256 and Tier A
  canonical(text) == text on all of it, RAW == 0, and the fallback rate on the other half (printed as
  `P4-TOKENIZER {...}`).

Corpus records are projected at parse time (json object_pairs_hook) to program.{dsl, program_sha256}; no other
field is kept, read or printed. Temporary files live under TMPDIR and are removed by each test.
"""
import contextlib
import hashlib
import io
import json
import math
import os
import random
import shutil
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path

from dltrain import dsl_tokenizer as T
from dltrain import oracle as O

EPOCH = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/epochs/epoch-000.json")
SAMPLE_PER_STRATUM = 4            # 5 folds x 2 cohorts x 4 = 40 contexts (the hard cap)
SAMPLE_SEED = 20261005            # the oracle builder's sample, so the counts are comparable
SPLIT_SEED = 7
DICT_SCHEMA = "jope.dl-program-dict/v1"


def report(name: str, doc: dict) -> None:
    print("P4-TOKENIZER " + json.dumps({"test": name, **doc}, sort_keys=True, ensure_ascii=True), flush=True)


def canonical_or_none(text: str):
    try:
        return O.canonical(text)
    except (O.CanonicalError, O.Undecidable):
        return None


# ---------------------------------------------------------------------------------------------
# Independent reference recognizer: dsl.rs tokens (oracle.dsl_tokenize) + the printer's exact spacing
# ---------------------------------------------------------------------------------------------
class _Ref:
    def __init__(self, text: str, toks: list):
        self.t, self.toks, self.k, self.end = text, toks, 0, 0
        self.atoms = []

    def _peek(self, gap: str, kind: str):
        if self.k >= len(self.toks):
            return None
        kd, proc, a, b = self.toks[self.k]
        if kd != kind or self.t[self.end:a] != gap:
            return None
        return kd, proc, a, b

    def take(self, gap: str, kind: str, raw: str | None = None) -> bool:
        tok = self._peek(gap, kind)
        if tok is None or (raw is not None and self.t[tok[2]:tok[3]] != raw):
            return False
        self.k += 1
        self.end = tok[3]
        return True

    def atom(self, gap: str, role: str):
        tok = self._peek(gap, "atom")
        if tok is None or self.t[tok[2]] == '"':
            return None
        self.k += 1
        self.end = tok[3]
        self.atoms.append((role, tok[2], tok[3]))
        return tok[1]

    def pipe(self, gap: str) -> bool:
        if not (self.take(gap, "(") and self.take("", "atom", "pipe") and self.take(" ", "atom", "X")):
            return False
        stages = 0
        while True:
            if self.take(" ", "("):
                if self.atom("", "op") is None:
                    return False
                while not self.take("", ")"):
                    proc = self.atom(" ", "param")
                    if proc is None or "=" not in proc:
                        return False
                stages += 1
            elif stages and self.take("", ")"):
                return True
            else:
                return False

    def portfolio(self) -> bool:
        if not (self.take("", "(") and self.take("", "atom", "portfolio")):
            return False
        members = 0
        while True:
            if members and self.t[self.end:] in (T.BLEND_AUTO_TEXT, T.BLEND_ONE_TEXT):
                return True
            if not (self.take(" ", "(") and self.take("", "atom", "member") and self.pipe(" ") and self.take("", ")")):
                return False
            members += 1


def reference_form(text: str):
    """(matches, atom spans) per the design's forms, decided without dltrain.dsl_tokenizer."""
    for opener in ('(sym affine "', '(sym raw "'):
        if text.startswith(opener):
            if len(text) < len(opener) + 2 or not text.endswith('")'):
                return False, None
            try:
                O.math_lex(text[len(opener):-2])
            except O.CanonicalError:
                return False, None
            return True, None
    try:
        toks = O.dsl_tokenize(text)
    except O.CanonicalError:
        return False, None
    r = _Ref(text, toks)
    if r.pipe(""):
        ok = r.k == len(toks) and r.end == len(text)
        return ok, (r.atoms if ok else None)
    r = _Ref(text, toks)
    if r.portfolio():
        return True, r.atoms
    return False, None


def lexer_atoms(lexemes: list) -> list:
    out = []
    for i, lx in enumerate(lexemes):
        if lx.kind == "OP":
            out.append(("op", lx.start, lx.end))
        elif lx.kind == "KEY":
            out.append(("param", lx.start + 1, lexemes[i + 1].end))
    return out


# ---------------------------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------------------------
ALPHABET = (list('()"\\= .-+eE019azXcpi,*/^_') + [" "] * 6 +
            ["\t", "\n", "\xa0", "\u3000", "\u2028", "\x85", "\x1c", "\x1f", "\x00", "\x7f", "\xe9", "\u4e2d",
             "\U0001f642", "\u200b"])
OPS = ("ridge", "histgbm2", "extra_trees", "winsor_clip", "target", "map", "portfolio", "zz_unseen_op", "op=x",
       "a\\b", "\xe9t\xe9", 'op"q"', "UPPER", "x(y)z")
KEYS = ("alpha", "max_features", "k", "", "zz_unseen_key", "a\\b", 'k"x"', "\xe9", "a=b", "q")
STRS = ("huber", "rbf+linear", "features", "zz_unseen_cat", "\xe9\u4e2d", "a b", "(x)", "a=b", "", 'a"b',
        "sigmoid(col(0))", "nan", "TRUE", "1e-5", "x\\y", "\xa0lead", "tab\there", "inf", "-")
FLOATS = (-0.0, 0.0, 1.0, 0.5, 1e-6, 0.09988591634790045, 0.5408000000000001, 1e20, -2.5, math.inf, -math.inf,
          math.nan, 1.234567890123456e-7, 9223372036854775808.0, 3.0000000000004, 1e-12, 5e-324)
FUNCS = {**{f: 1 for f in ("abs", "sqrt", "log1p", "sin", "cos", "tanh", "sigmoid", "exp", "log", "asinh", "atan",
                           "erf", "sign", "floor", "round")}, "min": 2, "max": 2, "step": 2, "clip": 3}
LEAVES = ("x", "row_mean", "mean", "ROW_STD", "Row_Sum", "row_l1", "row_norm", "C7", "col7", "0.5", "1e-5", "1E5",
          "2.5e+3", ".5", "5.", "007", "8.727099312855202e-5", "1.2705454457961025")
WS_CHOICES = ("", "", "", " ", " ", "\xa0", "\t", "\u3000")


def rand_value(rng: random.Random):
    k = rng.random()
    if k < 0.3:
        return ("Int", rng.choice((0, 1, -1, 7, 999761, O.I64_MIN, O.I64_MAX, rng.randint(-10 ** 6, 10 ** 6))))
    if k < 0.6:
        if rng.random() < 0.5:
            return ("Float", rng.choice(FLOATS))
        return ("Float", rng.uniform(-1.0, 1.0) * 10.0 ** rng.uniform(-8.0, 12.0))
    if k < 0.7:
        return ("Bool", rng.random() < 0.5)
    return ("Str", rng.choice(STRS))


def rand_pipe(rng: random.Random, max_stages: int = 5) -> O.Pipeline:
    stages = []
    for _ in range(rng.randint(1, max_stages)):
        params = {rng.choice(KEYS): rand_value(rng) for _ in range(rng.randint(0, 4))}
        stages.append((rng.choice(OPS), tuple(sorted(params.items()))))
    return O.Pipeline(tuple(stages))


def rand_body(rng: random.Random, depth: int) -> str:
    k = rng.random()
    if depth <= 0 or k < 0.3:
        if rng.random() < 0.5:
            return "c" + ("00" if rng.random() < 0.1 else "") + str(rng.randint(0, 1025))
        return rng.choice(LEAVES)
    if k < 0.42:
        return "-" + rand_body(rng, depth - 1)
    if k < 0.7:
        a, b = rand_body(rng, depth - 1), rand_body(rng, depth - 1)
        w = rng.choice(WS_CHOICES)
        return rng.choice(("({}{w}{}{w}{})", "{}{w}{}{w}{}")).format(a, rng.choice("+-*/^"), b, w=w)
    name = rng.choice(sorted(FUNCS))
    name = "".join(ch.upper() if rng.random() < 0.15 else ch for ch in name)
    args = [rand_body(rng, depth - 1) for _ in range(FUNCS[name.lower()])]
    return name + "(" + ("," + rng.choice(WS_CHOICES)).join(args) + ")"


def rand_canonical(rng: random.Random) -> str | None:
    k = rng.random()
    if k < 0.55:
        text = O.render(rand_pipe(rng))
    elif k < 0.9:
        body = rng.choice(WS_CHOICES) + rand_body(rng, rng.randint(0, 4)) + rng.choice(WS_CHOICES)
        text = '(sym ' + rng.choice(("affine", "raw")) + ' "' + body + '")'
    else:
        members = tuple(rand_pipe(rng, 3) for _ in range(rng.randint(1, 3)))
        text = O.render(O.Portfolio(members, rng.random() < 0.5))
    return text if canonical_or_none(text) == text else None


def rand_text(rng: random.Random) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 30)))


def mutate(rng: random.Random, base: str) -> str:
    s = list(base)
    for _ in range(rng.randint(1, 3)):
        op = rng.random()
        i = rng.randint(0, len(s))
        if op < 0.4:
            s.insert(i, rng.choice(ALPHABET))
        elif op < 0.7 and s:
            del s[min(i, len(s) - 1)]
        elif s:
            s[min(i, len(s) - 1)] = rng.choice(ALPHABET)
    return "".join(s)


# Hand-picked canonical fixtures (design test list); each is checked canonical by Tier A and Tier B.
FLOAT_24 = "0.0000001234567890123456"
FIXTURES = (
    "(pipe X (extra_trees depth=9 max_features=1 min_leaf=3 seed=999761 trees=128))",
    "(pipe X (extra_trees depth=9 max_features=1.0 min_leaf=3 seed=999761 trees=128))",
    "(pipe X (impute add_indicator=true strategy=median value=-1.0) (step t=-0.0) (histgbm2 delta=0.000001 loss=huber "
    "lr=0.09988591634790045 rounds=300 val_shuffle=true))",
    "(pipe X (winsor_clip q=0.5408000000000001) (ridge lambda_frac=" + FLOAT_24 + "))",
    "(pipe X (ridge a=-9223372036854775808 b=9223372036854775807 c=9223372036854775808.0 d=100000000000000000000.0))",
    "(pipe X (ridge k=NaN) (ridge k=inf) (ridge k=-inf) (ridge val_shuffle=false))",
    "(pipe X (map expr=\"sigmoid(col(0))\") (derive expr=\"row_mean + c0\"))",
    "(pipe X (map expr=\"a\\\"b=(c)\"))",
    "(pipe X (ridge =5))",
    "(pipe X (op=x k=\xe9) (portfolio preset=core))",
    "(sym affine \"8.727099312855202e-5\")",
    "(sym affine \"(c1*-0.43)\")",
    "(sym affine \"((-0.87)--1.44)\")",
    "(sym affine \"x--1.44\")",
    "(sym raw \"c395\")",
    "(sym affine \"c007+1E5*c1-2.5e+3\")",
    "(sym affine \"clip(max(c3,row_mean),-8.727099312855202e-5,min(c0,c395))\")",
    "(sym affine \"sigmoid(((c5^1.2705454457961025)/-1.1374340345718594))\")",
    "(sym affine \"Sigmoid(C5) +\xa0row_l1\")",
    "(portfolio (member (pipe X (ridge_gcv))) (member (pipe X (blom) (gpr kernel=rbf+linear steps=40))) "
    "(blend folds=5 guard=clip kind=nnls repeats=auto))",
    "(portfolio (member (pipe X (norm_row) (krr))) (member (pipe X (winsor_clip q=0.01804076952840708) (norm_row) "
    "(ridge_gcv))) (blend folds=5 guard=clip kind=nnls repeats=1))",
)


def make_dict(root: Path, parts: dict, *, chunks: int = 1, shuffle_seed: int | None = None) -> Path:
    """A synthetic P4a program dictionary: manifest.json + ROW_SCHEMA chunks per partition."""
    manifest = {"schema": DICT_SCHEMA, "partitions": {}}
    for part in T.PARTITIONS:
        dsls = list(parts.get(part, ()))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(dsls)
        entries = []
        n = max(1, chunks)
        for c in range(n):
            piece = dsls[c::n]
            if not piece:
                continue
            rel = f"{part}/part-{c:05d}.parquet"
            rec = T.write_dictionary_chunk(root / rel, piece, part, context=f"ctx{c}")
            entries.append({"path": rel, "rows": rec["rows"], "bytes": rec["bytes"], "sha256": rec["sha256"]})
        manifest["partitions"][part] = {"chunks": entries}
    (root / "manifest.json").write_bytes(json.dumps(manifest, sort_keys=True).encode())
    return root


# ---------------------------------------------------------------------------------------------
# Corpus sample (module-level cache: read once per process, <= 40 contexts)
# ---------------------------------------------------------------------------------------------
_KEEP = frozenset({"program", "dsl", "program_sha256"})
_CORPUS = None


def _project(pairs):
    """Parse-time projection: every JSON object keeps only program / dsl / program_sha256."""
    return {k: v for k, v in pairs if k in _KEEP}


def corpus_sample() -> dict:
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS
    doc = json.loads(EPOCH.read_bytes())
    included = [c for c in doc["contexts"] if c.get("status") == "included"]
    strata = {}
    for c in included:
        strata.setdefault((c["cohort"], c["fold"]), []).append(c)
    rng = random.Random(SAMPLE_SEED)
    picked = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda c: c["context_sha256"])
        picked.extend(rng.sample(group, SAMPLE_PER_STRATUM))
    assert len(picked) <= 40
    contexts = []
    for ctx in picked:
        src = ctx["native_source"]
        with open(src["path"], "rb") as f:
            data = f.read(src["bytes"])            # the pinned prefix only (live files may have grown)
        ok = len(data) == src["bytes"] and hashlib.sha256(data).hexdigest() == src["sha256"]
        rows = []
        for line in data.splitlines():
            rec = json.loads(line, object_pairs_hook=_project)
            prog = rec["program"]
            rows.append((prog["program_sha256"], prog["dsl"]))
        del data
        contexts.append({"context_sha256": ctx["context_sha256"], "dataset_hash": ctx["dataset_hash"],
                         "cohort": ctx["cohort"], "fold": ctx["fold"], "native_ok": ok, "rows": rows})
    _CORPUS = {"contexts": contexts}
    return _CORPUS


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------
class LayoutAndPinTests(unittest.TestCase):
    def test_fixed_id_layout(self):
        self.assertEqual((T.PAD, T.BOS, T.EOS), (0, 1, 2))
        self.assertEqual(list(T.FIXED)[3:13], list(T.MARKERS))
        self.assertEqual([T.FIXED[m] for m in T.MARKERS], [b"", b"", b" ", b"=", b"", b"", b"", b"", b"", b""])
        want = {"PIPE_OPEN": b"(pipe X", "STAGE_OPEN": b" (", "CLOSE": b")", "SYM_AFFINE_OPEN": b'(sym affine "',
                "SYM_RAW_OPEN": b'(sym raw "', "SYM_CLOSE": b'")', "PORTFOLIO_OPEN": b"(portfolio",
                "MEMBER_OPEN": b" (member ", "BLEND_AUTO": b" (blend folds=5 guard=clip kind=nnls repeats=auto))",
                "BLEND_ONE": b" (blend folds=5 guard=clip kind=nnls repeats=1))", "TRUE": b"true", "FALSE": b"false",
                "COL": b"c"}
        self.assertEqual({n: T.FIXED[n] for n in list(T.FIXED)[13:26]}, want)
        self.assertEqual([T.FIXED_IDS[n] for n in want], list(range(13, 26)))
        self.assertEqual((T.BYTE_BASE, T.DATA_BASE), (26, 282))
        v = T.Vocab.from_names({})
        self.assertEqual(v.size, 282)
        for b in range(256):
            self.assertEqual(v.piece(T.byte_id(b)), bytes([b]))
        self.assertEqual(v.class_range("BYTE"), (26, 282))
        self.assertEqual(sum(1 for b in range(256) if 0x20 <= b < 0x7F), 95)
        with self.assertRaises(ValueError):
            v.piece(-1)
        with self.assertRaises(ValueError):
            v.piece(282)

    def test_feat_math_pins(self):
        self.assertEqual(T.verify_feat_math_sources(), T.FEAT_MATH_PINS)
        for name, sha in T.FEAT_MATH_PINS.items():
            self.assertEqual(O.FEAT_MATH_PINS[name], sha)
        # The structural pieces are copied from the pinned printer sources.
        dsl_rs = (T.FEAT_MATH_SRC / "dsl.rs").read_text()
        genome_rs = (T.FEAT_MATH_SRC / "genome.rs").read_text()
        self.assertIn('write!(f, "(pipe X")?;', dsl_rs)
        self.assertIn('write!(f, " {stage}")?;', dsl_rs)
        self.assertIn('write!(f, " {key}={value}")?;', dsl_rs)
        self.assertIn('out.push_str(" (member ");', genome_rs)
        self.assertIn('" (blend folds={} guard=clip kind=nnls repeats={}))"', genome_rs)
        self.assertIn('"(sym {} \\"{}\\")"', genome_rs)

    def test_rust_whitespace_rule(self):
        # U+001C is not Rust White_Space (Python isspace says it is); U+00A0 is.
        self.assertTrue("\x1c".isspace())
        v = T.Vocab.from_names({})
        self.assertEqual(T.lex("(pipe X (ridge\x1c))")[2].text, "ridge\x1c")          # part of the op atom
        self.assertEqual(T.lex("(pipe X\xa0(ridge))")[0].kind, "RAW")                 # whitespace, but not the printer's
        self.assertEqual(T.lex('(sym raw "c0\x1c")')[0].kind, "RAW")                  # not whitespace for math lex
        lx = T.lex('(sym raw "c0\xa0+\u3000c1")')
        self.assertEqual([l.kind for l in lx], ["SYM_RAW_OPEN", "COL", "WS", "PUNCT", "WS", "COL", "SYM_CLOSE"])
        for s in ("(pipe X (ridge\x1c))", '(sym raw "c0\xa0+\u3000c1")'):
            self.assertEqual(T.detokenize(T.tokenize(s, v), v), s)


class LexerTests(unittest.TestCase):
    def test_spans_tile_the_text(self):
        rng = random.Random(1)
        texts = list(FIXTURES) + list(O.FUZZ_BASES) + [rand_text(rng) for _ in range(300)]
        for t in texts:
            lx = T.lex(t)
            pos = 0
            for l in lx:
                self.assertEqual(l.start, pos)
                self.assertEqual(l.text, t[l.start:l.end])
                self.assertIn(l.kind, T.LEXEME_KINDS)
                pos = l.end
            self.assertEqual(pos, len(t))

    def test_forms(self):
        kinds = lambda s: [l.kind for l in T.lex(s)]
        self.assertEqual(kinds("(pipe X (ridge alpha=1.0 kind=huber flag=true))"),
                         ["PIPE_OPEN", "STAGE_OPEN", "OP", "KEY", "NUM", "KEY", "VALUE", "KEY", "TRUE", "CLOSE", "CLOSE"])
        self.assertEqual(kinds('(sym affine "sigmoid(c12)-x")'),
                         ["SYM_AFFINE_OPEN", "IDENT", "PUNCT", "COL", "PUNCT", "PUNCT", "IDENT", "SYM_CLOSE"])
        self.assertEqual(kinds(O.FUZZ_BASES[-1])[-1], "BLEND_ONE")
        self.assertEqual(kinds(O.FUZZ_BASES[-2])[-1], "BLEND_AUTO")
        # quoted values: the atom rule (quote toggling, escapes) decides where the value ends
        lx = T.lex('(pipe X (map expr="a\\"b=(c)" k=v))')
        self.assertEqual([(l.kind, l.text) for l in lx[3:7]],
                         [("KEY", " expr="), ("VALUE", '"a\\"b=(c)"'), ("KEY", " k="), ("VALUE", "v")])
        self.assertEqual(T.lex("(pipe X (ridge k==5))")[4].text, "=5")            # the key ends at the first '='
        self.assertEqual(T.lex("(pipe X (ridge =5))")[3].name, "")                 # empty key
        self.assertEqual(T.lex("(pipe X (a=b))")[2].text, "a=b")                    # an op may hold '='

    def test_no_form_is_raw(self):
        for s in ("", " ", "(pipe X)", "(pipe X (ridge) )", " (pipe X (ridge))", "(pipe X (ridge))\n", "(pipe X(ridge))",
                  "(pipe X  (ridge))", "(pipe Xy (ridge))", "(pipe X (ridge k))", "(pipe X ((ridge)))",
                  '(pipe X ("ridge"))', '(pipe X (ridge "k=v"))', '(pipe X (ridge k="v))', '(pipe X (ridge k="v\\',
                  "(pipe X (ridge)) (ridge)", "(portfolio (blend folds=5 guard=clip kind=nnls repeats=1))",
                  "(portfolio (member (pipe X (ridge))) (blend))", '(sym affine "c0"', '(sym affine "")x',
                  '(sym affine "c0$")', '(sym affine "1.2.3")', '(sym affine "1e999")', '(sym affine "c""0")',
                  '(sym bogus "c0")', "(PIPE X (ridge))", "pipe"):
            lx = T.lex(s)
            self.assertEqual([l.kind for l in lx], ["RAW"], s)
            self.assertFalse(reference_form(s)[0], s)

    def test_raw_iff_no_form_against_reference(self):
        rng = random.Random(20261005)
        bases = list(FIXTURES) + list(O.FUZZ_BASES)
        texts = [rand_text(rng) for _ in range(2500)] + [mutate(rng, rng.choice(bases)) for _ in range(5000)]
        agree = forms = 0
        for t in texts:
            lx = T.lex(t)
            mine = lx[0].kind != "RAW"
            ref, atoms = reference_form(t)
            self.assertEqual(mine, ref, repr(t))
            agree += 1
            if mine:
                forms += 1
                if atoms is not None:
                    self.assertEqual(lexer_atoms(lx), atoms, repr(t))
        self.assertGreater(forms, 1000)          # both sides of the property are exercised
        self.assertGreater(len(texts) - forms, 1000)
        report("raw_iff_no_form", {"texts": len(texts), "agree": agree, "forms": forms, "raw": len(texts) - forms})


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab, cls.counts = T.build_vocab(list(FIXTURES) + list(O.FUZZ_BASES))

    def roundtrip(self, text: str, vocab=None) -> list:
        v = vocab or self.vocab
        ids = T.tokenize(text, v)
        self.assertEqual(ids[0], T.BOS)
        self.assertEqual(ids[-1], T.EOS)
        back = T.detokenize(ids, v)
        self.assertEqual(back, text)
        self.assertEqual(O.program_sha256(back), O.program_sha256(text))
        self.assertEqual(T.tokenize(back, v), ids)
        for cls, a, b in T.fallback_spans(ids, v):
            if cls != "RAW":
                text = T.detokenize(ids[a:b], v)
                self.assertTrue(T.spelling_allowed(cls, text[1:-1] if cls == "KEY" else text, v), (cls, text))
        return ids

    def test_fixture_roundtrip_and_canonical(self):
        for f in FIXTURES:
            self.assertEqual(canonical_or_none(f), f, f)
            ids = self.roundtrip(f)
            self.assertNotEqual(ids[1], T.RAW_BEGIN)
            # every name is in this vocab; only values that can never be CAT (quoted, NaN, inf) are spelled
            for cls, a, b in T.fallback_spans(ids, self.vocab):
                self.assertEqual(cls, "STR", f)
                self.assertFalse(T.is_cat_name(T.detokenize(ids[a:b], self.vocab)), f)
        for s in O.FUZZ_BASES:
            self.roundtrip(s)

    def test_exact_float_survival(self):
        v = self.vocab
        i1, f1 = (T.tokenize(FIXTURES[k], v) for k in (0, 1))
        self.assertNotEqual(i1, f1)
        self.assertNotEqual(O.program_sha256(FIXTURES[0]), O.program_sha256(FIXTURES[1]))
        mf = v.id("KEY", "max_features")
        self.assertEqual(i1[i1.index(mf) + 1:i1.index(mf) + 2], [T.byte_id(ord("1"))])
        self.assertEqual(f1[f1.index(mf) + 1:f1.index(mf) + 4], [T.byte_id(c) for c in b"1.0"])
        self.assertEqual(T.skeleton(FIXTURES[0]), T.skeleton(FIXTURES[1]))       # collapsed only in the skeleton
        self.assertEqual(len(FLOAT_24), 24)
        self.assertEqual(O.format_f64(float(FLOAT_24)), FLOAT_24)
        for literal in ("-0.0", "0.000001", FLOAT_24, "0.09988591634790045", "0.5408000000000001",
                        "-9223372036854775808", "9223372036854775807", "9223372036854775808.0", "100000000000000000000.0",
                        "8.727099312855202e-5", "1E5", "2.5e+3", "-0.43", "--1.44", "007"):
            hits = [f for f in FIXTURES if literal in f]
            self.assertTrue(hits, literal)
            ids = T.tokenize(hits[0], v)
            seq = [T.byte_id(c) for c in literal.encode()]
            self.assertTrue(any(ids[i:i + len(seq)] == seq for i in range(len(ids))), literal)
        # non-canonical spellings are kept verbatim too: the tokenizer never parses or re-prints a number
        for s in ("(pipe X (ridge alpha=1.50))", "(pipe X (ridge alpha=1e-5))", "(pipe X (ridge alpha=+5))",
                  "(pipe X (ridge alpha=-0))", '(sym affine "1.0e3*c1")'):
            self.roundtrip(s)
        # inline negatives are unary '-' then numeral bytes
        names = T.token_names(T.tokenize('(sym affine "(c1*-0.43)")', v), v)
        self.assertEqual(names[2:12], ["b:(", "COL", "b:1", "b:*", "b:-", "b:0", "b:.", "b:4", "b:3", "b:)"])

    def test_columns_have_no_oov(self):
        v = T.Vocab.from_names({})
        for n in ("0", "395", "1024", "007", "99999999999999999999999"):
            ids = T.tokenize('(sym raw "c' + n + '")', v)
            self.assertEqual(ids[2], T.COL)
            self.assertEqual(ids[3:-2], [T.byte_id(c) for c in n.encode()])
            self.assertEqual(T.fallback_spans(ids, v), [])
        self.assertEqual(T.token_names(T.tokenize('(sym raw "C5")', v), v)[2], "ID_BEGIN")   # case-sensitive c

    def test_unseen_names_are_spelled(self):
        v = self.vocab
        cases = {"(pipe X (zz_new_op))": ["OP"], "(pipe X (ridge zz_new_key=1))": ["KEY"],
                 "(pipe X (ridge loss=zz_new_cat))": ["STR"], '(pipe X (ridge loss="a b"))': ["STR"],
                 "(pipe X (ridge loss=NaN alpha=inf))": ["STR", "STR"], '(sym raw "SIGMOID(c1)")': ["ID"],
                 "(pipe X (\xe9t\xe9 k=\xfc))": ["OP", "STR"], "(pipe X (ridge =5))": [],
                 "(pipe X (ridge zz=5))": ["KEY"], "(pipe X (op=x k=\xe9))": []}
        for text, want in cases.items():
            ids = self.roundtrip(text)
            spans = T.fallback_spans(ids, v)
            self.assertEqual([c for c, _a, _b in spans], want, text)
        # OOV fallbacks versus STR literals (values that can never be CAT names)
        ids = T.tokenize('(pipe X (zz zk1=zzcat zk2="a b" zk3=NaN zk4=inf))', v)
        spelled = [(c, T.spelled_text(ids, (c, a, b), v)) for c, a, b in T.fallback_spans(ids, v)]
        self.assertEqual(spelled, [("OP", "zz"), ("KEY", "zk1"), ("STR", "zzcat"), ("KEY", "zk2"), ("STR", '"a b"'),
                                   ("KEY", "zk3"), ("STR", "NaN"), ("KEY", "zk4"), ("STR", "inf")])
        oov = [(c, T.spelled_text(ids, (c, a, b), v)) for c, a, b in T.oov_spans(ids, v)]
        self.assertEqual(oov, [("OP", "zz"), ("KEY", "zk1"), ("STR", "zzcat"), ("KEY", "zk2"), ("KEY", "zk3"),
                               ("KEY", "zk4")])
        st = T.TokenStats(v)
        st.add('(pipe X (ridge k="a b"))')
        st.add("(pipe X (zz))")
        s = st.summary()
        self.assertEqual((s["fallback_programs"], s["str_literal_programs"], s["str_literal_spans"]), (1, 1, 1))
        self.assertEqual(s["fallback_spans"]["OP"], 1)
        ids = T.tokenize("(pipe X (\xe9t\xe9))", v)
        self.assertEqual(ids[3:-3], [T.OP_BEGIN] + [T.byte_id(c) for c in "\xe9t\xe9".encode()] + [T.OP_END])
        ids = T.tokenize("(pipe X (ridge zz=1))", v)
        self.assertEqual(T.detokenize(ids[ids.index(T.KEY_BEGIN):ids.index(T.KEY_END) + 1], v), " zz=")

    def test_uniqueness(self):
        v = self.vocab
        self.assertIsNotNone(v.id("CAT", "huber"))
        self.assertIsNotNone(v.id("ID", "sigmoid"))
        # spelled forms are never emitted for these texts
        for cls, text in (("STR", "true"), ("STR", "false"), ("STR", "1.5"), ("STR", "-0.0"), ("STR", "huber"),
                          ("ID", "c12"), ("ID", "c007"), ("ID", "sigmoid"), ("OP", "ridge"), ("KEY", "alpha")):
            self.assertFalse(T.spelling_allowed(cls, text, v), (cls, text))
        for cls, text in (("STR", "True"), ("STR", "1e-5"), ("STR", "Huber"), ("STR", '"huber"'), ("ID", "C12"),
                          ("ID", "SIGMOID"), ("OP", "ridgeX"), ("KEY", "alpha2")):
            self.assertTrue(T.spelling_allowed(cls, text, v), (cls, text))
        names = lambda s: T.token_names(T.tokenize(s, v), v)
        self.assertEqual(names("(pipe X (ridge k=true))")[5], "TRUE")
        self.assertEqual(names("(pipe X (ridge k=false))")[5], "FALSE")
        self.assertEqual(names("(pipe X (ridge k=1.5))")[5:8], ["b:1", "b:.", "b:5"])
        self.assertEqual(names("(pipe X (ridge k=huber))")[5], "CAT:huber")
        self.assertEqual(names('(sym raw "sigmoid(c12)")')[2:7], ["ID:sigmoid", "b:(", "COL", "b:1", "b:2"])
        with self.assertRaises(ValueError):
            T.spelling_allowed("RAW", "x", v)
        # the class vocabularies never hold texts the lexer gives another token class
        self.assertFalse(any(T.is_cat_name(t) for t in ("true", "FALSE", "1", "-0.0", "inf", "NaN", "1e-5", '"x"', "",
                                                        "a b", "a=b")))
        with self.assertRaises(ValueError):
            T.Vocab.from_names({"ID": ["c12"]})
        with self.assertRaises(ValueError):
            T.Vocab.from_names({"CAT": ["true"]})
        with self.assertRaises(ValueError):
            T.Vocab.from_names({"KEY": ["a=b"]})
        with self.assertRaises(ValueError):
            T.Vocab.from_names({"OP": [""]})

    def test_totality_random_unicode(self):
        rng = random.Random(5)
        v = self.vocab
        raw = 0
        pool = [chr(c) for c in range(0, 0x250)] + ["\u2028", "\u3000", "\ufeff", "\U0001f642", "\U0010ffff", "\ud7ff",
                                                     "\ue000"]
        for _ in range(2000):
            t = "".join(rng.choice(pool) for _ in range(rng.randint(0, 40)))
            ids = self.roundtrip(t, v)
            raw += ids[1] == T.RAW_BEGIN
        for t in [mutate(rng, rng.choice(FIXTURES)) for _ in range(2000)]:
            self.roundtrip(t, v)
        self.assertGreater(raw, 1900)
        with self.assertRaises(ValueError):
            T.tokenize("(pipe X (ridge k=\ud800))", v)                 # a lone surrogate is not a scalar value
        with self.assertRaises(TypeError):
            T.tokenize(b"(pipe X (ridge))", v)
        with self.assertRaises(UnicodeDecodeError):
            T.detokenize([T.byte_id(0xC3)], v)                          # strict UTF-8
        with self.assertRaises(UnicodeDecodeError):
            T.detokenize([T.byte_id(b) for b in "\ud800".encode("utf-8", "surrogatepass")], v)
        with self.assertRaises(ValueError):
            T.detokenize([v.size], v)
        self.assertEqual(T.detokenize([T.PAD, T.BOS, T.FIXED_IDS["PIPE_OPEN"], T.PAD, T.EOS, T.PAD], v), "(pipe X")

    def test_canonical_generator(self):
        rng = random.Random(11)
        v = self.vocab
        made = Counter()
        fallback = Counter()
        tries = 0
        while sum(made.values()) < 1500 and tries < 20000:
            tries += 1
            text = rand_canonical(rng)
            if text is None:
                continue
            lx = T.lex(text)
            self.assertNotEqual(lx[0].kind, "RAW", text)
            self.assertTrue(reference_form(text)[0], text)
            ids = self.roundtrip(text, v)
            made[T.program_form(text)] += 1
            for cls, _a, _b in T.fallback_spans(ids, v):
                fallback[cls] += 1
        self.assertGreater(made["pipe"], 300)
        self.assertGreater(made["sym"], 300)
        self.assertGreater(made["portfolio"], 50)
        for cls in ("OP", "KEY", "STR", "ID"):
            self.assertGreater(fallback[cls], 0, cls)
        self.assertEqual(fallback["RAW"], 0)
        report("canonical_generator", {"programs": dict(made), "tries": tries, "fallback_spans": dict(fallback)})

    def test_oracle_fuzz_inputs(self):
        v = self.vocab
        inputs = O.fuzz_inputs(n=4000, seed=3)
        canon = raw = 0
        for _cat, t in inputs:
            ids = self.roundtrip(t, v)
            is_raw = ids[1] == T.RAW_BEGIN
            raw += is_raw
            self.assertEqual(not is_raw, reference_form(t)[0], repr(t))
            if canonical_or_none(t) == t:
                canon += 1
                self.assertFalse(is_raw, repr(t))
        self.assertGreater(canon, 200)
        report("oracle_fuzz_inputs", {"inputs": len(inputs), "canonical": canon, "raw": raw})

    def test_fallback_spans_and_errors(self):
        v = self.vocab
        ids = T.tokenize("(pipe X (zz k2=zz))", v)
        spans = T.fallback_spans(ids, v)
        self.assertEqual([c for c, _a, _b in spans], ["OP", "KEY", "STR"])
        for cls, a, b in spans:
            self.assertEqual(ids[a], T.SPELL_CLASSES[cls][0])
            self.assertEqual(ids[b - 1], T.SPELL_CLASSES[cls][1])
        for bad in ([T.OP_BEGIN, T.byte_id(97)], [T.OP_END], [T.OP_BEGIN, T.COL, T.OP_END], [v.size]):
            with self.assertRaises(ValueError):
                T.fallback_spans(bad, v)
        self.assertEqual(T.fallback_spans(T.tokenize("x y", v), v), [("RAW", 1, 6)])

    def test_skeleton(self):
        cases = {
            "(pipe X (histgbm2 delta=0.000001 loss=huber lr=0.1 rounds=300 val_shuffle=true))":
                "(pipe X (histgbm2 delta=# loss=huber lr=# rounds=# val_shuffle=true))",
            '(sym affine "clip(max(c3,row_mean),-8.727099312855202e-5,min(c0,c395))")':
                '(sym affine "clip(max(c3,row_mean),-#,min(c0,c395))")',
            "(pipe X (ridge k=NaN j=inf s=\"1.5\" t=x))": "(pipe X (ridge k=# j=# s=\"1.5\" t=x))",
            "not a program 1.5": "not a program 1.5",
        }
        for src, want in cases.items():
            self.assertEqual(T.skeleton(src), want)
            self.assertEqual(T.skeleton_sha256(src), hashlib.sha256(want.encode()).hexdigest())
        self.assertEqual(T.skeleton(FIXTURES[0]), T.skeleton(FIXTURES[1]))
        self.assertNotEqual(T.skeleton("(pipe X (ridge k=a))"), T.skeleton("(pipe X (ridge k=b))"))

    @unittest.skipUnless(O.DEFAULT_ENGINE.exists(), "pinned engine copy not present")
    def test_fixtures_canonical_per_pinned_engine(self):
        engine = O.PinnedEngine(O.DEFAULT_ENGINE)
        engine.verify()                                   # sha256 == pin before anything runs
        sb = engine.seed_batch(list(FIXTURES), feature_count=8)
        self.assertEqual(sb.exit_code, 0)
        self.assertEqual(sorted(sb.accepted), sorted(FIXTURES))
        self.assertEqual((sb.changed, sb.missing, sb.failed), ({}, [], []))
        self.assertEqual(engine.invocations, engine.verifications - 1)   # +1 for the up-front check
        report("fixtures_tier_b", {"fixtures": len(FIXTURES), "accepted_unchanged": len(sb.accepted),
                                   "engine_invocations": engine.invocations})


class VocabTests(unittest.TestCase):
    def test_build_vocab_classes_and_order(self):
        progs = sorted(set(FIXTURES) | set(O.FUZZ_BASES))       # build_vocab takes DISTINCT programs
        v, counts = T.build_vocab(progs)
        v2, counts2 = T.build_vocab(list(reversed(progs)))
        self.assertEqual((v, counts), (v2, counts2))
        self.assertEqual(v.sha256, v2.sha256)
        lo = T.DATA_BASE
        for cls in T.DATA_CLASSES:
            a, b = v.class_range(cls)
            self.assertEqual(a, lo)
            names = v.names(cls)
            self.assertEqual(list(names), sorted(names, key=lambda s: s.encode()))
            self.assertEqual(set(names), set(counts[cls]))
            lo = b
        self.assertEqual(lo, v.size)
        self.assertIn("histgbm2", v.names("OP"))
        self.assertIn("max_features", v.names("KEY"))
        self.assertEqual(v.piece(v.id("KEY", "max_features")), b" max_features=")
        self.assertIn("huber", v.names("CAT"))
        self.assertNotIn("1.0", v.names("CAT"))
        self.assertNotIn("NaN", v.names("CAT"))
        self.assertNotIn("true", v.names("CAT"))
        self.assertIn("sigmoid", v.names("ID"))
        self.assertIn("Sigmoid", v.names("ID"))
        self.assertFalse(any(T._COL_RE.fullmatch(n) for n in v.names("ID")))
        self.assertEqual(counts["OP"]["norm_row"], 3)
        v3, _ = T.build_vocab(progs, min_count=2)
        self.assertNotIn("histgbm", v3.names("OP"))
        self.assertIn("norm_row", v3.names("OP"))
        with self.assertRaises(T.TokenizerStop):
            T.build_vocab(progs + ["(pipe X)"])
        with self.assertRaises(ValueError):
            T.build_vocab(progs, min_count=0)
        # a canonical key with an odd number of '"' lexes as a key only with the value that closes its quote: it can
        # never be a vocabulary token, and a train_fit program holding one is a STOP (not a bare ValueError)
        odd = '(pipe X (op k"="v\\""))'
        self.assertEqual(O.canonical(odd), odd)
        ids = T.tokenize(odd, v)
        self.assertEqual(T.detokenize(ids, v), odd)
        with self.assertRaises(T.TokenizerStop) as cm:
            T.build_vocab(progs + [odd])
        self.assertIn("KEY", str(cm.exception))

    def test_vocab_json_roundtrip_and_refusals(self):
        v, counts = T.build_vocab(list(FIXTURES))
        raw = v.to_json(extra={"counts": {"names": counts}})
        self.assertEqual(raw, v.to_json(extra={"counts": {"names": counts}}))
        doc = json.loads(raw)
        self.assertEqual(doc["schema"], T.VOCAB_SCHEMA)
        self.assertEqual(doc["feat_math_pins"], T.FEAT_MATH_PINS)
        self.assertEqual(doc["vocab_sha256"], hashlib.sha256(json.dumps(
            doc["tokens"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest())
        self.assertEqual(doc["class_ranges"]["OP"][0], 282)
        back = T.Vocab.from_json(raw)
        self.assertEqual(back, v)
        for f in FIXTURES:
            self.assertEqual(T.tokenize(f, back), T.tokenize(f, v))

        def tampered(fn):
            d = json.loads(raw)
            fn(d)
            return json.dumps(d).encode()

        bad = [
            lambda d: d.update(schema="jope.dl-dsl-vocab/v0"),
            lambda d: d["feat_math_pins"].update({"dsl.rs": "0" * 64}),
            lambda d: d.update(vocab_sha256="0" * 64),
            lambda d: d["tokens"][300].update(piece_hex=d["tokens"][301]["piece_hex"]),
            lambda d: d["tokens"].__setitem__(5, {"class": "MARKER", "name": "KEY_BEGIN", "piece_hex": "3d"}),
            lambda d: d["tokens"].insert(283, d["tokens"].pop(282)),
            lambda d: d.update(size=d["size"] + 1),
            lambda d: d["tokens"][300].update(name=5),
            lambda d: d["tokens"].append({"class": "OP", "name": "zz", "piece_hex": "7a7a"}),   # OP after ID
        ]
        for fn in bad:
            with self.assertRaises(ValueError):
                T.Vocab.from_json(tampered(fn))
        with self.assertRaises(ValueError):
            v.to_json(extra={"tokens": []})


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="p4-tok-test-"))
        rng = random.Random(3)
        gen = []
        while len(gen) < 400:
            t = rand_canonical(rng)
            if t is not None and t not in gen:
                gen.append(t)
        # corpus programs are canonical (the round trip requires Tier A canonical(text) == text); one fuzz base is not
        self.train = list(FIXTURES) + [O.canonical(b) for b in O.FUZZ_BASES] + gen[:300]
        self.parts = {"train_fit": self.train + self.train[:40],                 # duplicates across contexts
                      "dev": gen[300:350] + ["(pipe X (zz_dev_only_op))"],
                      "diag": gen[350:380],
                      "heldout": gen[380:400]}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, name: str, **kw) -> tuple:
        d = make_dict(self.tmp / ("dict-" + name), self.parts, **kw)
        out = self.tmp / ("tok-" + name)
        rec = T.build_artifacts(d, EPOCH if EPOCH.exists() else None, out, automaton=False)
        return d, out, rec

    def test_build_determinism_roundtrip_and_poison(self):
        d1, o1, rec = self.build("a")
        d2, o2, _ = self.build("b", chunks=3, shuffle_seed=9)
        for name in ("vocab.json", "lengths.json"):
            self.assertEqual((o1 / name).read_bytes(), (o2 / name).read_bytes(), name)
        vocab = T.Vocab.from_json((o1 / "vocab.json").read_bytes())
        self.assertEqual(rec["vocab_sha256"], vocab.sha256)
        self.assertEqual(rec["automaton"], "skipped")
        self.assertFalse((o1 / "grammar.json").exists())
        doc = json.loads((o1 / "vocab.json").read_bytes())
        self.assertEqual(doc["input_closure"]["distinct_programs"], len(set(self.train)))
        self.assertEqual(sum(doc["counts"]["per_id"]), doc["counts"]["train_fit_tokens"])
        self.assertEqual(doc["counts"]["per_id"][T.BOS], len(set(self.train)))
        for cls in T.DATA_CLASSES:
            a, b = vocab.class_range(cls)
            for i in range(a, b):
                self.assertEqual(doc["counts"]["per_id"][i], doc["counts"]["names"][cls][vocab.name(i)])
        lengths = json.loads((o1 / "lengths.json").read_bytes())
        self.assertEqual(lengths["max_seq"], T.ceil32(lengths["max_length"]))
        self.assertEqual(lengths["train_fit"]["all"]["max"], lengths["max_length"])
        receipt = json.loads((o1 / "build-receipt.json").read_bytes())
        self.assertEqual([f["path"] for f in receipt["train_fit_closure"]["files"]], ["train_fit/part-00000.parquet"])
        self.assertEqual(receipt["artifacts"]["vocab.json"]["sha256"], hashlib.sha256((o1 / "vocab.json").read_bytes()).hexdigest())
        # round trip over all four partitions (through the projection role 'roundtrip', never heldout_report)
        rt = T.roundtrip(d1, o1, T.PARTITIONS)
        self.assertTrue(rt["pass"])
        self.assertTrue((o1 / "roundtrip.json").exists())
        for part, s in rt["partitions"].items():
            self.assertEqual(s["roundtrip_ok"], s["programs"], part)
            self.assertEqual(s["sha_ok"], s["programs"], part)
            self.assertEqual(s["tier_a_canonical_ok"] + s["tier_a_undecidable"], s["programs"], part)
            self.assertNotIn("idempotent_ok", s)
            self.assertEqual(s["raw_programs"], 0, part)
        journal = [json.loads(x) for x in (d1 / ".access-journal.jsonl").read_bytes().splitlines()]
        self.assertEqual({(e["partition"], e["role"]) for e in journal},
                         {("dev", "roundtrip"), ("diag", "roundtrip"), ("heldout", "roundtrip")})
        # a program that round-trips but is not Tier A canonical fails the round trip
        d4 = make_dict(self.tmp / "dict-d", dict(self.parts, dev=["(pipe X (ridge q=\"a\"))"]))
        rt4 = T.roundtrip(d4, o1, ("dev",))
        self.assertFalse(rt4["pass"])
        self.assertEqual(rt4["partitions"]["dev"]["failures"][0]["kind"], "tier_a_canonical")
        shutil.rmtree(d4)
        self.assertEqual(rt["partitions"]["train_fit"]["fallback_programs"], 0)
        self.assertTrue(rt["partitions"]["train_fit"]["closure_ok"])
        self.assertGreater(rt["partitions"]["dev"]["fallback_programs"], 0)
        self.assertEqual(rt["partitions"]["train_fit"]["over_max_seq"], 0)
        # a vocabulary built from another train_fit fails the closure check
        d3 = make_dict(self.tmp / "dict-c", dict(self.parts, train_fit=self.train[:-1]))
        rt3 = T.roundtrip(d3, o1, ("train_fit",))
        self.assertFalse(rt3["partitions"]["train_fit"]["closure_ok"])
        self.assertFalse(rt3["pass"])
        shutil.rmtree(d3)
        # vocabulary closure: dev/diag/heldout poison leaves every artifact byte-identical; train_fit poison does not
        res = T.poison_test(d1, o1, seed=4, automaton=False)
        self.assertTrue(res["pass"], res)
        self.assertEqual(res["artifacts"], ["lengths.json", "vocab.json"])
        self.assertTrue(all(res["identical_after_dev_diag_heldout_poison"].values()))
        self.assertTrue(res["changed_after_train_fit_poison"]["vocab.json"])
        self.assertEqual(sorted(os.listdir(self.tmp)), ["dict-a", "dict-b", "tok-a", "tok-b"])
        report("artifacts_synthetic", {"vocab_size": vocab.size, "max_seq": lengths["max_seq"],
                                       "fallback_rate": {p: s["fallback_rate"] for p, s in rt["partitions"].items()},
                                       "poison": {k: res[k] for k in ("identical_after_dev_diag_heldout_poison",
                                                                      "changed_after_train_fit_poison")}})

    def test_automaton_wiring_and_poison_cover_grammar(self):
        """build_artifacts writes signatures.json and grammar.json through dltrain.automaton's API when present
        (a stand-in here: the real module belongs to the automaton packet), and the poison test covers them."""
        from unittest import mock
        seen = []

        def build_signatures(dsls):
            dsls = list(dsls)
            seen.append(len(dsls))
            return {"ops": sorted({lx.text for d in dsls for lx in T.lex(d) if lx.kind == "OP"})}

        def build_grammar(signatures, vocab, max_seq):
            return {"schema": "stand-in", "vocab_sha256": vocab.sha256, "max_seq": max_seq, "ops": signatures["ops"]}

        d = make_dict(self.tmp / "dict-auto", self.parts)
        out = self.tmp / "tok-auto"
        with mock.patch.object(T, "_automaton_api", lambda mode: (build_signatures, build_grammar)):
            rec = T.build_artifacts(d, None, out)
            self.assertEqual(rec["automaton"], "built")
            self.assertEqual(seen, [len(set(self.train))])                # train_fit distinct programs only
            grammar = json.loads((out / "grammar.json").read_bytes())
            self.assertEqual(grammar["max_seq"], json.loads((out / "lengths.json").read_bytes())["max_seq"])
            res = T.poison_test(d, out, seed=1)
        self.assertTrue(res["pass"], res)
        self.assertEqual(res["artifacts"], ["grammar.json", "lengths.json", "signatures.json", "vocab.json"])
        self.assertTrue(all(res["identical_after_dev_diag_heldout_poison"].values()))
        self.assertTrue(res["changed_after_train_fit_poison"]["signatures.json"])
        self.assertIsNone(T._automaton_api(False))
        if T._automaton_api(None) is None:          # the real dltrain.automaton has no build API yet
            with self.assertRaises(T.TokenizerStop):
                T._automaton_api(True)

    def test_stops(self):
        parts = dict(self.parts, train_fit=self.train + ["(pipe X)"])
        d = make_dict(self.tmp / "dict-raw", parts)
        with self.assertRaises(T.TokenizerStop):
            T.build_artifacts(d, None, self.tmp / "tok-raw", automaton=False)
        parts = dict(self.parts, train_fit=["(pipe X" + " (op k=1)" * 200 + ")"])
        d = make_dict(self.tmp / "dict-long", parts)
        with self.assertRaises(T.TokenizerStop):
            T.build_artifacts(d, None, self.tmp / "tok-long", automaton=False)      # max_seq > 512
        d = make_dict(self.tmp / "dict-empty", dict(self.parts, train_fit=[]))
        with self.assertRaises(T.TokenizerStop):
            T.build_artifacts(d, None, self.tmp / "tok-empty", automaton=False)

    def test_cli(self):
        d = make_dict(self.tmp / "dict-cli", self.parts)
        out = self.tmp / "tok-cli"

        def run(argv):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = T.main(argv)
            return code, json.loads(buf.getvalue().strip().splitlines()[-1])

        code, doc = run(["build", "--dict", str(d), "--out", str(out), "--no-automaton", "--all-partitions"])
        self.assertEqual(code, 0, doc)
        self.assertEqual(doc["roundtrip"]["train_fit"]["fallback_programs"], 0)
        receipt = json.loads((out / "build-receipt.json").read_bytes())
        if EPOCH.exists():                                   # the default epoch is pinned in the receipt
            self.assertEqual(receipt["epoch"]["sha256"], hashlib.sha256(EPOCH.read_bytes()).hexdigest())
        code, doc = run(["roundtrip", "--dict", str(d), "--tok", str(out), "--partitions", "dev", "diag"])
        self.assertEqual(code, 0)
        self.assertEqual(sorted(doc["partitions"]), ["dev", "diag"])
        code, doc = run(["build", "--poison-test", "--dict", str(d), "--out", str(out), "--no-automaton"])
        self.assertEqual(code, 0, doc)
        code, doc = run(["show", "--vocab", str(out / "vocab.json"), "--dsl", FIXTURES[0]])
        self.assertTrue(doc["roundtrip"])
        bad = make_dict(self.tmp / "dict-bad", dict(self.parts, train_fit=["(pipe X)"]))
        code, doc = run(["build", "--dict", str(bad), "--out", str(self.tmp / "tok-bad"), "--no-automaton"])
        self.assertEqual(code, 2)
        self.assertEqual(doc["stop"], "TokenizerStop")
        code, doc = run(["roundtrip", "--dict", str(d), "--tok", str(out), "--all-partitions", "--require-automaton"])
        self.assertEqual((code, doc["stop"]), (2, "TokenizerStop"))           # no grammar.json: the gate cannot run

    def test_cli_runs_the_automaton_gate_and_walks(self):
        """build --all-partitions and roundtrip run the automaton gate (SYNTAX on every partition, POLICY 100% of
        train_fit) and masked walks next to the round trip, and fail when the gate fails."""
        from tests_p4 import test_automaton as TA          # programs POLICY is known to accept within every cap
        parts = {"train_fit": list(TA.TRAIN), "dev": ["(pipe X (zz_dev_only_op))"], "diag": [TA.TRAIN[0]],
                 "heldout": [TA.TRAIN[1]]}
        d = make_dict(self.tmp / "dict-gate", parts)
        out = self.tmp / "tok-gate"

        def run(argv):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = T.main(argv)
            return code, json.loads(buf.getvalue().strip().splitlines()[-1])

        code, doc = run(["build", "--dict", str(d), "--out", str(out), "--all-partitions", "--require-automaton",
                         "--walks", "60"])
        self.assertEqual(code, 0, doc)
        gates = doc["automaton_gates"]
        self.assertTrue(gates["pass"])
        self.assertEqual(gates["gate"]["train_fit"]["policy_accepted"], gates["gate"]["train_fit"]["programs"])
        self.assertEqual(sorted(gates["walks"]), ["policy", "syntax"])
        self.assertTrue((out / "automaton-gate.json").exists())
        code, doc = run(["roundtrip", "--dict", str(d), "--tok", str(out), "--partitions", "dev", "--walks", "20"])
        self.assertEqual((code, doc["automaton_gates"]["pass"]), (0, True))
        # a gate failure fails the command: a train_fit the grammar was not built from
        d2 = make_dict(self.tmp / "dict-gate2", dict(parts, train_fit=parts["train_fit"] + ["(pipe X (zz_new_op))"]))
        code, doc = run(["roundtrip", "--dict", str(d2), "--tok", str(out), "--partitions", "train_fit", "--walks", "0"])
        self.assertEqual(code, 1)
        self.assertFalse(doc["automaton_gates"]["pass"])


class CorpusSampleTests(unittest.TestCase):
    def test_corpus_sample_half_split(self):
        t0 = time.monotonic()
        sample = corpus_sample()
        contexts = sample["contexts"]
        self.assertLessEqual(len(contexts), 40)
        self.assertTrue(all(c["native_ok"] for c in contexts))
        datasets = sorted({c["dataset_hash"] for c in contexts})
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(datasets)
        fit_ds = set(datasets[:len(datasets) // 2])
        progs = {}
        fit, test = set(), set()
        rows = 0
        for c in contexts:
            for sha, dsl in c["rows"]:
                rows += 1
                self.assertEqual(O.program_sha256(dsl), sha)
                progs[sha] = dsl
                (fit if c["dataset_hash"] in fit_ds else test).add(sha)
        test_only = test - fit
        vocab, counts = T.build_vocab(progs[s] for s in sorted(fit))
        t1 = time.monotonic()
        stats = {"fit": T.TokenStats(vocab, canonical=True), "test_only": T.TokenStats(vocab, canonical=True)}
        forms = Counter()
        skeletons = set()
        spelled_bad = 0
        for sha in sorted(fit | test):
            dsl = progs[sha]
            group = "fit" if sha in fit else "test_only"
            ids = stats[group].add(dsl, sha)
            forms[T.program_form(dsl)] += 1
            skeletons.add(T.skeleton_sha256(dsl))
            for cls, a, b in T.fallback_spans(ids, vocab):
                text = T.detokenize(ids[a:b], vocab)
                if cls == "KEY":
                    text = text[1:-1]
                spelled_bad += cls == "RAW" or not T.spelling_allowed(cls, text, vocab)
        t2 = time.monotonic()
        fit_s, test_s = stats["fit"].summary(), stats["test_only"].summary()
        for s in (fit_s, test_s):
            self.assertEqual(s["raw_programs"], 0)
            self.assertEqual(s["roundtrip_ok"], s["programs"])
            self.assertEqual(s["sha_ok"], s["programs"])
            self.assertEqual(s["tier_a_canonical_ok"] + s["tier_a_undecidable"], s["programs"])
            self.assertTrue(s["pass"])
        self.assertEqual(fit_s["fallback_programs"], 0)              # by construction (min_count=1)
        self.assertEqual(spelled_bad, 0)
        self.assertEqual(forms["raw"], 0)
        self.assertGreater(fit_s["programs"], 1000)
        self.assertGreater(test_s["programs"], 1000)
        n_all = fit_s["programs"] + test_s["programs"]
        # the 161 non-printable byte tokens never occur in the corpus; every fit name occurs
        per_id = [a + b for a, b in zip(stats["fit"].per_id, stats["test_only"].per_id)]
        nonprintable = sum(per_id[T.byte_id(b)] for b in range(256) if not 0x20 <= b < 0x7F)
        self.assertEqual(nonprintable, 0)
        for cls in T.DATA_CLASSES:
            a, b = vocab.class_range(cls)
            self.assertTrue(all(stats["fit"].per_id[i] > 0 for i in range(a, b)), cls)
        mix = Counter()
        for i, n in enumerate(stats["fit"].per_id):
            mix[vocab.cls(i)] += n
        numeral_bytes = sum(stats["fit"].per_id[T.byte_id(c)] for c in b"0123456789.-+eE")
        total = sum(mix.values())
        report("corpus_sample", {
            "contexts": len(contexts), "datasets": {"fit": len(fit_ds), "test": len(datasets) - len(fit_ds)},
            "rows": rows, "distinct_programs": n_all, "forms": dict(forms),
            "fit_programs": fit_s["programs"], "test_programs": len(test), "test_only_programs": test_s["programs"],
            "vocab": {"size": vocab.size, **{c: len(vocab.names(c)) for c in T.DATA_CLASSES}},
            "roundtrip_ok": fit_s["roundtrip_ok"] + test_s["roundtrip_ok"],
            "sha_ok": fit_s["sha_ok"] + test_s["sha_ok"],
            "tier_a_canonical_ok": fit_s["tier_a_canonical_ok"] + test_s["tier_a_canonical_ok"],
            "tier_a_undecidable": fit_s["tier_a_undecidable"] + test_s["tier_a_undecidable"],
            "raw": fit_s["raw_programs"] + test_s["raw_programs"],
            "fallback": {"fit_programs": fit_s["fallback_programs"],
                         "test_only_programs": test_s["fallback_programs"],
                         "test_only_rate": test_s["fallback_rate"], "test_only_spans": test_s["fallback_spans"]},
            "str_literal_programs": {"fit": fit_s["str_literal_programs"], "test_only": test_s["str_literal_programs"]},
            "nonprintable_byte_tokens": nonprintable,
            "fit_token_mix": {c: round(n / total, 4) for c, n in sorted(mix.items())},
            "fit_numeral_char_share": round(numeral_bytes / total, 4),
            "length_all": {k: v for k, v in fit_s["length"]["all"].items() if k != "histogram"},
            "length_pipe": {k: v for k, v in fit_s["length"]["pipe"].items() if k != "histogram"},
            "length_sym": {k: v for k, v in fit_s["length"]["sym"].items() if k != "histogram"},
            "over_128": fit_s["over_product_max_program_tokens"] + test_s["over_product_max_program_tokens"],
            "distinct_skeletons": len(skeletons),
            "seconds": {"read_and_vocab": round(t1 - t0, 2), "tokenize_checks": round(t2 - t1, 2)},
            "programs_per_s": round(n_all / max(t2 - t1, 1e-9))})


if __name__ == "__main__":
    unittest.main()
