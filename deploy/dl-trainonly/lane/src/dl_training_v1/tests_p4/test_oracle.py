"""PLAN §5 P4 oracle tests: Tier A (dltrain.oracle.canonical) against the Rust #[test] vectors and the
normalization rules, the width/lane ports, caps and static semantics, the Tier B protocol (in-process
stub and the pinned engine), a <= 40-context corpus sample, and the >= 20,000-input differential fuzz.

The pinned engine is run only through dltrain.oracle.PinnedEngine (sha256-verified before every batch).
Corpus records are projected at parse time to program.{dsl, program_sha256}, status and the abandoned
top-level program_sha256; no other field is read or printed. Counts are printed as `P4-ORACLE {...}`.
"""
import contextlib
import hashlib
import io
import json
import os
import random
import re
import tempfile
import time
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path

from dltrain import oracle as O

EPOCH = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/epochs/epoch-000.json")
SAMPLE_PER_STRATUM = 4            # 5 folds x 2 cohorts x 4 = 40 contexts (the hard cap)
SAMPLE_SEED = 20261005
FUZZ_N = 24000
FUZZ_SEED = 20261005
VENDOR_TESTS = O.FEAT_MATH_SRC.parent / "tests"
VALUE_RE = re.compile(r'=([^\s()"=]+)')       # bare key=value atoms (feature counting only)


def report(name: str, doc: dict) -> None:
    print("P4-ORACLE " + json.dumps({"test": name, **doc}, sort_keys=True, ensure_ascii=True), flush=True)


def pipe_with(value: str) -> str:
    return "(pipe X (op k=" + value + "))"


def tie_values(text: str) -> list:
    out = []
    for m in VALUE_RE.finditer(text):
        v = O.parse_f64(m.group(1))
        if v is not None and O.is_shortest_tie(v):
            out.append(v)
    return out


def sym_body(text: str):
    i, j = text.find('"'), text.rfind('"')
    return text[i + 1:j] if 0 <= i < j else None


def balanced(n: int) -> str:
    return "c0" if n == 1 else "(" + balanced(n // 2) + "+" + balanced(n - n // 2) + ")"


# ---------------------------------------------------------------------------------------------
# Corpus sample (module-level cache: read once per process, <= 40 contexts)
# ---------------------------------------------------------------------------------------------
_CORPUS = None


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
        native_ok = len(data) == src["bytes"] and hashlib.sha256(data).hexdigest() == src["sha256"]
        rows = []
        for line in data.splitlines():
            rec = json.loads(line)
            prog = rec["program"]
            rows.append((prog["program_sha256"], prog["dsl"], rec.get("status"), rec.get("program_sha256")))
        del data
        pref = ctx["prepared_ref"]
        pbytes = Path(pref["path"]).read_bytes()
        prepared_ok = hashlib.sha256(pbytes).hexdigest() == pref["sha256"]
        pctx = json.loads(pbytes)["context"]
        contexts.append({"context_sha256": ctx["context_sha256"], "cohort": ctx["cohort"], "fold": ctx["fold"],
                         "native_ok": native_ok, "prepared_ok": prepared_ok, "rows": rows,
                         "encoded_features": int(pctx["encoded_features"]), "train_rows": int(pctx["train_rows"])})
    _CORPUS = {"contexts": contexts}
    return _CORPUS


def engine() -> O.PinnedEngine:
    """The pinned copy, verified once up front; every batch then verifies it again before running."""
    e = O.PinnedEngine(O.DEFAULT_ENGINE)
    e.verify()
    return e


# ---------------------------------------------------------------------------------------------
# Pins and primitives
# ---------------------------------------------------------------------------------------------
class SourcePinTests(unittest.TestCase):
    def test_ported_sources_match_pins(self):
        for rel, sha in O.FEAT_MATH_PINS.items():
            self.assertEqual(hashlib.sha256((O.FEAT_MATH_SRC / rel).read_bytes()).hexdigest(), sha, rel)
        for rel, sha in O.ENGINE_SOURCE_PINS.items():
            self.assertEqual(hashlib.sha256((O.LOCAL_LAMBDA / rel).read_bytes()).hexdigest(), sha, rel)

    def test_rust_white_space_is_not_python_isspace(self):
        self.assertEqual(len(O.RUST_WHITESPACE), 25)
        py = {chr(c) for c in range(0x110000) if chr(c).isspace()}
        self.assertEqual(py - set(O.RUST_WHITESPACE), {"\x1c", "\x1d", "\x1e", "\x1f"})
        self.assertEqual(set(O.RUST_WHITESPACE) - py, set())
        self.assertTrue(O.is_rust_whitespace("\xa0"))
        self.assertFalse(O.is_rust_whitespace("\x1c"))
        self.assertEqual(O.rust_trim("\x85\u3000 x \u2029"), "x")
        self.assertEqual(O.rust_trim("\x1cx\x1f"), "\x1cx\x1f")

    def test_numeric_primitives(self):
        self.assertEqual(O.parse_i64("+5"), 5)
        self.assertEqual(O.parse_i64("-0"), 0)
        self.assertEqual(O.parse_i64("0" * 5000 + "7"), 7)
        self.assertIsNone(O.parse_i64("9" * 5000))
        self.assertIsNone(O.parse_i64("9223372036854775808"))
        self.assertEqual(O.parse_i64("-9223372036854775808"), O.I64_MIN)
        for bad in ("1_000", " 1", "1 ", "\u0665", "\uff15", "+", "-", "", "0x1"):
            self.assertIsNone(O.parse_i64(bad), bad)
            self.assertIsNone(O.parse_f64(bad), bad)
        for good, want in (("1.", 1.0), (".5", 0.5), ("1e5", 1e5), ("1E-5", 1e-5), ("+.5e+1", 5.0),
                           ("INFINITY", float("inf")), ("-Inf", float("-inf"))):
            self.assertEqual(O.parse_f64(good), want, good)
        self.assertTrue(O.parse_f64("nAn") != O.parse_f64("nAn"))
        for bad in (".", "e5", "1e", "1e+", "1.2.3", "infinit", "\u0131nf", "1.5f", "nan(1)"):
            self.assertIsNone(O.parse_f64(bad), bad)
        cases = {0.1: "0.1", 1e-5: "0.00001", -1.5e-5: "-0.000015", 1e-13: "0.0", -1e-13: "-0.0", -0.0: "-0.0",
                 3.0000000000001: "3.0", 2.5: "2.5", 1e16: "10000000000000000.0", 4503599627370495.5: "4503599627370495.5",
                 float("nan"): "NaN", float("inf"): "inf", float("-inf"): "-inf", 1.5e-10: "0.00000000015",
                 0.5408000000000001: "0.5408000000000001", 1e300: "%.1f" % 1e300,
                 # exact ties between the two shortest candidates: Rust rounds up in magnitude, repr to even
                 1000000000000000.25: "1000000000000000.3", -1186771716226682.25: "-1186771716226682.3",
                 1948612774330343.25: "1948612774330343.3", -77383519778672.125: "-77383519778672.13",
                 1000000000000000.75: "1000000000000000.8"}
        for v, want in cases.items():
            self.assertEqual(O.format_f64(v), want, repr(v))
        self.assertEqual(repr(1000000000000000.25), "1000000000000000.2")    # CPython's tie goes to even
        self.assertTrue(O.is_shortest_tie(1000000000000000.25))
        self.assertFalse(O.is_shortest_tie(0.1))
        self.assertFalse(O.is_shortest_tie(4503599627370495.5))             # exact in 17 digits: no tie
        rng = random.Random(7)
        for _ in range(200):     # the tie generator yields ties; their printing is checked against the engine
            v = O._tie_double(rng)
            self.assertTrue(O.is_shortest_tie(v), repr(v))
            self.assertEqual(float(O.format_f64(v)), v, repr(v))
        self.assertEqual(O.as_usize(("Float", 5.9)), 5)
        self.assertEqual(O.as_usize(("Float", -0.0)), 0)
        self.assertEqual(O.as_usize(("Float", float("inf"))), O.U64_MAX)
        self.assertEqual(O.as_usize(("Float", 1e300)), O.U64_MAX)
        self.assertIsNone(O.as_usize(("Float", float("nan"))))
        self.assertIsNone(O.as_usize(("Int", -1)))
        self.assertIsNone(O.as_usize(("Str", "5")))
        self.assertIsNone(O.as_usize(("Bool", True)))


# ---------------------------------------------------------------------------------------------
# Rust #[test] vectors
# ---------------------------------------------------------------------------------------------
class RustVectorTests(unittest.TestCase):
    def test_dsl_rs_vectors(self):
        # parser_round_trips_canonical_pipeline
        src = "(pipe X (robust_scale) (fft_mag axis=features) (polar) (rff_cos n=32 seed=7) (cheb degree=4) (ridge alpha=1.0))"
        expr = O.parse_dsl(src)
        self.assertEqual(len(expr.stages), 6)
        self.assertEqual(O.as_usize(dict(expr.stages[3][1])["n"]), 32)
        self.assertEqual(O.render(expr), src)
        self.assertEqual(O.canonical(src), src)
        # parser_rejects_unsupported_shape
        for bad in ("(robust_scale X)", "(pipe X)"):
            with self.assertRaises(O.CanonicalError):
                O.parse_dsl(bad)
            with self.assertRaises(O.CanonicalError):
                O.canonical(bad)
        # parser_round_trips_quoted_param_values
        src = '(pipe X (map expr="sigmoid(col(0))") (derive expr="row_mean + c0"))'
        expr = O.parse_dsl(src)
        self.assertEqual(dict(expr.stages[0][1])["expr"], ("Str", "sigmoid(col(0))"))
        self.assertEqual(O.render(expr), src)
        self.assertEqual(O.canonical(src), src)

    def test_genome_rs_vectors(self):
        # symbolic_wire_form_rejects_unknown_modes_and_incomplete_delimiters
        for source in ('(sym bogus "col(0)")', '(sym affine "col(0)"', "(sym raw col(0))", '(sym raw "col(0))',
                       '(sym raw col(0)")', '(sym raw ""col(0)"")', '(sym raw "col(0)") trailing'):
            with self.assertRaises(O.CanonicalError, msg=source):
                O.parse_genome(source)
        for mode in ("raw", "affine"):
            source = f'(sym {mode} "col(0)")'
            self.assertEqual(O.render(O.parse_genome(source)), source)
        # portfolio_form_parses_and_canonicalizes_order_preserving
        src = ("(portfolio (member (pipe X (ridge_gcv))) (member (pipe X (blom) (gpr kernel=rbf+linear steps=40))) "
               "(blend kind=nnls folds=5 repeats=auto guard=clip))")
        g = O.parse_genome(src)
        self.assertIsInstance(g, O.Portfolio)
        self.assertEqual(len(g.members), 2)
        self.assertTrue(g.repeats_auto)
        canon = O.render(g)
        self.assertEqual(canon, "(portfolio (member (pipe X (ridge_gcv))) (member (pipe X (blom) (gpr kernel=rbf+linear "
                                "steps=40))) (blend folds=5 guard=clip kind=nnls repeats=auto))")
        self.assertEqual(O.render(O.parse_genome(canon)), canon)
        flipped = ("(portfolio (member (pipe X (blom) (gpr kernel=rbf+linear steps=40))) (member (pipe X (ridge_gcv))) "
                   "(blend))")
        self.assertNotEqual(O.render(O.parse_genome(flipped)), canon)
        # portfolio_form_rejects_bad_shapes
        bad = ["(portfolio (blend))", "(portfolio" + " (member (pipe X (ridge)))" * 13 + " (blend))",
               "(portfolio (member (pipe X (portfolio preset=core))) (blend))",
               "(portfolio (member (pipe X (ridge))) (blend kind=caruana))",
               "(portfolio (member (pipe X (ridge))) (blend folds=3))",
               "(portfolio (member (pipe X (ridge))) (blend repeats=2))",
               "(portfolio (member (pipe X (ridge))) (blend guard=none))",
               "(portfolio (member (pipe X (ridge))) (blend bogus=1))",
               "(portfolio (blend) (member (pipe X (ridge))))"]
        for b in bad:
            with self.assertRaises(O.CanonicalError, msg=b):
                O.parse_genome(b)
        g = O.parse_genome("(portfolio (member (pipe X (ridge))) (blend repeats=1))")
        self.assertFalse(g.repeats_auto)
        # twelve members is the cap (accepted)
        self.assertEqual(len(O.parse_genome("(portfolio" + " (member (pipe X (ridge)))" * 12 + " (blend))").members), 12)

    def test_math_expr_rs_vectors(self):
        for ok in ("clip(1,(-1)^0.5,(-1)^0.5)", "min(clip(1,(-1)^0.5,(-1)^0.5),7)+1", "clip(1,(-1)^0.5,3)",
                   "sigmoid(x) + c0", "clip(row_l2 + row_mean, 0, 10)", "step(c0, c1)"):
            O.parse_math(ok)
            self.assertEqual(O.canonical(f'(sym affine "{ok}")'), f'(sym affine "{ok}")')
        deep = "sigmoid(" * 17 + "x" + ")" * 17
        for bad in ("step(c0)", "made_up(x)", deep):
            with self.assertRaises(O.CanonicalError, msg=bad):
                O.parse_math(bad)
        self.assertEqual(O.parse_math("clip(row_l2 + row_mean, 0, 10)").nodes, 6)

    def test_vector_table(self):
        doc = O.replica_check()
        report("replica_check_tier_a", {"vectors": doc["vectors"], "passed": doc["passed"]})
        self.assertTrue(doc["pass"], doc["failures"])

    def test_vendor_integration_vectors(self):
        """Extra vectors from vendor tests/legacy_v2_freeze.rs and tests/mathexpr_current_semantics.rs."""
        legacy = (VENDOR_TESTS / "legacy_v2_freeze.rs").read_text()
        champion = (O.FEAT_MATH_SRC.parent / "assets/champions/8944cbdf.dsl").read_text()
        fixtures = [
            ("(pipe X (standardize append=true) (rff_cos n=4 seed=9 gamma=1.5 append=true) (ridge lambda_frac=0.05))",
             "(pipe X (standardize append=true) (rff_cos append=true gamma=1.5 n=4 seed=9) (ridge lambda_frac=0.05))",
             "65ebc7f874a86b96c44d534cc83b95a60e898d6a85c6cc1c31db7695f5d03611"),
            ("(pipe X (lag k=2 append=true) (roll k=3) (ridge))", "(pipe X (lag append=true k=2) (roll k=3) (ridge))",
             "8bc5bc757dc1e05bcc62a4b493ddccc623ee9dc95c9a96488974657bcb29e888"),
            ("(pipe X (pair_prod topk=3 append=true) (target_bin_encode bins=4) (ensemble rounds=8 depth=2 lambda_frac=0.02))",
             "(pipe X (pair_prod append=true topk=3) (target_bin_encode bins=4) (ensemble depth=2 lambda_frac=0.02 rounds=8))",
             "4bc89be6d8775ab9a3dda50b8e93152a9f4addd63f594939ef86cadb865800dd"),
            ('(sym affine "sigmoid(col(0)) + 0.5*col(1)")', '(sym affine "sigmoid(col(0)) + 0.5*col(1)")',
             "319147c804ab23dda6e4645f00ab0a63362297003378e6f3175e1c1be441115e"),
            (champion, champion, "0bc6e75d8f12173ac6e2acf432233e48457cf773e1bedef56f536384513c158d"),
        ]
        for src, canon, sha in fixtures:
            self.assertIn(sha, legacy)
            self.assertEqual(O.canonical(src), canon)
            self.assertEqual(O.program_sha256(canon), sha)
        for src in ("", "garbage", "(pipe X (ridge)", "(portfolio (blend))"):
            with self.assertRaises(O.CanonicalError):
                O.canonical(src)
        self.assertIn("clip(col(0), nan, nan)", (VENDOR_TESTS / "mathexpr_current_semantics.rs").read_text())
        with self.assertRaises(O.CanonicalError):
            O.parse_math("clip(col(0), nan, nan)")
        O.parse_math("clip(col(0), col(1), col(2))")
        root = O.parse_math("-2^2").ast
        self.assertEqual((root[0], root[1], root[2][0]), ("b", "^", "n"))   # (-2)^2
        with self.assertRaises(O.CanonicalError):
            O.parse_math("1e309")


# ---------------------------------------------------------------------------------------------
# Normalization rules (where Python's defaults differ from Rust)
# ---------------------------------------------------------------------------------------------
VALUE_CASES = [  # value text -> canonical value text (None: rejected)
    ("1e-5", "0.00001"), ("1.50", "1.5"), ("+5", "5"), ("05", "5"), ("-05", "-5"), ("-0", "0"), ("TRUE", "true"),
    ("False", "false"), ("nan", "NaN"), ("-nan", "NaN"), ("NaN", "NaN"), ("inf", "inf"), ("+Infinity", "inf"),
    ("-INF", "-inf"), ("1_000", "1_000"), ("9223372036854775807", "9223372036854775807"),
    ("9223372036854775808", "9223372036854775808.0"), ("-9223372036854775808", "-9223372036854775808"),
    ("-9223372036854775809", "-9223372036854775808.0"), ("1e400", "inf"), ("-1e-400", "-0.0"),
    ("3.0000000000001", "3.0"), ("1e-13", "0.0"), ("-1e-13", "-0.0"), ("1.", "1.0"), (".5", "0.5"),
    ("1e16", "10000000000000000.0"), ("0.000020741850001116463", "0.000020741850001116463"),
    ("1.5e-10", "0.00000000015"), ("\u0665", "\u0665"), ("\uff15", "\uff15"), ("\u0131nf", "\u0131nf"),
    ("0x10", "0x10"), ("1e", "1e"), (".", "."), ("-", "-"), ("+", "+"), ("1.2.3", "1.2.3"),
    ('"x"', "x"), ('"a b"', '"a b"'), ('""', '""'), ("", '""'), ('"5"', "5"), ('"true"', "true"),
    ('"1.50"', None), ('"TRUE"', None), ('"nan"', None), ('"NaN"', "NaN"), ('"inf"', "inf"), ('"+inf"', None),
    ('"a\\"b"', '"a\\"b"'), ('"a\\\\b"', "ab"), ('"a\\\\\\\\b"', "a\\b"), ('"a\\\\\\\\ b"', None),
    ('"x=y"', '"x=y"'), ('"(x)"', '"(x)"'), ('"\u0131nf"', "\u0131nf"), ('"a\xa0b"', '"a\xa0b"'),
    ('"a\x1cb"', "a\x1cb"), ("a\x1cb", "a\x1cb"), ('"abc', None), ('a"b', None), ('a"b"c', '"a\\"b\\"c"'),
]

BLEND_CASES = [  # blend text -> repeats in the canonical blend (None: rejected)
    ("", "auto"), ("(blend)", "auto"), ("(blend repeats=1 guard=clip folds=5 kind=nnls)", "1"),
    ("(blend folds=5.9)", "auto"), ("(blend folds=5.0 repeats=1.9)", "1"), ("(blend folds=+5 repeats=+1)", "1"),
    ("(blend folds=05)", "auto"), ("(blend folds=5e0)", "auto"), ('(blend repeats="auto")', "auto"),
    ('(blend kind="nnls" guard="clip")', "auto"), ("(blend folds=6)", None), ("(blend folds=4.99)", None),
    ('(blend folds="5")', None), ("(blend folds=-0.0)", None), ("(blend folds=nan)", None), ("(blend folds=inf)", None),
    ("(blend repeats=0.9)", None), ("(blend repeats=2)", None), ("(blend repeats=AUTO)", None),
    ("(blend repeats=true)", None), ('(blend repeats="1")', None), ("(blend guard=CLIP)", None),
    ("(blend kind=NNLS)", None), ("(blend kind=1)", None), ("(blend bogus=1)", None), ("(blend) (blend)", None),
    ("(blend repeats=1) (member (pipe X (ridge)))", None),
]

SYM_CASES = [
    ('(sym  affine "c0")', '(sym affine "c0")'), ('(sym affine  "c0")', '(sym affine "c0")'),
    ('(sym affine "c0" )', '(sym affine "c0")'), ('  (sym affine "c0")\n', '(sym affine "c0")'),
    ('(sym affine \u3000"c0")', '(sym affine "c0")'), ('(sym affine "c0"\xa0)', '(sym affine "c0")'),
    ('(sym affine " c0 + c1 ")', '(sym affine " c0 + c1 ")'), ('(sym raw "C0+ROW_MEAN")', '(sym raw "C0+ROW_MEAN")'),
    ('(sym affine "c0\xa0+c1")', '(sym affine "c0\xa0+c1")'), ('(sym\taffine "c0")', None), ('(sym affine\t"c0")', None),
    ('(sym affine\xa0"c0")', None), ('(sym AFFINE "c0")', None), ('(sym Raw "c0")', None), ('(sym affine "c0")\x1c', None),
    ('\x1c(sym affine "c0")', None), ('(sym affine "c0\x1c")', None), ('(sym affine "c0" "c1")', None),
    ('(symaffine "c0")', None), ("(sym affine c0)", None), ('(sym affine "")', None),
]

MATH_CASES = [
    ("c0", True), ("C0", True), ("SIN(C0)", True), ("mean", True), ("row_norm", True), ("col5", True), ("COL(5)", True),
    ("col(5.0)", True), ("col(2.0000000000001)", True), ("col((3))", True), ("col(+3)", True), ("x", True),
    ("row_sum+row_l1", True), ("c007", True), ("1.", True), (".5", True), ("1.e5", True), ("1E+5", True),
    ("8.727099312855202e-5", True), ("-2^2", True), ("+c0", True), ("--c0", True), ("c18446744073709551615", True),
    ("c", False), ("col", False), ("col()", False), ("col(1,2)", False), ("col(-1)", False), ("col(1.5)", False),
    ("col(c1)", False), ("c18446744073709551616", False), ("1e309", False), ("1e", False), ("1.2.3", False),
    (".", False), ("inf", False), ("nan", False), ("made_up(x)", False), ("sin()", False), ("sin(c0,c1)", False),
    ("min(c0)", False), ("clip(c0,c1)", False), ("step(c0,c1,c2)", False), ("c0 c1", False), ("(c0", False),
    ("c0)", False), ("", False), ("c0\u00e9", False), ("2c0", False), ("2exp(1)", False), ("c0,c1", False),
    ("-" + balanced(32), True), (balanced(33), False), ("sigmoid(" * 15 + "x" + ")" * 15, True),
    ("sigmoid(" * 16 + "x" + ")" * 16, False), ("-" * 15 + "c0", True), ("-" * 16 + "c0", False),
]


class NormalizationTests(unittest.TestCase):
    def check(self, src, want):
        if want is None:
            with self.assertRaises(O.CanonicalError, msg=repr(src)):
                O.canonical(src)
        else:
            self.assertEqual(O.canonical(src), want, repr(src))

    def test_whitespace(self):
        self.check("\xa0(pipe X (ridge))\xa0", "(pipe X (ridge))")
        self.check("(pipe\u3000X\t(ridge)\n)", "(pipe X (ridge))")
        self.check("(pipe X(ridge))", "(pipe X (ridge))")
        self.check("( pipe X (ridge))", None)
        for bad in ("\x1c(pipe X (ridge))", "(pipe X (ridge))\x1c", "(pipe X\x1c(ridge))", "\x1f(pipe X (ridge))"):
            self.check(bad, None)
        self.check("(pipe X (ridge\x1c))", "(pipe X (ridge\x1c))")       # U+001C is an atom char, not a separator

    def test_param_order_and_duplicates(self):
        self.check("(pipe X (histgbm rounds=10 depth=3 lr=0.1))", "(pipe X (histgbm depth=3 lr=0.1 rounds=10))")
        self.check("(pipe X (ridge a=1 a=2))", "(pipe X (ridge a=2))")
        self.check("(pipe X (ridge a=2 b=1 a=1))", "(pipe X (ridge a=1 b=1))")
        self.check("(pipe X (ridge \u00e9=1 z=2 a=3))", "(pipe X (ridge a=3 z=2 \u00e9=1))")   # code-point order
        self.check("(pipe X (ridge =5))", "(pipe X (ridge =5))")
        self.check("(pipe X (ridge \"k=v\"))", "(pipe X (ridge k=v))")
        self.check("(pipe X (\"ridge\"))", "(pipe X (ridge))")
        self.check("(pipe X (\"a=b\"))", "(pipe X (a=b))")
        self.check("(pipe X (\"a b\"))", None)
        self.check("(pipe X (\"\"))", None)
        self.check("(pipe X (ridge k))", None)

    def test_values(self):
        for value, want in VALUE_CASES:
            self.check(pipe_with(value), None if want is None else pipe_with(want))

    def test_blend(self):
        for blend, repeats in BLEND_CASES:
            src = "(portfolio (member (pipe X (ridge))) " + blend + ")"
            want = None if repeats is None else (
                "(portfolio (member (pipe X (ridge))) (blend folds=5 guard=clip kind=nnls repeats=" + repeats + "))")
            self.check(src, want)

    def test_sym(self):
        for src, want in SYM_CASES:
            self.check(src, want)

    def test_math(self):
        for body, ok in MATH_CASES:
            self.check(f'(sym raw "{body}")', f'(sym raw "{body}")' if ok else None)
        self.assertEqual(O.parse_math("-" + balanced(32)).nodes, 64)
        self.assertEqual(O.parse_math("sigmoid(" * 15 + "x" + ")" * 15).depth, 16)

    def test_undecidable_is_not_a_rejection(self):
        ok = 681            # 6 + 6 * 681 = 4092 frames <= RUST_FRAME_LIMIT
        self.assertEqual(O.parse_math("(" * ok + "c0" + ")" * ok).frames, 6 + 6 * ok)
        self.check('(sym affine "' + "(" * ok + "c0" + ")" * ok + '")', '(sym affine "' + "(" * ok + "c0" + ")" * ok + '")')
        for body in ("(" * (ok + 1) + "c0" + ")" * (ok + 1), "-" * 5000 + "c0", "c0" + "+c0" * 5000):
            with self.assertRaises(O.Undecidable):
                O.canonical('(sym affine "' + body + '")')
            with self.assertRaises(O.Undecidable):
                O.is_canonical('(sym affine "' + body + '")')
            self.assertFalse(issubclass(O.Undecidable, O.CanonicalError))

    def test_lone_surrogate_and_types(self):
        with self.assertRaises(O.CanonicalError):
            O.canonical("(pipe X (ridge k=\ud800))")
        with self.assertRaises(TypeError):
            O.canonical(b"(pipe X (ridge))")

    def test_is_canonical_and_sha(self):
        self.assertTrue(O.is_canonical("(pipe X (ridge))"))
        self.assertFalse(O.is_canonical("(pipe X  (ridge))"))
        self.assertFalse(O.is_canonical("(pipe X)"))
        self.assertEqual(O.program_sha256("(pipe X (ridge))"), hashlib.sha256(b"(pipe X (ridge))").hexdigest())


# ---------------------------------------------------------------------------------------------
# Width/lane ports, caps, static semantics
# ---------------------------------------------------------------------------------------------
class WidthPortTests(unittest.TestCase):
    def test_proposal_main_rs_vectors(self):
        self.assertFalse(O.admitted_width("(pipe X (piecewise_linear bins=32) (ridge))", 28))
        self.assertFalse(O.admitted_width("(pipe X (cheb append=true degree=8) (piecewise_linear bins=8) (ridge))", 16))
        self.assertFalse(O.admitted_width("(pipe X (poly_interact append=true) (ridge))", 64))
        self.assertFalse(O.admitted_width("(pipe X (oblique_hinge append=true n=256 seed=1) (ridge))", 300))
        self.assertFalse(O.standard_lane_model_safe("(pipe X (rank) (ridge))", 512, 10_000))
        self.assertFalse(O.standard_lane_model_safe("(pipe X (histgbm2 bags=10 leaves=31 rounds=400))", 32, 10_000))
        self.assertFalse(O.standard_lane_model_safe("(pipe X (gpr m=1025 steps=100))", 32, 10_000))
        self.assertTrue(O.standard_lane_model_safe("(pipe X (histgbm depth=4 rounds=120))", 32, 10_000))

    def test_append_and_target_pass_through(self):
        self.assertFalse(O.admitted_width("(pipe X (target kind=log1p) (ridge))", 600))   # target is width pass-through
        self.assertTrue(O.admitted_width("(pipe X (target kind=log1p) (ridge))", 512))
        self.assertTrue(O.admitted_width("(pipe X (ridge))", 1025))                        # the head is never counted
        self.assertTrue(O.admitted_width("(pipe X (standardize append=true) (ridge))", 256))
        self.assertFalse(O.admitted_width("(pipe X (standardize append=true) (ridge))", 257))
        self.assertFalse(O.admitted_width("(pipe X (standardize append=1) (ridge))", 257))
        self.assertTrue(O.admitted_width("(pipe X (standardize append=1.0) (ridge))", 257))
        self.assertFalse(O.admitted_width("(pipe X (pls_project k=0) (ridge))", 8))
        self.assertTrue(O.admitted_width("(pipe X (piecewise_linear bins=8.9) (ridge))", 64))
        self.assertFalse(O.admitted_width("(pipe X (piecewise_linear bins=9) (ridge))", 64))
        self.assertTrue(O.admitted_width("(pipe X (piecewise_linear bins=-3) (ridge))", 64))
        self.assertTrue(O.admitted_width("(pipe X (fft_mag k=100) (ridge))", 8))
        self.assertTrue(O.admitted_width("(pipe X (pair_prod topk=24 append=true) (ridge))", 236))   # 236 + 276 = 512
        self.assertFalse(O.admitted_width("(pipe X (pair_prod topk=24 append=true) (ridge))", 237))
        self.assertTrue(O.admitted_width('(sym affine "c0")', 4000))
        self.assertTrue(O.admitted_width("garbage", 4000))

    def test_lane_rules(self):
        self.assertFalse(O.standard_lane_model_safe("(pipe X (krr m=600))", 8, 6000))
        self.assertTrue(O.standard_lane_model_safe("(pipe X (krr m=500))", 8, 6000))
        self.assertFalse(O.standard_lane_model_safe("(pipe X (krr m=500))", 8, 16000))
        self.assertFalse(O.standard_lane_model_safe("(pipe X (histgbm depth=5))", 8, 6000))
        self.assertFalse(O.standard_lane_model_safe("(pipe X (histgbm rounds=161))", 8, 6000))
        self.assertTrue(O.standard_lane_model_safe("(pipe X (histgbm depth=5))", 8, 5000))
        self.assertFalse(O.standard_lane_model_safe("(pipe X (extra_trees trees=129))", 8, 6000))
        self.assertTrue(O.standard_lane_model_safe("(pipe X (blom) (ridge))", 300, 10_000))     # 24,000,000 bytes
        self.assertFalse(O.standard_lane_model_safe("(pipe X (blom) (ridge))", 300, 10_486))

    def test_width_preflight(self):
        self.assertEqual(O.width_preflight('(sym affine "c0")', 8, 100), "n/a")
        self.assertEqual(O.width_preflight("(portfolio (member (pipe X (ridge))) (blend))", 8, 100), "n/a")
        self.assertEqual(O.width_preflight("(pipe X (rank) (ridge))", 512, 10_000), "refused:lane")
        self.assertEqual(O.width_preflight("(pipe X (piecewise_linear bins=32) (ridge))", 28, 1000), "refused:width")
        self.assertEqual(O.width_preflight("(pipe X (ridge))", 8, 100), "admitted")
        with self.assertRaises(ValueError):
            O.width_preflight("(pipe X (ridge))", -1, 100)


class CapsSemanticsTests(unittest.TestCase):
    def test_form_caps(self):
        self.assertEqual(O.form_caps("(pipe X" + " (standardize)" * 9 + " (ridge))"), [])
        caps = O.form_caps("(pipe X" + " (standardize)" * 10 + " (ridge))")
        self.assertIn("evaluator:pipe_stages", caps)
        self.assertIn("island:lparen", caps)
        self.assertEqual(O.form_caps("(pipe X" + " (winsor_clip q=0.1)" * 4 + " (ridge))"), ["island:winsor_clip"])
        self.assertEqual(O.form_caps("(portfolio (member (pipe X (ridge))) (blend folds=5 guard=clip kind=nnls repeats=1))"),
                         ["evaluator:portfolio_members"])
        big = "(portfolio" + "".join(" (member (pipe X" + " (standardize)" * k + " (ridge)))" for k in (8, 7, 7)) + " (blend))"
        self.assertIn("evaluator:portfolio_stages", O.form_caps(big))
        self.assertEqual(O.form_caps('(sym affine "' + "(" * 17 + "c0" + ")" * 17 + '")'), [])
        self.assertEqual(O.form_caps('(sym affine "' + "(" * 18 + "c0" + ")" * 18 + '")'), ["island:lparen"])
        caps = O.form_caps("(pipe X (ridge k=" + "a" * 33000 + "))")
        self.assertIn("evaluator:bytes", caps)
        self.assertIn("island:bytes", caps)
        self.assertEqual(O.form_caps("garbage"), ["evaluator:parse", "island:form"])

    def test_static_semantics(self):
        S = O.static_semantics
        self.assertEqual(S("(pipe X (sigmoid) (target kind=log1p) (ridge))"), ["target_input_not_leading"])
        self.assertEqual(S("(pipe X (target kind=log1p) (target kind=rank_gauss) (ridge))"), ["target_input_not_leading"])
        self.assertEqual(S("(pipe X (input) (target kind=log1p) (ridge))"), ["target_input_not_leading"])
        self.assertEqual(S("(pipe X (target kind=log1p) (input) (standardize) (ridge))"), [])
        self.assertEqual(S("(pipe X (ridge) (standardize))"), ["head_not_last:ridge"])
        self.assertEqual(S("(pipe X (histgbm) (ridge))"), ["head_not_last:histgbm"])
        self.assertEqual(S("(pipe X (ridge lambda_frac=nan))"), ["nonfinite_param:ridge.lambda_frac"])
        self.assertEqual(S("(pipe X (ridge lambda_frac=-inf))"), ["nonfinite_param:ridge.lambda_frac"])
        self.assertEqual(S('(sym affine "c0")'), [])
        self.assertEqual(S("(portfolio (member (pipe X (ridge))) (member (pipe X (krr) (ridge))) (blend))"),
                         ["member1:head_not_last:krr"])


# ---------------------------------------------------------------------------------------------
# Tier B protocol against an in-process stub (no process is spawned)
# ---------------------------------------------------------------------------------------------
class StubEngine(O.PinnedEngine):
    """Follows the engine's stdout and exit contract in-process; Tier A plays the canonicalizer."""

    def __init__(self, path, sha, *, crash_on=(), mutate=None, corrupt=None, force=None):
        super().__init__(path, sha)
        self.crash_on, self.mutate, self.corrupt, self.force = tuple(crash_on), mutate, corrupt, dict(force or {})
        self.requests = []

    def _invoke(self, payload):
        req = json.loads(payload)
        self.requests.append(req)
        if any(c in s for s in req["prior_seeds"] for c in self.crash_on):
            return 1, b"", b"Error: simulated failure\n"
        out, seen, draw = [], set(), req.get("offset", 0)
        for s in req["prior_seeds"]:
            if len(out) >= req["count"]:
                break
            if s in self.force:
                d = self.force[s]
            else:
                try:
                    d = O.canonical(s)
                except O.CanonicalError:
                    continue
                d = self.mutate(d) if self.mutate else d
            h = O.program_sha256(d)
            if h in seen:
                continue
            seen.add(h)
            out.append({"draw": draw, "dsl": d, "grammar_family": "pipeline", "operator": "historical_seed",
                        "parents": [], "program_sha256": h})
            draw += 1
        while len(out) < req["count"]:
            d = f"(pipe X (fill_in n={draw}))"
            if O.program_sha256(d) not in seen:
                seen.add(O.program_sha256(d))
                out.append({"draw": draw, "dsl": d, "grammar_family": "pipeline", "operator": "structured_restart",
                            "parents": [], "program_sha256": O.program_sha256(d)})
            draw += 1
        random.Random(len(out)).shuffle(out)                 # outputs are never positional
        if self.corrupt == "sha":
            out[0]["program_sha256"] = "0" * 64
        if self.corrupt == "count":
            out = out[:-1]
        text = json.dumps({"proposals": out, "schema": O.PROPOSALS_SCHEMA}, sort_keys=True, ensure_ascii=False) + "\n"
        if self.corrupt == "lines":
            text += text
        return 0, text.encode("utf-8"), b""


class TierBProtocolStubTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.bin = Path(self.tmp.name) / "fake-engine"
        self.bin.write_bytes(b"not an executable; hashed only")
        self.sha = hashlib.sha256(self.bin.read_bytes()).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def stub(self, **kw):
        return StubEngine(self.bin, self.sha, **kw)

    def test_match_by_dsl_fill_ins_and_missing(self):
        e = self.stub()
        canon = ["(pipe X (ridge))", "(pipe X (krr m=50))", '(sym affine "c0")']
        sb = e.seed_batch(canon + ["(pipe X  (ridge_gcv))", "(pipe X)"])
        self.assertEqual(sb.accepted, {c: c for c in canon})
        self.assertEqual(sb.changed, {})                     # two unmatched inputs, one unmatched output
        self.assertEqual(sb.missing, ["(pipe X  (ridge_gcv))", "(pipe X)"])
        self.assertEqual(sb.unattributed, ["(pipe X (ridge_gcv))"])
        self.assertEqual(sb.fill_ins, 1)
        self.assertEqual((sb.exit_code, sb.failed, sb.batches), (0, [], 1))
        one = e.seed_batch(["(pipe X  (ridge_gcv))"])
        self.assertEqual(one.changed, {"(pipe X  (ridge_gcv))": "(pipe X (ridge_gcv))"})
        self.assertEqual((one.fill_ins, one.missing), (0, []))
        rej = e.seed_batch(["(pipe X)"])
        self.assertEqual((rej.accepted, rej.changed, rej.missing, rej.fill_ins), ({}, {}, ["(pipe X)"], 1))
        self.assertEqual(O.tier_b_verdict(e, "(pipe X)"), ("reject", None))
        self.assertEqual(O.tier_b_verdict(e, "(pipe X (ridge))"), ("accept", "(pipe X (ridge))"))
        # a duplicate canonical form is dropped by the engine: the non-canonical input goes missing
        dup = e.seed_batch(["(pipe X  (ridge))", "(pipe X (ridge))"])
        self.assertEqual((dup.accepted, dup.missing, dup.fill_ins), ({"(pipe X (ridge))": "(pipe X (ridge))"},
                                                                     ["(pipe X  (ridge))"], 1))
        self.assertEqual(e.requests[0]["count"], 5)
        self.assertEqual(e.requests[0]["feature_count"], 8)
        self.assertEqual(sorted(e.requests[0]), ["count", "feature_count", "prior_seeds", "seed", "train_rows"])

    def test_bisection_after_nonzero_exit(self):
        e = self.stub(crash_on=("CRASH",))
        inputs = [f"(pipe X (ridge k={i}))" for i in range(7)] + ["(pipe X (CRASH))"]
        sb = e.seed_batch(inputs)
        self.assertEqual(sb.exit_code, 1)
        self.assertTrue(sb.stderr_head.startswith("Error:"))
        self.assertEqual(sb.failed, ["(pipe X (CRASH))"])
        self.assertEqual(sorted(sb.accepted), sorted(inputs[:7]))
        self.assertEqual(sb.missing, [])                     # a failed batch is never read as rejections
        self.assertEqual(sb.batches, 7)                      # 8 -> 4+4 -> 2+2 -> 1+1
        self.assertEqual(e.invocations, 7)
        self.assertEqual(e.verifications, 7)
        self.assertEqual(O.tier_b_verdict(e, "(pipe X (CRASH))")[0], "engine_failure")

    def test_sha256_checked_before_every_batch(self):
        e = self.stub()
        for i in range(3):
            e.seed_batch([f"(pipe X (ridge k={i}))"])
        self.assertEqual((e.invocations, e.verifications), (3, 3))
        self.bin.write_bytes(b"tampered")
        with self.assertRaises(O.EngineIntegrityError):
            e.seed_batch(["(pipe X (ridge))"])
        self.assertEqual(e.invocations, 3)                   # nothing ran after the tamper

    def test_request_bounds(self):
        e = self.stub()
        bad = [([], {}), ([f"(pipe X (r k={i}))" for i in range(4097)], {}), (["a", "a"], {}),
               (["(pipe X (r))"], {"feature_count": 0}), (["(pipe X (r))"], {"feature_count": 2049}),
               (["(pipe X (r))"], {"feature_count": True}), (["(pipe X (r))"], {"train_rows": -1}),
               (["(pipe X (r))"], {"seed": -1}), (["(pipe X (r k=\ud800))"], {})]
        for inputs, kw in bad:
            with self.assertRaises(ValueError):
                e.seed_batch(inputs, **kw)
        self.assertEqual(e.invocations, 0)
        sb = e.seed_batch([f"(pipe X (r k={i}))" for i in range(4096)], feature_count=2048, train_rows=10**6)
        self.assertEqual(len(sb.accepted), 4096)
        self.assertEqual(e.requests[-1]["count"], 4096)

    def test_protocol_errors_stop(self):
        for mode in ("sha", "count", "lines"):
            with self.assertRaises(O.EngineProtocolError, msg=mode):
                self.stub(corrupt=mode).seed_batch(["(pipe X (ridge))", "(pipe X (krr))"])

    def test_corpus_generated_and_differential_drivers(self):
        e = self.stub()
        progs = [(O.program_sha256(d), d) for d in ("(pipe X (ridge))", "(pipe X (krr))", '(sym affine "c0")')]
        doc = O.corpus_check(e, progs, batch=2)
        self.assertTrue(doc["pass"], doc)
        self.assertEqual((doc["counts"]["tier_b_accepted_unchanged"], doc["counts"]["tier_b_batches"]), (3, 2))
        doc = O.corpus_check(e, progs + [(O.program_sha256("(pipe X  (gpr))"), "(pipe X  (gpr))"), ("0" * 64, "(pipe X (pls))")])
        self.assertFalse(doc["pass"])
        self.assertEqual((doc["counts"]["tier_a_changed"], doc["counts"]["sha_mismatch"], doc["counts"]["tier_b_changed"]),
                         (1, 1, 1))
        g = O.generated_check(e, "(pipe X  (target kind=log1p) (ridge))", feature_count=8, train_rows=100)
        self.assertEqual((g["status"], g["noncanonical_repaired"], g["canonical"]),
                         ("valid", True, "(pipe X (target kind=log1p) (ridge))"))
        g = O.generated_check(e, "(pipe X  (ridge))", feature_count=8, train_rows=100, noncanonical="reject")
        self.assertEqual((g["status"], g["reason"]), ("invalid_static", "noncanonical"))
        self.assertEqual(O.generated_check(e, "(pipe X)", feature_count=8, train_rows=100)["status"], "invalid_parse")
        self.assertEqual(O.generated_check(e, "(pipe X (sigmoid) (target kind=log1p) (ridge))", feature_count=8,
                                           train_rows=100)["status"], "invalid_semantics")
        self.assertEqual(O.generated_check(e, "(pipe X (rank) (ridge))", feature_count=512,
                                           train_rows=10_000)["status"], "refused:lane")
        self.assertEqual(O.generated_check(e, "(pipe X" + " (blom)" * 10 + " (ridge))", feature_count=8,
                                           train_rows=100)["status"], "refused:caps")
        g = O.generated_check(e, "(pipe X)", feature_count=8, train_rows=100, confirm_rejection=True)
        self.assertEqual((g["status"], g["tier_b"]), ("invalid_parse", "reject"))
        lenient = self.stub(force={"(pipe X)": "(pipe X (ridge))"})
        g = O.generated_check(lenient, "(pipe X)", feature_count=8, train_rows=100, confirm_rejection=True)
        self.assertEqual((g["status"], g["tier_b"]), ("drift", "accept"))
        drift = self.stub(mutate=lambda d: d.replace("ridge", "ridge_gcv"))
        self.assertEqual(O.generated_check(drift, "(pipe X (ridge))", feature_count=8, train_rows=100)["status"], "drift")
        diff = O.differential(e, [("a", "(pipe X (ridge))"), ("b", "(pipe X  (ridge_gcv))"), ("b", "(pipe X)")])
        self.assertTrue(diff["pass"], diff)
        self.assertEqual(diff["counts"]["agree"], 3)
        self.assertEqual(diff["counts"]["noncanonical"], 2)
        bad = O.differential(drift, ["(pipe X (ridge))"])
        self.assertFalse(bad["pass"])
        self.assertEqual(bad["counts"]["disagree"], 1)


# ---------------------------------------------------------------------------------------------
# The pinned engine (Tier B): vectors, protocol, corpus sample, differential fuzz
# ---------------------------------------------------------------------------------------------
class PinnedEngineTests(unittest.TestCase):
    def test_rust_vectors_agree_with_tier_b(self):
        e = engine()
        doc = O.replica_check(e)
        report("replica_check_tier_b", {"vectors": doc["vectors"], "passed": doc["passed"],
                                        "engine_invocations": e.invocations, "engine_verifications": e.verifications})
        self.assertTrue(doc["pass"], doc["failures"])
        self.assertEqual(e.verifications, e.invocations + 1)   # + the upfront check

    def test_protocol_on_the_pinned_engine(self):
        e = engine()
        sb = e.seed_batch(["(pipe X (ridge))", '(sym affine "c0")', "(pipe X  (ridge_gcv))", "(pipe X)"])
        self.assertEqual(sb.accepted, {"(pipe X (ridge))": "(pipe X (ridge))", '(sym affine "c0")': '(sym affine "c0")'})
        self.assertEqual(sb.missing, ["(pipe X  (ridge_gcv))", "(pipe X)"])
        self.assertEqual(sb.unattributed, ["(pipe X (ridge_gcv))"])
        self.assertEqual((sb.fill_ins, sb.exit_code, sb.failed), (1, 0, []))
        self.assertEqual(O.tier_b_verdict(e, "(pipe X  (ridge_gcv))"), ("accept", "(pipe X (ridge_gcv))"))
        self.assertEqual(O.tier_b_verdict(e, "(pipe X)"), ("reject", None))
        # Tier A's decidability limits hold on the engine: no stack overflow at either boundary.
        at_frame_limit = '(sym affine "' + "(" * 681 + "c0" + ")" * 681 + '")'        # 4,092 parser frames
        self.assertEqual(O.parse_math(sym_body(at_frame_limit)).frames, 4092)
        self.assertEqual(O.tier_a_verdict(at_frame_limit), ("accept", at_frame_limit))
        self.assertEqual(O.tier_b_verdict(e, at_frame_limit), ("accept", at_frame_limit))
        deep_ast = '(sym affine "c0' + "+c0" * 4094 + '")'                             # AST depth 4,095
        self.assertEqual(O.math_shape(sym_body(deep_ast)), (8189, 4095))
        self.assertEqual(O.tier_a_verdict(deep_ast), ("reject", "math"))
        self.assertEqual(O.tier_b_verdict(e, deep_ast), ("reject", None))
        self.assertEqual(e.verifications, e.invocations + 1)   # + the upfront check

    def test_corpus_sample(self):
        sample = corpus_sample()
        ctxs = sample["contexts"]
        c = Counter()
        distinct = {}
        preflight = Counter()
        for ctx in ctxs:
            self.assertTrue(ctx["native_ok"], ctx["context_sha256"])
            self.assertTrue(ctx["prepared_ok"], ctx["context_sha256"])
            c["contexts"] += 1
            c["cohort:" + ctx["cohort"]] += 1
            for sha, dsl, status, top_sha in ctx["rows"]:
                c["rows"] += 1
                c["sha_ok"] += O.program_sha256(dsl) == sha
                if top_sha is not None:
                    c["abandoned_top_sha_rows"] += 1
                    c["abandoned_top_sha_ok"] += top_sha == sha
                distinct[sha] = dsl
                preflight[O.width_preflight(dsl, ctx["encoded_features"], ctx["train_rows"])] += 1
        forms = Counter()
        t0 = time.perf_counter()
        for sha, dsl in distinct.items():
            c["tier_a_canonical"] += O.canonical(dsl) == dsl
        tier_a_s = time.perf_counter() - t0
        for dsl in distinct.values():
            forms[O.genome_form(O.parse_genome(dsl))] += 1
        e = engine()
        doc = O.corpus_check(e, sorted(distinct.items()))
        counts = dict(c, distinct_programs=len(distinct), forms=dict(forms), width_preflight=dict(preflight),
                      tier_a_seconds=round(tier_a_s, 3), tier_a_programs_per_s=round(len(distinct) / tier_a_s),
                      tier_b=doc["counts"], tier_b_pass=doc["pass"], strata=len({(x["cohort"], x["fold"]) for x in ctxs}))
        report("corpus_sample", counts)
        self.assertLessEqual(len(ctxs), 40)
        self.assertEqual(counts["strata"], 10)
        self.assertEqual(c["sha_ok"], c["rows"])
        self.assertEqual(c["abandoned_top_sha_ok"], c["abandoned_top_sha_rows"])
        self.assertEqual(c["tier_a_canonical"], len(distinct))
        self.assertTrue(doc["pass"], doc)
        self.assertEqual(doc["counts"]["tier_b_accepted_unchanged"], len(distinct))
        self.assertEqual(e.verifications, e.invocations + 1)   # + the upfront check

    def test_differential_fuzz(self):
        sample = corpus_sample()
        distinct = sorted({dsl for ctx in sample["contexts"] for _, dsl, _, _ in ctx["rows"]})
        rng = random.Random(FUZZ_SEED)
        pipes = [d for d in distinct if d.startswith("(pipe ")]
        syms = [d for d in distinct if d.startswith("(sym ")]
        bases = list(O.FUZZ_BASES) + rng.sample(pipes, min(3000, len(pipes))) + rng.sample(syms, min(1000, len(syms)))
        inputs = O.fuzz_inputs(bases, n=FUZZ_N, seed=FUZZ_SEED)
        texts = [t for _, t in inputs]
        features = {
            "U+001C": sum("\x1c" in t for t in texts), "U+00A0": sum("\xa0" in t for t in texts),
            "other_unicode_ws": sum(any(ch in O.RUST_WHITESPACE[6:] and ch != "\xa0" for ch in t) for t in texts),
            "non_ascii": sum(not t.isascii() for t in texts), "quoted_escape": sum('\\' in t for t in texts),
            "i64_bounds": sum(("922337203685477580" in t) for t in texts),
            "exponent_value": sum(("e-" in t or "e+" in t or "E" in t) for t in texts),
            "nan_inf": sum(any(w in t.lower() for w in ("=nan", "=inf", "=-inf", "=+inf", "infinity")) for t in texts),
            "blend": sum("(blend" in t for t in texts), "sym": sum("(sym" in t.lower() for t in texts),
        }
        shapes = Counter()
        for t in texts:
            body = sym_body(t) if "(sym" in t else None
            if body is None:
                continue
            try:
                nodes, depth = O.math_shape(body)
            except (O.CanonicalError, O.Undecidable):
                continue
            for name, hit in (("nodes_64", nodes == 64), ("nodes_65", nodes == 65), ("depth_16", depth == 16),
                              ("depth_17", depth == 17), ("nodes_64_depth_ok", nodes == 64 and depth <= 16),
                              ("nodes_65_depth_ok", nodes == 65 and depth <= 16)):
                shapes[name] += hit
        features.update(shapes)
        ties = [tie_values(t) for t in texts]
        features["shortest_tie"] = sum(bool(x) for x in ties)
        features["shortest_tie_repr_differs"] = sum(
            any(O.format_f64(v) != format(Decimal(repr(v)), "f") for v in x) for x in ties)
        e = engine()
        doc = O.differential(e, inputs, workers=3)
        counts = doc["counts"]
        report("differential_fuzz", {"inputs": len(inputs), "bases": len(bases), "counts": counts,
                                     "per_category": doc["per_category"], "features": features,
                                     "engine_invocations": doc["engine_invocations"],
                                     "engine_verifications": doc["engine_verifications"],
                                     "disagreements": doc["disagreements"][:10]})
        self.assertEqual(counts.get("disagree", 0), 0, doc["disagreements"][:10])
        self.assertTrue(doc["pass"])
        self.assertGreaterEqual(counts["noncanonical"], 20000)
        self.assertEqual(counts["inputs"], len(inputs))
        self.assertEqual(doc["engine_invocations"], doc["engine_verifications"])
        self.assertEqual(doc["engine_invocations"], len(inputs) + counts["resubmitted"])
        self.assertGreater(counts["resubmitted"], 100)
        self.assertEqual(counts["resubmit_unchanged"], counts["resubmitted"])
        self.assertEqual(counts.get("b_engine_failure", 0), 0)
        for name in ("U+001C", "U+00A0", "non_ascii", "quoted_escape", "i64_bounds", "nan_inf", "blend", "sym",
                     "shortest_tie_repr_differs"):
            self.assertGreater(features[name], 50, name)
        self.assertGreater(features["shortest_tie"], 100)
        for name in ("nodes_64_depth_ok", "nodes_65_depth_ok", "depth_16", "depth_17"):
            self.assertGreater(features[name], 10, name)
        for name, _meth, _share in O.FUZZ_CATEGORIES:
            self.assertGreater(doc["per_category"][name]["inputs"], 0, name)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = O.main(argv)
        return code, buf.getvalue()

    def test_replica_check(self):
        code, out = self.run_main(["replica-check"])
        doc = json.loads(out)
        self.assertEqual((code, doc["pass"], doc["vectors"], doc["tier_b"]), (0, True, 37, False))
        code, out = self.run_main(["replica-check", "--tier-b", "--out", str(self.d / "replica.json")])
        self.assertEqual(code, 0)
        receipt = json.loads((self.d / "replica.json").read_text())
        self.assertEqual((receipt["pass"], receipt["passed"], receipt["tier_b"]), (True, 37, True))
        self.assertEqual(receipt["feat_math_pins"], O.FEAT_MATH_PINS)

    def test_tier_a_corpus_file(self):
        good = self.d / "good.jsonl"
        good.write_text("".join(json.dumps({"program_sha256": O.program_sha256(x), "dsl": x}) + "\n"
                                for x in ("(pipe X (ridge))", '(sym affine "c0")')))
        code, out = self.run_main(["tier-a", "--corpus", str(good)])
        doc = json.loads(out)
        self.assertEqual((code, doc["pass"], doc["counts"]["tier_b_accepted_unchanged"]), (0, True, 2))
        bad = self.d / "bad.jsonl"
        bad.write_text(json.dumps({"program_sha256": O.program_sha256("(pipe X  (ridge))"), "dsl": "(pipe X  (ridge))"}) + "\n")
        code, out = self.run_main(["tier-a", "--corpus", str(bad)])
        self.assertEqual((code, json.loads(out)["pass"]), (1, False))

    def test_tier_a_corpus_program_dictionary(self):
        """The PLAN's P4 command `oracle tier-a --corpus <program-dict>` over all four partitions (the default):
        every partition, heldout included, is read through the projection role 'roundtrip' (journaled), and the
        receipt is written. Synthetic dictionary, one Tier B batch (sha256-verified)."""
        from tests_p4.test_automaton import make_dict
        parts = {"train_fit": ["(pipe X (ridge_gcv))", '(sym affine "c0")'], "dev": ["(pipe X (ridge))"],
                 "diag": ["(pipe X (standardize) (ridge))"], "heldout": ["(pipe X (norm_row) (ridge_gcv))"]}
        root = make_dict(self.d / "dict", parts)
        receipt = self.d / "tier-a.json"
        code, out = self.run_main(["tier-a", "--corpus", str(root), "--out", str(receipt)])
        self.assertEqual(code, 0, out)
        doc = json.loads(receipt.read_text())
        self.assertEqual((doc["pass"], doc["partitions"]), (True, ["train_fit", "dev", "diag", "heldout"]))
        self.assertEqual(doc["counts"]["tier_b_accepted_unchanged"], 5)
        journal = [json.loads(x) for x in (root / ".access-journal.jsonl").read_bytes().splitlines()]
        self.assertEqual({(e["partition"], e["role"], tuple(e["columns"])) for e in journal},
                         {(p, "roundtrip", ("program_sha256", "dsl")) for p in parts})
        rows = list(O._iter_context_rows(root, "heldout"))          # the width preflight's projection
        self.assertEqual([d for _c, d in rows], parts["heldout"])
        self.assertTrue(all(len(c) == 64 for c, _d in rows))

    def test_tier_b_generated(self):
        code, out = self.run_main(["tier-b", "--feature-count", "8", "--train-rows", "100", "--dsl", "(pipe X  (ridge))"])
        doc = json.loads(out)
        self.assertEqual((code, doc["status"], doc["canonical"], doc["noncanonical_repaired"]),
                         (0, "valid", "(pipe X (ridge))", True))
        code, out = self.run_main(["tier-b", "--feature-count", "8", "--train-rows", "100", "--noncanonical", "reject",
                                   "--dsl", "(pipe X  (ridge))"])
        self.assertEqual(json.loads(out)["status"], "invalid_static")

    def test_differential_small(self):
        code, out = self.run_main(["differential", "--n", "300", "--seed", "1", "--workers", "2"])
        doc = json.loads(out)
        self.assertEqual((code, doc["pass"], doc["counts"].get("disagree", 0)), (0, True, 0))
        self.assertGreaterEqual(doc["counts"]["inputs"], 300)


if __name__ == "__main__":
    unittest.main()
