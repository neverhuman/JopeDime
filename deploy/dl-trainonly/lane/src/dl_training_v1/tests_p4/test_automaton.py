"""PLAN §5 P4 grammar automaton tests (dltrain.automaton, design "automaton" section).

- numeric DFA (design list: accepted and refused pipe values, per-(op,key) Int/Float tables, no trailing-zero dead
  end at the last fraction position, <= 17 significant digits);
- sym pushdown automaton (design list, arity, unbalanced parens, open frames, the 65th node, AST depth 16) and a
  differential against Tier A (oracle.parse_math) on random bodies;
- caps (11th stage, 4th winsor_clip, 12th '(' in a pipe, 19th '(' in a sym, 1 or 4 members, 25 member stages,
  32,769 bytes, the max_seq boundary), uniqueness masks, the generation-only rules;
- consistency: allowed() equals a brute-force scan of every token on reached states, mask_key determines the mask,
  compile_masks agrees with allowed(); exhaustive enumeration for small budgets (no dead end, every sequence parses);
  frontier (Bellman) consistency by brute force over every token: each completion vector is realized by a token and
  no state with a fitting completion has an empty mask (this catches a poisoned completion memo, which a
  mask-vs-brute-force comparison cannot);
- the canonical-fidelity review's reproductions: the quoted-STR memo in both evaluation orders (CF-1), POLICY on
  its own train_fit when the token-minimal completion breaks the '(' cap (CF-2, pipe and portfolio), keys with an
  odd number of '"' (CF-4, documented refusal), keys after non-ASCII keys and search-limit reasons (CF-5);
- SYNTAX superset of Tier A-canonical generated programs; artifacts (determinism, refusals, tokenizer integration,
  gate, poison test, CLI); signature cross-check against the sha-pinned feat-math op tables;
- a <= 40-context corpus sample: POLICY accepts 100% of the train half its tables were built from, SYNTAX 100% of
  every tokenization within max_seq, held-out POLICY acceptance reported, compile_masks, 2,000 masked walks per
  mode (EOS within max_seq, Tier A parses, unique tokenization), and Tier B on a walk sample.

Corpus records are projected at parse time to program.{dsl, program_sha256}; nothing else is kept, read or printed.
Temporary files live under TMPDIR and are removed by each test. Results print as `P4-AUTOMATON {...}` lines.
"""
import contextlib
import hashlib
import io
import json
import random
import re
import resource
import shutil
import tempfile
import time
import unittest
from collections import Counter
from unittest import mock
from pathlib import Path

import numpy as np

from dltrain import automaton as A
from dltrain import dsl_tokenizer as T
from dltrain import oracle as O

EPOCH = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/epochs/epoch-000.json")
SAMPLE_PER_STRATUM = 4            # 5 folds x 2 cohorts x 4 = 40 contexts (the hard cap)
SAMPLE_SEED = 20261005            # the oracle and tokenizer builders' sample
SPLIT_SEED = 7
DICT_SCHEMA = "jope.dl-program-dict/v1"
SYN, POL = A.Mode.SYNTAX, A.Mode.POLICY
FEAT_MATH = Path(O.FEAT_MATH_SRC)
# sha256 of the vendored op tables used by the cross-check (registry.rs and spec.rs are pinned here; the
# others are oracle.FEAT_MATH_PINS)
CROSSCHECK_PINS = {
    "registry.rs": "337700af92862c8ef79f81ed04283870d3953333577896c37ac1fa883ffaf38c",
    "spec.rs": "33390040a4bcc22a095d59dca99fbe8891b429dfdddfe578ada6798099602b18",
    "heads.rs": O.FEAT_MATH_PINS["heads.rs"],
    "advanced_v3/pipeline.rs": O.FEAT_MATH_PINS["advanced_v3/pipeline.rs"],
    "target_transform.rs": O.FEAT_MATH_PINS["target_transform.rs"],
}


def report(name: str, doc: dict) -> None:
    print("P4-AUTOMATON " + json.dumps({"test": name, **doc}, sort_keys=True, ensure_ascii=True), flush=True)


BLEND1 = " (blend folds=5 guard=clip kind=nnls repeats=1))"
TRAIN = (
    "(pipe X (extra_trees depth=9 max_features=1 min_leaf=3 seed=999761 trees=128))",
    "(pipe X (extra_trees depth=12 max_features=1.0 min_leaf=3 seed=90 trees=24))",
    "(pipe X (impute add_indicator=true strategy=median value=-1.0) (step t=-0.0) (histgbm2 delta=0.000001 "
    "loss=huber lr=0.09988591634790045 rounds=300 val_shuffle=true))",
    "(pipe X (winsor_clip q=0.5408000000000001) (ridge lambda_frac=0.000020741850001116463))",
    "(pipe X (standardize) (winsor_clip q=0.05) (ridge_gcv))",
    "(pipe X (target kind=log1p) (robust_scale) (histgbm depth=4 lr=0.1 min_leaf=8 rounds=160))",
    "(pipe X (gpr kernel=rbf+linear m=50 steps=40))",
    "(pipe X (map expr=\"sigmoid(col(0))\") (ridge lambda_frac=0.5))",
    "(sym affine \"(c1*-0.43)\")",
    "(sym affine \"((-0.87)--1.44)\")",
    "(sym affine \"sigmoid(((c5^1.2705454457961025)/-1.1374340345718594))\")",
    "(sym affine \"clip(max(c3,row_mean),-0.5,min(c0,c39))\")",
    "(sym affine \"c0\")",
    "(portfolio (member (pipe X (norm_row) (krr))) (member (pipe X (winsor_clip q=0.01804076952840708) (norm_row) "
    "(ridge_gcv)))" + BLEND1,
)


def make_grammar(train, max_seq=None, caps=None):
    train = sorted(set(train))
    vocab, _ = T.build_vocab(train)
    if max_seq is None:
        max_seq = T.ceil32(max(len(T.tokenize(d, vocab)) for d in train))
    sig = A.build_signatures(train)
    gdoc = A.build_grammar(sig, vocab, max_seq, caps)
    return vocab, A.Grammar(gdoc, sig, vocab)


def ids_of(vocab, text):
    return T.tokenize(text, vocab)


def path_state(g, mode, ids, rules=A.Rules()):
    s = g.start(mode, rules)
    for t in ids:
        s = g.advance(s, t)
    return s


def brute_allowed(g, s):
    """Every token t whose successor state is valid and keeps a completion within every budget."""
    out = np.zeros(g.V, dtype=bool)
    for t in range(g.V):
        try:
            s2 = g.advance(s, t)
        except A.GrammarError:
            continue
        if g.completion_fits(s2):
            out[t] = True
    return out


def _proj(g, s, v):
    """A cost vector as the mode's frontier holds it (SYNTAX projects out lparen, winsor and stages)."""
    return v if g._ctxs[s.ctx][0] else (v[0], v[1], 0, 0, 0, v[5])


def bellman_violations(g, s):
    """Frontier consistency at s, checked by brute force over EVERY token (so a poisoned memo cannot hide it):
    each vector v of C(s) must be realized by some token t, i.e. delta(t) + v' <= v for a v' in C(advance(s, t));
    and a state with a completion that fits its budgets must allow at least one token. Returns the problems."""
    bad = []
    if s.form == A.F_DONE:
        return bad
    c = g.completion_frontier(s)
    succ = []
    for t in range(g.V):
        try:
            s2 = g.advance(s, t)
        except A.GrammarError:
            continue
        c2 = g.completion_frontier(s2)
        if c2 is None:
            continue
        delta = tuple(a - b for a, b in zip(s2.used, s.used))
        succ.extend(_proj(g, s, tuple(d + x for d, x in zip(delta, w))) for w in c2)
    for v in c or ():
        if not any(A._leq(w, v) for w in succ):
            bad.append(("unrealized", tuple(v)))
    if g.completion_fits(s) and not g.allowed(s).any():
        bad.append(("empty_mask", g.min_to_complete(s)))
    return bad


def tier_a_parses(text):
    try:
        O.parse_genome(text)
        return True
    except (O.CanonicalError, O.Undecidable):
        return False


def canonical_or_none(text):
    try:
        return O.canonical(text)
    except (O.CanonicalError, O.Undecidable):
        return None


class NumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab, cls.g = make_grammar(TRAIN, max_seq=128)
        cls.pv, cls.pg = make_grammar(["(pipe X (ridge a=12 b=0.25 c=-3 d=-0.5 m=1))",
                                       "(pipe X (ridge a=7 b=0.5 c=-12 d=-0.75 m=1.0))"], max_seq=64)

    def verdict(self, g, vocab, mode, text):
        return g.accepts(ids_of(vocab, text), mode)

    def test_design_values(self):
        ok = ["0", "999761", "-1.0", "-0.0", "0.0", "0.000001", "0.09988591634790045", "0.5408000000000001",
              "0.12345678901234567", "100000000000000000000.0"]
        bad = ["01", "-0", "1.", ".5", "1.50", "1e-5", "+5", "1.0e3", "0.123456789012345678", "00.5", "-01",
               "12345678901234567890"]
        for v in ok:
            self.assertTrue(self.verdict(self.g, self.vocab, SYN, f"(pipe X (ridge lambda_frac={v}))").ok, v)
        for v in bad:
            self.assertFalse(self.verdict(self.g, self.vocab, SYN, f"(pipe X (ridge lambda_frac={v}))").ok, v)
        report("numeric_dfa_syntax", {"accepted": ok, "refused": bad})

    def test_policy_tables(self):
        g, v = self.pg, self.pv

        def acc(a="12", b="0.25", c="-3", d="-0.5", m="1"):
            return g.accepts(ids_of(v, f"(pipe X (ridge a={a} b={b} c={c} d={d} m={m}))"), POL).ok

        self.assertTrue(acc())
        cases = {"a=99": acc(a="99"), "a=100 (3 digits)": acc(a="100"), "a=-1 (no negative)": acc(a="-1"),
                 "a=1.5 (Int only)": acc(a="1.5"), "b=0.75": acc(b="0.75"), "b=0.125 (3 frac)": acc(b="0.125"),
                 "b=12.5 (2 int)": acc(b="12.5"), "b=1 (Float only)": acc(b="1"), "b=-0.5": acc(b="-0.5"),
                 "c=-7": acc(c="-7"), "c=-0": acc(c="-0"), "d=-0.25": acc(d="-0.25"), "d=-0.0": acc(d="-0.0"),
                 "m=1.0": acc(m="1.0"), "m=7": acc(m="7")}
        want = {"a=99": True, "a=100 (3 digits)": False, "a=-1 (no negative)": False, "a=1.5 (Int only)": False,
                "b=0.75": True, "b=0.125 (3 frac)": False, "b=12.5 (2 int)": False, "b=1 (Float only)": False,
                "b=-0.5": False, "c=-7": True, "c=-0": False, "d=-0.25": True, "d=-0.0": True, "m=1.0": True,
                "m=7": True}
        self.assertEqual(cases, want)
        # no dead end at the last permitted fraction position (b: 2 fraction digits)
        pre = ids_of(v, "(pipe X (ridge a=12 b=0.25 c=-3 d=-0.5 m=1))")
        k = pre.index(v.id("KEY", "b")) + 1
        digit = {ch: T.byte_id(ord(ch)) for ch in "0123456789."}
        for prefix, zero_ok in (("0.", True), ("0.0", False), ("0.2", False)):
            s = path_state(g, POL, pre[:k] + [digit[ch] for ch in prefix])
            m = g.allowed(s)
            self.assertEqual(bool(m[digit["0"]]), zero_ok, prefix)
            self.assertTrue(m[digit["5"]], prefix)
        report("numeric_dfa_policy", {"cases": cases})


class SymTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab, cls.g = make_grammar(TRAIN + ("(sym affine \"clip(c1,c2,c3)+min(c4,c5)\")",), max_seq=512)

    def acc(self, body, mode, raw=False):
        text = "(sym " + ("raw" if raw else "affine") + " \"" + body + "\")"
        return self.g.accepts(ids_of(self.vocab, text), mode)

    def test_design_bodies(self):
        good = ["c0", "(c1*-0.43)", "((-0.87)--1.44)", "sigmoid(((c5^1.2705454457961025)/-1.1374340345718594))",
                "clip(c1,c2,c3)", "min(c4,c5)"]
        for b in good:
            for mode in (POL, SYN):
                self.assertTrue(self.acc(b, mode).ok, (b, mode))
        self.assertTrue(self.acc("8.727099312855202e-5", SYN).ok)
        self.assertFalse(self.acc("8.727099312855202e-5", POL).ok)      # no exponent in the train tables
        bad = ["min(c1)", "clip(c1,c2)", "sigmoid c1", "c", "c1)", "(c1", "min(c1,c2,c3)", "c1 c2", "", "1..5",
               "sigmoid()", "x(c1)", "unknownvar"]
        refused = {}
        for b in bad:
            for mode in (POL, SYN):
                v = self.acc(b, mode)
                self.assertFalse(v.ok, (b, mode))
                refused[f"{mode.value}:{b}"] = v.reason
        # COL without digits, ids built by hand
        ids = [A.BOS, A.SYM_AFFINE_OPEN, A.COL, A.SYM_CLOSE, A.EOS]
        self.assertFalse(self.g.accepts(ids, SYN).ok)
        report("sym_design", {"refused": refused})

    def test_nodes_and_depth(self):
        def balanced(n):
            if n == 1:
                return "c1"
            return "(" + balanced(n // 2) + "+" + balanced(n - n // 2) + ")"

        b63 = balanced(32)                         # 63 nodes, depth 6
        self.assertEqual(O.parse_math(b63).nodes, 63)
        self.assertTrue(self.acc(b63, SYN).ok)
        self.assertTrue(self.acc("-" + b63, SYN).ok)                       # 64 nodes
        v = self.acc("--" + b63, SYN)                                     # 65 nodes
        self.assertFalse(v.ok)
        self.assertIn("sym_nodes", v.reason)
        self.assertTrue(self.acc("-" * 15 + "c1", SYN).ok)                # depth 16
        self.assertFalse(self.acc("-" * 16 + "c1", SYN).ok)               # depth 17
        chain = "+".join(["c1"] * 16)                                      # left-assoc chain: depth 16
        self.assertEqual(O.parse_math(chain).depth, 16)
        self.assertTrue(self.acc(chain, SYN).ok)
        self.assertFalse(self.acc(chain + "+c1", SYN).ok)
        pw = "^".join(["c1"] * 16)                                         # right-assoc pow: depth 16
        self.assertTrue(self.acc(pw, SYN).ok)
        self.assertFalse(self.acc(pw + "^c1", SYN).ok)
        # POLICY node cap = train_fit max (9 here)
        self.assertTrue(self.acc(balanced(5), POL).ok)
        self.assertFalse(self.acc(balanced(6), POL).ok)
        report("sym_nodes_depth", {"syntax_nodes_cap": 64, "policy_nodes_cap": self.g.ps["nodes_cap"]})

    def test_differential_vs_tier_a(self):
        """SYNTAX accepts a sym text iff Tier A parses it, except the documented finite-state exclusions."""
        rng = random.Random(4)
        leaves = ("c0", "c007", "C7", "col7", "x", "row_mean", "mean", "ROW_STD", "row_l1", "0.5", "1e-5", "1E5",
                  "2.5e+3", ".5", "5.", "007", "1e400", "8.727099312855202e-5", "col(3)", "col(3.0)", "col(3.5)",
                  "col((2))", "col(+4)", "col(-1)", "c", "q", "sigmoid", "1.2.3", "1e", "c99999999999999999999")
        funcs = {"abs": 1, "sigmoid": 1, "Exp": 1, "min": 2, "MAX": 2, "step": 2, "clip": 3, "col": 1, "foo": 1}
        ws = ("", "", " ", "\t", "\xa0", "\u3000")

        def body(d):
            k = rng.random()
            if d <= 0 or k < 0.3:
                return rng.choice(leaves)
            if k < 0.42:
                return rng.choice("-+") + body(d - 1)
            if k < 0.7:
                w = rng.choice(ws)
                return rng.choice(("({}{w}{}{w}{})", "{}{w}{}{w}{}")).format(body(d - 1), rng.choice("+-*/^"),
                                                                             body(d - 1), w=w)
            name = rng.choice(sorted(funcs))
            n = funcs[name] + (rng.random() < 0.1) * rng.choice((-1, 1))
            return name + "(" + ("," + rng.choice(ws)).join(body(d - 1) for _ in range(max(n, 0))) + ")"

        agree = Counter()
        explained = Counter()
        unexplained = []
        for _ in range(3000):
            b = rng.choice(ws) + body(rng.randint(0, 4)) + rng.choice(ws)
            text = "(sym affine \"" + b + "\")"
            ids = ids_of(self.vocab, text)
            if len(ids) > self.g.max_seq:
                agree["skipped_over_max_seq"] += 1
                continue
            agree["raw_lexing"] += T.program_form(text) == "raw"
            mine = self.g.accepts(ids, SYN).ok
            try:
                O.parse_math(b)
                ref = True
            except O.CanonicalError:
                ref = False
            if mine == ref:
                agree["agree_" + ("accept" if ref else "reject")] += 1
            elif ref and re.search(r"col\(", b, re.I):
                explained["col_call_argument"] += 1
            elif ref and re.search(r"c(ol)?0*[0-9]{20,}", b, re.I):
                explained["col_index_digits"] += 1
            else:
                unexplained.append(b)
        self.assertEqual(unexplained[:5], [])
        self.assertGreater(agree["agree_accept"], 300)
        self.assertGreater(agree["agree_reject"], 300)
        report("sym_differential", {"counts": dict(agree), "explained_syntax_refusals": dict(explained)})


class CapsTests(unittest.TestCase):
    def refused(self, g, vocab, text, mode=POL):
        v = g.accepts(ids_of(vocab, text), mode)
        self.assertFalse(v.ok, text[:80])
        return v

    def accepted(self, g, vocab, text, mode=POL):
        v = g.accepts(ids_of(vocab, text), mode)
        self.assertTrue(v.ok, (text[:80], v))

    def test_design_caps(self):
        out = {}
        # 11th stage (ridge observed final and non-final); the 12th '(' comes with it
        vocab, g = make_grammar(["(pipe X (ridge) (ridge))"], max_seq=512)
        self.accepted(g, vocab, "(pipe X" + " (ridge)" * 10 + ")")
        v = self.refused(g, vocab, "(pipe X" + " (ridge)" * 11 + ")")
        self.assertEqual(v.position, 2 + 3 * 10)                     # the 11th STAGE_OPEN
        self.assertIn("stages", v.reason)
        out["stage_11"] = v.reason
        # 4th winsor_clip
        vocab, g = make_grammar(["(pipe X (winsor_clip q=0.5) (ridge))"], max_seq=512)
        self.accepted(g, vocab, "(pipe X" + " (winsor_clip q=0.5)" * 3 + " (ridge))")
        v = self.refused(g, vocab, "(pipe X" + " (winsor_clip q=0.5)" * 4 + " (ridge))")
        self.assertIn("winsor", v.reason)
        out["winsor_4"] = v.reason
        # 12th '(' in a pipe through a quoted literal holding parentheses
        lit = "(map expr=\"f(g(x))\")"
        vocab, g = make_grammar(["(pipe X " + lit + " " + lit + " (ridge))"], max_seq=512)
        self.accepted(g, vocab, "(pipe X " + " ".join([lit] * 3) + " (ridge))")            # 11 '('
        v = self.refused(g, vocab, "(pipe X " + " ".join([lit] * 4) + " (ridge))")         # 14 '('
        self.assertIn("lparen", v.reason)
        out["pipe_lparen_12"] = v.reason
        # 19th '(' in a sym (groups observed, one node)
        vocab, g = make_grammar(["(sym affine \"((((c1))))\")"], max_seq=512)
        self.accepted(g, vocab, "(sym affine \"" + "(" * 17 + "c1" + ")" * 17 + "\")")
        v = self.refused(g, vocab, "(sym affine \"" + "(" * 18 + "c1" + ")" * 18 + "\")")
        self.assertEqual(v.position, 2 + 17)
        self.assertIn("lparen", v.reason)
        out["sym_lparen_19"] = v.reason
        # portfolio members: observed {2, 3}; 1 and 4 refused in POLICY, allowed by SYNTAX (parser 1..12)
        m = " (member (pipe X (ridge)))"
        vocab, g = make_grammar(["(portfolio" + m * 2 + BLEND1, "(portfolio" + m * 3 + BLEND1], max_seq=512)
        for k in (2, 3):
            self.accepted(g, vocab, "(portfolio" + m * k + BLEND1)
        for k in (1, 4):
            out[f"members_{k}"] = self.refused(g, vocab, "(portfolio" + m * k + BLEND1).reason
            self.accepted(g, vocab, "(portfolio" + m * k + BLEND1, SYN)
        self.refused(g, vocab, "(portfolio" + m * 13 + BLEND1, SYN)
        # 25 member stages (no per-member cap; the sum is capped at 24)
        vocab, g = make_grammar(["(portfolio (member (pipe X (ridge) (ridge))) (member (pipe X (ridge)))" + BLEND1],
                                max_seq=512)
        mk = lambda a, b: ("(portfolio (member (pipe X" + " (ridge)" * a + ")) (member (pipe X" + " (ridge)" * b
                           + "))" + BLEND1)
        self.accepted(g, vocab, mk(12, 12))
        v = self.refused(g, vocab, mk(12, 13))
        self.assertIn("stages", v.reason)
        out["member_stages_25"] = v.reason
        # 32,769 bytes (both modes): a 4,000-byte op name
        long_op = "o" * 4000
        vocab, g = make_grammar([f"(pipe X ({long_op}) (ridge))"], max_seq=512)
        ok_text = "(pipe X" + f" ({long_op})" * 8 + " (ridge))"
        bad_text = "(pipe X" + f" ({long_op})" * 9 + " (ridge))"
        self.assertLessEqual(len(ok_text.encode()), 32768)
        self.assertGreater(len(bad_text.encode()), 32768)
        for mode in (POL, SYN):
            self.accepted(g, vocab, ok_text, mode)
            v = self.refused(g, vocab, bad_text, mode)
            self.assertIn("max_bytes", v.reason)
            self.assertEqual(v.position, 2 + 3 * 8 + 1)                  # the 9th long op token
        out["bytes_32769"] = v.reason
        # the token at the max_seq boundary
        vocab, g = make_grammar(["(pipe X (ridge a=1))"], max_seq=32)
        texts = {}
        for d in range(1, 20):
            for k in range(0, 10):
                text = "(pipe X (ridge a=" + "1" * d + ")" + " (ridge)" * k + ")"
                texts.setdefault(len(ids_of(vocab, text)), text)
        exact, over = texts[32], texts[33]
        self.accepted(g, vocab, exact, SYN)
        v = self.refused(g, vocab, over, SYN)
        self.assertIn("max_seq", v.reason)
        ids = ids_of(vocab, over)
        s = path_state(g, SYN, ids[:v.position])
        self.assertFalse(g.allowed(s)[ids[v.position]])
        self.assertLessEqual(g.min_to_complete(s), g.max_seq - s.used[0])
        out["max_seq_boundary"] = {"accepted_tokens": 32, "refused_tokens": 33, "refused_at": v.position}
        report("caps", out)


class UniquenessAndRulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab, cls.g = make_grammar(TRAIN, max_seq=128)

    def spell(self, mode, prefix_text, begin, chars):
        ids = ids_of(self.vocab, prefix_text)[:-1]          # drop EOS
        return path_state(self.g, mode, ids + [begin] + [T.byte_id(b) for b in chars.encode("utf-8")])

    def test_uniqueness_masks(self):
        v = self.vocab
        pre = "(pipe X (ridge"
        base = ids_of(v, pre + "))")[:4]                     # BOS PIPE_OPEN STAGE_OPEN OP:ridge
        key = v.id("KEY", "lambda_frac")
        res = {}

        def end_ok(mode, ids, begin, chars, end):
            s = path_state(self.g, mode, ids + [begin] + [T.byte_id(b) for b in chars.encode("utf-8")])
            return bool(self.g.allowed(s)[end])

        for text, want in (("true", False), ("false", False), ("1.5", False), ("12", False), ("huber", False),
                           ("huberx", True), ("NaN", True), ("inf", True), ("TRUE", False), ("\"a b\"", True),
                           ("\"ab\"", False), ("\"\"", True)):
            got = end_ok(SYN, base + [key], A.STR_BEGIN, text, A.STR_END)
            self.assertEqual(got, want, text)
            res["STR:" + text] = got
        sym = ids_of(v, "(sym affine \"c1\")")[:2]
        for text, want in (("c12", False), ("sigmoid", False), ("SIGMOID", True), ("C12", True), ("row_mean", False),
                           ("ROW_MEAN", True), ("c", False), ("col", True), ("x", True), ("zz", False)):
            got = end_ok(SYN, sym, A.ID_BEGIN, text, A.ID_END)
            self.assertEqual(got, want, text)
            res["ID:" + text] = got
        stage = ids_of(v, pre + "))")[:3]
        for text, want in (("ridge", False), ("ridgex", True), ("op\"q\"", True), ("op\"q", False)):
            got = end_ok(SYN, stage, A.OP_BEGIN, text, A.OP_END)
            self.assertEqual(got, want, text)
            res["OP:" + text] = got
        for text, want in (("lambda_frac", False), ("lambda_fraz", True), ("", True)):
            got = end_ok(SYN, base, A.KEY_BEGIN, text, A.KEY_END)
            self.assertEqual(got, want, text)
            res["KEY:" + text] = got
        # POLICY never spells a name; it replays observed STR literals only
        s = path_state(self.g, POL, base)
        self.assertFalse(self.g.allowed(s)[A.KEY_BEGIN])
        s = path_state(self.g, POL, ids_of(v, pre + "))")[:3])
        self.assertFalse(self.g.allowed(s)[A.OP_BEGIN])
        s = path_state(self.g, POL, sym)
        self.assertFalse(self.g.allowed(s)[A.ID_BEGIN])
        s = path_state(self.g, POL, base + [key])
        self.assertFalse(self.g.allowed(s)[A.STR_BEGIN])
        mp = ids_of(v, "(pipe X (map expr=\"sigmoid(col(0))\") (ridge lambda_frac=0.5))")
        k = mp.index(v.id("KEY", "expr"))
        s = path_state(self.g, POL, mp[:k + 1])
        self.assertTrue(self.g.allowed(s)[A.STR_BEGIN])
        self.assertTrue(self.g.accepts(mp, POL).ok)
        report("uniqueness", {"end_allowed": res})

    def test_generation_rules(self):
        v, g = self.vocab, self.g
        tgt = v.id("OP", "target")
        lead = A.Rules(target_lead=True)
        second = ids_of(v, "(pipe X (standardize) (ridge_gcv))")[:5]     # ... CLOSE of stage 0, then STAGE_OPEN
        s0 = path_state(g, POL, ids_of(v, "(pipe X (ridge_gcv))")[:3], lead)
        s1 = path_state(g, POL, second + [A.STAGE_OPEN], lead)
        s1_free = path_state(g, POL, second + [A.STAGE_OPEN])
        self.assertTrue(g.allowed(s0)[tgt])
        self.assertFalse(g.allowed(s1)[tgt])
        self.assertTrue(g.allowed(s1_free)[tgt])
        late = "(pipe X (standardize) (target kind=log1p) (ridge_gcv))"
        self.assertTrue(g.accepts(ids_of(v, late), POL).ok)
        self.assertFalse(g.accepts(ids_of(v, late), POL, lead).ok)
        for rules in (lead, A.Rules(feature_count=5, col_lt_p=True)):
            for mode in (POL, SYN):
                doc = A.walk_check(g, 150, seed=9, mode=mode, rules=rules)
                self.assertTrue(doc["pass"], doc["failures"])
        rng = random.Random(2)
        cols = Counter()
        for mode in (POL, SYN):
            for _ in range(300):
                text = T.detokenize(g.sample(rng, mode, A.Rules(feature_count=5, col_lt_p=True)), v)
                if text.startswith("(sym"):
                    for n in re.findall(r"(?<![A-Za-z0-9_])c([0-9]+)", text):
                        cols[int(n) < 5] += 1
        self.assertEqual(cols[False], 0)
        self.assertGreater(cols[True], 20)
        rl = A.Rules(feature_count=5, col_lt_p=True)
        self.assertTrue(g.accepts(ids_of(v, "(sym affine \"c4\")"), POL, rl).ok)
        self.assertFalse(g.accepts(ids_of(v, "(sym affine \"c5\")"), POL, rl).ok)
        self.assertFalse(g.accepts(ids_of(v, "(sym affine \"c39\")"), POL, rl).ok)
        self.assertTrue(g.accepts(ids_of(v, "(sym affine \"c39\")"), POL).ok)
        with self.assertRaises(ValueError):
            g.start(POL, A.Rules(col_lt_p=True))
        walks = 0
        lead_bad = 0
        for _ in range(400):
            text = T.detokenize(g.sample(rng, POL, lead), v)
            if text.startswith("(pipe") or text.startswith("(portfolio"):
                walks += 1
                lead_bad += any("target_input_not_leading" in x for x in O.static_semantics(text))
        self.assertEqual(lead_bad, 0)
        report("generation_rules", {"col_refs_lt_p": cols[True], "target_lead_walks": walks})


def enumerate_all(g, mode, limit=400000):
    """Every complete sequence reachable through allowed() (depth-first); counts dead ends."""
    seqs, dead, states = [], 0, 0
    stack = [(g.start(mode), [])]
    while stack:
        s, ids = stack.pop()
        states += 1
        if states > limit:
            raise AssertionError("state budget exceeded")
        if s.form == A.F_DONE:
            seqs.append(ids)
            continue
        cand = np.flatnonzero(g.allowed(s)).tolist()
        if not cand:
            dead += 1
            continue
        for t in cand:
            stack.append((g.advance(s, t), ids + [t]))
    return seqs, dead, states


class ConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab, cls.g = make_grammar(TRAIN, max_seq=96)

    def test_brute_force_masks_and_mask_keys(self):
        g, rng = self.g, random.Random(6)
        states = []
        for mode in (POL, SYN):
            for d in TRAIN:
                s = g.start(mode)
                for t in ids_of(self.vocab, d):
                    states.append(s)
                    s = g.advance(s, t)
            for rules in (A.Rules(), A.Rules(target_lead=True), A.Rules(feature_count=7, col_lt_p=True)):
                for bias in ("class", "uniform"):
                    for _ in range(8):
                        s = g.start(mode, rules)
                        for t in g.sample(rng, mode, rules, bias=bias):
                            states.append(s)
                            s = g.advance(s, t)
        rng.shuffle(states)
        states = states[:1800]
        keys, mism, by_form = {}, 0, Counter()
        for s in states:
            a = g.allowed(s)
            b = brute_allowed(g, s)
            mism += not np.array_equal(a, b)
            by_form[A.FORM_NAMES[s.form]] += 1
            k = g.mask_key(s)
            self.assertIsInstance(k, bytes)
            self.assertEqual(keys.setdefault(k, a.tobytes()), a.tobytes())
            self.assertTrue(a.any())
            self.assertLessEqual(s.used[0] + g.min_to_complete(s), g.max_seq)
            self.assertFalse(a.flags.writeable)
        self.assertEqual(mism, 0)
        report("brute_force_masks", {"states": len(states), "mismatches": mism, "by_form": dict(by_form),
                                     "distinct_mask_keys": len(keys), "cache": g.cache_stats()})

    def test_compile_masks(self):
        g = self.g
        seqs = [ids_of(self.vocab, d) for d in TRAIN]
        out = {}
        for mode in (POL, SYN):
            table, ids = g.compile_masks(seqs, mode)
            self.assertEqual(table.dtype, np.bool_)
            self.assertEqual(table.shape[1], g.V)
            self.assertEqual(len({r.tobytes() for r in table}), len(table))
            for seq, m in zip(seqs, ids):
                self.assertEqual(len(m), len(seq))
                self.assertEqual(np.flatnonzero(table[m[0]]).tolist(), [A.BOS])
                s = g.start(mode)
                for i, t in enumerate(seq):
                    self.assertTrue(np.array_equal(table[m[i]], g.allowed(s)))
                    self.assertTrue(table[m[i]][t])
                    s = g.advance(s, t)
                self.assertTrue(g.is_final(s))
            out[mode.value] = {"positions": sum(len(x) for x in seqs), "table_rows": int(table.shape[0])}
        bad = ids_of(self.vocab, "(pipe X (zz_unseen_op))")
        with self.assertRaises(A.GrammarError):
            g.compile_masks([bad], POL)
        table, ids = g.compile_masks([bad], SYN)
        self.assertEqual(len(ids[0]), len(bad))
        with self.assertRaises(A.GrammarError):
            g.compile_masks([bad[:-1]], SYN)                     # no EOS
        report("compile_masks_fixtures", out)

    def test_exhaustive_small_budgets(self):
        out = {}
        vocab, g = make_grammar(["(pipe X (ridge a=1))", "(sym affine \"c1\")", "(pipe X (standardize) (ridge))"],
                                max_seq=12)
        seqs, dead, states = enumerate_all(g, POL)
        self.assertEqual(dead, 0)
        # pipes: (ridge) 7 tokens, (ridge a=0..9) 9, (standardize)(ridge) 10, (standardize)(ridge a=0..9) 12;
        # (standardize)(standardize)(ridge) needs 13 > 12; sym: c0..c9 (6 tokens)
        self.assertEqual(len(seqs), 1 + 10 + 1 + 10 + 10)
        for ids in seqs:
            text = T.detokenize(ids, vocab)
            self.assertTrue(len(ids) <= 12 and g.accepts(ids, POL).ok)
            self.assertEqual(canonical_or_none(text), text)
            self.assertEqual(O.form_caps(text), [])
        out["policy_max_seq_12"] = {"sequences": len(seqs), "states": states}
        for max_seq in (6, 7):
            vocab, g = make_grammar(["(pipe X (ridge))", "(sym affine \"c1\")"], max_seq=max_seq)
            seqs, dead, states = enumerate_all(g, SYN)
            self.assertEqual(dead, 0)
            bad = [ids for ids in seqs if not (tier_a_parses(T.detokenize(ids, vocab))
                                               and T.tokenize(T.detokenize(ids, vocab), vocab) == ids)]
            self.assertEqual(bad[:3], [])
            forms = Counter(T.program_form(T.detokenize(ids, vocab)) for ids in seqs)
            out[f"syntax_max_seq_{max_seq}"] = {"sequences": len(seqs), "states": states, "forms": dict(forms)}
        report("exhaustive", out)

    def test_bellman_frontier_consistency(self):
        """Every frontier vector of every reached state is realized by some token (brute force over all V tokens,
        so a poisoned completion memo cannot hide behind allowed()), and no state with a fitting completion has an
        empty mask. States: the fixture programs, quoted-STR paths in both evaluation orders, and masked walks of
        both modes and biases (the uniform ones press on the budgets)."""
        g, rng = self.g, random.Random(11)
        extra = ['(pipe X (ridge q=""))', '(pipe X (ridge q="a b"))', '(pipe X (ridge q="a" r=1))',
                 '(pipe X (op \xe9=1 \xfc=2))', '(pipe X (op 中=1 中x="a b"))']
        states = []
        for mode in (POL, SYN):
            for d in TRAIN + tuple(extra if mode is SYN else ()):
                s = g.start(mode)
                for t in ids_of(self.vocab, d):
                    states.append(s)
                    try:
                        s = g.advance(s, t)
                    except A.GrammarError:
                        break
            for bias in ("class", "uniform"):
                for _ in range(10):
                    s = g.start(mode)
                    for t in g.sample(rng, mode, bias=bias):
                        states.append(s)
                        s = g.advance(s, t)
        rng.shuffle(states)
        states = states[:1500]
        problems = []
        for s in states:
            problems.extend(bellman_violations(g, s))
        self.assertEqual(problems[:5], [])
        report("bellman_frontier", {"states": len(states), "violations": len(problems),
                                    "max_frontier": max(len(g.completion_frontier(s) or ()) for s in states)})


class RegressionTests(unittest.TestCase):
    """The canonical-fidelity review's reproductions (CF-1, CF-2, CF-4, CF-5)."""

    def test_cf1_quoted_str_memo_is_exact_in_both_orders(self):
        """`""` (canonical) and a closed `"a"` without a special char (no completion) are different spelled
        states; whichever is evaluated first, the closing quote after `"a` stays masked and `""` stays accepted."""
        quote = T.byte_id(0x22)
        for order in ("empty_first", "nonspecial_first"):
            with self.subTest(order=order):
                vocab, g = make_grammar(TRAIN, max_seq=512)
                empty = ids_of(vocab, '(pipe X (ridge q=""))')
                spaced = ids_of(vocab, '(pipe X (ridge q="a b"))')
                i = spaced.index(A.STR_BEGIN)
                if order == "empty_first":
                    self.assertTrue(g.accepts(empty, SYN).ok)
                s = path_state(g, SYN, spaced[:i + 3])                     # ... STR_BEGIN '"' 'a'
                self.assertFalse(g.allowed(s)[quote])
                self.assertFalse(brute_allowed(g, s)[quote])
                v = g.accepts(empty, SYN)
                self.assertTrue(v.ok, v)
                self.assertTrue(g.accepts(spaced, SYN).ok)
                table, mids = g.compile_masks([spaced, empty], SYN)
                self.assertFalse(table[mids[0][i + 3]][quote])             # the mask 'b' was chosen under
                self.assertTrue(table[mids[1][i + 2]][quote])              # `""`: the close after the open quote
                with self.assertRaises(A.GrammarError):                     # Tier A turns q="a" into q=a
                    g.compile_masks([ids_of(vocab, '(pipe X (ridge q="a"))')], SYN)
                for k in range(len(spaced)):
                    self.assertEqual(bellman_violations(g, path_state(g, SYN, spaced[:k])), [])
        self.assertEqual(O.canonical('(pipe X (ridge q="a"))'), "(pipe X (ridge q=a))")
        self.assertEqual(O.canonical('(pipe X (ridge q=""))'), '(pipe X (ridge q=""))')
        report("cf1_quoted_str", {"orders": 2})

    def test_cf2_policy_accepts_its_train_fit_under_every_cap(self):
        """The token-minimal completion is not '('-minimal: POLICY must still accept every program its tables were
        built from, whichever literal the cheapest completion would take (pipe and portfolio variants)."""
        l1, l2 = '"((((((("', '"a b c d e f g h"'
        mid = " ".join("(%s)" % op for op in ("standardize", "robust_scale", "norm_row", "blom", "sigmoid", "polar",
                                                 "rank", "maxabs_scale"))
        train = ["(pipe X (map expr=%s) (ridge))" % l1, "(pipe X %s (map expr=%s) (ridge))" % (mid, l2)]
        for t in train:
            self.assertEqual((O.canonical(t), O.form_caps(t)), (t, []))
        vocab, g = make_grammar(train)
        for t in train:
            v = g.accepts(ids_of(vocab, t), POL)
            self.assertTrue(v.ok, (t[:60], v))
        table, _ = g.compile_masks([ids_of(vocab, t) for t in train], POL)
        self.assertGreater(len(table), 0)
        self.assertTrue(g.accepts(ids_of(vocab, "(pipe X (map expr=%s) %s (ridge))" % (l2, mid)), POL).ok)
        # the paren-heavy literal after eight stages really breaks the cap (1 + 10 + 7 = 18 > 11): refused there
        late = "(pipe X %s (map expr=%s) (ridge))" % (mid, l1)
        v = g.accepts(ids_of(vocab, late), POL)
        self.assertFalse(v.ok)
        self.assertIn("lparen", v.reason)
        # portfolio variant: a 15-'(' literal against a paren-free one, 3 members (cap 32)
        l3, l4 = '"' + "(" * 15 + '"', '"a b c d e f g h i j k l m n o p"'
        mid7 = " ".join("(%s)" % op for op in ("standardize", "robust_scale", "norm_row", "blom", "sigmoid",
                                                  "polar", "rank"))
        tail = " (member (pipe X (ridge))) (member (pipe X (ridge)))"
        ports = ["(portfolio (member (pipe X (map expr=%s) (ridge)))%s%s" % (l3, tail, BLEND1),
                 "(portfolio (member (pipe X %s (map expr=%s) (ridge)))%s%s" % (mid7, l4, tail, BLEND1)]
        for t in ports:
            self.assertEqual((O.canonical(t), O.form_caps(t)), (t, []))
        vocab, g = make_grammar(ports)
        for t in ports:
            v = g.accepts(ids_of(vocab, t), POL)
            self.assertTrue(v.ok, (t[:60], v))
        report("cf2_policy_frontier", {"pipe_programs": 2, "portfolio_programs": 2})

    def test_cf4_odd_quote_keys_are_a_documented_refusal(self):
        """Keys holding an odd number of '"' are canonical (Tier A unchanged) and round-trip losslessly, but sit
        outside SYNTAX (documented refusal); a train_fit program with one is a TokenizerStop, not a bare error."""
        vocab, g = make_grammar(TRAIN, max_seq=512)
        for text in ('(pipe X (op a"b="c\\"=d"))', '(pipe X (op k"="v\\""))', '(pipe X (op k"="a=b\\""))',
                     '(pipe X (ridge alpha=1.0 k"="v\\""))'):
            with self.subTest(text=text):
                self.assertEqual(O.canonical(text), text)
                ids = ids_of(vocab, text)
                self.assertEqual(T.detokenize(ids, vocab), text)
                v = g.accepts(ids, SYN)
                self.assertFalse(v.ok)
                self.assertIn("documented SYNTAX refusal", v.reason)
                self.assertEqual(ids[v.position], A.KEY_END)
        with self.assertRaises(T.TokenizerStop):
            T.build_vocab(['(pipe X (op k"="v\\""))'])
        report("cf4_odd_quote_keys", {"refused": 4})

    def test_cf5_keys_after_non_ascii_keys(self):
        """A key after a non-ASCII key (or after '~') can be spelled: the completion search follows the previous
        key's bytes and may go above them anywhere in 0x00-0xFF; a search that stops at its limit says so."""
        vocab, g = make_grammar(TRAIN, max_seq=512)
        cases = ["(pipe X (op \xe9=1 \xfc=2))", "(pipe X (op \xe9.=1 \xe91=2))", "(pipe X (op \xe9=1 \xe9x=2))",
                 "(pipe X (op 中=1 中x=2))", "(pipe X (\xe9 \xe9=1 \xfc=\"a b\"))", "(pipe X (op ~=1 \x7f=2))",
                 "(pipe X (op a=1 \xfc=2))", "(pipe X (op \U0010ffff=1 \U0010ffffa=2))"]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(O.canonical(text), text)
                ids = ids_of(vocab, text)
                v = g.accepts(ids, SYN)
                self.assertTrue(v.ok, v)
                for k in range(len(ids)):
                    self.assertEqual(bellman_violations(g, path_state(g, SYN, ids[:k])), [])
        # '~' then DEL: the one-byte completion 0x7F is found (the cost is not overestimated)
        ids = ids_of(vocab, "(pipe X (op ~=1 \x7f=2))")
        k = [i for i, t in enumerate(ids) if t == A.KEY_BEGIN][1]
        s = path_state(g, SYN, ids[:k + 1])                                    # ... KEY_BEGIN after the '~' key
        self.assertEqual((s.sub[0], s.sub[4]), (A.P_KEYSP, b"~"))
        self.assertEqual(g._spell_extra(s.ctx, "KEY", s.sub[6], s.sub[4]), 1)
        # a spelled-name search that stops at its limit is not reported as a dead end
        with mock.patch.object(A, "_SPELL_BFS_LIMIT", -1):
            vocab2, g2 = make_grammar(TRAIN, max_seq=512)
            v = g2.accepts(ids_of(vocab2, "(pipe X (op a=1 b=2))"), SYN)
        self.assertFalse(v.ok)
        self.assertIn("search limit", v.reason)
        report("cf5_non_ascii_keys", {"accepted": len(cases)})


# --- canonical generator (Tier A printer), for the SYNTAX superset test ---
OPS = ("ridge", "histgbm2", "extra_trees", "winsor_clip", "target", "map", "portfolio", "zz_unseen_op", "op=x",
       "a\\b", "\xe9t\xe9", "op\"q\"", "UPPER", "x(y)z", "op\"a b\"")
KEYS = ("alpha", "max_features", "k", "", "zz_unseen_key", "a\\b", "k\"x\"", "\xe9", "a=b", "q", "k\"(x)\"",
        "\xfc", "\xe9x", "\u4e2d", "\u4e2dx", "~", "\x7f")        # non-ASCII and edge-byte keys share stages
STRS = ("huber", "rbf+linear", "features", "zz_unseen_cat", "\xe9\u4e2d", "a b", "(x)", "a=b", "", "a\"b",
        "sigmoid(col(0))", "nan", "TRUE", "1e-5", "x\\y", "\xa0lead", "tab\there", "inf", "-", "\u3000")
FLOATS = (-0.0, 0.0, 1.0, 0.5, 1e-6, 0.09988591634790045, 0.5408000000000001, 1e20, -2.5, float("inf"),
          -float("inf"), float("nan"), 1.234567890123456e-7, 9223372036854775808.0, 3.0000000000004, 1e-12)
FUNCS = {"abs": 1, "sqrt": 1, "sigmoid": 1, "exp": 1, "min": 2, "max": 2, "step": 2, "clip": 3}
LEAVES = ("x", "row_mean", "mean", "ROW_STD", "Row_Sum", "row_l1", "row_norm", "C7", "col7", "0.5", "1e-5", "1E5",
          "2.5e+3", ".5", "5.", "007", "8.727099312855202e-5", "1.2705454457961025", "col(2)", "col( (3) )")
WS = ("", "", "", " ", "\xa0", "\t", "\u3000")


def rand_value(rng):
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


def rand_pipe(rng, max_stages=5):
    stages = []
    for _ in range(rng.randint(1, max_stages)):
        params = {rng.choice(KEYS): rand_value(rng) for _ in range(rng.randint(0, 4))}
        stages.append((rng.choice(OPS), tuple(sorted(params.items()))))
    return O.Pipeline(tuple(stages))


def rand_body(rng, depth):
    k = rng.random()
    if depth <= 0 or k < 0.3:
        return ("c" + ("00" if rng.random() < 0.1 else "") + str(rng.randint(0, 1025))) if rng.random() < 0.5 \
            else rng.choice(LEAVES)
    if k < 0.42:
        return rng.choice("-+") + rand_body(rng, depth - 1)
    if k < 0.7:
        w = rng.choice(WS)
        return rng.choice(("({}{w}{}{w}{})", "{}{w}{}{w}{}")).format(rand_body(rng, depth - 1), rng.choice("+-*/^"),
                                                                     rand_body(rng, depth - 1), w=w)
    name = rng.choice(sorted(FUNCS))
    name = "".join(ch.upper() if rng.random() < 0.15 else ch for ch in name)
    args = [rand_body(rng, depth - 1) for _ in range(FUNCS[name.lower()])]
    return name + "(" + ("," + rng.choice(WS)).join(args) + ")"


def rand_canonical(rng):
    k = rng.random()
    if k < 0.55:
        text = O.render(rand_pipe(rng))
    elif k < 0.9:
        body = rng.choice(WS) + rand_body(rng, rng.randint(0, 4)) + rng.choice(WS)
        text = "(sym " + rng.choice(("affine", "raw")) + " \"" + body + "\")"
    else:
        members = tuple(rand_pipe(rng, 3) for _ in range(rng.randint(1, 3)))
        text = O.render(O.Portfolio(members, rng.random() < 0.5))
    return text if canonical_or_none(text) == text else None


class SupersetTests(unittest.TestCase):
    def test_syntax_superset_of_canonical_image(self):
        vocab, g = make_grammar(TRAIN, max_seq=512)
        rng = random.Random(12)
        got = Counter()
        refused = []
        tries = 0
        while got["canonical"] < 1500:
            tries += 1
            text = rand_canonical(rng)
            if text is None:
                continue
            got["canonical"] += 1
            ids = ids_of(vocab, text)
            if len(ids) > g.max_seq:
                got["over_max_seq"] += 1
                continue
            got["spelled_programs"] += any(i in (A.OP_BEGIN, A.KEY_BEGIN, A.STR_BEGIN, A.ID_BEGIN) for i in ids)
            v = g.accepts(ids, SYN)
            got["syntax_accepted"] += v.ok
            got["policy_accepted"] += g.accepts(ids, POL).ok
            if not v.ok:
                refused.append((text[:120], v.reason))
        self.assertEqual(refused[:5], [])
        self.assertEqual(got["syntax_accepted"], got["canonical"] - got["over_max_seq"])
        report("syntax_superset", {"tries": tries, **dict(got)})


def make_dict(root: Path, parts: dict, *, chunks: int = 1, shuffle_seed=None) -> Path:
    """A synthetic P4a program dictionary: manifest.json + ROW_SCHEMA chunks per partition."""
    manifest = {"schema": DICT_SCHEMA, "partitions": {}}
    for part in T.PARTITIONS:
        dsls = list(parts.get(part, ()))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(dsls)
        entries = []
        for c in range(max(1, chunks)):
            piece = dsls[c::max(1, chunks)]
            if not piece:
                continue
            rel = f"{part}/part-{c:05d}.parquet"
            rec = T.write_dictionary_chunk(root / rel, piece, part, context=f"ctx{c}")
            entries.append({"path": rel, "rows": rec["rows"], "bytes": rec["bytes"], "sha256": rec["sha256"]})
        manifest["partitions"][part] = {"chunks": entries}
    (root / "manifest.json").write_bytes(json.dumps(manifest, sort_keys=True).encode())
    return root


def source_op_tables() -> dict:
    """Op names of the sha-pinned feat-math tables: registry.rs (op! macro rows), spec.rs plan_op and
    plan_head_stage match arms (plus the portfolio head), heads.rs is_head_op, advanced_v3 is_v3_head and its
    explicit transform list, target_transform.rs is_target_op, and the input pseudo-stage."""
    got = {}
    for name, pin in CROSSCHECK_PINS.items():
        data = (FEAT_MATH / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != pin:
            raise AssertionError(f"{name} differs from its pin")
        got[name] = data.decode("utf-8")
    tables = {"registry.rs": set(re.findall(r'op!\(\s*\d+,\s*"([^"]+)"', got["registry.rs"]))}
    spec = got["spec.rs"].splitlines()
    names = set()
    for fn in ("fn plan_op(", "fn plan_head_stage("):
        start = next(i for i, line in enumerate(spec) if fn in line)
        end = next(i for i in range(start + 1, len(spec)) if spec[i].startswith("}"))
        for line in spec[start:end]:
            m = re.match(r'^ {8}("[a-z_0-9]+"(?:\s*\|\s*"[a-z_0-9]+")*)\s*=>', line)
            if m:
                names.update(re.findall(r'"([a-z_0-9]+)"', m.group(1)))
    names.add("portfolio")                       # build_plan: (portfolio preset=...) compact head
    tables["spec.rs"] = names
    heads = got["heads.rs"]
    body = heads[heads.index("pub fn is_head_op"):]
    tables["heads.rs"] = set(re.findall(r'"([a-z_0-9]+)"', body[:body.index("}")]))
    pipe = got["advanced_v3/pipeline.rs"]
    body = pipe[pipe.index("fn is_v3_head"):]
    tables["advanced_v3/is_v3_head"] = set(re.findall(r'"([a-z_0-9]+)"', body[:body.index("\n}")]))
    m = re.search(r'("maxabs_scale"[^=]*)=>', pipe)
    tables["advanced_v3/transforms"] = set(re.findall(r'"([a-z_0-9]+)"', m.group(1)))
    tables["target_transform.rs"] = set(re.findall(r'op == "([a-z_0-9]+)"', got["target_transform.rs"]))
    tables["input"] = {"input"}
    return tables


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="p4-automaton-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_signatures_and_grammar_determinism_and_refusals(self):
        train = sorted(set(TRAIN))
        s1 = A.build_signatures(train)
        s2 = A.build_signatures(list(reversed(train)))
        self.assertEqual(T._json_bytes(s1), T._json_bytes(s2))
        vocab, _ = T.build_vocab(train)
        g1 = A.build_grammar(s1, vocab, 96)
        self.assertEqual(T._json_bytes(g1), T._json_bytes(A.build_grammar(s2, vocab, 96)))
        self.assertEqual(g1["vocab_sha256"], vocab.sha256)
        self.assertEqual(g1["signatures_sha256"], A.signatures_sha256(s1))
        p_g, p_s = self.tmp / "grammar.json", self.tmp / "signatures.json"
        p_g.write_bytes(T._json_bytes(g1))
        p_s.write_bytes(T._json_bytes(s1))
        g = A.Grammar.load(p_g, p_s, vocab)
        self.assertTrue(all(g.accepts(ids_of(vocab, d), POL).ok for d in train))
        other, _ = T.build_vocab(train + ["(pipe X (zz_other_op))"])
        with self.assertRaises(A.AutomatonStop):
            A.Grammar(g1, s1, other)
        tampered = json.loads(json.dumps(g1))
        tampered["policy"]["sym"]["nodes_cap"] = 64
        with self.assertRaises(A.AutomatonStop):
            A.Grammar(tampered, s1, vocab)
        s_bad = json.loads(json.dumps(s1))
        s_bad["ops"]["zz_not_in_vocab"] = {"final": 1, "nonfinal": 0, "member": 0, "occurrences": 1}
        s_bad["keysets"]["zz_not_in_vocab"] = [[]]
        with self.assertRaises(A.AutomatonStop):
            A.build_grammar(s_bad, vocab, 96)
        with self.assertRaises(A.AutomatonStop):
            A.build_grammar(s1, vocab, 600)
        with self.assertRaises(A.AutomatonStop):
            A.build_signatures(["(pipe X)"])                                  # RAW
        with self.assertRaises(A.AutomatonStop):
            A.build_signatures(["(pipe X (ridge b=1 a=2))"])                  # keys not increasing
        report("artifacts_determinism", {"signatures_bytes": len(T._json_bytes(s1)), "grammar_bytes": len(T._json_bytes(g1)),
                                         "trie_nodes": len(g1["policy"]["trie"]),
                                         "value_profiles": len(g1["policy"]["value_profiles"])})

    def test_tokenizer_integration_gate_poison_cli(self):
        rng = random.Random(3)
        gen = []
        while len(gen) < 160:
            t = rand_canonical(rng)
            if t is not None and t not in gen and len(t) < 400 and not O.form_caps(t):
                gen.append(t)
        train = list(TRAIN) + gen[:100]
        parts = {"train_fit": train + train[:10], "dev": gen[100:130] + ["(pipe X (zz_dev_only_op))"],
                 "diag": gen[130:150], "heldout": gen[150:160]}
        d = make_dict(self.tmp / "dict", parts)
        out = self.tmp / "tok"
        rec = T.build_artifacts(d, None, out, automaton=True)
        self.assertEqual(rec["automaton"], "built")
        d2 = make_dict(self.tmp / "dict-b", parts, chunks=3, shuffle_seed=5)
        T.build_artifacts(d2, None, self.tmp / "tok-b", automaton=True)
        for name in ("signatures.json", "grammar.json", "vocab.json", "lengths.json"):
            self.assertEqual((out / name).read_bytes(), (self.tmp / "tok-b" / name).read_bytes(), name)
        vocab, g = A.load_tok_dir(out)
        doc = A.gate(d, out)
        self.assertTrue(doc["pass"], {p: (s["policy_examples"][:2], s["syntax_examples"][:2])
                                      for p, s in doc["partitions"].items()})
        self.assertTrue((out / "automaton-gate.json").exists())
        tf = doc["partitions"]["train_fit"]
        self.assertEqual(tf["policy_accepted"], tf["programs"])
        self.assertLess(doc["partitions"]["dev"]["policy_accepted"], doc["partitions"]["dev"]["programs"])
        res = T.poison_test(d, out, seed=2, automaton=True)
        self.assertTrue(res["pass"], res)
        self.assertEqual(res["artifacts"], ["grammar.json", "lengths.json", "signatures.json", "vocab.json"])
        self.assertTrue(all(res["identical_after_dev_diag_heldout_poison"].values()))
        self.assertTrue(res["changed_after_train_fit_poison"]["signatures.json"])
        self.assertTrue(res["changed_after_train_fit_poison"]["grammar.json"])

        def run(argv):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = A.main(argv)
            return code, json.loads(buf.getvalue().strip().splitlines()[-1])

        code, cli = run(["gate", "--dict", str(d), "--tok", str(out)])
        self.assertEqual(code, 0, cli)
        # a train_fit program the signature builder cannot table is a STOP through the tokenizer's build CLI
        bad = make_dict(self.tmp / "dict-bad", dict(parts, train_fit=train + ["(pipe X (ridge b=1 a=2))"]))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = T.main(["build", "--dict", str(bad), "--out", str(self.tmp / "tok-bad")])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(buf.getvalue().strip().splitlines()[-1])["stop"], "AutomatonStop")
        self.assertFalse((self.tmp / "tok-bad").exists())
        code, cli = run(["walks", "--tok", str(out), "--n", "200", "--mode", "syntax"])
        self.assertEqual(code, 0, cli)
        (out / "lengths.json").write_bytes((out / "lengths.json").read_bytes().replace(b'"max_seq": ', b'"max_seq": 1'))
        code, cli = run(["walks", "--tok", str(out), "--n", "1"])
        self.assertEqual(code, 2)
        report("tokenizer_integration", {"partitions": {p: {k: s[k] for k in ("programs", "policy_accepted",
                                                                               "syntax_accepted", "over_max_seq")}
                                                        for p, s in doc["partitions"].items()},
                                         "max_seq": doc["max_seq"], "poison_pass": res["pass"]})


class CrosscheckTests(unittest.TestCase):
    def test_source_tables_parse(self):
        tables = source_op_tables()
        self.assertIn("ridge", tables["registry.rs"])
        self.assertIn("winsor_clip", tables["spec.rs"])
        self.assertIn("histgbm2", tables["heads.rs"])
        self.assertEqual(tables["advanced_v3/transforms"], {"maxabs_scale", "power_transform", "piecewise_linear",
                                                            "corr_screen", "pls_project"})
        self.assertIn("extra_trees", tables["advanced_v3/is_v3_head"])
        self.assertEqual(tables["target_transform.rs"], {"target"})
        union = set().union(*tables.values())
        sig = A.build_signatures(sorted(set(TRAIN)))
        self.assertEqual(sorted(set(sig["ops"]) - union), [])
        report("source_tables", {k: len(v) for k, v in sorted(tables.items())} | {"union": len(union)})


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
            rows.append((rec["program"]["program_sha256"], rec["program"]["dsl"]))
        del data
        contexts.append({"dataset_hash": ctx["dataset_hash"], "cohort": ctx["cohort"], "fold": ctx["fold"],
                         "native_ok": ok, "rows": rows})
    _CORPUS = {"contexts": contexts}
    return _CORPUS


@unittest.skipUnless(EPOCH.exists(), "epoch manifest not present")
class CorpusSampleTests(unittest.TestCase):
    def test_corpus_sample_gates_masks_walks(self):
        t0 = time.monotonic()
        contexts = corpus_sample()["contexts"]
        self.assertLessEqual(len(contexts), 40)
        self.assertTrue(all(c["native_ok"] for c in contexts))
        datasets = sorted({c["dataset_hash"] for c in contexts})
        random.Random(SPLIT_SEED).shuffle(datasets)
        fit_ds = set(datasets[:len(datasets) // 2])
        progs, fit, test = {}, set(), set()
        rows = 0
        for c in contexts:
            for sha, dsl in c["rows"]:
                rows += 1
                self.assertEqual(O.program_sha256(dsl), sha)
                progs[sha] = dsl
                (fit if c["dataset_hash"] in fit_ds else test).add(sha)
        fit, test_only = sorted(fit), sorted(test - fit)
        fit_dsls = [progs[s] for s in fit]
        vocab, _ = T.build_vocab(fit_dsls)
        t1 = time.monotonic()
        sig = A.build_signatures(fit_dsls)
        t2 = time.monotonic()
        toks = {s: T.tokenize(progs[s], vocab) for s in progs}
        max_seq = T.ceil32(max(len(toks[s]) for s in fit))
        gdoc = A.build_grammar(sig, vocab, max_seq)
        g = A.Grammar(gdoc, sig, vocab)
        t3 = time.monotonic()
        # gates: POLICY accepts 100% of the half its tables were built from; SYNTAX accepts every tokenization
        # within max_seq; held-out POLICY acceptance is reported with its reasons
        res = {}
        policy_rejected = []
        for name, group in (("fit", fit), ("test_only", test_only)):
            for mode in (POL, SYN):
                st = Counter()
                reasons = Counter()
                for s in group:
                    ids = toks[s]
                    v = g.accepts(ids, mode)
                    st["programs"] += 1
                    st["accepted"] += v.ok
                    st["over_max_seq"] += len(ids) > max_seq
                    if not v.ok:
                        reasons[v.reason] += 1
                        if mode is POL and name == "test_only" and len(ids) <= max_seq:
                            policy_rejected.append(s)
                res[f"{name}/{mode.value}"] = {**dict(st), "reasons": dict(reasons.most_common(8))}
        t4 = time.monotonic()
        self.assertEqual(res["fit/policy"]["accepted"], len(fit))
        self.assertEqual(res["fit/syntax"]["accepted"], len(fit))
        self.assertEqual(res["test_only/syntax"]["accepted"],
                         len(test_only) - res["test_only/syntax"]["over_max_seq"])
        self.assertTrue(all(r.startswith("budget exceeded: max_seq") for r in res["test_only/syntax"]["reasons"]))
        pol_rate = res["test_only/policy"]["accepted"] / len(test_only)
        self.assertGreater(pol_rate, 0.95)
        # signature cross-check against the sha-pinned feat-math tables
        tables = source_op_tables()
        union = set().union(*tables.values())
        ops = set(sig["ops"])
        self.assertEqual(sorted(ops - union), [])
        only_v3 = sorted(o for o in ops if o not in tables["registry.rs"] | tables["spec.rs"])
        # compile_masks: POLICY over the train half, SYNTAX over the held-out programs POLICY refuses
        t5 = time.monotonic()
        seqs = [toks[s] for s in fit]
        table, ids = g.compile_masks(seqs, POL)
        t6 = time.monotonic()
        for k in random.Random(1).sample(range(len(seqs)), 25):
            s = g.start(POL)
            for i, t in enumerate(seqs[k]):
                self.assertTrue(np.array_equal(table[ids[k][i]], g.allowed(s)))
                s = g.advance(s, t)
        stable, sids = g.compile_masks([toks[s] for s in policy_rejected], SYN)
        positions = sum(len(x) for x in seqs)
        # liveness on corpus prefixes: allowed() equals the brute-force scan
        rng = random.Random(5)
        checked = mism = 0
        for s_ in rng.sample(fit, 25) + rng.sample(test_only, 25):
            for mode in (POL, SYN):
                st = g.start(mode)
                for t in toks[s_]:
                    a = g.allowed(st)
                    checked += 1
                    mism += not np.array_equal(a, brute_allowed(g, st))
                    if not a[t]:
                        break
                    st = g.advance(st, t)
        self.assertEqual(mism, 0)
        t7 = time.monotonic()
        # 2,000 masked walks per mode (class-balanced draws) plus uniform draws that press on the budgets
        walks = {}
        for mode in (POL, SYN):
            for bias, n in (("class", 2000), ("uniform", 200 if mode is POL else 100)):
                tw = time.monotonic()
                doc = A.walk_check(g, n, seed=20261005, mode=mode, bias=bias)
                self.assertTrue(doc["pass"], doc["failures"][:3])
                walks[f"{mode.value}/{bias}"] = {k: v for k, v in doc.items() if k not in ("failures", "mode", "bias")}
                walks[f"{mode.value}/{bias}"]["seconds"] = round(time.monotonic() - tw, 1)
        t8 = time.monotonic()
        # frontier (Bellman) consistency on corpus-grammar walk states, brute force over every token
        rng = random.Random(8)
        walk_states = []
        for mode in (POL, SYN):
            for bias in ("class", "uniform"):
                for _ in range(15):
                    st = g.start(mode)
                    for t in g.sample(rng, mode, bias=bias):
                        walk_states.append(st)
                        st = g.advance(st, t)
        rng.shuffle(walk_states)
        bellman = []
        for st in walk_states[:300]:
            bellman.extend(bellman_violations(g, st))
        self.assertEqual(bellman[:5], [])
        t9 = time.monotonic()
        tier_b = self._tier_b(g, vocab)
        report("corpus_sample", {
            "contexts": len(contexts), "rows": rows, "distinct_programs": len(progs),
            "datasets": {"fit": len(fit_ds), "test": len(datasets) - len(fit_ds)},
            "fit_programs": len(fit), "test_only_programs": len(test_only), "vocab_size": vocab.size,
            "max_seq": max_seq, "gates": res, "policy_heldout_acceptance": round(pol_rate, 5),
            "signatures": {"ops": len(sig["ops"]), "keysets": sum(len(v) for v in sig["keysets"].values()),
                           "value_tables": sum(len(v) for v in sig["values"].values()),
                           "trie_nodes": len(gdoc["policy"]["trie"]),
                           "value_profiles": len(gdoc["policy"]["value_profiles"]),
                           "sym_nodes_cap": gdoc["policy"]["sym"]["nodes_cap"],
                           "forms": sig["forms"], "portfolio": gdoc["policy"]["portfolio"]},
            "crosscheck": {"ops_outside_sources": sorted(ops - union), "ops_only_in_advanced_v3": only_v3,
                           "source_ops_never_seen": len(union - ops)},
            "compile_masks": {"policy_fit_positions": positions, "policy_table_rows": int(table.shape[0]),
                              "policy_seconds": round(t6 - t5, 1),
                              "positions_per_s": round(positions / max(t6 - t5, 1e-9)),
                              "syntax_rejected_programs": len(policy_rejected),
                              "syntax_table_rows": int(stable.shape[0])},
            "brute_force": {"states": checked, "mismatches": mism},
            "bellman": {"states": min(300, len(walk_states)), "violations": len(bellman), "seconds": round(t9 - t8, 1)},
            "walks": walks, "tier_b": tier_b, "cache": g.cache_stats(),
            "maxrss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
            "seconds": {"read_vocab": round(t1 - t0, 1), "signatures": round(t2 - t1, 1),
                        "tokenize_grammar": round(t3 - t2, 1), "accepts_2_modes": round(t4 - t3, 1),
                        "masks_and_bruteforce": round(t7 - t5, 1), "walks": round(t8 - t7, 1)}})

    def _tier_b(self, g, vocab) -> dict:
        """Tier B (the pinned engine, sha256-verified by oracle.PinnedEngine before the batch) on Tier A-repaired
        POLICY walks: every distinct canonical program comes back unchanged as historical_seed."""
        engine_path = O.DEFAULT_ENGINE
        if not engine_path.exists():
            return {"skipped": "pinned engine copy not present"}
        rng = random.Random(77)
        texts, repaired = [], 0
        for _ in range(400):
            text = T.detokenize(g.sample(rng, POL), vocab)
            canon = O.canonical(text)
            repaired += canon != text
            texts.append(canon)
        distinct = sorted(set(texts))
        engine = O.PinnedEngine(engine_path)
        sb = engine.seed_batch(distinct, feature_count=8, train_rows=0)
        self.assertEqual(engine.verifications, engine.invocations)
        self.assertEqual(len(sb.accepted), len(distinct), (sb.missing[:3], sb.changed and list(sb.changed)[:3]))
        return {"walks": 400, "tier_a_repaired": repaired, "distinct": len(distinct), "accepted_unchanged":
                len(sb.accepted), "changed": len(sb.changed), "missing": len(sb.missing), "fill_ins": sb.fill_ins,
                "engine_invocations": engine.invocations, "sha256_verifications": engine.verifications}


if __name__ == "__main__":
    unittest.main()
