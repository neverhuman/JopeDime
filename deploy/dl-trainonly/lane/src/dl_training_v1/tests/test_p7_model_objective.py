"""P7 tests 1a-1e and 2, plus the objective identity (PLAN §5 P7 tests 1-2; §6.3 labels; §6.3.1 PolicyLM; §6.3.2 loss
and optimization; §6.6 parameter identity and determinism controls). Builder: model-objective.

CPU only, deterministic mode (setUpModule), no file I/O and no network. Hand-built inputs pin every rule against a
float64 reference; the fixture tests use the shared ``tests.fixtures_p7`` corpus and the real sampler collation, as the
plan requires. Random draws come only from local seeded generators. Nothing here skips.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import sys
import types
import unittest

import numpy as np
import torch

from dltrain import common
from dltrain import labels as L
from dltrain import model as M
from dltrain import objective as O

V_HAND, T_HAND, W_HAND = 12, 9, 4


def setUpModule() -> None:
    M.configure_determinism(4)


def _fixtures():
    """The shared synthetic P7 fixtures (owned by the sampler builder), imported lazily so that only the tests which
    need them depend on them."""
    from tests import fixtures_p7
    return fixtures_p7


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


# ------------------------------------------------------------------------------------------- float64 references

def _log_softmax(xs: list[float]) -> list[float]:
    top = max(xs)
    lse = top + math.log(sum(math.exp(x - top) for x in xs))
    return [x - lse for x in xs]


def _softmax(xs: list[float]) -> list[float]:
    return [math.exp(v) for v in _log_softmax(xs)]


def _softplus(x: float) -> float:
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _huber(d: float, delta: float = 1.0) -> float:
    return 0.5 * d * d if abs(d) <= delta else delta * (abs(d) - 0.5 * delta)


def _smooth_l1(d: float, beta: float = 1.0) -> float:
    return 0.5 * d * d / beta if abs(d) < beta else abs(d) - 0.5 * beta


def _mean(xs: list[float]) -> float:
    return math.fsum(xs) / len(xs)


def _manual_nll(logits: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor, legal: torch.Tensor,
                smoothing: float = 0.0) -> np.ndarray:
    """Python loop: -log softmax over the legal set only, at positions i < length - 1 (0 elsewhere)."""
    x = logits.detach().double().numpy()
    lg, tk = legal.numpy(), tokens.numpy()
    out = np.zeros(x.shape[:2])
    for n, length in enumerate(lengths.tolist()):
        for i in range(length - 1):
            allowed = [v for v in range(x.shape[2]) if lg[n, i, v]]
            lsm = dict(zip(allowed, _log_softmax([float(x[n, i, v]) for v in allowed])))
            nll = -lsm[int(tk[n, i + 1])]
            if smoothing:
                nll = (1 - smoothing) * nll + smoothing * math.fsum(-lsm[v] for v in allowed) / len(allowed)
            out[n, i] = nll
    return out


def _manual_seq_ce(logits, tokens, lengths, legal, smoothing: float = 0.0) -> list[float]:
    nll = _manual_nll(logits, tokens, lengths, legal, smoothing)
    return [math.fsum(nll[n, :length - 1]) / (length - 1) for n, length in enumerate(lengths.tolist())]


# ---------------------------------------------------------------------------------------------- hand-built batches

def _tokens_and_masks(lengths: list[int], seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BOS-led toy sequences with random legal sets that always contain the next token; rows >= length-1 all False."""
    rng = np.random.default_rng(seed)
    tokens = np.zeros((len(lengths), T_HAND), dtype=np.int64)
    legal = np.zeros((len(lengths), T_HAND - 1, V_HAND), dtype=np.bool_)
    for n, length in enumerate(lengths):
        tokens[n, 0] = 1
        tokens[n, 1:length] = rng.integers(2, V_HAND, size=length - 1)
        for i in range(length - 1):
            legal[n, i] = rng.random(V_HAND) < 0.45
            legal[n, i, 0] = False
            legal[n, i, tokens[n, i + 1]] = True
    return torch.from_numpy(tokens), torch.tensor(lengths, dtype=torch.int64), torch.from_numpy(legal)


def _hand_batch(members: list[dict], contexts: list[tuple[bool, float]], mode: str, seed: int = 11):
    """A batch with the sampler.Batch field names, built from float64 label specs and stored as the sampler stores
    them (float32 label floats, int64 grids and classes)."""
    tokens, lengths, legal = _tokens_and_masks([m["length"] for m in members], seed)
    f32 = lambda key: torch.tensor([m[key] for m in members], dtype=torch.float32)  # noqa: E731
    flag = lambda key: torch.tensor([m[key] for m in members], dtype=torch.bool)  # noqa: E731
    g = torch.Generator().manual_seed(seed)
    return types.SimpleNamespace(
        mode=mode, step=1, tokens=tokens, lengths=lengths, legal=legal,
        ctx_z=torch.randn(len(contexts), W_HAND, generator=g),
        seq_ctx=torch.tensor([m["ctx"] for m in members], dtype=torch.int64),
        ctx_weight=torch.tensor([w for _, w in contexts], dtype=torch.float32),
        ctx_is_group=torch.tensor([grp for grp, _ in contexts], dtype=torch.bool),
        role=torch.tensor([m["role"] for m in members], dtype=torch.int64),
        q=f32("q_opt"), q_opt=f32("q_opt"), midrank_pct=f32("pct"), value=f32("value"), value_pct=f32("vpct"),
        q_grid=torch.tensor([m["q_grid"] for m in members], dtype=torch.int64),
        q_known=flag("known"), gate_pass=flag("known"), feas_valid=flag("fv"), rt_valid=flag("rv"),
        rt_censored=flag("cens"), feas_class=torch.tensor([m["feas"] for m in members], dtype=torch.int64),
        rt_target=f32("rt"), imitation_divisor=f32("div"),
        ctx_ids=[f"ctx-{i}" for i in range(len(contexts))], member_ids=[(f"ctx-{m['ctx']}", f"p{i}") for i, m in enumerate(members)],
        n_tokens=int(lengths.sum()) - len(members))


def _member(ctx, role, *, q=None, grid=None, pct=None, feas=None, fv=False, rt=None, cens=False, div=1.0,
            value=0.0, vpct=0.0, length=6) -> dict:
    known = q is not None
    return {"ctx": ctx, "role": role, "q_opt": q if known else 0.0, "q_grid": grid if known else 0,
            "pct": pct if known else 0.0, "known": known, "feas": -1 if feas is None else feas, "fv": fv,
            "rt": 0.0 if rt is None else rt, "cens": cens, "rv": rt is not None, "div": div, "value": value,
            "vpct": vpct, "length": length}


def _successor_spec() -> tuple[list[dict], list[tuple[bool, float]]]:
    """Four contexts: two full groups (one with a q-grid tie, one with two clipped teachers of distinct grid, i.e. a
    zero-weight pair), a feasibility-stream context at weight 0.1, and a group with only 2 known members whose
    percentile is exactly 0.9 (boosted, although float32(0.9) < 0.9)."""
    qc = L.Q_CLIP
    m = [
        _member(0, 0, q=1.30, grid=1_300_000, pct=1.0, feas=0, fv=True, rt=2.0, length=9),
        _member(0, 0, q=1.10, grid=1_100_000, pct=6 / 7, feas=0, fv=True, rt=1.5, div=3.0, length=7),
        _member(0, 0, q=0.90, grid=900_000, pct=5 / 7, feas=0, fv=False, length=5),
        _member(0, 0, q=0.70, grid=700_000, pct=4 / 7, feas=0, fv=True, rt=0.5, length=8),
        _member(0, 1, q=0.20, grid=200_000, pct=2.5 / 7, feas=0, fv=True, rt=3.0, length=4),
        _member(0, 1, q=0.20, grid=200_000, pct=2.5 / 7, feas=0, fv=True, rt=2.5, length=6),
        _member(0, 2, feas=1, fv=True, length=3),
        _member(0, 2, feas=2, fv=True, rt=math.log1p(90.0), cens=True, length=9),
        _member(1, 0, q=qc, grid=13_000_000, pct=1.0, feas=0, fv=True, rt=1.0, length=6),
        _member(1, 0, q=qc, grid=9_000_000, pct=0.8, feas=0, fv=False, length=7),
        _member(1, 0, q=0.5, grid=500_000, pct=0.6, feas=0, fv=True, rt=0.2, length=5),
        _member(1, 0, q=-0.3, grid=-300_000, pct=0.4, feas=0, fv=True, rt=4.0, length=9),
        _member(1, 1, q=-1.0, grid=-1_000_000, pct=0.2, feas=0, fv=True, rt=0.9, length=4),
        _member(1, 1, q=-2.5, grid=-2_500_000, pct=0.0, feas=0, fv=True, rt=1.2, length=6),
        _member(2, 3, q=0.4, grid=400_000, pct=0.5, feas=0, fv=True, rt=1.1, length=5),
        _member(2, 3, feas=1, fv=True, length=4),
        _member(2, 3, feas=2, fv=False, length=9),
        _member(2, 3, feas=0, fv=True, rt=0.7, length=6),
        _member(3, 0, q=0.3, grid=300_000, pct=0.9, feas=0, fv=True, rt=0.3, length=7),
        _member(3, 0, q=0.1, grid=100_000, pct=0.0, feas=0, fv=True, rt=0.6, length=8),
        _member(3, 2, feas=1, fv=False, length=5),
    ]
    return m, [(True, 1.0), (True, 1.0), (False, 0.1), (True, 1.0)]


def _baseline_spec() -> tuple[list[dict], list[tuple[bool, float]]]:
    """Champion + 7 rows in two contexts (context 0 holds an exact value tie and a 1e-6 grid tie) plus a 2-row
    context (pairs but no listwise group)."""
    groups = [[0.81, 0.62, 0.62, 0.55, 0.4000001, 0.4000003, 0.31, 0.12],
              [0.9, 0.7, 0.66, 0.5, 0.45, 0.3, 0.2, 0.05],
              [0.6, 0.3]]
    members = []
    for c, values in enumerate(groups):
        pcts = L.midrank_pcts([L.q_grid(v) for v in values])
        for i, (value, vpct) in enumerate(zip(values, pcts)):
            rt = None if (c + i) % 3 == 2 else 0.25 + 0.5 * i
            members.append(_member(c, 0 if i == 0 else 1, q=1.0 - 0.1 * i, grid=L.q_grid(1.0 - 0.1 * i), pct=vpct,
                                   feas=0, fv=rt is not None, rt=rt, value=value, vpct=vpct, length=4 + (i + c) % 6))
    return members, [(True, 1.0)] * len(groups)


def _fake_out(n: int, seed: int, cf: bool = False) -> M.PolicyOutputs:
    """Leaf tensors standing in for PolicyLM outputs (the objective never needs the model)."""
    g = torch.Generator().manual_seed(seed)

    def leaf(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).requires_grad_()

    return M.PolicyOutputs(logits=leaf(n, T_HAND - 1, V_HAND, scale=2.0), q_hat=leaf(n), feas_logits=leaf(n, 3),
                           rt_hat=leaf(n, scale=1.5), cf_hat=leaf(n) if cf else None, h_eos=leaf(n, 4))


def _manual_successor(members, contexts, batch, out) -> dict[str, float]:
    """PLAN §6.3.2 successor-v1 terms by explicit per-context loops over the float64 label specs."""
    ce = _manual_seq_ce(out.logits, batch.tokens, batch.lengths, batch.legal)
    q_hat = out.q_hat.detach().double().tolist()
    rt_hat = out.rt_hat.detach().double().tolist()
    feas_logits = out.feas_logits.detach().double().tolist()
    by_ctx = [[n for n, m in enumerate(members) if m["ctx"] == c] for c in range(len(contexts))]
    group = [c for c, (is_group, _) in enumerate(contexts) if is_group]
    terms: dict[str, float] = {}
    per = []
    for c in group:
        teachers = [n for n in by_ctx[c] if members[n]["role"] == 0]
        if teachers:
            soft = _softmax([members[n]["q_opt"] / O.TEACHER_TAU for n in teachers])
            per.append(math.fsum(s / members[n]["div"] * ce[n] for s, n in zip(soft, teachers)))
    terms["ce"] = _mean(per) if per else 0.0
    per = []
    for c in group:
        known = [n for n in by_ctx[c] if members[n]["known"]]
        num, den = [], []
        for i in known:
            for k in known:
                if members[i]["q_grid"] > members[k]["q_grid"]:
                    boost = 2.0 if max(members[i]["pct"], members[k]["pct"]) >= 0.9 else 1.0
                    w = min(abs(members[i]["q_opt"] - members[k]["q_opt"]), 1.0) * boost
                    num.append(w * _softplus(q_hat[k] - q_hat[i]))
                    den.append(w)
        if math.fsum(den) > 0:
            per.append(math.fsum(num) / math.fsum(den))
    terms["bt"] = _mean(per) if per else 0.0
    per = []
    for c in group:
        known = [n for n in by_ctx[c] if members[n]["known"]]
        if len(known) >= 3:
            target = _softmax([members[n]["pct"] / O.LIST_TAU for n in known])
            logq = _log_softmax([q_hat[n] for n in known])
            per.append(-math.fsum(t * lq for t, lq in zip(target, logq)))
    terms["list"] = _mean(per) if per else 0.0
    per = []
    for c in group:
        known = [n for n in by_ctx[c] if members[n]["known"]]
        if known:
            per.append(_mean([_huber(q_hat[n] - members[n]["q_opt"]) for n in known]))
    terms["q"] = _mean(per) if per else 0.0

    def weighted(flag: str, loss) -> float:
        """sum_c w_c * mean_c over the contexts with a flagged member, divided by their number (PLAN §6.1: every
        drawn context has equal loss weight; a feasibility-stream context is 0.1 of one)."""
        num = []
        for c, (_, weight) in enumerate(contexts):
            rows = [n for n in by_ctx[c] if members[n][flag]]
            if rows:
                num.append(weight * _mean([loss(n) for n in rows]))
        return math.fsum(num) / len(num) if num else 0.0

    terms["feas"] = weighted("fv", lambda n: -_log_softmax(feas_logits[n])[members[n]["feas"]])
    terms["rt"] = weighted("rv", lambda n: max(0.0, members[n]["rt"] - rt_hat[n]) ** 2 if members[n]["cens"]
                           else _huber(rt_hat[n] - members[n]["rt"]))
    return terms


def _manual_baseline(members, batch, out) -> tuple[dict[str, float], int, int]:
    """The lineage-B composite of PLAN §6.3.2 by explicit loops over the float64 label specs."""
    nll = _manual_seq_ce(out.logits, batch.tokens, batch.lengths, batch.legal, smoothing=0.05)
    weights = [0.05 + m["vpct"] ** 4 for m in members]
    terms = {"seq_nll": math.fsum(w * x for w, x in zip(weights, nll)) / math.fsum(weights)}
    pairs, tied = [], 0
    for i in range(len(members)):
        for k in range(i + 1, len(members)):
            a, b = members[i], members[k]
            if a["ctx"] != b["ctx"]:
                continue
            ga, gb = round(a["value"] * 1e6), round(b["value"] * 1e6)
            if ga == gb:
                tied += 1
                continue
            w = abs(a["value"] - b["value"]) * (2.0 if max(a["vpct"], b["vpct"]) >= 0.9 else 1.0)
            pairs.append((i, k, w) if ga > gb else (k, i, w))
    scale = _mean([w for _, _, w in pairs])
    for name, score in (("bt", out.q_hat), ("cf_bt", out.cf_hat)):
        s = score.detach().double().tolist()
        terms[name] = math.fsum(w / scale * _softplus(s[lo] - s[wi]) for wi, lo, w in pairs) / len(pairs)
    q_hat = out.q_hat.detach().double().tolist()
    per = []
    for c in sorted({m["ctx"] for m in members}):
        rows = [n for n, m in enumerate(members) if m["ctx"] == c]
        if len(rows) >= 3:
            target = _softmax([members[n]["vpct"] / O.LIST_TAU for n in rows])
            logq = _log_softmax([q_hat[n] for n in rows])
            per.append(-math.fsum(t * lq for t, lq in zip(target, logq)))
    terms["list"] = _mean(per)
    rt_hat = out.rt_hat.detach().double().tolist()
    terms["rt"] = _mean([_smooth_l1(rt_hat[n] - m["rt"]) for n, m in enumerate(members) if m["rv"]])
    return terms, len(pairs), tied


def _expected_parameter_count(cfg: M.PolicyConfig) -> int:
    d, ff = cfg.d_model, cfg.d_ff
    emb = cfg.vocab_size * d + (cfg.max_seq + 1) * d
    ctx = (cfg.ctx_dim * d + d) + 2 * d + (d * d + d) + 2 * d
    block = 2 * d + (3 * d * d + 3 * d) + (d * d + d) + 2 * d + (d * ff + ff) + (ff * d + d)

    def head(d_in: int, d_out: int) -> int:
        return (d_in * d + d) + (d * d_out + d_out)

    heads = head(2 * d, 1) + head(2 * d, 3) + head(2 * d, 1) + (head(d, 1) if cfg.context_free_head else 0)
    return emb + ctx + cfg.n_layers * block + 2 * d + heads


def _small_config(**kw) -> M.PolicyConfig:
    base = {"vocab_size": V_HAND, "ctx_dim": W_HAND, "d_model": 16, "n_layers": 2, "n_heads": 2, "d_ff": 32,
            "max_seq": T_HAND, "dropout": 0.0}
    return M.PolicyConfig(**(base | kw))


def _seeded_model(cfg: M.PolicyConfig, seed: int) -> M.PolicyLM:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return M.PolicyLM(cfg)


# =================================================================================================== P7 test 1a/1b

class TestP7MaskedCrossEntropy(unittest.TestCase):
    """P7 test 1a/1b: masked CE equals a float64 manual loop; illegal logits contribute nothing."""

    def _fixture_case(self):
        fx = _fixtures()
        rows = sorted(fx.records(), key=lambda r: (r["length"], r["program_sha256"]))
        picks = rows[:4] + rows[len(rows) // 2:len(rows) // 2 + 4] + rows[-4:]
        toks = [list(r["tokens"]) for r in picks]
        vocab = len(fx.vocab())
        t = max(len(x) for x in toks)
        tokens = torch.zeros(len(toks), t, dtype=torch.int64)
        legal = torch.zeros(len(toks), t - 1, vocab, dtype=torch.bool)
        for n, x in enumerate(toks):
            tokens[n, :len(x)] = torch.tensor(x)
            legal[n, :len(x) - 1] = torch.from_numpy(np.asarray(fx.legal_mask(x), dtype=np.bool_))
        lengths = torch.tensor([len(x) for x in toks], dtype=torch.int64)
        self.assertGreater(len(set(lengths.tolist())), 1, "fixture case needs varied lengths")
        logits = torch.randn(len(toks), t - 1, vocab, generator=torch.Generator().manual_seed(20261005))
        return logits, tokens, lengths, legal

    def _hand_case(self):
        tokens, lengths, legal = _tokens_and_masks([9, 4, 2, 7, 6, 9, 3], seed=5)
        logits = torch.randn(7, T_HAND - 1, V_HAND, generator=torch.Generator().manual_seed(77))
        return logits, tokens, lengths, legal

    def _check_against_manual(self, logits, tokens, lengths, legal):
        valid = O.valid_positions(lengths, tokens.shape[1])
        self.assertEqual(valid.sum().item(), int((lengths - 1).sum()))
        for smoothing in (0.0, 0.05):
            nll = O.masked_token_nll(logits, tokens[:, 1:], legal, valid, label_smoothing=smoothing)
            ref = _manual_nll(logits, tokens, lengths, legal, smoothing)
            # atol 1e-6 is a float32 statement only for moderate values (ulp(8) ~ 1e-6), hence unit-scale logits.
            self.assertLess(float(ref.max()), 8.0)
            np.testing.assert_allclose(nll.double().numpy(), ref, rtol=0, atol=1e-6)
            seq = O.sequence_ce(logits, tokens[:, 1:], legal, valid, label_smoothing=smoothing)
            np.testing.assert_allclose(seq.double().numpy(), _manual_seq_ce(logits, tokens, lengths, legal, smoothing),
                                       rtol=0, atol=1e-6)
            self.assertEqual(nll.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(nll).all()))
            self.assertTrue(bool((nll[~valid] == 0).all()))

    def test_1a_matches_float64_manual_loop_on_fixture_masks(self):
        self._check_against_manual(*self._fixture_case())

    def test_1a_matches_float64_manual_loop_on_random_masks(self):
        self._check_against_manual(*self._hand_case())

    def _check_illegal_inert(self, logits, tokens, lengths, legal):
        valid = O.valid_positions(lengths, tokens.shape[1])
        g = torch.Generator().manual_seed(3)
        signs = torch.where(torch.rand(logits.shape, generator=g) < 0.5, -1e4, 1e4)
        poisoned = torch.where(legal, logits, signs)
        self.assertGreater(int((~legal).sum()), 0)
        for smoothing in (0.0, 0.05):
            base = O.masked_token_nll(logits, tokens[:, 1:], legal, valid, smoothing)
            other = O.masked_token_nll(poisoned, tokens[:, 1:], legal, valid, smoothing)
            self.assertTrue(torch.equal(base, other), "illegal logits changed the NLL")
            leaf = poisoned.clone().requires_grad_()
            O.sequence_ce(leaf, tokens[:, 1:], legal, valid, smoothing).sum().backward()
            self.assertEqual(int(torch.count_nonzero(leaf.grad[~legal])), 0, "illegal logits received gradient")
            self.assertGreater(int(torch.count_nonzero(leaf.grad[legal])), 0)
            self.assertTrue(bool(torch.isfinite(leaf.grad).all()))

    def test_1b_illegal_logits_contribute_nothing_on_fixture_masks(self):
        self._check_illegal_inert(*self._fixture_case())

    def test_1b_illegal_logits_contribute_nothing_on_random_masks(self):
        self._check_illegal_inert(*self._hand_case())

    def test_1b_illegal_target_or_empty_legal_row_raises(self):
        logits, tokens, lengths, legal = self._hand_case()
        targets, valid = tokens[:, 1:], O.valid_positions(lengths, tokens.shape[1])
        bad = legal.clone()
        bad[0, 2, tokens[0, 3]] = False
        with self.assertRaisesRegex(ValueError, "illegal"):
            O.masked_token_nll(logits, targets, bad, valid)
        empty = legal.clone()
        empty[3, 1, :] = False
        with self.assertRaisesRegex(ValueError, "empty legal row"):
            O.sequence_ce(logits, targets, empty, valid)
        out_of_range = targets.clone()
        out_of_range[1, 0] = V_HAND
        with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
            O.masked_token_nll(logits, out_of_range, legal, valid)
        # An empty legal row at an unsupervised (padding) position is fine.
        padded = legal.clone()
        padded[2, 5, :] = False
        self.assertFalse(bool(valid[2, 5]))
        O.masked_token_nll(logits, targets, padded, valid)
        with self.assertRaises(ValueError):
            O.sequence_ce(logits, targets, legal, torch.zeros_like(valid))


# ======================================================================================================== P7 test 1d

class TestP7PolicyLM(unittest.TestCase):
    """P7 test 1d: shapes, tying, causality, parameter_sha256, parameter groups and the S size."""

    def _inputs(self, n=3, t=T_HAND, seed=4):
        g = torch.Generator().manual_seed(seed)
        tokens = torch.randint(1, V_HAND, (n, t), generator=g)
        lengths = torch.tensor([t, max(2, t - 3), 2][:n] + [t] * max(0, n - 3), dtype=torch.int64)
        ctx_z = torch.randn(2, W_HAND, generator=g)
        seq_ctx = torch.tensor([i % 2 for i in range(n)], dtype=torch.int64)
        return tokens, lengths, ctx_z, seq_ctx

    def test_1d_shapes_and_heads(self):
        for cf in (False, True):
            cfg = _small_config(context_free_head=cf)
            model = _seeded_model(cfg, 1).eval()
            tokens, lengths, ctx_z, seq_ctx = self._inputs()
            out = model(tokens, lengths, ctx_z, seq_ctx)
            n, t = tokens.shape
            self.assertEqual(tuple(out.logits.shape), (n, t - 1, V_HAND))
            self.assertEqual(tuple(out.q_hat.shape), (n,))
            self.assertEqual(tuple(out.feas_logits.shape), (n, 3))
            self.assertEqual(tuple(out.rt_hat.shape), (n,))
            self.assertEqual(tuple(out.h_eos.shape), (n, cfg.d_model))
            self.assertEqual(out.logits.dtype, torch.float32)
            if cf:
                self.assertEqual(tuple(out.cf_hat.shape), (n,))
                self.assertIn("cf_head.0.weight", M.parameter_names(model))
            else:
                self.assertIsNone(out.cf_hat)
                self.assertFalse(any(name.startswith("cf_head") for name in M.parameter_names(model)))

    def test_1d_tied_head_and_parameter_names(self):
        model = _seeded_model(_small_config(), 2)
        self.assertIs(model.lm_head.weight, model.tok_emb.weight)
        M.assert_tied(model)
        names = M.parameter_names(model)
        self.assertNotIn("lm_head.weight", names)
        self.assertIn("tok_emb.weight", names)
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(M.count_parameters(model), sum(p.numel() for _, p in model.named_parameters()))
        self.assertEqual(M.count_parameters(model), _expected_parameter_count(model.cfg))
        untied = _seeded_model(_small_config(), 2)
        untied.lm_head.weight = torch.nn.Parameter(untied.tok_emb.weight.detach().clone())
        with self.assertRaises(AssertionError):
            M.assert_tied(untied)

    def test_1d_causal_in_eval_mode(self):
        model = _seeded_model(_small_config(dropout=0.1), 3).eval()
        tokens, lengths, ctx_z, seq_ctx = self._inputs(n=2, seed=8)
        lengths = torch.tensor([T_HAND, T_HAND], dtype=torch.int64)
        base = model(tokens, lengths, ctx_z, seq_ctx)
        for j in (1, 4, T_HAND - 1):
            changed = tokens.clone()
            changed[:, j] = torch.where(changed[:, j] == 5, 6, 5)
            out = model(changed, lengths, ctx_z, seq_ctx)
            self.assertTrue(torch.equal(out.logits[:, :j], base.logits[:, :j]), f"logits before {j} changed")
            if j < T_HAND - 1:
                self.assertFalse(torch.equal(out.logits[:, j:], base.logits[:, j:]))
            self.assertFalse(torch.equal(out.q_hat, base.q_hat), "the EOS heads must see every token")

    def test_1d_dropout_is_the_only_rng_consumer(self):
        model = _seeded_model(_small_config(dropout=0.2), 4)
        inputs = self._inputs()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(99)
            state = torch.get_rng_state()
            model.eval()
            first = model(*inputs)
            self.assertTrue(torch.equal(torch.get_rng_state(), state), "eval forward consumed the torch RNG")
            self.assertTrue(torch.equal(model(*inputs).logits, first.logits))
            model.train()
            a = model(*inputs).logits
            self.assertFalse(torch.equal(torch.get_rng_state(), state))
            b = model(*inputs).logits
            self.assertFalse(torch.equal(a, b))
        no_dropout = _seeded_model(_small_config(dropout=0.0), 4).train()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            state = torch.get_rng_state()
            no_dropout(*inputs)
            self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_1d_forward_validates_inputs(self):
        model = _seeded_model(_small_config(), 5)
        tokens, lengths, ctx_z, seq_ctx = self._inputs()
        with self.assertRaises(ValueError):
            model(torch.ones(3, T_HAND + 1, dtype=torch.int64), lengths, ctx_z, seq_ctx)
        with self.assertRaises(ValueError):
            model(tokens, torch.tensor([T_HAND + 1, 3, 2]), ctx_z, seq_ctx)
        with self.assertRaises(ValueError):
            model(tokens, torch.tensor([1, 3, 2]), ctx_z, seq_ctx)
        with self.assertRaises(ValueError):
            model(tokens, lengths, ctx_z, torch.tensor([0, 2, 1]))
        with self.assertRaises(ValueError):
            model(tokens, lengths, torch.randn(2, W_HAND + 1), seq_ctx)
        with self.assertRaises(ValueError):
            model(tokens.clone().fill_(V_HAND), lengths, ctx_z, seq_ctx)

    def test_1d_parameter_sha256_manual_recomputation(self):
        model = _seeded_model(_small_config(), 6)
        h = hashlib.sha256()
        inventory = []
        for name, p in sorted(model.named_parameters()):
            header = f"{name}\0F32\0{','.join(str(s) for s in p.shape)}\n"
            h.update(header.encode())
            h.update(p.detach().numpy().tobytes())
            inventory.append(header)
        self.assertEqual(M.parameter_sha256(model), h.hexdigest())
        self.assertEqual(M.tensor_inventory_sha256(dict(model.named_parameters())),
                         hashlib.sha256("".join(inventory).encode()).hexdigest())
        self.assertEqual(M.INVENTORY_ALGORITHM, "sha256-name-nul-dtype-nul-shape-csv-lf-v1")
        named = dict(model.named_parameters())
        self.assertEqual(M.parameter_sha256_tensors(named), M.parameter_sha256(model))
        scalar = {"a": torch.zeros(()), "b": torch.ones(2, 3)}
        expected = hashlib.sha256(b"a\0F32\0\n" + scalar["a"].numpy().tobytes() + b"b\0F32\0" + b"2,3\n"
                                  + scalar["b"].numpy().tobytes()).hexdigest()
        self.assertEqual(M.parameter_sha256_tensors(scalar), expected)
        self.assertEqual(M.tensor_inventory_sha256({"x": torch.zeros(4, dtype=torch.int64), "s": torch.zeros((), dtype=torch.uint8)}),
                         hashlib.sha256(b"s\0U8\0\nx\0I64\x004\n").hexdigest())
        with self.assertRaises(ValueError):
            M.parameter_sha256_tensors({"x": torch.zeros(2, dtype=torch.float64)})
        with self.assertRaises(ValueError):
            M.tensor_inventory_sha256({"x": torch.zeros(2, dtype=torch.complex64)})
        # One optimizer update changes it; tying survives the update.
        before = M.parameter_sha256(model)
        opt = torch.optim.AdamW(M.param_groups(model, 0.01), lr=1e-3, fused=False, foreach=False)
        tokens, lengths, ctx_z, seq_ctx = self._inputs()
        out = model(tokens, lengths, ctx_z, seq_ctx)
        (out.logits.square().mean() + out.q_hat.sum()).backward()
        opt.step()
        self.assertNotEqual(M.parameter_sha256(model), before)
        M.assert_tied(model)

    def test_1d_param_groups(self):
        model = _seeded_model(_small_config(context_free_head=True), 7)
        groups = M.param_groups(model, 0.01)
        self.assertEqual([g["group_name"] for g in groups], ["decay", "no_decay"])
        self.assertEqual([g["weight_decay"] for g in groups], [0.01, 0.0])
        order = [name for name, _ in model.named_parameters()]
        norms = {f"{mn}.{pn}" for mn, mod in model.named_modules() if isinstance(mod, torch.nn.LayerNorm)
                 for pn, _ in mod.named_parameters(recurse=False)}
        expect_no_decay = [n for n in order if n.endswith(".bias") or n in norms]
        expect_decay = [n for n in order if n not in expect_no_decay]
        self.assertEqual(groups[0]["param_names"], expect_decay)
        self.assertEqual(groups[1]["param_names"], expect_no_decay)
        params = dict(model.named_parameters())
        for g in groups:
            self.assertEqual(len(g["params"]), len(g["param_names"]))
            for p, name in zip(g["params"], g["param_names"]):
                self.assertIs(p, params[name])
        for name in ("tok_emb.weight", "pos_emb.weight", "ctx_mlp.0.weight", "blocks.0.attn.qkv.weight", "q_head.2.weight"):
            self.assertIn(name, groups[0]["param_names"])
        for name in ("ln_f.weight", "ln_f.bias", "ctx_mlp.2.weight", "ctx_mlp.5.weight", "blocks.1.ln1.weight",
                     "blocks.1.ln2.weight", "blocks.0.attn.qkv.bias", "cf_head.2.bias"):
            self.assertIn(name, groups[1]["param_names"])
        torch.optim.AdamW(groups, lr=1e-3, fused=False, foreach=False)
        with self.assertRaises(ValueError):
            M.param_groups(model, -0.1)

    def test_1d_size_s_builds_and_parameter_count(self):
        cfg = M.size_config("S", vocab_size=32, ctx_dim=8, max_seq=24)
        self.assertEqual((cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff), (256, 4, 4, 1024))
        m_cfg = M.size_config("M", vocab_size=32, ctx_dim=8, max_seq=24)
        self.assertEqual((m_cfg.d_model, m_cfg.n_layers, m_cfg.n_heads, m_cfg.d_ff), (512, 6, 8, 2048))
        model = _seeded_model(cfg, 8)
        count = M.count_parameters(model)
        self.assertEqual(count, _expected_parameter_count(cfg))
        sys.stderr.write(f"\n[P7 1d] size_config('S', V=32, W=8, max_seq=24): count_parameters={count}\n")
        out = model.eval()(torch.ones(2, 24, dtype=torch.int64), torch.tensor([24, 17]), torch.zeros(1, 8),
                           torch.zeros(2, dtype=torch.int64))
        self.assertEqual(tuple(out.logits.shape), (2, 23, 32))
        with self.assertRaises(ValueError):
            M.size_config("L", vocab_size=32, ctx_dim=8, max_seq=24)

    def test_1d_config_identity(self):
        cfg = _small_config(dropout=0, context_free_head=True)
        self.assertIsInstance(cfg.dropout, float)
        doc = cfg.to_json()
        self.assertEqual(M.PolicyConfig.from_json(doc), cfg)
        self.assertEqual(cfg.sha256(), common.sha256_bytes(common.canonical(doc)))
        self.assertEqual(cfg.sha256(), M.PolicyConfig.from_json(doc).sha256())
        self.assertNotEqual(cfg.sha256(), _small_config(dropout=0.0).sha256())
        with self.assertRaises(ValueError):
            M.PolicyConfig.from_json(doc | {"surprise": 1})
        with self.assertRaises(ValueError):
            M.PolicyConfig.from_json({k: v for k, v in doc.items() if k != "d_model"})
        for bad in ({"n_heads": 3}, {"dropout": 1.0}, {"max_seq": 1}, {"pad_id": V_HAND}, {"d_model": 16.0}):
            with self.assertRaises(ValueError):
                _small_config(**bad)

    def test_1d_configure_determinism_is_idempotent(self):
        first = M.configure_determinism(4)
        second = M.configure_determinism(4)
        self.assertEqual(first, second)
        self.assertEqual(first, {"use_deterministic_algorithms": True, "num_threads": 4, "sdpa_backend": "math",
                                 "cudnn_benchmark": False, "cudnn_deterministic": True, "allow_tf32_matmul": False,
                                 "allow_tf32_cudnn": False, "float32_matmul_precision": "highest",
                                 "ce_flatten": "(N*T,V)", "device": "cpu"})
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(torch.get_num_threads(), 4)
        with self.assertRaises(ValueError):
            M.configure_determinism(0)


# ======================================================================================================== P7 test 1c

class TestP7SuccessorObjective(unittest.TestCase):
    """P7 test 1c: successor-v1 terms against a float64 reference; pairs equal labels.ranking_pairs; connected zeros."""

    def test_1c_every_term_matches_the_float64_reference(self):
        members, contexts = _successor_spec()
        batch = _hand_batch(members, contexts, O.SUCCESSOR_V1)
        out = _fake_out(len(members), seed=21)
        res = O.successor_v1(out, batch)
        ref = _manual_successor(members, contexts, batch, out)
        self.assertEqual(list(res.terms), list(O.COEF_SUCCESSOR))
        for name, value in ref.items():
            self.assertAlmostEqual(res.terms[name], value, delta=2e-6 * max(1.0, abs(value)), msg=name)
            self.assertEqual(res.weighted[name], O.COEF_SUCCESSOR[name] * res.terms[name])
        total = math.fsum(O.COEF_SUCCESSOR[k] * v for k, v in ref.items())
        self.assertAlmostEqual(res.loss.item(), total, delta=5e-6 * max(1.0, abs(total)))
        self.assertEqual(res.support, {"teachers": 10, "group_contexts": 3, "pairs": 29, "list_groups": 2,
                                       "q_members": 14, "feas_members": 17, "rt_members": 15})
        self.assertEqual(res.loss.dim(), 0)
        self.assertEqual(res.loss.dtype, torch.float32)
        res.loss.backward()
        illegal = ~batch.legal
        self.assertEqual(int(torch.count_nonzero(out.logits.grad[illegal])), 0)
        non_teacher = (batch.role != 0) | ~batch.ctx_is_group[batch.seq_ctx]
        self.assertEqual(int(torch.count_nonzero(out.logits.grad[non_teacher])), 0, "L_CE reached non-teachers")
        for t in (out.q_hat, out.feas_logits, out.rt_hat):
            self.assertTrue(bool(torch.isfinite(t.grad).all()))
        self.assertEqual(float(out.rt_hat.grad[2]), 0.0, "a remote member entered L_rt")

    def test_1c_teacher_weights(self):
        members, contexts = _successor_spec()
        batch = _hand_batch(members, contexts, O.SUCCESSOR_V1)
        w = O.teacher_weights(batch)
        self.assertEqual(w.dtype, torch.float64)
        for c, (is_group, _) in enumerate(contexts):
            rows = [n for n, m in enumerate(members) if m["ctx"] == c]
            teachers = [n for n in rows if members[n]["role"] == 0 and is_group]
            if teachers:
                soft = _softmax([members[n]["q_opt"] / 0.25 for n in teachers])
                for s, n in zip(soft, teachers):
                    self.assertAlmostEqual(float(w[n]), s / members[n]["div"], delta=1e-6)
            for n in set(rows) - set(teachers):
                self.assertEqual(float(w[n]), 0.0)
        bad = _hand_batch(members, contexts, O.SUCCESSOR_V1)
        bad.imitation_divisor[0] = 0.5
        with self.assertRaisesRegex(ValueError, "imitation_divisor"):
            O.successor_v1(_fake_out(len(members), 1), bad)

    def test_1c_dense_pairs_equal_ranking_pairs(self):
        members, contexts = _successor_spec()
        batch = _hand_batch(members, contexts, O.SUCCESSOR_V1)
        pair, weight = O.bt_pair_matrix(batch)
        found = {(i, k): float(weight[i, k]) for i, k in pair.nonzero().tolist()}
        expected = {}
        for c, (is_group, _) in enumerate(contexts):
            rows = [n for n, m in enumerate(members) if m["ctx"] == c]
            if not is_group:
                continue
            labs = [_record_label(members[n]) for n in rows]
            for wi, lo, w in L.ranking_pairs(labs):
                expected[(rows[wi], rows[lo])] = w
        self.assertEqual(set(found), set(expected))
        for key, w in expected.items():
            self.assertAlmostEqual(found[key], w, delta=1e-6, msg=str(key))
        self.assertNotIn((4, 5), found)
        self.assertNotIn((5, 4), found)
        self.assertEqual(found[(8, 9)], 0.0, "two clipped teachers of distinct grid form a zero-weight pair")
        self.assertAlmostEqual(found[(18, 19)], 0.2 * O.BOOST_FACTOR, delta=1e-6, msg="pct 0.9 must be boosted")

    def test_1c_empty_support_terms_are_connected_zeros(self):
        members, contexts = _successor_spec()
        feas_only = [dict(m, ctx=0, role=3) for m in members if m["ctx"] == 2]
        batch = _hand_batch(feas_only, [(False, 0.1)], O.SUCCESSOR_V1)
        out = _fake_out(len(feas_only), seed=3)
        res = O.successor_v1(out, batch)
        for name in ("ce", "bt", "list", "q"):
            self.assertEqual(res.terms[name], 0.0, name)
        for name in ("teachers", "group_contexts", "pairs", "list_groups", "q_members"):
            self.assertEqual(res.support[name], 0, name)
        self.assertGreater(res.support["feas_members"], 0)
        self.assertTrue(res.loss.requires_grad)
        res.loss.backward()
        self.assertIsNotNone(out.q_hat.grad, "connected zeros must keep q_hat in the graph")
        self.assertEqual(int(torch.count_nonzero(out.q_hat.grad)), 0)
        self.assertTrue(out.logits.grad is None or int(torch.count_nonzero(out.logits.grad)) == 0)
        no_rt = _hand_batch([dict(m, rt=0.0, rv=False, fv=False) for m in feas_only], [(False, 0.1)], O.SUCCESSOR_V1)
        res = O.successor_v1(_fake_out(len(feas_only), seed=4), no_rt)
        self.assertEqual((res.terms["feas"], res.terms["rt"]), (0.0, 0.0))
        self.assertEqual((res.support["feas_members"], res.support["rt_members"]), (0, 0))
        self.assertEqual(res.loss.item(), 0.0)
        self.assertTrue(res.loss.requires_grad)

    def test_1c_feasibility_stream_weight_is_a_fraction_of_a_context(self):
        """PLAN §6.1: L_feas and L_rt count a feasibility-stream context as its declared 0.1 of a context. The weight
        does not cancel on an all-feasibility batch, and a group context's share does not depend on the weights of
        the other contexts in its batch."""
        members, contexts = _successor_spec()
        feas_rows = [m for m in members if m["ctx"] == 2]
        feas_only = [dict(m, ctx=i % 2, role=3) for i, m in enumerate(feas_rows)]
        out = _fake_out(len(feas_only), seed=17)
        res = {w: O.successor_v1(out, _hand_batch(feas_only, [(False, w), (False, w)], O.SUCCESSOR_V1))
               for w in (0.1, 1.0)}
        for name in ("feas", "rt"):
            self.assertEqual(res[0.1].support[f"{name}_members"], res[1.0].support[f"{name}_members"])
            self.assertGreater(res[1.0].terms[name], 0.0, name)
            self.assertAlmostEqual(res[0.1].terms[name], 0.1 * res[1.0].terms[name],
                                   delta=1e-6 * res[1.0].terms[name], msg=f"{name}: the 0.1 weight cancelled")

        out = _fake_out(len(members), seed=21)
        mixed = {}
        for w in (0.1, 1.0):
            ctxs = [(grp, w if c == 2 else weight) for c, (grp, weight) in enumerate(contexts)]
            mixed[w] = (O.successor_v1(out, _hand_batch(members, ctxs, O.SUCCESSOR_V1)),
                        _manual_successor(members, ctxs, _hand_batch(members, ctxs, O.SUCCESSOR_V1), out))
        feas_logits = out.feas_logits.detach().double().tolist()
        rows = [n for n, m in enumerate(members) if m["ctx"] == 2 and m["fv"]]
        m2 = _mean([-_log_softmax(feas_logits[n])[members[n]["feas"]] for n in rows])
        n_supported = len({m["ctx"] for m in members if m["fv"]})
        self.assertEqual(n_supported, 4)
        for w, (got, ref) in mixed.items():
            for name in ("feas", "rt"):
                self.assertAlmostEqual(got.terms[name], ref[name], delta=2e-6 * max(1.0, abs(ref[name])), msg=name)
        self.assertAlmostEqual(mixed[1.0][0].terms["feas"] - mixed[0.1][0].terms["feas"], 0.9 * m2 / n_supported,
                               delta=1e-5, msg="only the feasibility context's own share may change with its weight")

    def test_1c_nonfinite_raises(self):
        members, contexts = _successor_spec()
        batch = _hand_batch(members, contexts, O.SUCCESSOR_V1)
        out = _fake_out(len(members), seed=8)
        nan = float("nan")
        with self.assertRaises(O.ObjectiveNonFinite):
            O.successor_v1(out._replace(q_hat=out.q_hat.detach().clone().fill_(nan)), batch)
        for member in (3, 16):          # a known teacher, and a feasibility member no q-term reads
            q_hat = out.q_hat.detach().clone()
            q_hat[member] = nan
            with self.assertRaisesRegex(O.ObjectiveNonFinite, "'q_hat'"):
                O.successor_v1(out._replace(q_hat=q_hat), batch)
        rt_hat = out.rt_hat.detach().clone()
        rt_hat[0] = float("inf")
        with self.assertRaisesRegex(O.ObjectiveNonFinite, "'rt_hat'"):
            O.successor_v1(out._replace(rt_hat=rt_hat), batch)
        feas_logits = out.feas_logits.detach().clone()
        feas_logits[5, 1] = nan
        with self.assertRaisesRegex(O.ObjectiveNonFinite, "'feas_logits'"):
            O.successor_v1(out._replace(feas_logits=feas_logits), batch)
        huge = out.rt_hat.detach().clone()
        huge[7] = -1e30                  # finite output, censored member: relu(t - rt)^2 overflows float32
        with self.assertRaisesRegex(O.ObjectiveNonFinite, "'rt'"):
            O.successor_v1(out._replace(rt_hat=huge), batch)
        logits = out.logits.detach().clone()
        position = int(batch.legal[0, 0].nonzero()[0, 0])
        logits[0, 0, position] = nan
        with self.assertRaisesRegex(O.ObjectiveNonFinite, "'ce'"):
            O.successor_v1(out._replace(logits=logits), batch)

    def test_1c_refuses_a_baseline_batch(self):
        members, contexts = _baseline_spec()
        batch = _hand_batch(members, contexts, O.BASELINE_V1)
        with self.assertRaisesRegex(ValueError, "successor-v1"):
            O.successor_v1(_fake_out(len(members), 2), batch)

    def _fixture_batch(self, **train_kw):
        from dltrain import sampler as S
        fx = _fixtures()
        data = fx.train_data(**train_kw)
        draws = S.MacroSampler(data.fit, fx.RUN_SEED, data.config)
        for step in range(1, 200):
            batch = S.collate(draws.draw(step), data)
            if bool(batch.ctx_is_group.any()) and bool((~batch.ctx_is_group).any()):
                return fx, data, batch
        self.fail("no fixture step mixes group and feasibility contexts")

    def test_1c_ce_matches_manual_teacher_weighting_on_collated_fixture_batch(self):
        fx, data, batch = self._fixture_batch()
        model = _seeded_model(fx.model_config(), 31).eval()
        out = model(batch.tokens, batch.lengths, batch.ctx_z, batch.seq_ctx)
        ce = _manual_seq_ce(out.logits, batch.tokens, batch.lengths, batch.legal)
        labels_of = [data.corpus.label(c, p) for c, p in batch.member_ids]

        def manual(divisors: list[float]) -> float:
            per = []
            for c in range(len(batch.ctx_ids)):
                if not bool(batch.ctx_is_group[c]):
                    continue
                teachers = [n for n in range(len(labels_of)) if int(batch.seq_ctx[n]) == c and int(batch.role[n]) == 0]
                self.assertTrue(teachers)
                soft = _softmax([labels_of[n].q_opt / O.TEACHER_TAU for n in teachers])
                per.append(math.fsum(s / divisors[n] * ce[n] for s, n in zip(soft, teachers)))
            return _mean(per)

        divisors = [float(x) for x in batch.imitation_divisor]
        res = O.successor_v1(out, batch)
        self.assertAlmostEqual(res.terms["ce"], manual(divisors), delta=5e-6)
        self.assertEqual(res.support["group_contexts"], int(batch.ctx_is_group.sum()))
        for n, lab in enumerate(labels_of):
            if not lab.shared_seed:
                self.assertEqual(divisors[n], 1.0)
        first_teacher = int(((batch.role == 0) & batch.ctx_is_group[batch.seq_ctx]).nonzero()[0, 0])
        divisors[first_teacher] = 2.5
        perturbed = dataclasses.replace(batch, imitation_divisor=torch.tensor(divisors, dtype=torch.float32))
        res2 = O.successor_v1(out, perturbed)
        self.assertAlmostEqual(res2.terms["ce"], manual(divisors), delta=5e-6)
        self.assertNotAlmostEqual(res2.terms["ce"], res.terms["ce"], delta=1e-4)
        self.assertTrue(math.isfinite(res.loss.item()))

    def test_1c_dense_pairs_equal_ranking_pairs_on_collated_fixture_batch(self):
        fx, data, batch = self._fixture_batch()
        pair, weight = O.bt_pair_matrix(batch)
        found = {(i, k): float(weight[i, k]) for i, k in pair.nonzero().tolist()}
        expected = {}
        for c in range(len(batch.ctx_ids)):
            rows = [n for n in range(len(batch.member_ids)) if int(batch.seq_ctx[n]) == c]
            if not bool(batch.ctx_is_group[c]):
                continue
            labs = [data.corpus.label(*batch.member_ids[n]) for n in rows]
            for wi, lo, w in L.ranking_pairs(labs):
                expected[(rows[wi], rows[lo])] = w
        self.assertTrue(expected)
        self.assertEqual(set(found), set(expected))
        for key, w in expected.items():
            self.assertAlmostEqual(found[key], w, delta=1e-6)
        model = _seeded_model(fx.model_config(), 32).eval()
        out = model(batch.tokens, batch.lengths, batch.ctx_z, batch.seq_ctx)
        res = O.successor_v1(out, batch)
        self.assertEqual(res.support["pairs"], sum(1 for w in expected.values() if w > 0))
        # Empty support on the same fixture batch: no group context left -> connected zeros with support 0.
        feas = dataclasses.replace(batch, ctx_is_group=torch.zeros_like(batch.ctx_is_group))
        res0 = O.successor_v1(out, feas)
        for name in ("ce", "bt", "list", "q"):
            self.assertEqual(res0.terms[name], 0.0)
        self.assertEqual([res0.support[k] for k in ("teachers", "pairs", "list_groups", "q_members")], [0, 0, 0, 0])
        self.assertTrue(res0.loss.requires_grad)
        for member in range(batch.tokens.shape[0]):
            q_hat = out.q_hat.detach().clone()
            q_hat[member] = float("nan")
            with self.assertRaises(O.ObjectiveNonFinite):
                O.successor_v1(out._replace(q_hat=q_hat), batch)

    def test_1c_forward_objective_backward_is_bitwise_deterministic(self):
        members, contexts = _successor_spec()
        batch = _hand_batch(members, contexts, O.SUCCESSOR_V1)
        grads = []
        for _ in range(2):
            model = _seeded_model(_small_config(dropout=0.1), 41).train()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1234)
                out = model(batch.tokens, batch.lengths, batch.ctx_z, batch.seq_ctx)
                res = O.successor_v1(out, batch)
            res.loss.backward()
            grads.append([(name, p.grad.clone()) for name, p in sorted(model.named_parameters())])
        for (name, a), (_, b) in zip(*grads):
            self.assertTrue(torch.equal(a, b), name)


def _record_label(m: dict) -> L.RecordLabel:
    known = m["known"]
    return L.RecordLabel(context_sha256="c", program_sha256="p", skeleton_sha256="s", outcome=L.COMPLETED,
                         labelled=True, feas_class=0, feas_valid=True, q=m["q_opt"] if known else None,
                         q_opt=m["q_opt"] if known else None, q_grid=m["q_grid"] if known else None,
                         q_status="known" if known else "not_completed", gate_pass=None,
                         midrank_pct=m["pct"] if known else None, rt_target=None, rt_censored=False, rt_valid=False,
                         value=None, value_pct=None, backend="local", shared_seed=False)


# ======================================================================================================== P7 test 1e

class TestP7BaselineObjective(unittest.TestCase):
    """P7 test 1e: the lineage-B composite (0.05+p^4 row weights, value ties skipped) and its mode refusals."""

    def test_1e_terms_match_reference_with_row_weights_and_ties(self):
        members, contexts = _baseline_spec()
        batch = _hand_batch(members, contexts, O.BASELINE_V1)
        out = _fake_out(len(members), seed=12, cf=True)
        res = O.baseline_v1(out, batch)
        ref, n_pairs, tied = _manual_baseline(members, batch, out)
        self.assertEqual(list(res.terms), list(O.COEF_BASELINE))
        for name, value in ref.items():
            self.assertAlmostEqual(res.terms[name], value, delta=2e-6 * max(1.0, abs(value)), msg=name)
        self.assertEqual(tied, 2)
        self.assertEqual(res.support, {"rows": len(members), "pairs": n_pairs, "tied_pairs": 2, "list_groups": 2,
                                       "rt_rows": sum(m["rv"] for m in members)})
        self.assertEqual(n_pairs, 26 + 28 + 1)
        pair, weight, tied_count = O.baseline_pair_matrix(batch)
        self.assertEqual(tied_count, 2)
        for i, k in ((1, 2), (4, 5)):
            self.assertFalse(bool(pair[i, k]) or bool(pair[k, i]), f"tied rows {i},{k} formed a pair")
        unweighted = _mean(_manual_seq_ce(out.logits, batch.tokens, batch.lengths, batch.legal, 0.05))
        self.assertGreater(abs(res.terms["seq_nll"] - unweighted), 1e-3, "row weights had no effect")
        self.assertAlmostEqual(res.loss.item(), math.fsum(O.COEF_BASELINE[k] * v for k, v in ref.items()), delta=1e-5)
        res.loss.backward()
        self.assertEqual(int(torch.count_nonzero(out.logits.grad[~batch.legal])), 0)
        self.assertGreater(int(torch.count_nonzero(out.cf_hat.grad)), 0)

    def test_1e_row_weight_is_floor_plus_fourth_power(self):
        members, contexts = _baseline_spec()
        batch = _hand_batch(members, contexts, O.BASELINE_V1)
        out = _fake_out(len(members), seed=13, cf=True)
        nll = O.sequence_ce(out.logits, batch.tokens[:, 1:], batch.legal,
                            O.valid_positions(batch.lengths, batch.tokens.shape[1]), O.BASELINE_LABEL_SMOOTHING)
        weights = 0.05 + batch.value_pct ** 4
        res = O.baseline_v1(out, batch)
        self.assertAlmostEqual(res.terms["seq_nll"], ((weights * nll).sum() / weights.sum()).item(), delta=1e-6)
        self.assertEqual((O.ROW_WEIGHT_FLOOR, O.ROW_WEIGHT_EXPONENT, O.BASELINE_LABEL_SMOOTHING), (0.05, 4, 0.05))

    def test_1e_finite_on_collated_fixture_baseline_batch(self):
        from dltrain import sampler as S
        fx = _fixtures()
        data = fx.train_data(group_mode=O.BASELINE_V1)
        self.assertEqual(data.config.group_mode, O.BASELINE_V1)
        draws = S.MacroSampler(data.fit, fx.RUN_SEED, data.config)
        batch = None
        for step in range(1, 50):
            try:
                candidate = S.collate(draws.draw(step), data)
            except S.SamplerError:      # every slot of this step was 'skipped'
                continue
            if candidate.tokens.shape[0] >= 8:
                batch = candidate
                break
        self.assertIsNotNone(batch, "no baseline-mode fixture step drew a group")
        self.assertEqual(batch.mode, O.BASELINE_V1)
        model = _seeded_model(fx.model_config(context_free_head=True), 51).train()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(52)
            out = model(batch.tokens, batch.lengths, batch.ctx_z, batch.seq_ctx)
        res = O.baseline_v1(out, batch)
        self.assertTrue(math.isfinite(res.loss.item()))
        self.assertTrue(all(math.isfinite(v) for v in res.terms.values()))
        self.assertEqual(res.support["rows"], batch.tokens.shape[0])
        self.assertGreater(res.support["pairs"], 0)
        self.assertGreater(res.support["list_groups"], 0)
        res.loss.backward()
        self.assertGreater(float(model.cf_head[0].weight.grad.abs().sum()), 0.0)
        for name, p in model.named_parameters():
            if name.startswith("feas_head."):
                self.assertIsNone(p.grad, "the lineage-B composite has no feasibility term")
            else:
                self.assertTrue(bool(torch.isfinite(p.grad).all()), name)

    def test_1e_refusals(self):
        members, contexts = _successor_spec()
        succ = _hand_batch(members, contexts, O.SUCCESSOR_V1)
        with self.assertRaisesRegex(ValueError, "baseline-v1"):
            O.baseline_v1(_fake_out(len(members), 2, cf=True), succ)
        b_members, b_contexts = _baseline_spec()
        base = _hand_batch(b_members, b_contexts, O.BASELINE_V1)
        with self.assertRaisesRegex(ValueError, "context-free head"):
            O.baseline_v1(_fake_out(len(b_members), 2, cf=False), base)
        no_value = _hand_batch([dict(m, value=0.0) if i == 3 else m for i, m in enumerate(b_members)], b_contexts,
                               O.BASELINE_V1)
        with self.assertRaisesRegex(ValueError, "value"):
            O.baseline_v1(_fake_out(len(b_members), 2, cf=True), no_value)
        out = _fake_out(len(b_members), 3, cf=True)
        cf_hat = out.cf_hat.detach().clone()
        cf_hat[0] = float("nan")
        with self.assertRaisesRegex(O.ObjectiveNonFinite, "'cf_hat'"):
            O.baseline_v1(out._replace(cf_hat=cf_hat), base)
        logits = out.logits.detach().clone()
        logits[2, 0, int(base.legal[2, 0].nonzero()[0, 0])] = float("inf")
        with self.assertRaisesRegex(O.ObjectiveNonFinite, "'seq_nll'"):
            O.baseline_v1(out._replace(logits=logits), base)


# ===================================================================================================== P7 test 2

Y = [((i % 7) - 3) + 0.75 * math.sin(0.37 * i) for i in range(24)]
CTX = _sha("hand-context")
BASE = [1.0, 1.2]


def _ctx_row(csha: str = CTX, y=None, **kw) -> dict:
    return {"context_sha256": csha, "dataset_hash": _sha("hand-dataset"), "family_id": "fam-hand",
            "split_role": "train_fit", "fold": 1, "frozen_fold0": False, "y_train": list(Y if y is None else y)} | kw


def _row(tag: str, outcome: str = L.COMPLETED, *, csha: str = CTX, rmse=None, base=None, cv_log=None, cv_ratio=None,
         wall_us=1_500_000, backend="local", shared_seed=False) -> dict:
    return {"context_sha256": csha, "context_schema": "fixture/v1", "dataset_hash": _sha("hand-dataset"),
            "family_id": "fam-hand", "split_role": "train_fit", "program_sha256": _sha("prog:" + tag),
            "skeleton_sha256": _sha("skel:" + tag), "tokens": [1, 2], "length": 2, "outcome": outcome,
            "rmse_j": rmse, "baseline_rmse_j": base, "cv_mean_log_ratio": cv_log, "cv_log_ratio_sd": None,
            "cv_mean_ratio": cv_ratio, "wall_us": wall_us, "backend": backend, "shared_seed": shared_seed}


def _detailed(tag: str, q: float, base=BASE, **kw) -> dict:
    """A completed detailed row whose q is (to rounding) ``q``: rmse_j = base_j * exp(-q)."""
    rmse = [b * math.exp(-q) for b in base]
    ratios = [r / b for r, b in zip(rmse, base)]
    return _row(tag, rmse=rmse, base=list(base), cv_log=math.fsum(math.log(x) for x in ratios) / 2,
                cv_ratio=math.fsum(ratios) / 2, **kw)


def _by_tag(recs) -> dict:
    by_sha = {r.program_sha256: r for r in recs}
    return types.SimpleNamespace(get=lambda tag: by_sha[_sha("prog:" + tag)])


class TestP7Labels(unittest.TestCase):
    """P7 test 2: the train-only label rules of PLAN §6.3 and §5 P5."""

    def test_2_constant_target_is_degenerate_baseline_with_feasibility_only(self):
        y = [2.5] * 24
        self.assertEqual(L.robust_scale(y), 0.0)
        rows = [_row("a", rmse=[0.0, 0.0], base=[0.0, 0.0], cv_ratio=1.0),
                _row("b", rmse=[0.0, 0.0], base=[0.0, 0.0]),
                _row("f", "failed:compile_error", base=[0.0, 0.0], wall_us=60_000),
                _row("t", L.TIMEOUT_OUTCOME, base=[0.0, 0.0], wall_us=90_000_000)]
        ctx, recs = L.label_context(_ctx_row(y=y), rows)
        self.assertEqual(ctx.status, "DEGENERATE_BASELINE")
        self.assertEqual(ctx.eps_c, 1e-12 * 2.5)
        self.assertIsNone(ctx.q_cap)
        got = _by_tag(recs)
        for tag in ("a", "b"):
            lab = got.get(tag)
            self.assertEqual((lab.q, lab.q_opt, lab.q_grid, lab.midrank_pct, lab.value), (None,) * 5)
            self.assertEqual(lab.q_status, "degenerate_baseline")
            self.assertEqual((lab.feas_class, lab.feas_valid, lab.rt_valid), (0, True, True))
        self.assertTrue(got.get("a").gate_pass)
        self.assertEqual(got.get("f").feas_class, 1)
        self.assertEqual(got.get("t").feas_class, 2)
        self.assertEqual(L.ranking_pairs(recs), [])
        self.assertEqual((ctx.counts["q_known"], ctx.counts["q_unknown"], ctx.counts["completed"]), (0, 2, 2))

    def test_2_grid_ties_give_no_pair_and_are_transitive(self):
        qs = {"a": 0.1000004, "b": 0.1000009, "c": 0.1000014, "d": 0.25, "e": 0.25, "f": -0.4}
        ctx, recs = L.label_context(_ctx_row(), [_detailed(t, q) for t, q in qs.items()])
        got = _by_tag(recs)
        self.assertEqual({t: got.get(t).q_grid for t in qs},
                         {"a": 100000, "b": 100001, "c": 100001, "d": 250000, "e": 250000, "f": -400000})
        index = {r.program_sha256: i for i, r in enumerate(recs)}
        idx = {t: index[_sha("prog:" + t)] for t in qs}
        pairs = {frozenset((w, lo)): (w, lo) for w, lo, _ in L.ranking_pairs(recs)}
        tie = lambda x, y: frozenset((idx[x], idx[y])) not in pairs  # noqa: E731
        self.assertTrue(tie("b", "c") and tie("d", "e"))
        self.assertFalse(tie("a", "b"), "a tolerance rule would tie a~b; the grid rule must not")
        self.assertFalse(tie("a", "c"))
        for x in qs:
            for y in qs:
                for z in qs:
                    if len({x, y, z}) == 3 and tie(x, y) and tie(y, z):
                        self.assertTrue(tie(x, z), f"tie not transitive on {x},{y},{z}")
        self.assertEqual(pairs[frozenset((idx["a"], idx["b"]))], (idx["b"], idx["a"]))
        self.assertEqual(got.get("b").midrank_pct, got.get("c").midrank_pct)
        self.assertEqual(len(pairs), 15 - 2)

    def test_2_timeout_is_class_2_and_right_censored(self):
        rows = [_detailed("ok", 0.3),
                _row("t", L.TIMEOUT_OUTCOME, base=BASE, wall_us=90_000_000),
                _row("tr", L.TIMEOUT_OUTCOME, base=BASE, wall_us=None, backend="remote")]
        _, recs = L.label_context(_ctx_row(), rows)
        got = _by_tag(recs)
        for tag, local in (("t", True), ("tr", False)):
            lab = got.get(tag)
            self.assertEqual(lab.feas_class, L.FEAS_TIMEOUT)
            self.assertEqual(lab.rt_target, math.log1p(90.0))
            self.assertTrue(lab.rt_censored)
            self.assertEqual((lab.q, lab.q_status, lab.gate_pass), (None, "not_completed", None))
            self.assertEqual((lab.feas_valid, lab.rt_valid), (local, local))
        ok = got.get("ok")
        self.assertEqual((ok.rt_target, ok.rt_censored), (math.log1p(1.5), False))

    def test_2_completed_nonfinite_metric_becomes_failed_nonfinite(self):
        rows = [_detailed("ok", 0.2),
                _row("nan", rmse=[float("nan"), 1.2], base=BASE, cv_log=0.1, cv_ratio=1.0),
                _row("inf", rmse=[float("inf"), 1.2], base=BASE),
                _row("ninf", rmse=[1.0, float("-inf")], base=BASE),
                _row("neg", rmse=[-0.1, 1.2], base=BASE),
                _row("detailless_nan", cv_log=float("nan"), cv_ratio=float("nan"), base=BASE)]
        ctx, recs = L.label_context(_ctx_row(), rows)
        got = _by_tag(recs)
        for tag in ("nan", "inf", "ninf", "neg"):
            lab = got.get(tag)
            self.assertEqual(lab.outcome, L.NONFINITE)
            self.assertEqual(lab.feas_class, L.FEAS_FAILED)
            self.assertEqual((lab.q, lab.q_status, lab.gate_pass, lab.rt_target, lab.rt_valid),
                             (None, "not_completed", None, None, False))
        kept = got.get("detailless_nan")
        self.assertEqual((kept.outcome, kept.feas_class, kept.q_status), (L.COMPLETED, 0, "unknown_detailless_nonfinite"))
        self.assertEqual((ctx.counts["nonfinite"], ctx.counts["failed"], ctx.counts["completed"]), (4, 4, 2))
        self.assertEqual(L.normalize_outcome(rows[1]), L.NONFINITE)

    def test_2_zero_rmse_is_floored_so_q_equals_q_cap(self):
        rows = [_row("exact", rmse=[0.0, 0.0], base=BASE, cv_ratio=0.0), _row("half", rmse=[0.0, 0.6], base=BASE),
                _detailed("ok", 0.1)]
        ctx, recs = L.label_context(_ctx_row(), rows)
        s_c = L.robust_scale(Y)
        eps = L.epsilon(Y, s_c)
        self.assertEqual((ctx.s_c, ctx.eps_c), (s_c, eps))
        self.assertEqual(ctx.q_cap, L.q_cap(BASE, eps))
        exact = _by_tag(recs).get("exact")
        self.assertEqual(exact.q, ctx.q_cap)
        self.assertTrue(math.isfinite(exact.q))
        self.assertGreater(ctx.q_cap, L.Q_CLIP)
        self.assertEqual(exact.q_opt, L.Q_CLIP)
        self.assertTrue(exact.gate_pass)
        half = _by_tag(recs).get("half")
        self.assertAlmostEqual(half.q, -0.5 * (math.log(eps / 1.0) + math.log(0.6 / 1.2)), places=12)
        self.assertEqual(L.q_from_detail([0.0, 0.0], BASE, eps), L.q_cap(BASE, eps))

    def test_2_q_opt_is_clipped_at_log_1e3(self):
        _, recs = L.label_context(_ctx_row(), [_detailed("hi", 8.0), _detailed("lo", -8.0), _detailed("mid", 2.0)])
        got = _by_tag(recs)
        self.assertEqual(got.get("hi").q_opt, math.log(1e3))
        self.assertEqual(got.get("lo").q_opt, -math.log(1e3))
        self.assertEqual(got.get("mid").q_opt, got.get("mid").q)
        self.assertEqual(got.get("hi").q_grid, round(got.get("hi").q * 1e6))
        self.assertAlmostEqual(got.get("hi").q, 8.0, places=9)
        self.assertEqual((L.clip_q(1e9), L.clip_q(-1e9), L.clip_q(0.5)), (L.Q_CLIP, -L.Q_CLIP, 0.5))

    def test_2_detailless_q_is_capped_and_nonfinite_is_unknown(self):
        eps = L.epsilon(Y, L.robust_scale(Y))
        cap = L.q_cap(BASE, eps)
        rows = [_detailed("ok", 0.4), _row("cap", base=BASE, cv_log=-(cap + 1.0), cv_ratio=0.5),
                _row("below", base=BASE, cv_log=-0.3), _row("neginf", base=BASE, cv_log=float("-inf")),
                _row("nan", base=BASE, cv_log=float("nan")), _row("none", base=BASE, cv_log=None),
                _row("nobase", cv_log=-0.25)]
        ctx, recs = L.label_context(_ctx_row(), rows)
        got = _by_tag(recs)
        self.assertEqual(got.get("cap").q, cap)
        self.assertEqual(got.get("below").q, 0.3)
        self.assertEqual(got.get("nobase").q, 0.25, "a detail-less row uses the context baseline's cap")
        for tag in ("neginf", "nan", "none"):
            lab = got.get(tag)
            self.assertEqual((lab.q, lab.q_status, lab.outcome, lab.feas_class),
                             (None, "unknown_detailless_nonfinite", L.COMPLETED, 0))
        self.assertEqual((ctx.counts["q_known"], ctx.counts["q_unknown"]), (4, 3))
        self.assertIsNone(got.get("below").value, "the baseline arm needs per-fold detail")
        self.assertIsNotNone(got.get("ok").value)

    def test_2_abandoned_and_missing_are_unlabelled_and_counted(self):
        rows = [_detailed("ok", 0.4)] + [_row(o, o, base=None, wall_us=None) for o in sorted(L.ABANDONED)] + \
            [_row("missing", L.MISSING, wall_us=None)]
        ctx, recs = L.label_context(_ctx_row(), rows)
        for lab in recs:
            if lab.outcome == L.COMPLETED:
                continue
            self.assertFalse(lab.labelled)
            self.assertEqual((lab.feas_class, lab.feas_valid, lab.q_status, lab.rt_target, lab.rt_valid),
                             (None, False, "unlabelled", None, False))
        self.assertEqual(ctx.counts, {"rows": 6, "labelled": 1, "completed": 1, "failed": 0, "timeout": 0,
                                      "abandoned": 4, "missing": 1, "nonfinite": 0, "q_known": 1, "q_unknown": 0})
        other = _sha("other-context")
        table = L.label_corpus([_ctx_row(), _ctx_row(other)],
                               rows + [_row("m2", L.MISSING, csha=other), _detailed("x", 0.1, csha=other)])
        self.assertEqual(table.counts()["missing"], 2)
        self.assertEqual(table.counts()["rows"], 8)
        self.assertEqual(list(table.contexts), sorted(table.contexts))

    def test_2_midrank_percentiles_with_ties(self):
        self.assertEqual(L.midrank_pcts([30, 10, 20, 20]), [1.0, 0.0, 0.5, 0.5])
        self.assertEqual(L.midrank_pcts([1, 2, 2, 2, 3]), [0.0, 0.5, 0.5, 0.5, 1.0])
        self.assertEqual(L.midrank_pcts([5, 5, 5]), [0.5, 0.5, 0.5])
        self.assertEqual(L.midrank_pcts([7]), [0.5])
        self.assertEqual(L.midrank_pcts([]), [])
        self.assertEqual(L.midrank_pcts(list(range(11)))[9], 0.9)
        rows = [_detailed("a", 0.5), _detailed("b", 0.5), _detailed("c", 0.1), _row("f", "failed:x", base=BASE)]
        _, recs = L.label_context(_ctx_row(), rows)
        got = _by_tag(recs)
        self.assertEqual((got.get("a").midrank_pct, got.get("b").midrank_pct, got.get("c").midrank_pct), (0.75, 0.75, 0.0))
        self.assertIsNone(got.get("f").midrank_pct)

    def test_2_baseline_disagreement_raises(self):
        with self.assertRaisesRegex(L.LabelError, "baseline disagreement"):
            L.label_context(_ctx_row(), [_detailed("a", 0.1, base=[1.0, 1.2]), _detailed("b", 0.2, base=[1.0, 1.25])])
        with self.assertRaisesRegex(L.LabelError, "baseline disagreement"):
            L.label_context(_ctx_row(), [_detailed("a", 0.1), _row("f", "failed:x", base=[1.0, 1.3])])
        ctx, _ = L.label_context(_ctx_row(), [_detailed("a", 0.1), _row("n", "failed:x", base=[float("nan"), 1.2])])
        self.assertEqual(ctx.baseline_rmse_j, (1.0, 1.2))
        ctx, recs = L.label_context(_ctx_row(), [_row("only", "failed:x", base=None), _row("c", cv_log=-0.1)])
        self.assertEqual((ctx.status, ctx.baseline_rmse_j), ("UNKNOWN_BASELINE", None))
        self.assertEqual(_by_tag(recs).get("c").q_status, "unknown_baseline")

    def test_2_baseline_comes_only_from_rows_with_inner_cv_detail(self):
        """§5 P5 Labels: the per-context baseline RMSE_j comes from rows with inner-CV detail, and a context with no
        detailed row has UNKNOWN labels, even when a failed or detail-less row carries a baseline pair."""
        rows = [_row("f", "failed:compile_error", base=[1.0, 1.2], wall_us=60_000),
                _row("c1", base=list(BASE), cv_log=-0.3), _row("c2", cv_log=-9.0)]
        self.assertFalse(any(L.is_detailed(r) for r in rows))
        ctx, recs = L.label_context(_ctx_row(), rows)
        self.assertEqual((ctx.status, ctx.baseline_rmse_j, ctx.q_cap), ("UNKNOWN_BASELINE", None, None))
        got = _by_tag(recs)
        for tag in ("c1", "c2"):
            lab = got.get(tag)
            self.assertEqual((lab.outcome, lab.feas_class, lab.q_status), (L.COMPLETED, 0, "unknown_baseline"))
            self.assertEqual((lab.q, lab.q_opt, lab.q_grid, lab.midrank_pct), (None, None, None, None))
        self.assertEqual((ctx.counts["q_known"], ctx.counts["q_unknown"]), (0, 2))
        self.assertEqual(L.ranking_pairs(recs), [])
        ctx, _ = L.label_context(_ctx_row(), [_row("f", "failed:x", base=[1.0, 1.2]),
                                              _row("t", L.TIMEOUT_OUTCOME, base=[1.0, 1.3], wall_us=90_000_000)])
        self.assertEqual(ctx.status, "UNKNOWN_BASELINE", "baselines that establish nothing are not compared")
        ctx, recs = L.label_context(_ctx_row(), rows + [_detailed("d", 0.2, base=[1.0, 1.2])])
        self.assertEqual((ctx.status, ctx.baseline_rmse_j), ("OK", (1.0, 1.2)))
        self.assertEqual((_by_tag(recs).get("c1").q, _by_tag(recs).get("c2").q), (0.3, 9.0))

    def test_2_gate_pass_rule(self):
        base = [1.0, 2.0]
        rows = [_row("edge", rmse=[1.05, 2.1], base=base), _row("over", rmse=[1.1, 2.2], base=base),
                _row("mixed", rmse=[0.5, 3.0], base=base), _row("cv_ok", base=base, cv_log=-0.01, cv_ratio=1.04),
                _row("cv_bad", base=base, cv_log=0.1, cv_ratio=1.2), _row("cv_none", base=base, cv_log=0.0),
                _row("cv_nan", base=base, cv_log=0.0, cv_ratio=float("nan")), _row("t", L.TIMEOUT_OUTCOME, base=base)]
        _, recs = L.label_context(_ctx_row(), rows)
        got = _by_tag(recs)
        self.assertEqual({t: got.get(t).gate_pass for t in ("edge", "over", "mixed", "cv_ok", "cv_bad", "cv_none", "cv_nan", "t")},
                         {"edge": True, "over": False, "mixed": True, "cv_ok": True, "cv_bad": False, "cv_none": None,
                          "cv_nan": None, "t": None})
        _, recs = L.label_context(_ctx_row(y=[2.5] * 24),
                                  [_row("z", rmse=[0.0, 0.0], base=[0.0, 0.0], cv_ratio=0.9), _row("zn", rmse=[0.0, 0.0], base=[0.0, 0.0])])
        self.assertEqual((_by_tag(recs).get("z").gate_pass, _by_tag(recs).get("zn").gate_pass), (True, None))

    def test_2_feasibility_and_runtime_follow_the_local_backend_rule(self):
        rows = [_detailed("local", 0.2), _detailed("remote", 0.3, backend="remote"),
                _detailed("unknown", 0.4, backend="unknown"), _row("f_remote", "failed:oom", base=BASE, backend="remote"),
                _detailed("no_wall", 0.5, wall_us=None)]
        _, recs = L.label_context(_ctx_row(), rows)
        got = _by_tag(recs)
        self.assertEqual((got.get("local").feas_valid, got.get("local").rt_valid), (True, True))
        for tag in ("remote", "unknown", "f_remote"):
            lab = got.get(tag)
            self.assertEqual((lab.feas_valid, lab.rt_valid), (False, False), tag)
            self.assertIsNotNone(lab.feas_class)
        self.assertEqual(got.get("remote").q_status, "known", "q does not depend on the backend")
        self.assertEqual((got.get("no_wall").rt_target, got.get("no_wall").rt_valid, got.get("no_wall").feas_valid),
                         (None, False, True))

    def test_2_robust_scale_fallbacks_and_epsilon(self):
        y = [1.0, 2.0, 4.0, 8.0, 16.0]
        self.assertAlmostEqual(L.robust_scale(y), 1.4826 * 3.0)
        iqr_only = [0.0] * 6 + [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(L.robust_scale(iqr_only), 1.75 / 1.349)
        sd_only = [0.0] * 20 + [100.0]
        self.assertAlmostEqual(L.robust_scale(sd_only), float(np.std(sd_only)))
        self.assertEqual(L.epsilon([1e9, 1e9], 0.0), 1e-12 * 1e9)
        self.assertEqual(L.epsilon([0.1, 0.2], 3.0), 1e-6 * 3.0)
        for bad in ([], [1.0, float("nan")], [float("inf")], "abc"):
            with self.assertRaises(L.LabelError):
                L.robust_scale(bad)
        with self.assertRaises(L.LabelError):
            L.label_context(_ctx_row(y=[]), [])

    def test_2_outcome_validation(self):
        self.assertEqual(L.normalize_outcome({"outcome": "failed:oom"}), "failed:oom")
        self.assertEqual(L.normalize_outcome({"outcome": L.TIMEOUT_OUTCOME}), L.TIMEOUT_OUTCOME)
        for bad in ("succeeded", "failed:", "failed: x", "timeout@1000ms", "abandoned:other", 3, None):
            with self.assertRaises(L.LabelError):
                L.normalize_outcome({"outcome": bad})
        with self.assertRaises(L.LabelError):
            L.label_context(_ctx_row(), [_detailed("a", 0.1, csha=_sha("elsewhere"))])
        with self.assertRaises(L.LabelError):
            L.label_corpus([_ctx_row()], [_detailed("a", 0.1, csha=_sha("absent"))])
        with self.assertRaises(L.LabelError):
            L.label_corpus([_ctx_row(), _ctx_row()], [])
        with self.assertRaises(L.LabelError):
            L.label_context(_ctx_row(), [_row("w", base=BASE, wall_us=-5)])

    def test_2_each_context_is_labelled_from_its_own_rows_only(self):
        other = _sha("poisoned-context")
        mine = [_detailed("a", 0.4), _detailed("b", 0.1), _row("t", L.TIMEOUT_OUTCOME, base=BASE)]
        clean = L.label_corpus([_ctx_row(), _ctx_row(other)], mine + [_detailed("x", 0.2, csha=other)])
        poisoned_rows = [_detailed("x", 5.0, base=[3.0, 3.0], csha=other), _detailed("y", -2.0, base=[3.0, 3.0], csha=other),
                         _row("z", "failed:poison", csha=other)]
        poisoned = L.label_corpus([_ctx_row(), _ctx_row(other, y=[v * 3 + 7 for v in Y])], mine + poisoned_rows)
        self.assertEqual(poisoned.contexts[CTX], clean.contexts[CTX])
        self.assertEqual(poisoned.records[CTX], clean.records[CTX])
        self.assertEqual(common.canonical([r.to_json() for r in poisoned.records[CTX]]),
                         common.canonical([r.to_json() for r in clean.records[CTX]]))
        self.assertEqual(common.canonical(poisoned.contexts[CTX].to_json()), common.canonical(clean.contexts[CTX].to_json()))

    def test_2_baseline_arm_value_and_percentile(self):
        qs = {"a": 0.9, "b": 0.5, "c": 0.5, "d": -0.2, "e": 0.1}
        rows = [_detailed(t, q) for t, q in qs.items()] + [_row("dl", base=BASE, cv_log=-0.3)]
        ctx, recs = L.label_context(_ctx_row(), rows)
        got = _by_tag(recs)
        geo = {t: math.exp(0.5 * (math.log(BASE[0] * math.exp(-q)) + math.log(BASE[1] * math.exp(-q)))) for t, q in qs.items()}
        med = float(np.median(list(geo.values())))
        values = {t: 1.0 / (1.0 + g / med) for t, g in geo.items()}
        pcts = dict(zip(values, L.midrank_pcts([L.q_grid(v) for v in values.values()])))
        for t in qs:
            self.assertAlmostEqual(got.get(t).value, values[t], places=12)
            self.assertEqual(got.get(t).value_pct, pcts[t])
        self.assertEqual((got.get("b").value, got.get("c").value), (0.5, 0.5), "the median rows have value exactly 1/2")
        self.assertEqual(got.get("b").value_pct, got.get("c").value_pct)
        self.assertEqual((got.get("dl").value, got.get("dl").value_pct), (None, None))

    def test_2_fixture_corpus_specials(self):
        fx = _fixtures()
        contexts, records = fx.contexts(), fx.records()
        table = L.label_corpus(contexts, records)
        special = fx.special_contexts()
        raw = {}
        for r in records:
            raw.setdefault(r["context_sha256"], {})[r["program_sha256"]] = r
        labs = lambda name: table.records[special[name]]  # noqa: E731

        self.assertEqual(table.contexts[special["degenerate"]].status, "DEGENERATE_BASELINE")
        for lab in labs("degenerate"):
            self.assertIsNone(lab.q)
            if lab.outcome == L.COMPLETED:
                self.assertEqual(lab.q_status, "degenerate_baseline")
        self.assertEqual(L.ranking_pairs(labs("degenerate")), [])
        self.assertEqual(table.contexts[special["no_detail"]].status, "UNKNOWN_BASELINE")
        self.assertTrue(all(lab.q is None for lab in labs("no_detail")))

        cap = table.contexts[special["exact_fit"]].q_cap
        exact = [lab for lab in labs("exact_fit") if raw[special["exact_fit"]][lab.program_sha256]["rmse_j"] == [0.0, 0.0]]
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0].q, cap)
        self.assertEqual(exact[0].q_opt, L.Q_CLIP)

        nonfinite = [lab for lab in labs("nonfinite") if lab.outcome == L.NONFINITE]
        self.assertEqual(len(nonfinite), 2)
        self.assertTrue(all(lab.feas_class == 1 and lab.q is None for lab in nonfinite))

        grids: dict[int, list[int]] = {}
        tie_labs = labs("ties")
        for i, lab in enumerate(tie_labs):
            if lab.q_status == "known":
                grids.setdefault(lab.q_grid, []).append(i)
        tied = [ix for ix in grids.values() if len(ix) >= 2]
        self.assertTrue(tied, "the ties special must hold a grid tie")
        pair_set = {frozenset((w, lo)) for w, lo, _ in L.ranking_pairs(tie_labs)}
        for ix in tied:
            self.assertNotIn(frozenset(ix[:2]), pair_set)
            self.assertEqual(tie_labs[ix[0]].midrank_pct, tie_labs[ix[1]].midrank_pct)

        dctx = table.contexts[special["detailless"]]
        dl = [lab for lab in labs("detailless") if raw[special["detailless"]][lab.program_sha256]["rmse_j"] is None
              and lab.outcome == L.COMPLETED]
        self.assertIn(dctx.q_cap, [lab.q for lab in dl])
        self.assertIn("unknown_detailless_nonfinite", [lab.q_status for lab in dl])

        for csha, recs in table.records.items():
            for lab in recs:
                source = raw[csha][lab.program_sha256]
                if lab.outcome == L.TIMEOUT_OUTCOME:
                    self.assertEqual((lab.feas_class, lab.rt_target, lab.rt_censored), (2, math.log1p(90.0), True))
                if source["backend"] != "local":
                    self.assertFalse(lab.feas_valid or lab.rt_valid)
                if lab.outcome in L.ABANDONED or lab.outcome == L.MISSING:
                    self.assertEqual((lab.labelled, lab.feas_class, lab.q_status), (False, None, "unlabelled"))
        totals = table.counts()
        self.assertEqual(totals["rows"], len(records))
        self.assertGreater(totals["abandoned"], 0)
        self.assertGreater(totals["missing"], 0)
        self.assertEqual(totals["completed"], totals["q_known"] + totals["q_unknown"])


# ============================================================================================== objective identity

class TestObjectiveIdentity(unittest.TestCase):
    """objective_sha256 is a stable function of the declared objective; the LR schedule is a hashed formula."""

    def test_objective_sha256_is_stable(self):
        for name in (O.SUCCESSOR_V1, O.BASELINE_V1):
            first = O.objective_sha256(name)
            self.assertEqual(first, O.objective_sha256(name))
            self.assertEqual(first, common.sha256_bytes(common.canonical(O.identity(name))))
            self.assertEqual(O.identity(name)["schema"], "jope.dl-objective/v1")
            self.assertEqual(O.identity(name)["schedule_id"], O.SCHEDULE_ID)
        self.assertNotEqual(O.objective_sha256(O.SUCCESSOR_V1), O.objective_sha256(O.BASELINE_V1))
        self.assertEqual(O.identity(O.SUCCESSOR_V1)["coefficients"], O.COEF_SUCCESSOR)
        self.assertEqual(O.identity(O.BASELINE_V1)["coefficients"], O.COEF_BASELINE)
        with self.assertRaises(ValueError):
            O.identity("successor-v2")
        with self.assertRaises(ValueError):
            O.objective_sha256("")

    def test_coefficients_and_constants(self):
        self.assertEqual(O.COEF_SUCCESSOR, {"ce": 1.0, "bt": 0.25, "list": 0.10, "q": 0.25, "feas": 0.10, "rt": 0.05})
        self.assertEqual(O.COEF_BASELINE, {"seq_nll": 1.0, "bt": 0.25, "list": 0.10, "rt": 0.10, "cf_bt": 0.05})
        self.assertEqual((O.TEACHER_TAU, O.LIST_TAU, O.HUBER_DELTA, O.BOOST_THRESHOLD, O.BOOST_FACTOR),
                         (0.25, 0.25, 1.0, 0.9, 2.0))
        self.assertEqual(O.LOG1P_TIMEOUT, math.log1p(90.0))
        self.assertEqual(O.SCHEDULE_ID, "warmup_linear_cosine_v1")

    def test_lr_factor_endpoints(self):
        kw = {"warmup": 10, "max_updates": 80, "lr": 3e-3, "lr_min": 3e-4}
        self.assertEqual(O.lr_factor(0, **kw), 1 / 10)
        self.assertEqual(O.lr_factor(9, **kw), 1.0)
        self.assertAlmostEqual(O.lr_factor(10, **kw), 1.0, places=12)
        self.assertEqual(O.lr_factor(80, **kw), 3e-4 / 3e-3)
        self.assertEqual(O.lr_factor(500, **kw), 3e-4 / 3e-3)
        values = [O.lr_factor(e, **kw) for e in range(10, 81)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(O.lr_factor(45, **kw), (3e-4 + 0.5 * (3e-3 - 3e-4)) / 3e-3, places=12)
        self.assertAlmostEqual(O.lr_factor(0, warmup=0, max_updates=100, lr=3e-4, lr_min=3e-5), 1.0, places=12)
        self.assertEqual(O.lr_factor(5, warmup=10, max_updates=10, lr=3e-4, lr_min=3e-5), 0.6)
        self.assertEqual(O.lr_factor(10, warmup=10, max_updates=10, lr=3e-4, lr_min=3e-5), 3e-5 / 3e-4)
        with self.assertRaises(ValueError):
            O.lr_factor(-1, **kw)
        with self.assertRaises(ValueError):
            O.lr_factor(0, warmup=1, max_updates=2, lr=0.0, lr_min=0.0)
        # The trainer's LambdaLR wiring: lr after k scheduler steps is lr * lr_factor(k).
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=kw["lr"], fused=False, foreach=False)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: O.lr_factor(e, **kw))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], kw["lr"] / 10, places=15)
        for _ in range(80):
            opt.step()
            sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], kw["lr_min"], places=15)


if __name__ == "__main__":
    unittest.main()
