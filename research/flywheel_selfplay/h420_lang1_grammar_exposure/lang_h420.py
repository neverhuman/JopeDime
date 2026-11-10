#!/usr/bin/env python3
"""H420 language exposure: what the superset worker already runs but the search grammar never emits.

No Rust change and no change to the frozen grammar file: install() wraps four generator functions of the loaded
learners-v1 grammar module (rand_head, rand_stage, _mutate_params, cost_ok) and _stage_width. Every wrapper returns
the stock result unless extended generation is ON for the calling thread, so stock runs replay bit-for-bit in the
same process (the deploy replays each source in its own thread; see gx()).

Exposed (worker source: feat-math spec.rs / gpr.rs / ops.rs / math_expr.rs):
  heads   gpr kernels rq, rq+linear, rbf+rq, rbf*linear, rbf+periodic, rbf*periodic, rq*linear, rbf+rbf (stock:
          rbf, rbf+linear), steps up to 160 (stock 20/40), m up to 2000 (stock default 500); krr m up to 2000 (stock
          200/400); ksmooth k up to 256; histgbm depth 7-8, rounds to 500, min_leaf / lambda / bins (never set by the
          stock grammar); histgbm2 leaves 63/127, min_leaf, col_subsample 0.5, rounds x bags <= 2400 (stock 1200);
          xtrees / rforest trees to 300; oblivious depth 7-8, rounds to 1200, subsample, rsm; ensemble wider.
  stages  norm_row, row_diff_stats, row_spectral, fft_mag, polar, step, and the symbolic stages map / derive with
          parenthesis-free expressions (x-row_mean, x/row_l2, c3*c7, c1/c4, c2-c5, ...).
The 412-feature screener descriptor is blind to all of these (as it already is to gpr / krr / histgbm2 / the tree
heads); the in-run residual corrector sees the head family only. STAGE_OPS is unchanged, so descriptor widths are too.
"""
from __future__ import annotations

import math
import random
import threading
from typing import Optional

ALL_THREADS = False      # True in a GX search process: every thread generates extended programs
P_EXT_HEAD = 0.5         # share of generated heads drawn from the extended ranges (per rand_head call)
P_EXT_STAGE = 0.2        # share of generated stages replaced by a new stage of the same group
EXT_HISTGBM2_WORK = 2400  # rounds x bags cap for extended histgbm2 heads (stock 1200)
PERIODIC_MAX_WIDTH = 64   # periodic gpr kernels time out on wide tables (probe, width 365): swapped for rq above this
_TL = threading.local()
_G = None
_STOCK = {}

GPR_KERNELS = ("rq", "rq+linear", "rbf+rq", "rbf*linear", "rbf+periodic", "rbf*periodic", "rq*linear", "rbf+rbf")
GPR_KERNELS_ALL = ("rbf", "rbf+linear") + GPR_KERNELS
EXT_HEAD_OPS = frozenset(["gpr", "krr", "ksmooth", "histgbm", "histgbm2", "ensemble", "xtrees", "rforest", "oblivious"])
MUT_HEAD_OPS = frozenset(["gpr", "krr", "ksmooth", "histgbm", "histgbm2", "ensemble"])   # tree heads: stock domains
MAP_EXPRS = ("x-row_mean", "x/row_l2", "x/row_std", "x/row_max", "x/row_l1", "x-row_min", "x*x", "x^3", "x*row_mean")
DERIVE_FORMS = ("c{a}*c{b}", "c{a}/c{b}", "c{a}-c{b}", "c{a}+c{b}", "c{a}*c{b}*c{c}", "c{a}*c{b}/c{c}",
                "c{a}/row_mean", "c{a}/row_l2", "c{a}^2", "c{a}-row_mean")
DERIVE_MAX_COL = 12
NEW_STAGE_OPS = frozenset(["norm_row", "row_diff_stats", "row_spectral", "fft_mag", "polar", "step", "map", "derive"])
NEW_BY_GROUP = {"scale": ("norm_row", "map"), "projection": ("row_diff_stats", "row_spectral", "fft_mag", "polar"),
                "interaction": ("derive", "derive", "map"), "nonlinear": ("step", "map"), "kernel_map": ()}
EXT_DOMAINS = {
    ("gpr", "steps"): (10, 160, "i"), ("gpr", "m"): (50, 2000, "i"), ("krr", "m"): (50, 2000, "i"),
    ("ksmooth", "gamma"): (0.05, 32.0, "f"), ("ksmooth", "k"): (2, 256, "i"),
    ("histgbm", "depth"): (2, 8, "i"), ("histgbm", "lr"): (0.01, 0.3, "f"), ("histgbm", "rounds"): (40, 500, "i"),
    ("histgbm", "min_leaf"): (1, 60, "i"), ("histgbm", "lambda"): (0.01, 100.0, "f"), ("histgbm", "bins"): (16, 255, "i"),
    ("histgbm2", "leaves"): (3, 127, "i"), ("histgbm2", "bags"): (1, 5, "i"), ("histgbm2", "lr"): (0.01, 0.2, "f"),
    ("histgbm2", "rounds"): (60, 1000, "i"), ("histgbm2", "row_subsample"): (0.5, 1.0, "f"),
    ("histgbm2", "col_subsample"): (0.5, 1.0, "f"), ("histgbm2", "min_leaf"): (1, 20, "i"),
    ("histgbm2", "val_pct"): (5, 40, "i"),
    ("ensemble", "depth"): (2, 8, "i"), ("ensemble", "lambda_frac"): (1e-4, 1.0, "f"),
    ("ensemble", "lr"): (0.01, 0.3, "f"), ("ensemble", "rounds"): (40, 500, "i"),
}
# Library extension (deploy Tier B): fixed presets at the exposed ranges, declared before any H420 evaluation. The
# stock library has two kernel presets (gpr_std, krr_blom); these add the unexposed kernel family and stage classes.
LIB_EXT = (
    ("gpr_rq", "(pipe X (standardize) (gpr kernel=rq m=2000 steps=120))"),
    ("gpr_rq_lin", "(pipe X (standardize) (gpr kernel=rq+linear m=2000 steps=120))"),
    ("gpr_rbf_m2k", "(pipe X (standardize) (gpr kernel=rbf+linear m=2000 steps=120))"),
    ("gpr_blom_rq", "(pipe X (blom) (gpr kernel=rq+linear m=2000 steps=120))"),
    ("gpr_rbf_rq", "(pipe X (standardize) (gpr kernel=rbf+rq m=1200 steps=120))"),
    ("gpr_per", "(pipe X (standardize) (gpr kernel=rbf+periodic m=1200 steps=120))"),
    ("gpr_rbfxlin", "(pipe X (standardize) (gpr kernel=rbf*linear m=1200 steps=120))"),
    ("gpr_sel16", "(pipe X (standardize) (select_corr k=16) (gpr kernel=rq+linear m=2000 steps=120))"),
    ("gpr_svd8", "(pipe X (standardize) (svd_project k=8) (gpr kernel=rq+linear m=2000 steps=120))"),
    ("gpr_normrow", "(pipe X (standardize) (norm_row append=true) (gpr kernel=rq+linear m=1200 steps=80))"),
    ("krr_std_2k", "(pipe X (standardize) (krr m=2000))"),
    ("krr_blom_2k", "(pipe X (blom) (krr m=2000))"),
    ("ksm_std_k8", "(pipe X (standardize) (ksmooth gamma=1.0 k=8))"),
    ("ksm_blom_k64", "(pipe X (blom) (ksmooth gamma=4.0 k=64))"),
    ("hgb_d8_ml5", "(pipe X (histgbm bins=255 depth=8 lr=0.03 min_leaf=5 rounds=500))"),
    ("hgb_d6_l10", "(pipe X (histgbm depth=6 lambda=10.0 lr=0.05 min_leaf=20 rounds=500))"),
    ("svd_hgb2", "(pipe X (svd_project append=true k=8) (histgbm2 bags=5 leaves=31 lr=0.03 rounds=1000 "
                 "row_subsample=0.85))"),
    ("rp_obl", "(pipe X (rand_proj append=true k=16 seed=701) (oblivious depth=6 l2=3.0 lr=0.03 rounds=1500))"),
    ("rowc_hgb2", "(pipe X (row_stats append=true) (map append=true expr=x-row_mean) (histgbm2 bags=5 leaves=31 "
                  "lr=0.03 rounds=1000 row_subsample=0.85))"),
    ("spline_ridge", "(pipe X (blom) (spline append=true knots=8) (ridge_gcv))"),
)


def active() -> bool:
    return ALL_THREADS or getattr(_TL, "on", False)


class gx:
    """Context manager: extended generation ON for the calling thread (a GX source's replay)."""

    def __enter__(self):
        self.prev = getattr(_TL, "on", False)
        _TL.on = True

    def __exit__(self, *exc):
        _TL.on = self.prev


def _r6(x: float) -> float:
    return float(f"{x:.6g}")


def _logu(rng: random.Random, lo: float, hi: float) -> float:
    return _r6(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def ext_head(op: str, rng: random.Random):
    S = _G.Stage.make
    if op == "gpr":
        kw = dict(kernel=rng.choice(GPR_KERNELS), steps=rng.choice([40, 80, 120]))
        m = rng.choice([0, 0, 800, 1200, 2000])
        if m:
            kw["m"] = m
        return S(op, **kw)
    if op == "krr":
        return S(op, m=rng.choice([600, 800, 1200, 2000]))
    if op == "ksmooth":
        return S(op, gamma=rng.choice([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]), k=rng.choice([4, 8, 16, 32, 64, 128, 256]))
    if op == "histgbm":
        kw = dict(depth=rng.randint(4, 8), lr=_logu(rng, 0.015, 0.15), rounds=rng.choice([200, 300, 400, 500]))
        if rng.random() < 0.5:
            kw["min_leaf"] = rng.choice([3, 5, 20, 40])
        if rng.random() < 0.3:
            kw["lambda"] = rng.choice([0.1, 3.0, 10.0])
        if rng.random() < 0.3:
            kw["bins"] = rng.choice([32, 128, 255])
        return S(op, **kw)
    if op == "histgbm2":
        bags = rng.choice([1, 2, 3, 5])
        rounds = rng.choice([r for r in (250, 400, 600, 800) if r * bags <= EXT_HISTGBM2_WORK])
        kw = dict(bags=bags, leaves=rng.choice([15, 31, 63, 127]), lr=_logu(rng, 0.015, 0.1), rounds=rounds,
                  row_subsample=rng.choice([0.65, 0.85, 1.0]))
        if rng.random() < 0.4:
            kw["col_subsample"] = rng.choice([0.5, 0.6, 0.8])
        if rng.random() < 0.4:
            kw["min_leaf"] = rng.choice([1, 5, 10, 20])
        if rng.random() < 0.2:
            kw["val_pct"] = rng.choice([10, 15, 25])
        return S(op, **kw)
    if op in ("xtrees", "rforest"):
        return S(op, trees=rng.choice([150, 250, 300]), min_leaf=rng.choice([1, 2, 3, 5, 8, 16]),
                 max_features=rng.choice([0.2, 0.33, 0.5, 0.7, 1.0]))
    if op == "oblivious":
        kw = dict(depth=rng.choice([5, 6, 7, 8]), lr=_logu(rng, 0.02, 0.15), rounds=rng.choice([300, 500, 800, 1200]),
                  l2=rng.choice([1.0, 3.0, 10.0, 30.0]))
        if rng.random() < 0.4:
            kw["subsample"] = rng.choice([0.7, 0.85])
        if rng.random() < 0.4:
            kw["rsm"] = rng.choice([0.5, 0.8])
        return S(op, **kw)
    if op == "ensemble":
        return S(op, depth=rng.randint(3, 6), lambda_frac=rng.choice([0.002, 0.02, 0.1, 0.3]),
                 lr=_logu(rng, 0.03, 0.15), rounds=rng.choice([160, 240, 320]))
    return None


def ext_stage(op: str, rng: random.Random):
    S = _G.Stage.make
    app = rng.random() < 0.75
    kw = {}
    if op == "fft_mag":
        kw["k"] = rng.choice([4, 8, 16])
    elif op == "step":
        kw["t"] = rng.choice([-0.5, 0.0, 0.5])
    elif op == "map":
        kw["expr"] = rng.choice(MAP_EXPRS)
    elif op == "derive":
        a, b, c = rng.sample(range(DERIVE_MAX_COL), 3)
        kw["expr"] = rng.choice(DERIVE_FORMS).format(a=a, b=b, c=c)
        app = True
    elif op in ("row_diff_stats", "row_spectral", "polar"):
        app = True
    elif op != "norm_row":
        raise _G.GrammarError(f"unknown extended stage op {op}")
    if app:
        kw["append"] = True
    return S(op, **kw)


def _group_of(op: str) -> Optional[str]:
    for g, ops in _STOCK["groups"].items():
        if op in ops:
            return g
    return None


def rand_head(op, rng):
    if active() and op in EXT_HEAD_OPS and rng.random() < P_EXT_HEAD:
        h = ext_head(op, rng)
        if h is not None:
            return h
    return _STOCK["rand_head"](op, rng)


def rand_stage(op, rng):
    if op in NEW_STAGE_OPS:
        return ext_stage(op, rng)
    if active() and rng.random() < P_EXT_STAGE:
        new = NEW_BY_GROUP.get(_group_of(op) or "", ())
        if new:
            return ext_stage(rng.choice(new), rng)
    return _STOCK["rand_stage"](op, rng)


def _mutate_params(s, rng):
    if s.op in NEW_STAGE_OPS:
        if active() and s.op in ("map", "derive") and rng.random() < 0.5:
            return ext_stage(s.op, rng)
        return s
    if not (active() and s.op in MUT_HEAD_OPS):
        return _STOCK["_mutate_params"](s, rng)
    upd = {}
    for key, val in s.params:
        if isinstance(val, bool):
            continue
        if isinstance(val, str):
            if s.op == "gpr" and key == "kernel" and rng.random() < 0.25:
                upd[key] = rng.choice(GPR_KERNELS_ALL)
            continue
        dom = EXT_DOMAINS.get((s.op, key))
        if dom is None:
            continue
        lo, hi, kind = dom
        if kind == "f":
            upd[key] = _r6(min(max(float(val) * math.exp(rng.uniform(-0.30, 0.30)), lo), hi))
        else:
            upd[key] = int(min(max(int(round(int(val) * rng.uniform(0.75, 1.30))) + rng.choice([-1, 0, 1]), lo), hi))
    return s.with_params(**upd) if upd else s


def cost_ok(g) -> bool:
    if not active():
        return _STOCK["cost_ok"](g)
    G = _G
    g = G.parse(g) if isinstance(g, str) else g
    pipes = g.members if isinstance(g, G.Portfolio) else (g,)
    if isinstance(g, G.Portfolio) and len(pipes) > G.MAX_PORTFOLIO_GEN_MEMBERS:
        return False
    for p in pipes:
        h = p.head
        if h is None:
            continue
        if h.op == "stack":
            return False
        if isinstance(g, G.Portfolio) and h.op in G.EXPENSIVE_MEMBER_HEADS:
            return False
        if h.op == "histgbm2":
            work = int(h.get("rounds", 500)) * int(h.get("bags", 3))
            if work > (EXT_HISTGBM2_WORK // 2 if isinstance(g, G.Portfolio) else EXT_HISTGBM2_WORK):
                return False
    return True


def cap_gpr(g, width: int):
    """Stock gpr cost cap, then: a periodic kernel primitive becomes rq on tables wider than PERIODIC_MAX_WIDTH.
    Stock programs never carry a periodic kernel, so every stock program passes through unchanged."""
    G = _G
    g = _STOCK["cap_gpr"](g, width)
    if width <= PERIODIC_MAX_WIDTH:
        return g

    def fix(p):
        h = p.head
        if h is not None and h.op == "gpr" and "periodic" in str(h.get("kernel", "")):
            return G.Pipe(p.transforms + (h.with_params(kernel=str(h.get("kernel")).replace("periodic", "rq")),))
        return p

    if isinstance(g, G.Portfolio):
        return G.Portfolio(tuple(fix(m) for m in g.members), g.folds, g.repeats_auto)
    return fix(g)


def _stage_width(s, w: int) -> int:
    if s.op == "derive":
        return w + 1 if s.get("append", False) is True else 1
    if s.op == "fft_mag":
        k = int(s.get("k", 8))
        return w + k if s.get("append", False) is True else k
    return _STOCK["_stage_width"](s, w)


def featurize(dsl: str):
    """The Rust 412-feature featurizer may not know a new stage op: fall back to the program without those stages."""
    G = _G
    hit = G._FEAT_CACHE.get(dsl)
    if hit is not None:
        return hit
    try:
        return _STOCK["featurize"](dsl)
    except Exception:
        g = G.parse(dsl)

        def strip(p):
            return G.Pipe(tuple(s for s in p.stages if s.op not in NEW_STAGE_OPS))

        g2 = G.Portfolio(tuple(strip(m) for m in g.members), g.folds, g.repeats_auto) if isinstance(g, G.Portfolio) \
            else strip(g)
        hit = _STOCK["featurize"](G.canonical(g2))
        G._FEAT_CACHE[dsl] = hit
        return hit


def install(G, all_threads: bool = False) -> None:
    """Wrap the generators of the loaded grammar module (idempotent). all_threads=True: a GX search process."""
    global _G, ALL_THREADS
    if _G is None:
        _G = G
        _STOCK.update(rand_head=G.rand_head, rand_stage=G.rand_stage, _mutate_params=G._mutate_params,
                      cost_ok=G.cost_ok, _stage_width=G._stage_width, featurize=G.featurize, cap_gpr=G.cap_gpr,
                      groups=dict(G.STAGE_GROUPS))
        G.rand_head, G.rand_stage, G._mutate_params = rand_head, rand_stage, _mutate_params
        G.cost_ok, G._stage_width, G.featurize, G.cap_gpr = cost_ok, _stage_width, featurize, cap_gpr
    elif _G is not G:
        raise RuntimeError("lang_h420 is installed on another grammar module")
    ALL_THREADS = bool(all_threads)


def selfcheck(G, n: int = 3000) -> dict:
    """With the wrappers installed and generation OFF, the generators emit exactly the stock stream; with generation
    ON, every program round-trips the parser. Library presets are canonical."""
    names = ("rand_head", "rand_stage", "_mutate_params", "cost_ok", "_stage_width", "featurize", "cap_gpr")
    wrapped = {k: getattr(G, k) for k in names}

    def stream(seed):
        rng, out = random.Random(seed), []
        for i in range(n):
            g = G.op_synth(i % 4, rng)
            if i % 3 == 0:
                g = G.op_parameter(g, rng)
            if i % 5 == 0:
                g = G.op_insert_stage(g, rng)
            if i % 7 == 0:
                g = G.op_replace_head(g, rng)
            out.append((G.canonical(g), G.cost_ok(g), G.expanded_width(g, 20)))
        return out

    off = stream(7)
    for k in names:
        setattr(G, k, _STOCK[k])
    try:
        stock = stream(7)
    finally:
        for k in names:
            setattr(G, k, wrapped[k])
    out = {"stock_stream_equal": off == stock, "n": n}
    with gx():
        ext = stream(7)
    ops, kern, rt = {}, {}, 0
    for d, _c, _w in ext:
        g = G.parse(d)
        rt += 1 if G.canonical(g) == d else 0
        for p in (g.members if isinstance(g, G.Portfolio) else (g,)):
            for s in p.stages:
                if s.op in NEW_STAGE_OPS:
                    ops[s.op] = ops.get(s.op, 0) + 1
            if p.head is not None and p.head.op == "gpr":
                k = str(p.head.get("kernel", "rbf"))
                kern[k] = kern.get(k, 0) + 1
    out.update(ext_roundtrip=rt, ext_differs=sum(a != b for a, b in zip(ext, stock)), new_stage_counts=ops,
               gpr_kernels=kern, ext_cost_ok=sum(c for _d, c, _w in ext))
    for name, d in LIB_EXT:
        if G.canonical(G.parse(d)) != d:
            raise AssertionError(f"library preset not canonical: {name}")
    out["lib_ext"] = len(LIB_EXT)
    if not out["stock_stream_equal"] or rt != n:
        raise AssertionError(f"lang_h420 selfcheck failed: {out}")
    return out
