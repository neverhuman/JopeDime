#!/usr/bin/env python3
"""H410 deploy-the-population unit (PREREG §3): ensembles over the STORED program population of a context.

A unit replays every stored source run of one context from the memo (H335 Stage V and H381 E2L on dev91; H364 X0 and
H381 E2L / E2R on TRAIN). The replay re-runs the frozen AGG recipe (seed A/E/F, panmictic evo to g50) with the run's
own screener, and every evaluation is a memo hit, so a replay costs 0 worker runs. It reproduces each run's
SearchResults (with the stored out-of-fold `p0`/`p1`) and its stock CV3@12 output. The builders below are then fit
on SEARCH data only, inside a selection scope, and each output is AUDIT-scored exactly once under its own pipeline
name (AuditLedger rule). Members that were already audited are memo hits; only new members cost an AUDIT refit.

Builders (StackSpec unless noted):
  stock    ST.cv3_s1direct(N=12)                             = the stored deployed output (replay check)
  W24/W48  cv3_members_w (copy of ST.cv3_members with a fitness window argument) N=24/48, window 0.5, stock
           weights_s1_direct
  CAR      Caruana greedy forward selection with replacement on the out-of-fold predictions of the top 200 passing
           programs (objective = mean over the 2 SEARCH folds of ln(rmse / fold baseline)), 100 steps, weight =
           pick count
  NNLS     non-negative least squares + free intercept on the stacked out-of-fold predictions of the top 48 (linear
           output, own AUDIT path)
  BAG(x)   x's members deployed as bagged twins (histgbm/histgbm2 -> bags, rforest/xtrees -> 500 trees), same weights
  RUNBAG   uniform mean of the stock outputs' AUDIT predictions over the replayed (source, seed) runs (0 worker runs)
Library stage (PREREG §3 Tier B; optional): LIB presets + target-transform variants + polish neighbours, SEARCH-
evaluated once per context (memo shared across seeds), then the builders over pool ∪ library.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gp_h381 as GPH  # noqa: E402  (FIRST: h335_core pins + guard; S3 registered)

import numpy as np  # noqa: E402

K, C, LAB, L, D, ST, G, E = GPH.K, GPH.C, GPH.LAB, GPH.L, GPH.D, GPH.ST, GPH.G, GPH.E
import screener_h317 as S  # noqa: E402  (pinned H317 module)

WAVE = "H410"
GENS = 50
SCHEMA = "jope.h410.deploy-unit/v1"
WINDOW = 0.5
CAR_POOL, CAR_STEPS = 200, 100
NNLS_TOP = 48

# ----------------------------------------------------------------------------------------------------------------------
# Library (Tier B): fixed strong presets at the superset worker's upper ranges (declared before any H410 eval)
# ----------------------------------------------------------------------------------------------------------------------
LIB_PRESETS = (
    ("obl_d6_1500", "(pipe X (oblivious depth=6 l2=3.0 lr=0.03 rounds=1500))"),
    ("obl_d8_1000", "(pipe X (oblivious depth=8 l2=10.0 lr=0.03 rounds=1000))"),
    ("obl_d6_sub", "(pipe X (oblivious depth=6 l2=3.0 lr=0.05 rounds=1000 rsm=0.7 subsample=0.8))"),
    ("obl_d4_2000", "(pipe X (oblivious depth=4 l2=1.0 lr=0.05 rounds=2000))"),
    ("hgb2_l15_b5", "(pipe X (histgbm2 bags=5 leaves=15 lr=0.03 rounds=1000 row_subsample=0.85))"),
    ("hgb2_l31_b5", "(pipe X (histgbm2 bags=5 col_subsample=0.7 leaves=31 lr=0.03 rounds=1000 row_subsample=0.85))"),
    ("hgb2_l63_b8", "(pipe X (histgbm2 bags=8 leaves=63 lr=0.02 rounds=1000 row_subsample=0.8))"),
    ("hgb2_l7_b5", "(pipe X (histgbm2 bags=5 leaves=7 lr=0.05 rounds=1000 row_subsample=0.85))"),
    ("hgb2_huber", "(pipe X (histgbm2 bags=5 leaves=31 loss=huber lr=0.03 rounds=1000 row_subsample=0.85))"),
    ("rf_mf033", "(pipe X (rforest max_features=0.33 min_leaf=1 trees=500))"),
    ("rf_mf06", "(pipe X (rforest max_features=0.6 min_leaf=2 trees=500))"),
    ("rf_mf1_l5", "(pipe X (rforest max_features=1.0 min_leaf=5 trees=500))"),
    ("xt_mf1", "(pipe X (xtrees max_features=1.0 min_leaf=1 trees=500))"),
    ("xt_mf05_l3", "(pipe X (xtrees max_features=0.5 min_leaf=3 trees=500))"),
    ("gpr_std", "(pipe X (standardize) (gpr kernel=rbf+linear steps=80))"),
    ("krr_blom", "(pipe X (blom) (krr m=400))"),
    ("rank_ridge", "(pipe X (rank) (ridge_gcv))"),
    ("pp_hgb2", "(pipe X (pair_prod append=true topk=24) (histgbm2 bags=5 leaves=31 lr=0.03 rounds=1000 "
                "row_subsample=0.85))"),
    ("cheb_obl", "(pipe X (cheb append=true degree=3) (oblivious depth=6 l2=3.0 lr=0.03 rounds=1500))"),
    ("pr_obl", "(pipe X (pair_ratio append=true topk=12) (oblivious depth=6 l2=3.0 lr=0.03 rounds=1500))"),
)
TARGET_PRESET_BASES = ("obl_d6_1500", "obl_d8_1000", "hgb2_l15_b5", "hgb2_l31_b5", "rf_mf033", "xt_mf1")
TARGET_KINDS = ("log1p", "rank_gauss")
TOP_TARGET, TOP_POLISH, MAX_POLISH = 16, 8, 64


def _r6(x: float) -> float:
    return float(f"{x:.6g}")


def _cand(g, op: str, width: int, parents: Sequence[str] = ()) -> Optional["G.Candidate"]:
    return G.safe_candidate(g, op, tuple(parents), width, cost_exempt=True)


def _pipe(dsl: str):
    try:
        g = G.parse(dsl)
    except G.GrammarError:
        return None
    return g if isinstance(g, G.Pipe) and g.head is not None else None


def with_target(dsl: str, kind: str):
    g = _pipe(dsl)
    if g is None:
        return None
    rest = g.stages[1:] if g.stages and g.stages[0].op == "target" else g.stages
    st = G.Stage.make("target", kind=kind, q=0.05) if kind == "winsor" else G.Stage.make("target", kind=kind)
    return G.Pipe((st,) + tuple(rest))


def scale_up(dsl: str) -> List:
    """Polish neighbours: the head scaled toward more capacity at a lower rate, plus the grammar's line search."""
    g = _pipe(dsl)
    if g is None:
        return []
    h, out = g.head, []
    t = tuple(g.transforms)
    if h.op in ("histgbm", "gbm"):
        r, lr = int(h.get("rounds", 150)), float(h.get("lr", 0.1))
        out.append(G.Pipe(t + (h.with_params(rounds=min(500, 2 * r), lr=_r6(max(lr / 2, 0.005))),)))
        b = G.bag_final(g)
        if b is not None:
            out.append(b)
    elif h.op == "histgbm2":
        r, lr, b = int(h.get("rounds", 500)), float(h.get("lr", 0.05)), int(h.get("bags", 1))
        out.append(G.Pipe(t + (h.with_params(rounds=min(1000, 2 * r), lr=_r6(max(lr / 2, 0.005))),)))
        out.append(G.Pipe(t + (h.with_params(bags=min(10, max(5, 2 * b)), row_subsample=0.85),)))
    elif h.op == "oblivious":
        r, lr = int(h.get("rounds", 300)), float(h.get("lr", 0.05))
        out.append(G.Pipe(t + (h.with_params(rounds=min(2000, 2 * r), lr=_r6(max(lr / 2, 0.005))),)))
    elif h.op in ("xtrees", "rforest"):
        out.append(G.Pipe(t + (h.with_params(trees=500),)))
    out.extend(G.line_search_neighbors(g)[:4])
    return out


def twin_genome(dsl: str):
    """Bagged deployment twin of a member (None = deploy the member unchanged)."""
    g = _pipe(dsl)
    if g is None:
        return None
    h, t = g.head, tuple(g.transforms)
    if h.op in ("histgbm", "gbm"):
        return G.bag_final(g)
    if h.op == "histgbm2":
        return G.Pipe(t + (h.with_params(bags=min(10, max(5, 2 * int(h.get("bags", 1)))), row_subsample=0.85),))
    if h.op in ("xtrees", "rforest"):
        return G.Pipe(t + (h.with_params(trees=500),)) if int(h.get("trees", 100)) < 500 else None
    return None


def library_candidates(pool: Sequence["E.SearchResult"], width: int) -> List["G.Candidate"]:
    out: Dict[str, G.Candidate] = {}

    def add(gm, op, parents=()):
        c = _cand(gm, op, width, parents) if gm is not None else None
        if c is not None:
            out.setdefault(c.sha, c)

    for n, d in LIB_PRESETS:
        add(_pipe(d), f"h410_lib:{n}")
    base = dict(LIB_PRESETS)
    for n in TARGET_PRESET_BASES:
        for k in TARGET_KINDS:
            add(with_target(base[n], k), f"h410_libtgt:{n}:{k}")
    top = ST.passing_sorted(pool)
    for r in top[:TOP_TARGET]:
        for k in TARGET_KINDS + ("winsor",):
            add(with_target(r.dsl, k), f"h410_tgt:{k}", (r.sha,))
    n_pol = 0
    for r in top[:TOP_POLISH]:
        for gm in scale_up(r.dsl):
            if n_pol >= MAX_POLISH:
                break
            before = len(out)
            add(gm, "h410_polish", (r.sha,))
            n_pol += len(out) - before
    return list(out.values())


# ----------------------------------------------------------------------------------------------------------------------
# Builders (SEARCH data only)
# ----------------------------------------------------------------------------------------------------------------------
def cv3_members_w(results, n: int, fd, window: float):
    """ST.cv3_members verbatim except FITNESS_WINDOW -> `window` (no global patch: units run in threads)."""
    ps = ST.passing_sorted(results)
    if not ps:
        return []
    fmin = ps[0].fitness
    pool = [r for r in ps if r.fitness <= fmin + window]
    res = {r.sha: ST.residuals(r, fd) for r in pool}

    def corr(a, b):
        sa, sb = np.std(a), np.std(b)
        return 1.0 if sa < 1e-12 or sb < 1e-12 else float(np.corrcoef(a, b)[0, 1])

    def compatible(r, chosen):
        return all(corr(res[r.sha], res[c.sha]) < ST.RHO_MAX for c in chosen)

    chosen = []
    for r in pool:
        if len(chosen) >= n:
            break
        if compatible(r, chosen):
            chosen.append(r)
    eligible_isl = {G.island_of(r.dsl) for r in pool}
    target = min(3, len(eligible_isl), n)
    guard = 0
    while len({G.island_of(c.dsl) for c in chosen}) < target and guard < 12:
        guard += 1
        have = {G.island_of(c.dsl) for c in chosen}
        add = next((r for r in pool if G.island_of(r.dsl) not in have and r not in chosen
                    and compatible(r, [c for c in chosen])), None)
        if add is None:
            add = next((r for r in pool if G.island_of(r.dsl) not in have and r not in chosen), None)
        if add is None:
            break
        if len(chosen) < n:
            chosen.append(add)
            continue
        counts = {i: sum(G.island_of(c.dsl) == i for c in chosen) for i in have}
        victims = [c for c in reversed(chosen) if counts[G.island_of(c.dsl)] >= 2]
        if not victims:
            break
        chosen.remove(victims[0])
        chosen.append(add)
    return sorted(chosen, key=lambda r: (r.fitness, r.sha))


def b_wide(res, fd, n: int, window: float = WINDOW):
    mem = cv3_members_w(res, n, fd, window)
    if len(mem) < 2:
        return ST.single_spec(res, f"W{n}")
    P0, P1, _, comps = ST._mats(mem)
    w0, w1, wa = ST.weights_s1_direct(mem, fd)
    return ST._spec(f"W{n}", mem, w0, w1, wa, fd, P0, P1, comps, n_requested=n, window=window)


def b_caruana(res, fd, k: int = CAR_POOL, steps: int = CAR_STEPS):
    ps = ST.passing_sorted(res)[:k]
    if len(ps) < 2:
        return ST.single_spec(res, "CAR")
    P0, P1, _, _ = ST._mats(ps)
    s0, s1 = np.zeros(len(fd.y0)), np.zeros(len(fd.y1))
    cnt = np.zeros(len(ps))
    for t in range(1, steps + 1):
        r0 = np.sqrt((((s0[:, None] + P0) / t - fd.y0[:, None]) ** 2).mean(0))
        r1 = np.sqrt((((s1[:, None] + P1) / t - fd.y1[:, None]) ** 2).mean(0))
        obj = 0.5 * (np.log(np.maximum(r0, 1e-300) / fd.b0) + np.log(np.maximum(r1, 1e-300) / fd.b1))
        j = int(np.argmin(obj))
        cnt[j] += 1
        s0 += P0[:, j]
        s1 += P1[:, j]
    keep = np.flatnonzero(cnt > 0)
    mem = [ps[i] for i in keep]
    w = cnt[keep] / cnt.sum()
    if len(mem) == 1:
        return ST.single_spec(mem, "CAR")
    P0m, P1m, _, cm = ST._mats(mem)
    return ST._spec("CAR", mem, w, w, w, fd, P0m, P1m, cm, n_pool=len(ps), steps=steps)


def _nnls(A: np.ndarray, b: np.ndarray, iters: int = 400) -> np.ndarray:
    """Lawson-Hanson active-set NNLS (numpy only)."""
    n = A.shape[1]
    x, P = np.zeros(n), np.zeros(n, bool)
    w = A.T @ (b - A @ x)
    for _ in range(iters):
        if P.all() or w[~P].max(initial=-np.inf) <= 1e-10:
            break
        j = int(np.argmax(np.where(P, -np.inf, w)))
        P[j] = True
        while True:
            z = np.zeros(n)
            z[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (z[P] > 0).all():
                x = z
                break
            neg = P & (z <= 0)
            alpha = np.min(x[neg] / np.maximum(x[neg] - z[neg], 1e-300))
            x = x + alpha * (z - x)
            P &= x > 1e-12
            x[~P] = 0.0
        w = A.T @ (b - A @ x)
    return x


def b_nnls(res, fd, k: int = NNLS_TOP) -> Optional[dict]:
    ps = ST.passing_sorted(res)[:k]
    if len(ps) < 2:
        return None
    P = np.vstack([np.column_stack([r.p0 for r in ps]), np.column_stack([r.p1 for r in ps])])
    y = np.concatenate([fd.y0, fd.y1])
    pm, ym = P.mean(0), y.mean()
    w = _nnls(P - pm, y - ym)
    keep = np.flatnonzero(w > 1e-9)
    if len(keep) == 0:
        return None
    return {"members": [ps[i].sha for i in keep], "w": w[keep], "intercept": float(ym - pm[keep] @ w[keep]),
            "n_pool": len(ps)}


def twin_spec(spec, cand_of: Dict[str, "G.Candidate"], width: int):
    """`spec` with each member replaced by its bagged twin (weights unchanged). Adds the twins to cand_of."""
    shas, dsls, w = spec.declared()
    new_m, new_d, n_tw = [], [], 0
    for s, d in zip(shas, dsls):
        c = _cand(twin_genome(d), "h410_twin", width, (s,))
        if c is not None:
            cand_of.setdefault(c.sha, c)
            new_m.append(c.sha)
            new_d.append(c.dsl)
            n_tw += 1
        else:
            new_m.append(s)
            new_d.append(d)
    w = np.asarray(w, np.float64)
    return ST.StackSpec(f"BAG({spec.config})", new_m, new_d, [G.island_of(d) for d in new_d], w, w, w, {},
                        info={"twinned": n_tw, "of": spec.config}), n_tw


# ----------------------------------------------------------------------------------------------------------------------
# AUDIT (one request per pipeline name)
# ----------------------------------------------------------------------------------------------------------------------
def audit_spec(rt, pipe: str, g, spec, cand_of) -> Tuple[dict, Optional[np.ndarray]]:
    shas, _, _ = spec.declared()
    res = rt.ledger.score(pipe, g.ctx, g.seed, [cand_of[s] for s in shas])
    out = ST.stack_audit_utility(spec, {a.sha: a for a in res}, g.ctx.ya, g.ctx.ba, g.ctx.y_search_mean,
                                 return_pred=True)
    pred = out.pop("pred")
    out["n_members"] = len(shas)
    return out, pred


def audit_linear(rt, pipe: str, g, lin: dict, cand_of) -> Tuple[dict, Optional[np.ndarray]]:
    res = rt.ledger.score(pipe, g.ctx, g.seed, [cand_of[s] for s in lin["members"]])
    by = {a.sha: a for a in res}
    ok = [by[s].status == "completed" and by[s].pa is not None for s in lin["members"]]
    cols = [by[s].pa if good else np.full(len(g.ctx.ya), g.ctx.y_search_mean) for s, good in zip(lin["members"], ok)]
    pred = lin["intercept"] + np.column_stack(cols) @ lin["w"]
    rmse = float(np.sqrt(np.mean((pred - g.ctx.ya) ** 2)))
    return {"status": "completed", "utility": C.audit_utility_from_rmse(rmse, g.ctx.ba), "rmse": rmse,
            "n_members": len(lin["members"]), "n_mean_substituted": int(sum(not o for o in ok)),
            "intercept": lin["intercept"], "w_sum": float(np.sum(lin["w"]))}, pred


def pred_utility(pred: np.ndarray, g) -> dict:
    rmse = float(np.sqrt(np.mean((pred - g.ctx.ya) ** 2)))
    return {"status": "completed", "utility": C.audit_utility_from_rmse(rmse, g.ctx.ba), "rmse": rmse}


# ----------------------------------------------------------------------------------------------------------------------
# Unit
# ----------------------------------------------------------------------------------------------------------------------
def replay(g, gens: int = GENS):
    a = D.seed_A(g)
    e = D.seed_E(g, a)
    f = K.seed_F(g, e)
    evo = D.evo_panmictic(g, f.order, f.results, gens=gens, n_prop=K.N_PROP, n_live=K.N_LIVE, arm="evoA")
    cand_of = {**{c.sha: c for c in f.order}, **evo.cand_of}
    return LAB._upto(evo, gens), cand_of


def _gp(rt, cid: str, screener, seed):
    ctx = rt.engine.context(cid, rt.fam_of[cid])
    return D.GPContext(ctx, rt.engine, screener, rt.target_bank(cid), rt.coder, seed)


def _summ(raw: dict, spec=None, extra: Optional[dict] = None) -> dict:
    u = raw.get("utility")
    o = {"value": float(u) if u is not None and raw.get("status") == "completed" else 0.0,
         "invalid": bool(u is None or raw.get("all_substituted", False)), "raw": raw}
    if spec is not None:
        o["spec"] = spec.to_json() if hasattr(spec, "to_json") else spec
    if extra:
        o.update(extra)
    return o


def deploy_unit(rt, tag: str, cid: str, sources: Dict[str, object], seeds: Sequence[int], library: bool,
                own_source: Optional[str], pools: Dict[str, Sequence[str]], jobs: int = 4) -> dict:
    """sources: name -> screener_for(cid). pools: pool name -> source names whose replays form it."""
    stage = f"H410_{tag}"
    hit = rt.store.result(stage, cid, "pool")
    if hit:
        return hit
    t0 = time.time()
    keys = [(n, s) for n in sources for s in seeds]
    gs = {(n, s): _gp(rt, cid, sources[n](cid), s) for n, s in keys}
    with cf.ThreadPoolExecutor(max_workers=max(1, min(jobs, len(keys)))) as ex:
        reps = dict(zip(keys, ex.map(lambda k: replay(gs[k]), keys)))
    cand_of: Dict[str, G.Candidate] = {}
    for _, co in reps.values():
        cand_of.update(co)
    g0 = gs[keys[0]]
    fd = ST.FoldData.from_ctx(g0.ctx)
    width = g0.ctx.width
    gpool = D.GPContext(g0.ctx, rt.engine, g0.screener, g0.tb, g0.coder, 0)
    rec = {"type": "result", "schema": SCHEMA, "wave": WAVE, "tag": tag, "cid": cid, "seed": "pool",
           "seeds": list(seeds), "sources": list(sources), "ba": g0.ctx.ba, "y_search_mean": g0.ctx.y_search_mean,
           "env_sha": dict(g0.ctx.env_sha), "outputs": {}, "pool_sizes": {}}
    out = rec["outputs"]
    preds: Dict[str, np.ndarray] = {}

    # stock outputs of every replayed run (= the stored deployed outputs) + RUNBAG
    for (n, s), (res, _) in reps.items():
        with D.selection_scope(f"{stage}.{n}.{s}.stock"):
            spec = ST.cv3_s1direct(res, fd, 12)
        raw, pred = audit_spec(rt, f"{stage}.{n}.s{s}.stock", gs[(n, s)], spec, cand_of)
        out[f"{n}.s{s}.stock"] = _summ(raw, spec, {"n_pool": len(res)})
        if pred is not None:
            preds[f"{n}.s{s}"] = pred
    for pname, srcs in pools.items():
        ks = [f"{n}.s{s}" for n in srcs for s in seeds if f"{n}.s{s}" in preds]
        if len(ks) >= 2:
            out[f"{pname}.RUNBAG"] = _summ(pred_utility(np.mean([preds[k] for k in ks], axis=0), g0),
                                           extra={"runs": ks})

    def builders(prefix: str, res, g):
        rec["pool_sizes"][prefix] = len(res)
        specs = {}
        with D.selection_scope(f"{stage}.{prefix}.fit"):
            specs["W24"] = b_wide(res, fd, 24)
            specs["W48"] = b_wide(res, fd, 48)
            specs["CAR"] = b_caruana(res, fd)
            lin = b_nnls(res, fd)
        for k, sp in list(specs.items()):
            if sp is None:
                continue
            raw, _ = audit_spec(rt, f"{stage}.{prefix}.{k}", g, sp, cand_of)
            out[f"{prefix}.{k}"] = _summ(raw, sp)
            tw, n_tw = twin_spec(sp, cand_of, width)
            if n_tw:
                raw, _ = audit_spec(rt, f"{stage}.{prefix}.BAG_{k}", g, tw, cand_of)
                out[f"{prefix}.BAG_{k}"] = _summ(raw, tw)
        if lin is not None:
            raw, _ = audit_linear(rt, f"{stage}.{prefix}.NNLS", g, lin, cand_of)
            out[f"{prefix}.NNLS"] = _summ(raw, {"members": lin["members"], "w": [float(x) for x in lin["w"]],
                                                "intercept": lin["intercept"], "n_pool": lin["n_pool"]})

    # own-pool (same programs as one stored run) and pooled builders
    if own_source:
        for s in seeds:
            builders(f"own.{own_source}.s{s}", reps[(own_source, s)][0], gs[(own_source, s)])
    union: Dict[str, Dict[str, object]] = {}
    for pname, srcs in pools.items():
        u: Dict[str, object] = {}
        for n in srcs:
            for s in seeds:
                for r in reps[(n, s)][0]:
                    u.setdefault(r.sha, r)
        union[pname] = u
        builders(pname, list(u.values()), gpool)

    # library stage over the largest pool
    if library:
        big = max(pools, key=lambda p: len(union[p]))
        base = list(union[big].values())
        lc = library_candidates(base, width)
        for c in lc:
            cand_of.setdefault(c.sha, c)
        with D.selection_scope(f"{stage}.lib.eval"):
            lres = gpool.evaluate(lc)
        rec["library"] = {"n_candidates": len(lc), "n_completed": sum(1 for r in lres if r.status == "completed"),
                          "n_passing": len(ST.passing_sorted(lres)),
                          "operators": {}}
        for c in lc:
            op = c.operator.split(":")[0]
            rec["library"]["operators"][op] = rec["library"]["operators"].get(op, 0) + 1
        ok = [r for r in lres if r.status == "completed"]
        if ok:
            with D.selection_scope(f"{stage}.lib.single"):
                sp1 = ST.single_spec(ok, "LIBSINGLE")
            raw, _ = audit_spec(rt, f"{stage}.LIB.single", gpool, sp1, cand_of)
            out["LIB.single"] = _summ(raw, sp1, {"operator": cand_of[sp1.members[0]].operator})
        u2 = dict(union[big])
        for r in lres:
            u2.setdefault(r.sha, r)
        builders(f"{big}+LIB", list(u2.values()), gpool)
    rec["wall_s"] = time.time() - t0
    rt.store.append(stage, cid, "pool", rec)
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# Screeners for extra source runs (fold-0 pair on dev91; per-fold OOF on TRAIN)
# ----------------------------------------------------------------------------------------------------------------------
def build_rt_v(run_dir: Path, cores: Sequence[int], worker: Path, timeout_s: float, seeds: Sequence[int],
               neural_dirs: Dict[str, Path], vdir: Path, torch_threads: int = 1):
    """LAB.build_rt phase V, except that each neural dir loads only its fold-0 pair (C1's L run keeps only fold 0) and
    several source screeners share one engine / ledger / bank / coder. Returns (rt with the first dir's screener,
    {name: screener_for}, {name: screener manifest})."""
    import torch

    import crop_shuffle_synthetic_views as V

    if not C._STATE["stage_v"]:
        raise C.GuardViolation("phase V runtime before the Stage V unlock")
    G.MAX_EXPANDED_WIDTH = L.RUN_MAX_EXPANDED_WIDTH
    torch.set_num_threads(max(1, int(torch_threads)))
    runner = K.rust_runner_cpu(worker, timeout_s)
    gp_root = Path(run_dir) / f"gproot_{K.GROUP}"
    rows = C.train_rows()
    meta_rows = list(rows) + list(C.dev_rows_for_stage_v())
    fam_of = {r["context_sha256"]: r["family_id"] for r in meta_rows}
    ds_of = {r["context_sha256"]: r["dataset_id"] for r in meta_rows}
    folds = C.family_folds(rows)
    bank, _ = D.build_bank(rows)
    rg = E.ResourceGuard(log_path=Path(run_dir) / "resource_pauses.jsonl")
    engine = E.EvalEngine(cores, root=gp_root, timeout_s=timeout_s, guard=rg, runner=runner)
    views = S.SearchViewProvider(view_fn=lambda cid, kind="search": V.context_view_feats(cid, kind, view_dir=vdir))
    scr, man = {}, {}
    for name, nd in neural_dirs.items():
        fs = S.load_neural_fold_screeners(Path(nd), folds, views, folds=(0,), family_of=fam_of)
        man[name] = L.screener_manifest(Path(nd), fs)
        f0 = fs.by_fold[0]
        scr[name] = (lambda f0=f0: (lambda cid: f0))()
    first = next(iter(neural_dirs))
    rt = L.Runtime(engine, E.AuditLedger(engine), scr[first], D.NeuralContextCoder(views), bank, fam_of, ds_of,
                   L.Store(Path(run_dir)), tuple(seeds))
    return rt, scr, man


def build_rt_t(run_dir: Path, cores: Sequence[int], worker: Path, timeout_s: float, seeds: Sequence[int],
               neural_dirs: Dict[str, Path], torch_threads: int = 1):
    """LAB.build_rt phase T for the first dir (per-fold OOF screener, mean of its 2 seeds) plus the same construction
    for the other source dirs on the shared engine."""
    first = next(iter(neural_dirs))
    rt = LAB.build_rt(Path(run_dir), "T", cores, worker, timeout_s, seeds, Path(neural_dirs[first]), torch_threads)
    rows = C.train_rows()
    fam_of = {r["context_sha256"]: r["family_id"] for r in rows}
    folds = C.family_folds(rows)
    views = S.SearchViewProvider()
    scr, man = {first: rt.screener_for}, {}
    for name, nd in neural_dirs.items():
        fs = S.load_neural_fold_screeners(Path(nd), folds, views, family_of=fam_of)
        man[name] = L.screener_manifest(Path(nd), fs)
        if name != first:
            scr[name] = fs.for_context
    return rt, scr, man


def physical_by_kind(rt) -> Dict[str, int]:
    """Physical worker runs by envelope kind (search / audit) in this process (engine.walls holds one entry each)."""
    with rt.engine._lock:
        kinds = [w[0] for w in rt.engine.walls]
    return {k: kinds.count(k) for k in sorted(set(kinds))}


def code_files() -> Tuple[str, ...]:
    return ("deploy_h410.py", "gp_h381.py") + tuple(LAB.PHASE_T_FILES)


if __name__ == "__main__":
    print(json.dumps({"wave": WAVE, "lib_presets": len(LIB_PRESETS), "code_files": code_files()}))
    if os.environ.get("H410_SELFTEST") == "1":
        for n, d in LIB_PRESETS:
            c = _cand(_pipe(d), n, 64)
            print(n, "OK" if c is not None else "REJECTED", c.dsl if c else d)
