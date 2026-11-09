#!/usr/bin/env python3
"""H415 deploy unit (PREREG §6): the pooled population of a committee of GP runs, deployed by H410's builders and by
KFB (5-fold cross-fitted Caruana selection, deployed as the 5-fold bag of fold models).

One unit = one TRAIN context. Every source run (one screener each, same GP seed) is replayed from the run dir's memo
(the concatenation of the source runs' memos for the context); each replay re-runs the frozen AGG recipe (seed A/E/F,
panmictic evo to g50) with that run's screener, every evaluation a memo hit, so it reproduces the run's SearchResults
and its stock CV3@12 output (checked by the analysis against the stored stage records).

Pools: P2 = the first two sources, P3 = all three (when three are given). For every pool, H410's builders
(`deploy_h410` imported at its published bytes: W24, W48, CAR, NNLS, bagged twins) and RUNBAG; library
LIB<n> = deploy_h410.library_candidates(P<n>), SEARCH-evaluated once (one evaluation of the union of the LIB sets),
then the same builders over P<n>+LIB<n>. KFB runs on the last pool with its library (P3+LIB3 in the scored run;
P2+LIB2 in the two-source smoke):

  H       the top 48 passing programs by fitness (stacking_h317.passing_sorted: lower fitness first) + the 2 best passing
          programs of every head operator that is not already among those 48 (head = the pipe head op, "portfolio" for
          portfolios, "none" for a headless pipe); H is capped at 64, dropping the worst-fitness additions first
  kf<j>   5 y-stratified folds F_0..F_4 of the SEARCH rows (seeded per context). Alternative SEARCH envelope j:
          fold0 = SEARCH \\ F_j, fold1 = F_j, SEARCH rows only; the worker's fold-1 predictions (fit on 80% of SEARCH)
          are the out-of-fold predictions on F_j. Evaluated inside a selection scope. A program is usable only if all
          5 folds completed with finite predictions.
  KCAR    Caruana forward selection with replacement on the out-of-fold matrix of the usable programs, 100 steps from
          an empty ensemble; objective ln(RMSE / b_kf), b_kf = RMSE of the per-fold training-mean predictor;
          weights = pick counts / 100
  akf<j>  envelope fold0 = SEARCH \\ F_j, fold1 = AUDIT rows: the member's fold-j model's AUDIT predictions. Requested
          once per output outside every selection scope, under its own ledger key; only fold-1 predictions are read
          (the worker's reverse fit inside akf<j>, like the one inside the stock AUDIT envelope, is never read)
  KFB     weighted mean over members of the mean of their 5 fold models' AUDIT predictions (no refit). A member with a
          failed fold model uses the mean of its completed folds; with none, its full-refit AUDIT prediction (both
          counted)
  HFR     the KCAR members and weights deployed by the stock full-SEARCH refit (exploratory)
  KFB1    the usable program with the lowest out-of-fold RMSE, KFB-deployed (exploratory)

Worker seconds: every SearchResult / AuditResult carries the worker wall time of its (single-core, taskset-pinned)
evaluation, memo hits included, so each unit records the worker seconds behind its search pools, library, kf and akf
evaluations (a CPU proxy; the host-speed check is the analysis's).
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_h410 as DP  # noqa: E402  (FIRST: gp_h381 -> h335_core pins, grammar group, guard)

import numpy as np  # noqa: E402

GPH, K, C, LAB, L, D, ST, G, E = DP.GPH, DP.K, DP.C, DP.LAB, DP.L, DP.D, DP.ST, DP.G, DP.E

WAVE = "H415"
SCHEMA = "jope.h415.deploy-unit/v1"
GENS = 50
KF_SEED, KF_K = 415201, 5
HF_TOP, HF_PER_HEAD, HF_CAP = 48, 2, 64
KCAR_STEPS = 100


# ----------------------------------------------------------------------------------------------------------------------
# 5-fold cross-fitting envelopes
# ----------------------------------------------------------------------------------------------------------------------
def kfold_labels(cid: str, y: np.ndarray, k: int = KF_K) -> np.ndarray:
    """Sort SEARCH rows by y (stable); each consecutive block of k rows gets a seeded permutation of 0..k-1; a short
    last block takes a seeded subset of 0..k-1 without replacement. Fold sizes differ by at most 1."""
    y = np.asarray(y, np.float64)
    rng = np.random.default_rng(C.stream_seed(KF_SEED, "kf", cid))
    order = np.argsort(y, kind="stable")
    lab = np.full(len(y), -1, dtype=np.int16)
    for s in range(0, len(y), k):
        blk = order[s:s + k]
        lab[blk] = rng.permutation(k) if len(blk) == k else rng.choice(k, size=len(blk), replace=False)
    if (lab < 0).any():
        raise AssertionError("unlabelled SEARCH row")
    return lab


_ALT_LOCK = threading.Lock()
_LABELS: Dict[str, np.ndarray] = {}


def ensure_kfold_alts(engine, ctx) -> np.ndarray:
    """Register kf0..kf4 (SEARCH rows only) and akf0..akf4 (SEARCH \\ F_j -> AUDIT) as alternative envelopes of ctx
    (content-addressed files in the engine's env dir). Returns the fold labels; idempotent per context."""
    with _ALT_LOCK:
        if ctx.cid in _LABELS:
            return _LABELS[ctx.cid]
        s, a = ctx.search, ctx.audit
        n, na = int(s.nt), int(a.nt)
        if not (np.array_equal(a.mat[:n], s.mat) and (a.folds[:n] == 0).all() and (a.folds[n:] == 1).all()):
            raise AssertionError(f"{ctx.cid}: AUDIT envelope fold0 is not the SEARCH envelope in row order")
        y, ya = s.y, a.y[n:]
        if not np.array_equal(ya, ctx.ya):
            raise AssertionError(f"{ctx.cid}: AUDIT targets differ from ctx.ya")
        lab = kfold_labels(ctx.cid, y)
        sizes = np.bincount(lab, minlength=KF_K)
        if sizes.max() - sizes.min() > 1:
            raise AssertionError(f"{ctx.cid}: kf fold sizes {sizes.tolist()}")
        for j in range(KF_K):
            fj = lab == j
            kf = C.Jtt1(s.magic, n, 0, s.width, s.flag, fj.astype(np.uint8), s.mat.copy())
            b = kf.to_worker_bytes()
            ctx.alts[f"kf{j}"] = E.AltEnv(f"kf{j}", E._write_env(engine.env_dir, b), y[~fj].copy(), y[fj].copy(),
                                          hashlib.sha256(b).hexdigest())
            keep = np.concatenate([np.flatnonzero(~fj), np.arange(n, na)])
            akf = C.Jtt1(a.magic, len(keep), 0, a.width, a.flag, a.folds[keep].copy(), a.mat[keep].copy())
            b = akf.to_worker_bytes()
            ctx.alts[f"akf{j}"] = E.AltEnv(f"akf{j}", E._write_env(engine.env_dir, b), y[~fj].copy(), ya.copy(),
                                           hashlib.sha256(b).hexdigest())
        _LABELS[ctx.cid] = lab
        return lab


def kf_envelope_check(ctx) -> dict:
    """PREREG §8.2 smoke facts for one context: kf<j> holds SEARCH rows only (row count = n_search, fold1 = F_j),
    akf<j> fold0 = SEARCH \\ F_j and fold1 = AUDIT; fold sizes differ by <= 1."""
    lab = _LABELS[ctx.cid]
    n, na = int(ctx.search.nt), int(ctx.audit.nt)
    sizes = np.bincount(lab, minlength=KF_K).tolist()
    ok = max(sizes) - min(sizes) <= 1
    for j in range(KF_K):
        kf, akf = ctx.alts[f"kf{j}"], ctx.alts[f"akf{j}"]
        ok &= len(kf.y0) + len(kf.y1) == n and len(kf.y1) == sizes[j]
        ok &= len(akf.y0) == n - sizes[j] and len(akf.y1) == na - n
    return {"n_search": n, "n_audit": na - n, "fold_sizes": sizes, "ok": bool(ok),
            "kf_env_sha": [ctx.alts[f"kf{j}"].env_sha for j in range(KF_K)]}


def head_of(dsl: str) -> str:
    gg = G.parse(dsl)
    if isinstance(gg, G.Portfolio):
        return "portfolio"
    return gg.head.op if gg.head is not None else "none"


def hf_candidates(pool: Sequence["E.SearchResult"]) -> Tuple[list, int]:
    ps = ST.passing_sorted(pool)
    top = ps[:HF_TOP]
    in_top = {head_of(r.dsl) for r in top}
    per: Dict[str, int] = {}
    adds = []
    for r in ps[HF_TOP:]:
        h = head_of(r.dsl)
        if h in in_top or per.get(h, 0) >= HF_PER_HEAD:
            continue
        per[h] = per.get(h, 0) + 1
        adds.append(r)
    adds = sorted(adds, key=lambda r: (r.fitness, r.sha))[:max(0, HF_CAP - len(top))]
    return top + adds, len(adds)


def hf_stage(rt, ctx, cands: Sequence["G.Candidate"]):
    """kf<j> evaluations (SEARCH rows only) -> (usable shas, OOF matrix [n_search, m], per-program OOF RMSE, b_kf,
    per-fold failure counts, worker seconds)."""
    lab = ensure_kfold_alts(rt.engine, ctx)
    y = ctx.search.y
    res = {j: rt.engine._evaluate(ctx, list(cands), f"kf{j}") for j in range(KF_K)}
    ok, cols = [], []
    for i, c in enumerate(cands):
        col = np.full(len(y), np.nan)
        for j in range(KF_K):
            r = res[j][i]
            if r.status != "completed" or r.p1 is None or len(r.p1) != int((lab == j).sum()):
                break
            col[lab == j] = r.p1
        if np.all(np.isfinite(col)):
            ok.append(c.sha)
            cols.append(col)
    base = np.empty(len(y))
    for j in range(KF_K):
        base[lab == j] = y[lab != j].mean()
    b_kf = float(np.sqrt(np.mean((base - y) ** 2)))
    Q = np.column_stack(cols) if cols else np.zeros((len(y), 0))
    rm = np.sqrt(((Q - y[:, None]) ** 2).mean(0)) if cols else np.zeros(0)
    n_fail = [sum(1 for r in res[j] if r.status != "completed") for j in range(KF_K)]
    wall = float(sum(r.wall_s for j in range(KF_K) for r in res[j]))
    return ok, Q, rm, b_kf, n_fail, wall


def kcar(Q: np.ndarray, y: np.ndarray, steps: int = KCAR_STEPS) -> np.ndarray:
    """Caruana forward selection with replacement from an empty ensemble; returns pick counts. Minimising RMSE of the
    running mean is minimising ln(RMSE / b_kf); argmin ties -> the lowest column index (columns in H order)."""
    s = np.zeros(len(y))
    cnt = np.zeros(Q.shape[1])
    for t in range(1, steps + 1):
        r = np.sqrt((((s[:, None] + Q) / t - y[:, None]) ** 2).mean(0))
        j = int(np.argmin(r))
        cnt[j] += 1
        s += Q[:, j]
    return cnt


class KFLedger:
    """One akf request per (output, cid), the KFB analogue of E.AuditLedger; never inside a selection scope."""

    def __init__(self):
        self.declared: Dict[Tuple[str, str], int] = {}
        self.requested: Dict[Tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def score(self, engine, pipe: str, ctx, cands: Sequence["G.Candidate"]) -> Dict[int, list]:
        if E.in_selection_scope():
            raise E.AuditLeakError(f"akf (AUDIT) evaluation requested inside a selection scope ({pipe})")
        key = (pipe, ctx.cid)
        with self._lock:
            if key in self.declared:
                raise E.AuditLeakError(f"akf already requested for {key}")
            self.declared[key] = len(cands)
        out = {j: engine._evaluate(ctx, list(cands), f"akf{j}") for j in range(KF_K)}
        with self._lock:
            self.requested[key] = len(cands) if all(len(out[j]) == len(cands) for j in range(KF_K)) else -1
        return out

    def check(self) -> dict:
        bad = {str(k): (v, self.requested.get(k)) for k, v in self.declared.items() if self.requested.get(k) != v}
        if bad:
            raise E.AuditLeakError(f"akf ledger mismatch: {bad}")
        return {"outputs": len(self.declared), "member_requests": int(sum(self.requested.values()))}


KF_LEDGER = KFLedger()


def kfb_deploy(rt, pipe: str, g, members: Sequence[str], w: np.ndarray, cand_of) -> Tuple[dict, np.ndarray]:
    """Weighted mean of each member's 5-fold bag of AUDIT predictions (fallbacks counted)."""
    cands = [cand_of[s] for s in members]
    res = KF_LEDGER.score(rt.engine, pipe, g.ctx, cands)
    na = len(g.ctx.ya)
    cols, n_part, need_full = [], 0, []
    for i in range(len(members)):
        ps = [res[j][i].p1 for j in range(KF_K) if res[j][i].status == "completed" and res[j][i].p1 is not None
              and len(res[j][i].p1) == na and np.all(np.isfinite(res[j][i].p1))]
        if 0 < len(ps) < KF_K:
            n_part += 1
        cols.append(np.mean(ps, axis=0) if ps else None)
        if not ps:
            need_full.append(i)
    n_mean_sub = 0
    if need_full:
        fres = rt.ledger.score(f"{pipe}.fallback", g.ctx, g.seed, [cands[i] for i in need_full])
        for i, a in zip(need_full, fres):
            if a.status == "completed" and a.pa is not None:
                cols[i] = a.pa
            else:
                cols[i] = np.full(na, g.ctx.y_search_mean)
                n_mean_sub += 1
    ww = np.asarray(w, np.float64)
    pred = np.column_stack(cols) @ (ww / ww.sum())
    raw = DP.pred_utility(pred, g)
    raw.update({"n_members": len(members), "n_partial_bag": n_part, "n_full_refit_fallback": len(need_full),
                "n_mean_substituted": n_mean_sub, "all_substituted": n_mean_sub == len(members),
                "akf_worker_s": float(sum(r.wall_s for j in range(KF_K) for r in res[j]))})
    return raw, pred


# ----------------------------------------------------------------------------------------------------------------------
# Unit
# ----------------------------------------------------------------------------------------------------------------------
def deploy_unit(rt, tag: str, cid: str, sources: Dict[str, object], seed: int, jobs: int = 3) -> dict:
    """sources: ordered name -> screener_for(cid). P2 = first two names, P3 = all three (when three)."""
    stage = f"H415_{tag}"
    hit = rt.store.result(stage, cid, "pool")
    if hit:
        return hit
    t0 = time.time()
    names = list(sources)
    pools = {"P2": names[:2]}
    if len(names) >= 3:
        pools["P3"] = names[:3]
    kpool = list(pools)[-1]
    gs = {n: DP._gp(rt, cid, sources[n](cid), seed) for n in names}
    with cf.ThreadPoolExecutor(max_workers=max(1, min(jobs, len(names)))) as ex:
        reps = dict(zip(names, ex.map(lambda n: DP.replay(gs[n], GENS), names)))
    cand_of: Dict[str, "G.Candidate"] = {}
    for _, co in reps.values():
        cand_of.update(co)
    g0 = gs[names[0]]
    fd = ST.FoldData.from_ctx(g0.ctx)
    width = g0.ctx.width
    gpool = D.GPContext(g0.ctx, rt.engine, g0.screener, g0.tb, g0.coder, 0)
    rec = {"type": "result", "schema": SCHEMA, "wave": WAVE, "tag": tag, "cid": cid, "family_id": rt.fam_of[cid],
           "seed": "pool", "search_seed": seed, "sources": names, "pools": pools, "kfb_pool": f"{kpool}+LIB{kpool[1:]}",
           "ba": g0.ctx.ba, "y_search_mean": g0.ctx.y_search_mean, "env_sha": dict(g0.ctx.env_sha), "outputs": {},
           "pool_sizes": {}, "worker_s": {}}
    out = rec["outputs"]
    preds: Dict[str, np.ndarray] = {}

    # stock outputs of every replayed run (= the stored deployed outputs) + RUNBAG
    for n, (res, _) in reps.items():
        rec["worker_s"][f"search.{n}"] = float(sum(r.wall_s for r in res))
        with D.selection_scope(f"{stage}.{n}.stock"):
            spec = ST.cv3_s1direct(res, fd, 12)
        raw, pred = DP.audit_spec(rt, f"{stage}.{n}.stock", gs[n], spec, cand_of)
        out[f"{n}.stock"] = DP._summ(raw, spec, {"n_pool": len(res)})
        if pred is not None:
            preds[n] = pred
    for pname, srcs in pools.items():
        ks = [n for n in srcs if n in preds]
        if len(ks) >= 2:
            out[f"{pname}.RUNBAG"] = DP._summ(DP.pred_utility(np.mean([preds[k] for k in ks], axis=0), g0),
                                              extra={"runs": ks})

    def builders(prefix: str, res):
        """deploy_h410.deploy_unit's builders closure, unchanged."""
        rec["pool_sizes"][prefix] = len(res)
        specs = {}
        with D.selection_scope(f"{stage}.{prefix}.fit"):
            specs["W24"] = DP.b_wide(res, fd, 24)
            specs["W48"] = DP.b_wide(res, fd, 48)
            specs["CAR"] = DP.b_caruana(res, fd)
            lin = DP.b_nnls(res, fd)
        for k, sp in list(specs.items()):
            if sp is None:
                continue
            raw, _ = DP.audit_spec(rt, f"{stage}.{prefix}.{k}", gpool, sp, cand_of)
            out[f"{prefix}.{k}"] = DP._summ(raw, sp)
            tw, n_tw = DP.twin_spec(sp, cand_of, width)
            if n_tw:
                raw, _ = DP.audit_spec(rt, f"{stage}.{prefix}.BAG_{k}", gpool, tw, cand_of)
                out[f"{prefix}.BAG_{k}"] = DP._summ(raw, tw)
        if lin is not None:
            raw, _ = DP.audit_linear(rt, f"{stage}.{prefix}.NNLS", gpool, lin, cand_of)
            out[f"{prefix}.NNLS"] = DP._summ(raw, {"members": lin["members"], "w": [float(x) for x in lin["w"]],
                                                   "intercept": lin["intercept"], "n_pool": lin["n_pool"]})

    union: Dict[str, Dict[str, object]] = {}
    for pname, srcs in pools.items():
        u: Dict[str, object] = {}
        for n in srcs:
            for r in reps[n][0]:
                u.setdefault(r.sha, r)
        union[pname] = u
        builders(pname, list(u.values()))

    # libraries: LIB<n> = library_candidates(P<n>); one SEARCH evaluation of the union of the LIB sets
    lcs = {p: DP.library_candidates(list(union[p].values()), width) for p in pools}
    allc: Dict[str, "G.Candidate"] = {}
    for lc in lcs.values():
        for c in lc:
            allc.setdefault(c.sha, c)
            cand_of.setdefault(c.sha, c)
    with D.selection_scope(f"{stage}.lib.eval"):
        lres_all = {r.sha: r for r in gpool.evaluate(list(allc.values()))}
    rec["worker_s"]["library"] = float(sum(r.wall_s for r in lres_all.values()))
    rec["library"] = {}
    for p, lc in lcs.items():
        lib = f"LIB{p[1:]}"
        lres = [lres_all[c.sha] for c in lc]
        rec["library"][lib] = {"n_candidates": len(lc), "n_completed": sum(1 for r in lres if r.status == "completed"),
                               "n_passing": len(ST.passing_sorted(lres))}
        if p == kpool:
            ok = [r for r in lres if r.status == "completed"]
            if ok:
                with D.selection_scope(f"{stage}.lib.single"):
                    sp1 = ST.single_spec(ok, "LIBSINGLE")
                raw, _ = DP.audit_spec(rt, f"{stage}.LIB.single", gpool, sp1, cand_of)
                out["LIB.single"] = DP._summ(raw, sp1, {"operator": cand_of[sp1.members[0]].operator,
                                                        "library": lib})
        u2 = dict(union[p])
        for r in lres:
            u2.setdefault(r.sha, r)
        union[f"{p}+{lib}"] = u2
        builders(f"{p}+{lib}", list(u2.values()))

    # KFB over the last pool + its library
    kp = rec["kfb_pool"]
    with D.selection_scope(f"{stage}.{kp}.hf"):
        H, n_add = hf_candidates(list(union[kp].values()))
        ok, Q, rm, b_kf, n_fail, hf_wall = hf_stage(rt, g0.ctx, [cand_of[r.sha] for r in H])
        y = g0.ctx.search.y
        rec["worker_s"]["hf"] = hf_wall
        rec["hf"] = {"n_candidates": len(H), "n_head_additions": n_add, "n_usable": len(ok), "kf_failures": n_fail,
                     "b_kf": b_kf, "heads": sorted({head_of(r.dsl) for r in H}),
                     "envelopes": kf_envelope_check(g0.ctx)}
        if ok:
            cnt = kcar(Q, y)
            keep = np.flatnonzero(cnt > 0)
            mem = [ok[i] for i in keep]
            w = cnt[keep] / cnt.sum()
            ens = Q[:, keep] @ w
            i1 = int(np.argmin(rm))
            rec["hf"].update({"kcar_members": mem, "kcar_w": [float(x) for x in w],
                              "kcar_oof_u": -math.log(max(float(np.sqrt(np.mean((ens - y) ** 2))), 1e-300) / b_kf),
                              "best_single": ok[i1],
                              "best_single_oof_u": -math.log(max(float(rm[i1]), 1e-300) / b_kf)})
    if ok:
        raw, _ = kfb_deploy(rt, f"{stage}.{kp}.KFB", gpool, mem, w, cand_of)
        out[f"{kp}.KFB"] = DP._summ(raw, {"members": mem, "w": [float(x) for x in w]})
        dsls = [cand_of[s].dsl for s in mem]
        spec = ST.StackSpec("HFR", list(mem), dsls, [G.island_of(d) for d in dsls], w, w, w, {},
                            info={"of": "KCAR", "steps": KCAR_STEPS})
        raw, _ = DP.audit_spec(rt, f"{stage}.{kp}.HFR", gpool, spec, cand_of)
        out[f"{kp}.HFR"] = DP._summ(raw, spec)
        raw, _ = kfb_deploy(rt, f"{stage}.{kp}.KFB1", gpool, [ok[i1]], np.ones(1), cand_of)
        out[f"{kp}.KFB1"] = DP._summ(raw, {"members": [ok[i1]], "w": [1.0]})
    rec["wall_s"] = time.time() - t0
    rt.store.append(stage, cid, "pool", rec)
    return rec


def code_files() -> Tuple[str, ...]:
    return ("deploy_h415.py", "deploy_h410.py", "gp_h381.py") + tuple(LAB.PHASE_T_FILES)


if __name__ == "__main__":
    lab = kfold_labels("selftest", np.arange(23, dtype=float))
    print(json.dumps({"wave": WAVE, "fold_sizes": np.bincount(lab).tolist(), "code_files": code_files()}))
