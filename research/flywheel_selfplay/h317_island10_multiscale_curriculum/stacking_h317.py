"""H317 GP track, H0.6: stacking over evaluated trees, weights fit on SEARCH out-of-fold predictions only.

Inputs per member (SearchResult): p0 = fold-0 predictions of a model fit on fold 1, p1 = fold-1 predictions of a model
fit on fold 0 (worker inner_cv records 0/1). Baselines b0 = rmse(y0 - mean y1), b1 = rmse(y1 - mean y0).

Cross-fitting (every config): the weights that predict fold 0 are fit on fold-1 data only, and vice versa; the
in-sample SEARCH S1 of a stack is the S1 formula on those two cross-fitted fold predictions (H257-comparable;
complexity = sum_i wbar_i * C_i with wbar = mean of the two directions' weights, as H257 did).
Deployed (AUDIT) weights: the same objective fit on the pooled OOF predictions of both folds (super-learner rule);
AUDIT prediction = sum_i w_i * (member i fit on all SEARCH rows, predicting AUDIT rows).

Configs
  single      best-by-SEARCH tree (reference)
  CV1         H257 Top-3 cross-fitted 2-fold MSE simplex grid (step 0.05) + reg 0.02 toward exp(-(f - f_min)/0.15)
              (ported exactly from H257 `compute_leak_free_2fold_oof_stack`; note H257 normalizes the fitting fold's MSE
              by b_k = rmse(y_k - mean of the OTHER fold), so its weights depend on the predicted fold's target mean
              through that one scalar. Kept for replica fidelity; CV2-CV4 use fitting-fold-only scales.)
  CV2@N       Top-N (dedup) Frank-Wolfe simplex, normalized MSE + 0.02 ||w - softmax-fitness prior||^2
  CV3@N       5 target-stratified subfolds per fitting fold, S1-direct Frank-Wolfe on
              0.8 mean_lr + 0.1 tail_lr + 0.1 sd_lr + 0.0005 sum w_i C_i + 0.02 ||w - prior||^2,
              members: fitness <= min + 0.25, residual-correlation pruning rho < 0.985, >= 3 islands when eligible
  CV4         hierarchical: top-2 per island -> island predictor (FW MSE), then 5-simplex over island predictors +
              best overall (FW MSE); every level of a direction is fit on that direction's fitting fold only
  N in {3, 5, 7, 10, 12} for CV2 / CV3. (The plan's 20 bootstrap resamples inside CV3 are not in the GP brief: omitted.)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

import numpy as np  # noqa: E402

import gp_grammar_h317 as G  # noqa: E402
from gp_engine_h317 import AuditResult, SearchResult, selection_scope  # noqa: E402

N_GRID = (3, 5, 7, 10, 12)
LAMBDA = 0.02
TAU_PRIOR = 0.15
RHO_MAX = 0.985
FITNESS_WINDOW = 0.25
N_SUB = 5
W_EPS = 1e-9


@dataclass
class FoldData:
    y0: np.ndarray
    y1: np.ndarray
    b0: float
    b1: float

    @staticmethod
    def from_ctx(ctx) -> "FoldData":
        return FoldData(ctx.y0, ctx.y1, ctx.b0, ctx.b1)


@dataclass
class StackSpec:
    config: str
    members: List[str]
    dsls: List[str]
    islands: List[int]
    w_audit: np.ndarray
    w_fold0: np.ndarray
    w_fold1: np.ndarray
    search: dict
    fallback_single: bool = False
    info: dict = field(default_factory=dict)

    def declared(self) -> Tuple[List[str], List[str], np.ndarray]:
        """Effective members (w_audit > 0): exactly these get one AUDIT request each."""
        keep = [i for i, w in enumerate(self.w_audit) if w > W_EPS]
        return (
            [self.members[i] for i in keep],
            [self.dsls[i] for i in keep],
            self.w_audit[keep],
        )

    def to_json(self) -> dict:
        return {
            "config": self.config,
            "members": self.members,
            "islands": self.islands,
            "w_audit": [float(x) for x in self.w_audit],
            "w_fold0": [float(x) for x in self.w_fold0],
            "w_fold1": [float(x) for x in self.w_fold1],
            "search": self.search,
            "fallback_single": self.fallback_single,
            "info": self.info,
        }


# ----------------------------------------------------------------------------------------------------------------------
# S1 helpers
# ----------------------------------------------------------------------------------------------------------------------
def s1_from_lr(lr0: float, lr1: float, comp: float) -> dict:
    mean_lr, tail_lr, sd_lr = 0.5 * (lr0 + lr1), max(lr0, lr1), 0.5 * abs(lr0 - lr1)
    gate = lr0 <= 0.0 and lr1 <= 0.0 and math.exp(mean_lr) <= 0.90
    if gate:
        fit = 0.8 * mean_lr + 0.1 * tail_lr + 0.1 * sd_lr + 0.0005 * comp
        util = -fit
    else:
        fit = 100.0 + max(mean_lr, 0.0) + max(tail_lr, 0.0) + max(sd_lr, 0.0)
        util = C.GATE_FAIL_UTILITY
    return {
        "lr0": lr0,
        "lr1": lr1,
        "mean_lr": mean_lr,
        "tail_lr": tail_lr,
        "sd_lr": sd_lr,
        "complexity": comp,
        "gate": bool(gate),
        "fitness": float(fit),
        "utility": float(util),
    }


def _lr(pred: np.ndarray, y: np.ndarray, base: float) -> float:
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return math.log(max(rmse / max(base, 1e-12), 1e-12))


def crossfit_search_s1(P0, P1, fd: FoldData, w0, w1, comps) -> dict:
    wbar = 0.5 * (np.asarray(w0) + np.asarray(w1))
    return s1_from_lr(
        _lr(P0 @ w0, fd.y0, fd.b0),
        _lr(P1 @ w1, fd.y1, fd.b1),
        float(np.dot(wbar, comps)),
    )


def softmax_prior(fitness: np.ndarray, tau: float = TAU_PRIOR) -> np.ndarray:
    p = np.exp((np.min(fitness) - fitness) / tau)
    return p / p.sum()


# ----------------------------------------------------------------------------------------------------------------------
# Objectives + Frank-Wolfe on the simplex
# ----------------------------------------------------------------------------------------------------------------------
Block = Tuple[np.ndarray, np.ndarray, float]  # (P [n, K], y [n], base)


def mse_objective(blocks: Sequence[Block], prior: np.ndarray, lam: float):
    def f(w):
        val = np.mean(
            [np.mean((P @ w - y) ** 2) / max(b * b, 1e-24) for P, y, b in blocks]
        )
        return float(val + lam * np.sum((w - prior) ** 2))

    def grad(w):
        g = np.zeros_like(w)
        for P, y, b in blocks:
            g += 2.0 * P.T @ (P @ w - y) / (len(y) * max(b * b, 1e-24))
        return g / len(blocks) + 2.0 * lam * (w - prior)

    return f, grad


def s1_objective(
    blocks: Sequence[Block],
    comps: np.ndarray,
    prior: np.ndarray,
    lam: float,
    temp: float = 0.02,
):
    def lrs(w):
        out = []
        for P, y, b in blocks:
            mse = max(float(np.mean((P @ w - y) ** 2)), 1e-300)
            out.append(0.5 * math.log(mse) - math.log(max(b, 1e-12)))
        return np.array(out)

    def f(w):
        lr = lrs(w)
        return float(
            0.8 * lr.mean()
            + 0.1 * lr.max()
            + 0.1 * lr.std()
            + 0.0005 * np.dot(w, comps)
            + lam * np.sum((w - prior) ** 2)
        )

    def grad(w):
        lr = lrs(w)
        G_ = []
        for P, y, b in blocks:
            r = P @ w - y
            mse = max(float(np.mean(r * r)), 1e-300)
            G_.append(P.T @ r / (len(y) * mse))
        G_ = np.stack(G_)  # [S, K]
        gm = G_.mean(0)
        sm = np.exp((lr - lr.max()) / temp)
        sm /= sm.sum()
        gt = sm @ G_
        sd = lr.std()
        gs = (
            ((lr - lr.mean())[:, None] * (G_ - gm[None, :])).mean(0) / sd
            if sd > 1e-12
            else np.zeros_like(gm)
        )
        return 0.8 * gm + 0.1 * gt + 0.1 * gs + 0.0005 * comps + 2.0 * lam * (w - prior)

    return f, grad


def frank_wolfe(
    f: Callable, grad: Callable, w0: np.ndarray, iters: int = 300, tol: float = 1e-10
) -> np.ndarray:
    """Frank-Wolfe over the probability simplex with an exact grid line search; never leaves the simplex."""
    w = np.asarray(w0, np.float64).copy()
    w = np.maximum(w, 0.0)
    w /= w.sum()
    fw = f(w)
    gammas = np.concatenate(
        [np.linspace(0.0, 1.0, 41)[1:], [0.5**k for k in range(6, 14)]]
    )
    for _ in range(iters):
        g = grad(w)
        s = np.zeros_like(w)
        s[int(np.argmin(g))] = 1.0
        d = s - w
        if float(g @ -d) <= tol:
            break
        vals = [(f(w + t * d), t) for t in gammas]
        best, t = min(vals)
        if best >= fw - tol:
            break
        w = w + t * d
        w = np.maximum(w, 0.0)
        w /= w.sum()
        fw = best
    return w


def h257_grid(k: int) -> List[np.ndarray]:
    grid = []
    if k == 2:
        for i in range(21):
            grid.append(np.array([i * 0.05, 1.0 - i * 0.05]))
    else:
        for i in range(21):
            for j in range(21 - i):
                grid.append(
                    np.array([i * 0.05, j * 0.05, max(0.0, 1.0 - i * 0.05 - j * 0.05)])
                )
    return grid


def h257_grid_fit(
    blocks: Sequence[Block], prior: np.ndarray, reg: float = 0.02
) -> np.ndarray:
    grid = h257_grid(len(prior)) + [prior]
    best_w, best = prior, 1e30
    for w in grid:
        loss = np.mean(
            [np.mean((P @ w - y) ** 2) / max(b**2, 1e-12) for P, y, b in blocks]
        )
        loss += reg * float(np.sum((w - prior) ** 2))
        if loss < best:
            best, best_w = loss, w
    return np.asarray(best_w, np.float64)


# ----------------------------------------------------------------------------------------------------------------------
# Member selection
# ----------------------------------------------------------------------------------------------------------------------
def passing_sorted(results: Sequence[SearchResult]) -> List[SearchResult]:
    return sorted(
        [r for r in results if r.gate and r.ok], key=lambda r: (r.fitness, r.sha)
    )


def top_n_dedup(
    results: Sequence[SearchResult], n: int, b0: float
) -> List[SearchResult]:
    """H257 duplicate rule: skip a tree whose fold-0 OOF predictions equal a chosen one's within 1e-6 * b0."""
    chosen: List[SearchResult] = []
    for r in passing_sorted(results):
        if len(chosen) >= n:
            break
        if any(np.max(np.abs(r.p0 - c.p0)) < 1e-6 * max(b0, 1e-6) for c in chosen):
            continue
        chosen.append(r)
    return chosen


def residuals(r: SearchResult, fd: FoldData) -> np.ndarray:
    return np.concatenate([r.p0 - fd.y0, r.p1 - fd.y1])


def cv3_members(
    results: Sequence[SearchResult], n: int, fd: FoldData
) -> List[SearchResult]:
    ps = passing_sorted(results)
    if not ps:
        return []
    fmin = ps[0].fitness
    pool = [r for r in ps if r.fitness <= fmin + FITNESS_WINDOW]
    res = {r.sha: residuals(r, fd) for r in pool}

    def corr(a, b):
        sa, sb = np.std(a), np.std(b)
        return 1.0 if sa < 1e-12 or sb < 1e-12 else float(np.corrcoef(a, b)[0, 1])

    def compatible(r, chosen):
        return all(corr(res[r.sha], res[c.sha]) < RHO_MAX for c in chosen)

    chosen: List[SearchResult] = []
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
        add = next(
            (
                r
                for r in pool
                if G.island_of(r.dsl) not in have
                and r not in chosen
                and compatible(r, [c for c in chosen])
            ),
            None,
        )
        if add is None:
            add = next(
                (r for r in pool if G.island_of(r.dsl) not in have and r not in chosen),
                None,
            )
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


# ----------------------------------------------------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------------------------------------------------
def _mats(
    members: Sequence[SearchResult],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    P0 = np.column_stack([m.p0 for m in members])
    P1 = np.column_stack([m.p1 for m in members])
    f = np.array([m.fitness for m in members])
    comps = np.array([float(m.complexity) for m in members])
    return P0, P1, f, comps


def stratified_subfolds(y: np.ndarray, k: int = N_SUB) -> List[np.ndarray]:
    order = np.argsort(y, kind="stable")
    lab = np.empty(len(y), dtype=int)
    lab[order] = np.arange(len(y)) % k
    return [np.flatnonzero(lab == s) for s in range(k)]


def _spec(config, members, w0, w1, wa, fd, P0, P1, comps, **info) -> StackSpec:
    return StackSpec(
        config,
        [m.sha for m in members],
        [m.dsl for m in members],
        [G.island_of(m.dsl) for m in members],
        np.asarray(wa, np.float64),
        np.asarray(w0, np.float64),
        np.asarray(w1, np.float64),
        crossfit_search_s1(P0, P1, fd, w0, w1, comps),
        info=info,
    )


def single_spec(
    results: Sequence[SearchResult], config: str = "single"
) -> Optional[StackSpec]:
    done = [r for r in results if r.status == "completed" and r.ok]
    if not done:
        return None
    b = min(done, key=lambda r: (0 if r.gate else 1, r.fitness, r.sha))
    one = np.ones(1)
    search = {
        k: getattr(b, k)
        for k in (
            "lr0",
            "lr1",
            "mean_lr",
            "tail_lr",
            "sd_lr",
            "fitness",
            "utility",
            "gate",
        )
    }
    search["complexity"] = float(b.complexity)
    return StackSpec(
        config,
        [b.sha],
        [b.dsl],
        [G.island_of(b.dsl)],
        one,
        one,
        one,
        search,
        fallback_single=(config != "single"),
    )


def weights_h257(mem: Sequence[SearchResult], fd: FoldData):
    """(w for fold 0 [fit on fold 1], w for fold 1 [fit on fold 0], pooled w)."""
    P0, P1, f, _ = _mats(mem)
    prior = softmax_prior(f)
    return (
        h257_grid_fit([(P1, fd.y1, fd.b1)], prior),
        h257_grid_fit([(P0, fd.y0, fd.b0)], prior),
        h257_grid_fit([(P0, fd.y0, fd.b0), (P1, fd.y1, fd.b1)], prior),
    )


def fit_scale(y: np.ndarray) -> float:
    """Normalization that depends on the fitting fold only (CV2-CV4). The worker's fold baseline b_k uses the OTHER
    fold's mean, i.e. the predicted fold's targets, so it must not enter a cross-fitted weight fit."""
    return float(np.sqrt(np.mean((y - y.mean()) ** 2)))


def weights_fw_mse(mem: Sequence[SearchResult], fd: FoldData):
    P0, P1, f, _ = _mats(mem)
    prior = softmax_prior(f)
    s0, s1 = fit_scale(fd.y0), fit_scale(fd.y1)

    def fit(blocks):
        fo, gr = mse_objective(blocks, prior, LAMBDA)
        return frank_wolfe(fo, gr, prior)

    return (
        fit([(P1, fd.y1, s1)]),
        fit([(P0, fd.y0, s0)]),
        fit([(P0, fd.y0, s0), (P1, fd.y1, s1)]),
    )


def weights_s1_direct(mem: Sequence[SearchResult], fd: FoldData):
    P0, P1, f, comps = _mats(mem)
    prior = softmax_prior(f)
    b_fold1 = _s1_blocks(P1, fd.y1)
    b_fold0 = _s1_blocks(P0, fd.y0)

    def fit(blocks):
        fo, gr = s1_objective(blocks, comps, prior, LAMBDA)
        return frank_wolfe(fo, gr, prior)

    return fit(b_fold1), fit(b_fold0), fit(b_fold0 + b_fold1)


def cv1_h257_top3(results: Sequence[SearchResult], fd: FoldData) -> Optional[StackSpec]:
    mem = top_n_dedup(results, 3, fd.b0)
    if len(mem) < 2:
        return single_spec(results, "CV1")
    P0, P1, _, comps = _mats(mem)
    w0, w1, wa = weights_h257(mem, fd)
    return _spec("CV1", mem, w0, w1, wa, fd, P0, P1, comps)


def cv2_fw(
    results: Sequence[SearchResult], fd: FoldData, n: int
) -> Optional[StackSpec]:
    mem = top_n_dedup(results, n, fd.b0)
    if len(mem) < 2:
        return single_spec(results, f"CV2@{n}")
    P0, P1, _, comps = _mats(mem)
    w0, w1, wa = weights_fw_mse(mem, fd)
    return _spec(f"CV2@{n}", mem, w0, w1, wa, fd, P0, P1, comps, n_requested=n)


def _s1_blocks(P: np.ndarray, y: np.ndarray) -> List[Block]:
    """5 target-stratified subfolds of one fitting fold; subfold baseline = rmse(y_s - mean of the fold's other
    subfolds), so the fit sees only this fold's targets."""
    out = []
    for idx in stratified_subfolds(y):
        ys = y[idx]
        rest = np.delete(y, idx)
        out.append((P[idx], ys, float(np.sqrt(np.mean((ys - rest.mean()) ** 2)))))
    return out


def cv3_s1direct(
    results: Sequence[SearchResult], fd: FoldData, n: int
) -> Optional[StackSpec]:
    mem = cv3_members(results, n, fd)
    if len(mem) < 2:
        return single_spec(results, f"CV3@{n}")
    P0, P1, _, comps = _mats(mem)
    w0, w1, wa = weights_s1_direct(mem, fd)
    return _spec(
        f"CV3@{n}",
        mem,
        w0,
        w1,
        wa,
        fd,
        P0,
        P1,
        comps,
        n_requested=n,
        n_islands=len({G.island_of(m.dsl) for m in mem}),
    )


def cv4_hier(results: Sequence[SearchResult], fd: FoldData) -> Optional[StackSpec]:
    ps = passing_sorted(results)
    if len(ps) < 2:
        return single_spec(results, "CV4")
    groups: List[List[SearchResult]] = []
    for isl in range(4):
        g = top_n_dedup([r for r in ps if G.island_of(r.dsl) == isl], 2, fd.b0)
        if g:
            groups.append(g)
    best = ps[0]
    members: List[SearchResult] = []
    for g in groups:
        for r in g:
            if r not in members:
                members.append(r)
    if best not in members:
        members.append(best)
    if len(members) < 2:
        return single_spec(results, "CV4")
    idx = {m.sha: i for i, m in enumerate(members)}
    P0, P1, f, comps = _mats(members)

    def fit_direction(blocks_of: Callable[[np.ndarray], List[Block]]) -> np.ndarray:
        """Island sub-stacks then a meta-simplex, all on the same fitting rows; returns member weights."""
        maps = []
        for g in groups:
            cols = [idx[r.sha] for r in g]
            if len(cols) == 1:
                u = np.ones(1)
            else:
                pr = softmax_prior(np.array([r.fitness for r in g]))
                fo, gr = mse_objective(blocks_of(np.array(cols)), pr, LAMBDA)
                u = frank_wolfe(fo, gr, pr)
            m = np.zeros(len(members))
            m[cols] = u
            maps.append(m)
        mb = np.zeros(len(members))
        mb[idx[best.sha]] = 1.0
        maps.append(mb)
        Mx = np.stack(maps, axis=1)  # [K_members, n_groups + 1]
        prior = np.full(Mx.shape[1], 1.0 / Mx.shape[1])
        meta_blocks = [
            (P @ Mx, y, b) for (P, y, b) in blocks_of(np.arange(len(members)))
        ]
        fo, gr = mse_objective(meta_blocks, prior, LAMBDA)
        v = frank_wolfe(fo, gr, prior)
        w = Mx @ v
        return w / w.sum()

    s0, s1 = fit_scale(fd.y0), fit_scale(fd.y1)
    w0 = fit_direction(lambda cols: [(P1[:, cols], fd.y1, s1)])
    w1 = fit_direction(lambda cols: [(P0[:, cols], fd.y0, s0)])
    wa = fit_direction(
        lambda cols: [(P0[:, cols], fd.y0, s0), (P1[:, cols], fd.y1, s1)]
    )
    return _spec("CV4", members, w0, w1, wa, fd, P0, P1, comps, n_groups=len(groups))


def all_configs(
    results: Sequence[SearchResult], fd: FoldData, n_grid: Sequence[int] = N_GRID
) -> Dict[str, StackSpec]:
    with selection_scope("H0.6.stack_fit"):
        out: Dict[str, Optional[StackSpec]] = {
            "single": single_spec(results),
            "CV1": cv1_h257_top3(results, fd),
            "CV4": cv4_hier(results, fd),
        }
        for n in n_grid:
            out[f"CV2@{n}"] = cv2_fw(results, fd, n)
            out[f"CV3@{n}"] = cv3_s1direct(results, fd, n)
    return {k: v for k, v in out.items() if v is not None}


# ----------------------------------------------------------------------------------------------------------------------
# AUDIT scoring of a fixed stack (outside selection scope; never feeds a choice)
# ----------------------------------------------------------------------------------------------------------------------
def stack_audit_utility(
    spec: StackSpec,
    audit_by_sha: Dict[str, AuditResult],
    ya: np.ndarray,
    ba: float,
    y_search_mean: float,
) -> dict:
    """AUDIT utility of a FIXED stack. Weights never change with AUDIT outcomes: a member whose AUDIT output crashed or
    is non-finite contributes the SEARCH-target-mean prediction instead (leakage red team rule 3). The output is
    invalid only if the entire prediction is that substitution (`all_substituted`); partial substitution is reported
    as `substituted_weight_mass` (GP red team MEDIUM-3). A single-member output uses the worker's own AUDIT utility."""
    shas, _, w = spec.declared()
    w = (
        w / w.sum()
    )  # renormalise the SEARCH-fit weights after dropping ~0 entries (no AUDIT input)
    ok = np.array(
        [
            audit_by_sha[s].status == "completed" and audit_by_sha[s].pa is not None
            for s in shas
        ]
    )
    sub_mass = float(w[~ok].sum())
    base = {
        "n_members": len(shas),
        "n_mean_substituted": int((~ok).sum()),
        "substituted_weight_mass": sub_mass,
        "all_substituted": bool(not ok.any()),
    }
    if len(shas) == 1:
        a = audit_by_sha[shas[0]]
        return {
            "status": "completed",
            "utility": float(a.utility) if ok[0] else 0.0,
            **base,
        }
    cols = [
        audit_by_sha[s].pa if good else np.full(len(ya), y_search_mean)
        for s, good in zip(shas, ok)
    ]
    pred = np.column_stack(cols) @ w
    rmse = float(np.sqrt(np.mean((pred - ya) ** 2)))
    return {
        "status": "completed",
        "utility": C.audit_utility_from_rmse(rmse, ba),
        "rmse": rmse,
        **base,
    }
