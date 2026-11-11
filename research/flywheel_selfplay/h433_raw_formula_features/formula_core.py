#!/usr/bin/env python3
"""H433 formula core: raw-scale formula features, candidate arithmetic between unscaled columns, a cross-fitted one-dimensional smoother
screen with a permutation threshold, and the column heads (numpy, scipy, lightgbm; learners shared with H430's
keys_core).

Everything is fitted on the rows it is given ("fit rows") and applied to separate query rows. No function sees a
query target.

  screen(X, y, seed)        survivors: derived features whose smoother score beats both parents by GAIN_MIN and by
                            the permutation threshold; best gain first
  compute(spec, X)          a derived feature on any rows (winsorised to the fit rows' range)
  heads(X, y, Xq, seed)     the six column predictions for the query rows:
                              form_gbm     LightGBM on the features + survivors
                              form_gam     ridge on splines of the survivors and of the best raw columns
                              form_sr      sparse symbolic regression (forward selection, BIC)
                              form_interp  interpolation through bin means of the best survivor + ridge on the rest
                              gbm_plain    control: the same LightGBM on the features alone
                              cap          control: much larger LightGBM averaged with a full-row RBF kernel ridge
"""
from __future__ import annotations

import math

import numpy as np

import keys_core as K

PARENTS = 24          # raw columns that may enter a formula
KEEP = 12             # survivors kept
GAIN_MIN = 0.10       # smoother gain over both parents (u)
SCORE_MIN = 0.5       # smoother score of a survivor (u)
BINS = 32
INNER = 3
NPERM = 20
D2_TOP, D2_PARENTS = 4, 8
INTERP_MIN = 3.0      # form_interp uses the best survivor only above this smoother score
GAM_RAW = 12
SR_MAX = 12           # terms of the sparse formula at most
BINARY = ("sub", "add", "mul", "div", "rdiv", "lograt")
UNARY = ("log", "sq", "sqrt", "inv")


# ----------------------------------------------------------------------------------------------------------------
# derived features
# ----------------------------------------------------------------------------------------------------------------
def _operand(o, X):
    return X[:, o] if isinstance(o, (int, np.integer)) else _raw(o, X)


def _raw(spec, X):
    op = spec["op"]
    a = _operand(spec["a"], X)
    b = _operand(spec["b"], X) if "b" in spec else None
    with np.errstate(all="ignore"):
        if op == "sub":
            return a - b
        if op == "add":
            return a + b
        if op == "mul":
            return a * b
        if op == "div":
            return a / b
        if op == "rdiv":
            return b / a
        if op == "lograt":
            return np.log(np.abs(a)) - np.log(np.abs(b))
        if op == "log":
            return np.log(np.abs(a))
        if op == "sq":
            return a * a
        if op == "sqrt":
            return np.sqrt(np.abs(a))
        if op == "inv":
            return 1.0 / a
    raise ValueError(op)


def compute(spec: dict, X: np.ndarray) -> np.ndarray:
    """The derived feature on the rows of X (already imputed), clipped to the fit rows' winsorised range."""
    d = _raw(spec, X)
    return np.clip(np.where(np.isfinite(d), d, spec["med"]), spec["lo"], spec["hi"])


def name(spec) -> str:
    def o(v):
        return f"x{int(v)}" if isinstance(v, (int, np.integer)) else f"({name(v)})"
    s = {"sub": "{a}-{b}", "add": "{a}+{b}", "mul": "{a}*{b}", "div": "{a}/{b}", "rdiv": "{b}/{a}",
         "lograt": "ln|{a}|-ln|{b}|", "log": "ln|{a}|", "sq": "{a}^2", "sqrt": "sqrt|{a}|", "inv": "1/{a}"}[spec["op"]]
    return s.format(a=o(spec["a"]), b=o(spec["b"]) if "b" in spec else "")


def _den_ok(v) -> bool:
    a = np.abs(v)
    return bool(a.min() > 1e-6 * max(float(np.median(a)), 1e-300))


def _same_sign(v) -> bool:
    return bool(np.all(v > 0) or np.all(v < 0))


def _make(op, a, b, X):
    """A candidate spec with its fit-row values, or None when a guard fails."""
    va = _operand(a, X)
    vb = _operand(b, X) if b is not None else None
    if op == "div" and not _den_ok(vb):
        return None
    if op == "rdiv" and not _den_ok(va):
        return None
    if op == "lograt" and not (_same_sign(va) and _same_sign(vb)):
        return None
    if op == "log" and not _same_sign(va):
        return None
    if op == "inv" and not _den_ok(va):
        return None
    spec = {"op": op, "a": a}
    if b is not None:
        spec["b"] = b
    d = _raw(spec, X)
    if not np.all(np.isfinite(d)):
        return None
    if not np.ptp(d) > 0:
        return None
    if op in ("sub", "add", "mul", "sq"):      # cannot blow up: kept at full range (the tails carry signal)
        lo, hi = float("-inf"), float("inf")
    else:                                      # ratios, logs, roots: winsorised to the fit rows' 0.1%-99.9% range
        lo, hi = np.quantile(d, [0.001, 0.999])
        if not hi > lo:
            return None
    spec.update({"lo": float(lo), "hi": float(hi), "med": float(np.median(d))})
    return spec, np.clip(d, lo, hi)


# ----------------------------------------------------------------------------------------------------------------
# the cross-fitted smoother score, for the target and for permuted targets at once
# ----------------------------------------------------------------------------------------------------------------
class Scorer:
    def __init__(self, y: np.ndarray, seed: int):
        n = len(y)
        rng = np.random.default_rng(seed)
        self.Y = np.vstack([y] + [y[rng.permutation(n)] for _ in range(NPERM)])      # row 0 is the target
        f = np.arange(n) % INNER
        self.fold = f[rng.permutation(n)]
        self.sd = float(y.std())
        self.n = n

    def score(self, d: np.ndarray) -> np.ndarray:
        """u of the BINS-bin smoother of each row of Y on d, out of fold (length NPERM + 1)."""
        sse = np.zeros(self.Y.shape[0])
        for k in range(INNER):
            tr, te = self.fold != k, self.fold == k
            dt, de = d[tr], d[te]
            edges = np.unique(np.quantile(dt, np.linspace(0, 1, BINS + 1)))
            if len(edges) < 3:
                m = self.Y[:, tr].mean(1, keepdims=True)
                sse += ((m - self.Y[:, te]) ** 2).sum(1)
                continue
            idx = np.clip(np.searchsorted(edges, dt, side="right") - 1, 0, len(edges) - 2)
            order = np.argsort(idx, kind="stable")
            si = idx[order]
            starts = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
            cnt = np.diff(np.r_[starts, len(si)]).astype(np.float64)
            cen = np.add.reduceat(dt[order], starts) / cnt
            m = np.add.reduceat(self.Y[:, tr][:, order], starts, axis=1) / cnt
            nb = len(cnt)
            if nb < 2:
                sse += ((m - self.Y[:, te]) ** 2).sum(1)
                continue
            j = np.clip(np.searchsorted(cen, de) - 1, 0, nb - 2)
            w = np.clip((de - cen[j]) / (cen[j + 1] - cen[j]), 0.0, 1.0)
            pr = m[:, j] * (1 - w) + m[:, j + 1] * w
            sse += ((pr - self.Y[:, te]) ** 2).sum(1)
        return -np.log(np.maximum(np.sqrt(sse / self.n), 1e-12 * self.sd) / self.sd)


def screen(X: np.ndarray, y: np.ndarray, seed: int):
    """Survivors (spec, fit values, score, gain), best gain first, and a record of the screen."""
    n, p = X.shape
    sc = Scorer(y, seed)
    single = np.vstack([sc.score(X[:, j]) if np.ptp(X[:, j]) > 0 else np.full(NPERM + 1, -9.0) for j in range(p)])
    parents = list(np.argsort(-single[:, 0], kind="stable")[:PARENTS]) if p > PARENTS else list(range(p))
    cands = []
    for i, a in enumerate(parents):
        for op in UNARY:
            c = _make(op, int(a), None, X)
            if c is not None:
                cands.append((c, single[a]))
        for b in parents[i + 1:]:
            for op in BINARY:
                c = _make(op, int(a), int(b), X)
                if c is not None:
                    cands.append((c, np.maximum(single[a], single[b])))
    S = [sc.score(c[0][1]) for c in cands]
    G = np.array([s - par for s, (_c, par) in zip(S, cands)]) if cands else np.zeros((0, NPERM + 1))
    thr = float(np.quantile(G[:, 1:].max(0), 0.95)) if len(cands) else 0.0
    cut = max(GAIN_MIN, thr)
    keep = [i for i in np.argsort(-G[:, 0], kind="stable") if G[i, 0] >= cut and S[i][0] >= SCORE_MIN][:KEEP] if len(cands) else []
    surv = [{"spec": cands[i][0][0], "d": cands[i][0][1], "score": float(S[i][0]), "gain": float(G[i, 0])} for i in keep]
    # depth two: the best survivors against the best raw parents
    extra = []
    for s in surv[:D2_TOP]:
        for b in parents[:D2_PARENTS]:
            for op in BINARY:
                c = _make(op, s["spec"], int(b), X)
                if c is None:
                    continue
                v = sc.score(c[1])
                g = float(v[0] - max(s["score"], single[b][0]))
                if g >= cut and v[0] >= SCORE_MIN:
                    extra.append({"spec": c[0], "d": c[1], "score": float(v[0]), "gain": g})
    surv = sorted(surv + sorted(extra, key=lambda e: -e["gain"])[:KEEP], key=lambda e: -e["score"])[:KEEP]
    info = {"n_candidates": len(cands), "threshold": cut, "perm_threshold": thr, "n_survivors": len(surv),
            "survivors": [{"name": name(s["spec"]), "score": s["score"], "gain": s["gain"]} for s in surv],
            "best_single": float(single[:, 0].max()) if p else None}
    return surv, single[:, 0], info


# ----------------------------------------------------------------------------------------------------------------
# learners
# ----------------------------------------------------------------------------------------------------------------
def _spline_block(zf, zq):
    kn = np.unique(np.quantile(zf, np.linspace(0.03, 0.97, 8)))
    if len(kn) < 3:
        return [zf[:, None]], [zq[:, None]]
    return [zf[:, None], K._rcs(zf, kn)], [zq[:, None], K._rcs(zq, kn)]


def _z(f, q):
    mu, sd = float(f.mean()), float(f.std()) or 1.0
    return (f - mu) / sd, np.clip((q - mu) / sd, -8, 8)


def gam(cols_f: list, cols_q: list, y: np.ndarray) -> np.ndarray:
    Ff, Fq = [], []
    for f, q in zip(cols_f, cols_q):
        a, b = _spline_block(*_z(f, q))
        Ff += a; Fq += b
    Af, Aq = K._standardise(np.column_stack(Ff), np.column_stack(Fq))
    return K.ridge_gcv(Af, y, Aq)


def sparse_sr(names: list, cols_f: list, cols_q: list, y: np.ndarray):
    """Sparse linear formula over the given terms by forward selection: at each step the term most correlated with
    the residual is added and all chosen terms are refitted by least squares; selection stops when BIC stops
    improving or at SR_MAX terms. (Thresholded least squares was tried first and is unstable when survivors are
    near-collinear.) Returns predictions and the formula as text, in standardised terms z[.]."""
    Zf, Zq = zip(*[_z(f, q) for f, q in zip(cols_f, cols_q)])
    A, Aq = np.column_stack(Zf), np.column_stack(Zq)
    n, mu = len(y), float(y.mean())
    yc = y - mu
    sup, beta, r = [], np.zeros(0), yc.copy()
    bic = n * math.log(max(float(yc @ yc) / n, 1e-300))
    for _ in range(min(SR_MAX, A.shape[1])):
        c = np.abs(A.T @ r)
        c[sup] = -1.0
        trial = sup + [int(np.argmax(c))]
        b = np.linalg.lstsq(A[:, trial], yc, rcond=None)[0]
        rr = yc - A[:, trial] @ b
        nb = n * math.log(max(float(rr @ rr) / n, 1e-300)) + len(trial) * math.log(n)
        if nb >= bic - 1e-9:
            break
        sup, beta, r, bic = trial, b, rr, nb
    text = " ".join(f"{beta[i]:+.4g}*z[{names[j]}]" for i, j in enumerate(sup))
    pred = (Aq[:, sup] @ beta if sup else np.zeros(len(Aq))) + mu
    return pred, {"terms": len(sup), "formula": f"{mu:.6g} " + text}


def krr(Xf, y, Xq) -> np.ndarray:
    """RBF kernel ridge on all fit rows; bandwidth from {0.25, 1, 4} x the median heuristic, ridge by GCV."""
    A, Aq = K._standardise(*K._impute(Xf, Xq))
    if A.shape[1] == 0:
        return np.full(len(Xq), float(y.mean()))
    sq = (A ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * A @ A.T, 0.0)
    Dq = np.maximum((Aq ** 2).sum(1)[:, None] + sq[None, :] - 2 * Aq @ A.T, 0.0)
    med = float(np.median(D[np.triu_indices(len(y), 1)])) or 1.0
    mu, n = float(y.mean()), len(y)
    yc = y - mu
    best = None
    for g in (0.25, 1.0, 4.0):
        w, V = np.linalg.eigh(np.exp(-g * D / med))
        w = np.maximum(w, 0.0)
        vy = V.T @ yc
        for lam in w.max() * np.logspace(-8, 0, 9):
            d = w / (w + lam)
            rss = float((((1 - d) * vy) ** 2).sum())
            gcv = n * rss / max(n - float(d.sum()), 1e-9) ** 2
            if best is None or gcv < best[0]:
                best = (gcv, g, V @ (vy / (w + lam)))
    return np.exp(-best[1] * Dq / med) @ best[2] + mu


def gbm_big(Xf, y, Xq, seed: int) -> np.ndarray:
    import lightgbm as lgb

    n = len(y)
    mu, sd = float(y.mean()), float(y.std()) or 1.0
    z = (y - mu) / sd
    perm = np.random.default_rng(seed).permutation(n)
    nv = max(20, int(0.15 * n))
    va, tr = perm[:nv], perm[nv:]
    par = dict(objective="regression", learning_rate=0.03, num_leaves=127, min_data_in_leaf=2, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=1, seed=int(seed % 2**31),
               deterministic=True, force_row_wise=True)
    b = lgb.train(par, lgb.Dataset(Xf[tr], z[tr]), 3000, valid_sets=[lgb.Dataset(Xf[va], z[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    b = lgb.train(par, lgb.Dataset(Xf, z), max(100, int(1.1 * (b.best_iteration or 3000))))
    return b.predict(Xq) * sd + mu


def interp1d(d: np.ndarray, y: np.ndarray, dq: np.ndarray):
    """One-dimensional interpolation of y on d: the target averaged over equal values of d, smoothed by a centred
    moving average over k neighbouring values (k from 1, 3, 7, 15, 31 by leave-one-out RMSE), interpolated linearly
    and continued linearly beyond the ends. Returns the fit-row values, the query values, the leave-one-out u and k."""
    ud, inv = np.unique(d, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    sm = np.bincount(inv, weights=y)
    m = len(ud)
    sd = float(y.std()) or 1.0
    if m < 8:
        return np.full(len(y), float(y.mean())), np.full(len(dq), float(y.mean())), -9.0, 0
    cs, cc = np.r_[0.0, np.cumsum(sm)], np.r_[0.0, np.cumsum(cnt)]
    best = None
    for k in (1, 3, 7, 15, 31):
        h = k // 2
        lo, hi = np.maximum(np.arange(m) - h, 0), np.minimum(np.arange(m) + h + 1, m)
        tot, n = cs[hi] - cs[lo], cc[hi] - cc[lo]
        curve = tot / n
        if k == 1:      # leave one out: a row alone at its value is predicted from the two neighbouring values
            nb = np.empty(m)
            nb[1:-1] = (sm[:-2] / cnt[:-2]) + ((sm[2:] / cnt[2:]) - (sm[:-2] / cnt[:-2])) * (ud[1:-1] - ud[:-2]) / (ud[2:] - ud[:-2])
            g = sm / cnt       # the two end values are continued linearly from their two nearest neighbours
            nb[0] = g[1] + (g[1] - g[2]) * (ud[0] - ud[1]) / (ud[1] - ud[2])
            nb[-1] = g[-2] + (g[-2] - g[-3]) * (ud[-1] - ud[-2]) / (ud[-2] - ud[-3])
            ci = cnt[inv]
            loo = np.where(ci >= 2, (sm[inv] - y) / np.maximum(ci - 1, 1), nb[inv])
        else:
            loo = (tot[inv] - y) / np.maximum(n[inv] - 1, 1)
        r = float(np.sqrt(np.mean((loo - y) ** 2)))
        if best is None or r < best[0]:
            best = (r, k, curve)
    r, k, curve = best
    j = min(5, m - 1)
    s0, s1 = (curve[j] - curve[0]) / (ud[j] - ud[0]), (curve[-1] - curve[-1 - j]) / (ud[-1] - ud[-1 - j])

    def at(v):
        o = np.interp(v, ud, curve)
        o = np.where(v < ud[0], curve[0] + s0 * (v - ud[0]), o)
        return np.where(v > ud[-1], curve[-1] + s1 * (v - ud[-1]), o)

    return at(d), at(dq), float(-math.log(max(r, 1e-12 * sd) / sd)), k


NAMES = ("form_gbm", "form_gam", "form_sr", "form_interp", "gbm_plain", "cap")


def heads(Xf: np.ndarray, y: np.ndarray, Xq: np.ndarray, seed: int):
    """Predictions of the six columns for the query rows, and a record of the screen."""
    Xf, Xq, y = np.asarray(Xf, np.float64), np.asarray(Xq, np.float64), np.asarray(y, np.float64)
    If, Iq = K._impute(Xf, Xq)
    surv, single, info = screen(If, y, seed)
    out = {}
    plain = K.gbm(Xf, y, Xq, seed)
    out["gbm_plain"] = plain
    out["cap"] = 0.5 * gbm_big(Xf, y, Xq, seed) + 0.5 * krr(Xf, y, Xq)
    if not surv:
        for k in ("form_gbm", "form_gam", "form_sr", "form_interp"):
            out[k] = plain
        info["sr"] = None
    else:
        Df = [s["d"] for s in surv]
        Dq = [compute(s["spec"], Iq) for s in surv]
        out["form_gbm"] = K.gbm(np.column_stack([Xf] + Df), y, np.column_stack([Xq] + Dq), seed)
        raw = [int(j) for j in np.argsort(-single, kind="stable")[:GAM_RAW] if np.ptp(If[:, j]) > 0]
        Rf, Rq = [If[:, j] for j in raw], [Iq[:, j] for j in raw]
        out["form_gam"] = gam(Df + Rf, Dq + Rq, y)
        nm = [name(s["spec"]) for s in surv]
        Pf = [Df[i] * Df[j] for i in range(len(Df)) for j in range(i + 1, len(Df))]
        Pq = [Dq[i] * Dq[j] for i in range(len(Dq)) for j in range(i + 1, len(Dq))]
        pn = [f"({nm[i]})*({nm[j]})" for i in range(len(nm)) for j in range(i + 1, len(nm))]
        out["form_sr"], info["sr"] = sparse_sr(nm + [f"x{j}" for j in raw] + pn, Df + Rf + Pf, Dq + Rq + Pq, y)
        d, dq = _raw(surv[0]["spec"], If), _raw(surv[0]["spec"], Iq)
        d, dq = np.where(np.isfinite(d), d, surv[0]["spec"]["med"]), np.where(np.isfinite(dq), dq, surv[0]["spec"]["med"])
        base_f, base_q, loo_u, k = interp1d(d, y, dq)
        info["interp"] = {"feature": name(surv[0]["spec"]), "loo_u": loo_u, "k": k}
        if loo_u >= INTERP_MIN:
            out["form_interp"] = base_q + K.ridge_plain(Xf, y - base_f, Xq)
        else:
            out["form_interp"] = out["form_gam"]
    lo, hi = float(y.min()), float(y.max())
    pad, mu = 0.5 * (hi - lo), float(y.mean())
    for k in NAMES:
        out[k] = np.clip(np.where(np.isfinite(out[k]), out[k], mu), lo - pad, hi + pad)
    return out, info
