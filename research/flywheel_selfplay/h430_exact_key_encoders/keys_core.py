#!/usr/bin/env python3
"""H430 keys core: exact-value target encoding, key pairs, and the column heads (numpy, scipy, lightgbm).

Everything here is fitted on the rows it is given ("fit rows") and applied to separate query rows. No function sees a
query target. A column value is used as an exact level: two rows share a level only when their values are equal
(non-finite values form one level of their own).

  fit_keys(X, y)                    eligible single columns and column pairs, scored by leave-one-out RMSE of the
                                    shrunk group mean; returns key specs, best first
  make_stat(spec, X, y) / apply_stat / apply_count
                                    group statistics of one key on fit rows, and the encoding of query rows
  nested_encode(spec, X, y, seed)   out-of-fold encoding of the fit rows themselves (inner 5-fold)
  heads(X, y, Xq, seed)             the five column predictions for the query rows:
                                      key_direct   encoding of the best key (ridge when no key reaches DIRECT_MIN_U)
                                      key_gbm      LightGBM on the features + nested encodings and counts of the keys
                                      factor_glm   ridge on one-hot factors, factor crosses and splines, link by CV
                                      gbm_plain    control: the same LightGBM without any encoding
                                      key_noise    control: the same as key_gbm with encodings built from a permuted
                                                   target (they carry no information)
"""
from __future__ import annotations

import math

import numpy as np

M_GRID = (0.0, 0.5, 2.0, 8.0)   # shrinkage of the group mean toward the global mean
FACTOR_MAX = 20                 # at most this many levels: a factor
REP_MIN = 0.5                   # share of fit rows whose level occurs at least twice
PAIR_POOL = 12                  # pairs are formed among the best single keys
MAX_PAIRS = 30
PAIR_GAIN = 0.05                # a pair must beat both of its columns by this much (u)
TOPK = 8                        # keys given to key_gbm
DIRECT_MIN_U = 1.0              # key_direct uses the best key only above this leave-one-out u
INNER = 5
SENT = -9.87654321e300          # stands for a non-finite value


def _fin(col: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(col), col, SENT)


def _u(rmse: float, sd: float) -> float:
    return float(-math.log(max(rmse, 1e-12 * max(sd, 1e-300)) / max(sd, 1e-300)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ----------------------------------------------------------------------------------------------------------------
# levels and group statistics
# ----------------------------------------------------------------------------------------------------------------
def _levels(cols, X):
    """Level index of every row for a key (one column or a pair), and the lookup tables to index query rows."""
    if len(cols) == 1:
        vals, inv = np.unique(_fin(X[:, cols[0]]), return_inverse=True)
        return {"vals": [vals], "pairs": None}, inv, len(vals)
    va, ia = np.unique(_fin(X[:, cols[0]]), return_inverse=True)
    vb, ib = np.unique(_fin(X[:, cols[1]]), return_inverse=True)
    code = ia.astype(np.int64) * len(vb) + ib
    pv, inv = np.unique(code, return_inverse=True)
    return {"vals": [va, vb], "pairs": pv}, inv, len(pv)


def _index(tab, cols, Xq) -> np.ndarray:
    """Level index of query rows; -1 for a level the fit rows do not have."""
    def one(vals, col):
        q = _fin(col)
        i = np.clip(np.searchsorted(vals, q), 0, len(vals) - 1)
        return np.where(vals[i] == q, i, -1)

    if tab["pairs"] is None:
        return one(tab["vals"][0], Xq[:, cols[0]])
    ia, ib = one(tab["vals"][0], Xq[:, cols[0]]), one(tab["vals"][1], Xq[:, cols[1]])
    code = ia.astype(np.int64) * len(tab["vals"][1]) + ib
    pv = tab["pairs"]
    j = np.clip(np.searchsorted(pv, code), 0, len(pv) - 1)
    return np.where((ia >= 0) & (ib >= 0) & (pv[j] == code), j, -1)


def _interp_tables(vals, cnt, s):
    """Finite levels of a single column in value order with their group means (for interpolation between keys)."""
    k = vals != SENT
    return vals[k], (s[k] / cnt[k])


def _lgo_interp(v, m):
    """Mean of every group predicted from its two neighbours in value order (the group itself left out)."""
    g = len(v)
    out = np.empty(g)
    for i in range(g):
        if i == 0:
            out[i] = m[1]
        elif i == g - 1:
            out[i] = m[g - 2]
        else:
            w = (v[i] - v[i - 1]) / (v[i + 1] - v[i - 1])
            out[i] = m[i - 1] * (1 - w) + m[i + 1] * w
    return out


def _loo_pred(inv, y, G, m, back):
    """Leave-one-out shrunk group mean of every fit row; `back` is used where the row is alone in its group."""
    n = len(y)
    cnt = np.bincount(inv, minlength=G).astype(np.float64)
    s = np.bincount(inv, weights=y, minlength=G)
    mu_i = (y.sum() - y) / max(n - 1, 1)
    ci = cnt[inv]
    den = ci - 1.0 + m
    return np.where(ci >= 2, (s[inv] - y + m * mu_i) / np.where(den > 0, den, 1.0), back)


def _eligible(inv, G, n) -> bool:
    if G < 2 or G > n / 2:
        return False
    cnt = np.bincount(inv, minlength=G)
    return float((cnt[inv] >= 2).mean()) >= REP_MIN


def _score_single(j, X, y, sd):
    tab, inv, G = _levels((j,), X)
    n = len(y)
    if not _eligible(inv, G, n):
        return None
    cnt = np.bincount(inv, minlength=G).astype(np.float64)
    s = np.bincount(inv, weights=y, minlength=G)
    mu_i = (y.sum() - y) / max(n - 1, 1)
    back, interp = mu_i, False
    v, mg = _interp_tables(tab["vals"][0], cnt, s)
    if len(v) >= 3:
        lgo = _lgo_interp(v, mg)
        fin = tab["vals"][0] != SENT
        pos = np.cumsum(fin) - 1                      # index into the finite levels
        row_fin = fin[inv]
        # group-level leave-out: does interpolation between neighbouring keys beat the mean without that group?
        mu_g = (y.sum() - s) / np.maximum(n - cnt, 1.0)
        e_int = float(((y[row_fin] - lgo[pos[inv[row_fin]]]) ** 2).sum())
        e_mu = float(((y[row_fin] - mu_g[inv[row_fin]]) ** 2).sum())
        if e_int < e_mu:
            interp = True
            back = np.where(row_fin, lgo[pos[inv]], mu_i)
    best = None
    for m in M_GRID:
        p = _loo_pred(inv, y, G, m, back)
        r = _rmse(p, y)
        if best is None or r < best[0]:
            best = (r, m, p)
    return {"cols": (j,), "m": best[1], "interp": interp, "u": _u(best[0], sd), "loo": best[2], "card": int(G),
            "parent": None}


def _score_pair(a, b, X, y, sd, single):
    tab, inv, G = _levels((a, b), X)
    if not _eligible(inv, G, len(y)):
        return None
    par = single[a] if single[a]["u"] >= single[b]["u"] else single[b]
    best = None
    for m in M_GRID:
        p = _loo_pred(inv, y, G, m, par["loo"])
        r = _rmse(p, y)
        if best is None or r < best[0]:
            best = (r, m, p)
    u = _u(best[0], sd)
    if u < max(single[a]["u"], single[b]["u"]) + PAIR_GAIN:
        return None
    return {"cols": (a, b), "m": best[1], "interp": False, "u": u, "loo": best[2], "card": int(G),
            "parent": {k: par[k] for k in ("cols", "m", "interp")}}


def fit_keys(X: np.ndarray, y: np.ndarray) -> list:
    """Key specs (single columns and pairs) of the fit rows, best leave-one-out u first."""
    y = np.asarray(y, np.float64)
    sd = float(y.std())
    if sd <= 0 or len(y) < 20:
        return []
    single = {}
    for j in range(X.shape[1]):
        sp = _score_single(j, X, y, sd)
        if sp is not None:
            single[j] = sp
    pool = sorted(single, key=lambda j: -single[j]["u"])[:PAIR_POOL]
    pairs = []
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            sp = _score_pair(min(a, b), max(a, b), X, y, sd, single)
            if sp is not None:
                pairs.append(sp)
    pairs = sorted(pairs, key=lambda s: -s["u"])[:MAX_PAIRS]
    return sorted(list(single.values()) + pairs, key=lambda s: (-s["u"], s["cols"]))


def make_stat(spec: dict, X: np.ndarray, y: np.ndarray) -> dict:
    """Group statistics of one key on the fit rows (and of its backoff key, for a pair)."""
    y = np.asarray(y, np.float64)
    cols = tuple(spec["cols"])
    tab, inv, G = _levels(cols, X)
    cnt = np.bincount(inv, minlength=G).astype(np.float64)
    s = np.bincount(inv, weights=y, minlength=G)
    st = {"cols": cols, "m": float(spec["m"]), "tab": tab, "cnt": cnt, "s": s, "mu": float(y.mean()),
          "interp": None, "parent": None}
    if len(cols) == 1 and spec.get("interp"):
        v, mg = _interp_tables(tab["vals"][0], cnt, s)
        if len(v) >= 2:
            st["interp"] = (v, mg)
    if spec.get("parent") is not None:
        st["parent"] = make_stat(dict(spec["parent"], parent=None), X, y)
    return st


def apply_stat(st: dict, Xq: np.ndarray) -> np.ndarray:
    """Encoding of query rows: shrunk group mean; an unseen level falls back to interpolation, the backoff key, or
    the fit mean."""
    idx = _index(st["tab"], st["cols"], Xq)
    hit = idx >= 0
    i = np.where(hit, idx, 0)
    enc = (st["s"][i] + st["m"] * st["mu"]) / (st["cnt"][i] + st["m"])
    if st["parent"] is not None:
        back = apply_stat(st["parent"], Xq)
    elif st["interp"] is not None:
        v, mg = st["interp"]
        q = Xq[:, st["cols"][0]]
        back = np.where(np.isfinite(q), np.interp(np.where(np.isfinite(q), q, v[0]), v, mg), st["mu"])
    else:
        back = np.full(len(Xq), st["mu"])
    return np.where(hit, enc, back)


def apply_count(st: dict, Xq: np.ndarray) -> np.ndarray:
    idx = _index(st["tab"], st["cols"], Xq)
    return np.where(idx >= 0, st["cnt"][np.where(idx >= 0, idx, 0)], 0.0)


def _inner_folds(n: int, seed: int, k: int = INNER) -> np.ndarray:
    f = np.arange(n) % k
    return f[np.random.default_rng(seed).permutation(n)]


def nested_encode(spec: dict, X: np.ndarray, y: np.ndarray, seed: int):
    """Out-of-fold encoding and group size of the fit rows themselves (statistics from the other inner folds)."""
    f = _inner_folds(len(y), seed)
    enc, cnt = np.empty(len(y)), np.empty(len(y))
    for k in range(INNER):
        tr, te = f != k, f == k
        st = make_stat(spec, X[tr], y[tr])
        enc[te], cnt[te] = apply_stat(st, X[te]), apply_count(st, X[te])
    return enc, cnt


# ----------------------------------------------------------------------------------------------------------------
# learners
# ----------------------------------------------------------------------------------------------------------------
def _impute(Xf, Xq):
    med = np.nanmedian(np.where(np.isfinite(Xf), Xf, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(Xf), Xf, med), np.where(np.isfinite(Xq), Xq, med)


def _standardise(Af, Aq):
    mu, sd = Af.mean(0), Af.std(0)
    keep = sd > 1e-12 * (np.abs(mu) + 1.0)
    return (Af[:, keep] - mu[keep]) / sd[keep], (Aq[:, keep] - mu[keep]) / sd[keep]


def ridge_coef(Af: np.ndarray, y: np.ndarray):
    """Ridge coefficients and intercept, the penalty chosen by generalised cross-validation (design already
    standardised)."""
    mu = float(y.mean())
    yc = y - mu
    if Af.shape[1] == 0:
        return np.zeros(0), mu
    U, s, Vt = np.linalg.svd(Af, full_matrices=False)
    uy = U.T @ yc
    n, tot, best = len(y), float(yc @ yc), None
    for lam in (s.max() ** 2) * np.logspace(-8, 2, 21):
        d = s ** 2 / (s ** 2 + lam)
        rss = tot - float(((2 * d - d ** 2) * uy ** 2).sum())
        g = n * max(rss, 0.0) / max(n - 1.0 - float(d.sum()), 1e-9) ** 2
        if best is None or g < best[0]:
            best = (g, lam)
    return Vt.T @ ((s / (s ** 2 + best[1])) * uy), mu


def ridge_gcv(Af: np.ndarray, y: np.ndarray, Aq: np.ndarray) -> np.ndarray:
    coef, mu = ridge_coef(Af, y)
    return Aq @ coef + mu


def ridge_plain(Xf, y, Xq):
    A, Aq = _standardise(*_impute(Xf, Xq))
    return ridge_gcv(A, y, Aq)


def gbm(Xf: np.ndarray, y: np.ndarray, Xq: np.ndarray, seed: int) -> np.ndarray:
    """LightGBM with a fixed setting; the round count comes from a 15% inner holdout, then all fit rows are used."""
    import lightgbm as lgb

    n = len(y)
    mu, sd = float(y.mean()), float(y.std()) or 1.0
    z = (y - mu) / sd
    perm = np.random.default_rng(seed).permutation(n)
    nv = max(20, int(0.15 * n))
    va, tr = perm[:nv], perm[nv:]
    par = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=5, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=1, seed=int(seed % 2**31),
               deterministic=True, force_row_wise=True)
    b = lgb.train(par, lgb.Dataset(Xf[tr], z[tr]), 600, valid_sets=[lgb.Dataset(Xf[va], z[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    rounds = max(50, int(1.1 * (b.best_iteration or 600)))
    b = lgb.train(par, lgb.Dataset(Xf, z), rounds)
    return b.predict(Xq) * sd + mu


def _rcs(z: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted (natural) cubic spline basis without the linear term."""
    t, k = knots, len(knots)
    den = t[k - 1] - t[k - 2]
    cols = []
    for j in range(k - 2):
        c = (np.clip(z - t[j], 0, None) ** 3
             - np.clip(z - t[k - 2], 0, None) ** 3 * (t[k - 1] - t[j]) / den
             + np.clip(z - t[k - 1], 0, None) ** 3 * (t[k - 2] - t[j]) / den)
        cols.append(c / (t[k - 1] - t[0]) ** 2)
    return np.column_stack(cols)


def _yj(x, lam):
    out = np.empty_like(x)
    p, q = x >= 0, x < 0
    out[p] = np.log1p(x[p]) if abs(lam) < 1e-9 else ((x[p] + 1) ** lam - 1) / lam
    out[q] = -np.log1p(-x[q]) if abs(lam - 2) < 1e-9 else -((1 - x[q]) ** (2 - lam) - 1) / (2 - lam)
    return out


def _yj_inv(t, lam):
    out = np.empty_like(t)
    p, q = t >= 0, t < 0
    if abs(lam) < 1e-9:
        out[p] = np.expm1(np.clip(t[p], None, 700))
    else:
        out[p] = np.power(np.clip(lam * t[p] + 1, 1e-12, None), 1 / lam) - 1
    if abs(lam - 2) < 1e-9:
        out[q] = -np.expm1(np.clip(-t[q], None, 700))
    else:
        out[q] = 1 - np.power(np.clip(1 - (2 - lam) * t[q], 1e-12, None), 1 / (2 - lam))
    return out


def _links(y):
    """Candidate target links as (name, forward, inverse)."""
    mu, sd = float(y.mean()), float(y.std()) or 1.0
    out = [("identity", lambda v: v, lambda z: z)]
    lo = float(y.min())
    c = 0.05 * sd
    sk = float(((y - mu) ** 3).mean() / sd ** 3)
    if sk > 1.0:
        out.append(("log", lambda v: np.log(np.clip(v - lo + c, 1e-300, None)),
                    lambda z: np.exp(np.clip(z, -700, 700)) + lo - c))
    try:
        from scipy import stats

        lam = float(stats.yeojohnson_normmax((y - mu) / sd))
        if np.isfinite(lam) and abs(lam - 1.0) > 0.05:
            out.append(("yeojohnson", lambda v: _yj((v - mu) / sd, lam), lambda z: _yj_inv(z, lam) * sd + mu))
    except Exception:
        pass
    return out


def _glm_design(Xf, y, Xq, crosses):
    """One-hot factors, factor crosses, and splines of the other columns; returns standardised designs."""
    n, p = Xf.shape
    Ff, Fq = [], []
    num = []
    for j in range(p):
        vals, inv = np.unique(_fin(Xf[:, j]), return_inverse=True)
        if len(vals) <= FACTOR_MAX:
            if len(vals) >= 2:
                iq = _index({"vals": [vals], "pairs": None}, (j,), Xq)
                Ff.append(np.eye(len(vals))[inv])
                Fq.append(np.where(iq[:, None] >= 0, np.eye(len(vals))[np.where(iq >= 0, iq, 0)], 0.0))
        else:
            num.append(j)
    for a, b in crosses:
        tab, inv, G = _levels((a, b), Xf)
        iq = _index(tab, (a, b), Xq)
        Ff.append(np.eye(G)[inv])
        Fq.append(np.where(iq[:, None] >= 0, np.eye(G)[np.where(iq >= 0, iq, 0)], 0.0))
    if num:
        Nf, Nq = _impute(Xf[:, num], Xq[:, num])
        mu, sd = Nf.mean(0), Nf.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Zf, Zq = (Nf - mu) / sd, np.clip((Nq - mu) / sd, -8, 8)
        Ff.append(Zf); Fq.append(Zq)
        order = np.arange(len(num))
        if len(num) > 60:
            order = np.argsort(-np.abs(((Zf - Zf.mean(0)) * (y - y.mean())[:, None]).mean(0)))[:60]
        for k in order:
            if len(np.unique(Zf[:, k])) >= 10:
                kn = np.unique(np.quantile(Zf[:, k], [0.05, 0.275, 0.5, 0.725, 0.95]))
                if len(kn) >= 3:
                    Ff.append(_rcs(Zf[:, k], kn)); Fq.append(_rcs(Zq[:, k], kn))
    if not Ff:
        return np.zeros((n, 0)), np.zeros((len(Xq), 0))
    return _standardise(np.column_stack(Ff), np.column_stack(Fq))


def _factor_crosses(Xf, y) -> list:
    """Up to 6 factor-by-factor crosses, by leave-one-out gain of the pair encoding over both columns."""
    sd = float(y.std())
    single = {}
    for j in range(Xf.shape[1]):
        vals = np.unique(_fin(Xf[:, j]))
        if 2 <= len(vals) <= FACTOR_MAX:
            sp = _score_single(j, Xf, y, sd)
            if sp is not None:
                single[j] = sp
    pool = sorted(single, key=lambda j: -single[j]["u"])[:PAIR_POOL]
    out = []
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            a2, b2 = min(a, b), max(a, b)
            tab, inv, G = _levels((a2, b2), Xf)
            if G > 60 or not _eligible(inv, G, len(y)):
                continue
            best = min(_rmse(_loo_pred(inv, y, G, m, single[a]["loo"]), y) for m in M_GRID)
            gain = _u(best, sd) - max(single[a]["u"], single[b]["u"])
            if gain > 0.02:
                out.append((gain, (a2, b2)))
    return [c for _, c in sorted(out, reverse=True)[:6]]


def factor_glm(Xf, y, Xq, seed: int) -> np.ndarray:
    crosses = _factor_crosses(Xf, y)
    links = _links(y)
    lo, hi = float(y.min()), float(y.max())
    pad = 0.25 * (hi - lo)

    def fit_predict(Af, yf, Aq, link):
        _, fwd, inv = link
        z = fwd(yf)
        coef, mu = ridge_coef(Af, z)
        zq, zf = Aq @ coef + mu, Af @ coef + mu
        res = np.quantile(z - zf, np.linspace(0.005, 0.995, 100))
        return np.clip(inv(zq[:, None] + res[None, :]).mean(1), lo - pad, hi + pad)   # smearing correction

    best = links[0]
    if len(links) > 1:
        f = _inner_folds(len(y), seed, 3)
        err = {}
        for link in links:
            e = 0.0
            for k in range(3):
                tr, te = f != k, f == k
                Af, Aq = _glm_design(Xf[tr], y[tr], Xf[te], crosses)
                pr = fit_predict(Af, y[tr], Aq, link)
                e += float(((np.where(np.isfinite(pr), pr, y[tr].mean()) - y[te]) ** 2).sum())
            err[link[0]] = e
        best = min(links, key=lambda l: err[l[0]])
    Af, Aq = _glm_design(Xf, y, Xq, crosses)
    out = fit_predict(Af, y, Aq, best)
    return np.where(np.isfinite(out), out, float(y.mean()))


# ----------------------------------------------------------------------------------------------------------------
# the five columns for one (fit rows, query rows) split
# ----------------------------------------------------------------------------------------------------------------
NAMES = ("key_direct", "key_gbm", "factor_glm", "gbm_plain", "key_noise")


def heads(Xf: np.ndarray, y: np.ndarray, Xq: np.ndarray, seed: int):
    """Predictions of the five columns for the query rows, and a small record of what the fit rows contained."""
    Xf, Xq, y = np.asarray(Xf, np.float64), np.asarray(Xq, np.float64), np.asarray(y, np.float64)
    specs = fit_keys(Xf, y)
    cards = [len(np.unique(_fin(Xf[:, j]))) for j in range(Xf.shape[1])]
    info = {"n_keys": len(specs), "best": None, "n_cols": int(Xf.shape[1]),
            "n_factor": int(sum(2 <= c <= FACTOR_MAX for c in cards))}
    out = {}
    plain = gbm(Xf, y, Xq, seed)
    out["gbm_plain"] = plain
    if specs:
        b = specs[0]
        info["best"] = {"cols": list(b["cols"]), "u": b["u"], "m": b["m"], "card": b["card"], "interp": b["interp"]}
    if specs and specs[0]["u"] >= DIRECT_MIN_U:
        out["key_direct"] = apply_stat(make_stat(specs[0], Xf, y), Xq)
        info["direct"] = True
    else:
        out["key_direct"] = ridge_plain(Xf, y, Xq)
        info["direct"] = False
    top = specs[:TOPK]
    if top:
        ef, eq = [], []
        for i, sp in enumerate(top):
            e, c = nested_encode(sp, Xf, y, seed + 17 * (i + 1))
            st = make_stat(sp, Xf, y)
            ef += [e, c]; eq += [apply_stat(st, Xq), apply_count(st, Xq)]
        out["key_gbm"] = gbm(np.column_stack([Xf] + ef), y, np.column_stack([Xq] + eq), seed)
        # control: the same pipeline with encodings built from a permuted target
        yp = y[np.random.default_rng(seed + 7).permutation(len(y))]
        ef, eq = [], []
        for i, sp in enumerate(top):
            e, c = nested_encode(sp, Xf, yp, seed + 17 * (i + 1))
            st = make_stat(sp, Xf, yp)
            ef += [e, c]; eq += [apply_stat(st, Xq), apply_count(st, Xq)]
        out["key_noise"] = gbm(np.column_stack([Xf] + ef), y, np.column_stack([Xq] + eq), seed)
    else:
        out["key_gbm"] = plain
        out["key_noise"] = plain
    out["factor_glm"] = factor_glm(Xf, y, Xq, seed)
    mu = float(y.mean())
    for k in NAMES:
        out[k] = np.where(np.isfinite(out[k]), out[k], mu)
    return out, info
