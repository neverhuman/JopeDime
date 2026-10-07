"""Verbatim copy of the ridge machinery in experiments/gp_hwm_probe/fastsweep.py (stdz, eig_solve, cv_err, ridge_k, LAMS)
so TA checkpoint probes use the identical estimator without importing fastsweep's process-level env side effects."""
import math
import numpy as np
import scipy.linalg as sl

LAMS = np.logspace(-3, 2, 11)


def stdz(A, tr):
    A = np.array(A, dtype=np.float64)
    mu = np.nanmean(A[tr], 0); mu = np.where(np.isnan(mu), 0, mu)
    A = np.where(np.isnan(A), mu, A)
    sd = A[tr].std(0); keep = sd > 1e-8
    Z = np.clip((A[:, keep] - mu[keep]) / sd[keep], -10, 10)
    Z *= math.sqrt(tr.sum()) / (np.linalg.norm(Z[tr]) + 1e-12)
    return Z


def eig_solve(K, y, tr_i, te_i, lams):
    Ktt = np.array(K[np.ix_(tr_i, tr_i)], dtype=np.float64); Kte = K[np.ix_(te_i, tr_i)]
    ym = y[tr_i].mean(); yc = y[tr_i] - ym; out = []; d = np.arange(len(tr_i))
    for l in lams:
        A = Ktt.copy(); A[d, d] += l
        out.append(Kte @ sl.cho_solve(sl.cho_factor(A, lower=True, check_finite=False, overwrite_a=True), yc, check_finite=False) + ym)
    return np.stack(out)


def cv_err(K, y, tr_idx, ig, lams):
    err = np.zeros(len(lams))
    for f in range(3):
        a_, b_ = tr_idx[ig != f], tr_idx[ig == f]
        err += ((eig_solve(K, y, a_, b_, lams) - y[b_][None]) ** 2).sum(1)
    return err


def ridge_k(Ks, w, y, tr_idx, te_idx, ig):
    K = sum(w[b] * Ks[b] for b in Ks if w[b] > 0); lams = LAMS * np.diag(K)[tr_idx].mean()
    e = cv_err(K, y, tr_idx, ig, lams); li = int(e.argmin())
    return eig_solve(K, y, tr_idx, te_idx, lams[li:li + 1])[0], float(e[li]), float(LAMS[li]), float(lams[li])
