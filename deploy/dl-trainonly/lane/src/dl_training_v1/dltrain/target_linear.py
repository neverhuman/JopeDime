"""LINEAR-DETERMINED gate (owner ruling LINEAR-DETERMINED, owner-decisions-20261006T112914Z.json): is the target an exact
(or near-exact) linear function of ALL features on the context's OUTER-TRAIN rows?

  method    the method of the read-only scan RUN/tmp/lindet/scan_lindet.py (66c8b37d), receipt 1e3a458f: the encoder
            JTT1 payload is decoded ONLY through jtt1.train_view (no outer-test byte is read); rows with a non-finite
            feature or target are dropped; when there are more than p + 2 rows and std(y) > 0, y is fitted by OLS on
            [X, 1] (numpy.linalg.lstsq, rcond=None); rel_rmse = RMSE / std(y) (population std). Otherwise rel_rmse is
            None (no verdict).
  verdict   rel_rmse < 1e-8 -> 'target_linear_in_features': THIS context is REFUSED (its builder does not write it; the
            verdict of one context never refuses another fold's context, PR #246 review 5437848458);
            1e-8 <= rel_rmse < 1e-3 -> 'target_linear_near_review': built, flagged for review (training kept, listed in
            release notes); otherwise None.
This extends the single-column target-copy rule (t1prep.target_copy_columns, TARGET-COPY-36) to multi-column linear
combinations. It reads outer-train targets only, which the leak rule allows; it is shared by every builder route.
"""
from __future__ import annotations

import numpy as np

from dltrain import jtt1

REFUSE_REL_RMSE = 1e-8
REVIEW_REL_RMSE = 1e-3
REFUSE = "target_linear_in_features"
REVIEW = "target_linear_near_review"
RULE = ("per context, OLS of y on all features + intercept over OUTER-TRAIN rows only (dltrain.jtt1.train_view; no "
        "outer-test byte decoded); rows with a non-finite value dropped; rel_rmse = RMSE/std(y); refused "
        "(target_linear_in_features) if rel_rmse < 1e-8; flagged (target_linear_near_review) if 1e-8 <= rel_rmse < 1e-3")
SCAN_REFERENCE = {"code": "RUN/tmp/lindet/scan_lindet.py",
                  "code_sha256": "66c8b37d0a5969d921494ff274a346663783056a8a43e3aef8c36979130998a4",
                  "receipt_sha256": "1e3a458fbaf69d5c8ce558dd4ed720eb3dba80e2cd52b9b0bcaf1f829c0f9390"}


def ols_rel_rmse(X, y) -> dict:
    """The scan's OLS on outer-train rows (X: n x p features, y: n targets), exactly as scan_lindet.py computes it."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be n x p and y n")
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    Xo, yo = X[ok], y[ok]
    sd = float(np.std(yo)) if len(yo) else 0.0
    rel = None
    rank = None
    if len(yo) > Xo.shape[1] + 2 and sd > 0:
        A = np.c_[Xo, np.ones(len(yo))]
        coef, _, rank, _ = np.linalg.lstsq(A, yo, rcond=None)
        rel = float(np.sqrt(np.mean((yo - A @ coef) ** 2)) / sd)
    return {"n_train": int(X.shape[0]), "p": int(X.shape[1]), "rows_finite": int(ok.sum()), "y_std": sd,
            "ols_rel_rmse": rel, "ols_rank": None if rank is None else int(rank)}


def verdict(rel_rmse: float | None) -> str | None:
    if rel_rmse is None:
        return None
    if rel_rmse < REFUSE_REL_RMSE:
        return REFUSE
    if rel_rmse < REVIEW_REL_RMSE:
        return REVIEW
    return None


def check_train_rows(X, y) -> dict:
    """The gate on outer-train rows already in memory: the OLS report plus 'verdict' and the rule."""
    rep = ols_rel_rmse(X, y)
    return rep | {"verdict": verdict(rep["ols_rel_rmse"]), "rule": RULE}


def check_jtt1(raw: bytes) -> dict:
    """The gate on a context's encoder JTT1 payload (uncompressed): decodes the outer-train block only."""
    _, _, X, y = jtt1.train_view(raw)
    return check_train_rows(X, y) | {"decoded": "jtt1.train_view (outer-train rows only)"}
