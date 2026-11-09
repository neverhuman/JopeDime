#!/usr/bin/env python3
"""H416 meta-learners and deploy combiners (PREREG §6), numpy only, so the offline design grid (train_h416.py select)
and the deploy unit (deploy_h416.py) run the same code.

Three helpers are verbatim copies of pinned functions; deploy_h416.py asserts at import that each copy returns the
same values as the original on fixed inputs:
  kcar                    = deploy_h415.kcar (Caruana forward selection with replacement from an empty ensemble)
  stream_seed             = common_h317.stream_seed
  audit_utility_from_rmse = common_h317.audit_utility_from_rmse

Meta-learners on the SEARCH out-of-fold matrix Q (rows = SEARCH rows, columns = usable H programs in H order):
  KCAR100   kcar, 100 steps; weights = counts / 100, unpruned (= KFB's weights)
  KCAR1000  kcar, 1000 steps
  BCAR      N_BAGS bags; bag b = n SEARCH rows drawn with replacement; kcar(100) on those rows; w = mean of count/100
  RBCAR     N_BAGS bags; bag b = ceil(m/2) of the m programs drawn without replacement; kcar(100) on all rows
  BRCAR     N_BAGS bags; rows as BCAR, then programs as RBCAR (one generator per bag: rows first)
Bags are seeded by stream_seed(BAG_SEED, name, cid, b). Every learner except KCAR100: weights < W_MIN dropped, at
most MAX_MEMBERS kept (largest weights; ties by H order), renormalised.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

BAG_SEED, N_BAGS = 416201, 32
W_MIN, MAX_MEMBERS = 0.005, 32
META = ("KCAR100", "KCAR1000", "BCAR", "RBCAR", "BRCAR")
DEPLOYS = ("bag", "refit", "mix")
CONS = {"CONS2": ("P3+LIB3.KFB", "P3+LIB3.BAG_CAR"),
        "CONS3": ("P3+LIB3.KFB", "P3+LIB3.BAG_CAR", "P3+LIB3.W48")}
DESIGNS = tuple(f"{m}.{d}" for m in META for d in DEPLOYS) + tuple(CONS)
GATE_FAIL_UTILITY = -2.0


# ----------------------------------------------------------------------------------------------------------------------
# verbatim copies (checked against the originals by deploy_h416.py)
# ----------------------------------------------------------------------------------------------------------------------
def stream_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def audit_utility_from_rmse(model_rmse, baseline_rmse) -> float:
    if model_rmse is None or baseline_rmse is None:
        return GATE_FAIL_UTILITY
    if (
        not (math.isfinite(model_rmse) and math.isfinite(baseline_rmse))
        or baseline_rmse <= 1e-12
    ):
        return GATE_FAIL_UTILITY
    return float(-math.log(max(model_rmse / baseline_rmse, 1e-12)))


def kcar(Q: np.ndarray, y: np.ndarray, steps: int = 100) -> np.ndarray:
    s = np.zeros(len(y))
    cnt = np.zeros(Q.shape[1])
    for t in range(1, steps + 1):
        r = np.sqrt((((s[:, None] + Q) / t - y[:, None]) ** 2).mean(0))
        j = int(np.argmin(r))
        cnt[j] += 1
        s += Q[:, j]
    return cnt


# ----------------------------------------------------------------------------------------------------------------------
# meta-learners
# ----------------------------------------------------------------------------------------------------------------------
def prune(w: np.ndarray) -> np.ndarray:
    w = np.where(w >= W_MIN, w, 0.0)
    nz = np.flatnonzero(w > 0)
    if len(nz) > MAX_MEMBERS:
        keep = sorted(nz, key=lambda i: (-w[i], i))[:MAX_MEMBERS]
        w2 = np.zeros_like(w)
        w2[keep] = w[keep]
        w = w2
    s = w.sum()
    if s <= 0:
        raise AssertionError("pruning removed every member")
    return w / s


def meta_weights(name: str, Q: np.ndarray, y: np.ndarray, cid: str) -> np.ndarray:
    """Weights over the columns of Q (H order), summing to 1 (SEARCH out-of-fold data only)."""
    Q = np.asarray(Q, np.float64)
    y = np.asarray(y, np.float64)
    n, m = Q.shape
    if m == 0:
        raise ValueError("no usable program")
    if name == "KCAR100":
        cnt = kcar(Q, y, 100)
        return cnt / cnt.sum()
    if name == "KCAR1000":
        return prune(kcar(Q, y, 1000) / 1000.0)
    if name not in ("BCAR", "RBCAR", "BRCAR"):
        raise ValueError(name)
    acc = np.zeros(m)
    for b in range(N_BAGS):
        rng = np.random.default_rng(stream_seed(BAG_SEED, name, cid, b))
        rows = np.arange(n)
        cols = np.arange(m)
        if name in ("BCAR", "BRCAR"):
            rows = rng.integers(0, n, size=n)
        if name in ("RBCAR", "BRCAR") and m >= 2:
            cols = np.sort(rng.choice(m, size=int(math.ceil(m / 2)), replace=False))
        cnt = kcar(Q[np.ix_(rows, cols)], y[rows], 100)
        acc[cols] += cnt / 100.0
    return prune(acc / N_BAGS)


# ----------------------------------------------------------------------------------------------------------------------
# deploy combiners
# ----------------------------------------------------------------------------------------------------------------------
def deploy_pred(mode: str, w: np.ndarray, bag: Sequence[Optional[np.ndarray]], refit: Sequence[Optional[np.ndarray]],
                y_mean: float, na: int) -> Tuple[np.ndarray, dict]:
    """bag / refit / mix prediction for members (index-aligned lists) with weights w; fallbacks of the parts the mode
    uses are counted. bag: no fold model -> the member's refit, that failed too -> the SEARCH mean (KFB rule);
    refit: failed -> the SEARCH mean (stack_audit_utility rule); mix = 0.5 * bag + 0.5 * refit."""
    if mode not in DEPLOYS:
        raise ValueError(mode)
    ww = np.asarray(w, np.float64)
    ww = ww / ww.sum()
    mean = np.full(na, y_mean)
    info: Dict[str, int] = {}
    parts: Dict[str, np.ndarray] = {}
    if mode in ("bag", "mix"):
        cols, nf, nm = [], 0, 0
        for b, r in zip(bag, refit):
            if b is None:
                nf += 1
                if r is None:
                    nm += 1
                b = r if r is not None else mean
            cols.append(b)
        parts["bag"] = np.column_stack(cols) @ ww
        info.update({"bag_full_refit_fallback": nf, "bag_mean_substituted": nm})
    if mode in ("refit", "mix"):
        cols, nm = [], 0
        for r in refit:
            if r is None:
                nm += 1
            cols.append(r if r is not None else mean)
        parts["refit"] = np.column_stack(cols) @ ww
        info["refit_mean_substituted"] = nm
    pred = parts[mode] if mode != "mix" else 0.5 * parts["bag"] + 0.5 * parts["refit"]
    return pred, info


def utility(pred: np.ndarray, ya: np.ndarray, ba: float) -> Tuple[float, float]:
    rmse = float(np.sqrt(np.mean((np.asarray(pred, np.float64) - ya) ** 2)))
    return audit_utility_from_rmse(rmse, ba), rmse


def cons_pred(name: str, refs: Dict[str, Optional[np.ndarray]], y_mean: float, na: int) -> Tuple[np.ndarray, int]:
    """Mean of the named reference outputs' AUDIT predictions; a None prediction = the SEARCH mean (counted)."""
    parts = [refs[k] if refs.get(k) is not None else np.full(na, y_mean) for k in CONS[name]]
    return np.mean(parts, axis=0), int(sum(refs.get(k) is None for k in CONS[name]))


def design_pred(design: str, Q, y, cid, bag_cols, refit_cols, refs, y_mean, na) -> Tuple[np.ndarray, dict]:
    """Offline (dev) prediction of one design from the full matrices: bag_cols / refit_cols hold every usable H
    program's column (None = failed)."""
    if design in CONS:
        p, n = cons_pred(design, refs, y_mean, na)
        return p, {"none_substituted": n}
    meta, mode = design.split(".")
    w = meta_weights(meta, Q, y, cid)
    keep = np.flatnonzero(w > 0)
    p, info = deploy_pred(mode, w[keep], [bag_cols[i] for i in keep], [refit_cols[i] for i in keep], y_mean, na)
    info["n_members"] = int(len(keep))
    return p, info


if __name__ == "__main__":
    import json

    rng = np.random.default_rng(0)
    Q = rng.normal(size=(40, 6))
    y = Q[:, 2] + 0.1 * rng.normal(size=40)
    print(json.dumps({m: [round(float(x), 3) for x in meta_weights(m, Q, y, "selftest")] for m in META}))
