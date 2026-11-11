#!/usr/bin/env python3
"""H423, follow-up 3 of H422: the two H422 levers in the offline deploy path, behind flags that default off.
Code only; nothing here is scored by this wave. numpy + meta_h416 (no GP engine is imported).

1. Winsorised combiner as design names.  `design_pred` is `meta_h416.design_pred` for every stored design name and
   returns that function's result unchanged. With JOPE_H423_WINSOR_DESIGNS=1 it also accepts `<META>.mixw<k>`
   (for example `BCAR.mixw6`): the stored weights of <META>, the stored `mix` columns (each member's fold-model bag
   and refit, half weight each, with the stored fallbacks), and before the weighted mean every column is pulled
   back to within k member-scales of the row's weighted median. A member's scale is its root-mean-square deviation
   from the blend on the SEARCH out-of-fold rows. Without the flag those names are refused.
2. Pooled admission.  `hf_candidates(pool, hf, sub_pools)` is `hf(pool)` (the stored rule, e.g.
   deploy_h415.hf_candidates) unless JOPE_H423_POOLED_ADMISSION=1 and sub-pools are given; then it returns the
   ordered union (first occurrence kept, by program sha) of the stored rule applied to each sub-pool. The set H422
   tested is the union over four sub-pools: {GS, GE, GL} + library, {GS, GE, GX} + library, and both again with the
   20 added presets. A deploy unit that wants it must run the fourth search run and pass those sub-pools; no stored
   unit does.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import meta_h416 as M  # noqa: E402

FLAG_DESIGNS = "JOPE_H423_WINSOR_DESIGNS"
FLAG_ADMISSION = "JOPE_H423_POOLED_ADMISSION"
_WDESIGN = re.compile(r"^(KCAR100|KCAR1000|BCAR|RBCAR|BRCAR)\.mixw([0-9]+(?:\.[0-9]+)?)$")


def enabled(flag: str) -> bool:
    """A lever is on only when its environment variable is exactly "1"."""
    return os.environ.get(flag, "") == "1"


def wquant(P: np.ndarray, w: np.ndarray, q: float) -> np.ndarray:
    """Row-wise weighted quantile: the smallest value whose cumulative weight reaches q (sel_h422.wquant)."""
    o = np.argsort(P, axis=1, kind="stable")
    cw = np.cumsum(w[o], axis=1)
    k = (cw >= q - 1e-12).argmax(axis=1)
    r = np.arange(len(P))
    return P[r, o[r, k]]


def member_scales(Q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per member (columns of Q aligned with w): RMS deviation from the weighted blend on the SEARCH OOF rows."""
    ws = np.asarray(w, np.float64) / np.sum(w)
    ms = np.asarray(Q, np.float64) @ ws
    return np.sqrt(((Q - ms[:, None]) ** 2).mean(0)) + 1e-300


def winsor_pred(w: np.ndarray, bag: Sequence[Optional[np.ndarray]], refit: Sequence[Optional[np.ndarray]],
                scales: np.ndarray, y_mean: float, na: int, k: float) -> Tuple[np.ndarray, dict]:
    """The stored `mix` columns (meta_h416.deploy_pred's fallbacks: no fold model -> the member's refit, that failed
    too -> the SEARCH mean; refit failed -> the SEARCH mean), winsorised around the row's weighted median."""
    ws = np.asarray(w, np.float64) / np.sum(w)
    mean = np.full(na, y_mean)
    bcols, rcols, nf, nm, nr = [], [], 0, 0, 0
    for b, r in zip(bag, refit):
        if r is None:
            nr += 1
        rr = r if r is not None else mean
        if b is None:
            nf += 1
            nm += int(r is None)
            b = rr
        bcols.append(b)
        rcols.append(rr)
    cols = np.column_stack(bcols + rcols)
    w2, s2 = np.concatenate([ws, ws]) / 2.0, np.concatenate([scales, scales])
    med = wquant(cols, w2, 0.5)[:, None]
    pred = np.clip(cols, med - k * s2, med + k * s2) @ w2
    return pred, {"bag_full_refit_fallback": nf, "bag_mean_substituted": nm, "refit_mean_substituted": nr,
                  "winsor_k": float(k)}


def design_pred(design: str, Q, y, cid, bag_cols, refit_cols, refs, y_mean, na) -> Tuple[np.ndarray, dict]:
    """meta_h416.design_pred, plus `<META>.mixw<k>` when JOPE_H423_WINSOR_DESIGNS=1."""
    m = _WDESIGN.match(design)
    if m is None:
        return M.design_pred(design, Q, y, cid, bag_cols, refit_cols, refs, y_mean, na)   # stored path, unchanged
    if not enabled(FLAG_DESIGNS):
        raise ValueError(f"design {design} is refused unless {FLAG_DESIGNS}=1")
    w = M.meta_weights(m.group(1), Q, y, cid)
    keep = np.flatnonzero(w > 0)
    scales = member_scales(np.asarray(Q, np.float64)[:, keep], w[keep])
    pred, info = winsor_pred(w[keep], [bag_cols[i] for i in keep], [refit_cols[i] for i in keep], scales, y_mean, na,
                             float(m.group(2)))
    info["n_members"] = int(len(keep))
    return pred, info


def hf_candidates_pooled(sub_pools: Sequence[Sequence[object]], hf: Callable) -> Tuple[list, int]:
    """Ordered union (first occurrence kept, by sha) of hf(sub-pool); second value = admitted beyond the first."""
    out, seen, first = [], set(), None
    for p in sub_pools:
        H, _ = hf(list(p))
        if first is None:
            first = len(H)
        for r in H:
            if r.sha not in seen:
                seen.add(r.sha)
                out.append(r)
    return out, len(out) - (first or 0)


def hf_candidates(pool: Sequence[object], hf: Callable, sub_pools: Optional[Sequence[Sequence[object]]] = None
                  ) -> Tuple[list, int]:
    """hf(pool) (the stored admission rule) unless JOPE_H423_POOLED_ADMISSION=1 and sub-pools are given."""
    if not enabled(FLAG_ADMISSION) or not sub_pools:
        return hf(list(pool))   # stored path, unchanged
    return hf_candidates_pooled(sub_pools, hf)
