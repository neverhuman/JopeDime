"""H335 statistics (PREREG §5): primary test, verdict map, adoption-grade secondary, headline numbers, AUDIT-Elo.

Pure functions over per-context values {cid: value} (per-context value = mean over search seeds) and a cid -> family
map. No file I/O, no guard dependency except `adoption_gate`, which calls the frozen H317 contract gate.

Primary (one test per contrast, fixed sequence P1 -> P2, each at alpha = 0.025 one-sided):
  d_c   = clip(arm_c - ctrl_c, -CLIP, +CLIP)                    (CLIP = 0.5, fixed before Phase T)
  D_f   = mean of d_c over the contexts of family f             (every family counts once)
  T     = mean_f D_f
  p     = exact one-sided sign-flip p over all 2^F sign vectors  (P(T* >= T), identity included)
  pass  = p < alpha AND leave-one-family-out min of T > 0 AND median_f D_f >= 0 AND T > 0 (ties never pass)
  inferior = the mirrored test (ctrl vs arm) passes
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CLIP = 0.5
ALPHA = 0.025
TIE_TOL = 1e-4
ELO_ITERS = 400
ELO_PRIOR = 0.01
EXACT_MAX_F = 24  # exact enumeration up to 2^24 sign vectors; above that a 2^20-draw Monte Carlo (never at F=19)


def family_contrast(arm: Dict[str, float], ctrl: Dict[str, float], fam_of: Dict[str, str],
                    clip: Optional[float] = CLIP) -> Tuple[List[str], np.ndarray]:
    cids = sorted(set(arm) & set(ctrl))
    if set(arm) != set(ctrl):
        raise ValueError(f"arm/ctrl context sets differ ({len(set(arm) ^ set(ctrl))} contexts)")
    by: Dict[str, List[float]] = {}
    for c in cids:
        d = float(arm[c]) - float(ctrl[c])
        if not math.isfinite(d):
            raise ValueError(f"non-finite contrast at {c[:12]}")
        if clip is not None:
            d = min(max(d, -clip), clip)
        by.setdefault(fam_of[c], []).append(d)
    fams = sorted(by)
    return fams, np.array([float(np.mean(by[f])) for f in fams])


def _sign_matrix(F: int, lo: int, hi: int) -> np.ndarray:
    idx = np.arange(lo, hi, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(F, dtype=np.int64)[None, :]) & 1
    return 1.0 - 2.0 * bits.astype(np.float64)


def exact_signflip_p(D: np.ndarray, chunk: int = 1 << 16) -> float:
    """One-sided P(mean(s * D) >= mean(D)) over all 2^F sign vectors s (identity included)."""
    D = np.asarray(D, np.float64)
    F = D.size
    if F == 0:
        return 1.0
    t_obs = float(D.mean())
    tol = 1e-12 * max(1.0, float(np.abs(D).max()))
    if F > EXACT_MAX_F:  # Monte Carlo (TRAIN reports only), chunked: 2^20 draws in 2^14-row blocks
        rng = np.random.default_rng(335160)
        hits = 0
        for _ in range((1 << 20) >> 14):
            s = rng.choice([-1.0, 1.0], size=(1 << 14, F))
            hits += int(((s @ D) / F >= t_obs - tol).sum())
        return hits / float(1 << 20)
    n = 1 << F
    hits = 0
    for lo in range(0, n, chunk):
        s = _sign_matrix(F, lo, min(n, lo + chunk))
        hits += int(((s @ D) / F >= t_obs - tol).sum())
    return hits / n


def lofo_min(D: np.ndarray) -> float:
    D = np.asarray(D, np.float64)
    if D.size < 2:
        return float("-inf")
    tot = D.sum()
    return float(((tot - D) / (D.size - 1)).min())


def primary_test(arm: Dict[str, float], ctrl: Dict[str, float], fam_of: Dict[str, str],
                 alpha: float = ALPHA, clip: Optional[float] = CLIP) -> dict:
    fams, D = family_contrast(arm, ctrl, fam_of, clip)
    T = float(D.mean())
    p = exact_signflip_p(D)
    p_mirror = exact_signflip_p(-D)
    lm, lm_mirror = lofo_min(D), lofo_min(-D)
    med = float(np.median(D))
    ok = bool(T > 0 and p < alpha and lm > 0 and med >= 0)
    inferior = bool(-T > 0 and p_mirror < alpha and lm_mirror > 0 and -med >= 0)
    return {"families": len(fams), "contexts": len(set(arm) & set(ctrl)), "clip": clip, "T_clipped_family_mean": T,
            "signflip_p": p, "lofo_min": lm, "family_median": med, "pass": ok,
            "mirror_signflip_p": p_mirror, "inferior": inferior,
            "families_won": int((D > TIE_TOL).sum()), "families_tied": int((np.abs(D) <= TIE_TOL).sum()),
            "families_lost": int((D < -TIE_TOL).sum())}


VERDICTS = ("BETTER_MODEL", "BETTER_WITH_COMPUTE", "INFERIOR_TO_INCUMBENT", "INCONCLUSIVE")


def verdict(p1: dict, p2: dict) -> str:
    """Fixed sequence: P2 counts only when P1 passes."""
    if p1["pass"]:
        return "BETTER_MODEL" if p2["pass"] else "BETTER_WITH_COMPUTE"
    if p1["inferior"]:
        return "INFERIOR_TO_INCUMBENT"
    return "INCONCLUSIVE"


def headline(arm: Dict[str, float], ctrl: Dict[str, float], fam_of: Dict[str, str]) -> dict:
    """Unclipped family-equal mean gain, the matching % RMSE reduction 1 - exp(-gain), families won/tied/lost."""
    fams, D = family_contrast(arm, ctrl, fam_of, clip=None)
    g = float(D.mean())
    cids = sorted(set(arm) & set(ctrl))
    d = np.array([arm[c] - ctrl[c] for c in cids])
    return {"family_mean_gain": g, "pct_rmse_reduction": float(1.0 - math.exp(-g)),
            "family_median_gain": float(np.median(D)), "families_won": int((D > TIE_TOL).sum()),
            "families_tied": int((np.abs(D) <= TIE_TOL).sum()), "families_lost": int((D < -TIE_TOL).sum()),
            "n_families": len(fams), "context_wtl": [int((d > TIE_TOL).sum()), int((np.abs(d) <= TIE_TOL).sum()),
                                                      int((d < -TIE_TOL).sum())],
            "context_mean_gain": float(d.mean())}


def family_mean(values: Dict[str, float], fam_of: Dict[str, str]) -> float:
    by: Dict[str, List[float]] = {}
    for c, v in values.items():
        by.setdefault(fam_of[c], []).append(float(v))
    return float(np.mean([np.mean(x) for x in by.values()]))


# ---------------------------------------------------------------------------------------------------------------------
# AUDIT-Elo: verbatim copies of stage_v_h317.bt_ratings / family_wins_tensor / elo_panel (themselves copied from
# common_h252, sha abe974ed...); test_h335.py asserts equality with the frozen H317 functions.
# ---------------------------------------------------------------------------------------------------------------------
def bt_ratings(wins: np.ndarray, anchor: int = 0, prior: float = ELO_PRIOR, iters: int = ELO_ITERS) -> np.ndarray:
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    s = np.ones(wins.shape[:-1], dtype=np.float64)
    for _ in range(iters):
        denom = ((games + 2 * prior * (1 - np.eye(n))) / np.maximum(s[..., :, None] + s[..., None, :], 1e-12)).sum(-1)
        s = np.clip(totals / denom, 1e-6, 1e6)
        s /= np.maximum(s[..., anchor, None], 1e-12)
    return 1500.0 + 400.0 * np.log10(np.maximum(s, 1e-9))


def family_wins_tensor(values_2d: np.ndarray, families: Sequence[str], tie: float = TIE_TOL) -> np.ndarray:
    v = np.asarray(values_2d, np.float64)
    diff = v[:, :, None] - v[:, None, :]
    outcomes = np.where(diff > tie, 1.0, np.where(diff < -tie, 0.0, 0.5)) * (1.0 - np.eye(v.shape[1]))
    fams = np.asarray(families)
    return np.stack([outcomes[fams == f].mean(axis=0) for f in sorted(set(families))], axis=0)


def elo_panel(values: Dict[str, Dict[str, float]], fam_of: Dict[str, str], anchor: str,
              seed: int = 335160, n_boot: int = 10_000) -> dict:
    arms = list(values)
    cids = sorted(set.intersection(*[set(v) for v in values.values()]))
    V = np.array([[values[a][c] for a in arms] for c in cids])
    W = family_wins_tensor(V, [fam_of[c] for c in cids])
    ai = arms.index(anchor)
    point = bt_ratings(W.sum(axis=0), anchor=ai)
    rng = np.random.default_rng(seed)
    F = W.shape[0]
    ix = rng.integers(0, F, size=(n_boot, F))
    boot = np.concatenate([bt_ratings(W[ix[i:i + 500]].sum(axis=1), anchor=ai) for i in range(0, n_boot, 500)])
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return {"anchor": anchor, "families": int(F), "contexts": len(cids),
            "elo": {a: {"elo": float(point[i]), "ci95": [float(lo[i]), float(hi[i])]} for i, a in enumerate(arms)}}


def phase_t_decision(agg: Dict[int, Dict[str, Dict[str, float]]], fam_of: Dict[str, str],
                     checkpoints: Sequence[int], stack: str = "CV3@12") -> dict:
    """PREREG §3.3. agg[gc][cfg] = {cid: value} for cfg in {"single", stack}.
    G* = the smallest checkpoint whose family-mean AUDIT (locked stack) is >= best - 1 paired family SE (SE of the
    family-level difference to the best checkpoint). Stack: keep `stack` unless its family-mean gain over single at G*
    is < 0 (then single)."""
    means = {gc: family_mean(agg[gc][stack], fam_of) for gc in checkpoints}
    best = max(checkpoints, key=lambda gc: (means[gc], -gc))
    se: Dict[int, float] = {}
    gstar = best
    for gc in sorted(checkpoints):
        _, D = family_contrast(agg[gc][stack], agg[best][stack], fam_of, clip=None)
        se[gc] = float(D.std(ddof=1) / math.sqrt(D.size)) if D.size > 1 else 0.0
        if means[gc] >= means[best] - se[gc]:
            gstar = gc
            break
    st_gain = family_mean(agg[gstar][stack], fam_of) - family_mean(agg[gstar]["single"], fam_of)
    return {"family_means": {str(k): v for k, v in means.items()}, "best": best, "paired_se_vs_best":
            {str(k): v for k, v in se.items()}, "G_star": gstar, "stack_gain_vs_single_at_G_star": st_gain,
            "stack": stack if st_gain >= 0 else "single"}
