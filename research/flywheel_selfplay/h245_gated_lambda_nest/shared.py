"""Exact pure helpers from merged H195/H217, with no corpus/model imports."""

import numpy as np


def family_matrix(values, families):
    order = sorted(set(families))
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray([v[..., np.asarray(families) == f].mean(axis=-1) for f in order])
    return order, np.moveaxis(m, 0, -1)


def family_win_matrix(values_by_context, family_ids, tie_margin=1e-4):
    v = np.asarray(values_by_context, dtype=np.float64)
    diff = v[:, :, None] - v[:, None, :]
    wins = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5)) * (
        1.0 - np.eye(v.shape[1])
    )
    fams = sorted(set(family_ids))
    return np.array([wins[np.asarray(family_ids) == f].mean(axis=0) for f in fams])


def _plateau(arm_values, ref_values, families, seed, draws, plateau_margin):
    """Co-primary on DL-Val flywheel91: per-context native utilities of the arm and of r882 (91 contexts, each averaged
    over the 15 cells). Gain = context-mean delta. CI95 = 10,000 family-cluster draws (families resampled with
    replacement, every drawn family's contexts pooled, context-mean delta). WIN iff gain > +0.002 AND LCB95 > 0;
    a tie (gain exactly 0 or <= the margin) is not a win."""
    a, r = np.asarray(arm_values, np.float64), np.asarray(ref_values, np.float64)
    fams = np.asarray(families)
    if a.shape != r.shape or a.shape != fams.shape or not np.isfinite(a - r).all():
        raise ValueError("matching finite per-context values required")
    delta = a - r
    order = sorted(set(fams.tolist()))
    sums = np.array([delta[fams == f].sum() for f in order])
    counts = np.array([(fams == f).sum() for f in order])
    draw = np.random.default_rng(seed).integers(0, len(order), (draws, len(order)))
    boot = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    gain = float(delta.mean())
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    return dict(
        arm_context_mean=float(a.mean()),
        r882_context_mean=float(r.mean()),
        gain=gain,
        ci95=ci,
        contexts=int(len(a)),
        families=len(order),
        win=bool(gain > plateau_margin and ci[0] > 0),
    )
