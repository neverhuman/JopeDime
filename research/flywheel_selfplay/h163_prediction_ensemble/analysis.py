"""Data-free prediction ensembling and family-level selector helpers for H163."""

import numpy as np

FOLD_SEED = 163140
SEEDS = (163101, 163102, 163103)
BOOT_SEED = 163141
ARMS = ("R882", "ENS2", "ENS4", "ENS4_W", "RANKAVG4")
THRESHOLD = 0.001
EPS = 1e-12


def folds(rows):
    families = sorted({r["family_id"] for r in rows})
    if len(families) != 205 or len(families) % 5:
        raise ValueError("H163 requires the pinned 205-family TRAIN cohort")
    order = np.random.default_rng(FOLD_SEED).permutation(families)
    return [sorted(order[i * 41:(i + 1) * 41].tolist()) for i in range(5)]


def _scores(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("scores must be a finite (contexts, candidates) matrix")
    return x


def aligned_scores(scores, candidate_ids):
    """Align score columns to the first model's IDs; reject missing/duplicate IDs."""
    if len(scores) != 4 or len(candidate_ids) != 4:
        raise ValueError("H163 requires four model score/ID vectors")
    ids0 = list(candidate_ids[0])
    if len(set(ids0)) != len(ids0):
        raise ValueError("duplicate candidate IDs")
    out = [_scores(scores[0])]
    if out[0].shape[1] != len(ids0):
        raise ValueError("candidate ID/score width mismatch")
    for matrix, ids in zip(scores[1:], candidate_ids[1:], strict=True):
        x = _scores(matrix)
        ids = list(ids)
        if len(set(ids)) != len(ids) or set(ids) != set(ids0):
            raise ValueError("candidate IDs differ across models")
        if x.shape[1] != len(ids) or x.shape[0] != out[0].shape[0]:
            raise ValueError("candidate score shape mismatch")
        loc = {c: i for i, c in enumerate(ids)}
        out.append(x[:, [loc[c] for c in ids0]])
    return np.stack(out, axis=0), ids0


def zscore_costs(values):
    """Per context/model population z-score; lower remains the preferred cost."""
    x = _scores(values)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, ddof=0, keepdims=True)
    return np.divide(x - mu, sd, out=np.zeros_like(x), where=sd > 1e-12)


def _average_ranks(x):
    # Stable ordering is only for determining ties; tied values get average ranks.
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def arm_costs(model_scores, arm, weight=0.5):
    """Return (contexts,candidates) costs, preserving native first-index argmin ties."""
    s = np.asarray(model_scores, dtype=np.float64)
    if s.ndim != 3 or s.shape[0] != 4 or not np.isfinite(s).all():
        raise ValueError("model_scores must be finite (4, contexts, candidates)")
    if arm == "R882":
        return s[0].copy()
    z = np.stack([zscore_costs(m) for m in s])
    if arm == "ENS2":
        return (z[0] + z[1]) / 2
    if arm == "ENS4":
        return z.mean(axis=0)
    if arm == "ENS4_W":
        if weight not in (0.5, 0.7):
            raise ValueError("ENS4_W weight must be 0.5 or 0.7")
        return weight * z[0] + ((1 - weight) / 3) * z[1:].sum(axis=0)
    if arm == "RANKAVG4":
        ranks = np.stack([np.stack([_average_ranks(row) for row in m]) for m in s])
        return ranks.mean(axis=0)
    raise ValueError(f"unknown arm: {arm}")


def choose_indices(costs):
    """Native stable tie policy: argmin returns the first candidate in aligned order."""
    return np.argmin(_scores(costs), axis=1)


def family_means(rows, values):
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (len(rows),) or not np.isfinite(x).all():
        raise ValueError("family values mismatch or nonfinite")
    families = sorted({r["family_id"] for r in rows})
    return families, np.asarray([
        x[[r["family_id"] == f for r in rows]].mean() for f in families
    ])


def family_bootstrap(rows, candidate, baseline, draws=10_000, seed=BOOT_SEED):
    fams, a = family_means(rows, candidate)
    fams_b, b = family_means(rows, baseline)
    if fams != fams_b or draws < 1:
        raise ValueError("family panel mismatch")
    d = a - b
    rng = np.random.default_rng(seed)
    sample = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    return {"delta": float(d.mean()), "ci_low_95": float(np.quantile(sample, .025)),
            "ci_high_95": float(np.quantile(sample, .975)),
            "p_one_sided": float((np.sum(sample <= 0) + 1) / (draws + 1)),
            "draws": draws}


def admit(stat):
    """Strict margin; equality within 1e-12 rejects, as does nonpositive LCB."""
    return bool(stat["delta"] > THRESHOLD + EPS and stat["ci_low_95"] > 0.0)
