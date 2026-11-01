"""H189 pooled B selection and A confirmation; deterministic, data-free helpers."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

SEEDS = (189101, 189102, 189103)
FOLD_SEED = 189140
BOOT_SEED = 189141
DRAWS = 10_000
EPS = 1e-12
ARMS = ("SWA", "EMA")
SNAPSHOT_STEPS = tuple(range(40, 81, 5))


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def family_folds(family_ids, seed=FOLD_SEED):
    families = sorted(set(family_ids))
    if len(families) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(seed).permutation(len(families))
    return [sorted(families[i] for i in group) for group in np.array_split(order, 5)]


def partition(family_ids, outer, folds=None):
    groups = family_folds(family_ids) if folds is None else folds
    if len(groups) != 5 or not 0 <= outer < 5:
        raise ValueError("five family folds required")
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    fit = set().union(
        *(set(groups[i]) for i in range(5) if i not in (outer, (outer + 1) % 5))
    )
    if fit & (a | b) or a & b or (len(fit), len(b), len(a)) != (123, 41, 41):
        raise ValueError("family fold overlap or count mismatch")
    return fit, b, a


def batch_plan(cids, orders, seed, updates=80):
    cids = sorted(cids)
    if len(cids) < 8 or updates != 80:
        raise ValueError("H189 requires the fixed 80-update H156 batch plan")
    plan = []
    for step in range(1, updates + 1):
        stream = stream_seed(seed, step)
        rng = np.random.default_rng(stream)
        chosen = [cids[i] for i in rng.choice(len(cids), 8, replace=False)]
        indexes = []
        for cid in chosen:
            order = np.asarray(orders[cid], dtype=np.int64)
            indexes.append(
                np.concatenate((order[:4], rng.choice(order[4:], 20, replace=False)))
            )
        plan.append((stream, chosen, indexes))
    return plan


def validate_fit_plan(plan, fit_context_ids, a_context_ids, b_context_ids=()):
    fit = set(fit_context_ids)
    excluded = set(a_context_ids) | set(b_context_ids)
    if fit & excluded:
        raise ValueError("FIT and held-out contexts overlap")
    for _, chosen, indexes in plan:
        if (
            len(chosen) != 8
            or len(indexes) != 8
            or not set(chosen) <= fit
            or set(chosen) & excluded
        ):
            raise ValueError("training batch contains a non-FIT context")
    return True


def family_matrix(values, families):
    values = np.asarray(values, dtype=np.float64)
    families = np.asarray(families, dtype=object)
    if values.shape != families.shape or not np.isfinite(values).all():
        raise ValueError("family panel shape or finiteness")
    ordered = sorted(set(families.tolist()))
    return ordered, np.asarray([values[families == f].mean() for f in ordered])


def holm(p_values):
    if not p_values:
        raise ValueError("Holm requires at least one prespecified test")
    keys = sorted(p_values, key=lambda k: (float(p_values[k]), k))
    m = len(keys)
    adjusted, running = {}, 0.0
    for i, key in enumerate(keys):
        p = float(p_values[key])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid p value")
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def pooled_b_select(rows, r882_values, candidate_panels_by_arm, seed=BOOT_SEED):
    """H177 pooled family bootstrap and selector, specialized to H189's two arms."""
    if not isinstance(candidate_panels_by_arm, dict) or set(
        candidate_panels_by_arm
    ) != set(ARMS):
        raise ValueError(
            "H189 requires exactly the prespecified SWA and EMA arms (Holm m=2)"
        )
    families = [r["family_id"] for r in rows]
    ordered = sorted(set(families))
    if len(ordered) != 205:
        raise ValueError("pooled B selector requires all 205 TRAIN families")
    base = np.asarray(r882_values, dtype=np.float64)
    if base.ndim == 1:
        if base.shape != (len(rows),):
            raise ValueError("r882 panel row count")
        base = np.broadcast_to(base[None, :], (3, len(rows)))
    if base.shape != (3, len(rows)) or not np.isfinite(base).all():
        raise ValueError("r882 panel must have shape (3, N) and finite values")
    base_fam = np.stack([family_matrix(v, families)[1] for v in base])
    ix = np.random.default_rng(seed).integers(0, len(ordered), (DRAWS, len(ordered)))
    raw, pvals = {}, {}
    for arm in ARMS:
        panel = np.asarray(candidate_panels_by_arm[arm], dtype=np.float64)
        if panel.shape != (3, len(rows)) or not np.isfinite(panel).all():
            raise ValueError(f"{arm} pooled B panel must have shape (3, N)")
        candidate_fam = np.stack([family_matrix(v, families)[1] for v in panel])
        delta = candidate_fam - base_fam
        family_delta = delta.mean(axis=0)
        boot = family_delta[ix].mean(axis=1)
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
        p = float((1 + np.count_nonzero(boot <= 0.0)) / (DRAWS + 1))
        raw[arm] = {
            "mean_gain": float(family_delta.mean()),
            "ci95": ci,
            "per_seed_gain": delta.mean(axis=1).tolist(),
            "p_one_sided": p,
            "seed_averaged_family_gains": family_delta.tolist(),
        }
        pvals[arm] = p
    adjusted = holm(pvals)
    decisions = {}
    for arm in ARMS:
        d = raw[arm]
        d["holm_multiplicity"] = 2
        d["holm_p"] = adjusted[arm]
        d["admitted"] = bool(
            d["mean_gain"] > 0.001 + EPS and d["ci95"][0] > 0.0 and d["holm_p"] < 0.05
        )
        decisions[arm] = d
    return decisions


def pooled_a_confirm(rows, r882_values, candidate_panels_by_arm, seed=BOOT_SEED):
    """Return the preregistered median-of-seed pooled A effects and family CI."""
    families = [r["family_id"] for r in rows]
    ordered = sorted(set(families))
    if len(ordered) != 205:
        raise ValueError("pooled A confirmation requires all 205 TRAIN families")
    base = np.asarray(r882_values, dtype=np.float64)
    if base.shape == (len(rows),):
        base = np.broadcast_to(base[None, :], (3, len(rows)))
    if base.shape != (3, len(rows)) or not np.isfinite(base).all():
        raise ValueError("A reference must have shape (3, N)")
    ix = np.random.default_rng(seed).integers(0, len(ordered), (DRAWS, len(ordered)))
    result = {}
    if not candidate_panels_by_arm:
        raise ValueError("at least one pooled A panel is required")
    for arm, candidate_values in candidate_panels_by_arm.items():
        panel = np.asarray(candidate_values, dtype=np.float64)
        if panel.shape != (3, len(rows)) or not np.isfinite(panel).all():
            raise ValueError(f"{arm} pooled A panel must have shape (3, N)")
        delta = np.stack(
            [family_matrix(panel[i] - base[i], families)[1] for i in range(3)]
        )
        seed_gains = delta.mean(axis=1)
        boot = np.median(delta[:, ix].mean(axis=2), axis=0)
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
        result[arm] = {
            "median_gain": float(np.median(seed_gains)),
            "mean_gain": float(delta.mean()),
            "per_seed_gain": seed_gains.tolist(),
            "positive_seed_count": int(np.count_nonzero(seed_gains > 0)),
            "median_seed_family_bootstrap_ci95": ci,
            "advance_a_gate": bool(
                float(np.median(seed_gains)) > 0.0
                and np.all(seed_gains > 0.0)
                and ci[0] > 0.0
            ),
        }
    return result


def family_elo_summary(rows, r882_values, panels_by_arm, draws=1000, seed=BOOT_SEED):
    """Canonical paired family win/tie Elo against R882, with family bootstrap CI."""
    families = np.asarray([row["family_id"] for row in rows], dtype=object)
    ordered = sorted(set(families.tolist()))
    if len(ordered) != 205:
        raise ValueError("Elo requires all 205 families")
    all_panels = {"R882": np.asarray(r882_values, dtype=np.float64), **panels_by_arm}
    names = list(all_panels)
    wins = np.zeros((len(ordered), len(names), len(names)), dtype=np.float64)
    for si in range(3):
        utility = np.stack(
            [np.asarray(all_panels[name][si], dtype=np.float64) for name in names],
            axis=1,
        )
        delta = utility[:, :, None] - utility[:, None, :]
        context_wins = np.where(delta > 1e-4, 1.0, np.where(delta < -1e-4, 0.0, 0.5))
        context_wins *= 1.0 - np.eye(len(names))[None, :, :]
        for fi, family in enumerate(ordered):
            wins[fi] += context_wins[families == family].mean(axis=0) / 3.0

    def ratings(matrix):
        games = matrix + matrix.T
        n = matrix.shape[0]
        strength = np.ones(n, dtype=np.float64)
        totals = matrix.sum(axis=1) + 0.01 * (n - 1)
        for _ in range(400):
            denom = (
                (games + 0.02 * (1 - np.eye(n)))
                / np.maximum(strength[:, None] + strength[None, :], 1e-12)
            ).sum(axis=1)
            strength = np.clip(totals / denom, 1e-6, 1e6)
            strength /= max(strength[0], 1e-12)
        return 1500.0 + 400.0 * np.log10(np.maximum(strength, 1e-9))

    point = ratings(wins.sum(axis=0))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(ordered), (draws, len(ordered)))
    bootstrap = np.stack([ratings(wins[ix].sum(axis=0)) for ix in sampled])
    return {
        name: {
            "elo": float(point[i]),
            "elo_delta_vs_r882": float(point[i] - point[0]),
            "elo_delta_ci95": np.quantile(
                bootstrap[:, i] - bootstrap[:, 0], [0.025, 0.975]
            ).tolist(),
        }
        for i, name in enumerate(names)
    }


def utility_summary(rows, panel):
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (3, len(rows)) or not np.isfinite(values).all():
        raise ValueError("utility panel must have shape (3, N)")
    families = [row["family_id"] for row in rows]
    per_seed = [float(family_matrix(values[i], families)[1].mean()) for i in range(3)]
    return {
        "per_seed_equal_family_utility": per_seed,
        "median_seed_equal_family_utility": float(np.median(per_seed)),
        "mean_seed_equal_family_utility": float(np.mean(per_seed)),
        "minus_two_context_count_per_seed": [
            int(np.count_nonzero(values[i] == -2.0)) for i in range(3)
        ],
    }


class StageOrderGuard:
    """H189 stage-order guard following H177's fsynced pooled-B lock pattern."""

    def __init__(self):
        self.b_read_started = False
        self.b_locked_path = None

    def mark_b_read(self):
        self.b_read_started = True

    def mark_b_locked(self, lock_path):
        path = Path(lock_path)
        if not path.is_file():
            raise RuntimeError("H189 pooled B lock is not durable")
        payload = json.loads(path.read_text())
        if payload.get("wave") != "H189" or payload.get("pooled_b_families") != 205:
            raise RuntimeError("invalid H189 B lock")
        if set(payload.get("decisions", {})) != set(ARMS):
            raise RuntimeError("H189 lock must contain exactly SWA and EMA decisions")
        self.b_locked_path = path

    def require_b_locked(self):
        if self.b_locked_path is None or not self.b_locked_path.is_file():
            raise RuntimeError(
                "A scoring is forbidden before the fsynced pooled B lock"
            )
        return json.loads(self.b_locked_path.read_text())


def write_json_fsync(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
