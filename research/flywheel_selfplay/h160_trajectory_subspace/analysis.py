"""H160 trajectory-subspace basis/projection, H148 family harness, strict selection, inference and anchored BT Elo.

Partition, batch plan, strict snapshot selection and canonical Bradley-Terry Elo are copied from merged H148
(research/h148/analysis.py) with only the H160 seeds, arms and constants changed.
"""

import hashlib
import random

import numpy as np

SEEDS = (160101, 160102, 160103)
FOLD_SEED = 160140
BOOT_SEED = 160141
STEPS = (0, 10, 20, 30, 40)
DRAWS = 10000
ARMS = ("CTRL", "PROJ_K2", "PROJ_K4", "CTRL_DRIFT")
PRIMARY = ("PROJ_K2", "PROJ_K4")
K = {"PROJ_K2": 2, "PROJ_K4": 4}
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4
CHAIN = (
    95,
    129,
    130,
    256,
    782,
    785,
    859,
    882,
)  # accepted incumbents, round order (R774 == R256 adapter)
DRIFT_MATCH_BOUNDS = (0.85, 1.15)
MARGIN = 0.001
TIE = 1e-12


# ------------------------------------------------------------------ trajectory subspace


def basis(chain_vectors, k):
    """Top-k right singular vectors of the uncentered consecutive-delta matrix (rows = consecutive deltas)."""
    v = np.asarray(chain_vectors, dtype=np.float64)
    if v.ndim != 2 or len(v) < 2 or not np.isfinite(v).all():
        raise ValueError("finite chain of at least two adapter vectors required")
    deltas = np.diff(v, axis=0)
    _, s, vt = np.linalg.svd(deltas, full_matrices=False)
    rank = int((s > s[0] * 1e-9).sum())
    if k > rank:
        raise ValueError(f"k={k} exceeds delta rank {rank}")
    energy = (s**2).cumsum() / (s**2).sum()
    return vt[:k].T.copy(), dict(
        singular_values=s.tolist(),
        cumulative_energy=energy.tolist(),
        rank=rank,
        k=k,
        explained_energy=float(energy[k - 1]),
        delta_l2=np.linalg.norm(deltas, axis=1).tolist(),
    )


def project(u, x):
    """Orthogonal projection of x onto span(u) (u has orthonormal columns)."""
    u = np.asarray(u, dtype=np.float64)
    return u @ (u.T @ np.asarray(x, dtype=np.float64))


def out_of_span(u, x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    return 0.0 if n == 0 else float(np.linalg.norm(x - project(u, x)) / n)


def drift_lr(proj_k4_drifts, ctrl_drifts, base_lr=BASE_LR):
    """CTRL_DRIFT learning rate: base lr scaled by median PROJ_K4 drift / median CTRL drift (fixed before its cells)."""
    a, c = float(np.median(proj_k4_drifts)), float(np.median(ctrl_drifts))
    if not (np.isfinite(a) and np.isfinite(c)) or a <= 0 or c <= 0:
        raise ValueError("positive finite drifts required")
    return base_lr * a / c


def drift_match(target_drifts, matched_drifts, bounds=DRIFT_MATCH_BOUNDS):
    t, m = float(np.median(target_drifts)), float(np.median(matched_drifts))
    return dict(
        target_median=t,
        matched_median=m,
        ratio=m / t,
        bounds=list(bounds),
        valid=bool(bounds[0] * t <= m <= bounds[1] * t),
    )


# ------------------------------------------------------------------ H148 harness (copied, H160 constants)


def folds(rows):
    ids = [r["context_sha256"] for r in rows]
    families = sorted({r["family_id"] for r in rows})
    if len(ids) != len(set(ids)) or len(families) < 5:
        raise ValueError("unique contexts and at least five families required")
    return [
        sorted(g.tolist())
        for g in np.array_split(
            np.random.default_rng(FOLD_SEED).permutation(families), 5
        )
    ]


def partition(rows, outer):
    groups = folds(rows)
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    fit = [r for r in rows if r["family_id"] not in a | b]
    select = [r for r in rows if r["family_id"] in b]
    assess = [r for r in rows if r["family_id"] in a]
    assert not a & b and not {r["family_id"] for r in fit} & (a | b)
    return fit, select, assess


def batch_plan(fit_ids, orders, seed):
    """H140 paired stream: 20 rounds, consume 8 draws but retain first 2 steps."""
    if seed not in SEEDS or not fit_ids:
        raise ValueError("registered seed and fit contexts required")
    ids = tuple(sorted(fit_ids))
    sampler = seed * 1_000_003 + 10_007
    plan = []
    for rnd in range(1, 21):
        rng = np.random.RandomState((sampler + rnd * 104729) % 2**32)
        rot = rnd % len(ids)
        rotated = ids[rot:] + ids[:rot]
        for step in range(8):
            stream = int.from_bytes(
                hashlib.sha256(
                    f"flywheel-batch/v1:{sampler}:{rnd}:{step}".encode()
                ).digest()[:8],
                "little",
            )
            cids = random.Random(stream).choices(rotated, k=16)
            indexes = []
            for cid in cids:
                order = orders[cid]
                k = min(48, 192, len(order))
                rest = order[k:] if len(order) > k else order
                indexes.append(
                    np.concatenate(
                        [
                            order[:k],
                            rng.choice(rest, 192 - k, replace=len(rest) < 192 - k),
                        ]
                    )
                )
            if step < 2:
                plan.append((stream, cids, indexes))
    return plan


def family_matrix(values, families):
    order = sorted(set(families))
    matrix = np.asarray(
        [
            np.asarray(values)[np.asarray(families) == family].mean(axis=0)
            for family in order
        ]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite complete panel")
    return order, matrix


def bt_ratings(wins, anchor, prior=0.01):
    """Canonical 400-iteration MM, with optional leading bootstrap dimension."""
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    offdiag = 1.0 - np.eye(n)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    strengths = np.ones(wins.shape[:-1])
    for _ in range(400):
        denom = (
            (games + 2.0 * prior * offdiag)
            / np.maximum(strengths[..., :, None] + strengths[..., None, :], 1e-12)
        ).sum(axis=-1)
        strengths = np.clip(totals / denom, 1e-6, 1e6)
        strengths /= np.maximum(strengths[..., anchor, None], 1e-12)
    return 1500.0 + 400.0 * np.log10(np.maximum(strengths, 1e-9))


def family_wins(values, families, tie_margin=1e-4):
    diff = values[:, :, None] - values[:, None, :]
    outcomes = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5))
    outcomes *= 1.0 - np.eye(values.shape[1])
    return family_matrix(outcomes, families)[1]


def choose(rows, panel, seed):
    """No assessment argument. Earliest best step; reject unless gain > 0.001 (ties within 1e-12) and paired LB > 0."""
    values = np.asarray(panel, dtype=float)
    if values.shape != (len(rows), len(STEPS)):
        raise ValueError("complete five-snapshot selector required")
    _, fm = family_matrix(values, [r["family_id"] for r in rows])
    means = fm.mean(axis=0)
    best = int(np.argmax(means))
    delta = fm[:, best] - fm[:, 0]
    ix = np.random.default_rng(seed).integers(0, len(fm), (DRAWS, len(fm)))
    ci = np.quantile(delta[ix].mean(axis=1), [0.025, 0.975]).tolist()
    gain = float(delta.mean())
    accepted = gain > MARGIN + TIE and ci[0] > 0
    return dict(
        best_step=STEPS[best],
        selected_step=STEPS[best] if accepted else 0,
        gain=gain,
        ci95=ci,
        accepted=bool(accepted),
        family_means=means.tolist(),
    )


def selection_seed(outer, seed_index, arm):
    return BOOT_SEED + 100 * outer + 10 * seed_index + ARMS.index(arm) + 1


# ------------------------------------------------------------------ inference and decision


def summarize(rows, panels):
    """panels: [3 seeds, contexts, 1 + len(ARMS)] with column 0 = r882. Equal-family, shared family bootstrap."""
    values = np.asarray(panels, dtype=float)
    cols = ("r882",) + ARMS
    if values.shape != (3, len(rows), len(cols)) or not np.isfinite(values).all():
        raise ValueError("complete finite three-seed panel required")
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])  # [3, F, C]
    n = fm.shape[1]
    ix = np.random.default_rng(BOOT_SEED).integers(0, n, (DRAWS, n))
    means = fm.mean(axis=1)  # [3, C]
    boot = fm[:, ix].mean(axis=2)  # [3, DRAWS, C]
    contrasts = {}
    pairs = [(f"{a}_minus_r882", a, "r882") for a in ARMS] + [
        ("PROJ_K4_minus_CTRL_DRIFT", "PROJ_K4", "CTRL_DRIFT")
    ]
    for name, left, right in pairs:
        li, ri = cols.index(left), cols.index(right)
        per_seed = means[:, li] - means[:, ri]
        draws = np.median(boot[:, :, li] - boot[:, :, ri], axis=0)
        contrasts[name] = dict(
            per_seed_gain=per_seed.tolist(),
            median_gain=float(np.median(per_seed)),
            ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
            p_one_sided=float((1 + (draws <= 0).sum()) / (DRAWS + 1)),
            positive_seeds=int((per_seed > 0).sum()),
        )
    fw = np.stack([family_wins(v, families) for v in values])
    elo = bt_ratings(fw.sum(axis=1), 0)
    ei = np.random.default_rng(BOOT_SEED + 1).integers(0, n, (1000, n))
    elo_boot = np.median(bt_ratings(fw[:, ei].sum(axis=2), 0), axis=0)
    return dict(
        contexts=len(rows),
        families=n,
        columns=list(cols),
        contrasts=contrasts,
        family_utility_by_seed=means.tolist(),
        elo_by_seed=elo.tolist(),
        median_elo=dict(zip(cols, np.median(elo, axis=0).tolist())),
        elo_ci95=dict(
            zip(cols, np.quantile(elo_boot, [0.025, 0.975], axis=0).T.tolist())
        ),
        identical_seed_panels=bool(
            np.array_equal(values[0], values[1])
            and np.array_equal(values[1], values[2])
        ),
    )


def decide(primary, nonzero_picks, match):
    """Holm (m=2) over PROJ_K2/PROJ_K4 vs r882 on the assessment panel; mechanism contrast PROJ_K4 vs CTRL_DRIFT."""
    p = {a: primary["contrasts"][f"{a}_minus_r882"]["p_one_sided"] for a in PRIMARY}
    out, adjusted = {}, 0.0
    for rank, arm in enumerate(sorted(PRIMARY, key=lambda a: (p[a], a))):
        adjusted = max(adjusted, min(1.0, p[arm] * (len(PRIMARY) - rank)))
        c = primary["contrasts"][f"{arm}_minus_r882"]
        gates = dict(
            margin=c["median_gain"] > MARGIN + TIE,
            ci95_lower_gt_0=c["ci95"][0] > 0,
            two_of_three_seeds=c["positive_seeds"] >= 2,
            holm=adjusted < 0.05,
            nonzero_selection=nonzero_picks[arm] > 0,
        )
        out[arm] = dict(
            holm_adjusted_p=adjusted, gates=gates, advance=all(gates.values())
        )
    m = primary["contrasts"]["PROJ_K4_minus_CTRL_DRIFT"]
    if not match["valid"]:
        mechanism = "DRIFT_MATCH_INVALID"
    elif nonzero_picks["PROJ_K4"] == 0 and nonzero_picks["CTRL_DRIFT"] == 0:
        mechanism = "NO_SELECTION_BOTH"
    elif m["median_gain"] > 0 and m["ci95"][0] > 0:
        mechanism = "SUBSPACE_SPECIFIC"
    else:
        mechanism = "STEP_SIZE_ONLY"
    if all(nonzero_picks[a] == 0 for a in ARMS):
        verdict = "RULE_NEUTRAL"
    elif any(out[a]["advance"] for a in PRIMARY):
        verdict = "ADVANCE"
    else:
        verdict = "NO_ADVANCE"
    return dict(verdict=verdict, mechanism=mechanism, primary_arms=out)
