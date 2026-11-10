"""H278 Part B statistics: per-seed family-equal and family-median gains of the frozen RERANK8_VETO pick against the
incumbent's own pick, pooled family bootstrap (10,000 draws, seed 278150), W/T/L with ties NOT counted as wins,
two-contender MM Bradley-Terry Elo (incumbent = 1500), invalid picks and harmed families (hw_t2140)."""

import numpy as np

SEEDS = (278101, 278102, 278103)
BOOT_SEED = 278150
DRAWS, HARM = 10000, 1e-4
WATCH_FAMILY_PREFIX = "fam-769243ee"
P2_BLOCK_REASON = (
    "P2 re-score gate failed: a native CPU forward of the read-only r3175 copy (identical to the cached r3175 costs, "
    "91/91 arrays) gives DL-Val91 mean 1.1366829009431036 vs the stored 1.1389572403748425 (90/91 contexts equal; "
    "context 18 is a near-tie, r3175 cost gap 3.7e-5, where the GPU-scored stored pick differs). The stored utility "
    "cannot be re-scored exactly without GPU (xbabe2 GPU unavailable), so P2 is BLOCKED and not substituted."
)


def family_matrix(values, families):
    values = np.asarray(values, dtype=np.float64)
    fams = sorted(set(families))
    return fams, np.array(
        [values[np.asarray(families) == f].mean(axis=0) for f in fams]
    )


def bt_two(wins_ab, wins_ba, prior=0.01, iters=400):
    """Two-contender MM Bradley-Terry with the incumbent anchored at 1500; returns the challenger's Elo."""
    s_a, s_b = 1.0, 1.0
    games = wins_ab + wins_ba
    for _ in range(iters):
        s_b = (wins_ba + prior) / ((games + 2 * prior) / (s_a + s_b))
        s_b = float(np.clip(s_b / s_a, 1e-6, 1e6))
        s_a = 1.0
    return 1500 + 400 * np.log10(s_b)


def summarize(rows, utilities, picks_invalid):
    """utilities: (3 seeds, contexts, 2) = [incumbent, FROZEN_VETO]; picks_invalid: (3, contexts, 2) booleans."""
    u = np.asarray(utilities, dtype=np.float64)
    families = [r["family_id"] for r in rows]
    fams, _ = family_matrix(u[0], families)
    fm = np.stack([family_matrix(v, families)[1] for v in u])  # (3, F, 2)
    d = fm[:, :, 1] - fm[:, :, 0]  # (3, F)
    ix = np.random.default_rng(BOOT_SEED).integers(0, len(fams), (DRAWS, len(fams)))
    boot = np.median(d[:, ix].mean(axis=2), axis=0)
    per_seed = d.mean(axis=1)
    out = dict(
        contexts=len(rows),
        families=len(fams),
        per_seed_family_mean_gain=per_seed.tolist(),
        per_seed_family_median_gain=np.median(d, axis=1).tolist(),
        median_family_mean_gain=float(np.median(per_seed)),
        pooled_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        pooled_lcb95=float(np.quantile(boot, 0.025)),
        positive_seeds=int((per_seed > 0).sum()),
        wtl_by_seed=[
            [int((x > HARM).sum()), int((abs(x) <= HARM).sum()), int((x < -HARM).sum())]
            for x in d
        ],
        elo_challenger_by_seed=[
            bt_two(float((x < -HARM).sum()), float((x > HARM).sum())) for x in d
        ],
        invalid_picks_by_seed=[
            [int(v[:, 0].sum()), int(v[:, 1].sum())] for v in np.asarray(picks_invalid)
        ],
        harmed_families_by_seed=[
            [f for f, x in zip(fams, row) if x < -HARM] for row in d
        ],
        context_mean_by_seed=u.mean(axis=1).tolist(),
    )
    out["watch_family_harmed_by_seed"] = [
        [f for f in h if f.startswith(WATCH_FAMILY_PREFIX)]
        for h in out["harmed_families_by_seed"]
    ]
    return out


def p1_gate(s):
    """P1 (supporting): FROZEN_VETO > r882 on 3/3 seeds (H276 direction)."""
    return bool(s["positive_seeds"] == 3)


def adopt_gate(s):
    """ADOPT_SHADOW_PASS needs on P2: gain > 0 on 3/3 seeds, pooled LCB95 > 0, invalid picks not above the incumbent's
    (every seed); ties never count as wins."""
    inv_ok = all(c <= i for i, c in s["invalid_picks_by_seed"])
    return bool(s["positive_seeds"] == 3 and s["pooled_lcb95"] > 0 and inv_ok)
