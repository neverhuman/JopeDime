"""H286 statistics (hw_t2325): H283's analysis unchanged except BOOT_SEED 286150 and MIN_FAMILIES 80, plus the
pre-declared SECONDARY measures (sign test, normalized gain), which cannot flip the verdict.

H283 statistics (hw_t2305): frozen RERANK8_VETO (or the random null) against r3175's own slot-0 pick on the fresh
cohort. One context per family, so a family's gain is that context's utility difference.

Per seed: family-mean and family-median gain, family bootstrap CI95 (10,000 draws, seed 283150), W/T/L with ties
(|d| <= 1e-4) NOT wins, harmed families (loss > 0.01), H278's two-contender Elo (slot 0 = 1500), invalid and changed
picks. Across seeds: median gain and an H278-style pooled CI (median over seeds of common-draw bootstrap means).

Primary rule (ALL must hold, else NOT_TRANSFERRED_FRESH):
  (1) per-seed bootstrap LCB95 of the family-mean gain > 0 on 3/3 seeds;
  (2) median across the 3 seeds of the family-mean gain > +0.010;
  (3) harmed families (gain < -0.01) <= 35% of families on every seed;
  (4) invalid (-2.0) picks not above slot 0's on any seed;
  (5) the SEEDED_RANDOM_VALID_SLOT null does NOT pass (1)-(4).
INSUFFICIENT_FRESH if fewer than 25 families build."""

import numpy as np

BOOT_SEED = 286150
DRAWS = 10000
TIE = 1e-4
HARM_LOSS = 0.01
HARM_MAX_SHARE = 0.35
MEDIAN_MIN = 0.010
MIN_FAMILIES = 80
WATCH = ("fresh:",)


def bt_two(wins_ab, wins_ba, prior=0.01, iters=400):
    """H278's two-contender MM Bradley-Terry (incumbent anchored at 1500); returns the challenger's Elo."""
    s_a, s_b = 1.0, 1.0
    games = wins_ab + wins_ba
    for _ in range(iters):
        s_b = (wins_ba + prior) / ((games + 2 * prior) / (s_a + s_b))
        s_b = float(np.clip(s_b / s_a, 1e-6, 1e6))
        s_a = 1.0
    return 1500 + 400 * np.log10(s_b)


def family_gains(u, families):
    fams = sorted(set(families))
    f = np.asarray(families)
    return fams, np.array([u[f == x, 1].mean() - u[f == x, 0].mean() for x in fams])


def summarize(u, inv, families, picks):
    """u, inv: (3 seeds, contexts, 2) = [slot0, challenger]; picks: (3, contexts) slot indices."""
    u, inv = np.asarray(u, dtype=np.float64), np.asarray(inv, dtype=bool)
    fams = sorted(set(families))
    D = np.stack([family_gains(u[s], families)[1] for s in range(3)])
    ix = np.random.default_rng(BOOT_SEED).integers(0, len(fams), (DRAWS, len(fams)))
    boots = D[:, ix].mean(axis=2)  # (3, DRAWS), common draws
    per_seed_ci = [np.quantile(b, [0.025, 0.975]).tolist() for b in boots]
    pooled = np.median(boots, axis=0)
    gains = D.mean(axis=1)
    out = dict(
        families=len(fams),
        per_seed_family_mean_gain=gains.tolist(),
        per_seed_family_median_gain=np.median(D, axis=1).tolist(),
        per_seed_ci95=per_seed_ci,
        median_gain_across_seeds=float(np.median(gains)),
        pooled_ci95=np.quantile(pooled, [0.025, 0.975]).tolist(),
        wtl_by_seed=[
            [int((d > TIE).sum()), int((abs(d) <= TIE).sum()), int((d < -TIE).sum())]
            for d in D
        ],
        elo_by_seed=[
            float(bt_two(float((d < -TIE).sum()), float((d > TIE).sum()))) for d in D
        ],
        harmed_share_by_seed=[float((d < -HARM_LOSS).mean()) for d in D],
        harmed_families_by_seed=[
            [f for f, x in zip(fams, d) if x < -HARM_LOSS] for d in D
        ],
        invalid_by_seed=[[int(v[:, 0].sum()), int(v[:, 1].sum())] for v in inv],
        changed_picks_by_seed=[int(sum(j != 0 for j in p)) for p in picks],
        slot0_context_mean=float(u[0, :, 0].mean()),
        challenger_context_mean_by_seed=[float(u[s, :, 1].mean()) for s in range(3)],
    )
    out["criteria"] = dict(
        lcb_positive_3of3=bool(all(ci[0] > 0 for ci in per_seed_ci)),
        median_above_min=bool(out["median_gain_across_seeds"] > MEDIAN_MIN),
        harmed_share_ok=bool(
            all(h <= HARM_MAX_SHARE for h in out["harmed_share_by_seed"])
        ),
        invalid_ok=bool(all(c <= i for i, c in out["invalid_by_seed"])),
    )
    out["passes"] = bool(all(out["criteria"].values()))
    return out


def verdict(n_families, frozen, null):
    if n_families < MIN_FAMILIES:
        return "INSUFFICIENT_FRESH"
    return (
        "FRESH_TRANSFER_PASS"
        if frozen["passes"] and not null["passes"]
        else "NOT_TRANSFERRED_FRESH"
    )


def random_valid_slot_picks(gate_k_by_ctx, seed):
    """SEEDED_RANDOM_VALID_SLOT: uniform over the valid (gated) slots 1..7; slot 0 if none is valid."""
    rng = np.random.default_rng(seed)
    picks = []
    for g in gate_k_by_ctx:
        valid = [j for j in range(1, len(g)) if g[j]]
        picks.append(int(rng.choice(valid)) if valid else 0)
    return picks


def sign_test_one_sided(wins, losses):
    """Exact one-sided binomial p = P(X >= wins | n = wins + losses, 1/2); ties excluded."""
    from math import comb

    n = wins + losses
    if n == 0:
        return 1.0
    return float(sum(comb(n, k) for k in range(wins, n + 1)) / 2**n)


def secondary(u, families, oracle_by_ctx):
    """(a) one-sided exact sign test on non-tied families per seed (p < 0.05 on 3/3 seeds);
    (b) normalized gain (arm - slot0) / (oracle_top8 - slot0) per family, gap <= 1e-4 dropped; median + CI per seed."""
    u = np.asarray(u, dtype=np.float64)
    fams = sorted(set(families))
    f = np.asarray(families)
    oracle = np.asarray(oracle_by_ctx, dtype=np.float64)
    sign, norm = [], []
    for s in range(3):
        d = np.array([u[s, f == x, 1].mean() - u[s, f == x, 0].mean() for x in fams])
        w, lo = int((d > TIE).sum()), int((d < -TIE).sum())
        sign.append(
            dict(
                wins=w,
                losses=lo,
                ties=int(len(d) - w - lo),
                p_one_sided=sign_test_one_sided(w, lo),
            )
        )
        gap = np.array([oracle[f == x].mean() - u[s, f == x, 0].mean() for x in fams])
        keep = gap > TIE
        g = d[keep] / gap[keep]
        ci = boot_median_ci(g)
        norm.append(
            dict(
                kept=int(keep.sum()),
                dropped=int((~keep).sum()),
                median=float(np.median(g)) if len(g) else None,
                ci95=ci,
            )
        )
    return dict(
        sign_test=sign,
        sign_test_pass_3of3=bool(all(x["p_one_sided"] < 0.05 for x in sign)),
        normalized_gain=norm,
    )


def boot_median_ci(g):
    if len(g) == 0:
        return None
    ix = np.random.default_rng(BOOT_SEED).integers(0, len(g), (DRAWS, len(g)))
    return np.quantile(np.median(g[ix], axis=1), [0.025, 0.975]).tolist()
