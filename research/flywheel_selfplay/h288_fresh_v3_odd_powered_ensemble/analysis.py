"""H288 statistics (hw_t0005): an arm's pick against r3175's own slot-0 pick on fresh cohort v3, one context per family.

Per "seed" (a model seed for seeded arms; a bootstrap seed 288150/288151/288152 on the SAME picks for a deterministic
arm): family-mean and family-median gain, family bootstrap CI95 (10,000 draws), W/T/L with ties (|d| <= 1e-4) NOT
wins, one-sided exact sign test on non-tied families, harmed families (loss > 0.01), H278's two-contender Elo (slot 0 =
1500), invalid and changed picks. Across seeds: the median gain and a pooled CI (median over seeds of bootstrap means).

PRIMARY rule, per arm (ALL must hold, else NOT_TRANSFERRED_FRESH):
  (1) per-seed bootstrap LCB95 of the family-mean gain > 0 on 3/3 seeds;
  (2) median across seeds of the family-mean gain > +0.010;
  (3) one-sided exact sign test p < 0.05 on 3/3 seeds (NEW vs H286; tail resistance);
  (4) harmed families (gain < -0.01) <= 35% on every seed;
  (5) invalid (-2.0) picks not above slot 0's on any seed;
  (6) BOTH nulls (SEEDED_RANDOM_VALID_SLOT and SEEDED_RANDOM_SLOT_NO_LABELS) fail (1)-(5).
INSUFFICIENT_FRESH if fewer than 80 families build.

SECONDARY (cannot flip): 10%-trimmed family-mean gain + CI; normalized gain (arm - slot0) / (oracle_top8 - slot0) with
gap <= 1e-4 dropped, median + CI."""

from math import comb

import numpy as np

BOOT_SEED = 288150
DETERMINISTIC_BOOT_SEEDS = (288150, 288151, 288152)
DRAWS = 10000
TIE = 1e-4
HARM_LOSS = 0.01
HARM_MAX_SHARE = 0.35
MEDIAN_MIN = 0.010
SIGN_P = 0.05
MIN_FAMILIES = 80
TRIM = 0.10
VALID_NULL_SEEDS = (288101, 288102, 288103)
NOLABEL_NULL_SEEDS = (288201, 288202, 288203)


def bt_two(wins_ab, wins_ba, prior=0.01, iters=400):
    """H278's two-contender MM Bradley-Terry (incumbent anchored at 1500); returns the challenger's Elo."""
    s_a, s_b = 1.0, 1.0
    games = wins_ab + wins_ba
    for _ in range(iters):
        s_b = (wins_ba + prior) / ((games + 2 * prior) / (s_a + s_b))
        s_b = float(np.clip(s_b / s_a, 1e-6, 1e6))
        s_a = 1.0
    return 1500 + 400 * np.log10(s_b)


def sign_test_one_sided(wins, losses):
    """Exact one-sided binomial p = P(X >= wins | n = wins + losses, 1/2); ties excluded."""
    n = wins + losses
    if n == 0:
        return 1.0
    return float(sum(comb(n, k) for k in range(wins, n + 1)) / 2**n)


def family_diffs(u, families):
    fams = sorted(set(families))
    f = np.asarray(families)
    return fams, np.array([u[f == x, 1].mean() - u[f == x, 0].mean() for x in fams])


def trimmed_mean(x, trim=TRIM):
    x = np.sort(np.asarray(x, dtype=np.float64))
    k = int(np.floor(trim * len(x)))
    return float(x[k : len(x) - k].mean()) if len(x) - 2 * k > 0 else float("nan")


def summarize(u, inv, families, picks, boot_seeds=(BOOT_SEED,) * 3):
    """u, inv: (3, contexts, 2) = [slot0, arm]; picks: (3, contexts)."""
    u, inv = np.asarray(u, dtype=np.float64), np.asarray(inv, dtype=bool)
    fams = sorted(set(families))
    D = np.stack([family_diffs(u[s], families)[1] for s in range(3)])
    boots = []
    for s in range(3):
        ix = np.random.default_rng(boot_seeds[s]).integers(
            0, len(fams), (DRAWS, len(fams))
        )
        boots.append(D[s][ix].mean(axis=1))
    boots = np.stack(boots)
    ci = [np.quantile(b, [0.025, 0.975]).tolist() for b in boots]
    gains = D.mean(axis=1)
    wtl = [
        [int((d > TIE).sum()), int((abs(d) <= TIE).sum()), int((d < -TIE).sum())]
        for d in D
    ]
    out = dict(
        families=len(fams),
        boot_seeds=list(boot_seeds),
        per_seed_family_mean_gain=gains.tolist(),
        per_seed_family_median_gain=np.median(D, axis=1).tolist(),
        per_seed_ci95=ci,
        median_gain_across_seeds=float(np.median(gains)),
        pooled_ci95=np.quantile(np.median(boots, axis=0), [0.025, 0.975]).tolist(),
        wtl_by_seed=wtl,
        sign_test_p_by_seed=[sign_test_one_sided(w, lo) for w, _, lo in wtl],
        elo_by_seed=[float(bt_two(float(lo), float(w))) for w, _, lo in wtl],
        harmed_share_by_seed=[float((d < -HARM_LOSS).mean()) for d in D],
        harmed_families_by_seed=[
            [f for f, x in zip(fams, d) if x < -HARM_LOSS] for d in D
        ],
        invalid_by_seed=[[int(v[:, 0].sum()), int(v[:, 1].sum())] for v in inv],
        changed_picks_by_seed=[int(sum(int(j) != 0 for j in p)) for p in picks],
        slot0_context_mean=float(u[0, :, 0].mean()),
        arm_context_mean_by_seed=[float(u[s, :, 1].mean()) for s in range(3)],
    )
    out["criteria"] = dict(
        lcb_positive_3of3=bool(all(c[0] > 0 for c in ci)),
        median_above_min=bool(out["median_gain_across_seeds"] > MEDIAN_MIN),
        sign_test_3of3=bool(all(p < SIGN_P for p in out["sign_test_p_by_seed"])),
        harmed_share_ok=bool(
            all(h <= HARM_MAX_SHARE for h in out["harmed_share_by_seed"])
        ),
        invalid_ok=bool(all(c <= i for i, c in out["invalid_by_seed"])),
    )
    out["passes"] = bool(all(out["criteria"].values()))
    out["_D"] = D
    return out


def verdict(n_families, arm, nulls):
    if n_families < MIN_FAMILIES:
        return "INSUFFICIENT_FRESH"
    return (
        "FRESH_TRANSFER_PASS"
        if arm["passes"] and not any(n["passes"] for n in nulls)
        else "NOT_TRANSFERRED_FRESH"
    )


def secondary(summary, u, families, oracle_by_ctx):
    D = summary["_D"]
    fams = sorted(set(families))
    f = np.asarray(families)
    oracle = np.asarray(oracle_by_ctx, dtype=np.float64)
    out = dict(trimmed_mean_gain=[], normalized_gain=[])
    for s in range(3):
        rng = np.random.default_rng(summary["boot_seeds"][s])
        ix = rng.integers(0, len(fams), (DRAWS, len(fams)))
        tb = np.array(
            [trimmed_mean(D[s][row]) for row in ix[:2000]]
        )  # 2,000 draws for the trimmed-mean CI
        out["trimmed_mean_gain"].append(
            dict(
                value=trimmed_mean(D[s]), ci95=np.quantile(tb, [0.025, 0.975]).tolist()
            )
        )
        gap = np.array(
            [oracle[f == x].mean() - np.asarray(u)[s, f == x, 0].mean() for x in fams]
        )
        keep = gap > TIE
        g = D[s][keep] / gap[keep]
        if len(g):
            gix = np.random.default_rng(summary["boot_seeds"][s]).integers(
                0, len(g), (DRAWS, len(g))
            )
            gci = np.quantile(np.median(g[gix], axis=1), [0.025, 0.975]).tolist()
            out["normalized_gain"].append(
                dict(
                    kept=int(keep.sum()),
                    dropped=int((~keep).sum()),
                    median=float(np.median(g)),
                    ci95=gci,
                )
            )
        else:
            out["normalized_gain"].append(
                dict(kept=0, dropped=int(len(keep)), median=None, ci95=None)
            )
    return out


def public(summary):
    return {k: v for k, v in summary.items() if not k.startswith("_")}


def random_valid_slot_picks(gate_k_by_ctx, seed):
    """SEEDED_RANDOM_VALID_SLOT (H283): uniform over the valid (gated) slots 1..7; slot 0 if none is valid."""
    rng = np.random.default_rng(seed)
    picks = []
    for g in gate_k_by_ctx:
        valid = [j for j in range(1, len(g)) if g[j]]
        picks.append(int(rng.choice(valid)) if valid else 0)
    return picks


def random_slot_no_label_picks(n_ctx, seed, k=8):
    """SEEDED_RANDOM_SLOT_NO_LABELS: uniform over all top-8 slots 0..7, using no validity information."""
    return np.random.default_rng(seed).integers(0, k, n_ctx).tolist()
