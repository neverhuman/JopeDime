"""H292 statistics (hw_t0125): the confirmatory rule for the TWO pre-declared primaries ENS_VOTE2 and CLIP, Holm-corrected
at one-sided family-wise alpha 0.05, against r3175's own slot-0 pick on fresh cohort v4 (one context per family).

Per "seed" (CLIP: its 3 refit seeds with common bootstrap draws, seed 292150; ENS_VOTE2: a deterministic single policy,
so its 3 "seeds" are bootstrap seeds 292150/292151/292152 on the same picks):
  family-mean gain d_s; a 10,000-draw family bootstrap B_s of the mean; one-sided bootstrap p_boot_s = P(B_s <= 0);
  one-sided exact sign test p_sign_s on non-tied families (ties |d| <= 1e-4 are never wins).

Holm (2 hypotheses, one-sided alpha 0.05):
  arm p-value p_a = max over seeds of max(p_boot_s, p_sign_s)  (intersection-union: every seed must pass);
  order the arms by p_a (ties: ENS_VOTE2 first); test the first at alpha/2 = 0.025, the second at alpha = 0.05,
  stopping at the first non-confirmation.
  An arm tested at level a_k is CONFIRMED_FRESH iff ALL hold:
    (1) Holm-adjusted LCB = the a_k quantile of B_s is > 0 on 3/3 seeds (first tested: 97.5% one-sided intervals);
    (2) the median across seeds of d_s is > +0.010;
    (3) p_sign_s < a_k on 3/3 seeds;
    (4) harmed families (gain < -0.01) <= 35% on every seed;
    (5) invalid (-2.0) picks are not above slot 0's on any seed;
    (6) both nulls fail the same criteria evaluated at the most lenient level (alpha 0.05).
  Otherwise NOT_CONFIRMED_FRESH. Fewer than 80 families built -> INSUFFICIENT_FRESH for both.
Secondary/descriptive arms (BAG_SWITCH, ENS_MEAN, ENS_VOTE3, RERANK8_VETO) are reported at the unadjusted H288 levels
and cannot flip a verdict."""

from math import comb

import numpy as np

ALPHA = 0.05
BOOT_SEED = 292150
DETERMINISTIC_BOOT_SEEDS = (292150, 292151, 292152)
DRAWS = 10000
TIE = 1e-4
HARM_LOSS = 0.01
HARM_MAX_SHARE = 0.35
MEDIAN_MIN = 0.010
MIN_FAMILIES = 80
VALID_NULL_SEEDS = (292101, 292102, 292103)
NOLABEL_NULL_SEEDS = (292201, 292202, 292203)
PRIMARIES = ("ENS_VOTE2", "CLIP")


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
    n = wins + losses
    if n == 0:
        return 1.0
    return float(sum(comb(n, k) for k in range(wins, n + 1)) / 2**n)


def summarize(u, inv, families, picks, boot_seeds=(BOOT_SEED,) * 3):
    """u, inv: (3, contexts, 2) = [slot0, arm]; picks: (3, contexts)."""
    u, inv = np.asarray(u, dtype=np.float64), np.asarray(inv, dtype=bool)
    fams = sorted(set(families))
    f = np.asarray(families)
    D = np.stack(
        [
            np.array([u[s, f == x, 1].mean() - u[s, f == x, 0].mean() for x in fams])
            for s in range(3)
        ]
    )
    B = np.stack(
        [
            D[s][
                np.random.default_rng(boot_seeds[s]).integers(
                    0, len(fams), (DRAWS, len(fams))
                )
            ].mean(axis=1)
            for s in range(3)
        ]
    )
    wtl = [
        [int((d > TIE).sum()), int((abs(d) <= TIE).sum()), int((d < -TIE).sum())]
        for d in D
    ]
    gains = D.mean(axis=1)
    out = dict(
        families=len(fams),
        boot_seeds=list(boot_seeds),
        per_seed_family_mean_gain=gains.tolist(),
        per_seed_family_median_gain=np.median(D, axis=1).tolist(),
        per_seed_ci95=[np.quantile(b, [0.025, 0.975]).tolist() for b in B],
        per_seed_lcb_one_sided={
            "0.025": [float(np.quantile(b, 0.025)) for b in B],
            "0.05": [float(np.quantile(b, 0.05)) for b in B],
        },
        per_seed_p_boot=[float((b <= 0).mean()) for b in B],
        median_gain_across_seeds=float(np.median(gains)),
        pooled_ci95=np.quantile(np.median(B, axis=0), [0.025, 0.975]).tolist(),
        wtl_by_seed=wtl,
        sign_test_p_by_seed=[sign_test_one_sided(w, lo) for w, _, lo in wtl],
        elo_by_seed=[float(bt_two(float(lo), float(w))) for w, _, lo in wtl],
        harmed_share_by_seed=[float((d < -HARM_LOSS).mean()) for d in D],
        harmed_families_by_seed=[
            [x for x, v in zip(fams, d) if v < -HARM_LOSS] for d in D
        ],
        invalid_by_seed=[[int(v[:, 0].sum()), int(v[:, 1].sum())] for v in inv],
        changed_picks_by_seed=[int(sum(int(j) != 0 for j in p)) for p in picks],
        slot0_context_mean=float(u[0, :, 0].mean()),
        arm_context_mean_by_seed=[float(u[s, :, 1].mean()) for s in range(3)],
    )
    out["arm_p"] = float(
        max(
            max(a, b)
            for a, b in zip(out["per_seed_p_boot"], out["sign_test_p_by_seed"])
        )
    )
    out["_B"] = B
    return out


def criteria(s, level):
    """The rule at one-sided level `level` (0.025 or 0.05)."""
    lcb = [float(np.quantile(b, level)) for b in s["_B"]]
    return dict(
        level=level,
        lcb=lcb,
        lcb_positive_3of3=bool(all(x > 0 for x in lcb)),
        median_above_min=bool(s["median_gain_across_seeds"] > MEDIAN_MIN),
        sign_test_3of3=bool(all(p < level for p in s["sign_test_p_by_seed"])),
        harmed_share_ok=bool(
            all(h <= HARM_MAX_SHARE for h in s["harmed_share_by_seed"])
        ),
        invalid_ok=bool(all(c <= i for i, c in s["invalid_by_seed"])),
    )


def passes(c):
    return all(v for k, v in c.items() if k not in ("level", "lcb"))


def holm(summaries, nulls, n_families):
    """Holm over the two primaries; returns per-arm verdicts and the procedure record."""
    null_fail = {k: not passes(criteria(v, ALPHA)) for k, v in nulls.items()}
    if n_families < MIN_FAMILIES:
        return {a: "INSUFFICIENT_FRESH" for a in PRIMARIES}, dict(null_fail=null_fail)
    order = sorted(PRIMARIES, key=lambda a: (summaries[a]["arm_p"], PRIMARIES.index(a)))
    verdicts, steps, stopped = {}, [], False
    for k, arm in enumerate(order):
        level = ALPHA / (len(order) - k)
        c = criteria(summaries[arm], level)
        ok = (not stopped) and passes(c) and all(null_fail.values())
        steps.append(
            dict(
                arm=arm,
                arm_p=summaries[arm]["arm_p"],
                level=level,
                criteria=c,
                tested=not stopped,
                confirmed=ok,
            )
        )
        verdicts[arm] = "CONFIRMED_FRESH" if ok else "NOT_CONFIRMED_FRESH"
        if not ok:
            stopped = True
    return verdicts, dict(order=order, steps=steps, null_fail=null_fail)


def public(summary):
    return {k: v for k, v in summary.items() if not k.startswith("_")}


def random_valid_slot_picks(gate_k_by_ctx, seed):
    rng = np.random.default_rng(seed)
    out = []
    for g in gate_k_by_ctx:
        valid = [j for j in range(1, len(g)) if g[j]]
        out.append(int(rng.choice(valid)) if valid else 0)
    return out


def random_slot_no_label_picks(n_ctx, seed, k=8):
    return np.random.default_rng(seed).integers(0, k, n_ctx).tolist()


def source_group(family_id):
    """Source group of a fresh family id: 'srcfile:<stem prefix>', 'legacy:...', else 'dedupe_fingerprint'."""
    k = family_id.removeprefix("fresh:")
    return (
        k.split(":")[0] + ":" + k.split(":")[1][:8]
        if ":" in k
        else "dedupe_fingerprint"
    )
