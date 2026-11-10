"""H296 statistics (hw_t0240): dev91 terminal scores of the H296 candidates against the r3175 copy (per-context paired).

PRIMARY SWA_CYCLE passes iff on 3/3 seeds (ties never win):
  (1) canonical dev91 mean > 1.1389572403748425 + 0.001;
  (2) canonical dev91 mean > its paired CONTROL (16th iterate of the same trajectory);
  (3) family-grouped (19 dev families) bootstrap LCB95 (10,000 draws, seed 296150; families resampled with replacement,
      statistic = mean of the paired per-context deltas over the resampled contexts) > 0;
  (4) gain not concentrated: the top 4 positive per-context deltas are < 80% of the total positive gain (no positive
      gain fails).
Verdict DEV_ADVANCE (repeated-dev evidence only: a new fresh cohort is required before any adoption; no serving/live
change) or NOT_ADVANCED. Reported per candidate: lift, family-bootstrap CI, W/T/L contexts (|d| <= 1e-12 is a tie),
changed contexts, Elo (MM Bradley-Terry over contexts, r3175 = 1500) with a family-bootstrap 95% interval; for the primary
the median-of-seeds lift with a 95% family-bootstrap CI. EMA_POLYAK and SOUP_RET are descriptive."""

import numpy as np

STORED = 1.1389572403748425
MARGIN = 0.001
BOOT_SEED = 296150
DRAWS = 10000
TIE = 1e-12
TOP_K, CONC_MAX = 4, 0.80


def bt_two(wins_ab, wins_ba, prior=0.01, iters=400):
    """MM Bradley-Terry for two contenders (incumbent anchored at 1500; H278's bt_two)."""
    s_a, s_b = 1.0, 1.0
    games = wins_ab + wins_ba
    for _ in range(iters):
        s_b = (wins_ba + prior) / ((games + 2 * prior) / (s_a + s_b))
        s_b = float(np.clip(s_b / s_a, 1e-6, 1e6))
        s_a = 1.0
    return float(1500 + 400 * np.log10(s_b))


def family_draws(families):
    fams = sorted(set(families))
    f = np.asarray(families)
    members = [np.flatnonzero(f == x) for x in fams]
    ix = np.random.default_rng(BOOT_SEED).integers(0, len(fams), (DRAWS, len(fams)))
    return members, ix


def candidate_stats(per_context, base, families, selected=None, base_selected=None):
    d = np.asarray(per_context, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    members, ix = family_draws(families)
    boot, elo_boot = np.empty(DRAWS), np.empty(DRAWS)
    for k, row in enumerate(ix):
        sel = np.concatenate([members[j] for j in row])
        boot[k] = d[sel].mean()
        elo_boot[k] = bt_two(float((d[sel] < -TIE).sum()), float((d[sel] > TIE).sum()))
    pos = np.sort(d[d > TIE])[::-1]
    w, t, lo = int((d > TIE).sum()), int((abs(d) <= TIE).sum()), int((d < -TIE).sum())
    return dict(
        mean_delta=float(d.mean()),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        lcb95=float(np.quantile(boot, 0.025)),
        wtl=[w, t, lo],
        elo=bt_two(float(lo), float(w)),
        elo_ci95=np.quantile(elo_boot, [0.025, 0.975]).tolist(),
        top4_share=float(pos[:TOP_K].sum() / pos.sum()) if len(pos) else None,
        changed_contexts=None if selected is None else int(sum(a != b for a, b in zip(selected, base_selected))),
        _boot=boot,
    )


def seed_criteria(mean, control_mean, st):
    return dict(
        above_stored_plus_margin=bool(mean > STORED + MARGIN),
        above_paired_control=bool(mean > control_mean),
        lcb95_positive=bool(st["lcb95"] > 0),
        not_concentrated=bool(st["top4_share"] is not None and st["top4_share"] < CONC_MAX),
    )


def evaluate(results, base, families, base_selected):
    by = {r["name"]: r for r in results}
    seeds = sorted({int(n.split("-")[1]) for n in by if n.startswith("SWA_CYCLE-")})
    out, per_seed_pass, boots, lifts = {}, [], [], []
    for name, r in by.items():
        st = candidate_stats(r["per_context"], base, families, r["selected_programs"], base_selected)
        out[name] = dict(
            mean=r["mean"], lift=r["mean"] - STORED, **{k: v for k, v in st.items() if not k.startswith("_")}
        )
        if name.startswith("SWA_CYCLE-"):
            seed = name.split("-")[1]
            c = seed_criteria(r["mean"], by[f"CONTROL-{seed}"]["mean"], st)
            out[name]["criteria"] = c
            out[name]["passes"] = all(c.values())
            per_seed_pass.append(out[name]["passes"])
            boots.append(st["_boot"])
            lifts.append(r["mean"] - STORED)
    primary = dict(seeds=seeds, per_seed_pass=per_seed_pass)
    if len(boots) == 3:
        primary.update(
            median_of_seeds_lift=float(np.median(lifts)),
            median_of_seeds_ci95=np.quantile(np.median(np.stack(boots), axis=0), [0.025, 0.975]).tolist(),
        )
    verdict = "DEV_ADVANCE" if len(per_seed_pass) == 3 and all(per_seed_pass) else "NOT_ADVANCED"
    return dict(verdict=verdict, primary=primary, candidates=out)
