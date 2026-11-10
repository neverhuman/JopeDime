"""H282 statistics (hw_t2235): the frozen H278 RERANK8_VETO pick against each checkpoint's own slot-0 pick on that
checkpoint's DL-Val91 top-8 pool. Family-equal gains, family bootstrap (10,000 draws, seed 282150), W/T/L with ties NOT
wins, two-contender MM Bradley-Terry Elo (slot 0 = 1500; H278's bt_two), invalid picks and harmed families.

Primary rule (all must hold on every one of the 3 seeds):
  (a) pooled bootstrap over (checkpoint x family) units of the family-mean gain has LCB95 > 0;
  (b) gain > 0 on at least ceil(2/3 x E) of the E regression-passing checkpoints;
  (c) on no checkpoint does FROZEN_VETO make more invalid (-2.0) picks than slot 0.
TRANSFER_PASS if all hold, else NOT_TRANSFERRED; fewer than 4 regression-passing checkpoints -> INSUFFICIENT_CHECKPOINTS."""

import math

import numpy as np

BOOT_SEED = 282150
DRAWS = 10000
HARM = 1e-4
PLATEAU = 1.1389572403748425
WATCH = ("fam-280b1a13", "fam-469b72b7", "fam-769243ee")
MIN_CHECKPOINTS = 4


def family_gains(u, families):
    """u: (contexts, 2) = [slot0, frozen] for one checkpoint and seed -> (sorted family ids, family-mean gain)."""
    fams = sorted(set(families))
    f = np.asarray(families)
    d = np.array([u[f == x, 1].mean() - u[f == x, 0].mean() for x in fams])
    return fams, d


def bt_two(wins_ab, wins_ba, prior=0.01, iters=400):
    """H278's two-contender MM Bradley-Terry (incumbent anchored at 1500); returns the challenger's Elo."""
    s_a, s_b = 1.0, 1.0
    games = wins_ab + wins_ba
    for _ in range(iters):
        s_b = (wins_ba + prior) / ((games + 2 * prior) / (s_a + s_b))
        s_b = float(np.clip(s_b / s_a, 1e-6, 1e6))
        s_a = 1.0
    return 1500 + 400 * np.log10(s_b)


def boot_ci(d, seed=BOOT_SEED, draws=DRAWS):
    ix = np.random.default_rng(seed).integers(0, len(d), (draws, len(d)))
    return np.quantile(d[ix].mean(axis=1), [0.025, 0.975]).tolist()


def cell(u, inv, families, picks):
    """One checkpoint x seed."""
    fams, d = family_gains(u, families)
    return dict(
        family_mean_gain=float(d.mean()),
        family_median_gain=float(np.median(d)),
        ci95=boot_ci(d),
        wtl=[
            int((d > HARM).sum()),
            int((abs(d) <= HARM).sum()),
            int((d < -HARM).sum()),
        ],
        elo=float(bt_two(float((d < -HARM).sum()), float((d > HARM).sum()))),
        invalid=[int(inv[:, 0].sum()), int(inv[:, 1].sum())],
        changed_picks=int(sum(j != 0 for j in picks)),
        context_mean=[float(u[:, 0].mean()), float(u[:, 1].mean())],
        harmed_families=[f for f, x in zip(fams, d) if x < -HARM],
        _d=d,
    )


def checkpoint_summary(cells):
    """Across the 3 seeds: median gain and an H278-style pooled CI (median over seeds of common-draw bootstrap means)."""
    D = np.stack([c["_d"] for c in cells])  # (3, F)
    ix = np.random.default_rng(BOOT_SEED).integers(0, D.shape[1], (DRAWS, D.shape[1]))
    boot = np.median(D[:, ix].mean(axis=2), axis=0)
    return dict(
        median_gain=float(np.median(D.mean(axis=1))),
        pooled_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        watch_harmed=sorted(
            {f for c in cells for f in c["harmed_families"] if f.startswith(WATCH)}
        ),
    )


def primary(grid):
    """grid[ckpt][seed_index] = cell. Returns per-seed criteria and the verdict."""
    names = sorted(grid)
    E = len(names)
    if E < MIN_CHECKPOINTS:
        return dict(E=E, verdict="INSUFFICIENT_CHECKPOINTS")
    need = math.ceil(2 * E / 3)
    per_seed = []
    for s in range(3):
        units = np.concatenate([grid[k][s]["_d"] for k in names])
        lcb, ucb = boot_ci(units)
        pos = sum(grid[k][s]["family_mean_gain"] > 0 for k in names)
        inv_ok = all(
            grid[k][s]["invalid"][1] <= grid[k][s]["invalid"][0] for k in names
        )
        per_seed.append(
            dict(
                pooled_unit_mean_gain=float(units.mean()),
                pooled_ci95=[lcb, ucb],
                lcb_positive=bool(lcb > 0),
                positive_checkpoints=int(pos),
                needed_positive=need,
                count_ok=bool(pos >= need),
                invalid_ok=bool(inv_ok),
            )
        )
    ok = all(p["lcb_positive"] and p["count_ok"] and p["invalid_ok"] for p in per_seed)
    return dict(
        E=E, per_seed=per_seed, verdict="TRANSFER_PASS" if ok else "NOT_TRANSFERRED"
    )


def public(cellrow):
    return {k: v for k, v in cellrow.items() if not k.startswith("_")}
