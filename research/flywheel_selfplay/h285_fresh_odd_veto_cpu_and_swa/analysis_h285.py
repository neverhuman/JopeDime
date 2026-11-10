"""Frozen fresh-family comparison; each frozen model has its own paired family-bootstrap gate."""

import hashlib
import numpy as np
from dependencies import clusters

SEEDS = (278101, 278102, 278103)
NULL_SEEDS = (285101, 285102, 285103)
BOOT = 285150
ARMS = ("BASE", "A", "RANDOM_VALID", "SWA")
TIE = 1e-4
HARM = 0.01
EPS = 1e-12


def random_valid_picks(cids, gates, seed):
    picks = []
    for c, g in zip(cids, gates):
        valid = np.flatnonzero(g)
        ss = int.from_bytes(
            hashlib.sha256(f"{seed}:{c}".encode()).digest()[:8], "little"
        )
        picks.append(int(np.random.default_rng(ss).choice(valid)) if len(valid) else 0)
    return np.asarray(picks, int)


def family_values(values, families):
    ids = sorted(set(families))
    f = np.asarray(families)
    return ids, np.stack([values[:, f == name].mean(axis=1) for name in ids], axis=1)


def contrast(delta, draws):
    gains = delta.mean(axis=1)
    boot = delta[:, draws].mean(axis=2)
    med = np.median(boot, axis=0)
    return dict(
        per_seed_gain=gains.tolist(),
        median_gain=float(np.median(gains)),
        positive_seeds=int((gains > TIE).sum()),
        per_seed_ci95=np.quantile(boot, [0.025, 0.975], axis=1).T.tolist(),
        median_ci95=np.quantile(med, [0.025, 0.975]).tolist(),
        harm_families_loss_gt_0_01=(delta < -HARM).sum(axis=1).tolist(),
        wtl_by_seed=[
            dict(
                wins=int((d > TIE).sum()),
                ties=int((abs(d) <= TIE).sum()),
                losses=int((d < -TIE).sum()),
            )
            for d in delta
        ],
    )


def elo_per_seed(values, families):
    diff = values[:, :, :, None] - values[:, :, None, :]
    outcomes = np.where(diff > TIE, 1.0, np.where(diff < -TIE, 0.0, 0.5)) * (
        1 - np.eye(values.shape[2])
    )
    fw = np.stack([clusters.family_matrix(o, families)[1] for o in outcomes])
    return clusters.bt_ratings(fw.sum(axis=1))


def summarize(values, invalid, picks, families):
    u = np.asarray(values, float)
    inv = np.asarray(invalid, bool)
    assert u.shape == inv.shape and u.shape[0] == 3 and u.shape[2] == 4
    ids, fm = family_values(u, families)
    draws = np.random.default_rng(BOOT).integers(0, len(ids), (10000, len(ids)))
    stats = {}
    for i, a in enumerate(ARMS[1:], 1):
        stats[a] = contrast(fm[:, :, i] - fm[:, :, 0], draws)
        stats[a]["invalid_by_seed"] = inv[:, :, i].sum(axis=1).tolist()
        stats[a]["changed_picks_by_seed"] = (
            (np.asarray(picks)[:, :, i] != np.asarray(picks)[:, :, 0])
            .sum(axis=1)
            .tolist()
        )
    stats["SWA"]["single_run"] = True
    stats["SWA"]["family_bootstrap_ci95"] = stats["SWA"]["per_seed_ci95"][0]
    stats["SWA"]["gain"] = stats["SWA"]["per_seed_gain"][0]
    ratings = elo_per_seed(u, families)
    return dict(
        per_seed_family_mean_utility=fm.mean(axis=1).tolist(),
        elo_by_seed=ratings.tolist(),
        families=len(ids),
        contexts=len(families),
        columns=list(ARMS),
        contrasts=stats,
        base_invalid_by_seed=inv[:, :, 0].sum(axis=1).tolist(),
        elo={a: float(v) for a, v in zip(ARMS, np.median(ratings, axis=0))},
        family_ids=ids,
    )


def gates(summary, arm):
    s = summary["contrasts"][arm]
    return dict(
        all_three_lcb95_positive=bool(
            np.all(np.asarray(s["per_seed_ci95"])[:, 0] > EPS)
        ),
        median_gt_0_010=s["median_gain"] > 0.010 + EPS,
        harmed_at_most_35_percent=bool(
            np.all(
                np.asarray(s["harm_families_loss_gt_0_01"])
                <= 0.35 * summary["families"]
            )
        ),
        invalid_not_above_base=bool(
            np.all(np.asarray(s["invalid_by_seed"]) <= summary["base_invalid_by_seed"])
        ),
    )


def decide(summary):
    a = gates(summary, "A")
    null = gates(summary, "RANDOM_VALID")
    null_pass = all(null.values())
    a["random_null_fails"] = not null_pass
    enough = summary["families"] >= 25
    swa = summary["contrasts"]["SWA"]
    bgates = dict(
        gain_positive=swa["gain"] > TIE,
        family_lcb95_positive=swa["family_bootstrap_ci95"][0] > EPS,
        invalid_not_above_base=swa["invalid_by_seed"][0]
        <= summary["base_invalid_by_seed"][0],
        harmed_at_most_35_percent=swa["harm_families_loss_gt_0_01"][0]
        <= 0.35 * summary["families"],
    )
    return dict(
        verdict="INSUFFICIENT_FRESH"
        if not enough
        else ("ADVANCE" if all(a.values()) else "NOT_TRANSFERRED_FRESH"),
        A_gates=a,
        random_null_gates=null,
        random_null_would_pass=null_pass,
        B_verdict="PROMISING" if enough and all(bgates.values()) else "NOT",
        B_gates=bgates,
        B_qualification="single deterministic exploratory run; cannot ADVANCE",
    )
