import numpy as np
from dependencies import clusters

ARMS = ("BASE", "REF", "A", "B", "PLACEBO")
PARTITIONS = ("KM276140", "KM276141", "KM276142", "WARD")
SEEDS = (277101, 277102, 277103)
BOOT = 277150
EPS = 1e-12


def per_family(rows, values):
    names = np.array([r["family_id"] for r in rows])
    families = sorted(set(names))
    return families, np.stack(
        [values[:, names == f].mean(axis=1) for f in families], axis=1
    )


def contrast(delta, boot):
    gains = delta.mean(axis=1)
    med = np.median(boot, axis=0)
    return dict(
        per_seed_family_gains=gains.tolist(),
        median_family_gain=float(np.median(gains)),
        positive_seeds=int((gains > EPS).sum()),
        ci95=np.quantile(med, [0.025, 0.975]).tolist(),
        ci975_holm2=np.quantile(med, [0.0125, 0.9875]).tolist(),
        raw_p_one_sided=float((1 + (med <= EPS).sum()) / (len(med) + 1)),
        harm_families_gt_1e4_by_seed=(delta < -1e-4).sum(axis=1).tolist(),
        family_wtl_by_seed=[
            dict(
                wins=int((d > 1e-4).sum()),
                ties=int((abs(d) <= 1e-4).sum()),
                losses=int((d < -1e-4).sum()),
            )
            for d in delta
        ],
    )


def summarize(rows, panel, vrows, vpanel, parts):
    primary = {}
    raw = {}
    for pi, name in enumerate(PARTITIONS):
        families, fm = per_family(rows, panel[:, pi])
        loc = {f: i for i, f in enumerate(families)}
        groups = [[loc[f] for f in g] for g in parts[name]]
        sizes = np.array([len(g) for g in groups])
        draws = np.random.default_rng(BOOT).integers(0, 5, (10000, 5))
        den = sizes[draws].sum(axis=1)
        result = {}
        raw[name] = {
            a: int((panel[:, pi, :, ai] == -2).sum()) for ai, a in enumerate(ARMS)
        }
        for ai, arm in enumerate(ARMS[1:], 1):
            d = fm[:, :, ai] - fm[:, :, 0]
            sums = np.stack([d[:, g].sum(axis=1) for g in groups], axis=1)
            boot = sums[:, draws].sum(axis=2) / den
            result[arm] = contrast(d, boot)
            result[arm]["per_cluster_gains_by_seed"] = np.stack(
                [d[:, g].mean(axis=1) for g in groups], axis=1
            ).tolist()
        previous = 0.0
        for rank, arm in enumerate(
            sorted(("A", "B"), key=lambda a: result[a]["raw_p_one_sided"])
        ):
            previous = max(
                previous, min(1.0, (2 - rank) * result[arm]["raw_p_one_sided"])
            )
            result[arm]["holm_p_one_sided"] = previous
            result[arm]["holm_lcb95"] = result[arm]["ci975_holm2"][0]
        result["elo"] = {
            a: float(v)
            for a, v in zip(
                ARMS, clusters.elo(panel[:, pi], [r["family_id"] for r in rows])
            )
        }
        primary[name] = result
    families, fm = per_family(vrows, vpanel.mean(axis=1))
    draws = np.random.default_rng(BOOT).integers(
        0, len(families), (10000, len(families))
    )
    secondary = {}
    for ai, arm in enumerate(ARMS[1:], 1):
        d = fm[:, :, ai] - fm[:, :, 0]
        secondary[arm] = contrast(d, d[:, draws].mean(axis=2))
    secondary["elo"] = {
        a: float(v)
        for a, v in zip(
            ARMS, clusters.elo(vpanel.mean(axis=1), [r["family_id"] for r in vrows])
        )
    }
    secondary["raw_minus2_picks"] = {
        a: int((vpanel[:, :, :, ai] == -2).sum()) for ai, a in enumerate(ARMS)
    }
    return primary, secondary, raw


def decide(primary, secondary, raw):
    basic = {}
    for arm in ("A", "B", "PLACEBO"):
        basic[arm] = dict(
            three_positive_every_partition=all(
                primary[n][arm]["positive_seeds"] == 3 for n in PARTITIONS
            ),
            adjusted_cluster_lcb_every_partition=all(
                primary[n][arm]["ci975_holm2"][0] > EPS for n in PARTITIONS
            ),
            raw_invalid_not_worse=all(
                raw[n][arm] <= raw[n]["BASE"] for n in PARTITIONS
            ),
            harmed_not_above_REF=all(
                np.all(
                    np.array(primary[n][arm]["harm_families_gt_1e4_by_seed"])
                    <= np.array(primary[n]["REF"]["harm_families_gt_1e4_by_seed"])
                )
                for n in PARTITIONS
            ),
            secondary_three_positive=secondary[arm]["positive_seeds"] == 3,
        )
    placebo_pass = all(basic["PLACEBO"].values())
    decisions = {
        arm: dict(
            gates={**basic[arm], "placebo_fails": not placebo_pass},
            advance=all(basic[arm].values()) and not placebo_pass,
        )
        for arm in ("A", "B")
    }
    return dict(
        arms=decisions,
        placebo_would_pass=placebo_pass,
        placebo_gates=basic["PLACEBO"],
        verdict="ADVANCE"
        if any(d["advance"] for d in decisions.values())
        else "NO_ADVANCE",
    )
