"""Per-partition family-equal contrasts and H276 cluster resampling; Holm m=1."""

import numpy as np
from dependencies import clusters

ARMS = ("BASE", "REF_TRANSFER", "A", "PLACEBO")
PARTITIONS = ("KM276140", "KM276141", "KM276142", "WARD")
SEEDS = (280101, 280102, 280103)
BOOT = 280150
EPS = 1e-12


def per_family(rows, values):
    names = np.array([r["family_id"] for r in rows])
    families = sorted(set(names))
    return families, np.stack(
        [values[:, names == f].mean(axis=1) for f in families], axis=1
    )


def contrast(delta, boot, families):
    gains = delta.mean(axis=1)
    med = np.median(boot, axis=0)
    watch = [i for i, f in enumerate(families) if f.startswith("fam-769243ee")]
    return dict(
        per_seed_family_gains=gains.tolist(),
        median_family_gain=float(np.median(gains)),
        positive_seeds=int((gains > EPS).sum()),
        ci95=np.quantile(med, [0.025, 0.975]).tolist(),
        holm_m=1,
        p_one_sided=float((1 + (med <= EPS).sum()) / (len(med) + 1)),
        harm_families_gt_1e4_by_seed=(delta < -1e-4).sum(axis=1).tolist(),
        family_wtl_by_seed=[
            dict(
                wins=int((d > 1e-4).sum()),
                ties=int((abs(d) <= 1e-4).sum()),
                losses=int((d < -1e-4).sum()),
            )
            for d in delta
        ],
        fam_769243ee_gain_by_seed=delta[:, watch[0]].tolist() if watch else None,
    )


def summarize(rows, panel, vrows, vpanel, parts):
    primary = {}
    raw = {}
    rng = np.random.default_rng(BOOT)
    for pi, name in enumerate(PARTITIONS):
        families, fm = per_family(rows, panel[:, pi])
        loc = {f: i for i, f in enumerate(families)}
        groups = [[loc[f] for f in g if f in loc] for g in parts[name]]
        assert all(groups)
        sizes = np.array([len(g) for g in groups])
        draws = rng.integers(0, 5, (10000, 5))
        den = sizes[draws].sum(axis=1)
        result = {"families": len(families)}
        raw[name] = {
            a: int((panel[:, pi, :, ai] == -2).sum()) for ai, a in enumerate(ARMS)
        }
        for ai, arm in enumerate(ARMS[1:], 1):
            d = fm[:, :, ai] - fm[:, :, 0]
            sums = np.stack([d[:, g].sum(axis=1) for g in groups], axis=1)
            boot = sums[:, draws].sum(axis=2) / den
            result[arm] = contrast(d, boot, families)
            result[arm]["per_cluster_gains_by_seed"] = np.stack(
                [d[:, g].mean(axis=1) for g in groups], axis=1
            ).tolist()
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
        secondary[arm] = contrast(d, d[:, draws].mean(axis=2), families)
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
    for arm in ["A", "PLACEBO"]:
        basic[arm] = dict(
            three_positive_every_partition=all(
                primary[n][arm]["positive_seeds"] == 3 for n in PARTITIONS
            ),
            median_gt_0_010_every_partition=all(
                primary[n][arm]["median_family_gain"] > 0.010 + EPS for n in PARTITIONS
            ),
            cluster_lcb_every_partition=all(
                primary[n][arm]["ci95"][0] > EPS for n in PARTITIONS
            ),
            raw_invalid_not_worse=all(
                raw[n][arm] <= raw[n]["BASE"] for n in PARTITIONS
            ),
            harm_at_most_35_percent_every_partition_seed=all(
                max(primary[n][arm]["harm_families_gt_1e4_by_seed"])
                <= 0.35 * primary[n]["families"]
                for n in PARTITIONS
            ),
            secondary_three_positive=secondary[arm]["positive_seeds"] == 3,
        )
    placebo_pass = all(basic["PLACEBO"].values())
    gates = {**basic["A"], "placebo_fails": not placebo_pass}
    return dict(
        gates=gates,
        placebo_would_pass=placebo_pass,
        placebo_gates=basic["PLACEBO"],
        verdict="ADVANCE" if all(gates.values()) else "NO_ADVANCE",
    )
