"""H279 leave-family-CLUSTER-out test of bagged strict-margin switching (BAG_SWITCH q=6 and BAG_SWITCH_NESTED q in {5,6,7})
against r882 (BASE) and RERANK8_VETO (REF): outcome-blind family clusterings from H276 (k-means k = 5 seeds 276140-276142
and Ward k = 5), cluster-out folds with family-grouped inner CV, two-level cluster bootstrap (10k draws, seed 279150),
Holm m = 2 over BAG_SWITCH and BAG_SWITCH_NESTED, and the preregistered ADVANCE gates (hw_t2140)."""

import hashlib
import numpy as np

SEEDS = (279101, 279102, 279103)
KMEANS_SEEDS = (276140, 276141, 276142)
PARTITIONS = ("KM276140", "KM276141", "KM276142", "WARD")
SECONDARY_PARTITION = "KM276140"
BOOT_SEED = 279150
N_CLUSTERS = 5
ARMS = ("RERANK8_VETO", "BAG_SWITCH", "BAG_SWITCH_NESTED", "PLACEBO")
GATED = ("BAG_SWITCH", "BAG_SWITCH_NESTED")
REFERENCES = {
    "RERANK8_VETO": ("r882",),
    "BAG_SWITCH": ("r882", "RERANK8_VETO"),
    "BAG_SWITCH_NESTED": ("r882", "RERANK8_VETO", "BAG_SWITCH"),
    "PLACEBO": ("r882",),
}
ALL_ARMS = ("r882",) + ARMS
DRAWS, EPS, HARM, HARM_GATE = 10000, 1e-12, 1e-4, 0.01
PASS_MARGIN = 0.010
PLACEBO_MAX = 0.005
HARM_RATIO_MAX = 0.75
GAIN_RATIO_MIN = 0.80
VETO_TAU = 0.10
WATCH_FAMILY_PREFIX = "fam-769243ee"


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def family_descriptors(rows, ctx_desc, pool_desc):
    """Outcome-blind family descriptor: the family mean of [context descriptor, pool descriptor], each standardised
    over TRAIN contexts. No fitness, gate or utility enters."""
    cids = [r["context_sha256"] for r in rows]
    X = np.stack(
        [
            np.concatenate(
                [
                    np.asarray(ctx_desc[c], dtype=np.float64),
                    np.asarray(pool_desc[c], dtype=np.float64),
                ]
            )
            for c in cids
        ]
    )
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)
    fams = sorted({r["family_id"] for r in rows})
    fam_of = np.array([r["family_id"] for r in rows])
    return fams, np.stack([X[fam_of == f].mean(axis=0) for f in fams])


def clusterings(fams, F):
    """Pre-declared partitions: k-means (k = 5, n_init = 10) with seeds 276140-276142, and Ward agglomerative (k = 5).
    Cluster ids are relabelled by ascending smallest member family id so partitions are canonical."""
    from sklearn.cluster import AgglomerativeClustering, KMeans

    out = {}
    for s in KMEANS_SEEDS:
        out[f"KM{s}"] = KMeans(
            n_clusters=N_CLUSTERS, n_init=10, random_state=s
        ).fit_predict(F)
    out["WARD"] = AgglomerativeClustering(
        n_clusters=N_CLUSTERS, linkage="ward"
    ).fit_predict(F)
    canon = {}
    for name, labels in out.items():
        groups = {}
        for f, lab in zip(fams, labels):
            groups.setdefault(int(lab), []).append(f)
        ordered = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
        canon[name] = ordered
    return canon


def cluster_out_folds(rows, clusters, held, seed):
    """Fold 0 = held-out cluster; folds 1..4 = the remaining families split into 4 random family groups (stream per
    model seed, partition-independent), for the unchanged H265 inner selector (family-grouped inner CV)."""
    rest = sorted(f for i, g in enumerate(clusters) if i != held for f in g)
    order = np.random.default_rng(
        stream_seed(seed, "inner", held, len(rest), rest[0])
    ).permutation(len(rest))
    groups = [sorted(clusters[held])] + [
        sorted(rest[i] for i in g) for g in np.array_split(order, 4)
    ]
    where = {f: k for k, g in enumerate(groups) for f in g}
    out = {k: [] for k in range(5)}
    for r in rows:
        out[where[r["family_id"]]].append(r["context_sha256"])
    return out


def family_matrix(values, families):
    values = np.asarray(values, dtype=np.float64)
    fams = sorted(set(families))
    return fams, np.array(
        [values[np.asarray(families) == f].mean(axis=0) for f in fams]
    )


def bt_ratings(wins, anchor=0, prior=0.01):
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    strengths = np.ones(wins.shape[:-1])
    for _ in range(400):
        denom = (
            (games + 2 * prior * (1 - np.eye(n)))
            / np.maximum(strengths[..., :, None] + strengths[..., None, :], 1e-12)
        ).sum(axis=-1)
        strengths = np.clip(totals / denom, 1e-6, 1e6)
        strengths /= np.maximum(strengths[..., anchor, None], 1e-12)
    return 1500 + 400 * np.log10(np.maximum(strengths, 1e-9))


def elo(values, families):
    """MM Bradley-Terry Elo (r882 = 1500) from per-family pairwise wins, summed over seeds' family win matrices."""
    diff = values[:, :, :, None] - values[:, :, None, :]  # (seeds, contexts, P, P)
    outcomes = np.where(diff > HARM, 1.0, np.where(diff < -HARM, 0.0, 0.5)) * (
        1 - np.eye(values.shape[2])
    )
    fw = np.stack([family_matrix(o, families)[1] for o in outcomes])  # (seeds, F, P, P)
    return np.median(bt_ratings(fw.sum(axis=1)), axis=0).tolist()


def holm(pvals):
    result, previous = {}, 0.0
    for rank, key in enumerate(sorted(pvals, key=pvals.get)):
        previous = max(previous, min(1.0, pvals[key] * (len(pvals) - rank)))
        result[key] = previous
    return result


def summarize(rows, panel, partitions, seed=BOOT_SEED):
    """panel: (3 seeds, P partitions, contexts, policies) of OOF single-pick utilities (each context held out once per
    partition). Per seed the gain is the MEDIAN over partitions of the equal-family gain. The CI is a two-level
    cluster bootstrap: each draw resamples, within every partition independently, its 5 clusters with replacement
    (families weighted by the number of times their cluster is drawn). Also per-cluster gains and harmed counts."""
    values = np.asarray(panel, dtype=np.float64)
    families = [r["family_id"] for r in rows]
    fam_ids = sorted(set(families))
    fidx = {f: i for i, f in enumerate(fam_ids)}
    fm = np.stack(
        [np.stack([family_matrix(v, families)[1] for v in rep]) for rep in values]
    )  # (3, P, F, A)
    rng = np.random.default_rng(seed)
    cluster_members = [
        [np.array([fidx[f] for f in g]) for g in partitions[name]]
        for name in partitions
    ]
    weights = np.zeros((DRAWS, len(partitions), len(fam_ids)))
    for pi, members in enumerate(cluster_members):
        draws = rng.integers(0, len(members), (DRAWS, len(members)))
        for ci, m in enumerate(members):
            weights[:, pi, m] += (draws == ci).sum(axis=1)[:, None]
    weights /= weights.sum(axis=2, keepdims=True)
    contrasts = {}
    for col, arm in enumerate(ARMS, 1):
        for label in REFERENCES[arm]:
            ref = ALL_ARMS.index(label)
            diff = fm[..., col] - fm[..., ref]  # (3, P, F)
            per_rep = diff.mean(axis=2)
            per_seed = np.median(per_rep, axis=1)
            boot = np.median(
                np.median(np.einsum("spf,dpf->dsp", diff, weights), axis=2), axis=1
            )
            per_cluster = {
                name: [
                    [float(diff[s, pi, m].mean()) for m in cluster_members[pi]]
                    for s in range(3)
                ]
                for pi, name in enumerate(partitions)
            }
            mean_diff = diff.mean(axis=1)  # (3, F)
            harmed_001_by_seed = [int((d < -HARM_GATE - EPS).sum()) for d in mean_diff]
            contrasts[arm + "_minus_" + label] = dict(
                per_seed_gain=per_seed.tolist(),
                per_seed_per_partition_gain=per_rep.tolist(),
                median_gain=float(np.median(per_seed)),
                ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                ci_holm_m2=np.quantile(boot, [0.0125, 0.9875]).tolist(),
                p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
                positive_seeds=int((per_seed > 0).sum()),
                per_cluster_gain_by_seed=per_cluster,
                wtl_by_seed=[
                    [
                        int((d > HARM).sum()),
                        int((abs(d) <= HARM).sum()),
                        int((d < -HARM).sum()),
                    ]
                    for d in mean_diff
                ],
                harmed_001_by_seed=harmed_001_by_seed,
                harmed_001_median=float(np.median(harmed_001_by_seed)),
                harmed_001_families_by_seed=[
                    [f for f, d in zip(fam_ids, row) if d < -HARM_GATE - EPS]
                    for row in mean_diff
                ],
                harm_families_by_seed=[
                    [f for f, d in zip(fam_ids, row) if d < -HARM] for row in mean_diff
                ],
            )
    pooled = values.mean(axis=1)
    return dict(
        columns=list(ALL_ARMS),
        contexts=len(rows),
        families=len(fam_ids),
        partitions=list(partitions),
        contrasts=contrasts,
        family_utility_by_seed=fm.mean(axis=(1, 2)).tolist(),
        median_elo=elo(pooled[:, :, :], families),
    )


def clusters_nonnegative(contrast, partitions):
    """Per partition: the number of clusters whose gain is >= 0 in at least 2 of 3 seeds."""
    out = {}
    for name in partitions:
        per = np.array(contrast["per_cluster_gain_by_seed"][name])  # (3, 5)
        out[name] = int(((per >= -EPS).sum(axis=0) >= 2).sum())
    return out


def decide(summary, invalid, partitions, dlval_summary=None):
    """ADVANCE per gated arm (BAG_SWITCH, BAG_SWITCH_NESTED; Holm m = 2) iff all 8 strict gates hold:
    1) median per-family gain vs r882 > +0.010 AND > 0 on each of 3 seeds;
    2) Holm-level LCB > 0 (and Holm-adjusted one-sided p < 0.025);
    3) harmed families (loss > 0.01 vs r882) <= 0.75 * REF's harmed count, median over seeds;
    4) median gain >= 0.80 * REF's median gain;
    5) invalid -2.0 picks <= REF on DL-Train;
    6) >= 4/5 clusters non-negative (>= 2/3 seeds) in every partition;
    7) PLACEBO median <= +0.005 (and PLACEBO does not pass);
    8) DL-Val91 median gain vs r882 > 0."""
    c = summary["contrasts"]
    adj = holm({a: c[a + "_minus_r882"]["p_one_sided"] for a in GATED})
    ref_x = c["RERANK8_VETO_minus_r882"]
    placebo_x = c["PLACEBO_minus_r882"]
    placebo = float(placebo_x["median_gain"])
    placebo_would_pass = bool(
        placebo > PASS_MARGIN + EPS
        and placebo_x["positive_seeds"] == 3
        and placebo_x["ci_holm_m2"][0] > 0
    )
    arms = {}
    for a in GATED:
        x = c[a + "_minus_r882"]
        nonneg = clusters_nonnegative(x, partitions)
        if dlval_summary is not None:
            dlval_gain = float(
                dlval_summary["contrasts"][a + "_minus_r882"]["median_gain"]
            )
        else:
            dlval_gain = float(x["median_gain"])
        gates = dict(
            median_gt_0_010=bool(x["median_gain"] > PASS_MARGIN + EPS),
            three_seeds=bool(x["positive_seeds"] == 3),
            holm_lcb_gt_0=bool(x["ci_holm_m2"][0] > 0 and adj[a] < 0.025),
            harmed_le_075_ref=bool(
                x["harmed_001_median"]
                <= HARM_RATIO_MAX * ref_x["harmed_001_median"] + EPS
            ),
            gain_ge_080_ref=bool(
                x["median_gain"] >= GAIN_RATIO_MIN * ref_x["median_gain"] - EPS
            ),
            invalid_le_ref=bool(invalid[a] <= invalid["RERANK8_VETO"]),
            clusters_4_of_5_every_partition=bool(all(v >= 4 for v in nonneg.values())),
            placebo_le_0_005=bool(
                placebo <= PLACEBO_MAX + EPS and not placebo_would_pass
            ),
            dlval_gain_gt_0=bool(dlval_gain > EPS),
        )
        arms[a] = dict(
            gates=gates,
            clusters_nonnegative=nonneg,
            holm_p=adj[a],
            harmed_001_median=x["harmed_001_median"],
            ref_harmed_001_median=ref_x["harmed_001_median"],
            harmed_ratio_vs_ref=float(
                x["harmed_001_median"] / max(ref_x["harmed_001_median"], 1.0)
            ),
            gain_ratio_vs_ref=float(
                x["median_gain"] / max(ref_x["median_gain"], 1e-12)
            ),
            dlval_median_gain=dlval_gain,
            passed=bool(all(gates.values())),
        )
    return dict(
        arms=arms,
        placebo_median=placebo,
        placebo_would_pass=placebo_would_pass,
        verdict="ADVANCE" if any(arms[a]["passed"] for a in GATED) else "NO_ADVANCE",
    )


def partitions_from_manifest(fm):
    """Rebuild the canonical partitions from the compact manifest (sorted family ids + one cluster label per family)."""
    fams = fm["families"]
    out = {}
    for name, labels in fm["labels"].items():
        groups = {}
        for f, lab in zip(fams, labels):
            groups.setdefault(int(lab), []).append(f)
        out[name] = [sorted(groups[k]) for k in sorted(groups)]
    return out
