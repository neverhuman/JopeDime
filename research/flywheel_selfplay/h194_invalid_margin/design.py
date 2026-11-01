"""H194 independent folds, single IMH candidate and paired-family reporting."""

import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def sibling(name, relative):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


h177 = sibling(
    "h194_h177", "research/flywheel_selfplay/h177_pooled_selector/analysis.py"
)
H177_DECISION_RULE = h177.H177_DECISION_RULE
StageOrderGuard = h177.StageOrderGuard
SEEDS = (194101, 194102, 194103)
FOLD_SEEDS = (194141, 194142, 194143)
BOOT_SEED = 194150
ARMS = ("CTRL", "IMH", "IMH_SHUF")
ALL_ARMS = ("R882",) + ARMS
DRAWS = 10000
EPS = 1e-12


def folds(rows, seed):
    families = sorted({r["family_id"] for r in rows})
    if len(families) != 205:
        raise ValueError("205 TRAIN families required")
    return [
        [families[j] for j in g]
        for g in np.array_split(
            np.random.default_rng(FOLD_SEEDS[SEEDS.index(seed)]).permutation(205), 5
        )
    ]


def partition(rows, outer, seed):
    g = folds(rows, seed)
    a, b = set(g[outer]), set(g[(outer + 1) % 5])
    return (
        [r for r in rows if r["family_id"] not in a | b],
        [r for r in rows if r["family_id"] in b],
        [r for r in rows if r["family_id"] in a],
    )


def require_fit(cids, fit_rows):
    if not set(cids) <= {r["context_sha256"] for r in fit_rows}:
        raise ValueError("FIT-only input refusal")


def select(rows, base, candidates):
    if set(candidates) != set(ARMS):
        raise ValueError("exact H194 arms required")
    native = H177_DECISION_RULE.select(
        rows,
        base,
        dict(
            CTRL=candidates["CTRL"],
            ACC4=candidates["IMH"],
            ACC16=candidates["IMH_SHUF"],
            OC15=base,
        ),
        seed=BOOT_SEED,
    )
    r = native["ACC4"]
    admitted = bool(r["unadjusted_admitted"] and r["p_one_sided"] < 0.05)
    return dict(
        arm="IMH",
        candidate_step=40,
        selected_step=40 if admitted else 0,
        mean_gain=r["mean_gain"],
        ci95=r["ci95"],
        median_seed_gain=r["median_seed_gain"],
        median_seed_ci95=r["median_seed_ci95"],
        per_seed_gain=r["per_seed_gain"],
        positive_seeds=r["positive_seeds"],
        holm_multiplicity=1,
        holm_p=r["p_one_sided"],
        admitted=admitted,
        scope="Owner-prescribed single IMH candidate: unadjusted H177 mean-functional test is Holm m1; native m4 decisions diagnostic only.",
    ), native


def summarize(rows, panel):
    panel = np.asarray(panel, dtype=float)
    if panel.shape != (3, len(rows), 4) or not np.isfinite(panel).all():
        raise ValueError("complete 3-seed four-policy panel")
    labels = [r["family_id"] for r in rows]
    fm = np.stack([h177.family_matrix(v, labels)[1] for v in panel])
    n = fm.shape[1]
    ix = np.random.default_rng(BOOT_SEED).integers(0, n, (DRAWS, n))
    w = np.stack([(ix == j).sum(axis=1) for j in range(n)], axis=1) / n

    def summary(v):
        means = v.mean(axis=1)
        boot = v[:, ix].mean(axis=2)
        med = np.median(boot, axis=0)
        avg = v.mean(axis=0)[ix].mean(axis=1)
        return dict(
            per_seed=means.tolist(),
            median=float(np.median(means)),
            ci95=np.quantile(med, [0.025, 0.975]).tolist(),
            seed_average=float(means.mean()),
            seed_average_ci95=np.quantile(avg, [0.025, 0.975]).tolist(),
            p_one_sided=float((1 + (avg <= 0).sum()) / (DRAWS + 1)),
            positive_seeds=int((means > 0).sum()),
        )

    utilities = {a: summary(fm[:, :, j]) for j, a in enumerate(ALL_ARMS)}
    contrasts = {
        a + "_minus_R882": summary(fm[:, :, j] - fm[:, :, 0])
        for j, a in enumerate(ARMS, 1)
    }
    for j, arm in [(2, ARMS[1]), (3, ARMS[2])]:
        contrasts[arm + "_minus_CTRL"] = summary(fm[:, :, j] - fm[:, :, 1])
    delta = fm[:, :, 2] - fm[:, :, 0]
    median_boot = np.median(delta[:, ix].mean(axis=2), axis=0)
    contrasts["IMH_minus_R882"]["holm_m1_p_median"] = float(
        (1 + (median_boot <= 0).sum()) / (DRAWS + 1)
    )
    fw = np.stack([h177.family_wins(v, labels) for v in panel])
    elo = h177.bt_ratings(fw.sum(axis=1))
    elo_boot = []
    # 10k paired family draws for Elo too; chunk avoids large temporary arrays.
    for start in range(0, DRAWS, 500):
        wins = np.einsum("qf,sfij->sqij", w[start : start + 500] * n, fw)
        elo_boot.append(np.median(h177.bt_ratings(wins), axis=0))
    eb = np.concatenate(elo_boot)
    return dict(
        contexts=len(rows),
        families=n,
        utility=utilities,
        contrasts=contrasts,
        Elo={
            a: dict(
                per_seed=elo[:, j].tolist(),
                median=float(np.median(elo[:, j])),
                lift_vs_R882=float(np.median(elo[:, j]) - 1500),
                lift_ci95=(np.quantile(eb[:, j], [0.025, 0.975]) - 1500).tolist(),
                ci95=np.quantile(eb[:, j], [0.025, 0.975]).tolist(),
            )
            for j, a in enumerate(ALL_ARMS)
        },
        minus2_count_per_seed={
            a: (panel[:, :, j] == -2).sum(axis=1).tolist()
            for j, a in enumerate(ALL_ARMS)
        },
        bootstrap_draws=DRAWS,
        seed_reduction="axis=0, paired shared family resamples; Elo context wins then equal-family MM prior.01/400iterations",
    )


def verdict(decision, A):
    c = A["contrasts"]["IMH_minus_R882"]
    advance = bool(
        decision["admitted"]
        and c["median"] > 0.001 + EPS
        and c["ci95"][0] > 0
        and c["holm_m1_p_median"] < 0.05
    )
    return dict(
        verdict="ADVANCE" if advance else "NO_ADVANCE",
        advance_candidate=advance,
        serving_promotion=False,
        primary_arm="IMH",
        holm_multiplicity=1,
    )
