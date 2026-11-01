"""H187 preregistered folds, imported pooled selector and paired-family reporting."""

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
    "h187_h177", "research/flywheel_selfplay/h177_pooled_selector/analysis.py"
)
H177_DECISION_RULE = h177.H177_DECISION_RULE
StageOrderGuard = h177.StageOrderGuard
SEEDS = (187101, 187102, 187103)
FOLD_SEED, BOOT_SEED = 187140, 187141
ARMS = ("FRESH160", "FRESH160_RNORM", "FRESH160_FBAL")
ALL_ARMS = ("R882",) + ARMS
SLOTS = dict(zip(ARMS, ("ACC4", "ACC16", "OC15"), strict=True))
DRAWS = 10000
EPS = 1e-12


def folds(rows):
    families = sorted({r["family_id"] for r in rows})
    if len(families) != 205:
        raise ValueError("205 TRAIN families required")
    return [
        [families[j] for j in g]
        for g in np.array_split(np.random.default_rng(FOLD_SEED).permutation(205), 5)
    ]


def partition(rows, outer):
    g = folds(rows)
    a, b = set(g[outer]), set(g[(outer + 1) % 5])
    return (
        [r for r in rows if r["family_id"] not in a | b],
        [r for r in rows if r["family_id"] in b],
        [r for r in rows if r["family_id"] in a],
    )


def require_fit(cids, fit_rows):
    if not set(cids) <= {r["context_sha256"] for r in fit_rows}:
        raise ValueError("FIT-only input refusal")


def family_probabilities(rows):
    if len({r["context_sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate context")
    counts = {
        f: sum(r["family_id"] == f for r in rows)
        for f in {r["family_id"] for r in rows}
    }
    w = np.array([1 / counts[r["family_id"]] for r in rows], float)
    return w / w.sum()


def balanced_plan(rows, orders, seed, updates):
    ordered = sorted(rows, key=lambda r: r["context_sha256"])
    cids = [r["context_sha256"] for r in ordered]
    prob = family_probabilities(ordered)
    plan = []
    for step in range(1, updates + 1):
        stream = h177.stream_seed(seed, step)
        rng = np.random.default_rng(stream)
        chosen = [cids[j] for j in rng.choice(len(cids), 8, replace=True, p=prob)]
        indices = []
        for cid in chosen:
            order = np.asarray(orders[cid])
            indices.append(np.r_[order[:4], rng.choice(order[4:], 20, replace=False)])
        require_fit(chosen, rows)
        plan.append((stream, chosen, indices))
    return plan


def execution_plan(seconds):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive timing pilot required")
    # Allow fixed 25% treatment overhead and1200s evaluation/reporting reserve.
    default = 7200 * seconds * 1.25 + 1200
    steps = 160 if default <= 10800 else 80
    projected = 45 * steps * seconds * 1.25 + 1200
    if projected > 10800:
        raise RuntimeError("INCOMPLETE even80updates exceeds3h; no seed/fold reduction")
    return dict(
        updates_per_cell=steps,
        total_optimizer_updates=45 * steps,
        folds=5,
        seeds=3,
        pilot_updates=80,
        fallback=steps == 80,
        projected_seconds=projected,
        safety_factor=1.25,
        reserve_seconds=1200,
    )


def select(rows, base, candidates):
    if set(candidates) != set(ARMS):
        raise ValueError("exact three H187 arms required")
    # Execute H177 unchanged, including its existing three-treatment Holm branch.
    native = H177_DECISION_RULE.select(
        rows,
        base,
        dict(CTRL=base, **{SLOTS[a]: candidates[a] for a in ARMS}),
        seed=BOOT_SEED,
    )
    decisions = {}
    for arm in ARMS:
        d = native[SLOTS[arm]]
        decisions[arm] = dict(
            mean_gain=d["mean_gain"],
            ci95=d["ci95"],
            per_seed_gain=d["per_seed_gain"],
            median_seed_gain=d["median_seed_gain"],
            median_seed_ci95=d["median_seed_ci95"],
            positive_seeds=d["positive_seeds"],
            p_one_sided=d["p_one_sided"],
            holm_p=d["holm_p_treatments3"],
            admitted=d["admitted_treatments3"],
            holm_multiplicity=3,
            imported_H177_slot=SLOTS[arm],
            m4_admitted_diagnostic=d["admitted"],
        )
    return decisions, native


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
        contrasts[arm + "_minus_FRESH160"] = summary(fm[:, :, j] - fm[:, :, 1])
    adjusted = h177.holm({a: contrasts[a + "_minus_R882"]["p_one_sided"] for a in ARMS})
    for a in ARMS:
        contrasts[a + "_minus_R882"]["holm_p"] = adjusted[a]
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


def verdict(decisions, A):
    result = {}
    for arm in ARMS:
        c = A["contrasts"][arm + "_minus_R882"]
        result[arm] = bool(
            decisions[arm]["admitted"]
            and c["median"] > 0
            and c["positive_seeds"] == 3
            and c["ci95"][0] > 0
        )
    return dict(
        verdict="ADVANCE" if any(result.values()) else "NO_ADVANCE",
        advance_by_arm=result,
        serving_promotion=False,
    )
