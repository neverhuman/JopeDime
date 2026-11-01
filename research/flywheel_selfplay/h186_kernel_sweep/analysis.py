"""H186 bindings around the imported H177 decision implementation, with seven-arm Holm."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import numpy as np
from parents import h177

SEEDS = (186101, 186102, 186103)
FOLD_SEED, BOOT_SEED = 186140, 186141
ARMS = ("KX0.5", "KX1.5", "KX2.0", "TOPK4", "TOPK16", "W0X0.5", "W0X2")
ALL_ARMS = ("R882",) + ARMS
H177_DECISION_RULE = h177.H177_DECISION_RULE
pooled_b_select = h177.pooled_b_select
StageOrderGuard = h177.StageOrderGuard
verify_pooled_b_coverage = h177.verify_pooled_b_coverage
resolve_h177_verdict = h177.resolve_h177_verdict
DRAWS = 10000


@contextmanager
def bindings(seed=SEEDS[0]):
    """Only data/config globals are rebound in this private module; function bodies stay original.

    H177's positive step slot is a candidate VERSION=1, never an optimizer update.
    The lock retains wave=H177 for the imported guard schema; study=H186 identifies this study.
    """
    if seed not in SEEDS:
        raise ValueError("unknown reshuffle seed")
    names = ("ARMS", "PRESPECIFIED_CANDIDATE_STEP", "FOLD_SEED")
    old = {name: getattr(h177, name) for name in names}
    h177.ARMS = ARMS
    h177.PRESPECIFIED_CANDIDATE_STEP = dict.fromkeys(ARMS, 1)
    h177.FOLD_SEED = h177.stream_seed(FOLD_SEED, seed, "fold-reshuffle")
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(h177, name, value)


def fold_manifest(rows):
    result = {}
    for seed in SEEDS:
        with bindings(seed):
            result[str(seed)] = dict(
                effective_seed=h177.FOLD_SEED,
                families=h177.folds(rows),
                coverage=verify_pooled_b_coverage(rows),
            )
    if len({json.dumps(x["families"]) for x in result.values()}) != 3:
        raise RuntimeError("reshuffles must be distinct")
    return result


def select(rows, panel):
    with bindings():
        decisions = H177_DECISION_RULE.select(
            rows,
            panel[:, :, 0],
            {a: panel[:, :, i] for i, a in enumerate(ARMS, 1)},
            seed=BOOT_SEED,
        )
    for d in decisions.values():
        # H177 has a literal metadata field 4; its holm() actually uses len(pvals)=7.
        d["h177_literal_metadata_m"] = d.pop("holm_multiplicity")
        d["holm_multiplicity"] = 7
        d["candidate_version"] = d.pop("prespecified_candidate_step")
        d["optimizer_updates"] = 0
        d["selected_candidate_version"] = d.pop("selected_step")
        d["unadjusted_selected_candidate_version"] = d.pop("selected_step_unadjusted")
        d.pop("holm_p_treatments3")
        d.pop("admitted_treatments3")
    return decisions


def summarize(rows, panel, seed=BOOT_SEED):
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (3, len(rows), 8) or not np.isfinite(values).all():
        raise ValueError("finite 3-seed 8-policy panel required")
    families = [r["family_id"] for r in rows]
    fm = np.stack([h177.family_matrix(v, families)[1] for v in values])
    n = fm.shape[1]
    ix = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    result = {}
    for col, arm in enumerate(ARMS, 1):
        d = fm[:, :, col] - fm[:, :, 0]
        per_seed = d.mean(axis=1)
        boot = np.median(d[:, ix].mean(axis=2), axis=0)
        result[arm] = dict(
            per_seed_gain=per_seed.tolist(),
            median_gain=float(np.median(per_seed)),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            positive_seeds=int((per_seed > 0).sum()),
            p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
        )
    adjusted = h177.holm({a: d["p_one_sided"] for a, d in result.items()})
    for arm in ARMS:
        result[arm]["holm_p"] = adjusted[arm]
    wins = np.stack([h177.family_wins(v, families) for v in values])
    elo = h177.bt_ratings(wins.sum(axis=1))
    ei = np.random.default_rng(seed + 1).integers(0, n, (1000, n))
    # Median seed Elo for each shared-family resample, matching the reported functional.
    boot_elo = np.stack([h177.bt_ratings(w[ei].sum(axis=1)) for w in wins])
    return dict(
        contexts=len(rows),
        families=n,
        contrasts=result,
        family_utility_by_seed=fm.mean(axis=1).tolist(),
        elo_by_seed=elo.tolist(),
        median_elo=np.median(elo, axis=0).tolist(),
        elo_ci95=np.quantile(
            np.median(boot_elo, axis=0), [0.025, 0.975], axis=0
        ).T.tolist(),
        exact_minus2_by_seed=(values == -2.0).sum(axis=1).tolist(),
        utility_bootstrap_draws=DRAWS,
        elo_bootstrap_draws=1000,
    )


def verdict(decisions, assess, receipt, analysis_sha):
    confirmed = {
        a: bool(
            decisions[a]["admitted"]
            and assess["contrasts"][a]["median_gain"] > 0
            and assess["contrasts"][a]["positive_seeds"] == 3
            and assess["contrasts"][a]["ci95"][0] > 0
        )
        for a in ARMS
    }
    result = resolve_h177_verdict(any(confirmed.values()), receipt, analysis_sha)
    return dict(**result, confirmed=confirmed, serving_promotion=False)


def lock_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
