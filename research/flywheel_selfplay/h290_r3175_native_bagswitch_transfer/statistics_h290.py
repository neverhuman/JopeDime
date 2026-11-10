"""Frozen family-weighted inference and six-arm MM Bradley-Terry Elo."""

import numpy as np
from scipy.stats import binomtest

ARMS = ("BASE", "NATIVE", "R882_FIT", "PLACEBO", "NULL_RV", "NULL_LF")
SEEDS = (290101, 290102, 290103)
TIE = 1e-4


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


def sign(delta):
    w, l = int((delta > TIE).sum()), int((delta < -TIE).sum())  # noqa: E741
    return dict(
        wins=w,
        ties=len(delta) - w - l,
        losses=l,
        p_one_sided=1.0
        if w + l == 0
        else float(binomtest(w, w + l, 0.5, alternative="greater").pvalue),
    )


def bootstrap_summary(points, samples):
    return dict(
        per_seed=points.tolist(),
        median=float(np.median(points)),
        per_seed_ci95=np.quantile(samples, [0.025, 0.975], axis=0).T.tolist(),
        median_ci95=np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist(),
    )


def summarize(u, invalid, families, draws=10000):
    u = np.asarray(u)
    families = np.asarray(families)
    ids = sorted(set(families))
    masks = [families == f for f in ids]
    assert (
        u.shape == (3, len(families), 6)
        and invalid.shape == u.shape
        and np.isfinite(u).all()
    )
    fm = np.stack([u[:, m].mean(axis=1) for m in masks], axis=1)
    indices = np.random.default_rng(290150).integers(0, len(ids), (draws, len(ids)))
    count = np.zeros((draws, len(ids)))
    np.add.at(count, (np.arange(draws)[:, None], indices), 1)
    boot = np.stack([count @ x / len(ids) for x in fm], axis=1)
    delta = fm - fm[:, :, 0, None]
    bd = boot - boot[:, :, 0, None]
    pairwise = u[:, :, :, None] - u[:, :, None, :]
    outcomes = np.where(pairwise > TIE, 1.0, np.where(pairwise < -TIE, 0.0, 0.5)) * (
        1 - np.eye(6)
    )
    fw = np.stack([outcomes[:, m].mean(axis=1) for m in masks], axis=1)
    elo = bt_ratings(fw.sum(axis=1))
    elo_samples = []
    for s in range(3):
        games = (count @ fw[s].reshape(len(ids), 36)).reshape(draws, 6, 6)
        elo_samples.append(bt_ratings(games))
    eb = np.stack(elo_samples, axis=1)
    arms = {}
    for j, arm in enumerate(ARMS):
        d = delta[:, :, j]
        arms[arm] = dict(
            gain=bootstrap_summary(d.mean(axis=1), bd[:, :, j]),
            family_median_per_seed=np.median(d, axis=1).tolist(),
            family_median=float(np.median(np.median(d, axis=1))),
            sign_test=[sign(x) for x in d],
            harmed_loss_gt_001=(d < -0.01).sum(axis=1).tolist(),
            raw_minus2=(u[:, :, j] == -2).sum(axis=1).tolist(),
            gate_invalid=invalid[:, :, j].sum(axis=1).tolist(),
            elo=bootstrap_summary(elo[:, j], eb[:, :, j]),
        )
    contrast = bootstrap_summary(
        (fm[:, :, 1] - fm[:, :, 2]).mean(axis=1), boot[:, :, 1] - boot[:, :, 2]
    )
    return (
        dict(
            contexts=len(families),
            families=len(ids),
            bootstrap_seed=290150,
            bootstrap_draws=draws,
            arms=arms,
            native_minus_r882_fit=contrast,
        ),
        fm,
        indices,
    )


def transfer_gate(summary, arm):
    a = summary["arms"][arm]
    ref = summary["arms"]["R882_FIT"]
    base = summary["arms"]["BASE"]
    gates = dict(
        median_gt_001=a["gain"]["median"] > 0.010,
        lcb_positive_all_three=all(x[0] > 0 for x in a["gain"]["per_seed_ci95"]),
        sign_p_below_005_all_three=all(x["p_one_sided"] < 0.05 for x in a["sign_test"]),
        harmed_not_above_r882_each_seed=all(
            x <= y for x, y in zip(a["harmed_loss_gt_001"], ref["harmed_loss_gt_001"])
        ),
        raw_minus2_not_above_base_each_seed=all(
            x <= y for x, y in zip(a["raw_minus2"], base["raw_minus2"])
        ),
    )
    return dict(pass_gate=all(gates.values()), gates=gates)


def verdict(primary):
    gates = {arm: transfer_gate(primary, arm) for arm in ARMS[1:]}
    controls_fail = all(
        not gates[a]["pass_gate"] for a in ["PLACEBO", "NULL_RV", "NULL_LF"]
    )
    better = primary["native_minus_r882_fit"]["median"] > 0
    if gates["NATIVE"]["pass_gate"] and controls_fail:
        result = "NATIVE_TRANSFER_PASS" if better else "NATIVE_NOT_BETTER"
    else:
        result = "NOT_TRANSFERRED_SEMI_FRESH"
    return dict(
        verdict=result,
        arm_gates=gates,
        all_controls_fail=controls_fail,
        native_minus_r882_median_positive=better,
        shadow_only=True,
        independent_fresh_confirmation_required=True,
    )
