"""Preregistered H201 paired-family inference and admission-history replay."""

import hashlib

import numpy as np

SEEDS = (201101, 201102, 201103)
FOLD_SEEDS = (201141, 201142, 201143)
BOOT_SEED = 201150
DRAWS = 10_000


def stream(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def weights(n, seed, draws=DRAWS):
    if n < 2 or draws < 1:
        raise ValueError("at least two families and one draw required")
    ix = np.random.default_rng(seed).integers(n, size=(draws, n))
    return np.stack([(ix == j).sum(axis=1) for j in range(n)], axis=1) / n


def family_panel(panel, families):
    panel = np.asarray(panel, dtype=np.float64)
    labels = np.asarray(families)
    if panel.ndim != 2 or panel.shape[1] != len(labels) or not np.isfinite(panel).all():
        raise ValueError("finite policy-by-context panel required")
    names = sorted(set(labels.tolist()))
    return names, np.stack([panel[:, labels == f].mean(axis=1) for f in names], axis=1)


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    records = []
    for seed in SEEDS:
        draws = weights(len(values), seed) @ values
        records.append(
            dict(
                seed=seed,
                gain=float(values.mean()),
                ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
            )
        )
    return dict(
        gain=float(values.mean()),
        median_over_seeds=float(np.median([r["gain"] for r in records])),
        ci95=np.median([r["ci95"] for r in records], axis=0).tolist(),
        bootstrap_seeds=records,
    )


def strict_accept(candidate, best, boot):
    delta = np.asarray(candidate) - np.asarray(best)
    lower = float(np.quantile(boot @ delta, 0.025))
    return bool(delta.mean() > 0 and lower > 0), lower


def sma_replay(rounds, panel, initial, actions):
    """True counterfactual chain and separately the observed-incumbent paired screen."""
    panel = np.asarray(panel, dtype=np.float64)
    if len(rounds) != len(panel) or len(actions) != len(panel):
        raise ValueError("complete history panel required")
    boot = weights(panel.shape[1], BOOT_SEED)
    incumbent = np.asarray(initial).copy()
    observed = incumbent.copy()
    decisions, best_round = [], None
    for round_id, candidate, action in zip(rounds, panel, actions, strict=True):
        accept, lower = strict_accept(candidate, incumbent, boot)
        screen, historical_lower = strict_accept(candidate, observed, boot)
        decisions.append(
            dict(
                round=int(round_id),
                current_action=action,
                SMA_accept=accept,
                LCB95=lower,
                observed_incumbent_screen_accept=screen,
                observed_incumbent_LCB95=historical_lower,
            )
        )
        if accept:
            incumbent = candidate.copy()
            best_round = int(round_id)
        if action in ("accept_improvement", "accept_tie"):
            observed = candidate.copy()
    counts = {
        action: dict(
            current=sum(a == action for a in actions),
            SMA_accept=sum(
                d["SMA_accept"] for d in decisions if d["current_action"] == action
            ),
            SMA_reject=sum(
                not d["SMA_accept"] for d in decisions if d["current_action"] == action
            ),
            observed_screen_accept=sum(
                d["observed_incumbent_screen_accept"]
                for d in decisions
                if d["current_action"] == action
            ),
        )
        for action in sorted(set(actions))
    }
    ties = counts.get("accept_tie", dict(current=0, SMA_reject=0))
    improvements = counts.get("accept_improvement", dict(current=0, SMA_reject=0))
    h2 = (
        ties["current"] >= 4
        and ties["SMA_reject"] == ties["current"]
        and improvements["current"] >= 4
        and improvements["SMA_reject"] >= 3
    )
    return dict(
        counts=counts,
        SMA_total_accept=sum(d["SMA_accept"] for d in decisions),
        SMA_total_reject=sum(not d["SMA_accept"] for d in decisions),
        SMA_final_admitted_round=best_round,
        H2=bool(h2),
    ), decisions


def null_max(history, reference):
    """Joint centered family bootstrap, preserving candidate dependence within families."""
    delta = np.asarray(history) - np.asarray(reference)
    centered = delta - delta.mean(axis=1, keepdims=True)
    observed = float(max(0, delta.mean(axis=1).max()))
    results = []
    for seed in SEEDS:
        boot = weights(delta.shape[1], seed)
        maxima = np.zeros(DRAWS)
        for start in range(0, DRAWS, 250):
            maxima[start : start + 250] = np.maximum(
                0, (boot[start : start + 250] @ centered.T).max(axis=1)
            )
        results.append(
            dict(
                seed=seed,
                expected_max=float(maxima.mean()),
                null_max_ci95=np.quantile(maxima, [0.025, 0.975]).tolist(),
                p_max=float((1 + (maxima >= observed).sum()) / (DRAWS + 1)),
            )
        )
    return dict(
        N=len(history),
        observed_max_gain=observed,
        expected_max=float(np.median([r["expected_max"] for r in results])),
        ci95=np.median([r["null_max_ci95"] for r in results], axis=0).tolist(),
        p_max=float(np.median([r["p_max"] for r in results])),
        per_seed=results,
        model="joint family bootstrap after centering every candidate to zero equal-family gain",
    )


def split_selection(history, reference, rounds, family_names):
    delta = np.vstack([np.zeros(len(reference)), np.asarray(history) - reference])
    policy_rounds = [882] + list(rounds)
    results = []
    for seed in FOLD_SEEDS:
        perm = np.random.default_rng(seed).permutation(len(reference))
        a, b = perm[: len(reference) // 2], perm[len(reference) // 2 :]
        winner = int(np.argmax(delta[:, a].mean(axis=1)))
        held = delta[winner, b]
        boot = weights(len(b), stream(BOOT_SEED, "split-B", seed)) @ held
        results.append(
            dict(
                seed=seed,
                A_families=[family_names[j] for j in a],
                B_families=[family_names[j] for j in b],
                selected_round=int(policy_rounds[winner]),
                A_gain=float(delta[winner, a].mean()),
                B_gain=float(held.mean()),
                B_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            )
        )
    return dict(
        per_seed=results,
        median_B_gain=float(np.median([r["B_gain"] for r in results])),
        median_B_ci95=np.median([r["B_ci95"] for r in results], axis=0).tolist(),
        all_three_B_positive=all(r["B_gain"] > 0 for r in results),
    )


def bt_ratings(wins):
    """Bradley-Terry MM, weak symmetric prior .01; reference fixed to 1500."""
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    eye = np.eye(n)
    games = wins + np.swapaxes(wins, -1, -2)
    totals = wins.sum(axis=-1) + 0.01 * (n - 1)
    strength = np.ones(wins.shape[:-1])
    for _ in range(400):
        denom = (
            (games + 0.02 * (1 - eye))
            / np.maximum(strength[..., :, None] + strength[..., None, :], 1e-12)
        ).sum(axis=-1)
        strength = np.clip(totals / denom, 1e-6, 1e6)
        strength /= np.maximum(strength[..., :1], 1e-12)
    return 1500 + 400 * np.log10(np.maximum(strength, 1e-9))


def elo(panel, families, names):
    delta = panel.T[:, :, None] - panel.T[:, None, :]
    wins = np.where(delta > 1e-4, 1.0, np.where(delta < -1e-4, 0.0, 0.5)) * (
        1 - np.eye(len(names))
    )
    labels = np.asarray(families)
    fw = np.stack([wins[labels == f].mean(axis=0) for f in sorted(set(families))])
    point = bt_ratings(fw.sum(axis=0))
    intervals = []
    for seed in SEEDS:
        boot = weights(len(fw), stream(BOOT_SEED, "elo", seed)) * len(fw)
        ratings = bt_ratings(np.einsum("qf,fij->qij", boot, fw))
        intervals.append(np.quantile(ratings, [0.025, 0.975], axis=0))
    ci = np.median(intervals, axis=0)
    return {
        name: dict(
            Elo=float(point[j]),
            ci95=ci[:, j].tolist(),
            per_seed_ci95=[v[:, j].tolist() for v in intervals],
        )
        for j, name in enumerate(names)
    }


def verdict(lower, null_p, splits_positive, elo_lower, complete=True):
    if not complete:
        return "INCONCLUSIVE"
    if lower <= 0 and null_p >= 0.05:
        return "WINNERS_CURSE"
    if lower > 0 and splits_positive and elo_lower > 1500:
        return "CANDIDATE_REAL"
    return "INCONCLUSIVE"
