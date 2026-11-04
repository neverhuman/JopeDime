"""H305 independent family-level statistics; no peer evaluator imports."""

import hashlib
import json
import math
from pathlib import Path
import numpy as np

TIE = 1e-4


def win_tie_loss(delta):
    d = np.asarray(delta)
    return [int(np.sum(d > TIE)), int(np.sum(np.abs(d) <= TIE)), int(np.sum(d < -TIE))]


def elo_mm(wins, losses, prior=0.01):
    """Symmetric two-player MM, normalized to incumbent strength1 each step."""
    count = np.array([losses + prior, wins + prior], dtype=float)
    strength = np.ones(2)
    for _ in range(400):
        denominator = count.sum() / strength.sum()
        updated = count / denominator
        updated /= updated[0]
        if np.max(np.abs(updated - strength)) < 1e-13:
            strength = updated
            break
        strength = updated
    return float(1500 + 400 * math.log10(strength[1]))


def family_deltas(utility, picks, families):
    chosen = utility[np.arange(len(families))[None, :], picks]
    return np.stack(
        [
            chosen[:, families == f].mean(1) - utility[families == f, 0].mean()
            for f in sorted(set(families))
        ],
        axis=1,
    )


def concentration(delta, families):
    out = []
    for d in delta:
        order = np.argsort(-d, kind="stable")
        total, positive = float(d.sum()), float(d[d > 0].sum())
        remaining = np.delete(d, order[:5])
        boot = np.random.default_rng(302150).integers(
            0, len(remaining), (10000, len(remaining))
        )
        out.append(
            dict(
                drop_top_gain={
                    str(n): float(np.delete(d, order[:n]).mean()) for n in (1, 5, 10)
                },
                top10_net_share=float(d[order[:10]].sum() / total)
                if total != 0
                else None,
                top10_positive_share=float(
                    np.maximum(d[order[:10]], 0).sum() / positive
                )
                if positive > 0
                else None,
                top10=[
                    dict(family=str(families[i]), gain=float(d[i])) for i in order[:10]
                ],
                drop_top5_lcb95=float(np.quantile(remaining[boot].mean(1), 0.05)),
            )
        )
    return out


def seed_dependence(picks, delta):
    pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            union = (picks[i] != 0) | (picks[j] != 0)
            pairs.append(
                dict(
                    seeds=[30001 + i, 30001 + j],
                    agreement=float(np.mean(picks[i] == picks[j])),
                    union_switched_contexts=int(union.sum()),
                    disagreements=int(np.sum(picks[i] != picks[j])),
                    agreement_on_switched_union=float(
                        np.mean(picks[i, union] == picks[j, union])
                    )
                    if union.any()
                    else None,
                )
            )
    if np.any(delta.std(1) == 0):
        correlation, effective = None, None
    else:
        correlation = np.corrcoef(delta)
        eigen = np.linalg.eigvalsh(correlation)
        effective = float(eigen.sum() ** 2 / np.square(eigen).sum())
        correlation = correlation.tolist()
    return dict(
        pairs=pairs,
        family_delta_correlation=correlation,
        effective_seed_count_spectral=effective,
        interpretation="Descriptive dependence across the same families; three training seeds are not three independent data samples.",
    )


def permutation_null(utility, picks, families, draws=1000, seed=305150):
    """Preserve vectors of three seeds while exchanging context slots within family."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(families == f) for f in sorted(set(families))]
    values = np.empty(draws)
    all_indices = np.arange(len(families))
    mutable = sum(len(g) > 1 and np.any(picks[:, g] != picks[:, g[:1]]) for g in groups)
    # Bincount implements equal family weighting without 205 repeated masks per draw.
    weights = np.zeros(len(families))
    for g in groups:
        weights[g] = 1.0 / (len(groups) * len(g))

    # The identity and permuted statistics must use exactly the same arithmetic.
    # Different reduction orders can turn a degenerate null into a false p=1/(B+1).
    def statistic(p):
        # Advanced indexing can change C/F memory layout and therefore BLAS reductions.
        u = np.ascontiguousarray(utility[all_indices[None, :], p] - utility[:, 0])
        return float(np.median(u @ weights))

    observed = statistic(picks)
    for draw in range(draws):
        perm = all_indices.copy()
        for g in groups:
            perm[g] = rng.permutation(g)
        values[draw] = statistic(picks[:, perm])
    return dict(
        seed=seed,
        draws=draws,
        observed=observed,
        upper_tail_count=int(np.sum(values >= observed)),
        p_upper_plus_one=float((1 + np.sum(values >= observed)) / (draws + 1)),
        null_mean=float(values.mean()),
        null_sd=float(values.std()),
        null_min=float(values.min()),
        null_max=float(values.max()),
        null_q025_q50_q975=np.quantile(values, [0.025, 0.5, 0.975]).tolist(),
        unique_statistics=len(np.unique(values)),
        families_with_nonconstant_pick_vectors=int(mutable),
        null_statistics_sha256=hashlib.sha256(
            values.astype("<f8").tobytes()
        ).hexdigest(),
        limitation="Conditional within-family context-exchange null; not a retraining/label-shuffle null or proof against design selection leakage.",
    )


def summarize(utility, picks, families, gates):
    delta = family_deltas(utility, picks, families)
    names = sorted(set(families))
    index = np.random.default_rng(302150).integers(0, len(names), (10000, len(names)))
    boot = np.stack([d[index].mean(1) for d in delta])
    wtl = [win_tie_loss(d) for d in delta]
    invalid0 = (~gates[:, 0]) | (utility[:, 0] <= -2 + 1e-9)
    invalid = []
    for p in picks:
        flags = (~gates[np.arange(len(p)), p]) | (
            utility[np.arange(len(p)), p] <= -2 + 1e-9
        )
        worse, better = int(np.sum(flags & ~invalid0)), int(np.sum(invalid0 & ~flags))
        n = worse + better
        mcnemar = sum(math.comb(n, k) for k in range(worse, n + 1)) / 2**n
        invalid.append(
            dict(
                baseline=int(invalid0.sum()),
                challenger=int(flags.sum()),
                excess=int(flags.sum() - invalid0.sum()),
                discordant_worse=worse,
                discordant_better=better,
                p_one_sided=mcnemar,
            )
        )
    be = np.stack(
        [
            1500
            + 400
            * np.log10(
                ((d[index] > TIE).sum(1) + 0.01) / ((d[index] < -TIE).sum(1) + 0.01)
            )
            for d in delta
        ]
    )
    return dict(
        gains=delta.mean(1).tolist(),
        median_gain=float(np.median(delta.mean(1))),
        pooled_ci95=np.quantile(np.median(boot, axis=0), [0.025, 0.975]).tolist(),
        per_seed_lcb95=np.quantile(boot, 0.05, axis=1).tolist(),
        family_medians=np.median(delta, axis=1).tolist(),
        family_wins_ties_losses=wtl,
        changed_picks=np.sum(picks != 0, axis=1).tolist(),
        elo_actual_r882_slot0_1500=[elo_mm(wins, losses) for wins, ties, losses in wtl],
        elo_ci95=[np.quantile(v, [0.025, 0.975]).tolist() for v in be],
        requested_r3175_elo=None,
        requested_r3175_elo_status="UNMEASURED: original slot0 is r882",
        invalid=invalid,
        harmed_share=np.mean(delta < -0.01, axis=1).tolist(),
        concentration=concentration(delta, names),
        seed_dependence=seed_dependence(picks, delta),
        permutation=permutation_null(utility, picks, families),
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--picks", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    with np.load(a.data, allow_pickle=False) as z:
        result = summarize(
            z["utility"],
            np.load(a.picks, allow_pickle=False),
            z["families"],
            z["gates"],
        )
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "gains",
                    "pooled_ci95",
                    "elo_actual_r882_slot0_1500",
                    "family_wins_ties_losses",
                    "seed_dependence",
                    "permutation",
                ]
            }
        )
    )
