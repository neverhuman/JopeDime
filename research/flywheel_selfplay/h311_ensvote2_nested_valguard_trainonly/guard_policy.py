"""Frozen H295 fallback with the frozen H303 primary agreement threshold retained."""

import numpy as np

GRID = (0.030, 0.035, 0.040, 0.045, 0.050, 0.060)


def fallback(primary, members, votes, raw, pinv, tau):
    out = np.asarray(primary, dtype=np.int32).copy()
    for i, c0 in enumerate(primary):
        nz = [int(x) for x in members[:, i] if x > 0]
        alternatives = set(nz) - {int(c0)}

        def key(c):
            return (
                nz.count(c),
                sum(int(votes[m, i]) for m in range(3) if members[m, i] == c),
                float(raw[:, i, c].mean()),
                -c,
            )

        alternatives = sorted(alternatives, key=key, reverse=True)
        if c0 > 0:
            if pinv[i, c0] <= tau:
                continue
            good = [c for c in alternatives if pinv[i, c] <= tau]
            out[i] = good[0] if good else (c0 if pinv[i, 0] > pinv[i, c0] else 0)
        elif pinv[i, 0] > max(tau, 0.055):
            good = [c for c in alternatives if pinv[i, c] <= min(tau, 0.015)]
            out[i] = good[0] if good else 0
    return out


def evaluate(picks, u, gate, fams):
    idx = np.arange(len(picks))
    delta = u[idx, picks] - u[:, 0]
    fg = np.array([delta[fams == f].mean() for f in sorted(set(fams))])
    return (
        float(fg.mean()),
        int((fg < -0.01 - 1e-12).sum()),
        int((~gate[idx, picks]).sum()),
    )


def choose_tau(primary, members, votes, raw, pinv, u, gate, fams):
    gain0, harm0, _ = evaluate(primary, u, gate, fams)
    inv0 = int((~gate[:, 0]).sum())
    table = []
    for tau in GRID:
        pick = fallback(primary, members, votes, raw, pinv, tau)
        gain, harm, invalid = evaluate(pick, u, gate, fams)
        feasible = int(invalid <= inv0 and gain >= 0.85 * gain0 and harm <= harm0)
        table.append(
            dict(
                tau=tau,
                feasible=feasible,
                invalid=invalid,
                harmed=harm,
                gain=gain,
                interventions=int((pick != primary).sum()),
            )
        )
    best = max(
        table,
        key=lambda x: (x["feasible"], -x["invalid"], -x["harmed"], x["gain"], x["tau"]),
    )
    return best["tau"], table


def decision(
    invalid_a, invalid_b, invalid_n, harm_a, harm_b, paired_median, paired_lcb
):
    gates = dict(
        invalid_cut_median_ge2=bool(np.median(invalid_b) <= np.median(invalid_a) - 2),
        invalid_beats_null_ge6of9=bool(np.sum(np.asarray(invalid_b) < invalid_n) >= 6),
        paired_gain_median_ge_minus002=bool(paired_median >= -0.002),
        paired_gain_lcb95_ge_minus005=bool(paired_lcb >= -0.005),
        harmed_not_above_a_ge6of9=bool(np.sum(np.asarray(harm_b) <= harm_a) >= 6),
    )
    verdict = (
        "VALGUARD_NESTED_ADVANCE"
        if all(gates.values())
        else (
            "VALGUARD_NULL_EQUIVALENT"
            if not gates["invalid_beats_null_ge6of9"]
            else "VALGUARD_NOT_ADVANCED"
        )
    )
    return dict(
        verdict=verdict, gates=gates, failed=[k for k, v in gates.items() if not v]
    )
