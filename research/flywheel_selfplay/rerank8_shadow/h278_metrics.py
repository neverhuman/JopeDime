"""Frozen-shadow comparison and advisory gate; never an adoption command."""

import numpy as np


def compare(incumbent_utility, shadow_utility, families, incumbent_gate, shadow_gate):
    """H278 family-equal 2-contender MM Elo: prior .01, 400 iterations, tie 1e-4."""
    inc, picked = (
        np.asarray(incumbent_utility, float),
        np.asarray(shadow_utility, float),
    )
    fam = np.asarray(families)
    inc_gate, picked_gate = (
        np.asarray(incumbent_gate, bool),
        np.asarray(shadow_gate, bool),
    )
    assert inc.shape == picked.shape == fam.shape == inc_gate.shape == picked_gate.shape
    assert np.isfinite(inc).all() and np.isfinite(picked).all()
    delta = {
        f: float(np.mean(picked[fam == f] - inc[fam == f]))
        for f in sorted(set(families))
    }
    d = np.asarray(list(delta.values()))
    ix = np.random.default_rng(278150).integers(0, len(d), (10000, len(d)))

    def elo(values):
        w = (values > 1e-4).sum(axis=-1) + 0.5 * (np.abs(values) <= 1e-4).sum(axis=-1)
        losses = (values < -1e-4).sum(axis=-1) + 0.5 * (np.abs(values) <= 1e-4).sum(
            axis=-1
        )
        strength = np.ones_like(np.asarray(w, float))
        for _ in range(400):
            strength = np.clip(
                (w + 0.01) * (1 + strength) / (w + losses + 0.02), 1e-6, 1e6
            )
        return 400 * np.log10(strength)

    lift = float(elo(d))
    bootstrap_elo = elo(d[ix])
    result = {
        "contexts": len(inc),
        "families": len(d),
        "family_median_delta": float(np.median(d)),
        "family_median_delta_ci95": np.quantile(
            np.median(d[ix], axis=1), [0.025, 0.975]
        ).tolist(),
        "family_mean_delta": float(d.mean()),
        "family_mean_delta_ci95": np.quantile(
            d[ix].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "family_WTL": [
            int((d > 1e-4).sum()),
            int((np.abs(d) <= 1e-4).sum()),
            int((d < -1e-4).sum()),
        ],
        "Elo_lift": lift,
        "incumbent_Elo": 1500.0,
        "shadow_Elo": 1500.0 + lift,
        "Elo_lift_ci95": np.quantile(bootstrap_elo, [0.025, 0.975]).tolist(),
        "Elo_protocol": "H278 family-mean utility comparison, tie1e-4, prior.01, MM400",
        "bootstrap_draws": 10000,
        "bootstrap_seed": 278150,
        "scope": "repeated Dev shadow, no independent test claim",
        "incumbent_invalid_picks": int((~inc_gate).sum()),
        "shadow_invalid_picks": int((~picked_gate).sum()),
        "harmed_families": sum(v < -1e-4 for v in delta.values()),
        "harmed_families_gt_001": sum(v < -0.01 for v in delta.values()),
        "family_deltas": delta,
        "fam_769243ee": {
            f: v for f, v in delta.items() if f.startswith("fam-769243ee")
        },
    }
    result["strict_positive_no_excess_invalid"] = bool(
        result["family_median_delta"] > 0
        and result["shadow_invalid_picks"] <= result["incumbent_invalid_picks"]
    )
    return result


def frozen_picks(scores, invalid_probability, margin, tau=0.10):
    """H276/H278 masks every nonbaseline slot at p>=tau before strict selection."""
    scores = np.array(scores, dtype=np.float64, copy=True)
    p = np.asarray(invalid_probability)
    assert scores.shape == p.shape and scores.ndim == 2 and scores.shape[1] == 8
    assert (
        np.isfinite(scores).all()
        and np.isfinite(p).all()
        and np.all((p >= 0) & (p <= 1))
    )
    veto = p >= tau
    veto[:, 0] = False
    scores[veto] = -np.inf
    best = scores.argmax(axis=1)
    n = np.arange(len(scores))
    return np.where((scores[n, best] - scores[:, 0]) > margin, best, 0).astype(np.int64)


def advisory_gate(rounds, p2_verdict):
    """Require consecutive complete rounds and strict success for all 3 fixed seeds."""
    rows = sorted(rounds, key=lambda r: r["round"])
    streak = 0
    previous = None
    for r in rows:
        valid = (
            r.get("status") == "SCORED"
            and r.get("completed_round") is True
            and set(r.get("seeds", {})) == {"278101", "278102", "278103"}
            and all(
                s["family_median_delta"] > 0
                and s["shadow_invalid_picks"] <= s["incumbent_invalid_picks"]
                for s in r["seeds"].values()
            )
        )
        if previous is not None and r["round"] != previous + 1:
            streak = 0
        streak = streak + 1 if valid else 0
        previous = r["round"]
    return {
        "H278_P2_ADOPT_SHADOW_PASS": p2_verdict == "ADOPT_SHADOW_PASS",
        "consecutive_strict_success_rounds": streak,
        "numerical_prerequisites_met": bool(
            p2_verdict == "ADOPT_SHADOW_PASS" and streak >= 3
        ),
        "adoption_authorized": False,
        "decision_owner": "watcher",
        "promotion_wiring": False,
    }
