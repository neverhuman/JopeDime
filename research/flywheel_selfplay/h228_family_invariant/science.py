"""H228 prospective family partitions, Holm-2 candidate readout, and DL-Val91 evaluation."""

import hashlib
import math
import numpy as np

SEEDS = (228101, 228102, 228103)
FOLD_SEEDS = (228141, 228142, 228143)
SMOKE_SEED = 228001
ARMS = ("CTRL", "ANDMASK", "GDRO", "ANDMASK_SHUF")
CANDIDATES = ("ANDMASK", "GDRO")
CONTROLS = ("CTRL", "ANDMASK_SHUF")
LEGACY_ARMS = {
    "CTRL": "CONTROL",
    "ANDMASK": "FBAL",
    "GDRO": "FBAL",
    "ANDMASK_SHUF": "FBAL",
}
BOOT_SEED = 228150
ELO_SEED = 228151
VAL_BOOT_SEED = 228152
MARGIN = 0.002
TOL = 1e-12
HARM_LIMIT = 0.10


def stream(*parts):
    return int.from_bytes(
        hashlib.sha256(":".join(map(str, ("H228", *parts))).encode()).digest()[:8],
        "little",
    )


def folds(families, seed):
    """H109/H219 round-robin permutation on sorted families at an explicit prospective seed."""
    ordered = sorted(set(families))
    if len(ordered) != 205:
        raise ValueError("205 real TRAIN families required")
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    return [sorted(ordered[int(i)] for i in permutation[j::5]) for j in range(5)]


def partition(rows, seed, outer):
    if seed not in SEEDS or outer not in range(5):
        raise ValueError("unregistered split")
    held = set(folds([r["family_id"] for r in rows], seed + 40)[outer])
    return (
        [r for r in rows if r["family_id"] not in held],
        [r for r in rows if r["family_id"] in held],
    )


def training_seed(seed, outer):
    if seed not in (*SEEDS, SMOKE_SEED) or outer not in range(5):
        raise ValueError("unregistered training stream")
    return seed * 10 + outer


def legacy_config(legacy):
    # Only the seed admission whitelist is extended; H109/H219 algorithms and constants stay pinned.
    legacy.SEEDS = tuple(training_seed(s, f) for s in SEEDS for f in range(5)) + (
        training_seed(SMOKE_SEED, 0),
    )
    return legacy


def family_mean(panel, families):
    panel = np.asarray(panel, dtype=np.float64)
    fam_arr = np.asarray(families)
    ordered = sorted(set(families))
    out = np.empty(panel.shape[:-1] + (len(ordered),), dtype=np.float64)
    for fi, fam in enumerate(ordered):
        idx = np.flatnonzero(fam_arr == fam)
        sub = panel[..., idx]
        out[..., fi] = np.apply_along_axis(
            lambda x: math.fsum(x.tolist()) / len(idx), -1, sub
        )
    return out


def _contrast_from_family_delta(delta, draws):
    n_fams = delta.shape[1]
    means = np.array(
        [math.fsum(delta[s].tolist()) / n_fams for s in range(delta.shape[0])],
        dtype=np.float64,
    )
    boot = np.median(delta[:, draws].mean(-1), axis=0)
    fam_avg = np.array(
        [math.fsum(delta[:, j].tolist()) / delta.shape[0] for j in range(n_fams)],
        dtype=np.float64,
    )
    ci95 = np.quantile(boot, [0.025, 0.975]).tolist()
    return boot, dict(
        per_seed_gain=means.tolist(),
        median_gain=float(np.median(means)),
        ci95=ci95,
        lcb95=float(np.quantile(boot, 0.05)),
        p_zero=float((1 + (boot <= 0.0).sum()) / (len(boot) + 1)),
        p_margin=float((1 + (boot <= MARGIN + TOL).sum()) / (len(boot) + 1)),
        p_margin_001=float((1 + (boot <= 0.001 + TOL).sum()) / (len(boot) + 1)),
        harm=float((fam_avg < -0.01).mean()),
        harmed_families=int((fam_avg < -0.01).sum()),
        positive_seeds=int((means > 0.0).sum()),
    )


def val91_readout(val_panels, val_families):
    if set(val_panels) != {"R882", *ARMS} or len(set(val_families)) != 19:
        raise ValueError("complete 19-family DL-Val91 panels required")
    shape = np.asarray(val_panels["R882"]).shape
    if shape != (3, len(val_families)) or any(
        np.asarray(p).shape != shape or not np.isfinite(p).all()
        for p in val_panels.values()
    ):
        raise ValueError("incomplete or nonfinite DL-Val91 panel")
    draws = np.random.default_rng(VAL_BOOT_SEED).integers(0, 19, (10000, 19))
    ordered_fams = sorted(set(val_families))
    fam_indices = [np.flatnonzero(np.asarray(val_families) == f) for f in ordered_fams]
    equal_family = {}
    context_weighted = {}
    passes = {}
    for arm, reference in [(a, "R882") for a in ARMS] + [
        ("ANDMASK", "CTRL"),
        ("GDRO", "CTRL"),
        ("ANDMASK_SHUF", "CTRL"),
        ("ANDMASK", "ANDMASK_SHUF"),
    ]:
        key = arm if reference == "R882" else f"{arm}_vs_{reference}"
        diff = np.asarray(val_panels[arm], dtype=np.float64) - np.asarray(
            val_panels[reference], dtype=np.float64
        )
        delta_fam = family_mean(diff, val_families)
        _, eq_stat = _contrast_from_family_delta(delta_fam, draws)
        eq_stat["reference"] = reference
        equal_family[key] = eq_stat

        # Context-weighted 91-context gain with 19-family cluster bootstrap.
        ctx_means = np.array(
            [math.fsum(diff[s].tolist()) / diff.shape[1] for s in range(3)],
            dtype=np.float64,
        )
        fam_sums = np.stack(
            [diff[:, idx].sum(-1) for idx in fam_indices], axis=-1
        )  # (3, 19)
        fam_counts = np.array([len(idx) for idx in fam_indices], dtype=np.float64)
        boot_num = fam_sums[:, draws].sum(-1)  # (3, 10000)
        boot_den = fam_counts[draws].sum(-1)[None, :]  # (1, 10000)
        ctx_boot = np.median(boot_num / boot_den, axis=0)
        ctx_ci95 = np.quantile(ctx_boot, [0.025, 0.975]).tolist()
        context_weighted[key] = dict(
            reference=reference,
            per_seed_gain=ctx_means.tolist(),
            median_gain=float(np.median(ctx_means)),
            ci95=ctx_ci95,
            lcb95=float(np.quantile(ctx_boot, 0.05)),
            p_zero=float((1 + (ctx_boot <= 0.0).sum()) / (len(ctx_boot) + 1)),
            p_margin=float(
                (1 + (ctx_boot <= MARGIN + TOL).sum()) / (len(ctx_boot) + 1)
            ),
            harm=eq_stat["harm"],
            harmed_families=eq_stat["harmed_families"],
            positive_seeds=int((ctx_means > 0.0).sum()),
        )
    for a in CANDIDATES:
        eq = equal_family[a]
        cw = context_weighted[a]
        passes[a] = bool(
            eq["median_gain"] > MARGIN + TOL
            and eq["ci95"][0] > 0.0
            and eq["harm"] <= HARM_LIMIT
            and eq["positive_seeds"] == 3
            and cw["median_gain"] > MARGIN + TOL
            and cw["ci95"][0] > 0.0
            and cw["positive_seeds"] == 3
        )
    return dict(
        equal_family_contrasts=equal_family,
        context_weighted_contrasts=context_weighted,
        passes=passes,
        bootstrap_draws=10000,
        bootstrap_seed=VAL_BOOT_SEED,
    )


def readout(panels, families, val_panels=None, val_families=None):
    if set(panels) != {"R882", *ARMS} or len(set(families)) != 205:
        raise ValueError("complete registered 205-family panels required")
    shape = np.asarray(panels["R882"]).shape
    if shape != (3, len(families)) or any(
        np.asarray(p).shape != shape or not np.isfinite(p).all()
        for p in panels.values()
    ):
        raise ValueError("incomplete or nonfinite panel")
    draws = np.random.default_rng(BOOT_SEED).integers(0, 205, (10000, 205))
    results, boots = {}, {}
    for arm, reference in [(a, "R882") for a in ARMS] + [
        ("ANDMASK", "CTRL"),
        ("GDRO", "CTRL"),
        ("ANDMASK_SHUF", "CTRL"),
        ("ANDMASK", "ANDMASK_SHUF"),
    ]:
        key = arm if reference == "R882" else f"{arm}_vs_{reference}"
        delta = family_mean(
            np.asarray(panels[arm], dtype=np.float64)
            - np.asarray(panels[reference], dtype=np.float64),
            families,
        )
        boot, stat = _contrast_from_family_delta(delta, draws)
        stat["reference"] = reference
        boots[key] = boot
        results[key] = stat

    adjusted, running_p, running_pm, continuing = {}, 0.0, 0.0, True
    ordered_candidates = sorted(
        CANDIDATES, key=lambda a: (results[a]["p_zero"], CANDIDATES.index(a))
    )
    for rank, arm in enumerate(ordered_candidates):
        m_rem = len(CANDIDATES) - rank
        alpha = 0.05 / m_rem
        running_p = max(running_p, min(1.0, results[arm]["p_zero"] * m_rem))
        running_pm = max(running_pm, min(1.0, results[arm]["p_margin"] * m_rem))
        corrected_ci_low = float(np.quantile(boots[arm], alpha / 2.0))
        corrected_lcb = float(np.quantile(boots[arm], alpha))
        step_ok = bool(
            running_p < 0.05
            and corrected_ci_low > 0.0
            and results[arm]["median_gain"] > MARGIN + TOL
        )
        continuing = bool(continuing and step_ok)
        adjusted[arm] = dict(
            rank=rank + 1,
            p=running_p,
            p_margin=running_pm,
            alpha=alpha,
            ci95_lower_corrected=corrected_ci_low,
            lcb95=corrected_lcb,
            passes=bool(continuing),
        )
    admitted = {
        a: bool(
            adjusted[a]["passes"]
            and results[a]["median_gain"] > MARGIN + TOL
            and results[a]["ci95"][0] > 0.0
            and results[a]["lcb95"] > 0.0
            and results[a]["harm"] <= HARM_LIMIT
            and results[a]["positive_seeds"] == 3
        )
        for a in CANDIDATES
    }
    val_summary = (
        val91_readout(val_panels, val_families)
        if val_panels is not None and val_families is not None
        else None
    )
    if admitted["ANDMASK"]:
        verdict = (
            "BREAKTHROUGH"
            if (val_summary is not None and val_summary["passes"]["ANDMASK"])
            else "CANDIDATE"
        )
    else:
        verdict = "NO_ADVANCE"
    return dict(
        contrasts=results,
        holm=adjusted,
        holm_multiplicity=len(CANDIDATES),
        admitted=admitted,
        dl_val91=val_summary,
        primary="ANDMASK",
        verdict=verdict,
        bootstrap_draws=10000,
        bootstrap_seed=BOOT_SEED,
        strict_margin=MARGIN,
    )


def jaccard_report(families):
    ordered = sorted(set(families))
    old = {"H109_10900": folds(ordered, 10900)}
    for wave in (219, 220):
        for seed in range(wave * 1000 + 141, wave * 1000 + 144):
            old[f"H{wave}_{seed}"] = folds(ordered, seed)
    for wave in (210, 215, 217, 218, 221, 222, 223, 224, 225):
        for seed in range(wave * 1000 + 141, wave * 1000 + 144):
            old[f"H{wave}_{seed}"] = [
                sorted(g.tolist())
                for g in np.array_split(
                    np.random.default_rng(seed).permutation(ordered), 5
                )
            ]
    return {
        str(seed + 40): {
            name: [
                [len(set(a) & set(b)) / len(set(a) | set(b)) for b in prior]
                for a in folds(ordered, seed + 40)
            ]
            for name, prior in old.items()
        }
        for seed in SEEDS
    }


def require_native_reference(observed, expected):
    if (
        len(observed) != 91
        or not np.array_equal(
            np.asarray(observed, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
        )
        or float(np.mean(observed)) != 1.1377941898164525
    ):
        raise RuntimeError("unchanged-r882 native91 bit-exact regression failed")
