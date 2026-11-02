"""H220 prospective family partitions and strict, paired primary readout."""

import hashlib
import numpy as np

SEEDS = (220101, 220102, 220103)
ARMS = ("CONTROL", "MIX")
CANDIDATES = ("MIX",)
LEGACY_ARMS = {"CONTROL": "CONTROL", "MIX": "CONTROL"}
BOOT_SEED = 220150


def stream(*parts):
    return int.from_bytes(
        hashlib.sha256(":".join(map(str, ("H220", *parts))).encode()).digest()[:8],
        "little",
    )


def folds(families, seed):
    """H109 round-robin permutation, amended to an explicit prospective seed."""
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
    if seed not in (*SEEDS, 220001) or outer not in range(5):
        raise ValueError("unregistered training stream")
    return seed * 10 + outer


def legacy_config(legacy):
    # Only the seed admission whitelist changes; algorithms/constants stay pinned.
    legacy.SEEDS = tuple(training_seed(s, f) for s in SEEDS for f in range(5)) + (
        training_seed(220001, 0),
    )
    return legacy


def family_mean(panel, families):
    panel = np.asarray(panel, dtype=np.float64)
    return np.stack(
        [
            panel[..., np.flatnonzero(np.asarray(families) == f)].mean(-1)
            for f in sorted(set(families))
        ],
        axis=-1,
    )


def readout(panels, families):
    if set(panels) != {"R882", *ARMS} or len(set(families)) != 205:
        raise ValueError("complete registered panels required")
    shape = np.asarray(panels["R882"]).shape
    if shape != (3, len(families)) or any(
        np.asarray(p).shape != shape or not np.isfinite(p).all()
        for p in panels.values()
    ):
        raise ValueError("incomplete or nonfinite panel")
    draws = np.random.default_rng(BOOT_SEED).integers(0, 205, (10000, 205))
    results, boots = {}, {}
    for arm, reference in [(a, "R882") for a in ARMS] + [("MIX", "CONTROL")]:
        key = arm if reference == "R882" else arm + "_vs_CONTROL"
        delta = family_mean(np.asarray(panels[arm]) - panels[reference], families)
        means = delta.mean(1)
        boot = np.median(delta[:, draws].mean(-1), axis=0)
        boots[key] = boot
        results[key] = dict(
            reference=reference,
            per_seed_gain=means.tolist(),
            median_gain=float(np.median(means)),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            lcb95=float(np.quantile(boot, 0.05)),
            p_margin=float((1 + (boot <= 0.001 + 1e-12).sum()) / 10001),
            harm=float((delta.mean(0) < -0.01).mean()),
            harmed_families=int((delta.mean(0) < -0.01).sum()),
            positive_seeds=int((means > 0).sum()),
        )
    adjusted, running, continuing = {}, 0.0, True
    for rank, arm in enumerate(
        sorted(CANDIDATES, key=lambda a: (results[a]["p_margin"], CANDIDATES.index(a)))
    ):
        alpha = 0.05 / (len(CANDIDATES) - rank)
        running = max(
            running, min(1.0, results[arm]["p_margin"] * (len(CANDIDATES) - rank))
        )
        corrected_lcb = float(np.quantile(boots[arm], alpha))
        continuing = continuing and running < 0.05 and corrected_lcb > 0.001 + 1e-12
        adjusted[arm] = dict(
            p=running, alpha=alpha, lcb95=corrected_lcb, passes=bool(continuing)
        )
    admitted = {
        a: bool(
            adjusted[a]["passes"]
            and results[a]["lcb95"] > 0.001 + 1e-12
            and results[a]["harm"] <= 0.1
            and results[a]["positive_seeds"] == 3
        )
        for a in CANDIDATES
    }
    return dict(
        contrasts=results,
        holm=adjusted,
        holm_multiplicity=1,
        admitted=admitted,
        primary="MIX",
        verdict="CANDIDATE" if admitted["MIX"] else "NO_ADVANCE",
        bootstrap_draws=10000,
        bootstrap_seed=BOOT_SEED,
    )


def jaccard_report(families):
    old = {
        "H219_" + str(seed): folds(families, seed) for seed in (219141, 219142, 219143)
    }
    for wave in (210, 217, 218):
        for seed in range(wave * 1000 + 141, wave * 1000 + 144):
            old[f"H{wave}_{seed}"] = [
                list(g)
                for g in np.array_split(
                    np.random.default_rng(seed).permutation(sorted(set(families))), 5
                )
            ]
    return {
        str(seed + 40): {
            name: [
                [len(set(a) & set(b)) / len(set(a) | set(b)) for b in prior]
                for a in folds(families, seed + 40)
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
