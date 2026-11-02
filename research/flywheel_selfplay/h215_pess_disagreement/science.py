"""H215 fixed disagreement rule, nested family partitions and paired inference."""

import hashlib
import math
import numpy as np

SEEDS = (215101, 215102, 215103)
KAPPAS = (0.0, 0.25, 0.5, 1.0, 2.0)
ARMS = ("PESS_NEST", "PESS_FIX", "PESS_SHUF", "OPT")
SELECTOR_NAMES = dict(zip(ARMS, ("CTRL", "ACC4", "ACC16", "OC15"), strict=True))


def stream(*parts):
    return int.from_bytes(
        hashlib.sha256(":".join(map(str, ("H215", *parts))).encode()).digest()[:8],
        "little",
    )


def partition(rows, seed, outer):
    families = sorted({r["family_id"] for r in rows})
    if len(families) != 205 or seed not in SEEDS or outer not in range(5):
        raise ValueError("unregistered partition")
    groups = np.array_split(np.random.default_rng(seed + 40).permutation(families), 5)
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    return tuple(
        [
            r
            for r in rows
            if (
                r["family_id"] in a
                if role == "A"
                else r["family_id"] in b
                if role == "B"
                else r["family_id"] not in a | b
            )
        ]
        for role in ("FIT", "B", "A")
    )


def inner_partition(fit, seed, outer, inner):
    families = sorted({r["family_id"] for r in fit})
    if len(families) != 123 or inner not in range(3):
        raise ValueError("123-family FIT/three inner folds required")
    groups = np.array_split(
        np.random.default_rng(stream(seed, outer, "inner")).permutation(families), 3
    )
    held = set(groups[inner])
    return [r for r in fit if r["family_id"] not in held], [
        r for r in fit if r["family_id"] in held
    ]


def member_seed(seed, outer, phase, member):
    if (
        seed not in SEEDS
        or outer not in range(5)
        or phase not in range(4)
        or member not in range(10)
    ):
        raise ValueError("unregistered member")
    return 215201 + (SEEDS.index(seed) * 5 + outer) * 40 + phase * 10 + member


def disagreement(costs):
    c = np.asarray(costs, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != 10 or c.shape[1] < 1 or not np.isfinite(c).all():
        raise ValueError("ten finite member cost vectors required")
    sd = c.std(axis=1, keepdims=True, ddof=0)
    z = (c - c.mean(axis=1, keepdims=True)) / np.maximum(sd, 1e-12)
    return (z - z[:1]).std(axis=0, ddof=0)


def pick(native, sigma, kappa):
    c, s = np.asarray(native), np.asarray(sigma, dtype=np.float64)
    if (
        c.ndim != 1
        or c.shape != s.shape
        or not np.isfinite(c).all()
        or not np.isfinite(s).all()
        or np.any(s < 0)
    ):
        raise ValueError("finite matching costs/disagreement required")
    # Preserve original dtype, ties and first argmin exactly at the null rule.
    if kappa == 0:
        return int(c.argmin())
    return int(np.argmin(c.astype(np.float64) + float(kappa) * s))


def family_mean(values, families):
    x = np.asarray(values, dtype=np.float64)
    return np.stack(
        [
            x[..., np.flatnonzero(np.asarray(families) == f)].mean(axis=-1)
            for f in sorted(set(families))
        ],
        axis=-1,
    )


def choose_kappa(gains, families):
    if set(gains) != set(KAPPAS) or len(set(families)) != 123:
        raise ValueError("complete inner-OOF grid over123 FIT families required")
    scores = {k: float(family_mean(v, families).mean()) for k, v in gains.items()}
    if not all(math.isfinite(x) for x in scores.values()):
        raise ValueError("nonfinite inner CV score")
    best = max(scores, key=lambda k: (scores[k], -k))
    return (best if scores[best] > 0 else 0.0), scores


def contrast(panel, base, families):
    delta = family_mean(np.asarray(panel) - base, families)
    draws = np.random.default_rng(215150).integers(
        0, delta.shape[1], (10000, delta.shape[1])
    )
    boot = np.median(delta[:, draws].mean(-1), axis=0)
    means = delta.mean(-1)
    return dict(
        per_seed_gain=means.tolist(),
        median_gain=float(np.median(means)),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        p=float((1 + (boot <= 0).sum()) / 10001),
        harm=float((delta.mean(0) < -0.01).mean()),
        positive_seeds=int((means > 0).sum()),
    )


def verdict(primary, adjusted_p, b_admitted, calibrated):
    return (
        "CANDIDATE"
        if (
            calibrated
            and b_admitted
            and primary["median_gain"] > 0
            and primary["ci95"][0] > 0
            and adjusted_p < 0.05
            and primary["harm"] <= 0.1
        )
        else "NO_ADVANCE"
    )


def require_native_reference(observed, expected):
    a, b = np.asarray(observed, np.float64), np.asarray(expected, np.float64)
    if (
        a.shape != (91,)
        or b.shape != (91,)
        or not np.array_equal(a, b)
        or float(a.mean()) != 1.1377941898164525
    ):
        raise RuntimeError("native91 exact utility mismatch")
