"""H131 fixed family bootstrap, parameter averaging and paired-family readout."""

from collections import Counter
import hashlib
import json
import random

import numpy as np
import torch

ARMS = ("CONTROL", "SOUP8", "SOUP8_HALF")
SEEDS = (97, 98, 99)
MEMBERS = tuple(range(1, 9))


def family_split(rows):
    families = sorted({r["family_id"] for r in rows})
    if len(families) != 205 or len(rows) != 1219:
        raise ValueError("full registered DL-Train roster required")
    folds = [
        list(x)
        for x in np.array_split(np.random.default_rng(131097).permutation(families), 5)
    ]
    return dict(
        fold_seed=131097,
        folds=folds,
        assessment_families=folds[0],
        fit_families=sorted(set(families) - set(folds[0])),
    )


def bootstrap_contexts(rows, rng_seed):
    """Resample whole families; preserve all repeated-family context multiplicities."""
    by_family = {}
    for row in rows:
        by_family.setdefault(row["family_id"], []).append(row["context_sha256"])
    families = sorted(by_family)
    if not families or len({r["context_sha256"] for r in rows}) != len(rows):
        raise ValueError("nonempty unique context roster required")
    draws = (
        np.random.default_rng(rng_seed)
        .choice(families, len(families), replace=True)
        .tolist()
    )
    contexts = sorted(c for family in draws for c in sorted(by_family[family]))
    return dict(
        rng_seed=rng_seed,
        family_draws=draws,
        family_multiplicity=dict(Counter(draws)),
        contexts=contexts,
        context_count=len(contexts),
    )


def bootstrap_identity(value):
    return dict(
        rng_seed=value["rng_seed"],
        context_count=value["context_count"],
        unique_families=len(value["family_multiplicity"]),
        sha256=hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def update_plan(pools, sampler, seed, batch_type, expanded_ids=None):
    """Exact H76/H125 first two streams per round; optionally use a bootstrap multiset."""
    ids = tuple(sorted(pools) if expanded_ids is None else sorted(expanded_ids))
    if not ids or not set(ids).issubset(pools):
        raise ValueError("all sampling contexts must belong to fit pools")
    orders = {c: np.argsort(p["fitness"]) for c, p in pools.items()}
    state = np.random.get_state()
    plan = []
    try:
        for rnd in range(1, 21):
            key = seed * 1_000_003 + 10_007
            np.random.seed((key + rnd * 104729) % (2**32))
            rotated = ids[rnd % len(ids) :] + ids[: rnd % len(ids)]
            for step in range(8):
                stream = int.from_bytes(
                    hashlib.sha256(
                        f"flywheel-batch/v1:{key}:{rnd}:{step}".encode()
                    ).digest()[:8],
                    "little",
                )
                cids = tuple(random.Random(stream).choices(rotated, k=16))
                ix = tuple(
                    np.asarray(sampler(pools[c], orders[c], 192), dtype=np.int64).copy()
                    for c in cids
                )
                if step < 2:
                    plan.append(batch_type(rnd, step, stream, cids, ix))
    finally:
        np.random.set_state(state)
    return plan


def assert_plan_equal(actual, reference):
    if len(actual) != 40 or len(reference) != 40:
        raise ValueError("exactly40 updates required")
    for a, r in zip(actual, reference, strict=True):
        b = r[0]
        if (
            a.seed != b.seed
            or a.round_index != b.round_index
            or a.context_ids != b.context_ids
            or any(
                not np.array_equal(x, y)
                for x, y in zip(a.candidate_indices, b.candidate_indices, strict=True)
            )
        ):
            raise ValueError("CONTROL differs from H125 recipe")


def average_adapters(initial, members):
    if len(members) != 8:
        raise ValueError("all eight members required; no survivor average")
    result = {}
    for name, tensor in initial.items():
        if any(
            set(m) != set(initial)
            or m[name].shape != tensor.shape
            or m[name].dtype != tensor.dtype
            for m in members
        ):
            raise ValueError("adapter inventories differ")
        if any(not torch.isfinite(m[name]).all() for m in members):
            raise ValueError("nonfinite member")
        # Deterministic member order, float64 accumulation, one cast to original dtype.
        result[name] = (
            torch.stack([m[name].double() for m in members]).mean(0).to(tensor.dtype)
        )
    half = {
        n: (0.5 * initial[n].double() + 0.5 * v.double()).to(initial[n].dtype)
        for n, v in result.items()
    }
    return result, half


def elo(values):
    p = np.mean((values > 0) + 0.5 * (values == 0), axis=-1)
    with np.errstate(divide="ignore"):
        return 1500 + 400 * np.log10(p / (1 - p))


def scalar(value):
    value = float(value)
    return value if np.isfinite(value) else ("+infinity" if value > 0 else "-infinity")


def summarize(reference, cells):
    if len(cells) != 9 or {(x["arm"], x["seed"]) for x in cells} != {
        (a, s) for a in ARMS for s in SEEDS
    }:
        raise ValueError("all nine final cells required")
    families = sorted(reference["per_family"])
    contexts = sorted(reference["per_context"])
    if len(families) != 41:
        raise ValueError("41 assessment families required")
    draws = np.random.default_rng(131000).integers(0, 41, size=(10000, 41))
    result, bootstraps = {}, {}
    for arm in ARMS:
        group = sorted((x for x in cells if x["arm"] == arm), key=lambda x: x["seed"])
        if any(
            sorted(x["assessment"]["per_family"]) != families
            or sorted(x["assessment"]["per_context"]) != contexts
            for x in group
        ):
            raise ValueError("incomplete assessment roster")
        gains = np.array(
            [
                [
                    x["assessment"]["per_family"][f] - reference["per_family"][f]
                    for f in families
                ]
                for x in group
            ]
        )
        if not np.isfinite(gains).all():
            raise ValueError("nonfinite utility gain")
        boot = np.median(gains[:, draws].mean(2), axis=0)
        bootstraps[arm] = boot
        ev = elo(gains)
        eb = np.median(elo(gains[:, draws]), axis=0)
        result[arm] = dict(
            median_gain=float(np.median(gains.mean(1))),
            seed_gains=gains.mean(1).tolist(),
            family_bootstrap_95_ci=np.quantile(boot, [0.025, 0.975]).tolist(),
            ordinary_one_sided_95_lcb=float(np.quantile(boot, 0.05)),
            bootstrap_lower_tail_area=float((1 + (boot <= 0).sum()) / 10001),
            seed_utilities=[x["assessment"]["mean"] for x in group],
            family_harm_fraction=float((gains.mean(0) < 0).mean()),
            seed_family_harm_fractions=(gains < 0).mean(1).tolist(),
            family_wins=(gains > 0).sum(1).tolist(),
            family_ties=(gains == 0).sum(1).tolist(),
            family_losses=(gains < 0).sum(1).tolist(),
            median_elo=scalar(np.median(ev)),
            seed_elo=[scalar(v) for v in ev],
            elo_95_ci=[
                scalar(v)
                for v in np.quantile(eb, [0.025, 0.975], method="inverted_cdf")
            ],
            identical_seed_gain_vectors=bool(
                np.array_equal(gains[0], gains[1])
                and np.array_equal(gains[0], gains[2])
            ),
            identical_to_reference_utilities=bool(np.all(gains == 0)),
            verdict="NO_ADVANCE",
        )
    alive = True
    for i, arm in enumerate(
        sorted(ARMS, key=lambda a: (result[a]["bootstrap_lower_tail_area"], a))
    ):
        x = result[arm]
        threshold = 0.05 / (3 - i)
        lower = float(np.quantile(bootstraps[arm], threshold))
        alive = bool(
            alive and x["bootstrap_lower_tail_area"] <= threshold and lower > 0
        )
        x.update(holm_threshold=threshold, holm_lcb95=lower, holm_pass=alive)
        if (
            alive
            and x["median_gain"] > 0.001
            and min(x["seed_gains"]) > 0
            and x["family_harm_fraction"] <= 0.1
            and not x["identical_to_reference_utilities"]
        ):
            x["verdict"] = "ADMIT"
    return dict(
        status="COMPLETE",
        arms=result,
        reference_utility=reference["mean"],
        assessment_families=41,
        assessment_contexts=len(contexts),
        harm_definition="fraction of families with strictly negative seed-mean gain; no tolerance",
        interpretation="Additional fine-tuning of an already DL-Train-exposed r882; no unseen-family or serving qualification.",
    )
