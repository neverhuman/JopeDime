"""Frozen H197 independent family folds and FIT-only batch plans."""

import hashlib
import numpy as np

SEEDS = (197101, 197102, 197103)
FOLD_SEEDS = (197141, 197142, 197143)
BOOT_SEED = 197150
UPDATES = 40
BATCH_SIZE = 8
DRAWS = 10_000
ARMS = ("CTRL40", "PCGC40", "GCLIP40", "PCGC_SHUF40")
ALL_ARMS = ("R882",) + ARMS


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def family_folds(family_ids, fold_seed):
    families = sorted(set(family_ids))
    if len(families) != 205 or fold_seed not in FOLD_SEEDS:
        raise ValueError("H197 requires 205 families and a preregistered fold seed")
    order = np.random.default_rng(fold_seed).permutation(len(families))
    return [sorted(families[i] for i in group) for group in np.array_split(order, 5)]


def partition(rows, outer, folds):
    if len(folds) != 5 or outer not in range(5):
        raise ValueError("five cyclic folds required")
    a, b = set(folds[outer]), set(folds[(outer + 1) % 5])
    fit = set().union(
        *(set(folds[i]) for i in range(5) if i not in (outer, (outer + 1) % 5))
    )
    if fit & (a | b) or a & b or (len(fit), len(b), len(a)) != (123, 41, 41):
        raise ValueError("FIT/B/A family overlap or size mismatch")
    return (
        [r for r in rows if r["family_id"] in fit],
        [r for r in rows if r["family_id"] in b],
        [r for r in rows if r["family_id"] in a],
    )


def batch_plan(cids, orders, seed, updates=UPDATES):
    cids = sorted(cids)
    if len(cids) < BATCH_SIZE or updates != UPDATES:
        raise ValueError("H197 uses exactly 40 updates and eight FIT contexts/update")
    plan = []
    for step in range(1, updates + 1):
        stream = stream_seed(seed, step, "H197-H156-batch-v1")
        rng = np.random.default_rng(stream)
        chosen = [cids[i] for i in rng.choice(len(cids), BATCH_SIZE, replace=False)]
        indexes = []
        for cid in chosen:
            order = np.asarray(orders[cid], dtype=np.int64)
            if len(order) < 24:
                raise ValueError("H197 requires at least 24 candidates/context")
            indexes.append(
                np.concatenate((order[:4], rng.choice(order[4:], 20, replace=False)))
            )
        plan.append((stream, chosen, indexes))
    return plan


def validate_fit_plan(plan, fit_ids, b_ids, a_ids):
    fit, b, a = set(fit_ids), set(b_ids), set(a_ids)
    if fit & (b | a) or b & a:
        raise ValueError("FIT/B/A context overlap")
    if len(plan) != UPDATES:
        raise ValueError("H197 plan must contain exactly 40 updates")
    for _, chosen, indexes in plan:
        if len(chosen) != BATCH_SIZE or len(indexes) != BATCH_SIZE:
            raise ValueError("H197 batch must contain eight contexts")
        if not set(chosen) <= fit or set(chosen) & (b | a):
            raise ValueError("FIT-only context refusal")
        for cid, ix in zip(chosen, indexes, strict=True):
            ix = np.asarray(ix)
            if ix.shape != (24,) or len(set(ix.tolist())) != 24:
                raise ValueError("each slate requires 24 distinct candidates")
    return True


def fold_manifest(family_ids, source_selected_sha256=None):
    family_ids = sorted(set(family_ids))
    reps = {
        str(seed): family_folds(family_ids, fseed)
        for seed, fseed in zip(SEEDS, FOLD_SEEDS, strict=True)
    }
    pairs = {}
    for i, left in enumerate(SEEDS):
        for right in SEEDS[i + 1 :]:
            vals = []
            for lf, rf in zip(reps[str(left)], reps[str(right)], strict=True):
                left_group, right_group = set(lf), set(rf)
                vals.append(
                    len(left_group & right_group) / len(left_group | right_group)
                )
            pairs[f"{left}-{right}"] = vals
    return {
        "wave": "H197",
        "scope": "TRAIN family IDs only; no labels or protected roles",
        "source_selected_sha256": source_selected_sha256,
        "seeds": list(SEEDS),
        "fold_seeds": list(FOLD_SEEDS),
        "folds_per_seed": 5,
        "families_per_fold": [len(g) for g in reps[str(SEEDS[0])]],
        "replicates": {
            str(seed): {"fold_seed": fseed, "families": reps[str(seed)]}
            for seed, fseed in zip(SEEDS, FOLD_SEEDS, strict=True)
        },
        "pairwise_corresponding_fold_jaccard": pairs,
    }
