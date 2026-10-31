"""H90 fixed recipe deltas; CPU helpers have no dataset or GPU entrypoint."""

from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
import random

import numpy as np
import torch

ARMS = ("CONTROL", "FBGA", "LS", "WCR")
SEEDS = (27, 28, 29)


def learning_rate(round_index):
    if not 1 <= round_index <= 20:
        raise ValueError("round must be1..20")
    if round_index <= 2:
        return 2e-4 * round_index / 2
    phase = (round_index - 2) / 8 if round_index <= 10 else (round_index - 11) / 9
    return 1e-5 + (2e-4 - 1e-5) * (1 + math.cos(math.pi * phase)) / 2


def recipe(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return {
        "arm": arm,
        "rounds": 20,
        "updates_per_round": 2,
        "batch": 16,
        "slate": 192,
        "weight_decay": 1e-4,
        "clip": 1.0,
        "loss": "H59_CONTROL",
        "smoothing_epsilon": 0.1 if arm == "LS" else 0.0,
        "sampling": "inverse_family_size" if arm == "FBGA" else "H76_uniform",
        "microbatch": 8 if arm == "FBGA" else 16,
        "learning_rates": [
            learning_rate(r) if arm == "WCR" else 1e-4 for r in range(1, 21)
        ],
    }


def recipe_hash(arm):
    return hashlib.sha256(
        json.dumps(
            recipe(arm), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def family_weights(pools):
    if not pools:
        raise ValueError("empty train roster")
    families = {
        cid: pool.get("rec", {}).get("family_id") for cid, pool in pools.items()
    }
    if any(not isinstance(f, str) or not f for f in families.values()):
        raise ValueError("missing training family identity")
    sizes = Counter(families.values())
    return {cid: 1.0 / sizes[families[cid]] for cid in sorted(pools)}


def family_batch_plan(pools, sampler, seed, science):
    """H76 keyed streams/candidate sampler, with context weights as the only draw delta."""
    ids = tuple(sorted(pools))
    weights = family_weights(pools)
    orders = {cid: np.argsort(pool["fitness"]) for cid, pool in pools.items()}
    state = np.random.get_state()
    plan = []
    try:
        for rnd in range(1, 21):
            key = seed * 1_000_003 + 10_007
            np.random.seed((key + rnd * 104729) % 2**32)
            rotated = ids[rnd % len(ids) :] + ids[: rnd % len(ids)]
            for step in range(8):
                stream = int.from_bytes(
                    hashlib.sha256(
                        f"flywheel-batch/v1:{key}:{rnd}:{step}".encode()
                    ).digest()[:8],
                    "little",
                )
                cids = tuple(
                    random.Random(stream).choices(
                        rotated, weights=[weights[c] for c in rotated], k=16
                    )
                )
                ix = tuple(
                    np.asarray(sampler(pools[c], orders[c], 192), dtype=np.int64).copy()
                    for c in cids
                )
                if any(x.shape != (192,) for x in ix):
                    raise RuntimeError("incorrect candidate slate")
                plan.append(science.h59.TrainingBatch(rnd, step, stream, cids, ix))
    finally:
        np.random.set_state(state)
    return plan


def smooth_targets(targets):
    if (
        targets.ndim != 2
        or targets.shape[1] != 192
        or not torch.isfinite(targets).all()
    ):
        raise ValueError("finite batch-by192 targets required")
    values = targets.detach()
    return 0.9 * values + 0.1 * values.mean(dim=1, keepdim=True)


def accumulated_step(named, optimizer, batch, predict, target_fn, loss_fn):
    """Two8-context backwards, one final clip and one AdamW/decay application."""
    if len(batch.context_ids) != 16 or len(batch.candidate_indices) != 16:
        raise ValueError("FBGA requires16 contexts")
    params = [p for _, p in named if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for start in (0, 8):
        micro = replace(
            batch,
            context_ids=batch.context_ids[start : start + 8],
            candidate_indices=batch.candidate_indices[start : start + 8],
        )
        prediction = predict(list(micro.context_ids), list(micro.candidate_indices))
        loss = loss_fn("CONTROL", prediction, target_fn(micro, prediction.device))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite microbatch loss")
        (loss * 0.5).backward()
        losses.append(float(loss.detach()))
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return sum(losses) / 2, float(norm)


def adapter_l2(initial, final):
    if set(initial) != set(final) or not initial:
        raise ValueError("adapter inventory mismatch")
    squared = 0.0
    for name in sorted(initial):
        if initial[name].shape != final[name].shape:
            raise ValueError("adapter shape mismatch")
        delta = (
            final[name].detach().cpu().double() - initial[name].detach().cpu().double()
        )
        if not torch.isfinite(delta).all():
            raise ValueError("nonfinite adapter distance")
        squared += float(delta.square().sum())
    return math.sqrt(squared)


def top_four(pools, scores):
    if set(pools) != set(scores):
        raise ValueError("score/context roster mismatch")
    selected = {}
    for cid in sorted(pools):
        values = np.asarray(scores[cid], dtype=np.float64)
        pids = pools[cid]["pids"]
        if (
            values.ndim != 1
            or len(values) != len(pids)
            or len(values) < 8
            or not np.isfinite(values).all()
        ):
            raise ValueError("four finite unrevealed candidates required")
        ix = 4 + np.argsort(values[4:], kind="stable")[:4]
        selected[cid] = [pids[int(i)] for i in ix]
    return selected


def overlap(candidate, reference):
    if not candidate or set(candidate) != set(reference):
        raise ValueError("top4 roster mismatch")
    values = []
    for cid in sorted(candidate):
        if len(candidate[cid]) != 4 or len(reference[cid]) != 4:
            raise ValueError("exactly four selected identities required")
        values.append(
            sum((Counter(candidate[cid]) & Counter(reference[cid])).values()) / 4
        )
    return float(np.mean(values))


def pseudoreplication(contexts, families, scalars):
    def groups(values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape[0] != 3 or not np.isfinite(values).all():
            raise ValueError("three complete finite seed vectors required")
        result = []
        for i, seed in enumerate(SEEDS):
            for group in result:
                if np.array_equal(values[i], values[SEEDS.index(group[0])]):
                    group.append(seed)
                    break
            else:
                result.append([seed])
        return result

    result = {
        "context_utility_groups": groups(contexts),
        "family_utility_groups": groups(families),
        "scalar_utility_groups": groups(scalars),
    }
    result["identical_utility_flag"] = any(
        len(g) > 1 for grouping in result.values() for g in grouping
    )
    result["independent_seed_replicates_claimed"] = 0
    result["interval_unit"] = "19 paired dev families; complete seed vectors retained"
    return result
