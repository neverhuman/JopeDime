"""H243 pure helpers: module-wise (per tensor group) soups of r882 and r782, the matched-Frobenius random-direction
placebo, adapter distances and launch binding. The group list is frozen in groups.json before any outcome read."""

import hashlib
import math

import numpy as np
import torch

BETAS = (0.5, 1.0)
PLACEBO_SEED = 243160


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def check_groups(groups, names):
    """Groups must partition the adapter tensors exactly (every tensor in exactly one group)."""
    flat = [n for g in groups.values() for n in g]
    if sorted(flat) != sorted(names) or len(flat) != len(set(flat)):
        raise ValueError("groups must partition the adapter tensors")


def label(group, beta):
    return f"{group}@{beta:g}"


def soup(anchor, previous, recipe, groups):
    """theta = r882 with each group g in recipe replaced by (1 - beta_g) r882_g + beta_g r782_g (float64, cast back).
    Tensors outside the recipe's groups are cloned bit for bit; beta = 1 copies r782 exactly."""
    out = {n: t.clone() for n, t in anchor.items()}
    for g, beta in recipe.items():
        if not 0 < beta <= 1:
            raise ValueError("beta in (0, 1]")
        for n in groups[g]:
            a, p = anchor[n], previous[n]
            if a.shape != p.shape or a.dtype != p.dtype:
                raise ValueError("tensor mismatch: " + n)
            if beta == 1:
                out[n] = p.clone()
            else:
                out[n] = ((1 - beta) * a.double() + beta * p.double()).to(a.dtype)
    return out


def group_l2(anchor, previous, group, beta):
    return beta * math.sqrt(
        math.fsum(
            float((previous[n].double() - anchor[n].double()).square().sum())
            for n in group
        )
    )


def placebo(anchor, previous, groups, g, beta):
    """r882 with group g moved along a seeded Gaussian direction scaled to the same L2 as beta (r782_g - r882_g)."""
    target = group_l2(anchor, previous, groups[g], beta)
    rng = np.random.default_rng(stream_seed(PLACEBO_SEED, g, beta))
    draws = {
        n: torch.as_tensor(
            rng.standard_normal(tuple(anchor[n].shape)), dtype=torch.float64
        )
        for n in groups[g]
    }
    norm = math.sqrt(math.fsum(float(d.square().sum()) for d in draws.values()))
    if target <= 0 or norm <= 0:
        raise ValueError("degenerate placebo: " + g)
    out = {n: t.clone() for n, t in anchor.items()}
    for n, d in draws.items():
        out[n] = (anchor[n].double() + (target / norm) * d).to(anchor[n].dtype)
    return out


def adapter_l2(first, second):
    if set(first) != set(second):
        raise ValueError("adapter state inventory differs")
    return math.sqrt(
        math.fsum(
            float((second[n].double() - first[n].double()).square().sum())
            for n in first
        )
    )


def validate_launch(
    launch, head, manifest_sha256, prereg_sha256, fold_sha256, smoke=False
):
    if launch.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("launch manifest binding")
    if smoke:
        return
    if (
        not launch.get("merged_source_verified")
        or launch.get("source_commit") != head
        or type(launch.get("review_id")) is not int
        or launch["review_id"] <= 0
        or type(launch.get("prereg_pr")) is not int
        or launch["prereg_pr"] <= 0
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
    ):
        raise RuntimeError("external merged/reviewed launch binding")
