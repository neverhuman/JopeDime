"""H241 pure helpers: accepted-trajectory adapter extrapolation (ExPO-style), adapter distances, launch binding."""

import math

import torch


def extrapolate(anchor, previous, alpha, names):
    """theta(alpha) = r882 + alpha * (r882 - r_prev) on the adapter tensors only, in float64, cast back to the anchor
    dtype. alpha=0 returns the anchor bits exactly."""
    if alpha < 0:
        raise ValueError("extrapolation needs alpha >= 0")
    if set(names) - set(anchor) or set(names) - set(previous):
        raise ValueError("adapter inventory differs")
    out = {}
    for n in names:
        a, p = anchor[n], previous[n]
        if a.shape != p.shape or a.dtype != p.dtype:
            raise ValueError("adapter tensor mismatch: " + n)
        if alpha == 0:
            out[n] = a.clone()
            continue
        x = a.double() + alpha * (a.double() - p.double())
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite extrapolation: " + n)
        out[n] = x.to(a.dtype)
    return out


def adapter_l2(first, second):
    """All adapter coordinates, float64 L2 between two adapter states."""
    if set(first) != set(second):
        raise ValueError("adapter state inventory differs")
    return math.sqrt(
        math.fsum(
            float((second[n].double() - first[n].double()).square().sum())
            for n in first
        )
    )


def nonadapter_differences(anchor, previous, names):
    """Every non-adapter tensor/buffer must be bitwise identical between r882 and r_prev (H152 found this true)."""
    if set(anchor) != set(previous):
        return ["<key set differs>"]
    return sorted(
        k
        for k in anchor
        if k not in names
        and (
            anchor[k].dtype != previous[k].dtype
            or anchor[k].shape != previous[k].shape
            or anchor[k].numpy().tobytes() != previous[k].numpy().tobytes()
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
