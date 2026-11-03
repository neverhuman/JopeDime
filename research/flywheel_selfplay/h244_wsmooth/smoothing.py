"""Frozen H244 additive tensor-RMS Gaussian perturbations; prediction averaging, not weight averaging."""

import numpy as np
import torch

K = 8
SEEDS = (244101, 244102, 244103)
MEMBER_SEEDS = {seed: [seed * 100 + k for k in range(1, K + 1)] for seed in SEEDS}


def inventory(state, names):
    return [
        dict(
            name=n,
            shape=list(state[n].shape),
            parameters=state[n].numel(),
            rms=float(state[n].double().square().mean().sqrt()),
        )
        for n in names
    ]


def perturb(state, names, sigma, member_seed):
    if sigma not in (0.0, 0.01, 0.03, 0.1):
        raise ValueError("unregistered sigma")
    if sigma == 0:
        return {n: state[n].clone() for n in names}
    rng = np.random.default_rng(member_seed)
    result = {}
    for n in names:
        w = state[n]
        rms = float(w.double().square().mean().sqrt())
        noise = torch.as_tensor(
            rng.standard_normal(tuple(w.shape)), dtype=torch.float64
        )
        result[n] = (w.double() + sigma * rms * noise).to(w.dtype)
        if not torch.isfinite(result[n]).all():
            raise ValueError("nonfinite perturbed tensor")
    return result


def mean_cost(costs):
    if len(costs) != K + 1:
        raise ValueError("baseline plus eight members required")
    a = np.asarray(costs, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite member scores")
    return a.mean(axis=0)


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
