"""Fit-only SWAG posterior over a late AdamW trajectory; no data access."""

from collections.abc import Mapping
import math

import torch

SNAPSHOT_STEPS = tuple(range(10, 41, 2))
SELECT_STEPS = (0, 10, 20, 30, 40)
RANK = 10
SAMPLES = 16
SCALES = (0.5, 1.0)


def average_cost_vectors(cost_vectors):
    """BMA averages the cost vector first; choice is taken by the caller."""
    values = torch.as_tensor(cost_vectors, dtype=torch.float32, device="cpu")
    if (
        values.ndim != 2
        or values.shape[0] != SAMPLES
        or not torch.isfinite(values).all()
    ):
        raise ValueError("exactly 16 finite posterior cost vectors required")
    return values.mean(dim=0)


def flatten_state(state: Mapping[str, torch.Tensor], names: tuple[str, ...]):
    if set(state) != set(names):
        raise ValueError("adapter state inventory mismatch")
    pieces = [
        state[name].detach().reshape(-1).to(dtype=torch.float32, device="cpu")
        for name in names
    ]
    vector = torch.cat(pieces)
    if not torch.isfinite(vector).all():
        raise ValueError("nonfinite adapter snapshot")
    return vector


def fit_swag(snapshots: Mapping[int, Mapping[str, torch.Tensor]], names):
    """Fit SWAG using only the fixed 16 late FIT trajectory snapshots."""
    if tuple(sorted(snapshots)) != SNAPSHOT_STEPS:
        raise ValueError("SWAG requires exactly updates 10,12,...,40")
    names = tuple(names)
    matrix = torch.stack(
        [flatten_state(snapshots[step], names) for step in SNAPSHOT_STEPS]
    ).double()
    mean = matrix.mean(dim=0)
    variance = (matrix.square().mean(dim=0) - mean.square()).clamp_min(0)
    deviations = matrix[-RANK:] - mean
    if deviations.shape != (RANK, matrix.shape[1]):
        raise ValueError("rank-10 deviation inventory")
    return dict(
        names=names,
        mean=mean.float(),
        variance=variance.float(),
        deviations=deviations.float(),
        snapshot_count=len(SNAPSHOT_STEPS),
        rank=RANK,
        diagnostics=dict(
            diagonal_variance_mean=float(variance.mean()),
            diagonal_variance_median=float(variance.median()),
            diagonal_variance_p95=float(torch.quantile(variance, 0.95)),
            diagonal_variance_max=float(variance.max()),
            diagonal_variance_trace=float(variance.sum()),
        ),
    )


def sample_vectors(swag, seed: int, scales=SCALES):
    """Return paired deterministic posterior vectors; scale multiplies covariance."""
    if swag["rank"] != RANK or swag["snapshot_count"] != 16:
        raise ValueError("unrecognized SWAG fit")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    z_diag = torch.randn((SAMPLES, swag["mean"].numel()), generator=gen)
    z_rank = torch.randn((SAMPLES, RANK), generator=gen)
    outputs = {}
    for scale in scales:
        if scale not in SCALES:
            raise ValueError("fixed SWAG covariance scale only")
        diagonal = z_diag * swag["variance"].sqrt()
        low_rank = z_rank @ swag["deviations"]
        outputs[scale] = (
            swag["mean"]
            + math.sqrt(scale / 2) * diagonal
            + math.sqrt(scale / (2 * (RANK - 1))) * low_rank
        )
    return outputs


def unflatten_state(vector, template, names):
    vector = vector.detach().to(dtype=torch.float32, device="cpu")
    if set(template) != set(names):
        raise ValueError("adapter state inventory mismatch")
    state, cursor = {}, 0
    for name in names:
        old = template[name]
        count = old.numel()
        state[name] = vector[cursor : cursor + count].reshape(old.shape).clone()
        cursor += count
    if cursor != vector.numel() or not torch.isfinite(vector).all():
        raise ValueError("SWAG vector length or finite check")
    return state


def validate_fit_only(fit_ids, trajectory_ids, select_ids, assess_ids, val_ids):
    fit, trajectory = set(fit_ids), set(trajectory_ids)
    forbidden = set(select_ids) | set(assess_ids) | set(val_ids)
    if (
        not trajectory
        or not trajectory <= fit
        or trajectory & forbidden
        or fit & forbidden
    ):
        raise ValueError("SWAG trajectory is not FIT-only")
    return True


def gradient_norm(params):
    grads = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(grads))) if grads else 0.0


def adapter_drift(initial, final):
    if set(initial) != set(final):
        raise ValueError("adapter state inventory differs")
    return math.sqrt(
        math.fsum(
            float((final[n].double() - initial[n].double()).square().sum())
            for n in initial
        )
    )


def validate_launch(
    launch, head, manifest_sha256, prereg_sha256, fold_sha256, smoke=False
):
    if launch.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("manifest binding")
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
