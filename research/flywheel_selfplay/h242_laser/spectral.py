"""H242 pure helpers: energy-fraction SVD rank reduction of 2-D adapter matrices (LASER-style), the matched-Frobenius
random-projection placebo, the frozen spectral inventory, adapter distances and launch binding."""

import hashlib
import math

import numpy as np
import torch

RHOS = (0.99, 0.95, 0.90)
SINGLE_RHO = 0.90
PLACEBO_SEED = 242160


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def matrices(state, names):
    """The 2-D adapter weight matrices, in adapter order; 1-D and 3-D tensors (biases, norms, scalars) are untouched."""
    return [n for n in names if state[n].dim() == 2]


def kept_rank(sigma, rho):
    """Smallest k whose top-k singular values hold at least rho of sum(sigma^2) (float64)."""
    energy = np.cumsum(np.asarray(sigma, dtype=np.float64) ** 2)
    if not 0 < rho <= 1 or energy[-1] <= 0:
        raise ValueError("rho in (0, 1] and a nonzero matrix required")
    if rho == 1:
        return len(energy)
    return int(np.searchsorted(energy, rho * energy[-1], side="left")) + 1


def truncate(w, rho):
    """Top-k SVD reconstruction in float64, cast back; returns (matrix, k, full rank). k == full rank -> exact bits."""
    if w.dim() != 2:
        raise ValueError("2-D matrix required")
    u, s, vh = torch.linalg.svd(w.double(), full_matrices=False)
    full = s.numel()
    k = kept_rank(s.numpy(), rho)
    if k >= full:
        return w.clone(), full, full
    out = (u[:, :k] * s[:k]) @ vh[:k]
    return out.to(w.dtype), k, full


def placebo(w, rho, name):
    """Remove a seeded random orthogonal direction set of the same size (full - k) from the row space, scaled so the
    Frobenius change equals the rank-k truncation's. k == full rank -> exact bits."""
    truncated, k, full = truncate(w, rho)
    d = full - k
    if d == 0:
        return w.clone()
    target = float(torch.linalg.matrix_norm(w.double() - truncated.double()))
    rng = np.random.default_rng(stream_seed(PLACEBO_SEED, name, rho))
    q, _ = np.linalg.qr(rng.standard_normal((w.shape[1], d)))
    q = torch.as_tensor(q, dtype=torch.float64)
    delta = w.double() @ q @ q.T
    norm = float(torch.linalg.matrix_norm(delta))
    if norm <= 0:
        raise ValueError("degenerate placebo direction: " + name)
    return (w.double() - (target / norm) * delta).to(w.dtype)


def reduce_all(state, names, rho, which=None, fn="svd"):
    """New adapter state: every 2-D matrix (or only `which`) replaced; every other tensor cloned bit for bit."""
    targets = set(matrices(state, names) if which is None else which)
    out = {}
    for n in names:
        if n not in targets:
            out[n] = state[n].clone()
        elif fn == "svd":
            out[n] = truncate(state[n], rho)[0]
        elif fn == "placebo":
            out[n] = placebo(state[n], rho, n)
        else:
            raise ValueError("unknown reduction")
        if not torch.isfinite(out[n]).all():
            raise ValueError("nonfinite reduction: " + n)
    return out


def inventory(state, names):
    """Frozen-before-outcomes spectral census: shape, full rank, kept rank and Frobenius change per rho."""
    rows = []
    for n in matrices(state, names):
        w = state[n]
        s = torch.linalg.svdvals(w.double()).numpy()
        row = dict(
            name=n,
            shape=list(w.shape),
            full_rank=len(s),
            frobenius=float(np.sqrt((s**2).sum())),
        )
        for rho in RHOS:
            k = kept_rank(s, rho)
            row[f"k_{rho:g}"] = k
            row[f"removed_energy_{rho:g}"] = float((s[k:] ** 2).sum() / (s**2).sum())
            row[f"frobenius_change_{rho:g}"] = float(np.sqrt((s[k:] ** 2).sum()))
        rows.append(row)
    return rows


def singles(rows):
    """LASER_SINGLE candidates: matrices whose rho=.90 truncation changes them (k < full rank), in adapter order."""
    return [r["name"] for r in rows if r[f"k_{SINGLE_RHO:g}"] < r["full_rank"]]


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
