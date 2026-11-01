"""Lossless XOR encoding of full float32 costs against immutable r882 costs."""

import os
from pathlib import Path

import numpy as np


def encode(cost, reference):
    cost = np.asarray(cost, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if (
        cost.shape != reference.shape
        or not np.isfinite(cost).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("matching finite full-pool cost vectors required")
    return np.bitwise_xor(cost.view(np.uint32), reference.view(np.uint32))


def decode(encoded, reference):
    reference = np.asarray(reference, dtype=np.float32)
    return np.bitwise_xor(
        np.asarray(encoded, dtype=np.uint32), reference.view(np.uint32)
    ).view(np.float32)


def save_npz(path, **arrays):
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())


def save_scores(path, cids, costs, reference, utilities=None, choices=None):
    offsets = np.r_[0, np.cumsum([len(x) for x in costs])].astype(np.int64)
    arrays = dict(
        context_ids=np.asarray(cids),
        offsets=offsets,
        costs_xor_u32=np.concatenate(
            [encode(v, reference[c]) for c, v in zip(cids, costs, strict=True)]
        ),
    )
    if utilities is not None:
        arrays["utilities"] = np.asarray(utilities, dtype=np.float64)
    if choices is not None:
        arrays["choices"] = np.asarray(choices, dtype=np.int32)
    save_npz(path, **arrays)


def save_state(path, state):
    save_npz(path, **{k: v.detach().cpu().numpy() for k, v in state.items()})


def load_state(path):
    import torch

    with np.load(path, allow_pickle=False) as z:
        return {k: torch.from_numpy(z[k].copy()) for k in z.files}
