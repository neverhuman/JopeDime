"""Exact device gathers for repeated training slates; no scoring/math changes.

Pools use append/replacement updates. Array identity invalidates the cache after
promotion; no full slates are materialized on the CPU. Single-context inference
can retain its existing path independently.
"""

from __future__ import annotations

import fcntl
import time

import numpy as np
import torch


class DevicePoolCache:
    FIELDS = ("program", "proj16", "fitness", "gate")

    def __init__(self, device):
        self.device = torch.device(device)
        self.entries = {}
        self.transfers = 0
        self.host_payload_bytes = []

    def _get(self, cid, pool):
        sources = tuple(pool[field] for field in self.FIELDS)
        previous = self.entries.get(cid)
        if previous is not None and all(a is b for a, b in zip(previous[0], sources)):
            return previous[1]
        size = len(sources[0])
        if not size or any(len(a) != size for a in sources):
            raise ValueError("unaligned or empty device pool")
        self.host_payload_bytes.extend(value.nbytes for value in sources)
        values = {
            name: torch.as_tensor(np.ascontiguousarray(value), device=self.device)
            for name, value in zip(self.FIELDS, sources)
        }
        self.entries[cid] = (sources, values)
        self.transfers += 1
        return values

    def pack(self, pools, cids, selections):
        start = time.monotonic()
        self.host_payload_bytes = []
        if not cids or len(cids) != len(selections):
            raise ValueError("empty or mismatched slate batch")
        unique = list(dict.fromkeys(cids))
        values = {c: self._get(c, pools[c]) for c in unique}
        offsets = {}
        offset = 0
        for c in unique:
            offsets[c] = offset
            offset += len(values[c]["program"])
        indices = np.stack(selections).astype(np.int64, copy=False)
        sizes = np.asarray([len(values[c]["program"]) for c in cids], dtype=np.int64)
        if (
            indices.ndim != 2
            or (indices < 0).any()
            or (indices >= sizes[:, None]).any()
        ):
            raise ValueError("slate index outside its context pool")
        flat = indices + np.asarray([offsets[c] for c in cids], dtype=np.int64)[:, None]
        index_tensor = torch.as_tensor(flat, device=self.device)
        packed = {
            field: torch.cat([values[c][field] for c in unique], dim=0)[index_tensor]
            for field in self.FIELDS
        }
        packed["host_payload_bytes"] = [*self.host_payload_bytes, flat.nbytes]
        packed["packing_cpu_wall_seconds"] = time.monotonic() - start
        packed["cpu_index_bytes"] = flat.nbytes
        packed["avoided_cpu_feature_slate_bytes"] = (
            indices.size
            * pools[cids[0]]["program"].shape[1]
            * pools[cids[0]]["program"].dtype.itemsize
        )
        return packed


def wait_for_gpu_headroom(device, minimum_free_bytes=20 * 1024**3, pause_seconds=5):
    """DL allocation guard; release our unused cache and wait for a safe round.

    The measured batch8192 peak is about17GiB. Reserve20GiB free before starting
    another round. This cannot replace a shared lock for simultaneous launchers.
    """
    while True:
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        if minimum_free_bytes > total:
            raise RuntimeError(
                "GPU capacity is below the unchanged campaign requirement"
            )
        if free >= minimum_free_bytes:
            return
        time.sleep(pause_seconds)


def release_gpu_round(lock):
    """Finish queued GPU work and release cache before admitting another job."""
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)


def landmark_float32(features):
    """Avoid overflow before the frozen landmark engine's per-column scaling.

    Rescale only columns outside float32 range by their finite maximum magnitude.
    Raw JTT1 data and labels used by GP evaluation remain untouched.
    """
    values = np.asarray(features, dtype=np.float64)
    maxima = np.max(np.where(np.isfinite(values), np.abs(values), 0.0), axis=0)
    columns = maxima > np.finfo(np.float32).max
    if columns.any():
        values = values.copy()
        values[:, columns] /= maxima[columns]
    return np.ascontiguousarray(values, dtype=np.float32)
