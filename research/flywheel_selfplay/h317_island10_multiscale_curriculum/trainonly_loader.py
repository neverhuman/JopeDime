# Vendored verbatim (below this header) for H317 from neverhuman/JopeDime origin/v1/next@66c8b4b49b5183fbdbe594abe62b9a22e6e9db71
#   research/flywheel_selfplay/h308_ensvote2_trainonly_repartition/trainonly_loader.py (H308 TRAIN-prefix pread loader).
# H317 use: open the mixed H58 npz files only inside common_h317.mixed_prefix_read().
"""H308 clean-room TRAIN-only loader for the two mixed H58 cache files (H305 audit (h)).

`~/jope-data/h58/cache/{pool_priors,context_features}.npz` hold 1,219 DL-Train contexts followed by 91 DL-Val (dev91)
contexts. `npz[key][i]` decodes the whole member, DL-Val rows included. This loader:
  1. reads only the `cids` (and `offsets`) members, which are identifiers / row offsets, not features;
  2. verifies that the first N cids are exactly the N TRAIN cids of the pinned selected.json and that no later cid is a
     TRAIN cid (the TRAIN rows form a verified prefix);
  3. reads, for every feature member, ONLY the byte range of that prefix with os.pread on the uncompressed (ZIP_STORED)
     .npy payload. No byte of a DL-Val feature row is read. The zip CRC is not checked (it covers the whole member);
     a full-member read through zipfile would raise on the poisoned fixture in test_h308.py, this loader does not.
`run_h303.setup()` indexes rows i < N only, so it gets byte-identical TRAIN values."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np

MIXED = {"pool_priors.npz", "context_features.npz"}
ID_MEMBERS = {"cids", "offsets"}
COUNTERS = {"feature_bytes_read": 0, "feature_rows_read": 0, "dlval_feature_rows_read": 0}


def _member_layout(path: Path, name: str):
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"{path}:{name} is compressed; a prefix read is impossible")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        if local[:4] != b"PK\x03\x04":
            raise RuntimeError("bad local zip header")
        n, m = int.from_bytes(local[26:28], "little"), int.from_bytes(local[28:30], "little")
        f.seek(info.header_offset + 30 + n + m)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        data_off = f.tell()
    if fortran or dtype.hasobject:
        raise RuntimeError(f"{path}:{name} unsupported layout")
    return data_off, tuple(shape), dtype


def read_rows(path: Path, name: str, n_rows: int) -> np.ndarray:
    """Rows [0, n_rows) of an uncompressed .npy member, reading exactly that byte range."""
    data_off, shape, dtype = _member_layout(path, name)
    if n_rows > shape[0]:
        raise ValueError("prefix longer than array")
    row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize if len(shape) > 1 else dtype.itemsize
    nbytes = n_rows * row_bytes
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = os.pread(fd, nbytes, data_off)
    finally:
        os.close(fd)
    if len(buf) != nbytes:
        raise RuntimeError("short read")
    return np.frombuffer(buf, dtype=dtype).reshape((n_rows,) + shape[1:]).copy()


def read_all(path: Path, name: str) -> np.ndarray:
    data_off, shape, dtype = _member_layout(path, name)
    return read_rows(path, name, shape[0])


class TrainOnlyNpz:
    """Mapping view of a mixed cache npz returning ids in full and feature members as the verified TRAIN prefix only."""

    def __init__(self, path: Path, train_cids: set[str]):
        self.path = Path(path)
        cids = [str(c) for c in read_all(self.path, "cids.npy")]
        n = len(train_cids)
        if len(set(cids[:n])) != n or set(cids[:n]) != set(train_cids) or any(c in train_cids for c in cids[n:]):
            raise RuntimeError(f"{self.path}: TRAIN cids are not the leading prefix")
        self.n_train, self.n_total = n, len(cids)
        self._ids = {"cids": np.asarray(cids)}
        names = {i.filename[:-4] for i in zipfile.ZipFile(self.path).infolist()}
        if "offsets" in names:
            off = read_all(self.path, "offsets.npy")
            self._ids["offsets"] = off
            self.row_prefix = int(off[n])  # ragged: rows [0, offsets[n]) belong to the TRAIN contexts
        else:
            self.row_prefix = n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getitem__(self, key: str) -> np.ndarray:
        if key in ID_MEMBERS:
            return self._ids[key]
        arr = read_rows(self.path, key + ".npy", self.row_prefix)
        COUNTERS["feature_bytes_read"] += arr.nbytes
        COUNTERS["feature_rows_read"] += arr.shape[0]
        return arr


class TrainOnlyNumpy:
    """Stands in for `np` inside run_h303 so its setup() opens the two mixed files through TrainOnlyNpz."""

    def __init__(self, train_cids: set[str]):
        self.train_cids = set(train_cids)

    def __getattr__(self, name):
        return getattr(np, name)

    def load(self, file, *args, **kw):
        if Path(file).name in MIXED:
            return TrainOnlyNpz(Path(file), self.train_cids)
        return np.load(file, *args, **kw)


class GuardedPackets(dict):
    """Packets dict that refuses access to forbidden (outer-test) contexts: enforces 0 outer-test uses before a lock."""

    def __init__(self, base: dict, forbidden: set[str]):
        super().__init__(base)
        self.forbidden = set(forbidden)
        self.denied = 0

    def __getitem__(self, key):
        if key in self.forbidden:
            self.denied += 1
            raise KeyError(f"outer-test context {key[:12]} accessed before fit_lock")
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default
