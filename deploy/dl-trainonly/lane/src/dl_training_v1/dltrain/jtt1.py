"""PLAN §5 P6 JTT1 rule, shared by P5 (s_c, train-only leak recompute) and P6 (A0 meta-features, train-row hashes).

Layout (research_2k/prepare_fold0.py, per_fold_hwm/build_campaign.py, incoming_intake_v1/vendor/prepare_fold0.py):
    b"JTT1" + struct.pack("<IIIB", n_train, n_test, p, inner_folds)        17 bytes
    inner-fold label per outer-train row, u8                               n_train bytes
    float64 LE matrix, row-major, (n_train + n_test) rows x (p + 1) columns, outer-train rows first, last column = target

The rule: assert the magic is JTT1 (JTT2 is refused) and len(raw) == 17 + n_train + 8*(p+1)*(n_train+n_test); decode only
[17 + n_train, 17 + n_train + 8*(p+1)*n_train); never decode at or beyond that offset (outer-test rows and targets).
`input_raw_sha256` hashes the full raw input for identity only. `train_view` slices the train range before any parse, so
the test range is never handed to a decoder.

Fetches are exact-VersionId GETs through dltrain.s3io; the gz body must match input_ref.sha256/bytes and the raw body
input_ref.raw_sha256/raw_bytes. The local cache (RUN/inputs/jtt1/<raw_sha256>.jtt1.gz) is create-only and re-verified on
every read.
"""
import gzip
import hashlib
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

from dltrain import common

MAGIC = b"JTT1"
HEADER = struct.Struct("<IIIB")
HEADER_BYTES = 4 + HEADER.size  # 17


class JTTError(ValueError):
    pass


def parse_header(raw):
    if len(raw) < HEADER_BYTES:
        raise JTTError("truncated header")
    magic = bytes(raw[:4])
    if magic != MAGIC:
        raise JTTError(f"magic {magic!r} refused (only JTT1 is admitted)")
    n_train, n_test, p, inner = HEADER.unpack(bytes(raw[4:HEADER_BYTES]))
    expected = HEADER_BYTES + n_train + 8 * (p + 1) * (n_train + n_test)
    if len(raw) != expected:
        raise JTTError(f"length {len(raw)} != {expected} for n_train={n_train} n_test={n_test} p={p}")
    if n_train < 1 or p < 1:
        raise JTTError(f"degenerate shape n_train={n_train} p={p}")
    return {"n_train": n_train, "n_test": n_test, "p": p, "inner_folds": inner}


def train_range(header):
    start = HEADER_BYTES + header["n_train"]
    return start, start + 8 * (header["p"] + 1) * header["n_train"]


def train_view(raw):
    """Return (header, inner_fold_labels u8[n_train], X f64[n_train, p], y f64[n_train]) from the train bytes only."""
    h = parse_header(raw)
    mv = memoryview(raw)
    labels = np.frombuffer(mv[HEADER_BYTES:HEADER_BYTES + h["n_train"]], dtype=np.uint8).copy()
    start, stop = train_range(h)
    block = np.frombuffer(mv[start:stop], dtype="<f8").reshape(h["n_train"], h["p"] + 1).astype(np.float64, copy=True)
    return h, labels, block[:, :-1], block[:, -1]


def train_rows_sha256(raw):
    """sha256 of exactly the decodable train bytes (labels + train block); a counterfactual test-row change leaves it unchanged."""
    h = parse_header(raw)
    start, stop = train_range(h)
    return hashlib.sha256(bytes(memoryview(raw)[HEADER_BYTES:stop])).hexdigest()


def check_raw(raw, ref):
    if common.sha256_bytes(raw) != ref["raw_sha256"] or len(raw) != int(ref["raw_bytes"]):
        raise JTTError("raw sha256/bytes differ from input_ref")


def cache_path(cache_dir, ref):
    return Path(cache_dir) / f"{ref['raw_sha256']}.jtt1.gz"


def _atomic_create(path, data):
    """Create-only and atomic: write a same-directory temp file, fsync, then link it into place. A concurrent reader sees
    either no file or the complete file; an existing file must hold equal bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o444)
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise JTTError(f"{path} exists with different bytes")
    finally:
        os.unlink(tmp)


def load_raw(ref, cache_dir, client=None, ledger=None):
    """Return the verified raw JTT1 bytes for an epoch context's input_ref, from the create-only cache or one exact GET."""
    from dltrain import quarantine
    quarantine.assert_not_quarantined(ref.get("key") if isinstance(ref, dict) else None)
    path = cache_path(cache_dir, ref)
    if path.exists():
        gz = path.read_bytes()
        if common.sha256_bytes(gz) != ref["sha256"] or len(gz) != int(ref["bytes"]):
            raise JTTError(f"cached gz differs from input_ref: {path}")
    else:
        if client is None:
            raise JTTError(f"not cached and no client: {path.name}")
        from dltrain import s3io
        gz, _ = s3io.get_exact(client, ref["key"], ref["version_id"], bucket=ref["bucket"])
        if ledger is not None:
            ledger["gets"] = ledger.get("gets", 0) + 1
            ledger["bytes"] = ledger.get("bytes", 0) + len(gz)
        if common.sha256_bytes(gz) != ref["sha256"] or len(gz) != int(ref["bytes"]):
            raise JTTError(f"S3 body differs from input_ref: {ref['key']}")
        _atomic_create(path, gz)
    raw = gzip.decompress(gz)
    check_raw(raw, ref)
    return raw
