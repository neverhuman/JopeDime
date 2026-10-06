"""PAYLOAD-STRIP-PUBLISH scorer-only outer-test artifact (JTS1). The ONLY module that reads or writes it.

Owner ruling (ENG:watcher delegated, 2026-10-06 06:52 MT; owner-decisions-20261006T125305Z.json): published encoder and
teacher payloads hold no outer-test target values; the outer-test rows move to a separate scorer-only artifact with its own
sha256, read only by the outer-test scorer adapter, never by encoder, teacher, selector or scaler code. This module is that
adapter's reader. tests/test_payload_strip.py statically refuses any other dltrain/ or tools/ module (except the strip tool,
tools/v2_strip_outer_test.py, which writes it) that imports this module or names the artifact.

Layout (raw; stored gzip with mtime 0 as <payload stem>.scorer-outer-test.jts1.gz, e.g. encoder.scorer-outer-test.jts1.gz):
    b"JTS1" + u32 LE H + H bytes canonical JSON header + float64 LE (n_test x (p + 1)), row-major, last column = target
The f64 block is byte-for-byte the outer-test block of the original JTT1 payload (rows in their original order). The magic
is not JTT1, so jtt1.parse_header (and every JTT1 decoder) refuses it. The header binds the artifact to its payload:
view_id, cut_id, payload name, original and stripped raw/gz sha256, n_train, train_rows_sha256 and test_block_sha256.

LOCAL ONLY: it carries outer-test targets, so it never goes to S3 and never sits inside a publish tree.
"""
import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"JTS1"
SCHEMA = "jope.dl-scorer-outer-test/v1"
SUFFIX = ".scorer-outer-test.jts1.gz"
LEN = struct.Struct("<I")
PREFIX_BYTES = 4 + LEN.size
HEADER_FIELDS = ("schema", "payload", "view_id", "cut_id", "dataset_hash", "fold", "n_train", "n_test", "p", "inner_folds",
                 "original_raw_sha256", "original_gz_sha256", "stripped_raw_sha256", "stripped_gz_sha256", "train_rows_sha256",
                 "test_block_sha256")
JTT1_MAGIC, JTT1_HEADER = b"JTT1", struct.Struct("<IIIB")


class ScorerError(ValueError):
    pass


def _sha(b) -> str:
    return hashlib.sha256(b).hexdigest()


def artifact_name(payload_name: str) -> str:
    """encoder.jtt1.gz -> encoder.scorer-outer-test.jts1.gz"""
    if not payload_name.endswith(".jtt1.gz"):
        raise ScorerError(f"not a JTT1 payload name: {payload_name!r}")
    return payload_name[: -len(".jtt1.gz")] + SUFFIX


def encode(header: dict, test_block: bytes) -> bytes:
    """Raw JTS1 bytes for one payload's outer-test block. `header` carries every HEADER_FIELDS entry except schema and
    test_block_sha256, which are set here."""
    h = dict(header, schema=SCHEMA, test_block_sha256=_sha(test_block))
    if set(h) != set(HEADER_FIELDS):
        raise ScorerError(f"header fields {sorted(set(h) ^ set(HEADER_FIELDS))} missing or unknown")
    if len(test_block) != 8 * (h["p"] + 1) * h["n_test"]:
        raise ScorerError("test block length differs from n_test x (p + 1) float64")
    body = json.dumps(h, sort_keys=True, separators=(",", ":")).encode()
    return MAGIC + LEN.pack(len(body)) + body + bytes(test_block)


def gz(raw: bytes) -> bytes:
    return gzip.compress(raw, compresslevel=9, mtime=0)


def decode(raw: bytes):
    """(header, test_block bytes) after checking magic, header fields, length and test_block_sha256."""
    if len(raw) < PREFIX_BYTES or bytes(raw[:4]) != MAGIC:
        raise ScorerError("not a JTS1 scorer artifact")
    (n,) = LEN.unpack(bytes(raw[4:PREFIX_BYTES]))
    try:
        h = json.loads(bytes(raw[PREFIX_BYTES:PREFIX_BYTES + n]))
    except ValueError as err:
        raise ScorerError("unreadable JTS1 header") from err
    if not isinstance(h, dict) or set(h) != set(HEADER_FIELDS) or h["schema"] != SCHEMA:
        raise ScorerError("JTS1 header fields differ from the schema")
    block = bytes(raw[PREFIX_BYTES + n:])
    if len(block) != 8 * (h["p"] + 1) * h["n_test"] or _sha(block) != h["test_block_sha256"]:
        raise ScorerError("JTS1 test block length or sha256 differs from its header")
    return h, block


def load(path, expect: dict | None = None):
    """Outer-test scorer adapter entry point: (header, X_test f64[n_test, p], y_test f64[n_test]) of one artifact.
    `expect` pins header fields (e.g. stripped_raw_sha256 of the published payload, view_id, cut_id, sha256 of the gz)."""
    data = Path(path).read_bytes()
    expect = dict(expect or {})
    want_gz = expect.pop("artifact_sha256", None)
    if want_gz is not None and _sha(data) != want_gz:
        raise ScorerError(f"{path}: artifact sha256 differs from the pinned value")
    h, block = decode(gzip.decompress(data))
    for k, v in expect.items():
        if h.get(k) != v:
            raise ScorerError(f"{path}: header {k}={h.get(k)!r} != expected {v!r}")
    m = np.frombuffer(block, dtype="<f8").reshape(h["n_test"], h["p"] + 1).astype(np.float64, copy=True)
    return h, m[:, :-1], m[:, -1]


def target_bytes(raw: bytes) -> set:
    """The exact 8-byte float64 LE encodings of the target column (no tolerance, no feature columns)."""
    h, block = decode(raw)
    w = 8 * (h["p"] + 1)
    return {block[i * w + w - 8:i * w + w] for i in range(h["n_test"])}


def reconstruct(stripped_raw: bytes, raw: bytes) -> bytes:
    """The original JTT1 bytes: header with the recorded n_test, the stripped payload's labels and train block, then the
    artifact's outer-test block. Used by the strip tool to prove nothing was lost or altered."""
    h, block = decode(raw)
    if bytes(stripped_raw[:4]) != JTT1_MAGIC or _sha(stripped_raw) != h["stripped_raw_sha256"]:
        raise ScorerError("stripped payload differs from the artifact binding")
    n_train, n_test, p, inner = JTT1_HEADER.unpack(bytes(stripped_raw[4:4 + JTT1_HEADER.size]))
    if (n_train, n_test, p, inner) != (h["n_train"], 0, h["p"], h["inner_folds"]):
        raise ScorerError("stripped payload header differs from the artifact binding")
    out = JTT1_MAGIC + JTT1_HEADER.pack(n_train, h["n_test"], p, inner) + bytes(stripped_raw[4 + JTT1_HEADER.size:]) + block
    if _sha(out) != h["original_raw_sha256"]:
        raise ScorerError("reconstruction differs from the original raw sha256")
    return out
