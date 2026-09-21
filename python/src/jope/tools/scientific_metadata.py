"""Metadata contracts only. Hashes do not establish original execution or access."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

CONTEXT_FIELDS = (
    "family",
    "dataset",
    "partition",
    "cut",
    "prepared_sha256",
    "training_sha256",
    "validation_sha256",
    "preprocessing_sha256",
    "evaluator_sha256",
    "runtime_sha256",
    "numerical_contract_sha256",
)


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def sha256_value(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def metadata_bytes(path, limit=32 * 1024 * 1024):
    path = Path(path)
    if "data" in path.parts:
        raise ValueError("refusing to open data/ payloads")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("metadata symlink rejected")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("metadata size limit exceeded")
    return data, hashlib.sha256(data).hexdigest()


def read_with_digest(path, limit=32 * 1024 * 1024):
    data, observed = metadata_bytes(path, limit)
    value = json.loads(
        data,
        object_pairs_hook=_object,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + x)),
    )
    canonical(value)  # Reject overflowing JSON exponents, too.
    return value, observed


def read(path, limit=32 * 1024 * 1024):
    return read_with_digest(path, limit)[0]


def inside(root, name):
    root = Path(root).resolve(strict=True)
    name = Path(name)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("workspace-relative metadata path required")
    path = root
    for part in name.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("metadata symlink ancestry rejected")
    return path


def reference(root, ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or not sha256_value(ref["sha256"]):
        raise ValueError("exact metadata reference required")
    path = inside(root, ref["path"])
    value, observed = read_with_digest(path)
    if observed != ref["sha256"]:
        raise ValueError("metadata reference changed")
    return value


def context(row):
    if not isinstance(row, dict) or any(k not in row for k in CONTEXT_FIELDS):
        raise ValueError("complete scientific context identity required")
    value = {k: row[k] for k in CONTEXT_FIELDS}
    if (
        value["partition"] != "meta_train"
        or type(value["cut"]) is not int
        or value["cut"] < 0
        or any(not isinstance(value[k], str) or not value[k] for k in ("family", "dataset"))
        or any(not sha256_value(v) for k, v in value.items() if k.endswith("_sha256"))
    ):
        raise ValueError("invalid or nontraining scientific context")
    return value


def program(row):
    genome, sha = row.get("genome"), row.get("program_sha256")
    if (
        not isinstance(genome, str)
        or not genome
        or not sha256_value(sha)
        or hashlib.sha256(genome.encode()).hexdigest() != sha
    ):
        raise ValueError("program bytes/digest disagree")
    return {"genome": genome, "program_sha256": sha}


def immutable(path, value):
    """Publish once, preserving historical diagnostics and concurrent writers."""
    path = Path(path)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("immutable metadata symlink rejected")
    data = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("immutable metadata changed; use a successor path")
        return hashlib.sha256(data).hexdigest()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        staging = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(staging, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise ValueError("concurrent immutable metadata conflict")
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        staging.unlink()
    return hashlib.sha256(data).hexdigest()
