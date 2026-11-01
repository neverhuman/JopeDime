"""Memory-only, create-only candidate-array custody with exact-version S3 replay for H198."""

import hashlib
import io

import numpy as np

BUCKET = "veox-jopedime-learning-use2"
PREFIX = "experiments/rd_breakthrough_20261030/H198/rd_argon-t1205/"


def array_identity(array):
    a = np.ascontiguousarray(array)
    if a.dtype.hasobject:
        raise ValueError("object arrays are forbidden")
    return dict(
        dtype=a.dtype.str,
        shape=list(a.shape),
        sha256=hashlib.sha256(a.tobytes()).hexdigest(),
    )


def replay(client, receipt):
    if receipt["bucket"] != BUCKET or not receipt["key"].startswith(PREFIX):
        raise ValueError("wrong custody scope")
    version = receipt["version_id"]
    if not version or version == "null":
        raise ValueError("versioned custody required")
    response = client.get_object(Bucket=BUCKET, Key=receipt["key"], VersionId=version)
    body = response["Body"]
    try:
        payload = body.read()
    finally:
        body.close()
    if (
        response.get("VersionId") != version
        or len(payload) != receipt["bytes"]
        or hashlib.sha256(payload).hexdigest() != receipt["sha256"]
    ):
        raise RuntimeError("exact-version object verification failed")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if {key: array_identity(a) for key, a in arrays.items()} != receipt["members"]:
        raise RuntimeError("array member verification failed")
    return arrays


def upload(client, name, arrays):
    if not name or name.startswith("/") or ".." in name or not name.endswith(".npz"):
        raise ValueError("unique relative npz key required")
    members = {key: array_identity(a) for key, a in arrays.items()}
    if not members:
        raise ValueError("nonempty array archive required")
    memory = io.BytesIO()
    np.savez_compressed(memory, **arrays)
    payload = memory.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    key = PREFIX + name
    result = client.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=payload,
        IfNoneMatch="*",
        Metadata={"sha256": digest},
        ContentType="application/octet-stream",
    )
    receipt = dict(
        bucket=BUCKET,
        key=key,
        version_id=result.get("VersionId"),
        bytes=len(payload),
        sha256=digest,
        members=members,
    )
    replay(client, receipt)
    return receipt
