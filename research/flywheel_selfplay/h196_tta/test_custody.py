"""Fake-S3 custody controls: no network or corpus access."""

import io
import numpy as np
import pytest

from s3_arrays import upload, replay


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.corrupt = False
        self.versioned = True

    def put_object(self, **kwargs):
        assert kwargs["IfNoneMatch"] == "*"
        key = kwargs["Key"]
        if key in self.objects:
            raise RuntimeError("412 precondition failed")
        self.objects[key] = kwargs["Body"]
        return {"VersionId": "version-1" if self.versioned else "null"}

    def get_object(self, **kwargs):
        assert kwargs["VersionId"] == "version-1"
        payload = self.objects[kwargs["Key"]]
        if self.corrupt:
            payload += b"corrupt"
        return {"VersionId": "version-1", "Body": io.BytesIO(payload)}


def test_exact_version_member_replay_and_no_overwrite():
    client = MemoryS3()
    arrays = {"cid": np.arange(24, dtype=np.float32).reshape(3, 8)}
    receipt = upload(client, "B/s196101-f0.npz", arrays)
    assert np.array_equal(replay(client, receipt)["cid"], arrays["cid"])
    with pytest.raises(RuntimeError, match="412"):
        upload(client, "B/s196101-f0.npz", arrays)
    receipt["members"]["cid"]["shape"] = [24]
    with pytest.raises(RuntimeError, match="member"):
        replay(client, receipt)


def test_corrupt_payload_rejected():
    client = MemoryS3()
    client.corrupt = True
    with pytest.raises(RuntimeError, match="object verification"):
        upload(client, "bad.npz", {"c": np.ones(4)})


def test_unversioned_custody_rejected():
    client = MemoryS3()
    client.versioned = False
    with pytest.raises(ValueError, match="versioned"):
        upload(client, "unversioned.npz", {"c": np.ones(4)})


def test_pickle_and_path_escape_rejected():
    client = MemoryS3()
    with pytest.raises(ValueError, match="object arrays"):
        upload(client, "object.npz", {"c": np.array([object()], dtype=object)})
    with pytest.raises(ValueError, match="relative"):
        upload(client, "../escape.npz", {"c": np.ones(4)})
    assert client.objects == {}
