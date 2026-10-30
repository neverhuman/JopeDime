import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from device_pool_cache import DevicePoolCache


def pool(n, offset):
    return {
        "program": np.arange(n * 3, dtype=np.float32).reshape(n, 3) + offset,
        "proj16": np.arange(n * 2, dtype=np.float64).reshape(n, 2) + offset,
        "fitness": np.arange(n, dtype=np.float64) + offset,
        "gate": np.arange(n) % 2 == 0,
    }


def test_exact_legacy_batch_order_dtype_and_own_targets():
    pools = {"a": pool(5, 0), "b": pool(7, 100)}
    cids = ["a", "b", "a"]
    selection = [np.array([4, 0, 4]), np.array([1, 6, 0]), np.array([2, 1, 0])]
    cache = DevicePoolCache("cpu")
    packed = cache.pack(pools, cids, selection)
    for field in cache.FIELDS:
        expected = torch.from_numpy(
            np.stack([pools[c][field][s] for c, s in zip(cids, selection)])
        )
        assert (
            torch.equal(packed[field], expected)
            and packed[field].dtype == expected.dtype
        )
    assert cache.transfers == 2
    cache.pack(pools, cids, selection)
    assert cache.transfers == 2
    assert packed["cpu_index_bytes"] < packed["avoided_cpu_feature_slate_bytes"]


def test_promotion_invalidates_and_preserves_new_program_target_pair():
    pools = {"a": pool(5, 0)}
    cache = DevicePoolCache("cpu")
    cache.pack(pools, ["a"], [np.array([0])])
    p = pools["a"]
    p["program"] = np.vstack([p["program"], [42, 43, 44]]).astype(np.float32)
    p["proj16"] = np.vstack([p["proj16"], [7, 8]])
    p["fitness"] = np.append(p["fitness"], -1.5)
    p["gate"] = np.append(p["gate"], True)
    batch = cache.pack(pools, ["a"], [np.array([5])])
    assert cache.transfers == 2 and batch["program"][0, 0, 0] == 42
    assert batch["fitness"][0, 0] == -1.5 and batch["gate"][0, 0]


def test_same_length_replacement_invalidates_cache():
    pools = {"a": pool(5, 0)}
    cache = DevicePoolCache("cpu")
    cache.pack(pools, ["a"], [np.array([0])])
    pools["a"]["program"] = pools["a"]["program"] + 10
    batch = cache.pack(pools, ["a"], [np.array([0])])
    assert cache.transfers == 2 and batch["program"][0, 0, 0] == 10


@pytest.mark.parametrize("index", [-1, 5])
def test_context_bounds_are_checked(index):
    with pytest.raises(ValueError):
        DevicePoolCache("cpu").pack({"a": pool(5, 0)}, ["a"], [np.array([index])])


def test_gpu_memory_guard_backs_off_until_safe_without_changing_batch(monkeypatch):
    import device_pool_cache as module

    available = iter([(3, 24), (12, 24), (21, 24)])
    sleeps, releases = [], []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: next(available))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: releases.append(True))
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    module.wait_for_gpu_headroom("cuda", minimum_free_bytes=20, pause_seconds=5)
    assert sleeps == [5, 5] and len(releases) == 3


def test_gpu_memory_guard_refuses_insufficient_hardware(monkeypatch):
    import device_pool_cache as module

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (16, 16))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    with pytest.raises(RuntimeError, match="capacity"):
        module.wait_for_gpu_headroom("cuda", minimum_free_bytes=20)


def test_round_handoff_finishes_gpu_before_unlock(tmp_path, monkeypatch):
    import fcntl

    import device_pool_cache as module

    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: events.append("release_cache")
    )
    with (
        (tmp_path / "gpu.lock").open("a") as owner,
        (tmp_path / "gpu.lock").open("a") as other,
    ):
        fcntl.flock(owner, fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        module.release_gpu_round(owner)
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert events == ["sync", "release_cache"]


def test_round_handoff_unlocks_even_if_gpu_sync_fails(tmp_path, monkeypatch):
    import fcntl

    import device_pool_cache as module

    def fail():
        raise RuntimeError("GPU sync failed")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with (
        (tmp_path / "gpu.lock").open("a") as owner,
        (tmp_path / "gpu.lock").open("a") as other,
    ):
        fcntl.flock(owner, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="GPU sync failed"):
            module.release_gpu_round(owner)
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_landmark_overflow_rescaling_keeps_raw_table_and_ordinary_columns():
    from device_pool_cache import landmark_float32

    raw = np.array([[1e300, 1.5], [-5e299, 2.0]], dtype=np.float64)
    original = raw.copy()
    result = landmark_float32(raw)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(result[:, 0], [1.0, -0.5])
    np.testing.assert_array_equal(result[:, 1], raw[:, 1].astype(np.float32))
