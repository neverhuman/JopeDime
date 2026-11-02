"""Data-free CPU controls for prospective H-wave numeric thread admission."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from hwave_numeric_threads import NUMERIC_ENV, admit_numeric_threads, capture_numeric_threads


def worker(pid=100, threads=1):
    return {
        "pid": pid,
        "environment": {key: str(threads) for key in NUMERIC_ENV},
        "torch_intraop": threads,
        "native_pools": [{"user_api": "blas", "internal_api": "openblas", "num_threads": threads}],
    }


@pytest.mark.parametrize("os_threads", [4, 5, 64, 256])
def test_cuda_runtime_overhead_does_not_refuse_admitted_numeric_worker(os_threads):
    result = admit_numeric_threads([worker()], os_threads=os_threads)
    assert result["numeric_threads_upper_bound"] == 1
    assert result["os_threads_diagnostic"] == os_threads


@pytest.mark.parametrize("backend", [*NUMERIC_ENV, "torch_intraop", "native_pool"])
def test_numeric_excess_is_rejected_even_with_driver_overhead(backend):
    report = worker()
    if backend in NUMERIC_ENV:
        report["environment"][backend] = "5"
    elif backend == "native_pool":
        report["native_pools"][0]["num_threads"] = 5
    else:
        report[backend] = 5
    with pytest.raises(ValueError, match="numeric threads>4"):
        admit_numeric_threads([report], os_threads=64)


def test_numeric_four_boundary_and_multiple_worker_budget():
    assert admit_numeric_threads([worker(threads=4)], os_threads=64)["numeric_threads_upper_bound"] == 4
    assert (
        admit_numeric_threads([worker(1, 2), worker(2, 2)], os_threads=64)["numeric_threads_upper_bound"] == 4
    )
    with pytest.raises(ValueError, match="numeric threads>4"):
        admit_numeric_threads([worker(1, 4), worker(2, 1)], os_threads=64)
    with pytest.raises(ValueError, match="duplicate"):
        admit_numeric_threads([worker(), worker()], os_threads=64)


@pytest.mark.parametrize("bad", [None, "", "0", "-1", "many", 1, True])
def test_unknown_or_invalid_numeric_limits_fail_closed(bad):
    report = worker()
    report["environment"]["OMP_NUM_THREADS"] = bad
    with pytest.raises(ValueError):
        admit_numeric_threads([report], os_threads=64)


def test_missing_reports_and_loaded_pool_inventory_fail_closed():
    with pytest.raises(ValueError, match="reports required"):
        admit_numeric_threads([], os_threads=64)
    report = worker()
    del report["native_pools"]
    with pytest.raises(ValueError, match="inventory"):
        admit_numeric_threads([report], os_threads=64)
    report = worker()
    report["native_pools"][0]["num_threads"] = None
    with pytest.raises(ValueError, match="positive integer"):
        admit_numeric_threads([report], os_threads=64)


def test_capture_uses_worker_torch_and_native_values_without_cuda():
    env = {key: "1" for key in NUMERIC_ENV}
    pools = worker()["native_pools"]
    before = deepcopy(pools)
    report = capture_numeric_threads(
        SimpleNamespace(get_num_threads=lambda: 1), environ=env, native_pools=pools
    )
    assert report["environment"] == env and report["torch_intraop"] == 1
    assert pools == before
    assert admit_numeric_threads([report], os_threads=5)["numeric_threads_upper_bound"] == 1


def test_real_cpu_worker_capture(monkeypatch):
    import torch
    from threadpoolctl import threadpool_limits

    for key in NUMERIC_ENV:
        monkeypatch.setenv(key, "1")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            report = capture_numeric_threads(torch)
            assert admit_numeric_threads([report], os_threads=64)["numeric_threads_upper_bound"] == 1
    finally:
        torch.set_num_threads(previous)
