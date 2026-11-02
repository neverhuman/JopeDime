"""Prospective H-wave numeric-thread admission; OS/driver threads are diagnostic.

Use inside each numerical worker after library setup and at stage boundaries.
The supervisor checks the complete live worker reports, not its own library
settings. Historical, hash-pinned H-wave execution sources remain immutable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

NUMERIC_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
THREAD_POLICY = "<=4 numeric threads (OMP/MKL/OpenBLAS/torch intra-op); CUDA runtime/driver threads exempt"


def capture_numeric_threads(torch_module, *, environ=None, native_pools=None) -> dict:
    """Measure loaded pool limits in the worker; never initialize/query CUDA."""
    if native_pools is None:
        from threadpoolctl import threadpool_info

        native_pools = threadpool_info()
    env = os.environ if environ is None else environ
    return {
        "pid": os.getpid(),
        "environment": {key: env.get(key) for key in NUMERIC_ENV},
        "torch_intraop": torch_module.get_num_threads(),
        "native_pools": [
            {
                "user_api": pool.get("user_api"),
                "internal_api": pool.get("internal_api"),
                "num_threads": pool.get("num_threads"),
            }
            for pool in native_pools
        ],
    }


def _positive_int(value, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} requires a positive integer")
    return value


def admit_numeric_threads(reports: Sequence[Mapping], *, os_threads: int) -> dict:
    """Fail closed on unknown/excess numerical limits, including multiple workers.

    A worker's bound is the maximum of its OMP/MKL/OpenBLAS/torch pool limits,
    not the sum of alternative library pools. Sum those bounds across concurrent
    workers. Disable nested parallelism and application-created concurrent numeric
    jobs; those must instead be admitted as separate workers. OS thread count
    never proves numerical admission or identifies a driver thread.
    """
    if not reports:
        raise ValueError("numeric worker reports required")
    bounds = []
    seen = set()
    for report in reports:
        pid = _positive_int(report.get("pid"), "worker pid")
        if pid in seen:
            raise ValueError("duplicate numeric worker pid")
        seen.add(pid)
        env = report.get("environment", {})
        caps = []
        for key in NUMERIC_ENV:
            value = env.get(key)
            if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
                raise ValueError(f"{key} numeric limit missing/invalid")
            caps.append(_positive_int(int(value), key))
        caps.append(_positive_int(report.get("torch_intraop"), "torch intra-op"))
        pools = report.get("native_pools")
        if not isinstance(pools, list):
            raise ValueError("native numeric pool inventory required")
        for pool in pools:
            if pool.get("user_api") not in ("blas", "openmp"):
                raise ValueError("unrecognized native numeric pool")
            caps.append(_positive_int(pool.get("num_threads"), "native numeric pool"))
        bounds.append(max(caps))
    total = sum(bounds)
    if total > 4:
        raise ValueError(f"numeric threads>4: worker bounds={bounds}, total={total}")
    if type(os_threads) is not int or os_threads < len(reports):
        raise ValueError("OS thread diagnostic inconsistent with worker count")
    return {
        "policy": THREAD_POLICY,
        "numeric_threads_upper_bound": total,
        "os_threads_diagnostic": os_threads,
        "workers": list(reports),
    }
