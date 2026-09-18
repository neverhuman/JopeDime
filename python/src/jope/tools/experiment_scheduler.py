#!/usr/bin/env python3
"""One spare-CPU allocation shared by qualification comparisons and encoders."""
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import experiment_runtime as runtime

PHASE = "GP optimizes cut targets for the next teaching corpus"
SPARE_CPUS = tuple(range(0, 8)) + tuple(range(62, 70))


def admission_reason(root, now=None):
    """Return None when admission is allowed; otherwise an explicit wait reason."""
    root = Path(root).resolve()
    now = time.time() if now is None else now
    try:
        read = lambda p: json.loads(p.read_text())
        campaign = read(root / "runs/campaign-progress.json")
        progress = read(root / "runs/cycle-progress.json")
        gp = read(root / "runs/gp-service.json")
        if now >= campaign["deadline"]:
            return "campaign deadline reached"
        if progress.get("complete") or progress.get("phase") != PHASE:
            return "mandatory campaign block: " + progress.get("phase", "unknown")
        if not 0 <= now - progress["timestamp"] <= 5:
            return "campaign progress stale"
        if not 0 <= now - gp["timestamp"] <= 5 or gp.get("phase") != "running":
            return "GP telemetry stale or unavailable"
        count = progress["neural_seeded_gp"]
        if count["total"] <= 0 or count["completed"] / count["total"] >= 0.9:
            return "campaign GP reached 90%; releasing CPUs and GPUs for retraining"
        cycle = (root / campaign["current_cycle"]).resolve()
        if not cycle.is_relative_to(root / "runs"):
            return "campaign cycle outside workspace"
        complete = read(cycle / "auxiliary/complete.json")
        if complete.get("schema") != "parallel.auxiliary_training_complete.v1":
            return "auxiliary learning has no durable completion"
        wait = root / "runs/gpu-wait.json"
        if wait.exists():
            receipt = read(wait)
            if 0 <= now - receipt.get("timestamp", 0) <= 3:
                return "campaign waiting for GPU lease: " + receipt.get("phase", "GPU work")
        return None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return "required campaign receipt unavailable: " + str(error)


@contextmanager
def spare_lease(root, owner, cpus=16, gpu=False):
    """Nonblocking lease; caller polls admission_reason and cancels owned trials.

    GPU callers launch direct ranks: this lease already owns the native GPU lock.
    Owned children inherit the flock descriptors as well as CPU affinity.
    """
    root = Path(root).resolve()
    if not 1 <= cpus <= 16:
        raise ValueError("qualification work must share at most 16 spare CPUs")
    reason = admission_reason(root)
    if reason:
        raise BlockingIOError(reason)
    handles = []
    previous_fds = runtime._lease_fds
    original = os.sched_getaffinity(0)
    try:
        names = ["experiment-spare-cpus.lock"]
        if gpu:
            names.append("encoding-service.lock")
        for name in names:
            with runtime.parent_fd(root / "runs" / name) as (directory, filename):
                fd = os.open(filename, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            handle = os.fdopen(fd, "a+")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reason = admission_reason(root)
        if reason:
            raise BlockingIOError(reason)
        available = [cpu for cpu in SPARE_CPUS if cpu in original]
        if len(available) < cpus:
            raise BlockingIOError("the requested spare CPU affinity is unavailable")
        os.sched_setaffinity(0, available[:cpus])
        runtime._lease_fds = previous_fds + tuple(h.fileno() for h in handles)
        yield {"owner": owner, "cpus": available[:cpus], "gpu_world": 8 if gpu else 0,
               "pass_fds": runtime._lease_fds}
    finally:
        runtime._lease_fds = previous_fds
        os.sched_setaffinity(0, original)
        for handle in reversed(handles):
            handle.close()
