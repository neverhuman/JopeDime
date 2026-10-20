"""Readback gate for an externally owned job; creates no resource route.

The existing resource owner must independently qualify the entire control tree.
A sample, a boolean in a user-written file, or a test fixture is not that proof.
"""
from __future__ import annotations

import json
import math
import os
import socket
import stat
import time
from pathlib import Path

from protocol import file_hash


def validate_status(status, grant, grant_sha, now):
    if status.get('grant_sha256') != grant_sha or status.get('cgroup') != grant['cgroup']:
        raise ValueError('wrong guard/leaf identity')
    age = now - status.get('sample_unix', 0)
    if not 0 <= age <= 1.5 or status.get('tick_seconds') != 1:
        raise ValueError('stale or unsupported continuous guard')
    peak = status.get('host_cpu_peak_percent', 100)
    if type(peak) not in (int, float) or not math.isfinite(peak) or not 0 <= peak < 65 or status.get('state') != 'admitted':
        raise ValueError('resource owner holds new work; stop this job')
    for key in ('available_ram_bytes', 'free_disk_bytes', 'temperature_celsius'):
        value = status.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('missing/nonfinite resource telemetry: ' + key)
    if status['available_ram_bytes'] < 20 * 1024**3 or status.get('swapin_bytes_per_second', 1) != 0:
        raise ValueError('RAM/swap guard hold')
    if status['free_disk_bytes'] < grant.get('minimum_free_disk_bytes', 150 * 1024**3):
        raise ValueError('storage hold')
    if not 0 <= status['temperature_celsius'] <= 85:
        raise ValueError('temperature hold')


def protected_json(path):
    p = Path(path)
    if p.is_symlink():
        raise ValueError("symlink grant refused")
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != 0 or st.st_mode & 0o022 or st.st_size > 65536:
            raise ValueError("owner-protected root grant/telemetry required")
        return json.loads(os.read(fd, st.st_size + 1))
    finally:
        os.close(fd)


def check(grant_path, *, plan_sha256, source_commit, clock=time.time):
    """Must run INSIDE the new, independently admitted job, before torch import."""
    g = protected_json(grant_path)
    if g.get("schema") != "jope.gpu-research-admission/v1":
        raise ValueError("no current GPU research admission")
    if g.get("node") != socket.gethostname() or g.get("node") not in ("xbabe2", "xbabe3"):
        raise ValueError("wrong research node")
    if g.get("preregistration_sha256") != plan_sha256 or g.get("source_commit") != source_commit:
        raise ValueError("grant does not bind this source/plan")
    now = clock()
    if not g.get("not_before_unix", now + 1) <= now < g.get("expires_unix", 0):
        raise ValueError("grant not current; no clock rewrite/retry")
    # These are source-bound independent qualification artifacts, not generated
    # by this module. The resource owner must review the actual mechanism.
    for key in ("independent_review", "whole_job_qualification", "aggregate_cpu70_qualification"):
        artifact = g.get(key, {})
        if not artifact.get("path") or file_hash(artifact["path"]) != artifact.get("sha256"):
            raise ValueError("missing exact external proof: " + key)
        protected_json(artifact["path"])
    rel = Path('/proc/self/cgroup').read_text().strip()
    if rel != '0::' + g.get("cgroup", "") or not g.get("cgroup", "").startswith('/system.slice/jope-research-'):
        raise ValueError("not inside the exact new owned research leaf")
    root = Path('/sys/fs/cgroup' + g['cgroup'])
    expected = {"memory.max": str(g["memory_max_bytes"]), "memory.swap.max": "0", "pids.max": str(g["tasks_max"])}
    for name, value in expected.items():
        if (root / name).read_text().strip() != value:
            raise ValueError("kernel bound mismatch: " + name)
    quota, period = (root / 'cpu.max').read_text().split()
    if quota == 'max' or int(quota) / int(period) > 2 or not 1 <= len(os.sched_getaffinity(0)) <= 2:
        raise ValueError("source requires at most two assigned CPU cores")
    if (root / 'cgroup.freeze').read_text().strip() != '0':
        raise ValueError("owned leaf currently held")
    status = protected_json(g['guard_status_path'])
    validate_status(status, g, file_hash(grant_path), now)
    return g
