"""Background cluster resource monitor: per-host GPU + CPU utilization, non-blocking.

Polls every host (local + ssh) on a daemon thread at a fixed interval and caches the
parsed snapshot, so a TUI frame reads it in O(1) and never blocks on an ssh round trip.
Remote calls reuse the Prime-owned ControlMaster sockets the eval lanes already set up,
so each poll is one multiplexed round trip that returns
GPU stats + load average + core count + active bridge-lane count together.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .prime_config import ssh_control_path

_GPU_QUERY = "index,utilization.gpu,memory.used,memory.total,temperature.gpu"
# One remote command: GPUs, then a marker, then loadavg + nproc + live bridge lanes.
_REMOTE_CMD = (
    f"nvidia-smi --query-gpu={_GPU_QUERY} --format=csv,noheader,nounits 2>/dev/null; "
    "echo __SEP__; cat /proc/loadavg; nproc; "
    "pgrep -fc 'jope-(newmath-)?validate' 2>/dev/null || echo 0"
)


@dataclass
class HostState:
    host: str
    ok: bool = False
    gpus: list[dict[str, Any]] = field(default_factory=list)  # {index, util, mem_used, mem_total, temp}
    load1: float | None = None
    cores: int | None = None
    lanes: int | None = None  # active bridge processes (eval lanes)
    age_s: float = 0.0
    updated_at: float = 0.0


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse(host: str, text: str) -> HostState:
    st = HostState(host=host, ok=True, updated_at=time.time())
    gpu_part, _, rest = text.partition("__SEP__")
    for line in gpu_part.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0] != "":
            st.gpus.append(
                {
                    "index": int(_num(parts[0]) or 0),
                    "util": _num(parts[1]),
                    "mem_used": _num(parts[2]),
                    "mem_total": _num(parts[3]),
                    "temp": _num(parts[4]) if len(parts) > 4 else None,
                }
            )
    rest_lines = [l for l in rest.strip().splitlines() if l.strip()]
    if rest_lines:
        st.load1 = _num(rest_lines[0].split()[0]) if rest_lines[0].split() else None
    if len(rest_lines) >= 2:
        c = _num(rest_lines[1])
        st.cores = int(c) if c is not None else None
    if len(rest_lines) >= 3:
        lanes = _num(rest_lines[2])
        st.lanes = int(lanes) if lanes is not None else None
    return st


# ---- compute-saturation guard --------------------------------------------------
# Our stack (self-play eval, champion miner, GP factory) shares xbabe2/xbabe3 with
# each other and with the user's own jobs. These helpers let each launch point read a
# host's 1-min loadavg / core count and back off when a box is saturated — so we never
# pile onto an already-hammered host. Everything is FAIL-OPEN: a probe error returns
# None and callers treat "unknown" as "not saturated", so a flaky ssh never stalls the
# primary v33 loop.
#
# Policy constants (ratio = load1 / cores):
FACTORY_MAX = 1.00       # opportunistic GP factory (nice -5): shed workers past 1.0x, skip past 1.5x
EVAL_REMOTE_MAX = 1.40   # remote eval hosts: attenuate only when clearly oversaturated by foreign load
EVAL_WORKER_FLOOR = 0.25 # ...but never below 25% of requested workers
KEEPALIVE_MAX = 1.30     # skip an OPTIONAL factory relaunch when the local box is already hot
_LOCAL = ("local", "localhost", "")
_LOAD_CMD = "cat /proc/loadavg | cut -d' ' -f1; nproc"
_LOAD_TTL = 30.0
_load_cache: dict[str, tuple[float, float | None, int | None]] = {}


def host_load(host: str, *, ttl: float = _LOAD_TTL) -> tuple[float | None, int | None]:
    """(load1, cores) for a host, ~ttl-second cached. Local reads os.getloadavg directly;
    remote reuses the eval ControlMaster socket. Returns (None, None) on any failure."""
    now = time.time()
    hit = _load_cache.get(host)
    if hit is not None and now - hit[0] < ttl:
        return hit[1], hit[2]
    load1: float | None = None
    cores: int | None = None
    try:
        if host in _LOCAL:
            load1 = os.getloadavg()[0]
            cores = os.cpu_count()
        else:
            cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4"]
            sock = ssh_control_path(host)
            if sock.exists():
                cmd += ["-o", f"ControlPath={sock}"]
            cmd += [host, _LOAD_CMD]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=8).stdout.strip().splitlines()
            if out and out[0].split():
                load1 = _num(out[0].split()[0])
            if len(out) >= 2:
                c = _num(out[1])
                cores = int(c) if c is not None else None
    except Exception:  # noqa: BLE001 — fail-open by design
        load1, cores = None, None
    _load_cache[host] = (now, load1, cores)
    return load1, cores


def load_ratio(host: str) -> float | None:
    """load1/cores, or None if unknown (callers treat None as unsaturated)."""
    load1, cores = host_load(host)
    if load1 is None or not cores:
        return None
    return load1 / cores


def factory_workers(host: str, requested: int) -> int:
    """Requested worker count for an OPPORTUNISTIC factory host, attenuated by load:
    full at <=FACTORY_MAX, linearly shed to 0 at 1.5x (skip the host entirely)."""
    r = load_ratio(host)
    if r is None or r <= FACTORY_MAX:
        return requested
    mult = max(0.0, (1.5 - r) / (1.5 - FACTORY_MAX))
    return max(0, round(requested * mult))


def eval_worker_multiplier(host: str) -> float:
    """Worker multiplier [EVAL_WORKER_FLOOR, 1.0] for an eval host. `local` is the v33
    primary node and is NEVER attenuated. Remote hosts attenuate only above EVAL_REMOTE_MAX."""
    if host in _LOCAL:
        return 1.0
    r = load_ratio(host)
    if r is None or r <= EVAL_REMOTE_MAX:
        return 1.0
    mult = (2 * EVAL_REMOTE_MAX - r) / EVAL_REMOTE_MAX
    return max(EVAL_WORKER_FLOOR, min(1.0, mult))


def local_saturated(threshold: float = KEEPALIVE_MAX) -> bool:
    """True when the local box's load/cores exceeds `threshold` (for optional relaunch gates)."""
    r = load_ratio("local")
    return r is not None and r > threshold


class ClusterMonitor:
    def __init__(self, hosts: tuple[str, ...] = ("local", "xbabe1", "xbabe3"), interval: float = 2.0) -> None:
        self.hosts = hosts
        self.interval = interval
        self._states: dict[str, HostState] = {h: HostState(host=h) for h in hosts}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> ClusterMonitor:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def poll_once(self) -> ClusterMonitor:
        """Synchronously poll every host once (concurrent). For one-shot renders."""
        results: dict[str, HostState] = {}
        threads = []

        def work(h: str) -> None:
            results[h] = self._poll_host(h)

        for h in self.hosts:
            t = threading.Thread(target=work, args=(h,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=7.0)
        with self._lock:
            for h in self.hosts:
                if h in results:
                    self._states[h] = results[h]
        return self

    def snapshot(self) -> list[HostState]:
        now = time.time()
        with self._lock:
            out = []
            for h in self.hosts:
                st = self._states[h]
                st.age_s = now - st.updated_at if st.updated_at else 0.0
                out.append(st)
            return out

    def _poll_host(self, host: str) -> HostState:
        try:
            if host in ("local", ""):
                cmd = ["bash", "-lc", _REMOTE_CMD]
            else:
                cmd = [
                    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
                    "-o", "ControlMaster=auto", "-o", f"ControlPath={ssh_control_path(host)}",
                    "-o", "ControlPersist=300", host, _REMOTE_CMD,
                ]
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=6.0)
            if proc.returncode == 0 and proc.stdout.strip():
                return _parse(host, proc.stdout)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return HostState(host=host, ok=False, updated_at=time.time())

    def _run(self) -> None:
        while not self._stop.is_set():
            # poll hosts concurrently so the slowest ssh doesn't serialize the others
            threads = []
            results: dict[str, HostState] = {}

            def work(h: str) -> None:
                results[h] = self._poll_host(h)

            for h in self.hosts:
                t = threading.Thread(target=work, args=(h,), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join(timeout=7.0)
            with self._lock:
                for h in self.hosts:
                    if h in results:
                        self._states[h] = results[h]
            self._stop.wait(self.interval)
