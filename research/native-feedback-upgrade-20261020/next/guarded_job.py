"""Offline guard candidate for a single future owned cgroup leaf.

No OS reads/writes, process signals, launcher or live authority are provided.
An owner-qualified adapter must supply all facts and enforce/read back each plan
under an independent watchdog. A passing fixture is never live admission.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import PurePosixPath
import re


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


@dataclass(frozen=True)
class CpuPoint:
    monotonic_ns: int
    online_ids: tuple[int, ...]
    # Linux aggregate user,nice,system,idle,iowait,irq,softirq,steal counters.
    # guest counters are not added a second time.
    ticks: tuple[int, ...]
    owned_usage_usec: int


def cpu_interval(before, after, now_ns):
    """Counter-reset/hotplug/stale intervals refuse admission, never clamp busy."""
    for p in (before, after):
        need(isinstance(p, CpuPoint) and integer(p.monotonic_ns), "missing CPU point")
        need(type(p.online_ids) is tuple and len(p.online_ids) > 0
             and all(integer(x) for x in p.online_ids)
             and tuple(sorted(set(p.online_ids))) == p.online_ids, "invalid online CPU roster")
        need(type(p.ticks) is tuple and len(p.ticks) == 8
             and all(integer(x) for x in p.ticks) and integer(p.owned_usage_usec), "invalid CPU counters")
    need(before.online_ids == after.online_ids, "online CPU roster changed")
    need(integer(now_ns), "invalid current time")
    dt = after.monotonic_ns - before.monotonic_ns
    need(900_000_000 <= dt <= 1_250_000_000, "CPU interval outside one-second tolerance")
    need(0 <= now_ns - after.monotonic_ns <= 250_000_000, "stale/future CPU sample")
    delta = tuple(b - a for a, b in zip(before.ticks, after.ticks))
    need(all(x >= 0 for x in delta) and sum(delta) > 0, "CPU counter reset/stall")
    usage = after.owned_usage_usec - before.owned_usage_usec
    need(usage >= 0, "owned CPU counter reset")
    n = len(after.online_ids)
    host = Fraction(sum(delta) - delta[3] - delta[4], sum(delta))
    owned_cores = Fraction(usage * 1000, dt)
    need(owned_cores <= n, "owned CPU exceeds online capacity")
    # One core of conservative alignment uncertainty; larger disagreement fails.
    need(owned_cores <= host * n + 1, "host/owned CPU samples inconsistent")
    foreign_upper_cores = max(Fraction(0), host * n - max(Fraction(0), owned_cores - 1))
    return {"online_cpus": n, "interval_ns": dt, "host_fraction": host,
            "owned_cores": owned_cores, "foreign_upper_cores": foreign_upper_cores}


@dataclass(frozen=True)
class Leaf:
    lease_sha256: str
    path: str
    device: int
    inode: int
    member_starts: tuple[tuple[int, int], ...]

    def validate(self):
        need(type(self.lease_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", self.lease_sha256), "unbound leaf lease")
        need(type(self.path) is str, "invalid leaf path")
        p = PurePosixPath(self.path)
        need(str(p) == self.path and p.is_relative_to("/sys/fs/cgroup")
             and ".." not in p.parts and p.name.startswith("jope-research-")
             and p.name.endswith(".scope"), "exact research leaf path required")
        need(integer(self.device) and integer(self.inode, 1), "unbound leaf inode")
        need(type(self.member_starts) is tuple and len(self.member_starts) > 0
             and all(type(x) is tuple and len(x) == 2 and integer(x[0], 1) and integer(x[1], 1)
                     for x in self.member_starts)
             and tuple(sorted(set(self.member_starts))) == self.member_starts
             and len({x[0] for x in self.member_starts}) == len(self.member_starts), "unbound exact PID/start roster")


@dataclass(frozen=True)
class ScopeFacts:
    leaf: Leaf
    child_cgroups: int
    descendants_confined: bool
    ancestor_frozen: bool
    delegated_leaf_qualified: bool
    controller_readback_qualified: bool
    parent_watchdog_qualified: bool
    parent_watchdog_alive: bool
    leaf_frozen: bool
    freeze_owner_sha256: str | None


@dataclass(frozen=True)
class Health:
    ram_ok: bool
    swap_ok: bool
    temperature_ok: bool
    storage_ok: bool


class Guard:
    """Produce plans only. Exact leaf binding is a prerequisite for any action.

    Defaults target60%, hold65%, breach70%, <=8 owned cores. Three consecutive
    safe intervals precede RUN. Hard breaches and identity/measurement/health
    faults latch STOP. Release requires a newly qualified owner-bound instance.
    """

    def __init__(self, leaf, *, deadline_ns, max_owned_cores=8):
        leaf.validate()
        need(integer(deadline_ns, 1), "bound monotonic deadline required")
        need(integer(max_owned_cores, 1) and max_owned_cores <= 8, "owned quota outside candidate bound")
        self.leaf = leaf
        self.deadline_ns = deadline_ns
        self.max_owned_cores = max_owned_cores
        self.safe_intervals = 0
        self.latched = False
        self.samples = 0
        self.peak_host = Fraction(0)
        self.breach_samples = []
        self.reasons = Counter()
        self.last_cpu_point = None

    def observe(self, before, after, now_ns, scope, health):
        self.samples += 1
        reason, mode, writable = None, "STOP", False
        view, quota_usec = None, 0
        try:
            # A changed path, inode or PID/start roster may now belong to someone
            # else. Never plan even a freeze against an unverified replacement.
            need(isinstance(scope, ScopeFacts), "missing scope facts")
            scope.leaf.validate()
            need(scope.leaf == self.leaf, "leaf identity/membership changed")
            need(type(scope.child_cgroups) is int and scope.child_cgroups == 0
                 and scope.descendants_confined is True, "unconfined leaf membership")
            need(scope.delegated_leaf_qualified is True, "leaf delegation not qualified")
            need(type(scope.leaf_frozen) is bool, "unknown leaf freezer state")
            need(scope.freeze_owner_sha256 == (self.leaf.lease_sha256 if scope.leaf_frozen else None),
                 "foreign/unknown leaf freeze ownership; never release it")
            writable = True
            need(scope.controller_readback_qualified is True, "control/readback not qualified")
            need(scope.parent_watchdog_qualified is True and scope.parent_watchdog_alive is True, "watchdog unqualified/dead")
            need(scope.ancestor_frozen is False, "ancestor frozen/unknown; never thaw ancestor")
            need(isinstance(health, Health) and all(x is True for x in (
                health.ram_ok, health.swap_ok, health.temperature_ok, health.storage_ok)), "RAM/swap/temperature/storage hold")
            need(integer(now_ns) and now_ns < self.deadline_ns, "lease expired/time invalid")
            view = cpu_interval(before, after, now_ns)
            need(self.last_cpu_point is None or before == self.last_cpu_point,
                 "CPU interval repeated/gapped or previous sample changed")
            self.last_cpu_point = after
            self.peak_host = max(self.peak_host, view["host_fraction"])
            if view["host_fraction"] >= Fraction(70, 100):
                self.breach_samples.append(self.samples)
                raise ValueError("observed aggregate CPU >=70%; owned work must stop")
            need(not self.latched, "prior fault remains latched")
            cores = min(Fraction(self.max_owned_cores), Fraction(60, 100) * view["online_cpus"] - view["foreign_upper_cores"])
            if view["host_fraction"] >= Fraction(65, 100) or cores < Fraction(1, 4):
                self.safe_intervals = 0
                reason, mode = "foreign/host load leaves insufficient operating margin", "HOLD"
            else:
                self.safe_intervals += 1
                if self.safe_intervals < 3:
                    reason, mode = "awaiting three consecutive safe one-second intervals", "HOLD"
                else:
                    quota_usec = (cores * 100_000).__floor__()
                    reason, mode = "bounded own-leaf quota; enforce/read back before release", "RUN_CANDIDATE"
        except (ValueError, TypeError, AttributeError) as error:
            self.latched, self.safe_intervals = True, 0
            reason = str(error)
        self.reasons[reason] += 1
        plan = {"schema": "jope.owned-leaf-guard-plan/v1", "sample_number": self.samples, "mode": mode,
                "reason": reason, "latched": self.latched, "leaf_lease_sha256": self.leaf.lease_sha256,
                "leaf_path": self.leaf.path, "leaf_device": self.leaf.device, "leaf_inode": self.leaf.inode,
                "exact_member_starts": self.leaf.member_starts,
                "actions": ([{"op": "set_cpu_max", "quota_usec": quota_usec, "period_usec": 100_000},
                             {"op": "readback_cpu_max"}] +
                            ([{"op": "release_own_freeze_after_readback"}] if scope.leaf_frozen else [])
                            if mode == "RUN_CANDIDATE" else [{"op": "freeze_own_leaf"}, {"op": "readback_frozen"}]) if writable else [],
                "unverified_identity_action": None if writable else "owner watchdog must contain original leaf; do not touch replacement",
                "host_pct": float(view["host_fraction"] * 100) if view else None,
                "owned_cores": float(view["owned_cores"]) if view else None,
                "foreign_upper_cores": float(view["foreign_upper_cores"]) if view else None,
                "actual_online_cpus": view["online_cpus"] if view else None,
                "peak_host_pct": float(self.peak_host * 100), "breach_samples": self.breach_samples.copy(),
                "live_execution_qualified": False, "actions_executed": False,
                "observation_semantics": "0.9-1.25 second counter interval, <=0.25 second age; not instantaneous foreign-load control"}
        encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        plan["plan_sha256"] = hashlib.sha256(encoded).hexdigest()
        return plan
