"""Task-local guard candidate; imports and policy/proposal calls are pure.

The explicit CLI implements ONLY a finite CPU inert qualification. It requires
a separately supplied protected root-owned grant and exact staged source/binary
hashes. No live authority or qualification follows from this source or fixtures.
The custodian may freeze only the pinned inert leaf and kill only its own parent
through a pidfd captured before fork; no shared controller or model is targeted.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import PurePosixPath
import re
import os
import time
import socket
import struct
import stat
import subprocess
import signal
import select
import sys
from pathlib import Path


SYSTEMD_255_REFERENCES = (
    "https://www.freedesktop.org/software/systemd/man/255/systemd-run.html",
    "https://www.freedesktop.org/software/systemd/man/255/systemctl.html",
    "https://www.freedesktop.org/software/systemd/man/255/systemd.resource-control.html",
    "https://www.freedesktop.org/software/systemd/man/255/systemd.scope.html",
    "https://www.freedesktop.org/software/systemd/man/255/org.freedesktop.systemd1.html",
)


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


@dataclass(frozen=True)
class RootScopeCandidate:
    """Render-only alternative to an unavailable unprivileged delegation.

    The existing resource owner would authorize a root supervisor controlling
    exactly this NEW unit through systemd. No privilege/delegation is granted by
    this value or its hashes. The first executable must be an independently
    qualified start barrier, not an immediately executing training payload.
    """
    task_sha256: str
    source_sha256: str
    lease_sha256: str
    nonce: str
    online_ids: tuple[int, ...]
    allowed_ids: tuple[int, ...]
    uid: int
    gid: int
    cwd: str
    barrier_argv: tuple[str, ...]
    memory_high_bytes: int
    memory_max_bytes: int
    runtime_seconds: int
    device_visibility: str = ""

    def validate(self):
        for value in (self.task_sha256, self.source_sha256, self.lease_sha256):
            need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "unbound task/source/lease")
        need(type(self.nonce) is str and re.fullmatch(r"[0-9a-f]{32}", self.nonce), "fresh exact-unit nonce required")
        for roster in (self.online_ids, self.allowed_ids):
            need(type(roster) is tuple and 0 < len(roster)
                 and all(integer(x) for x in roster) and tuple(sorted(set(roster))) == roster, "invalid CPU roster")
        need(set(self.allowed_ids) <= set(self.online_ids) and len(self.allowed_ids) <= 8, "allowed CPUs outside bound")
        need(integer(self.uid, 1) and integer(self.gid, 1), "nonroot payload identity required")
        p = PurePosixPath(self.cwd) if type(self.cwd) is str else PurePosixPath("/")
        need(str(p) == self.cwd and p.is_relative_to("/home/ubuntu/jope-evidence") and ".." not in p.parts,
             "explicit owned custody cwd required")
        need(type(self.barrier_argv) is tuple and 0 < len(self.barrier_argv) <= 64
             and all(type(x) is str and 0 < len(x) <= 4096 and not any(c in x for c in "\x00\r\n") for x in self.barrier_argv)
             and self.barrier_argv[0].startswith("/"), "literal absolute barrier argv required")
        need(integer(self.memory_high_bytes, 1) and integer(self.memory_max_bytes, 1)
             and self.memory_high_bytes <= self.memory_max_bytes <= 8*1024**3, "candidate RAM ceiling")
        need(integer(self.runtime_seconds, 1) and self.runtime_seconds <= 3600, "finite <=1h runtime bound")
        need(self.device_visibility in ("", "0") and type(self.device_visibility) is str, "explicit candidate GPU visibility")

    @property
    def unit(self):
        self.validate()
        return "jope-research-" + self.task_sha256[:12] + "-" + self.nonce + ".scope"

    def control_argv(self, operation, *, quota_usec=None):
        """Literal commands for an owner-qualified root actor, NEVER executed.

        No unit-pattern, ancestor, arbitrary property or stop/kill operation is
        accepted. Revalidate exact path/inode/membership before every operation.
        Successful command exit is not actual quota/freezer readback proof.
        """
        unit = self.unit
        base = ["/usr/bin/systemctl", "--system", "--no-ask-password"]
        need(operation in ("show", "quota", "freeze", "thaw"), "unknown owned-unit operation")
        if operation == "quota":
            need(integer(quota_usec, 25_000) and quota_usec <= len(self.allowed_ids)*100_000,
                 "owned quota exceeds CPU allocation")
            percent = f"{quota_usec//1000}.{quota_usec%1000:03d}%"
            return base + ["--runtime", "set-property", unit, "CPUQuota=" + percent, "CPUQuotaPeriodSec=100ms"]
        need(quota_usec is None, "quota supplied to nonquota action")
        if operation == "show":
            return base + ["show", unit, "--property=Id,LoadState,ActiveState,ControlGroup,Slice,FreezerState,CPUQuotaPerSecUSec,CPUQuotaPeriodUSec,EffectiveCPUs,MemoryHigh,MemoryMax,MemorySwapMax,TasksMax,RuntimeMaxUSec"]
        return base + [operation, unit]

    def proposal(self):
        unit = self.unit
        # 0.25 core startup quota confines the barrier. Raise it only after
        # qualified fresh host/owned measurements and successful quota readback.
        args = ["/usr/bin/systemd-run", "--system", "--no-ask-password", "--scope",
                "--unit=" + unit, "--slice=jarambit-evo.slice", "--expand-environment=no",
                "--working-directory=" + self.cwd,
                "--property=CPUQuota=25%", "--property=CPUQuotaPeriodSec=100ms",
                "--property=AllowedCPUs=" + ",".join(map(str, self.allowed_ids)),
                "--property=MemoryHigh=" + str(self.memory_high_bytes),
                "--property=MemoryMax=" + str(self.memory_max_bytes),
                "--property=MemorySwapMax=0", "--property=TasksMax=64",
                "--property=RuntimeMaxSec=" + str(self.runtime_seconds) + "s",
                "--property=RuntimeRandomizedExtraSec=0", "--property=TimeoutStopSec=1s",
                "--property=SendSIGKILL=yes",
                "--", "/usr/bin/setpriv", "--reuid=" + str(self.uid), "--regid=" + str(self.gid),
                "--init-groups", "--no-new-privs", "--", "/usr/bin/env", "-i",
                "PATH=/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=2",
                "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "CUDA_VISIBLE_DEVICES=" + self.device_visibility,
                *self.barrier_argv]
        return {"schema": "jope.root-owned-scope-proposal/v1", "unit": unit,
                "task_sha256": self.task_sha256, "source_sha256": self.source_sha256, "lease_sha256": self.lease_sha256,
                "launch_argv_candidate": args, "show_argv": self.control_argv("show"),
                "root_exec_environment": {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"},
                "privilege_actor": "existing owner-qualified root supervisor, exact new unit only",
                "live_execution_qualified": False, "commands_executed": False,
                "existing_wrapper_modified": False, "shared_service_or_ancestor_write": False,
                "start_order": ["Verify pinned supervisor/barrier/executable identities and current exact task resource grant; root exec uses only root_exec_environment, never inherited LD/PYTHON/sudo-preserved variables",
                                "Reject existing unit name; verify shared ancestor health without changing it",
                                "Arm independent root custodian/watchdog outside the new payload scope",
                                "Create only this scope at startup quota; nonroot barrier must not exec payload",
                                "Read actual ControlGroup, device/inode, members/starts, cpuset/quota/RAM/swap/runtime bounds",
                                "Freeze/read back only this new leaf; bind freeze ownership and Guard facts",
                                "Require three consecutive safe host intervals and all health/watchdog conditions",
                                "Set bounded quota then read actual cpu.max before releasing attested own freeze and barrier"],
                "readback_contract": ["Cgroup path from exact unit show; pin path/device/inode, exact PID/start roster, no unknown descendants",
                                      "Read cpu.max and effective cpuset; configured properties alone are insufficient",
                                      "Freeze/thaw requires actual cgroup.events frozen state and bounded timeout; exit0 is insufficient",
                                      "External/ancestor freeze must never be released; no unit patterns or shared property changes",
                                      "A failed write/readback or missing controller heartbeat keeps the payload barrier closed; custodian freezes only original owned leaf"],
                "remaining_blockers": ["Exact-unit root control grant and independent custodian/watchdog implementation/readback proof",
                                       "Actual trusted start-barrier and root-supervisor executable closure; neither supplied here",
                                       "Current native TRAIN data/evaluator/teaching/runtime/scientific protocol and GPU memory admission"],
                "systemd_version_required": 255, "references": SYSTEMD_255_REFERENCES,
                "limits": "Rendered argv is not a privilege grant, installed supervisor, containment proof or numerical launch. systemctl unit jobs may auto-thaw; freeze is not durable protection against unrelated unit operations."}


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


# Live code below is deliberately restricted to one inert process, <=0.25 CPU,
# <=256MiB payload RAM, <=5s inert work and <=60s total qualification lifetime.
# Root staging under /run is supplied by the resource owner; this code does not
# create/chown/delegate a privileged directory or modify any installed service.
GRANT_KEYS = {"schema", "task_sha256", "source_sha256", "nonce", "expires_unix_ns", "uid", "gid",
              "online_ids", "allowed_ids", "memory_high_bytes", "memory_max_bytes", "runtime_seconds",
              "inert_seconds", "temperature_paths", "max_temperature_millic", "min_ram_bytes",
              "min_disk_bytes", "disk_path", "binary_sha256", "review_receipt_sha256", "resource_lease_id"}
BINARIES = ("/usr/bin/systemctl", "/usr/bin/systemd-run", "/usr/bin/setpriv", "/usr/bin/env", "/usr/bin/python3")


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    obj = json.loads(raw, object_pairs_hook=pairs,
                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    json.dumps(obj, allow_nan=False)
    return obj


def grant_shape(doc, now_unix_ns):
    need(type(doc) is dict and set(doc) == GRANT_KEYS, "closed inert grant schema")
    need(doc["schema"] == "jope.owned-inert-qualification/v1", "inert qualification only")
    for k in ("task_sha256", "source_sha256", "review_receipt_sha256"):
        need(type(doc[k]) is str and re.fullmatch(r"[0-9a-f]{64}", doc[k]) and doc[k] != "0"*64, "bound grant hash")
    need(type(doc["resource_lease_id"]) is str and re.fullmatch(r"[A-Za-z0-9_-]{8,128}", doc["resource_lease_id"]), "actual finite resource lease required")
    need(integer(doc["expires_unix_ns"]) and 0 < doc["expires_unix_ns"]-now_unix_ns <= 60_000_000_000, "grant expired or >60s lifetime")
    need(integer(doc["inert_seconds"], 1) and doc["inert_seconds"] <= 5, "inert work <=5s")
    need(integer(doc["runtime_seconds"], 1) and doc["runtime_seconds"] <= 60, "scope runtime <=60s")
    need(doc["expires_unix_ns"]-now_unix_ns >= (doc["runtime_seconds"]+3)*1_000_000_000,
         "unit lifetime must fit original resource expiry with setup margin")
    need(doc["memory_max_bytes"] <= 256*1024**2 and len(doc["allowed_ids"]) <= 2, "inert resource bounds")
    need(integer(doc["min_ram_bytes"]) and doc["min_ram_bytes"] >= 20*1024**3, "preserve >=20GiB available RAM floor")
    need(integer(doc["min_disk_bytes"]) and doc["min_disk_bytes"] >= 150*1024**3, "preserve >=150GiB storage floor")
    need(doc["disk_path"] == "/home/ubuntu/jope-evidence", "fixed evidence filesystem guard")
    need(integer(doc["max_temperature_millic"], 1) and doc["max_temperature_millic"] <= 85000, "temperature ceiling <=85C")
    need(type(doc["temperature_paths"]) is list and 0 < len(doc["temperature_paths"]) <= 16
         and all(type(p) is str and p.startswith("/sys/devices/") and ".." not in PurePosixPath(p).parts for p in doc["temperature_paths"]), "bound canonical temperature sensors")
    need(type(doc["binary_sha256"]) is dict and set(doc["binary_sha256"]) == set(BINARIES)
         and all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v) for v in doc["binary_sha256"].values()), "exact launcher/runtime binary pins")
    # Reuse strict task/nonce/roster/uid/RAM validation, without authorizing it.
    RootScopeCandidate(doc["task_sha256"], doc["source_sha256"], "a"*64, doc["nonce"],
        tuple(doc["online_ids"]), tuple(doc["allowed_ids"]), doc["uid"], doc["gid"], doc["disk_path"],
        ("/usr/bin/python3",), doc["memory_high_bytes"], doc["memory_max_bytes"], doc["runtime_seconds"]).validate()
    return doc


def protected_root_file(path):
    path = Path(path)
    need(path.is_absolute() and str(path) == str(path.resolve()), "canonical protected path required")
    for p in (path, *path.parents):
        st = p.lstat()
        need(not p.is_symlink() and st.st_uid == 0 and not st.st_mode & 0o022, "root ownership/nonwritable ancestry required")
    return path


def load_inert_grant(path):
    need(sys.platform == "linux" and os.geteuid() == 0, "explicit Linux root qualification actor required")
    need(sys.flags.isolated == 1 and sys.flags.no_site == 1, "qualification interpreter requires -I -S")
    path = protected_root_file(path)
    need(path.name == "grant.json" and path.parent.parent == Path("/run/jope-research"), "task-local root staging only")
    raw = path.read_bytes(); need(len(raw) <= 32768, "bounded grant bytes")
    doc = grant_shape(strict_json(raw), time.time_ns())
    need(path.parent.name == doc["nonce"], "grant staging nonce mismatch")
    source = protected_root_file(Path(__file__).resolve())
    need(source.parent == path.parent and source.name == "guarded_job.py"
         and hashlib.sha256(source.read_bytes()).hexdigest() == doc["source_sha256"], "staged exact source pin")
    verify_binary_pins(doc)
    need(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "pidfd custodian support required")
    return doc, path.parent, hashlib.sha256(raw).hexdigest()


def verify_binary_pins(doc):
    need(set(doc["binary_sha256"]) == set(BINARIES),"runtime closure missing binary")
    for name, expected in doc["binary_sha256"].items():
        p = protected_root_file(Path(name).resolve())
        need(hashlib.sha256(p.read_bytes()).hexdigest() == expected, "runtime binary drift")


def append_event(directory, event):
    raw = json.dumps({"monotonic_ns": time.monotonic_ns(), **event}, sort_keys=True, allow_nan=False).encode()+b"\n"
    need(len(raw) <= 32768, "bounded event")
    fd = os.open(str(directory/"events.jsonl"), os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW, 0o600)
    try:
        st = os.fstat(fd); need(st.st_uid == 0 and not st.st_mode & 0o022 and st.st_nlink == 1, "protected journal required")
        need(os.write(fd, raw) == len(raw), "complete event write"); os.fsync(fd)
    finally:
        os.close(fd)


def process_start(pid):
    # comm can contain spaces/parentheses; starttime is field22 after final ')'.
    text = Path(f"/proc/{pid}/stat").read_text()
    return int(text[text.rfind(")")+2:].split()[19])


def timespan_usec(value):
    """Closed finite systemctl show duration grammar; unknown formatting refuses."""
    need(type(value) is str and value != "", "missing finite duration")
    if value.isdecimal(): return int(value)
    factors={"us":1,"ms":1000,"s":1_000_000,"min":60_000_000}
    tokens=value.split();total=0
    for token in tokens:
        match=re.fullmatch(r"([0-9]+)(us|ms|s|min)",token)
        need(match is not None,"unknown/nonfinite duration")
        total+=int(match[1])*factors[match[2]]
    return total


def unit_lifetime(props, runtime_seconds, launch_lower_ns, expiry_mono_ns, now_ns):
    # systemd255 Unit monotonic timestamps are boot-relative microseconds.
    # RuntimeMax stops the scope; a <=1s stop timeout + SIGKILL is the separate
    # finite backstop. Both, including setup latency, must fit original expiry.
    need(props.get("ActiveState") == "active", "scope not active")
    runtime=timespan_usec(props.get("RuntimeMaxUSec"))*1000
    extra=timespan_usec(props.get("RuntimeRandomizedExtraUSec"))*1000
    stop=timespan_usec(props.get("TimeoutStopUSec"))*1000
    raw=props.get("ActiveEnterTimestampMonotonic")
    need(type(raw) is str and raw.isdecimal(),"missing unit activation timestamp")
    start=int(raw)*1000
    need(runtime == runtime_seconds*1_000_000_000 and extra == 0,
         "effective runtime drift/nonfinite random extension")
    need(0 < stop <= 1_000_000_000 and props.get("SendSIGKILL") == "yes", "finite stop backstop missing")
    need(0 < start and launch_lower_ns-1000 <= start <= now_ns,"unit activation outside original launch")
    end=start+runtime
    need(now_ns < end and end+stop <= expiry_mono_ns,"effective unit lifetime exceeds original expiry")
    return end


class PinnedInertLeaf:
    """Pinned one-process inert leaf; path rebinds/children refuse all writes."""
    def unit_properties(self):
        result = subprocess.run(["/usr/bin/systemctl", "--system", "--no-ask-password", "show", self.unit,
            "--property=Id,ControlGroup,Slice,ActiveState,RuntimeMaxUSec,RuntimeRandomizedExtraUSec,TimeoutStopUSec,SendSIGKILL,ActiveEnterTimestampMonotonic", "--no-pager"],
            check=True,capture_output=True,text=True,timeout=2,env={"PATH":"/usr/bin:/bin"})
        pairs=[line.split("=",1) for line in result.stdout.splitlines()]
        need(all(len(x)==2 for x in pairs) and len({x[0] for x in pairs}) == len(pairs),"invalid unit readback")
        return dict(pairs)

    def open_binding(self):
        fd=os.open("/sys/fs/cgroup",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            for part in self.relative.strip("/").split("/"):
                nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                os.close(fd);fd=nxt
            return fd
        except BaseException:
            os.close(fd);raise

    def path_identity(self):
        fd=self.open_binding()
        try:
            st=os.fstat(fd);return st.st_dev,st.st_ino
        finally:os.close(fd)

    def verify_runtime(self,props):
        end=unit_lifetime(props,self.runtime_seconds,self.launch_lower_ns,self.expiry_mono_ns,time.monotonic_ns())
        start=props['ActiveEnterTimestampMonotonic']
        if self.unit_start is not None:need(start == self.unit_start and end == self.runtime_deadline_ns,"unit restarted/runtime changed")
        self.unit_start,self.runtime_deadline_ns=start,end

    def verify_binding(self, *, containment=False):
        props=self.unit_properties()
        need(props.get('Id') == self.unit and props.get('ControlGroup') == self.relative
             and props.get('Slice') == 'jarambit-evo.slice',"unit-to-directory binding changed")
        need(self.path_identity() == (self.device,self.inode),"unit pathname rebound")
        if containment:
            # Expiry/drift must block releases, never an identity-bound freeze
            # of the original unit. A restarted unit is foreign even here.
            need(props.get("ActiveEnterTimestampMonotonic") == self.unit_start,"unit restarted before containment")
        else:self.verify_runtime(props)

    def __init__(self, unit, pid, doc, launch_lower_ns, expiry_mono_ns):
        need(re.fullmatch(r"jope-research-[0-9a-f]{12}-[0-9a-f]{32}\.scope", unit), "exact own unit")
        self.unit,self.relative=unit,"/jarambit.slice/jarambit-evo.slice/"+unit
        self.runtime_seconds,self.launch_lower_ns,self.expiry_mono_ns=doc['runtime_seconds'],launch_lower_ns,expiry_mono_ns
        self.unit_start,self.runtime_deadline_ns=None,None
        props=self.unit_properties()
        need(props.get('Id')==unit and props.get('ControlGroup')==self.relative
             and props.get('Slice')=='jarambit-evo.slice',"exact new scope readback")
        self.verify_runtime(props)
        self.fd=self.open_binding()
        st=os.fstat(self.fd);self.device,self.inode=st.st_dev,st.st_ino
        self.pid,self.start,self.path=pid,process_start(pid),"/sys/fs/cgroup"+self.relative
        try:self.verify()
        except BaseException:os.close(self.fd);raise

    def read(self, name):
        need(name in ("cgroup.procs", "cgroup.events", "cgroup.freeze", "cpu.stat", "cpu.max", "cpuset.cpus.effective", "memory.high", "memory.max", "memory.swap.max", "pids.max"), "allowed leaf read")
        fd = os.open(name, os.O_RDONLY|os.O_NOFOLLOW, dir_fd=self.fd)
        try: return os.read(fd, 32768).decode().strip()
        finally: os.close(fd)

    def verify(self, *, containment=False):
        self.verify_binding(containment=containment)
        st = os.fstat(self.fd)
        need((st.st_dev, st.st_ino) == (self.device, self.inode), "pinned leaf changed")
        need(self.read("cgroup.procs").split() == [str(self.pid)] and process_start(self.pid) == self.start,
             "inert PID/start roster changed")
        for name in os.listdir(self.fd):
            mode=os.stat(name,dir_fd=self.fd,follow_symlinks=False).st_mode
            need(not stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),"unexpected child cgroup/link")

    def write(self, name, value):
        need(name in ("cgroup.freeze", "cpu.max"), "only own freezer/quota writes")
        self.verify(containment=(name == "cgroup.freeze" and value == "1"))
        fd = os.open(name, os.O_WRONLY|os.O_NOFOLLOW, dir_fd=self.fd)
        try: need(os.write(fd, value.encode()) == len(value), "complete control write")
        finally: os.close(fd)

    def freeze(self, frozen):
        need(type(frozen) is bool, "explicit freeze state")
        self.write("cgroup.freeze", "1" if frozen else "0")
        until = time.monotonic()+0.5
        while time.monotonic() < until:
            values = dict(line.split() for line in self.read("cgroup.events").splitlines())
            if values.get("frozen") == str(int(frozen)): return
            time.sleep(0.01)
        raise ValueError("freezer readback timeout")

    def quota(self, quota_usec):
        need(integer(quota_usec, 1) and quota_usec <= 25000, "inert <=0.25 CPU only")
        value = str(quota_usec)+" 100000"; self.write("cpu.max", value)
        need(self.read("cpu.max") == value, "quota readback mismatch")

    def usage(self):
        return int(dict(line.split() for line in self.read("cpu.stat").splitlines())["usage_usec"])

    def frozen(self):
        value=self.read("cgroup.freeze");need(value in ("0","1"),"unknown freezer state")
        return value == "1"

    def ancestor_frozen(self):
        paths=("/sys/fs/cgroup/jarambit.slice/cgroup.freeze", "/sys/fs/cgroup/jarambit.slice/jarambit-evo.slice/cgroup.freeze")
        values=[Path(p).read_text().strip() for p in paths]
        need(all(x in ("0","1") for x in values),"unknown ancestor freezer state")
        return "1" in values


def parse_cpu(text, owned_usage_usec, monotonic_ns):
    lines = text.splitlines(); fields = lines[0].split()
    need(fields[0] == "cpu" and len(fields) >= 9, "Linux aggregate CPU counters")
    ids = tuple(sorted(int(x.split()[0][3:]) for x in lines if re.match(r"cpu[0-9]+ ", x)))
    return CpuPoint(monotonic_ns, ids, tuple(map(int, fields[1:9])), owned_usage_usec)


def host_point(leaf):
    return parse_cpu(Path("/proc/stat").read_text(), leaf.usage(), time.monotonic_ns())


def resource_health(doc, previous_swap):
    mem = dict((a, int(b.split()[0])*1024) for a,b in (x.split(":",1) for x in Path("/proc/meminfo").read_text().splitlines()))
    vm = dict(x.split() for x in Path("/proc/vmstat").read_text().splitlines())
    swap = (int(vm["pswpin"]), int(vm["pswpout"]))
    temps = [int(protected_root_file(p).read_text()) for p in doc["temperature_paths"]]
    disk = os.statvfs(doc["disk_path"])
    return Health(mem["MemAvailable"] >= doc["min_ram_bytes"], previous_swap is not None and swap == previous_swap,
                  all(0 < t <= doc["max_temperature_millic"] for t in temps),
                  disk.f_bavail*disk.f_frsize >= doc["min_disk_bytes"]), swap


def startup_admission(doc):
    before=parse_cpu(Path("/proc/stat").read_text(),0,time.monotonic_ns())
    _,swap=resource_health(doc,None);time.sleep(1)
    after=parse_cpu(Path("/proc/stat").read_text(),0,time.monotonic_ns())
    need(after.online_ids == tuple(doc["online_ids"]),"admission online CPU mismatch")
    view=cpu_interval(before,after,time.monotonic_ns());n=view["online_cpus"]
    need(view["host_fraction"]*n+2 <= Fraction(60,100)*n,"insufficient startup headroom including supervisor/custodian")
    health,_=resource_health(doc,swap)
    need(all(x is True for x in (health.ram_ok,health.swap_ok,health.temperature_ok,health.storage_ok)),"startup health guard")
    need(doc["expires_unix_ns"]-time.time_ns() >= doc["runtime_seconds"]*1_000_000_000,"launch no longer fits resource expiry")


class InertSupervisor:
    """Single controller core; backend is real pinned leaf or test fake.

    A custodian trip is irreversible. Before every thaw it must be excluded by
    the independent custodian protocol; lost/stale controller kills the exact
    controller pidfd before freezing the original leaf, so a late thaw cannot
    undo custodian containment.
    """
    def __init__(self, leaf, guard, *, owned_freeze=False, release_check=None):
        self.leaf, self.guard, self.started, self.stopped = leaf, guard, False, False
        # This process records only its own confirmed transition. A kernel
        # frozen bit by itself never supplies ownership. Unknown holds stop.
        self.owned_freeze=owned_freeze is True
        self.release_check=release_check
        self.frozen_state=True

    def apply(self, plan, custodian_tripped):
        need(not self.stopped and not custodian_tripped, "supervisor stopped/custodian tripped")
        self.leaf.verify()
        try:
            need(self.leaf.frozen() == self.frozen_state,"unexpected external freezer change")
            need(not self.frozen_state or self.owned_freeze,"unknown freeze ownership")
            need(plan["leaf_path"] == self.guard.leaf.path and plan["leaf_inode"] == self.guard.leaf.inode
                 and plan["leaf_device"] == self.guard.leaf.device, "plan leaf mismatch")
            if plan["mode"] != "RUN_CANDIDATE":
                self.leaf.freeze(True)
                self.frozen_state=True
                self.owned_freeze=True
                if plan["mode"] == "STOP": self.stopped = True
                return False
            self.leaf.quota(min(plan["actions"][0]["quota_usec"], 25000))
            self.leaf.verify()
            need(self.leaf.frozen() == self.frozen_state,"external freeze during quota readback")
            release=any(x.get('op') == 'release_own_freeze_after_readback' for x in plan['actions'])
            if release:
                need(self.frozen_state and self.owned_freeze,"no owned freeze transition to release")
                need(callable(self.release_check) and self.release_check() is True,"fresh custodian release proof missing")
                self.leaf.verify()
                need(self.leaf.frozen() is True,"owned freeze changed before release")
                self.leaf.freeze(False)
                self.frozen_state=False;self.owned_freeze=False
            else:
                need(not self.frozen_state,"frozen plan lacks owned release action")
            self.started = True
            return True
        except BaseException:
            self.stopped = True
            # Never mask an unresolved freeze failure as safe containment.
            self.leaf.freeze(True)
            self.frozen_state=True
            raise


def contain_owned_leaf(leaf, parent_pidfd, payload_pidfd, directory, reason):
    # This path is independent of heartbeat/journal output. New pollers avoid
    # reusing a failed monitoring object. Kill-confirm own parent BEFORE freeze.
    parent=select.poll();parent.register(parent_pidfd,select.POLLIN)
    payload=select.poll();payload.register(payload_pidfd,select.POLLIN)
    if not parent.poll(0):
        try:signal.pidfd_send_signal(parent_pidfd,signal.SIGKILL)
        except ProcessLookupError:pass
        need(parent.poll(500),"controller death unconfirmed; containment unresolved")
    event={"role":"custodian","reason":reason}
    if payload.poll(0):event['event']='controller-lost-owned-payload-already-exited'
    else:
        leaf.freeze(True)
        event.update(event='controller-lost-owned-leaf-frozen',pid=leaf.pid,inode=leaf.inode)
    # Containment is attempted first. A broken journal cannot prevent it.
    try:append_event(directory,event)
    except Exception:pass


def custodian_loop(leaf, heartbeat_fd, parent_pidfd, payload_pidfd, directory, deadline_ns, armed_fd):
    reason='heartbeat EOF/stale/deadline or invalid completion'
    try:
        poller=select.poll();poller.register(parent_pidfd,select.POLLIN)
        payload_poller=select.poll();payload_poller.register(payload_pidfd,select.POLLIN)
        leaf.verify();os.write(armed_fd,b"A")
        last=time.monotonic_ns()
        while time.monotonic_ns() < deadline_ns:
            os.write(armed_fd,struct.pack("<Q",time.monotonic_ns()))
            try:
                data=os.read(heartbeat_fd,4096)
                if not all(x in b"HD" for x in data):break
                if b"D" in data:
                    if payload_poller.poll(500):return
                    break
                if data:last=time.monotonic_ns()
                if data == b"":break
            except BlockingIOError:pass
            if poller.poll(0) or time.monotonic_ns()-last > 1_500_000_000:break
            time.sleep(0.05)
    except BaseException as error:
        reason='custodian monitoring failure: '+type(error).__name__
    contain_owned_leaf(leaf,parent_pidfd,payload_pidfd,directory,reason)


def custodian_fresh(raw, now_ns):
    need(type(raw) is bytes and len(raw) >= 8 and len(raw)%8 == 0,"custodian heartbeat missing/truncated")
    stamp=struct.unpack("<Q",raw[-8:])[0]
    need(0 <= now_ns-stamp <= 250_000_000,"custodian heartbeat stale/future")
    return True


def inert_barrier(socket_path, nonce, seconds):
    need(os.geteuid() != 0 and integer(seconds,1) and seconds <= 5, "nonroot finite inert barrier only")
    p = Path(socket_path)
    need(p.name == "barrier.sock" and p.parent.parent == Path("/run/jope-research")
         and p.parent.name == nonce and re.fullmatch(r"[0-9a-f]{32}", nonce), "exact protected barrier socket")
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as conn:
        conn.settimeout(60); conn.connect(str(p)); conn.sendall(("READY:"+nonce).encode())
        need(conn.recv(128) == b"GO", "barrier closed; no work executed")
        deadline = time.monotonic()+seconds; count = 0
        while time.monotonic() < deadline:
            for _ in range(1000): count = (count*1664525+1013904223)&0xffffffff
        conn.sendall(json.dumps({"inert_completed":True,"seconds":seconds,"checksum":count}).encode())


def run_inert_qualification(grant_path):
    doc, directory, grant_sha = load_inert_grant(grant_path)
    # Anchor once; subsequent setup/readback never restarts the grant clock.
    expiry_mono_ns=time.monotonic_ns()+max(0,doc["expires_unix_ns"]-time.time_ns())
    need(not any((directory/n).exists() for n in ("barrier.sock", "events.jsonl")), "qualification refuses reuse")
    source = str(Path(__file__).resolve()); socket_path = directory/"barrier.sock"
    spec = RootScopeCandidate(doc["task_sha256"],doc["source_sha256"],grant_sha,doc["nonce"],tuple(doc["online_ids"]),
        tuple(doc["allowed_ids"]),doc["uid"],doc["gid"],doc["disk_path"],
        ("/usr/bin/python3","-I","-S",source,"barrier",str(socket_path),doc["nonce"],str(doc["inert_seconds"])),
        doc["memory_high_bytes"],doc["memory_max_bytes"],doc["runtime_seconds"])
    proposal = spec.proposal(); root_env = proposal["root_exec_environment"]
    # Existing unit name is a collision, never an invitation to stop/reuse it.
    check = subprocess.run(["/usr/bin/systemctl","--system","show",spec.unit,"--property=LoadState","--value"],
                           capture_output=True,text=True,timeout=2,env=root_env)
    need(check.stdout.strip() == "not-found", "new unit name collision/unknown")
    server = socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET); server.settimeout(3)
    server.bind(str(socket_path)); os.chmod(socket_path,0o666); server.listen(1)
    launcher = None; leaf = None; child = None; heart = None; parent_pidfd = None;payload_pidfd=None
    try:
        append_event(directory,{"event":"qualification-start", "unit":spec.unit,"grant_sha256":grant_sha,"source_sha256":doc["source_sha256"]})
        startup_admission(doc)
        launch_lower_ns=time.monotonic_ns()
        launcher = subprocess.Popen(proposal["launch_argv_candidate"],env=root_env,stdin=subprocess.DEVNULL)
        conn,_ = server.accept()
        with conn:
            pid,uid,gid = struct.unpack("3i",conn.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
            need(uid == doc["uid"] and gid == doc["gid"] and conn.recv(128) == ("READY:"+doc["nonce"]).encode(), "barrier peer identity")
            leaf = PinnedInertLeaf(spec.unit,pid,doc,launch_lower_ns,expiry_mono_ns)
            leaf.freeze(True); leaf.quota(25000)
            need(leaf.read("memory.high") == str(doc["memory_high_bytes"]) and leaf.read("memory.max") == str(doc["memory_max_bytes"])
                 and leaf.read("memory.swap.max") == "0" and leaf.read("pids.max") == "64", "RAM/swap/tasks readback")
            # Initial affinity must contain no CPU outside the explicit grant.
            effective = leaf.read("cpuset.cpus.effective")
            ids = set()
            for part in effective.split(","):
                pair=part.split("-"); ids.update(range(int(pair[0]),int(pair[-1])+1))
            need(ids == set(doc["allowed_ids"]), "effective CPU readback")
            bound = Leaf(grant_sha,leaf.path,leaf.device,leaf.inode,((pid,leaf.start),))
            deadline = min(expiry_mono_ns,leaf.runtime_deadline_ns)
            guard = Guard(bound,deadline_ns=deadline,max_owned_cores=1); core=InertSupervisor(leaf,guard,owned_freeze=True)
            readfd,heart=os.pipe2(os.O_CLOEXEC|os.O_NONBLOCK); parent_pidfd=os.pidfd_open(os.getpid())
            payload_pidfd=os.pidfd_open(pid);arm_read,arm_write=os.pipe2(os.O_CLOEXEC|os.O_NONBLOCK)
            child=os.fork()
            if child == 0:
                os.close(heart);os.close(arm_read);conn.close();server.close()
                try: custodian_loop(leaf,readfd,parent_pidfd,payload_pidfd,directory,deadline,arm_write)
                except BaseException as error:
                    append_event(directory,{"role":"custodian","event":"CONTAINMENT_UNRESOLVED","error":str(error)})
                    os._exit(2)
                os._exit(0)
            os.close(readfd);os.close(arm_write)
            need(select.select([arm_read],[],[],1)[0] and os.read(arm_read,1) == b"A","custodian did not arm")
            def release_ready():
                need(os.waitpid(child,os.WNOHANG)[0] == 0,"custodian exited before release")
                return custodian_fresh(os.read(arm_read,4096),time.monotonic_ns())
            core.release_check=release_ready
            before=host_point(leaf);need(before.online_ids == tuple(doc["online_ids"]),"grant/actual online CPU mismatch")
            _,swap=resource_health(doc,None); released=False
            conn.setblocking(False)
            def completion():
                nonlocal child
                try: done=conn.recv(1024)
                except BlockingIOError: return None
                if not done: return None
                result=strict_json(done)
                need(released and result.get("inert_completed") is True,"inert completion receipt")
                need(launcher.wait(timeout=2) == 0,"scope launcher failed")
                os.write(heart,b"D");_,status=os.waitpid(child,0);child=None
                need(status == 0,"custodian completion failed")
                append_event(directory,{"event":"inert-completed","result":result,"live_training_admitted":False})
                return result
            while time.monotonic_ns() < deadline:
                os.write(heart,b"H"); time.sleep(1)
                need(os.waitpid(child,os.WNOHANG)[0] == 0,"custodian exited unexpectedly")
                custodian_fresh(os.read(arm_read,4096),time.monotonic_ns())
                result=completion()
                if result is not None: return result
                try: after=host_point(leaf)
                except (FileNotFoundError,ProcessLookupError):
                    result=completion()
                    if result is not None: return result
                    raise
                health,swap=resource_health(doc,swap)
                frozen=leaf.frozen();need(frozen == core.frozen_state,"unowned freezer change")
                facts=ScopeFacts(bound,0,True,leaf.ancestor_frozen(),True,True,True,True,frozen,grant_sha if frozen and core.owned_freeze else None)
                # These facts describe THIS verified staged inert runtime, not
                # general host qualification. Verify current roster every step.
                leaf.verify(); plan=guard.observe(before,after,time.monotonic_ns(),facts,health)
                append_event(directory,{"event":"sample","plan":plan,"inert_only":True})
                run=core.apply(plan,False)
                if core.stopped: raise ValueError("guard stopped inert qualification")
                if run and not released: conn.sendall(b"GO");released=True
                before=after
                result=completion()
                if result is not None: return result
            raise ValueError("inert qualification deadline")
    except BaseException as error:
        if leaf is not None:
            try: leaf.freeze(True)
            except BaseException as failure: append_event(directory,{"event":"CONTAINMENT_UNRESOLVED","error":str(failure)})
        append_event(directory,{"event":"qualification-refused","error":str(error)})
        raise
    finally:
        # Heartbeat EOF makes the independent custodian contain only the pinned
        # leaf. No cleanup signal, shared thaw or broad systemctl stop is used.
        if heart is not None: os.close(heart)
        if parent_pidfd is not None: os.close(parent_pidfd)
        if payload_pidfd is not None: os.close(payload_pidfd)
        if leaf is not None: os.close(leaf.fd)
        server.close()


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Owner-granted CPU inert qualification only; no training command")
    sub=parser.add_subparsers(dest="role",required=True)
    p=sub.add_parser("supervisor");p.add_argument("grant")
    p=sub.add_parser("barrier");p.add_argument("socket");p.add_argument("nonce");p.add_argument("seconds",type=int)
    args=parser.parse_args()
    if args.role == "supervisor": run_inert_qualification(args.grant)
    else: inert_barrier(args.socket,args.nonce,args.seconds)
