#!/usr/bin/env python3
"""Verify observed cgroup caps and immutable host policy from inside a CPU job."""
import importlib.util
import json
from pathlib import Path


def _load_provenance_bind():
    path = Path(__file__).resolve().parent / "provenance_bind.py"
    spec = importlib.util.spec_from_file_location("provenance_bind", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


provenance_bind = _load_provenance_bind()


def verify(policy, memory, cpu, swap, devices):
    role = (policy["host"], policy["lane"])
    if role in {("xbabe1", "ci"), ("xbabe1", "ci2"), ("xbabe2", "ci"), ("xbabe3", "ci"), ("xbabe0", "ci"), ("xbabe4", "ci"), ("xbabe4", "ci2"), ("xbabe4", "ci3"), ("xbabe4", "ci4"), ("xbabe4", "ci5"), ("xbabe4", "ci6"), ("xbabe4", "ci7"), ("xbabe4", "ci8"), ("xbabe4", "ci9"), ("xbabe4", "ci10"), ("xbabe4", "ci11"), ("xbabe4", "ci12")}:
        domain, gib, cpus = "pr", 16, 16
    elif role in {("xbabe2", "full"), ("xbabe3", "full")}:
        domain, gib, cpus = "main", 48, 16
    elif role in {("xbabe1", "guard"), ("xbabe2", "guard"), ("xbabe3", "guard")}:
        domain, gib, cpus = "guard", 2, 2
    else:
        raise ValueError("unapproved host/lane")
    if policy != dict(
        host=role[0],
        lane=role[1],
        domain=domain,
        memory=f"{gib}g",
        cpus=cpus,
        volume_prefix=f"jope-{domain}-{role[1]}",
    ):
        raise ValueError("unexpected immutable launch policy")
    quota, period = cpu.split()
    if (
        memory != str(gib * 1024**3)
        or quota == "max"
        or int(quota) != cpus * int(period)
        or swap != "0"
    ):
        raise ValueError("cgroup limits do not equal the approved CPU budget")
    if devices:
        raise ValueError("CPU container exposes a GPU device or Docker socket")


def main():
    policy = json.loads(Path("/opt/jope-runner/policy.json").read_text())
    cgroup = Path("/sys/fs/cgroup")
    verify(
        policy,
        (cgroup / "memory.max").read_text().strip(),
        (cgroup / "cpu.max").read_text().strip(),
        (cgroup / "memory.swap.max").read_text().strip(),
        list(Path("/dev").glob("nvidia*"))
        + [p for p in (Path("/var/run/docker.sock"),) if p.exists()],
    )
    # Hooks use the same PR-base / exact-head bind as the scripts digest.
    provenance_bind.verify_hooks(Path(".").resolve(), Path("/opt/jope-runner"))
    if policy["host"] == "xbabe4":
        provenance_bind.verify_mac_git(Path(".").resolve(), Path("/opt/jope-runner"))
    print(
        json.dumps(
            {
                "policy": policy,
                "memory_max": (cgroup / "memory.max").read_text().strip(),
                "cpu_max": (cgroup / "cpu.max").read_text().strip(),
                "swap_max": "0",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
