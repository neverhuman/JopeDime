"""Durably fence a legacy dispatch cgroup before registry migration.

Freezing the complete cgroup prevents search children from dispatching another
candidate. Failures after freezing intentionally leave work frozen for recovery.
The persistent start condition is retained until a qualified registry handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def durable(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f"immutable evidence changed: {path}")
        return
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(name)


def receipt(path, value):
    durable(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def read(path):
    return json.loads(Path(path).read_text())


def systemctl(*args):
    return subprocess.check_output(["sudo", "-n", "systemctl", *args], text=True).strip()


def properties(unit):
    text = systemctl(
        "show",
        unit,
        "-p",
        "MainPID",
        "-p",
        "ControlGroup",
        "-p",
        "ActiveState",
        "-p",
        "SubState",
        "-p",
        "Restart",
        "-p",
        "FreezerState",
    )
    return dict(line.split("=", 1) for line in text.splitlines())


def process(pid):
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        return dict(
            pid=pid,
            state=fields[0],
            parent=int(fields[1]),
            group=int(fields[2]),
            birth=fields[19],
            command=(root / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
        )
    except FileNotFoundError:
        return None


def cgroup_pids(cgroup):
    root = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    require(root.is_relative_to("/sys/fs/cgroup") and cgroup not in ("", "/"), "unsafe cgroup")
    return sorted({int(pid) for file in root.rglob("cgroup.procs") for pid in file.read_text().split()})


def frozen(cgroup):
    path = Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "cgroup.events"
    return path.exists() and dict(line.split() for line in path.read_text().splitlines()).get("frozen") == "1"


def wait_for(predicate, message, seconds=15):
    deadline = time.monotonic() + seconds
    while not predicate():
        require(time.monotonic() < deadline, message)
        time.sleep(0.05)


def fence(root, output, unit, expected_pid):
    require(re.fullmatch(r"[A-Za-z0-9_.-]+\.service", unit), "invalid service")
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (output / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request_path = output / "request.json"
        if request_path.exists():
            request = read(request_path)
            require(
                request["unit"] == unit
                and request["pid"] == expected_pid
                and request["driver_sha256"] == sha(__file__),
                "fence recovery identity changed",
            )
        else:
            before = properties(unit)
            identity = process(expected_pid)
            require(
                int(before["MainPID"]) == expected_pid and identity is not None,
                "legacy service identity changed",
            )
            require(before["ControlGroup"] == "/system.slice/" + unit, "unexpected legacy service cgroup")
            request = dict(
                schema="parallel.legacy_fit_fence.v1",
                unit=unit,
                pid=expected_pid,
                process=identity,
                boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                properties=before,
                root=str(root),
                driver_sha256=sha(__file__),
                timestamp=time.time(),
            )
            receipt(request_path, request)
            durable(output / "service.before.txt", (systemctl("cat", unit) + "\n").encode())
        if (output / "complete.json").exists():
            require(int(properties(unit)["MainPID"]) == 0, "legacy dispatcher restarted after fence")
            return read(output / "complete.json")
        group = request["properties"]["ControlGroup"]
        require(
            request["boot"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "host rebooted during fence: manual process reconciliation required",
        )
        override = Path("/etc/systemd/system") / (unit + ".d") / "zzzzzzzz-registry-migration-fence.conf"
        permit = output / "REGISTRY_HANDOFF_REQUIRED"
        require(not permit.exists(), "legacy fit start condition unexpectedly satisfied")
        content = f"# Registry migration fence: {output}\n[Unit]\nConditionPathExists={permit}\n[Service]\nRestart=no\n"
        staged = output / "fence.conf"
        durable(staged, content.encode())
        if override.exists():
            require(override.read_text() == content, "another migration owns service fence")
        else:
            subprocess.run(
                ["sudo", "-n", "install", "-D", "-m", "644", str(staged), str(override)], check=True
            )
        systemctl("daemon-reload")
        current = process(expected_pid)
        if (
            current is not None
            and current["birth"] == request["process"]["birth"]
            and current["state"] not in ("Z", "X")
        ):
            require(int(properties(unit)["MainPID"]) == expected_pid, "unit PID changed before freeze")
            systemctl("freeze", unit)
            wait_for(lambda: frozen(group), "entire legacy cgroup did not freeze")
            require(properties(unit)["Restart"] == "no", "legacy automatic restart remains enabled")
            if not (output / "frozen.json").exists():
                identities = [process(pid) for pid in cgroup_pids(group)]
                require(all(identities), "frozen process set changed during capture")
                copied = []
                paths = {root / "runs/gp-service.json"}
                for identity in identities:
                    args = identity["command"]
                    for flag in ("--manifest", "--output"):
                        if flag not in args:
                            continue
                        path = Path(args[args.index(flag) + 1])
                        if not path.is_absolute():
                            path = root / path
                        require(path.resolve().is_relative_to(root), "legacy output escaped root")
                        if flag == "--manifest":
                            paths.add(path)
                        elif path.is_dir():
                            paths.update(path.glob("*.json"))
                            paths.update(path.glob("*.jsonl"))
                            paths.update(path.glob("*.dsl"))
                        else:
                            paths.add(path)
                for path in sorted(paths):
                    if path.is_file():
                        data = path.read_bytes()
                        digest = hashlib.sha256(data).hexdigest()
                        durable(output / "artifacts" / digest, data)
                        copied.append(dict(path=str(path), sha256=digest, bytes=len(data)))
                receipt(
                    output / "frozen.json",
                    dict(
                        timestamp=time.time(),
                        processes=identities,
                        artifacts=copied,
                        request_sha256=sha(request_path),
                        cgroup=group,
                        interrupted_outcome="seen_unavailable_no_automatic_retry",
                    ),
                )
            snapshot = read(output / "frozen.json")
            require(
                set(cgroup_pids(group)) <= {p["pid"] for p in snapshot["processes"]},
                "uncaptured processes in frozen cgroup",
            )
            receipt(output / "kill-intent.json", dict(frozen_sha256=sha(output / "frozen.json")))
            # SIGKILL is queued while frozen. Unlike service Stop, it cannot run
            # TERM/CONT handlers that might dispatch another candidate.
            for identity in snapshot["processes"]:
                current = process(identity["pid"])
                if current is None:
                    continue
                require(current["birth"] == identity["birth"], "PID was reused inside frozen cgroup")
                fd = os.pidfd_open(identity["pid"])
                try:
                    require(process(identity["pid"])["birth"] == identity["birth"], "PID changed before kill")
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                finally:
                    os.close(fd)
            # SIGKILL can retire a frozen cgroup immediately. systemd then
            # rejects thaw because the unit is already inactive.
            if frozen(group):
                try:
                    systemctl("thaw", unit)
                except subprocess.CalledProcessError:
                    require(
                        int(properties(unit)["MainPID"]) == 0 and not frozen(group),
                        "live cgroup could not be thawed after queued kills",
                    )
        else:
            require((output / "kill-intent.json").exists(), "legacy parent vanished before recorded kill")
        wait_for(
            lambda: all(
                (now := process(p["pid"])) is None or now["birth"] != p["birth"] or now["state"] in ("Z", "X")
                for p in read(output / "frozen.json")["processes"]
            ),
            "legacy descendants remain alive",
        )
        require(int(properties(unit)["MainPID"]) in (0, expected_pid), "replacement unit must be preserved")
        systemctl("stop", unit)
        after = properties(unit)
        require(
            int(after["MainPID"]) == 0 and after["ActiveState"] in ("inactive", "failed"),
            "legacy service not stopped",
        )
        result = dict(
            schema="parallel.legacy_fit_fence_completion.v1",
            timestamp=time.time(),
            request_sha256=sha(request_path),
            frozen_sha256=sha(output / "frozen.json"),
            properties=after,
            legacy_dispatch_fenced=True,
            descendants_reconciled=True,
            permanent_override=str(override),
            override_sha256=sha(override),
            historical_import_complete=False,
            registry_dispatch_enabled=False,
        )
        receipt(output / "complete.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--expected-pid", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(fence(args.root, args.output, args.unit, args.expected_pid), sort_keys=True))
