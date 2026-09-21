"""Read-only observations of the running node; never edit services or queues."""

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

UNITS = (
    "jope-parallel-gp",
    "jope-parallel-campaign",
    "jope-parallel-monitor",
    "jope-parallel-resident-gpu",
    "jope-parallel-reports",
    "jope-parallel-console",
    "jope-parallel-gp-next",
    "jope-parallel-controller-next",
)
FENCED_UNITS = (
    "jope-parallel-gp",
    "jope-parallel-campaign",
    "jope-parallel-gp-next",
    "jope-parallel-controller-next",
)
FENCE_COMPLETE = "runs/registry-integration-20260910/live-fence/complete.json"
FENCE_RECEIPT_SHA256 = "e49c37bb6eda8dbf19b30f6821f45d93ab7e1bd769cca8c5bb30f2d22ea08647"
FENCE_DROPIN = Path("/etc/systemd/system/jope-parallel-gp.service.d/zzzzzzzz-registry-migration-fence.conf")
SERVING_STATE = "runs/continuous-learning-v2-20260909/state.json"
RECOVERY_SHA256 = "c2f229c9b8c582e11e0e62ec011a9ccb066820fbcee5e2da9d0ad5d63780160e"
LEGACY_EXECUTABLE_SHA256 = "974e93830b51413187ccb0ac19d1dc878b84a6246a2d0981419feb12d38b7dc2"
COORDINATED_MARKER = "runs/recovery-execution-20260911/QUALIFIED_COLLECTION_ACTIVATION_REQUIRED"
COORDINATED_DROPIN = "zzzzzzzzzz-coordinated-recovery-fence.conf"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def planned_migration_fence(root):
    """Verify the retained fence bytes and current disabled start condition."""
    path = Path(root) / FENCE_COMPLETE
    if not path.is_file() or path.is_symlink() or sha(path) != FENCE_RECEIPT_SHA256:
        return None
    complete = json.loads(path.read_text())
    if (
        complete.get("schema") != "parallel.legacy_fit_fence_completion.v1"
        or complete.get("legacy_dispatch_fenced") is not True
        or complete.get("registry_dispatch_enabled") is not False
        or complete.get("descendants_reconciled") is not True
        or Path(complete.get("permanent_override", "")) != FENCE_DROPIN
        or not FENCE_DROPIN.is_file()
        or FENCE_DROPIN.is_symlink()
        or sha(FENCE_DROPIN) != complete.get("override_sha256")
    ):
        return None
    marker = path.parent / "REGISTRY_HANDOFF_REQUIRED"
    if os.path.lexists(marker):
        return None
    expected = "ConditionPathExists=" + str(marker)
    if expected not in FENCE_DROPIN.read_text().splitlines():
        return None
    return dict(complete, receipt_sha256=sha(path), marker=str(marker), marker_exists=False)


def blocking_condition(conditions, marker):
    return isinstance(conditions, list) and any(
        isinstance(row, list)
        and len(row) == 4
        and row[0] == "ConditionPathExists"
        and type(row[1]) is bool
        and row[1] is False
        and type(row[2]) is bool
        and row[2] is False
        and row[3] == marker
        for row in conditions
    )


def effective_conditions(unit):
    if unit not in FENCED_UNITS:
        raise ValueError("Unknown dispatch unit")
    escaped = (unit + ".service").replace("-", "_2d").replace(".", "_2e")
    response = json.loads(
        subprocess.check_output(
            [
                "busctl",
                "--json=short",
                "get-property",
                "org.freedesktop.systemd1",
                "/org/freedesktop/systemd1/unit/" + escaped,
                "org.freedesktop.systemd1.Unit",
                "Conditions",
            ],
            text=True,
            timeout=5,
        )
    )
    return [row[:4] for row in response.get("data", []) if isinstance(row, list)]


def coordinated_dispatch_fence(root, unit, props):
    """Bind the installed recovery fence to its effective AND condition.

    An absent marker in a file is insufficient: later overrides may clear the
    condition. Inspect systemd's loaded condition and retain both identities.
    """
    if unit not in FENCED_UNITS[1:]:
        raise ValueError("Unknown alternate dispatch unit")
    marker = Path(root) / COORDINATED_MARKER
    dropin = Path("/etc/systemd/system") / (unit + ".service.d") / COORDINATED_DROPIN
    expected = f"[Unit]\nConditionPathExists={marker}\n[Service]\nRestart=no\n"
    if (
        props.get("Restart") != "no"
        or str(dropin) not in props.get("DropInPaths", "").split()
        or any(p.is_symlink() for p in (dropin, *dropin.parents, marker, *marker.parents))
        or not dropin.is_file()
        or dropin.read_text() != expected
        or os.path.lexists(marker)
    ):
        return None
    conditions = effective_conditions(unit)
    if not blocking_condition(conditions, str(marker)):
        return None
    return dict(
        dropin=str(dropin),
        dropin_sha256=sha(dropin),
        marker=str(marker),
        marker_exists=False,
        effective_conditions=conditions,
    )


def _telemetry(root):
    telemetry = {}
    for name in ("hardware.json", "gp-service.json", "campaign-progress.json", "cycle-progress.json"):
        path = Path(root) / "runs" / name
        telemetry[name] = json.loads(path.read_text()) if path.exists() else None
    return telemetry


def dispatch_processes(root):
    """Reconcile actual processes, including descendants outside stopped cgroups."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if not argv:
                continue
            routes = {
                "gp-service",
                "search",
                "evaluate-memory",
                "evaluate",
                "fit-worker",
                "fit",
                "fit-view",
                "evaluate-view",
                "evaluate-worker",
                "registry-executor",
                "registry-service-fit",
            }
            cgroup = (proc / "cgroup").read_text()
            owned = any(unit + ".service" in cgroup for unit in FENCED_UNITS)
            if not owned and set(argv[1:]) & routes:
                executable = (proc / "exe").resolve()
                owned = executable.is_relative_to(root) or sha(proc / "exe") == LEGACY_EXECUTABLE_SHA256
            if owned:
                found.append(
                    {"pid": int(proc.name), "command": argv, "start_stat": (proc / "stat").read_text()}
                )
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def capture(root):
    root = Path(root).resolve()
    stable, telemetry = {}, {}
    errors = []
    fence = planned_migration_fence(root)
    for unit in UNITS:
        properties = subprocess.check_output(
            [
                "systemctl",
                "show",
                unit,
                "--property=MainPID,InvocationID,NRestarts,ActiveState,SubState,ControlGroup,ExecStart,Restart,DropInPaths,Requires,ExecCondition,LoadState",
            ],
            text=True,
        )
        props = dict(line.split("=", 1) for line in properties.splitlines() if "=" in line)
        definition = subprocess.run(["systemctl", "cat", unit], text=True, capture_output=True)
        stable[unit] = {"properties": props, "definition": definition.stdout}
        if definition.returncode and not (
            not fence and unit in FENCED_UNITS[2:] and props.get("LoadState") == "not-found"
        ):
            errors.append("Service definition unavailable: " + unit)
        if props.get("ActiveState") != "active" or int(props.get("MainPID", 0)) <= 0:
            if not (
                (fence and unit in FENCED_UNITS or unit in FENCED_UNITS[2:])
                and int(props.get("MainPID", 0)) == 0
            ):
                errors.append("Required live service is unavailable: " + unit)
            continue
        if fence and unit in FENCED_UNITS or unit in FENCED_UNITS[2:]:
            errors.append("Fenced dispatch role is running: " + unit)
        pid = props["MainPID"]
        proc = Path("/proc") / pid
        command = (proc / "cmdline").read_bytes().decode().strip("\0").split("\0")
        executable = str((proc / "exe").resolve())
        # Only the child-executable route is read from the live environment.
        route = [
            v for v in (proc / "environ").read_bytes().split(b"\0") if v.startswith(b"JOPE_GP_EXECUTABLE=")
        ]
        stable[unit] = {
            "properties": props,
            "definition": definition.stdout,
            "exe": executable,
            "exe_sha256": sha(proc / "exe"),
            "command": command,
            "child_route": [v.decode() for v in route],
        }
        for flag in ("--manifest", "--spec"):
            if flag in command:
                path = Path(command[command.index(flag) + 1])
                path = path if path.is_absolute() else root / path
                stable[str(path)] = sha(path)
                doc = json.loads(path.read_text())
                if isinstance(doc.get("vocabulary"), str):
                    p = root / doc["vocabulary"]
                    stable[str(p)] = sha(p)
        for value in route:
            path = Path(value.decode().split("=", 1)[1])
            stable[str(path)] = sha(path)
    # Snapshot all live source paths, including grammar and the vendored kernel.
    # Mutable campaign progress is observed separately, not required to freeze.
    for name in (
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "src",
        "native",
        "tools",
        "assets/grammar",
        "assets/source/jope-dime/vendor/feat-math",
    ):
        base = root / name
        if not base.exists():
            continue
        paths = [base] if base.is_file() else base.rglob("*")
        for p in paths:
            relative = p.relative_to(root)
            if p.is_file() and not any(
                x in relative.parts for x in ("target", "build", "__pycache__", ".git")
            ):
                stable[str(p)] = sha(p)
    telemetry = _telemetry(root)
    if fence:
        stable["fence"] = fence
        gp = stable["jope-parallel-gp"]
        conditions = json.loads(
            subprocess.check_output(
                [
                    "busctl",
                    "--json=short",
                    "get-property",
                    "org.freedesktop.systemd1",
                    "/org/freedesktop/systemd1/unit/jope_2dparallel_2dgp_2eservice",
                    "org.freedesktop.systemd1.Unit",
                    "Conditions",
                ],
                text=True,
            )
        )
        effective = conditions.get("data", [])
        # A non-triggered, non-negated missing-path condition blocks the unit
        # irrespective of any additional OR conditions or prior condition result.
        blocker = any(
            row[:4] == ["ConditionPathExists", False, False, fence["marker"]]
            for row in effective
            if isinstance(row, list)
        )
        gp["effective_conditions"] = [row[:4] for row in effective]
        if (
            str(FENCE_DROPIN) not in gp["properties"].get("DropInPaths", "").split()
            or gp["properties"].get("Restart") != "no"
            or not blocker
        ):
            errors.append("Effective GP fence differs")
        # A failed condition on a Requires dependency does not block the
        # dependent unit. The campaign must have its own effective fence.
        for unit in FENCED_UNITS[1:]:
            props = stable[unit]["properties"]
            # The coordinated recovery fence uses a missing-path condition;
            # earlier ExecCondition entries were explicitly cleared by overrides.
            try:
                coordinated = coordinated_dispatch_fence(root, unit, props)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                coordinated = None
                errors.append("Alternate dispatch condition unavailable: " + unit + ": " + str(error))
            stable[unit]["coordinated_fence"] = coordinated
            if not coordinated:
                errors.append("Alternate dispatch fence missing: " + unit)
        state = json.loads((root / SERVING_STATE).read_text())
        checkpoint = state.get("serving_checkpoint")
        if not checkpoint or state.get("serving_checkpoint_sha256") != RECOVERY_SHA256:
            errors.append("Serving recovery reference changed")
        else:
            path = (root / checkpoint / "state.json").resolve()
            if not path.is_relative_to(root) or sha(path) != RECOVERY_SHA256:
                errors.append("Serving recovery state bytes changed")
            stable["serving_reference"] = {"checkpoint": checkpoint, "sha256": sha(path)}
        descendants = dispatch_processes(root)
        if descendants:
            errors.append("Owned numerical dispatch descendants survive the fence")
    else:
        descendants = None
    return {
        "schema": "kernel.live-state.v2",
        "custody": "planned_migration" if fence else "live",
        "timestamp": time.time(),
        "stable": stable,
        "telemetry": telemetry,
        "fence": fence,
        "capture_errors": errors,
        "dispatch_processes": descendants,
    }


def compare(before, after):
    old, new = before["stable"], after["stable"]
    changes = [key for key in sorted(old.keys() | new.keys()) if old.get(key) != new.get(key)]
    return {
        "stable_identities_unchanged": not changes,
        "changed": changes,
        "scope": "Live source, service invocation, executable routes, fixed manifests and vocabulary; mutable progress may advance",
    }


def compare_health(before, after, max_age=10):
    """Check observed operational health, without inferring performance or cause.

    Unavailable/stale evidence is inconclusive, never a healthy observation.
    Cumulative counters are comparable only within the same service invocation.
    A verified planned-migration fence certifies a stopped GP; unexpected loss still fails.
    """
    failures, unavailable, observations = [], [], {}
    before_custody = before.get("custody")
    after_custody = after.get("custody")
    for label, snapshot in (("before", before), ("after", after)):
        if snapshot.get("schema") != "kernel.live-state.v2" or snapshot.get("capture_errors") != []:
            failures.append(label + ": complete service capture required")
        for unit in UNITS:
            row = snapshot.get("stable", {}).get(unit, {})
            props = row.get("properties", {})
            expected_stop = unit in FENCED_UNITS[2:] or (
                snapshot.get("custody") == "planned_migration" and unit in FENCED_UNITS
            )
            if not props:
                unavailable.append(label + ": missing service properties: " + unit)
            elif expected_stop:
                if props.get("MainPID") != "0" or props.get("ActiveState") not in ("inactive", "failed"):
                    failures.append(label + ": unexpected dispatch service: " + unit)
            elif (
                props.get("ActiveState") != "active"
                or int(props.get("MainPID", 0)) <= 0
                or not row.get("exe_sha256")
                or not row.get("definition")
            ):
                failures.append(label + ": required live service unavailable: " + unit)
    if before_custody == "planned_migration" or after_custody == "planned_migration":
        if type(max_age) not in (int, float) or not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("A positive finite telemetry age limit is required")
        if before_custody != after_custody:
            failures.append("custody changed between live and planned_migration")
        for label, snapshot in (("before", before), ("after", after)):
            stable = snapshot.get("stable", {})
            fence = snapshot.get("fence") or {}
            if (
                snapshot.get("schema") != "kernel.live-state.v2"
                or fence.get("receipt_sha256") != FENCE_RECEIPT_SHA256
                or fence.get("override_sha256")
                != "67940194f08f23503b44d75b13e5f6add51c4abb06a32487e13fa9d106c8c7f5"
                or fence.get("registry_dispatch_enabled") is not False
            ):
                failures.append(label + ": missing verified original fence")
            marker = fence.get("marker")
            suffix = str(Path(FENCE_COMPLETE).parent / "REGISTRY_HANDOFF_REQUIRED")
            valid_marker = (
                isinstance(marker, str) and marker.startswith("/") and marker.endswith("/" + suffix)
            )
            gp = stable.get("jope-parallel-gp", {})
            gp_props = gp.get("properties", {})
            if (
                not valid_marker
                or fence.get("marker_exists") is not False
                or not blocking_condition(gp.get("effective_conditions"), marker)
                or gp_props.get("Restart") != "no"
                or str(FENCE_DROPIN) not in gp_props.get("DropInPaths", "").split()
            ):
                failures.append(label + ": original effective fence evidence missing or changed")
            for unit in FENCED_UNITS[1:]:
                row = stable.get(unit, {})
                props = row.get("properties", {})
                binding = row.get("coordinated_fence") or {}
                root = (
                    Path(marker).parents[len(Path(FENCE_COMPLETE).parts) - 1] if valid_marker else Path("/")
                )
                expected_marker = str(root / COORDINATED_MARKER)
                expected_dropin = str(
                    Path("/etc/systemd/system") / (unit + ".service.d") / COORDINATED_DROPIN
                )
                expected_bytes = (
                    f"[Unit]\nConditionPathExists={expected_marker}\n[Service]\nRestart=no\n".encode()
                )
                if (
                    not valid_marker
                    or binding.get("marker") != expected_marker
                    or binding.get("dropin") != expected_dropin
                    or binding.get("dropin_sha256") != hashlib.sha256(expected_bytes).hexdigest()
                    or binding.get("marker_exists") is not False
                    or not blocking_condition(binding.get("effective_conditions"), expected_marker)
                    or props.get("Restart") != "no"
                    or expected_dropin not in props.get("DropInPaths", "").split()
                ):
                    failures.append(
                        label + ": alternate effective fence evidence missing or changed: " + unit
                    )
            if snapshot.get("capture_errors") != []:
                failures.extend(
                    label + ": " + str(e) for e in snapshot.get("capture_errors", ["capture incomplete"])
                )
            if snapshot.get("dispatch_processes") != []:
                failures.append(label + ": missing or nonempty process reconciliation")
            if stable.get("serving_reference", {}).get("sha256") != RECOVERY_SHA256:
                failures.append(label + ": serving recovery reference unavailable")
            for unit in UNITS:
                row = stable.get(unit, {})
                props = row.get("properties", {})
                if not row.get("definition") or not props:
                    unavailable.append(label + ": service observation missing: " + unit)
                elif unit in FENCED_UNITS:
                    if props.get("MainPID") != "0" or props.get("ActiveState") not in ("inactive", "failed"):
                        failures.append(label + ": fenced role is not stopped: " + unit)
                elif (
                    props.get("ActiveState") != "active"
                    or int(props.get("MainPID", 0)) <= 0
                    or not row.get("exe_sha256")
                ):
                    failures.append(label + ": required service unavailable: " + unit)
            hardware = snapshot.get("telemetry", {}).get("hardware.json") or {}
            stamp, observed = snapshot.get("timestamp"), hardware.get("timestamp")
            if (
                type(stamp) not in (int, float)
                or type(observed) not in (int, float)
                or not math.isfinite(stamp)
                or not math.isfinite(observed)
                or not -1 <= stamp - observed <= max_age
            ):
                unavailable.append(label + ": stale or missing hardware observation")
            if not isinstance(hardware.get("errors"), list):
                unavailable.append(label + ": hardware errors unavailable")
            elif hardware["errors"]:
                failures.append(label + ": hardware telemetry reports errors")
        if not compare(before, after)["stable_identities_unchanged"]:
            failures.append("fence, service definitions, process identity or serving reference changed")
        if after.get("timestamp", 0) < before.get("timestamp", 0):
            failures.append("snapshot time moved backwards")
        observations["planned_migration"] = True
        return {
            "schema": "kernel.operational-health.v2",
            "status": "failed" if failures else "inconclusive" if unavailable else "passed",
            "failures": failures,
            "unavailable": unavailable,
            "observations": observations,
            "scope": "All required services, original fence, current descendants, serving and hardware",
        }
    if not isinstance(max_age, (int, float)) or not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("A positive finite telemetry age limit is required")

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    gp = []
    for label, snapshot in (("before", before), ("after", after)):
        stamp = snapshot.get("timestamp")
        telemetry = snapshot.get("telemetry")
        if not number(stamp) or not isinstance(telemetry, dict):
            unavailable.append(label + ": missing snapshot timestamp or telemetry")
            gp.append(None)
            continue
        for name in ("gp-service.json", "cycle-progress.json", "hardware.json"):
            record = telemetry.get(name)
            if not isinstance(record, dict) or not number(record.get("timestamp")):
                unavailable.append(label + ": missing timestamp for " + name)
                continue
            age = stamp - record["timestamp"]
            # Integer-second producer timestamps permit up to one second of skew.
            if not -1 <= age <= max_age:
                unavailable.append(label + ": stale or future " + name)
            if name == "hardware.json":
                if not isinstance(record.get("errors"), list):
                    unavailable.append(label + ": hardware errors unavailable")
                elif record["errors"]:
                    failures.append(label + ": hardware telemetry reports errors")
        record = telemetry.get("gp-service.json")
        counters = ("completed_views", "completed_evaluations", "failed_views", "active", "queued")
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("phase"), str)
            or any(type(record.get(k)) is not int or record[k] < 0 for k in counters)
        ):
            unavailable.append(label + ": invalid GP counters")
            gp.append(None)
        else:
            gp.append(record)
    if (
        number(before.get("timestamp"))
        and number(after.get("timestamp"))
        and after["timestamp"] < before["timestamp"]
    ):
        unavailable.append("snapshot time moved backwards")
    invocations = [
        s.get("stable", {}).get("jope-parallel-gp", {}).get("properties", {}).get("InvocationID")
        for s in (before, after)
    ]
    if not all(isinstance(value, str) and value for value in invocations) or invocations[0] != invocations[1]:
        unavailable.append("GP invocation changed or is unavailable; counters cannot be compared")
    elif all(record is not None for record in gp):
        for name in ("completed_views", "completed_evaluations", "failed_views"):
            delta = gp[1][name] - gp[0][name]
            observations[name + "_delta"] = delta
            if delta < 0:
                failures.append(name + " moved backwards within one service invocation")
            elif name == "failed_views" and delta > 0:
                failures.append("new failed views observed")
        observations["active_workers_after"] = gp[1]["active"]
        observations["failed_views_after"] = gp[1]["failed_views"]
        # A legitimate empty/draining queue need not keep all workers active.
        if gp[1]["phase"] == "running" and gp[1]["active"] == 0 and gp[1].get("queued", 0) > 0:
            failures.append("GP reports queued work with no active workers")
    return {
        "schema": "kernel.operational-health.v1",
        "status": "failed" if failures else "inconclusive" if unavailable else "passed",
        "failures": failures,
        "unavailable": unavailable,
        "observations": observations,
        "scope": "Observed health over this interval; not causal attribution or quality/throughput qualification",
    }
