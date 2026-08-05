#!/usr/bin/env python3
"""Fail-closed, read-only absolute-path preflight for JOPE bundles and jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import stat
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

SCHEMA = "JopeAbsolutePathAudit.v1"
DEFAULT_NODES = ("xbabe0", "xbabe1", "xbabe2", "xbabe3", "xbabe4")
DEFAULT_CURRENT = Path(
    "/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current"
)
STATE_ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
RUNTIME_BUNDLES_ROOT = STATE_ROOT / "runtime-bundles"
MUTABLE_ROOTS = (
    Path("/home/ubuntu/jope-dime"),
    Path("/home/ubuntu/jope-prime"),
    Path("/home/ubuntu/fable_god"),
    Path("/home/ubuntu/sotarena"),
    Path("/tmp"),
)
HOST_ROOTS = (
    Path("/usr/bin"),
    Path("/usr/sbin"),
    Path("/bin"),
    Path("/sbin"),
    Path("/dev"),
    Path("/proc"),
    Path("/sys"),
    Path("/home/ubuntu/smart_cluster/target/release"),
)
UNIT_PATH_KEYS = {
    "ExecStart",
    "ExecStartPre",
    "ExecStartPost",
    "WorkingDirectory",
    "Environment",
    "EnvironmentFile",
}
EXEC_KEYS = {"ExecStart", "ExecStartPre", "ExecStartPost", "command"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ABSOLUTE_RE = re.compile(r"/(?:[A-Za-z0-9._+@%=-]+/?)+")


@dataclass(frozen=True)
class PathUse:
    original: str
    path: str
    source: str
    field: str
    scope: str
    executable: bool = False
    pinned_sha256: str | None = None
    colon_group: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def absolute_tokens(value: str) -> list[str]:
    """Extract every path from commands, assignments, and colon path lists."""

    return [match.group(0).rstrip(".,;:)") for match in ABSOLUTE_RE.finditer(value)]


def logical_systemd_entries(text: str) -> list[tuple[str, str]]:
    """Join systemd continuation lines, rejecting a dangling continuation."""

    logical: list[str] = []
    pending = ""
    for physical in text.splitlines():
        line = physical.rstrip()
        continued = line.endswith("\\")
        fragment = line[:-1].rstrip() if continued else line
        pending = f"{pending} {fragment.lstrip()}".strip() if pending else fragment
        if not continued:
            logical.append(pending)
            pending = ""
    if pending:
        raise ValueError("dangling systemd line continuation")

    entries: list[tuple[str, str]] = []
    for line in logical:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")) or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key in UNIT_PATH_KEYS:
            entries.append((key, value.strip()))
    return entries


def resolve_current(path: str, bundle_root: Path) -> tuple[str, bool]:
    """Resolve a current reference against an exact candidate without reading it."""

    candidate = Path(path)
    prefixes = (DEFAULT_CURRENT, bundle_root.parent / "current")
    for prefix in prefixes:
        if candidate == prefix or is_under(candidate, prefix):
            relative = candidate.relative_to(prefix)
            return str(bundle_root / relative), True
    return path, False


def exact_bundle_path(path: Path, bundle_root: Path) -> bool:
    for container in (RUNTIME_BUNDLES_ROOT, bundle_root.parent):
        if not is_under(path, container):
            continue
        relative = path.relative_to(container)
        if relative.parts and SHA256_RE.fullmatch(relative.parts[0]):
            return True
    return False


def path_use(
    raw: str,
    *,
    source: Path,
    field: str,
    scope: str,
    bundle_root: Path,
    executable: bool = False,
    pinned_sha256: str | None = None,
    colon_group: str | None = None,
) -> PathUse:
    if not raw.startswith("/"):
        raise ValueError(f"path is not absolute: {raw}")
    if ".." in PurePosixPath(raw).parts:
        raise ValueError(f"path traversal is forbidden: {raw}")
    resolved, _ = resolve_current(raw, bundle_root)
    return PathUse(
        original=raw,
        path=resolved,
        source=str(source),
        field=field,
        scope=scope,
        executable=executable,
        pinned_sha256=pinned_sha256,
        colon_group=colon_group,
    )


def _uses_from_value(
    value: str,
    *,
    source: Path,
    field: str,
    scope: str,
    bundle_root: Path,
    executable: bool = False,
    pinned_sha256: str | None = None,
) -> list[PathUse]:
    tokens = absolute_tokens(value)
    group = None
    if len(tokens) > 1 and ":" in value:
        group = f"{source}:{field}"
    return [
        path_use(
            token,
            source=source,
            field=field,
            scope=scope,
            bundle_root=bundle_root,
            executable=executable and index == 0,
            pinned_sha256=pinned_sha256,
            colon_group=group,
        )
        for index, token in enumerate(tokens)
    ]


def collect_unit(path: Path, bundle_root: Path) -> list[PathUse]:
    entries = logical_systemd_entries(path.read_text())
    uses: list[PathUse] = []
    for key, value in entries:
        uses.extend(
            _uses_from_value(
                value,
                source=path,
                field=key,
                scope="unit_control_node",
                bundle_root=bundle_root,
                executable=key in EXEC_KEYS,
            )
        )
    return uses


def _walk_spec(
    value: Any,
    *,
    source: Path,
    bundle_root: Path,
    field: str = "",
) -> Iterable[PathUse]:
    if isinstance(value, dict):
        for key, child in value.items():
            next_field = f"{field}.{key}" if field else key
            yield from _walk_spec(
                child, source=source, bundle_root=bundle_root, field=next_field
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            next_field = f"{field}[{index}]"
            yield from _walk_spec(
                child, source=source, bundle_root=bundle_root, field=next_field
            )
    elif isinstance(value, str):
        yield from _uses_from_value(
            value,
            source=source,
            field=field,
            scope="job_eligible_nodes",
            bundle_root=bundle_root,
            executable=field.endswith(".command") or field == "command",
        )


def collect_spec(path: Path, bundle_root: Path) -> list[PathUse]:
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    uses = list(_walk_spec(document, source=path, bundle_root=bundle_root))

    # Only source fields in actual [[input]] tables are staged by SCQ. A
    # workdir or an unrelated source-shaped key remains a runtime dependency.
    for index, item in enumerate(document.get("input", [])):
        if not isinstance(item, dict) or not isinstance(item.get("source"), str):
            continue
        source_path = item["source"]
        pinned = item.get("sha256")
        replacement = _uses_from_value(
            source_path,
            source=path,
            field=f"input[{index}].source",
            scope="staged_submission_node",
            bundle_root=bundle_root,
            pinned_sha256=pinned if isinstance(pinned, str) else None,
        )
        uses = [
            use
            for use in uses
            if not (
                use.original == source_path
                and use.field.startswith(f"input[{index}]")
                and use.field.endswith(".source")
            )
        ]
        uses.extend(replacement)
    return uses


def load_bundle(bundle_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    digest = bundle_root.name
    if not SHA256_RE.fullmatch(digest):
        raise ValueError("bundle root must end in an exact 64-character SHA-256")
    with (bundle_root / "bundle.json").open() as handle:
        manifest = json.load(handle)
    if manifest.get("bundle_sha256") != digest:
        raise ValueError("bundle manifest identity does not match the exact bundle root")
    inventory: dict[str, dict[str, Any]] = {}
    for row in manifest.get("files", []):
        target = row.get("target")
        if not isinstance(target, str) or Path(target).is_absolute() or ".." in Path(target).parts:
            raise ValueError("bundle inventory contains an unsafe target")
        if target in inventory:
            raise ValueError(f"bundle inventory repeats {target}")
        inventory[target] = row
    return manifest, inventory


def load_services(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != "JopeSelfplayServiceManifest.v3":
        raise ValueError("unsupported service manifest schema")
    active = [row for row in manifest.get("services", []) if row.get("active") is True]
    if not active:
        raise ValueError("service manifest has no active units")
    return manifest, active


def classify(use: PathUse, bundle_root: Path) -> tuple[str | None, str | None]:
    path = Path(use.path)
    if is_under(path, bundle_root):
        return "bundle_local_immutable", None
    if exact_bundle_path(path, bundle_root):
        return "content_addressed_immutable_external", None
    for root in MUTABLE_ROOTS:
        if is_under(path, root):
            return None, f"mutable checkout path is forbidden: {path}"
    if is_under(path, RUNTIME_BUNDLES_ROOT):
        return None, f"runtime bundle path has no exact content identity: {path}"
    if use.scope == "staged_submission_node":
        if not use.pinned_sha256 or not SHA256_RE.fullmatch(use.pinned_sha256):
            return None, f"staged input lacks a valid SHA-256: {path}"
        return "content_addressed_immutable_external", None
    if is_under(path, STATE_ROOT):
        return "runtime_output_parent", None
    if any(is_under(path, root) for root in HOST_ROOTS):
        return "host_executable_or_capability", None
    if ".dependencies.smoke" in use.field or use.field.startswith("dependencies.smoke"):
        return "host_executable_or_capability", None
    return None, f"unclassified absolute path: {path}"


def _local_contract(
    path: Path,
    classification: str,
    inventory: dict[str, dict[str, Any]],
    bundle_root: Path,
    pinned: set[str],
) -> tuple[dict[str, Any], str | None]:
    contract: dict[str, Any] = {}
    try:
        link_info = path.lstat()
    except OSError as error:
        return contract, f"local audit target is missing: {path}: {error}"
    is_symlink = stat.S_ISLNK(link_info.st_mode)
    if is_symlink and classification != "host_executable_or_capability":
        return contract, f"immutable/runtime audit target must not be a symlink: {path}"
    try:
        info = path.stat() if is_symlink else link_info
    except OSError as error:
        return contract, f"dangling symlink audit target: {path}: {error}"
    if stat.S_ISREG(info.st_mode):
        kind = "file"
    elif stat.S_ISDIR(info.st_mode):
        kind = "directory"
    else:
        kind = "other"
    contract.update(
        type=kind,
        mode=stat.S_IMODE(info.st_mode),
        uid=info.st_uid,
        gid=info.st_gid,
        symlink=is_symlink,
    )
    if is_symlink:
        contract["symlink_target"] = os.readlink(path)
    if kind == "file":
        contract.update(size=info.st_size, sha256=sha256_file(path))

    if classification == "bundle_local_immutable" and kind == "file":
        target = str(path.relative_to(bundle_root))
        row = inventory.get(target)
        if row is None:
            return contract, f"bundle-local file is absent from bundle inventory: {target}"
        if row.get("sha256") != contract["sha256"] or row.get("bytes") != contract["size"]:
            return contract, f"local bundle file disagrees with manifest: {target}"
    if classification == "content_addressed_immutable_external" and kind == "file":
        exact_runtime_bundle = exact_bundle_path(path, bundle_root)
        if len(pinned) != 1 and not exact_runtime_bundle:
            return contract, f"immutable input has missing or conflicting pinned hashes: {path}"
        expected = next(iter(pinned)) if len(pinned) == 1 else contract["sha256"]
        if contract["sha256"] != expected:
            return contract, f"staged input SHA-256 mismatch: {path}"
    return contract, None


SHELL_PROBE = r'''
while IFS= read -r name; do
    if [ -L "$name" ] && [ ! -e "$name" ]; then
        printf '%s\t0\tdangling_symlink\n' "$name"
        continue
    fi
    if [ ! -e "$name" ]; then
        printf '%s\t0\tmissing\n' "$name"
        continue
    fi
    if [ -f "$name" ]; then kind=file
    elif [ -d "$name" ]; then kind=directory
    else kind=other
    fi
    metadata=$(stat -Lc '%a %u %g %s' -- "$name") || {
        printf '%s\t0\tstat_failed\n' "$name"
        continue
    }
    set -- $metadata
    mode=$1 uid=$2 gid=$3 size=$4
    if [ -x "$name" ]; then executable=1; else executable=0; fi
    if [ -w "$name" ]; then writable=1; else writable=0; fi
    if [ -L "$name" ]; then
        symlink=1
        symlink_target=$(readlink -- "$name") || symlink_target=readlink_failed
    else
        symlink=0
        symlink_target=-
    fi
    if [ "$kind" = file ]; then
        sha256=$(sha256sum -- "$name" | cut -d ' ' -f 1) || sha256=hash_failed
    else
        sha256=-
    fi
    printf '%s\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$kind" "$mode" "$uid" "$gid" "$executable" "$writable" \
        "$size" "$sha256" "$symlink" "$symlink_target"
done
'''


def probe_nodes(
    nodes: list[str], paths: list[str]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str]]:
    results: dict[str, dict[str, dict[str, Any]]] = {}
    failures: dict[str, str] = {}
    local_names = {"local", socket.gethostname(), socket.gethostname().split(".", 1)[0]}
    payload = "".join(f"{path}\n" for path in paths)
    for node in nodes:
        if node in local_names:
            command = ["/bin/sh", "-c", SHELL_PROBE]
        else:
            remote = f"/bin/sh -c {shlex.quote(SHELL_PROBE)}"
            command = [
                "/usr/bin/ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                node,
                remote,
            ]
        try:
            done = subprocess.run(
                command,
                input=payload,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            failures[node] = f"{type(error).__name__}: {error}"
            continue
        if done.returncode != 0:
            failures[node] = f"probe exited {done.returncode}: {done.stderr.strip()}"
            continue
        try:
            decoded: dict[str, dict[str, Any]] = {}
            for line in done.stdout.splitlines():
                columns = line.split("\t")
                if len(columns) < 3:
                    raise ValueError("probe returned a malformed row")
                name, exists, detail = columns[:3]
                if exists == "0":
                    decoded[name] = {"exists": False, "error": detail}
                    continue
                if exists != "1" or len(columns) != 12:
                    raise ValueError("probe returned a malformed present row")
                (
                    kind,
                    mode,
                    uid,
                    gid,
                    executable,
                    writable,
                    size,
                    sha256,
                    symlink,
                    symlink_target,
                ) = columns[2:]
                row: dict[str, Any] = {
                    "exists": True,
                    "type": kind,
                    "mode": int(mode, 8),
                    "uid": int(uid),
                    "gid": int(gid),
                    "executable": executable == "1",
                    "writable": writable == "1",
                    "symlink": symlink == "1",
                }
                if symlink == "1":
                    row["symlink_target"] = symlink_target
                if kind == "file":
                    if not SHA256_RE.fullmatch(sha256):
                        raise ValueError("probe could not hash a regular file")
                    row.update(size=int(size), sha256=sha256)
                decoded[name] = row
            if any(path not in decoded for path in paths):
                raise ValueError("probe omitted requested paths")
            results[node] = decoded
        except ValueError as error:
            failures[node] = f"invalid probe response: {error}"
    return results, failures


def _finding(code: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "detail": detail, **extra}


def audit(
    *,
    bundle_root: Path,
    service_manifest_path: Path,
    specs: list[Path],
    units: list[Path],
    nodes: list[str],
    node_probe: Callable[
        [list[str], list[str]],
        tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str]],
    ] = probe_nodes,
) -> dict[str, Any]:
    started_ms = int(time.time() * 1000)
    findings: list[dict[str, Any]] = []
    manifest, inventory = load_bundle(bundle_root)
    service_manifest, active_services = load_services(service_manifest_path)
    service_manifest_sha256 = sha256_file(service_manifest_path)
    declared_service_sha256 = manifest.get("service_manifest_sha256")
    if (
        declared_service_sha256 is not None
        and declared_service_sha256 != service_manifest_sha256
    ):
        raise ValueError("service manifest does not match the bundle declaration")
    if (
        manifest.get("service_manifest") is not None
        and manifest["service_manifest"] != service_manifest
    ):
        raise ValueError("service manifest content does not match the bundled inventory")
    active_units = {row["unit"] for row in active_services}

    if units:
        unit_by_name = {path.name: path for path in units}
        if set(unit_by_name) != active_units:
            missing = sorted(active_units - set(unit_by_name))
            extra = sorted(set(unit_by_name) - active_units)
            findings.append(
                _finding(
                    "active_unit_inventory_mismatch",
                    "provided units do not exactly match the active service manifest",
                    missing=missing,
                    extra=extra,
                )
            )
    else:
        unit_by_name = {
            name: bundle_root / "runtime/systemd" / name for name in active_units
        }

    uses: list[PathUse] = []
    for row in active_services:
        unit_path = unit_by_name.get(row["unit"])
        if unit_path is None:
            continue
        try:
            uses.extend(collect_unit(unit_path, bundle_root))
        except (OSError, ValueError) as error:
            findings.append(
                _finding("unit_parse_failed", str(error), source=str(unit_path))
            )
        for asset, field in (
            (unit_path, "active_unit_asset"),
            (bundle_root / "runtime/bin" / row["binary"], "active_binary_asset"),
        ):
            try:
                uses.append(
                    path_use(
                        str(asset),
                        source=service_manifest_path,
                        field=field,
                        scope="bundle_sync_all_nodes",
                        bundle_root=bundle_root,
                        executable=field == "active_binary_asset",
                    )
                )
            except ValueError as error:
                findings.append(_finding("invalid_path", str(error)))

    for spec in specs:
        try:
            uses.extend(collect_spec(spec, bundle_root))
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            findings.append(_finding("spec_parse_failed", str(error), source=str(spec)))

    classified: list[tuple[PathUse, str]] = []
    colon_classes: dict[str, set[str]] = {}
    for use in uses:
        classification, error = classify(use, bundle_root)
        if error:
            code = "mutable_checkout_path" if error.startswith("mutable") else "unclassified_path"
            findings.append(
                _finding(code, error, path=use.path, source=use.source, field=use.field)
            )
            continue
        assert classification is not None
        classified.append((use, classification))
        if use.colon_group:
            colon_classes.setdefault(use.colon_group, set()).add(classification)
    for group, classes in colon_classes.items():
        if len(classes) > 1:
            findings.append(
                _finding(
                    "mixed_root_colon_list",
                    "colon-packed path list crosses deployment classes",
                    source_field=group,
                    classes=sorted(classes),
                )
            )

    control_node = socket.gethostname().split(".", 1)[0]
    if control_node not in nodes:
        control_node = nodes[0] if nodes else ""
    grouped: dict[str, list[tuple[PathUse, str]]] = {}
    for use, classification in classified:
        grouped.setdefault(use.path, []).append((use, classification))

    path_rows: list[dict[str, Any]] = []
    contracts: dict[str, dict[str, Any]] = {}
    scopes: dict[str, list[str]] = {}
    for path_string, entries in sorted(grouped.items()):
        classes = {classification for _, classification in entries}
        if len(classes) != 1:
            findings.append(
                _finding(
                    "path_class_conflict",
                    "one path has conflicting deployment classes",
                    path=path_string,
                    classes=sorted(classes),
                )
            )
            continue
        classification = next(iter(classes))
        sync_required = classification == "bundle_local_immutable"
        eligible_required = any(use.scope == "job_eligible_nodes" for use, _ in entries)
        if sync_required or eligible_required:
            scope_nodes = list(nodes)
        else:
            scope_nodes = [control_node]
        scopes[path_string] = scope_nodes
        pinned = {use.pinned_sha256 for use, _ in entries if use.pinned_sha256}
        contract, error = _local_contract(
            Path(path_string), classification, inventory, bundle_root, pinned
        )
        contracts[path_string] = contract
        if error:
            findings.append(_finding("local_contract_failed", error, path=path_string))
        path_rows.append(
            {
                "path": path_string,
                "original_paths": sorted({use.original for use, _ in entries}),
                "current_rewritten": any(use.original != use.path for use, _ in entries),
                "class": classification,
                "execution_scope": sorted({use.scope for use, _ in entries}),
                "required_nodes": scope_nodes,
                "sources": sorted({f"{use.source}:{use.field}" for use, _ in entries}),
                "expected": contract,
                "executable_required": any(use.executable for use, _ in entries),
            }
        )

    probe_results, node_failures = node_probe(nodes, sorted(grouped))
    for node in nodes:
        if node in node_failures:
            findings.append(
                _finding(
                    "mandatory_node_unreachable",
                    node_failures[node],
                    node=node,
                )
            )

    for row in path_rows:
        path_string = row["path"]
        contract = contracts[path_string]
        for node in scopes[path_string]:
            observed = probe_results.get(node, {}).get(path_string)
            if observed is None:
                if node not in node_failures:
                    findings.append(
                        _finding(
                            "node_probe_omitted_path",
                            "node probe did not return the required path",
                            node=node,
                            path=path_string,
                        )
                    )
                continue
            if not observed.get("exists"):
                findings.append(
                    _finding(
                        "missing_target",
                        observed.get("error", "required path does not exist"),
                        node=node,
                        path=path_string,
                    )
                )
                continue
            for key in ("type", "mode", "symlink", "symlink_target"):
                if key in contract and observed.get(key) != contract[key]:
                    findings.append(
                        _finding(
                            f"{key}_mismatch",
                            f"expected {contract[key]!r}, observed {observed.get(key)!r}",
                            node=node,
                            path=path_string,
                        )
                    )
            if row["class"] != "runtime_output_parent" and contract.get("type") == "file":
                for key in ("size", "sha256"):
                    if observed.get(key) != contract.get(key):
                        findings.append(
                            _finding(
                                f"{key}_mismatch",
                                f"expected {contract.get(key)!r}, observed {observed.get(key)!r}",
                                node=node,
                                path=path_string,
                            )
                        )
            if row["executable_required"] and not observed.get("executable"):
                findings.append(
                    _finding(
                        "not_executable",
                        "command path is not executable",
                        node=node,
                        path=path_string,
                    )
                )
            if row["class"] == "runtime_output_parent":
                for key in ("uid", "gid"):
                    if key in contract and observed.get(key) != contract[key]:
                        findings.append(
                            _finding(
                                f"runtime_{key}_mismatch",
                                f"expected {contract[key]!r}, observed {observed.get(key)!r}",
                                node=node,
                                path=path_string,
                            )
                        )
                if contract.get("type") == "directory" and not observed.get("writable"):
                    findings.append(
                        _finding(
                            "runtime_parent_not_writable",
                            "runtime-output parent is not writable",
                            node=node,
                            path=path_string,
                        )
                    )

    finished_ms = int(time.time() * 1000)
    node_rows = {
        node: {
            "status": "unreachable" if node in node_failures else "checked",
            "error": node_failures.get(node),
            "paths_returned": len(probe_results.get(node, {})),
        }
        for node in nodes
    }
    return {
        "schema": SCHEMA,
        "generated_at_unix_ms": finished_ms,
        "freshness": {
            "status": "fresh",
            "max_age_seconds": 300,
            "audit_duration_ms": finished_ms - started_ms,
        },
        "bundle_sha256": manifest["bundle_sha256"],
        "bundle_root": str(bundle_root),
        "bundle_manifest_sha256": sha256_file(bundle_root / "bundle.json"),
        "service_manifest": str(service_manifest_path),
        "service_manifest_sha256": service_manifest_sha256,
        "active_units": sorted(active_units),
        "specs": [str(path) for path in specs],
        "required_nodes": nodes,
        "control_node": control_node,
        "nodes": node_rows,
        "paths": path_rows,
        "findings": findings,
        "healthy": not findings,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--service-manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, action="append", default=[])
    parser.add_argument("--unit", type=Path, action="append", default=[])
    parser.add_argument("--node", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_cli_nodes(nodes: list[str]) -> None:
    if len(nodes) != len(DEFAULT_NODES) or set(nodes) != set(DEFAULT_NODES):
        raise ValueError(
            "--node must name exactly xbabe0, xbabe1, xbabe2, xbabe3, and xbabe4"
        )


def main() -> int:
    args = parse_args()
    nodes = args.node or list(DEFAULT_NODES)
    try:
        validate_cli_nodes(nodes)
        report = audit(
            bundle_root=args.bundle_root,
            service_manifest_path=args.service_manifest,
            specs=args.spec,
            units=args.unit,
            nodes=nodes,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        now_ms = int(time.time() * 1000)
        report = {
            "schema": SCHEMA,
            "generated_at_unix_ms": now_ms,
            "freshness": {"status": "fresh", "max_age_seconds": 300},
            "bundle_root": str(args.bundle_root),
            "required_nodes": nodes,
            "findings": [_finding("audit_configuration_failed", str(error))],
            "healthy": False,
        }
    atomic_json(args.output, report)
    print(
        f"{SCHEMA}: {'green' if report['healthy'] else 'blocked'}; "
        f"{len(report['findings'])} finding(s); receipt={args.output}"
    )
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
