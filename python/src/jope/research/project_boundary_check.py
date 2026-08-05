#!/usr/bin/env python3
"""Verify the versioned JOPE-Dime source and runtime boundary."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


MANIFEST = Path(__file__).resolve().parents[1] / "manifests/project-boundary.v1.json"
EXECUTABLE_OR_CONFIG_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".jsonl",
    ".py",
    ".rs",
    ".service",
    ".sh",
    ".timer",
    ".toml",
    ".yaml",
    ".yml",
}
PROSE_SUFFIXES = {".md", ".txt"}
SELF_DECLARATION_PATHS = {
    Path("manifests/project-boundary.v1.json"),
    Path("manifests/legacy-flywheel-inventory.v1.json"),
    Path("scripts/project_boundary_check.py"),
    Path("scripts/test_project_boundary_check.py"),
}
IGNORED_PARTS = {".git", "target", "__pycache__", ".claude"}


def load_manifest(path: Path = MANIFEST) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema") != "JopeProjectBoundary.v1":
        raise ValueError("project boundary manifest schema drift")
    required = {
        "project",
        "authority",
        "active_source_roots",
        "documentation_roots",
        "archived_roots",
        "jope_flywheel_runtime_roots",
        "task_vocabulary",
        "service_manifest",
        "permitted_dependencies",
        "forbidden_references",
        "forbidden_runtime_dependencies",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"project boundary manifest missing {sorted(missing)}")
    if value["task_vocabulary"] != [
        "regression",
        "binary",
        "multiclass",
        "timeseries",
    ]:
        raise ValueError("project boundary task vocabulary drift")
    runtime_roots = value["jope_flywheel_runtime_roots"]
    if not runtime_roots or any(not row.get("path") or not row.get("role") for row in runtime_roots):
        raise ValueError("project boundary runtime roots are incomplete")
    return value


def _under(relative: Path, declarations: list[str]) -> bool:
    for declaration in declarations:
        root = Path(declaration)
        if relative == root or root in relative.parents:
            return True
    return False


def _tracked_paths(root: Path, declarations: list[str]) -> list[Path]:
    if (root / ".git").is_dir():
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
        return sorted(
            Path(value.decode())
            for value in result.stdout.split(b"\0")
            if value and os.path.lexists(root / value.decode())
        )
    found: set[Path] = set()
    for declaration in declarations:
        base = root / declaration
        if os.path.lexists(base) and (base.is_file() or base.is_symlink()):
            found.add(Path(declaration))
        elif base.is_dir():
            for directory, directories, filenames in os.walk(base, followlinks=False):
                directories[:] = [name for name in directories if name not in IGNORED_PARTS]
                for filename in filenames:
                    found.add((Path(directory) / filename).relative_to(root))
    return sorted(found)


def _contains_path(text: str, path: str) -> bool:
    return re.search(re.escape(path) + r"(?=/|[\s\"'])", text) is not None


def _read_service_manifest(root: Path, boundary: dict) -> tuple[Path, dict]:
    relative = Path(boundary["service_manifest"])
    path = root / relative
    value = json.loads(path.read_text())
    if value.get("schema") != "JopeSelfplayServiceManifest.v3":
        raise ValueError("service manifest schema drift")
    if not isinstance(value.get("binaries"), list) or not isinstance(value.get("services"), list):
        raise ValueError("service manifest inventory is incomplete")
    return relative, value


def _check_active_units(root: Path, boundary: dict, service_manifest: dict) -> list[str]:
    failures: list[str] = []
    known_binaries = set(service_manifest["binaries"])
    runtime_root = Path(boundary["authority"]["runtime_state"])
    bundle_root = runtime_root / "runtime-bundles/current"
    forbidden_runtime = [str(value) for value in boundary["forbidden_runtime_dependencies"]]
    forbidden_services = boundary["forbidden_references"]["service_prefixes"]
    seen_units: set[str] = set()
    for row in service_manifest["services"]:
        unit = row.get("unit")
        binary = row.get("binary")
        if not unit or not binary or binary not in known_binaries:
            failures.append("service manifest contains an invalid unit/binary row")
            continue
        if unit in seen_units:
            failures.append(f"service manifest repeats unit {unit}")
        seen_units.add(unit)
        if any(unit.startswith(prefix) for prefix in forbidden_services):
            failures.append(f"service manifest names forbidden unit {unit}")
        if not row.get("active"):
            continue
        relative = Path("systemd") / unit
        unit_path = root / relative
        if not unit_path.is_file():
            failures.append(f"{relative}: active unit source is missing")
            continue
        text = unit_path.read_text(errors="replace")
        expected_binary = str(bundle_root / "runtime/bin" / binary)
        if expected_binary not in text:
            failures.append(f"{relative}: active unit does not execute bundled {binary}")
        for dependency in forbidden_runtime:
            if _contains_path(text, dependency):
                failures.append(f"{relative}: active unit uses mutable runtime dependency {dependency}")
        for line in text.splitlines():
            if line.startswith("WorkingDirectory="):
                workdir = line.partition("=")[2]
                if not _contains_path(workdir + " ", str(runtime_root)):
                    failures.append(f"{relative}: active unit working directory is outside runtime state")
            if not line.startswith("ExecStart="):
                continue
            for script in re.findall(r"/[^\s\"']+\.(?:py|sh)", line):
                if not str(script).startswith(str(bundle_root) + "/"):
                    failures.append(f"{relative}: active unit executes unbundled script {script}")
    return failures


def check(root: Path, manifest: dict | None = None) -> list[str]:
    manifest = manifest or load_manifest()
    active_roots = [str(value) for value in manifest["active_source_roots"]]
    documentation_roots = [str(value) for value in manifest["documentation_roots"]]
    archived_roots = [str(value) for value in manifest["archived_roots"]]
    declared = active_roots + documentation_roots + archived_roots
    paths = _tracked_paths(root, declared)
    forbidden = manifest["forbidden_references"]
    forbidden_paths = [str(value) for value in forbidden["paths"]]
    forbidden_terms = [str(value) for value in forbidden["artifact_terms"]]
    forbidden_services = [str(value) for value in forbidden["service_prefixes"]]
    failures: list[str] = []

    for relative in paths:
        path = root / relative
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            target = os.readlink(path)
            if os.path.isabs(target):
                failures.append(f"{relative}: tracked symlink has absolute target {target}")
            else:
                resolved = Path(os.path.normpath(str(relative.parent / target)))
                if str(resolved).startswith("../") or resolved == Path(".."):
                    failures.append(f"{relative}: tracked symlink escapes the project root")
            if any(_contains_path(target + " ", value) for value in forbidden_paths):
                failures.append(f"{relative}: tracked symlink targets a forbidden project")
            continue
        if not path.is_file():
            continue
        executable_or_config = relative.suffix in EXECUTABLE_OR_CONFIG_SUFFIXES
        prose = relative.suffix in PROSE_SUFFIXES
        in_documentation = _under(relative, documentation_roots)
        in_archive = _under(relative, archived_roots)
        if executable_or_config and not (_under(relative, active_roots) or in_archive):
            failures.append(f"{relative}: tracked executable/configuration is outside active source roots")
        if not executable_or_config and not prose:
            continue
        if in_archive or (prose and in_documentation):
            continue
        if relative in SELF_DECLARATION_PATHS:
            continue
        text = path.read_text(errors="replace")
        for value in forbidden_paths:
            if _contains_path(text, value):
                failures.append(f"{relative}: forbidden project path {value}")
        for value in forbidden_services + forbidden_terms:
            if value.lower() in text.lower():
                failures.append(f"{relative}: forbidden project reference {value}")

    try:
        _, service_manifest = _read_service_manifest(root, manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"service manifest: {error}")
    else:
        failures.extend(_check_active_units(root, manifest, service_manifest))

        # Active binary entrypoints may not reach into the pre-Dime JOPE source
        # checkout. Script and build defaults are checked at the effective unit
        # boundary above, where explicit bundle overrides are authoritative.
        for row in service_manifest["services"]:
            if not row.get("active"):
                continue
            source = root / "src/bin" / f"{row['binary']}.rs"
            if not source.is_file():
                continue
            text = source.read_text(errors="replace")
            if _contains_path(text, "/home/ubuntu/jope"):
                failures.append(f"{source.relative_to(root)}: active runtime depends on live /home/ubuntu/jope")

    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=MANIFEST.parents[1])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    try:
        failures = check(args.root, load_manifest(args.manifest))
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        print(f"boundary: error: {error}", file=sys.stderr)
        return 2
    if failures:
        print("boundary: FAIL")
        print("\n".join(failures))
        return 1
    print("boundary: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
