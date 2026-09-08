#!/usr/bin/env python3
"""Manifested cleanup for rebuildable DLGP campaign storage only."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "JopeDlgpCleanupManifest.v1"
ACTIVE_TOKENS = (
    "cargo",
    "rustc",
    "jope_strategy_screen",
    "jope_dlgp",
    "scq",
    "evaluator",
    "artifact-finalizer",
    "artifact_finalizer",
)
PROTECTED_PARTS = (".git", "sotarena", "lightning_logs", "checkpoints")
QUARANTINED_ROOTS = (
    Path("/home/ubuntu/TO_DELETE"),
    Path("/home/ubuntu/.codex"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def active_candidate_processes(candidates: list[Path]) -> list[dict[str, Any]]:
    output = subprocess.run(
        ["/usr/bin/ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    ).stdout
    own_pid = os.getpid()
    active = []
    for line in output.splitlines():
        pid_text, _, command = line.strip().partition(" ")
        if not pid_text.isdigit() or int(pid_text) == own_pid:
            continue
        lowered = command.lower()
        if "dlgp_safe_cleanup.py" in lowered or not any(token in lowered for token in ACTIVE_TOKENS):
            continue
        pid = int(pid_text)
        resolved_candidates = [candidate.resolve() for candidate in candidates]
        command_uses_candidate = any(str(candidate) in command for candidate in resolved_candidates)
        try:
            cwd = Path(f"/proc/{pid}/cwd").resolve()
        except OSError:
            cwd = None
        build_tool = "cargo" in lowered or "rustc" in lowered
        cwd_uses_candidate = cwd is not None and any(
            cwd == candidate
            or candidate in cwd.parents
            or (build_tool and cwd != Path("/") and candidate.is_relative_to(cwd))
            for candidate in resolved_candidates
        )
        fd_uses_candidate = False
        try:
            for descriptor in Path(f"/proc/{pid}/fd").iterdir():
                try:
                    opened = descriptor.resolve()
                except OSError:
                    continue
                if any(opened == candidate or candidate in opened.parents for candidate in resolved_candidates):
                    fd_uses_candidate = True
                    break
        except OSError:
            pass
        if command_uses_candidate or cwd_uses_candidate or fd_uses_candidate:
            active.append({"pid": pid, "command": command})
    return active


def verify_evidence_bundle(bundle: Path) -> str:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "JopeDlgpRound2Evidence.v1":
        raise ValueError("cleanup requires a verified round-2 evidence bundle")
    for entry in manifest.get("files", []):
        path = bundle / entry["path"]
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError(f"frozen input drift before/after cleanup: {entry['path']}")
    return manifest["payload_sha256"]


def validate_candidate(path: Path, workspace: Path, sccache_root: Path) -> str:
    resolved = path.resolve()
    incremental = (workspace / "target/debug/incremental").resolve()
    dependencies = (workspace / "target/debug/deps").resolve()
    cache = sccache_root.resolve()
    allowed = {incremental: "rust_incremental", dependencies: "stale_debug_dependencies", cache: "sccache"}
    if resolved not in allowed:
        raise ValueError(f"cleanup target is not an exact rebuildable path: {resolved}")
    if any(root == resolved or root in resolved.parents for root in QUARANTINED_ROOTS):
        raise ValueError(f"cleanup target is quarantined: {resolved}")
    if any(part in PROTECTED_PARTS for part in resolved.parts):
        raise ValueError(f"cleanup target is protected: {resolved}")
    return allowed[resolved]


def prune_old_manifested_logs(root: Path, manifest: Path, min_age_days: int, apply: bool) -> list[str]:
    policy = json.loads(manifest.read_text())
    if policy.get("schema") != "JopeDlgpLogCleanupPolicy.v1" or min_age_days < 1:
        raise ValueError("log cleanup requires a typed policy and positive age")
    now = dt.datetime.now(dt.UTC).timestamp()
    removed = []
    for relative in policy.get("rebuildable_unreferenced_logs", []):
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise ValueError("log cleanup policy escapes its root or names a non-file")
        age_days = (now - path.stat().st_mtime) / 86_400
        if age_days >= min_age_days:
            removed.append(str(path))
            if apply:
                path.unlink()
    return removed


def cleanup(args: argparse.Namespace) -> None:
    workspace = args.workspace.resolve()
    evidence = args.evidence_bundle.resolve()
    evidence_sha256 = verify_evidence_bundle(evidence)
    candidates = []
    if args.incremental:
        candidates.append(workspace / "target/debug/incremental")
    if args.debug_deps:
        candidates.append(workspace / "target/debug/deps")
    if args.sccache:
        candidates.append(args.sccache_root)
    classified = [(path, validate_candidate(path, workspace, args.sccache_root)) for path in candidates]
    processes = active_candidate_processes(candidates)
    if processes:
        raise RuntimeError(f"cleanup blocked by processes using candidate paths: {processes}")

    if args.debug_deps:
        if args.debug_deps_min_age_days < 1:
            raise ValueError("debug dependency cleanup requires a positive age")
        release_binary = workspace / "target/release/jope_strategy_screen"
        evidence_manifest = json.loads((evidence / "manifest.json").read_text())
        if (
            not release_binary.is_file()
            or sha256_file(release_binary) != evidence_manifest["binary_sha256"]
        ):
            raise RuntimeError("debug dependency cleanup requires the preserved pinned release binary")

    disk_before = shutil.disk_usage(workspace)
    records = [
        {"path": str(path.resolve()), "class": kind, "bytes_before": tree_bytes(path)}
        for path, kind in classified
    ]
    log_candidates = []
    if args.logs_root or args.logs_policy:
        if not (args.logs_root and args.logs_policy):
            raise ValueError("log cleanup requires both --logs-root and --logs-policy")
        log_candidates = prune_old_manifested_logs(
            args.logs_root.resolve(), args.logs_policy.resolve(), args.log_min_age_days, args.apply
        )

    if args.apply:
        for path, kind in classified:
            if kind == "sccache":
                subprocess.run([str(args.sccache_binary), "--stop-server"], check=False)
            if kind == "stale_debug_dependencies" and path.is_dir():
                cutoff = dt.datetime.now(dt.UTC).timestamp() - args.debug_deps_min_age_days * 86_400
                for entry in path.rglob("*"):
                    if entry.is_file() and entry.stat().st_mtime < cutoff:
                        entry.unlink()
                for directory in sorted(
                    (entry for entry in path.rglob("*") if entry.is_dir()), reverse=True
                ):
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
            elif path.exists():
                shutil.rmtree(path)
            if kind == "sccache":
                path.mkdir(mode=0o700, parents=True, exist_ok=True)

    for record, (path, _) in zip(records, classified, strict=True):
        record["bytes_after"] = tree_bytes(path)
    if verify_evidence_bundle(evidence) != evidence_sha256:
        raise RuntimeError("hash-frozen evidence identity changed during cleanup")
    disk_after = shutil.disk_usage(workspace)
    report = {
        "schema": SCHEMA,
        "mode": "apply" if args.apply else "audit",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "evidence_payload_sha256": evidence_sha256,
        "active_processes": processes,
        "disk_free_bytes_before": disk_before.free,
        "disk_free_bytes_after": disk_after.free,
        "candidates": records,
        "manifested_log_candidates": log_candidates,
        "quarantined_candidates_untouched": [str(path) for path in QUARANTINED_ROOTS],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workspace", type=Path, required=True)
    result.add_argument("--evidence-bundle", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--incremental", action="store_true")
    result.add_argument("--debug-deps", action="store_true")
    result.add_argument("--debug-deps-min-age-days", type=int, default=7)
    result.add_argument("--sccache", action="store_true")
    result.add_argument("--sccache-root", type=Path, default=Path("/home/ubuntu/.cache/sccache"))
    result.add_argument("--sccache-binary", type=Path, default=Path("/home/ubuntu/.local/bin/sccache"))
    result.add_argument("--logs-root", type=Path)
    result.add_argument("--logs-policy", type=Path)
    result.add_argument("--log-min-age-days", type=int, default=30)
    result.add_argument("--apply", action="store_true")
    return result


if __name__ == "__main__":
    cleanup(parser().parse_args())
