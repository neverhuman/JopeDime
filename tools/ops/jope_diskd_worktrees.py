from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from jope_diskd_constants import (
    CARGO_INCREMENTAL_BINARY_CACHE_NAMES,
    PROTECTED_PARTS,
    PROTECTED_PREFIXES,
    REBUILDABLE_WORKTREE_IGNORED_DIRS,
    REBUILDABLE_WORKTREE_IGNORED_FILES,
    TARGET_EVIDENCE_PARTS,
    TARGET_RETENTION_SECONDS,
    TARGET_EVIDENCE_SUFFIXES,
    WORKTREE_IDLE_SECONDS,
)

def lsof_open_paths(resolved_root: Path, *, user_id: int | None = None) -> set[Path] | None:
    lsof = shutil.which("lsof")
    if not lsof:
        return None
    try:
        priority = ["/usr/bin/ionice", "-c3", "/usr/bin/nice", "-n", "10"] if sys.platform.startswith("linux") else ["/usr/bin/nice", "-n", "10"]
        command = priority + [lsof, "-nP"]
        if user_id is not None:
            command.extend(["-a", "-u", str(user_id)])
        command.extend(["-F", "n", "+D", str(resolved_root)])
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in proc.stderr.splitlines():
        if not line.strip() or line.startswith("      Output information may be incomplete."):
            continue
        match = re.search(r"can't stat\(\) .* file system (.+)$", line)
        if not line.startswith("lsof: WARNING:") or not match:
            return None
        mount = Path(match.group(1)).resolve(strict=False)
        if mount == resolved_root or mount in resolved_root.parents or resolved_root in mount.parents:
            return None
    if proc.returncode not in {0, 1}:
        return None
    opened = set()
    for line in proc.stdout.splitlines():
        if not line.startswith("n"):
            continue
        target = Path(line[1:])
        try:
            normalized = target.resolve(strict=False)
            if normalized == resolved_root or resolved_root in normalized.parents:
                opened.add(normalized)
        except OSError:
            return None
    return opened

def under_home(path: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(Path.home().resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False

def protected_path(path: Path) -> bool:
    return any(
        part.lower() in PROTECTED_PARTS or any(part.lower().startswith(prefix) for prefix in PROTECTED_PREFIXES)
        for part in path.parts
    )

def path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False

def detached_worktree_reflog_is_on_main(repo: Path, worktree: Path) -> bool:
    """Do not discard a detached worktree's only reference to non-main history."""
    try:
        reflog = subprocess.run(
            ["git", "-C", str(worktree), "reflog", "show", "--format=%H", "HEAD"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        heads = {line.strip() for line in reflog.stdout.splitlines() if line.strip()}
        if not heads:
            return False
        for head in heads:
            merged = subprocess.run(
                ["git", "-C", str(repo), "merge-base", "--is-ancestor", head, "origin/main"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            if merged.returncode != 0:
                return False
        return True
    except (OSError, subprocess.SubprocessError):
        return False

def safe_target_tree(root: Path, *, now: float, ttl_seconds: int = TARGET_RETENTION_SECONDS) -> bool:
    from jope_diskd_cache import fail_walk

    try:
        root_stat = root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != current_uid():
            return False
        device = root_stat.st_dev
        for directory, dirs, files in os.walk(root, topdown=True, followlinks=False, onerror=fail_walk):
            parent = Path(directory)
            relative_parent = parent.relative_to(root)
            if any(part.lower() in TARGET_EVIDENCE_PARTS for part in relative_parent.parts):
                return False
            parent_stat = parent.lstat()
            if parent_stat.st_dev != device or parent_stat.st_uid != current_uid() or now - parent_stat.st_mtime < ttl_seconds:
                return False
            if not relative_parent.parts:
                if any(name not in {"debug", "release", "tmp"} for name in dirs) or any(name not in {"CACHEDIR.TAG", ".rustc_info.json"} for name in files):
                    return False
            elif len(relative_parent.parts) == 1 and relative_parent.parts[0] in {"debug", "release", "tmp"}:
                if any(name not in {".fingerprint", "build", "deps", "examples", "incremental", "tmp"} for name in dirs):
                    return False
            for name in dirs:
                item = (parent / name).lstat()
                if not stat.S_ISDIR(item.st_mode) or item.st_dev != device or item.st_uid != current_uid():
                    return False
                if name.lower() in TARGET_EVIDENCE_PARTS:
                    return False
            for name in files:
                item = (parent / name).lstat()
                lowered = name.lower()
                if (
                    not stat.S_ISREG(item.st_mode)
                    or item.st_dev != device
                    or item.st_uid != current_uid()
                    or item.st_nlink != 1
                    or now - item.st_mtime < ttl_seconds
                    or any(term in lowered for term in ("receipt", "report", "evidence", "score", "checkpoint"))
                    or Path(name).suffix.lower() in {".csv", ".db", ".html", ".jsonl", ".md", ".npz", ".parquet", ".png", ".sqlite", ".svg", ".txt", ".yaml", ".yml"}
                    or (
                        Path(name).suffix.lower() in TARGET_EVIDENCE_SUFFIXES
                        and not (
                            name.lower() in CARGO_INCREMENTAL_BINARY_CACHE_NAMES
                            and "incremental" in relative_parent.parts
                        )
                    )
                    or (Path(name).suffix.lower() == ".json" and ".fingerprint" not in relative_parent.parts)
                ):
                    return False
        return True
    except OSError:
        return False

def parse_worktree_list(output: str) -> list[dict[str, str | bool]]:
    """Parse git's porcelain worktree inventory without inferring missing fields."""
    entries: list[dict[str, str | bool]] = []
    current: dict[str, str | bool] = {}
    for line in output.splitlines() + [""]:
        if not line:
            if current:
                entries.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current["path"] = value
        elif key == "HEAD":
            current["head"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key == "locked":
            current["locked"] = True
    return entries

def cleanup_disposition(
    *,
    main_ancestor: bool,
    branch: str,
    head: str,
    pr_states: dict[tuple[str, str], str],
) -> str | None:
    if main_ancestor:
        return "merged"
    state = pr_states.get((branch, head), "").upper()
    if state == "MERGED":
        return "merged"
    if state == "CLOSED" and branch:
        return "closed"
    return None

def worktree_has_only_rebuildable_ignored_files(worktree: Path) -> bool:
    """Allow only the target and known disposable caches among ignored entries."""
    try:
        result = subprocess.run(
            ["git", "-C", str(worktree), "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z", "--", "."],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode != 0:
            return False
        device = worktree.stat().st_dev
        for raw in result.stdout.split("\0"):
            relative = raw.rstrip("/")
            if not relative or relative == "target":
                continue
            path = worktree / relative
            item = path.lstat()
            if item.st_uid != current_uid() or item.st_dev != device or stat.S_ISLNK(item.st_mode):
                return False
            if relative in REBUILDABLE_WORKTREE_IGNORED_FILES:
                if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
                    return False
            elif Path(relative).name in REBUILDABLE_WORKTREE_IGNORED_DIRS:
                if not stat.S_ISDIR(item.st_mode):
                    return False
            else:
                return False
        return True
    except (OSError, subprocess.SubprocessError):
        return False

def current_uid() -> int:
    return os.getuid() if hasattr(os, "getuid") else -1

def _handle_cargo_targets_impl(
    _namespace: dict[str, Any],
    entry: dict[str, Any],
    repo: Path,
    policy: dict[str, Any],
    logger: Any,
    *,
    dry_run: bool,
    now: float,
    needed_bytes: int,
) -> int:
    root = _namespace["expand_path"](entry["path"])
    expected = Path.home() / "JopeDime" / ".agent" / "worktrees"
    if not policy.get("local_only") or root.absolute() != expected.absolute() or not root.is_dir():
        print(f"SKIP {entry['id']}: local-only canonical worktree root required")
        return 0
    if needed_bytes <= 0:
        print(f"OK {entry['id']}: low watermark already reached")
        return 0
    ttl_seconds = int(entry.get("ttl_hours", TARGET_RETENTION_SECONDS // 3600)) * 3600
    idle_seconds = int(entry.get("worktree_idle_hours", WORKTREE_IDLE_SECONDS // 3600)) * 3600
    candidates = _namespace["worktree_cleanup_inventory"](
        repo,
        root,
        ttl_seconds,
        now,
        include_closed_pr_worktrees=entry.get("remove_closed_pr_worktrees", False),
        idle_seconds=idle_seconds,
        per_target_cap_bytes=entry.get("per_target_cap_bytes"),
    )
    if candidates is None:
        print(f"SKIP {entry['id']}: worktree cleanup inventory unavailable")
        return 0
    freed = 0
    for candidate in candidates:
        worktree = candidate["worktree"]
        target = candidate["target"]
        branch = candidate["branch"]
        head = candidate["head"]
        remove_worktree = bool(
            candidate["clean"]
            and candidate.get("history_safe", True)
            and (
                (candidate["disposition"] == "merged" and entry.get("remove_merged_worktrees", False))
                or (candidate["disposition"] == "closed" and entry.get("remove_closed_pr_worktrees", False))
            )
        )
        if remove_worktree:
            ignored_safe = candidate.get("ignored_safe")
            if ignored_safe is None:
                ignored_safe = _namespace["worktree_has_only_rebuildable_ignored_files"](worktree)
            remove_worktree = bool(ignored_safe)
        remove_stale_target = bool(
            candidate["stale_target"]
            and candidate.get("target_over_cap", True)
            and candidate["clean"]
            and (
                candidate["disposition"] in {"merged", "closed"}
                or entry.get("remove_stale_targets_on_open_worktrees", False)
            )
        )
        if not remove_worktree and not remove_stale_target:
            if candidate["stale_target"] and not candidate["clean"]:
                print(f"SKIP {target}: worktree has uncommitted changes")
            continue
        if remove_worktree and candidate["target_exists"] and not candidate["stale_target"]:
            print(f"SKIP {worktree}: target is newer than the {ttl_seconds // 3600}h retention")
            remove_worktree = False
        if remove_worktree and candidate["target_exists"] and not _namespace["safe_target_tree"](target, now=now, ttl_seconds=ttl_seconds):
            print(f"SKIP {worktree}: target contains recent, evidence-like, or non-rebuildable content")
            remove_worktree = False
        remove_target_only = remove_stale_target and not remove_worktree
        if not remove_worktree and not remove_target_only:
            continue
        watched_path = worktree if remove_worktree else target
        opened = _namespace["open_handles_under"](watched_path, all_users=False, allow_lsof_fallback=True)
        references = _namespace["local_process_references"](target, worktree)
        if opened is None or opened or references is not False:
            print(f"SKIP {watched_path}: open handles or process references are present/unknown")
            continue
        if candidate["target_exists"] and not _namespace["safe_target_tree"](target, now=now, ttl_seconds=ttl_seconds):
            print(f"SKIP {target}: evidence marker, recent entry, special file, or non-user-owned entry")
            continue
        size = _namespace["path_size"](worktree if remove_worktree else target)
        if size is None:
            print(f"SKIP {watched_path}: cannot measure")
            continue
        action = f"merged/closed worktree {worktree}" if remove_worktree else f"stale Cargo target {target}"
        print(f"{'WOULD REMOVE' if dry_run else 'REMOVE'} {action} ({size} bytes)")
        if not dry_run:
            # Repeat consumer, source, and evidence checks immediately before removal.
            if (
                _namespace["open_handles_under"](watched_path, all_users=False, allow_lsof_fallback=True) != set()
                or _namespace["local_process_references"](target, worktree) is not False
                or not _namespace["worktree_idle_for_cleanup"](worktree, now=time.time(), idle_seconds=idle_seconds)
                or not _namespace["worktree_head_is_unchanged"](
                    worktree, branch, head, require_ignored_safe=remove_worktree,
                )
                or (remove_worktree and not branch and not _namespace["detached_worktree_reflog_is_on_main"](repo, worktree))
                or (candidate["target_exists"] and not _namespace["safe_target_tree"](target, now=time.time(), ttl_seconds=ttl_seconds))
            ):
                print(f"SKIP {watched_path}: candidate changed before removal")
                continue
            if candidate["target_exists"]:
                target_size = _namespace["path_size"](target)
                if target_size is None:
                    print(f"SKIP {target}: cannot remeasure target")
                    continue
                logger({"event": "delete_intent_tree", "path": str(target), "bytes": target_size, "policy": entry["id"]})
                shutil.rmtree(target)
                logger({"event": "deleted_tree", "path": str(target), "bytes": target_size, "policy": entry["id"]})
            if remove_worktree:
                # Git's non-force remove is the final guard against a lock, dirt, or
                # a concurrent write. Never delete the local branch ref.
                if (
                    _namespace["open_handles_under"](worktree, all_users=False, allow_lsof_fallback=True) != set()
                    or _namespace["local_process_references"](target, worktree) is not False
                    or not _namespace["worktree_idle_for_cleanup"](worktree, now=time.time(), idle_seconds=idle_seconds)
                    or not _namespace["worktree_head_is_unchanged"](worktree, branch, head)
                ):
                    print(f"SKIP {worktree}: source became active after target removal")
                    continue
                logger({"event": "remove_worktree_intent", "path": str(worktree), "head": head, "disposition": candidate["disposition"], "policy": entry["id"]})
                try:
                    subprocess.run(
                        ["git", "-C", str(repo), "worktree", "remove", str(worktree)],
                        check=True, capture_output=True, text=True, timeout=30,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    logger({"event": "remove_worktree_failed", "path": str(worktree), "head": head, "error": str(exc), "policy": entry["id"]})
                    print(f"SKIP {worktree}: git worktree remove refused; source retained")
                    continue
                logger({"event": "removed_worktree", "path": str(worktree), "head": head, "disposition": candidate["disposition"], "policy": entry["id"]})
        freed += size
        if freed >= needed_bytes:
            break
    print(f"CARGO-TARGETS {entry['id']}: planned_bytes={freed}")
    return freed
