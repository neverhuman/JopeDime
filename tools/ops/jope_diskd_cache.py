from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from jope_diskd_constants import CACHE_METADATA_NAMES, SAFE_CACHE_PATHS
from jope_diskd_worktrees import current_uid, protected_path

def safe_state_dir(home: Path, *, create: bool) -> Path | None:
    state = home / ".local" / "state" / "jope-diskd"
    try:
        if state.absolute() != state.resolve(strict=False):
            return None
        if create:
            state.mkdir(mode=0o700, parents=True, exist_ok=True)
        item = state.lstat()
        if (
            not stat.S_ISDIR(item.st_mode)
            or item.st_uid != current_uid()
            or state.resolve(strict=True) != state.absolute()
        ):
            return None
        return state
    except (OSError, RuntimeError):
        return None

def allocated_bytes(item: os.stat_result) -> int:
    return int(getattr(item, "st_blocks", 0)) * 512 or item.st_size

def fail_walk(exc: OSError) -> None:
    raise exc

def cache_candidates(root: Path, ttl_seconds: int, now: float) -> list[Path] | None:
    """List old ordinary cache files, excluding metadata and Git internals."""
    candidates: list[Path] = []
    try:
        for directory, dirs, files in os.walk(root, topdown=True, followlinks=False, onerror=fail_walk):
            parent = Path(directory)
            for name in list(dirs):
                child = parent / name
                child_st = child.lstat()
                if (
                    stat.S_ISLNK(child_st.st_mode)
                    or child_st.st_uid != current_uid()
                    or protected_path(child)
                    or name.lower() == ".git"
                ):
                    dirs.remove(name)
            for name in files:
                child = parent / name
                item = child.lstat()
                if (
                    stat.S_ISLNK(item.st_mode)
                    or not stat.S_ISREG(item.st_mode)
                    or protected_path(child)
                    or child.name.lower() in CACHE_METADATA_NAMES
                    or any(part.lower() == ".git" for part in child.relative_to(root).parts)
                ):
                    continue
                if item.st_uid != current_uid() or item.st_nlink != 1:
                    continue
                if now - item.st_mtime >= ttl_seconds:
                    candidates.append(child)
    except OSError:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime)

def remove_empty_dirs(root: Path, dry_run: bool, logger: Any) -> None:
    if dry_run:
        return
    directories: list[Path] = []
    for directory, dirs, _files in os.walk(root, topdown=True, followlinks=False):
        path = Path(directory)
        for name in list(dirs):
            child = path / name
            try:
                item = child.lstat()
            except OSError:
                dirs.remove(name)
                continue
            if (
                not stat.S_ISDIR(item.st_mode)
                or item.st_uid != current_uid()
                or name.lower() == ".git"
                or protected_path(child)
            ):
                dirs.remove(name)
        if path != root:
            directories.append(path)
    for path in reversed(directories):
        try:
            item = path.lstat()
            if not stat.S_ISDIR(item.st_mode) or item.st_uid != current_uid() or protected_path(path):
                continue
            logger({"event": "delete_intent_dir", "path": str(path)})
            path.rmdir()
        except OSError:
            continue
        logger({"event": "removed_empty_dir", "path": str(path)})

def _handle_cache_impl(
    _namespace: dict[str, Any],
    entry: dict[str, Any],
    policy: dict[str, Any],
    threshold: int,
    max_bytes: int,
    needed_bytes: int,
    dry_run: bool,
    logger: Any,
    now: float,
) -> int:
    name = entry["id"]
    root = _namespace["expand_path"](entry["path"])
    expected = Path.home() / SAFE_CACHE_PATHS.get(name, Path("__not_allowlisted__"))
    if root.absolute() != expected.absolute() or not _namespace["safe_cache_root"](root, policy):
        print(f"SKIP {name}: cache root not an owned, canonical home directory")
        return 0
    size = _namespace["path_size"](root)
    if size is None:
        print(f"SKIP {name}: could not measure cache")
        return 0
    if size <= max_bytes:
        print(f"OK {name}: {size} bytes <= cap {max_bytes}")
        return 0
    target_bytes = min(max_bytes, max(0, size - needed_bytes))
    opened = _namespace["open_handles_under"](root)
    if opened is None:
        print(f"SKIP {name}: open-handle scan was incomplete")
        return 0
    candidates = _namespace["cache_candidates"](root, threshold, now)
    if candidates is None:
        print(f"SKIP {name}: symlink, non-file, ownership, or traversal uncertainty")
        return 0
    freed = 0
    removed_files = 0
    for path in candidates:
        if size - freed <= target_bytes:
            break
        try:
            st = path.lstat()
            if st.st_uid != _namespace["current_uid"]() or not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                continue
            if path.resolve(strict=False) in opened:
                continue
            amount = _namespace["allocated_bytes"](st)
            if not dry_run or removed_files < 20:
                print(f"{'WOULD REMOVE' if dry_run else 'REMOVE'} {path} ({amount} bytes)")
            if not dry_run:
                logger({"event": "delete_intent", "path": str(path), "bytes": amount, "policy": name})
                path.unlink()
                logger({"event": "deleted", "path": str(path), "bytes": amount, "policy": name})
            freed += amount
            removed_files += 1
        except OSError as exc:
            print(f"SKIP {path}: {exc}")
    if dry_run and removed_files > 20:
        print(f"... {removed_files - 20} additional eligible {name} files omitted")
    print(f"CACHE {name}: eligible_files={removed_files} planned_bytes={freed}")
    if not dry_run:
        _namespace["remove_empty_dirs"](root, dry_run, logger)
    return freed
