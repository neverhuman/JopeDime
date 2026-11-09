from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import os
import re
import stat
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from jope_diskd_cache import fail_walk
from jope_diskd_constants import OFFLOAD_PREFIX, PROTECTED_PARTS, TARGET_EVIDENCE_SUFFIXES

from jope_diskd_worktrees import current_uid, parse_worktree_list, protected_path, under_home

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def s3_key(path: Path, allowed_root: Path, host: str, day: dt.date, digest: str) -> str:
    relative = path.relative_to(allowed_root).as_posix()
    safe_host = re.sub(r"[^A-Za-z0-9._-]", "_", host)
    return f"{OFFLOAD_PREFIX}/{safe_host}/{day:%Y%m%d}/{digest}/{relative}"

def sha256_file_gzip(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def jankurai_report_roots(repo: Path, worktrees_root: Path) -> list[tuple[Path, Path]] | None:
    """Return report roots only for the canonical repository and its registered direct worktrees."""
    try:
        canonical_repo = repo.resolve(strict=True)
        canonical_worktrees = (canonical_repo / ".agent" / "worktrees").resolve(strict=True)
        if worktrees_root.resolve(strict=True) != canonical_worktrees:
            return None
        result = subprocess.run(
            ["git", "-C", str(canonical_repo), "worktree", "list", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        roots: list[tuple[Path, Path]] = []
        for item in parse_worktree_list(result.stdout):
            checkout = Path(str(item.get("path", ""))).resolve(strict=True)
            if checkout == canonical_repo:
                roots.append((checkout / "target" / "jankurai" / "reports", checkout))
                continue
            try:
                relative = checkout.relative_to(canonical_worktrees)
            except ValueError:
                continue
            if len(relative.parts) == 1:
                roots.append((checkout / "target" / "jankurai" / "reports", checkout))
        return roots
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

def source_tar_is_eligible(
    path: Path,
    report_root: Path,
    *,
    max_file_bytes: int,
    ttl_seconds: int,
    now: float,
) -> os.stat_result | None:
    """Recognize only aged, large, ordinary input/source.tar files under jankurai reports."""
    try:
        relative = path.relative_to(report_root)
        if len(relative.parts) < 3 or relative.parts[-2:] != ("input", "source.tar"):
            return None
        if report_root.is_symlink() or not under_home(report_root):
            return None
        root_stat = report_root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != current_uid():
            return None
        resolved_root = report_root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
        parent = path.parent
        while parent != report_root:
            parent_stat = parent.lstat()
            if (
                not stat.S_ISDIR(parent_stat.st_mode)
                or stat.S_ISLNK(parent_stat.st_mode)
                or parent_stat.st_uid != current_uid()
                or parent_stat.st_dev != root_stat.st_dev
            ):
                return None
            parent = parent.parent
        item = path.lstat()
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_uid != current_uid()
            or item.st_dev != root_stat.st_dev
            or item.st_nlink != 1
            or item.st_size <= max_file_bytes
            or now - item.st_mtime < ttl_seconds
        ):
            return None
        return item
    except (OSError, ValueError):
        return None

def safe_remote_source_tree(
    path: Path, lane: Path, *, now: float, ttl_seconds: int
) -> bool:
    """Accept only old, ordinary src-* trees with no protected or active content."""
    try:
        root_stat = path.lstat()
        lane_stat = lane.lstat()
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_uid != current_uid()
            or root_stat.st_dev != lane_stat.st_dev
            or not path.name.startswith("src-")
            or now - root_stat.st_mtime < ttl_seconds
            or protected_path(path)
        ):
            return False
        for directory, dirs, files in os.walk(
            path, topdown=True, followlinks=False, onerror=fail_walk
        ):
            parent = Path(directory)
            for name in dirs:
                item_path = parent / name
                item = item_path.lstat()
                if (
                    not stat.S_ISDIR(item.st_mode)
                    or item.st_uid != current_uid()
                    or item.st_dev != root_stat.st_dev
                    or now - item.st_mtime < ttl_seconds
                    or name.lower() in PROTECTED_PARTS
                    or protected_path(item_path)
                    or name.lower() == "target"
                ):
                    return False
            for name in files:
                item_path = parent / name
                item = item_path.lstat()
                if (
                    not stat.S_ISREG(item.st_mode)
                    or item.st_uid != current_uid()
                    or item.st_dev != root_stat.st_dev
                    or item.st_nlink != 1
                    or now - item.st_mtime < ttl_seconds
                    or protected_path(item_path)
                    or Path(name).suffix.lower() in TARGET_EVIDENCE_SUFFIXES
                    or any(
                        term in name.lower()
                        for term in (
                            "receipt",
                            "checkpoint",
                            "dataset",
                            "flywheel",
                            "evidence",
                        )
                    )
                ):
                    return False
        return True
    except (OSError, RuntimeError, ValueError):
        return False

def remote_source_scratch_candidates(
    root: Path, *, now: float, ttl_seconds: int
) -> list[tuple[Path, Path, os.stat_result]] | None:
    """Find direct lane_work/<lane>/src-* trees and old *source.tar files."""
    expected_root = Path.home() / "lane_work"
    try:
        root_stat = root.lstat()
        if (
            root.absolute() != expected_root.absolute()
            or not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_uid != current_uid()
            or root.resolve(strict=True) != root.absolute()
        ):
            return None
        candidates: list[tuple[Path, Path, os.stat_result]] = []
        for lane in root.iterdir():
            lane_stat = lane.lstat()
            if (
                not stat.S_ISDIR(lane_stat.st_mode)
                or lane_stat.st_uid != current_uid()
                or lane_stat.st_dev != root_stat.st_dev
                or protected_path(lane)
            ):
                continue
            for child in lane.iterdir():
                if child.name.startswith("src-") and safe_remote_source_tree(
                    child, lane, now=now, ttl_seconds=ttl_seconds
                ):
                    candidates.append((child, lane, child.lstat()))
            for directory, dirs, files in os.walk(
                lane, topdown=True, followlinks=False, onerror=fail_walk
            ):
                parent = Path(directory)
                dirs[:] = [
                    name
                    for name in dirs
                    if not (parent / name).is_symlink()
                    and not name.startswith("src-")
                    and not protected_path(parent / name)
                ]
                for name in files:
                    if not name.endswith("source.tar"):
                        continue
                    path = parent / name
                    item = path.lstat()
                    if (
                        stat.S_ISREG(item.st_mode)
                        and item.st_uid == current_uid()
                        and item.st_dev == root_stat.st_dev
                        and item.st_nlink == 1
                        and now - item.st_mtime >= ttl_seconds
                        and not protected_path(path)
                    ):
                        candidates.append((path, lane, item))
        return sorted(candidates, key=lambda item: item[2].st_mtime)
    except (OSError, RuntimeError, ValueError):
        return None

def _handle_jankurai_source_tars_impl(
    _namespace: dict[str, Any],
    entry: dict[str, Any],
    repo: Path,
    worktrees_root: Path,
    logger: Any,
    *,
    dry_run: bool,
    now: float,
    needed_bytes: int,
) -> int:
    """Prune only old, oversized, unreferenced `input/source.tar` report inputs."""
    expected_repo = Path.home() / "JopeDime"
    if (
        entry.get("id") != "jankurai-source-tars"
        or entry.get("path") != "~/JopeDime"
        or not entry.get("max_file_bytes")
        or not entry.get("ttl_hours")
        or repo.absolute() != expected_repo.absolute()
        or needed_bytes <= 0
    ):
        print(f"SKIP {entry.get('id', 'jankurai-source-tars')}: canonical root, limits, or disk-pressure budget unavailable")
        return 0
    roots = _namespace["jankurai_report_roots"](repo, worktrees_root)
    if roots is None:
        print("SKIP jankurai source tar: registered report-root inventory unavailable")
        return 0
    max_file_bytes = int(entry["max_file_bytes"])
    ttl_seconds = int(entry["ttl_hours"]) * 3600
    freed = 0
    candidates: list[tuple[Path, Path, Path, os.stat_result]] = []
    for report_root, worktree in roots:
        try:
            if not report_root.is_dir() or report_root.is_symlink() or not _namespace["under_home"](report_root):
                continue
            for directory, dirs, names in os.walk(report_root, topdown=True, followlinks=False, onerror=fail_walk):
                dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
                if "source.tar" not in names:
                    continue
                path = Path(directory) / "source.tar"
                item = _namespace["source_tar_is_eligible"](
                    path,
                    report_root,
                    max_file_bytes=max_file_bytes,
                    ttl_seconds=ttl_seconds,
                    now=now,
                )
                if item is not None:
                    candidates.append((path, report_root, worktree, item))
        except OSError:
            print(f"SKIP reports {report_root}: directory scan failed")
    candidates.sort(key=lambda candidate: candidate[3].st_mtime)
    for path, report_root, worktree, before in candidates:
        if freed >= needed_bytes:
            break
        opened = _namespace["open_handles_under"](report_root, all_users=False, allow_lsof_fallback=True)
        referenced = _namespace["local_process_references"](report_root, worktree)
        if opened is None or any(opened_path.is_dir() for opened_path in opened) or referenced is not False:
            print(f"SKIP {path}: process/open-handle scan is busy or incomplete")
            continue
        resolved = path.resolve(strict=True)
        if resolved in opened:
            print(f"SKIP {path}: file is open")
            continue
        print(f"{'WOULD REMOVE' if dry_run else 'REMOVE'} jankurai source tar {path} ({before.st_size} bytes)")
        if dry_run:
            freed += before.st_size
            continue
        # Recheck identity and consumers immediately before logging/removal.
        opened = _namespace["open_handles_under"](report_root, all_users=False, allow_lsof_fallback=True)
        referenced = _namespace["local_process_references"](report_root, worktree)
        after = _namespace["source_tar_is_eligible"](
            path,
            report_root,
            max_file_bytes=max_file_bytes,
            ttl_seconds=ttl_seconds,
            now=time.time(),
        )
        if (
            opened is None
            or any(opened_path.is_dir() for opened_path in opened)
            or resolved in opened
            or referenced is not False
            or after is None
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            print(f"SKIP {path}: source tar changed or became active before removal")
            continue
        logger({"event": "delete_intent_file", "path": str(path), "bytes": before.st_size, "policy": entry["id"]})
        try:
            path.unlink()
        except OSError as exc:
            logger(
                {
                    "event": "delete_file_failed",
                    "path": str(path),
                    "bytes": before.st_size,
                    "policy": entry["id"],
                    "error": str(exc),
                }
            )
            print(f"SKIP {path}: unlink failed")
            continue
        logger(
            {
                "event": "deleted_file",
                "path": str(path),
                "bytes": before.st_size,
                "policy": entry["id"],
            }
        )
        freed += before.st_size
    print(f"JANKURAI-SOURCE-TARS {entry['id']}: planned_bytes={freed}")
    return freed
def _handle_remote_source_scratch_impl(
    _namespace: dict[str, Any],
    entry: dict[str, Any], logger: Any, *, dry_run: bool, now: float, needed_bytes: int
) -> int:
    """Prune aged remote-test source scratch only under lane_work, on disk pressure."""
    root = Path.home() / "lane_work"
    if (
        entry.get("id") != "remote-source-scratch"
        or entry.get("path") != "~/lane_work"
        or entry.get("ttl_seconds") != 3600
        or needed_bytes <= 0
    ):
        print(
            "SKIP remote source scratch: exact root, one-hour TTL, or disk-pressure budget unavailable"
        )
        return 0
    candidates = _namespace["remote_source_scratch_candidates"](root, now=now, ttl_seconds=3600)
    if candidates is None:
        print("SKIP remote source scratch: lane_work inventory unavailable")
        return 0
    freed = 0
    for path, lane, before in candidates:
        opened = _namespace["open_handles_under"](path, all_users=True, allow_lsof_fallback=True)
        referenced = _namespace["local_process_references"](path, path)
        if opened is None or opened or referenced is not False:
            print(f"SKIP {path}: process/open-handle scan is busy or incomplete")
            continue
        size = _namespace["path_size"](path) if stat.S_ISDIR(before.st_mode) else before.st_size
        if size is None:
            print(f"SKIP {path}: size could not be read")
            continue
        print(
            f"{'WOULD REMOVE' if dry_run else 'REMOVE'} remote source scratch {path} ({size} bytes)"
        )
        if dry_run:
            freed += size
            if freed >= needed_bytes:
                break
            continue
        opened = _namespace["open_handles_under"](path, all_users=True, allow_lsof_fallback=True)
        referenced = _namespace["local_process_references"](path, path)
        try:
            after = path.lstat()
        except OSError:
            continue
        still_safe = (
            opened == set()
            and referenced is False
            and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            and (
                _namespace["safe_remote_source_tree"](path, lane, now=time.time(), ttl_seconds=3600)
                if stat.S_ISDIR(before.st_mode)
                else stat.S_ISREG(after.st_mode)
                and after.st_uid == _namespace["current_uid"]()
                and after.st_dev == before.st_dev
                and after.st_nlink == 1
                and time.time() - after.st_mtime >= 3600
                and not _namespace["protected_path"](path)
            )
        )
        if not still_safe:
            print(f"SKIP {path}: scratch changed or became active before removal")
            continue
        logger(
            {
                "event": "delete_intent_tree"
                if stat.S_ISDIR(before.st_mode)
                else "delete_intent_file",
                "path": str(path),
                "bytes": size,
                "policy": entry["id"],
            }
        )
        try:
            shutil.rmtree(path) if stat.S_ISDIR(before.st_mode) else path.unlink()
        except OSError as exc:
            logger(
                {
                    "event": "delete_scratch_failed",
                    "path": str(path),
                    "bytes": size,
                    "policy": entry["id"],
                    "error": str(exc),
                }
            )
            print(f"SKIP {path}: removal failed")
            continue
        logger(
            {
                "event": "deleted_tree"
                if stat.S_ISDIR(before.st_mode)
                else "deleted_file",
                "path": str(path),
                "bytes": size,
                "policy": entry["id"],
            }
        )
        freed += size
        if freed >= needed_bytes:
            break
    print(f"REMOTE-SOURCE-SCRATCH {entry['id']}: planned_bytes={freed}")
    return freed
