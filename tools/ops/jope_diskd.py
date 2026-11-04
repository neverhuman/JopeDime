#!/usr/bin/env python3
"""Conservative, policy-driven disk pressure observer and cache janitor.

The YAML policy is deliberately JSON syntax (a YAML 1.2 subset), so the daemon
has no third-party parser dependency on runner hosts. Dry-run is the default.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
import re
import select
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from jope_diskd_constants import (
    GIB,
    SAFE_CACHE_PATHS,
    SUPPORTED_LOCAL_HOSTS,
    CACHE_METADATA_NAMES,
    REBUILDABLE_WORKTREE_IGNORED_DIRS,
    REBUILDABLE_WORKTREE_IGNORED_FILES,
    LOG_LIMIT_BYTES,
    LOG_RETENTION_SECONDS,
    TARGET_RETENTION_SECONDS,
    WORKTREE_IDLE_SECONDS,
    OFFLOAD_BUCKET,
    OFFLOAD_PREFIX,
    TARGET_EVIDENCE_PARTS,
    TARGET_EVIDENCE_SUFFIXES,
    CARGO_INCREMENTAL_BINARY_CACHE_NAMES,
    PROTECTED_PARTS,
    PROTECTED_PREFIXES,
)
from jope_diskd_worktrees import (
    lsof_open_paths,
    under_home,
    protected_path,
    path_contains,
    detached_worktree_reflog_is_on_main,
    safe_target_tree,
    parse_worktree_list,
    cleanup_disposition,
    worktree_has_only_rebuildable_ignored_files,
    current_uid,
    _handle_cargo_targets_impl,
)
from jope_diskd_sources import (
    sha256_file,
    s3_key,
    sha256_file_gzip,
    jankurai_report_roots,
    source_tar_is_eligible,
    safe_remote_source_tree,
    remote_source_scratch_candidates,
    _handle_jankurai_source_tars_impl,
    _handle_remote_source_scratch_impl,
)
from jope_diskd_cache import (
    safe_state_dir,
    allocated_bytes,
    fail_walk,
    cache_candidates,
    remove_empty_dirs,
    _handle_cache_impl,
)

def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def expand_path(value: str) -> Path:
    return Path(value).expanduser()

def read_policy(path: Path | None) -> dict[str, Any]:
    if path is not None:
        raw = path.read_text(encoding="utf-8")
    else:
        encoded = os.environ.get("JOPE_DISKD_POLICY_B64", "")
        raw = base64.b64decode(encoded, validate=True).decode("utf-8") if encoded else os.environ.get("JOPE_DISKD_POLICY_JSON", "")
    if not raw:
        raise ValueError("policy required: pass --config or JOPE_DISKD_POLICY_JSON")
    policy = json.loads(raw)
    if policy.get("version") != 1:
        raise ValueError("unsupported policy version")
    marks = policy.get("watermarks", {})
    if not (0 < marks.get("soft_percent", 0) < marks.get("aggressive_percent", 0) < marks.get("alert_percent", 0) <= 100):
        raise ValueError("watermarks must be increasing percentages in (0, 100]")
    if not (0 <= marks.get("low_percent", 0) < marks["soft_percent"]):
        raise ValueError("low watermark must be below soft watermark")
    if not isinstance(policy.get("paths"), list):
        raise ValueError("paths must be a list")
    if not isinstance(policy.get("deny_prefixes"), list):
        raise ValueError("deny_prefixes must be a list")
    if policy.get("local_only") is not True:
        raise ValueError("local-only mode is required")
    if not isinstance(policy.get("allow_log_rotation", False), bool):
        raise ValueError("allow_log_rotation must be a boolean")
    if not isinstance(policy.get("allow_report_offload", False), bool):
        raise ValueError("allow_report_offload must be a boolean")
    if not isinstance(policy.get("allow_log_offload", False), bool):
        raise ValueError("allow_log_offload must be a boolean")
    for entry in policy["paths"]:
        kind = entry.get("kind")
        if kind == "cargo-target":
            if "per_target_cap_bytes" in entry and (
                type(entry["per_target_cap_bytes"]) is not int or entry["per_target_cap_bytes"] < 1
            ):
                raise ValueError("cargo-target per_target_cap_bytes must be a positive integer")
            for key in (
                "remove_merged_worktrees",
                "remove_closed_pr_worktrees",
                "remove_stale_targets_on_open_worktrees",
            ):
                if key in entry and not isinstance(entry[key], bool):
                    raise ValueError(f"{key} must be a boolean")
            if "ttl_hours" in entry and (type(entry["ttl_hours"]) is not int or entry["ttl_hours"] < 1):
                raise ValueError("cargo-target ttl_hours must be a positive integer")
            if "worktree_idle_hours" in entry and (
                type(entry["worktree_idle_hours"]) is not int or entry["worktree_idle_hours"] < 1
            ):
                raise ValueError("cargo-target worktree_idle_hours must be a positive integer")
            continue
        if kind == "jankurai-source-tar":
            if (
                entry.get("id") != "jankurai-source-tars"
                or entry.get("path") != "~/JopeDime"
                or type(entry.get("max_file_bytes")) is not int
                or entry["max_file_bytes"] < 1
                or type(entry.get("ttl_hours")) is not int
                or entry["ttl_hours"] < 1
            ):
                raise ValueError("jankurai source tar policy must use its exact canonical root and positive limits")
        if kind == "remote-source-scratch" and (
            entry.get("id") != "remote-source-scratch"
            or entry.get("path") != "~/lane_work"
            or type(entry.get("ttl_seconds")) is not int
            or entry["ttl_seconds"] != 3600
        ):
            raise ValueError("remote source scratch policy must use ~/lane_work and an exact one-hour TTL")
    hosts = policy.get("allowed_hosts")
    if (
        not isinstance(hosts, list)
        or not hosts
        or any(not isinstance(host, str) for host in hosts)
        or len(hosts) != len(set(hosts))
        or not set(hosts).issubset(SUPPORTED_LOCAL_HOSTS)
    ):
        raise ValueError("allowed_hosts must be a unique subset of the exact supported local hosts")
    if policy.get("repo_root") != "~/JopeDime" or policy.get("sccache_max_bytes") != 50 * GIB:
        raise ValueError("repo root or sccache cap differs from the reviewed local policy")
    if policy.get("offload") != {
        "bucket": OFFLOAD_BUCKET,
        "prefix": OFFLOAD_PREFIX,
        "retention_days": 14,
        "sha256_verify_before_remove": True,
    }:
        raise ValueError("offload must match the approved bucket, prefix, and retention")
    return policy

def path_size(path: Path) -> int | None:
    if not path.exists():
        return 0
    try:
        priority = ["/usr/bin/ionice", "-c3", "/usr/bin/nice", "-n", "10"] if sys.platform.startswith("linux") else ["/usr/bin/nice", "-n", "10"]
        result = subprocess.run(
            priority + ["/usr/bin/du", "-sk", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return int(result.stdout.split()[0]) * 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None

def open_handles_under(root: Path, *, all_users: bool = False, allow_lsof_fallback: bool = True) -> set[Path] | None:
    """Return open paths under root, or None unless the scan was complete."""
    resolved_root = root.resolve(strict=True)
    if sys.platform.startswith("linux") and Path("/proc/self/fd").is_dir():
        opened: set[Path] = set()
        complete = True
        try:
            for process in Path("/proc").iterdir():
                if not process.name.isdigit():
                    continue
                try:
                    status = (process / "status").read_text(encoding="utf-8")
                except FileNotFoundError:
                    continue
                uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
                uid_fields = uid_line.split()[1:]
                if not all_users and (len(uid_fields) < 2 or int(uid_fields[1]) != current_uid()):
                    continue
                fds = process / "fd"
                try:
                    fd_list = list(fds.iterdir())
                except FileNotFoundError:
                    continue
                except PermissionError:
                    complete = False
                    continue
                for fd in fd_list:
                    try:
                        target = Path(os.readlink(fd).removesuffix(" (deleted)"))
                        if not target.is_absolute():
                            continue
                        normalized = target.resolve(strict=False)
                        if normalized == resolved_root or resolved_root in normalized.parents:
                            opened.add(normalized)
                    except FileNotFoundError:
                        continue
                    except PermissionError:
                        complete = False
            if complete:
                return opened
            if all_users or not allow_lsof_fallback:
                return None
        except (OSError, StopIteration, ValueError):
            if all_users or not allow_lsof_fallback:
                return None
            pass
        if not allow_lsof_fallback:
            return None
    return lsof_open_paths(resolved_root, user_id=None if all_users else current_uid())

def local_process_references(target: Path, worktree: Path, proc_root: Path = Path("/proc")) -> bool | None:
    """Return busy/clear, or None if local process metadata was not fully readable."""
    if not proc_root.is_dir():
        return None
    try:
        for process in proc_root.iterdir():
            if not process.name.isdigit():
                continue
            try:
                status = (process / "status").read_text(encoding="utf-8")
                uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
                uid_fields = uid_line.split()[1:]
                if len(uid_fields) < 2:
                    return None
                if int(uid_fields[1]) != current_uid():
                    continue
            except FileNotFoundError:
                continue
            except PermissionError:
                return None
            except OSError:
                return None
            try:
                cwd = Path(os.readlink(process / "cwd"))
            except FileNotFoundError:
                continue
            except PermissionError:
                cwd = None
            except OSError:
                return None
            try:
                environ = (process / "environ").read_bytes().split(b"\0")
            except FileNotFoundError:
                continue
            except PermissionError:
                environ = []
            except OSError:
                return None
            if cwd is not None and (path_contains(target, cwd) or path_contains(worktree, cwd)):
                return True
            for entry in environ:
                if not entry.startswith(b"CARGO_TARGET_DIR="):
                    continue
                value = Path(os.fsdecode(entry.split(b"=", 1)[1])).expanduser()
                if not value.is_absolute():
                    continue
                if path_contains(target, value) or path_contains(value, target):
                    return True
        return False
    except (OSError, RuntimeError, StopIteration, ValueError):
        return None

def worktree_idle_for_cleanup(worktree: Path, *, now: float, idle_seconds: int) -> bool:
    """Require both per-worktree Git activity markers to be older than the idle window."""
    try:
        result = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "--absolute-git-dir"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        git_dir = Path(result.stdout.strip())
        activity = [git_dir / name for name in ("HEAD", "index")]
        mtimes = [path.stat().st_mtime for path in activity]
        return len(mtimes) == 2 and max(mtimes) <= now - idle_seconds
    except (OSError, subprocess.SubprocessError, ValueError):
        return False

def worktree_cleanup_inventory(
    repo: Path,
    worktrees_root: Path,
    ttl_seconds: int,
    now: float,
    *,
    include_closed_pr_worktrees: bool = True,
    idle_seconds: int = WORKTREE_IDLE_SECONDS,
    per_target_cap_bytes: int | None = None,
) -> list[dict[str, Any]] | None:
    """Find stale build targets and clean merged/closed worktrees under the canonical root."""
    try:
        root = worktrees_root.resolve(strict=True)
        result = subprocess.run(
            ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        worktrees = parse_worktree_list(result.stdout)
    except (OSError, subprocess.SubprocessError):
        return None

    candidates: list[dict[str, Any]] = []
    pr_states: dict[tuple[str, str], str] = {}
    queried_branches: set[str] = set()
    for item in worktrees:
        try:
            worktree = Path(str(item["path"])).resolve(strict=True)
            relative = worktree.relative_to(root)
            if len(relative.parts) != 1 or worktree == Path(repo).resolve(strict=True):
                continue
            if item.get("locked"):
                continue
            head = str(item.get("head", ""))
            if not head:
                continue
            branch = str(item.get("branch", ""))
            target = worktree / "target"
            target_exists = target.exists() or target.is_symlink()
            stale_target = False
            target_over_cap = True
            if target_exists:
                target_stat = target.lstat()
                stale_target = (
                    stat.S_ISDIR(target_stat.st_mode)
                    and target_stat.st_uid == current_uid()
                    and target_stat.st_dev == worktree.stat().st_dev
                    and target.resolve(strict=True) == target
                    and now - target_stat.st_mtime >= ttl_seconds
                )
                if per_target_cap_bytes is not None:
                    target_size = path_size(target) if stale_target else None
                    target_over_cap = target_size is not None and target_size > per_target_cap_bytes
                    if stale_target and not target_over_cap:
                        print(f"KEEP {target}: below per-worktree target cap ({target_size} <= {per_target_cap_bytes} bytes)")
                    elif stale_target and target_size is None:
                        print(f"SKIP {target}: per-worktree target size unavailable")
            ancestry = subprocess.run(
                ["git", "-C", str(repo), "merge-base", "--is-ancestor", head, "origin/main"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            main_ancestor = ancestry.returncode == 0
            clean = False
            if main_ancestor or stale_target or (include_closed_pr_worktrees and branch):
                # A plain status refreshes another writer's index (and takes index.lock), which would reset
                # the idle clock read below; inspect without optional locks, as ~/.agent/bin/lanes does.
                status = subprocess.run(
                    ["git", "--no-optional-locks", "-C", str(worktree), "status", "--porcelain", "--untracked-files=all"],
                    capture_output=True, text=True, timeout=15, check=False,
                )
                clean = status.returncode == 0 and not status.stdout
            if not main_ancestor and clean and branch and include_closed_pr_worktrees and branch not in queried_branches:
                queried_branches.add(branch)
                try:
                    pr_result = subprocess.run(
                        [
                            "gh", "pr", "list", "--repo", "neverhuman/JopeDime", "--state", "all",
                            "--head", branch, "--limit", "20", "--json", "headRefName,headRefOid,state",
                        ],
                        check=True, capture_output=True, text=True, timeout=15,
                    )
                    for pr in json.loads(pr_result.stdout):
                        if pr.get("headRefName") and pr.get("headRefOid") and pr.get("state"):
                            pr_states[(pr["headRefName"], pr["headRefOid"])] = pr["state"].upper()
                except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
                    pass
            disposition = cleanup_disposition(
                main_ancestor=main_ancestor, branch=branch, head=head, pr_states=pr_states,
            )
            ignored_safe: bool | None = None
            idle = (stale_target or (disposition and clean)) and worktree_idle_for_cleanup(
                worktree, now=now, idle_seconds=idle_seconds,
            )
            history_safe = not (disposition and clean and not branch) or detached_worktree_reflog_is_on_main(
                repo, worktree,
            )
            if disposition and clean and not branch and not history_safe:
                print(f"SKIP {worktree}: detached HEAD reflog contains history outside origin/main")
            if idle and ((stale_target and target_over_cap) or (disposition and clean)):
                candidates.append({
                    "worktree": worktree,
                    "target": target,
                    "target_exists": target_exists,
                    "stale_target": stale_target,
                    "target_over_cap": target_over_cap,
                    "branch": branch,
                    "head": head,
                    "disposition": disposition,
                    "clean": clean,
                    "ignored_safe": ignored_safe,
                    "idle": True,
                    "history_safe": history_safe,
                })
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    return candidates

def worktree_head_is_unchanged(
    worktree: Path,
    expected_branch: str,
    expected_head: str,
    *,
    require_ignored_safe: bool = True,
) -> bool:
    try:
        clean = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(worktree), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        ignored_safe = not require_ignored_safe or worktree_has_only_rebuildable_ignored_files(worktree)
        branch = subprocess.run(
            ["git", "-C", str(worktree), "symbolic-ref", "--quiet", "--short", "HEAD"],
            check=False, capture_output=True, text=True, timeout=15,
        )
        head = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return (
            clean.returncode == 0
            and not clean.stdout
            and ignored_safe
            and branch.stdout.strip() == expected_branch
            and head.stdout.strip() == expected_head
        )
    except (OSError, subprocess.SubprocessError):
        return False

def upload_and_verify(path: Path, key: str) -> bool:
    uri = f"s3://{OFFLOAD_BUCKET}/{key}"
    try:
        upload = subprocess.run(
            ["aws", "s3", "cp", str(path), uri, "--only-show-errors", "--no-progress"],
            capture_output=True, text=True, timeout=900, check=False,
        )
        if upload.returncode != 0:
            return False
        download = subprocess.Popen(
            ["aws", "s3", "cp", uri, "-", "--only-show-errors", "--no-progress"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if download.stdout is None:
            download.kill()
            return False
        digest = hashlib.sha256()
        deadline = time.monotonic() + 900
        fd = download.stdout.fileno()
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(download.args, 900)
                ready, _, _ = select.select([fd], [], [], min(5, remaining))
                if not ready:
                    continue
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            return download.wait(timeout=max(0.1, deadline - time.monotonic())) == 0 and digest.hexdigest() == sha256_file(path)
        except BaseException:
            with contextlib.suppress(OSError):
                download.kill()
            with contextlib.suppress(subprocess.SubprocessError):
                download.wait()
            raise
    except (OSError, subprocess.SubprocessError):
        return False

def offload_file(
    path: Path,
    allowed_root: Path,
    logger: Any,
    *,
    dry_run: bool,
    now: float,
    open_paths: set[Path] | None = None,
    handles_checked: bool = False,
) -> bool:
    try:
        item = path.lstat()
        if not stat.S_ISREG(item.st_mode) or item.st_uid != current_uid() or item.st_nlink != 1:
            return False
        if now - item.st_mtime < LOG_RETENTION_SECONDS or protected_path(path):
            return False
        resolved_root = allowed_root.resolve(strict=True)
        if resolved_root != allowed_root.absolute():
            return False
        expected_log_root = (Path.home() / ".local" / "state" / "jope-diskd").resolve(strict=False)
        repo = Path.home() / "JopeDime"
        relative = resolved_root.relative_to(repo) if resolved_root != expected_log_root else None
        report_root = relative is not None and (
            relative.parts == ("target", "jankurai", "reports")
            or (len(relative.parts) == 6 and relative.parts[:2] == (".agent", "worktrees") and relative.parts[-3:] == ("target", "jankurai", "reports"))
        )
        if resolved_root != expected_log_root and not report_root:
            return False
        if resolved_root == expected_log_root and not path.name.endswith(".log.gz"):
            return False
        opened = open_paths if handles_checked else open_handles_under(path, allow_lsof_fallback=False)
        if opened is None or opened:
            if opened is None or path.resolve(strict=False) in opened:
                return False
        digest = sha256_file(path)
        key = s3_key(path, allowed_root.resolve(strict=True), socket.gethostname(), utc_now().date(), digest)
        print(f"{'WOULD OFFLOAD' if dry_run else 'OFFLOAD'} {path} -> s3://{OFFLOAD_BUCKET}/{key} sha256={digest}")
        if dry_run:
            return True
        if not upload_and_verify(path, key):
            logger({"event": "offload_verify_failed", "path": str(path), "s3_key": key, "sha256": digest})
            return False
        current = path.lstat()
        if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
            item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns
        ) or sha256_file(path) != digest or open_handles_under(path, allow_lsof_fallback=False) != set():
            logger({"event": "offload_source_changed", "path": str(path), "s3_key": key, "sha256": digest})
            return False
        logger({"event": "offload_verified", "path": str(path), "s3_key": key, "sha256": digest, "bytes": item.st_size})
        path.unlink()
        logger({"event": "offload_local_removed", "path": str(path), "s3_key": key, "sha256": digest, "bytes": item.st_size})
        return True
    except (OSError, ValueError):
        return False

def rotate_daemon_log(
    home: Path, logger: Any, *, dry_run: bool, now: float, offload_archives: bool = False
) -> None:
    requested_state = home / ".local" / "state" / "jope-diskd"
    if dry_run and not requested_state.is_dir():
        return
    state = safe_state_dir(home, create=not dry_run)
    if state is None:
        print("SKIP log maintenance: state path is absent, symlinked, or not user-owned")
        return
    active = state / "jope-diskd.log"
    try:
        if active.exists() and active.stat().st_size > LOG_LIMIT_BYTES:
            opened = open_handles_under(active, allow_lsof_fallback=False)
            if opened is None or opened:
                print(f"SKIP log rotation: {active} has open/unknown handles")
            else:
                rotated = state / f"jope-diskd-{utc_now():%Y%m%dT%H%M%SZ}.log.gz"
                print(f"{'WOULD ROTATE' if dry_run else 'ROTATE'} {active} -> {rotated}")
                if not dry_run:
                    with active.open("rb") as source, gzip.open(rotated, "xb", compresslevel=6) as target:
                        shutil.copyfileobj(source, target)
                    if sha256_file(active) != sha256_file_gzip(rotated):
                        rotated.unlink(missing_ok=True)
                        print("SKIP log rotation: compressed log verification failed")
                    else:
                        active.unlink()
                        logger({"event": "log_rotated", "path": str(active), "archive": str(rotated), "bytes": rotated.stat().st_size})
        if offload_archives:
            for archive in state.glob("jope-diskd-*.log.gz"):
                if now - archive.stat().st_mtime >= LOG_RETENTION_SECONDS:
                    offload_file(archive, state, logger, dry_run=dry_run, now=now)
    except OSError as exc:
        print(f"SKIP log maintenance: {exc}")

def offload_jankurai_reports(repo: Path, worktrees_root: Path, logger: Any, *, dry_run: bool, now: float) -> None:
    roots = [Path(repo) / "target" / "jankurai" / "reports"]
    try:
        for entry in worktrees_root.iterdir():
            roots.append(entry / "target" / "jankurai" / "reports")
    except OSError:
        return
    for root in roots:
        try:
            if not root.is_dir() or root.is_symlink() or not under_home(root):
                continue
            open_paths = open_handles_under(root, allow_lsof_fallback=False)
            if open_paths is None:
                print(f"SKIP reports {root}: open-handle scan incomplete")
                return
            if any(item.is_dir() for item in open_paths):
                print(f"SKIP reports {root}: an open directory handle could still be writing reports")
                continue
            for directory, dirs, names in os.walk(root, topdown=True, followlinks=False, onerror=fail_walk):
                dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
                if now - Path(directory).stat().st_mtime < LOG_RETENTION_SECONDS:
                    dirs.clear()
                    continue
                for name in names:
                    path = Path(directory) / name
                    offload_file(path, root, logger, dry_run=dry_run, now=now, open_paths=open_paths, handles_checked=True)
            if not dry_run:
                for directory, _dirs, _names in os.walk(root, topdown=False, followlinks=False, onerror=fail_walk):
                    empty = Path(directory)
                    try:
                        item = empty.lstat()
                        if not stat.S_ISDIR(item.st_mode) or item.st_uid != current_uid():
                            continue
                        empty.rmdir()
                        logger({"event": "offload_empty_dir_removed", "path": str(empty)})
                    except OSError:
                        continue
                try:
                    root.parent.rmdir()
                    logger({"event": "offload_empty_dir_removed", "path": str(root.parent)})
                except OSError:
                    pass
        except OSError:
            continue

def handle_jankurai_source_tars(
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
    return _handle_jankurai_source_tars_impl(globals(), entry, repo, worktrees_root, logger, dry_run=dry_run, now=now, needed_bytes=needed_bytes)

def handle_remote_source_scratch(
    entry: dict[str, Any], logger: Any, *, dry_run: bool, now: float, needed_bytes: int
) -> int:
    """Prune aged remote-test source scratch only under lane_work, on disk pressure."""
    return _handle_remote_source_scratch_impl(globals(), entry, logger, dry_run=dry_run, now=now, needed_bytes=needed_bytes)

def denied(path: Path, policy: dict[str, Any]) -> bool:
    resolved = path.resolve(strict=False)
    if protected_path(resolved):
        return True
    for value in policy["deny_prefixes"]:
        prefix = expand_path(value).resolve(strict=False)
        try:
            resolved.relative_to(prefix)
            return True
        except ValueError:
            continue
    return False

def safe_cache_root(path: Path, policy: dict[str, Any]) -> bool:
    try:
        resolved = path.resolve(strict=True)
        if resolved != path.absolute() or not under_home(path) or denied(path, policy):
            return False
        st = path.lstat()
        return stat.S_ISDIR(st.st_mode) and st.st_uid == current_uid() and not protected_path(resolved)
    except OSError:
        return False

def audit_logger(home: Path, dry_run: bool):
    def log(event: dict[str, Any]) -> None:
        if dry_run:
            return
        state = safe_state_dir(home, create=True)
        if state is None:
            raise RuntimeError("jope-diskd state path is not a canonical owned directory")
        audit_path = state / "audit.jsonl"
        daemon_log = state / "jope-diskd.log"
        event = {"time": utc_now().isoformat(), "host": socket.gethostname(), **event}
        with audit_path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            serialized = json.dumps(event, sort_keys=True)
            stream.write(serialized + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        with daemon_log.open("a", encoding="utf-8") as stream:
            stream.write(f"{event['time']} {serialized}\n")
            stream.flush()

    return log

def alert_if_needed(home: Path, percent: float, policy: dict[str, Any], dry_run: bool) -> None:
    if percent < policy["watermarks"]["alert_percent"]:
        return
    now = utc_now()
    host = socket.gethostname()
    alert_file = home / "nudges" / "DISK_ALERT.md"
    message = f"{now.isoformat()} {host}: disk usage {percent:.1f}% (>= {policy['watermarks']['alert_percent']}%); automatic cleanup remained within explicit rebuildable-cache policy."
    print(f"ALERT {'would append' if dry_run else 'append'} {alert_file}: {message}")
    if dry_run:
        return
    state_dir = safe_state_dir(home, create=True)
    if state_dir is None:
        raise RuntimeError("jope-diskd state path is not a canonical owned directory")
    state_file = state_dir / f"alert-{host}-{now:%Y-%m-%d}.json"
    if state_file.exists() and state_file.stat().st_size > 0:
        return
    with (state_dir / f"alert-{host}-{now:%Y-%m-%d}.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if state_file.exists() and state_file.stat().st_size > 0:
            return
        alert_file.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        with alert_file.open("a", encoding="utf-8") as out:
            out.write(f"- {message}\n")
            out.flush()
            os.fsync(out.fileno())
        state_file.write_text(json.dumps({"time": now.isoformat(), "usage_percent": percent}) + "\n", encoding="utf-8")

@contextlib.contextmanager
def run_lock(home: Path, dry_run: bool):
    if dry_run:
        yield
        return
    state_dir = safe_state_dir(home, create=True)
    if state_dir is None:
        raise RuntimeError("jope-diskd state path is not a canonical owned directory")
    with (state_dir / "run.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another jope-diskd apply run is active") from exc
        yield

def handle_cache(
    entry: dict[str, Any],
    policy: dict[str, Any],
    threshold: int,
    max_bytes: int,
    needed_bytes: int,
    dry_run: bool,
    logger: Any,
    now: float,
) -> int:
    return _handle_cache_impl(globals(), entry, policy, threshold, max_bytes, needed_bytes, dry_run, logger, now)

def handle_cargo_targets(
    entry: dict[str, Any],
    repo: Path,
    policy: dict[str, Any],
    logger: Any,
    *,
    dry_run: bool,
    now: float,
    needed_bytes: int,
) -> int:
    return _handle_cargo_targets_impl(globals(), entry, repo, policy, logger, dry_run=dry_run, now=now, needed_bytes=needed_bytes)

def report_only_paths(policy: dict[str, Any]) -> None:
    for entry in policy["paths"]:
        if entry.get("mode") != "observe":
            continue
        path = expand_path(entry["path"])
        if entry.get("manual_only"):
            cap = entry.get("per_target_cap_bytes", entry.get("max_bytes"))
            ttl = entry.get("ttl_hours")
            print(f"OBSERVE {entry['id']}: manual inventory only (cap={cap}, ttl_hours={ttl}); no recurring whole-tree scan ({path})")
            continue
        size = path_size(path)
        if size is None:
            print(f"OBSERVE {entry['id']}: measurement unavailable ({path})")
        elif size:
            print(f"OBSERVE {entry['id']}: {size} bytes at {path}; no automatic deletion policy")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON-syntax YAML policy")
    parser.add_argument("--dry-run", action="store_true", help="preview only (the default)")
    parser.add_argument("--apply", action="store_true", help="apply the approved local cleanup policy")
    args = parser.parse_args(argv)
    if args.dry_run and args.apply:
        parser.error("choose only one of --dry-run or --apply")
    dry_run = not args.apply
    try:
        if args.apply and os.environ.get("JOPE_DISKD_LOCAL_ONLY") != "1":
            raise ValueError("apply requires JOPE_DISKD_LOCAL_ONLY=1")
        policy = read_policy(args.config)
        home = Path.home()
        host = socket.gethostname()
        usage = shutil.disk_usage(home)
        percent = 100.0 * usage.used / (usage.used + usage.free)
        print(f"host={host} home={home} used={percent:.1f}% free_bytes={usage.free}")
        if policy.get("local_only") and host not in policy["allowed_hosts"]:
            print(f"SKIP: local-only policy does not include {host}")
            return 0
        with run_lock(home, dry_run):
            alert_if_needed(home, percent, policy, dry_run)
            report_only_paths(policy)
            logger = audit_logger(home, dry_run)
            now = time.time()
            repo = expand_path(policy.get("repo_root", "~/JopeDime"))
            worktrees_root = repo / ".agent" / "worktrees"
            if policy.get("allow_log_rotation", False):
                rotate_daemon_log(
                    home,
                    logger,
                    dry_run=dry_run,
                    now=now,
                    offload_archives=policy.get("allow_log_offload", False),
                )
            else:
                print("daemon log rotation disabled by policy")
            if policy.get("allow_report_offload", False) and percent >= policy["watermarks"]["soft_percent"]:
                offload_jankurai_reports(repo, worktrees_root, logger, dry_run=dry_run, now=now)
            else:
                print("report offload disabled by policy")
            if percent < policy["watermarks"]["soft_percent"]:
                print("below soft watermark; no automatic cleanup")
                return 0
            ttl_key = "aggressive_ttl_hours" if percent >= policy["watermarks"]["aggressive_percent"] else "soft_ttl_hours"
            freed = 0
            for entry in policy["paths"]:
                kind = entry.get("kind")
                if kind == "cache":
                    required = max(
                        0,
                        int(usage.used - (usage.used + usage.free) * policy["watermarks"]["low_percent"] / 100) - freed,
                    )
                    freed += handle_cache(
                        entry,
                        policy,
                        int(entry[ttl_key]) * 3600,
                        int(entry["max_bytes"]),
                        required,
                        dry_run,
                        logger,
                        now,
                    )
                elif kind == "cargo-target":
                    required = max(
                        0,
                        int(usage.used - (usage.used + usage.free) * policy["watermarks"]["low_percent"] / 100) - freed,
                    )
                    freed += handle_cargo_targets(
                        entry, repo, policy, logger, dry_run=dry_run, now=now, needed_bytes=required
                    )
                elif kind == "jankurai-source-tar":
                    required = max(
                        0,
                        int(usage.used - (usage.used + usage.free) * policy["watermarks"]["low_percent"] / 100) - freed,
                    )
                    freed += handle_jankurai_source_tars(
                        entry,
                        repo,
                        worktrees_root,
                        logger,
                        dry_run=dry_run,
                        now=now,
                        needed_bytes=required,
                    )
                elif kind == "remote-source-scratch":
                    required = max(
                        0,
                        int(usage.used - (usage.used + usage.free) * policy["watermarks"]["low_percent"] / 100) - freed,
                    )
                    freed += handle_remote_source_scratch(
                        entry, logger, dry_run=dry_run, now=now, needed_bytes=required
                    )
            print(f"planned_reclaim_bytes={freed} mode={'dry-run' if dry_run else 'apply'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"jope-diskd: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
