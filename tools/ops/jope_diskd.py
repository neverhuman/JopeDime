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

GIB = 1024**3
SAFE_CACHE_PATHS = {
    "uv-cache": Path(".cache/uv"),
    "pip-cache": Path(".cache/pip"),
}
CACHE_METADATA_NAMES = {"cachedir.tag", ".gitignore", ".lock", "lock"}
LOG_LIMIT_BYTES = 256 * 1024**2
LOG_RETENTION_SECONDS = 14 * 24 * 3600
TARGET_RETENTION_SECONDS = 24 * 3600
OFFLOAD_BUCKET = "veox-jopedime-learning-use2"
OFFLOAD_PREFIX = "offload"
TARGET_EVIDENCE_PARTS = {
    "checkpoint", "checkpoints", "evidence", "health", "receipt", "receipts",
    "report", "reports", "score", "scores", "sealed", "holdout", "flywheel",
}
PROTECTED_PARTS = {
    "admiralty",
    "checkpoints",
    "custody",
    "datasets",
    "evidence",
    "guestd",
    "flywheel",
    "holdout",
    "jeryu",
    "jope-data",
    "jope-rescue-20260921",
    "jope_plan",
    "jope-plan",
    "receipts",
    "sealed",
    "veox",
}
PROTECTED_PREFIXES = {
    "admiralty",
    "checkpoint",
    "datasets",
    "evidence",
    "flywheel",
    "holdout",
    "jeryu",
    "jope-data",
    "jope-plan",
    "jope-rescue-20260921",
    "jope_plan",
    "receipt",
    "sealed",
    "veox",
    "guestd",
}


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
    if not isinstance(policy.get("allowed_host"), str) or not policy["allowed_host"]:
        raise ValueError("an exact local-only host is required")
    if policy.get("repo_root") != "~/JopeDime" or policy.get("sccache_max_bytes") != 60 * GIB:
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


def lsof_open_paths(resolved_root: Path) -> set[Path] | None:
    lsof = shutil.which("lsof")
    if not lsof:
        return None
    try:
        priority = ["/usr/bin/ionice", "-c3", "/usr/bin/nice", "-n", "10"] if sys.platform.startswith("linux") else ["/usr/bin/nice", "-n", "10"]
        command = priority + [lsof, "-nP", "-F", "n", "+D", str(resolved_root)]
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
                if not all_users and int(uid_line.split()[1]) != current_uid():
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
    return lsof_open_paths(resolved_root)


def path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def local_process_references(target: Path, worktree: Path, proc_root: Path = Path("/proc")) -> bool | None:
    """Return busy/clear, or None if local process metadata was not fully readable."""
    if not proc_root.is_dir():
        return None
    try:
        for process in proc_root.iterdir():
            if not process.name.isdigit():
                continue
            try:
                cwd = Path(os.readlink(process / "cwd"))
                environ = (process / "environ").read_bytes().split(b"\0")
            except FileNotFoundError:
                continue
            except PermissionError:
                return None
            except OSError:
                return None
            if path_contains(target, cwd) or path_contains(worktree, cwd):
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
    except (OSError, RuntimeError):
        return None


def safe_target_tree(root: Path, *, now: float) -> bool:
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
            if parent_stat.st_dev != device or parent_stat.st_uid != current_uid() or now - parent_stat.st_mtime < TARGET_RETENTION_SECONDS:
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
                    or now - item.st_mtime < TARGET_RETENTION_SECONDS
                    or any(term in lowered for term in ("receipt", "report", "evidence", "score", "checkpoint"))
                    or Path(name).suffix.lower() in {".csv", ".db", ".html", ".jsonl", ".md", ".npz", ".parquet", ".png", ".sqlite", ".svg", ".txt", ".yaml", ".yml"}
                    or (Path(name).suffix.lower() == ".json" and ".fingerprint" not in relative_parent.parts)
                ):
                    return False
        return True
    except OSError:
        return False


def merged_worktree_targets(repo: Path, worktrees_root: Path, ttl_seconds: int, now: float) -> list[tuple[Path, Path, str, str]] | None:
    """Select old target dirs only from clean registered worktrees merged into origin/main."""
    try:
        root = worktrees_root.resolve(strict=True)
        result = subprocess.run(
            ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        worktrees = [Path(line.removeprefix("worktree ")) for line in result.stdout.splitlines() if line.startswith("worktree ")]
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        merged_prs = subprocess.run(
            ["gh", "pr", "list", "--repo", "neverhuman/JopeDime", "--state", "merged", "--limit", "100", "--json", "headRefName,headRefOid,mergedAt"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        merged_heads = {
            (item["headRefName"], item["headRefOid"])
            for item in json.loads(merged_prs.stdout)
            if item.get("mergedAt") and item.get("headRefName") and item.get("headRefOid")
        }
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return None
    candidates: list[tuple[Path, Path, str, str]] = []
    for worktree in worktrees:
        try:
            resolved_worktree = worktree.resolve(strict=True)
            relative = resolved_worktree.relative_to(root)
            if len(relative.parts) != 1 or resolved_worktree == Path(repo).resolve(strict=True):
                continue
            clean = subprocess.run(
                ["git", "-C", str(worktree), "status", "--porcelain"],
                check=True, capture_output=True, text=True, timeout=15,
            )
            if clean.stdout:
                continue
            branch = subprocess.run(
                ["git", "-C", str(worktree), "symbolic-ref", "--quiet", "--short", "HEAD"],
                check=True, capture_output=True, text=True, timeout=15,
            )
            head = subprocess.run(
                ["git", "-C", str(worktree), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, timeout=15,
            )
            if (branch.stdout.strip(), head.stdout.strip()) not in merged_heads:
                continue
            target = resolved_worktree / "target"
            item = target.lstat()
            if not stat.S_ISDIR(item.st_mode) or item.st_uid != current_uid() or now - item.st_mtime < ttl_seconds:
                continue
            if target.resolve(strict=True) != target or item.st_dev != resolved_worktree.stat().st_dev:
                continue
            candidates.append((resolved_worktree, target, branch.stdout.strip(), head.stdout.strip()))
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    return candidates


def worktree_head_is_unchanged(worktree: Path, expected_branch: str, expected_head: str) -> bool:
    try:
        clean = subprocess.run(
            ["git", "-C", str(worktree), "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        branch = subprocess.run(
            ["git", "-C", str(worktree), "symbolic-ref", "--quiet", "--short", "HEAD"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        head = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return not clean.stdout and branch.stdout.strip() == expected_branch and head.stdout.strip() == expected_head
    except (OSError, subprocess.SubprocessError):
        return False


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


def rotate_daemon_log(home: Path, logger: Any, *, dry_run: bool, now: float) -> None:
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
                    with active.open("rb") as source, gzip.open(rotated, "wb", compresslevel=6) as target:
                        shutil.copyfileobj(source, target)
                    if sha256_file(active) != sha256_file_gzip(rotated):
                        rotated.unlink(missing_ok=True)
                        print("SKIP log rotation: compressed log verification failed")
                    else:
                        active.unlink()
                        logger({"event": "log_rotated", "path": str(active), "archive": str(rotated), "bytes": rotated.stat().st_size})
        for archive in state.glob("jope-diskd-*.log.gz"):
            if now - archive.stat().st_mtime >= LOG_RETENTION_SECONDS:
                offload_file(archive, state, logger, dry_run=dry_run, now=now)
    except OSError as exc:
        print(f"SKIP log maintenance: {exc}")


def sha256_file_gzip(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def current_uid() -> int:
    return os.getuid() if hasattr(os, "getuid") else -1


def safe_state_dir(home: Path, *, create: bool) -> Path | None:
    state = home / ".local" / "state" / "jope-diskd"
    try:
        if state.absolute() != state.resolve(strict=False):
            return None
        if create:
            state.mkdir(mode=0o700, parents=True, exist_ok=True)
        item = state.lstat()
        if not stat.S_ISDIR(item.st_mode) or item.st_uid != current_uid() or state.resolve(strict=True) != state.absolute():
            return None
        return state
    except (OSError, RuntimeError):
        return None


def allocated_bytes(item: os.stat_result) -> int:
    return int(getattr(item, "st_blocks", 0)) * 512 or item.st_size


def fail_walk(exc: OSError) -> None:
    raise exc


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
    name = entry["id"]
    root = expand_path(entry["path"])
    expected = Path.home() / SAFE_CACHE_PATHS.get(name, Path("__not_allowlisted__"))
    if root.absolute() != expected.absolute() or not safe_cache_root(root, policy):
        print(f"SKIP {name}: cache root not an owned, canonical home directory")
        return 0
    size = path_size(root)
    if size is None:
        print(f"SKIP {name}: could not measure cache")
        return 0
    if size <= max_bytes:
        print(f"OK {name}: {size} bytes <= cap {max_bytes}")
        return 0
    target_bytes = min(max_bytes, max(0, size - needed_bytes))
    opened = open_handles_under(root)
    if opened is None:
        print(f"SKIP {name}: open-handle scan was incomplete")
        return 0
    candidates = cache_candidates(root, threshold, now)
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
            if st.st_uid != current_uid() or not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                continue
            if path.resolve(strict=False) in opened:
                continue
            amount = allocated_bytes(st)
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
        remove_empty_dirs(root, dry_run, logger)
    return freed


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
    root = expand_path(entry["path"])
    expected = Path.home() / "JopeDime" / ".agent" / "worktrees"
    if not policy.get("local_only") or root.absolute() != expected.absolute() or not root.is_dir():
        print(f"SKIP {entry['id']}: local-only canonical worktree root required")
        return 0
    if needed_bytes <= 0:
        print(f"OK {entry['id']}: low watermark already reached")
        return 0
    candidates = merged_worktree_targets(repo, root, TARGET_RETENTION_SECONDS, now)
    if candidates is None:
        print(f"SKIP {entry['id']}: merged-worktree inventory unavailable")
        return 0
    freed = 0
    for worktree, target, branch, head in candidates:
        opened = open_handles_under(target, all_users=True)
        references = local_process_references(target, worktree)
        if opened is None or opened or references is not False:
            print(f"SKIP {target}: open handles or process references are present/unknown")
            continue
        if not safe_target_tree(target, now=now):
            print(f"SKIP {target}: evidence marker, recent entry, special file, or non-user-owned entry")
            continue
        size = path_size(target)
        if size is None:
            print(f"SKIP {target}: cannot measure")
            continue
        print(f"{'WOULD REMOVE' if dry_run else 'REMOVE'} merged-worktree target {target} ({size} bytes)")
        if not dry_run:
            # Repeat every consumer/evidence check immediately before the tree removal.
            if (
                open_handles_under(target, all_users=True) != set()
                or local_process_references(target, worktree) is not False
                or not safe_target_tree(target, now=time.time())
                or not worktree_head_is_unchanged(worktree, branch, head)
            ):
                print(f"SKIP {target}: candidate changed before removal")
                continue
            logger({"event": "delete_intent_tree", "path": str(target), "bytes": size, "policy": entry["id"]})
            shutil.rmtree(target)
            logger({"event": "deleted_tree", "path": str(target), "bytes": size, "policy": entry["id"]})
        freed += size
        if freed >= needed_bytes:
            break
    print(f"CARGO-TARGETS {entry['id']}: planned_bytes={freed}")
    return freed


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
    parser.add_argument("--apply", action="store_true", help="apply only the safe cache policy")
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
        if policy.get("local_only") and host != policy["allowed_host"]:
            print(f"SKIP: local-only policy is for {policy['allowed_host']}, not {host}")
            return 0
        with run_lock(home, dry_run):
            alert_if_needed(home, percent, policy, dry_run)
            report_only_paths(policy)
            logger = audit_logger(home, dry_run)
            now = time.time()
            repo = expand_path(policy.get("repo_root", "~/JopeDime"))
            worktrees_root = repo / ".agent" / "worktrees"
            rotate_daemon_log(home, logger, dry_run=dry_run, now=now)
            if percent >= policy["watermarks"]["soft_percent"]:
                offload_jankurai_reports(repo, worktrees_root, logger, dry_run=dry_run, now=now)
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
            print(f"planned_reclaim_bytes={freed} mode={'dry-run' if dry_run else 'apply'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"jope-diskd: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
