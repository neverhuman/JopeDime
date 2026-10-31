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
SUPPORTED_LOCAL_HOSTS = frozenset({"xbabe1", "xbabe2", "xbabe3"})
CACHE_METADATA_NAMES = {"cachedir.tag", ".gitignore", ".lock", "lock"}
REBUILDABLE_WORKTREE_IGNORED_DIRS = {
    "__pycache__", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache", ".tox", ".venv", "node_modules", "venv",
}
REBUILDABLE_WORKTREE_IGNORED_FILES = {"AGENT_CHAT.lock", "AGENT_CHAT.md.lock"}
LOG_LIMIT_BYTES = 256 * 1024**2
LOG_RETENTION_SECONDS = 14 * 24 * 3600
TARGET_RETENTION_SECONDS = 24 * 3600
WORKTREE_IDLE_SECONDS = 2 * 3600
OFFLOAD_BUCKET = "veox-jopedime-learning-use2"
OFFLOAD_PREFIX = "offload"
TARGET_EVIDENCE_PARTS = {
    "checkpoint", "checkpoints", "evidence", "health", "receipt", "receipts",
    "report", "reports", "score", "scores", "sealed", "holdout", "flywheel",
}
TARGET_EVIDENCE_SUFFIXES = {
    ".bin", ".ckpt", ".h5", ".hdf5", ".joblib", ".npy", ".onnx", ".pkl",
    ".pickle", ".pt", ".pth", ".safetensors", ".tfrecord", ".tfrecords",
}
CARGO_INCREMENTAL_BINARY_CACHE_NAMES = {
    "dep-graph.bin", "query-cache.bin", "work-products.bin",
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
    roots = jankurai_report_roots(repo, worktrees_root)
    if roots is None:
        print("SKIP jankurai source tar: registered report-root inventory unavailable")
        return 0
    max_file_bytes = int(entry["max_file_bytes"])
    ttl_seconds = int(entry["ttl_hours"]) * 3600
    freed = 0
    candidates: list[tuple[Path, Path, Path, os.stat_result]] = []
    for report_root, worktree in roots:
        try:
            if not report_root.is_dir() or report_root.is_symlink() or not under_home(report_root):
                continue
            for directory, dirs, names in os.walk(report_root, topdown=True, followlinks=False, onerror=fail_walk):
                dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
                if "source.tar" not in names:
                    continue
                path = Path(directory) / "source.tar"
                item = source_tar_is_eligible(
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
        opened = open_handles_under(report_root, all_users=False, allow_lsof_fallback=True)
        referenced = local_process_references(report_root, worktree)
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
        opened = open_handles_under(report_root, all_users=False, allow_lsof_fallback=True)
        referenced = local_process_references(report_root, worktree)
        after = source_tar_is_eligible(
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


def handle_remote_source_scratch(
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
    candidates = remote_source_scratch_candidates(root, now=now, ttl_seconds=3600)
    if candidates is None:
        print("SKIP remote source scratch: lane_work inventory unavailable")
        return 0
    freed = 0
    for path, lane, before in candidates:
        opened = open_handles_under(path, all_users=True, allow_lsof_fallback=True)
        referenced = local_process_references(path, path)
        if opened is None or opened or referenced is not False:
            print(f"SKIP {path}: process/open-handle scan is busy or incomplete")
            continue
        size = path_size(path) if stat.S_ISDIR(before.st_mode) else before.st_size
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
        opened = open_handles_under(path, all_users=True, allow_lsof_fallback=True)
        referenced = local_process_references(path, path)
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
                safe_remote_source_tree(path, lane, now=time.time(), ttl_seconds=3600)
                if stat.S_ISDIR(before.st_mode)
                else stat.S_ISREG(after.st_mode)
                and after.st_uid == current_uid()
                and after.st_dev == before.st_dev
                and after.st_nlink == 1
                and time.time() - after.st_mtime >= 3600
                and not protected_path(path)
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
    ttl_seconds = int(entry.get("ttl_hours", TARGET_RETENTION_SECONDS // 3600)) * 3600
    idle_seconds = int(entry.get("worktree_idle_hours", WORKTREE_IDLE_SECONDS // 3600)) * 3600
    candidates = worktree_cleanup_inventory(
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
                ignored_safe = worktree_has_only_rebuildable_ignored_files(worktree)
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
        if remove_worktree and candidate["target_exists"] and not safe_target_tree(target, now=now, ttl_seconds=ttl_seconds):
            print(f"SKIP {worktree}: target contains recent, evidence-like, or non-rebuildable content")
            remove_worktree = False
        remove_target_only = remove_stale_target and not remove_worktree
        if not remove_worktree and not remove_target_only:
            continue
        watched_path = worktree if remove_worktree else target
        opened = open_handles_under(watched_path, all_users=False, allow_lsof_fallback=True)
        references = local_process_references(target, worktree)
        if opened is None or opened or references is not False:
            print(f"SKIP {watched_path}: open handles or process references are present/unknown")
            continue
        if candidate["target_exists"] and not safe_target_tree(target, now=now, ttl_seconds=ttl_seconds):
            print(f"SKIP {target}: evidence marker, recent entry, special file, or non-user-owned entry")
            continue
        size = path_size(worktree if remove_worktree else target)
        if size is None:
            print(f"SKIP {watched_path}: cannot measure")
            continue
        action = f"merged/closed worktree {worktree}" if remove_worktree else f"stale Cargo target {target}"
        print(f"{'WOULD REMOVE' if dry_run else 'REMOVE'} {action} ({size} bytes)")
        if not dry_run:
            # Repeat consumer, source, and evidence checks immediately before removal.
            if (
                open_handles_under(watched_path, all_users=False, allow_lsof_fallback=True) != set()
                or local_process_references(target, worktree) is not False
                or not worktree_idle_for_cleanup(worktree, now=time.time(), idle_seconds=idle_seconds)
                or not worktree_head_is_unchanged(
                    worktree, branch, head, require_ignored_safe=remove_worktree,
                )
                or (remove_worktree and not branch and not detached_worktree_reflog_is_on_main(repo, worktree))
                or (candidate["target_exists"] and not safe_target_tree(target, now=time.time(), ttl_seconds=ttl_seconds))
            ):
                print(f"SKIP {watched_path}: candidate changed before removal")
                continue
            if candidate["target_exists"]:
                target_size = path_size(target)
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
                    open_handles_under(worktree, all_users=False, allow_lsof_fallback=True) != set()
                    or local_process_references(target, worktree) is not False
                    or not worktree_idle_for_cleanup(worktree, now=time.time(), idle_seconds=idle_seconds)
                    or not worktree_head_is_unchanged(worktree, branch, head)
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
