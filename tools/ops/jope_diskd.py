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
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

GIB = 1024**3
SCRATCH_NAME = re.compile(r"src\.(\d+)\Z")
SAFE_CACHE_PATHS = {
    "uv-cache": Path(".cache/uv"),
    "pip-cache": Path(".cache/pip"),
}
CACHE_METADATA_NAMES = {"cachedir.tag", ".gitignore", ".lock", "lock"}
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


def open_handles_under(root: Path) -> set[Path] | None:
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
                if int(uid_line.split()[1]) != current_uid():
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
        except (OSError, StopIteration, ValueError):
            pass
    return lsof_open_paths(resolved_root)


def pid_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def current_uid() -> int:
    return os.getuid() if hasattr(os, "getuid") else -1


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
    audit_path = home / ".local" / "state" / "jope-diskd" / "audit.jsonl"

    def log(event: dict[str, Any]) -> None:
        if dry_run:
            return
        audit_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        event = {"time": utc_now().isoformat(), "host": socket.gethostname(), **event}
        with audit_path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    return log


def alert_if_needed(home: Path, percent: float, policy: dict[str, Any], dry_run: bool) -> None:
    if percent < policy["watermarks"]["alert_percent"]:
        return
    now = utc_now()
    host = socket.gethostname()
    state_dir = home / ".local" / "state" / "jope-diskd"
    state_file = state_dir / f"alert-{host}-{now:%Y-%m-%d}.json"
    alert_file = home / "nudges" / "DISK_ALERT.md"
    message = f"{now.isoformat()} {host}: disk usage {percent:.1f}% (>= {policy['watermarks']['alert_percent']}%); automatic cleanup remained within explicit rebuildable-cache policy."
    print(f"ALERT {'would append' if dry_run else 'append'} {alert_file}: {message}")
    if dry_run or (state_file.exists() and state_file.stat().st_size > 0):
        return
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
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
    state_dir = home / ".local" / "state" / "jope-diskd"
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
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


def remote_scratch_candidates(root: Path, policy: dict[str, Any], ttl_seconds: int, now: float) -> list[Path]:
    expected = Path.home() / "remote_tests"
    if root.absolute() != expected.absolute() or not safe_cache_root(root, policy):
        return []
    candidates = []
    try:
        for path in root.iterdir():
            match = SCRATCH_NAME.fullmatch(path.name)
            if not match:
                continue
            st = path.lstat()
            if not stat.S_ISDIR(st.st_mode) or st.st_uid != current_uid() or protected_path(path):
                continue
            if now - st.st_mtime < ttl_seconds or pid_is_live(int(match.group(1))):
                continue
            candidates.append(path)
    except OSError:
        return []
    return candidates


def scratch_entries(root: Path) -> tuple[list[Path], list[Path]] | None:
    files: list[Path] = []
    directories: list[Path] = []
    try:
        for directory, dirs, names in os.walk(root, topdown=True, followlinks=False, onerror=fail_walk):
            parent = Path(directory)
            for name in list(dirs):
                child = parent / name
                st = child.lstat()
                if (
                    not stat.S_ISDIR(st.st_mode)
                    or st.st_uid != current_uid()
                    or protected_path(child)
                    or name.lower() == ".git"
                ):
                    return None
                directories.append(child)
            for name in names:
                child = parent / name
                st = child.lstat()
                if (
                    not stat.S_ISREG(st.st_mode)
                    or st.st_uid != current_uid()
                    or st.st_nlink != 1
                    or protected_path(child)
                    or child.name.lower() in CACHE_METADATA_NAMES
                    or any(part.lower() == ".git" for part in child.relative_to(root).parts)
                ):
                    return None
                files.append(child)
    except OSError:
        return None
    return files, directories


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
        policy = read_policy(args.config)
        home = Path.home()
        usage = shutil.disk_usage(home)
        percent = 100.0 * usage.used / (usage.used + usage.free)
        print(f"host={socket.gethostname()} home={home} used={percent:.1f}% free_bytes={usage.free}")
        with run_lock(home, dry_run):
            alert_if_needed(home, percent, policy, dry_run)
            report_only_paths(policy)
            if percent < policy["watermarks"]["soft_percent"]:
                print("below soft watermark; no automatic cleanup")
                return 0
            ttl_key = "aggressive_ttl_hours" if percent >= policy["watermarks"]["aggressive_percent"] else "soft_ttl_hours"
            logger = audit_logger(home, dry_run)
            freed = 0
            now = time.time()
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
                elif kind == "remote-tests":
                    root = expand_path(entry["path"])
                    total = path_size(root)
                    if total is None:
                        print(f"SKIP {entry['id']}: cannot measure")
                        continue
                    print(f"OBSERVE {entry['id']}: {total} bytes (cap {entry['max_bytes']}); shared target is report-only")
                    if total <= int(entry["max_bytes"]):
                        continue
                    candidate_freed = 0
                    for candidate in remote_scratch_candidates(root, policy, int(entry["ttl_hours"]) * 3600, now):
                        entries = scratch_entries(candidate)
                        if entries is None:
                            print(f"SKIP {candidate}: unsafe or unknown tree contents")
                            continue
                        candidate_size = path_size(candidate)
                        if candidate_size is None:
                            print(f"SKIP {candidate}: cannot measure")
                            continue
                        opened = open_handles_under(candidate)
                        if opened is None or opened:
                            print(f"SKIP {candidate}: open-handle state unavailable or busy")
                            continue
                        print(f"{'WOULD REMOVE' if dry_run else 'REMOVE'} stale scratch {candidate} ({candidate_size} bytes)")
                        candidate_freed += candidate_size
                        if not dry_run:
                            files, directories = entries
                            for child in files:
                                item = child.lstat()
                                if not stat.S_ISREG(item.st_mode) or item.st_uid != current_uid() or item.st_nlink != 1:
                                    raise RuntimeError(f"scratch entry changed after validation: {child}")
                                amount = allocated_bytes(item)
                                logger({"event": "delete_intent", "path": str(child), "bytes": amount, "policy": entry["id"]})
                                child.unlink()
                                logger({"event": "deleted", "path": str(child), "bytes": amount, "policy": entry["id"]})
                            for child in reversed(directories):
                                logger({"event": "delete_intent_dir", "path": str(child), "policy": entry["id"]})
                                child.rmdir()
                                logger({"event": "deleted_dir", "path": str(child), "policy": entry["id"]})
                            logger({"event": "delete_intent_dir", "path": str(candidate), "policy": entry["id"]})
                            candidate.rmdir()
                            logger({"event": "deleted_dir", "path": str(candidate), "policy": entry["id"]})
                        if candidate_freed >= total - int(entry["max_bytes"]):
                            break
                    print(f"REMOTE-SCRATCH {entry['id']}: planned_bytes={candidate_freed}")
                    freed += candidate_freed
            print(f"planned_reclaim_bytes={freed} mode={'dry-run' if dry_run else 'apply'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"jope-diskd: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
