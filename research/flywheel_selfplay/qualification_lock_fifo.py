#!/usr/bin/env python3
"""FIFO ticket queue and legacy-waiter backoff helper for ~/jope_plan/qualification.lock."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Callable

LOCK_PATH = Path.home() / "jope_plan/qualification.lock"
QUEUE_DIR = Path.home() / "jope_plan/qualification.queue"
JOPE_DATA_DIR = Path.home() / "jope-data"
LOCK_POLL_SECONDS = 30.0
LEGACY_WAIT_WINDOW_SECONDS = 90.0
LEGACY_BACKOFF_SECONDS = 45.0


def pid_is_alive(pid: int) -> bool:
    """Return True iff the given OS PID currently exists and is not a zombie."""
    if pid <= 0:
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            content = proc_stat.read_text(errors="replace")
            rparen = content.rfind(")")
            if rparen != -1:
                after = content[rparen + 1 :].split()
                if after and after[0] == "Z":
                    return False
        except OSError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def parse_ticket_name(name: str) -> tuple[int, int, str] | None:
    """Parse '<epoch_ns>-<pid>-<wave>' into (epoch_ns, pid, wave)."""
    parts = name.split("-", 2)
    if len(parts) != 3:
        return None
    epoch_str, pid_str, wave = parts
    if not epoch_str.isdigit() or not pid_str.isdigit() or not wave:
        return None
    return int(epoch_str), int(pid_str), wave


def enqueue_ticket(
    queue_dir: Path = QUEUE_DIR,
    *,
    pid: int | None = None,
    wave: str = "H61",
    epoch_ns: int | None = None,
) -> Path:
    """Create a FIFO ticket '<epoch_ns>-<pid>-<wave>' in queue_dir."""
    queue_dir.mkdir(parents=True, exist_ok=True)
    actual_pid = os.getpid() if pid is None else int(pid)
    actual_ns = time.time_ns() if epoch_ns is None else int(epoch_ns)
    safe_wave = "".join(ch for ch in str(wave) if ch.isalnum() or ch in ("_", ".")) or "wave"
    while True:
        ticket_path = queue_dir / f"{actual_ns}-{actual_pid}-{safe_wave}"
        try:
            fd = os.open(str(ticket_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as handle:
                handle.write(f"epoch_ns={actual_ns}\npid={actual_pid}\nwave={safe_wave}\n")
            return ticket_path
        except FileExistsError:
            actual_ns += 1


def remove_ticket(ticket_path: Path | None) -> None:
    """Delete a FIFO ticket file if present."""
    if ticket_path is None:
        return
    try:
        ticket_path.unlink(missing_ok=True)
    except OSError:
        pass


def prune_dead_tickets_and_list_live(
    queue_dir: Path = QUEUE_DIR,
    *,
    is_alive_fn: Callable[[int], bool] = pid_is_alive,
) -> list[tuple[int, int, str, Path]]:
    """Prune tickets whose PID is dead and return live tickets sorted by (epoch_ns, pid, wave)."""
    if not queue_dir.exists():
        return []
    live: list[tuple[int, int, str, Path]] = []
    for entry in queue_dir.iterdir():
        if not entry.is_file():
            continue
        parsed = parse_ticket_name(entry.name)
        if parsed is None:
            continue
        epoch_ns, pid, wave = parsed
        if not is_alive_fn(pid):
            remove_ticket(entry)
            continue
        live.append((epoch_ns, pid, wave, entry))
    live.sort(key=lambda item: (item[0], item[1], item[2], item[3].name))
    return live


def is_oldest_live_ticket(
    ticket_path: Path,
    queue_dir: Path = QUEUE_DIR,
    *,
    is_alive_fn: Callable[[int], bool] = pid_is_alive,
) -> tuple[bool, tuple[int, int, str, Path] | None]:
    """Return (True, oldest_ticket) iff ticket_path is the oldest live ticket in queue_dir."""
    live = prune_dead_tickets_and_list_live(queue_dir, is_alive_fn=is_alive_fn)
    if not live:
        return False, None
    oldest = live[0]
    return oldest[3].resolve() == ticket_path.resolve(), oldest


def other_wave_waiting_recently(
    data_root: Path = JOPE_DATA_DIR,
    *,
    own_status_path: Path | None = None,
    now_utc: datetime | None = None,
    window_seconds: float = LEGACY_WAIT_WINDOW_SECONDS,
) -> str | None:
    """Return a description if any other *_STATUS.log under ~/jope-data/h*/ logged WAIT qualification.lock in the last 90s."""
    if not data_root.exists():
        return None
    reference_now = now_utc or datetime.now(timezone.utc)
    own_resolved = own_status_path.resolve() if own_status_path is not None and own_status_path.exists() else None

    candidates: set[Path] = set()
    for pattern in ("h*/*_STATUS.log", "h*/*/*_STATUS.log"):
        for path in data_root.glob(pattern):
            if path.is_file():
                candidates.add(path)

    for log_path in sorted(candidates):
        try:
            if own_resolved is not None and log_path.resolve() == own_resolved:
                continue
        except OSError:
            continue
        try:
            lines = log_path.read_text(errors="replace").splitlines()[-20:]
        except OSError:
            continue
        for line in reversed(lines):
            if "WAIT qualification.lock" not in line:
                continue
            prefix = line.split("—", 1)[0].strip()
            try:
                ts = datetime.fromisoformat(prefix.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = (reference_now - ts).total_seconds()
                if 0.0 <= age <= window_seconds:
                    return f"legacy waiter in {log_path.name} ({age:.1f}s ago)"
            except ValueError:
                continue
    return None
