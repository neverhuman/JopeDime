"""Git and deployed-file helpers for provenance_bind.py."""
from __future__ import annotations

import base64
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional

DEPLOYED_PATH_RE = re.compile(
    r"/(?:opt/jope-runner|etc/systemd/system|etc/tmpfiles\.d)/[^ \n]+"
)
HOOK_NAMES = ("job-started.sh", "job-policy.py")
BASE_SHA_RE = re.compile(r"[0-9a-f]{40}")
FETCH_TIMEOUT_SECONDS = 120
FETCH_TIMEOUT_MAX_SECONDS = 600
DIGEST_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}")

def has_commit(repo: Path, rev: str) -> bool:
    """True when rev names a commit whose objects are present in repo."""
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{rev}^{{commit}}"],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def fetch_env(repo: Path, environ: Mapping[str, str]) -> dict:
    """Environment for the base fetch.

    actions/checkout runs with persist-credentials: false, so the clone carries no auth. When
    JOPE_PROVENANCE_FETCH_TOKEN is set, pass it as an http.<server>/.extraheader through
    GIT_CONFIG_* (never argv, never .git/config). An extraheader already configured for the
    server wins, so two Authorization headers are never sent.
    """
    env = dict(os.environ)
    token = environ.get("JOPE_PROVENANCE_FETCH_TOKEN", "")
    if not token:
        return env
    server = (environ.get("GITHUB_SERVER_URL") or "https://github.com").rstrip("/")
    key = f"http.{server}/.extraheader"
    existing = subprocess.run(
        ["git", "config", "--get-all", key],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.strip()
    if existing:
        return env
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    n = int(env.get("GIT_CONFIG_COUNT", "0") or "0")
    env[f"GIT_CONFIG_KEY_{n}"] = key
    env[f"GIT_CONFIG_VALUE_{n}"] = f"AUTHORIZATION: basic {basic}"
    env["GIT_CONFIG_COUNT"] = str(n + 1)
    return env


def fetch_timeout(environ: Mapping[str, str]) -> float:
    """Seconds the base fetch may take. A malformed or non-positive override fails closed."""
    raw = environ.get("JOPE_PROVENANCE_FETCH_TIMEOUT", "")
    if not raw:
        return float(FETCH_TIMEOUT_SECONDS)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"JOPE_PROVENANCE_FETCH_TIMEOUT is not a number: {raw!r}") from None
    if not value > 0 or value == float("inf"):
        raise ValueError(f"JOPE_PROVENANCE_FETCH_TIMEOUT must be positive and finite: {raw!r}")
    if value > FETCH_TIMEOUT_MAX_SECONDS:
        raise ValueError(
            f"JOPE_PROVENANCE_FETCH_TIMEOUT {raw!r} exceeds the {FETCH_TIMEOUT_MAX_SECONDS}s cap"
        )
    return value


def git_dir(repo: Path) -> Path:
    out = subprocess.check_output(
        ["git", "rev-parse", "--absolute-git-dir"], cwd=repo, stderr=subprocess.PIPE, text=True
    )
    return Path(out.strip())


def lock_holders(lock: Path, proc_root: Path = Path("/proc")) -> list[str]:
    """Processes that may hold `lock`: any pid with an open fd on it, and any git process
    whose fds cannot be inspected. Raises when /proc itself is unreadable (fail closed)."""
    target = os.path.realpath(lock)
    holders: list[str] = []
    try:
        pids = [entry for entry in os.listdir(proc_root) if entry.isdigit()]
    except OSError as exc:
        raise ValueError(f"cannot inspect processes for {lock}: {exc}") from exc
    for pid in pids:
        if int(pid) == os.getpid():
            continue
        try:
            comm = (proc_root / pid / "comm").read_text().strip()
        except OSError:
            continue  # exited
        try:
            fds = os.listdir(proc_root / pid / "fd")
        except FileNotFoundError:
            continue
        except OSError:
            if comm.startswith("git"):
                holders.append(f"{pid} ({comm}, fds unreadable)")
            continue
        for fd in fds:
            try:
                if os.readlink(proc_root / pid / "fd" / fd) == target:
                    holders.append(f"{pid} ({comm})")
                    break
            except OSError:
                continue
    return holders


def clear_stale_shallow_lock(repo: Path, proc_root: Path = Path("/proc")) -> bool:
    """Remove a `.git/shallow.lock` left by a killed fetch in a reused workspace.

    Only when no process holds it; a held lock (or one whose holders cannot be ruled out)
    fails closed. Returns True when a stale lock was removed.
    """
    lock = git_dir(repo) / "shallow.lock"
    if not lock.exists():
        return False
    holders = lock_holders(lock, proc_root)
    if holders:
        raise ValueError(f"{lock} is held by {', '.join(holders)}; refusing to fetch")
    lock.unlink()
    print(f"removed stale {lock} (no process held it)", file=sys.stderr, flush=True)
    return True


def validate_digest_rev(arg: str) -> Optional[str]:
    """`digest` argument: WORKTREE means the checkout; otherwise a 40-hex id or a plain ref."""
    if arg == "WORKTREE":
        return None
    if BASE_SHA_RE.fullmatch(arg):
        return arg
    if (
        DIGEST_REF_RE.fullmatch(arg)
        and ".." not in arg
        and "//" not in arg
        and not arg.endswith((".", "/", ".lock"))
    ):
        return arg
    raise ValueError(f"digest REV must be WORKTREE, a 40-hex commit id or a plain ref: {arg!r}")



def parse_deployed_relpaths(common_sh: str) -> list[str]:
    """Mirror the action.yml sed/grep parse of JOPE_RUNNER_DEPLOYED → deploy/runner/… paths."""
    lines = common_sh.splitlines()
    try:
        start = next(
            i for i, line in enumerate(lines) if line.startswith("JOPE_RUNNER_DEPLOYED=(")
        )
    except StopIteration as exc:
        raise ValueError("JOPE_RUNNER_DEPLOYED missing from common.sh") from exc
    block: list[str] = []
    for line in lines[start:]:
        block.append(line)
        if line.strip() == ")":
            break
    else:
        raise ValueError("JOPE_RUNNER_DEPLOYED list is not closed")
    text = "\n".join(block)
    deployed = DEPLOYED_PATH_RE.findall(text)
    if len(deployed) < 10:
        raise ValueError(f"parsed only {len(deployed)} deployed files from common.sh")
    relpaths: list[str] = []
    for path in deployed:
        rel = path
        for prefix in (
            "/opt/jope-runner/",
            "/etc/systemd/system/",
            "/etc/tmpfiles.d/",
        ):
            if rel.startswith(prefix):
                rel = rel[len(prefix) :]
                break
        relpaths.append(f"deploy/runner/{rel}")
    return relpaths


def _git_show(repo: Path, rev: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{rev}:{path}"],
        cwd=repo,
        stderr=subprocess.PIPE,
    )


def file_bytes(repo: Path, rev: Optional[str], path: str) -> bytes:
    if rev is None:
        return (repo / path).read_bytes()
    return _git_show(repo, rev, path)


def scripts_digest(repo: Path, rev: Optional[str] = None) -> str:
    common = file_bytes(repo, rev, "deploy/runner/common.sh").decode()
    paths = parse_deployed_relpaths(common)
    blob = b"".join(file_bytes(repo, rev, path) for path in paths)
    return hashlib.sha256(blob).hexdigest()


def entrypoint_digest(repo: Path, rev: Optional[str] = None) -> str:
    return hashlib.sha256(file_bytes(repo, rev, "deploy/runner/entrypoint.sh")).hexdigest()


def hook_digest(repo: Path, rev: Optional[str], name: str) -> str:
    if name not in HOOK_NAMES:
        raise ValueError(f"unapproved hook: {name}")
    return hashlib.sha256(file_bytes(repo, rev, f"deploy/runner/{name}")).hexdigest()
