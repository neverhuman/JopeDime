#!/usr/bin/env python3
"""Bind host runner provenance to the commit under test, or to the PR base on pull_request.

On pull_request, hosts must still run only reviewed merged code: the expected digest is
computed from the PR base ref (merged main), not the PR head. That lets runner-change PRs
go green on current hosts, merge, then deploy in a short post-merge freeze. Main-branch
and other non-PR runs still require an exact match to the checkout under test.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional

DEPLOYED_PATH_RE = re.compile(
    r"/(?:opt/jope-runner|etc/systemd/system|etc/tmpfiles\.d)/[^ \n]+"
)
HOOK_NAMES = ("job-started.sh", "job-policy.py")
# A PR base is a full commit id: exactly 40 lowercase hex characters (SHA-1, as GitHub sends it).
# Anything else (a branch, HEAD, an abbreviation, an option such as --upload-pack=...) is refused
# before it reaches git.
BASE_SHA_RE = re.compile(r"[0-9a-f]{40}")
# Upper bound on the explicit base fetch; JOPE_PROVENANCE_FETCH_TIMEOUT (seconds) overrides it.
FETCH_TIMEOUT_SECONDS = 120


def bind_rev(event_name: str, base_sha: str = "") -> Optional[str]:
    """Return the git rev whose deploy/runner tree is the provenance expectation.

    None means the current worktree (exact match to the commit under test).
    """
    if event_name == "pull_request":
        sha = base_sha or ""
        if not sha.strip():
            raise ValueError("pull_request provenance requires BASE_SHA (PR base commit)")
        if not BASE_SHA_RE.fullmatch(sha):
            raise ValueError(
                f"BASE_SHA must be exactly 40 lowercase hex characters, got {sha!r}"
            )
        return sha
    return None


def bind_rev_from_env(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    env = environ if environ is not None else os.environ
    event = env.get("GITHUB_EVENT_NAME", "")
    base = env.get("BASE_SHA", "")
    if not base.strip():
        event_path = env.get("GITHUB_EVENT_PATH", "")
        if event_path and Path(event_path).is_file():
            payload = json.loads(Path(event_path).read_text())
            base = payload.get("pull_request", {}).get("base", {}).get("sha", "") or ""
    return bind_rev(event, base)


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
    return value


def ensure_bind_rev(
    repo: Path, environ: Optional[Mapping[str, str]] = None, remote: str = "origin"
) -> Optional[str]:
    """Make the bind rev readable on a shallow (depth-1) checkout, then return it.

    A pull_request checkout of refs/pull/N/merge at the default depth holds only the merge
    commit, so `git show <BASE_SHA>:path` would fail on a cold workspace. Fetch exactly the
    base commit (--depth=1) instead of the full history. Non-PR runs bind to the worktree and
    fetch nothing. A malformed BASE_SHA is refused before git runs; a fetch that fails, exceeds
    fetch_timeout(), or still leaves the base unreadable fails closed.
    """
    env = environ if environ is not None else os.environ
    rev = bind_rev_from_env(env)
    if rev is None or has_commit(repo, rev):
        return rev
    timeout = fetch_timeout(env)
    git_env = fetch_env(repo, env)
    # Never wait on a credential prompt; a missing credential fails the fetch instead.
    git_env["GIT_TERMINAL_PROMPT"] = "0"
    argv = ["git", "fetch", "--no-tags", "--no-recurse-submodules", "--depth=1", remote, rev]
    # Own process group, so a timeout also kills git's transport helpers, not just git.
    proc = subprocess.Popen(
        argv, cwd=repo, env=git_env, stdout=sys.stderr, start_new_session=True
    )
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise ValueError(
            f"fetching PR base commit {rev} timed out after {timeout:g}s"
        ) from None
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, argv)
    if not has_commit(repo, rev):
        raise ValueError(f"PR base commit {rev} is not readable after fetching it")
    return rev


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


def require_scripts_match(host_sha: str, expected_sha: str) -> None:
    if not host_sha:
        raise ValueError(
            "the runner injected no provenance: re-run deploy/runner/provision.sh on the host"
        )
    if host_sha != expected_sha:
        raise ValueError(
            "the runner host is executing different files than the provenance bind requires"
        )


def require_entrypoint_match(image_sha: str, expected_sha: str) -> None:
    if image_sha != expected_sha:
        raise ValueError(
            "the image's entrypoint is not the one the provenance bind requires: "
            "rebuild the runner image"
        )


def require_hook_match(name: str, actual_sha: str, expected_sha: str) -> None:
    if actual_sha != expected_sha:
        raise ValueError(f"installed admission hook differs: {name}")


def require_image_identity(running: str, recorded: str) -> None:
    if running != recorded:
        raise ValueError(
            "the running image is not the one provision.sh recorded: re-run provision.sh"
        )


def verify_scripts_and_image(
    repo: Path, environ: Optional[Mapping[str, str]] = None
) -> dict:
    """CI action entry: scripts digest, image entrypoint, provisioned image identity."""
    env = environ if environ is not None else os.environ
    rev = bind_rev_from_env(env)
    expected = scripts_digest(repo, rev)
    host = env.get("JOPE_RUNNER_SCRIPTS_SHA256", "")
    require_scripts_match(host, expected)

    image_ep_path = Path(
        env.get("JOPE_RUNNER_ENTRYPOINT_PATH", "/usr/local/bin/jope-runner-entrypoint")
    )
    image_ep = hashlib.sha256(image_ep_path.read_bytes()).hexdigest()
    require_entrypoint_match(image_ep, entrypoint_digest(repo, rev))

    running = env.get("JOPE_RUNNER_IMAGE_ID", "x")
    recorded = env.get("JOPE_RUNNER_IMAGE_ID_RECORDED", "y")
    require_image_identity(running, recorded)

    return {
        "bind_rev": rev or "WORKTREE",
        "expected_scripts_sha256": expected,
        "host_scripts_sha256": host,
        "entrypoint_sha256": image_ep,
        "image_id": running,
    }


def verify_hooks(
    repo: Path,
    host_hook_dir: Path,
    environ: Optional[Mapping[str, str]] = None,
) -> None:
    """Compare mounted admission hooks to the provenance bind rev."""
    env = environ if environ is not None else os.environ
    rev = bind_rev_from_env(env)
    for name in HOOK_NAMES:
        actual = hashlib.sha256((host_hook_dir / name).read_bytes()).hexdigest()
        require_hook_match(name, actual, hook_digest(repo, rev, name))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    repo = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()
    if not args or args[0] == "verify":
        result = verify_scripts_and_image(repo)
        print(json.dumps(result, sort_keys=True))
        print(f"host:           {os.environ.get('JOPE_RUNNER_HOST', '?')}", flush=True)
        print(f"bind rev:       {result['bind_rev']}", flush=True)
        print(f"deployed sha:   {result['host_scripts_sha256']}", flush=True)
        print(f"expected sha:   {result['expected_scripts_sha256']}", flush=True)
        print(f"entrypoint:     image={result['entrypoint_sha256']}", flush=True)
        print(
            "image:          running="
            f"{result['image_id']} recorded="
            f"{os.environ.get('JOPE_RUNNER_IMAGE_ID_RECORDED', '?')}",
            flush=True,
        )
        return 0
    if args[0] == "ensure-base":
        rev = ensure_bind_rev(repo)
        print(f"bind rev:       {rev or 'WORKTREE'} (readable)", flush=True)
        return 0
    if args[0] == "digest":
        rev = None
        if len(args) > 1 and args[1] != "WORKTREE":
            rev = args[1]
        print(scripts_digest(repo, rev))
        return 0
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} [verify|ensure-base|digest [REV]]")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode() if exc.stderr else str(exc)
        sys.stderr.write(err)
        raise SystemExit(1) from exc
    except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
