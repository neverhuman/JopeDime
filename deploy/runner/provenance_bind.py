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
import urllib.request
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
# Upper bound on the explicit base fetch; JOPE_PROVENANCE_FETCH_TIMEOUT (seconds) overrides it,
# up to FETCH_TIMEOUT_MAX_SECONDS. A larger override fails closed rather than being clamped.
FETCH_TIMEOUT_SECONDS = 120
FETCH_TIMEOUT_MAX_SECONDS = 600
# `digest REV` takes WORKTREE, a full 40-hex commit id, or a plain ref name. No revision syntax
# (~ ^ : @{ ..), no leading '-' (option injection), no whitespace.
DIGEST_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}")


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
        if sha == "0" * 40:
            raise ValueError("BASE_SHA cannot be the all-zero push sentinel")
        return sha
    return None


def github_public_json(path: str, environ: Mapping[str, str]) -> dict:
    """Read only this pinned repository API; never log token, headers or response bodies."""
    token = environ.get("JOPE_PROVENANCE_FETCH_TOKEN", "")
    if not token:
        raise ValueError("manual PR provenance requires the read-only job token")
    request = urllib.request.Request(
        "https://api.github.com/repos/neverhuman/JopeDime/" + path,
        headers={"Accept": "application/vnd.github+json", "Authorization": "Bearer " + token,
                 "X-GitHub-Api-Version": "2022-11-28"})
    class RefuseRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("manual PR provenance refuses an API redirect")
    opener = urllib.request.build_opener(RefuseRedirect())
    with opener.open(request, timeout=30) as response:
        raw = response.read(2_097_153)
    if len(raw) > 2_097_152:
        raise ValueError("manual PR provenance API response exceeds its bound")
    return json.loads(raw)


def manual_pr_bind(environ: Mapping[str, str], number: str) -> str:
    """Full manual candidate tests on merged runner policy, just like native PR CI.

    This does not certify candidate runner deployment. After merge, receiving CI still
    requires exact merged-head host bytes and the new roles need live qualification.
    """
    if not re.fullmatch(r"[1-9][0-9]{0,8}", number):
        raise ValueError("provenance_pr must be a positive PR number")
    repo = "neverhuman/JopeDime"
    if environ.get("GITHUB_REPOSITORY") != repo or environ.get("GITHUB_REPOSITORY_ID") != "1376159324":
        raise ValueError("manual PR provenance repository identity differs")
    sha = environ.get("GITHUB_SHA", "")
    bind_rev("pull_request", sha)  # full nonzero SHA, before any API call
    pr = github_public_json("pulls/" + number, environ)
    if (pr.get("number") != int(number) or pr.get("state") != "open" or pr.get("merged") is not False
            or pr.get("base", {}).get("ref") != "main"):
        raise ValueError("manual PR provenance requires this open PR targeting main")
    for side in ("base", "head"):
        repository = pr.get(side, {}).get("repo") or {}
        if repository.get("id") != 1376159324 or repository.get("full_name") != repo:
            raise ValueError("manual PR provenance refuses a fork or transferred repository")
    head = pr["head"]
    ref = environ.get("GITHUB_REF", "")
    if (head.get("sha") != sha or head.get("ref") == "main"
            or ref != "refs/heads/" + head.get("ref", "")
            or environ.get("GITHUB_WORKFLOW_REF") != repo + "/.github/workflows/ci.yml@" + ref):
        raise ValueError("manual PR provenance head, branch or workflow differs")
    base = bind_rev("pull_request", pr["base"].get("sha", ""))
    if github_public_json("git/ref/heads/main", environ).get("object", {}).get("sha") != base:
        raise ValueError("manual PR provenance base is no longer current main")
    return base


def bind_rev_from_env(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    env = environ if environ is not None else os.environ
    event = env.get("GITHUB_EVENT_NAME", "")
    # An explicit input is read from GitHub's dispatch payload, not a caller-supplied base.
    # No input keeps the original exact-head rule for workflow_dispatch/main.
    if event == "workflow_dispatch":
        event_path = env.get("GITHUB_EVENT_PATH", "")
        payload = json.loads(Path(event_path).read_text()) if event_path else {}
        # GitHub declares dispatch inputs as object or null (including no-input workflows).
        inputs = payload.get("inputs")
        if inputs is None:
            inputs = {}
        if not isinstance(inputs, dict):
            raise ValueError("workflow_dispatch inputs must be an object or null")
        number = inputs.get("provenance_pr", "")
        if number:
            return manual_pr_bind(env, number)
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


def ensure_bind_rev(
    repo: Path, environ: Optional[Mapping[str, str]] = None, remote: str = "origin"
) -> Optional[str]:
    """Make the bind rev readable on a shallow (depth-1) checkout, then return it.

    A pull_request checkout of refs/pull/N/merge at the default depth holds only the merge
    commit, so `git show <BASE_SHA>:path` would fail on a cold workspace. Fetch exactly the
    base commit (--depth=1) instead of the full history. Non-PR runs bind to the worktree and
    fetch nothing. A malformed BASE_SHA is refused before git runs; a stale shallow.lock from a
    killed fetch is removed only when no process holds it; a fetch that fails, exceeds
    fetch_timeout(), or still leaves the base unreadable fails closed.
    """
    env = environ if environ is not None else os.environ
    rev = bind_rev_from_env(env)
    if rev is None or has_commit(repo, rev):
        return rev
    timeout = fetch_timeout(env)
    clear_stale_shallow_lock(repo)
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


def verify_mac_git(
    repo: Path, host_hook_dir: Path, environ: Optional[Mapping[str, str]] = None
) -> None:
    """Bind Mac-only sidecars and the actual immutable tool to the reviewed base.

    Call only after verifying host=xbabe4 from the immutable six-field launch policy,
    never from a job-controlled environment variable.
    """
    rev = bind_rev_from_env(environ)
    for name in ("mac-git-entrypoint.sh", "mac-capacity.json"):
        actual = hashlib.sha256((host_hook_dir / name).read_bytes()).hexdigest()
        expected = hashlib.sha256(file_bytes(repo, rev, f"deploy/runner/{name}")).hexdigest()
        require_hook_match(name, actual, expected)
    policy = json.loads(file_bytes(repo, rev, "deploy/runner/mac-capacity.json"))["git_emulation"]
    result = subprocess.run(
        ["/bin/bash", str(host_hook_dir / "mac-git-entrypoint.sh"), "--inspect-tool",
         str(host_hook_dir / "mac-capacity.json"), "/opt/jope-git-emulation/qemu-x86_64"],
        check=True, text=True, capture_output=True, timeout=20,
    )
    expected = dict(binary_sha256=policy["binary_sha256"], binary_bytes=policy["binary_bytes"],
                    architecture="aarch64", uid=0, gid=0, mode="0555",
                    volume=policy["volume"], version=policy["version"])
    if json.loads(result.stdout) != expected:
        raise ValueError("Mac Git tool facts differ from the reviewed policy")


def runner_pool(repo: Path, environ: Optional[Mapping[str, str]] = None) -> str:
    """Select only a capability pool declared by the provenance-bound merged policy.

    A candidate's unmerged label never changes where its own premerge jobs run.
    The composite action calls this only after the installed scripts/hooks/kernel proof.
    """
    rev = bind_rev_from_env(environ)
    policy = json.loads(file_bytes(repo, rev, "deploy/runner/mac-capacity.json"))
    label = policy.get("runner_label", "jope-ci")
    if label not in ("jope-ci", "jope-mac-ci"):
        raise ValueError("unapproved provenance-bound CI capability label")
    return label


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
    if args[0] == "runner-pool":
        print(runner_pool(repo))
        return 0
    if args[0] == "ensure-base":
        rev = ensure_bind_rev(repo)
        print(f"bind rev:       {rev or 'WORKTREE'} (readable)", flush=True)
        return 0
    if args[0] == "digest":
        if len(args) > 2:
            raise ValueError("digest takes at most one REV")
        rev = validate_digest_rev(args[1]) if len(args) > 1 else None
        print(scripts_digest(repo, rev))
        return 0
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} [verify|ensure-base|runner-pool|digest [REV]]")


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
