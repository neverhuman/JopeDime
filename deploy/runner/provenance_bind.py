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
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Mapping, Optional

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_jope_provenance_bind_helpers", Path(__file__).with_name("provenance_bind_helpers.py")
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError("cannot load provenance bind helpers")
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPERS)
for _name in (
    "BASE_SHA_RE", "DEPLOYED_PATH_RE", "DIGEST_REF_RE", "FETCH_TIMEOUT_SECONDS",
    "FETCH_TIMEOUT_MAX_SECONDS", "HOOK_NAMES", "_git_show", "clear_stale_shallow_lock",
    "entrypoint_digest", "fetch_env", "fetch_timeout", "file_bytes", "git_dir",
    "has_commit", "hook_digest", "lock_holders", "parse_deployed_relpaths",
    "scripts_digest", "validate_digest_rev",
):
    if hasattr(_HELPERS, _name):
        globals()[_name] = getattr(_HELPERS, _name)


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
