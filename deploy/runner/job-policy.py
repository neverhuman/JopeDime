#!/usr/bin/env python3
"""Admit only the workflow/event belonging to this root-selected storage domain."""
import json
import os
from pathlib import Path
import re
import signal
import sys


def authorize(policy, env):
    repo = "neverhuman/JopeDime"
    if env.get("GITHUB_REPOSITORY") != repo or env.get("GITHUB_REPOSITORY_ID") != "1376159324":
        raise ValueError("foreign repository")
    workflow = env.get("GITHUB_WORKFLOW_REF", "")
    event = env.get("GITHUB_EVENT_NAME")
    if policy["domain"] == "main":
        # Full hosts also carry jope-ci (owner permanent 2026-10-14). Admit ci.yml
        # PR/manual onto the full lane in addition to trusted main workflows.
        ci_ok = (
            workflow.startswith(f"{repo}/.github/workflows/ci.yml@")
            and event in {"pull_request", "workflow_dispatch"}
        )
        if not ci_ok:
            if env.get("GITHUB_REF") != "refs/heads/main":
                raise ValueError("trusted storage requires main")
            allowed = {f"{repo}/.github/workflows/full-main.yml@refs/heads/main": {"push", "workflow_dispatch"},
                       f"{repo}/.github/workflows/main-guard.yml@refs/heads/main": {"push"}}
            if event not in allowed.get(workflow, set()):
                raise ValueError("workflow/event is not admitted to trusted storage")
    elif policy["domain"] == "pr":
        if not workflow.startswith(f"{repo}/.github/workflows/ci.yml@") or event not in {"pull_request", "workflow_dispatch"}:
            raise ValueError("fast storage admits only ci PR/manual jobs")
    else:
        raise ValueError("unknown storage domain")
    if not re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", "")):
        raise ValueError("missing source identity")


def main():
    try:
        policy = json.loads(Path("/opt/jope-runner/policy.json").read_text())
        authorize(policy, os.environ)
        print(f"runner admission: {policy['domain']}/{policy['lane']} {os.environ['GITHUB_RUN_ID']}/{os.environ['GITHUB_RUN_ATTEMPT']}", flush=True)
    except (OSError, ValueError, KeyError) as exc:
        print(f"runner admission denied: {exc}", file=sys.stderr, flush=True)
        # Merely failing a hook could permit a malicious `if: always()` step.
        # Terminate every same-uid process (including Listener/Worker), then exit.
        # The non-root container has no host pid namespace or other user's processes.
        os.kill(-1, signal.SIGKILL)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
