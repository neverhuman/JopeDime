#!/usr/bin/env python3
"""Provenance bind: PR base vs exact-head for runner-change PRs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_bind():
    path = ROOT / "deploy/runner/provenance_bind.py"
    spec = importlib.util.spec_from_file_location("provenance_bind", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bind = load_bind()

COMMON_TEMPLATE = """#!/bin/bash
JOPE_RUNNER_DEPLOYED=(
  /opt/jope-runner/common.sh
  /opt/jope-runner/run.sh
  /opt/jope-runner/disk-guard.sh
  /opt/jope-runner/prune.sh
  /opt/jope-runner/job-started.sh
  /opt/jope-runner/job-policy.py
  /etc/systemd/system/jope-runner@.service
  /etc/systemd/system/jope-runner@.service.d/runtime-max.conf
  /etc/systemd/system/jope-runner@.timer
  /etc/systemd/system/jope-runner-prune.service
  /etc/systemd/system/jope-runner-prune.timer
  /etc/systemd/system/jope-docker-proxy.service
  /etc/tmpfiles.d/jope-exclusive-cpu.conf
)
# marker={marker}
"""

DEPLOYED_REL = [
    "deploy/runner/common.sh",
    "deploy/runner/run.sh",
    "deploy/runner/disk-guard.sh",
    "deploy/runner/prune.sh",
    "deploy/runner/job-started.sh",
    "deploy/runner/job-policy.py",
    "deploy/runner/jope-runner@.service",
    "deploy/runner/jope-runner@.service.d/runtime-max.conf",
    "deploy/runner/jope-runner@.timer",
    "deploy/runner/jope-runner-prune.service",
    "deploy/runner/jope-runner-prune.timer",
    "deploy/runner/jope-docker-proxy.service",
    "deploy/runner/jope-exclusive-cpu.conf",
]

BASH_DIGEST = r"""
mapfile -t deployed < <(sed -n '/^JOPE_RUNNER_DEPLOYED=(/,/^)/p' deploy/runner/common.sh   | grep -oE '/(opt/jope-runner|etc/systemd/system|etc/tmpfiles.d)/[^ ]+')
files=()
for d in "${deployed[@]}"; do
  rel=${d#/opt/jope-runner/}
  rel=${rel#/etc/systemd/system/}
  rel=${rel#/etc/tmpfiles.d/}
  files+=("deploy/runner/$rel")
done
cat "${files[@]}" | sha256sum | cut -d' ' -f1
"""


def write_tree(repo: Path, marker: str) -> None:
    for rel in DEPLOYED_REL:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("common.sh"):
            path.write_text(COMMON_TEMPLATE.format(marker=marker))
        else:
            path.write_text(f"{rel}\n{marker}\n")
    (repo / "deploy/runner/entrypoint.sh").write_text(f"entrypoint\n{marker}\n")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def digest_of(repo: Path, rev=None) -> str:
    return bind.scripts_digest(repo, rev)


class ProvenanceBindTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        write_tree(self.repo, "base-v1")
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "test@example.com")
        git(self.repo, "config", "user.name", "test")
        git(self.repo, "config", "advice.defaultBranchName", "false")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-m", "base")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        self.base_digest = digest_of(self.repo, self.base_sha)

        write_tree(self.repo, "head-v2-runner-change")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-m", "runner change")
        self.head_sha = git(self.repo, "rev-parse", "HEAD")
        self.head_digest = digest_of(self.repo, None)
        self.assertNotEqual(self.base_digest, self.head_digest)

        self.ep_base = hashlib.sha256(
            bind.file_bytes(self.repo, self.base_sha, "deploy/runner/entrypoint.sh")
        ).hexdigest()
        self.ep_head = hashlib.sha256(
            (self.repo / "deploy/runner/entrypoint.sh").read_bytes()
        ).hexdigest()
        self.assertNotEqual(self.ep_base, self.ep_head)

        self.ep_file = self.repo / "fake-image-entrypoint"
        self.image_id = "sha256:image-fixed"

    def tearDown(self):
        self.tmp.cleanup()

    def _env(self, event, host_digest, entrypoint_marker, **extra):
        if entrypoint_marker == "base":
            self.ep_file.write_bytes(
                bind.file_bytes(self.repo, self.base_sha, "deploy/runner/entrypoint.sh")
            )
        else:
            self.ep_file.write_bytes(
                (self.repo / "deploy/runner/entrypoint.sh").read_bytes()
            )
        env = {
            "GITHUB_EVENT_NAME": event,
            "JOPE_RUNNER_SCRIPTS_SHA256": host_digest,
            "JOPE_RUNNER_ENTRYPOINT_PATH": str(self.ep_file),
            "JOPE_RUNNER_IMAGE_ID": self.image_id,
            "JOPE_RUNNER_IMAGE_ID_RECORDED": self.image_id,
            "JOPE_RUNNER_HOST": "xbabe1",
            "GITHUB_WORKSPACE": str(self.repo),
        }
        env.update(extra)
        return env

    def test_pull_request_passes_when_host_matches_base_not_head(self):
        env = self._env("pull_request", self.base_digest, "base", BASE_SHA=self.base_sha)
        result = bind.verify_scripts_and_image(self.repo, env)
        self.assertEqual(result["bind_rev"], self.base_sha)
        self.assertEqual(result["expected_scripts_sha256"], self.base_digest)

    def test_pull_request_fails_when_host_at_other_digest(self):
        other = hashlib.sha256(b"not-base-not-head").hexdigest()
        env = self._env("pull_request", other, "base", BASE_SHA=self.base_sha)
        with self.assertRaisesRegex(ValueError, "different files"):
            bind.verify_scripts_and_image(self.repo, env)

    def test_pull_request_fails_when_host_only_matches_head(self):
        env = self._env("pull_request", self.head_digest, "head", BASE_SHA=self.base_sha)
        with self.assertRaisesRegex(ValueError, "different files"):
            bind.verify_scripts_and_image(self.repo, env)

    def test_main_push_requires_exact_match_to_checkout(self):
        env = self._env("push", self.head_digest, "head")
        result = bind.verify_scripts_and_image(self.repo, env)
        self.assertEqual(result["bind_rev"], "WORKTREE")
        self.assertEqual(result["expected_scripts_sha256"], self.head_digest)
        with self.assertRaisesRegex(ValueError, "different files"):
            bind.verify_scripts_and_image(
                self.repo, self._env("push", self.base_digest, "base")
            )

    def test_workflow_dispatch_requires_exact_match(self):
        env = self._env("workflow_dispatch", self.head_digest, "head")
        bind.verify_scripts_and_image(self.repo, env)
        with self.assertRaisesRegex(ValueError, "different files"):
            bind.verify_scripts_and_image(
                self.repo, self._env("workflow_dispatch", self.base_digest, "base")
            )

    def test_pull_request_without_base_sha_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "BASE_SHA"):
            bind.bind_rev("pull_request", "")

    def test_hooks_follow_same_bind(self):
        host = Path(self.tmp.name) / "host-hooks"
        host.mkdir()
        for name in bind.HOOK_NAMES:
            (host / name).write_bytes(
                bind.file_bytes(self.repo, self.base_sha, f"deploy/runner/{name}")
            )
        env = {"GITHUB_EVENT_NAME": "pull_request", "BASE_SHA": self.base_sha}
        bind.verify_hooks(self.repo, host, env)
        for name in bind.HOOK_NAMES:
            (host / name).write_bytes((self.repo / "deploy/runner" / name).read_bytes())
        with self.assertRaisesRegex(ValueError, "admission hook differs"):
            bind.verify_hooks(self.repo, host, env)
        env_push = {"GITHUB_EVENT_NAME": "push"}
        bind.verify_hooks(self.repo, host, env_push)

    def test_event_path_supplies_base_sha(self):
        event = Path(self.tmp.name) / "event.json"
        event.write_text(json.dumps({"pull_request": {"base": {"sha": self.base_sha}}}))
        env = self._env(
            "pull_request", self.base_digest, "base", GITHUB_EVENT_PATH=str(event)
        )
        env.pop("BASE_SHA", None)
        result = bind.verify_scripts_and_image(self.repo, env)
        self.assertEqual(result["bind_rev"], self.base_sha)

    def test_parse_matches_repo_common_sh(self):
        common = (ROOT / "deploy/runner/common.sh").read_text()
        paths = bind.parse_deployed_relpaths(common)
        self.assertGreaterEqual(len(paths), 10)
        live = bind.scripts_digest(ROOT, None)
        bash = subprocess.check_output(
            ["bash", "-ec", BASH_DIGEST], cwd=ROOT, text=True
        ).strip()
        self.assertEqual(live, bash)


if __name__ == "__main__":
    unittest.main()
