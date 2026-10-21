#!/usr/bin/env python3
"""Security entry point must execute every gate and propagate scanner failures."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class SecurityLane(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "target" / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="security-controls-", dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        for name in ("tools/security-lane.sh", "deploy/ci/secret-scan.sh", "deploy/ci/actionlint-workflows.sh", ".gitleaks.toml"):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        workflows = self.root / ".github" / "workflows"
        workflows.mkdir(parents=True)
        (workflows / "ci.yml").write_text("name: fixture\non: push\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for command in ("bash", "dirname", "realpath", "find", "sort", "grep", "wc"):
            (self.bin / command).symlink_to(shutil.which(command))
        self.fake("git", 'case "$*" in *--is-shallow-repository*) echo false ;; *--verify*) printf "%040d\\n" 1 ;; *--count*) echo 7 ;; *) exit 92 ;; esac')
        self.fake("actionlint", 'exit "${FAIL_ACTIONLINT:-0}"')
        self.fake("gitleaks", '[[ "$*" == *"--redact"* && "$*" == *"--log-opts=HEAD"* ]] || exit 93; exit "${FAIL_GITLEAKS:-0}"')
        self.fake("npm", '[[ "$*" == "--prefix web/training-dashboard audit --package-lock-only --audit-level=high" ]] || exit 95; exit "${FAIL_NPM:-0}"')
        self.fake("cargo", 'case "$*" in "deny --version") echo "cargo-deny 0.20.2" ;; "deny --locked --all-features list --layout crate") printf "torch-sys@1: MIT\\ncudarc@1: MIT\\n"; [[ "${OMIT_CANARY:-0}" == 1 ]] || echo "safetensors@1: Apache-2.0" ;; "deny --locked --all-features check advisories bans licenses sources") exit "${FAIL_DENY:-0}" ;; *) exit 94 ;; esac')

    def fake(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\nset -euo pipefail\n" + body + "\n")
        path.chmod(0o755)

    def run_lane(self, **extra):
        env = dict(os.environ, PATH=str(self.bin), **extra)
        return subprocess.run(["/bin/bash", str(self.root / "tools/security-lane.sh")], env=env, capture_output=True, text=True)

    def test_authorized_path_executes_all_gates(self):
        result = self.run_lane()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("optional-feature deps present", result.stdout)

    def test_workflow_failure_propagates(self):
        self.assertEqual(self.run_lane(FAIL_ACTIONLINT="17").returncode, 17)

    def test_secret_failure_propagates(self):
        self.assertEqual(self.run_lane(FAIL_GITLEAKS="18").returncode, 18)

    def test_dependency_advisory_failure_propagates(self):
        self.assertEqual(self.run_lane(FAIL_DENY="19").returncode, 19)

    def test_frontend_advisory_failure_propagates(self):
        self.assertEqual(self.run_lane(FAIL_NPM="20").returncode, 20)

    def test_optional_dependency_canary_is_required(self):
        result = self.run_lane(OMIT_CANARY="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing: safetensors", result.stderr)

    def test_missing_tool_is_not_a_skip(self):
        (self.bin / "gitleaks").unlink()
        result = self.run_lane()
        self.assertEqual(result.returncode, 127)
        self.assertIn("requires gitleaks", result.stderr)


if __name__ == "__main__":
    unittest.main()
