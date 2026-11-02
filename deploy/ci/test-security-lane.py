#!/usr/bin/env python3
"""Security entry point must execute every gate and propagate scanner failures."""
import hashlib
import json
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
        self.real_gitleaks = shutil.which("gitleaks")
        self.real_python3 = shutil.which("python3")
        for name in ("tools/security-lane.sh", "deploy/ci/secret-scan.sh", "deploy/ci/actionlint-workflows.sh", "deploy/ci/verify-vendor-security.py", ".gitleaks.toml"):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        workflows = self.root / ".github" / "workflows"
        workflows.mkdir(parents=True)
        (workflows / "ci.yml").write_text("name: fixture\non: push\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for command in ("bash", "dirname", "realpath", "find", "sort", "grep", "wc", "mkdir"):
            (self.bin / command).symlink_to(shutil.which(command))
        self.fake("git", 'case "$*" in *--is-shallow-repository*) echo false ;; *--verify*) printf "%040d\\n" 1 ;; *--count*) echo 7 ;; *) exit 92 ;; esac')
        self.fake("python3", 'case "$*" in "-B deploy/ci/verify-vendor-security.py --root "*) exit "${FAIL_PROVENANCE:-0}" ;; *) exec "$REAL_PYTHON3" "$@" ;; esac')
        self.fake("syft", 'case "$*" in "version") echo "syft 1.40.0" ;; *"--output spdx-json="*) rc="${FAIL_SBOM:-0}"; [ "$rc" = 0 ] || exit "$rc"; output="${@: -1}"; output="${output#spdx-json=}"; printf %s "{\\"spdxVersion\\":\\"SPDX-2.3\\",\\"packages\\":[]}" > "$output" ;; *) exit 97 ;; esac')
        self.fake("actionlint", 'exit "${FAIL_ACTIONLINT:-0}"')
        self.fake("gitleaks", '[[ "$*" == *"--redact"* && "$*" == *"--log-opts=HEAD"* ]] || exit 93; exit "${FAIL_GITLEAKS:-0}"')
        self.fake("npm", '[[ "$*" == "--prefix web/training-dashboard audit --package-lock-only --audit-level=high" ]] || exit 95; exit "${FAIL_NPM:-0}"')
        self.fake("cargo", r'''
printf '%s\n' "$*" >> "$CARGO_CALL_TRACE"
case "$*" in
  "deny --version") echo "cargo-deny 0.20.2" ;;
  "deny --locked --all-features list --layout crate")
    printf "torch-sys@1: MIT\ncudarc@1: MIT\n"
    [[ "${OMIT_CANARY:-0}" == 1 ]] || echo "safetensors@1: Apache-2.0" ;;
  "deny --locked --all-features check advisories bans licenses sources")
    exit "${FAIL_DENY:-0}" ;;
  "deny --manifest-path web/browser-predict/Cargo.toml --config deny.toml --locked --all-features list --layout crate")
    [[ "${FAIL_BROWSER_GRAPH:-0}" == 0 ]] || exit "$FAIL_BROWSER_GRAPH"
    for package in jope-browser-predict feat-math jope-runtime; do
      [[ "${OMIT_BROWSER_CANARY:-}" == "$package" ]] || printf '%s@1: Apache-2.0\n' "$package"
    done ;;
  "deny --manifest-path web/browser-predict/Cargo.toml --config deny.toml --locked --all-features check advisories bans licenses sources")
    exit "${FAIL_BROWSER_DENY:-0}" ;;
  *) exit 94 ;;
esac
'''.strip())

    def fake(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\nset -euo pipefail\n" + body + "\n")
        path.chmod(0o755)

    def run_lane(self, **extra):
        env = dict(os.environ, PATH=str(self.bin), REAL_PYTHON3=self.real_python3, CARGO_CALL_TRACE=str(self.root / "cargo-calls"), **extra)
        return subprocess.run(["/bin/bash", str(self.root / "tools/security-lane.sh")], env=env, capture_output=True, text=True)

    def test_sbom_is_retained_in_lane_evidence(self):
        evidence = self.root / "receipt"
        result = self.run_lane(JOPE_LANE_EVIDENCE=str(evidence))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        sbom = json.loads((evidence / "jope.spdx.json").read_text())
        self.assertEqual(sbom["spdxVersion"], "SPDX-2.3")
        self.assertEqual(sbom["packages"], [])

    def test_authorized_path_executes_all_gates(self):
        result = self.run_lane()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("optional-feature deps present", result.stdout)
        self.assertIn("browser/runtime deps present", result.stdout)
        calls = (self.root / "cargo-calls").read_text().splitlines()
        root_check = "deny --locked --all-features check advisories bans licenses sources"
        browser_check = "deny --manifest-path web/browser-predict/Cargo.toml --config deny.toml --locked --all-features check advisories bans licenses sources"
        self.assertIn(root_check, calls)
        self.assertIn(browser_check, calls)
        self.assertLess(calls.index(root_check), calls.index(browser_check))

    def test_provenance_failure_propagates(self):
        self.assertEqual(self.run_lane(FAIL_PROVENANCE="21").returncode, 21)

    def test_h205_content_identity_allowlist_does_not_hide_api_keys(self):
        self.assertIsNotNone(self.real_gitleaks, "gitleaks is required for the real config control")
        source = self.root / "h205-content-identities"
        family = source / "research" / "flywheel_selfplay" / "h205_fresh_vs_static"
        family.mkdir(parents=True)
        identifier = hashlib.sha256(b"H205 synthetic content identity control").hexdigest()[:40]
        (family / "README.md").write_text(f"family_key: {identifier}`\n")
        (family / "PREREG.md").write_text(f"family_key = {identifier}`\n")
        (family / "metrics.json").write_text(f'{{"family_key": "{identifier}"}}\n')
        (family / "unreviewed.json").write_text(f'{{"api_key": "{identifier}"}}\n')
        report = self.root / "h205-content-identities-report.json"
        result = subprocess.run(
            [
                self.real_gitleaks,
                "detect",
                "--no-git",
                "--redact=100",
                "--no-banner",
                "--config",
                str(ROOT / ".gitleaks.toml"),
                "--source",
                ".",
                "--report-format",
                "json",
                "--report-path",
                str(report),
            ],
            cwd=source,
            capture_output=True,
            text=True,
            timeout=30,
        )
        findings = json.loads(report.read_text()) if report.exists() else []
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            [(finding["RuleID"], finding["File"]) for finding in findings],
            [("generic-api-key", "research/flywheel_selfplay/h205_fresh_vs_static/unreviewed.json")],
        )

    def test_sbom_failure_propagates(self):
        self.assertEqual(self.run_lane(FAIL_SBOM="22").returncode, 22)

    def test_workflow_failure_propagates(self):
        self.assertEqual(self.run_lane(FAIL_ACTIONLINT="17").returncode, 17)

    def test_secret_failure_propagates(self):
        self.assertEqual(self.run_lane(FAIL_GITLEAKS="18").returncode, 18)

    def test_dependency_advisory_failure_propagates(self):
        self.assertEqual(self.run_lane(FAIL_DENY="19").returncode, 19)

    def test_browser_advisory_failure_propagates_after_root_success(self):
        self.assertEqual(self.run_lane(FAIL_BROWSER_DENY="21").returncode, 21)
        calls = (self.root / "cargo-calls").read_text().splitlines()
        self.assertIn("deny --locked --all-features check advisories bans licenses sources", calls)

    def test_browser_graph_failure_is_not_a_skip(self):
        self.assertEqual(self.run_lane(FAIL_BROWSER_GRAPH="22").returncode, 22)

    def test_browser_dependency_canaries_are_required(self):
        for package in ("jope-browser-predict", "feat-math", "jope-runtime"):
            with self.subTest(package=package):
                result = self.run_lane(OMIT_BROWSER_CANARY=package)
                self.assertNotEqual(0, result.returncode)
                self.assertIn(f"missing: {package}", result.stderr)

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
