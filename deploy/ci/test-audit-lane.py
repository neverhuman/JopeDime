#!/usr/bin/env python3
"""Exercise audit failures and source/report integrity using an isolated fake auditor."""
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FAKE = r'''
#!/usr/bin/env python3
import hashlib, json, os, pathlib, sys, tomllib
mode = os.environ["AUDIT_TEST_MODE"]
if sys.argv[1:] == ["--version"]:
    print("jankurai 1.7.4" if mode == "version" else "jankurai 1.7.5")
    sys.exit(0)
args = sys.argv[1:]
assert args[0] == "audit"
assert all(v in args for v in ["--full", "--read-only", "--no-badge", "--no-score-history"])
assert args[args.index("--mode") + 1] == "standard"
source = pathlib.Path(args[1])
if mode == "missing":
    sys.exit(0)
failed = mode in ("failed", "false-green", "failed-above-floor")
low = mode in ("failed", "false-green")
floor = 0 if mode == "floor" else 85
report = {
    "auditor_version": "1.7.5", "schema_version": "1.9.0",
    "standard_version": "0.9.0", "paper_edition": "2026.05-ed8",
    "scope": {"mode": "full", "paths": []},
    "policy_fingerprint": "sha256:131f419494eae9e811ca69b52a2f78a621115fa30765d5daae2519b21f828111",
    "policy": {"minimum_score": floor, "mode": "standard", "fail_on": ["critical", "high"], "advisory_on": ["low", "medium"]},
    "score": 52 if low else 94, "raw_score": 52 if low else 94,
    "decision": {"status": "fail" if failed else "pass", "passed": not failed,
                 "minimum_score": floor, "hard_findings": int(failed), "soft_findings": 0},
    "findings": [{"severity": "high", "hardness": "hard"}] if failed else [],
}
policy_file = source / "agent/audit-policy.toml"
if policy_file.exists():
    policy_bytes = policy_file.read_bytes()
    configured = tomllib.loads(policy_bytes.decode())
    advisory = sorted(set(configured.get("advisory_on", ["medium", "low"])) - {"critical", "high"})
    report["policy"]["advisory_on"] = advisory
    binding = {"format": "jankurai-effective-audit-policy-v1",
               "source_sha256": "sha256:" + hashlib.sha256(policy_bytes).hexdigest(),
               "minimum_score": floor, "fail_on": ["critical", "high"], "advisory_on": advisory,
               "auditor_version": "1.7.5", "schema_version": "1.9.0", "standard_version": "0.9.0"}
    report["policy_fingerprint"] = "sha256:" + hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if mode == "advisory":
    report["policy"]["advisory_on"] = ["info"]
if mode == "severity":
    report["findings"] = [{"severity": "critical", "hardness": "soft"}]
    report["decision"]["soft_findings"] = 1
if mode == "raw":
    report["raw_score"] = 1
if mode == "policy":
    report["policy_fingerprint"] = "sha256:" + "0" * 64
if mode == "scope":
    report["scope"] = {"mode": "changed", "paths": ["one-file"]}
if mode == "counts":
    report["decision"]["hard_findings"] = 1
for flag, value in [
    ("--json", json.dumps(report)), ("--md", "Synthetic auditor report\n"),
    ("--timings-json", "{}"), ("--github-step-summary", "Synthetic summary\n"),
    ("--repair-queue-jsonl", '{"rule_id":"fixture"}\n' if failed else ""),
]:
    pathlib.Path(args[args.index(flag) + 1]).write_text(value)
output = pathlib.Path(args[args.index("--json") + 1]).parent
if mode == "duplicate":
    path = output / "repo-score.json"
    path.write_text('{"score":0,' + path.read_text()[1:])
if mode == "nonfinite":
    (output / "timings.json").write_text('{"elapsed":NaN}')
if mode == "malformed":
    (output / "repo-score.json").write_text("{")
if mode == "source":
    target = source / "README.md"
    target.chmod(0o644)
    target.write_text("modified after capture")
    target.chmod(0o444)
if mode == "mode":
    (source / "README.md").chmod(0o644)
if mode == "extra":
    source.chmod(0o755)
    (source / "untracked.txt").write_text("not admitted")
    source.chmod(0o555)
if mode == "fifo":
    source.chmod(0o755)
    os.mkfifo(source / "unexpected-fifo")
    source.chmod(0o555)
if mode == "manifest":
    path = output / "input/SOURCE_MANIFEST.json"
    payload = json.loads(path.read_text())
    payload["tree_sha"] = "0" * 40
    path.write_text(json.dumps(payload))
if mode == "archive":
    path = output / "input/source.tar"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"changed")
    path.chmod(0o444)
if mode == "link":
    path = output / "repo-score.md"
    path.unlink()
    path.symlink_to(output / "summary.md")
sys.exit(1 if mode in ("failed", "failed-above-floor") else 0)
'''
# Stand-in for ops/ci/verify-pyexec-exceptions.sh, which re-audits pinned product files the
# fixture does not carry. It records its arguments and fails on request.
EXCEPTIONS = r'''
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "$AUDIT_TEST_EXCEPTION_ARGS"
[[ "$AUDIT_TEST_EXCEPTIONS" == pass ]] || { echo 'fixture exception drift' >&2; exit 3; }
printf '{"fixture": true}\n' > "$3"
'''


class AuditLaneTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "target" / "audit-lane-controls"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        names = ["ops/ci/jankurai.sh", "deploy/ci/audit-source.py"]
        if (ROOT / "deploy/ci/verify-audit-report.py").exists():
            names.append("deploy/ci/verify-audit-report.py")
        for name in names:
            dest = self.repo / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / name).read_bytes())
        (self.repo / "README.md").write_text("Audit fixture, not a product checkout.\n")
        (self.repo / "ops/ci/verify-pyexec-exceptions.sh").write_text(EXCEPTIONS.lstrip())
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "--", "README.md",
                        "ops/ci/verify-pyexec-exceptions.sh", *names], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Fixture",
                        "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
        self.auditor = self.base / "auditor"
        self.auditor.write_text(FAKE.lstrip())
        self.auditor.chmod(0o755)

    def run_audit(self, mode, report_only=False, output=None, exceptions="pass", source_sha=None):
        self.output = output or self.base / ("report-" + mode + ("-report-only" if report_only else ""))
        env = dict(os.environ, JANKURAI_BIN=str(self.auditor), AUDIT_TEST_MODE=mode,
                   AUDIT_TEST_EXCEPTIONS=exceptions,
                   AUDIT_TEST_EXCEPTION_ARGS=str(self.base / "exception-args.txt"))
        for key in ("JOPE_SOURCE_SHA", "GITHUB_STEP_SUMMARY", "JOPE_AUDIT_REPORT_ONLY"):
            env.pop(key, None)
        if source_sha is not None:
            env["JOPE_SOURCE_SHA"] = source_sha
        if report_only:
            env["JOPE_AUDIT_REPORT_ONLY"] = "1"
        return subprocess.run(["bash", "ops/ci/jankurai.sh", str(self.output)],
                              cwd=self.repo, env=env, capture_output=True, text=True)

    def test_captured_repository_policy_is_bound_in_the_effective_fingerprint(self):
        policy = self.repo / "agent/audit-policy.toml"
        policy.parent.mkdir()
        policy.write_text('minimum_score = 85\nfail_on = ["critical", "high"]\nadvisory_on = ["medium", "low", "info"]\n[scan]\nextra_excluded_paths = ["target"]\n')
        subprocess.run(["git", "add", "agent/audit-policy.toml"], cwd=self.repo, check=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                        "commit", "-qm", "captured repository policy"], cwd=self.repo, check=True)
        result = self.run_audit("failed", report_only=True)
        self.assertEqual(0, result.returncode, result.stderr)
        verification = json.loads((self.output / "audit-verification.json").read_text())
        report = json.loads((self.output / "repo-score.json").read_text())
        self.assertEqual(report["policy_fingerprint"], verification["policy_fingerprint"])
        self.assertFalse(verification["audit_passed"])
        self.assertNotEqual(0, self.run_audit("policy", report_only=True).returncode)
        self.assertNotEqual(0, self.run_audit("advisory", report_only=True).returncode)

    def test_synthetic_merge_identity_requires_the_reviewed_tree(self):
        reviewed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                        "commit", "--allow-empty", "-qm", "synthetic identical-tree checkout"], cwd=self.repo, check=True)
        result = self.run_audit("pass", source_sha=reviewed)
        self.assertEqual(0, result.returncode, result.stderr)
        manifest = json.loads((self.output / "input/SOURCE_MANIFEST.json").read_text())
        self.assertNotEqual(manifest["source_sha"], manifest["checkout_sha"])
        self.assertEqual(manifest["source_tree_sha"], manifest["tree_sha"])
        spec = importlib.util.spec_from_file_location("merge_audit_verifier", ROOT / "deploy/ci/verify-audit-report.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        manifest_path = self.output / "input/SOURCE_MANIFEST.json"
        invalid = dict(manifest)
        invalid.pop("source_tree_sha")
        manifest_path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, "attestation"):
            verifier.validate(self.output, 0, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        invalid["source_tree_sha"] = "0" * 40
        manifest_path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, "source tree differs"):
            verifier.validate(self.output, 0, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        (self.repo / "README.md").write_text("different source tree\n")
        subprocess.run(["git", "add", "README.md"], cwd=self.repo, check=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                        "commit", "-qm", "unreviewed tree"], cwd=self.repo, check=True)
        self.assertNotEqual(0, self.run_audit("pass", output=self.base / "wrong-tree", source_sha=reviewed).returncode)

    def test_honest_failed_gate_retains_failure_and_reports(self):
        result = self.run_audit("failed")
        self.assertEqual(1, result.returncode, result.stderr)
        report = json.loads((self.output / "repo-score.json").read_text())
        self.assertEqual(52, report["score"])
        self.assertFalse(report["decision"]["passed"])

    def test_report_only_passes_verified_below_floor_audit(self):
        result = self.run_audit("failed", report_only=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("::warning title=Audit below floor (report-only)::score 52/85", result.stdout)
        report = json.loads((self.output / "repo-score.json").read_text())
        self.assertFalse(report["decision"]["passed"])
        receipt = json.loads((self.output / "audit-verification.json").read_text())
        self.assertTrue(receipt["artifact_valid"])
        self.assertFalse(receipt["audit_passed"])
        marker = json.loads((self.output / "report-only.json").read_text())
        self.assertEqual({"audit_exit": 1, "audit_passed": False, "score": 52, "minimum_score": 85},
                         {key: marker[key] for key in ("audit_exit", "audit_passed", "score", "minimum_score")})

    def test_report_only_still_refuses_invalid_evidence(self):
        for mode in ("missing", "false-green", "floor", "version", "policy", "scope", "counts",
                     "malformed", "source", "extra", "manifest", "archive", "link", "mode",
                     "severity", "raw", "duplicate", "nonfinite", "fifo", "failed-above-floor"):
            with self.subTest(mode=mode):
                self.assertNotEqual(0, self.run_audit(mode, report_only=True).returncode)
                self.assertFalse((self.output / "report-only.json").exists())

    def test_report_only_leaves_passing_audit_unmarked(self):
        self.assertEqual(0, self.run_audit("pass", report_only=True).returncode)
        self.assertFalse((self.output / "report-only.json").exists())

    def test_retained_evidence_replays_then_post_upload_unlock_makes_it_deletable(self):
        # Retained evidence keeps its read-only captured source so the verifier replays on it.
        # Self-hosted workspaces persist, so the workflow's post-upload step must unlock it, or
        # the next job's checkout cannot delete it.
        spec = importlib.util.spec_from_file_location("verify_audit_report", ROOT / "deploy/ci/verify-audit-report.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        for workflow_name, job in (("ci.yml", "jankurai-audit"), ("full-main.yml", "audit")):
            workflow = (ROOT / ".github/workflows" / workflow_name).read_text()
            block = workflow.split("\n  " + job + ":\n", 1)[1]
            unlock = block.split("      - name: Leave the persistent workspace deletable for the next checkout\n", 1)[1]
            self.assertTrue(unlock.startswith("        if: always()\n        run: "))
            command = unlock.split("        run: ", 1)[1].splitlines()[0]
            self.assertEqual("if [ -d target/full-audit ]; then chmod -R u+w target/full-audit; fi", command)
        workflow = (ROOT / ".github/workflows/full-main.yml").read_text()
        block = workflow.split("\n  audit:\n", 1)[1]
        unlock = block.split("      - name: Leave the persistent workspace deletable for the next checkout\n", 1)[1]
        self.assertTrue(unlock.startswith("        if: always()\n        run: "))
        command = unlock.split("        run: ", 1)[1].splitlines()[0]
        self.assertEqual("if [ -d target/full-audit ]; then chmod -R u+w target/full-audit; fi", command)
        for mode, audit_exit in (("pass", 0), ("failed", 1)):
            with self.subTest(mode=mode):
                workspace = self.base / ("workspace-" + mode)
                self.run_audit(mode, output=workspace / "target/full-audit/1/1")
                manifest = hashlib.sha256((self.output / "input/SOURCE_MANIFEST.json").read_bytes()).hexdigest()
                verifier.validate(self.output, audit_exit, manifest)
                self.assertTrue([p for p in self.output.rglob("*") if p.is_dir() and not os.stat(p).st_mode & 0o200])
                subprocess.run(["bash", "-c", command], cwd=workspace, check=True)
                stuck = [p for p in [self.output, *self.output.rglob("*")]
                         if not p.is_symlink() and not os.stat(p).st_mode & 0o200]
                self.assertEqual([], stuck)
                shutil.rmtree(workspace / "target")
                self.assertFalse((workspace / "target").exists())

    def test_valid_pass_is_accepted(self):
        result = self.run_audit("pass")
        self.assertEqual(0, result.returncode, result.stderr)
        receipt = json.loads((self.output / "audit-verification.json").read_text())
        self.assertTrue(receipt["artifact_valid"])
        self.assertTrue(receipt["audit_passed"])

    def test_exception_verifier_receives_captured_source_and_manifest(self):
        self.assertEqual(0, self.run_audit("pass").returncode)
        args = (self.base / "exception-args.txt").read_text().splitlines()
        self.assertEqual(5, len(args))
        self.assertEqual(str(self.auditor), args[1])
        expected = [self.output / "input/source", self.output / "pyexec-exceptions.json",
                    self.repo, self.output / "input/SOURCE_MANIFEST.json"]
        self.assertEqual([p.resolve() for p in expected],
                         [Path(arg).resolve() for arg in (args[0], args[2], args[3], args[4])])

    def test_exception_drift_fails_a_passing_audit(self):
        result = self.run_audit("pass", exceptions="drift")
        self.assertEqual(3, result.returncode, result.stderr)
        self.assertIn("fixture exception drift", result.stderr)

    def test_exception_drift_fails_report_only_below_floor(self):
        self.assertEqual(3, self.run_audit("failed", report_only=True, exceptions="drift").returncode)
        self.assertFalse((self.output / "report-only.json").exists())

    def test_zero_exit_without_reports_is_not_a_pass(self):
        self.assertNotEqual(0, self.run_audit("missing").returncode)

    def test_zero_exit_cannot_hide_failed_decision(self):
        self.assertNotEqual(0, self.run_audit("false-green").returncode)

    def test_lowered_policy_is_refused(self):
        self.assertNotEqual(0, self.run_audit("floor").returncode)

    def test_unpinned_auditor_is_refused(self):
        self.assertNotEqual(0, self.run_audit("version").returncode)

    def test_changed_protocol_or_evidence_cannot_pass(self):
        for mode in ("policy", "scope", "counts", "malformed", "source", "extra",
                     "manifest", "archive", "link", "mode", "severity", "raw", "duplicate", "nonfinite", "fifo"):
            with self.subTest(mode=mode):
                self.assertNotEqual(0, self.run_audit(mode).returncode)

    def test_existing_evidence_is_never_reused(self):
        self.assertEqual(0, self.run_audit("pass").returncode)
        before = (self.output / "repo-score.json").read_bytes()
        self.assertNotEqual(0, self.run_audit("pass").returncode)
        self.assertEqual(before, (self.output / "repo-score.json").read_bytes())

    def test_main_audit_is_scheduled_inside_existing_trust_boundary(self):
        workflow = (ROOT / ".github/workflows/full-main.yml").read_text()
        self.assertIn("\n  audit:\n", workflow)
        block = workflow.split("\n  audit:\n", 1)[1]
        self.assertIn("if: github.ref == 'refs/heads/main'", block)
        self.assertIn("runs-on: [self-hosted, linux, x64, jope-full-cpu]", block)
        self.assertIn("bash ops/ci/jankurai.sh", block)
        self.assertIn("if: always()", block)
        self.assertIn("uses: ./.github/actions/upload-ci-evidence", block)
        self.assertNotIn("continue-on-error", block)
        self.assertNotIn("|| true", block)
        self.assertNotIn("JANKURAI_HOSTED_RUNNER", block)
        self.assertEqual(1, block.count('JOPE_AUDIT_REPORT_ONLY: "1"'))
        self.assertLess(block.index("uses: ./.github/actions/upload-ci-evidence"),
                        block.index("chmod -R u+w target/full-audit"))
        spec = importlib.util.spec_from_file_location("audit_job_policy", ROOT / "deploy/runner/job-policy.py")
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        env = dict(GITHUB_REPOSITORY="neverhuman/JopeDime", GITHUB_REPOSITORY_ID="1376159324",
                   GITHUB_WORKFLOW_REF="neverhuman/JopeDime/.github/workflows/full-main.yml@refs/heads/main",
                   GITHUB_REF="refs/heads/main", GITHUB_EVENT_NAME="push", GITHUB_SHA="a" * 40)
        policy.authorize({"domain": "main"}, env)
        with self.assertRaises(ValueError):
            policy.authorize({"domain": "pr"}, env)
        with self.assertRaises(ValueError):
            policy.authorize({"domain": "main"}, dict(env, GITHUB_EVENT_NAME="pull_request"))


if __name__ == "__main__":
    unittest.main()
