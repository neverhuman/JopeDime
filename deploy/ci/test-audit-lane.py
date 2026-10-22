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
import hashlib, json, os, pathlib, sys
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
    "policy": {"minimum_score": floor, "mode": "standard", "fail_on": ["critical", "high"]},
    "score": 52 if low else 94, "raw_score": 52 if low else 94,
    "decision": {"status": "fail" if failed else "pass", "passed": not failed,
                 "minimum_score": floor, "hard_findings": int(failed), "soft_findings": 0},
    "findings": [{"severity": "high", "hardness": "hard"}] if failed else [],
}
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
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "--", "README.md", *names], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Fixture",
                        "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
        self.auditor = self.base / "auditor"
        self.auditor.write_text(FAKE.lstrip())
        self.auditor.chmod(0o755)

    def run_audit(self, mode, report_only=False, output=None):
        self.output = output or self.base / ("report-" + mode + ("-report-only" if report_only else ""))
        env = dict(os.environ, JANKURAI_BIN=str(self.auditor), AUDIT_TEST_MODE=mode)
        for key in ("JOPE_SOURCE_SHA", "GITHUB_STEP_SUMMARY", "JOPE_AUDIT_REPORT_ONLY"):
            env.pop(key, None)
        if report_only:
            env["JOPE_AUDIT_REPORT_ONLY"] = "1"
        return subprocess.run(["bash", "ops/ci/jankurai.sh", str(self.output)],
                              cwd=self.repo, env=env, capture_output=True, text=True)

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
