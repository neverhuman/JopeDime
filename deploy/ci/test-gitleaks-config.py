#!/usr/bin/env python3
"""Exercise the real secret scanner against synthetic credential and digest controls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def cases():
    # These values are generated test inputs, never provisioned credentials.
    digest = hashlib.sha256(b"JopeDime synthetic gitleaks config control").hexdigest()
    key_kind = "OPENSSH PRIVATE" + " KEY"
    private = f"-----BEGIN {key_kind}-----\nsynthetic-test-data\n-----END {key_kind}-----\n"
    result = [
        ("hex-api-key-64", "config.txt", f'api_key = "{digest}"\n', True),
        ("hex-api-key-40", "config.txt", f'api_key = "{digest[:40]}"\n', True),
        ("embedded-hex-api-key", "config.txt", f'api_key = "A7Q9{digest}K3R2"\n', True),
        ("sha256-metadata", "manifest.txt", f'artifact_sha256 = "{digest}"\n', False),
        ("git-object-metadata", "manifest.txt", f'git_commit = "{digest[:40]}"\n', False),
        ("digest-and-object-metadata", "manifest.txt", f'sha256 = "{digest}"\ngit_sha = "{digest[:40]}"\n', False),
        ("documented-evaluation-schema", "fixture.txt", 'evaluation_key_schema = "EvaluationKey.v2"\n', False),
        ("documented-observation-type", "fixture.txt", 'key_type = "ObservationKeyV2"\n', False),
        ("documented-archived-private-key-fixture", "python/src/jope/research/test_workspace_sync.py", private, False),
        ("private-key-outside-archived-fixture", "source.txt", private, True),
        ("credential-containing-schema-marker", "config.txt", f'api_key = "EvaluationKey.v2-{digest}"\n', True),
    ]
    # Real scanner controls for reviewed public provenance assignments. Exact
    # file + exact match context must both hold; credentials elsewhere in the
    # same file or minified line must still be found.
    provenance_contexts = [('python/src/jope/policy/tokens.py', 'ERATION31_TOKEN_SHA256 = "', '"'), ('deploy/dl-flywheel/pipelines/continuous_v6/tests/fixtures/kl_pins.json', 'MAX_TOKENS_PER_CHUNK": "', '"'), ('deploy/release/release.sh', 'PUBKEY_SHA256=', ''), ('deploy/dl-flywheel/pipelines/final_capture_v1/SOURCE_PINS.json', 'dsl_tokenizer.py": "', '"'), ('deploy/dl-flywheel/docs/continuous-runtime-pins-20261009.json', 'dsl_tokenizer.py":"', '"'), ('deploy/dl-trainonly/lane/src/dl_training_v1/tools/p2_decisions.py', 'family_keys": "', '"'), ('deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/provenance/independent/selected-config-draft-review.json', 'program_key": "', '"'), ('docs/internal/kernel2/receipts/evaluator-identity-3ec654cb/receipt.json', 'score_key":"', '"'), ('deploy/lambda/SNAPSHOT.json', 'secrets_existence_sha256": "', '"'), ('docs/internal/clover/gp-expansion-protocol.v1.json', 'tokenizer_sha256": "', '"'), ('crates/jope-campaigns/manifests/advanced-v3/task-receipts/P-1C/0001.v1.json', 'tokens": "', '"'), ('crates/jope-engine/assets/grammar/featmath-jain-v1.json', 'tokens.py": "', '"'), ('crates/jope-campaigns/manifests/advanced-v3/task-receipts/P-1A/0003.v1.json', 'tokens.rs": "', '"'), ('crates/jope-learning/tests/fixtures/support-policy-v1.json', 'trace_tokens_sha256":"', '"'), ('deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/provenance/selected-qualification-v3/actual-readback-summary.json', 'valuation_key": "', '"'), ('deploy/dl-flywheel/docs/native-dispatch-qualification-20261009.json', 'valuation_key":"', '"')]
    for index, (relative, prefix, suffix) in enumerate(provenance_contexts):
        metadata = prefix + digest + suffix
        credential = f'api_key = "{digest}"'
        result.extend([
            (f"provenance-{index}-digest", relative, metadata + "\n", False),
            (f"provenance-{index}-credential", relative, credential + "\n", True),
            (f"provenance-{index}-same-line-credential", relative, metadata + "; " + credential + "\n", True),
        ])
    relative, prefix, suffix = provenance_contexts[0]
    result.append(("provenance-context-in-unreviewed-file", "unreviewed.txt", prefix + digest + suffix + "\n", True))
    return result


def fingerprint_baseline_control(scanner, config, root, evidence):
    """Keep an inherited finding ignored while rejecting a replacement on the same line."""
    directory = evidence / "fingerprint-baseline"
    source = directory / "source"
    source.mkdir(parents=True)

    def git(*args):
        return subprocess.run(
            ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args],
            cwd=source, capture_output=True, text=True, check=True, timeout=20,
        )

    git("init", "-q", "-b", "main")
    first = hashlib.sha256(b"synthetic inherited credential fingerprint").hexdigest()
    credential = source / "config.txt"
    credential.write_text(f'api_key = "A7Q9{first}K3R2"\n')
    git("add", "config.txt")
    git("commit", "-qm", "synthetic inherited finding")

    report = directory / "report.json"
    scanned = subprocess.run(
        [scanner, "detect", "--redact=100", "--no-banner", "--config", str(config),
         "--source", str(source), "--log-opts=HEAD", "--report-format", "json",
         "--report-path", str(report)],
        cwd=source, capture_output=True, text=True, timeout=20,
    )
    findings = json.loads(report.read_text()) if report.exists() else []
    report_safe = bool(findings) and all(
        first not in json.dumps(finding) and "REDACTED" in finding.get("Secret", "")
        for finding in findings
    )
    initial_red = scanned.returncode == 1 and len(findings) == 1 and report_safe
    (source / ".gitleaksignore").write_text(findings[0]["Fingerprint"] + "\n" if findings else "")
    git("add", ".gitleaksignore")
    git("commit", "-qm", "baseline one inherited fingerprint")

    def full_scan():
        return subprocess.run(
            ["bash", str(root / "deploy/ci/secret-scan.sh"), str(source)],
            cwd=root, capture_output=True, text=True, timeout=20,
        )

    inherited_green = full_scan().returncode == 0
    second = hashlib.sha256(b"synthetic replacement credential fingerprint").hexdigest()
    credential.write_text(f'api_key = "B8R2{second}Q1M6"\n')
    git("add", "config.txt")
    git("commit", "-qm", "synthetic new finding at the same line")
    replacement_red = full_scan().returncode == 1
    return {
        "source": "synthetic-only Git history; no repository findings or live credential inputs",
        "initial_finding_red": initial_red,
        "inherited_fingerprint_green": inherited_green,
        "new_finding_same_file_line_red": replacement_red,
        "secret_report_redacted": report_safe,
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / ".gitleaks.toml")
    parser.add_argument("--scanner", default="gitleaks")
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    scanner = shutil.which(args.scanner)
    if scanner is None:
        parser.error("gitleaks is required; missing scanner is not a passing control")
    config = args.config.resolve(strict=True)
    if args.evidence_dir is None:
        base = root / "target" / "security" / "gitleaks-config-controls"
        base.mkdir(parents=True, exist_ok=True)
        evidence = Path(tempfile.mkdtemp(prefix="run-", dir=base))
    else:
        evidence = args.evidence_dir.resolve()
        evidence.mkdir(parents=True, exist_ok=False)
    version = subprocess.check_output([scanner, "version"], text=True, timeout=10).strip()
    records = []
    for label, relative, content, should_detect in cases():
        directory = evidence / label
        source = directory / "source"
        path = source / relative
        path.parent.mkdir(parents=True)
        path.write_text(content)
        report = directory / "report.json"
        # Relative file names match the repository-relative paths seen in Git mode.
        argv = [scanner, "detect", "--no-git", "--redact=100", "--no-banner", "--config", str(config), "--source", ".", "--report-format", "json", "--report-path", str(report)]
        result = subprocess.run(argv, cwd=source, capture_output=True, text=True, timeout=20)
        (directory / "stdout.txt").write_text(result.stdout)
        (directory / "stderr.txt").write_text(result.stderr)
        findings = json.loads(report.read_text()) if report.exists() else []
        expected = 1 if should_detect else 0
        passed = result.returncode == expected and bool(findings) == should_detect
        records.append({"case": label, "expected_detection": should_detect, "exit_code": result.returncode, "finding_rules": sorted({finding["RuleID"] for finding in findings}), "passed": passed})
    baseline = fingerprint_baseline_control(scanner, config, root, evidence)
    baseline_ok = all(baseline[name] for name in (
        "initial_finding_red", "inherited_fingerprint_green",
        "new_finding_same_file_line_red", "secret_report_redacted",
    ))
    receipt = {"scanner_version": version, "scanner_sha256": hashlib.sha256(Path(scanner).read_bytes()).hexdigest(), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "source": "synthetic controls only; no repository/history or real credential reads", "cases": records, "passed": sum(case["passed"] for case in records), "failed": sum(not case["passed"] for case in records) + (not baseline_ok), "fingerprint_baseline_control": baseline}
    (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    print(f"Evidence: {evidence}")
    raise SystemExit(0 if receipt["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
