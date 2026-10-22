#!/usr/bin/env python3
"""Validate complete audit evidence without turning a failed policy gate green."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

POLICY_FINGERPRINT = "sha256:131f419494eae9e811ca69b52a2f78a621115fa30765d5daae2519b21f828111"
REPORTS = ("repo-score.json", "repo-score.md", "timings.json", "summary.md", "repair-queue.jsonl")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)


def regular(path):
    require(not path.is_symlink() and path.is_file(), f"missing/nonregular evidence: {path.name}")
    return path


def validate(output, audit_exit, expected_manifest):
    require(0 <= audit_exit <= 255, "invalid auditor exit")
    require(re.fullmatch(r"[0-9a-f]{64}", expected_manifest) is not None, "invalid manifest pin")
    for name in REPORTS:
        regular(output / name)
    require((output / "repo-score.md").stat().st_size > 0, "empty Markdown report")
    require((output / "summary.md").stat().st_size > 0, "empty step summary")
    require(isinstance(decode((output / "timings.json").read_text()), dict), "invalid timings")
    for line in (output / "repair-queue.jsonl").read_text().splitlines():
        require(isinstance(decode(line), dict), "invalid repair queue entry")

    report = decode((output / "repo-score.json").read_text())
    for field, value in (
        ("auditor_version", "1.7.5"), ("schema_version", "1.9.0"),
        ("standard_version", "0.9.0"), ("paper_edition", "2026.05-ed8"),
        ("policy_fingerprint", POLICY_FINGERPRINT),
        ("scope", {"mode": "full", "paths": []}),
    ):
        require(report[field] == value, f"unexpected {field}")
    policy = report["policy"]
    require(policy["minimum_score"] == 85 and type(policy["minimum_score"]) is int,
            "audit floor must remain 85")
    require(policy["mode"] == "standard" and set(policy["fail_on"]) == {"critical", "high"},
            "audit policy changed")
    decision = report["decision"]
    require(type(decision["minimum_score"]) is int and decision["minimum_score"] == 85,
            "decision floor differs from policy")
    for field in ("score", "raw_score"):
        require(type(report[field]) is int and 0 <= report[field] <= 100, f"invalid {field}")
    require(report["score"] <= report["raw_score"], "score exceeds raw score")
    findings = report["findings"]
    require(isinstance(findings, list), "invalid findings")
    require(all(item.get("severity") in ("critical", "high", "medium", "low", "info")
                for item in findings), "unknown finding severity")
    hard = sum(item.get("hardness") == "hard" for item in findings)
    soft = sum(item.get("hardness") == "soft" for item in findings)
    require(hard + soft == len(findings), "unknown finding hardness")
    require(type(decision["hard_findings"]) is int and decision["hard_findings"] == hard,
            "hard finding count mismatch")
    require(type(decision["soft_findings"]) is int and decision["soft_findings"] == soft,
            "soft finding count mismatch")
    require(type(decision["passed"]) is bool and decision["status"] in ("pass", "fail"),
            "invalid decision")
    passed = decision["passed"]
    require(passed == (decision["status"] == "pass"), "decision fields disagree")
    require(passed == (audit_exit == 0), "auditor exit and policy decision disagree")
    if passed:
        require(report["score"] >= 85 and hard == 0, "passing report violates policy")
        require(not any(item["severity"] in policy["fail_on"] for item in findings),
                "passing report contains a blocking severity")

    manifest_path = regular(output / "input/SOURCE_MANIFEST.json")
    require(digest(manifest_path) == expected_manifest, "input manifest changed during audit")
    manifest = decode(manifest_path.read_text())
    require(manifest["schema"] == "jope.audit-product-input/v1", "unknown input manifest")
    for field in ("source_sha", "checkout_sha", "tree_sha"):
        require(re.fullmatch(r"[0-9a-f]{40}", manifest[field]) is not None, f"invalid {field}")
    require(manifest["source_sha"] == manifest["checkout_sha"],
            "logical audit source must equal the captured checkout")
    archive = regular(output / "input/source.tar")
    require(digest(archive) == manifest["archive_sha256"], "input archive changed during audit")
    source = output / "input/source"
    require(source.is_dir() and not source.is_symlink(), "invalid immutable source directory")
    paths = list(source.rglob("*"))
    require(not any(path.is_symlink() for path in paths), "source contains a symlink")
    require(all(path.is_file() or path.is_dir() for path in paths),
            "source contains an unsupported entry")
    require((source.stat().st_mode & 0o777) == 0o555, "source directory mode changed")
    require(all((path.stat().st_mode & 0o777) == 0o555 for path in paths if path.is_dir()),
            "source directory mode changed")
    files = manifest["files"]
    require(isinstance(files, list), "invalid input file inventory")
    require(type(manifest["file_count"]) is int and manifest["file_count"] == len(files) > 0,
            "input file count mismatch")
    expected, total = set(), 0
    for item in files:
        name = item["path"]
        parts = PurePosixPath(name).parts
        require(name and not name.startswith("/") and ".." not in parts
                and str(PurePosixPath(name)) == name and name not in expected,
                "unsafe or duplicate input path")
        expected.add(name)
        path = regular(source / name)
        require(item["mode"] in ("100644", "100755"), "invalid source mode")
        mode = 0o555 if item["mode"] == "100755" else 0o444
        require((path.stat().st_mode & 0o777) == mode, f"input mode changed: {name}")
        data = path.read_bytes()
        require(type(item["bytes"]) is int and len(data) == item["bytes"], f"input size changed: {name}")
        require(hashlib.sha256(data).hexdigest() == item["sha256"], f"input bytes changed: {name}")
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        require(blob == item["git_blob"], f"input blob changed: {name}")
        total += len(data)
    require({str(path.relative_to(source)) for path in paths if path.is_file()} == expected,
            "input paths changed during audit")
    require(type(manifest["input_bytes"]) is int and total == manifest["input_bytes"], "input byte total mismatch")
    return {
        "schema": "jope.audit-verification/v1", "artifact_valid": True, "audit_passed": passed,
        "audit_exit": audit_exit, "score": report["score"], "minimum_score": 85,
        "hard_findings": hard, "soft_findings": soft,
        "source_sha": manifest["source_sha"], "checkout_sha": manifest["checkout_sha"],
        "tree_sha": manifest["tree_sha"], "file_count": len(files),
        "manifest_sha256": expected_manifest, "input_reverified_after_audit": True,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "reports": {name: digest(output / name) for name in REPORTS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit-exit", type=int, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        record = validate(args.output, args.audit_exit, args.expected_manifest_sha256)
        status = 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        record = {"schema": "jope.audit-verification/v1", "artifact_valid": False,
                  "audit_passed": False, "audit_exit": args.audit_exit, "error": str(error)}
        status = 2
    try:
        with (args.output / "audit-verification.json").open("x") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
    except OSError as error:
        print(f"cannot retain audit verification: {error}", file=sys.stderr)
        status = 2
    print(json.dumps(record))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
