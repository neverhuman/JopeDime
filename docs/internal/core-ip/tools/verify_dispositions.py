#!/usr/bin/env python3
"""Validate scoped finding dispositions against hash-bound evidence (W12-T02).

The v1 register stays an immutable open inventory. Dispositions live in a separate
append-only record whose closures are accepted only when stage-appropriate evidence
exists, is byte-identical to its recorded digest and names the same finding, stage
and source commit. A digest authenticates bytes, not the truth of their producer;
independent review of the evidence remains required.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from verify_findings import ROOT as FINDINGS_ROOT, verify as verify_register

SCHEMA = "jope.regression-finding-dispositions/v1"
EVIDENCE_SCHEMA = "jope.regression-finding-evidence/v1"
DISPOSITIONS = {
    "confirmed-defect", "verified-safeguard", "missing-execution-evidence",
    "historical-unknown", "research-result", "deferred-non-regression",
}
# Dispositions that record an honest open state; they can never close a finding.
NEVER_CLOSED = {"missing-execution-evidence", "historical-unknown", "deferred-non-regression"}
STAGES = ("source_implementation", "fixture_proof", "hosted_ci", "live_qualification",
          "measurement_complete", "promotion")
# Evidence kinds each stage accepts. Synthetic fixtures never back live, measurement
# or promotion stages, whatever their bytes or names say.
STAGE_KINDS = {
    "source_implementation": {"source-trace"},
    "fixture_proof": {"synthetic-fixture", "real-corpus-fixture"},
    "hosted_ci": {"hosted-ci"},
    "live_qualification": {"live-host"},
    "measurement_complete": {"measurement"},
    "promotion": {"decision"},
}
# Packages whose acceptance requires live operation on managed hosts or real storage.
LIVE_PACKAGES = {"W7", "W8", "W10", "W11"}
TEST_ROLES = {"pre-fix-failure", "negative", "positive", "regression"}
DEFERRABLE = {"binary", "multiclass", "time-series"}
SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_path(root, finding, reference):
    """Resolve a finding-local evidence file without traversal or symlinks."""
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
            f"{finding}: evidence reference fields")
    relative = PurePosixPath(reference["path"])
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"{finding}: evidence path traversal")
    require(relative.parts[:2] == ("evidence", finding) and len(relative.parts) >= 3,
            f"{finding}: evidence outside evidence/{finding}/")
    require(DIGEST.fullmatch(reference["sha256"] or ""), f"{finding}: evidence digest format")
    path = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), f"{finding}: evidence symlink")
    require(path.is_file(), f"{finding}: evidence missing {relative}")
    require(sha256_file(path) == reference["sha256"], f"{finding}: evidence digest mismatch {relative}")
    return path


def load_evidence(root, finding, stage, commit, reference):
    path = evidence_path(root, finding, reference)
    body = json.loads(path.read_text())
    require(body.get("schema") == EVIDENCE_SCHEMA, f"{finding}: evidence schema")
    require(body.get("finding") == finding, f"{finding}: evidence names another finding")
    require(body.get("stage") == stage, f"{finding}: evidence names another stage")
    require(body.get("source_commit") == commit, f"{finding}: evidence names another commit")
    kind = body.get("evidence_kind")
    require(kind in STAGE_KINDS[stage], f"{finding}: {kind!r} cannot back {stage}")
    return body


def check_tests(finding, stage, attestation, evidence):
    """Named tests must appear with the outcome their role requires."""
    tests = attestation.get("tests", [])
    require(isinstance(tests, list), f"{finding}: tests list")
    outcomes = {}
    for body in evidence:
        for name in body.get("passed_tests", []):
            outcomes.setdefault(name, set()).add("passed")
        for name in body.get("failed_tests", []):
            outcomes.setdefault(name, set()).add("failed")
    roles = Counter()
    for test in tests:
        require(isinstance(test, dict) and set(test) >= {"target", "name", "role"},
                f"{finding}: test fields")
        require(test["role"] in TEST_ROLES, f"{finding}: test role")
        qualified = f"{test['target']}::{test['name']}"
        roles[test["role"]] += 1
        if test["role"] == "pre-fix-failure":
            pre = test.get("pre_fix_commit")
            require(SHA.fullmatch(pre or "") and pre != attestation["source_commit"],
                    f"{finding}: pre-fix failure needs a distinct pre-fix commit")
            require(any(b.get("pre_fix_commit") == pre and qualified in b.get("failed_tests", [])
                        for b in evidence),
                    f"{finding}: pre-fix failure not observed at {pre}")
        else:
            require("passed" in outcomes.get(qualified, set()) and
                    "failed" not in outcomes.get(qualified, set()),
                    f"{finding}: {qualified} not observed passing at the attested commit")
    return roles


def validate_record(root, record, register_entry, package_live):
    finding = record["finding"]
    disposition = record.get("disposition")
    require(disposition in DISPOSITIONS, f"{finding}: disposition")
    require(isinstance(record.get("scope"), str) and record["scope"].strip(),
            f"{finding}: scope")
    decided = record.get("decided_by", {})
    require(isinstance(decided, dict) and decided.get("role") and decided.get("identity"),
            f"{finding}: decided_by")
    closed = record.get("closed")
    require(isinstance(closed, bool), f"{finding}: closed flag")
    if disposition == "deferred-non-regression":
        subcases = record.get("deferred_subcases")
        require(isinstance(subcases, list) and subcases and set(subcases) <= DEFERRABLE,
                f"{finding}: deferral names only binary/multiclass/time-series subcases")
        require("deferred" in register_entry["scope"], f"{finding}: entry has no deferred scope")
    if disposition in {"historical-unknown", "missing-execution-evidence"}:
        require(bool(record.get("reason")), f"{finding}: open disposition needs its reason")

    stages = {}
    for attestation in record.get("attestations", []):
        stage = attestation.get("stage")
        require(stage in STAGES, f"{finding}: stage {stage!r}")
        require(stage not in stages, f"{finding}: duplicate {stage} attestation")
        commit = attestation.get("source_commit")
        require(SHA.fullmatch(commit or ""), f"{finding}: {stage} source commit")
        references = attestation.get("evidence", [])
        require(isinstance(references, list) and references,
                f"{finding}: {stage} has no evidence")
        evidence = [load_evidence(root, finding, stage, commit, r) for r in references]
        if stage == "hosted_ci":
            require(all(isinstance(b.get("run_id"), int) and isinstance(b.get("attempt"), int)
                        and b.get("head_sha") == commit and b.get("conclusion") == "success"
                        for b in evidence), f"{finding}: hosted CI evidence binding")
        if stage == "promotion":
            require(all(b.get("decision_owner_role") == "scientific-decision-owner"
                        for b in evidence), f"{finding}: promotion owner")
        if stage == "source_implementation":
            require(all(b.get("paths") for b in evidence), f"{finding}: source trace paths")
        roles = check_tests(finding, stage, attestation, evidence)
        stages[stage] = (attestation, roles)

    if "promotion" in stages:
        require("live_qualification" in stages, f"{finding}: promotion without live qualification")
    if "live_qualification" in stages:
        require("hosted_ci" in stages, f"{finding}: live qualification without hosted CI")
    if not closed:
        return disposition, False
    require(disposition not in NEVER_CLOSED, f"{finding}: {disposition} cannot close")
    if disposition == "confirmed-defect":
        require({"source_implementation", "fixture_proof", "hosted_ci"} <= set(stages),
                f"{finding}: defect closure needs source, fixture and hosted CI")
        roles = stages["fixture_proof"][1]
        require(roles["pre-fix-failure"] and roles["negative"] and roles["positive"],
                f"{finding}: defect closure needs pre-fix failure, negative and positive tests")
    elif disposition == "verified-safeguard":
        require({"source_implementation", "fixture_proof", "hosted_ci"} <= set(stages),
                f"{finding}: safeguard closure needs trace, regression test and hosted CI")
        require(stages["fixture_proof"][1]["regression"],
                f"{finding}: safeguard closure needs regression protection")
    elif disposition == "research-result":
        require("measurement_complete" in stages, f"{finding}: research needs complete measurement")
    if package_live:
        require("live_qualification" in stages,
                f"{finding}: {register_entry['work_package']} requires live qualification")
    return disposition, True


def verify(root=FINDINGS_ROOT, require_git=False, repository=None):
    verify_register(root)
    register_bytes = (root / "register.json").read_bytes()
    register = json.loads(register_bytes)
    entries = {entry["id"]: entry for entry in register["entries"]}
    document = json.loads((root / "dispositions.json").read_text())
    require(document.get("schema") == SCHEMA, "dispositions schema")
    require(document.get("register_sha256") == hashlib.sha256(register_bytes).hexdigest(),
            "dispositions bind another register")
    records = document.get("records")
    require(isinstance(records, list), "records list")
    last_sequence = {}
    effective = {}
    for record in records:
        finding = record.get("finding")
        require(finding in entries, f"unknown finding {finding!r}")
        sequence = record.get("sequence")
        require(isinstance(sequence, int) and sequence > last_sequence.get(finding, 0),
                f"{finding}: sequence must increase (append-only history)")
        last_sequence[finding] = sequence
        entry = entries[finding]
        effective[finding] = validate_record(root, record, entry,
                                             entry["work_package"] in LIVE_PACKAGES)
        if require_git:
            for attestation in record.get("attestations", []):
                commits = [attestation["source_commit"]] + [
                    t.get("pre_fix_commit") for t in attestation.get("tests", [])
                    if t.get("pre_fix_commit")]
                for commit in commits:
                    subprocess.run(["git", "-C", str(repository or root), "cat-file", "-e",
                                    f"{commit}^{{commit}}"], check=True, capture_output=True)
    summary = Counter(disposition for disposition, _ in effective.values())
    return {
        "entries": len(entries),
        "with_disposition": len(effective),
        "closed": sum(closed for _, closed in effective.values()),
        "open": len(entries) - sum(closed for _, closed in effective.values()),
        "dispositions": dict(sorted(summary.items())),
        "scope": "evidence binding only; reviewed evidence, not this validator, carries truth",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=FINDINGS_ROOT)
    parser.add_argument("--require-git", action="store_true",
                        help="also require every cited commit in the local repository")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.require_git), sort_keys=True))
