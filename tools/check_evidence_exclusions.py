#!/usr/bin/env python3
"""Fail-closed verification for receipt-bound exact-path auditor exclusions."""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTER = ROOT / "agent/evidence-exclusions.toml"
POLICY = ROOT / "agent/audit-policy.toml"
LEGACY_POLICY_PATHS_SHA256 = "d215126faf0ebbfddaae4e952a73e95e217e58434e80f51e63ceedef02f14b55"
MANAGED_POLICY_PATHS_SHA256 = "04b94886c5df6cf2dc87131404c913017465fc189df2117fc121c7942f7b496a"
RECEIPT_NAME = re.compile(r"(?i)(provenance|manifest|receipt|snapshot|checkpoint|source.?pins|source.?packet|source.?counts|file_map|findings|inventory|metrics|compat_freeze|freeze|gp-config|classifications|deletions|handoff|pr3\.md|technical-review\.md|runtime_kumos|\.sha256|\.lock\.json)")


def fail(message: str) -> None:
    print(f"evidence exclusion check: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def repo_file(raw: str, label: str) -> tuple[pathlib.Path, bytes]:
    if not isinstance(raw, str) or not raw or pathlib.PurePosixPath(raw).is_absolute():
        fail(f"{label} must be a nonempty repository-relative path: {raw!r}")
    path = ROOT / raw
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(ROOT.resolve())
        data = path.read_bytes()
    except (OSError, ValueError) as exc:
        fail(f"{label} is missing or escapes the repository: {raw} ({exc})")
    if not path.is_file() or path.is_symlink():
        fail(f"{label} is not a regular non-symlink file: {raw}")
    return path, data


def digest_paths(paths: set[str]) -> str:
    body = json.dumps(sorted(paths), separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(body).hexdigest()


def main() -> int:
    try:
        register = tomllib.loads(REGISTER.read_text(encoding="utf-8"))
        policy = tomllib.loads(POLICY.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        fail(f"invalid register or audit policy: {exc}")
    entries = register.get("evidence_exclusion")
    if not isinstance(entries, list) or not entries:
        fail("register has no [[evidence_exclusion]] entries")
    paths: set[str] = set()
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            fail(f"entry {index} is not a table")
        path, reason, receipt, expected = (entry.get(k) for k in ("path", "reason", "receipt", "sha256"))
        if not all(isinstance(x, str) and x for x in (path, reason, receipt, expected)):
            fail(f"entry {index} requires path, reason, receipt, and sha256")
        if any(ch in path for ch in "*?[") or path.endswith("/"):
            fail(f"entry {index} is not an exact file path: {path}")
        if "\n" in reason or "\r" in reason:
            fail(f"entry {index} reason must be one line: {path}")
        if path in paths:
            fail(f"duplicate register path: {path}")
        paths.add(path)
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            fail(f"invalid SHA-256 for {path}")
        _, content = repo_file(path, "source")
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            fail(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
        receipt_path, receipt_bytes = repo_file(receipt, "receipt")
        if not RECEIPT_NAME.search(receipt) and "jankurai-reporot-manifest-exceptions-20261102.md" not in receipt:
            fail(f"receipt is not a recognized pin/manifest/provenance/receipt artifact: {receipt}")
        if expected.encode("ascii") not in receipt_bytes:
            fail(f"receipt {receipt} does not contain the listed SHA-256 for {path}")
    try:
        policy_paths = set(policy["scan"]["extra_excluded_paths"])
        policy_paths.update(path for block in policy.get("evidence_snapshot", []) for path in block["paths"])
    except (KeyError, TypeError) as exc:
        fail(f"audit policy path lists have invalid shape: {exc}")
    missing = sorted(paths - policy_paths)
    if missing:
        fail("register paths absent from audit policy: " + ", ".join(missing[:10]))
    if digest_paths(policy_paths - paths) != LEGACY_POLICY_PATHS_SHA256:
        fail("audit-policy contains unregistered path changes (legacy set digest mismatch)")
    try:
        policy_text = POLICY.read_text(encoding="utf-8")
        body = policy_text.split("# BEGIN EVEXCL t2320 managed exact-path entries", 1)[1].split("# END EVEXCL t2320 managed exact-path entries", 1)[0]
    except (OSError, IndexError):
        fail("managed policy block markers are missing")
    listed = set(re.findall(r'^\s+"([^"\n]+)"\s*,\s*$', body, re.MULTILINE))
    if digest_paths(listed) != MANAGED_POLICY_PATHS_SHA256 or listed - paths:
        fail("managed policy block does not exactly match registered additions")
    print(f"evidence exclusion check: PASS ({len(paths)} registered exact paths; all source and receipt hashes match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
