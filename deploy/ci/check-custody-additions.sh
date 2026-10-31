#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
base=${1:-origin/main}

python3 - "$base" <<'PY'
import os
import subprocess
import sys

LIMIT = 1_000_000
PREFIX = "research/flywheel_selfplay/custody/"
ARCHIVE_SUFFIXES = (
    ".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst", ".tar.lz4", ".tar",
    ".tgz", ".tbz", ".tbz2", ".txz", ".tzst", ".zip", ".whl", ".egg",
    ".jar", ".7z", ".rar", ".gz", ".bz2", ".xz", ".zst", ".lz4",
    ".deb", ".rpm", ".bundle",
)


def git(root, *args):
    result = subprocess.run(
        ["git", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode:
        message = os.fsdecode(result.stderr).strip()
        raise RuntimeError(message or f"git {' '.join(args)} failed")
    return result.stdout


def added_paths(raw):
    fields = raw.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 2:
        raise ValueError("git diff returned malformed NUL-delimited name-status data")
    paths = []
    for offset in range(0, len(fields), 2):
        status = fields[offset]
        path = os.fsdecode(fields[offset + 1])
        if status == b"A" and path.startswith(PREFIX):
            paths.append(path)
    return paths


def is_archive(path, prefix):
    lower = path.lower()
    if lower.endswith(ARCHIVE_SUFFIXES):
        return True
    if (
        prefix.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))
        or prefix.startswith((b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00"))
        or prefix.startswith((b"\x28\xb5\x2f\xfd", b"7z\xbc\xaf\x27\x1c"))
        or prefix.startswith(b"Rar!\x1a\x07")
        or prefix.startswith((b"070701", b"070702", b"070707"))
        or prefix.startswith(b"!<arch>\n")
        or prefix.startswith(b"MSCF")
        or prefix.startswith((b"# v2 git bundle", b"# v3 git bundle"))
        or prefix[257:262] == b"ustar"
    ):
        return True
    return False


def policy_violations(path, size, prefix):
    if not path.startswith(PREFIX):
        return []
    reasons = []
    if size > LIMIT:
        reasons.append(f"{size} bytes exceeds the {LIMIT}-byte limit")
    if size <= LIMIT and is_archive(path, prefix):
        reasons.append("archive/package archive is not allowed in Git custody")
    return reasons


def inspect_added(root, base, head="HEAD"):
    diff_range = f"{base}...{head}"
    try:
        raw = git(
            root,
            "diff", "--name-status", "-z", "--diff-filter=A", diff_range,
            "--", PREFIX,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"cannot inspect custody additions against {base!r}; fetch the PR base ref: {error}"
        ) from error

    violations = []
    for path in added_paths(raw):
        object_id = os.fsdecode(git(root, "rev-parse", "--verify", f"{head}:{path}")).strip()
        size = int(git(root, "cat-file", "-s", object_id).strip())
        # Large blobs are rejected by size without reading or buffering their contents.
        prefix = b"" if size > LIMIT else git(root, "cat-file", "blob", object_id)[:512]
        for reason in policy_violations(path, size, prefix):
            violations.append(f"{path}: {reason}")
    return violations


def self_test():
    ordinary = f"{PREFIX}files/example/config.json"
    assert added_paths(b"A\0" + ordinary.encode() + b"\0") == [ordinary]
    assert added_paths(b"M\0" + ordinary.encode() + b"\0") == []
    assert added_paths(b"A\0docs/readme.md\0") == []
    assert policy_violations(ordinary, LIMIT, b"{}") == []
    assert "exceeds" in policy_violations(ordinary, LIMIT + 1, b"")[0]
    for name, signature in [
        ("copy.bin", b"PK\x03\x04"), ("copy.dat", b"\x1f\x8b"),
        ("copy.tar", b""), ("copy.blob", b"# v2 git bundle"),
    ]:
        prefix = signature
        if name.endswith(".tar"):
            prefix = b" " * 257 + b"ustar"
        assert policy_violations(f"{PREFIX}{name}", len(prefix), prefix)

    global git
    real_git = git

    def fake_git(_root, *args):
        command = args[0]
        if command == "diff":
            return fake_git.diff
        if command == "rev-parse":
            return b"synthetic-blob-id\n"
        if command == "cat-file" and args[1] == "-s":
            return str(fake_git.size).encode()
        if command == "cat-file" and args[1] == "blob":
            return fake_git.content
        raise AssertionError(f"unexpected git command in self-test: {args}")

    try:
        fake_git.diff = b"A\0" + ordinary.encode() + b"\0"
        fake_git.size, fake_git.content = 4, b"{}\n"
        git = fake_git
        assert inspect_added(".", "base") == []
        fake_git.size, fake_git.content = LIMIT + 1, b""
        assert "exceeds" in inspect_added(".", "base")[0]
        fake_git.size, fake_git.content = 8, b"PK\x03\x04data"
        assert "archive" in inspect_added(".", "base")[0]
        fake_git.diff = b"M\0" + ordinary.encode() + b"\0"
        assert inspect_added(".", "base") == []
    finally:
        git = real_git
    print("custody addition guard self-test: PASS")


def main():
    self_test()
    root = os.getcwd()
    base = sys.argv[1]
    failures = inspect_added(root, base)
    if failures:
        print("new flywheel custody payloads are not allowed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print(
            "Store package payloads in approved external custody and add a manifest pointer instead.",
            file=sys.stderr,
        )
        return 1
    print(f"custody additions against {base}: none over limit or archived")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"custody addition guard FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
PY
