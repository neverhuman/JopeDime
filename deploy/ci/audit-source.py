#!/usr/bin/env python3
"""Export and bind every tracked product path, without foreign worktrees or untracked files."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


def snapshot(repo, output):
    repo = repo.resolve(strict=True)
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], env=env)
    head = git("rev-parse", "HEAD").decode().strip()
    dirty = git("diff", "--name-only", "-z", "HEAD", "--").split(b"\0")
    unexpected = [p for p in dirty if p and p != b"AGENT_CHAT.md"]
    if unexpected:
        raise ValueError("commit tracked product changes before audit export: " + repr(unexpected))
    entries = []
    for row in git("ls-tree", "-r", "-z", head).split(b"\0"):
        if not row:
            continue
        meta, name = row.split(b"\t", 1)
        mode, kind, blob = meta.decode().split()
        name = name.decode("utf-8")
        if kind != "blob" or mode not in ("100644", "100755"):
            # A source symlink could make the scanner read outside immutable input.
            raise ValueError("audit export requires reviewed regular-file source: " + name)
        entries.append((name, mode, blob))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "source.tar"
    with archive.open("xb") as stream:
        subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", head], stdout=stream, check=True, env=env)
    source = output / "source"
    source.mkdir()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts
            if member.name.startswith("/") or ".." in parts or not (member.isfile() or member.isdir()):
                raise ValueError("unexpected archive entry: " + member.name)
        tar.extractall(source, filter="data")
    expected = {row[0] for row in entries}
    actual = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
    if expected != actual:
        raise ValueError("git archive omitted or introduced product paths")
    files = []
    for name, mode, blob in entries:
        path = source / name
        data = path.read_bytes()
        identity = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if identity != blob:
            raise ValueError("archive bytes differ from git blob: " + name)
        files.append({"path": name, "git_blob": blob, "mode": mode, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        path.chmod(0o555 if mode == "100755" else 0o444)
    if git("rev-parse", "HEAD").decode().strip() != head:
        raise ValueError("source HEAD changed while snapshot was being bound")
    with archive.open("rb") as archive_stream:
        archive_sha256 = hashlib.file_digest(archive_stream, "sha256").hexdigest()
    manifest = {"schema": "jope.audit-product-input/v1", "source_sha": os.environ.get("JOPE_SOURCE_SHA", head), "checkout_sha": head,
                "tree_sha": git("rev-parse", head + "^{tree}").decode().strip(),
                "archive_sha256": archive_sha256,
                "files": files, "file_count": len(files), "input_bytes": sum(row["bytes"] for row in files),
                "tracked_dirty_before": [p.decode() for p in dirty if p],
                "git_context": "No .git reference: auditor git fields may be null. This complete per-file Git-blob/SHA256 manifest binds immutable input to the captured checkout/source. No shared index writes or foreign untracked reads.",
                "scope": "all tracked product paths, full default standard policy; no product exclusions or policy/floor changes"}
    (output / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in sorted((p for p in source.rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o555)
    source.chmod(0o555)
    archive.chmod(0o444)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = snapshot(args.repo, args.output)
    print(json.dumps({k: result[k] for k in ("source_sha", "checkout_sha", "file_count", "input_bytes", "archive_sha256")}))
