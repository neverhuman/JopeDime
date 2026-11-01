#!/usr/bin/env python3
"""Package fresh named-client sidecars with source, candidate and input pins.

This creates a custody manifest, not an A5 qualification manifest. It never
launches clients or reads dataset payloads.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

CLIENTS = {
    "claude_code": "claude-code",
    "gemini_cli": "gemini-cli",
    "codex_cli": "codex-cli",
}
SHA = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


class CapturePackageError(ValueError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fail_if(condition: bool, message: str) -> None:
    if condition:
        raise CapturePackageError(message)


def load_object(path: Path, label: str) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapturePackageError(f"{label} is unreadable or invalid JSON") from exc
    fail_if(not isinstance(value, dict), f"{label} must be a JSON object")
    return value, raw


def source_identity(repo: Path, claimed: str) -> None:
    fail_if(GIT_SHA.fullmatch(claimed) is None, "source SHA must be 40 lowercase hex chars")
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CapturePackageError("cannot establish repository source identity") from exc
    fail_if(actual != claimed, "claimed source does not match checkout HEAD")
    fail_if(bool(dirty), "source checkout is not clean")


def prepare(args) -> dict:
    repo = args.repo.resolve(strict=True)
    source_identity(repo, args.source_sha)
    fail_if(args.candidate.is_symlink(), "candidate must not be a symlink")
    candidate = args.candidate.resolve(strict=True)
    fail_if(not candidate.is_file(), "candidate must be a regular file")
    candidate_sha = digest(candidate.read_bytes())
    fail_if(args.candidate_sha256 != candidate_sha, "candidate SHA-256 mismatch")

    handoff, handoff_raw = load_object(args.input_handoff, "DSC01 input handoff")
    fail_if(handoff.get("split", "TRAIN") != "TRAIN", "input handoff must identify TRAIN only")
    handoff_sha = digest(handoff_raw)
    if args.input_handoff_sha256:
        fail_if(args.input_handoff_sha256 != handoff_sha, "input handoff SHA-256 mismatch")

    metadata, _ = load_object(args.client_metadata, "client metadata")
    fail_if(set(metadata) != set(CLIENTS), "metadata must name exactly the three required clients")
    for key, row in metadata.items():
        fail_if(not isinstance(row, dict), f"metadata for {key} must be an object")
        for field in ("version", "executable", "executable_sha256", "implementation"):
            fail_if(not isinstance(row.get(field), str) or not row[field].strip(),
                    f"metadata field {key}.{field} is required")
        fail_if(SHA.fullmatch(row["executable_sha256"]) is None,
                f"metadata field {key}.executable_sha256 is invalid")

    capture_root = args.capture_root.resolve(strict=True)
    fail_if(not capture_root.is_dir(), "capture root must be a directory")
    output = args.out_dir.resolve()
    fail_if(output == repo or repo in output.parents,
            "output directory must be outside the source checkout")
    fail_if(Path("/tmp") == output or Path("/tmp") in output.parents,
            "output directory must not be under /tmp")
    inventory = {}
    for key, slug in CLIENTS.items():
        for suffix in ("execution.json", "verification.json", "mcp.jsonl"):
            name = f"{slug}.{suffix}"
            source_file = capture_root / name
            fail_if(source_file.is_symlink() or not source_file.is_file(),
                    f"required fresh sidecar missing or unsafe: {name}")
            contents = source_file.read_bytes()
            fail_if(not contents, f"sidecar is empty: {name}")
            if suffix == "execution.json":
                record, _ = load_object(source_file, name)
                fail_if(record.get("source") != args.source_sha
                        or record.get("binary_sha256") != candidate_sha
                        or record.get("client") != key
                        or record.get("client_version") != metadata[key]["version"]
                        or record.get("client_resolved_path") != metadata[key]["executable"]
                        or record.get("client_launcher_sha256") != metadata[key]["executable_sha256"],
                        f"execution identity does not bind to source, candidate and client: {key}")
            elif suffix == "verification.json":
                record, _ = load_object(source_file, name)
                fail_if(record.get("source") != args.source_sha
                        or record.get("binary_sha256") != candidate_sha
                        or record.get("client_version") != metadata[key]["version"]
                        or record.get("client_sha256") != metadata[key]["executable_sha256"],
                        f"verification identity does not bind to source, candidate and client: {key}")
            inventory[name] = {"bytes": len(contents), "sha256": digest(contents)}

    fail_if(output.exists() or output.is_symlink(), "output directory already exists")
    fail_if(output == capture_root or capture_root in output.parents,
            "output directory must not be inside capture input")
    output.mkdir(mode=0o700, parents=True)
    try:
        for name in inventory:
            shutil.copyfile(capture_root / name, output / name, follow_symlinks=False)
            os.chmod(output / name, 0o600)
        payload = {
            "schema": "jopedime.a5-capture-custody/v1",
            "status": "PREPARED_UNVERIFIED",
            "source": args.source_sha,
            "candidate_artifact_sha256": candidate_sha,
            "input_handoff": {
                "sha256": handoff_sha,
                "split": "TRAIN",
                "dataset_id": handoff.get("dataset_id"),
            },
            "clients": metadata,
            "files": inventory,
            "a5_status": "NOT_QUALIFIED",
        }
        manifest = output / "capture-manifest.json"
        manifest.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.chmod(manifest, 0o600)
        pin = output / "trusted-manifest.sha256"
        pin.write_text(digest(manifest.read_bytes()) + "\n", encoding="ascii")
        os.chmod(pin, 0o600)
    except Exception:
        # Never recursively remove a caller-supplied tree; output was freshly created.
        for child in output.iterdir():
            child.unlink()
        output.rmdir()
        raise
    return {"manifest": str(manifest), "manifest_sha256": digest(manifest.read_bytes()),
            "status": payload["status"], "a5_status": payload["a5_status"]}


def self_test() -> None:
    """Data-free contract checks; no client, engine, or dataset is launched/read."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="a5-capture-selftest-", dir=Path.home()) as temp:
        root = Path(temp)
        repo = root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c",
                        "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "fixture"], check=True)
        source = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        candidate = root / "candidate.bin"
        candidate.write_bytes(b"synthetic candidate bytes")
        handoff = root / "handoff.json"
        handoff.write_text('{"dataset_id":"fixture-only","split":"TRAIN"}\n', encoding="utf-8")
        metadata = {}
        captures = root / "captures"
        captures.mkdir()
        candidate_sha = digest(candidate.read_bytes())
        for key, slug in CLIENTS.items():
            metadata[key] = {"version": "fixture", "executable": "/fixture/client",
                             "executable_sha256": "a" * 64, "implementation": "fixture"}
            execution = {"source": source, "binary_sha256": candidate_sha,
                "client": key, "client_version": "fixture",
                "client_resolved_path": "/fixture/client", "client_launcher_sha256": "a" * 64}
            verification = {"source": source, "binary_sha256": candidate_sha,
                "client_version": "fixture", "client_sha256": "a" * 64}
            (captures / f"{slug}.execution.json").write_text(json.dumps(execution), encoding="utf-8")
            (captures / f"{slug}.verification.json").write_text(json.dumps(verification), encoding="utf-8")
            (captures / f"{slug}.mcp.jsonl").write_text("synthetic frame\n", encoding="utf-8")
        metadata_path = root / "clients.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        parser = argparse.Namespace(repo=repo, source_sha=source, candidate=candidate,
            candidate_sha256=digest(candidate.read_bytes()), input_handoff=handoff,
            input_handoff_sha256=digest(handoff.read_bytes()), client_metadata=metadata_path,
            capture_root=captures, out_dir=root / "out")
        result = prepare(parser)
        assert result["a5_status"] == "NOT_QUALIFIED"
        assert digest((root / "out/capture-manifest.json").read_bytes()) == result["manifest_sha256"]
        for broken in ({**metadata, "extra": {}}, {"claude_code": metadata["claude_code"]}):
            metadata_path.write_text(json.dumps(broken), encoding="utf-8")
            try:
                prepare(argparse.Namespace(**{**vars(parser), "out_dir": root / "rejected"}))
            except CapturePackageError:
                pass
            else:
                raise AssertionError("invalid client roster was accepted")
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        altered = json.loads((captures / "claude-code.execution.json").read_text(encoding="utf-8"))
        altered["source"] = "0" * 40
        (captures / "claude-code.execution.json").write_text(json.dumps(altered), encoding="utf-8")
        try:
            prepare(argparse.Namespace(**{**vars(parser), "out_dir": root / "bad-binding"}))
        except CapturePackageError:
            pass
        else:
            raise AssertionError("source-mismatched capture was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--input-handoff", type=Path)
    parser.add_argument("--input-handoff-sha256")
    parser.add_argument("--client-metadata", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("capture client packaging controls: PASS (synthetic fixtures only)")
        return 0
    required = (args.repo, args.source_sha, args.candidate, args.candidate_sha256,
                args.input_handoff, args.client_metadata, args.capture_root, args.out_dir)
    if any(value is None for value in required):
        parser.error("all package inputs are required unless --self-test is used")
    try:
        print(json.dumps(prepare(args), sort_keys=True))
    except CapturePackageError as exc:
        print(f"capture package refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
