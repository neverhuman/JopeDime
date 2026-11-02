"""Validation and custody packaging for named-client capture sidecars."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

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


def parse_object(raw: bytes, label: str) -> dict:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapturePackageError(f"{label} is invalid JSON") from exc
    fail_if(not isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_object(path: Path, label: str) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CapturePackageError(f"{label} is unreadable") from exc
    return parse_object(raw, label), raw


def source_identity(repo: Path, claimed: str) -> None:
    fail_if(
        GIT_SHA.fullmatch(claimed) is None, "source SHA must be 40 lowercase hex chars"
    )
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CapturePackageError(
            "cannot establish repository source identity"
        ) from exc
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
    fail_if(
        handoff.get("split") != "TRAIN", "input handoff must explicitly identify TRAIN"
    )
    fail_if(
        not isinstance(handoff.get("dataset_id"), str)
        or not handoff["dataset_id"].strip(),
        "input handoff must contain a nonblank dataset_id",
    )
    handoff_sha = digest(handoff_raw)
    fail_if(
        not isinstance(args.input_handoff_sha256, str)
        or SHA.fullmatch(args.input_handoff_sha256) is None,
        "an expected input handoff SHA-256 is required",
    )
    fail_if(args.input_handoff_sha256 != handoff_sha, "input handoff SHA-256 mismatch")

    metadata, _ = load_object(args.client_metadata, "client metadata")
    fail_if(
        set(metadata) != set(CLIENTS),
        "metadata must name exactly the three required clients",
    )
    for key, row in metadata.items():
        fail_if(not isinstance(row, dict), f"metadata for {key} must be an object")
        for field in ("version", "executable", "executable_sha256", "implementation"):
            fail_if(
                not isinstance(row.get(field), str) or not row[field].strip(),
                f"metadata field {key}.{field} is required",
            )
        fail_if(
            SHA.fullmatch(row["executable_sha256"]) is None,
            f"metadata field {key}.executable_sha256 is invalid",
        )

    requested = getattr(args, "clients", None)
    selected = list(CLIENTS) if requested is None else requested
    fail_if(
        not selected
        or any(not isinstance(key, str) or key not in CLIENTS for key in selected),
        "selected clients must name at least one supported client",
    )
    fail_if(len(selected) != len(set(selected)), "selected clients must be unique")
    selected_set = set(selected)
    selected = [key for key in CLIENTS if key in selected_set]
    excluded = [key for key in CLIENTS if key not in selected_set]
    excluded_clients = {}
    for key in selected:
        fail_if(
            metadata[key].get("status") not in (None, "INCLUDED"),
            f"selected client has an invalid disposition in metadata: {key}",
        )
    for key in excluded:
        row = metadata[key]
        reason = row.get("exclusion_reason")
        fail_if(
            row.get("status") != "EXCLUDED"
            or not isinstance(reason, str)
            or not reason.strip(),
            f"omitted client must be EXCLUDED with a reason in metadata: {key}",
        )
        excluded_clients[key] = reason.strip()
    fail_if(
        excluded and getattr(args, "provider_ready", False),
        "provider-ready packaging requires all three clients; partial captures remain NOT_QUALIFIED",
    )

    capture_root = args.capture_root.resolve(strict=True)
    fail_if(not capture_root.is_dir(), "capture root must be a directory")
    output = args.out_dir.resolve()
    fail_if(
        output == repo or repo in output.parents,
        "output directory must be outside the source checkout",
    )
    fail_if(
        Path("/tmp") == output or Path("/tmp") in output.parents,
        "output directory must not be under /tmp",
    )
    inventory = {}
    for key, slug in CLIENTS.items():
        suffixes = (
            ("execution.json",)
            if key in excluded_clients
            else (
                "execution.json",
                "verification.json",
                "mcp.jsonl",
            )
        )
        for suffix in suffixes:
            name = f"{slug}.{suffix}"
            source_file = capture_root / name
            fail_if(
                source_file.is_symlink() or not source_file.is_file(),
                f"required fresh sidecar missing or unsafe: {name}",
            )
            contents = source_file.read_bytes()
            fail_if(not contents, f"sidecar is empty: {name}")
            if suffix == "execution.json":
                record = parse_object(contents, name)
                fail_if(
                    record.get("source") != args.source_sha
                    or record.get("binary_sha256") != candidate_sha
                    or record.get("client") != key
                    or record.get("client_version") != metadata[key]["version"]
                    or record.get("client_resolved_path") != metadata[key]["executable"]
                    or record.get("client_launcher_sha256")
                    != metadata[key]["executable_sha256"],
                    f"execution identity does not bind to source, candidate and client: {key}",
                )
                if key in excluded_clients:
                    fail_if(
                        record.get("status") != "EXCLUDED"
                        or record.get("exclusion_reason") != excluded_clients[key],
                        f"excluded client execution receipt does not record its exclusion: {key}",
                    )
                else:
                    fail_if(
                        record.get("status", "INCLUDED") == "EXCLUDED",
                        f"selected client execution receipt is marked EXCLUDED: {key}",
                    )
            elif suffix == "verification.json":
                record = parse_object(contents, name)
                fail_if(
                    record.get("source") != args.source_sha
                    or record.get("binary_sha256") != candidate_sha
                    or record.get("client_version") != metadata[key]["version"]
                    or record.get("client_sha256")
                    != metadata[key]["executable_sha256"],
                    f"verification identity does not bind to source, candidate and client: {key}",
                )
                if getattr(args, "provider_ready", False):
                    fail_if(
                        type(record.get("tool_calls")) is not int
                        or record["tool_calls"] != 16
                        or type(record.get("successful_calls")) is not int
                        or record["successful_calls"] != 16
                        or type(record.get("failed_calls")) is not int
                        or record["failed_calls"] != 0
                        or record.get("ag1_named_client_flow_pass") is not True,
                        f"provider manifest requires a complete successful client receipt: {key}",
                    )
            inventory[name] = {"bytes": len(contents), "sha256": digest(contents)}

    fail_if(output.exists() or output.is_symlink(), "output directory already exists")
    fail_if(
        output == capture_root or capture_root in output.parents,
        "output directory must not be inside capture input",
    )
    output.mkdir(mode=0o700, parents=True)
    try:
        for name in inventory:
            copied = output / name
            shutil.copyfile(capture_root / name, copied, follow_symlinks=False)
            fail_if(
                copied.is_symlink() or not copied.is_file(),
                f"sidecar changed while copying: {name}",
            )
            copied_bytes = copied.read_bytes()
            fail_if(
                len(copied_bytes) != inventory[name]["bytes"]
                or digest(copied_bytes) != inventory[name]["sha256"],
                f"sidecar changed while copying: {name}",
            )
            os.chmod(copied, 0o600)
        source_identity(repo, args.source_sha)
        fail_if(
            args.candidate.is_symlink() or not args.candidate.is_file(),
            "candidate must remain a regular non-symlink file",
        )
        fail_if(
            digest(args.candidate.read_bytes()) != candidate_sha,
            "candidate changed during capture copy",
        )
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
            "selected_clients": selected,
            "excluded_clients": excluded_clients,
            "clients": metadata,
            "files": inventory,
            "a5_status": "NOT_QUALIFIED",
        }
        manifest = output / "capture-manifest.json"
        manifest.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.chmod(manifest, 0o600)
        pin = output / "trusted-manifest.sha256"
        pin.write_text(digest(manifest.read_bytes()) + "\n", encoding="ascii")
        os.chmod(pin, 0o600)
        provider_manifest_path = None
        provider_manifest_sha256 = None
        if getattr(args, "provider_ready", False):
            provider_payload = {
                "schema": "jopedime.named-client-transcripts/v1",
                "status": "PREPARED_UNVERIFIED",
                "source": args.source_sha,
                "candidate_artifact_sha256": candidate_sha,
                "input": {
                    "split": "TRAIN",
                    "dataset_id": handoff["dataset_id"],
                    "handoff_sha256": handoff_sha,
                },
                "a5": {
                    "status": "PREPARED_UNVERIFIED",
                    "full_path_observed_clients": len(CLIENTS),
                    "successful_tool_calls": 16 * len(CLIENTS),
                    "failed_tool_calls": 0,
                },
                "clients": metadata,
                "files": inventory,
            }
            provider_manifest_path = output / "a5-manifest.json"
            provider_manifest_path.write_text(
                json.dumps(provider_payload, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            os.chmod(provider_manifest_path, 0o600)
            provider_manifest_sha256 = digest(provider_manifest_path.read_bytes())
            provider_pin = output / "a5-manifest.sha256"
            provider_pin.write_text(provider_manifest_sha256 + "\n", encoding="ascii")
            os.chmod(provider_pin, 0o600)
    except Exception:
        # Never recursively remove a caller-supplied tree; output was freshly created.
        for child in output.iterdir():
            child.unlink()
        output.rmdir()
        raise
    result = {
        "manifest": str(manifest),
        "manifest_sha256": digest(manifest.read_bytes()),
        "status": payload["status"],
        "a5_status": payload["a5_status"],
    }
    if provider_manifest_path is not None:
        result.update(
            provider_manifest=str(provider_manifest_path),
            provider_manifest_sha256=provider_manifest_sha256,
            provider_status="PREPARED_UNVERIFIED",
        )
    return result


