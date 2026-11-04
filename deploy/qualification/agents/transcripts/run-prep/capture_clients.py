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

if __package__:
    from .capture_clients_package import (
        CLIENTS, GIT_SHA, SHA, CapturePackageError, digest, fail_if, load_object,
        parse_object, prepare, source_identity,
    )
else:
    from capture_clients_package import (
        CLIENTS, GIT_SHA, SHA, CapturePackageError, digest, fail_if, load_object,
        parse_object, prepare, source_identity,
    )


def self_test() -> None:
    """Data-free contract checks; no client, engine, or dataset is launched/read."""
    import tempfile

    with tempfile.TemporaryDirectory(
        prefix="a5-capture-selftest-", dir=Path.home()
    ) as temp:
        root = Path(temp)
        repo = root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--allow-empty",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        source = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        candidate = root / "candidate.bin"
        candidate.write_bytes(b"synthetic candidate bytes")
        handoff = root / "handoff.json"
        handoff.write_text(
            '{"dataset_id":"fixture-only","split":"TRAIN"}\n', encoding="utf-8"
        )
        metadata = {}
        captures = root / "captures"
        captures.mkdir()
        candidate_sha = digest(candidate.read_bytes())
        for key, slug in CLIENTS.items():
            metadata[key] = {
                "version": "fixture",
                "executable": "/fixture/client",
                "executable_sha256": "a" * 64,
                "implementation": "fixture",
            }
            execution = {
                "source": source,
                "binary_sha256": candidate_sha,
                "client": key,
                "client_version": "fixture",
                "client_resolved_path": "/fixture/client",
                "client_launcher_sha256": "a" * 64,
            }
            verification = {
                "source": source,
                "binary_sha256": candidate_sha,
                "client_version": "fixture",
                "client_sha256": "a" * 64,
                "tool_calls": 16,
                "successful_calls": 16,
                "failed_calls": 0,
                "ag1_named_client_flow_pass": True,
            }
            (captures / f"{slug}.execution.json").write_text(
                json.dumps(execution), encoding="utf-8"
            )
            (captures / f"{slug}.verification.json").write_text(
                json.dumps(verification), encoding="utf-8"
            )
            (captures / f"{slug}.mcp.jsonl").write_text(
                "synthetic frame\n", encoding="utf-8"
            )
        metadata_path = root / "clients.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        parser = argparse.Namespace(
            repo=repo,
            source_sha=source,
            candidate=candidate,
            candidate_sha256=digest(candidate.read_bytes()),
            input_handoff=handoff,
            input_handoff_sha256=digest(handoff.read_bytes()),
            client_metadata=metadata_path,
            capture_root=captures,
            out_dir=root / "out",
            provider_ready=False,
        )
        result = prepare(parser)
        assert result["a5_status"] == "NOT_QUALIFIED"
        assert (
            digest((root / "out/capture-manifest.json").read_bytes())
            == result["manifest_sha256"]
        )
        provider_result = prepare(
            argparse.Namespace(
                **{
                    **vars(parser),
                    "out_dir": root / "out-provider",
                    "provider_ready": True,
                }
            )
        )
        provider_manifest = json.loads(
            (root / "out-provider/a5-manifest.json").read_text(encoding="utf-8")
        )
        assert provider_result["provider_status"] == "PREPARED_UNVERIFIED"
        assert provider_manifest["schema"] == "jopedime.named-client-transcripts/v1"
        assert provider_manifest["source"] == source
        assert provider_manifest["candidate_artifact_sha256"] == candidate_sha
        assert provider_manifest["a5"] == {
            "status": "PREPARED_UNVERIFIED",
            "full_path_observed_clients": 3,
            "successful_tool_calls": 48,
            "failed_tool_calls": 0,
        }
        assert (
            digest((root / "out-provider/a5-manifest.json").read_bytes())
            == (root / "out-provider/a5-manifest.sha256").read_text().strip()
        )
        partial_captures = root / "partial-captures"
        partial_captures.mkdir()
        partial_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, slug in CLIENTS.items():
            if key == "gemini_cli":
                reason = "Gemini CLI MCP configuration unavailable"
                partial_metadata[key]["status"] = "EXCLUDED"
                partial_metadata[key]["exclusion_reason"] = reason
                execution = {
                    "source": source,
                    "binary_sha256": candidate_sha,
                    "client": key,
                    "client_version": "fixture",
                    "client_resolved_path": "/fixture/client",
                    "client_launcher_sha256": "a" * 64,
                    "status": "EXCLUDED",
                    "exclusion_reason": reason,
                }
                (partial_captures / f"{slug}.execution.json").write_text(
                    json.dumps(execution), encoding="utf-8"
                )
                continue
            for suffix in ("execution.json", "verification.json", "mcp.jsonl"):
                source_file = captures / f"{slug}.{suffix}"
                (partial_captures / source_file.name).write_bytes(
                    source_file.read_bytes()
                )
        partial_metadata_path = root / "partial-clients.json"
        partial_metadata_path.write_text(json.dumps(partial_metadata), encoding="utf-8")
        partial_args = argparse.Namespace(
            **{
                **vars(parser),
                "client_metadata": partial_metadata_path,
                "capture_root": partial_captures,
                "out_dir": root / "out-partial",
                "clients": ["claude_code", "codex_cli"],
            }
        )
        partial_result = prepare(partial_args)
        partial_manifest = json.loads(
            (root / "out-partial/capture-manifest.json").read_text(encoding="utf-8")
        )
        assert partial_result["a5_status"] == "NOT_QUALIFIED"
        assert partial_manifest["selected_clients"] == [
            "claude_code",
            "codex_cli",
        ]
        assert partial_manifest["excluded_clients"] == {
            "gemini_cli": "Gemini CLI MCP configuration unavailable"
        }
        assert "gemini-cli.mcp.jsonl" not in partial_manifest["files"]
        try:
            prepare(
                argparse.Namespace(
                    **{
                        **vars(partial_args),
                        "out_dir": root / "out-partial-provider",
                        "provider_ready": True,
                    }
                )
            )
        except CapturePackageError:
            pass
        else:
            raise AssertionError("provider-ready accepted a partial client roster")
        verification_path = captures / "claude-code.verification.json"
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        verification["failed_calls"] = 1
        verification_path.write_text(json.dumps(verification), encoding="utf-8")
        try:
            prepare(
                argparse.Namespace(
                    **{
                        **vars(parser),
                        "out_dir": root / "rejected-summary",
                        "provider_ready": True,
                    }
                )
            )
        except CapturePackageError:
            pass
        else:
            raise AssertionError("provider manifest accepted a failed client receipt")
        verification["failed_calls"] = 0
        verification_path.write_text(json.dumps(verification), encoding="utf-8")
        for broken in (
            {**metadata, "extra": {}},
            {"claude_code": metadata["claude_code"]},
        ):
            metadata_path.write_text(json.dumps(broken), encoding="utf-8")
            try:
                prepare(
                    argparse.Namespace(**{**vars(parser), "out_dir": root / "rejected"})
                )
            except CapturePackageError:
                pass
            else:
                raise AssertionError("invalid client roster was accepted")
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        altered = json.loads(
            (captures / "claude-code.execution.json").read_text(encoding="utf-8")
        )
        altered["source"] = "0" * 40
        (captures / "claude-code.execution.json").write_text(
            json.dumps(altered), encoding="utf-8"
        )
        try:
            prepare(
                argparse.Namespace(**{**vars(parser), "out_dir": root / "bad-binding"})
            )
        except CapturePackageError:
            pass
        else:
            raise AssertionError("source-mismatched capture was accepted")
        handoff.write_text('{"dataset_id":"fixture-only"}\n', encoding="utf-8")
        try:
            prepare(
                argparse.Namespace(**{**vars(parser), "out_dir": root / "bad-train"})
            )
        except CapturePackageError:
            pass
        else:
            raise AssertionError("handoff without explicit TRAIN split was accepted")


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
    parser.add_argument(
        "--clients",
        nargs="+",
        choices=CLIENTS,
        help="clients represented by the package (default: all)",
    )
    parser.add_argument(
        "--provider-ready",
        action="store_true",
        help="also write a candidate-bound A5 input manifest from three successful receipts",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("capture client packaging controls: PASS (synthetic fixtures only)")
        return 0
    required = (
        args.repo,
        args.source_sha,
        args.candidate,
        args.candidate_sha256,
        args.input_handoff,
        args.client_metadata,
        args.capture_root,
        args.out_dir,
    )
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
