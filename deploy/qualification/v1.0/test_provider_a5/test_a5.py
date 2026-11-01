from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "providers"))
import a5
import common


CLIENTS = {
    "claude_code": ("claude-code", "claude"),
    "gemini_cli": ("gemini-cli", "gemini"),
    "codex_cli": ("codex-cli", "codex"),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class A5ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.artifact = self.root / "candidate.bin"
        self.artifact.write_bytes(b"synthetic candidate; not a release artifact")
        self.source_sha = "a" * 40
        self.artifact_sha256 = digest(self.artifact.read_bytes())
        self.result = self.root / "result.json"
        self.ctx = common.Context(
            "A5", self.artifact, self.artifact_sha256, self.source_sha, self.result
        )
        self.manifest_path = self.evidence / "manifest.json"
        self.payload = self.make_payload()
        self.write_manifest()
        self.calls: list[tuple[list[str], dict]] = []
        self.dirty = ""
        self.verifier_code = 0
        self.verifier_output = None
        self.addCleanup(patch.stopall)
        patch.object(a5, "ROOT", self.checkout).start()
        patch.object(common, "ROOT", self.checkout).start()
        patch.object(a5, "verify_candidate").start()

    def write_manifest(self):
        self.payload["files"] = {}
        for name, contents in self.sidecars.items():
            path = self.evidence / name
            path.write_bytes(contents)
            self.payload["files"][name] = {
                "bytes": len(contents),
                "sha256": digest(contents),
            }
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")

    def make_payload(self):
        self.sidecars = {}
        clients = {}
        for key, (slug, client_name) in CLIENTS.items():
            client_sha = digest(key.encode())
            clients[key] = {
                "version": f"{client_name}-1.0",
                "executable": f"/clients/{key}",
                "executable_sha256": client_sha,
                "implementation": a5.IMPLEMENTATIONS[key],
            }
            self.sidecars[f"{slug}.execution.json"] = json.dumps(
                {
                    "source": self.source_sha,
                    "binary_sha256": self.artifact_sha256,
                    "client": key,
                    "client_version": clients[key]["version"],
                    "client_resolved_path": clients[key]["executable"],
                    "client_launcher_sha256": client_sha,
                    "exit_code": 0,
                    "engine_exit": 0,
                    "engine_stop_exit": 0,
                    "server_exit": 0,
                    "relay_exit": 0,
                    "timed_out": False,
                }
            ).encode()
            self.sidecars[f"{slug}.mcp.jsonl"] = b"synthetic paired wire capture\n"
            receipt = {
                "schema": "jope.mcp05-a5-client-review/v1",
                "source": self.source_sha,
                "binary_sha256": self.artifact_sha256,
                "client": client_name,
                "client_version": clients[key]["version"],
                "client_sha256": client_sha,
                "lifecycle_identity": [
                    {"clientInfo": {"name": a5.PROTOCOL_IDENTITIES[key]}}
                ],
                "tool_calls": 16,
                "successful_calls": 16,
                "failed_calls": 0,
                "tool_counts": a5.TOOL_COUNTS,
                "terminal_state": "completed",
                "evaluations_used": "1",
                "event_count": 6,
                "run_retry_contract": {
                    "successful_attempts": 2,
                    "inventory_counts": [0, 1, 1],
                    "run_id": {
                        "claude_code": "run_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                        "gemini_cli": "run_01ARZ3NDEKTSV4RRFFQ69G5FAW",
                        "codex_cli": "run_01ARZ3NDEKTSV4RRFFQ69G5FAX",
                    }[key],
                },
                "plan_placement": {"evaluation": "cpu", "workers": 1, "dope": "none"},
                "dataset_id": "ds-shared-train",
                "dataset_rows": 25,
                "dataset_columns": 13,
                "predictions": {
                    "rows": 25,
                    "columns": ["prediction"],
                    "sha256": "c" * 64,
                },
                "materialized_files": {
                    "manifest.json": {"bytes": 1, "sha256": "d" * 64},
                    "src/rust/predictor.rs": {"bytes": 1, "sha256": "e" * 64},
                },
                "export": {
                    "kind": "bundle",
                    "retains_training_data": False,
                    "languages": [{"language": "rust", "status": "exported"}],
                },
                "raw_capture_hashes_verified": True,
                "raw_messages_match_wire": True,
                "all_rpc_requests_paired": True,
                "text_matches_structured_results": True,
                "transport_complete": True,
                "identical_run_retry_outcome": True,
                "events_contiguous": True,
                "identical_export_retry": True,
                "ag1_named_client_flow_pass": True,
                "client_exit": 0,
                "engine_exit": 0,
                "engine_stop_exit": 0,
                "server_exit": 0,
                "relay_exit": 0,
            }
            self.sidecars[f"{slug}.verification.json"] = json.dumps(receipt).encode()
        return {
            "schema": "jopedime.named-client-transcripts/v1",
            "source": self.source_sha,
            "candidate_artifact_sha256": self.artifact_sha256,
            "a5": {
                "full_path_observed_clients": 3,
                "successful_tool_calls": 48,
                "failed_tool_calls": 0,
            },
            "clients": clients,
            "files": {},
        }

    def run_child(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[:2] == ["git", "rev-parse"]:
            output, code = self.source_sha, 0
        elif args[:2] == ["git", "status"]:
            output, code = self.dirty, 0
        elif len(args) > 2 and args[:2] == ["python", "-B"]:
            output, code = "", 0
        elif (
            args
            and str(args[0]) == sys.executable
            and str(args[2]).endswith("verify_capture.py")
        ):
            output = self.verifier_output or json.dumps(
                {
                    "integrity": "PASS",
                    "sidecars_verified": len(self.payload["files"]),
                    "a5_status": "UNVERIFIED:READY_FOR_REVIEW",
                    "run_retry_contract": {
                        key: {
                            "run_id": json.loads(
                                self.sidecars[f"{CLIENTS[key][0]}.verification.json"]
                            )["run_retry_contract"]["run_id"],
                            "successful_attempts": 2,
                            "inventory_counts": [0, 1, 1],
                        }
                        for key in CLIENTS
                    },
                }
            )
            code = self.verifier_code
        else:
            self.fail(f"unexpected child command: {args}")
        return subprocess.CompletedProcess(args, code, output, "")

    def invoke(self):
        with patch.dict(
            "os.environ",
            {
                "JOPE_REL07_A5_MANIFEST": str(self.manifest_path),
                "JOPE_REL07_A5_MANIFEST_SHA256": digest(
                    self.manifest_path.read_bytes()
                ),
            },
        ):
            return a5.qualify(self.ctx, run=self.run_child)

    def test_three_candidate_bound_named_clients_write_a5_result(self):
        self.assertEqual(self.invoke(), 0)
        result = json.loads(self.result.read_text())
        self.assertEqual(result["id"], "A5")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["source_sha"], self.source_sha)
        self.assertEqual(result["artifact_sha256"], self.artifact_sha256)
        self.assertEqual(result["successful_tool_calls"], 48)
        self.assertEqual(result["counts"]["passed"], 3)

    def test_wrong_candidate_or_source_binding_refuses(self):
        self.payload["candidate_artifact_sha256"] = "0" * 64
        self.write_manifest()
        with self.assertRaisesRegex(common.ProviderError, "candidate bytes"):
            self.invoke()
        self.assertFalse(self.result.exists())
        self.payload["candidate_artifact_sha256"] = self.artifact_sha256
        self.payload["source"] = "b" * 40
        self.write_manifest()
        with self.assertRaisesRegex(common.ProviderError, "candidate source"):
            self.invoke()
        self.assertFalse(self.result.exists())

    def test_missing_client_or_incomplete_receipt_refuses(self):
        del self.payload["clients"]["gemini_cli"]
        self.write_manifest()
        with self.assertRaisesRegex(common.ProviderError, "exactly the three"):
            self.invoke()
        self.payload = self.make_payload()
        receipt = json.loads(self.sidecars["claude-code.verification.json"])
        receipt["failed_calls"] = 1
        self.sidecars["claude-code.verification.json"] = json.dumps(receipt).encode()
        self.write_manifest()
        with self.assertRaisesRegex(common.ProviderError, "flow or exact retry"):
            self.invoke()
        self.assertFalse(self.result.exists())

    def test_manifest_or_sidecar_symlink_refuses(self):
        link = self.evidence / "linked.json"
        link.symlink_to(self.manifest_path)
        with patch.dict("os.environ", {"JOPE_REL07_A5_MANIFEST": str(link)}):
            with self.assertRaisesRegex(common.ProviderError, "non-symlink"):
                a5.evidence_manifest()
        sidecar = self.evidence / "claude-code.mcp.jsonl"
        target = self.evidence / "wire-copy.jsonl"
        sidecar.rename(target)
        sidecar.symlink_to(target)
        with self.assertRaisesRegex(common.ProviderError, "sidecar is missing"):
            self.invoke()
        self.assertFalse(self.result.exists())

    def test_raw_verifier_failure_and_dirty_source_refuse(self):
        self.verifier_code = 1
        with self.assertRaisesRegex(common.ProviderError, "raw transcript"):
            self.invoke()
        self.assertFalse(self.result.exists())
        self.verifier_code = 0
        self.dirty = "?? stray"
        with self.assertRaisesRegex(common.ProviderError, "clean candidate-pinned"):
            self.invoke()
        self.assertFalse(self.result.exists())

    def test_raw_retry_ids_must_match_each_semantic_receipt(self):
        self.verifier_output = json.dumps(
            {
                "integrity": "PASS",
                "sidecars_verified": len(self.payload["files"]),
                "run_retry_contract": {
                    key: {
                        "run_id": "run_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                        "successful_attempts": 2,
                        "inventory_counts": [0, 1, 1],
                    }
                    for key in CLIENTS
                },
            }
        )
        with self.assertRaisesRegex(common.ProviderError, "raw verifier"):
            self.invoke()
        self.assertFalse(self.result.exists())

    def test_result_must_be_new_and_outside_checkout(self):
        self.ctx = common.Context(
            "A5",
            self.artifact,
            self.artifact_sha256,
            self.source_sha,
            self.checkout / "result.json",
        )
        with self.assertRaisesRegex(
            common.ProviderError, "outside the source checkout"
        ):
            self.invoke()
        self.assertFalse(self.result.exists())


if __name__ == "__main__":
    unittest.main()
