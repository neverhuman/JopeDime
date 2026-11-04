from __future__ import annotations

import hashlib
import json
import shutil
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


class A5ProviderSupport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.verifier_path = (
            self.checkout
            / "deploy/qualification/agents/transcripts/run-prep/verify_capture.py"
        )
        self.verifier_path.parent.mkdir(parents=True)
        shutil.copyfile(a5._CAPTURE_PATH, self.verifier_path)
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
            self.sidecars[f"{slug}.mcp.jsonl"] = self.make_wire(key)
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
                    "run_id": self.run_id(key),
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

    @staticmethod
    def run_id(key):
        return {
            "claude_code": "run_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "gemini_cli": "run_01ARZ3NDEKTSV4RRFFQ69G5FAW",
            "codex_cli": "run_01ARZ3NDEKTSV4RRFFQ69G5FAX",
        }[key]

    def make_wire(self, key, *, partial=False):
        run_id = self.run_id(key)
        dataset_id = "ds-shared-train"
        plan_digest = "b" * 64
        artifact_id = "art-verified-model"
        running = {"run_id": run_id, "state": "running"}
        completed = {
            "run_id": run_id,
            "state": "completed",
            "best": {"model_artifact": artifact_id},
            "budget_used": {"evaluations": "1"},
            "event_watermark": "6",
        }
        inventory_before = {"runs": [], "next_after": None}
        inventory_after = {"runs": [{"run_id": run_id}], "next_after": None}
        start_args = {
            "dataset": dataset_id,
            "dry_run": False,
            "plan_digest": plan_digest,
            "idempotency_key": "start-run",
            "device": "cpu",
        }
        events = [
            {"sequence": str(i), "run_id": run_id, "kind": kind}
            for i, kind in enumerate(
                [
                    "run.started",
                    "experiment.started",
                    "experiment.finished",
                    "algorithm.accepted",
                    "checkpoint.saved",
                    "run.finished",
                ],
                start=1,
            )
        ]
        calls = [
            ("system_info", {}, {"health": "ok"}),
            ("run_status", {}, inventory_before),
            ("dataset_register", {}, {"dataset_id": dataset_id}),
            ("dataset_register", {}, {"dataset_id": dataset_id}),
            (
                "dataset_inspect",
                {"dataset_id": dataset_id},
                {"dataset_id": dataset_id, "rows": "25"},
            ),
            (
                "run_start",
                {"dataset": dataset_id, "dry_run": True},
                {
                    "plan_digest": plan_digest,
                    "placement": {"evaluation": "cpu", "workers": 1, "dope": "none"},
                    "effort": {"evaluations": "1"},
                },
            ),
            ("run_start", start_args, running),
            ("run_status", {}, inventory_after),
            ("run_start", start_args, running),
            ("run_status", {}, inventory_after),
            ("run_status", {"run_id": run_id}, completed),
            ("run_status", {"run_id": run_id}, completed),
            ("run_events", {"run_id": run_id}, {"events": events}),
            (
                "model_export",
                {"run_id": run_id, "idempotency_key": "export-run"},
                {
                    "kind": "bundle",
                    "artifact_id": artifact_id,
                    "retains_training_data": False,
                    "languages": [{"language": "rust", "status": "exported"}],
                },
            ),
            (
                "model_export",
                {"run_id": run_id, "idempotency_key": "export-run"},
                {
                    "kind": "bundle",
                    "artifact_id": artifact_id,
                    "retains_training_data": False,
                    "languages": [{"language": "rust", "status": "exported"}],
                },
            ),
            (
                "model_predict",
                {"model": {"run_id": run_id}, "input": {"dataset_id": dataset_id}},
                {"rows": "25", "sha256": "c" * 64},
            ),
        ]
        if partial:
            calls = [
                ("run_status", {}, inventory_before),
                ("run_start", start_args, running),
                ("run_status", {}, inventory_after),
                ("run_start", start_args, running),
                ("run_status", {}, inventory_after),
            ]
        frames = []
        for rpc_id, (name, arguments, data) in enumerate(calls):
            frames.extend(
                [
                    {
                        "direction": "client_to_server",
                        "message": {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "method": "tools/call",
                            "params": {"name": name, "arguments": arguments},
                        },
                    },
                    {
                        "direction": "server_to_client",
                        "message": {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "result": {"isError": False, "structuredContent": data},
                        },
                    },
                ]
            )
        return b"\n".join(json.dumps(frame).encode() for frame in frames) + b"\n"

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
            if self.verifier_code == 0 and self.verifier_output is None:
                return subprocess.run(args, **kwargs)
            output = self.verifier_output or json.dumps(
                {
                    "integrity": "PASS",
                    "sidecars_verified": len(self.payload["files"]),
                    "a5_status": "UNVERIFIED:missing",
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
