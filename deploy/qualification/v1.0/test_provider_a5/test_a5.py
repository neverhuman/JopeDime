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


sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5_support


class A5ProviderTests(a5_support.A5ProviderSupport):
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

    def test_five_raw_retry_calls_cannot_use_positive_semantic_receipts(self):
        for slug, _client_name in CLIENTS.values():
            self.sidecars[f"{slug}.mcp.jsonl"] = self.make_wire(
                next(key for key, value in CLIENTS.items() if value[0] == slug),
                partial=True,
            )
        self.write_manifest()
        manifest_sha = digest(self.manifest_path.read_bytes())
        verifier = subprocess.run(
            [
                sys.executable,
                "-B",
                str(a5._CAPTURE_PATH),
                str(self.manifest_path),
                "--check-run-retry",
                "--manifest-sha256",
                manifest_sha,
            ],
            cwd=self.checkout,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(verifier.returncode, 0, verifier.stderr)
        verifier_result = json.loads(verifier.stdout)
        self.assertEqual(verifier_result["integrity"], "PASS")
        self.assertEqual(verifier_result["a5_status"], "UNVERIFIED:missing")
        self.assertEqual(
            verifier_result["sidecars_verified"], len(self.payload["files"])
        )
        with self.assertRaisesRegex(common.ProviderError, "raw flow tool inventory"):
            self.invoke()
        self.assertFalse(self.result.exists())

    def test_snapshot_events_and_prediction_cannot_mix_run_or_input_identity(self):
        original = self.make_wire("claude_code").decode()
        for kind in ("snapshot", "events", "prediction"):
            frames = [json.loads(line) for line in original.splitlines()]
            for index, frame in enumerate(frames):
                message = frame["message"]
                params = message.get("params", {})
                response = (
                    frames[index + 1]["message"]
                    .get("result", {})
                    .get("structuredContent", {})
                    if index + 1 < len(frames)
                    else {}
                )
                if (
                    kind == "snapshot"
                    and params.get("name") == "run_status"
                    and params.get("arguments", {}).get("run_id")
                ):
                    response["run_id"] = "run_01ARZ3NDEKTSV4RRFFQ69G5FAX"
                elif kind == "events" and params.get("name") == "run_events":
                    response["events"][-1]["sequence"] = "1"
                elif kind == "prediction" and params.get("name") == "model_predict":
                    response["rows"] = "24"
            self.sidecars["claude-code.mcp.jsonl"] = b"\n".join(
                json.dumps(frame).encode() for frame in frames
            ) + b"\n"
            self.write_manifest()
            with self.subTest(kind=kind), self.assertRaises(common.ProviderError):
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

    def test_candidate_symlink_swap_after_raw_verifier_refuses(self):
        target = self.root / "same-candidate-bytes.bin"
        target.write_bytes(self.artifact.read_bytes())
        original = self.run_child

        def swap(args, **kwargs):
            result = original(args, **kwargs)
            if len(args) > 2 and str(args[2]).endswith("verify_capture.py"):
                self.artifact.unlink()
                self.artifact.symlink_to(target)
            return result

        self.run_child = swap
        with self.assertRaisesRegex(common.ProviderError, "non-symlink"):
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
