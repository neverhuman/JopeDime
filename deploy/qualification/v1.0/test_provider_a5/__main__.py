"""Synthetic packet/controller controls; no named client, data or engine runs."""

import contextlib
from dataclasses import replace
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "providers"))
a4, a5 = importlib.import_module("a4"), importlib.import_module("a5")
common = importlib.import_module("common")


class A5Tests(unittest.TestCase):
    def setUp(self):
        folder = Path(os.environ.get("JOPE_QUALIFICATION_TEST_TMP", HERE / "target"))
        folder.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix="a5-control-", dir=folder)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo, self.packet = self.root / "checkout", self.root / "packet"
        self.repo.mkdir()
        self.packet.mkdir()
        self.artifact = self.root / "candidate"
        self.artifact.write_bytes(b"controller-only fixture; no release signature")
        self.ctx = common.Context(
            "A5",
            self.artifact,
            common.sha256(self.artifact),
            "a" * 40,
            self.root / "result.json",
        )
        self.source, self.dirty = self.ctx.source_sha, ""
        self.after_verify = lambda: None
        self.calls = []
        self.manifest = {
            "schema": "jopedime.named-client-transcripts/v1",
            "source": self.ctx.source_sha,
            "clients": {},
            "files": {},
            "a5": {"status": "NOT_QUALIFIED:synthetic controls"},
        }
        self.manifest_path = self.packet / "manifest.json"
        self.train_hash = "f" * 64
        self.put(
            "source-and-input.json",
            {
                "runtime_source": self.ctx.source_sha,
                "binary_sha256": self.ctx.artifact_sha256,
                "version": {
                    "source_sha40": self.ctx.source_sha,
                    "features": {"dev_keys": False},
                    "trusts_development_key": False,
                },
                "input": {
                    "header_train_sha256": self.train_hash,
                    "rows": 25,
                    "external_test_dataset": None,
                    "sealed_or_external_holdout_read": False,
                },
            },
        )
        run_ids = [
            "run_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "run_01ARZ3NDEKTSV4RRFFQ69G5FAW",
            "run_01ARZ3NDEKTSV4RRFFQ69G5FAX",
        ]
        for index, (client, slug) in enumerate(a5.CLIENTS.items()):
            version = ["2.1.307 (Claude Code)", "1.4.12", "codex-cli 0.166.0"][index]
            client_hash = hashlib.sha256(client.encode()).hexdigest()
            self.manifest["clients"][client] = {
                "version": version,
                "executable_sha256": client_hash,
            }
            self.put(
                slug + ".execution.json",
                {
                    "client": client,
                    "source": self.ctx.source_sha,
                    "binary_sha256": self.ctx.artifact_sha256,
                    "client_version": version,
                    "client_launcher_sha256": client_hash,
                    "fixture_sha256": self.train_hash,
                    "exit_code": 0,
                    "client_exit": 0,
                    "engine_exit": 0,
                    "engine_stop_exit": 0,
                    "forced_engine_signal": False,
                    "binary_unchanged": True,
                    "timed_out": False,
                },
            )
            self.put(slug + ".command.json", [slug, "--fixture-only"])
            self.make_wire(slug, run_ids[index], sorted(a5.CLIENT_NAMES[client])[0])
        self.freeze()
        self.addCleanup(patch.stopall)
        patch.object(a4, "ROOT", self.repo).start()

    def put(self, name, value):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        (self.packet / name).write_bytes(raw)
        self.manifest["files"][name] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def freeze(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.digest = common.sha256(self.manifest_path)

    def change(self, name, edit):
        value = json.loads((self.packet / name).read_text())
        edit(value)
        self.put(name, value)
        self.freeze()

    def make_wire(self, slug, run_id, client_name):
        dataset, plan = "ds_" + self.train_hash[:24], "b" * 64
        artifact = "art_" + "c" * 24

        def page(ids):
            return {"runs": [{"run_id": r} for r in ids], "next_after": None}

        start = {
            "dataset": dataset,
            "idempotency_key": "start",
            "dry_run": False,
            "plan_digest": plan,
        }
        outcome = {"run_id": run_id, "state": "running"}
        self.frames = [
            {
                "direction": "client_to_server",
                "message": {
                    "jsonrpc": "2.0",
                    "id": "initialize",
                    "method": "initialize",
                    "params": {
                        "clientInfo": {
                            "name": client_name,
                            "version": "protocol-fixture",
                        }
                    },
                },
            },
            {
                "direction": "server_to_client",
                "message": {"jsonrpc": "2.0", "id": "initialize", "result": {}},
            },
        ]
        calls = [
            ("system_info", {}, {"version": "fixture"}),
            ("run_status", {}, page([])),
            (
                "dataset_register",
                {"path": "fixture-not-read", "idempotency_key": "register"},
                {"dataset_id": dataset},
            ),
            (
                "dataset_inspect",
                {"dataset_id": dataset},
                {"dataset_id": dataset, "rows": "25"},
            ),
            (
                "run_start",
                {"dataset": dataset, "dry_run": True, "idempotency_key": "plan"},
                {"plan_digest": plan},
            ),
            ("run_start", start, outcome),
            ("run_status", {}, page([run_id])),
            ("run_start", start, outcome),
            ("run_status", {}, page([run_id])),
            (
                "run_status",
                {"run_id": run_id},
                {
                    "run_id": run_id,
                    "state": "completed",
                    "best": {"model_artifact": artifact},
                    "budget_used": {"evaluations": "1"},
                    "event_watermark": "3",
                },
            ),
            (
                "run_events",
                {"run_id": run_id},
                {
                    "events": [
                        {"sequence": str(i + 1), "run_id": run_id, "kind": kind}
                        for i, kind in enumerate(
                            ["run.started", "experiment.finished", "run.finished"]
                        )
                    ]
                },
            ),
            (
                "model_export",
                {"run_id": run_id},
                {
                    "kind": "bundle",
                    "artifact_id": artifact,
                    "languages": [{"language": "rust", "status": "exported"}],
                },
            ),
            (
                "model_predict",
                {"model": {"run_id": run_id}, "input": {"dataset_id": dataset}},
                {"rows": "25", "sha256": "d" * 64},
            ),
        ]
        for index, (name, args, data) in enumerate(calls):
            self.frames.extend(
                [
                    {
                        "direction": "client_to_server",
                        "message": {
                            "jsonrpc": "2.0",
                            "id": index,
                            "method": "tools/call",
                            "params": {"name": name, "arguments": args},
                        },
                    },
                    {
                        "direction": "server_to_client",
                        "message": {
                            "jsonrpc": "2.0",
                            "id": index,
                            "result": {"isError": False, "structuredContent": data},
                        },
                    },
                ]
            )
        self.save_wire(slug, self.frames)

    def save_wire(self, slug, frames):
        wire = b"\n".join(json.dumps(f).encode() for f in frames) + b"\n"
        raw_files = {"wire.jsonl": wire, "server.stderr": b""}
        for name, direction in [
            ("client.raw", "client_to_server"),
            ("server.raw", "server_to_client"),
        ]:
            raw_files[name] = (
                b"\n".join(
                    json.dumps(f["message"]).encode()
                    for f in frames
                    if f["direction"] == direction
                )
                + b"\n"
            )
        for name, raw in raw_files.items():
            self.put(slug + "." + ("mcp.jsonl" if name == "wire.jsonl" else name), raw)
        self.put(
            slug + ".capture.json",
            {
                "schema": "jopedime.mcp-stdio-capture/v1",
                "complete": True,
                "error": None,
                "executable_sha256": self.ctx.artifact_sha256,
                "producer_sha256": common.sha256(a5.PREP / "capture_stdio.py"),
                "verifier_sha256": common.sha256(a5.PREP / "verify_capture.py"),
                "server_exit_code": 0,
                "relay_exit_code": 0,
                "files": {
                    name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                    for name, raw in raw_files.items()
                },
            },
        )

    def run_child(self, args, **kwargs):
        self.calls.append(args)
        if args[:2] == ["git", "rev-parse"]:
            output = self.source
        elif args[:2] == ["git", "status"]:
            output = self.dirty
        elif "inspect" in args:
            output = json.dumps(
                {
                    "manifest": {
                        "profile": "full",
                        "scope": "public",
                        "source": {"sha40": self.ctx.source_sha},
                    }
                }
            )
        elif "verify" in args:
            self.after_verify()
            output = json.dumps({"schema": "jope.appliance-verify/v1", "ok": True})
        else:
            self.fail("A5 must not launch a named client, compiler or engine")
        return subprocess.CompletedProcess(args, 0, output, "")

    def invoke(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return a5.qualify(
                self.ctx, self.manifest_path, self.digest, run=self.run_child
            )

    def test_three_complete_pinned_clients_emit_three_counts_without_launches(self):
        self.assertEqual(self.invoke(), 0)
        result = json.loads(self.ctx.result.read_text())
        self.assertEqual(
            result["counts"], {k: 3 * v for k, v in a4.SINGLE_PASS.items()}
        )
        self.assertEqual(result["artifact_sha256"], self.ctx.artifact_sha256)
        self.assertEqual(result["source_sha"], self.ctx.source_sha)
        self.assertEqual(result["manifest_sha256"], self.digest)
        self.assertEqual(set(result["clients"]), set(a5.CLIENTS))

    def test_untrusted_digest_stale_source_or_other_candidate_cannot_pass(self):
        self.digest = "0" * 64
        with self.assertRaisesRegex(common.ProviderError, "manifest SHA"):
            self.invoke()
        self.manifest["source"] = "b" * 40
        self.freeze()
        with self.assertRaisesRegex(common.ProviderError, "source is stale"):
            self.invoke()
        self.manifest["source"] = self.ctx.source_sha
        self.change("source-and-input.json", lambda x: x.update(binary_sha256="e" * 64))
        with self.assertRaisesRegex(common.ProviderError, "exact production"):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_unhashed_required_sidecar_cannot_supply_identity(self):
        del self.manifest["files"]["source-and-input.json"]
        self.freeze()
        with self.assertRaisesRegex(common.ProviderError, "hash-pinned"):
            self.invoke()

    def test_missing_symlinked_and_tampered_files_refuse(self):
        path = self.packet / "claude-code.server.raw"
        raw = path.read_bytes()
        path.write_bytes(raw + b"changed")
        with self.assertRaises(common.ProviderError):
            self.invoke()
        path.unlink()
        original = self.root / "original"
        original.write_bytes(raw)
        path.symlink_to(original)
        with self.assertRaises(common.ProviderError):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_partial_path_cannot_pass_from_status_label_or_retry_only(self):
        frames = [
            json.loads(x)
            for x in (self.packet / "claude-code.mcp.jsonl").read_text().splitlines()
        ]
        ids = {
            f["message"]["id"]
            for f in frames
            if f["message"].get("params", {}).get("name")
            in ("model_export", "model_predict")
        }
        self.save_wire(
            "claude-code", [f for f in frames if f["message"]["id"] not in ids]
        )
        self.manifest["a5"]["status"] = "PASS"
        self.freeze()
        with self.assertRaisesRegex(common.ProviderError, "full golden path"):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_incomplete_transport_or_client_failure_refuses(self):
        self.change("claude-code.capture.json", lambda x: x.update(complete=False))
        with self.assertRaises(common.ProviderError):
            self.invoke()
        self.change("claude-code.capture.json", lambda x: x.update(complete=True))
        self.change("claude-code.execution.json", lambda x: x.update(engine_exit=1))
        with self.assertRaises(a5.capture.CaptureError):
            self.invoke()

    def test_client_version_source_and_launcher_sha_are_bound(self):
        for field, value in [
            ("source", "b" * 40),
            ("client_version", "changed"),
            ("client_launcher_sha256", "e" * 64),
        ]:
            path = "claude-code.execution.json"
            original = (self.packet / path).read_bytes()
            self.change(path, lambda x: x.update({field: value}))
            with self.subTest(field=field), self.assertRaises(common.ProviderError):
                self.invoke()
            self.put(path, original)
            self.freeze()

    def test_raw_streams_cannot_disagree_with_wire_frames(self):
        self.put("claude-code.server.raw", b"{}\n")
        self.freeze()
        with self.assertRaisesRegex(common.ProviderError, "raw capture hash"):
            self.invoke()

    def test_candidate_or_source_drift_never_writes_a_result(self):
        original = self.artifact.read_bytes()
        for kind in ["candidate", "source"]:
            self.after_verify = (
                (lambda: self.artifact.write_bytes(b"changed"))
                if kind == "candidate"
                else (lambda: setattr(self, "source", "b" * 40))
            )
            with self.subTest(kind=kind), self.assertRaises(common.ProviderError):
                self.invoke()
            self.assertFalse(self.ctx.result.exists())
            self.artifact.write_bytes(original)

    def test_broken_retry_and_missing_ordered_inventory_refuse(self):
        original = (self.packet / "claude-code.mcp.jsonl").read_text()
        for mode in ["changed-key", "missing-middle-inventory", "paginated"]:
            frames = [json.loads(line) for line in original.splitlines()]
            starts = [
                f
                for f in frames
                if f["direction"] == "client_to_server"
                and f["message"].get("params", {}).get("name") == "run_start"
                and f["message"]["params"]["arguments"].get("dry_run") is False
            ]
            if mode == "changed-key":
                starts[-1]["message"]["params"]["arguments"]["idempotency_key"] = (
                    "different"
                )
            else:
                between = [
                    f
                    for f in frames
                    if f["direction"] == "client_to_server"
                    and f["message"].get("params", {}).get("name") == "run_status"
                    and starts[0]["message"]["id"]
                    < f["message"]["id"]
                    < starts[1]["message"]["id"]
                ][0]
                rpc = between["message"]["id"]
                if mode == "missing-middle-inventory":
                    frames = [f for f in frames if f["message"]["id"] != rpc]
                else:
                    for f in frames:
                        if (
                            f["direction"] == "server_to_client"
                            and f["message"]["id"] == rpc
                        ):
                            f["message"]["result"]["structuredContent"][
                                "next_after"
                            ] = "more"
            self.save_wire("claude-code", frames)
            self.freeze()
            with self.subTest(mode=mode), self.assertRaises(a5.capture.CaptureError):
                self.invoke()
            self.assertFalse(self.ctx.result.exists())

    def test_one_capture_or_client_binary_cannot_count_for_three_clients(self):
        first = self.manifest["clients"]["claude_code"]["executable_sha256"]
        self.manifest["clients"]["gemini_cli"]["executable_sha256"] = first
        self.change(
            "gemini-cli.execution.json",
            lambda x: x.update(client_launcher_sha256=first),
        )
        with self.assertRaisesRegex(common.ProviderError, "cannot reuse"):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_existing_or_in_checkout_results_never_overwrite(self):
        self.ctx.result.write_bytes(b"prior")
        with self.assertRaises(common.ProviderError):
            self.invoke()
        self.assertEqual(self.ctx.result.read_bytes(), b"prior")
        self.ctx = replace(self.ctx, result=self.repo / "result.json")
        with self.assertRaises(common.ProviderError):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
