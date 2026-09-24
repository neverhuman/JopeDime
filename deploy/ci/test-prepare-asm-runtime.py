#!/usr/bin/env python3
"""Metadata fixtures only: these do not qualify a compiler, worker or model."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("prepare_asm", Path(__file__).parents[1] / "prepare-asm-runtime.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class Preparation(unittest.TestCase):
    def setUp(self):
        parent = Path.home() / ".cache" / "jope-preparation-fixtures"
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        lock = self.root / "cpu.lock"
        lock.touch()
        self.lock_patch = patch.object(prepare, "HOST_LOCK", lock)
        self.lock_patch.start()
        self.addCleanup(self.lock_patch.stop)
        self.source = "a" * 40

    def put(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return prepare.artifact(path)

    def build(self):
        source = self.put("source.json", {"source_sha40": self.source})
        executables = {name: self.put(name, {"fixture": name}) for name in prepare.TARGETS}
        messages = [{"reason": "compiler-artifact", "target": {"name": name}, "executable": ref["path"],
                     "features": ["context-producer"]} for name, ref in executables.items()]
        messages.append({"reason": "build-finished", "success": True})
        path = self.root / "cargo.jsonl"
        path.write_text("\n".join(json.dumps(m) for m in messages))
        receipt = self.put("build.json", {"schema": "jope.asm-cargo-build/v1", "source_sha40": self.source,
                           "source_manifest_sha256": source["sha256"], "exit_code": 0,
                           "started_unix_ms": 1, "finished_unix_ms": 2,
                           "cargo_messages": prepare.artifact(path), "executables": [
                               {"target_name": name, **ref} for name, ref in executables.items()]})
        workers = [{"role": role, "executable": executables[name], "features": ["context-producer"],
                    "build_receipt": receipt} for role, name in
                   (("policy", "jope-policy-producer"), ("context", "jope-context-worker"))]
        inventory = self.put("workers.json", {"schema": "jope.asm-worker-build-inventory/v1",
                             "source_sha40": self.source, "source_manifest": source, "workers": workers})
        summary = {"source_sha40": self.source, "source_manifest": source, "worker_inventory": inventory,
                   "build_receipt": receipt, "executables": executables}
        self.put("summary.json", summary)
        return summary

    def test_valid_build_and_changed_worker_bytes(self):
        summary = self.build()
        self.assertEqual(prepare.worker_build(self.root / "summary.json")[0], summary)
        Path(summary["executables"]["jope-context-worker"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
            prepare.worker_build(self.root / "summary.json")

    def test_worker_roles_source_and_feature_bindings(self):
        summary = self.build()
        original = prepare.read(summary["worker_inventory"]["path"])
        for corruption in ("duplicate", "source", "features"):
            with self.subTest(corruption=corruption):
                inventory = copy.deepcopy(original)
                if corruption == "duplicate":
                    inventory["workers"][1]["role"] = "policy"
                elif corruption == "source":
                    inventory["source_sha40"] = "b" * 40
                else:
                    inventory["workers"][0]["features"] = []
                summary["worker_inventory"] = self.put("workers.json", inventory)
                self.put("summary.json", summary)
                with self.assertRaises(ValueError):
                    prepare.worker_build(self.root / "summary.json")

    def test_build_receipt_and_cargo_output_must_agree(self):
        summary = self.build()
        receipt = prepare.read(summary["build_receipt"]["path"])
        receipt["executables"][0]["sha256"] = "c" * 64
        summary["build_receipt"] = self.put("build.json", receipt)
        self.put("summary.json", summary)
        with self.assertRaises(ValueError):
            prepare.worker_build(self.root / "summary.json")

    def test_output_cannot_escape_custody_or_overwrite(self):
        with self.assertRaisesRegex(ValueError, "outside /tmp"):
            prepare.fresh_output(Path("/tmp/asm-preparation-refusal"))
        path = prepare.fresh_output(self.root / "output")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare.fresh_output(path)
        (self.root / "link").symlink_to(path, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "physical"):
            prepare.fresh_output(self.root / "link" / "child")
        (self.root / ".git").touch()
        with self.assertRaisesRegex(ValueError, "outside a checkout"):
            prepare.fresh_output(self.root / "checkout-output")

    def test_busy_allocation_refuses_before_work(self):
        with prepare.allocation():
            with self.assertRaisesRegex(ValueError, "allocation is busy"):
                with prepare.allocation():
                    self.fail("entered occupied allocation")

    def test_source_guard_rejects_changed_or_uncommitted_source(self):
        with patch.object(prepare, "git", side_effect=[self.source, ""]):
            prepare.source_guard(self.root, self.source)
        with patch.object(prepare, "git", return_value=self.source):
            with self.assertRaisesRegex(ValueError, "HEAD changed"):
                prepare.source_guard(self.root, "b" * 40)
        with patch.object(prepare, "git", side_effect=[self.source, "?? uncommitted"]):
            with self.assertRaisesRegex(ValueError, "committed and clean"):
                prepare.source_guard(self.root, self.source)

    def test_context_preparation_does_not_execute_proof(self):
        summary = self.build()
        for name in ("sources", "encoder", "resources"):
            self.put(name + ".json", {"fixture": name})
        output = self.root / "preparation"
        args = SimpleNamespace(worker_build=self.root / "summary.json", output=output,
                               reference_sources=self.root / "sources.json", encoder_files=self.root / "encoder.json",
                               resources=self.root / "resources.json")
        calls = []

        def native(command, **kwargs):
            calls.append(command)
            self.assertEqual(command[:2], [summary["executables"]["jope-context-proof"]["path"], "prepare"])
            path = output / "plan.json"
            path.write_text('{"fixture":true}')
            return subprocess.CompletedProcess(command, 0, str(path) + " " + prepare.digest(path) + "\n", "")

        with patch.object(prepare.subprocess, "run", side_effect=native):
            result = prepare.prepare_context(args)
        self.assertEqual(len(calls), 1)
        self.assertFalse(result["admitted"])
        self.assertEqual(result["command"][1], "run")
        self.assertFalse(prepare.read(output / "preparation.json")["execution_budget_started"])

    def test_configuration_requires_native_validation(self):
        summary = self.build()
        assets = {key: self.put(key, {"fixture": key}) for key in
                  ("hyperion_weights", "hyperion_manifest", "frozen_transforms")}
        asset_manifest = self.put("assets.json", {"configuration": self.put("encoder-files.json", assets),
                                                 "model_closure": list(assets.values())})
        policy = self.put("policy-preparation.json", {key: self.put(key, {"fixture": key})
                          for key in ("checkpoint_receipt", "policy", "vocabulary")})
        args = SimpleNamespace(worker_build=self.root / "summary.json", policy_preparation=Path(policy["path"]),
                               asset_preparation=Path(asset_manifest["path"]), completed_qualification=None,
                               output=self.root / "configuration")

        def native(command, **kwargs):
            self.assertEqual(command[:3], [summary["executables"]["jope"]["path"], "optimize", "validate-configuration"])
            ref = prepare.artifact(args.output / "configuration.json")
            return subprocess.CompletedProcess(command, 0, json.dumps({
                "schema": "jope.asm-configuration-validation/v1", "ok": True,
                "source_sha40": self.source, "configuration": ref, "context_qualified": False,
            }), "")

        with patch.object(prepare.subprocess, "run", side_effect=native):
            result = prepare.configuration(args)
        self.assertFalse(result["context_qualified"])
        self.assertFalse(result["execution_budget_started"])
        args.output = self.root / "refused-configuration"
        with patch.object(prepare.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "stale proof")):
            with self.assertRaisesRegex(ValueError, "native configuration validation failed"):
                prepare.configuration(args)
        self.assertFalse((args.output / "receipt.json").exists())
        self.assertEqual(prepare.read(args.output / "native-validation.json")["exit_code"], 1)

    def test_failed_build_and_changed_source_retain_failure_evidence(self):
        for changed in (False, True):
            with self.subTest(changed=changed):
                root = self.root / ("changed" if changed else "failed")
                root.mkdir()
                args = SimpleNamespace(source_root=root, expect_source=self.source, jobs=1,
                                       output=self.root / (root.name + "-output"))
                guards = [None, ValueError("source HEAD changed")] if changed else [None]
                with patch.object(prepare, "source_guard", side_effect=guards), \
                        patch.object(prepare, "git", return_value="b" * 40), \
                        patch.object(prepare.subprocess, "check_output", return_value=b""), \
                        patch.object(prepare.subprocess, "run", return_value=subprocess.CompletedProcess([], 0 if changed else 1)), \
                        patch.dict(prepare.os.environ, {"CARGO_TARGET_DIR": ""}):
                    with self.assertRaises(ValueError):
                        prepare.build_workers(args)
                self.assertTrue((args.output / "failed-build.json").is_file())
                self.assertFalse((args.output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
