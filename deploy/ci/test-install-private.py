#!/usr/bin/env python3
"""Synthetic installer custody checks; no real jope build, fit or download."""
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "install-private.py"
spec = importlib.util.spec_from_file_location("jope_private_install", path)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class PrivateInstallerTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / "target/private-install-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)
        self.source = self.root / "synthetic-binary"
        self.source.write_bytes(b"explicitly synthetic binary fixture; no executable code")
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.source_sha = "a" * 40
        self.prefix = self.root / "private"

    def tearDown(self):
        self.temporary.cleanup()

    def install(self):
        with patch.object(installer, "inspect_binary", return_value={"fixture": "synthetic"}), \
             patch.object(installer, "probe_binary", return_value={"fixture": "synthetic"}):
            return installer.install_binary(self.source, self.sha, self.source_sha, self.prefix)

    def test_verified_bytes_private_receipt_and_repeat_install(self):
        first = self.install()
        destination = Path(first["binary"])
        self.assertEqual(destination.read_bytes(), self.source.read_bytes())
        self.assertEqual(destination.stat().st_mode & 0o777, 0o700)
        receipt = Path(first["receipt"])
        self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(receipt.read_text())["binary_sha256"], self.sha)
        self.assertEqual(Path(first["command"]).resolve(), destination)
        original = receipt.read_bytes()
        repeated = self.install()
        self.assertTrue(repeated["already_present"])
        self.assertEqual(receipt.read_bytes(), original)
        self.assertNotEqual(first["receipt"], repeated["receipt"])
        self.assertEqual(repeated["fits"], 0)
        self.assertFalse(repeated["warm_demo_measured"])

    def test_bad_checksum_never_executes_or_activates(self):
        with patch.object(installer, "probe_binary") as probe:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                installer.install_binary(self.source, "0" * 64, self.source_sha, self.prefix)
            probe.assert_not_called()
        self.assertFalse(os.path.lexists(self.prefix / "bin/jope"))
        self.assertEqual(self.source.read_bytes(), b"explicitly synthetic binary fixture; no executable code")

    def test_tampered_installed_bytes_and_changed_source_pin_are_refused(self):
        first = self.install()
        with self.assertRaisesRegex(ValueError, "provenance"):
            installer.install_binary(self.source, self.sha, "b" * 40, self.prefix)
        Path(first["binary"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.install()

    def test_foreign_activation_is_not_replaced(self):
        installer.private_directory(self.prefix)
        binaries = installer.private_directory(self.prefix / "bin")
        current = binaries / "jope"
        current.write_bytes(b"unmanaged")
        with self.assertRaisesRegex(ValueError, "unmanaged"):
            self.install()
        self.assertEqual(current.read_bytes(), b"unmanaged")

    def test_new_activation_preserves_the_previous_version_and_receipt(self):
        first = self.install()
        first_bytes = Path(first["binary"]).read_bytes()
        self.source.write_bytes(b"a second explicitly synthetic binary fixture")
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        second = self.install()
        self.assertEqual(Path(first["binary"]).read_bytes(), first_bytes)
        self.assertTrue(Path(first["receipt"]).exists())
        self.assertEqual(Path(second["command"]).resolve(), Path(second["binary"]))
        self.assertIsNotNone(second["prior_activation"])

    def test_size_limit_and_architecture_fail_before_probe(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            installer.copy_pinned_binary(self.source, self.root / "partial", self.sha, max_bytes=3)
        self.assertFalse((self.root / "partial").exists())
        with self.assertRaisesRegex(ValueError, "ELF64"):
            installer.inspect_binary(self.source)

    def test_compiled_asm_probe_and_commands_are_required(self):
        values = ["jope fixture", json.dumps({"schema": "jope.doctor/v1"}),
                  json.dumps({"schema": "jope.optimize-capabilities/v1", "compiled": True,
                              "fits": 0, "qualified": False})] + ["help fixture"] * 4
        with patch.object(installer.subprocess, "check_output", side_effect=values) as output:
            probe = installer.probe_binary(self.source)
        self.assertEqual(probe["required_feature_policy"]["cargo_features"], ["asm"])
        self.assertEqual(set(probe["command_help_sha256"]), {"benchmark", "demo", "optimize", "train"})
        self.assertEqual(output.call_args_list[2].args[0][1:], ["optimize", "capabilities"])
        self.assertFalse(probe["asm_capabilities"]["qualified"])

    def test_uncompiled_asm_is_refused_before_activation_and_on_reinstall(self):
        responses = ["jope fixture", json.dumps({"schema": "jope.doctor/v1"}),
                     json.dumps({"schema": "jope.optimize-capabilities/v1", "compiled": False,
                                 "fits": 0, "qualified": False})]
        with patch.object(installer, "inspect_binary", return_value={}), \
             patch.object(installer.subprocess, "check_output", side_effect=responses):
            with self.assertRaisesRegex(ValueError, "compiled ASM"):
                installer.install_binary(self.source, self.sha, self.source_sha, self.prefix)
        self.assertFalse(os.path.lexists(self.prefix / "bin/jope"))
        self.install()
        activation = os.readlink(self.prefix / "bin/jope")
        with patch.object(installer.subprocess, "check_output", side_effect=responses):
            with self.assertRaisesRegex(ValueError, "compiled ASM"):
                installer.install_binary(self.source, self.sha, self.source_sha, self.prefix)
        self.assertEqual(os.readlink(self.prefix / "bin/jope"), activation)

    def mocked_build(self, exit_code):
        checkout = self.root / "synthetic-checkout"
        checkout.mkdir()
        output = self.root / "build"
        def read(command, **_kwargs):
            if command[:3] == ("git","rev-parse","--show-toplevel"):
                return str(checkout)
            if command[:3] == ("git","rev-parse","HEAD"):
                return self.source_sha
            if command[:2] == ("git","status"):
                return ""
            if command[:2] == ("cargo","tree"):
                self.assertIn("asm",command)
                return "jope-cli v0.1.0\nfeat-math v0.1.0"
            if command[:2] == ("cargo","metadata"):
                return '{"packages":[]}'
            if command[:1] == ("rustc",):
                return "rustc explicit synthetic fixture"
            return str(self.source)  # Mocked Cargo executable selection.
        def build(command, **kwargs):
            kwargs['stdout'].write('{"reason":"build-finished","success":' + ('true' if exit_code == 0 else 'false') + '}\n')
            kwargs['stderr'].write('explicit synthetic compiler diagnostic\n')
            return subprocess.CompletedProcess(command,exit_code)
        with patch.object(installer.subprocess,"check_output",side_effect=read), \
             patch.object(installer.subprocess,"run",side_effect=build), \
             patch.object(installer,"inspect_binary",return_value={"fixture":"synthetic"}), \
             patch.object(installer,"probe_binary",return_value={"fixture":"synthetic"}):
            if exit_code:
                with self.assertRaisesRegex(ValueError,"failure receipt retained"):
                    installer.build_source(checkout,self.source_sha,output)
                return output,json.loads((output/'build-failure.json').read_text())
            return output,installer.build_source(checkout,self.source_sha,output)

    def test_cold_source_failure_keeps_actual_diagnostics_and_graph_identity(self):
        output,receipt=self.mocked_build(101)
        self.assertFalse(receipt['ok'])
        self.assertEqual(receipt['exit_code'],101)
        self.assertEqual(receipt['fits'],0)
        for key in ['dependency_inventory','cargo_messages','diagnostics']:
            artifact=receipt[key]
            self.assertEqual(installer.checksum(artifact['path']),artifact['sha256'])
        self.assertFalse((output/'jope').exists())

    def test_cold_source_success_pins_retained_graph_messages_and_binary(self):
        output,receipt=self.mocked_build(0)
        self.assertTrue(receipt['ok'])
        self.assertEqual(receipt['binary_sha256'],self.sha)
        self.assertFalse(receipt['warm_demo_measured'])
        self.assertEqual(receipt['cargo_features'],['asm'])
        for key in ['dependency_inventory','cargo_messages','cargo_metadata','diagnostics']:
            artifact=receipt[key]
            self.assertEqual(installer.checksum(artifact['path']),artifact['sha256'])
        inventory=json.loads(Path(receipt['dependency_inventory']['path']).read_text())
        self.assertEqual(hashlib.sha256(inventory['graph'].encode()).hexdigest(),receipt['dependency_graph_sha256'])

    def test_source_build_graph_excludes_engine_and_torch(self):
        self.assertEqual(installer.validate_cli_graph("jope-cli v0.1.0\nfeat-math v0.1.0\nhyperion v0.1.0"),
                         ["feat-math", "hyperion", "jope-cli"])
        for forbidden in ("jope-parallel", "jope-engine", "tch", "torch-sys", "cudarc", "bindgen_cuda", "cudnn-sys", "candle-kernels"):
            with self.subTest(package=forbidden), self.assertRaisesRegex(ValueError, "unexpectedly"):
                installer.validate_cli_graph("jope-cli v0.1.0\n" + forbidden + " v0.1.0")


if __name__ == "__main__":
    unittest.main()
