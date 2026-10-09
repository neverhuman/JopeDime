import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from rlfw.policy_binding import verify_manifest, verify_policy_origins
from rlfw.run_rlfw_offline import source_identity
from rlfw.trie_logprob import TrieLogProb
from rlfw.verify_checkpoint import compare_readouts, policy_stop_active


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "model.py"
        self.source.write_text("# inert fixture\n")
        self.source.chmod(0o444)
        raw = f"{hashlib.sha256(self.source.read_bytes()).hexdigest()}  model.py\n"
        self.manifest = self.root / "MANIFEST.sha256"
        self.manifest.write_text(raw)
        self.pin = hashlib.sha256(raw.encode()).hexdigest()

    def test_byte_pin_and_readonly_check(self):
        self.assertEqual(verify_manifest(self.root, expected_sha256=self.pin), self.pin)
        self.source.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "read-only"):
            verify_manifest(self.root)
        self.source.write_text("# changed\n")
        self.source.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "bytes"):
            verify_manifest(self.root)

    def test_manifest_pin_unlisted_code_and_duplicates_refused(self):
        with self.assertRaises(ValueError):
            verify_manifest(self.root, expected_sha256="b" * 64)
        self.manifest.write_bytes(self.manifest.read_bytes() * 2)
        with self.assertRaises(ValueError):
            verify_manifest(self.root)
        self.manifest.write_bytes(
            self.manifest.read_bytes()[: len(self.manifest.read_bytes()) // 2]
        )
        (self.root / "extra.py").write_text("# unpinned\n")
        with self.assertRaises(ValueError):
            verify_manifest(self.root)

    def test_symlink_and_escaping_path_refused(self):
        self.source.chmod(0o644)
        self.source.unlink()
        target = self.root / "target.txt"
        target.write_text("# inert fixture\n")
        target.chmod(0o444)
        self.source.symlink_to(target)
        with self.assertRaises(ValueError):
            verify_manifest(self.root)
        self.manifest.write_text("a" * 64 + "  ../escape.py\n")
        with self.assertRaises(ValueError):
            verify_manifest(self.root)

    def test_import_does_not_load_torch_or_deployed_modules(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("continuous", sys.modules)
        self.assertNotIn("neural_decode", sys.modules)

    def test_directory_symlink_and_unlisted_binary_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "unpinned.py").write_text("# not declared\n")
        (self.root / "consumer").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "directory links"):
            verify_manifest(self.root, expected_sha256=self.pin)
        (self.root / "consumer").unlink()
        with self.assertRaisesRegex(ValueError, "unmanifested"):
            verify_manifest(self.root)

    def test_imported_module_origin_and_package_path_refused(self):
        module = ModuleType("neural_decode")
        module.__file__ = str(self.root.parent / "outside.py")
        with patch.dict(sys.modules, {"neural_decode": module}):
            with self.assertRaisesRegex(ValueError, "mixed policy"):
                verify_policy_origins(self.root)
            module.__file__ = str(self.source)
            module.__path__ = [str(self.root.parent)]
            with self.assertRaisesRegex(ValueError, "search path"):
                verify_policy_origins(self.root)

    def test_source_head_is_distinct_from_synthetic_checkout(self):
        with patch.dict(
            os.environ, {"JOPE_SOURCE_SHA": "a" * 40, "GITHUB_SHA": "b" * 40}
        ):
            self.assertEqual(source_identity(), ("a" * 40, "b" * 40))

    def test_device_parity_reports_likelihood_and_each_head_drift(self):
        class Readout:
            def __init__(self, delta=0.0):
                self.delta = delta

            def logprobs_many(self, contexts, rows):
                return [
                    TrieLogProb(-0.5 - self.delta, (-0.5 - self.delta, 0.0))
                    for _ in rows
                ]

            def score_heads(self, contexts, rows):
                return [
                    {
                        "q": 0.1 + self.delta,
                        "p_complete": 0.7 + self.delta,
                        "runtime_seconds": 1.0 + self.delta,
                    }
                    for _ in rows
                ]

        zero = compare_readouts(Readout(), Readout(), [None], [(1, 3, 2)])
        self.assertEqual(
            set(zero),
            {"total_logprob", "token_logprob", "q", "p_complete", "runtime_log1p"},
        )
        self.assertTrue(all(value == 0.0 for value in zero.values()))
        changed = compare_readouts(Readout(), Readout(1e-4), [None], [(1, 3, 2)])
        self.assertTrue(all(value > 1e-6 for value in changed.values()))

    def test_explicit_evaluator_stop_exception_is_exact_and_cpu_only(self):
        stop = self.root / "STOP.json"
        stop.write_text(
            json.dumps(
                {
                    "schema": "rlfw-campaign-stop/v1",
                    "reason": "D1_native_Rust_parity_smoke_mismatch",
                }
            )
        )
        pin = hashlib.sha256(stop.read_bytes()).hexdigest()
        self.assertTrue(policy_stop_active(stop, device="cpu"))
        self.assertFalse(
            policy_stop_active(stop, device="cpu", evaluator_stop_sha256=pin)
        )
        for device, comparison in (("cuda", None), ("cpu", "cuda")):
            self.assertTrue(
                policy_stop_active(
                    stop,
                    device=device,
                    compare_device=comparison,
                    evaluator_stop_sha256=pin,
                )
            )
        self.assertTrue(
            policy_stop_active(stop, device="cpu", evaluator_stop_sha256="b" * 64)
        )
        stop.write_text(
            json.dumps({"schema": "rlfw-campaign-stop/v1", "reason": "memory_guard"})
        )
        changed = hashlib.sha256(stop.read_bytes()).hexdigest()
        self.assertTrue(
            policy_stop_active(stop, device="cpu", evaluator_stop_sha256=changed)
        )
        stop.write_bytes(b"not-json")
        changed = hashlib.sha256(stop.read_bytes()).hexdigest()
        self.assertTrue(
            policy_stop_active(stop, device="cpu", evaluator_stop_sha256=changed)
        )
