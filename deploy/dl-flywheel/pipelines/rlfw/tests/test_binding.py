import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from rlfw.policy_binding import verify_manifest


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
