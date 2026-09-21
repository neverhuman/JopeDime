import tempfile
import unittest
from pathlib import Path

import freeze_registry_release as release


class SourceFreezeTests(unittest.TestCase):
    def test_reproducible_untracked_source_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            path = root / "src" / "untracked.rs"
            path.write_text("complete semantic dependency")
            first = release.freeze(root, root / "one", roots=("src",))
            second = release.freeze(root, root / "two", roots=("src",))
            self.assertEqual(first["archive_sha256"], second["archive_sha256"])
            path.write_text("working edits do not mutate snapshot")
            release.verify(root / "one" / "source", first)
            frozen = root / "one/source/src/untracked.rs"
            frozen.chmod(0o644)
            frozen.write_text("tampered")
            with self.assertRaisesRegex(RuntimeError, "frozen source changed"):
                release.verify(root / "one" / "source", first)

    def test_missing_dependency_and_symlink_are_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "dependency absent"):
                release.freeze(root, root / "one", roots=("missing",))
            (root / "src").mkdir()
            (root / "outside").write_text("not in closure")
            (root / "src/link").symlink_to(root / "outside")
            with self.assertRaisesRegex(RuntimeError, "symlink resolution"):
                release.freeze(root, root / "two", roots=("src",))
