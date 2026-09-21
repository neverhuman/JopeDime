import tempfile
import unittest
from pathlib import Path

from kernel_release import (
    HYPERION,
    KERNEL,
    dependency_coverage,
    manifest,
    stage,
    sync_mirror,
    verify_mirror,
    verify_release,
)


class KernelReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "parallel"
        self.root.mkdir()
        (self.base / "Cargo.toml").write_text("[workspace]\n")
        for name in ("Cargo.toml", "Cargo.lock", "build.rs"):
            (self.root / name).write_text(name)
        for name in ("src", "native", "tools", "tests", "examples", "systemd", "assets/grammar"):
            p = self.root / name
            p.mkdir(parents=True)
            (p / "input").write_text(name)
        canonical = self.base / "vendor/feat-math"
        for name in (
            "Cargo.toml",
            "build.rs",
            "src/lib.rs",
            "cpp/runtime.cpp",
            "assets/champions/example.dsl",
            "tests/fixtures/data.json",
        ):
            p = canonical / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(name)
        (self.root / KERNEL).mkdir(parents=True)
        sync_mirror(self.root)

    def tearDown(self):
        for p in self.base.rglob("*"):
            if p.is_dir():
                p.chmod(0o755)
        self.tmp.cleanup()

    def test_complete_snapshot_and_isolated_sources(self):
        release = stage(self.root, self.base / "releases")
        self.assertEqual(stage(self.root, self.base / "releases"), release)
        files = verify_release(release)["files"]
        for name in (
            "build.rs",
            "cpp/runtime.cpp",
            "assets/champions/example.dsl",
            "tests/fixtures/data.json",
        ):
            self.assertIn(str(KERNEL / name), files)
        (self.root / "src/input").write_text("changed")
        self.assertEqual(verify_release(release)["files"], files)
        self.assertNotEqual(stage(self.root, self.base / "releases"), release)

    def test_tampering_and_extra_files_rejected(self):
        release = stage(self.root, self.base / "releases")
        p = release / "source/src/input"
        p.chmod(0o644)
        p.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "integrity"):
            verify_release(release)
        (self.root / KERNEL / "extra").write_text("unreviewed")
        with self.assertRaises(ValueError):
            sync_mirror(self.root)

    def test_drift_and_symlink_rejected(self):
        (self.root / KERNEL / "src/lib.rs").write_text("drift")
        with self.assertRaises(ValueError):
            verify_mirror(self.root)
        sync_mirror(self.root)
        (self.root / "src/link").symlink_to(self.root / "Cargo.toml")
        with self.assertRaisesRegex(ValueError, "symlink"):
            manifest(self.root)

    def test_optional_helper_dependencies_are_explicit_and_packaged(self):
        coverage = dependency_coverage(self.root)
        self.assertEqual(coverage["hyperion_host"]["source_status"], "missing")
        for name in coverage["hyperion_host"]["missing_files"]:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("helper source")
        coverage = dependency_coverage(self.root)
        self.assertEqual(coverage["hyperion_host"]["source_status"], "included")
        release = stage(self.root, self.base / "releases")
        receipt = verify_release(release)
        self.assertEqual(receipt["dependency_coverage"], coverage)
        self.assertIn(str(HYPERION / "crates/feat-cli/src/bin/hyperion_vs.rs"), receipt["files"])


if __name__ == "__main__":
    unittest.main()
