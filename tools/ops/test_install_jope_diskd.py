"""Regression tests for the disk daemon installer’s no-surprise gates."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

INSTALLER = Path(__file__).with_name("install_jope_diskd.sh")


class DiskDaemonInstallerTests(unittest.TestCase):
    def temp_home(self):
        return tempfile.TemporaryDirectory(prefix="jope-diskd-installer-", dir=Path.home())

    def run_installer(self, home: Path, *args: str, installer: Path = INSTALLER) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["HOME"] = str(home)
        return subprocess.run(
            ["bash", str(installer), *args],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def assert_no_install_artifacts(self, home: Path) -> None:
        for relative in (
            ".local/bin/jope-diskd",
            ".local/state/jope-diskd",
            ".config/jope-diskd/policy.yaml",
            ".config/systemd/user/jope-diskd.service",
            ".config/systemd/user/jope-diskd.timer",
        ):
            self.assertFalse((home / relative).exists(), relative)

    def make_isolated_checkout(self, root: Path) -> Path:
        checkout = root / "checkout"
        ops = checkout / "tools/ops"
        ops.mkdir(parents=True)
        installer = ops / "install_jope_diskd.sh"
        shutil.copy2(INSTALLER, installer)
        subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Installer Test"], check=True)
        subprocess.run(["git", "-C", str(checkout), "config", "user.email", "installer-test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(checkout), "add", "tools/ops/install_jope_diskd.sh"], check=True)
        subprocess.run(["git", "-C", str(checkout), "commit", "--quiet", "-m", "test installer fixture"], check=True)
        return installer

    def test_default_preview_prints_paths_without_writing(self) -> None:
        with self.temp_home() as home_dir:
            home = Path(home_dir)
            result = self.run_installer(home)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Dry run only", result.stdout)
            self.assertIn(str(home / ".local/bin/jope-diskd"), result.stdout)
            self.assert_no_install_artifacts(home)

    def test_unknown_argument_refuses_without_writing(self) -> None:
        with self.temp_home() as home_dir:
            home = Path(home_dir)
            result = self.run_installer(home, "--delete")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Usage:", result.stderr)
            self.assert_no_install_artifacts(home)

    def test_enable_refuses_dirty_checkout_before_writing(self) -> None:
        with self.temp_home() as home_dir:
            home = Path(home_dir)
            installer = self.make_isolated_checkout(home)
            (installer.parent / "untracked.txt").write_text("dirty\n", encoding="utf-8")
            result = self.run_installer(home, "--enable", installer=installer)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing enable", result.stderr)
            self.assert_no_install_artifacts(home)

    def test_enable_refuses_checkout_without_merged_main_ref_before_writing(self) -> None:
        with self.temp_home() as home_dir:
            home = Path(home_dir)
            installer = self.make_isolated_checkout(home)
            result = self.run_installer(home, "--enable", installer=installer)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing enable", result.stderr)
            self.assert_no_install_artifacts(home)


if __name__ == "__main__":
    unittest.main()
