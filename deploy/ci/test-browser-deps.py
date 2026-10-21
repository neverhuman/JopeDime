#!/usr/bin/env python3
"""Actual deb extraction/cache integrity controls; no network, root or host install."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("browser_deps", Path(__file__).with_name("browser-deps.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BrowserDependencies(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "target/tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="browser-deps-controls-", dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        package = self.root / "package"
        control = package / "DEBIAN/control"
        control.parent.mkdir(parents=True)
        control.write_text("Package: browser-controls\nVersion: 1.0\nArchitecture: amd64\nMaintainer: Fixture <fixture@example.invalid>\nDescription: offline package integrity control\n")
        self.payload = "usr/lib/control-library"
        data = package / self.payload
        data.parent.mkdir(parents=True)
        data.write_bytes(b"bounded fixture library bytes")
        (data.parent / "control-link").symlink_to("control-library")
        self.archive = self.root / "fixture.deb"
        subprocess.run(["dpkg-deb", "--build", "--root-owner-group", str(package), str(self.archive)], check=True, stdout=subprocess.DEVNULL)
        sha = MODULE.digest(self.archive)
        self.cache = self.root / "cache"
        archives = self.cache / "archives"
        archives.mkdir(parents=True)
        self.cached = archives / (sha + ".deb")
        shutil.copyfile(self.archive, self.cached)
        lock = self.root / "lock.json"
        lock.write_text(json.dumps({"schema": "jope.chromium-ubuntu24.04-dependencies/v1", "architecture": "amd64", "packages": [{"url": "https://us.archive.ubuntu.com/ubuntu/pool/main/f/fixture/fixture.deb", "sha256": sha, "size": self.archive.stat().st_size}]}))
        previous = MODULE.LOCK
        MODULE.LOCK = lock
        self.addCleanup(setattr, MODULE, "LOCK", previous)

    def prepare(self, name="proof.json"):
        with contextlib.redirect_stdout(io.StringIO()):
            MODULE.prepare(self.cache, self.root / name)
        proof = json.loads((self.root / name).read_text())
        return Path(proof["runtime"])

    def test_verified_extraction_and_reuse(self):
        runtime = self.prepare()
        self.assertEqual((runtime / self.payload).read_bytes(), b"bounded fixture library bytes")
        self.assertEqual(runtime, self.prepare("proof-reused.json"))

    def test_corrupt_download_cache_is_refused(self):
        self.cached.write_bytes(b"corrupt archive")
        with self.assertRaisesRegex(ValueError, "cached package differs"):
            self.prepare()

    def test_changed_runtime_bytes_are_refused(self):
        runtime = self.prepare()
        (runtime / self.payload).write_bytes(b"modified library")
        with self.assertRaisesRegex(ValueError, "differs from verified"):
            self.prepare("bad.json")

    def test_extra_runtime_file_is_refused(self):
        runtime = self.prepare()
        (runtime / "extra").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "differs from verified"):
            self.prepare("bad.json")

    def test_changed_runtime_symlink_is_refused(self):
        runtime = self.prepare()
        link = runtime / "usr/lib/control-link"
        link.unlink()
        link.symlink_to("different-target")
        with self.assertRaisesRegex(ValueError, "differs from verified"):
            self.prepare("bad.json")

    def test_member_paths_cannot_escape(self):
        for path in ("/absolute", "../outside", "usr/../../outside"):
            with self.assertRaisesRegex(ValueError, "unsafe package member"):
                MODULE.safe_name(path)


if __name__ == "__main__":
    unittest.main()
