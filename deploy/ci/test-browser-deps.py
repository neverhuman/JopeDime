#!/usr/bin/env python3
"""Actual deb extraction/cache integrity controls; no network, root or host install."""
import contextlib
import hashlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile
import unittest
from unittest import mock
import urllib.error

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

    def test_explicit_generic_lock_preserves_default_and_verifies_payload(self):
        explicit = self.root / "publication.json"
        data = json.loads(MODULE.LOCK.read_text())
        data["schema"] = "jope.ubuntu24.04-dependencies/v1"
        explicit.write_text(json.dumps(data))
        with contextlib.redirect_stdout(io.StringIO()):
            MODULE.prepare(self.cache, self.root / "generic.json", explicit)
        proof = json.loads((self.root / "generic.json").read_text())
        self.assertEqual(proof["lock_sha256"], MODULE.digest(explicit))
        self.assertEqual((Path(proof["runtime"]) / self.payload).read_bytes(), b"bounded fixture library bytes")
        self.assertEqual(json.loads(MODULE.LOCK.read_text())["schema"], "jope.chromium-ubuntu24.04-dependencies/v1")
        for field, value in (("schema", "unknown"), ("architecture", "arm64")):
            bad = {**data, field: value}
            explicit.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "unsupported"):
                MODULE.prepare(self.cache, self.root / "bad-generic.json", explicit)
        data["packages"][0]["url"] = "https://example.invalid/ubuntu/pool/fixture.deb"
        explicit.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "official Ubuntu"):
            MODULE.prepare(self.cache, self.root / "bad-mirror.json", explicit)

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

    def fetch(self, outcomes):
        """Prepare with an empty package cache and a scripted mirror; returns the call count."""
        self.cached.unlink()
        calls = []

        def mirror(url, timeout):
            calls.append(url)
            outcome = outcomes[min(len(calls), len(outcomes)) - 1]
            if isinstance(outcome, BaseException):
                raise outcome
            return contextlib.nullcontext(io.BytesIO(outcome))

        with mock.patch.object(MODULE.urllib.request, "urlopen", mirror), \
                mock.patch.object(MODULE, "BACKOFF_SECONDS", 0), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                self.prepare()
            finally:
                self.calls = len(calls)

    def test_dropped_connections_are_retried(self):
        archive = self.archive.read_bytes()
        eof = urllib.error.URLError(ssl.SSLEOFError(8, "EOF occurred in violation of protocol"))
        self.fetch([http.client.RemoteDisconnected("closed"), eof, archive])
        self.assertEqual(3, self.calls)
        self.assertEqual(MODULE.digest(self.cached), MODULE.digest(self.archive))

    def test_persistent_drops_fail_after_bounded_attempts(self):
        with self.assertRaises(http.client.RemoteDisconnected):
            self.fetch([http.client.RemoteDisconnected("closed")])
        self.assertEqual(MODULE.ATTEMPTS, self.calls)
        self.assertFalse(self.cached.exists())

    def test_content_mismatch_and_client_errors_are_not_retried(self):
        archive = self.archive.read_bytes()
        for outcome, error in ((archive + b"x", ValueError), (archive[:-1] + b"x", ValueError),
                               (urllib.error.HTTPError("u", 404, "missing", {}, None), urllib.error.HTTPError)):
            with self.subTest(outcome=type(outcome).__name__), self.assertRaises(error):
                shutil.copyfile(self.archive, self.cached)
                self.fetch([outcome, archive])
            self.assertEqual(1, self.calls)
            self.assertEqual([], [p for p in self.cached.parent.iterdir() if p != self.cached])

    def test_member_paths_cannot_escape(self):
        for path in ("/absolute", "../outside", "usr/../../outside"):
            with self.assertRaisesRegex(ValueError, "unsafe package member"):
                MODULE.safe_name(path)


if __name__ == "__main__":
    unittest.main()
