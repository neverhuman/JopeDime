#!/usr/bin/env python3
"""Actual deb extraction/cache integrity controls; no network, root or host install."""
import contextlib
import hashlib
import http.client
import importlib.util
import itertools
import io
import json
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
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

    def test_generic_font_fallbacks_use_only_the_verified_runtime(self):
        runtime = self.root / "font-runtime"
        fonts = runtime / "etc/fonts"
        dropins = fonts / "conf.d"
        dropins.mkdir(parents=True)
        (fonts / "fonts.conf").write_text('<fontconfig><dir>/usr/share/fonts</dir><include>conf.d</include><cachedir>/var/cache/fontconfig</cachedir></fontconfig>')
        (dropins / "45-fonts.conf").write_text('<fontconfig><alias><family>Consolas</family><default><family>monospace</family></default></alias></fontconfig>')
        (dropins / "60-fonts.conf").write_text('<fontconfig><alias><family>sans-serif</family><prefer><family>Fixture Sans</family></prefer></alias></fontconfig>')
        (dropins / "50-user.conf").write_text('<fontconfig><include>~/.fonts.conf</include></fontconfig>')
        available = runtime / "usr/share/fontconfig/conf.avail"
        available.mkdir(parents=True)
        (available / "70-pinned.conf").write_text('<fontconfig><match target="font"><edit name="hintstyle"><const>hintslight</const></edit></match></fontconfig>')
        (dropins / "70-pinned.conf").symlink_to("/usr/share/fontconfig/conf.avail/70-pinned.conf")
        config = MODULE.hermetic_fontconfig(runtime, self.root / "font-cache")
        self.assertEqual([node.text for node in config.findall("dir")], [str(runtime / "usr/share/fonts")])
        self.assertEqual(list(config.iter("include")), [], "ambient font rules must never enter CI")
        self.assertEqual(config.findtext("alias/default/family"), "monospace")
        self.assertEqual(config.findall("alias")[1].findtext("prefer/family"), "Fixture Sans")
        (dropins / "70-foreign.conf").symlink_to(self.root / "lock.json")
        with self.assertRaisesRegex(ValueError, "escapes the verified runtime"):
            MODULE.hermetic_fontconfig(runtime, self.root / "font-cache")

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
        # A dated official snapshot must admit the same pinned bytes, even on reuse.
        data["packages"][0]["url"] = "https://snapshot.ubuntu.com/ubuntu/20261023T000000Z/pool/main/f/fixture/fixture.deb"
        explicit.write_text(json.dumps(data))
        with contextlib.redirect_stdout(io.StringIO()):
            MODULE.prepare(self.cache, self.root / "snapshot.json", explicit)
        snapshot = json.loads((self.root / "snapshot.json").read_text())
        self.assertEqual(snapshot["lock_sha256"], MODULE.digest(explicit))
        self.assertEqual((Path(snapshot["runtime"]) / self.payload).read_bytes(), b"bounded fixture library bytes")
        for url in (
            "https://example.invalid/ubuntu/pool/fixture.deb",
            "http://snapshot.ubuntu.com/ubuntu/20261023T000000Z/pool/fixture.deb",
            "https://snapshot.ubuntu.com.evil.invalid/ubuntu/20261023T000000Z/pool/fixture.deb",
            "https://snapshot.ubuntu.com/ubuntu/latest/pool/fixture.deb",
            "https://snapshot.ubuntu.com/ubuntu/20261332T000000Z/pool/fixture.deb",
            "https://snapshot.ubuntu.com/ubuntu/20261023T000000Z/pool/%2e%2e/fixture.deb",
            "https://snapshot.ubuntu.com/ubuntu/20261023T000000Z/pool/fixture.deb?mirror=other",
            "https://user@snapshot.ubuntu.com/ubuntu/20261023T000000Z/pool/fixture.deb",
        ):
            data["packages"][0]["url"] = url
            explicit.write_text(json.dumps(data))
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "official Ubuntu"):
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
        self.timeouts = []

        def mirror(url, timeout):
            calls.append(url)
            self.timeouts.append(timeout)
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
                self.urls = list(calls)

    def test_dropped_connections_are_retried(self):
        archive = self.archive.read_bytes()
        eof = urllib.error.URLError(ssl.SSLEOFError(8, "EOF occurred in violation of protocol"))
        self.fetch([http.client.RemoteDisconnected("closed"), eof, archive])
        self.assertEqual(3, self.calls)
        self.assertEqual(MODULE.digest(self.cached), MODULE.digest(self.archive))

    def test_a_dead_mirror_address_cannot_exhaust_the_job_budget(self):
        """Each attempt is bounded, and one timeout per locked package still fits in the 25-minute job."""
        archive = self.archive.read_bytes()
        self.fetch([TimeoutError("timed out"), archive])
        self.assertEqual(2, self.calls)
        self.assertEqual([MODULE.TIMEOUT_SECONDS] * 2, self.timeouts)
        packages = len(json.loads(Path(__file__).with_name("browser-deps.lock.json").read_bytes())["packages"])
        self.assertLess(packages * MODULE.TIMEOUT_SECONDS, 25 * 60 * 3 // 4)

    def test_an_unreachable_address_is_tried_last_for_the_rest_of_the_process(self):
        """One dead mirror address costs one bounded connect, not one wait per package."""
        dead, alive = ("192.0.2.83", 443), ("192.0.2.81", 443)
        attempts = []

        class Socket:
            def __init__(self, *args):
                self.timeouts = []

            def settimeout(self, value):
                self.timeouts.append(value)

            def connect(self, sockaddr):
                attempts.append((sockaddr, self.timeouts[-1]))
                if sockaddr == dead:
                    raise TimeoutError("timed out")

            def close(self):
                pass

        infos = [(2, 1, 6, "", dead), (2, 1, 6, "", alive)]
        with mock.patch.object(MODULE.socket, "getaddrinfo", lambda *args: list(infos)), \
                mock.patch.object(MODULE.socket, "socket", Socket), \
                mock.patch.object(MODULE, "UNREACHABLE", set()):
            first = MODULE.connect_skipping_unreachable(("mirror.test", 443), MODULE.TIMEOUT_SECONDS)
            second = MODULE.connect_skipping_unreachable(("mirror.test", 443), MODULE.TIMEOUT_SECONDS)
        connect = min(MODULE.CONNECT_SECONDS, MODULE.TIMEOUT_SECONDS)
        self.assertEqual([(dead, connect), (alive, connect), (alive, connect)], attempts)
        self.assertEqual([connect, MODULE.TIMEOUT_SECONDS], first.timeouts)
        self.assertEqual([connect, MODULE.TIMEOUT_SECONDS], second.timeouts)

    def test_the_installed_opener_uses_the_skipping_connection(self):
        handler = MODULE.SkippingHTTPSHandler()
        self.assertIs(MODULE.connect_skipping_unreachable, MODULE.SkippingHTTPSConnection("mirror.test")._create_connection)
        self.assertTrue(handler._context.check_hostname)

    def test_a_failed_fetch_moves_to_the_fallback_mirror_with_the_same_pool_path(self):
        archive = self.archive.read_bytes()
        with mock.patch.object(MODULE, "MIRROR_FAILURES", {}):
            self.fetch([TimeoutError("timed out"), archive])
        pool = "pool/main/f/fixture/fixture.deb"
        self.assertEqual([mirror + pool for mirror in MODULE.MIRRORS], self.urls)
        self.assertEqual(MODULE.digest(self.cached), MODULE.digest(self.archive))

    def test_a_fallback_mirror_cannot_change_the_content(self):
        archive = self.archive.read_bytes()
        with mock.patch.object(MODULE, "MIRROR_FAILURES", {}), self.assertRaises(ValueError):
            self.fetch([TimeoutError("timed out"), archive + b"x"])
        self.assertEqual(2, self.calls)
        self.assertFalse(self.cached.exists())

    def test_a_demoted_mirror_is_tried_last(self):
        url = MODULE.MIRRORS[0] + "pool/main/f/fixture/fixture.deb"
        with mock.patch.object(MODULE, "MIRROR_FAILURES", {}) as failures:
            self.assertEqual(MODULE.MIRRORS, tuple(mirror for mirror, _ in MODULE.sources(url)))
            failures[MODULE.MIRRORS[0]] = MODULE.DEMOTE_AFTER
            self.assertEqual(MODULE.MIRRORS[::-1], tuple(mirror for mirror, _ in MODULE.sources(url)))
        self.assertEqual([(None, "https://example.test/x.deb")], MODULE.sources("https://example.test/x.deb"))

    def test_a_slow_fetch_counts_against_its_mirror(self):
        archive = self.archive.read_bytes()
        ticks = itertools.count(0, MODULE.SLOW_SECONDS + 1)
        with mock.patch.object(MODULE, "MIRROR_FAILURES", {}) as failures, \
                mock.patch.object(MODULE.time, "monotonic", lambda: next(ticks)):
            self.fetch([archive])
            self.assertEqual({MODULE.MIRRORS[0]: 1}, failures)

    def test_persistent_drops_fail_after_bounded_attempts(self):
        with self.assertRaises(http.client.RemoteDisconnected):
            self.fetch([http.client.RemoteDisconnected("closed")])
        self.assertEqual(MODULE.ATTEMPTS, self.calls)
        self.assertFalse(self.cached.exists())

    def test_content_mismatch_and_client_errors_are_not_retried(self):
        archive = self.archive.read_bytes()
        for outcome, error in ((archive + b"x", ValueError), (archive[:-1] + bytes([archive[-1] ^ 0xFF]), ValueError),
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
