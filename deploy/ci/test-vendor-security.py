#!/usr/bin/env python3
"""Small offline controls for vendor inventory custody, never the real vendor tree."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("vendor_guard", Path(__file__).with_name("verify-vendor-security.py"))
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


class VendorCustodyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jope-vendor-custody-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "vendor/security"
        self.base.mkdir(parents=True)
        self.data = {"schema": "jope.dependency-source-patch/v1", "input_lock_sha256": "0" * 64,
                     "crates": [], "retained_file_pins": [], "files": 0, "bytes": 0}
        for name, version in sorted(GUARD.EXPECTED_CRATES):
            directory = f"{name}-{version}"
            contents = f'[package]\nname = "{name}"\nversion = "{version}"\n'.encode()
            if name == "ratatui":
                contents += b'\n[dependencies.lru]\nversion = "=0.18.4"\n'
            target = self.base / directory / "Cargo.toml"
            target.parent.mkdir()
            target.write_bytes(contents)
            digest = hashlib.sha256(contents).hexdigest()
            self.data["crates"].append({"name": name, "version": version, "directory": directory,
                "archive_sha256": "1" * 64, "manifest_upstream_sha256": digest,
                "manifest_candidate_sha256": digest, "retained_files": 1, "retained_bytes": len(contents)})
            self.data["retained_file_pins"].append({"path": f"vendor/security/{directory}/Cargo.toml",
                "bytes": len(contents), "upstream_sha256": digest, "candidate_sha256": digest, "changed": False})
            self.data["files"] += 1
            self.data["bytes"] += len(contents)
        self.manifest = self.base / GUARD.PROVENANCE
        self.write()
        self.target = self.root / self.data["retained_file_pins"][0]["path"]

    def write(self):
        self.manifest.write_text(json.dumps(self.data))

    def reject(self, text=None):
        with self.assertRaises((GUARD.CustodyError, OSError, ValueError, TypeError)) as caught:
            GUARD.verify(self.root)
        if text:
            self.assertIn(text, str(caught.exception))

    def test_valid_inventory_is_read_only_and_deterministic(self):
        before = {p.relative_to(self.base).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.base.rglob("*") if p.is_file()}
        first = GUARD.verify(self.root)
        self.assertEqual(first, GUARD.verify(self.root))
        self.assertEqual((first["crates"], first["files"]), (21, 21))
        after = {p.relative_to(self.base).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.base.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_ratatui_lru_floor_refuses_coherent_downgrades_and_overrides(self):
        path = "vendor/security/ratatui-0.29.0/Cargo.toml"
        target = self.root / path
        original_data = copy.deepcopy(self.data)
        original_bytes = target.read_bytes()
        substitutions = [
            b'version = "0.12.0"',
            b'version = "=0.16.4"',  # fixes only the first advisory
            b'version = "=0.18.1"',
            b'version = "0.18"',  # admits unpatched0.18.0/0.18.1
            b'version = "=0.18.4"\npath = "../../unreviewed"',
            b'version = "=0.18.4"\ngit = "https://invalid.example/lru"',
        ]
        for replacement in substitutions:
            with self.subTest(requirement=replacement):
                self.data = copy.deepcopy(original_data)
                candidate = original_bytes.replace(b'version = "=0.18.4"', replacement)
                target.write_bytes(candidate)
                digest = hashlib.sha256(candidate).hexdigest()
                delta = len(candidate) - len(original_bytes)
                pin = next(p for p in self.data["retained_file_pins"] if p["path"] == path)
                pin.update(bytes=len(candidate), candidate_sha256=digest,
                           changed=digest != pin["upstream_sha256"])
                crate = next(c for c in self.data["crates"] if c["name"] == "ratatui")
                crate["manifest_candidate_sha256"] = digest
                crate["retained_bytes"] += delta
                self.data["bytes"] += delta
                self.write()
                self.reject("ratatui lru must retain reviewed")
        target.write_bytes(original_bytes)
        self.data = original_data
        self.write()

    def test_same_size_content_tamper(self):
        self.target.write_bytes(b"X" * self.target.stat().st_size)
        self.reject("size/hash mismatch")

    def test_size_tamper(self):
        self.target.write_bytes(self.target.read_bytes() + b"!")
        self.reject("size/hash mismatch")

    def test_missing_source(self):
        self.target.unlink()
        self.reject("missing=")

    def test_extra_source(self):
        self.target.with_name("injected.rs").write_text("fn injected() {}")
        self.reject("extra=")

    def test_extra_root_file(self):
        (self.base / "untracked.txt").write_text("untracked")
        self.reject("extra=")

    def test_extra_empty_directory(self):
        (self.base / "extra").mkdir()
        self.reject("extra directories")

    def test_file_symlink(self):
        original = self.target.read_bytes()
        self.target.unlink()
        outside = self.root / "outside"
        outside.write_bytes(original)
        self.target.symlink_to(outside)
        self.reject("symlink forbidden")

    def test_directory_symlink(self):
        (self.target.parent / "linked").symlink_to(self.root, target_is_directory=True)
        self.reject("symlink forbidden")

    def test_vendor_root_symlink(self):
        moved = self.root / "moved-security"
        self.base.rename(moved)
        self.base.symlink_to(moved, target_is_directory=True)
        self.reject()

    def test_repository_root_symlink(self):
        link = self.root / "root-alias"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(OSError):
            GUARD.verify(link)

    def test_duplicate_pin(self):
        self.data["retained_file_pins"].append(copy.deepcopy(self.data["retained_file_pins"][0]))
        self.write()
        self.reject("duplicate pinned path")

    def test_unsafe_relative_paths(self):
        for path in ("/absolute", "vendor/security/../escape", "vendor//security/file", "vendor/security/./file",
                     "vendor\\security\\file", "C:/drive/file", "vendor/security/x\nfile"):
            with self.subTest(path=path):
                self.data["retained_file_pins"][0]["path"] = path
                self.write()
                self.reject()

    def test_undeclared_crate(self):
        self.data["retained_file_pins"][0]["path"] = "vendor/security/unknown/Cargo.toml"
        self.write()
        self.reject("outside declared crates")

    def test_duplicate_crate(self):
        self.data["crates"][1] = copy.deepcopy(self.data["crates"][0])
        self.write()
        self.reject("duplicate crate identity")

    def test_missing_one_of_reviewed_crates(self):
        self.data["crates"].pop()
        self.write()
        self.reject("all 21")

    def test_substituted_identity(self):
        crate = self.data["crates"][0]
        crate["version"] = "0.0.0"
        crate["directory"] = crate["name"] + "-0.0.0"
        self.write()
        self.reject("crate identities")

    def test_crate_count_mismatch(self):
        self.data["crates"][0]["retained_files"] += 1
        self.write()
        self.reject("crate inventory mismatch")

    def test_global_count_mismatch(self):
        self.data["files"] += 1
        self.write()
        self.reject("global inventory mismatch")

    def test_malformed_original_checksum(self):
        self.data["retained_file_pins"][0]["upstream_sha256"] = "not-a-checksum"
        self.write()
        self.reject("upstream source checksum")

    def test_changed_flag_mismatch(self):
        self.data["retained_file_pins"][0]["changed"] = True
        self.write()
        self.reject("changed flag")

    def test_manifest_upstream_pin_mismatch(self):
        self.data["crates"][0]["manifest_upstream_sha256"] = "f" * 64
        self.write()
        self.reject("manifest upstream pin mismatch")

    def test_oversized_pin(self):
        self.data["retained_file_pins"][0]["bytes"] = GUARD.MAX_FILE_BYTES + 1
        self.write()
        self.reject("oversized")

    def test_oversized_real_file(self):
        with self.target.open("wb") as handle:
            handle.truncate(GUARD.MAX_FILE_BYTES + 1)
        self.reject("oversized file")

    def test_boolean_size(self):
        self.data["retained_file_pins"][0]["bytes"] = True
        self.write()
        self.reject("invalid pinned file size")

    def test_duplicate_json_key(self):
        self.manifest.write_text('{"schema":"one","schema":"two"}')
        self.reject("duplicate JSON key")

    def test_missing_provenance(self):
        self.manifest.unlink()
        self.reject()

    def test_fifo_refused_without_blocking(self):
        import os
        self.target.unlink()
        os.mkfifo(self.target)
        self.reject("not a regular file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
