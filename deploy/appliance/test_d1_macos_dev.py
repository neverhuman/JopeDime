"""Synthetic checks for the D1 Darwin development-package boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("d1_macos_dev.py")
SPEC = importlib.util.spec_from_file_location("d1_macos_dev", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
d1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d1)

SOURCE = "2b242fb01560efd39218ccced0993f084e802a51"
FEATURES = ["embedded", "engine", "mcp", "client"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(path: Path, code: Path, source: str = SOURCE) -> None:
    executable = path / "jopedime"
    executable.write_bytes(code.read_bytes())
    executable.chmod(0o755)
    metadata = {
        "schema": d1.PACKAGE_SCHEMA,
        "label": "DEV-BUILD",
        "qualification": "NOT_QUALIFIED",
        "source_sha": source,
        "platform": "macos-arm64",
        "target_triple": "aarch64-apple-darwin",
        "profile": "fixture",
        "features": FEATURES,
        "artifact": "jopedime",
        "artifact_sha256": sha(executable),
        "release_signed": False,
        "development_signed": True,
    }
    (path / "DEV-BUILD.json").write_text(json.dumps(metadata), encoding="utf-8")


class D1DarwinDevTests(unittest.TestCase):
    def test_toolchain_path_must_be_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no-toolchain"
            path.mkdir()
            self.assertTrue(d1.toolchain_path_is_empty(path))
            (path / "rustc").touch()
            self.assertFalse(d1.toolchain_path_is_empty(path))
            self.assertFalse(d1.toolchain_path_is_empty(path / "missing"))

    def test_version_source_uses_bound_package_identity(self):
        self.assertEqual(d1.version_source({"source_sha40": SOURCE}), SOURCE)
        self.assertEqual(
            d1.version_source(
                {"source_sha40": "unknown", "package": {"source_sha40": SOURCE}}
            ),
            SOURCE,
        )

    def test_fixture_identity_requires_development_signature_and_source(self):
        identity = {
            "source_sha40": "unknown",
            "target": "aarch64-apple-darwin",
            "features": {name: True for name in FEATURES} | {"dev_keys": True},
            "trusts_development_key": True,
            "package": {
                "source_sha40": SOURCE,
                "profile": "fixture",
                "scope": "development",
                "signer_role": "development",
            },
        }
        d1.validate_fixture_identity(identity, SOURCE)
        identity["package"]["signer_role"] = "release"
        with self.assertRaisesRegex(d1.D1DevError, "identity/features"):
            d1.validate_fixture_identity(identity, SOURCE)

    def test_candidate_requires_dev_label_and_byte_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate"
            candidate.mkdir()
            package(candidate, Path(__file__))

            executable, metadata = d1.validate_dev_candidate(candidate, SOURCE)
            self.assertEqual(executable.name, "jopedime")
            self.assertEqual(metadata["qualification"], "NOT_QUALIFIED")

            receipt = json.loads((candidate / "DEV-BUILD.json").read_text())
            receipt["label"] = "SIGNED-CANDIDATE"
            (candidate / "DEV-BUILD.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(d1.D1DevError, "not an intact"):
                d1.validate_dev_candidate(candidate, SOURCE)

    def test_candidate_rejects_changed_executable_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate"
            candidate.mkdir()
            package(candidate, Path(__file__))
            with (candidate / "jopedime").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(d1.D1DevError, "not an intact"):
                d1.validate_dev_candidate(candidate, SOURCE)

    def test_development_key_fixture_is_explicitly_dev_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate"
            candidate.mkdir()
            package(candidate, Path(__file__))
            metadata_path = candidate / "DEV-BUILD.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["profile"] = "full"
            metadata["development_signed"] = False
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(d1.D1DevError, "not an intact"):
                d1.validate_dev_candidate(candidate, SOURCE)


if __name__ == "__main__":
    unittest.main()
