"""Storage-boundary controls; fixtures call production resolver without deserialization."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("export_fused_bundle.py")
SPEC = importlib.util.spec_from_file_location("fused_exporter_storage_test", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def custody_fixture(tmp_path, monkeypatch):
    root = tmp_path / "custody"
    (root / "files").mkdir(parents=True)
    cache = tmp_path / "cache"
    (cache / "sha256").mkdir(parents=True)
    payload = b"small immutable fixture bytes"
    digest = _sha(payload)
    external_rel = Path("binary/asset.bin")
    external = {
        "schema": "jope.external-custody/v1",
        "recovery_commit": "0" * 40,
        "files": [{
            "path": "/" + external_rel.as_posix(),
            "sha256": digest,
            "bytes": len(payload),
            "git_path": "research/flywheel_selfplay/custody/files/" + external_rel.as_posix(),
            "git_object_id": "1" * 40,
        }],
    }
    vendor = b"small vendor fixture"
    vendor_digest = _sha(vendor)
    source_digest = _sha(b"src")
    manifest = {
        "import_files": [{"path": "/" + external_rel.as_posix(), "sha256": digest, "bytes": len(payload)}],
        "assets": [{"path": "/vendor.txt", "sha256": vendor_digest, "bytes": len(vendor)}],
        "evaluator": {
            "source_root": "/source",
            "compiled_source_inputs": [{
                "path": "/source/input.rs", "repo_path": "input.rs",
                "actual_sha256": source_digest, "revision_sha256": source_digest,
                "matches_revision": True,
            }],
        },
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    (root / "EXTERNAL.json").write_text(json.dumps(external))
    (root / "files/vendor.txt").write_bytes(vendor)
    (root / "files/binary").mkdir()
    (root / "files/binary/asset.bin").write_bytes(b"stale vendored copy")
    (root / "files/source").mkdir()
    (root / "files/source/input.rs").write_bytes(b"src")
    (cache / "sha256" / digest).write_bytes(payload)
    # Isolated fixture repository supplies the pinned immutable Git recovery object.
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    git_file = repo / external["files"][0]["git_path"]
    git_file.parent.mkdir(parents=True)
    git_file.write_bytes(payload)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=Fixture", "-c",
        "user.email=fixture@example.invalid", "commit", "-qm", "fixture",
    ], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    object_id = subprocess.check_output([
        "git", "-C", str(repo), "rev-parse", f"{commit}:{external['files'][0]['git_path']}"
    ], text=True).strip()
    external["recovery_commit"] = commit
    external["files"][0]["git_object_id"] = object_id
    (root / "EXTERNAL.json").write_text(json.dumps(external))
    monkeypatch.setattr(exporter, "REPO_ROOT", repo)
    yield root, cache, external_rel, payload, repo


def test_external_cache_reads_exact_pinned_bytes(custody_fixture):
    root, cache, rel, payload, _ = custody_fixture
    actual = exporter._pinned_bytes(root, "/" + rel.as_posix(), _sha(payload), len(payload), external_root=cache)
    assert actual == payload


def test_immutable_git_recovery_reads_fixture_blob(custody_fixture):
    root, _, rel, payload, _ = custody_fixture
    assert exporter._pinned_bytes(root, "/" + rel.as_posix(), _sha(payload), len(payload)) == payload


def test_vendored_read_and_external_cli_forwarding(custody_fixture, monkeypatch, capsys):
    root, cache, _, _, _ = custody_fixture
    assert exporter._pinned_bytes(root, "/vendor.txt", _sha(b"small vendor fixture"), len(b"small vendor fixture")) == b"small vendor fixture"
    observed = {}
    monkeypatch.setattr(exporter, "load_inputs", lambda custody, external_root=None: observed.update(external_root=external_root) or {})
    monkeypatch.setattr(exporter, "read_json", lambda data: {})
    monkeypatch.setattr(exporter, "export_bundle", lambda **kwargs: {"bundle_digest": "ok"})
    metadata_path = root / "metadata.json"
    metadata_path.write_text("{}")
    assert exporter.main(["--custody", str(root), "--external-root", str(cache), "--metadata", str(metadata_path), "--output", "out"]) == 0
    assert observed["external_root"] == cache
    assert capsys.readouterr().out.strip() == "ok"


@pytest.mark.parametrize("mutation", ["wrong_digest", "wrong_size", "missing", "relative_root", "payload_symlink", "ancestor_symlink", "root_symlink", "directory_payload"])
def test_explicit_cache_fails_closed(custody_fixture, tmp_path, mutation):
    root, cache, rel, payload, _ = custody_fixture
    digest = _sha(payload)
    path = cache / "sha256" / digest
    chosen_root = cache
    if mutation == "wrong_digest":
        path.write_bytes(b"tampered")
    elif mutation == "wrong_size":
        with pytest.raises(exporter.ExportError):
            exporter._pinned_bytes(root, "/" + rel.as_posix(), digest, len(payload) + 1, external_root=cache)
        return
    elif mutation == "missing":
        path.unlink()
    elif mutation == "relative_root":
        chosen_root = Path("relative-cache")
    elif mutation == "payload_symlink":
        path.unlink()
        path.symlink_to(tmp_path / "elsewhere")
    elif mutation == "ancestor_symlink":
        (cache / "sha256").rename(cache / "sha256-real")
        (cache / "sha256").symlink_to(cache / "sha256-real", target_is_directory=True)
    elif mutation == "root_symlink":
        chosen_root = tmp_path / "cache-link"
        chosen_root.symlink_to(cache, target_is_directory=True)
    elif mutation == "directory_payload":
        path.unlink()
        path.mkdir()
    with pytest.raises(exporter.ExportError):
        exporter._pinned_bytes(root, "/" + rel.as_posix(), digest, len(payload), external_root=chosen_root)


def test_external_pin_mismatch_duplicate_and_no_cache_fallback(custody_fixture):
    root, cache, rel, payload, _ = custody_fixture
    external_path = root / "EXTERNAL.json"
    original = json.loads(external_path.read_text())
    altered = json.loads(json.dumps(original))
    altered["files"][0]["bytes"] += 1
    external_path.write_text(json.dumps(altered))
    with pytest.raises(exporter.ExportError):
        exporter._pinned_bytes(root, "/" + rel.as_posix(), _sha(payload), len(payload), external_root=cache)
    altered["files"] = original["files"] * 2
    external_path.write_text(json.dumps(altered))
    with pytest.raises(exporter.ExportError):
        exporter._pinned_bytes(root, "/" + rel.as_posix(), _sha(payload), len(payload), external_root=cache)
    altered["files"] = [{"malformed": True}]
    external_path.write_text(json.dumps(altered))
    with pytest.raises(exporter.ExportError):
        exporter._pinned_bytes(root, "/" + rel.as_posix(), _sha(payload), len(payload), external_root=cache)
    external_path.write_text(json.dumps(original))
    (cache / "sha256" / _sha(payload)).unlink()
    assert (root / "files" / rel).read_bytes() == b"stale vendored copy"
    with pytest.raises(exporter.ExportError):
        exporter._pinned_bytes(root, "/" + rel.as_posix(), _sha(payload), len(payload), external_root=cache)
