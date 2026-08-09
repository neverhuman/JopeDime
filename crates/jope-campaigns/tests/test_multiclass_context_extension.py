from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "encode_multiclass_context_extension.py"
SPEC = importlib.util.spec_from_file_location("encode_multiclass_context_extension", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def runtime_bundle(tmp_path: Path) -> tuple[Path, str, str, Path]:
    payload = b"declared runtime source\n"
    row = {
        "target": "pytorch/encoding/encode_external.py",
        "sha256": MODULE.sha256_bytes(payload),
        "bytes": len(payload),
        "python_role": "pytorch_dataset_encoding",
    }
    identity_row = {
        "target": row["target"],
        "sha256": row["sha256"],
        "bytes": row["bytes"],
        "python_role": row["python_role"],
    }
    identity = MODULE.sha256_bytes(MODULE.canonical_bytes([identity_row]))
    root = tmp_path / identity
    declared = root / row["target"]
    declared.parent.mkdir(parents=True)
    declared.write_bytes(payload)
    manifest = {
        "schema": "RuntimeBundle.v3",
        "bundle_sha256": identity,
        "files": [row],
    }
    manifest_path = root / "bundle.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return root, identity, MODULE.sha256_file(manifest_path), declared


def test_runtime_identity_ignores_undeclared_bytecode_but_detects_declared_drift(
    tmp_path: Path,
) -> None:
    root, identity, manifest_sha256, declared = runtime_bundle(tmp_path)
    MODULE.verify_runtime_bundle(root, identity, manifest_sha256)

    cache = root / "pytorch/encoding/__pycache__/encode_external.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"volatile bytecode")
    MODULE.verify_runtime_bundle(root, identity, manifest_sha256)

    declared.write_bytes(b"drifted runtime source\n")
    with pytest.raises(RuntimeError, match="target size drift|target hash drift"):
        MODULE.verify_runtime_bundle(root, identity, manifest_sha256)
