"""Metadata-only import/runtime pinning; these tests never import Torch."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from jope.policy import experimental_runtime as runtime


@pytest.fixture
def tmp_path():
    root = Path(__file__).resolve().parents[2] / "target" / "runtime-contract-fixtures"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def test_runtime_requires_isolation_and_new_durable_output(tmp_path, monkeypatch):
    source, packages = tmp_path / "source", tmp_path / "packages"
    (source / "jope/policy").mkdir(parents=True)
    (source / "jope/policy/experimental_proposal.py").write_text("# fixture")
    packages.mkdir()
    monkeypatch.setattr(sys, "executable", str(Path(sys.executable).resolve()))
    monkeypatch.setattr(
        sys,
        "flags",
        SimpleNamespace(
            **{
                name: getattr(sys.flags, name)
                for name in dir(sys.flags)
                if not name.startswith("_") and isinstance(getattr(sys.flags, name), int)
            },
            **{},
        ),
    )
    with pytest.raises(ValueError, match="physical interpreter"):
        runtime.prepare(source, packages, tmp_path / "output")
    monkeypatch.setattr(
        sys,
        "flags",
        SimpleNamespace(
            **{
                **{
                    name: getattr(sys.flags, name)
                    for name in dir(sys.flags)
                    if not name.startswith("_") and isinstance(getattr(sys.flags, name), int)
                },
                "isolated": True,
            }
        ),
    )
    with pytest.raises(ValueError, match="new durable"):
        runtime.prepare(source, packages, packages)
    with pytest.raises(ValueError, match="new durable"):
        runtime.prepare(source, packages, Path("/tmp/jope-uncreated-fixture"))


def test_runtime_captures_exact_source_and_distribution_bytes(tmp_path, monkeypatch):
    source, packages = tmp_path / "source", tmp_path / "packages"
    (source / "jope/policy").mkdir(parents=True)
    module = source / "jope/policy/experimental_proposal.py"
    module.write_text("# fixture proposal source")
    packages.mkdir()
    package = packages / "synthetic.py"
    package.write_text("# fixture distribution")
    monkeypatch.setattr(sys, "executable", str(Path(sys.executable).resolve()))
    monkeypatch.setattr(
        sys,
        "flags",
        SimpleNamespace(
            **{
                **{
                    name: getattr(sys.flags, name)
                    for name in dir(sys.flags)
                    if not name.startswith("_") and isinstance(getattr(sys.flags, name), int)
                },
                "isolated": True,
            }
        ),
    )
    identity = {
        "python": runtime.artifact(sys.executable),
        "sources": {"policy/experimental_proposal.py": runtime.sha(module)},
        "packages": {
            "synthetic": {
                "version": "1",
                "files": 1,
                "files_sha256": hashlib.sha256(
                    runtime.encoded({"synthetic.py": runtime.sha(package)})
                ).hexdigest(),
            }
        },
    }
    monkeypatch.setitem(
        sys.modules,
        "jope.policy",
        SimpleNamespace(training_campaign=SimpleNamespace(runtime_identity=lambda: identity)),
    )
    distribution = SimpleNamespace(
        metadata={"Name": "synthetic"},
        version="1",
        files=[Path("synthetic.py")],
        locate_file=lambda item: packages / item,
    )
    monkeypatch.setattr(runtime.importlib.metadata, "distributions", lambda: [distribution])
    original_path = sys.path[:]
    try:
        result = runtime.prepare(source, packages, tmp_path / "output")
    finally:
        sys.path[:] = original_path
    value = json.loads(Path(result["path"]).read_text())
    manifest = json.loads(Path(value["manifest"]["path"]).read_text())
    assert value["program"] == identity["python"]
    assert value["source_files"] == [runtime.artifact(module)]
    assert value["package_files"] == [runtime.artifact(package)]
    assert value["runtime_sha256"] == hashlib.sha256(runtime.encoded(identity)).hexdigest()
    assert manifest["packages"]["synthetic"]["files"]["synthetic.py"] == runtime.artifact(package)
    assert os.stat(result["path"]).st_mode & 0o777 == 0o600
    assert not value["environment"].get("PYTHONPATH")
    before = value["package_files"][0]["sha256"]
    package.write_text("# changed distribution")
    assert runtime.sha(package) != before


def test_runtime_receipts_are_immutable_and_nonfinite_values_refused(tmp_path):
    path = tmp_path / "receipt.json"
    reference = runtime.write(path, {"source": "fixture"})
    assert runtime.sha(path) == reference["sha256"]
    with pytest.raises(FileExistsError):
        runtime.write(path, {"source": "changed"})
    with pytest.raises(ValueError):
        runtime.encoded({"unsupported": float("nan")})
