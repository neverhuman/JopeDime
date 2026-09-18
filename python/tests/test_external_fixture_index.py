import copy
import hashlib
import json
from pathlib import Path
import stat

import pytest


import external_fixture_index as fixtures  # noqa: E402


@pytest.mark.xbabe2  # needs the benchmarks/sotarena clone at e5db30a0 (B18: pin unfetchable, submodule absent from the port tree)
def test_index_is_path_free_and_structurally_complete():
    index = fixtures.read_index()
    fixtures.validate_index(index)

    assert index["group_counts"] == {
        "binary42": 168,
        "modules": 2,
        "train_only": 252,
    }
    assert len(index["entries"]) == 422
    assert index["unresolved_count"] == 46
    assert b"/home/" not in fixtures.canonical_bytes(index)


@pytest.mark.xbabe2  # needs the benchmarks/sotarena clone at e5db30a0 (B18: pin unfetchable, submodule absent from the port tree)
def test_complete_verification_fails_closed_on_missing_binary_splits():
    with pytest.raises(
        fixtures.FixtureError, match="46 external fixture files remain unresolved"
    ):
        fixtures.validate_index(fixtures.read_index(), require_complete=True)


@pytest.mark.xbabe2  # needs the benchmarks/sotarena clone at e5db30a0 (B18: pin unfetchable, submodule absent from the port tree)
def test_workspace_profile_materializes_committed_read_only_bytes(tmp_path):
    destination = tmp_path / "fixtures"
    receipt = fixtures.materialize(
        fixtures.read_index(), destination, "workspace_tests"
    )

    assert receipt["file_count"] == 254
    sample = destination / "sotarena" / "elo.py"
    assert sample.is_file() and not sample.is_symlink()
    assert stat.S_IMODE(sample.stat().st_mode) == 0o444
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    expected = next(
        row
        for row in fixtures.read_index()["entries"]
        if row["target_path"] == "sotarena/elo.py"
    )
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == expected["sha256"]


def test_one_field_mutation_is_rejected():
    index = copy.deepcopy(fixtures.read_index())
    index["entries"][0]["sha256"] = "0" * 64
    with pytest.raises(fixtures.FixtureError):
        fixtures.validate_index(index)


@pytest.mark.xbabe2  # recovers the exact-46 fixtures from /home/ubuntu/sotarena_old
def test_qualified_local_recovery_imports_exact_46_into_temporary_cas(tmp_path):
    source_root = Path("/home/ubuntu/sotarena_old")
    assert source_root.is_dir(), "the frozen local-recovery observation is absent"
    fixture_store = tmp_path / "external-fixtures-v1"
    import_path = tmp_path / "JopeExternalFixtureRecoveryImport.v1.json"
    index_path = tmp_path / "JopeExternalFixtureIndex.v1.json"
    import_manifest, recovered = fixtures.import_local_recovery(
        fixtures.read_index(),
        source_root,
        fixture_store,
        import_path,
        index_path,
    )

    assert import_manifest["schema"] == "JopeExternalFixtureRecoveryImport.v1"
    assert import_manifest["object_count"] == 46
    assert sum(row["size_bytes"] for row in import_manifest["objects"]) == 17_172_847
    assert import_manifest["metadata_authority"] == {
        "repository": "sotarena-rust",
        "commit": "c1f108464d43d0c01417e048aa6044fa19ad2bcc",
        "tree": "0d31b7da2297fb71203d317a885bda1ba668673e",
    }
    assert import_manifest["recovery_observation"] == {
        "head": "2109b8f001778c97cea681650786c2202e6be3a0",
        "tree": "c21408eb41e87b6ee12728a6dc225a7f0492958a",
        "dirty_observation_not_authority": True,
    }
    assert recovered["unresolved_count"] == 0
    fixtures.validate_index(recovered, require_complete=True)
    assert json.loads(index_path.read_text()) == recovered
    import_digest = hashlib.sha256(fixtures.canonical_bytes(import_manifest)).hexdigest()
    recovered_rows = [
        row
        for row in recovered["entries"]
        if row["source"]["kind"] == "verified_local_recovery"
    ]
    assert len(recovered_rows) == 46
    assert all(
        row["source"]["import_manifest_sha256"] == import_digest
        and row["source"]["dirty_observation_not_authority"] is True
        for row in recovered_rows
    )
    for row in recovered_rows:
        object_path = fixture_store / row["source"]["cas_object"]
        assert object_path.is_file() and not object_path.is_symlink()
        assert stat.S_IMODE(object_path.stat().st_mode) == 0o444
        assert object_path.stat().st_size == row["size_bytes"]
        assert hashlib.sha256(object_path.read_bytes()).hexdigest() == row["sha256"]

    repositories = {row["id"]: row for row in recovered["repositories"]}
    sample = recovered_rows[0]
    assert hashlib.sha256(
        fixtures.source_bytes(sample, repositories, fixture_store)
    ).hexdigest() == sample["sha256"]


def test_recovery_double_read_rejects_symlink(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b"x,y\n1,2\n")
    link = tmp_path / "link.csv"
    link.symlink_to(source)
    with pytest.raises(fixtures.FixtureError, match="regular non-symlink"):
        fixtures.stable_double_read(link)
