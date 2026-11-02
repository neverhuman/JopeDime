"""RES-06 acceptance on the committed, pinned, row-free custody assets (CPU only)."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import torch
from safetensors.numpy import load as load_safetensors

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "research/flywheel_selfplay/export_fused_bundle.py"
SPEC = importlib.util.spec_from_file_location("fused_exporter", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.fixture(scope="module")
def inputs():
    torch.set_num_threads(2)
    return exporter.load_inputs()


@pytest.fixture
def metadata():
    return dict(
        schema="jope.fused-bundle/v1",
        architecture="dope-fused-ranker/h40b",
        dope_core_rev="15e12aa3dc33851655580f6aa7c9c66353143f23",
        reference_source_sha256={},
        constants=dict(exporter.CONSTANTS),
        surrogate_recipe="full_train_cv/constrained_two_inner_fold_log_rmse_ratio/negative_train_cv_fitness_gate_penalty",
        trained_objectives=["rmse"],
        landmark_recipe="full-high/v1",
        landmark_sample_rows=1024,
        qualified=[],
        asset_policy="internal-research-only; not qualified for redistribution",
    )


def publish(inputs, metadata, output):
    return exporter.export_bundle(**inputs, metadata=metadata, output=output)


@pytest.fixture
def bundle(inputs, metadata, tmp_path):
    path = tmp_path / "bundle"
    manifest = publish(inputs, metadata, path)
    return path, manifest


def test_schema_and_all_tensor_bytes_are_preserved(inputs, bundle):
    path, manifest = bundle
    schema = json.loads((ROOT / "schemas/jope.fused-bundle.v1.schema.json").read_bytes())
    jsonschema.Draft202012Validator(schema).validate(manifest)
    assert set(p.name for p in path.iterdir()) == exporter.PAYLOADS | {"manifest.json"}
    actual = {}
    for name in ("dope.safetensors", "jope.safetensors"):
        data = (path / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest[name.replace(".safetensors", "_weights_sha256")]
        tensors = load_safetensors(data)
        assert not set(actual) & set(tensors)
        actual.update(tensors)
    assert len(actual) == 338
    assert set(actual) == set(inputs["state"])
    for name, tensor in inputs["state"].items():
        np.testing.assert_array_equal(actual[name], tensor.numpy())
        assert actual[name].dtype == tensor.numpy().dtype
    assert exporter.dope_composite(inputs["state"]) == exporter.FROZEN_DOPE_COMPOSITE
    assert manifest["qualified"] == []


def test_bank_is_complete_ordered_row_free_and_bound_to_checkpoint(inputs, bundle):
    path, _ = bundle
    decoded = subprocess.check_output(["zstd", "-q", "-d", "-c", str(path / "bank.jsonl.zst")])
    rows = [json.loads(line) for line in decoded.splitlines()]
    assert len(rows) == 881
    for index, (row, source) in enumerate(zip(rows, inputs["champions"], strict=True)):
        assert set(row) == {
            "dsl",
            "family_hash",
            "feat412",
            "proj16",
            "lm48_summary",
            "role",
            "coordinate_space",
        }
        assert row["role"] == "model_prior"
        assert row["coordinate_space"] == "standardized_lm48_l2_unit"
        assert row["dsl"] == source["dsl"]
        assert row["family_hash"] == source["family_id"].removeprefix("fam-")
        np.testing.assert_array_equal(row["feat412"], inputs["state"]["bank_feat412"][index + 4].numpy())
        np.testing.assert_array_equal(row["proj16"], inputs["state"]["bank_proj16"][index + 4].numpy())
        np.testing.assert_array_equal(row["lm48_summary"], inputs["state"]["bank_lm48_z_881"][index].numpy())
    features = json.loads((path / "featurizer.json").read_bytes())
    assert features["feature_names"] == inputs["feature_names"]
    assert features["proj16_domain"] == "jope-context-cross-v1"


def test_real_cli_is_deterministic_across_processes_hash_seeds_and_working_dirs(metadata, tmp_path):
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps(metadata))
    outputs, digests = [], []
    for seed in ("1", "811"):
        directory = tmp_path / seed
        directory.mkdir()
        output = directory / "bundle"
        env = dict(
            os.environ,
            PYTHONHASHSEED=seed,
            PYTHONDONTWRITEBYTECODE="1",
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        run = subprocess.run(
            [sys.executable, str(SCRIPT), "--metadata", str(metadata_file), "--output", str(output)],
            cwd=directory,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert run.returncode == 0, run.stderr
        digests.append(run.stdout.strip())
        outputs.append({p.name: p.read_bytes() for p in output.iterdir()})
    assert digests[0] == digests[1]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "name",
    [
        "step2_lm_std_1x256",
        "dope_model.router.small_embed",
        "adapter_10tok.null_hist_token",
        "adapter_10tok.anchor.ctx_in.0.weight",
    ],
)
def test_missing_tensor_refused_without_output(inputs, metadata, tmp_path, name):
    changed = dict(inputs, state=dict(inputs["state"]))
    del changed["state"][name]
    with pytest.raises(exporter.ExportError, match="complete tensor manifest"):
        publish(changed, metadata, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_extra_tensor_refused_without_output(inputs, metadata, tmp_path):
    state = dict(inputs["state"], unlisted=torch.zeros(1))
    with pytest.raises(exporter.ExportError, match="extra=.*unlisted"):
        publish(dict(inputs, state=state), metadata, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad", [torch.zeros(2), torch.zeros(10, dtype=torch.float64)])
def test_wrong_shape_or_dtype_refused(inputs, metadata, tmp_path, bad):
    state = dict(inputs["state"])
    state["adapter_10tok.modality_gate_logits"] = bad
    with pytest.raises(exporter.ExportError, match="shape/dtype mismatch"):
        publish(dict(inputs, state=state), metadata, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_weight_refused(inputs, metadata, tmp_path, value):
    state = dict(inputs["state"])
    state["adapter_10tok.modality_gate_logits"] = state["adapter_10tok.modality_gate_logits"].clone()
    state["adapter_10tok.modality_gate_logits"][0] = value
    with pytest.raises(exporter.ExportError, match="finite"):
        publish(dict(inputs, state=state), metadata, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["dope_model.router.small_embed", "mu_lm48"])
def test_frozen_dope_parameter_and_normalizer_drift_refused(inputs, metadata, tmp_path, name):
    state = dict(inputs["state"])
    state[name] = state[name] + 1
    with pytest.raises(exporter.ExportError, match="frozen DOPE"):
        publish(dict(inputs, state=state), metadata, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_frozen_anchor_alias_conflict_refused(inputs, metadata, tmp_path):
    state = dict(inputs["state"])
    state["adapter_10tok.anchor.alpha_prior"] = state["adapter_10tok.anchor.alpha_prior"] + 1
    with pytest.raises(exporter.ExportError, match="anchor aliases disagree"):
        publish(dict(inputs, state=state), metadata, tmp_path / "out")


@pytest.mark.parametrize("change", ["missing", "rows", "feature", "summary", "family", "dsl"])
def test_bad_bank_refused(inputs, metadata, tmp_path, change):
    champions = list(inputs["champions"])
    champions[0] = copy.deepcopy(champions[0])
    if change == "missing":
        champions.pop()
    elif change == "rows":
        champions[0]["training_rows"] = [[1, 2]]
    elif change == "feature":
        champions[0]["feat412"][0] += 1
    elif change == "summary":
        champions[0]["lm48_z"][0] += 10
    elif change == "family":
        champions[0]["family_id"] = "unknown"
    else:
        champions[0]["dsl"] += " "
    with pytest.raises(exporter.ExportError):
        publish(dict(inputs, champions=champions), metadata, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["schema", "extra", "computed", "constants", "source", "features"])
def test_bad_manifest_or_features_refused(inputs, metadata, tmp_path, change):
    args = dict(inputs)
    if change == "schema":
        metadata["architecture"] = "unknown"
    elif change == "extra":
        metadata["extra"] = True
    elif change == "computed":
        metadata["bundle_digest"] = "sha256:" + "0" * 64
    elif change == "constants":
        metadata["constants"]["revealed"] = 5
    elif change == "source":
        name = next(iter(inputs["source_hashes"]))
        metadata["reference_source_sha256"][name] = "0" * 64
    else:
        args["feature_names"] = ["duplicate"] * 412
    with pytest.raises((exporter.ExportError, jsonschema.ValidationError)):
        publish(args, metadata, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_existing_output_preserved(inputs, metadata, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_bytes(b"keep")
    with pytest.raises(exporter.ExportError, match="new directory"):
        publish(inputs, metadata, output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {"sentinel": b"keep"}


def test_atomic_publish_refuses_empty_destination_created_after_validation(tmp_path):
    staging, output = tmp_path / "staging", tmp_path / "racing-output"
    staging.mkdir()
    (staging / "payload").write_bytes(b"new")
    output.mkdir()
    with pytest.raises(FileExistsError):
        exporter._publish_directory(staging, output)
    assert output.is_dir() and not list(output.iterdir())
    assert (staging / "payload").read_bytes() == b"new"


def test_pinned_input_corruption_and_symlink_refused(tmp_path):
    root = tmp_path / "custody"
    (root / "files/source").mkdir(parents=True)
    path = root / "files/source/asset"
    pinned = b"pinned bytes"
    path.write_bytes(pinned)
    _write_storage_metadata(root, "/source/asset", pinned)
    assert exporter._pinned_bytes(root, "/source/asset", _storage_sha(pinned), len(pinned)) == pinned
    path.write_bytes(b"P" + pinned[1:])
    with pytest.raises(exporter.ExportError, match="hash/length"):
        exporter._pinned_bytes(root, "/source/asset", _storage_sha(pinned), len(pinned))
    path.unlink()
    path.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(exporter.ExportError, match="symlink"):
        exporter._pinned_bytes(root, "/source/asset", "0" * 64)
    with pytest.raises(exporter.ExportError, match="unsafe"):
        exporter._pinned_bytes(root, "/../asset", "0" * 64)


def test_digest_binds_unsigned_metadata_names_lengths_and_every_payload():
    manifest = {"scale": 0.4208, "recipe": "test"}
    files = {"a": b"bc", "d": b"e"}
    digest = exporter.compute_bundle_digest(manifest, files)
    assert digest == exporter.compute_bundle_digest(
        dict(reversed(list(manifest.items()))), dict(reversed(list(files.items())))
    )
    assert digest == exporter.compute_bundle_digest(dict(manifest, bundle_digest="sha256:" + "0" * 64), files)
    for name in files:
        assert digest != exporter.compute_bundle_digest(manifest, dict(files, **{name: files[name] + b"!"}))
    assert digest != exporter.compute_bundle_digest(dict(manifest, scale=0.5), files)
    assert digest != exporter.compute_bundle_digest(manifest, {"ab": b"c", "d": b"e"})
    # Independently assembled fixed vector pins the documented wire framing.
    parts = [
        b"jope.fused-bundle/v1\0",
        struct.pack("<Q", 32),
        b'{"recipe":"test","scale":0.4208}',
        struct.pack("<Q", 2),
        struct.pack("<Q", 1),
        b"a",
        struct.pack("<Q", 2),
        b"bc",
        struct.pack("<Q", 1),
        b"d",
        struct.pack("<Q", 1),
        b"e",
    ]
    assert digest == "sha256:" + hashlib.sha256(b"".join(parts)).hexdigest()


def test_duplicate_or_nonfinite_json_refused():
    for data in (b'{"a":1,"a":2}', b'{"a":NaN}'):
        with pytest.raises(exporter.ExportError):
            exporter.read_json(data)


def _storage_sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_storage_metadata(root, vendored_path, vendored_bytes):
    """Build valid custody metadata for focused storage-boundary checks."""
    external_bytes = b"external fixture pin"
    external_path = "fixture/external.bin"
    source_bytes = b"fixture source"
    manifest = {
        "import_files": [{
            "path": "/" + external_path,
            "sha256": _storage_sha(external_bytes),
            "bytes": len(external_bytes),
        }],
        "assets": [{
            "path": vendored_path,
            "sha256": _storage_sha(vendored_bytes),
            "bytes": len(vendored_bytes),
        }],
        "evaluator": {
            "source_root": "/fixture-source",
            "compiled_source_inputs": [{
                "path": "/fixture-source/input.py",
                "repo_path": "input.py",
                "actual_sha256": _storage_sha(source_bytes),
                "revision_sha256": _storage_sha(source_bytes),
                "matches_revision": True,
            }],
        },
    }
    external = {
        "schema": "jope.external-custody/v1",
        "recovery_commit": "a" * 40,
        "files": [{
            "path": "/" + external_path,
            "sha256": _storage_sha(external_bytes),
            "bytes": len(external_bytes),
            "git_path": "research/flywheel_selfplay/custody/files/" + external_path,
            "git_object_id": "b" * 40,
        }],
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    (root / "EXTERNAL.json").write_text(json.dumps(external))


@pytest.fixture
def storage_custody_fixture(tmp_path, monkeypatch):
    root = tmp_path / "custody"
    (root / "files").mkdir(parents=True)
    cache = tmp_path / "cache"
    (cache / "sha256").mkdir(parents=True)
    payload = b"small immutable fixture bytes"
    digest = _storage_sha(payload)
    external_rel = Path("binary/asset.bin")
    unrelated_rel = Path("binary/unrelated.bin")
    unrelated = b"unrelated external pin"
    vendor = b"small vendor fixture"
    source = b"src"
    manifest = {
        "import_files": [
            {"path": "/" + external_rel.as_posix(), "sha256": digest, "bytes": len(payload)},
            {"path": "/" + unrelated_rel.as_posix(), "sha256": _storage_sha(unrelated), "bytes": len(unrelated)},
        ],
        "assets": [{"path": "/vendor.txt", "sha256": _storage_sha(vendor), "bytes": len(vendor)}],
        "evaluator": {
            "source_root": "/source",
            "compiled_source_inputs": [{
                "path": "/source/input.rs", "repo_path": "input.rs",
                "actual_sha256": _storage_sha(source), "revision_sha256": _storage_sha(source),
                "matches_revision": True,
            }],
        },
    }
    external = {
        "schema": "jope.external-custody/v1",
        "recovery_commit": "0" * 40,
        "files": [
            {
                "path": "/" + external_rel.as_posix(),
                "sha256": digest,
                "bytes": len(payload),
                "git_path": "research/flywheel_selfplay/custody/files/" + external_rel.as_posix(),
                "git_object_id": "1" * 40,
            },
            {
                "path": "/" + unrelated_rel.as_posix(),
                "sha256": _storage_sha(unrelated),
                "bytes": len(unrelated),
                "git_path": "research/flywheel_selfplay/custody/files/" + unrelated_rel.as_posix(),
                "git_object_id": "f" * 40,
            },
        ],
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    (root / "EXTERNAL.json").write_text(json.dumps(external))
    (root / "files/vendor.txt").write_bytes(vendor)
    (root / "files/binary").mkdir()
    (root / "files/binary/asset.bin").write_bytes(b"stale vendored copy")
    (root / "files/source").mkdir()
    (root / "files/source/input.rs").write_bytes(source)
    (cache / "sha256" / digest).write_bytes(payload)

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
    return root, cache, external_rel, payload, repo


def test_storage_cache_succeeds_without_history_or_git(storage_custody_fixture, monkeypatch, tmp_path):
    root, cache, rel, payload, _ = storage_custody_fixture
    monkeypatch.setattr(exporter, "REPO_ROOT", tmp_path / "pruned-history")
    def forbidden_git(*args, **kwargs):
        raise AssertionError("explicit external cache attempted Git access")
    monkeypatch.setattr(exporter.subprocess, "run", forbidden_git)
    monkeypatch.setattr(exporter.subprocess, "check_output", forbidden_git)
    monkeypatch.setattr(exporter.subprocess, "Popen", forbidden_git)
    assert exporter._pinned_bytes(root, "/" + rel.as_posix(), _storage_sha(payload), len(payload), external_root=cache) == payload


def test_storage_git_recovery_checks_consumed_row_identity(storage_custody_fixture):
    root, _, rel, payload, _ = storage_custody_fixture
    assert exporter._pinned_bytes(root, "/" + rel.as_posix(), _storage_sha(payload), len(payload)) == payload


@pytest.mark.parametrize("field", ["recovery_commit", "git_path", "git_object_id"])
def test_storage_git_recovery_rejects_commit_path_or_object_disagreement(storage_custody_fixture, field):
    root, _, rel, payload, repo = storage_custody_fixture
    external_path = root / "EXTERNAL.json"
    document = json.loads(external_path.read_text())
    if field == "git_path":
        document["files"][0][field] += ".wrong"
    elif field == "recovery_commit":
        git_file = repo / document["files"][0]["git_path"]
        git_file.unlink()
        subprocess.run(["git", "-C", str(repo), "add", "-u"], check=True)
        subprocess.run([
            "git", "-C", str(repo), "-c", "user.name=Fixture", "-c",
            "user.email=fixture@example.invalid", "commit", "-qm", "remove recovery path",
        ], check=True)
        document[field] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    else:
        document["files"][0][field] = "0" * 40
    external_path.write_text(json.dumps(document))
    expected_message = {
        "recovery_commit": "pinned Git recovery commit/path unavailable",
        "git_path": "invalid external custody pins",
        "git_object_id": "external Git recovery object identity differs",
    }[field]
    with pytest.raises(exporter.ExportError, match=expected_message):
        exporter._pinned_bytes(root, "/" + rel.as_posix(), _storage_sha(payload), len(payload))


def test_storage_cli_forwards_external_root_without_deserializing(storage_custody_fixture, monkeypatch, capsys):
    root, cache, _, _, _ = storage_custody_fixture
    observed = {}
    monkeypatch.setattr(exporter, "load_inputs", lambda custody, external_root=None: observed.update(external_root=external_root) or {})
    monkeypatch.setattr(exporter, "read_json", lambda data: {})
    monkeypatch.setattr(exporter, "export_bundle", lambda **kwargs: {"bundle_digest": "ok"})
    monkeypatch.setattr(exporter.torch, "set_num_threads", lambda count: None)
    metadata_path = root / "metadata.json"
    metadata_path.write_text("{}")
    assert exporter.main([
        "--custody", str(root), "--external-root", str(cache), "--metadata", str(metadata_path), "--output", "out",
    ]) == 0
    assert observed["external_root"] == cache
    assert capsys.readouterr().out.strip() == "ok"


@pytest.mark.parametrize("mutation", [
    "same_length_hash_drift", "wrong_payload_length", "missing", "relative_root",
    "payload_symlink", "cache_ancestor_symlink", "cache_parent_symlink", "cache_root_symlink", "directory_payload",
])
def test_storage_explicit_cache_fails_closed(storage_custody_fixture, tmp_path, mutation):
    root, cache, rel, payload, _ = storage_custody_fixture
    digest = _storage_sha(payload)
    payload_path = cache / "sha256" / digest
    selected_root = cache
    expected_message = {
        "same_length_hash_drift": "hash/length",
        "wrong_payload_length": "size differs",
        "missing": "missing external cache payload",
        "relative_root": "must be absolute",
        "payload_symlink": "not a regular file",
        "cache_ancestor_symlink": "symlink in external cache path",
        "cache_parent_symlink": "symlink in external cache path",
        "cache_root_symlink": "symlink in external cache path",
        "directory_payload": "not a regular file",
    }[mutation]
    if mutation == "same_length_hash_drift":
        changed = b"S" + payload[1:]
        assert len(changed) == len(payload)
        payload_path.write_bytes(changed)
    elif mutation == "wrong_payload_length":
        payload_path.write_bytes(payload + b"!")
    elif mutation == "missing":
        payload_path.unlink()
    elif mutation == "relative_root":
        selected_root = Path("relative-cache")
    elif mutation == "payload_symlink":
        payload_path.unlink()
        payload_path.symlink_to(tmp_path / "elsewhere")
    elif mutation == "cache_ancestor_symlink":
        (cache / "sha256").rename(cache / "sha256-real")
        (cache / "sha256").symlink_to(cache / "sha256-real", target_is_directory=True)
    elif mutation == "cache_root_symlink":
        selected_root = tmp_path / "cache-link"
        selected_root.symlink_to(cache, target_is_directory=True)
    elif mutation == "cache_parent_symlink":
        linked_parent = tmp_path / "cache-parent-link"
        linked_parent.symlink_to(tmp_path, target_is_directory=True)
        selected_root = linked_parent / "cache"
    elif mutation == "directory_payload":
        payload_path.unlink()
        payload_path.mkdir()
    with pytest.raises(exporter.ExportError, match=expected_message):
        exporter._pinned_bytes(root, "/" + rel.as_posix(), digest, len(payload), external_root=selected_root)


def test_storage_external_cache_never_uses_stale_vendored_copy(storage_custody_fixture):
    root, cache, rel, payload, _ = storage_custody_fixture
    (cache / "sha256" / _storage_sha(payload)).unlink()
    assert (root / "files" / rel).read_bytes() == b"stale vendored copy"
    with pytest.raises(exporter.ExportError, match="missing external cache payload"):
        exporter._pinned_bytes(root, "/" + rel.as_posix(), _storage_sha(payload), len(payload), external_root=cache)


def test_storage_external_row_malformed_duplicate_or_pin_mismatch(storage_custody_fixture):
    root, cache, rel, payload, _ = storage_custody_fixture
    external_path = root / "EXTERNAL.json"
    original = json.loads(external_path.read_text())
    variants = []
    malformed = json.loads(json.dumps(original))
    malformed["files"] = [{"malformed": True}]
    variants.append(malformed)
    duplicate = json.loads(json.dumps(original))
    duplicate["files"] *= 2
    variants.append(duplicate)
    mismatch = json.loads(json.dumps(original))
    mismatch["files"][0]["bytes"] += 1
    variants.append(mismatch)
    for variant in variants:
        external_path.write_text(json.dumps(variant))
        with pytest.raises(exporter.ExportError, match="invalid external custody pins"):
            exporter._pinned_bytes(root, "/" + rel.as_posix(), _storage_sha(payload), len(payload), external_root=cache)


def test_storage_vendored_success_hash_drift_and_symlink_refusal(tmp_path):
    root = tmp_path / "custody"
    files = root / "files"
    asset = files / "source/asset.bin"
    asset.parent.mkdir(parents=True)
    pinned = b"pinned bytes"
    asset.write_bytes(pinned)
    _write_storage_metadata(root, "/source/asset.bin", pinned)
    assert exporter._pinned_bytes(root, "/source/asset.bin", _storage_sha(pinned), len(pinned)) == pinned
    asset.write_bytes(b"P" + pinned[1:])
    with pytest.raises(exporter.ExportError, match="hash/length"):
        exporter._pinned_bytes(root, "/source/asset.bin", _storage_sha(pinned), len(pinned))
    asset.unlink()
    asset.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(exporter.ExportError, match="symlink"):
        exporter._pinned_bytes(root, "/source/asset.bin", _storage_sha(pinned), len(pinned))


@pytest.mark.parametrize("symlink_at", ["custody_parent", "files_root", "files_ancestor"])
def test_storage_vendored_refuses_custody_files_symlink_traversal(tmp_path, symlink_at):
    actual = tmp_path / "custody-real"
    files = actual / "files"
    asset = files / "source/asset.bin"
    asset.parent.mkdir(parents=True)
    data = b"fixture bytes"
    asset.write_bytes(data)
    _write_storage_metadata(actual, "/source/asset.bin", data)
    if symlink_at == "custody_parent":
        custody = tmp_path / "custody-link"
        custody.symlink_to(actual, target_is_directory=True)
    elif symlink_at == "files_root":
        custody = actual
        real_files = actual / "files-real"
        files.rename(real_files)
        files.symlink_to(real_files, target_is_directory=True)
    else:
        custody = actual
        real_source = files / "source-real"
        (files / "source").rename(real_source)
        (files / "source").symlink_to(real_source, target_is_directory=True)
    with pytest.raises(exporter.ExportError, match="symlink"):
        exporter._pinned_bytes(custody, "/source/asset.bin", _storage_sha(data), len(data))
