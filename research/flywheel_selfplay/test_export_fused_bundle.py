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
    path.write_bytes(b"changed")
    with pytest.raises(exporter.ExportError, match="hash/length"):
        exporter._pinned_bytes(root, "/source/asset", "0" * 64)
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
