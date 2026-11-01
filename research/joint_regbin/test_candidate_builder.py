"""Synthetic contract tests for the JRB candidate tensor builder."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import struct

import numpy as np
import pytest
import torch

from research.joint_regbin import candidate_builder as builder


@pytest.fixture(autouse=True)
def rederive_synthetic_role_admission(monkeypatch):
    def rederive(manifest, *_args):
        if (
            manifest.get("admission_schema") != builder.admission.ROLE_SCHEMA
            or manifest.get("admission_status") != "ADMITTED"
            or manifest.get("eligibility_verified") is not True
        ):
            raise ValueError("incomplete JRB inputs cannot produce a runnable plan")
        return manifest

    monkeypatch.setattr(builder.admission, "rederive_role_manifest", rederive)
    monkeypatch.setattr(
        builder.admission,
        "verify_family_key_pool_manifest",
        lambda manifest, _catalog: {
            "catalog_sha256": manifest["catalog_sha256"],
            "eligibility_verified": False,
            "fold_count": 5,
            "nonsealed_eligibility": {"eligibility_verified": True},
            "payloads_opened": 0,
        },
    )


def _inputs(task="regression", role="train"):
    dataset = "fixture-dataset"
    family = "fixture-family"
    train_sha = "a" * 64
    family_pools = {
        "schema": builder.admission.FAMILY_KEY_POOL_SCHEMA,
        "status": "FAMILY_KEY_SPLITS_NOT_ADMITTED",
        "eligibility_verified": False,
        "catalog_sha256": builder.CATALOG_V246_SHA256,
        "train_metadata_by_dataset": {
            dataset: {"train_sha256": train_sha, "train_rows": 50, "task": task}
        },
        "family_by_dataset": {dataset: family},
        "fold_by_family": {family: 0},
        "fold_splits": {
            str(fold): {
                role_name: {
                    task_name: (
                        [dataset]
                        if task_name == task
                        and role_name == role
                        and (fold == 0 or role == "train")
                        else []
                    )
                    for task_name in builder.protocol.TASKS
                }
                for role_name in ("train", "val")
            }
            for fold in range(5)
        },
        "nonsealed_eligibility": {
            "status": "CATALOG_USABLE_NONSEALED_MEMBERSHIP_VERIFIED",
            "eligibility_verified": True,
            "verified_family_count": 1,
        },
    }
    context = {
        "id": "b" * 64,
        "dataset": dataset,
        "family": family,
        "task": task,
        "role": role,
        "source_split": "train",
        "origin": "real",
        "has_meta_test": False,
        "payload": f"{task}/{dataset}.npz",
        "payload_sha256": "c" * 64,
    }
    role_manifest = {
        "schema": builder.protocol.SCHEMA,
        "admission_schema": builder.admission.ROLE_SCHEMA,
        "admission_status": "ADMITTED",
        "eligibility_verified": True,
        "catalog_sha256": builder.CATALOG_V246_SHA256,
        "contexts": [context],
        "family_by_dataset": {dataset: family},
        "evidence": {"data": {}},
    }
    catalog_row = {
        "id": dataset,
        "task": task,
        "s3_bucket": "veox-jopedime-datasets-use2",
        "train_sha256": train_sha,
        "train_rows": 50,
        "cols": 3,
    }
    return family_pools, role_manifest, context, catalog_row


class _Body:
    def __init__(self, body: bytes):
        self.body = io.BytesIO(body)

    def read(self):
        return self.body.read()


class _S3:
    def __init__(self, manifest: bytes, train: bytes):
        self.manifest = manifest
        self.train = train
        self.requests = []

    def get_object(self, *, Bucket, Key):
        self.requests.append((Bucket, Key))
        if Key.endswith("/manifest.json"):
            body = self.manifest
        elif Key == f"blobs/sha256/{hashlib.sha256(self.train).hexdigest()}":
            body = self.train
        else:
            raise AssertionError(f"unexpected S3 read: {Key}")
        return {"Body": _Body(body)}


def test_train_context_binds_rederived_admission_and_source_split():
    pools, roles, context, catalog_row = _inputs()
    identity = builder.validate_catalog_train_context(
        pools, roles, context, catalog_row
    )
    assert identity["task"] == "regression"
    assert identity["train_sha256"] == "a" * 64
    assert identity["admission_status"] == "ADMITTED"
    assert identity["eligibility_verified"] is True
    context["source_split"] = "test"
    with pytest.raises(ValueError):
        builder.validate_catalog_train_context(pools, roles, context, catalog_row)


def test_catalog_family_folds_still_require_complete_role_admission():
    pools, role_manifest, context, catalog_row = _inputs()
    assert (
        builder.validate_catalog_train_context(
            pools, role_manifest, context, catalog_row
        )["task"]
        == context["task"]
    )
    role_manifest["admission_status"] = "BLOCKED_INCOMPLETE_EVIDENCE"
    role_manifest["eligibility_verified"] = False
    s3 = _S3(b"{}", b"")
    with pytest.raises(ValueError, match="incomplete JRB inputs"):
        builder.validate_catalog_train_context(
            pools, role_manifest, context, catalog_row
        )
    assert s3.requests == []


@pytest.mark.parametrize(
    "field,value", [("role", "panel"), ("has_meta_test", True), ("origin", "synthetic")]
)
def test_train_context_rejects_panel_protected_and_synthetic_rows(field, value):
    pools, roles, context, catalog_row = _inputs()
    context[field] = value
    with pytest.raises(ValueError):
        builder.validate_catalog_train_context(pools, roles, context, catalog_row)


def test_s3_reads_exact_manifest_and_sha_addressed_train_blob_only():
    pools, roles, context, catalog_row = _inputs()
    csv_bytes = b"0,1,0\n1,0,1\n" + b"0,1,0\n" * 48
    train_sha = hashlib.sha256(csv_bytes).hexdigest()
    catalog_row["train_sha256"] = train_sha
    pools["train_metadata_by_dataset"][context["dataset"]]["train_sha256"] = train_sha
    identity = builder.validate_catalog_train_context(
        pools, roles, context, catalog_row
    )
    manifest = {
        "id": identity["dataset"],
        "task": identity["task"],
        "files": {"train": {"sha256": train_sha, "rows": 50, "columns": 3}},
        "target": {"index": 2},
    }
    s3 = _S3(json.dumps(manifest).encode(), csv_bytes)
    actual_manifest, actual_train = builder.fetch_train_csv(
        s3, "veox-jopedime-datasets-use2", identity
    )
    assert actual_manifest == manifest
    assert actual_train == csv_bytes
    assert s3.requests == [
        (
            "veox-jopedime-datasets-use2",
            f"datasets/{identity['dataset']}/manifest.json",
        ),
        ("veox-jopedime-datasets-use2", f"blobs/sha256/{train_sha}"),
    ]


def test_s3_reader_rejects_unvalidated_metadata_candidates_before_request():
    s3 = _S3(b"{}", b"")
    with pytest.raises(ValueError, match="fresh role-admission validation"):
        builder.fetch_train_csv(
            s3,
            "veox-jopedime-datasets-use2",
            {"dataset": "metadata-only"},
        )
    assert s3.requests == []


def test_binary_labels_are_explicitly_normalized_and_regression_is_unchanged():
    binary = {
        "id": "fixture-dataset",
        "task": "binary",
        "files": {"train": {"rows": 50, "columns": 3}},
        "target": {"index": 2},
    }
    raw = b"".join(f"{i},{i + 1},{1 if i % 2 else 2}\n".encode() for i in range(50))
    x, y, metadata = builder.parse_train_csv(
        raw,
        binary,
        {"dataset": "fixture-dataset", "task": "binary", "train_rows": 50},
    )
    assert set(np.unique(y)) == {0.0, 1.0}
    assert metadata["binary_label_mapping"] == {"1.0": 0, "2.0": 1}

    regression = {
        "id": "fixture-dataset",
        "task": "regression",
        "files": {"train": {"rows": 50, "columns": 3}},
        "target": {"index": 2},
    }
    x_reg, y_reg, metadata_reg = builder.parse_train_csv(
        raw,
        regression,
        {"dataset": "fixture-dataset", "task": "regression", "train_rows": 50},
    )
    assert np.array_equal(
        y_reg, np.asarray([2.0 if i % 2 == 0 else 1.0 for i in range(50)])
    )
    assert metadata_reg["binary_label_mapping"] is None
    assert x.shape == x_reg.shape == (50, 2)


def test_two_fold_jtt1_is_deterministic_stratified_and_train_only():
    x = np.arange(100, dtype=np.float64).reshape(50, 2)
    y = np.asarray([float(i % 2) for i in range(50)])
    first = builder.encode_two_fold_jtt1(
        x, y, task="binary", dataset="fixture", seed=20261029
    )
    second = builder.encode_two_fold_jtt1(
        x, y, task="binary", dataset="fixture", seed=20261029
    )
    assert first == second
    raw = gzip.decompress(first)
    n, n_test, columns, folds_count = struct.unpack("<IIIB", raw[4:17])
    assert (n, n_test, columns, folds_count) == (50, 0, 2, 2)
    fold_ids = np.frombuffer(raw, dtype=np.uint8, count=n, offset=17)
    targets = np.frombuffer(
        raw, dtype="<f8", count=n * (columns + 1), offset=17 + n
    ).reshape(n, columns + 1)[:, -1]
    assert np.bincount(fold_ids).tolist() == [25, 25]
    for label in (0.0, 1.0):
        assert np.bincount(fold_ids[targets == label]).tolist() == [
            13,
            12,
        ] or np.bincount(fold_ids[targets == label]).tolist() == [12, 13]


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_candidate_pool_uses_existing_features_and_task_specific_targets(task):
    identity = {
        "context_id": "b" * 64,
        "dataset": "fixture",
        "family": "family",
        "task": task,
        "role": "train",
        "source_split": "train",
        "origin": "real",
        "has_meta_test": False,
        "train_sha256": "d" * 64,
        "catalog_sha256": builder.CATALOG_V246_SHA256,
        "admission_status": "ADMITTED",
        "eligibility_verified": True,
        "role_admission_sha256": "e" * 64,
    }
    x = np.arange(100, dtype=np.float64).reshape(50, 2)
    y = (
        np.asarray([float(i % 2) for i in range(50)])
        if task == "binary"
        else np.arange(50, dtype=np.float64)
    )
    calls = {}

    def propose(parents, landmark, champs, family, *, n_proposals, seed):
        assert parents == [] and champs == [] and family == "family"
        assert landmark.shape == (48,)
        calls["seed"] = seed
        return [
            {
                "dsl": f"(pipe X model_{i})",
                "program_sha256": hashlib.sha256(
                    f"(pipe X model_{i})".encode()
                ).hexdigest(),
            }
            for i in range(n_proposals)
        ]

    def evaluate(_jtt, proposals):
        return [
            {
                "dsl": row["dsl"],
                "program_sha256": row["program_sha256"],
                "status": "completed",
                "fitness": float(index),
                "cv_gate_passed": index % 2 == 0,
            }
            for index, row in enumerate(proposals)
        ]

    pool = builder.build_candidate_pool(
        identity,
        x,
        y,
        seed=20261029,
        landmark_fn=lambda _jtt: np.ones(48, dtype=np.float32),
        propose_fn=propose,
        evaluate_fn=evaluate,
        featurize_fn=lambda dsl: (
            np.full(412, len(dsl), dtype=np.float32),
            np.zeros(16, dtype=np.float64),
        ),
    )
    assert pool["eligibility_verified"] is False
    assert pool["candidate_count"] == builder.PROPOSALS_PER_CONTEXT
    if task == "binary":
        assert pool["task_target"] == "s1_cv_gate_passed"
        assert set(pool["target"].tolist()) == {0.0, 1.0}
    else:
        assert pool["task_target"] == "s1_fitness"
        assert pool["target"].tolist() == pool["fitness"].tolist()
    tokens = torch.ones((10, 32))
    mask = torch.ones(10)
    inputs, targets = builder.pack_worker_inputs(
        identity,
        pool,
        tokens,
        mask,
        checkpoint_sha256=builder.R882_CHECKPOINT_SHA256,
        dope_sha256=builder.DOPE_SHA256,
    )
    assert inputs["prog_412"].shape == (1, 16, 412)
    assert inputs["proj_16"].shape == (1, 16, 16)
    assert inputs["proj_16"].dtype == torch.float64
    assert inputs["revealed_indices"].shape == (1, 0)
    assert targets.shape == (1, 16)
    packer = builder.make_worker_packer(
        {identity["context_id"]: pool},
        {identity["context_id"]: tokens},
        {identity["context_id"]: mask},
        checkpoint_sha256=builder.R882_CHECKPOINT_SHA256,
        dope_sha256=builder.DOPE_SHA256,
    )
    packed_inputs, packed_target = packer(
        {
            "id": identity["context_id"],
            **{
                key: identity[key]
                for key in (
                    "dataset",
                    "family",
                    "task",
                    "role",
                    "source_split",
                    "origin",
                    "has_meta_test",
                )
            },
        }
    )
    assert packed_inputs["mm_tokens_10x32"].shape == (1, 10, 32)
    assert packed_target.shape == (1, 16)
    with pytest.raises(ValueError, match="pinned read-only R882"):
        builder.pack_worker_inputs(
            identity,
            pool,
            tokens,
            mask,
            checkpoint_sha256="0" * 64,
            dope_sha256=builder.DOPE_SHA256,
        )


def test_candidate_pool_rejects_incomplete_evaluation_and_unpinned_checkpoint():
    identity = {
        "dataset": "fixture",
        "family": "family",
        "task": "binary",
        "role": "train",
        "source_split": "train",
        "origin": "real",
        "has_meta_test": False,
        "train_sha256": "d" * 64,
        "catalog_sha256": builder.CATALOG_V246_SHA256,
        "admission_status": "ADMITTED",
        "eligibility_verified": True,
        "role_admission_sha256": "e" * 64,
    }
    with pytest.raises(ValueError, match="evaluation result count mismatch"):
        builder.build_candidate_pool(
            identity,
            np.ones((50, 2)),
            np.asarray([float(i % 2) for i in range(50)]),
            seed=20261029,
            landmark_fn=lambda _jtt: np.ones(48),
            propose_fn=lambda *_args, **_kwargs: (
                [{"dsl": "(pipe X bad)", "program_sha256": "a" * 64}] * 16
            ),
            evaluate_fn=lambda *_args, **_kwargs: [],
            featurize_fn=lambda _dsl: (np.ones(412), np.ones(16)),
        )
