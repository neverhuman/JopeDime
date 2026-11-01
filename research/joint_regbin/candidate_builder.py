"""Train-only candidate tensor construction for the JRB-01 worker.

This module binds catalog train rows to the existing Flywheel proposal and
two-fold S1 evaluation path. It creates candidate labels for the worker; it
does not admit a study packet, run training, or write score history.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import struct
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import admission, protocol

CATALOG_V246_SHA256 = "e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba"
R882_CHECKPOINT_SHA256 = (
    "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
)
DOPE_SHA256 = protocol.DOPE
FLYWHEEL_EVALUATOR_SHA256 = (
    "7b0cb0ecb98c7836b16a781f632a9fc74683c2ab9facee820790ceebc759cb45"
)
FLYWHEEL_LANDMARK_SHA256 = (
    "c54dac453b762490f51089f74777fcebc982a3d8c94c4f6cf68170daa0fba0f3"
)
NATIVE_S1_SHA256 = "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"
RANK_PREPARE_SHA256 = "7aab09c5346f8a2ec54fae244c3ebd650e3876cb4d85a05616f67e45f7e1304b"
PROGRAM_FEATURE_NAMES_SHA256 = (
    "c4e4373d1bc7e99bc38e44eb4acda22d0f9ecc9e332fa8cb9c6a0b463fe37962"
)
POOL_SCHEMA = "jope.joint-regbin.candidate-tensor/v1"
PROPOSALS_PER_CONTEXT = 16


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_catalog_train_context(
    family_pools: dict,
    role_manifest: dict,
    context: dict,
    catalog_row: dict,
    *,
    catalog_path: Path | None = None,
    family_exclusion_path: Path | None = None,
    family_exclusion_sha256: str | None = None,
) -> dict:
    """Bind a train-only context to rederived role admission and family identity.

    Catalog-certified family folds provide candidate roles only. A separately
    rederived ADMITTED role manifest remains mandatory before a train blob can
    be fetched.
    """
    _require(
        family_pools.get("schema") == admission.FAMILY_KEY_POOL_SCHEMA
        and family_pools.get("status") == "FAMILY_KEY_SPLITS_NOT_ADMITTED"
        and family_pools.get("eligibility_verified") is False,
        "pinned non-admitted family-key pool manifest required",
    )
    _require(
        family_pools.get("catalog_sha256") == CATALOG_V246_SHA256,
        "family-key pool catalog pin mismatch",
    )
    expected_exclusion_sha = (
        family_exclusion_sha256 or admission.FAMILY_EXCLUSION_SHA256
    )
    _require(
        family_pools.get("family_exclusion_sha256") == expected_exclusion_sha
        and expected_exclusion_sha == admission.FAMILY_EXCLUSION_SHA256,
        "family-key pool exclusion artifact pin mismatch",
    )
    _require(
        family_pools.get("fold_splits")
        and family_pools.get("fold_by_family")
        and family_pools.get("nonsealed_eligibility", {}).get("status")
        == "CATALOG_USABLE_NONSEALED_MEMBERSHIP_VERIFIED"
        and family_pools.get("nonsealed_eligibility", {}).get("eligibility_verified")
        is True
        and family_pools.get("nonsealed_eligibility", {}).get("verified_family_count")
        > 0,
        "catalog-certified family folds required",
    )
    source_catalog = catalog_path or admission.DEFAULT_ROLE_CATALOG
    verified_pool = admission.verify_family_key_pool_manifest(
        family_pools,
        source_catalog,
        family_exclusion_path=family_exclusion_path,
        family_exclusion_sha256=expected_exclusion_sha,
    )
    _require(
        verified_pool.get("catalog_sha256") == CATALOG_V246_SHA256
        and verified_pool.get("eligibility_verified") is False
        and verified_pool.get("family_exclusion_sha256") == expected_exclusion_sha
        and verified_pool.get("fold_count") == 5
        and verified_pool.get("nonsealed_eligibility", {}).get("eligibility_verified")
        is True
        and verified_pool.get("payloads_opened") == 0,
        "family-key pool did not rederive as a metadata-only catalog split",
    )
    if catalog_path is None:
        admitted = admission.rederive_role_manifest(role_manifest)
    else:
        admitted = admission.rederive_role_manifest(role_manifest, catalog_path)
    _require(
        admitted.get("admission_status") == "ADMITTED"
        and admitted.get("eligibility_verified") is True
        and admitted.get("catalog_sha256") == CATALOG_V246_SHA256,
        "complete admitted role manifest required; unverified families are withheld",
    )
    required = {
        "id",
        "dataset",
        "family",
        "task",
        "role",
        "source_split",
        "origin",
        "has_meta_test",
        "payload",
        "payload_sha256",
    }
    _require(
        isinstance(context, dict) and set(context) == required,
        "JRB context fields mismatch",
    )
    admitted_context = next(
        (row for row in admitted.get("contexts", []) if row.get("id") == context["id"]),
        None,
    )
    _require(
        admitted_context == context,
        "context is not in the rederived admitted role manifest",
    )
    dataset = context["dataset"]
    task = context["task"]
    _require(
        task in protocol.TASKS,
        "candidate context task must be explicit regression/binary",
    )
    _require(
        context["role"] in ("train", "val"), "candidate builder accepts TRAIN/VAL only"
    )
    _require(
        context["source_split"] == "train", "candidate builder accepts train CSVs only"
    )
    _require(
        context["origin"] == "real" and context["has_meta_test"] is False,
        "protected or synthetic context refused",
    )
    protocol.relative_payload(context["payload"])
    protocol.check_sha(context["payload_sha256"])
    _require(catalog_row.get("id") == dataset, "catalog row identity mismatch")
    _require(catalog_row.get("task") == task, "catalog task mismatch")
    _require(
        catalog_row.get("s3_bucket") == "veox-jopedime-datasets-use2",
        "unapproved dataset bucket",
    )
    pool_row = family_pools.get("train_metadata_by_dataset", {}).get(dataset, {})
    _require(
        catalog_row.get("train_sha256") == pool_row.get("train_sha256")
        and catalog_row.get("train_rows") == pool_row.get("train_rows")
        and catalog_row.get("task") == pool_row.get("task"),
        "catalog train metadata differs from the pinned family pool",
    )
    family = family_pools.get("family_by_dataset", {}).get(dataset)
    _require(
        isinstance(family, str) and context["family"] == family,
        "family identity mismatch",
    )
    _require(
        admitted.get("family_by_dataset", {}).get(dataset) == family,
        "admitted role family differs from pinned family identity",
    )
    _require(
        any(
            dataset in split.get(context["role"], {}).get(task, [])
            and family_pools.get("fold_by_family", {}).get(family) is not None
            for split in family_pools.get("fold_splits", {}).values()
        ),
        "context role is outside catalog-certified family folds",
    )
    _require(
        catalog_row.get("cols", 0) > 1, "catalog feature/target column count invalid"
    )
    _require(
        type(catalog_row.get("train_rows")) is int and catalog_row["train_rows"] >= 50,
        "train rows below S1 minimum",
    )
    return {
        "context_id": context["id"],
        "dataset": dataset,
        "task": task,
        "family": family,
        "role": context["role"],
        "source_split": context["source_split"],
        "origin": context["origin"],
        "has_meta_test": context["has_meta_test"],
        "train_rows": catalog_row["train_rows"],
        "columns": catalog_row["cols"],
        "train_sha256": catalog_row["train_sha256"],
        "catalog_sha256": admitted["catalog_sha256"],
        "role_admission_sha256": sha256_bytes(
            json.dumps(admitted, sort_keys=True, separators=(",", ":")).encode()
        ),
        "admission_status": "ADMITTED",
        "eligibility_verified": True,
        "_role_manifest": role_manifest,
        "_context": context,
        "_catalog_row": catalog_row,
        "_family_pools": family_pools,
        "_catalog_path": catalog_path,
        "_family_exclusion_path": family_exclusion_path,
        "_family_exclusion_sha256": expected_exclusion_sha,
    }


def fetch_train_csv(
    s3_client, bucket: str, context_identity: dict
) -> tuple[dict, bytes]:
    """Read only the exact dataset manifest and catalog-pinned train blob."""
    _require(
        all(
            key in context_identity
            for key in (
                "_family_pools",
                "_role_manifest",
                "_context",
                "_catalog_row",
                "_catalog_path",
            )
        ),
        "fresh role-admission validation required before S3 read",
    )
    _require(bucket == "veox-jopedime-datasets-use2", "unapproved S3 bucket")
    rechecked = validate_catalog_train_context(
        context_identity["_family_pools"],
        context_identity["_role_manifest"],
        context_identity["_context"],
        context_identity["_catalog_row"],
        catalog_path=context_identity["_catalog_path"],
        family_exclusion_path=context_identity["_family_exclusion_path"],
        family_exclusion_sha256=context_identity["_family_exclusion_sha256"],
    )
    _require(
        rechecked.get("role_admission_sha256")
        == context_identity.get("role_admission_sha256"),
        "role admission identity changed before S3 read",
    )
    for key in (
        "context_id",
        "dataset",
        "task",
        "family",
        "role",
        "source_split",
        "origin",
        "has_meta_test",
        "train_rows",
        "columns",
        "train_sha256",
        "catalog_sha256",
    ):
        _require(
            rechecked.get(key) == context_identity.get(key),
            f"validated train identity changed before S3 read: {key}",
        )
    _require(
        context_identity.get("catalog_sha256") == CATALOG_V246_SHA256
        and context_identity.get("admission_status") == "ADMITTED"
        and context_identity.get("eligibility_verified") is True
        and len(context_identity.get("role_admission_sha256", "")) == 64
        and context_identity.get("role") in ("train", "val")
        and context_identity.get("source_split") == "train"
        and context_identity.get("origin") == "real"
        and context_identity.get("has_meta_test") is False,
        "only pinned nonsealed TRAIN/VAL identities may read train CSVs",
    )
    dataset = context_identity["dataset"]
    train_sha = context_identity["train_sha256"]
    _require(
        len(train_sha) == 64 and all(c in "0123456789abcdef" for c in train_sha),
        "invalid train SHA",
    )
    manifest_key = f"datasets/{dataset}/manifest.json"
    manifest_bytes = s3_client.get_object(Bucket=bucket, Key=manifest_key)[
        "Body"
    ].read()
    manifest = json.loads(manifest_bytes)
    _require(
        manifest.get("id") == dataset
        and manifest.get("task") == context_identity["task"],
        "S3 manifest identity mismatch",
    )
    train_meta = manifest.get("files", {}).get("train")
    _require(isinstance(train_meta, dict), "S3 train manifest entry missing")
    _require(train_meta.get("sha256") == train_sha, "S3 train manifest hash mismatch")
    _require(
        train_meta.get("rows") == context_identity["train_rows"],
        "S3 train row count mismatch",
    )
    _require(
        train_meta.get("columns") == context_identity["columns"],
        "S3 train column count mismatch",
    )
    _require(
        manifest.get("target", {}).get("index") is not None, "S3 target index missing"
    )
    train_key = f"blobs/sha256/{train_sha}"
    train_bytes = s3_client.get_object(Bucket=bucket, Key=train_key)["Body"].read()
    _require(sha256_bytes(train_bytes) == train_sha, "S3 train blob hash mismatch")
    return manifest, train_bytes


def parse_train_csv(
    train_bytes: bytes, manifest: dict, context_identity: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decode numeric train rows only; normalize explicit binary labels to 0/1."""
    _require(
        manifest.get("id") == context_identity.get("dataset")
        and manifest.get("task") == context_identity.get("task"),
        "train manifest/context identity mismatch",
    )
    train_meta = manifest["files"]["train"]
    expected_rows = context_identity["train_rows"]
    expected_cols = train_meta["columns"]
    target_index = manifest["target"]["index"]
    _require(
        type(target_index) is int and 0 <= target_index < expected_cols,
        "target index outside train columns",
    )
    rows = []
    for row in csv.reader(io.StringIO(train_bytes.decode("utf-8"))):
        _require(len(row) == expected_cols, "unexpected train CSV column count")
        try:
            rows.append([float(value) for value in row])
        except ValueError as exc:
            raise ValueError("non-numeric value in train CSV") from exc
    _require(len(rows) == expected_rows, "unexpected train CSV row count")
    table = np.asarray(rows, dtype=np.float64)
    _require(np.isfinite(table).all(), "non-finite train CSV value")
    y = table[:, target_index]
    x = np.delete(table, target_index, axis=1)
    _require(0 < x.shape[1] <= 512, "S1 feature width outside 1..512")
    label_mapping = None
    if context_identity["task"] == "binary":
        classes = np.unique(y)
        _require(
            len(classes) == 2,
            "binary catalog task must contain exactly two target classes",
        )
        label_mapping = {repr(float(classes[0])): 0, repr(float(classes[1])): 1}
        y = (y == classes[1]).astype(np.float64)
    return (
        np.ascontiguousarray(x),
        np.ascontiguousarray(y),
        {"binary_label_mapping": label_mapping},
    )


def encode_two_fold_jtt1(
    x: np.ndarray, y: np.ndarray, *, task: str, dataset: str, seed: int
) -> bytes:
    """Create a deterministic, stratified two-fold JTT1 from train rows only."""
    _require(
        x.ndim == 2 and y.ndim == 1 and len(x) == len(y), "train matrix shape mismatch"
    )
    _require(
        50 <= len(y) < 2000 and 1 <= x.shape[1] <= 512, "train matrix outside S1 bounds"
    )
    _require(np.isfinite(x).all() and np.isfinite(y).all(), "non-finite train matrix")
    _require(
        task in protocol.TASKS,
        "explicit regression/binary task required for fold construction",
    )
    folds = np.empty(len(y), dtype=np.uint8)
    if task == "binary":
        _require(
            set(np.unique(y).tolist()) == {0.0, 1.0},
            "binary train target must be explicitly normalized to zero/one",
        )
        for class_index, label in enumerate((0.0, 1.0)):
            ids = np.flatnonzero(y == label)
            _require(
                len(ids) >= 2, "binary target class cannot populate both inner folds"
            )
            order = sorted(
                ids.tolist(),
                key=lambda i: hashlib.sha256(f"{seed}:{dataset}:{i}".encode()).digest(),
            )
            for offset, index in enumerate(order):
                folds[index] = (offset + class_index) % 2
    else:
        order = sorted(
            range(len(y)),
            key=lambda i: hashlib.sha256(f"{seed}:{dataset}:{i}".encode()).digest(),
        )
        for offset, index in enumerate(order):
            folds[index] = offset % 2
    _require(
        min(np.bincount(folds, minlength=2)) >= 25, "inner fold has fewer than 25 rows"
    )
    table = np.column_stack((x, y)).astype("<f8", copy=False)
    raw = (
        b"JTT1"
        + struct.pack("<IIIB", len(y), 0, x.shape[1], 2)
        + folds.tobytes()
        + table.tobytes()
    )
    _require(
        len(raw) == 17 + len(y) + 8 * len(y) * (x.shape[1] + 1),
        "JTT1 encoding size mismatch",
    )
    return gzip.compress(raw, mtime=0)


def build_candidate_pool(
    context_identity: dict,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    landmark_fn: Callable[[bytes], np.ndarray],
    propose_fn: Callable[..., list[dict]],
    evaluate_fn: Callable[..., list[dict]],
    featurize_fn: Callable[[str], tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Build task-labeled candidates with injected, source-pinned Flywheel APIs."""
    _require(
        context_identity.get("task") in protocol.TASKS, "explicit catalog task required"
    )
    _require(
        context_identity.get("admission_status") == "ADMITTED"
        and context_identity.get("eligibility_verified") is True
        and len(context_identity.get("role_admission_sha256", "")) == 64,
        "candidate labels require a rederived admitted role manifest",
    )
    _require(
        context_identity.get("role") in ("train", "val"), "TRAIN/VAL role required"
    )
    _require(
        context_identity.get("source_split") == "train",
        "candidate labels must come from train CSV rows",
    )
    _require(
        context_identity.get("origin") == "real"
        and context_identity.get("has_meta_test") is False,
        "protected or synthetic candidate context refused",
    )
    if context_identity["task"] == "binary":
        _require(
            set(np.unique(y).tolist()) == {0.0, 1.0},
            "binary target must be normalized by the explicit catalog task parser",
        )
    jtt1_gz = encode_two_fold_jtt1(
        x,
        y,
        task=context_identity["task"],
        dataset=context_identity["dataset"],
        seed=seed,
    )
    _require(
        isinstance(context_identity.get("train_sha256"), str)
        and len(context_identity["train_sha256"]) == 64
        and isinstance(context_identity.get("catalog_sha256"), str)
        and len(context_identity["catalog_sha256"]) == 64,
        "candidate pool requires pinned catalog/train identities",
    )
    landmark = np.asarray(landmark_fn(jtt1_gz), dtype=np.float32)
    _require(
        landmark.shape == (48,) and np.isfinite(landmark).all(),
        "Flywheel Landmark-V2 feature shape/value mismatch",
    )
    proposals = propose_fn(
        [],
        landmark,
        [],
        context_identity["family"],
        n_proposals=PROPOSALS_PER_CONTEXT,
        seed=seed,
    )
    _require(
        len(proposals) == PROPOSALS_PER_CONTEXT, "Flywheel proposal count mismatch"
    )
    _require(
        all(
            isinstance(row.get("dsl"), str) and row["dsl"].startswith("(pipe X ")
            for row in proposals
        ),
        "invalid Flywheel candidate DSL",
    )
    results = evaluate_fn(jtt1_gz, proposals)
    _require(
        len(results) == len(proposals), "Flywheel evaluation result count mismatch"
    )
    dsl_by_sha = {row.get("program_sha256"): row["dsl"] for row in proposals}
    program_rows, projection_rows, fitness_rows, gate_rows, target_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    for proposal, result in zip(proposals, results, strict=True):
        program_sha = proposal.get("program_sha256")
        _require(
            result.get("program_sha256") == program_sha
            and result.get("dsl", proposal["dsl"]) == dsl_by_sha[program_sha],
            "Flywheel result identity mismatch",
        )
        _require(result.get("status") == "completed", "incomplete candidate evaluation")
        feat412, proj16 = featurize_fn(proposal["dsl"])
        feat412 = np.asarray(feat412, dtype=np.float32)
        # The adapter consumes this W1 projection as float64 for parity.
        proj16 = np.asarray(proj16, dtype=np.float64)
        _require(
            feat412.shape == (412,) and proj16.shape == (16,),
            "Flywheel candidate feature shape mismatch",
        )
        _require(
            np.isfinite(feat412).all() and np.isfinite(proj16).all(),
            "non-finite Flywheel candidate features",
        )
        fitness = result.get("fitness")
        gate = result.get("cv_gate_passed")
        _require(
            type(fitness) in (int, float) and math.isfinite(float(fitness)),
            "invalid Flywheel constrained fitness",
        )
        _require(type(gate) is bool, "invalid Flywheel two-fold gate label")
        program_rows.append(feat412)
        projection_rows.append(proj16)
        fitness_rows.append(float(fitness))
        gate_rows.append(gate)
        # Binary BCE learns the existing, explicit S1 gate outcome, not an
        # inferred label or a new AUC threshold. Regression retains S1 fitness.
        target_rows.append(
            float(gate) if context_identity["task"] == "binary" else float(fitness)
        )
    targets = np.asarray(target_rows, dtype=np.float32)
    _require(np.isfinite(targets).all(), "non-finite candidate targets")
    return {
        "schema": POOL_SCHEMA,
        "status": "CANDIDATE_TENSORS_NOT_ADMITTED",
        "eligibility_verified": False,
        "qualification_status": "NOT_QUALIFIED",
        "context": dict(context_identity),
        "seed": seed,
        "candidate_count": len(proposals),
        "candidate_sha256": [row["program_sha256"] for row in proposals],
        "task_target": "s1_fitness"
        if context_identity["task"] == "regression"
        else "s1_cv_gate_passed",
        "train_jtt1_sha256": sha256_bytes(jtt1_gz),
        "program": np.stack(program_rows),
        "proj16": np.stack(projection_rows),
        "fitness": np.asarray(fitness_rows, dtype=np.float32),
        "gate": np.asarray(gate_rows, dtype=np.bool_),
        "target": targets,
        "proposal_source_sha256": FLYWHEEL_EVALUATOR_SHA256,
        "landmark_source_sha256": FLYWHEEL_LANDMARK_SHA256,
        "native_s1_sha256": NATIVE_S1_SHA256,
        "blockers": [
            "candidate tensor cache is not an admitted study payload",
            "two-task worker/checkpoint qualification remains required",
        ],
    }


def flywheel_callbacks(
    cache_root: Path,
) -> tuple[Callable, Callable, Callable, Callable]:
    """Load the exact existing Flywheel proposal, Landmark-V2, and S1 code."""
    root = Path(__file__).resolve().parents[2]
    evaluator_path = root / "research/flywheel_selfplay/rust_evaluator.py"
    landmark_path = root / "research/flywheel_selfplay/dataset_builder.py"
    native_path = Path(
        "/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s1-prod/binaries/bootstrap.s1main"
    )
    corrected_study = Path(
        "/home/ubuntu/jope-evidence/dope-lift-20261021/corrected-study"
    )
    rank_prepare_path = corrected_study / "rank_prepare.py"
    feature_names_path = corrected_study / "program-feature-names.json"
    _require(
        sha256_file(evaluator_path) == FLYWHEEL_EVALUATOR_SHA256,
        "Flywheel evaluator source pin mismatch",
    )
    _require(
        sha256_file(landmark_path) == FLYWHEEL_LANDMARK_SHA256,
        "Flywheel Landmark-V2 source pin mismatch",
    )
    _require(
        sha256_file(native_path) == NATIVE_S1_SHA256, "native S1 evaluator pin mismatch"
    )
    _require(
        sha256_file(rank_prepare_path) == RANK_PREPARE_SHA256,
        "DSL feature source pin mismatch",
    )
    _require(
        sha256_file(feature_names_path) == PROGRAM_FEATURE_NAMES_SHA256,
        "DSL feature names pin mismatch",
    )
    import importlib.util
    import sys

    evaluator_spec = importlib.util.spec_from_file_location(
        "jrb01_flywheel_evaluator", evaluator_path
    )
    landmark_spec = importlib.util.spec_from_file_location(
        "jrb01_flywheel_landmarks", landmark_path
    )
    _require(
        evaluator_spec is not None and evaluator_spec.loader is not None,
        "Flywheel evaluator import failed",
    )
    _require(
        landmark_spec is not None and landmark_spec.loader is not None,
        "Flywheel Landmark-V2 import failed",
    )
    evaluator = importlib.util.module_from_spec(evaluator_spec)
    landmark_module = importlib.util.module_from_spec(landmark_spec)
    sys.modules[evaluator_spec.name] = evaluator
    sys.modules[landmark_spec.name] = landmark_module
    evaluator_spec.loader.exec_module(evaluator)
    landmark_spec.loader.exec_module(landmark_module)
    cache_root = Path(cache_root).resolve()
    repo = Path(__file__).resolve().parents[2]
    _require(
        cache_root.is_absolute()
        and not cache_root.is_relative_to(repo)
        and not cache_root.is_relative_to(Path("/tmp")),
        "JRB cache must remain outside checkout and /tmp",
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    scratch_root = cache_root / "rust-evaluator-scratch"
    _require(
        not scratch_root.is_symlink(), "Flywheel scratch path must not be a symlink"
    )
    evaluator.EVAL_SCRATCH_ROOT = scratch_root

    def landmark_fn(jtt1_gz: bytes) -> np.ndarray:
        path = cache_root / f"landmark-{sha256_bytes(jtt1_gz)}.jtt1.gz"
        _require(not path.is_symlink(), "train-only Landmark cache cannot be a symlink")
        if path.exists():
            _require(
                path.is_file() and sha256_file(path) == sha256_bytes(jtt1_gz),
                "cached train-only JTT1 identity mismatch",
            )
        else:
            with path.open("xb") as stream:
                stream.write(jtt1_gz)
                stream.flush()
        _cid, values = landmark_module.compute_landmark_v2_48d((path.stem, str(path)))
        return values

    def propose_fn(parents, landmark, champions, family, *, n_proposals, seed):
        return evaluator.propose_flywheel_candidates(
            parents, landmark, champions, family, n_proposals=n_proposals, seed=seed
        )

    def evaluate_fn(jtt1_gz: bytes, proposals: list[dict]) -> list[dict]:
        path = cache_root / f"s1-{sha256_bytes(jtt1_gz)}.jtt1.gz"
        _require(not path.is_symlink(), "train-only S1 cache cannot be a symlink")
        if path.exists():
            _require(
                path.is_file() and sha256_file(path) == sha256_bytes(jtt1_gz),
                "cached S1 JTT1 identity mismatch",
            )
        else:
            with path.open("xb") as stream:
                stream.write(jtt1_gz)
                stream.flush()
        return evaluator.evaluate_gp_batch_rust(
            str(path), proposals, run_outer_test=False, max_workers=1
        )

    return landmark_fn, propose_fn, evaluate_fn, evaluator.featurize_dsl_412


def pack_worker_inputs(
    context_identity: dict,
    pool: dict,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    checkpoint_sha256: str,
    dope_sha256: str,
    device: str | torch.device = "cpu",
) -> tuple[dict, torch.Tensor]:
    """Bind a pool to the frozen R882/DOPE worker API without label leakage."""
    _require(
        checkpoint_sha256 == R882_CHECKPOINT_SHA256,
        "initial checkpoint is not the pinned read-only R882",
    )
    _require(dope_sha256 == DOPE_SHA256, "DOPE frozen-backbone pin mismatch")
    _require(
        pool.get("schema") == POOL_SCHEMA and pool.get("eligibility_verified") is False,
        "non-admitted candidate tensor pool required",
    )
    expected_target = (
        "s1_fitness"
        if context_identity.get("task") == "regression"
        else "s1_cv_gate_passed"
    )
    _require(
        pool.get("status") == "CANDIDATE_TENSORS_NOT_ADMITTED"
        and pool.get("task_target") == expected_target
        and pool.get("proposal_source_sha256") == FLYWHEEL_EVALUATOR_SHA256
        and pool.get("landmark_source_sha256") == FLYWHEEL_LANDMARK_SHA256
        and pool.get("native_s1_sha256") == NATIVE_S1_SHA256,
        "candidate pool source or task-target contract mismatch",
    )
    _require(
        pool.get("context") == context_identity,
        "candidate pool/context identity mismatch",
    )
    _require(
        context_identity.get("task") in protocol.TASKS, "explicit worker task required"
    )
    _require(
        tokens.shape == (10, 32) and mask.shape == (10,),
        "R882 context token shape mismatch",
    )
    _require(
        bool(torch.isfinite(tokens).all()) and bool(torch.isfinite(mask.float()).all()),
        "non-finite R882 context tokens",
    )
    programs = torch.as_tensor(pool["program"], dtype=torch.float32, device=device)
    projections = torch.as_tensor(pool["proj16"], dtype=torch.float64, device=device)
    target = torch.as_tensor(
        pool["target"], dtype=torch.float32, device=device
    ).reshape(1, -1)
    n = programs.shape[0]
    _require(
        n == PROPOSALS_PER_CONTEXT
        and programs.shape == (n, 412)
        and projections.shape == (n, 16),
        "candidate tensor dimensions mismatch",
    )
    _require(
        target.shape == (1, n) and bool(torch.isfinite(target).all()),
        "candidate target shape/value mismatch",
    )
    if context_identity["task"] == "binary":
        _require(
            bool(((target == 0) | (target == 1)).all()),
            "binary candidate targets must be explicit zero/one S1 gate labels",
        )
    inputs = {
        "ctx_feat": torch.zeros((1, 584), dtype=torch.float32, device=device),
        "prog_412": programs.unsqueeze(0),
        "proj_16": projections.unsqueeze(0),
        "base_prior_cost": torch.zeros((1, n), dtype=torch.float32, device=device),
        "scale": torch.full((1,), 0.4208, dtype=torch.float32, device=device),
        "sos_prior": torch.zeros((1, n, 2), dtype=torch.float32, device=device),
        "mm_tokens_10x32": tokens.to(device=device, dtype=torch.float32).unsqueeze(0),
        "mm_mask_10": mask.to(device=device).unsqueeze(0),
        # Training labels are all targets; reveal no target values as model inputs.
        "revealed_indices": torch.empty((1, 0), dtype=torch.int64, device=device),
        "revealed_raw_f": torch.empty((1, 0), dtype=torch.float32, device=device),
        "revealed_gate": torch.empty((1, 0), dtype=torch.float32, device=device),
    }
    return inputs, target


def make_worker_packer(
    pools_by_context_id: dict[str, dict],
    tokens_by_context_id: dict[str, torch.Tensor],
    masks_by_context_id: dict[str, torch.Tensor],
    *,
    checkpoint_sha256: str,
    dope_sha256: str,
    device: str | torch.device = "cpu",
) -> Callable[[dict], tuple[dict, torch.Tensor]]:
    """Create the injected callable required by ``WorkerCore``."""
    _require(
        checkpoint_sha256 == R882_CHECKPOINT_SHA256,
        "initial checkpoint is not the pinned read-only R882",
    )
    _require(dope_sha256 == DOPE_SHA256, "DOPE frozen-backbone pin mismatch")
    _require(
        set(pools_by_context_id)
        == set(tokens_by_context_id)
        == set(masks_by_context_id),
        "candidate pools and R882 tokens do not cover the same contexts",
    )

    def pack(row: dict) -> tuple[dict, torch.Tensor]:
        context_id = row.get("id")
        _require(
            isinstance(context_id, str) and context_id in pools_by_context_id,
            "worker context has no candidate tensor pool",
        )
        pool = pools_by_context_id[context_id]
        identity = pool.get("context")
        _require(
            isinstance(identity, dict) and identity.get("context_id") == context_id,
            "candidate pool context identity mismatch",
        )
        for field in (
            "dataset",
            "family",
            "task",
            "role",
            "source_split",
            "origin",
            "has_meta_test",
        ):
            _require(
                row.get(field) == identity.get(field),
                f"worker context {field} differs from candidate pool",
            )
        return pack_worker_inputs(
            identity,
            pool,
            tokens_by_context_id[context_id],
            masks_by_context_id[context_id],
            checkpoint_sha256=checkpoint_sha256,
            dope_sha256=dope_sha256,
            device=device,
        )

    return pack
