#!/usr/bin/env python3
"""Immutable expanded-dataset caches and append-only catalog v2."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping

from test_exact_common import (
    atomic_json,
    family_disjoint_split,
    prime_source_root,
    sha256_bytes,
    sha256_file,
)

SOURCE_ROOT = Path("/home/ubuntu/remote_super/quant")
MIRROR_ROOT = Path("/home/ubuntu/quant")
CACHE_ROOT = Path("/home/ubuntu/jope-flywheel/encodings")
FLYWHEEL_ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
TASKS = ("regression", "binary", "multiclass", "timeseries")
# The multiclass corpus keeps its legacy on-disk directory name; every other
# task's directory matches its task name. This mirrors TaskKind::corpus_dir_name
# in src/task.rs:44 -- the two MUST agree or the Rust and Python halves of the
# expansion pipeline disagree about where a dataset lives.
CORPUS_DIR_NAMES = {"multiclass": "multi_label"}


def corpus_dir_name(task: str) -> str:
    """On-disk directory for a task's source corpus."""

    return CORPUS_DIR_NAMES.get(task, task)
TRANCHE_MAX = 2_048
SEALED_PER_TASK = 256
ENCODER_PROTOCOL = "hyperion-train-context-v3"
CONTEXT_PROTOCOL = "jope-dataset1536-target512-f32le-v1"
DATASET_DIM = 1_536
TARGET_DIM = 512
CONTEXT_STRIDE = (DATASET_DIM + TARGET_DIM) * 4
STANDARDIZATION = Path(
    "/home/ubuntu/jope-flywheel/encodings/foundation-v1/standardization/sha256/"
    "daac224cf7c6d13a1bf74fc3bb7defd36f33bc3f30db35d3e881c16ede976d1f.json"
)
TRAIN_ENCODING_NAMES = (
    "hyperion3train128dim.vs",
    "train3.vs",
    "train_target.vs",
)
OPTIONAL_TRAIN_ENCODING_NAMES = ("hyperion3train.vs",)
VOLATILE_RECIPE_KEYS = {
    "seed",
    "random_seed",
    "generator_seed",
    "row_count",
    "rows",
    "n_rows",
    "n_train_rows",
    "n_test_rows",
    "train_rows",
    "test_rows",
}
IDENTITY_RECIPE_KEYS = {
    "dataset_id",
    "dataset_hash",
    "id",
    "name",
    "created_at",
    "generated_at",
    "source_path",
    "train_path",
    "test_path",
}
_DATASET = re.compile(r"^(?:synth_)?[0-9a-f]{16}$")


def quarantine_identities(cache_root: Path = CACHE_ROOT) -> frozenset[str]:
    """Load the frozen V1 incident denylist, failing closed if it is partial."""

    root = cache_root / "incident-quarantine"
    if not root.exists():
        return frozenset()
    index = root / "index.v1.json"
    if not index.is_file():
        raise RuntimeError("V1 incident quarantine exists without index.v1.json")
    value = json.loads(index.read_text())
    if value.get("schema") != "JopeV1EncodingQuarantine.v1":
        raise RuntimeError("V1 incident quarantine schema drift")
    policy = value.get("policy")
    if not isinstance(policy, Mapping) or any(
        policy.get(field) is not False
        for field in (
            "catalog_allowed",
            "training_allowed",
            "benchmark_allowed",
            "v2_import_allowed",
            "gc_allowed",
        )
    ):
        raise RuntimeError("V1 incident quarantine policy permits consumption")
    trees = value.get("trees")
    if not isinstance(trees, list):
        raise RuntimeError("V1 incident quarantine tree inventory is absent")
    complete = [tree for tree in trees if tree.get("state") == "complete"]
    partial = [tree for tree in trees if tree.get("state") == "partial"]
    if len(complete) != 11 or len(partial) != 7:
        raise RuntimeError("V1 incident quarantine inventory is incomplete")
    identities = [str(tree.get("identity") or "") for tree in trees]
    if any(not identity for identity in identities) or len(set(identities)) != 18:
        raise RuntimeError("V1 incident quarantine identities are invalid")
    return frozenset(identities)


def now_ms() -> int:
    return int(time.time() * 1000)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _stable_recipe(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_recipe(item)
            for key, item in sorted(value.items())
            if str(key).lower() not in VOLATILE_RECIPE_KEYS
            and str(key).lower() not in IDENTITY_RECIPE_KEYS
            and not str(key).lower().endswith("_seed")
            and "row_count" not in str(key).lower()
        }
    if isinstance(value, list):
        return [_stable_recipe(item) for item in value]
    return value


def _has_recipe_signal(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_has_recipe_signal(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_recipe_signal(item) for item in value)
    return value is not None and value != "" and value != "synthetic"


def source_family(meta: Mapping[str, object], dataset_id: str) -> str:
    """Group synthetic recipes while ignoring seed and row-count noise."""

    synthetic = bool(
        meta.get("synthetic_origin")
        or meta.get("origin") == "synthetic"
        or dataset_id.startswith("synth_")
    )
    if synthetic:
        recipe = meta.get("generator_recipe") or {
            "generator_name": meta.get("generator_name"),
            "generator_version": meta.get("generator_version"),
            "target_generation": meta.get("target_generation"),
        }
        stable = _stable_recipe(recipe)
        if not _has_recipe_signal(stable):
            stable = _stable_recipe(meta)
        if not _has_recipe_signal(stable):
            raise ValueError(
                f"{dataset_id}: synthetic dataset has no stable generator recipe"
            )
        return f"synthetic:{canonical_hash(stable)}"
    acquisition = meta.get("acquisition")
    if isinstance(acquisition, Mapping):
        family = {
            "source_path": acquisition.get("source_path"),
            "raw_sha256": acquisition.get("raw_sha256"),
            "dedupe_fingerprint": acquisition.get("dedupe_fingerprint"),
        }
    else:
        family = {
            "dataset_name": meta.get("dataset_name") or meta.get("name"),
            "source": meta.get("source") or meta.get("origin"),
            "dedupe_fingerprint": meta.get("dedupe_fingerprint"),
        }
    if not any(family.values()):
        family = {"dataset_id": dataset_id}
    return f"real:{canonical_hash(family)}"


def mirror_dataset(task: str, dataset_id: str, mirror_root: Path = MIRROR_ROOT) -> Path | None:
    candidates = (
        mirror_root / corpus_dir_name(task) / dataset_id,
        mirror_root / task / dataset_id,
        mirror_root / dataset_id,
    )
    return next((path for path in candidates if path.is_dir()), None)


def conflict_receipt(
    *,
    task: str,
    dataset_id: str,
    source: Path,
    mirror: Path,
    disagreements: Mapping[str, object],
    receipt_root: Path,
) -> Path:
    payload = {
        "schema": "JopeExpandedMirrorConflict.v1",
        "task": task,
        "dataset_id": dataset_id,
        "source": str(source),
        "mirror": str(mirror),
        "disagreements": disagreements,
        "observed_at_ms": now_ms(),
    }
    digest = canonical_hash(payload)
    path = receipt_root / f"{digest}.json"
    if not path.is_file():
        atomic_json(path, payload)
    return path


def compare_mirror(
    task: str,
    dataset_id: str,
    source: Path,
    *,
    mirror_root: Path = MIRROR_ROOT,
    receipt_root: Path = FLYWHEEL_ROOT / "receipts" / "mirror-conflicts",
) -> dict:
    mirror = mirror_dataset(task, dataset_id, mirror_root)
    if mirror is None:
        return {"mirror": None, "matched": None}
    disagreements = {}
    for name in ("train.csv", "test.csv"):
        source_file = source / name
        mirror_file = mirror / name
        if not mirror_file.is_file():
            disagreements[name] = {"source": sha256_file(source_file), "mirror": None}
            continue
        source_hash = sha256_file(source_file)
        mirror_hash = sha256_file(mirror_file)
        if source_hash != mirror_hash:
            disagreements[name] = {"source": source_hash, "mirror": mirror_hash}
    if disagreements:
        receipt = conflict_receipt(
            task=task,
            dataset_id=dataset_id,
            source=source,
            mirror=mirror,
            disagreements=disagreements,
            receipt_root=receipt_root,
        )
        raise RuntimeError(
            f"{task}:{dataset_id}: mirror content conflict ({receipt})"
        )
    return {"mirror": str(mirror), "matched": True}


def discover(
    *,
    source_root: Path = SOURCE_ROOT,
    mirror_root: Path = MIRROR_ROOT,
    verify_mirror: bool = True,
    limit: int | None = None,
    cache_root: Path = CACHE_ROOT,
) -> list[dict]:
    entries = []
    denied = quarantine_identities(cache_root)
    for task in TASKS:
        task_root = source_root / corpus_dir_name(task)
        for source in sorted(task_root.iterdir()):
            if (
                not source.is_dir()
                or not _DATASET.fullmatch(source.name)
                or not all((source / name).is_file() for name in ("train.csv", "test.csv", "meta.json"))
            ):
                continue
            if source.name in denied:
                continue
            if verify_mirror:
                mirror = compare_mirror(
                    task, source.name, source, mirror_root=mirror_root
                )
            else:
                mirror = {"mirror": None, "matched": None}
            meta = json.loads((source / "meta.json").read_text())
            entries.append(
                {
                    "task": task,
                    "dataset_id": source.name,
                    "dataset_hash": source.name,
                    "source": str(source),
                    "source_family": source_family(meta, source.name),
                    "train_sha256": sha256_file(source / "train.csv"),
                    "test_sha256": sha256_file(source / "test.csv"),
                    "meta_sha256": sha256_file(source / "meta.json"),
                    **mirror,
                }
            )
            if limit is not None and len(entries) >= limit:
                return entries
    return entries


def cache_path(task: str, dataset_id: str, cache_root: Path = CACHE_ROOT) -> Path:
    return cache_root / f"{task}-expanded" / dataset_id


def _copy_reflink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cp", "--reflink=auto", "--preserve=mode,timestamps", str(source), str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        shutil.copy2(source, destination)


def validate_cache(path: Path, *, expected: Mapping[str, object] | None = None) -> dict:
    if "incident-quarantine" in path.parts:
        raise ValueError("quarantined V1 cache cannot be consumed")
    manifest_path = path / "cache-manifest.v1.json"
    if not manifest_path.is_file():
        raise ValueError("expanded cache manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "JopeExpandedEncodingCache.v1"
        or manifest.get("immutable") is not True
        or manifest.get("encoder_protocol") != ENCODER_PROTOCOL
    ):
        raise ValueError("expanded cache manifest envelope is invalid")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("expanded cache file inventory is absent")
    for name, digest in files.items():
        source = path / str(name)
        if not source.is_file() or sha256_file(source) != digest:
            raise ValueError(f"expanded cache content drift: {name}")
    if expected:
        for field in ("task", "dataset_id", "train_sha256", "test_sha256"):
            if expected.get(field) is not None and manifest.get(field) != expected[field]:
                raise ValueError(f"expanded cache identity drift: {field}")
    return manifest


def materialize_cache(
    entry: Mapping[str, object],
    *,
    cache_root: Path = CACHE_ROOT,
    encoder: Callable[[Path, Path, str], None] | None = None,
) -> dict:
    task = str(entry["task"])
    dataset_id = str(entry.get("dataset_id") or entry["dataset_hash"])
    if dataset_id in quarantine_identities(cache_root):
        raise RuntimeError(f"V1 cache materialization denied by quarantine: {dataset_id}")
    source = Path(str(entry["source"]))
    final = cache_path(task, dataset_id, cache_root)
    lock_path = final.parent / ".locks" / f"{dataset_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if final.is_dir():
            return validate_cache(final, expected=entry)
        temporary = final.with_name(f".{dataset_id}.tmp-{os.getpid()}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            for name in ("train.csv", "test.csv", "meta.json", "signature_manifest.json"):
                source_file = source / name
                if source_file.is_file():
                    _copy_reflink_or_copy(source_file, temporary / name)
            missing = []
            for name in (*TRAIN_ENCODING_NAMES, *OPTIONAL_TRAIN_ENCODING_NAMES):
                source_file = source / name
                if source_file.is_file():
                    _copy_reflink_or_copy(source_file, temporary / name)
                elif name in TRAIN_ENCODING_NAMES:
                    missing.append(name)
            if missing:
                if encoder is None:
                    raise RuntimeError(
                        f"{task}:{dataset_id}: missing train encodings: {','.join(missing)}"
                    )
                encoder(source, temporary, task)
            absent = [name for name in TRAIN_ENCODING_NAMES if not (temporary / name).is_file()]
            if absent:
                raise RuntimeError(
                    f"{task}:{dataset_id}: encoder did not produce {','.join(absent)}"
                )
            if not (temporary / "signature_manifest.json").is_file():
                signature = {
                    "schema": "JopeExpandedSignatureManifest.v1",
                    "task": task,
                    "dataset_id": dataset_id,
                    "train_only": True,
                    "encoder_protocol": ENCODER_PROTOCOL,
                    "files": {
                        name: sha256_file(temporary / name)
                        for name in TRAIN_ENCODING_NAMES
                    },
                }
                atomic_json(temporary / "signature_manifest.json", signature)
            files = {
                path.name: sha256_file(path)
                for path in sorted(temporary.iterdir())
                if path.is_file()
            }
            manifest = {
                "schema": "JopeExpandedEncodingCache.v1",
                "immutable": True,
                "task": task,
                "dataset_id": dataset_id,
                "source": str(source),
                "source_family": entry["source_family"],
                "train_sha256": entry["train_sha256"],
                "test_sha256": entry["test_sha256"],
                "encoder_protocol": ENCODER_PROTOCOL,
                "encoder_protocol_sha256": canonical_hash(
                    {"protocol": ENCODER_PROTOCOL, "train_only": True}
                ),
                "train_only_encoding": True,
                "files": files,
                "created_at_ms": now_ms(),
            }
            atomic_json(temporary / "cache-manifest.v1.json", manifest)
            validate_cache(temporary, expected=entry)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, final)
            return manifest
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


# Root of the foundation-expert tree. `fusion.experts.TabPFNv3Reg.__init__`
# hard-codes `sys.path.insert(0, "/home/ubuntu/fable_god")` and then imports
# `python.elo_finetune.model_io` -- but that root was MOVED here and the old
# path no longer exists, so the import raised ModuleNotFoundError, which
# `encode_external` caught and reported as "cuda expert load failed; falling
# back to cpu". The CPU fallback calls the same loader and fails identically,
# so the encoder was dead on BOTH devices while the log blamed the device.
# Putting the real root on sys.path fixes it without touching another repo:
# `default_weights_dir()` locates the weights relative to its own package, and
# the stale hard-coded path is simply skipped once it does not exist.
FOUNDATION_EXPERT_ROOT = Path("/home/ubuntu/remote_super/fable_god")


def ensure_foundation_expert_importable() -> None:
    """Put the foundation expert tree on sys.path and PROVE it resolves.

    The failure this guards against cost 854 seconds per attempt and reported
    the wrong cause. A missing dependency must fail here, immediately, naming
    the path it could not import from -- not degrade into a device message.
    """

    root = str(FOUNDATION_EXPERT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        importlib.import_module("python.elo_finetune.model_io")
    except ImportError as error:
        raise SystemExit(
            f"foundation expert is unimportable from {root}: {error}. "
            "The encoder cannot run on ANY device until this resolves; do not "
            "interpret a device fallback message as the cause."
        ) from error


# `jope.signatures.make_signatures` caps hyperion_vs at 300s by default and
# exposes no environment or keyword override to its callers. Measured on a real
# multiclass source, hyperion_vs takes ~344s on CPU and ~4s on a GPU -- so the
# default fails on CPU by a margin of seconds, and it fails as a TIMEOUT, which
# reads like a hang rather than a budget that is simply too small. The encoder
# must also run on nodes with no GPU at all (xbabe4), so the CPU cost is not
# exceptional, it is the floor. Widen it here rather than editing ~/jope.
SIGNATURE_TIMEOUT_SECONDS = 3_600.0


def widen_signature_timeout(seconds: float = SIGNATURE_TIMEOUT_SECONDS) -> None:
    """Raise hyperion_vs's ceiling for this process. Idempotent.

    `encode_external` lives in ~/jope/scripts but the `jope` package it imports
    lives in ~/jope/src, so both roots have to be reachable before the patch
    target can even be named.
    """

    jope_src = "/home/ubuntu/jope/src"
    if jope_src not in sys.path:
        sys.path.insert(0, jope_src)
    import jope.signatures as signatures

    if getattr(signatures.make_signatures, "_jope_dime_widened", False):
        return
    original = signatures.make_signatures

    def widened(dataset_dir, **kwargs):
        kwargs.setdefault("timeout_s", seconds)
        return original(dataset_dir, **kwargs)

    widened._jope_dime_widened = True
    signatures.make_signatures = widened


def existing_hyperion_encoder(source: Path, destination: Path, task: str) -> None:
    """Generate missing sidecars with JOPE's pinned train-only encoder."""

    del task
    ensure_foundation_expert_importable()
    widen_signature_timeout()
    jope_root = Path("/home/ubuntu/jope")
    sys.path.insert(0, str(jope_root / "scripts"))
    from encode_external import encode_external

    scratch_root = destination.parent / f".encoder-{destination.name}-{os.getpid()}"
    scratch = scratch_root / destination.name.lstrip(".").split(".tmp-", 1)[0]
    scratch.mkdir(parents=True, exist_ok=False)
    try:
        encode_external(
            scratch.name,
            source / "train.csv",
            scratch,
            source_test=source / "test.csv",
            jain_train_csv=source / "train.csv",
            jain_test_csv=source / "test.csv",
            device="cuda",
        )
        for name in TRAIN_ENCODING_NAMES:
            generated = scratch / name
            if generated.is_file() and not (destination / name).is_file():
                _copy_reflink_or_copy(generated, destination / name)
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


def freeze_sealed_cohort(
    entries: Iterable[Mapping[str, object]],
    *,
    output: Path,
    per_task: int = SEALED_PER_TASK,
    seed: int = 20260726,
) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for raw in entries:
        by_task[str(raw["task"])].append(dict(raw))
    selected = []
    selected_families = set()
    for task in TASKS:
        ordered = sorted(
            by_task[task],
            key=lambda row: hashlib.sha256(
                f"{seed}|sealed|{task}|{row['source_family']}|{row['dataset_hash']}".encode()
            ).digest(),
        )
        for row in ordered:
            family = str(row["source_family"])
            if family in selected_families:
                continue
            selected.append(row)
            selected_families.add(family)
            if sum(item["task"] == task for item in selected) == per_task:
                break
        if sum(item["task"] == task for item in selected) != per_task:
            raise RuntimeError(f"{task}: insufficient family-disjoint sealed datasets")
    payload = {
        "schema": "JopeSealedExpandedCohort.v1",
        "immutable": True,
        "classification": "sealed_zero_shot",
        "training_ingestion_eligible": False,
        "entries": selected,
        "counts": {task: per_task for task in TASKS},
        "created_at_ms": now_ms(),
    }
    payload["cohort_sha256"] = canonical_hash(payload)
    if output.is_file():
        existing = json.loads(output.read_text())
        if existing != payload:
            raise RuntimeError("refusing to rewrite immutable sealed cohort")
    else:
        atomic_json(output, payload)
    return payload


def adaptive_candidates(
    entries: Iterable[Mapping[str, object]], sealed: Mapping[str, object]
) -> list[dict]:
    sealed_entries = sealed.get("entries") or []
    identities = {
        (str(row["task"]), str(row["dataset_hash"])) for row in sealed_entries
    }
    families = {str(row["source_family"]) for row in sealed_entries}
    return [
        dict(row)
        for row in entries
        if (str(row["task"]), str(row["dataset_hash"])) not in identities
        and str(row["source_family"]) not in families
    ]


def encode_train_context(
    cache: Path,
    task: str,
    *,
    standardization_path: Path = STANDARDIZATION,
) -> bytes:
    """Encode one cache using train features and train targets only."""

    del task
    sys.path.insert(0, str(prime_source_root()))
    from jope.propose import embedding_from_vs, target_summary_from_vs

    standardization = json.loads(standardization_path.read_text())
    values = [
        *embedding_from_vs(cache / "hyperion3train128dim.vs", standardization),
        *target_summary_from_vs(cache / "train_target.vs"),
    ]
    if len(values) != DATASET_DIM + TARGET_DIM:
        raise RuntimeError(
            f"{cache.name}: context has {len(values)} values, expected "
            f"{DATASET_DIM + TARGET_DIM}"
        )
    return struct.pack(f"<{len(values)}f", *values)


def activate_tranche(
    entries: Iterable[Mapping[str, object]],
    *,
    cache_root: Path = CACHE_ROOT,
    catalog_root: Path = FLYWHEEL_ROOT / "catalog" / "v2",
    tranche_id: str | None = None,
    context_encoder: Callable[[Path, str], bytes] | None = None,
) -> dict:
    selected = [dict(row) for row in entries]
    if not selected or len(selected) > TRANCHE_MAX:
        raise ValueError("adaptive tranche must contain 1..2048 datasets")
    identities = set()
    families = {}
    catalog_entries = []
    contexts: list[bytes] = []
    encode = context_encoder or encode_train_context
    denied = quarantine_identities(cache_root)
    for index, row in enumerate(selected):
        identity = (str(row["task"]), str(row["dataset_hash"]))
        if identity[1] in denied:
            raise RuntimeError(
                f"catalog adoption denied by V1 quarantine: {identity[1]}"
            )
        if identity in identities:
            raise ValueError(f"tranche repeats {identity}")
        identities.add(identity)
        cache = cache_path(identity[0], identity[1], cache_root)
        manifest = validate_cache(cache, expected=row)
        context = encode(cache, identity[0])
        if len(context) != CONTEXT_STRIDE:
            raise RuntimeError(
                f"{identity[0]}:{identity[1]}: context is {len(context)} bytes, "
                f"expected {CONTEXT_STRIDE}"
            )
        contexts.append(context)
        family = str(row["source_family"])
        families[f"{identity[0]}:{identity[1]}"] = family
        catalog_entries.append(
            {
                **row,
                "cache": str(cache),
                "cache_manifest_sha256": sha256_file(
                    cache / "cache-manifest.v1.json"
                ),
                "encoding_hashes": {
                    name: digest
                    for name, digest in manifest["files"].items()
                    if name
                    in (*TRAIN_ENCODING_NAMES, *OPTIONAL_TRAIN_ENCODING_NAMES)
                },
                "encoder_protocol_sha256": manifest["encoder_protocol_sha256"],
                "context_protocol_sha256": canonical_hash(
                    {"protocol": CONTEXT_PROTOCOL}
                ),
                "context_sha256": sha256_bytes(context),
                "catalog_index": index,
                "metric": "r2" if identity[0] == "regression" else "auc",
            }
        )
    splits = family_disjoint_split(families)
    for row in catalog_entries:
        row["split"] = splits[f"{row['task']}:{row['dataset_hash']}"]
    counts = {
        task: sum(row["task"] == task for row in catalog_entries) for task in TASKS
    }
    identifier = tranche_id or (
        f"tranche-{now_ms()}-{canonical_hash(sorted(identities))[:12]}"
    )
    blob = b"".join(contexts)
    data_sha256 = sha256_bytes(blob)
    catalog_root.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_root / f"{identifier}.json"
    activated_at_ms = (
        int(json.loads(catalog_path.read_text())["activated_at_ms"])
        if catalog_path.is_file()
        else now_ms()
    )
    payload = {
        "schema": "JopeExpandedCatalog.v2",
        "immutable": True,
        "append_only": True,
        "tranche_id": identifier,
        "task_counts": counts,
        "row_count": len(catalog_entries),
        "dataset_dimensions": DATASET_DIM,
        "target_dimensions": TARGET_DIM,
        "record_stride_bytes": CONTEXT_STRIDE,
        "context_protocol": CONTEXT_PROTOCOL,
        "data_sha256": data_sha256,
        "source_family_disjoint": True,
        "entries": catalog_entries,
        "activated_at_ms": activated_at_ms,
    }
    payload["catalog_content_sha256"] = canonical_hash(payload)
    data_path = catalog_root / f"{data_sha256}.f32.bin"
    if data_path.is_file():
        if sha256_file(data_path) != data_sha256:
            raise RuntimeError("expanded context blob content drift")
    else:
        temporary = data_path.with_name(f".{data_path.name}.tmp-{os.getpid()}")
        temporary.write_bytes(blob)
        if sha256_file(temporary) != data_sha256:
            temporary.unlink()
            raise RuntimeError("expanded context blob write failed validation")
        os.replace(temporary, data_path)
    if catalog_path.exists():
        if json.loads(catalog_path.read_text()) != payload:
            raise RuntimeError("refusing to rewrite immutable expanded catalog")
    else:
        atomic_json(catalog_path, payload)
    pointer = {
        "schema": "JopeExpandedCatalogPointer.v2",
        "catalog": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "data": str(data_path),
        "data_sha256": data_sha256,
        "task_counts": counts,
        "row_count": len(catalog_entries),
    }
    atomic_json(catalog_root / "current.json", pointer)
    return pointer


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--limit", type=int)
    inventory.add_argument("--skip-mirror", action="store_true")
    cache = commands.add_parser("cache")
    cache.add_argument("task", choices=TASKS)
    cache.add_argument("dataset_id")
    args = parser.parse_args()
    if args.command == "inventory":
        rows = discover(verify_mirror=not args.skip_mirror, limit=args.limit)
        print(
            json.dumps(
                {
                    "datasets": len(rows),
                    "task_counts": {
                        task: sum(row["task"] == task for row in rows)
                        for task in TASKS
                    },
                },
                sort_keys=True,
            )
        )
        return
    source = SOURCE_ROOT / corpus_dir_name(args.task) / args.dataset_id
    compare_mirror(args.task, args.dataset_id, source)
    meta = json.loads((source / "meta.json").read_text())
    entry = {
        "task": args.task,
        "dataset_id": args.dataset_id,
        "dataset_hash": args.dataset_id,
        "source": str(source),
        "source_family": source_family(meta, args.dataset_id),
        "train_sha256": sha256_file(source / "train.csv"),
        "test_sha256": sha256_file(source / "test.csv"),
    }
    print(
        json.dumps(
            materialize_cache(entry, encoder=existing_hyperion_encoder),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
