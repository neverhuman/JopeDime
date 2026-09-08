from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data.vs import dataset_signature_from_vs
from ..prime_config import PrimeConfig, load_config
from ..prime_staging import prime_owned_path

EMBEDDING_VERSION = "hybrid-v3.2"
EMBEDDING_DIM = 1536
HYPERION_DIM = 512
FOUNDATION_RAW_DIM = 2048
FOUNDATION_DIM = 1024
MAX_ROWS = 1000
SUBSAMPLE_SEED = 20260710

FROZEN_STANDARDIZATION_SHA256 = "6c07305fc1de841ea387fe5b01d0074b9ba386d9d15089fe2a3c10acdf623145"
FROZEN_PROJECTION_SHA256 = "3df4615971e7974b09f91d6ebcf22672e02b4900d4f71a18616dae8dc37b8ff9"
FROZEN_HYPERION_WEIGHTS_SHA256 = "88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be"
FROZEN_HYPERION_MANIFEST_SHA256 = "ebb745f6a0aa6398b1c3d3c9e26585e0536326b65e2083401391203fa37d21ef"

GOLDEN42 = (
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "13b5df533e59a626",
    "17a4542ee4fb9e44",
    "1a803b27c2400dc4",
    "1cb97b1d7314d43d",
    "21fbfe8b53dbcd60",
    "24713c7f7bfa7807",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "3baa7e37fc1f7861",
    "3f781a51af45387e",
    "43b4cc3ab0ebbd97",
    "471f11cc2d38853a",
    "4f68d277d770477f",
    "542e75bc0cfdd1bf",
    "55b144de19659467",
    "55f06d147bcc2a32",
    "5f91a00bd3b66742",
    "693890aae9dc860b",
    "79e600bc5fcd5da7",
    "7af85a04de0c7018",
    "860e5558f732f638",
    "ab0239b9ef0ba28e",
    "aebf5d861a8365b4",
    "b1f909fcfceeef4b",
    "b253fbfae8460dc1",
    "b5beff4a53fa12c0",
    "b78d323566930bd7",
    "bd49e740a79bcb6c",
    "c32fa09b48fb2f27",
    "c7d815714ca52bce",
    "c8f90fc9cc5364be",
    "cf5e74b4e9594944",
    "d2daf53412ab18dc",
    "dee1fc8e9afd81c9",
    "e1c1db5260d0894e",
    "e5a3a7b3668a1cc2",
    "ee56a1015e344c11",
    "f0e7eb3d790df37d",
    "f8536dfeef471c2f",
    "fe3cc716a05d036f",
)


@dataclass(frozen=True)
class FrozenContract:
    standardization_sha256: str
    projection_sha256: str
    hyperion_weights_sha256: str
    hyperion_manifest_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "standardization_sha256": self.standardization_sha256,
            "projection_sha256": self.projection_sha256,
            "hyperion_weights_sha256": self.hyperion_weights_sha256,
            "hyperion_manifest_sha256": self.hyperion_manifest_sha256,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def frozen_contract(
    standardization_path: Path,
    projection_path: Path,
    config: PrimeConfig,
    *,
    expected: FrozenContract | None = None,
) -> FrozenContract:
    expected = expected or FrozenContract(
        FROZEN_STANDARDIZATION_SHA256,
        FROZEN_PROJECTION_SHA256,
        FROZEN_HYPERION_WEIGHTS_SHA256,
        FROZEN_HYPERION_MANIFEST_SHA256,
    )
    for path, digest, label in (
        (standardization_path, expected.standardization_sha256, "standardization"),
        (projection_path, expected.projection_sha256, "foundation projection"),
        (config.hyperion_weights, expected.hyperion_weights_sha256, "Hyperion weights"),
        (config.hyperion_manifest, expected.hyperion_manifest_sha256, "Hyperion manifest"),
    ):
        _require_hash(path, digest, label)
    return expected


def _load_projection(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        components = np.asarray(artifact["components"], dtype=np.float32)
        pca_mean = np.asarray(artifact["pca_mean"], dtype=np.float32)
        z_mean = np.asarray(artifact["z_mean"], dtype=np.float32)
        z_scale = np.asarray(artifact["z_scale"], dtype=np.float32)
    if (
        components.shape != (FOUNDATION_DIM, FOUNDATION_RAW_DIM)
        or pca_mean.shape != (FOUNDATION_RAW_DIM,)
        or z_mean.shape != (FOUNDATION_DIM,)
        or z_scale.shape != (FOUNDATION_DIM,)
        or not all(np.isfinite(value).all() for value in (components, pca_mean, z_mean, z_scale))
        or np.any(np.abs(z_scale) < 1e-12)
    ):
        raise ValueError("frozen foundation projection has invalid shape or values")
    return components, pca_mean, z_mean, z_scale


def assemble_embedding(
    vs_path: str | Path,
    foundation_path: str | Path,
    standardization: dict[str, Any],
    projection: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    signature = dataset_signature_from_vs(vs_path).astype(np.float32)
    mean = np.asarray(standardization.get("mean"), dtype=np.float32)
    scale = np.asarray(standardization.get("scale"), dtype=np.float32)
    if signature.shape != (HYPERION_DIM,) or mean.shape != signature.shape or scale.shape != signature.shape:
        raise ValueError("Hyperion signature or standardization shape mismatch")
    if not all(np.isfinite(value).all() for value in (signature, mean, scale)):
        raise ValueError("Hyperion signature or standardization is non-finite")
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    hyperion = (signature - mean) / scale

    raw = np.load(foundation_path, allow_pickle=False)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (FOUNDATION_RAW_DIM,) or not np.isfinite(raw).all():
        raise ValueError(f"foundation vector must be finite ({FOUNDATION_RAW_DIM},)")
    components, pca_mean, z_mean, z_scale = projection
    foundation = ((raw - pca_mean) @ components.T - z_mean) / z_scale
    embedding = np.concatenate((hyperion, foundation)).astype(np.float32)
    if embedding.shape != (EMBEDDING_DIM,) or not np.isfinite(embedding).all():
        raise ValueError(f"assembled embedding must be finite ({EMBEDDING_DIM},)")
    return embedding


def encode_tabarena_train(
    train_csv: str | Path,
    metadata_json: str | Path,
    *,
    max_rows: int = MAX_ROWS,
    seed: int = SUBSAMPLE_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce Dime's train-only categorical/median encoding, then min-max and cap."""
    metadata = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    categorical = metadata.get("categorical_indicator")
    if not isinstance(categorical, list) or not all(isinstance(value, bool) for value in categorical):
        raise ValueError("TabArena metadata has no boolean categorical_indicator")
    with Path(train_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = [row for row in reader if row]
    if header is None or len(header) != len(categorical) + 1 or not rows:
        raise ValueError("TabArena train CSV shape does not match metadata")
    if any(len(row) != len(header) for row in rows):
        raise ValueError("TabArena train CSV has a ragged row")
    features = len(categorical)
    dictionaries: list[dict[str, int]] = []
    medians: list[float] = []
    for column, is_categorical in enumerate(categorical):
        if is_categorical:
            dictionaries.append(
                {value: index for index, value in enumerate(sorted({row[column] for row in rows}))}
            )
            medians.append(0.0)
            continue
        dictionaries.append({})
        values = sorted(
            value
            for row in rows
            if (value := _finite_float(row[column])) is not None
        )
        if not values:
            medians.append(0.0)
        elif len(values) % 2:
            medians.append(values[len(values) // 2])
        else:
            middle = len(values) // 2
            medians.append((values[middle - 1] + values[middle]) / 2.0)

    encoded = np.empty((len(rows), features + 1), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column, is_categorical in enumerate(categorical):
            if is_categorical:
                encoded[row_index, column] = dictionaries[column][row[column]]
            else:
                encoded[row_index, column] = (
                    value if (value := _finite_float(row[column])) is not None else medians[column]
                )
        target = _finite_float(row[-1])
        if target is None:
            raise ValueError(f"TabArena train target is not finite at data row {row_index + 1}")
        encoded[row_index, -1] = target

    low = encoded.min(axis=0)
    span = encoded.max(axis=0) - low
    span = np.where(span > 1e-12, span, 1.0)
    scaled = (encoded - low) / span
    full_rows = len(scaled)
    if full_rows > max_rows:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(full_rows, size=max_rows, replace=False))
        scaled = scaled[selected]
    if not np.isfinite(scaled).all():
        raise ValueError("TabArena train encoding is non-finite")
    provenance = {
        "source_rows": full_rows,
        "encoded_rows": int(scaled.shape[0]),
        "features": features,
        "categorical_features": sum(categorical),
        "categorical_dictionaries": [
            {"column": header[index], "values": dictionary}
            for index, dictionary in enumerate(dictionaries)
            if dictionary
        ],
        "numeric_medians": {
            header[index]: medians[index]
            for index, is_categorical in enumerate(categorical)
            if not is_categorical
        },
        "min": low.tolist(),
        "span": span.tolist(),
        "subsample_seed": seed,
        "max_rows": max_rows,
        "test_rows_inspected": 0,
    }
    return scaled, provenance


def _finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _csv_bytes(matrix: np.ndarray) -> bytes:
    output = io.StringIO(newline="")
    for row in matrix:
        output.write(",".join(format(float(value), ".9g") for value in row))
        output.write("\n")
    return output.getvalue().encode("utf-8")


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _write_new_or_equal(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact collision: {path}") from None


def _sidecar(
    dataset: str,
    cohort: str,
    embedding: np.ndarray,
    contract: FrozenContract,
    sources: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "jope-prime.dime-embedding.v1",
        "version": EMBEDDING_VERSION,
        "dataset": dataset,
        "cohort": cohort,
        "shape": [EMBEDDING_DIM],
        "dtype": "float32",
        "embedding_sha256": hashlib.sha256(_npy_bytes(embedding)).hexdigest(),
        "frozen": contract.as_dict(),
        "sources": sources,
        "train_rows_only": True,
        "test_rows_inspected": 0,
    }


def _verify_cached_vector(path: Path, sidecar_path: Path, contract: FrozenContract) -> dict[str, Any]:
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("frozen") != contract.as_dict():
        raise ValueError(f"cached embedding frozen contract mismatch: {sidecar_path}")
    vector = np.load(path, mmap_mode="r", allow_pickle=False)
    if vector.shape != (EMBEDDING_DIM,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
        raise ValueError(f"cached embedding has invalid shape, dtype, or values: {path}")
    if sha256_file(path) != sidecar.get("embedding_sha256"):
        raise ValueError(f"cached embedding SHA-256 mismatch: {path}")
    return sidecar


def cache_dime_embeddings(
    *,
    output_root: str | Path | None = None,
    split_manifest: str | Path | None = None,
    standardization_path: str | Path | None = None,
    tabarena_root: str | Path = "/home/ubuntu/tabArenaLite/data/tabarena-lite-v0.1/datasets/regression",
    device: str = "cuda",
    config: PrimeConfig | None = None,
    expected_contract: FrozenContract | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    output_base = prime_owned_path(output_root or config.cache, config)
    output = output_base / "dime_score" / "embeddings" / EMBEDDING_VERSION
    output.mkdir(parents=True, exist_ok=True)
    split_manifest = Path(
        split_manifest
        or config.legacy_root / "data/processed/active_v3_dual_20260706/split_manifest.json"
    )
    standardization_path = Path(
        standardization_path
        or config.legacy_root / "data/processed/active_v3_dual_20260706/standardization.json"
    )
    standardization = json.loads(standardization_path.read_text(encoding="utf-8"))
    foundation_spec = standardization.get("foundation")
    if not isinstance(foundation_spec, dict):
        raise ValueError("standardization has no frozen foundation specification")
    projection_path = Path(foundation_spec["projection_file"])
    if not projection_path.is_absolute():
        projection_path = standardization_path.parent / projection_path
    contract = frozen_contract(
        standardization_path, projection_path, config, expected=expected_contract
    )
    projection = _load_projection(projection_path)
    split = json.loads(split_manifest.read_text(encoding="utf-8")).get("datasets")
    if not isinstance(split, dict):
        raise ValueError("split manifest has no datasets object")
    missing = [dataset for dataset in GOLDEN42 if dataset not in split]
    if missing:
        raise ValueError(f"Golden42 datasets missing from split manifest: {missing}")
    tabarena = Path(tabarena_root)
    tab_names = sorted(
        child.name
        for child in tabarena.iterdir()
        if child.is_dir() and (child / "train.csv").is_file() and (child / "metadata.json").is_file()
    )
    if len(tab_names) != 13:
        raise ValueError(f"TabArena regression cohort must contain 13 datasets, got {len(tab_names)}")

    entries: list[dict[str, Any]] = []
    vectors: dict[str, np.ndarray] = {}
    for dataset in GOLDEN42:
        vector_path = output / f"{dataset}.npy"
        sidecar_path = output / f"{dataset}.v1.json"
        entry = split[dataset]
        vs_path = Path(entry["vs_path"])
        raw_path = Path(foundation_spec["cache_dir"]) / f"{dataset}.npy"
        sources = {
            "hyperion_vs": str(vs_path),
            "hyperion_vs_sha256": sha256_file(vs_path),
            "foundation_raw": str(raw_path),
            "foundation_raw_sha256": sha256_file(raw_path),
        }
        if vector_path.exists() or sidecar_path.exists():
            if not vector_path.is_file() or not sidecar_path.is_file():
                raise ValueError(f"incomplete cached embedding for {dataset}")
            sidecar = _verify_cached_vector(vector_path, sidecar_path, contract)
            if sidecar.get("sources") != sources:
                raise ValueError(f"Golden42 embedding source provenance changed for {dataset}")
            vector = np.load(vector_path, allow_pickle=False)
        else:
            vector = assemble_embedding(vs_path, raw_path, standardization, projection)
            sidecar = _sidecar(dataset, "golden42", vector, contract, sources)
            _write_new_or_equal(vector_path, _npy_bytes(vector))
            _write_new_or_equal(
                sidecar_path,
                (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode(),
            )
            _verify_cached_vector(vector_path, sidecar_path, contract)
        vectors[f"golden42:{dataset}"] = np.asarray(vector, dtype=np.float32)
        entries.append(sidecar)

    expert = None
    for dataset in tab_names:
        identity = f"tabarena:{dataset}"
        vector_path = output / f"tabarena--{dataset}.npy"
        sidecar_path = output / f"tabarena--{dataset}.v1.json"
        source = tabarena / dataset
        encoded_dir = output / "sources" / "tabarena" / dataset
        encoded, preprocessing = encode_tabarena_train(
            source / "train.csv", source / "metadata.json"
        )
        encoded_bytes = _csv_bytes(encoded)
        encoded_path = encoded_dir / "train.csv"
        _write_new_or_equal(encoded_path, encoded_bytes)
        sources = {
            "train_csv": str(source / "train.csv"),
            "train_csv_sha256": sha256_file(source / "train.csv"),
            "metadata_json": str(source / "metadata.json"),
            "metadata_json_sha256": sha256_file(source / "metadata.json"),
            "encoded_train_csv": str(encoded_path),
            "encoded_train_csv_sha256": hashlib.sha256(encoded_bytes).hexdigest(),
            "preprocessing": preprocessing,
        }
        if vector_path.exists() or sidecar_path.exists():
            if not vector_path.is_file() or not sidecar_path.is_file():
                raise ValueError(f"incomplete cached embedding for {identity}")
            sidecar = _verify_cached_vector(vector_path, sidecar_path, contract)
            recorded_sources = sidecar.get("sources")
            if not isinstance(recorded_sources, dict) or any(
                recorded_sources.get(key) != value for key, value in sources.items()
            ):
                raise ValueError(f"TabArena embedding source provenance changed for {dataset}")
            for path_key, digest_key in (
                ("hyperion_vs", "hyperion_vs_sha256"),
                ("foundation_raw", "foundation_raw_sha256"),
            ):
                recorded_path = Path(str(recorded_sources.get(path_key)))
                if (
                    not recorded_path.is_file()
                    or sha256_file(recorded_path) != recorded_sources.get(digest_key)
                ):
                    raise ValueError(
                        f"TabArena cached source SHA-256 mismatch for {dataset}: {path_key}"
                    )
            vector = np.load(vector_path, allow_pickle=False)
        else:
            from ..encoders.foundation import encode_dataset, load_expert
            from ..signatures import make_signatures

            report = make_signatures(encoded_dir, force=False, config=config)
            if report.status not in {"generated", "skipped"}:
                raise RuntimeError(f"Hyperion encoding failed for {dataset}: {report.errors}")
            if expert is None:
                try:
                    expert = load_expert(device)
                except Exception:
                    expert = load_expert("cpu")
            raw = encode_dataset(expert, encoded_dir)
            raw_path = encoded_dir / "foundation-raw.npy"
            _write_new_or_equal(raw_path, _npy_bytes(raw))
            vector = assemble_embedding(
                encoded_dir / "hyperion3train128dim.vs",
                raw_path,
                standardization,
                projection,
            )
            sources.update(
                {
                    "hyperion_vs": str(encoded_dir / "hyperion3train128dim.vs"),
                    "hyperion_vs_sha256": sha256_file(
                        encoded_dir / "hyperion3train128dim.vs"
                    ),
                    "foundation_raw": str(raw_path),
                    "foundation_raw_sha256": sha256_file(raw_path),
                }
            )
            sidecar = _sidecar(dataset, "tabarena", vector, contract, sources)
            _write_new_or_equal(vector_path, _npy_bytes(vector))
            _write_new_or_equal(
                sidecar_path,
                (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode(),
            )
            _verify_cached_vector(vector_path, sidecar_path, contract)
        vectors[identity] = np.asarray(vector, dtype=np.float32)
        entries.append(sidecar)

    identities = [f"golden42:{dataset}" for dataset in GOLDEN42] + [
        f"tabarena:{dataset}" for dataset in tab_names
    ]
    matrix = np.stack([vectors[identity] for identity in identities]).astype(np.float32)
    if matrix.shape != (55, EMBEDDING_DIM) or not np.isfinite(matrix).all():
        raise ValueError("consolidated Dime embedding matrix is incomplete or invalid")
    matrix_bytes = _npy_bytes(matrix)
    _write_new_or_equal(output / "embeddings.npy", matrix_bytes)
    manifest = {
        "schema": "jope-prime.dime-embeddings.v1",
        "version": EMBEDDING_VERSION,
        "shape": [55, EMBEDDING_DIM],
        "dtype": "float32",
        "identities": identities,
        "embeddings_file": "embeddings.npy",
        "embeddings_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "frozen": contract.as_dict(),
        "entries": entries,
        "train_rows_only": True,
        "test_rows_inspected": 0,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _write_new_or_equal(output / "manifest.v1.json", manifest_bytes)
    return {
        "output": str(output),
        "datasets": 55,
        "golden42": 42,
        "tabarena": 13,
        "embeddings_sha256": manifest["embeddings_sha256"],
        "frozen": contract.as_dict(),
    }
