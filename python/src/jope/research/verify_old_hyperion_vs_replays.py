#!/usr/bin/env python3
"""Verify two old-Hyperion reconstruction runs byte for byte.

The generator deliberately does not retain its large sweep tensors.  Their
C-order SHA-256 digests are therefore compared from the independently written
manifests, while the published ``*.vs.orig`` files are hashed and compared
directly.  The verifier refuses partial cohorts and writes its receipt with
exclusive-create semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


GENERATION_SCHEMA = "HyperionOldPythonVsReconstruction.v1"
VERIFICATION_SCHEMA = "HyperionOldPythonVsReplayVerification.v1"
EXPECTED_DATASETS = 20
EXPECTED_OUTPUTS = EXPECTED_DATASETS * 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_complete_manifest(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root.resolve() / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != GENERATION_SCHEMA:
        raise ValueError(f"Unexpected generation schema in {path}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Generation is not complete: {path}")
    if manifest.get("dataset_count") != EXPECTED_DATASETS:
        raise ValueError(f"Expected {EXPECTED_DATASETS} datasets: {path}")
    if manifest.get("output_file_count") != EXPECTED_OUTPUTS:
        raise ValueError(f"Expected {EXPECTED_OUTPUTS} outputs: {path}")
    order = manifest.get("dataset_order")
    datasets = manifest.get("datasets")
    if not isinstance(order, list) or len(order) != EXPECTED_DATASETS:
        raise ValueError(f"Invalid dataset order: {path}")
    if len(set(order)) != len(order):
        raise ValueError(f"Duplicate dataset ID: {path}")
    if not isinstance(datasets, list) or len(datasets) != EXPECTED_DATASETS:
        raise ValueError(f"Invalid dataset receipts: {path}")
    if [dataset.get("dataset_id") for dataset in datasets] != order:
        raise ValueError(f"Dataset receipts are out of order: {path}")
    return path, manifest


def _without_path(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "path"}


def _require_equal(label: str, first: Any, second: Any) -> None:
    if first != second:
        raise ValueError(f"Replay mismatch in {label}")


def _verify_output(receipt: dict[str, Any]) -> tuple[Path, str]:
    path = Path(receipt["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing generated output: {path}")
    actual = sha256_file(path)
    if actual != receipt.get("sha256"):
        raise ValueError(f"Generated output hash mismatch: {path}")
    if receipt.get("finite") is not True:
        raise ValueError(f"Generated output is not finite: {path}")
    return path, actual


def compare_replays(first_root: Path, second_root: Path) -> dict[str, Any]:
    first_path, first = load_complete_manifest(first_root)
    second_path, second = load_complete_manifest(second_root)

    for field in (
        "claim",
        "dataset_order",
        "parameters",
        "authorities",
        "load_diagnostics",
        "known_irrecoverable_gap",
    ):
        _require_equal(field, first.get(field), second.get(field))

    stable_environment = (
        "python",
        "numpy",
        "pandas",
        "torch",
        "cuda_runtime",
        "cuda_visible_devices",
        "visible_cuda_device_count",
        "cuda_device",
        "cuda_capability",
        "float32_matmul_precision",
    )
    for field in stable_environment:
        _require_equal(
            f"environment.{field}",
            first["environment"].get(field),
            second["environment"].get(field),
        )

    comparisons: list[dict[str, Any]] = []
    for first_dataset, second_dataset in zip(
        first["datasets"], second["datasets"], strict=True
    ):
        dataset_id = first_dataset["dataset_id"]
        _require_equal("dataset_id", dataset_id, second_dataset.get("dataset_id"))
        _require_equal(
            f"{dataset_id}.campaign_index",
            first_dataset.get("campaign_index"),
            second_dataset.get("campaign_index"),
        )
        _require_equal(
            f"{dataset_id}.inputs",
            first_dataset.get("inputs"),
            second_dataset.get("inputs"),
        )
        _require_equal(
            f"{dataset_id}.sweep_tensor",
            first_dataset.get("sweep_tensor"),
            second_dataset.get("sweep_tensor"),
        )
        _require_equal(
            f"{dataset_id}.collapsed_cell_c_order_sha256",
            first_dataset.get("collapsed_cell_c_order_sha256"),
            second_dataset.get("collapsed_cell_c_order_sha256"),
        )

        output_comparisons: dict[str, Any] = {}
        for split in ("train", "test"):
            first_output = first_dataset["outputs"][split]
            second_output = second_dataset["outputs"][split]
            _require_equal(
                f"{dataset_id}.{split}.receipt",
                _without_path(first_output),
                _without_path(second_output),
            )
            first_output_path, first_sha = _verify_output(first_output)
            second_output_path, second_sha = _verify_output(second_output)
            _require_equal(f"{dataset_id}.{split}.sha256", first_sha, second_sha)
            if first_output_path.read_bytes() != second_output_path.read_bytes():
                raise ValueError(f"Output bytes differ: {dataset_id} {split}")
            output_comparisons[split] = {
                "sha256": first_sha,
                "shape": first_output["shape"],
                "byte_identical": True,
            }

        comparisons.append(
            {
                "dataset_id": dataset_id,
                "sweep_tensor_c_order_sha256": first_dataset["sweep_tensor"][
                    "c_order_sha256"
                ],
                "collapsed_cell_c_order_sha256": first_dataset[
                    "collapsed_cell_c_order_sha256"
                ],
                "outputs": output_comparisons,
            }
        )

    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "complete",
        "claim": "two independent seeded CUDA reconstructions are byte-identical",
        "generation_manifests": {
            "first": {"path": str(first_path), "sha256": sha256_file(first_path)},
            "second": {"path": str(second_path), "sha256": sha256_file(second_path)},
        },
        "dataset_count": len(comparisons),
        "output_file_count": len(comparisons) * 2,
        "all_sweep_tensor_digests_identical": True,
        "all_collapsed_cell_digests_identical": True,
        "all_output_files_byte_identical": True,
        "datasets": comparisons,
    }


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--second-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = compare_replays(args.first_root, args.second_root)
    write_json_exclusive(args.receipt.resolve(), receipt)
    print(
        f"Verified {receipt['output_file_count']} byte-identical outputs across "
        f"{receipt['dataset_count']} datasets"
    )


if __name__ == "__main__":
    main()
