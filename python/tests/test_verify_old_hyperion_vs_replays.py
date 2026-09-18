from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import jope.research

SCRIPT = (
    Path(jope.research.__file__).parent / "verify_old_hyperion_vs_replays.py"
)
SPEC = importlib.util.spec_from_file_location("verify_old_hyperion_vs_replays", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _make_manifest(root: Path, common_inputs: Path) -> dict:
    datasets = []
    order = [f"{index:016x}" for index in range(20)]
    for index, dataset_id in enumerate(order):
        outputs = {}
        for split in ("train", "test"):
            path = root / dataset_id / f"{split}3.vs.orig"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{dataset_id},{split}\n".encode())
            outputs[split] = {
                "path": str(path),
                "sha256": MODULE.sha256_file(path),
                "shape": [1, 1],
                "finite": True,
            }
        datasets.append(
            {
                "dataset_id": dataset_id,
                "campaign_index": index,
                "inputs": {"path": str(common_inputs / dataset_id)},
                "sweep_tensor": {
                    "shape": [100, 4, 2, 1],
                    "dtype": "float32",
                    "finite": True,
                    "c_order_sha256": f"sweep-{dataset_id}",
                },
                "collapsed_cell_c_order_sha256": f"cell-{dataset_id}",
                "outputs": outputs,
            }
        )
    return {
        "schema": MODULE.GENERATION_SCHEMA,
        "status": "complete",
        "claim": "reconstruction",
        "dataset_count": 20,
        "output_file_count": 40,
        "dataset_order": order,
        "parameters": {"sweeps": 100},
        "authorities": {"checkpoint": "pinned"},
        "load_diagnostics": {"missing_key_count": 30},
        "known_irrecoverable_gap": {"exact_replay_possible": False},
        "environment": {
            "python": "test",
            "numpy": "test",
            "pandas": "test",
            "torch": "test",
            "cuda_runtime": "test",
            "cuda_visible_devices": "0",
            "visible_cuda_device_count": 1,
            "cuda_device": "test",
            "cuda_capability": [8, 9],
            "float32_matmul_precision": "highest",
        },
        "datasets": datasets,
    }


def _write_manifest(root: Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_compare_replays_requires_matching_tensor_and_output_bytes(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    common_inputs = tmp_path / "inputs"
    first = _make_manifest(first_root, common_inputs)
    second = _make_manifest(second_root, common_inputs)
    _write_manifest(first_root, first)
    _write_manifest(second_root, second)

    receipt = MODULE.compare_replays(first_root, second_root)
    assert receipt["dataset_count"] == 20
    assert receipt["output_file_count"] == 40
    assert receipt["all_output_files_byte_identical"] is True

    second["datasets"][3]["sweep_tensor"]["c_order_sha256"] = "changed"
    _write_manifest(second_root, second)
    with pytest.raises(ValueError, match="sweep_tensor"):
        MODULE.compare_replays(first_root, second_root)
