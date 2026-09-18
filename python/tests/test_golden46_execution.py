from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jope.research

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(jope.research.__file__).parent / "golden46_execution.py"
SPEC = importlib.util.spec_from_file_location("golden46_execution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
execution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(execution)


def write_dataset(root: Path, dataset_id: str, test_target: np.ndarray) -> None:
    dataset = root / dataset_id
    dataset.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(
        {
            "a": np.linspace(-2.0, 2.0, 32),
            "b": np.tile([0.0, 1.0], 16),
        }
    )
    train["target"] = 0.25 + 0.5 * train["a"] - 0.1 * train["b"]
    test = pd.DataFrame(
        {
            "a": np.linspace(-1.75, 1.75, 12),
            "b": np.tile([1.0, 0.0], 6),
            "target": test_target,
        }
    )
    train.to_csv(dataset / "train.csv", index=False)
    test.to_csv(dataset / "test.csv", index=False)
    (dataset / "meta.json").write_text(
        json.dumps(
            {
                "headers_removed_from_csv": False,
                "train_rows": len(train),
                "test_rows": len(test),
                "n_features": 2,
                "target_is_final_column": True,
            }
        )
    )


def run_isolated(
    canonical: Path,
    worker: Path,
    controller: Path,
    artifacts: Path,
    dataset_id: str,
) -> tuple[bytes, np.ndarray, float]:
    execution.prepare_dataset(
        dataset_id,
        canonical_root=canonical,
        worker_root=worker,
        controller_root=controller,
    )
    execution.run_registered(
        dataset_id,
        "ridge",
        worker_root=worker,
        artifact_root=artifacts,
    )
    report = execution.finalize_report(
        dataset_id,
        "ridge",
        canonical_root=canonical,
        worker_root=worker,
        artifact_root=artifacts,
        controller_root=controller,
    )
    worker_test = (worker / dataset_id / "test.csv").read_bytes()
    prediction = np.load(
        artifacts / dataset_id / "ridge" / "predictions.npy",
        allow_pickle=False,
    )
    return worker_test, prediction, report["metrics"]["rmse"]


@pytest.mark.xbabe2  # imports cluster_model_queue/model_zoo from /home/ubuntu/remote_super (old state plane)
def test_test_target_changes_only_controller_score(tmp_path: Path) -> None:
    dataset_id = "masked"
    canonical = tmp_path / "canonical"
    write_dataset(canonical, dataset_id, np.linspace(-1.0, 1.0, 12))
    first = run_isolated(
        canonical,
        tmp_path / "worker-a",
        tmp_path / "controller-a",
        tmp_path / "artifacts-a",
        dataset_id,
    )
    write_dataset(canonical, dataset_id, np.linspace(10.0, 20.0, 12))
    second = run_isolated(
        canonical,
        tmp_path / "worker-b",
        tmp_path / "controller-b",
        tmp_path / "artifacts-b",
        dataset_id,
    )

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] != second[2]
    masked = pd.read_csv(tmp_path / "worker-a" / dataset_id / "test.csv", header=None)
    assert np.all(masked.iloc[:, -1].to_numpy() == execution.DUMMY_TEST_TARGET)


@pytest.mark.xbabe2  # imports cluster_model_queue/model_zoo from /home/ubuntu/remote_super (old state plane)
def test_controller_rejects_prediction_or_report_changes(tmp_path: Path) -> None:
    dataset_id = "tamper"
    canonical = tmp_path / "canonical"
    worker = tmp_path / "worker"
    controller = tmp_path / "controller"
    artifacts = tmp_path / "artifacts"
    write_dataset(canonical, dataset_id, np.linspace(-1.0, 1.0, 12))
    run_isolated(canonical, worker, controller, artifacts, dataset_id)

    prediction_path = artifacts / dataset_id / "ridge" / "predictions.npy"
    changed = np.load(prediction_path, allow_pickle=False)
    changed[0] += 1.0
    np.save(prediction_path, changed, allow_pickle=False)
    with pytest.raises(ValueError, match="score/report disagreement"):
        execution.verify_report(
            dataset_id,
            "ridge",
            canonical_root=canonical,
            artifact_root=artifacts,
        )


def test_nomad_style_headers_are_removed_from_worker_view(tmp_path: Path) -> None:
    dataset_id = "headers"
    canonical = tmp_path / "canonical"
    worker = tmp_path / "worker"
    write_dataset(canonical, dataset_id, np.linspace(-1.0, 1.0, 12))
    execution.prepare_dataset(
        dataset_id,
        canonical_root=canonical,
        worker_root=worker,
        controller_root=tmp_path / "controller",
    )
    first = (worker / dataset_id / "train.csv").read_text().splitlines()[0]
    assert first.split(",") == ["-2", "0", "-0.75"]
    assert read_meta(worker / dataset_id)["headers_removed_from_csv"] is True


def read_meta(path: Path) -> dict:
    return json.loads((path / "meta.json").read_text())
