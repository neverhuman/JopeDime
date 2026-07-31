from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import golden46_special_execution as special  # noqa: E402


def test_special_specs_match_frozen_cohort() -> None:
    assert set(special.SPECIAL_SPECS) == special.frozen_special_ids()


def test_chimera_blocker_is_unresolved_and_forbids_substitution(tmp_path: Path) -> None:
    receipt = special.compatibility_blocker(
        "ZTS_small",
        "chimera_regression",
        artifact_root=tmp_path,
    )
    assert receipt["status"] == "blocked-missing-frozen-artifact"
    assert receipt["benchmark_cell_resolved"] is False
    assert receipt["substitution_permitted"] is False
    stored = json.loads(
        (tmp_path / "ZTS_small/chimera_regression/compatibility-receipt.json").read_text()
    )
    assert stored["missing_artifact"]["sha256"] == special.CHIMERA_EXPECTED["sha256"]


def test_chimera_conversion_is_bound_to_frozen_source_identity() -> None:
    identity = special.validate_chimera_artifacts()
    assert identity["source_checkpoint"]["sha256"] == special.CHIMERA_EXPECTED["sha256"]
    assert (
        identity["converted_checkpoint"]["sha256"]
        == "57cdb53ae732a705e6aee7381a6a11e7e82cf357bcd67c46ccfa47b789c7d5d2"
    )
    assert identity["embedded_metadata"] == {
        "architecture": "StackFusionReg",
        "format": "pt_to_safetensors",
        "source_sha256": special.CHIMERA_EXPECTED["sha256"],
    }
    assert identity["tensor_count"] == 32
    assert (
        identity["conversion_receipt_sha256"]
        == "a86a63bca8148ca3238472a3daf838559a0666fa6e45b96afd34438bb5e3fb46"
    )


def test_chimera_special_identity_records_source_and_converted_artifacts() -> None:
    identity = special.special_identity("chimera_regression")
    assert identity["parameters"]["source_checkpoint"] == special.CHIMERA_EXPECTED
    assert (
        identity["parameters"]["converted_checkpoint"]
        == special.CHIMERA_CONVERTED_EXPECTED
    )


def test_worker_array_loader_never_returns_test_target(tmp_path: Path) -> None:
    train = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    test_a = np.array([[7.0, 8.0, 0.0], [9.0, 10.0, 0.0]])
    test_b = test_a.copy()
    test_b[:, -1] = [111.0, 222.0]
    np.savetxt(tmp_path / "train.csv", train, delimiter=",")
    np.savetxt(tmp_path / "test.csv", test_a, delimiter=",")
    x_train_a, y_train_a, x_test_a = special.load_worker_arrays(tmp_path)
    np.savetxt(tmp_path / "test.csv", test_b, delimiter=",")
    x_train_b, y_train_b, x_test_b = special.load_worker_arrays(tmp_path)
    np.testing.assert_array_equal(x_train_a, x_train_b)
    np.testing.assert_array_equal(y_train_a, y_train_b)
    np.testing.assert_array_equal(x_test_a, x_test_b)
