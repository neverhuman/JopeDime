from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import jope.research

SCRIPTS = Path(jope.research.__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "hyperion_authority_b5_screen", SCRIPTS / "hyperion_authority_b5_screen.py"
)
assert SPEC and SPEC.loader
screen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = screen
SPEC.loader.exec_module(screen)


def authority() -> dict[str, object]:
    return {
        "schema": "HyperionProtectedCellAuthority.v1",
        "encoder_authorized_dataset_count": 2,
        "datasets": [
            {
                "dataset_id": "4f68d277d770477f",
                "train_output_hold_feature_invariant": True,
            },
            {
                "dataset_id": screen.DATASET_ID,
                "train_output_hold_feature_invariant": True,
                "train_sha256": "a" * 64,
                "authoritative_cell_sha256": "b" * 64,
                "relative_path": f"{screen.DATASET_ID}/train3.vs",
            },
        ],
        "test_artifacts_read": False,
    }


def test_authority_requires_one_hold_invariant_b5_entry() -> None:
    assert screen.authority_entry(authority())["dataset_id"] == screen.DATASET_ID
    value = authority()
    value["datasets"][1]["train_output_hold_feature_invariant"] = False
    with pytest.raises(ValueError, match="hold-feature invariant"):
        screen.authority_entry(value)


def test_overlay_contains_only_hash_bound_train_side_files(tmp_path: Path) -> None:
    quant = tmp_path / "quant"
    authority_root = tmp_path / "authority"
    dataset = quant / screen.DATASET_ID
    cached = authority_root / screen.DATASET_ID
    dataset.mkdir(parents=True)
    cached.mkdir(parents=True)
    train = b"1,2,0\n3,4,1\n"
    cell = b"0.1,0.2\n0.3,0.4\n"
    (dataset / "train.csv").write_bytes(train)
    (dataset / "test.csv").write_text("forbidden\n")
    (cached / "train3.vs").write_bytes(cell)
    entry = {
        "train_sha256": screen.common.sha256_file(dataset / "train.csv"),
        "authoritative_cell_sha256": screen.common.sha256_file(cached / "train3.vs"),
        "relative_path": f"{screen.DATASET_ID}/train3.vs",
    }
    receipt = screen.materialize_overlay(quant, authority_root, tmp_path / "overlay", entry)
    overlay = tmp_path / "overlay" / screen.DATASET_ID
    assert sorted(path.name for path in overlay.iterdir()) == ["train.csv", "train3.vs"]
    assert receipt["test_artifacts_read"] is False
    assert (overlay / "train.csv").read_bytes() == train
    assert (overlay / "train3.vs").read_bytes() == cell
