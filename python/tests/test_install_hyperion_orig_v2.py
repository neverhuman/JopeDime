from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from jope.research import install_hyperion_orig_v2 as subject


def build_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    installed = tmp_path / "installed"
    datasets = []
    for index, dataset_id in enumerate(subject.DATASET_IDS):
        source_dataset = source / dataset_id
        installed_dataset = installed / dataset_id
        source_dataset.mkdir(parents=True)
        installed_dataset.mkdir(parents=True)
        outputs = {}
        for split in ("train", "test"):
            raw = installed_dataset / f"{split}.csv"
            cell = source_dataset / f"{split}3.vs.orig.v2"
            raw.write_text(f"{index + 1},2,3\n4,5,6\n", encoding="utf-8")
            cell.write_text("0.25,0.5\n0.75,1\n", encoding="utf-8")
            outputs[split] = {
                "sha256": subject.sha256_file(cell),
                "shape": [2, 2],
            }
        datasets.append({"dataset_id": dataset_id, "outputs": outputs})
    generation = {
        "schema": subject.GENERATION_SCHEMA,
        "status": "complete",
        "dataset_count": 20,
        "output_file_count": 40,
        "dataset_order": list(subject.DATASET_IDS),
        "datasets": datasets,
    }
    receipt = tmp_path / "generation.json"
    receipt.write_text(json.dumps(generation), encoding="utf-8")
    return receipt, source, installed


def test_complete_exclusive_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generation, source, installed = build_fixture(tmp_path)
    receipt = tmp_path / "installation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install_hyperion_orig_v2.py",
            "--generation-receipt",
            str(generation),
            "--source-root",
            str(source),
            "--installed-root",
            str(installed),
            "--receipt",
            str(receipt),
        ],
    )
    subject.main()
    installed_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert installed_receipt["status"] == "complete"
    assert installed_receipt["output_file_count"] == 40
    assert len(list(installed.glob("*/*.vs.orig.v2"))) == 40

    with pytest.raises(FileExistsError, match="existing destination"):
        subject.main()
