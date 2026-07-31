from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import json


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "test_exact_dl_generate", SCRIPT_DIR / "test_exact_dl_generate.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def proposal(semantic: str, score: float, head: str) -> dict:
    return {
        "semantic_sha256": semantic * 64,
        "selection_score": score,
        "_head_op": head,
    }


def test_reward_order_changes_production_slots_and_preserves_global_best() -> None:
    rows = [
        proposal("a", 0.1, "ridge"),
        proposal("b", 0.9, "ridge"),
        proposal("c", 0.4, "gpr"),
    ]
    selected = MODULE.rank_scored_proposals(rows, 3)
    assert selected[0]["semantic_sha256"] == "b" * 64
    assert [row["semantic_sha256"] for row in selected] != [
        row["semantic_sha256"] for row in rows
    ]


def test_diversity_starts_after_slot_zero_and_never_duplicates() -> None:
    rows = [
        proposal("a", 1.0, "ridge"),
        proposal("b", 0.99, "ridge"),
        proposal("c", 0.5, "gpr"),
        proposal("d", 0.4, "elasticnet"),
    ]
    selected = MODULE.rank_scored_proposals(rows, 3)
    assert [row["_head_op"] for row in selected] == [
        "ridge",
        "gpr",
        "elasticnet",
    ]
    assert len({row["semantic_sha256"] for row in selected}) == len(selected)


def test_context_deficiency_emits_fewer_rows_instead_of_backfill() -> None:
    rows = [proposal("a", 1.0, "ridge"), proposal("b", 0.9, "gpr")]
    selected = MODULE.rank_scored_proposals(rows, 8)
    assert len(selected) == 2


def test_exact_dataset_manifest_preserves_order_and_rejects_drift(tmp_path) -> None:
    entries = [
        {
            "task": "regression",
            "dataset_hash": "a",
            "context_sha256": "1" * 64,
        },
        {
            "task": "regression",
            "dataset_hash": "b",
            "context_sha256": "2" * 64,
        },
        {
            "task": "binary",
            "dataset_hash": "c",
            "context_sha256": "3" * 64,
        },
    ]
    manifest = tmp_path / "datasets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "TrainOnlyProposalDatasetManifest.v1",
                "catalog_sha256": "f" * 64,
                "entries": [
                    {
                        "task": "regression",
                        "dataset_hash": "b",
                        "context_sha256": "2" * 64,
                    },
                    {
                        "task": "regression",
                        "dataset_hash": "a",
                        "context_sha256": "1" * 64,
                    },
                ],
            }
        )
    )
    selected = MODULE.filter_exact_dataset_manifest(
        entries, manifest, catalog_sha256="f" * 64
    )
    assert [row["dataset_hash"] for row in selected] == ["b", "a"]

    value = json.loads(manifest.read_text())
    value["entries"][0]["context_sha256"] = "0" * 64
    manifest.write_text(json.dumps(value))
    try:
        MODULE.filter_exact_dataset_manifest(
            entries, manifest, catalog_sha256="f" * 64
        )
    except RuntimeError as error:
        assert "absent or changed" in str(error)
    else:
        raise AssertionError("context drift must fail closed")
