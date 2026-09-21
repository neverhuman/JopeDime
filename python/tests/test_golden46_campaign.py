from __future__ import annotations

import json
from pathlib import Path

import pytest

import jope.research

SCRIPTS = Path(jope.research.__file__).parent

import golden46_cluster_campaign as cluster  # noqa: E402
import golden46_historical_receipts as historical  # noqa: E402
import golden46_matrix_audit as matrix  # noqa: E402
import golden46_required_study_audit as required  # noqa: E402


def test_historical_aspfm_receipt_is_terminal_without_a_score(tmp_path: Path) -> None:
    receipt = historical.write_receipt("ZTS", tmp_path)
    assert receipt["benchmark_cell_resolved"] is True
    assert receipt["score_present"] is False
    assert receipt["method_identity_changed"] is False
    stored = json.loads((tmp_path / "ZTS/jope-dime-regressor/terminal-receipt.json").read_text())
    assert stored == receipt


@pytest.mark.xbabe2  # imports the quant model zoo from /home/ubuntu/remote_super (old state plane)
def test_full_zts_registered_gpu_methods_are_all_frozen_ineligible() -> None:
    assert cluster.eligible_methods("ZTS", {"gpu", "gpu_heavy"}) == []
    assert len(cluster.eligible_methods("ZTS", {"cpu", "cpu_light"})) == 96


def test_matrix_contract_accounts_for_all_historical_cells(tmp_path: Path) -> None:
    payload = matrix.matrix_audit(
        matrix.DEFAULT_ARTIFACT_ROOT,
        tmp_path,
    )
    contract = payload["resolution_contract"]
    assert contract["validated_scores"] == 476
    assert contract["explicit_terminal_receipts"] == 24
    assert sum(contract["terminal_breakdown"].values()) == 24
    assert 476 + 24 == payload["expected_cells"]


def test_required_study_excludes_full_zts(tmp_path: Path) -> None:
    payload = required.required_study_audit(
        required.DEFAULT_ARTIFACT_ROOT,
        tmp_path,
    )
    assert payload["dataset_order"] == list(required.DATASETS)
    assert payload["scope"]["excluded_dataset_ids"] == ["ZTS"]
    assert payload["resolution_contract"] == {
        "validated_scores": 372,
        "explicit_terminal_receipts": 3,
        "terminal_breakdown": {"historical_aspfm_not_rerun": 3},
    }
    assert payload["expected_cells"] == 375
