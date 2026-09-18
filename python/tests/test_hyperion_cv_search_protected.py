from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import jope.research

SCRIPTS = Path(jope.research.__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "hyperion_cv_search_protected", SCRIPTS / "hyperion_cv_search_protected.py"
)
assert SPEC and SPEC.loader
screen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = screen
SPEC.loader.exec_module(screen)


def base_config() -> dict[str, object]:
    return {
        "schema": "JopeStrategyScreenConfig.v1",
        "strategy": "c1000",
        "dataset_hash": "development-a",
        "master_seed": 101,
        "dl_genes": [],
        "budget": dict(screen.expanded.EXPECTED_BUDGET),
        "final_cv_folds": 7,
        "final_top_k": 10,
        "perturb_fitness": False,
    }


def parent_result() -> dict[str, object]:
    return {
        "schema": "JopeStrategyScreenResult.v1",
        "strategy": "c2000",
        "dataset_hash": "development-a",
        "master_seed": 101,
        "test_csv_read": False,
        "search_evaluations": 2_000,
        "final_cv_evaluations": 70,
        "winner_genome": "(pipe X (ridge))",
        "winner_semantic": "winner",
        "top": [
            {"canonical": f"(pipe X (ridge lambda_frac={index}.0))"}
            for index in range(10)
        ],
    }


def test_v2_search_is_exact_matched_and_does_not_mutate_authorities() -> None:
    base = base_config()
    parent = parent_result()
    raw = screen.strict_v2_config(base, parent, "development-a", 101, "raw")
    cell = screen.strict_v2_config(base, parent, "development-a", 101, "hyperion_cell")
    assert base["schema"] == "JopeStrategyScreenConfig.v1"
    assert parent["strategy"] == "c2000"
    for config in (raw, cell):
        assert config["schema"] == "JopeStrategyScreenConfig.v2"
        assert config["budget"]["population"] == 40
        assert config["budget"]["generations"] == 25
        assert config["budget"]["cv_fitness"] == 2
        assert config["budget"]["splits_per_gen"] == 0
        assert config["drop_failed_folds"] is False
        assert config["pooled_oof"] is True
        assert config["dl_genes"] == [member["canonical"] for member in parent["top"]]
    assert raw["dl_genes"] == cell["dl_genes"]
    assert raw["master_seed"] == cell["master_seed"]
    assert raw["strategy"] == screen.strategy_name("raw")
    assert cell["strategy"] == screen.strategy_name("hyperion_cell")


def test_config_rejects_foreign_view_archive_and_base_drift() -> None:
    with pytest.raises(ValueError, match="unsupported feature view"):
        screen.strict_v2_config(
            base_config(), parent_result(), "development-a", 101, "row128"
        )
    parent = parent_result()
    parent["top"] = parent["top"][:-1]
    with pytest.raises(ValueError, match="top-ten"):
        screen.strict_v2_config(base_config(), parent, "development-a", 101, "raw")
    base = base_config()
    base["budget"] = {**base["budget"], "generations": 24}
    with pytest.raises(ValueError, match="budget drift"):
        screen.strict_v2_config(base, parent_result(), "development-a", 101, "raw")


def test_staging_reads_only_train_and_exact_cell_matrix(tmp_path: Path) -> None:
    dataset = tmp_path / "fixture"
    dataset.mkdir()
    (dataset / "train.csv").write_text("1,2,0\n3,4,1\n5,6,2\n7,8,3\n", encoding="utf-8")
    (dataset / "train3.vs").write_text(
        "0.1,0.2\n0.3,0.4\n0.5,0.6\n0.7,0.8\n", encoding="utf-8"
    )
    # A forbidden path can exist, but the API cannot name it.
    (dataset / "test.csv").write_text("secret\n", encoding="utf-8")
    destination = tmp_path / "staged" / "train.csv"
    _, metadata = screen.stage_view(dataset, "hyperion_cell", destination)
    features, targets, _ = screen.common.read_training_csv(destination)
    assert features.shape == (4, 4)
    assert targets.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert metadata["raw_features"] == 2
    assert metadata["output_features"] == 4
    assert metadata["cell_sha256"] is not None
    assert metadata["test_artifacts_read"] is False
    with pytest.raises(ValueError, match="forbidden"):
        screen.checked_train_path(dataset, "test.csv")


def test_result_requires_complete_strict_search() -> None:
    result = {
        "schema": "JopeStrategyScreenResult.v2",
        "strategy": screen.strategy_name("raw"),
        "dataset_hash": "development-a",
        "master_seed": 101,
        "test_csv_read": False,
        "search_evaluations": 2_000,
        "final_cv_evaluations": 70,
        "winner_genome": "(pipe X (ridge))",
        "strict_fold_contract": {
            "complete_predictions_required": True,
            "fail_on_incomplete_candidate": True,
            "candidate_timeout_seconds": screen.EXPECTED_EVALUATOR_TIMEOUT_SECONDS,
        },
        "generations": [{"invalid_members": 0} for _ in range(25)],
    }
    screen.validate_result(result, "development-a", 101, "raw")
    result["strict_fold_contract"]["candidate_timeout_seconds"] = 90
    with pytest.raises(ValueError, match="incomplete, foreign"):
        screen.validate_result(result, "development-a", 101, "raw")
    result["strict_fold_contract"]["candidate_timeout_seconds"] = (
        screen.EXPECTED_EVALUATOR_TIMEOUT_SECONDS
    )
    result["generations"][7]["invalid_members"] = 1
    with pytest.raises(ValueError, match="incomplete candidate"):
        screen.validate_result(result, "development-a", 101, "raw")


def test_completed_temporary_recovery_requires_one_strict_result(
    tmp_path: Path,
) -> None:
    result = {
        "schema": "JopeStrategyScreenResult.v2",
        "strategy": screen.strategy_name("raw"),
        "dataset_hash": "development-a",
        "master_seed": 101,
        "test_csv_read": False,
        "search_evaluations": 2_000,
        "final_cv_evaluations": 70,
        "winner_genome": "(pipe X (ridge))",
        "strict_fold_contract": {
            "complete_predictions_required": True,
            "fail_on_incomplete_candidate": True,
            "candidate_timeout_seconds": screen.EXPECTED_EVALUATOR_TIMEOUT_SECONDS,
        },
        "generations": [{"invalid_members": 0} for _ in range(25)],
    }
    temporary = tmp_path / ".result.json.tmp.123"
    temporary.write_text(screen.json.dumps(result), encoding="utf-8")
    recovered = screen.recover_completed_result(tmp_path, "development-a", 101, "raw")
    assert recovered is not None
    assert recovered[0] == result
    assert recovered[1] == temporary
    assert temporary.is_file()
