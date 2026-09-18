from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "expanded_gp_protected_screen", SCRIPTS / "expanded_gp_protected_screen.py"
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
        "budget": dict(screen.EXPECTED_BUDGET),
        "final_cv_folds": 7,
        "final_top_k": 10,
        "perturb_fitness": False,
    }


def test_treatment_is_exact_and_does_not_mutate_frozen_base() -> None:
    base = base_config()
    treatment = screen.treatment_config(base, "development-a", 101)
    assert all(treatment[key] == value for key, value in screen.TREATMENT.items())
    assert not any(key in base for key in screen.TREATMENT)
    assert treatment["budget"] == screen.EXPECTED_BUDGET
    assert screen.SELECTION_ACCOUNTING == {
        "search_evaluations": 1000,
        "final_cv_refits": 70,
        "repeated_cv_refits": 350,
        "bootstrap_refits": 200,
        "ensemble_oof_refits": 70,
        "total_train_only_fits": 1690,
    }


def test_treatment_rejects_budget_or_identity_drift() -> None:
    base = base_config()
    base["budget"] = {**screen.EXPECTED_BUDGET, "generations": 24}
    with pytest.raises(ValueError, match="budget drift"):
        screen.treatment_config(base, "development-a", 101)
    with pytest.raises(ValueError, match="identity mismatch"):
        screen.treatment_config(base_config(), "confirmation-a", 101)


def test_continuation_uses_exact_parent_archive_and_rung_budget() -> None:
    parent = {
        "schema": "JopeStrategyScreenResult.v1",
        "strategy": "c1000",
        "dataset_hash": "development-a",
        "master_seed": 101,
        "test_csv_read": False,
        "search_evaluations": 1000,
        "final_cv_evaluations": 70,
        "winner_genome": "(pipe X (ridge))",
        "winner_semantic": "winner",
        "top": [
            {"canonical": f"(pipe X (ridge lambda_frac={index}.0))"}
            for index in range(10)
        ],
    }
    config = screen.continuation_config(
        base_config(), parent, "development-a", 101, 2000
    )
    assert config["strategy"] == "c2000"
    assert config["budget"]["generations"] == 50
    assert config["dl_genes"] == [member["canonical"] for member in parent["top"]]
    assert screen.selection_accounting(2000)["total_train_only_fits"] == 2690


def test_protected_ids_must_be_development_only() -> None:
    protocol = {
        "cohorts": {
            "development": ["a", "b", *[f"d{index}" for index in range(20)]],
            "confirmation": ["c"],
        },
        "best5_screen": {"datasets": ["a", "b"]},
    }
    assert screen.protected_ids(protocol) == ("a", "b")
    protocol["best5_screen"]["datasets"] = ["a", "c"]
    with pytest.raises(ValueError):
        screen.protected_ids(protocol)


def test_result_requires_train_only_exact_budget() -> None:
    result = {
        "schema": "JopeStrategyScreenResult.v1",
        "strategy": "c1000",
        "dataset_hash": "development-a",
        "master_seed": 101,
        "test_csv_read": False,
        "search_evaluations": 1000,
        "final_cv_evaluations": 70,
        "winner_genome": "(pipe X (ridge))",
        "winner_semantic": "abc",
    }
    screen.validate_search_result(result, "development-a", 101)
    result["test_csv_read"] = True
    with pytest.raises(ValueError, match="train-only"):
        screen.validate_search_result(result, "development-a", 101)
