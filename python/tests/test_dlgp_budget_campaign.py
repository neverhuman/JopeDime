import importlib.util
from pathlib import Path

import jope.research

import pytest


SPEC = importlib.util.spec_from_file_location(
    "dlgp_budget_campaign", Path(jope.research.__file__).parent / "dlgp_budget_campaign.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def fixture():
    datasets = ("a" * 16, "b" * 16)
    results = [
        {
            "schema": "JopeStrategyScreenResult.v1",
            "dataset_hash": dataset,
            "strategy": search,
            "master_seed": seed,
            "test_csv_read": False,
        }
        for dataset in datasets
        for search in MODULE.SEARCHES
        for seed in MODULE.SEEDS
    ]
    scores = {
        f"{dataset}|{finale}|{search}|{seed}": (
            0.9 if dataset == datasets[0] and search == "c1000" and finale == "cv7" else 1.1
        )
        for dataset in datasets
        for finale in MODULE.FINALES
        for search in MODULE.SEARCHES
        for seed in MODULE.SEEDS
    }
    baselines = {
        datasets[0]: {"target_rmse": 1.0},
        datasets[1]: {"target_rmse": 0.0},
    }
    return results, scores, baselines


def test_recompute_round2_verdict_uses_two_strict_seeds_and_zero_is_unbeatable():
    results, scores, baselines = fixture()
    verdict = MODULE.recompute_verdict(results, scores, baselines)
    assert verdict["c1000/cv7"]["robust_wins"] == 1
    assert verdict["c1000/cv7"]["any_wins"] == 1
    assert verdict["t5/cv7"]["robust_wins"] == 0


def test_recompute_rejects_test_read_and_duplicate_identity():
    results, scores, baselines = fixture()
    results[0]["test_csv_read"] = True
    with pytest.raises(ValueError, match="test_csv_read=false"):
        MODULE.recompute_verdict(results, scores, baselines)

    results, scores, baselines = fixture()
    results[1] = results[0].copy()
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.recompute_verdict(results, scores, baselines)
