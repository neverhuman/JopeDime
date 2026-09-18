from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import jope.research

SCRIPTS = Path(jope.research.__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "hyperion_final_cv7_train_benchmark",
    SCRIPTS / "hyperion_final_cv7_train_benchmark.py",
)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def prior() -> dict[str, list[dict[str, object]]]:
    rows = []
    for expert in benchmark.EXPERTS:
        rows.append(
            {
                "semantic": expert["semantic"],
                "genome": expert["genome"],
                "mean_pct": 75.0 if expert is benchmark.EXPERTS[0] else 69.0,
                "worst_pct": 53.0 if expert is benchmark.EXPERTS[0] else 55.0,
            }
        )
    return {
        f"development-{index}": rows if index else list(reversed(rows))
        for index in range(22)
    }


def test_expert_authority_requires_primary_to_rank_first_21_of_22() -> None:
    evidence = benchmark.validate_expert_authority(prior())
    assert evidence[benchmark.EXPERTS[0]["name"]]["top_rank_count"] == 21
    broken = prior()
    broken["development-1"][0]["genome"] = "(pipe X (ridge_gcv))"
    with pytest.raises(ValueError, match="missing expert"):
        benchmark.validate_expert_authority(broken)


def test_cv7_fold_plan_rejects_tiny_tables_and_enforces_guards() -> None:
    with pytest.raises(ValueError):
        benchmark.fold_plan([f"g-{index}" for index in range(37)], 303)
    plan = benchmark.fold_plan([f"g-{index}" for index in range(50)], 303)
    assert len(plan) == 7
    assert all(len(fold.fit_rows) >= 32 for fold in plan)
    assert all(len(fold.hold_rows) >= 2 for fold in plan)
    assert all(len(fold.hold_groups) >= 2 for fold in plan)


def test_fixed_weighted_average_uses_both_experts() -> None:
    assert benchmark.weighted_average([[1.0, 3.0], [3.0, 5.0]]) == [2.0, 4.0]
    with pytest.raises(ValueError, match="coverage"):
        benchmark.weighted_average([[1.0]])


def test_confirmation_cohort_requires_exactly_twenty_unique_ids() -> None:
    protocol = {
        "schema": "JopeDlgpBudgetFirstProtocol.v1",
        "cohorts": {"confirmation": [f"dataset-{index}" for index in range(20)]},
    }
    assert len(benchmark.confirmation_ids(protocol)) == 20
    protocol["cohorts"]["confirmation"][-1] = "dataset-0"
    with pytest.raises(ValueError, match="incomplete or foreign"):
        benchmark.confirmation_ids(protocol)
