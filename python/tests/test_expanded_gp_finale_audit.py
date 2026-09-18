from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jope.research

import pytest


SCRIPTS = Path(jope.research.__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "expanded_gp_finale_audit", SCRIPTS / "expanded_gp_finale_audit.py"
)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def fixture_result() -> dict[str, object]:
    members = [
        {"semantic": f"s{index}", "canonical": f"(pipe X (ridge lambda_frac={index}.0))"}
        for index in range(8)
    ]
    return {
        "top": members,
        "finales": {
            "t4_rcv_1se": "s0",
            "control_cv7": "s1",
            "t3_bootstrap": "s2",
            "t2_uniform5": [f"s{index}" for index in range(5)],
            "t2_greedy": ["s0", "s1", "s1", "s2"],
        },
    }


def test_extracts_only_the_five_predeclared_finalizers() -> None:
    arms = audit.extract_arms(fixture_result())
    assert set(arms) == set(audit.ARM_KEYS)
    assert [semantic for semantic, _ in arms["greedy8"]] == ["s0", "s1", "s1", "s2"]
    assert len(arms["uniform5"]) == 5


def test_finalizer_rejects_missing_member_or_arm_drift() -> None:
    result = fixture_result()
    result["finales"]["t3_bootstrap"] = "missing"  # type: ignore[index]
    with pytest.raises(ValueError, match="missing top member"):
        audit.extract_arms(result)
    result = fixture_result()
    result["finales"]["t2_uniform5"] = ["s0"]  # type: ignore[index]
    with pytest.raises(ValueError, match="exactly five"):
        audit.extract_arms(result)


def test_average_predictions_preserves_repeated_ensemble_weight() -> None:
    assert audit.average_predictions([[0.0, 3.0], [3.0, 0.0], [3.0, 0.0]]) == [2.0, 1.0]
    with pytest.raises(ValueError):
        audit.average_predictions([])
