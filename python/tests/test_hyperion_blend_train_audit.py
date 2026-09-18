from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "hyperion_blend_train_audit", SCRIPTS / "hyperion_blend_train_audit.py"
)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_blend_grid_is_small_fixed_and_model_preserving() -> None:
    assert audit.WEIGHTS == (0.10, 0.25, 0.50)
    assert set(audit.HEADS) == {
        "ridge_gcv",
        "elasticnet",
        "histgbm2",
        "stack",
        "portfolio_core",
    }
    assert audit.blend([1.0, 3.0], [3.0, 1.0], 0.25) == [1.5, 2.5]


def test_candidate_round_trip_rejects_unfrozen_values() -> None:
    for head in audit.HEADS:
        for weight in audit.WEIGHTS:
            assert audit.parse_candidate(audit.candidate_name(head, weight)) == (head, weight)
    try:
        audit.parse_candidate("ridge_gcv@0.33")
    except ValueError:
        pass
    else:
        raise AssertionError("unfrozen blend weight was accepted")
