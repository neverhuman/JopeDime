from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("raw_floor_train_audit", SCRIPTS / "raw_floor_train_audit.py")
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_screen_requires_one_percent_against_both_seeds() -> None:
    raw = {101: 1.0, 202: 0.9}
    assert audit.screen_choice({"ridge": 0.895}, raw) is None
    choice = audit.screen_choice({"ridge": 0.88, "stack": 0.87}, raw)
    assert choice is not None
    assert choice["candidate"] == "stack"


def test_candidate_set_is_fixed_and_production_grammar_only() -> None:
    assert set(audit.CANDIDATES) == {
        "ridge",
        "ridge_gcv",
        "elasticnet",
        "histgbm2",
        "stack",
        "portfolio_core",
    }
    assert all("hyperion" not in genome and "test" not in genome for genome in audit.CANDIDATES.values())
