from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jope.research

import numpy as np


SCRIPTS = Path(jope.research.__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "hyperion_residual_train_audit", SCRIPTS / "hyperion_residual_train_audit.py"
)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_residual_arm_grid_is_fixed_to_six_shrunken_ridges() -> None:
    assert len(audit.CORRECTIONS) == 6
    assert set(alpha for alpha, _ in audit.CORRECTIONS.values()) == {0.01, 0.10}
    assert set(weight for _, weight in audit.CORRECTIONS.values()) == {0.25, 0.50, 1.00}


def test_residual_ridge_learns_intercept_and_signal() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    residuals = np.asarray([1.0, 3.0, 5.0, 7.0])
    predicted = audit.ridge_residual_predictions(
        features, residuals, np.asarray([[1.5]]), alpha=1e-8
    )
    assert np.allclose(predicted, [4.0], atol=1e-6)


def test_inner_folds_are_atomic_and_cover_only_outer_fit() -> None:
    groups = ["a", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    outer = audit.common.Fold(
        index=0,
        fit_rows=tuple(range(10)),
        hold_rows=(10, 11),
        hold_groups=("j", "k"),
    )
    folds = audit.inner_folds(groups, outer)
    covered = [row for fold in folds for row in fold.hold_rows]
    assert sorted(covered) == list(range(10))
    assert not ({10, 11} & set(covered))
    for fold in folds:
        assert set(fold.fit_rows).isdisjoint(fold.hold_rows)
        assert not ({10, 11} & set(fold.fit_rows))
    assert all(0 in fold.hold_rows if 1 in fold.hold_rows else 0 not in fold.hold_rows for fold in folds)
