from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h49.run import SWA_CYCLE_UPDATES
from h49.sel import (
    STRICT_MARGIN,
    accept_family_bootstrap_lower_bound,
    historical_noise_audit,
)
from h49.score import arm_seeds


def test_swa_is_one_full_32_update_cycle() -> None:
    assert SWA_CYCLE_UPDATES == 32


def test_sel_margin_is_strict_and_rejects_ties_and_noise() -> None:
    assert not accept_family_bootstrap_lower_bound(0.0)
    assert not accept_family_bootstrap_lower_bound(STRICT_MARGIN)
    assert not accept_family_bootstrap_lower_bound(0.000199)
    assert accept_family_bootstrap_lower_bound(0.001001)


def test_known_live_rule_noise_events_all_fail_margin_without_live_reads() -> None:
    audit = historical_noise_audit()
    assert audit["buffer_k"] == 8
    assert audit["survived"] == 0
    assert audit["total"] == 3


def test_h49_terminal_seed_budget_is_twelve_reads_and_es2_is_fresh() -> None:
    specs = arm_seeds()
    assert [(name, list(seeds)) for name, seeds in specs] == [
        ("0", [0, 1, 2]),
        ("SAM", [0, 1, 2]),
        ("SWA", [0, 1, 2]),
        ("ES2", [3, 4, 5]),
    ]
    assert sum(len(seeds) for _, seeds in specs) == 12
