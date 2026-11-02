"""Synthetic, outcome-free tests for H212's preregistered protocol."""

import numpy as np
import pytest

import analysis


def test_registered_seed_streams_are_stable_and_distinct():
    cid = "a" * 64
    values = [analysis.stream_seed(seed, cid) for seed in analysis.SEEDS]
    assert values == [analysis.stream_seed(seed, cid) for seed in analysis.SEEDS]
    assert len(set(values)) == 3
    with pytest.raises(ValueError):
        analysis.stream_seed(7, cid)


def test_family_folds_are_complete_disjoint_and_balanced():
    ids = [f"fam-{i:03d}" for i in range(205)]
    folds = analysis.family_folds(ids)
    assert [len(fold) for fold in folds] == [41] * 5
    assert len(set().union(*(set(x) for x in folds))) == 205
    assert all(not (set(folds[i]) & set(folds[j])) for i in range(5) for j in range(i))


def test_native_first_argmin_preserves_first_tie_and_gate_failure_cost():
    utility, index = analysis.first_argmin_utility(
        [0.2, 0.1, 0.1], [0.5, 0.8, 0.9], [True, False, True]
    )
    assert (utility, index) == (-2.0, 1)


def test_family_bootstrap_reports_strict_ties_as_not_wins():
    gain = np.zeros(205)
    gain[0] = 1.0
    gain[1] = -1.0
    ix = np.tile(np.arange(205), (8, 1))
    stats = analysis.family_gain_stats(gain, ix)
    assert stats["win_tie_loss"] == [1, 203, 1]
    assert stats["harm_fraction"] == 1 / 205
    assert stats["ci95"] == [0.0, 0.0]


def test_primary_requires_all_seed_ci_harm_and_elo_gates():
    passing = {s: {"ci95": [0.01, 0.1], "harm_fraction": 0.2} for s in analysis.SEEDS}
    elo = {s: 1500.01 for s in analysis.SEEDS}
    assert analysis.decide(passing, elo)["verdict"] == "ADVANCE"
    passing[analysis.SEEDS[-1]] = {"ci95": [0.0, 0.1], "harm_fraction": 0.0}
    assert analysis.decide(passing, elo)["verdict"] == "NO_ADVANCE"
    assert analysis.decide({analysis.SEEDS[0]: passing[analysis.SEEDS[0]]}, elo)["verdict"] == "INCOMPLETE"


def test_width_outputs_are_validated_independently():
    arms = {
        width: [{"program_sha256": f"{width * 100 + i:064x}"} for i in range(width)]
        for width in analysis.WIDTHS
    }
    assert [len(analysis.validate_width_proposals(arms[w], w)) for w in analysis.WIDTHS] == [16, 32, 64]
    assert arms[16] != arms[64][:16]
    with pytest.raises(ValueError, match="exactly 16"):
        analysis.validate_width_proposals(arms[16][:15], 16)
    arms[64][-1] = arms[64][0]
    with pytest.raises(ValueError, match="unique"):
        analysis.validate_width_proposals(arms[64], 64)
