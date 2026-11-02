"""Synthetic H214 protocol tests; no checkpoint, cohort, or utility files opened."""

import hashlib

import numpy as np
import pytest

from research.flywheel_selfplay.h214_pool_diversity import analysis as a
from research.flywheel_selfplay.h214_pool_diversity import protocol as p


def program(i):
    dsl = f"(pipe X (ridge alpha={i}))"
    return {"dsl": dsl, "program_sha256": hashlib.sha256(dsl.encode()).hexdigest()}


def population():
    return [program(i) for i in range(65)]


def observation(**updates):
    row = dict(
        host="xbabe1",
        nice=19,
        disk_used_fraction=0.938,
        disk_free_bytes=40_000_000_000,
        mem_available_bytes=16 * 2**30,
        rss_bytes=5_999_999_999,
        load5=85.0,
        load1=85.0,
        cores=100,
        cpu_pct=85.0,
    )
    return dict(row, **updates)


def test_seed_registration_and_context_binding():
    cid = "a" * 64
    assert len({a.stream_seed(seed, cid) for seed in a.SEEDS}) == 3
    assert a.stream_seed(a.SEEDS[0], cid) != a.stream_seed(a.SEEDS[0], "b" * 64)
    with pytest.raises(ValueError):
        a.stream_seed(1, cid)
    with pytest.raises(ValueError):
        a.stream_seed(a.SEEDS[0], "x" * 64)


def test_native_has_exactly_16_scored_programs_parent_first_and_stable_dedup():
    rows = population()
    result = a.native_pool(rows[0], [rows[0], rows[1], rows[1], *rows[2:]])
    assert result == rows[:16]
    assert len({row["program_sha256"] for row in result}) == 16
    with pytest.raises(ValueError):
        a.native_pool(rows[0], rows[1:10])
    with pytest.raises(ValueError):
        a.native_pool(dict(rows[0], program_sha256="f" * 64), rows[1:])


def test_fps_chooses_opposite_projection_before_nearby_native_proposal():
    rows = population()
    vectors = {}
    for i, row in enumerate(rows):
        vector = np.zeros(16)
        vector[:2] = [np.cos(2 * np.pi * i / 64), np.sin(2 * np.pi * i / 64)]
        vectors[row["program_sha256"]] = vector
    result = a.fps_pool(rows[0], rows[1:], vectors)
    assert len(result) == 16 and result[0] == rows[0] and result[1] == rows[32]
    assert len({row["program_sha256"] for row in result}) == 16


def test_fps_exact_distance_ties_use_original_native_order():
    rows = population()
    vectors = {row["program_sha256"]: np.zeros(16) for row in rows}
    assert a.fps_pool(rows[0], rows[1:], vectors) == rows[:16]
    diagnostic = a.diversity(rows[:16], vectors)
    assert diagnostic == {"unique_programs": 16, "mean_pairwise_distance": 0.0}
    vectors[rows[1]["program_sha256"]][0] = np.nan
    with pytest.raises(ValueError):
        a.fps_pool(rows[0], rows[1:], vectors)


def test_first_native_argmin_keeps_parent_tie_and_gate_failure_penalty():
    assert a.first_argmin_utility([1, 1], [0.1, 0.2], [True, True]) == (-0.1, 0)
    assert a.first_argmin_utility([0, 1], [0.1, 0.2], [False, True]) == (-2.0, 0)


def test_family_cluster_estimator_weights_families_equally():
    families = ["large"] * 73 + [f"small-{i}" for i in range(18)]
    gain = np.zeros((3, 91))
    gain[:, :73] = 1
    result = a.family_gain_stats(gain, families)
    np.testing.assert_allclose(result["seed_equal_family_mean_gain"], [1 / 19] * 3)
    assert result["median_gain"] < 73 / 91
    with pytest.raises(ValueError):
        a.family_gain_stats(gain[:2], families)


def test_estimator_takes_median_over_seeds_and_shared_family_bootstrap():
    families = [f"f-{i % 19}" for i in range(91)]
    gain = np.stack([np.full(91, x) for x in [0.01, 0.02, 0.03]])
    result = a.family_gain_stats(gain, families)
    assert result["median_gain"] == pytest.approx(0.02)
    np.testing.assert_allclose(result["ci95"], [0.02, 0.02], atol=1e-15)
    assert result["p_margin"] == 1 / 10001


@pytest.mark.parametrize(
    "lower,median,pvalue",
    [
        (a.MIN_GAIN, 0.1, 0.001),
        (0.01, 0.0, 0.001),
        (0.01, 0.1, 0.05),
    ],
)
def test_strict_threshold_ties_never_advance(lower, median, pvalue):
    row = dict(median_gain=median, ci95=[lower, 0.2], p_margin=pvalue)
    assert a.decide({"FPS16": row, "TEMP16": row})["verdict"] == "NO_ADVANCE"


def test_holm_uses_both_registered_contrasts_and_missing_arm_is_incomplete():
    good = dict(median_gain=0.02, ci95=[0.01, 0.03], p_margin=0.03)
    weak = dict(median_gain=0.03, ci95=[0.02, 0.04], p_margin=0.04)
    assert a.decide({"FPS16": good})["verdict"] == "INCOMPLETE"
    result = a.decide({"FPS16": good, "TEMP16": weak})
    assert result["holm_p"] == {"FPS16": 0.06, "TEMP16": 0.06}
    assert result["accepted_arms"] == []


def test_smoke_must_be_complete_and_bound_to_exact_source_head():
    row = dict(
        status="PASS",
        source_commit="a" * 40,
        widths=[16, 32, 64],
        pool_record_count=3,
        rust_evaluation_completed=True,
        analysis_row_written=True,
    )
    p.require_smoke(row, "a" * 40)
    for changed in [
        dict(source_commit="b" * 40),
        dict(widths=[16]),
        dict(analysis_row_written=False),
        dict(status="INCOMPLETE"),
    ]:
        with pytest.raises(ValueError):
            p.require_smoke(dict(row, **changed), "a" * 40)


def test_native_temperature_is_required_and_never_emulated():
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        return population()[1:17]

    with pytest.raises(ValueError, match="native temperature hook"):
        p.propose_arm(native, {}, "TEMP16", "a" * 64)
    assert calls == []

    def supported(*, n_proposals, temperature=1.0):
        calls.append((n_proposals, temperature))
        return population()[1 : 1 + n_proposals]

    assert len(p.propose_arm(supported, {}, "TEMP16", "a" * 64)) == 16
    assert calls == [(16, 2.0)]


@pytest.mark.parametrize(
    "updates",
    [
        {"host": "xbabe3"},
        {"nice": 10},
        {"disk_used_fraction": 0.938001},
        {"disk_free_bytes": 39_999_999_999},
        {"mem_available_bytes": 16 * 2**30 - 1},
        {"rss_bytes": 6_000_000_000},
        {"load5": 85.001},
    ],
)
def test_instantaneous_resource_limits_refuse(updates):
    with pytest.raises(RuntimeError):
        p.resource_guard(observation(**updates), 1.0, None)


def test_sustained_high_cpu_aborts_at_300_seconds_and_recovery_resets_timer():
    high = observation(cpu_pct=85.001)
    assert p.resource_guard(high, 0.0, None) == 0.0
    assert p.resource_guard(high, 299.999, 0.0) == 0.0
    with pytest.raises(RuntimeError):
        p.resource_guard(high, 300.0, 0.0)
    assert p.resource_guard(observation(), 200.0, 0.0) is None


@pytest.mark.parametrize(
    "updates", [{"cpu_pct": np.nan}, {"load1": np.inf}, {"cores": 0}]
)
def test_invalid_resource_observation_cannot_bypass_guard(updates):
    with pytest.raises(RuntimeError):
        p.resource_guard(observation(**updates), 0.0, None)
