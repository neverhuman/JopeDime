from analyze import (
    accept_h46_s,
    bootstrap_ci,
    bootstrap_family_interval_of_median_seed_metric,
    make_rows,
)
from losses import family_balanced_weights


def test_h46_s_rejects_exact_ties_and_requires_positive_lower_bound():
    assert not accept_h46_s(0.0, [0.0, 0.0])
    assert not accept_h46_s(0.001, [-0.001, 0.003])
    assert not accept_h46_s(0.0001, [0.00001, 0.001])
    assert accept_h46_s(0.0002, [0.00001, 0.001])


def test_bootstrap_resamples_families_and_is_deterministic():
    rows = [
        {"family": "a", "delta": 1.0},
        {"family": "a", "delta": 1.0},
        {"family": "b", "delta": -1.0},
    ]
    first = bootstrap_ci(rows, seed=7, n=100)
    assert first == bootstrap_ci(rows, seed=7, n=100)
    assert first[0] <= first[1]


def test_make_rows_preserves_context_family_pairs():
    rows = make_rows(
        [{"context_sha256": "c1", "family_id": "f1"}],
        [1.0],
        [1.25],
    )
    assert rows == [
        {
            "context": "c1",
            "family": "f1",
            "baseline": 1.0,
            "candidate": 1.25,
            "delta": 0.25,
        }
    ]


def test_family_interval_of_median_seed_metric_is_deterministic():
    rows = [
        {"family": "a", "seed": 0, "candidate": 1.0},
        {"family": "b", "seed": 0, "candidate": 2.0},
        {"family": "a", "seed": 1, "candidate": 1.5},
        {"family": "b", "seed": 1, "candidate": 2.5},
    ]
    first = bootstrap_family_interval_of_median_seed_metric(
        rows, "candidate", seed=9, n=100
    )
    assert first == bootstrap_family_interval_of_median_seed_metric(
        rows, "candidate", seed=9, n=100
    )
    assert first[0] <= first[1]


def test_family_balanced_weights_equalize_total_family_contribution():
    weights = family_balanced_weights(["large", "large", "large", "small"])
    assert weights.sum() == 4.0
    assert weights[:3].sum() == weights[3]
