from analyze import accept_h46_s, bootstrap_ci, make_rows


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
