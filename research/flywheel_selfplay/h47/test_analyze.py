from analyze import h46, make_family_metric_rows


def test_family_metric_rows_keep_seed_values_inside_family_clusters():
    seeds = [
        [
            {"family": "f1", "candidate": 1.0, "delta": 0.0},
            {"family": "f2", "candidate": 2.0, "delta": 0.2},
        ],
        [
            {"family": "f1", "candidate": 1.0, "delta": 0.0},
            {"family": "f2", "candidate": 2.0, "delta": 0.2},
        ],
    ]
    rows = make_family_metric_rows(seeds)
    assert len(rows) == 4
    assert {row["family"] for row in rows} == {"f1", "f2"}
    assert {row["seed"] for row in rows} == {0, 1}


def test_median_seed_interval_is_deterministic_and_family_grouped():
    rows = [
        {"family": "f1", "seed": 0, "delta": 0.1, "candidate": 1.1},
        {"family": "f2", "seed": 0, "delta": 0.2, "candidate": 1.2},
        {"family": "f1", "seed": 1, "delta": 0.1, "candidate": 1.1},
        {"family": "f2", "seed": 1, "delta": 0.2, "candidate": 1.2},
    ]
    first = h46.bootstrap_family_interval_of_median_seed_metric(
        rows, "delta", seed=17, n=100
    )
    second = h46.bootstrap_family_interval_of_median_seed_metric(
        rows, "delta", seed=17, n=100
    )
    assert first == second
    assert first[0] <= first[1]


def test_per_seed_family_bootstrap_is_deterministic():
    rows = [
        {"family": "f1", "delta": -0.1},
        {"family": "f1", "delta": -0.2},
        {"family": "f2", "delta": 0.0},
        {"family": "f3", "delta": -0.3},
    ]
    first = h46.bootstrap_ci(rows, seed=99, n=100)
    assert first == h46.bootstrap_ci(rows, seed=99, n=100)
    assert first[0] <= first[1]
