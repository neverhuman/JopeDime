from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).with_name("sensitivity.py")
SPEC = importlib.util.spec_from_file_location("h50_sensitivity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
h50 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h50)
SEARCH_PATH = Path(__file__).with_name("search.py")
SEARCH_SPEC = importlib.util.spec_from_file_location("h50_search", SEARCH_PATH)
assert SEARCH_SPEC is not None and SEARCH_SPEC.loader is not None
search = importlib.util.module_from_spec(SEARCH_SPEC)
SEARCH_SPEC.loader.exec_module(search)


def test_flat_direction_is_unit_norm_and_seeded():
    parameters = [torch.zeros(3), torch.zeros(2, 2)]

    first = h50.flat_direction(123, parameters)
    second = h50.flat_direction(123, parameters)

    assert [tuple(value.shape) for value in first] == [(3,), (2, 2)]
    assert torch.equal(torch.cat([value.flatten() for value in first]), torch.cat([value.flatten() for value in second]))
    norm = torch.linalg.vector_norm(torch.cat([value.flatten() for value in first]))
    assert torch.isclose(norm, torch.tensor(1.0, dtype=norm.dtype))


def test_flat_direction_changes_when_seed_changes():
    parameters = [torch.zeros(5)]
    first = h50.flat_direction(123, parameters)[0]
    second = h50.flat_direction(124, parameters)[0]

    assert not torch.equal(first, second)


def test_flatten_and_unflatten_preserve_adapter_state():
    template = {"a": torch.arange(3, dtype=torch.float32), "b": torch.arange(4, dtype=torch.float32).reshape(2, 2)}

    vector = search.flatten_snapshot(template)
    restored = search.unflatten_snapshot(vector, template)

    assert torch.equal(restored["a"], template["a"])
    assert torch.equal(restored["b"], template["b"])


def test_centered_ranks_are_deterministic_and_centered():
    ranks = search.centered_ranks([4.0, 1.0, 3.0, 2.0])

    assert ranks.tolist() == [0.375, -0.375, 0.125, -0.125]
    assert ranks.sum() == 0.0


def test_es_generation_is_antithetic_with_population_32():
    center = torch.zeros(17)
    noises, candidates = search.es_generation(center, 0.001, seed=6, generation=1)

    assert len(noises) == 16
    assert len(candidates) == 32
    for positive, negative in zip(candidates[::2], candidates[1::2], strict=True):
        assert torch.equal(positive, -negative)


def test_candidate_at_or_below_margin_uses_finite_ci_sentinel():
    deltas = {f"f{i}": 0.0 for i in range(5)}
    median, lower, upper = search.candidate_median_ci(
        deltas,
        torch.zeros((2, 5), dtype=torch.int64),
    )

    assert (median, lower, upper) == (0.0, 0.0, 0.0)
