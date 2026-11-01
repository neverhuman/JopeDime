"""Synthetic H196 controls; no scientific inference or corpus reads."""

import numpy as np
import pytest
import torch

from treatment import (
    admission,
    averaged_pick,
    fit_scale,
    noisy_tokens,
    permutations,
    permute_kwargs,
)


def test_fit_excludes_other_contexts_and_masks():
    tokens = {c: np.full((10, 32), x) for c, x in [("a", 1), ("b", 3), ("A", 999)]}
    masks = {c: np.ones(10, bool) for c in tokens}
    masks["b"][1] = False
    scale = fit_scale(["a", "b"], tokens, masks)
    assert np.array_equal(scale[0], np.ones(32))
    assert np.array_equal(scale[1], np.zeros(32))
    tokens["A"] *= 1000
    assert np.array_equal(scale, fit_scale(["b", "a"], tokens, masks))


def test_noise_stream_stability_pairing_mask_and_zero_scale():
    x = np.ones((10, 32), np.float32)
    mask = np.arange(10) < 5
    scale = np.ones_like(x)
    first = noisy_tokens(x, mask, scale, 196101, 0, "a", 0)
    noisy_tokens(x, mask, scale, 196101, 0, "b", 0)
    sixteen = [noisy_tokens(x, mask, scale, 196101, 0, "a", k) for k in range(16)]
    assert np.array_equal(first, sixteen[0])
    assert not np.array_equal(first, sixteen[1])
    assert np.array_equal(first[~mask], x[~mask])
    assert np.array_equal(x, noisy_tokens(x, mask, scale * 0, 196101, 0, "a", 0))
    assert np.array_equal(x, np.ones_like(x))


def test_full_permutation_inverse_and_revealed_history():
    k = 7
    p = np.stack([permutations(196101, "a", k)[0], permutations(196102, "b", k)[0]])
    base = torch.arange(2 * k).reshape(2, k).float()
    kwargs = dict(
        prog_412=base[..., None].expand(2, k, 412),
        proj_16=base[..., None].expand(2, k, 16),
        base_prior_cost=base,
        sos_prior=base[..., None].expand(2, k, 2),
        revealed_indices=torch.tensor([[0, 2, 4, 6], [1, 3, 5, 0]]),
        revealed_raw_f=torch.tensor([[8.0, 9.0, 10.0, 11.0], [1.0, 2.0, 3.0, 4.0]]),
        revealed_gate=torch.ones(2, 4),
        ctx_feat=torch.zeros(2, 584),
    )
    permuted, inv = permute_kwargs(kwargs, p)
    assert torch.equal(permuted["base_prior_cost"].gather(1, inv), base)
    assert torch.equal(
        permuted["base_prior_cost"].gather(1, permuted["revealed_indices"]),
        base.gather(1, kwargs["revealed_indices"]),
    )
    for key in ("revealed_raw_f", "revealed_gate", "ctx_feat"):
        assert permuted[key] is kwargs[key]
    for key in ("prog_412", "proj_16", "sos_prior"):
        assert torch.equal(permuted[key][..., 0].gather(1, inv), base)
    with pytest.raises(ValueError):
        permute_kwargs(kwargs, np.zeros((2, k), int))


def test_average_cost_before_argmin_and_original_order_tie():
    costs = np.array([[0.0, 1.0, 2.0], [20.0, 1.0, 0.0]])
    pick, mean = averaged_pick(costs)
    assert pick == 1 and mean.tolist() == [10.0, 1.0, 1.0]
    assert int(np.argmin(costs[0])) == 0
    with pytest.raises(ValueError):
        averaged_pick([[float("nan")]])


@pytest.mark.parametrize(
    "ci,p,harm,expected",
    [
        ([0.0, 1.0], 0.01, 0.1, False),
        ([0.01, 1.0], 0.05, 0.1, False),
        ([0.01, 1.0], 0.01, 0.100001, False),
        ([0.00001, 1.0], 0.049, 0.1, True),
    ],
)
def test_owner_gate_boundaries(ci, p, harm, expected):
    assert admission(ci, p, harm) is expected
