"""Data-free checks of H244 smoothing, held-family selection, launch binding and A1."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import analysis as stat
import monitor as mon
import smoothing as sp
from runner import Engine

HERE = Path(__file__).parent


def rows(n=205):
    return [dict(context_sha256=f"c{i}", family_id=f"f{i:03}") for i in range(n)]


def test_noise_formula_all_dimensions_zero_rms_and_control():
    state = {
        "w": torch.tensor([[1.0, -2.0], [3.0, 4.0]]),
        "b": torch.tensor([1.0, 2.0]),
        "scalar": torch.tensor(2.0),
        "zero": torch.zeros(3),
        "t": torch.ones(1, 2, 3),
    }
    names = list(state)
    seed = 123
    copy = {n: w.clone() for n, w in state.items()}
    actual = sp.perturb(state, names, 0.03, seed)
    rng = np.random.default_rng(seed)
    for n, w in state.items():
        expected = (
            w.double()
            + 0.03
            * w.double().square().mean().sqrt()
            * torch.tensor(rng.standard_normal(tuple(w.shape)))
        ).to(w.dtype)
        assert torch.equal(actual[n], expected)
        assert torch.equal(state[n], copy[n])
        assert torch.equal(sp.perturb(state, names, 0, seed)[n], w)
    assert torch.equal(actual["zero"], state["zero"])
    assert not torch.equal(actual["b"], state["b"])
    with pytest.raises(ValueError):
        sp.perturb(state, names, 0.05, seed)


def test_frozen_seeds_sigma_and_tensor_census():
    inv = json.loads((HERE / "perturbations.json").read_text())
    assert sp.K == 8 and stat.SIGMAS == (0.0, 0.01, 0.03, 0.1)
    assert stat.SEEDS == (244101, 244102, 244103)
    assert len({x for seeds in sp.MEMBER_SEEDS.values() for x in seeds}) == 24
    assert inv["member_seeds"] == {str(k): v for k, v in sp.MEMBER_SEEDS.items()}
    assert (
        len(inv["tensors"]) == 47
        and sum(r["parameters"] for r in inv["tensors"]) == 29211
    )
    assert stat.DECISION_ARMS == ("WSMOOTH_SELECT",)


def test_mean_cost_not_vote_or_mean_weights():
    costs = [np.array([0.0, 2.0])] + [np.array([10.0, 0.0])] * 8
    assert np.array_equal(sp.mean_cost(costs), np.mean(np.array(costs), axis=0))
    assert sp.mean_cost(costs).argmin() == 1
    with pytest.raises(ValueError):
        sp.mean_cost(costs[:8])
    with pytest.raises(ValueError):
        sp.mean_cost([np.array([np.nan])] * 9)


def test_engine_uses_baseline_plus_eight_predictions_and_restores(tmp_path):
    # Nonlinear score makes averaging weights observably different from averaging predictions.
    e = Engine.__new__(Engine)
    e.net = torch.nn.Linear(1, 1, bias=False)
    e.net.weight.data.fill_(0)
    e.initial = {"weight": e.net.weight.detach().clone()}
    e.pools = {"c": dict(fitness=np.array([-1.0, -2.0]), gate=np.ones(2, dtype=bool))}
    e.cached = {"c": (torch.tensor(0.0), torch.tensor(0.0))}
    e.batch = lambda c, k: ({}, None, None)
    e.check = lambda: None
    seen = []

    def forward(*args):
        w = float(e.net.weight)
        seen.append(w)
        return torch.tensor([[w * w, 2.0]])

    e.forward = forward
    e.cost_path = tmp_path / "costs.npz"
    members = [
        {"weight": torch.tensor([[v]])}
        for v in [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0]
    ]
    result = e.evaluate([dict(context_sha256="c")], members=members)
    assert seen == [0.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0]
    assert result["choices"] == [1] and result["single"]["choices"] == [1]
    assert result["values"] == [2.0] and float(e.net.weight) == 0
    with np.load(e.cost_path) as z:
        np.testing.assert_array_equal(z["c_mean"], [32 / 9, 2.0])


def test_folds_and_selection_strict_ties_and_gain():
    r = rows()
    assert stat.FOLD_SEEDS == {244101: 244141, 244102: 244142, 244103: 244143}
    for seed in stat.SEEDS:
        assert [len(g) for g in stat.folds(r, seed)] == [41] * 5
        for outer in range(5):
            fit, assess = stat.partition(r, outer, seed)
            assert len(fit) == 164 and len(assess) == 41
            assert not {x["family_id"] for x in fit} & {x["family_id"] for x in assess}
    panel = np.ones((205, 4))
    assert stat.choose(r, panel, 1, stat.LABELS)["selected"] == "r882"
    panel[:, 1:] += 0.001
    assert stat.choose(r, panel, 1, stat.LABELS)["selected"] == "r882"
    panel[:, 1:] += 0.01
    assert stat.choose(r, panel, 1, stat.LABELS)["selected"] == "SM01"


def test_selector_rejects_wide_ci():
    r = rows(40)
    panel = np.ones((40, 4))
    panel[0, 1] += 1
    result = stat.choose(r, panel, 4, stat.LABELS)
    assert result["best"] == "SM01" and result["selected"] == "r882"


def test_holm_one_primary_placebo_report_only():
    r = rows(19)
    panel = np.ones((3, 19, 4))
    result = stat.summarize(r, panel)
    c = result["contrasts"]["WSMOOTH_SELECT_minus_r882"]
    assert not c["advance"] and c["holm_p"] == 1
    panel[:, :, 3] += 0.05
    result = stat.summarize(r, panel)
    assert "advance" not in result["contrasts"]["WSINGLE_minus_r882"]
    panel[:, :, 2] += 0.002
    assert not stat.summarize(r, panel)["contrasts"]["WSMOOTH_SELECT_minus_r882"][
        "advance"
    ]
    panel[:2, :, 2] += 0.01
    result = stat.summarize(r, panel)
    assert result["contrasts"]["WSMOOTH_SELECT_minus_r882"]["advance"]
    assert (
        result["contrasts"]["WSMOOTH_SELECT_minus_r882"]["holm_p"]
        == result["contrasts"]["WSMOOTH_SELECT_minus_r882"]["p_one_sided"]
    )


def sample(t, ram, load=50):
    return dict(
        utc=t, available=ram * 2**30, load5=load, rss=2**30, threads=2, free=10**15
    )


def test_a1_transient_memory_recovery_and_load():
    g = mon.Guard()
    t = 1793680000
    assert g.update(sample(t, 11)) == ([], True)
    assert g.update(sample(t + 1, 26)) == ([], True)
    assert g.update(sample(t + 30, 26)) == ([], True)
    assert g.update(sample(t + 31, 26)) == ([], False)
    assert g.update(sample(t + 32, 40, 109)) == ([], True)
    assert g.update(sample(t + 33, 40, 101)) == ([], True)
    assert g.update(sample(t + 34, 40, 100)) == ([], False)
    assert mon.admission_refusals(sample(t, 29))


def test_a1_hard_limits_and_deadline():
    g = mon.Guard()
    t = 1793680000
    g.update(sample(t, 11))
    assert "MemAvailable<12GiB sustained30s" in g.update(sample(t + 30, 11))[0]
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(t, 5))[0]
    assert "absolute compute stop 07:30Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40)
    )
    assert mon.COMPUTE_STOP == 1793691000


def test_launch_fails_closed():
    launch = dict(
        manifest_sha256="m",
        merged_source_verified=True,
        source_commit="h",
        review_id=1,
        prereg_pr=2,
        prereg_sha256="p",
        fold_manifest_sha256="f",
    )
    sp.validate_launch(launch, "h", "m", "p", "f")
    for k, v in [("source_commit", "x"), ("review_id", 0), ("prereg_sha256", "x")]:
        with pytest.raises(RuntimeError):
            sp.validate_launch(dict(launch, **{k: v}), "h", "m", "p", "f")


def test_metrics_schema_and_manifest():
    import jsonschema

    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H244" and m["total_updates"] == 0
    assert m["resource_limits"]["memory_admission_bytes"] == 30 * 2**30
    schema = json.loads((HERE / "metrics.schema.json").read_text())
    jsonschema.validate(json.loads((HERE / "metrics.json").read_text()), schema)
