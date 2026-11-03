"""Data-free tests: optimizer exclusion, exact AdamW, strict statistics and A1 guard."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import analysis as stat
import monitor as mon
import runner
from treatment import (
    ARMS,
    INVENTORY,
    adapter_drift,
    arm_loss,
    check_subset,
    make_optimizer,
    validate_launch,
)

HERE = Path(__file__).resolve().parent


def parameters():
    return {
        r["name"]: torch.nn.Parameter(torch.full(r["shape"], 0.25, dtype=torch.float64))
        for r in INVENTORY["tensors"]
    }


def test_exact_census_and_no_anchor_or_buffers():
    specs = INVENTORY["arms"]
    assert [(specs[a]["tensors"], specs[a]["parameters"]) for a in ARMS] == [
        (47, 29211),
        (23, 755),
        (2, 25),
        (47, 29211),
    ]
    assert specs["HEAD_ONLY"]["names"] == [
        "adapter_10tok.traj_delta_head.2.weight",
        "adapter_10tok.traj_delta_head.2.bias",
    ]
    assert len({r["name"] for r in INVENTORY["tensors"]}) == 47
    assert all(
        ".anchor." not in r["name"] and r["name"].startswith("adapter_10tok.")
        for r in INVENTORY["tensors"]
    )
    assert {r["llrd_depth"] for r in INVENTORY["tensors"]} == set(range(6))
    for r in INVENTORY["tensors"]:
        assert r["llrd_multiplier"] == 0.5 ** r["llrd_depth"]
        assert r["numel"] == int(np.prod(r["shape"]))


@pytest.mark.parametrize("arm", ARMS)
def test_optimizer_matches_independent_adamw_and_never_moves_frozen_coordinates(arm):
    named = parameters()
    initial = {n: p.detach().clone() for n, p in named.items()}
    for p in named.values():
        p.grad = torch.ones_like(p)  # stale gradients must be cleared
    optimizer = make_optimizer(arm, named)
    allowed = set(INVENTORY["arms"][arm]["names"])
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for n, p in named.items() if n in allowed
    }
    depth = {r["name"]: r["llrd_multiplier"] for r in INVENTORY["tensors"]}
    refs = {n: torch.nn.Parameter(initial[n].clone()) for n in allowed}
    independent = {
        n: torch.optim.AdamW(
            [p],
            lr=1e-4 * (depth[n] if arm == "LLRD" else 1),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )
        for n, p in refs.items()
    }
    for step in range(1, 41):
        optimizer.zero_grad(set_to_none=True)
        for n, p in named.items():
            if n in allowed and ".parent_delta_net." not in n:
                p.grad = torch.full_like(p, step / 40)
                refs[n].grad = p.grad.clone()
            elif n in refs:
                refs[n].grad = None
        optimizer.step()
        for opt in independent.values():
            opt.step()
        check_subset(arm, named, initial)
        for n, p in refs.items():
            assert torch.equal(named[n], p), n
        for n in set(named) - allowed:
            assert torch.equal(named[n], initial[n]) and named[n].grad is None
    assert any(not torch.equal(p, initial[n]) for n, p in named.items())
    assert all(
        not optimizer.state[p] for n, p in named.items() if ".parent_delta_net." in n
    )
    assert arm_loss(arm, lambda *args: 123, None, None, None) == 123


def test_subset_drift_is_rejected_and_arm_switch_reenables_full_control():
    named = parameters()
    initial = {n: p.detach().clone() for n, p in named.items()}
    make_optimizer("HEAD_ONLY", named)
    frozen = next(n for n, p in named.items() if not p.requires_grad)
    with torch.no_grad():
        named[frozen].add_(1)
    with pytest.raises(RuntimeError, match="frozen subset drift"):
        check_subset("HEAD_ONLY", named, initial)
    make_optimizer("CONTROL", named)
    assert all(p.requires_grad for p in named.values())
    with pytest.raises(ValueError):
        make_optimizer("unknown", named)
    with pytest.raises(ValueError):
        make_optimizer("CONTROL", dict(named, bad=torch.nn.Parameter(torch.ones(1))))
    assert adapter_drift({"x": torch.zeros(2)}, {"x": torch.tensor([3.0, 4.0])}) == 5


def rows():
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(205)]


def test_fresh_fold_exclusion_and_paired_batches():
    source = rows()
    assert stat.SEEDS == (239101, 239102, 239103)
    assert list(stat.FOLD_SEEDS.values()) == [239141, 239142, 239143]
    assert len({json.dumps(stat.folds(source, s)) for s in stat.SEEDS}) == 3
    for seed in stat.SEEDS:
        seen = []
        for outer in range(5):
            fit, select, assess = stat.partition(source, outer, seed)
            groups = [{r["family_id"] for r in rs} for rs in [fit, select, assess]]
            assert list(map(len, groups)) == [123, 41, 41]
            assert not (
                groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
            )
            seen += list(groups[2])
        assert len(seen) == len(set(seen)) == 205
    ids = [r["context_sha256"] for r in fit]
    orders = {c: np.arange(48) for c in ids}
    a, b = (
        stat.batch_plan(ids, orders, 239101),
        stat.batch_plan(ids[::-1], orders, 239101),
    )
    assert len(a) == 40
    for (s, c, ix), (t, d, jx) in zip(a, b, strict=True):
        assert s == t and c == d and len(c) == 8
        assert all(
            np.array_equal(i, j) and len(set(i)) == 24
            for i, j in zip(ix, jx, strict=True)
        )


def test_strict_selection_and_earliest_ties():
    source = rows()[:41]
    panel = np.zeros((41, 5))
    assert stat.choose(source, panel, 1)["selected_step"] == 0
    panel[:, 1] = 0.005
    panel[:, 2] = 0.005 + 5e-13
    assert stat.choose(source, panel, 1)["selected_step"] == 10
    panel[:, 1:] = 0.001
    assert stat.choose(source, panel, 1)["selected_step"] == 0
    panel[:] = 0
    panel[:, 1] = np.r_[-np.ones(20), np.ones(21)]
    panel[:, 2] = 0.005
    pick = stat.choose(source, panel, 1)
    assert (
        pick["best_step"] == 10 and pick["ci95"][0] < 0 and pick["selected_step"] == 0
    )


def test_family_weighting_three_way_holm_margin_seeds_and_elo():
    source = [dict(family_id="big")] * 9 + [dict(family_id="small")]
    panel = np.zeros((3, 10, 5))
    panel[:, :9, 2] = 1
    panel[:, 9, 2] = -1
    summary = stat.summarize(source, panel)
    assert summary["contrasts"]["BIAS_ONLY_minus_r882"]["median_gain"] == 0
    assert np.allclose(summary["median_elo"], 1500)
    panel[:] = 0
    panel[:, :, 2] = 0.002
    assert not stat.summarize(source, panel)["contrasts"]["BIAS_ONLY_minus_r882"][
        "advance"
    ]
    panel[:, :, 2] = 0.003
    s = stat.summarize(source, panel)
    c = s["contrasts"]["BIAS_ONLY_minus_r882"]
    assert c["advance"] and c["positive_seeds"] == 3 and c["holm_p"] < 0.05
    assert c["holm_p"] == min(1.0, 3 * c["p_one_sided"])
    assert s["median_elo"][2] > 1500 and s["median_elo_lift_vs_control"][2] > 0
    panel[1:, :, 2] = 0
    assert not stat.summarize(source, panel)["contrasts"]["BIAS_ONLY_minus_r882"][
        "advance"
    ]
    assert stat.holm({"a": 0.01, "b": 0.04, "c": 0.02}) == {
        "a": 0.03,
        "c": 0.04,
        "b": 0.04,
    }


def sample(t=0, ram=32, load=50):
    return dict(
        utc=t,
        available=ram * mon.GIB,
        load5=load,
        rss=1_000_000,
        free=400_000_000_000,
        threads=3,
    )


def test_a1_sustained_low_memory_and_immediate_emergency():
    g = mon.Guard()
    assert g.update(sample(0, 11)) == ([], True)
    assert g.update(sample(29.9, 11)) == ([], True)
    reasons, pause = g.update(sample(30, 11))
    assert reasons == ["MemAvailable<12GiB sustained30s"] and pause
    g = mon.Guard()
    g.update(sample(0, 11))
    g.update(sample(29, 12))
    assert g.update(sample(30, 11))[0] == []
    assert g.update(sample(59.9, 11))[0] == []
    assert g.update(sample(60, 11))[0]
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(1, 5.99))[0]
    assert not mon.Guard().update(sample(1, 6))[0]


def test_a1_hysteresis_requires_unbroken_30_seconds_and_load_recovery():
    g = mon.Guard()
    assert g.update(sample(0, 19))[1]
    assert g.update(sample(1, 25))[1]
    assert g.update(sample(30.9, 25))[1]
    assert not g.update(sample(31, 25))[1]
    g.update(sample(32, 19))
    g.update(sample(33, 26))
    g.update(sample(62, 24.99))
    assert g.update(sample(63, 25))[1] and g.update(sample(92.9, 25))[1]
    assert not g.update(sample(93, 25))[1]
    assert g.update(sample(94, 32, 108.1))[1]
    assert g.update(sample(95, 32, 101))[1]
    assert not g.update(sample(96, 32, 100))[1]
    assert mon.admission_refusals(sample(ram=30, load=100)) == []
    assert mon.admission_refusals(sample(ram=29.99))
    assert mon.admission_refusals(sample(load=100.01))
    assert mon.COMPUTE_STOP == 1793691000
    assert mon.violations(sample(mon.COMPUTE_STOP))
    for key, value in dict(rss=4 * mon.GIB, threads=5, free=299_999_999_999).items():
        assert mon.violations(dict(sample(), **{key: value}))


def test_cooperative_pause_reads_no_model_and_never_uses_sigstop(tmp_path, monkeypatch):
    pause = tmp_path / "pause.json"
    pause.write_text("{}")
    monkeypatch.setattr(runner, "PAUSE", pause)
    waited = []

    def sleep(seconds):
        waited.append(seconds)
        pause.unlink()

    monkeypatch.setattr(runner.time, "sleep", sleep)
    runner.cooperate()
    assert waited == [0.25]
    source = (HERE / "monitor.py").read_text()
    assert "signal.SIGSTOP" not in source and "signal.SIGCONT" not in source
    assert "start_new_session=True" in source or "start_new_session = True" in source


def test_source_launch_requires_merged_head_even_for_smoke():
    launch = dict(
        manifest_sha256="m",
        source_commit="h",
        prereg_sha256="p",
        fold_manifest_sha256="f",
        review_id=123,
        prereg_pr=456,
        merged_source_verified=True,
    )
    validate_launch(launch, "h", "m", "p", "f")
    for key, bad in dict(
        manifest_sha256="x",
        source_commit="x",
        prereg_sha256="x",
        fold_manifest_sha256="x",
        review_id=True,
        prereg_pr=0,
        merged_source_verified=False,
    ).items():
        with pytest.raises(RuntimeError):
            validate_launch(dict(launch, **{key: bad}), "h", "m", "p", "f", smoke=True)


def test_schema_and_input_boundary_without_loading_data():
    import jsonschema

    jsonschema.validate(
        json.loads((HERE / "metrics.json").read_text()),
        json.loads((HERE / "metrics.schema.json").read_text()),
    )
    source = (HERE / "runner.py").read_text()
    assert source.index(
        "baseline, costs = engine.evaluate(val, native=True"
    ) < source.index("for arm in stat.ARMS:")
    assert source.index('out / "smoke.json"') < source.index("engine.load(train)")
    assert 'r.get("has_meta_test") is True' in source
    assert (
        "cooperate()" in source
        and "resume selection hash" in source
        and "resume adapter hash" in source
    )
