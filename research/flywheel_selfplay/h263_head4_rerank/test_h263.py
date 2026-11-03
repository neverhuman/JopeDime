"""Data-free protocol and mechanism tests; no study observations."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import analysis as stat
import monitor as mon
import treatment as tr

GIB = 2**30


def rows(n=205):
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(n)]


def test_partition_and_training_batch_identity():
    r = rows()
    groups = [stat.folds(r, s) for s in stat.SEEDS]
    assert groups[0] == groups[1] == groups[2]
    for outer in range(5):
        fit, select, assess = stat.partition(r, outer, stat.SEEDS[0])
        sets = [{x["family_id"] for x in part} for part in (fit, select, assess)]
        assert list(map(len, sets)) == [123, 41, 41]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
        ids = [x["context_sha256"] for x in fit]
        orders = {cid: np.arange(30) for cid in ids}
        a = stat.batch_plan(ids, orders, 263101)
        b = stat.batch_plan(ids, orders, 263101)
        assert len(a) == 40
        for (stream, cids, ix), (stream2, cids2, ix2) in zip(a, b, strict=True):
            assert stream == stream2 and cids == cids2 and set(cids) <= set(ids)
            for q, z in zip(ix, ix2, strict=True):
                assert (
                    np.array_equal(q, z)
                    and np.array_equal(q[:4], np.arange(4))
                    and len(set(q)) == 24
                )


def test_selector_ties_noise_and_strict_margin():
    r = rows(40)
    flat = np.ones((40, 5))
    d = stat.choose(r, flat, 1)
    assert d["selected_step"] == d["best_step"] == 0
    noisy = flat.copy()
    noisy[:, 4] += np.where(np.arange(40) % 2 == 0, 0.5, -0.49)
    d = stat.choose(r, noisy, 1)
    assert d["best_step"] == 40 and d["selected_step"] == 0
    flat[:, 1:] += 0.001
    assert stat.choose(r, flat, 1)["selected_step"] == 0
    flat[:, 1:] += 0.01
    assert stat.choose(r, flat, 1)["selected_step"] == 10


def sample(t, ram, load=50):
    return dict(
        utc=t,
        load5=load,
        available=ram * GIB,
        free=10**15,
        rss=GIB,
        threads=2,
        affinity=[48, 50, 51, 52],
    )


def test_temporal_a1_memory_and_load_guards():
    t = 1793700000.0
    g = mon.Guard()
    assert g.update(sample(t, 19)) == ([], True)
    assert g.update(sample(t + 1, 26))[1]
    assert g.update(sample(t + 30, 26))[1]
    assert g.update(sample(t + 31, 26)) == ([], False)
    g = mon.Guard()
    assert not g.update(sample(t, 11))[0]
    assert not g.update(sample(t + 29, 11))[0]
    assert "MemAvailable<12GiB sustained30s" in g.update(sample(t + 30, 11))[0]
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(t, 5))[0]
    g = mon.Guard()
    assert g.update(sample(t, 40, 109))[1]
    assert g.update(sample(t + 1, 40, 101))[1]
    assert not g.update(sample(t + 2, 40, 100))[1]
    assert (
        mon.COMPUTE_STOP == 1793731500.0
        and "absolute compute stop 18:45Z"
        in mon.violations(sample(mon.COMPUTE_STOP, 40))
    )


def test_pair_orientation_six_pairs_and_gradient():
    fitness = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    pred = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    gate = torch.ones(4, dtype=torch.bool)
    observed = tr.head4_loss(pred, fitness, gate)
    expected = torch.stack(
        [
            torch.nn.functional.softplus(pred[i] - pred[j])
            for i in range(4)
            for j in range(i + 1, 4)
        ]
    ).mean()
    torch.testing.assert_close(observed, expected.double())
    assert observed < tr.head4_loss(-pred, fitness, gate)
    observed.backward()
    assert pred.grad[0] > 0 and pred.grad[-1] < 0


def test_pair_ties_invalid_worst_and_sentinel_gap():
    p = torch.zeros(4, requires_grad=True)
    f = torch.tensor([5.0, 99.0, 99.0, 99.0], dtype=torch.float64)
    g = torch.tensor([True, False, False, False])
    loss = tr.head4_loss(p, f, g)
    loss.backward()
    assert p.grad[0] > 0 and all(p.grad[1:] < 0)  # valid raw5 beats invalid sentinel2
    p.grad = None
    zero = tr.head4_loss(p, torch.ones(4), torch.ones(4, dtype=torch.bool))
    zero.backward()
    assert zero == 0 and torch.equal(p.grad, torch.zeros(4))
    assert tr.head4_loss(p, torch.ones(4), torch.zeros(4, dtype=torch.bool)) == 0
    assert tr.head4_loss(p, torch.tensor([2.0, 99.0, 99.0, 99.0]), g) == 0
    with pytest.raises(ValueError):
        tr.head4_loss(p[:3], f, g)


def test_l2sp_exact_anchor_and_retraction():
    p = torch.tensor([1.0, -2.0], requires_grad=True)
    q = torch.tensor([3.0], requires_grad=True)
    initial = [torch.tensor([0.5, -1.0]), torch.tensor([2.0])]
    loss = tr.l2sp_penalty([p, q], initial)
    assert loss.item() == 25 * (0.25 + 1 + 1)
    loss.backward()
    torch.testing.assert_close(p.grad, 50 * (p.detach() - initial[0]))
    before = [p.detach().clone(), q.detach().clone()]
    tr.retract([p, q], initial)
    for value, prev, anchor in zip([p, q], before, initial, strict=True):
        torch.testing.assert_close(value, anchor + 0.75 * (prev - anchor))


def test_exact_control_adamw_update():
    p = torch.tensor([0.4, -0.2], requires_grad=True)
    q = p.detach().clone().requires_grad_()
    a = tr.ControlAdamW([p])
    b = torch.optim.AdamW([q], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    for _ in range(4):
        a.zero_grad()
        b.zero_grad()
        p.square().sum().backward()
        q.square().sum().backward()
        a.step()
        b.step()
    assert torch.equal(p, q)


def test_holm_primary_secondary_only_and_fallback():
    r = rows(11)
    panel = np.ones((3, 11, 5))
    result = stat.summarize(r, panel)
    assert result["contrasts"]["HEAD4_minus_r882"]["holm_lcb95"] == 0
    assert not result["contrasts"]["HEAD4_minus_r882"]["advance"]
    assert "holm_p" not in result["contrasts"]["L2SP_minus_r882"]
    panel[:, :, 3] += 0.01
    result = stat.summarize(r, panel)
    assert result["contrasts"]["HEAD4_minus_r882"]["advance"]
    assert not result["contrasts"]["HEAD4_NOTR_minus_r882"]["advance"]
    assert result["contrasts"]["HEAD4_minus_L2SP"]["median_gain"] > 0.009
    assert np.array(result["elo_by_seed"])[:, 0].tolist() == [1500.0] * 3


def test_top4_h258_gate_and_stable_order():
    cost = np.array([0.0, 0.001, 0.002, 1.0], dtype=np.float32)
    assert (
        stat.top4_pick(
            cost, np.array([0.0, -4.0, -3.0, -9.0]), np.array([1, 0, 1, 0], bool)
        )["choice"]
        == 2
    )
    assert (
        stat.top4_pick(
            np.array([0.0, 2.0, 2.0, 2.0], np.float32),
            np.array([0.0, -4.0, -3.0, -9.0]),
            np.ones(4, bool),
        )["reveals"]
        == 0
    )


def test_global_lock_rejects_incomplete_or_mutated(tmp_path):
    import runner

    payload = tmp_path / "weights"
    payload.write_bytes(b"a")
    lock = tmp_path / "final-lock.json"
    lock.write_text(
        json.dumps(dict(completed_cells=14, optimizer_updates=2400, files={}))
    )
    with pytest.raises(RuntimeError):
        runner.assert_final_lock(tmp_path)
    lock.write_text(
        json.dumps(
            dict(
                completed_cells=15,
                optimizer_updates=2400,
                files={"weights": runner.eng.sha(payload)},
            )
        )
    )
    runner.assert_final_lock(tmp_path)
    payload.write_bytes(b"b")
    with pytest.raises(RuntimeError):
        runner.assert_final_lock(tmp_path)


def test_source_protocol_global_barrier_and_deadline():
    source = (Path(__file__).parent / "runner.py").read_text()
    assert (
        source.index("GLOBAL SELECTION LOCK PASS")
        < source.index("v,vc=engine.evaluate(val)")
        if "v,vc=engine.evaluate(val)" in source
        else source.index("GLOBAL SELECTION LOCK PASS")
        < source.index("v, vc = engine.evaluate(val)")
    )
    assert mon.COMPUTE_STOP == 1793731500.0
    assert stat.SEEDS == (263101, 263102, 263103)
