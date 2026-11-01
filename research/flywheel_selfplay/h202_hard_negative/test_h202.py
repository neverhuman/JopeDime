"""Data-free H202 exclusion, gradient, top-k, pairing, selection and codec controls."""

import numpy as np
import pytest
import torch
from torch import nn

import analysis as stat
import artifacts
import design as d
import monitor
import objectives as o
import runner


@pytest.mark.parametrize("host", ["xbabe0", "xbabe3", "unknown"])
def test_forbidden_host_before_scientific_input_read(monkeypatch, host):
    monkeypatch.setattr(monitor.socket, "gethostname", lambda: host)
    monkeypatch.setattr(
        runner.Path, "read_text", lambda *a, **k: pytest.fail("input read")
    )
    monkeypatch.setattr(
        runner.Path, "read_bytes", lambda *a, **k: pytest.fail("input read")
    )
    with pytest.raises(RuntimeError, match="excludes this host"):
        runner.main()


def test_host_resource_boundaries():
    assert (
        monitor.admit_host("xbabe1") == "xbabe1"
        and monitor.admit_host("xbabe2") == "xbabe2"
    )
    x = dict(
        cpu=10,
        available=13 * 2**30,
        rss=4e9,
        output=200e6,
        elapsed=100,
        threads=3,
        load=20,
        free=50e9,
        used_fraction=0.95,
    )
    assert monitor.violations(x, "xbabe1") == []
    for key, value, label in [
        ("threads", 4, "threads3"),
        ("load", 108.81, "load108.8"),
        ("rss", 5e9, "RSS5GB"),
        ("output", 300e6, "output300MB"),
        ("available", 11 * 2**30, "Mem12GiB"),
        ("elapsed", 14401, "wall4h"),
    ]:
        y = dict(x)
        y[key] = value
        assert label in monitor.violations(y, "xbabe1")


def test_top5_inverted_pairs_have_correct_descent_sign_and_exclude_outside():
    cost = torch.arange(6.0, requires_grad=True)
    loss, info = o.hard_negative(cost, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 100.0]))
    assert info["top5"] == [0, 1, 2, 3, 4] and info["pair_count"] == 10
    loss.backward()
    assert cost.grad[0] < 0 and cost.grad[4] > 0 and cost.grad[5] == 0


def test_no_inversions_is_finite_differentiable_zero():
    cost = torch.arange(6.0, requires_grad=True)
    loss, info = o.hard_negative(cost, np.arange(6.0, 0.0, -1))
    assert loss == 0 and info["pair_count"] == 0
    loss.backward()
    assert torch.equal(cost.grad, torch.zeros(6))


def test_stable_cost_ties_and_strict_utility_order():
    cost = torch.zeros(6, requires_grad=True)
    u = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 99.0])
    _, info = o.hard_negative(cost, u)
    assert info["top5"] == [0, 1, 2, 3, 4]
    assert all(
        u[better] > u[worse] and better < 5 and worse < 5
        for better, worse in info["pairs"]
    )
    with pytest.raises(ValueError):
        o.hard_negative(torch.zeros(4), np.zeros(4))


def test_within_context_shuffle_preserves_marginal_and_is_seeded():
    values = np.arange(20, dtype=float)
    np.testing.assert_array_equal(np.sort(o.shuffled_utilities(values, 202101)), values)
    np.testing.assert_array_equal(
        o.shuffled_utilities(values, 202101), o.shuffled_utilities(values, 202101)
    )
    assert not np.array_equal(
        o.shuffled_utilities(values, 202101), o.shuffled_utilities(values, 202102)
    )


def test_dope_and_anchor_get_no_gradient_and_optimizer_excludes_them():
    net = nn.Module()
    net.dope_model = nn.Linear(2, 3)
    net.adapter_10tok = nn.Module()
    net.adapter_10tok.anchor = nn.Linear(3, 1)
    net.adapter_10tok.head = nn.Linear(3, 1)
    names = o.adapter_mask(net)
    features = net.dope_model(torch.ones(6, 2))
    costs = (
        net.adapter_10tok.anchor(features) + net.adapter_10tok.head(features)
    ).flatten()
    loss, _ = o.hard_negative(costs, np.arange(6.0))
    loss.backward()
    o.assert_grad_mask(net, names)
    assert all(p.grad is None for p in net.dope_model.parameters())
    assert all(p.grad is None for p in net.adapter_10tok.anchor.parameters())
    assert all(p.grad is not None for p in net.adapter_10tok.head.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=1e-4
    )
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in net.adapter_10tok.head.parameters()
    }


def test_three_fresh_family_partitions_and_cyclic_role_exclusion():
    rows = [dict(context_sha256=str(j), family_id="F" + str(j)) for j in range(205)]
    signatures = []
    for seed in d.FOLD_SEEDS:
        groups = d.folds([r["family_id"] for r in rows], seed)
        signatures.append(str(groups))
        aa = []
        bb = []
        for k in range(5):
            fit, b, a = d.partition(rows, groups, k)
            assert [len(x) for x in [fit, b, a]] == [123, 41, 41]
            assert not {r["family_id"] for r in fit} & {r["family_id"] for r in a + b}
            aa.extend(r["family_id"] for r in a)
            bb.extend(r["family_id"] for r in b)
        assert sorted(aa) == sorted(bb) == sorted(r["family_id"] for r in rows)
    assert len(set(signatures)) == 3


def test_family_sampling_probabilities_and_paired_plan():
    rows = [
        dict(context_sha256=str(j), family_id="A" if j == 0 else "B") for j in range(3)
    ]
    p = d.family_probabilities(rows)
    assert p == {"0": 0.5, "1": 0.25, "2": 0.25} and sum(p.values()) == 1
    plan = d.plan(rows, 202101, 0)
    assert len(plan) == 40 and all(len(x["context_ids"]) == 4 for x in plan)
    assert plan == d.plan(rows, 202101, 0) and plan != d.plan(rows, 202102, 0)
    assert all(set(x["context_ids"]) <= {"0", "1", "2"} for x in plan)


def test_common_B_step_early_ties_and_complete_panel_required():
    rows = [dict(family_id=str(j)) for j in range(205)]
    panel = np.zeros((3, 205, 4, 5))
    panel[:, :, 1, 1:3] = 0.02
    result = d.choose_steps(rows, panel)
    assert result["selected_steps"] == {"HNEG40": 10, "CTRL40": 0, "HNEG_SHUF": 0}
    panel[0, 0, 1, 1] = np.nan
    with pytest.raises(ValueError):
        d.choose_steps(rows, panel)


def test_A_requires_durable_unchanged_B_lock(tmp_path):
    path = tmp_path / "B-lock.json"
    with pytest.raises(RuntimeError):
        d.authorize_A(path, "0" * 64)
    payload = dict(locked_before_A=True, selected_steps={a: 0 for a in d.ARMS})
    digest = d.write_lock(path, payload)
    assert d.authorize_A(path, digest) == payload["selected_steps"]
    path.write_text("{}")
    with pytest.raises(RuntimeError):
        d.authorize_A(path, digest)


def test_xor_cost_roundtrip_is_bit_exact_including_signed_zero():
    reference = np.array([-0.0, 0.0, 1.0, -3.0, 1e-30, 1e10], dtype=np.float32)
    values = np.nextafter(reference, np.float32(np.inf))
    encoded = artifacts.encode(values, reference)
    assert (
        encoded.dtype == np.uint32
        and artifacts.decode(encoded, reference).tobytes() == values.tobytes()
    )
    with pytest.raises(ValueError):
        artifacts.encode(
            np.array([np.nan], dtype=np.float32), np.zeros(1, dtype=np.float32)
        )


def test_median_axis_and_three_positive_seeds_gate(monkeypatch):
    monkeypatch.setattr(stat, "DRAWS", 100)
    rows = [dict(family_id=str(j)) for j in range(205)]
    panel = np.zeros((3, 205, 4))
    panel[:, :, 1] = np.array([0.03, 0.02, -0.04])[:, None]
    result = stat.summarize(rows, panel)
    c = result["contrasts"]["HNEG40_minus_R882"]
    assert c["positive_seeds"] == 2 and abs(c["median"] - 0.02) < 1e-12
    assert result["verdict"] == "NO_ADVANCE"
    panel[:, :, 1] = 0.001
    assert stat.summarize(rows, panel)["verdict"] == "NO_ADVANCE"
    panel[:, :, 1] = 0.02
    assert stat.summarize(rows, panel)["verdict"] == "CANDIDATE"
    panel[:, :, 2] = 0.02
    assert stat.summarize(rows, panel)["verdict"] == "NO_ADVANCE"


def test_native_module_namespace_isolated():
    base = runner.native_modules()
    assert base.stat.__file__.endswith("h189_swa_ema/analysis.py")


def test_resource_scan_tolerates_counter_atomic_rename():
    class Gone:
        def is_file(self):
            return True

        def stat(self):
            raise FileNotFoundError("atomic rename")

    class Root:
        def rglob(self, pattern):
            return [Gone()]

    assert monitor.tree_bytes(Root()) == 0


def test_guard_reaps_only_its_owned_inert_process():
    import subprocess
    import sys

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        monitor.stop_owned(child)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
