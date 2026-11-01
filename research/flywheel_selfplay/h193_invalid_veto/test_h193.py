"""Data-free controls for the new veto, data-role closure and imported decision rule."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import analysis as stat
from parents import h177, h188, h191, h192
from runner import Engine, forbidden_training
from treatment import (
    invalid_share,
    revealed_invalid,
    theta_grid,
    veto_pick,
    median_theta,
)
from monitor import SustainedLoad, LOAD_LIMIT


def rows(n=205):
    return [dict(context_sha256=f"c{i}", family_id=f"f{i}") for i in range(n)]


def test_original_exact_minus2_not_float32_rounding():
    assert revealed_invalid(
        [1, 2, 2 + 1e-10, 100], [False, True, True, False]
    ).tolist() == [True, True, False, True]


@pytest.mark.parametrize("invalid,expected", [(False, 0.0), (True, 1.0)])
def test_uniform_revealed_outcomes_probability_limits(invalid, expected):
    ad = SimpleNamespace(
        sem_kernel_scale_raw=torch.tensor(0.0),
        emb_kernel_scale_raw=torch.tensor(0.0),
        null_kernel_w0_raw=torch.tensor(0.0),
        traj_kernel_blend=torch.tensor(0.0),
    )
    static = dict(
        sem24=torch.zeros(1, 3, 24),
        cand_q_norm=torch.zeros(1, 3, 2),
        eff_prior=torch.zeros(1, 3),
    )
    p = invalid_share(
        ad,
        static,
        torch.ones(1),
        torch.tensor([[0, 1]]),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.full((1, 2), invalid),
    )
    assert torch.equal(p, torch.full((1, 3), expected))


def test_kernel_weighted_share_uses_imported_kernel_and_only_indicator(monkeypatch):
    ad = SimpleNamespace(
        sem_kernel_scale_raw=torch.tensor(0.0),
        emb_kernel_scale_raw=torch.tensor(0.0),
        null_kernel_w0_raw=torch.tensor(0.0),
        traj_kernel_blend=torch.tensor(0.0),
    )
    sem = torch.zeros(1, 3, 24)
    sem[0, 0, 0] = 1
    sem[0, 1, 1] = 1
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
    static = dict(sem24=sem, cand_q_norm=q, eff_prior=torch.zeros(1, 3))
    args = (
        ad,
        static,
        torch.ones(1),
        torch.tensor([[0, 1]]),
        torch.tensor([[100.0, -4.0]]),
        torch.ones(1, 2),
        torch.tensor([[False, True]]),
    )
    original = h188.kernel_term
    seen = []

    def traced(*args, **kwargs):
        seen.append(kwargs["residual_transform"](torch.tensor([[99.0, -99.0]])))
        return original(*args, **kwargs)

    monkeypatch.setattr(h188, "kernel_term", traced)
    p = invalid_share(*args)
    assert (
        len(seen) == 2
        and seen[0].tolist() == [[0.0, 1.0]]
        and seen[1].tolist() == [[1.0, 1.0]]
    )
    assert p[0, 0] < 0.5 and p[0, 1] > 0.5 and p[0, 2] == 0.5
    altered = list(args)
    altered[4] = torch.tensor([[-123.0, 321.0]])
    assert torch.equal(invalid_share(*altered), p)


@pytest.mark.parametrize(
    "cost",
    [np.array([1.0, 1.0, 2.0], np.float32), np.array([3.0, -1.0, -1.0], np.float32)],
)
def test_inf_exact_identity_including_first_argmin_ties(cost):
    pick, count = veto_pick(cost, np.array([0.2, 0.9, 1.0]), None)
    assert pick == int(np.argmin(cost)) and count["vetoed"] == count["fallback"] == 0


def test_strict_threshold_and_all_veto_fallback():
    assert veto_pick([0.0, 1.0], [0.5, 0.6], 0.5)[0] == 0
    pick, count = veto_pick([0.0, 1.0], [0.9, 0.6], 0.5)
    assert pick == 0 and count["fallback"] == 1 and count["vetoed"] == 2
    pick, count = veto_pick([0.0, 1.0], [0.9, 0.1], 0.5)
    assert pick == 1 and count["fallback"] == 0 and count["vetoed"] == 1


def test_positive_fit_candidate_quantiles_empty_and_median_inf():
    assert theta_grid([np.array([0, 0.1, 0.2]), np.array([0.3])]) == [
        None,
        0.28,
        0.25,
        0.2,
    ]
    assert theta_grid([np.zeros(3)]) == [None] * 4
    assert median_theta([None, 0.2, 0.1, None, 0.3]) == 0.3
    json.dumps(theta_grid([np.zeros(3)]), allow_nan=False)


def test_cell_choice_ignores_own_a_and_quantiles_ignore_b():
    fit = [dict(context_sha256="fit", family_id="FIT")]
    b = [dict(context_sha256="b", family_id="B")]
    p = {
        "fit": np.array([0.1, 0.2, 0.3, 0.4]),
        "b": np.array([0.9, 0.1]),
        "a": np.array([0.99, 0.99]),
    }
    costs = {"b": np.array([0.0, 1.0]), "a": np.array([-1.0, 3.0])}
    u = {"b": np.array([-2.0, 1.0]), "a": np.array([9.0, 9.0])}
    before = stat.select_cell(fit, b, p, costs, u)[0]
    assert before["selected_index"] == 1
    p["a"][:] = 0
    costs["a"][:] = 100
    u["a"][:] = -100
    assert stat.select_cell(fit, b, p, costs, u)[0] == before
    p["b"][:] = 0
    after = stat.select_cell(fit, b, p, costs, u)[0]
    assert after["grid"] == before["grid"] and after["selected_index"] == 0


def test_equal_family_selection_and_baseline_ties():
    fit = [dict(context_sha256="fit", family_id="FIT")]
    b = [dict(context_sha256=f"large{i}", family_id="large") for i in range(100)] + [
        dict(context_sha256="small", family_id="small")
    ]
    p = {"fit": np.array([0.1, 0.2, 0.3, 0.4])}
    costs = {}
    u = {}
    for r in b:
        cid = r["context_sha256"]
        p[cid] = np.array([0.9, 0.01])
        costs[cid] = np.array([0.0, 1.0])
        u[cid] = np.array([0.0, 1.0 if cid != "small" else -4.0])
    cell = stat.select_cell(fit, b, p, costs, u)[0]
    assert cell["selected_index"] == 0 and cell["b_family_means"][1] == -1.5
    for cid in u:
        u[cid][:] = 0
    assert stat.select_cell(fit, b, p, costs, u)[0]["selected_index"] == 0


def test_literal_fold_seeds_distinct_disjoint_and_exact_once():
    rs = rows()
    fm = stat.fold_manifest(rs)
    assert [fm[str(s)]["fold_seed"] for s in stat.SEEDS] == [193141, 193142, 193143]
    assert len({json.dumps(x["families"]) for x in fm.values()}) == 3
    for s in stat.SEEDS:
        seen = []
        for k in range(5):
            fit, b, a = stat.partition(rs, s, k)
            assert tuple(map(len, (fit, b, a))) == (123, 41, 41)
            assert not {r["family_id"] for r in fit} & {r["family_id"] for r in b + a}
            assert not {r["family_id"] for r in b} & {r["family_id"] for r in a}
            seen.extend(r["context_sha256"] for r in a)
        assert len(set(seen)) == len(seen) == 205


def lock_fixture(tmp_path):
    rs = rows()
    p = {r["context_sha256"]: np.array([0.1, 0.5]) for r in rs}
    c = {k: np.array([0.0, 1.0]) for k in p}
    u = {k: np.array([0.0, 1.0]) for k in p}
    lock, panel = stat.stage1(rs, p, c, u)
    np.savez_compressed(tmp_path / "B-panel.npz", utility=panel)
    lock["b_panel_sha256"] = stat.digest_file(tmp_path / "B-panel.npz")
    path = tmp_path / "veto_selection_lock.json"
    path.write_text(json.dumps(lock))
    return rs, p, c, u, path


def test_actual_a_and_val_entries_require_complete_immutable_lock(tmp_path):
    guard = stat.SelectionGuard()
    for stage in ("A", "VAL"):
        with pytest.raises(RuntimeError, match="immutable"):
            stat.stage2([], {}, {}, {}, guard, stage)
    rs, p, c, u, path = lock_fixture(tmp_path)
    guard.lock(path)
    a, _ = stat.stage2(rs, c, p, u, guard, "A")
    assert a.shape == (3, 205, 5)
    path.write_text(path.read_text() + " ")
    with pytest.raises(RuntimeError, match="immutable"):
        stat.stage2(rs, c, p, u, guard, "A")


@pytest.mark.parametrize("mutation", ["cell", "argmax", "panel", "val"])
def test_lock_refuses_incomplete_or_mutated_decisions(tmp_path, mutation):
    *_, path = lock_fixture(tmp_path)
    value = json.loads(path.read_text())
    if mutation == "cell":
        value["cells"].pop("s193101-f0")
    elif mutation == "argmax":
        value["cells"]["s193101-f0"]["selected_index"] = 3
    elif mutation == "panel":
        (tmp_path / "B-panel.npz").write_bytes(b"changed")
    else:
        value["val_theta"]["193101"] = 0.99
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError):
        stat.SelectionGuard().lock(path)


def test_imported_engine_decision_and_bootstrap():
    assert Engine.__bases__ == (h192.Engine,)
    assert (
        Path(inspect.getsourcefile(h188.kernel_term)).parent.name
        == "h188_kernel_inputs"
    )
    assert (
        stat.partition is h191.partition
        and stat.H177_DECISION_RULE is h177.H177_DECISION_RULE
    )
    rs = rows()
    panel = np.zeros((3, 205, 5))
    panel[:, :, 1] = 0.1
    admission = stat.select_a(rs, panel)
    assert (
        admission["admitted"]
        and admission["holm_multiplicity"] == 1
        and admission["holm_p"] == pytest.approx(1 / 10001)
    )
    panel[:, :, 1] = 0.001
    assert not stat.select_a(rs, panel)["admitted"]
    small = rows(9)
    panel = np.zeros((3, 9, 5))
    panel[:, :, 1] = (
        np.array([-0.01, 0.03, 0.07])[:, None] + np.arange(9)[None, :] * 0.001
    )
    summary = stat.summarize(small, panel)
    d = panel[:, :, 1]
    ix = np.random.default_rng(193150).integers(0, 9, (10000, 9))
    expected = np.median(d[:, ix].mean(2), axis=0)
    assert (
        summary["contrasts"]["VETO"]["ci95"]
        == np.quantile(expected, [0.025, 0.975]).tolist()
    )
    with pytest.raises(RuntimeError):
        forbidden_training()


def test_load_requires_three_high_readings_five_seconds_apart_and_resets():
    guard = SustainedLoad()
    assert not guard.check(LOAD_LIMIT + 1, 0)
    assert not guard.check(LOAD_LIMIT + 1, 4.9)
    assert not guard.check(LOAD_LIMIT + 1, 5)
    assert guard.check(LOAD_LIMIT + 1, 10)
    assert not guard.check(LOAD_LIMIT, 10.5)
    assert not guard.check(LOAD_LIMIT + 1, 11)
