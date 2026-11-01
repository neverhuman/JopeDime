"""Data-free nested selection, source reuse, intervention isolation and guard controls."""

import copy
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
import analysis as stat
import treatment
from parents import ROOT, h177, h186_treatment
from runner import Engine, forbidden_training


def rows(n=205):
    return [
        dict(family_id=f"family-{i:03}", context_sha256=f"context-{i:03}")
        for i in range(n)
    ]


@pytest.mark.parametrize("arm", treatment.GRID_ARMS[1:])
def test_only_kernel_coefficient_changes(arm):
    base, changed = treatment.constants("R882"), treatment.constants(arm)
    assert [key for key in base if base[key] != changed[key]] == ["kernel_multiplier"]
    assert changed["topk"] == 8 and changed["null_multiplier"] == 1
    assert treatment.adapter_components is h186_treatment.adapter_components
    assert treatment.scores is h186_treatment.scores


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_token = 32
        self.null_kernel_w0_raw = nn.Parameter(torch.tensor(-0.5))
        self.sem_kernel_scale_raw = nn.Parameter(torch.tensor(1.2))
        self.emb_kernel_scale_raw = nn.Parameter(torch.tensor(0.8))
        self.traj_kernel_blend = nn.Parameter(torch.tensor(0.25))
        self.hist_token_proj = nn.Sequential(
            nn.Linear(60, 32), nn.LayerNorm(32), nn.SiLU()
        )
        self.null_hist_token = nn.Parameter(torch.randn(1, 1, 32) * 0.05)
        self.traj_cross_attn = nn.MultiheadAttention(32, 4, dropout=0, batch_first=True)
        self.traj_delta_head = nn.Sequential(
            nn.Linear(67, 24), nn.SiLU(), nn.Linear(24, 1)
        )
        self.traj_attn_scale = nn.Parameter(torch.tensor(0.1))


def test_component_intervention_preserves_prior_attention_and_weights():
    torch.manual_seed(191101)
    ad = Adapter().eval()
    q = torch.randn(2, 20, 32)
    static = dict(
        eff_prior=torch.randn(2, 20),
        mm_prior_adj=torch.randn(2, 20),
        sem24=torch.randn(2, 20, 24),
        cand_q=q,
        cand_q_norm=torch.nn.functional.normalize(q, dim=-1),
    )
    args = (
        ad,
        static,
        torch.ones(2),
        torch.arange(4).repeat(2, 1),
        torch.randn(2, 4),
        torch.ones(2, 4),
    )
    before = {k: v.clone() for k, v in ad.state_dict().items()}
    with torch.inference_mode():
        base = treatment.adapter_components(*args)
        assert all(
            torch.equal(base[k], v)
            for k, v in treatment.adapter_components(
                *args, **treatment.constants("R882")
            ).items()
        )
        for arm in treatment.GRID_ARMS[1:]:
            changed = treatment.adapter_components(*args, **treatment.constants(arm))
            assert torch.equal(changed["PRIOR"], base["PRIOR"])
            assert torch.equal(changed["ATTN"], base["ATTN"])
            assert torch.allclose(
                changed["KERNEL"],
                base["KERNEL"] * treatment.constants(arm)["kernel_multiplier"],
                atol=1e-6,
            )
            assert not torch.equal(changed["KERNEL"], base["KERNEL"])
    assert all(torch.equal(v, before[k]) for k, v in ad.state_dict().items())


def test_b_choice_is_equal_family_and_smaller_coefficient_breaks_ties():
    rs = [dict(family_id="large") for _ in range(100)] + [dict(family_id="small")]
    panel = np.zeros((101, 6))
    panel[:100, 1] = 1
    panel[-1, 1] = -4
    panel[-1, 2] = 0.1
    assert np.argmax(panel.mean(axis=0)) == 1
    assert stat.choose(rs, panel)["grid_index"] == 2
    panel[:, 3] = panel[:, 2]
    assert stat.choose(rs, panel)["grid_index"] == 2
    assert stat.choose(rs, np.zeros_like(panel))["grid_index"] == 0
    with pytest.raises(ValueError):
        stat.choose(rs, panel * np.nan)
    assert stat.val_choice([2, 1, 2, 1, 4]) == 1


def test_distinct_literal_seed_partitions_have_disjoint_exact_coverage():
    rs = rows()
    manifests = stat.fold_manifest(rs)
    assert [manifests[str(s)]["fold_seed"] for s in stat.SEEDS] == [
        191141,
        191142,
        191143,
    ]
    assert len({json.dumps(v["families"]) for v in manifests.values()}) == 3
    for seed in stat.SEEDS:
        seen_b, seen_a = [], []
        for k in range(5):
            fit, b, a = stat.partition(rs, seed, k)

            def ids(x):
                return {r["family_id"] for r in x}

            assert tuple(map(len, (fit, b, a))) == (123, 41, 41)
            assert not ids(fit) & (ids(b) | ids(a)) and not ids(b) & ids(a)
            seen_b += [r["context_sha256"] for r in b]
            seen_a += [r["context_sha256"] for r in a]
        assert len(set(seen_b)) == len(set(seen_a)) == 205


def lock_fixture(tmp_path):
    (tmp_path / "B-panel.npz").write_bytes(b"synthetic B panel")
    cell = dict(family_means=[0, 0, 0.2, 0.1, 0, 0], grid_index=2, multiplier=2.0)
    value = dict(
        wave="H191",
        fold_manifest_sha256="folds",
        adapter_sha256="weights",
        b_panel_sha256=stat.lock_hash(tmp_path / "B-panel.npz"),
        locked_before_a_and_treatment_val=True,
        cells={f"s{s}-f{k}": copy.deepcopy(cell) for s in stat.SEEDS for k in range(5)},
        val_grid_index={str(s): 2 for s in stat.SEEDS},
    )
    p = tmp_path / "kx_selection_lock.json"
    p.write_text(json.dumps(value))
    return p, value


def test_actual_scoring_entry_refuses_a_and_val_before_complete_lock(tmp_path):
    guard = stat.SelectionGuard("folds", "weights")
    engine = object.__new__(Engine)
    engine.set_policy_guard(guard, "weights")
    for stage in ("A", "DL-Val"):
        with pytest.raises(RuntimeError, match="immutable B choice lock"):
            engine.evaluate_many([], stage, tmp_path, "refusal")
    path, value = lock_fixture(tmp_path)
    for _ in range(14):
        guard.mark_b_read()
    with pytest.raises(RuntimeError, match="fifteen"):
        guard.mark_b_locked(path)
    guard.mark_b_read()
    guard.mark_b_locked(path)
    guard.require_b_locked()
    with pytest.raises(RuntimeError):
        guard.mark_b_read()
    path.write_text(path.read_text() + " ")
    with pytest.raises(RuntimeError):
        guard.require_b_locked()


@pytest.mark.parametrize(
    "mutation", ["missing_cell", "argmax", "val_policy", "b_payload"]
)
def test_lock_rejects_incomplete_or_inconsistent_b_decisions(tmp_path, mutation):
    path, value = lock_fixture(tmp_path)
    if mutation == "missing_cell":
        value["cells"].pop("s191101-f0")
    elif mutation == "argmax":
        value["cells"]["s191101-f0"]["grid_index"] = 1
    elif mutation == "val_policy":
        value["val_grid_index"]["191101"] = 4
    else:
        (tmp_path / "B-panel.npz").write_bytes(b"changed")
    path.write_text(json.dumps(value))
    guard = stat.SelectionGuard("folds", "weights")
    for _ in range(15):
        guard.mark_b_read()
    with pytest.raises(RuntimeError):
        guard.mark_b_locked(path)


def test_imported_selector_single_primary_holm_and_strict_margin():
    assert stat.pooled_b_select is h177.pooled_b_select
    assert stat.H177_DECISION_RULE is h177.H177_DECISION_RULE
    assert stat.resolve_h177_verdict is h177.resolve_h177_verdict
    assert Path(inspect.getsourcefile(Engine)).parent.name == "h186_kernel_sweep"
    panel = np.zeros((3, 205, 5))
    panel[:, :, 1] = 0.1
    selected = stat.select_a(rows(), panel)
    assert selected["admitted"] and selected["holm_multiplicity"] == 1
    assert selected["holm_p"] == selected["p_one_sided"] == 1 / 10001
    panel[:, :, 1] = 0.001
    assert not stat.select_a(rows(), panel)["admitted"]


def test_median_bootstrap_axis_and_verdict_do_not_require_three_positive_seeds():
    rs = rows(9)
    panel = np.zeros((3, 9, 5))
    panel[:, :, 1] = (
        np.array([-0.01, 0.03, 0.07])[:, None] + np.arange(9)[None, :] * 0.001
    )
    summary = stat.summarize(rs, panel)
    indices = np.random.default_rng(stat.BOOT_SEED).integers(0, 9, (10000, 9))
    expected = np.median(panel[:, indices, 1].mean(axis=2), axis=0)
    assert np.allclose(
        summary["contrasts"]["KXSEL"]["ci95"], np.quantile(expected, [0.025, 0.975])
    )
    assert summary["contrasts"]["KXSEL"]["positive_seeds"] == 2
    receipt = json.loads(
        (
            ROOT
            / "research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json"
        ).read_text()
    )
    digest = hashlib.sha256(Path(h177.__file__).read_bytes()).hexdigest()
    assert (
        stat.verdict({"admitted": True}, summary, receipt, digest)["verdict"]
        == "ADVANCE"
    )
    summary["contrasts"]["KXSEL"]["median_gain"] = 0.001
    assert (
        stat.verdict({"admitted": True}, summary, receipt, digest)["verdict"]
        == "NO_ADVANCE"
    )
    with pytest.raises(RuntimeError):
        forbidden_training()


def test_projection_keeps_selected_primary_and_prespecified_dose_columns():
    values = np.arange(24).reshape(4, 6)
    assert np.array_equal(stat.project(values, 2), values[:, [0, 2, 3, 4, 5]])
    with pytest.raises(ValueError):
        stat.project(values, 6)
