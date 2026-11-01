"""Data-free H186 intervention isolation, imported selector, and stage-order controls."""

import ast
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
import analysis as stat
import treatment
from parents import h177, h184_treatment, load
from runner import Engine, forbidden_training


class FakeAdapter(nn.Module):
    """The attributes compute_adapter_delta_from_static_terms reads, with the live shapes and init rules."""

    def __init__(self, d_token=32):
        super().__init__()
        self.d_token = d_token
        self.null_kernel_w0_raw = nn.Parameter(torch.tensor(-0.5))
        self.sem_kernel_scale_raw = nn.Parameter(torch.tensor(1.2))
        self.emb_kernel_scale_raw = nn.Parameter(torch.tensor(0.8))
        self.traj_kernel_blend = nn.Parameter(torch.tensor(0.25))
        self.hist_token_proj = nn.Sequential(
            nn.Linear(24 + d_token + 4, d_token), nn.LayerNorm(d_token), nn.SiLU()
        )
        self.null_hist_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.05)
        self.traj_cross_attn = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=4, dropout=0.0, batch_first=True
        )
        self.traj_delta_head = nn.Sequential(
            nn.Linear(d_token * 2 + 3, 24), nn.SiLU(), nn.Linear(24, 1)
        )
        self.traj_attn_scale = nn.Parameter(torch.tensor(0.10))


def synthetic(b=3, m=24, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, m, 32, generator=g)
    sem = torch.nn.functional.normalize(torch.randn(b, m, 24, generator=g), dim=-1)
    adj = 0.1 * torch.randn(b, m, generator=g)
    static = dict(
        eff_prior=torch.randn(b, m, generator=g) + adj,
        mm_prior_adj=adj,
        sem24=sem,
        cand_q=q,
        cand_q_norm=torch.nn.functional.normalize(q, p=2, dim=-1, eps=1e-8),
    )
    rev = torch.arange(k).reshape(1, -1).expand(b, -1)
    raw = torch.randn(b, k, generator=g)
    gate = (torch.rand(b, k, generator=g) > 0.3).float()
    return static, torch.full((b,), 0.4208), rev, raw, gate


@pytest.mark.parametrize("arm", stat.ARMS)
def test_exactly_one_named_constant_differs(arm):
    c = treatment.constants(arm)
    diff = [k for k in c if c[k] != treatment.DEFAULT[k]]
    assert diff == [treatment.CHANGES[arm][0]]
    assert c[diff[0]] == treatment.CHANGES[arm][1]


def test_scorer_body_has_only_the_three_registered_constant_edits():
    old = ast.parse(inspect.getsource(h184_treatment.adapter_components)).body[0]
    new = ast.parse(inspect.getsource(treatment.adapter_components)).body[0]
    new.args.kwonlyargs = []
    new.args.kw_defaults = []

    class Undo(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "min"
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "topk"
            ):
                node.args[0] = ast.Constant(8)
            return node

        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.op, ast.Mult)
                and isinstance(node.right, ast.Name)
                and node.right.id in ("kernel_multiplier", "null_multiplier")
            ):
                return node.left
            return node

    assert ast.dump(Undo().visit(new), include_attributes=False) == ast.dump(
        old, include_attributes=False
    )


@pytest.mark.parametrize("k", [4, 20])
def test_reference_matches_parent_bit_exact_and_topk_four_reveals_noop(k):
    torch.manual_seed(186101)
    ad = FakeAdapter().eval()
    args = synthetic(k=k)
    before = {n: t.clone() for n, t in ad.state_dict().items()}
    with torch.inference_mode():
        expected = h184_treatment.adapter_components(ad, *args)
        actual = treatment.adapter_components(ad, *args)
        assert all(torch.equal(actual[n], expected[n]) for n in expected)
        for arm in ("TOPK4", "TOPK16"):
            changed = treatment.adapter_components(
                ad, *args, **treatment.constants(arm)
            )
            assert (
                all(torch.equal(changed[n], actual[n]) for n in actual)
                if k == 4
                else not torch.equal(changed["KERNEL"], actual["KERNEL"])
            )
        for arm in ("KX0.5", "KX1.5", "KX2.0"):
            changed = treatment.adapter_components(
                ad, *args, **treatment.constants(arm)
            )
            assert torch.equal(changed["PRIOR"], actual["PRIOR"]) and torch.equal(
                changed["ATTN"], actual["ATTN"]
            )
            assert not torch.equal(changed["KERNEL"], actual["KERNEL"])
        for arm in ("W0X0.5", "W0X2"):
            changed = treatment.adapter_components(
                ad, *args, **treatment.constants(arm)
            )
            assert torch.equal(changed["PRIOR"], actual["PRIOR"]) and not torch.equal(
                changed["KERNEL"], actual["KERNEL"]
            )
    assert all(torch.equal(t, before[n]) for n, t in ad.state_dict().items())


def rows():
    return [dict(family_id=f"f{i:03}", context_sha256=f"c{i:03}") for i in range(205)]


def test_fresh_reshuffles_and_imported_coverage_exact_once():
    fm = stat.fold_manifest(rows())
    assert len({json.dumps(v["families"]) for v in fm.values()}) == 3
    for value in fm.values():
        assert value["coverage"]["b_exact_once"] and value["coverage"]["a_exact_once"]
        assert sorted(sum(value["families"], [])) == sorted(
            r["family_id"] for r in rows()
        )


def test_imported_policy_identity_and_seven_arm_holm_margin():
    assert stat.pooled_b_select is h177.pooled_b_select
    assert stat.H177_DECISION_RULE is h177.H177_DECISION_RULE
    assert stat.StageOrderGuard is h177.StageOrderGuard
    assert stat.verify_pooled_b_coverage is h177.verify_pooled_b_coverage
    assert stat.resolve_h177_verdict is h177.resolve_h177_verdict
    p = np.zeros((3, 205, 8))
    p[:, :, 1] = 0.001
    p[:, :, 2] = 0.02
    d = stat.select(rows(), p)
    assert not d["KX0.5"]["admitted"] and d["KX1.5"]["admitted"]
    expected = h177.holm({a: v["p_one_sided"] for a, v in d.items()})
    assert all(
        v["holm_p"] == expected[a]
        and v["holm_multiplicity"] == 7
        and v["optimizer_updates"] == 0
        for a, v in d.items()
    )
    assert h177.ARMS == ("CTRL", "ACC4", "ACC16", "OC15")


def test_b_lock_required_by_actual_evaluation_path(tmp_path):
    with stat.bindings():
        guard = stat.StageOrderGuard(dict.fromkeys(stat.ARMS, 1))
    engine = Engine.__new__(Engine)
    engine.set_policy_guard(guard, "initial")
    for stage in ["A", "DL-Val"]:
        with pytest.raises(RuntimeError, match="forbidden before"):
            engine.evaluate_many([], stage, tmp_path, "negative")
    p = tmp_path / "pooled_b_selection_lock.json"
    p.write_text(
        json.dumps(
            dict(
                wave="H177",
                study="H186",
                pooled_b_families=205,
                decisions=dict.fromkeys(stat.ARMS, {}),
            )
        )
    )
    with stat.bindings():
        guard.mark_b_locked(p)
        guard.require_b_locked()
        with pytest.raises(RuntimeError, match="immutable"):
            guard.set_candidate_step(stat.ARMS[0], 2)
    engine.selection_hash = stat.lock_hash(p)
    p.write_text(p.read_text() + " ")
    with pytest.raises(RuntimeError, match="mutation"):
        engine.evaluate_many([], "A", tmp_path, "negative")


def test_no_optimizer_or_backward_and_guard_thresholds():
    with pytest.raises(RuntimeError, match="forbids"):
        forbidden_training()
    tree = ast.parse(Path(__file__).with_name("runner.py").read_text())
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("backward", "step", "AdamW", "Adam", "SGD")
        for n in ast.walk(tree)
    )
    monitor = load(
        "test_monitor", "research/flywheel_selfplay/h186_kernel_sweep/monitor.py"
    )
    ok = dict(
        load=1,
        cpu=1,
        available=32 * 2**30,
        free=100e9,
        disk_used_fraction=0.93,
        rss=1e9,
        threads=3,
    )
    assert monitor.violations(ok) == []
    for changes in [
        dict(cpu=86),
        dict(threads=5),
        dict(rss=6e9),
        dict(disk_used_fraction=0.938),
        dict(available=15 * 2**30),
        dict(free=39e9),
    ]:
        assert monitor.violations(dict(ok, **changes))


def test_axis_zero_keeps_family_uncertainty_and_zero_elo():
    rng = np.random.default_rng(186199)
    p = np.zeros((3, 205, 8))
    noise = rng.normal(0, 0.1, (3, 205))
    noise -= noise.mean(axis=1, keepdims=True)
    p[:, :, 1] = noise + 0.0002
    result = stat.summarize(rows(), p)
    assert (
        result["contrasts"]["KX0.5"]["ci95"][0]
        < 0
        < result["contrasts"]["KX0.5"]["ci95"][1]
    )
    null = stat.summarize(rows(), np.zeros_like(p))
    assert null["median_elo"] == [1500.0] * 8


def test_verdict_requires_b_admission_three_positive_and_calibration():
    parent = Path(h177.__file__)
    receipt = json.loads(
        (parent.parent.parent / "h178_pooled_calibration/h178_receipt.json").read_text()
    )
    d = {a: dict(admitted=False) for a in stat.ARMS}
    a = dict(
        contrasts={
            x: dict(median_gain=0.02, positive_seeds=3, ci95=[0.01, 0.03])
            for x in stat.ARMS
        }
    )
    assert (
        stat.verdict(d, a, receipt, stat.lock_hash(parent))["verdict"] == "NO_ADVANCE"
    )
    d["KX0.5"]["admitted"] = True
    assert stat.verdict(d, a, receipt, stat.lock_hash(parent))["verdict"] == "ADVANCE"
    a["contrasts"]["KX0.5"]["positive_seeds"] = 2
    assert (
        stat.verdict(d, a, receipt, stat.lock_hash(parent))["verdict"] == "NO_ADVANCE"
    )
