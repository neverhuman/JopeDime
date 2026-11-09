"""Data-free H259 acceptance: unbiased SR, dtype/state isolation, actual schedule, strict selection and guards."""

from pathlib import Path
import ast
import json
import sys
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import precision as px  # noqa:E402
import treatment as tx  # noqa:E402
import analysis as stat  # noqa:E402
import monitor  # noqa:E402
import launcher  # noqa:E402


def test_sr_unbiased_exact_complete_16bit_uniform_grid():
    fixed = torch.tensor(
        [-7.123, -1.234567, -0.001234, 0.0, 0.001234, 1.234567, 2.71828, 100.125],
        dtype=torch.float32,
    )
    values = fixed[None, :].expand(65536, -1).contiguous()
    uniforms = torch.arange(65536, dtype=torch.int32)[:, None].expand_as(values)
    rounded = px.stochastic_round_bf16(values, uniform_u16=uniforms)
    assert rounded.dtype == torch.bfloat16
    assert torch.allclose(rounded.double().mean(0), fixed.double(), atol=1e-12, rtol=0)


def test_sr_preserves_exact_bf16_and_signed_zero():
    x = torch.tensor([-2.0, -0.5, -0.0, 0.0, 1.0, 16.0], dtype=torch.float32)
    y = px.stochastic_round_bf16(x, torch.Generator().manual_seed(1)).float()
    assert torch.equal(x.view(torch.int32), y.view(torch.int32))


def test_sr_seed_reproducibility_and_step_independence():
    x = torch.full((10000,), 1.23456)
    a = px.stochastic_round_bf16(
        x, torch.Generator().manual_seed(px.sr_seed(259101, 2, 3))
    )
    b = px.stochastic_round_bf16(
        x, torch.Generator().manual_seed(px.sr_seed(259101, 2, 3))
    )
    c = px.stochastic_round_bf16(
        x, torch.Generator().manual_seed(px.sr_seed(259101, 2, 4))
    )
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert (
        len(
            {
                px.sr_seed(s, f, t)
                for s in stat.SEEDS
                for f in range(5)
                for t in range(41)
            }
        )
        == 615
    )


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), torch.finfo(torch.float32).max]
)
def test_sr_rejects_nonfinite_and_overflow_without_clipping(value):
    with pytest.raises(FloatingPointError):
        px.stochastic_round_bf16(torch.tensor([value]))


def test_sr_random_roundings_are_adjacent_bf16_values():
    x = torch.tensor([-3.123, -0.000001234, 0.000001234, 0.99991, 17.987]).repeat(100)
    y = px.stochastic_round_bf16(x, torch.Generator().manual_seed(2)).float()
    bits = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    got = y.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    assert ((got == ((bits >> 16) << 16)) | (got == (((bits >> 16) + 1) << 16))).all()


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.attn = torch.nn.MultiheadAttention(8, 2, dropout=0, batch_first=True)
        self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        x = self.norm(self.linear(x))
        return self.attn(x, x, x, need_weights=False)[0] + self.frozen


class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, kwargs, tok, mask, anchor):
        return self.net(tok) + anchor[0]


class FakeEngine:
    def __init__(self, net):
        self.net = net
        self.adapter_call = Wrapper(net)

    def forward(self, kwargs, tok, mask, anchor):
        return self.net(tok) + anchor[0]


@pytest.mark.parametrize("arm", px.ARMS)
def test_training_operator_dtype_backward_state_and_fp32_scoring(arm):
    torch.manual_seed(4)
    net = Toy().eval()
    named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    names, params = zip(*named, strict=True)
    frozen = net.frozen.detach().clone()
    engine = FakeEngine(net)
    opt = tx.make_optimizer(arm, params, names)
    opt.begin(259101, 0)
    opt.zero_grad()
    tok = torch.randn(2, 4, 8)
    anchor = (torch.zeros(2, 4, 8), torch.zeros(2, 4, 8))
    pred = opt.predict(engine, {}, tok, None, anchor)
    assert pred.dtype == torch.float32
    pred.square().mean().backward()
    opt.filter_gradients()
    torch.nn.utils.clip_grad_norm_(opt.optimizer_parameters, 1.0)
    opt.step()
    expected = torch.bfloat16 if arm == "BF16_SR" else torch.float32
    assert all(p.dtype == expected for p in params)
    assert all(p.dtype == torch.float32 for p in opt.optimizer_parameters)
    assert opt.last_diagnostics["state_dtypes"] == [
        "torch.float32"
    ] and opt.last_diagnostics["nan_inf"] == dict(nan=0, inf=0)
    assert torch.equal(net.frozen, frozen) and net.frozen.dtype == torch.float32
    with opt.fp32_scoring():
        assert all(
            p.dtype == torch.float32 for p in params
        ) and not torch.is_autocast_enabled("cpu")
        assert engine.forward({}, tok, None, anchor).dtype == torch.float32
    assert all(p.dtype == expected for p in params)
    if arm == "BF16_SR":
        assert all(
            torch.equal(a.float(), w)
            for a, w in zip(params, opt.optimizer_parameters, strict=True)
        )


def test_control_adamw_is_bit_exact_with_h170():
    a = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    b = torch.nn.Parameter(a.detach().clone())
    ours = tx.make_optimizer("CONTROL", [a])
    reference = torch.optim.AdamW([b], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    ours.begin(259101, 0)
    for _ in range(40):
        a.grad = b.grad = torch.tensor([0.3, -0.2])
        ours.filter_gradients()
        ours.step()
        reference.step()
    assert torch.equal(a, b)


def test_actual_runner_schedule_matches_frozen_table():
    module = ast.parse((HERE / "runner.py").read_text())
    main = next(
        n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    node = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "actual_schedule" for t in n.targets
        )
    )
    actual = eval(
        compile(ast.Expression(node.value), "<runner schedule>", "eval"),
        dict(stat=stat, settings=tx.settings),
    )
    assert actual == json.loads((HERE / "schedules.json").read_text())["arms"]


def test_paired_h170_sampler_is_exact_and_fit_only():
    rows = [
        dict(family_id=f"f{i:03}", context_sha256=f"c{i:03}-{j}")
        for i in range(205)
        for j in range(2)
    ]
    fit, select, assess = stat.partition(rows, 1, 259101)
    ids = {r["context_sha256"] for r in fit}
    orders = {c: np.arange(32) for c in ids}
    plan = stat.structured_plan(fit, orders, 259101, 1)
    original = stat.batch_plan(ids, orders, stat.stream_seed(259101, 1))
    assert len(plan) == 40
    for a, b in zip(plan, original, strict=True):
        assert (
            a[0] == b[0]
            and a[1] == b[1]
            and all(np.array_equal(i, j) for i, j in zip(a[2], b[2], strict=True))
        )
        assert len(a[1]) == 8 and len(set(a[1])) == 8 and set(a[1]) <= ids
        assert not set(a[1]) & {r["context_sha256"] for r in select + assess}
        for ix in a[2]:
            assert len(set(ix)) == 24 and ix[:4].tolist() == [0, 1, 2, 3]


def test_three_positive_seeds_are_required_and_holm_two():
    rows = [dict(family_id=f"f{i}") for i in range(4)]
    panel = np.zeros((3, 4, 4))
    panel[:, :, 2] = np.array([0.02, 0.02, -0.001])[:, None]
    s = stat.summarize(rows, panel, context_weighted=True)
    c = s["contrasts"]["BF16_AMP_minus_r882"]
    assert c["median_gain"] > 0.002 and c["positive_seeds"] == 2 and not c["advance"]
    panel[:, :, 2] = 0.02
    s = stat.summarize(rows, panel, context_weighted=True)
    assert s["contrasts"]["BF16_AMP_minus_r882"]["advance"]
    assert stat.holm(dict(a=0.01, b=0.04)) == dict(a=0.02, b=0.04)


def test_strict_selector_ties_and_small_gain_retain_r882():
    rows = [dict(family_id=f"f{i}") for i in range(4)]
    for gain in [0, 0.0005, 0.001]:
        panel = np.zeros((4, 5))
        panel[:, 1:] = gain
        assert stat.choose(rows, panel, 12)["selected_step"] == 0
    panel = np.zeros((4, 5))
    panel[:, 2:] = 0.003
    assert stat.choose(rows, panel, 12)["selected_step"] == 20


def test_context_weighted_statistic_keeps_distinct_family_label():
    rows = [dict(family_id="a")] + [dict(family_id="b")] * 3
    panel = np.zeros((3, 4, 4))
    panel[:, 0, 2] = 0.04
    result = stat.summarize(rows, panel, context_weighted=True)
    assert result["primary_utility_by_seed"][0][2] == pytest.approx(0.01) and result[
        "family_utility_by_seed"
    ][0][2] == pytest.approx(0.02)


def sample(t=0, available=40 * 2**30, load5=90):
    return dict(
        utc=t,
        available=available,
        load5=load5,
        rss=1000,
        threads=2,
        free=400_000_000_000,
    )


def test_memory_and_load_guard_hysteresis():
    g = monitor.Guard()
    assert g.update(sample(0, 19 * 2**30))[1]
    assert g.update(sample(1, 25 * 2**30))[1]
    assert not g.update(sample(31, 25 * 2**30))[1]
    assert g.update(sample(32, load5=109))[1]
    assert g.update(sample(33, load5=101))[1]
    assert not g.update(sample(34, load5=100))[1]


def test_memory_hard_stop_and_fixed_deadline():
    g = monitor.Guard()
    assert not g.update(sample(0, 11 * 2**30))[0]
    assert g.update(sample(30, 11 * 2**30))[0]
    assert monitor.Guard().update(sample(0, 5 * 2**30))[0]
    assert monitor.COMPUTE_STOP == launcher.STOP == 1793724300
    assert monitor.violations(sample(monitor.COMPUTE_STOP))


def test_published_reviewed_launch_binding_and_no_resume(tmp_path):
    launch = dict(
        published_prereg_verified=True,
        head_review_verified=True,
        source_commit="a" * 40,
        manifest_sha256="m",
        prereg_sha256="p",
        fold_manifest_sha256="f",
        prereg_published_at="2026-11-03T15:00:00Z",
        output=str(tmp_path / "run"),
        resume_number=0,
    )
    tx.validate_launch(launch, "a" * 40, "m", "p", "f")
    for field in ["published_prereg_verified", "head_review_verified"]:
        with pytest.raises(RuntimeError):
            tx.validate_launch(dict(launch, **{field: False}), "a" * 40, "m", "p", "f")
    assert launcher.check_launch(launch, tmp_path / "run.launch.json")[1] == 0
    with pytest.raises(RuntimeError):
        launcher.check_launch(
            dict(launch, resume_number=1), tmp_path / "run.launch.json"
        )


def test_sr_unbiased_at_bf16_subnormal_boundary():
    x = torch.tensor(
        [2.0**-134, -(2.0**-134), 2.0**-140, -(2.0**-140)], dtype=torch.float32
    )
    values = x[None, :].expand(65536, -1).contiguous()
    uniforms = torch.arange(65536, dtype=torch.int32)[:, None].expand_as(values)
    result = px.stochastic_round_bf16(values, uniform_u16=uniforms)
    assert torch.allclose(result.double().mean(0), x.double(), atol=1e-52, rtol=0)
