"""Synthetic loss/gradient contracts; no real outcomes, model, trainer or GPU."""

import ast
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))
from losses import compose_cost, targets, w3_loss  # noqa: E402


def row(fitness=1.0, logratio=0.2, gate=True, revealed=False, status=None):
    return dict(
        status=status or ("completed_pass" if gate else "completed_gate_fail"),
        fitness=fitness,
        logratio=logratio,
        gate=gate,
        revealed=revealed,
    )


def tensors(n):
    return torch.linspace(-0.2, 0.4, n, dtype=torch.float64)[None], torch.zeros(
        (1, n), dtype=torch.float64, requires_grad=True
    )


def test_disabled_does_not_read_inputs():
    assert w3_loss("not_an_arm", object(), object(), object()) is None


@pytest.mark.parametrize("n", [1, 2, 8, 16, 20])
def test_top_order_floor_and_clip(n):
    rs = [row(float(i)) for i in range(n)] + [row(1000.0)]
    s = targets([rs])[0]
    expected = max(float(min(16, n + 1) - 1) if n >= 16 else rs[-1]["fitness"], 0.05)
    assert s["scale"] == expected
    assert s["rows"][0]["regret"] == 0
    assert max(r["regret"] for r in s["rows"]) <= 6


def test_single_pass_floor_and_failure_eight():
    s = targets([[row(-3.0), row(100.0, 0.0, False)]])[0]
    assert s["scale"] == 0.05 and s["rows"][0]["regret"] == 0
    assert s["rows"][1]["regret"] == 8
    assert s["rows"][1]["cost"] == 2  # failure penalty is not quality


@pytest.mark.parametrize("arm", ["a", "b", "c", "d"])
def test_finite_loss_and_only_active_gradients(arm):
    rs = [
        [
            row(revealed=True),
            row(1.0),
            row(100.0, -0.2, False),
            row(None, None, None, status="unknown"),
        ]
    ]
    prior, delta = tensors(4)
    heads = {
        k: torch.zeros_like(delta, requires_grad=True)
        for k in ("quality", "residual", "gate")
    }
    out = w3_loss(arm, prior, delta, rs, enabled=True, heads=heads)
    assert out["status"] == "LOSS_ONLY" and torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [h.grad for h in heads.values()] if arm == "d" else [delta.grad]
    for grad in grads:
        assert torch.isfinite(grad).all()
        assert grad[0, 0] == grad[0, 3] == 0
    if arm == "d":
        assert heads["residual"].grad[0, 2] == 0
        assert out["component_counts"] == {"quality": 2, "residual": 1, "gate": 2}


def test_components_are_not_inferred_from_penalties():
    rs = [
        [
            row(100.0, -0.3, False),
            row(None, 0.5, True),
            row(1.0, None, True),
            row(None, None, None, status="completed_pass"),
        ]
    ]
    r = targets(rs)[0]["rows"]
    assert r[0]["quality"] == -0.3 and not r[0]["residual_mask"]
    assert r[1]["quality_mask"] and not r[1]["rank_mask"] and not r[1]["residual_mask"]
    assert r[2]["rank_mask"] and not r[2]["quality_mask"]
    assert not any(
        r[3][k] for k in ("rank_mask", "quality_mask", "residual_mask", "gate_mask")
    )


@pytest.mark.parametrize("arm", ["a", "b", "c", "d"])
def test_empty_supervision_is_explicit(arm):
    prior, delta = tensors(2)
    rs = [[row(None, None, None, status="unknown") for _ in range(2)]]
    heads = {
        k: torch.zeros_like(delta, requires_grad=True)
        for k in ("quality", "residual", "gate")
    }
    out = w3_loss(arm, prior, delta, rs, enabled=True, heads=heads)
    assert out["status"] == "NO_SUPERVISION" and out["loss"].item() == 0
    out["loss"].backward()


def test_no_passer_path_is_uniform_not_a_winner():
    rs = [[row(100.0, -0.2, False), row(200.0, 0.2, False)]]
    assert [r["regret"] for r in targets(rs)[0]["rows"]] == [8, 8]
    p = torch.zeros((1, 2), dtype=torch.float64)
    d = p.clone().requires_grad_()
    out = w3_loss("c", p, d, rs, enabled=True)
    out["loss"].backward()
    assert out["loss"].abs() < 1e-12 and torch.equal(d.grad, p)


def test_zero_update_and_fixed_prior_gradient():
    prior, delta = tensors(20)
    prior.requires_grad_()
    before = prior.detach().clone()
    for _ in ("a", "b", "c", "d"):
        assert torch.equal(compose_cost(prior, delta), before)
        assert torch.equal(
            compose_cost(prior, delta).argsort(stable=True), before.argsort(stable=True)
        )
    compose_cost(prior, delta).sum().backward()
    assert prior.grad is None and torch.equal(prior.detach(), before)


def test_b_reproduces_pinned_h58_d_on_complete_slates():
    path = Path(__file__).parents[1] / "h58/runner.py"
    tree = ast.parse(path.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_loss_arm"
    )
    scope = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    rs = [
        [row(float(i) / 10, revealed=i < 4) for i in range(24)] + [row(100, 0.0, False)]
    ]
    prior, delta = tensors(25)
    actual = w3_loss("b", prior, delta, rs, enabled=True)["loss"]
    expected = scope["compute_loss_arm"](
        "D",
        prior + delta,
        torch.tensor(
            [[r["fitness"] if r["gate"] else 2 for r in rs[0]]], dtype=torch.float64
        ),
        torch.tensor([[r["gate"] for r in rs[0]]]),
    )
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    ga = torch.autograd.grad(actual, delta, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, delta)[0]
    assert torch.allclose(ga, ge, atol=1e-10, rtol=1e-10)


def test_c_target_shift_invariance():
    a = [row(float(i) / 100) for i in range(20)]
    b = [row(r["fitness"] + 10) for r in a]
    assert [r["regret"] for r in targets([a])[0]["rows"]] == pytest.approx(
        [r["regret"] for r in targets([b])[0]["rows"]]
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(gate=1),
        dict(revealed=1),
        dict(fitness=float("nan")),
        dict(logratio=float("inf")),
        dict(gate=False),
        dict(status="signature_pass"),
        dict(status="unknown"),
        dict(outcome=1),
    ],
)
def test_invalid_or_inferred_rows_refused(change):
    r = row()
    r.update(change)
    with pytest.raises(ValueError):
        targets([[r]])


def test_shape_and_nonfinite_predictions_refused():
    p, d = tensors(2)
    with pytest.raises(ValueError):
        w3_loss("c", p, d, [[row()]], enabled=True)
    with pytest.raises(ValueError):
        compose_cost(p, torch.tensor([[float("nan"), 1.0]]))


def test_c_gradient_moves_best_cost_down():
    prior = torch.zeros((1, 3), dtype=torch.float64)
    delta = prior.clone().requires_grad_()
    out = w3_loss(
        "c", prior, delta, [[row(0.0), row(0.1), row(100.0, 0.4, False)]], enabled=True
    )
    out["loss"].backward()
    assert delta.grad[0, 0] > 0 and delta.grad[0, 2] < 0


def test_continuous_residual_uses_only_passing_measured_components():
    values = targets([[row(1.0, 0.2), row(100, 0.2, False)]])[0]["rows"]
    assert values[0]["residual"] == pytest.approx(0.82)
    assert values[1]["residual_mask"] is False


def test_prior_composition_overflow_refused():
    p = torch.full((1, 2), 1e308, dtype=torch.float64)
    with pytest.raises(ValueError):
        compose_cost(p, p)


@pytest.mark.parametrize("arm", ["a", "b", "c", "d"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_supported_cpu_dtypes_have_finite_backward(arm, dtype):
    prior = torch.zeros((1, 2), dtype=dtype)
    delta = torch.zeros_like(prior, requires_grad=True)
    heads = {
        k: torch.zeros_like(prior, requires_grad=True)
        for k in ("quality", "residual", "gate")
    }
    result = w3_loss(
        arm, prior, delta, [[row(0.2), row(0.5)]], enabled=True, heads=heads
    )
    result["loss"].backward()
    grads = [h.grad for h in heads.values()] if arm == "d" else [delta.grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("arm", ["a", "d"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unsupported_cpu_dtypes_refused_before_loss(arm, dtype):
    prior = torch.zeros((1, 2), dtype=dtype)
    delta = torch.zeros_like(prior, requires_grad=True)
    heads = {
        k: torch.zeros_like(prior, requires_grad=True)
        for k in ("quality", "residual", "gate")
    }
    with pytest.raises(ValueError, match="CPU float32 or float64 tensors required"):
        w3_loss(arm, prior, delta, [[row(0.2), row(0.5)]], enabled=True, heads=heads)


def test_unsupported_auxiliary_head_dtype_refused():
    prior, delta = tensors(2)
    heads = {
        k: torch.zeros((1, 2), dtype=torch.float16, requires_grad=True)
        for k in ("quality", "residual", "gate")
    }
    with pytest.raises(ValueError, match="CPU float32 or float64 tensors required"):
        w3_loss("d", prior, delta, [[row(0.2), row(0.5)]], enabled=True, heads=heads)
