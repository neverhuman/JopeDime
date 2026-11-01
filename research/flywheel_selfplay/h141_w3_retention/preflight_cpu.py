"""CPU-only admission, H113 loss/gradient parity and r882 zero-update controls."""

import importlib.util
import os
import json
from pathlib import Path
import numpy as np
import torch
import runner
import w3_losses

SOURCE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("H141_DATA_ROOT", str(SOURCE)))
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
spec = importlib.util.spec_from_file_location(
    "original_h113", ROOT / "h113_w3_continuous__losses.py"
)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
torch.manual_seed(37)
rows = [
    [
        {
            "status": "completed_pass",
            "fitness": -1 + j * 0.03,
            "logratio": -0.5 + j * 0.01,
            "gate": True,
            "revealed": j < 4,
        }
        for j in range(24)
    ]
]
rows[0][-1] = {
    "status": "completed_gate_fail",
    "fitness": None,
    "logratio": 0.25,
    "gate": False,
    "revealed": False,
}
rows[0][-2] = {
    "status": "unknown",
    "fitness": None,
    "logratio": None,
    "gate": None,
    "revealed": False,
}
cases = []
for arm in "abcd":
    pred0 = torch.randn(1, 24, dtype=torch.float64, requires_grad=True)
    pred1 = pred0.detach().clone().requires_grad_()
    heads0 = {
        key: torch.randn(1, 24, dtype=torch.float64, requires_grad=True)
        for key in ("quality", "residual", "gate")
    }
    heads1 = {
        key: value.detach().clone().requires_grad_() for key, value in heads0.items()
    }
    args = dict(rows=rows, enabled=True)
    left = original.w3_loss(
        arm,
        torch.zeros_like(pred0),
        pred0,
        heads=heads0 if arm == "d" else None,
        **args,
    )
    right = w3_losses.w3_loss(
        arm,
        torch.zeros_like(pred1),
        pred1,
        heads=heads1 if arm == "d" else None,
        **args,
    )
    assert torch.equal(left["loss"], right["loss"])
    left["loss"].backward()
    right["loss"].backward()
    if arm == "d":
        assert all(torch.equal(heads0[key].grad, heads1[key].grad) for key in heads0)
        assert heads1["residual"].grad[0, -1] == 0
        assert all(value.grad[0, -2] == 0 for value in heads1.values())
    else:
        assert torch.equal(pred0.grad, pred1.grad)
        assert pred1.grad[0, -2] == 0 and torch.equal(pred1.grad[0, :4], torch.zeros(4))
    cases.append({"arm": arm, "loss_and_gradient_bit_exact_vs_H113": True})
unknown = [
    [
        {
            "status": "unknown",
            "fitness": None,
            "logratio": None,
            "gate": None,
            "revealed": False,
        }
    ]
    * 8
]
pred = torch.ones(1, 8, requires_grad=True)
assert (
    w3_losses.w3_loss("a", torch.zeros_like(pred), pred, unknown, enabled=True)[
        "status"
    ]
    == "NO_SUPERVISION"
)
net = runner.load_net(torch.device("cpu"))
panel = runner.Panel(ROOT, net, torch.device("cpu"))
fit = [i for i in range(len(panel.cids)) if panel.fold[i] != 0 and panel.n(i) > 4][:8]
net.eval()
zero = []
with torch.no_grad():
    for index in fit:
        rows = [np.arange(panel.n(index))]
        base, _ = runner.components(net, panel, [index], rows)
        for arm in runner.ARMS:
            heads = runner.Heads() if arm.startswith("d") else None
            value, aux = runner.components(net, panel, [index], rows, heads, arm)
            assert torch.equal(base, value)
            if aux is not None:
                if arm == "d_scalar":
                    assert all(torch.equal(aux[..., j], base) for j in range(3))
                else:
                    assert torch.count_nonzero(aux[..., 1:]) == 0
            assert torch.argmin(base) == torch.argmin(value)
            zero.append(
                {
                    "context_sha256": panel.cids[index],
                    "arm": arm,
                    "full_cost_and_choice_identical": True,
                }
            )
assert sum(p.numel() for p in runner.trainable(net)) == 23546
head_counts = [sum(p.numel() for p in runner.Heads().parameters()) for _ in range(3)]
assert head_counts == [1251] * 3
real_gradient_cases = []
rng = np.random.default_rng(700)
indices = fit[:2]
slates = [
    np.r_[np.arange(4), rng.choice(np.arange(4, panel.n(i)), 188, replace=True)]
    for i in indices
]
actual_targets = panel.targets(indices, slates)
net.train()
for arm in runner.ARMS:
    net.zero_grad(set_to_none=True)
    heads = runner.Heads() if arm.startswith("d") else None
    pred, aux = runner.components(net, panel, indices, slates, heads, arm)
    if arm in ("a", "b", "c"):
        loss = w3_losses.w3_loss(
            arm, torch.zeros_like(pred), pred, actual_targets, enabled=True
        )["loss"]
    elif arm == "d_scalar":
        loss = (
            sum(
                w3_losses.w3_loss(
                    "a",
                    torch.zeros_like(pred),
                    aux[..., j],
                    actual_targets,
                    enabled=True,
                )["loss"]
                for j in range(3)
            )
            / 3
        )
    else:
        loss = w3_losses.w3_loss(
            "d",
            torch.zeros_like(pred),
            pred,
            actual_targets,
            enabled=True,
            heads={
                key: aux[..., j]
                for j, key in enumerate(("quality", "residual", "gate"))
            },
        )["loss"]
    loss.backward()
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in net.adapter_10tok.parameters()
    )
    assert all(p.grad is None for p in net.dope_model.parameters())
    real_gradient_cases.append(
        {
            "arm": arm,
            "fit_only_batch_gradient_finite": all(
                torch.isfinite(p.grad).all()
                for p in net.parameters()
                if p.grad is not None
            ),
            "optimizer_updates": 0,
        }
    )
    assert real_gradient_cases[-1]["fit_only_batch_gradient_finite"]
net.eval()
assert not torch.cuda.is_initialized()
receipt = {
    "status": "CPU_PREFLIGHT_PASS",
    "loss_gradient_cases": cases,
    "zero_update_cases": zero,
    "d_scalar_shuffled_head_parameter_counts": head_counts,
    "base_active_parameters": 23546,
    "real_fit_only_gradient_cases": real_gradient_cases,
    "DOPE_sha256": runner.dope_guard.snapshot_dope_weights(net).composite_sha256,
    "GPU_initialized": False,
    "GPU_seconds": 0,
    "native_fits": 0,
    "optimizer_updates": 0,
    "code_sha256": {
        name: runner.SHA(SOURCE / name)
        for name in (
            "runner.py",
            "w3_losses.py",
            "dope_guard.py",
            "elo_tracker.py",
            "preflight_cpu.py",
            "readout.py",
        )
    },
}
runner.write(ROOT / "cpu_preflight_receipt.json", receipt)
print(
    json.dumps({k: v for k, v in receipt.items() if k != "zero_update_cases"}, indent=2)
)
