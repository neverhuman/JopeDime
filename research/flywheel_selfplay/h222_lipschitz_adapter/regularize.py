"""Explicit once-per-update spectral power steps and differentiable context VJP."""

import torch
from torch import nn
from torch.nn.utils import parametrize
import torch.nn.functional as F


class SpectralWeight(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("u", F.normalize(torch.ones(weight.shape[0]), dim=0))
        self.register_buffer("v", F.normalize(torch.ones(weight.shape[1]), dim=0))
        self.steps = 0
        self.advance(
            weight
        )  # One declared initialization iteration, not an optimizer update.
        self.steps = 0

    @torch.no_grad()
    def advance(self, weight):
        self.v.copy_(F.normalize(weight.detach().T @ self.u, dim=0, eps=1e-12))
        self.u.copy_(F.normalize(weight.detach() @ self.v, dim=0, eps=1e-12))
        self.steps += 1

    def forward(self, weight):
        sigma = (self.u.clone() @ weight @ self.v.clone()).abs().clamp_min(1e-12)
        return weight / sigma


def spectralize(adapter):
    targets = []
    for name, module in list(adapter.named_modules()):
        if name == "anchor" or name.startswith("anchor."):
            continue
        fields = []
        if isinstance(module, nn.Linear) and module.weight.requires_grad:
            fields.append("weight")
        if (
            isinstance(module, nn.MultiheadAttention)
            and module.in_proj_weight is not None
            and module.in_proj_weight.requires_grad
        ):
            fields.append("in_proj_weight")
        for field in fields:
            normalizer = SpectralWeight(getattr(module, field))
            parametrize.register_parametrization(module, field, normalizer)
            targets.append((name, field, module, normalizer))
    if not targets:
        raise ValueError("no trainable adapter linear transforms")
    return targets


def advance(targets):
    for _, field, module, normalizer in targets:
        normalizer.advance(getattr(module.parametrizations, field).original)


def jacobian_penalty(prediction, tokens, mask, stream):
    gen = torch.Generator(device="cpu").manual_seed(stream % (2**63 - 1))
    outputs = prediction[:, 4:]
    probe = (
        torch.randint(0, 2, outputs.shape, generator=gen, dtype=torch.int64).to(
            outputs.dtype
        )
        * 2
        - 1
    )
    grad = torch.autograd.grad(
        (outputs * probe).sum(), tokens, create_graph=True, retain_graph=True
    )[0]
    active = mask.clone()
    active[:, 0] = True
    return (
        (grad * active.unsqueeze(-1)).square().sum((1, 2)) / outputs.shape[1]
    ).mean()
