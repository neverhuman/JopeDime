"""Frozen CPU bf16 training rules; fp32 backbone and assessment remain outside these contexts."""

from contextlib import contextmanager
import hashlib
import math
import torch

ARMS = ("CONTROL", "BF16_AMP", "BF16_SR")
STABILITY_L2_BOUND = (
    1.0  # Descriptive predeclared bound versus matched CONTROL; not a utility gate.
)


def sr_seed(seed, outer, step):
    return (
        int.from_bytes(
            hashlib.sha256(f"H259:bf16-sr:{seed}:{outer}:{step}".encode()).digest()[:8],
            "little",
        )
        & 0x7FFFFFFF
    )


def stochastic_round_bf16(value, generator=None, uniform_u16=None):
    """Adjacent-bf16 unbiased rounding of finite representable fp32 values, including negative/subnormal values."""
    if value.dtype != torch.float32 or value.device.type != "cpu":
        raise ValueError("SR requires CPU fp32 update values")
    if (
        not torch.isfinite(value).all()
        or (value.abs() > torch.finfo(torch.bfloat16).max).any()
    ):
        raise FloatingPointError("SR nonfinite/bf16 overflow; no clipping")
    bits = value.detach().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    upper, rem = bits >> 16, bits & 0xFFFF
    if uniform_u16 is None:
        uniform_u16 = torch.randint(
            0, 65536, value.shape, generator=generator, dtype=torch.int32
        )
    if (
        uniform_u16.shape != value.shape
        or ((uniform_u16 < 0) | (uniform_u16 >= 65536)).any()
    ):
        raise ValueError("uniform 16-bit integer inventory")
    rounded = upper + (uniform_u16.to(torch.int64) < rem).to(torch.int64)
    signed = ((rounded + 32768) % 65536 - 32768).to(torch.int16)
    result = signed.view(torch.bfloat16)
    if not torch.isfinite(result).all():
        raise FloatingPointError("SR produced nonfinite bf16")
    return result


def nonfinite_counts(tensors):
    tensors = list(tensors)
    return dict(
        nan=sum(int(torch.isnan(x).sum()) for x in tensors),
        inf=sum(int(torch.isinf(x).sum()) for x in tensors),
    )


class PrecisionAdamW:
    """CONTROL/AMP use fp32 master parameters. SR uses bf16 authoritative weights and fp32 work/moment state.

    SR discards the fp32 residual after every stochastic writeback: each next work value is the stored bf16 value.
    Differentiable fp32 views in functional_call permit CPU layernorm while eligible adapter GEMMs use bf16 autocast.
    """

    def __init__(self, arm, params, names=None):
        if arm not in ARMS:
            raise ValueError("unknown precision arm")
        self.arm = arm
        self.active_params = list(params)
        self.names = list(names or [str(i) for i in range(len(self.active_params))])
        if len(self.names) != len(self.active_params) or any(
            p.dtype != torch.float32 for p in self.active_params
        ):
            raise ValueError("precision arm starts from fp32 parameters")
        self.optimizer_parameters = (
            [torch.nn.Parameter(p.detach().float().clone()) for p in self.active_params]
            if arm == "BF16_SR"
            else self.active_params
        )
        self.opt = torch.optim.AdamW(
            self.optimizer_parameters, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4
        )
        self.param_groups = self.opt.param_groups
        self.step_index = 0
        self.seed = self.outer = 0
        self.last_diagnostics = dict(
            quantization_l2=0.0,
            rounding_seed=None,
            active_dtype="torch.float32",
            state_dtypes=[],
            nan_inf=dict(nan=0, inf=0),
        )

    def begin(self, seed, outer):
        self.seed, self.outer = seed, outer
        if self.arm == "BF16_SR":
            self._writeback(0)

    def zero_grad(self, set_to_none=True):
        self.opt.zero_grad(set_to_none=set_to_none)
        if self.arm == "BF16_SR":
            for p in self.active_params:
                p.grad = None

    def forward_context(self):
        return torch.autocast(
            "cpu",
            dtype=torch.bfloat16,
            enabled=self.arm != "CONTROL",
            cache_enabled=False,
        )

    def predict(self, engine, kwargs, tok, mask, anchor):
        if any(x.dtype != torch.float32 for x in anchor):
            raise RuntimeError("frozen DOPE/anchor features must remain fp32")
        with self.forward_context():
            if self.arm == "BF16_SR":
                replacements = {
                    "net." + n: p.float()
                    for n, p in zip(self.names, self.active_params, strict=True)
                }
                pred = torch.func.functional_call(
                    engine.adapter_call,
                    replacements,
                    (kwargs, tok, mask, anchor),
                    strict=False,
                )
            else:
                pred = engine.forward(kwargs, tok, mask, anchor)
        counts = nonfinite_counts([pred])
        self.last_diagnostics["forward_nan_inf"] = counts
        if counts["nan"] or counts["inf"]:
            raise FloatingPointError("nonfinite adapter forward: " + str(counts))
        return pred.float()

    def filter_gradients(self):
        if self.arm == "BF16_SR":
            for active, work in zip(
                self.active_params, self.optimizer_parameters, strict=True
            ):
                work.grad = (
                    None
                    if active.grad is None
                    else active.grad.detach().float().clone()
                )
        counts = nonfinite_counts(
            p.grad for p in self.optimizer_parameters if p.grad is not None
        )
        self.last_diagnostics["nan_inf"] = counts
        if counts["nan"] or counts["inf"]:
            raise FloatingPointError("nonfinite adapter gradient: " + str(counts))

    @torch.no_grad()
    def _writeback(self, step):
        rng = torch.Generator(device="cpu").manual_seed(
            sr_seed(self.seed, self.outer, step)
        )
        error = 0.0
        for active, work in zip(
            self.active_params, self.optimizer_parameters, strict=True
        ):
            rounded = stochastic_round_bf16(work.detach(), generator=rng)
            error += float(
                (rounded.float().double() - work.detach().double()).square().sum()
            )
            active.data = rounded.clone()
            work.copy_(rounded.float())
        self.last_diagnostics.update(
            quantization_l2=math.sqrt(error),
            rounding_seed=sr_seed(self.seed, self.outer, step),
            active_dtype="torch.bfloat16",
        )

    def step(self):
        self.opt.step()
        self.step_index += 1
        if self.arm == "BF16_SR":
            self._writeback(self.step_index)
        floating = [
            v
            for state in self.opt.state.values()
            for v in state.values()
            if torch.is_tensor(v) and v.is_floating_point()
        ]
        if any(v.dtype != torch.float32 for v in floating):
            raise RuntimeError("AdamW state must remain fp32")
        counts = nonfinite_counts([*self.active_params, *floating])
        self.last_diagnostics.update(
            state_dtypes=sorted({str(v.dtype) for v in floating}), nan_inf=counts
        )
        if counts["nan"] or counts["inf"]:
            raise FloatingPointError("nonfinite weights/AdamW state: " + str(counts))

    @contextmanager
    def fp32_scoring(self):
        original = [p.data for p in self.active_params]
        try:
            for p in self.active_params:
                p.data = p.data.float()
            with torch.autocast("cpu", enabled=False):
                yield
        finally:
            for p, value in zip(self.active_params, original, strict=True):
                p.data = value
