"""Parameter-free H165 adapter-input dropout and norm-scaled gradient noise."""

import math
import torch

PARAMETERS = 29211


class InputDropout(torch.nn.Dropout):
    """Parameter-free input intervention, with observed training-mask evidence."""

    def forward(self, value):
        result = super().forward(value)
        if self.training:
            nz = value.detach() != 0
            self.changed = int((result.detach() != value.detach()).sum())
            self.nonzero_inputs = int(nz.sum())
            self.dropped_nonzero = int((nz & (result.detach() == 0)).sum())
        return result


def noise_std(eta, scale, step, parameters=PARAMETERS):
    if (
        eta not in (0.01, 0.1)
        or not math.isfinite(scale)
        or scale < 0
        or step < 1
        or parameters <= 0
    ):
        raise ValueError("invalid fixed noise schedule")
    return scale * math.sqrt(eta / parameters) / (1 + step) ** 0.275


def add_noise(params, eta, scale, step, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sigma = noise_std(eta, scale, step)
    total = 0.0
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)  # Noise also acts on a zero data gradient.
            noise = (
                torch.randn(p.shape, dtype=p.dtype, device="cpu", generator=generator)
                * sigma
            )
            p.grad.add_(noise)
            total += float(noise.double().square().sum())
    return dict(
        sigma=sigma,
        noise_l2=math.sqrt(total),
        expected_noise_l2=sigma * math.sqrt(PARAMETERS),
    )


def adapter_forward(
    net,
    prog,
    sos,
    tokens,
    mask,
    scale,
    rev,
    raw,
    gate,
    anchor_total,
    anchor_prior,
    site,
):
    import torch.nn.functional as F

    ad = net.adapter_10tok
    pooled, query, semantic, adjustment = ad.encode_multimodal_context_and_candidates(
        mm_tokens_10x32=site(tokens),
        mm_mask_10=mask,
        prog_412=prog,
        sos_prior=sos,
        active_modality_override_mask=None,
    )
    static = dict(
        eff_prior=anchor_prior + adjustment,
        mm_prior_adj=adjustment,
        sem24=semantic,
        cand_q=query,
        cand_q_norm=F.normalize(query, p=2, dim=-1, eps=1e-8),
        ctx_pooled=pooled,
    )
    delta = ad.compute_adapter_delta_from_static_terms(
        static_terms=static,
        scale=scale,
        revealed_indices=rev,
        revealed_raw_f=raw,
        revealed_gate=gate,
    )
    return (anchor_total + delta).float()
