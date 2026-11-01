"""Parameter-free H167 adapter input-feature Gaussian noise and None-safe gradient accounting."""

import math
import torch

PARAMETERS = 29211
SCALES = {"CTRL": 0.0, "IN05": 0.05, "IN10": 0.10, "IN10_FIXED": 0.10}


def compute_fit_feature_std(tokens, masks, fit_cids, excluded_cids=()):
    """Compute per-feature fit-fold TRAIN std s_j (shape 10x32) strictly from fit_cids."""
    fit_set = set(fit_cids)
    excl_set = set(excluded_cids)
    if not fit_set or (fit_set & excl_set):
        raise ValueError("fit_cids must be non-empty and disjoint from excluded_cids")
    ordered = sorted(fit_set)
    tok_stack = torch.stack([tokens[c].detach().double() for c in ordered], dim=0)
    mask_stack = torch.stack([masks[c].detach().bool() for c in ordered], dim=0)
    if tok_stack.ndim != 3 or tok_stack.shape[1:] != (10, 32):
        raise ValueError("expected (N, 10, 32) fit token tensor")
    if mask_stack.shape != (len(ordered), 10):
        raise ValueError("expected (N, 10) fit modality mask tensor")
    slot_present = mask_stack.any(dim=0).unsqueeze(-1)
    std = torch.std(tok_stack, dim=0, correction=0) * slot_present.double()
    if not torch.isfinite(std).all() or int((std > 0).sum()) == 0:
        raise ValueError(
            "fit-fold feature std must be finite with positive active features"
        )
    return std.float()


def build_fixed_context_noise(fit_cids, masks, fit_std, scale, seed_fn):
    """Pre-draw one fixed Gaussian perturbation N(0, (scale * s_j)^2) per fit context for IN10_FIXED."""
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("positive finite scale required for fixed context noise")
    sigma = fit_std.detach().float() * float(scale)
    table = {}
    for cid in sorted(set(fit_cids)):
        gen = torch.Generator(device="cpu").manual_seed(int(seed_fn(cid)))
        raw = torch.randn(sigma.shape, dtype=torch.float32, device="cpu", generator=gen)
        m = masks[cid].detach().bool().unsqueeze(-1).float()
        table[cid] = (raw * sigma * m).detach()
    return table


class InputFeatureNoise(torch.nn.Module):
    """Parameter-free, buffer-free input-space Gaussian noise site for adapter_10tok tokens."""

    def __init__(self):
        super().__init__()
        self.reset_stats()

    def reset_stats(self):
        self.changed = 0
        self.active_inputs = 0
        self.noise_l2 = 0.0
        self.expected_noise_l2 = 0.0

    def forward(
        self,
        tokens,
        mask,
        cids=None,
        fit_std=None,
        scale=0.0,
        seed=None,
        fixed_table=None,
    ):
        self.reset_stats()
        if not self.training or scale == 0.0:
            return tokens
        if (
            fit_std is None
            or scale not in (0.05, 0.10)
            or tokens.shape[1:] != fit_std.shape
        ):
            raise ValueError(
                "valid fit_std and scale in {0.05, 0.10} required in training mode"
            )
        gate = mask.detach().bool().unsqueeze(-1)
        sigma = fit_std.detach().float() * float(scale)
        b_sigma = sigma.unsqueeze(0) * gate.float()
        if fixed_table is not None:
            if cids is None or len(cids) != tokens.shape[0]:
                raise ValueError("cids required for fixed per-context noise lookup")
            noise = torch.stack([fixed_table[c] for c in cids], dim=0) * gate.float()
        else:
            if seed is None:
                raise ValueError(
                    "deterministic generator seed required for dynamic noise"
                )
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            raw = torch.randn(
                tokens.shape, dtype=tokens.dtype, device="cpu", generator=gen
            )
            noise = raw * b_sigma
        result = tokens + noise
        active_sigma = b_sigma.detach() > 0
        self.changed = int((result.detach() != tokens.detach()).sum())
        self.active_inputs = int(active_sigma.sum())
        self.noise_l2 = float(torch.linalg.vector_norm(noise.detach().double()))
        self.expected_noise_l2 = float(
            torch.linalg.vector_norm(b_sigma.detach().double())
        )
        return result


def grad_norms(params, max_norm=1.0):
    """Clip gradients and compute pre/post-clip L2 norms while safely handling p.grad is None (07:26Z addendum)."""
    plist = list(params)
    active = [p.grad.detach().double().reshape(-1) for p in plist if p.grad is not None]
    none_count = sum(1 for p in plist if p.grad is None)
    if not active:
        raise RuntimeError("all parameter gradients are None")
    raw_norm = float(
        torch.nn.utils.clip_grad_norm_(plist, max_norm, error_if_nonfinite=True)
    )
    post = [p.grad.detach().double().reshape(-1) for p in plist if p.grad is not None]
    clipped_norm = float(torch.linalg.vector_norm(torch.cat(post)))
    if not math.isfinite(raw_norm) or not math.isfinite(clipped_norm):
        raise RuntimeError("nonfinite gradient norm")
    return dict(
        raw_grad_l2=raw_norm,
        clipped_grad_l2=clipped_norm,
        active_grad_tensors=len(post),
        none_grad_tensors=none_count,
    )


def adapter_l2_drift(state_a, state_b, names):
    """Compute float64 L2 norm of parameter difference across the 47 trainable adapter tensors."""
    diffs = [
        (state_a[n].detach().double() - state_b[n].detach().double()).reshape(-1)
        for n in names
    ]
    return float(torch.linalg.vector_norm(torch.cat(diffs)))


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
    cids=None,
    fit_std=None,
    noise_scale=0.0,
    noise_seed=None,
    fixed_table=None,
):
    import torch.nn.functional as F

    ad = net.adapter_10tok
    noisy_tokens = site(
        tokens,
        mask,
        cids=cids,
        fit_std=fit_std,
        scale=noise_scale,
        seed=noise_seed,
        fixed_table=fixed_table,
    )
    pooled, query, semantic, adjustment = ad.encode_multimodal_context_and_candidates(
        mm_tokens_10x32=noisy_tokens,
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
