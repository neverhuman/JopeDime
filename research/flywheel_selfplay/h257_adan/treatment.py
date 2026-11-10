"""H257 fixed Adan update, official single-tensor proximal variant; exact H170 CTRL loss."""

import math
import torch

ARMS = ("CONTROL", "ADAN", "ADAN_HALF")
CONTROL_LR, CONTROL_WD, BETAS = 1e-4, 1e-4, (0.9, 0.999)
ADAN_BETAS, ADAN_WD, ADAN_EPS = (0.98, 0.92, 0.99), 0.02, 1e-8


class Adan(torch.optim.Optimizer):
    """Bias-corrected moments of g, delta-g and (g + beta2*delta-g)^2.

    External CTRL global clip precedes this update. First delta is zero.
    Proximal decoupled decay divides the updated parameter by 1+lr*wd.
    No internal clipping, foreach, fused kernel, restart, or schedule.
    """

    def __init__(self, params, lr):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("positive finite learning rate required")
        super().__init__(
            params, dict(lr=lr, betas=ADAN_BETAS, weight_decay=ADAN_WD, eps=ADAN_EPS)
        )

    def filter_gradients(self):
        return None

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("H257 uses one gradient evaluation, no closure")
        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            t = group["step"]
            b1, b2, b3 = group["betas"]
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse or not torch.isfinite(g).all():
                    raise RuntimeError("finite dense gradient required")
                state = self.state[p]
                if not state:
                    state.update(
                        m=torch.zeros_like(p),
                        d=torch.zeros_like(p),
                        v=torch.zeros_like(p),
                        previous=g.clone(),
                    )
                delta = g - state["previous"]
                state["m"].mul_(b1).add_(g, alpha=1 - b1)
                state["d"].mul_(b2).add_(delta, alpha=1 - b2)
                corrected = delta.mul(b2).add_(g)
                state["v"].mul_(b3).addcmul_(corrected, corrected, value=1 - b3)
                denom = (state["v"].sqrt() / math.sqrt(1 - b3**t)).add_(eps)
                p.addcdiv_(state["m"], denom, value=-lr / (1 - b1**t))
                p.addcdiv_(state["d"], denom, value=-lr * b2 / (1 - b2**t))
                p.div_(1 + lr * wd)
                state["previous"].copy_(g)


def make_optimizer(arm, params):
    if arm == "CONTROL":
        return ControlAdamW(params)
    if arm in ("ADAN", "ADAN_HALF"):
        return Adan(params, lr=CONTROL_LR if arm == "ADAN" else CONTROL_LR * 0.5)
    raise ValueError("unknown arm")


def settings(arm, step):
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm or step")
    if arm == "CONTROL":
        return CONTROL_LR, *BETAS, CONTROL_WD, None, 1e-8
    return (
        CONTROL_LR * (0.5 if arm == "ADAN_HALF" else 1),
        *ADAN_BETAS[:2],
        ADAN_WD,
        ADAN_BETAS[2],
        ADAN_EPS,
    )


def validate_launch(
    launch, head, manifest_sha256, prereg_sha256, fold_sha256, smoke=False
):
    if smoke:
        raise RuntimeError("No pre-publication model smoke in H257")
    if (
        not launch.get("published_source_verified")
        or launch.get("publication_branch") != "v1/next"
        or launch.get("source_commit") != head
        or launch.get("reviewed_source") != head
        or launch.get("review_status") != "PASS"
        or launch.get("manifest_sha256") != manifest_sha256
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
    ):
        raise RuntimeError("published prereg/review binding")


class ControlAdamW(torch.optim.AdamW):
    """The exact H170 CTRL AdamW, with a no-op gradient filter."""

    def __init__(self, params):
        super().__init__(params, lr=CONTROL_LR, betas=BETAS, weight_decay=CONTROL_WD)

    def filter_gradients(self):
        return None


def arm_loss(arm, ctrl_loss, pred, fitness, gate):
    """H257 changes only the update geometry: every arm uses the inherited H146 CTRL loss."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return ctrl_loss(pred, fitness, gate)


def gradient_norm(params):
    """None contributes zero to diagnostics; native AdamW None semantics are retained."""
    grads = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(grads))) if grads else 0.0


def adapter_drift(initial, final):
    """All 29,211 coordinates, including inactive/unchanged tensors; float64 L2."""
    if set(initial) != set(final):
        raise ValueError("adapter state inventory differs")
    return math.sqrt(
        math.fsum(
            float((final[n].double() - initial[n].double()).square().sum())
            for n in initial
        )
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
):
    import torch.nn.functional as F

    ad = net.adapter_10tok
    pooled, query, semantic, adjustment = ad.encode_multimodal_context_and_candidates(
        mm_tokens_10x32=tokens,
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
