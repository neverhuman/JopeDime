"""H237 optimizer arms (AdamW CONTROL vs AdaBelief vs RAdam) for the r882 adapter continuation; every arm uses the
exact inherited H170 CTRL loss, batches and lr. Also None-safe diagnostics."""

import math
import torch

ARMS = ("CONTROL", "ABEL", "RADAM")
CONTROL_LR, CONTROL_WD = (
    1e-4,
    1e-4,
)  # merged H170 CTRL: AdamW lr=1e-4, wd=1e-4, betas (0.9, 0.999)
BETAS = (0.9, 0.999)
ABEL_EPS = 1e-16  # hw_t0030: eps added inside s_t (and to its square root)


class AdaBelief(torch.optim.Optimizer):
    """AdaBelief (Zhuang et al. 2020, Algorithm 2) with decoupled weight decay, bias correction, no rectification.
    Per parameter with a gradient, at step t:
      p <- p * (1 - lr * wd)
      m <- b1 m + (1 - b1) g
      s <- b2 s + (1 - b2) (g - m)^2 + eps
      p <- p - lr * (m / (1 - b1^t)) / (sqrt(s / (1 - b2^t)) + eps)
    Parameters whose gradient is None are skipped entirely (no decay, no state, no step count), as in native AdamW."""

    def __init__(self, params, lr, betas=BETAS, eps=ABEL_EPS, weight_decay=0.0):
        if (
            lr <= 0
            or not (0 <= betas[0] < 1 and 0 <= betas[1] < 1)
            or eps <= 0
            or weight_decay < 0
        ):
            raise ValueError("invalid AdaBelief hyperparameters")
        super().__init__(
            params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("H237 AdaBelief takes no closure")
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = (
                group["lr"],
                group["betas"],
                group["eps"],
                group["weight_decay"],
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_var"] = torch.zeros_like(p)
                state["step"] += 1
                t, m, s = state["step"], state["exp_avg"], state["exp_avg_var"]
                p.mul_(1 - lr * wd)
                m.mul_(b1).add_(g, alpha=1 - b1)
                s.mul_(b2).addcmul_(g - m, g - m, value=1 - b2).add_(eps)
                denom = (s / (1 - b2**t)).sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-lr / (1 - b1**t))


def make_optimizer(arm, params):
    """CONTROL: AdamW (exact H170 CTRL). ABEL: AdaBelief, same lr/betas/wd. RADAM: torch RAdam (Liu et al. 2020),
    same lr/betas, decoupled wd = CTRL wd. All settings constant over the 40 updates."""
    if arm == "CONTROL":
        return torch.optim.AdamW(
            params, lr=CONTROL_LR, betas=BETAS, weight_decay=CONTROL_WD
        )
    if arm == "ABEL":
        return AdaBelief(
            params, lr=CONTROL_LR, betas=BETAS, eps=ABEL_EPS, weight_decay=CONTROL_WD
        )
    if arm == "RADAM":
        return torch.optim.RAdam(
            params,
            lr=CONTROL_LR,
            betas=BETAS,
            weight_decay=CONTROL_WD,
            decoupled_weight_decay=True,
        )
    raise ValueError("unknown arm")


def settings(arm, step):
    """The applied (lr, beta1, beta2, weight_decay) of an arm at update 1..40 (constant, identical across arms)."""
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm or update")
    return CONTROL_LR, BETAS[0], BETAS[1], CONTROL_WD


def arm_loss(arm, ctrl_loss, pred, fitness, gate):
    """H237 changes only the optimizer: every arm uses the inherited H146 CTRL loss."""
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


def validate_launch(
    launch, head, manifest_sha256, prereg_sha256, fold_sha256, smoke=False
):
    if launch.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("launch manifest binding")
    if smoke:
        return
    if (
        not launch.get("merged_source_verified")
        or launch.get("source_commit") != head
        or type(launch.get("review_id")) is not int
        or launch["review_id"] <= 0
        or type(launch.get("prereg_pr")) is not int
        or launch["prereg_pr"] <= 0
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
    ):
        raise RuntimeError("external merged/reviewed launch binding")
