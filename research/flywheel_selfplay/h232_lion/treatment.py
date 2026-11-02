"""H232 optimizer arms (AdamW CONTROL vs Lion) for the r882 adapter continuation, plus None-safe diagnostics."""

import math
import torch

ARMS = ("CONTROL", "LION", "LION_LR3")
CONTROL_LR, CONTROL_WD = (
    1e-4,
    1e-4,
)  # merged H170 CTRL: AdamW lr=1e-4, wd=1e-4, betas (0.9, 0.999)
SETTINGS = {
    "CONTROL": dict(
        kind="adamw", lr=CONTROL_LR, betas=(0.9, 0.999), weight_decay=CONTROL_WD
    ),
    # Lion (Chen et al. 2023): beta1=0.9, beta2=0.99, lr = CONTROL_lr/10, decoupled wd = 3 x CONTROL_wd.
    "LION": dict(
        kind="lion", lr=CONTROL_LR / 10, betas=(0.9, 0.99), weight_decay=3 * CONTROL_WD
    ),
    "LION_LR3": dict(
        kind="lion", lr=CONTROL_LR / 3, betas=(0.9, 0.99), weight_decay=3 * CONTROL_WD
    ),
}


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023, Algorithm 2) with decoupled weight decay. Per parameter with a gradient:
    p <- p * (1 - lr * wd); c = beta1 * m + (1 - beta1) * g; p <- p - lr * sign(c); m <- beta2 * m + (1 - beta2) * g.
    Parameters whose gradient is None are skipped entirely (no decay, no state), matching native AdamW."""

    def __init__(self, params, lr, betas=(0.9, 0.99), weight_decay=0.0):
        if lr <= 0 or not (0 <= betas[0] < 1 and 0 <= betas[1] < 1) or weight_decay < 0:
            raise ValueError("invalid Lion hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("H232 Lion takes no closure")
        for group in self.param_groups:
            lr, (beta1, beta2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                p.mul_(1 - lr * wd)
                update = (m * beta1 + p.grad * (1 - beta1)).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)


def make_optimizer(arm, params):
    """Constant settings for all 40 updates (CONTROL's schedule is constant; 'same schedule')."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    s = SETTINGS[arm]
    cls = torch.optim.AdamW if s["kind"] == "adamw" else Lion
    return cls(params, lr=s["lr"], betas=s["betas"], weight_decay=s["weight_decay"])


def settings(arm, step):
    """The applied (lr, beta1, beta2, weight_decay) of an arm at update 1..40."""
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm or update")
    s = SETTINGS[arm]
    return s["lr"], s["betas"][0], s["betas"][1], s["weight_decay"]


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
