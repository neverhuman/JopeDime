"""H235 loss arms (CTRL vs heteroscedastic beta-NLL with a zero-initialised log-variance head) for the r882 adapter
continuation; every arm uses the exact merged H170 CTRL AdamW recipe. Also None-safe diagnostics."""

import math
import torch
import torch.nn.functional as F

ARMS = ("CONTROL", "BNLL_MIX", "BNLL_ADD")
HEAD_ARMS = ("BNLL_MIX", "BNLL_ADD")
CONTROL_LR, CONTROL_WD = (
    1e-4,
    1e-4,
)  # merged H170 CTRL: AdamW lr=1e-4, wd=1e-4, betas (0.9, 0.999)
BETA = 0.5  # Seitzer et al. 2022 beta-NLL
LOGVAR_CLAMP = 6.0
ADD_WEIGHT = 0.5  # BNLL_ADD = CTRL + 0.5 * beta-NLL


def make_optimizer(arm, params):
    """Identical AdamW for every arm; `params` are the adapter tensors (+ the variance head for HEAD_ARMS)."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return torch.optim.AdamW(
        params, lr=CONTROL_LR, betas=(0.9, 0.999), weight_decay=CONTROL_WD
    )


def settings(arm, step):
    """The applied (lr, beta1, beta2, weight_decay) of an arm at update 1..40 (constant, identical across arms)."""
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm or update")
    return CONTROL_LR, 0.9, 0.999, CONTROL_WD


def make_head(dim):
    """ONE linear log-variance head on the adapter's per-candidate query embedding; weight 0 and bias 0, so
    log-variance is 0 (sigma^2 = 1) at update 0. Scoring never uses it."""
    head = torch.nn.Linear(dim, 1)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    return head


def log_variance(head, hidden):
    return head(hidden).squeeze(-1).clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)


def ctrl_target(fitness, gate):
    """The CTRL loss's cost target (H140 control's H59 targets): clipped fitness, failed gates => cost 2."""
    return torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))


def ctrl_regression(pred, fitness, gate):
    """The CTRL loss's regression term: smooth-L1 (Huber, delta 1) to the cost target, mean over candidates."""
    return F.smooth_l1_loss(pred, ctrl_target(fitness, gate), reduction="none").mean(1)


def ctrl_listwise(pred, fitness, gate):
    """The CTRL loss's other term, unchanged: 0.1 * KL(softmax(-target/0.2) || softmax(-pred/0.2)) per context."""
    target = ctrl_target(fitness, gate)
    return 0.1 * F.kl_div(
        F.log_softmax(-pred / 0.2, dim=-1),
        F.softmax(-target / 0.2, dim=-1),
        reduction="none",
    ).sum(1)


def beta_nll(pred, logvar, fitness, gate, beta=BETA):
    """beta-NLL (Seitzer et al. 2022): stopgrad(sigma^(2 beta)) * (0.5 log sigma^2 + (y - mu)^2 / (2 sigma^2)),
    per candidate, mean over candidates; returns the per-context vector (like the CTRL terms)."""
    target = ctrl_target(fitness, gate)
    var = torch.exp(logvar)
    nll = 0.5 * (logvar + (target - pred).square() / var)
    return (var.detach().pow(beta) * nll).mean(1)


def arm_loss(arm, ctrl_loss, pred, fitness, gate, logvar=None):
    """CONTROL: the inherited H146 CTRL loss. BNLL_MIX: the CTRL regression term replaced by beta-NLL, the listwise
    KL term unchanged. BNLL_ADD: CTRL + 0.5 * beta-NLL."""
    if arm == "CONTROL":
        return ctrl_loss(pred, fitness, gate)
    if logvar is None:
        raise ValueError("beta-NLL arms need the log-variance head")
    if arm == "BNLL_MIX":
        return (
            beta_nll(pred, logvar, fitness, gate) + ctrl_listwise(pred, fitness, gate)
        ).mean()
    if arm == "BNLL_ADD":
        return (
            ctrl_loss(pred, fitness, gate)
            + ADD_WEIGHT * beta_nll(pred, logvar, fitness, gate).mean()
        )
    raise ValueError("unknown arm")


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
    return_hidden=False,
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
    pred = (anchor_total + delta).float()
    # H235: the per-candidate query embedding feeds the log-variance head; the mean prediction is unchanged.
    return (pred, query) if return_hidden else pred


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
