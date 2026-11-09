"""H256 gradient-filter arms for the r882 adapter continuation: exact H170 CTRL AdamW (CONTROL), CTRL AdamW with
Gradient Centralization (Yong et al. 2020) followed by unit-wise Adaptive Gradient Clipping (Brock et al. 2021, NFNet)
(GC_AGC), and CTRL AdamW with AGC only (AGC_ONLY, the mechanism control). The filters act on the raw gradients after
backward and BEFORE the inherited global clip 1.0. Every arm uses the exact inherited H170 CTRL loss, batches, data order
and 40 updates. All constants are frozen here before any outcome read (hw_t1335)."""

import math
import torch

ARMS = ("CONTROL", "GC_AGC", "AGC_ONLY")
CONTROL_LR, CONTROL_WD = (
    1e-4,
    1e-4,
)  # merged H170 CTRL: AdamW lr=1e-4, wd=1e-4, betas (0.9, 0.999)
BETAS = (0.9, 0.999)
AGC_LAMBDA, AGC_EPS = (
    0.01,
    1e-3,
)  # hw_t1335: clip ratio lambda 0.01 fixed; NFNet eps 1e-3
FILTERS = {
    "CONTROL": (False, False),
    "GC_AGC": (True, True),
    "AGC_ONLY": (False, True),
}  # (GC, AGC)


def unit_rows(t):
    """View a tensor as its units: dim 0 is the output unit for >= 2-D tensors (all other dims flattened); a 0-D or
    1-D tensor is one unit (NFNet unitwise_norm convention)."""
    if not t.is_contiguous():
        raise RuntimeError("non-contiguous gradient or parameter")
    return t.reshape(t.shape[0], -1) if t.dim() >= 2 else t.reshape(1, -1)


@torch.no_grad()
def centralize(grad):
    """Gradient Centralization: subtract each output row's mean over its remaining dims (>= 2-D tensors only).
    Returns True when applied."""
    if grad.dim() < 2:
        return False
    rows = grad.view(grad.shape[0], -1)
    rows.sub_(rows.mean(dim=1, keepdim=True))
    return True


@torch.no_grad()
def adaptive_clip(param, grad, lamb=AGC_LAMBDA, eps=AGC_EPS):
    """Unit-wise AGC: for each unit i with ||G_i|| > lamb * max(||W_i||, eps), rescale G_i to that bound.
    Returns (clipped units, total units)."""
    g = unit_rows(grad)
    w = unit_rows(param.detach())
    g_norm = torch.linalg.vector_norm(g, dim=1, keepdim=True)
    bound = lamb * torch.clamp(
        torch.linalg.vector_norm(w, dim=1, keepdim=True), min=eps
    )
    clip = g_norm > bound
    scale = torch.where(
        clip, bound / torch.clamp(g_norm, min=1e-6), torch.ones_like(g_norm)
    )
    g.mul_(scale)
    return int(clip.sum()), int(g.shape[0])


class FilteredAdamW(torch.optim.AdamW):
    """The exact H170 CTRL AdamW plus an optional GC then AGC gradient filter. The runner calls filter_gradients()
    after backward and before the inherited global clip 1.0. None gradients are skipped. CONTROL has no filter."""

    def __init__(self, params, gc=False, agc=False):
        super().__init__(params, lr=CONTROL_LR, betas=BETAS, weight_decay=CONTROL_WD)
        self.gc, self.agc = gc, agc

    @torch.no_grad()
    def filter_gradients(self):
        stats = dict(gc_tensors=0, agc_clipped_units=0, agc_units=0)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if self.gc:
                    stats["gc_tensors"] += centralize(p.grad)
                if self.agc:
                    clipped, units = adaptive_clip(p, p.grad)
                    stats["agc_clipped_units"] += clipped
                    stats["agc_units"] += units
        return stats


def make_optimizer(arm, params):
    """CONTROL: exact H170 CTRL AdamW. GC_AGC: CTRL AdamW with GC then AGC. AGC_ONLY: CTRL AdamW with AGC."""
    if arm not in FILTERS:
        raise ValueError("unknown arm")
    gc, agc = FILTERS[arm]
    return FilteredAdamW(list(params), gc=gc, agc=agc)


def settings(arm, step):
    """The applied update settings at update 1..40 (constant)."""
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm or update")
    gc, agc = FILTERS[arm]
    return dict(
        optimizer="AdamW",
        lr=CONTROL_LR,
        beta1=BETAS[0],
        beta2=BETAS[1],
        weight_decay=CONTROL_WD,
        gradient_centralization=gc,
        agc_lambda=AGC_LAMBDA if agc else None,
        agc_eps=AGC_EPS if agc else None,
        filter_order="GC, then AGC, then global clip 1.0"
        if gc
        else ("AGC, then global clip 1.0" if agc else "global clip 1.0"),
    )


def arm_loss(arm, ctrl_loss, pred, fitness, gate):
    """H256 changes only the gradient filter: every arm uses the inherited H146 CTRL loss."""
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
        or launch.get("publication_ref") != "v1/next"
        or not launch.get("published_ancestor_verified")
        or len(str(launch.get("review_receipt_sha256", ""))) != 64
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
    ):
        raise RuntimeError("external merged/reviewed launch binding")
