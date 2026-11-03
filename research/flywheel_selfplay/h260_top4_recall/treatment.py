"""H260 recall@4 objectives for the r882 adapter continuation. Every arm uses the exact H170 CTRL AdamW, batches, data
order and 40 updates; only the loss differs:
- CONTROL: the inherited H146/H170 CTRL loss L_H170.
- T4_HINGE (primary): L_H170 + 0.5 * mean_ctx relu(c(oracle_best) - c(predicted rank-4) + m), with m = 0.05 x the std
  of r882's predicted costs over all FIT-fold candidates of the cell (LambdaLoss@4-style boundary hinge).
- T4_PL (secondary): L_H170 + 0.5 * mean_ctx -log P_PL(oracle_best within the first 4 draws without replacement from
  softmax(-c / tau)), tau = 1, computed exactly.
The oracle best of a minibatch context is argmin over its 24 candidates of the gated cost (fitness if gate else 2.0),
first index on ties; labels come only from FIT-fold training contexts. Constants frozen before any outcome (hw_t1530)."""

from itertools import permutations
import math
import torch

ARMS = ("CONTROL", "T4_HINGE", "T4_PL")
CONTROL_LR, CONTROL_WD = (
    1e-4,
    1e-4,
)  # merged H170 CTRL: AdamW lr=1e-4, wd=1e-4, betas (0.9, 0.999)
BETAS = (0.9, 0.999)
AUX_WEIGHT = 0.5
MARGIN_FRACTION = (
    0.05  # m = 0.05 x std of r882 predicted cost over the cell's FIT-fold candidates
)
PL_TAU = 1.0
TOP_K = 4
_PERMS = {}


class ControlAdamW(torch.optim.AdamW):
    """The exact H170 CTRL AdamW, with a no-op gradient filter (shared by every H260 arm)."""

    def __init__(self, params):
        super().__init__(params, lr=CONTROL_LR, betas=BETAS, weight_decay=CONTROL_WD)

    def filter_gradients(self):
        return {}


def make_optimizer(arm, params):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return ControlAdamW(list(params))


def settings(arm, step):
    """The applied settings at update 1..40 (constant)."""
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm or update")
    return dict(
        optimizer="AdamW",
        lr=CONTROL_LR,
        beta1=BETAS[0],
        beta2=BETAS[1],
        weight_decay=CONTROL_WD,
        aux_loss={
            "CONTROL": None,
            "T4_HINGE": "hinge@4",
            "T4_PL": "plackett_luce_top4",
        }[arm],
        aux_weight=None if arm == "CONTROL" else AUX_WEIGHT,
        margin_fraction=MARGIN_FRACTION if arm == "T4_HINGE" else None,
        pl_tau=PL_TAU if arm == "T4_PL" else None,
    )


def oracle_best(fitness, gate):
    """Per-context index of the best candidate by gated cost (fitness if gate else 2.0); first index on ties."""
    gated = torch.where(gate, fitness, torch.full_like(fitness, 2.0))
    return torch.argmin(gated, dim=1)


def hinge_at4(pred, best, margin):
    """mean_ctx relu(c[best] - c[rank-4] + margin); rank-4 is the 4th-lowest predicted cost (stable, detached)."""
    rank4 = torch.argsort(pred.detach(), dim=1, stable=True)[:, TOP_K - 1]
    c_best = pred.gather(1, best[:, None]).squeeze(1)
    c_rank4 = pred.gather(1, rank4[:, None]).squeeze(1)
    return torch.relu(c_best - c_rank4 + margin).mean()


def _perms(n_other, length):
    key = (n_other, length)
    if key not in _PERMS:
        _PERMS[key] = torch.tensor(
            list(permutations(range(n_other), length)), dtype=torch.long
        ).reshape(-1, length)
    return _PERMS[key]


def pl_topk_prob(pred, best, tau=PL_TAU, k=TOP_K):
    """Exact Plackett-Luce probability that `best` is among the first k draws without replacement from
    softmax(-pred / tau): the sum over every ordered prefix of 0..k-1 other candidates. Float64."""
    w = torch.softmax(-pred.double() / tau, dim=1)
    b, n = w.shape
    keep = torch.ones_like(w, dtype=torch.bool)
    keep[torch.arange(b), best] = False
    others = torch.arange(n).expand(b, n)[keep].reshape(b, n - 1)
    w_o = w.gather(1, others)
    w_i = w.gather(1, best[:, None]).squeeze(1)
    total = w_i.clone()
    for length in range(1, k):
        idx = _perms(n - 1, length)
        seq = w_o[:, idx]  # (b, P, length)
        prob = torch.ones(seq.shape[:2], dtype=w.dtype)
        used = torch.zeros(seq.shape[:2], dtype=w.dtype)
        for t in range(length):
            prob = prob * seq[:, :, t] / torch.clamp(1 - used, min=1e-300)
            used = used + seq[:, :, t]
        total = total + (prob * w_i[:, None] / torch.clamp(1 - used, min=1e-300)).sum(
            dim=1
        )
    return total


def pl_top4_loss(pred, best, tau=PL_TAU):
    return (
        -torch.log(torch.clamp(pl_topk_prob(pred, best, tau), min=1e-300))
        .mean()
        .float()
    )


def arm_loss(arm, ctrl_loss, pred, fitness, gate, margin=None):
    """Returns (total objective, auxiliary term or None). CONTROL is exactly L_H170."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    base = ctrl_loss(pred, fitness, gate)
    if arm == "CONTROL":
        return base, None
    best = oracle_best(fitness, gate)
    if arm == "T4_HINGE":
        if margin is None or not margin > 0:
            raise ValueError("positive frozen margin required")
        aux = hinge_at4(pred, best, margin)
    else:
        aux = pl_top4_loss(pred, best)
    return base + AUX_WEIGHT * aux, aux


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
