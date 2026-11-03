"""H248 expected-regret softmin loss; all arms retain exact H170 CTRL AdamW."""

import math
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

ARMS = ("CONTROL", "DFL_MIX", "DFL")
CONTROL_LR, CONTROL_WD, BETAS = 1e-4, 1e-4, (0.9, 0.999)


class ControlAdamW(torch.optim.AdamW):
    def __init__(self, params):
        super().__init__(params, lr=CONTROL_LR, betas=BETAS, weight_decay=CONTROL_WD)

    def filter_gradients(self):
        return None


def make_optimizer(arm, params):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return ControlAdamW(list(params))


def settings(arm, step):
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm/update")
    return CONTROL_LR, BETAS[0], BETAS[1], CONTROL_WD


def family_weights(rows, cids):
    """Unbiased equal-family DFL objective under inherited uniform-context minibatches."""
    families = {r["context_sha256"]: r["family_id"] for r in rows}
    counts = Counter(families.values())
    return torch.tensor(
        [len(rows) / (len(counts) * counts[families[c]]) for c in cids],
        dtype=torch.float32,
    )


def failure_ceiling(fitness, gate):
    """Native invalid utility -2 corresponds to cost +2; replace invalid cost by worst valid cost +1."""
    fitness, gate = np.asarray(fitness), np.asarray(gate, dtype=bool)
    if not gate.any() or not np.isfinite(fitness[gate]).all():
        raise ValueError("finite non-failure label required")
    return float(fitness[gate].max()) + 1.0


def regret_targets(fitness, gate, ceilings):
    costs = torch.where(gate, fitness, ceilings[:, None].expand_as(fitness)).detach()
    if not torch.isfinite(costs).all():
        raise ValueError("nonfinite DFL costs")
    return costs - costs.min(dim=1, keepdim=True).values


def expected_regret(pred, fitness, gate, temperature, weights, ceilings):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive frozen temperature required")
    regret = regret_targets(fitness, gate, ceilings)
    per_context = (F.softmax(-pred / temperature, dim=1) * regret).sum(dim=1)
    return (per_context * weights.to(pred)).mean()


def ctrl_kl(pred, fitness, gate):
    target = torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))
    return (
        0.1
        * F.kl_div(
            F.log_softmax(-pred / 0.2, dim=-1),
            F.softmax(-target / 0.2, dim=-1),
            reduction="none",
        )
        .sum(1)
        .mean()
    )


def arm_loss(arm, ctrl_loss, pred, fitness, gate, temperature, weights, ceilings):
    if arm == "CONTROL":
        return ctrl_loss(pred, fitness, gate)
    dfl = expected_regret(pred, fitness, gate, temperature, weights, ceilings)
    if arm == "DFL_MIX":
        return ctrl_loss(pred, fitness, gate) + 0.5 * dfl
    if arm == "DFL":
        return dfl + ctrl_kl(pred, fitness, gate)
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
