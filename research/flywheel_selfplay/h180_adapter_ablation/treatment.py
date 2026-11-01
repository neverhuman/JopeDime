"""H180 inherits the H176 objective/forward verbatim and adds the prespecified ZERO_ADAPTER (anchor-only) scorer."""

import math

import torch


def adapter_forward(
    net, prog, sos, tokens, mask, scale, rev, raw, gate, anchor_total, anchor_prior
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


def gradient_norm(params):
    grads = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(grads))) if grads else 0.0


def adapter_drift(initial, final):
    if set(initial) != set(final):
        raise ValueError("adapter state inventory differs")
    return math.sqrt(
        math.fsum(
            float((final[n].double() - initial[n].double()).square().sum())
            for n in initial
        )
    )


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
        or not (
            (type(launch.get("review_id")) is int and launch["review_id"] > 0)
            or (
                isinstance(launch.get("review_waiver"), str)
                and len(launch["review_waiver"].strip()) >= 20
            )
        )
        or type(launch.get("prereg_pr")) is not int
        or launch["prereg_pr"] <= 0
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
    ):
        raise RuntimeError("external merged/reviewed launch binding")


def zero_adapter_scores(anchor_total):
    """ZERO_ADAPTER: the adapter contribution removed. The live FlywheelBlockFormerAdapterNet.predict_costs returns
    total_cost = anchor_total + adapter_total_delta, where anchor_total is anchor.predict_costs(...) with prior
    correction, struct kernel and cross-attention on and UCB off; dropping adapter_total_delta leaves anchor_total."""
    return anchor_total.detach().float().clone()


def assert_anchor_decomposition(native, anchor_total):
    """Proof that the anchor-only scorer is what the live code produces with the adapter delta set to zero: the live
    total must equal the separately computed anchor_total plus the live adapter_total_delta, bit for bit."""
    import torch

    if "adapter_total_delta" not in native:
        raise RuntimeError(
            "live scorer did not expose adapter_total_delta (revealed history required)"
        )
    if not torch.equal(
        native["total_cost"].float(),
        anchor_total.float() + native["adapter_total_delta"].float(),
    ):
        raise RuntimeError("live total_cost != anchor_total + adapter_total_delta")
    return True
