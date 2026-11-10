"""H264 frozen-r882 functions: the exact cached-anchor forward (bit-exact to native r882 Native91) and the embedding
extraction used as kNN keys (the same encode_multimodal_context_and_candidates call, on the full candidate pool). No
parameter is trained; there is no optimizer (hw_t1658)."""

import torch


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


def adapter_embed(net, prog, sos, tokens, mask):
    """Frozen r882 adapter context embedding [B, 32] and candidate query embedding [B, M, 32], computed on the full
    candidate pool exactly as in the forward (cand_q centres sem24 over the pool, so the pool must be complete)."""
    with torch.no_grad():
        pooled, query, _, _ = (
            net.adapter_10tok.encode_multimodal_context_and_candidates(
                mm_tokens_10x32=tokens,
                mm_mask_10=mask,
                prog_412=prog,
                sos_prior=sos,
                active_modality_override_mask=None,
            )
        )
    return pooled.float(), query.float()


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
