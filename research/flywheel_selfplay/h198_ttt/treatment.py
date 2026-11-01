"""H198 per-context test-time training (TTT) primitives on a fresh copy of the r882 adapter.

Only the 4 revealed candidates of the scored context enter the TTT loss; unrevealed candidate labels are never passed to
TTT or read before `argmin(cost)`. After every context, the 47 adapter tensors (29,211 parameters) are reset to r882 and
verified bit-exact by SHA-256 tensor hash.
"""

import hashlib
import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F

K_GRID = (1, 3, 5)
LR_GRID = (1e-5, 3e-5, 1e-4)
GRID = tuple((k, lr) for k in K_GRID for lr in LR_GRID)
GRID_LABELS = tuple(f"K{k}_LR{lr:.0e}" for k, lr in GRID)
L2_BETA = 10.0

DERANGEMENTS_4 = tuple(
    p
    for p in itertools.permutations(range(4))
    if all(p[i] != i for i in range(4))
)


def stream_seed(*parts):
    digest = hashlib.sha256(":".join(map(str, ("H198", *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def shuf_permutation(seed, cid):
    """Deterministic fixed-point-free permutation (derangement) of the 4 revealed targets for (seed, cid)."""
    idx = stream_seed("TTT_SHUF", int(seed), str(cid)) % len(DERANGEMENTS_4)
    return DERANGEMENTS_4[idx]


def adapter_forward(
    net, prog, sos, tokens, mask, scale, rev, raw, gate, anchor_total, anchor_prior
):
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
    grads = [p.grad.detach().reshape(-1).double() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    val = float(torch.linalg.vector_norm(torch.cat(grads)).cpu())
    if not math.isfinite(val):
        raise RuntimeError("nonfinite gradient norm")
    return val


def adapter_drift(initial, current):
    if set(initial) != set(current):
        raise ValueError("adapter state inventory differs")
    return math.sqrt(
        math.fsum(
            float((current[n].detach().double() - initial[n].detach().double()).square().sum())
            for n in initial
        )
    )


def revealed_live_loss(scores, revealed_indices, revealed_raw_f, revealed_gate, loss_fn):
    """Live H146/H140 loss restricted strictly to the 4 revealed candidates of the single scored context."""
    if (
        scores.ndim != 2
        or scores.shape[0] != 1
        or revealed_indices.shape != (1, 4)
        or revealed_raw_f.shape != (1, 4)
        or revealed_gate.shape != (1, 4)
    ):
        raise ValueError("single-context 4-revealed-row tensors required")
    if not torch.equal(revealed_indices, torch.arange(4, device=revealed_indices.device).reshape(1, 4)):
        raise ValueError("revealed_indices must be [0, 1, 2, 3]")
    rev_scores = torch.gather(scores, dim=1, index=revealed_indices)
    return loss_fn(rev_scores, revealed_raw_f.float(), revealed_gate > 0.5)


def ttt_context_trajectory(
    net,
    names,
    params,
    pdict,
    initial,
    initial_hash,
    rt,
    context_inputs,
    lr,
    k_steps=K_GRID,
    arm="TTT_K",
    shuf_perm=None,
    beta=L2_BETA,
):
    """Run plain SGD (momentum=0, weight_decay=0) for one context starting from r882 and reset bit-exact in `finally`."""
    if "fitness" in context_inputs or "gate" in context_inputs:
        raise ValueError("unrevealed fitness/gate must never be passed into TTT")
    if arm not in ("TTT_K", "TTT_SHUF", "TTT_L2"):
        raise ValueError(f"unknown TTT arm: {arm}")
    if lr <= 0 or not math.isfinite(lr):
        raise ValueError("positive finite learning rate required")
    k_tuple = tuple(sorted(set(int(k) for k in k_steps)))
    if not k_tuple or any(k not in K_GRID for k in k_tuple):
        raise ValueError("k_steps must be a non-empty subset of {1, 3, 5}")
    if rt.tensor_hash({n: pdict[n] for n in names}) != initial_hash:
        raise RuntimeError("adapter state not bit-exact r882 before context TTT")

    prog = context_inputs["prog_412"]
    sos = context_inputs["sos_prior"]
    tokens = context_inputs["tokens"]
    mask = context_inputs["masks"]
    scale = context_inputs["scale"]
    rev = context_inputs["revealed_indices"]
    raw = context_inputs["revealed_raw_f"]
    gate = context_inputs["revealed_gate"]
    anchor_total = context_inputs["anchor_total"]
    anchor_prior = context_inputs["anchor_prior"]

    if arm == "TTT_SHUF":
        if shuf_perm is None or tuple(shuf_perm) not in DERANGEMENTS_4:
            raise ValueError("TTT_SHUF requires a valid 4-element derangement")
        perm_t = torch.as_tensor(shuf_perm, dtype=torch.long)
        loss_raw = raw[:, perm_t]
        loss_gate = gate[:, perm_t]
    else:
        loss_raw = raw
        loss_gate = gate

    max_k = max(k_tuple)
    k_set = set(k_tuple)
    snapshots = {}
    optimizer = torch.optim.SGD(params, lr=float(lr), momentum=0.0, weight_decay=0.0)
    try:
        for step in range(1, max_k + 1):
            optimizer.zero_grad(set_to_none=True)
            scores = adapter_forward(
                net, prog, sos, tokens, mask, scale, rev, raw, gate, anchor_total, anchor_prior
            )
            loss = revealed_live_loss(scores, rev, loss_raw, loss_gate, rt.loss)
            if arm == "TTT_L2":
                l2_term = 0.5 * float(beta) * sum(
                    (pdict[n] - initial[n]).square().sum() for n in names
                )
                loss = loss + l2_term
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite TTT loss")
            loss.backward()
            gnorm = gradient_norm(params)
            optimizer.step()
            if step in k_set:
                with torch.no_grad():
                    post = adapter_forward(
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
                    )[0]
                    costs = post.detach().cpu().numpy().copy()
                    if not np.isfinite(costs).all():
                        raise RuntimeError("nonfinite post-TTT candidate costs")
                    pick = int(np.argmin(costs))
                    drift = adapter_drift(initial, {n: pdict[n] for n in names})
                    if not math.isfinite(drift) or drift <= 0.0:
                        raise RuntimeError("invalid post-TTT adapter drift")
                    snapshots[step] = dict(
                        costs=costs,
                        pick=pick,
                        drift=float(drift),
                        loss=float(loss.detach().cpu()),
                        grad_norm=float(gnorm),
                    )
    finally:
        with torch.no_grad():
            for n in names:
                pdict[n].copy_(initial[n])
            for p in params:
                p.grad = None
        if rt.tensor_hash({n: pdict[n] for n in names}) != initial_hash:
            raise RuntimeError("adapter reset hash mismatch after context TTT")

    return snapshots, max_k


def verify_leakage_mask_invariance(
    net, names, params, pdict, initial, initial_hash, rt, build_inputs_fn, cid, pool
):
    """Prove that poisoning unrevealed labels (indices 4..M-1) with NaN/+1e9/False leaves TTT costs, picks, and drifts
    bit-for-bit identical."""
    clean_inputs = build_inputs_fn(cid, pool["fitness"], pool["gate"])
    poisoned_fitness = pool["fitness"].copy()
    poisoned_gate = pool["gate"].copy()
    poisoned_fitness[4:] = np.nan
    poisoned_gate[4:] = False
    poisoned_inputs = build_inputs_fn(cid, poisoned_fitness, poisoned_gate)

    perm = shuf_permutation(198101, cid)
    for arm in ("TTT_K", "TTT_SHUF", "TTT_L2"):
        s_clean, _ = ttt_context_trajectory(
            net,
            names,
            params,
            pdict,
            initial,
            initial_hash,
            rt,
            clean_inputs,
            lr=3e-5,
            k_steps=K_GRID,
            arm=arm,
            shuf_perm=perm if arm == "TTT_SHUF" else None,
        )
        s_poison, _ = ttt_context_trajectory(
            net,
            names,
            params,
            pdict,
            initial,
            initial_hash,
            rt,
            poisoned_inputs,
            lr=3e-5,
            k_steps=K_GRID,
            arm=arm,
            shuf_perm=perm if arm == "TTT_SHUF" else None,
        )
        for k in K_GRID:
            if (
                not np.array_equal(s_clean[k]["costs"], s_poison[k]["costs"])
                or s_clean[k]["pick"] != s_poison[k]["pick"]
                or s_clean[k]["drift"] != s_poison[k]["drift"]
            ):
                raise RuntimeError(f"unrevealed label leakage detected in {arm} at k={k}")
    return True


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
