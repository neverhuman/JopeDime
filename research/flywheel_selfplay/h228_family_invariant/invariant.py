"""Family-invariant gradient combination rules: AND-mask (tau=0.8), shuffled placebo, and GroupDRO (eta=0.01)."""

import math
import numpy as np
import torch
import torch.nn.functional as F

TAU = 0.8
GDRO_ETA = 0.01
MIN_FAMILIES_PER_STEP = 4


def per_context_loss(pred, fitness, gate):
    """Per-context H140/H146 SmoothL1 + 0.1 KL (temperature 0.2) loss vector of shape (B,)."""
    target = torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))
    return F.smooth_l1_loss(pred, target, reduction="none").mean(1) + 0.1 * F.kl_div(
        F.log_softmax(-pred / 0.2, dim=-1),
        F.softmax(-target / 0.2, dim=-1),
        reduction="none",
    ).sum(1)


def andmask_combine(family_grads, tau=TAU):
    """Parascandolo et al. (2020) AND-mask over M >= 4 per-family gradient vectors of shape (M, D)."""
    if (
        not isinstance(family_grads, torch.Tensor)
        or family_grads.ndim != 2
        or family_grads.shape[0] < MIN_FAMILIES_PER_STEP
        or family_grads.shape[1] < 1
        or not torch.isfinite(family_grads).all()
    ):
        raise ValueError("finite (M>=4, D>=1) family_grads tensor required")
    if tau != TAU:
        raise ValueError("ANDMASK tau is fixed a priori at 0.8")
    n_families, n_coords = family_grads.shape
    pos = (family_grads > 0).sum(dim=0)
    neg = (family_grads < 0).sum(dim=0)
    max_agree = torch.maximum(pos, neg)
    # Exact integer inequality for tau = 0.8 = 4/5: agree / M >= 4/5 <=> 5 * agree >= 4 * M.
    mask = (5 * max_agree) >= (4 * n_families)
    mean_grad = family_grads.mean(dim=0)
    masked_grad = torch.where(mask, mean_grad, torch.zeros_like(mean_grad))
    kept_count = int(mask.sum().item())
    stats = dict(
        n_families=int(n_families),
        kept_count=kept_count,
        total_coords=int(n_coords),
        kept_fraction=float(kept_count / n_coords),
        raw_grad_sq=float(family_grads.detach().double().square().sum().item()),
        masked_grad_sq=float(masked_grad.detach().double().square().sum().item()),
    )
    return masked_grad, mask, stats


def andmask_shuf_combine(family_grads, seed, tau=TAU, kept_count_override=None):
    """Random coordinate mask with the exact same per-step kept count/fraction as ANDMASK."""
    _, and_mask, and_stats = andmask_combine(family_grads, tau=tau)
    n_coords = and_stats["total_coords"]
    kept_count = (
        and_stats["kept_count"]
        if kept_count_override is None
        else int(kept_count_override)
    )
    if not 0 <= kept_count <= n_coords:
        raise ValueError("invalid kept_count for ANDMASK_SHUF")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_coords)
    shuf_np = np.zeros(n_coords, dtype=bool)
    shuf_np[perm[:kept_count]] = True
    shuf_mask = torch.as_tensor(shuf_np, dtype=torch.bool, device=family_grads.device)
    if int(shuf_mask.sum().item()) != kept_count:
        raise RuntimeError("ANDMASK_SHUF kept_count density mismatch")
    mean_grad = family_grads.mean(dim=0)
    shuf_grad = torch.where(shuf_mask, mean_grad, torch.zeros_like(mean_grad))
    stats = dict(
        n_families=and_stats["n_families"],
        kept_count=kept_count,
        andmask_local_kept_count=and_stats["kept_count"],
        total_coords=n_coords,
        kept_fraction=float(kept_count / n_coords),
        density_match=bool(int(shuf_mask.sum().item()) == kept_count),
        raw_grad_sq=and_stats["raw_grad_sq"],
        masked_grad_sq=float(shuf_grad.detach().double().square().sum().item()),
    )
    return shuf_grad, shuf_mask, stats


def gdro_init_weights(n_families):
    """Initialize uniform GroupDRO simplex weights over FIT families."""
    if type(n_families) is not int or n_families < MIN_FAMILIES_PER_STEP:
        raise ValueError("at least 4 FIT families required for GroupDRO")
    return np.full(n_families, 1.0 / n_families, dtype=np.float64)


def gdro_step(q, fit_families, step_families, step_losses, family_grads, eta=GDRO_ETA):
    """Sagawa et al. (2020) GroupDRO exponentiated-gradient update and family-weighted gradient."""
    if eta != GDRO_ETA:
        raise ValueError("GroupDRO eta is fixed a priori at 0.01")
    fit_list = list(fit_families)
    step_list = list(step_families)
    q_arr = np.asarray(q, dtype=np.float64)
    losses_arr = np.asarray(step_losses, dtype=np.float64)
    if (
        len(fit_list) < MIN_FAMILIES_PER_STEP
        or q_arr.shape != (len(fit_list),)
        or not np.isfinite(q_arr).all()
        or (q_arr <= 0).any()
        or abs(math.fsum(q_arr.tolist()) - 1.0) > 1e-9
    ):
        raise ValueError("valid simplex group weights q required")
    if (
        len(step_list) < MIN_FAMILIES_PER_STEP
        or len(set(step_list)) != len(step_list)
        or not set(step_list) <= set(fit_list)
        or losses_arr.shape != (len(step_list),)
        or not np.isfinite(losses_arr).all()
    ):
        raise ValueError("at least 4 distinct finite FIT step families required")
    if (
        not isinstance(family_grads, torch.Tensor)
        or family_grads.shape[0] != len(step_list)
        or family_grads.ndim != 2
        or family_grads.shape[1] < 1
        or not torch.isfinite(family_grads).all()
    ):
        raise ValueError("finite (M, D) family_grads matching step_families required")
    fit_index = {f: i for i, f in enumerate(fit_list)}
    q_new = q_arr.copy()
    for fam, loss_val in zip(step_list, losses_arr, strict=True):
        q_new[fit_index[fam]] *= math.exp(eta * float(loss_val))
    q_sum = math.fsum(q_new.tolist())
    if not math.isfinite(q_sum) or q_sum <= 0:
        raise RuntimeError("nonfinite GroupDRO weight normalization")
    q_new /= q_sum
    active_raw = np.array([q_new[fit_index[f]] for f in step_list], dtype=np.float64)
    active_sum = math.fsum(active_raw.tolist())
    if not math.isfinite(active_sum) or active_sum <= 0:
        raise RuntimeError("nonfinite active GroupDRO step weight sum")
    step_w = active_raw / active_sum
    w_tensor = torch.as_tensor(
        step_w, dtype=family_grads.dtype, device=family_grads.device
    )
    combined_grad = (w_tensor[:, None] * family_grads).sum(dim=0)
    if not torch.isfinite(combined_grad).all():
        raise RuntimeError("nonfinite GroupDRO combined gradient")
    stats = dict(
        n_families=len(step_list),
        max_group_weight=float(q_new.max()),
        min_group_weight=float(q_new.min()),
        max_step_weight=float(step_w.max()),
        min_step_weight=float(step_w.min()),
        raw_grad_sq=float(family_grads.detach().double().square().sum().item()),
        combined_grad_sq=float(combined_grad.detach().double().square().sum().item()),
    )
    return combined_grad, q_new, stats


def extract_family_gradients(params, batch_cids, family_by_cid, ctx_losses):
    """Compute per-family mean losses and stacked active-parameter gradients for a microbatch."""
    if len(batch_cids) != len(ctx_losses):
        raise ValueError("batch_cids and ctx_losses length mismatch")
    step_families = sorted({family_by_cid[cid] for cid in batch_cids})
    if len(step_families) < MIN_FAMILIES_PER_STEP:
        raise RuntimeError(
            f"requires >= {MIN_FAMILIES_PER_STEP} families/step, got {len(step_families)}"
        )
    active_indices = None
    family_losses = []
    grad_rows = []
    for idx, fam in enumerate(step_families):
        slots = [i for i, cid in enumerate(batch_cids) if family_by_cid[cid] == fam]
        fam_loss = ctx_losses[slots].mean()
        if not torch.isfinite(fam_loss):
            raise RuntimeError("nonfinite family loss")
        grads = torch.autograd.grad(
            fam_loss,
            params,
            retain_graph=(idx < len(step_families) - 1),
            allow_unused=True,
        )
        cur_active = [i for i, g in enumerate(grads) if g is not None]
        if len(cur_active) != 41 or len(params) - len(cur_active) != 6:
            raise RuntimeError(
                "41 active / 6 inactive adapter gradient inventory required"
            )
        if active_indices is None:
            active_indices = cur_active
        elif cur_active != active_indices:
            raise RuntimeError(
                "active adapter gradient indices drifted across families"
            )
        flat = torch.cat([grads[i].detach().reshape(-1) for i in active_indices])
        if not torch.isfinite(flat).all():
            raise RuntimeError("nonfinite per-family gradient")
        family_losses.append(float(fam_loss.detach().item()))
        grad_rows.append(flat)
    return (
        step_families,
        np.asarray(family_losses, dtype=np.float64),
        torch.stack(grad_rows, dim=0),
        active_indices,
    )


def accumulate_active_grad(params, active_indices, flat_grad, scale=0.25):
    """Add scale * flat_grad into params[i].grad for i in active_indices."""
    offset = 0
    for i in active_indices:
        p = params[i]
        numel = p.numel()
        piece = (
            (flat_grad[offset : offset + numel] * scale).reshape(p.shape).to(p.dtype)
        )
        if p.grad is None:
            p.grad = piece.clone()
        else:
            p.grad.add_(piece)
        offset += numel
    if offset != flat_grad.numel():
        raise RuntimeError("flat_grad length mismatch with active adapter parameters")
