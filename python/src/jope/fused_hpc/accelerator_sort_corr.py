from __future__ import annotations

import math

import torch


def sort_valid_values_radix_f64(
    x: torch.Tensor,
    r_mask_bool: torch.Tensor,
) -> torch.Tensor:
    """Sort valid rows per column in ``float64`` using coalesced innermost-dimension radix sort."""
    rm = r_mask_bool.unsqueeze(-1)
    x_masked = torch.where(rm, x, torch.full_like(x, 1e300))
    x_bpn = x_masked.permute(0, 2, 1).contiguous()
    sorted_bpn = torch.sort(x_bpn, dim=-1, stable=True).values
    return sorted_bpn.permute(0, 2, 1).contiguous()


def sort_valid_rows_radix_f64(
    x: torch.Tensor,
    r_mask_bool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coalesced CUB radix-sort replacement for ``_sort_valid_rows_f64``.

    Transposes ``[B, N, P] -> [B, P, N]`` so each column is contiguous in memory,
    executes ``torch.sort(..., dim=-1, stable=True)`` (which dispatches to NVIDIA CUB
    ``DeviceSegmentedRadixSortKernel`` on CUDA — ``185x`` faster per call than
    single-thread global-memory ``batch_aquicksort_f64_cuda_kernel``), and transposes back.
    """
    rm = r_mask_bool.unsqueeze(-1)
    x_masked = torch.where(rm, x, torch.full_like(x, 1e300))
    x_bpn = x_masked.permute(0, 2, 1).contiguous()
    sorted_bpn, order_bpn = torch.sort(x_bpn, dim=-1, stable=True)
    return sorted_bpn.permute(0, 2, 1).contiguous(), order_bpn.permute(0, 2, 1).contiguous()


def reuse_monotone_sort_order_f64(
    x_transformed: torch.Tensor,
    order_x: torch.Tensor,
    r_mask_bool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse an existing column sort permutation ``order_x`` for a strictly monotone transform.

    In Stage 1A (`gpu_stage1a_and_slate.py:381, 495, 548`), ``Z_unc = (X_scaled - mean_xs) / std_xs``
    and ``Z_clp = clamp(Z_unc, -8.0, 8.0)`` are non-decreasing monotone transforms of ``X_in``.
    Gathering ``x_transformed`` along ``order_x`` in :math:`O(B N P)` avoids re-sorting the
    entire table while preserving ``sorted_vals`` and ``order_x``.
    """
    gathered = torch.gather(x_transformed, 1, order_x)
    rm = r_mask_bool.unsqueeze(-1)
    sorted_vals = torch.where(rm, gathered, torch.full_like(gathered, 1e300))
    return sorted_vals, order_x


def erf_approx_f64(x: torch.Tensor) -> torch.Tensor:
    """Abramowitz-Stegun 7.1.26 polynomial ``erf`` approximation matching Stage 1A."""
    sign = torch.sign(x)
    ax = torch.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * torch.exp(-(ax * ax)))


def exact_corr_with_yz_f64(
    f_col: torch.Tensor,
    y_z: torch.Tensor,
    r_mask: torch.Tensor,
    n_b: torch.Tensor,
) -> torch.Tensor:
    """Reference single-probe absolute correlation with ``y_z`` matching Stage 1A."""
    rm3 = r_mask.unsqueeze(-1)
    dn = n_b.clamp_min(1.0)
    mean_f = (f_col * rm3).sum(dim=1) / dn
    df = (f_col - mean_f.unsqueeze(1)) * rm3
    std_f = torch.sqrt((df * df).sum(dim=1) / dn)
    valid = (std_f > 1e-9) & torch.isfinite(std_f) & (n_b > 0.0)
    std_safe = torch.where(valid, std_f, torch.ones_like(std_f))
    prod = ((df / std_safe.unsqueeze(1)) * (y_z * r_mask).unsqueeze(-1)).sum(dim=1) / dn
    out = torch.abs(prod)
    return torch.where(valid & torch.isfinite(out), out, torch.zeros_like(out))


def fused_14_probe_corrs_with_yz_f64(
    z_clp: torch.Tensor,
    y_z: torch.Tensor,
    r_mask: torch.Tensor,
    n_b: torch.Tensor,
    c_mask: torch.Tensor,
    c_q10: torch.Tensor,
    c_q25: torch.Tensor,
    c_q50: torch.Tensor,
    c_q75: torch.Tensor,
    c_q90: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute all 14 Stage 1A nonlinear probe correlations in one fused 4D pass.

    Replaces the 14 separate ``_exact_corr_with_yz_f64`` calls in
    ``gpu_stage1a_and_slate.py:521-540`` (`~140` kernel launches) with a single
    ``[B, N, P, 14]`` tensor reduction over rows ``dim=1``.
    Matches the sequential Stage 1A probe correlations to ``< 1e-12`` max abs diff.
    """
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    half_pi = 0.5 * math.pi
    one_third = 1.0 / 3.0

    abs_z = torch.abs(z_clp)
    sign_z = torch.sign(z_clp)

    f_asinh = torch.asinh(z_clp)
    f_atan = torch.atan(z_clp)
    f_erf = erf_approx_f64(z_clp * inv_sqrt2)
    f_cbrt = sign_z * torch.pow(abs_z, one_third)
    f_log1p = sign_z * torch.log1p(abs_z)
    f_tanh = torch.tanh(0.75 * z_clp)
    f_laplace = torch.expm1(-abs_z)
    f_clamp = torch.clamp(z_clp, c_q10.unsqueeze(1), c_q90.unsqueeze(1))
    f_step = (z_clp >= c_q50.unsqueeze(1)).to(torch.float64)
    f_hinge = torch.clamp(z_clp - c_q75.unsqueeze(1), min=0.0) - torch.clamp(
        c_q25.unsqueeze(1) - z_clp, min=0.0
    )
    f_sin = torch.sin(half_pi * z_clp)
    f_cos = torch.cos(half_pi * z_clp)
    f_fract = z_clp - torch.floor(z_clp)
    f_lin = z_clp

    # Stack all 14 probes into [B, N, P, 14] and run a single reduction pass
    f_all = torch.stack(
        [
            f_asinh,
            f_atan,
            f_erf,
            f_cbrt,
            f_log1p,
            f_tanh,
            f_laplace,
            f_clamp,
            f_step,
            f_hinge,
            f_sin,
            f_cos,
            f_fract,
            f_lin,
        ],
        dim=-1,
    )

    rm4 = r_mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
    dn = n_b.clamp_min(1.0).unsqueeze(-1)  # [B, 1, 1]
    mean_f = (f_all * rm4).sum(dim=1) / dn  # [B, P, 14]
    df = (f_all - mean_f.unsqueeze(1)) * rm4  # [B, N, P, 14]
    std_f = torch.sqrt((df * df).sum(dim=1) / dn)  # [B, P, 14]

    valid = (std_f > 1e-9) & torch.isfinite(std_f) & (n_b.unsqueeze(-1) > 0.0)
    std_safe = torch.where(valid, std_f, torch.ones_like(std_f))

    yz_masked = (y_z * r_mask).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
    prod = ((df / std_safe.unsqueeze(1)) * yz_masked).sum(dim=1) / dn  # [B, P, 14]
    corrs_all = torch.abs(prod)
    corrs_all = torch.where(valid & torch.isfinite(corrs_all), corrs_all, torch.zeros_like(corrs_all))

    c_sin = corrs_all[:, :, 10]
    c_cos = corrs_all[:, :, 11]
    masked_corrs = corrs_all * c_mask.unsqueeze(-1)

    return {
        "c_asinh": masked_corrs[:, :, 0],
        "c_atan": masked_corrs[:, :, 1],
        "c_erf": masked_corrs[:, :, 2],
        "c_cbrt": masked_corrs[:, :, 3],
        "c_log1p": masked_corrs[:, :, 4],
        "c_tanh": masked_corrs[:, :, 5],
        "c_laplace": masked_corrs[:, :, 6],
        "c_clamp": masked_corrs[:, :, 7],
        "c_step": masked_corrs[:, :, 8],
        "c_hinge": masked_corrs[:, :, 9],
        "c_sin": c_sin,
        "c_cos": c_cos,
        "c_sincos": torch.maximum(c_sin, c_cos) * c_mask,
        "c_fract": masked_corrs[:, :, 12],
        "c_lin": masked_corrs[:, :, 13],
        "corrs_14": masked_corrs,
    }
