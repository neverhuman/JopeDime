"""Phase E / Phase 2 HPC #1 and #6 helpers."""

from __future__ import annotations

RBUF_STRIDE_PER_ROW: int = 260
IBUF_STRIDE_PER_ROW: int = 268
IBUF_HIST_EXTRA_INTS: int = 2048


def _load_split_exports():
    from jope.fused_hpc.slate import (
        BUILD_LIVE_96_SLATE_BLOCK256_CUDA_SOURCE as BUILD_LIVE_96_SLATE_BLOCK256_CUDA_SOURCE,
    )
    from jope.fused_hpc.slate import (
        _block256_table_seed_u32 as _block256_table_seed_u32,
    )
    from jope.fused_hpc.slate import (
        _fnv1a_embs_hash_u64 as _fnv1a_embs_hash_u64,
    )
    from jope.fused_hpc.slate import (
        build_live_96_slate_block256 as build_live_96_slate_block256,
    )
    from jope.fused_hpc.transcendental import (
        EXACT_HOST_DEVICE_TRANSCENDENTALS_CUDA_HEADER as EXACT_HOST_DEVICE_TRANSCENDENTALS_CUDA_HEADER,
    )
    from jope.fused_hpc.transcendental import (
        _as_f64_array as _as_f64_array,
    )
    from jope.fused_hpc.transcendental import (
        _cos_poly_f64 as _cos_poly_f64,
    )
    from jope.fused_hpc.transcendental import (
        _sin_poly_f64 as _sin_poly_f64,
    )
    from jope.fused_hpc.transcendental import (
        _sincos_reduce_f64 as _sincos_reduce_f64,
    )
    from jope.fused_hpc.transcendental import (
        exact_box_muller_f64 as exact_box_muller_f64,
    )
    from jope.fused_hpc.transcendental import (
        exact_cos_f64 as exact_cos_f64,
    )
    from jope.fused_hpc.transcendental import (
        exact_exp_f64 as exact_exp_f64,
    )
    from jope.fused_hpc.transcendental import (
        exact_log_f64 as exact_log_f64,
    )
    from jope.fused_hpc.transcendental import (
        exact_sin_f64 as exact_sin_f64,
    )
    from jope.fused_hpc.workspace import (
        PHASE2_HEAD_SLICE_SPECS as PHASE2_HEAD_SLICE_SPECS,
    )
    from jope.fused_hpc.workspace import (
        Phase2HeadSliceSpec as Phase2HeadSliceSpec,
    )
    from jope.fused_hpc.workspace import (
        Phase2WinnerWorkspace as Phase2WinnerWorkspace,
    )

    return (
        BUILD_LIVE_96_SLATE_BLOCK256_CUDA_SOURCE,
        _block256_table_seed_u32,
        _fnv1a_embs_hash_u64,
        build_live_96_slate_block256,
        EXACT_HOST_DEVICE_TRANSCENDENTALS_CUDA_HEADER,
        _as_f64_array,
        _cos_poly_f64,
        _sin_poly_f64,
        _sincos_reduce_f64,
        exact_box_muller_f64,
        exact_cos_f64,
        exact_exp_f64,
        exact_log_f64,
        exact_sin_f64,
        PHASE2_HEAD_SLICE_SPECS,
        Phase2HeadSliceSpec,
        Phase2WinnerWorkspace,
    )


(
    BUILD_LIVE_96_SLATE_BLOCK256_CUDA_SOURCE,
    _block256_table_seed_u32,
    _fnv1a_embs_hash_u64,
    build_live_96_slate_block256,
    EXACT_HOST_DEVICE_TRANSCENDENTALS_CUDA_HEADER,
    _as_f64_array,
    _cos_poly_f64,
    _sin_poly_f64,
    _sincos_reduce_f64,
    exact_box_muller_f64,
    exact_cos_f64,
    exact_exp_f64,
    exact_log_f64,
    exact_sin_f64,
    PHASE2_HEAD_SLICE_SPECS,
    Phase2HeadSliceSpec,
    Phase2WinnerWorkspace,
) = _load_split_exports()
del _load_split_exports
