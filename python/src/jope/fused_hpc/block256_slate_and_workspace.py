"""Phase E / Phase 2 HPC #1 and #6 helpers."""

from __future__ import annotations

RBUF_STRIDE_PER_ROW: int = 260
IBUF_STRIDE_PER_ROW: int = 268
IBUF_HIST_EXTRA_INTS: int = 2048

from jope.fused_hpc.transcendental import (
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
)
from jope.fused_hpc.workspace import (
    PHASE2_HEAD_SLICE_SPECS,
    Phase2HeadSliceSpec,
    Phase2WinnerWorkspace,
)
from jope.fused_hpc.slate import (
    BUILD_LIVE_96_SLATE_BLOCK256_CUDA_SOURCE,
    _block256_table_seed_u32,
    _fnv1a_embs_hash_u64,
    build_live_96_slate_block256,
)
