"""Phase E Fused-Path HPC Accelerator Primitives (NORTH_STAR NS11).

Addresses the five GPU/host throughput bottlenecks on the fused inference and
flywheel path documented in ``FUSION_REDTEAM.md`` Part B (``HPC #1-#5``):

* **HPC #1** (``build_live_96_slate`` champion retrieval):
  Replaces single-thread ``881 x 48`` cosine similarity + :math:`O(881^2)` insertion
  sort and ``128 x 12`` L2 distance + :math:`O(128^2)` insertion sort with vectorized
  batched GEMM/L2 kernels and deterministic tie-breaking.
* **HPC #2** (Stage 1A column sorting):
  Replaces single-thread-per-column global-memory ``batch_aquicksort_f64`` with
  coalesced transpose + CUB segmented radix sort (``torch.sort(..., stable=True)``)
  and reuses ``X_in``'s sort permutation across monotone column transforms.
* **HPC #3** (Stage 1A 14 nonlinear probe correlations):
  Fuses the 14 separate ``_exact_corr_with_yz_f64`` full-table passes into a single
  4D ``[B, N, P, 14]`` vectorized contraction pass.
* **HPC #4** (Host I/O, H2D staging, and out-of-loop BPE token counting):
  Provides reusable page-locked (``pin_memory=True``) host batch buffers with async
  stream H2D transfer, bounded multi-threaded CSV split loading, and out-of-loop
  batch BPE token counting.
* **HPC #5** (2D ``(P_tier, N_tr)`` shape bucketing, ``CUDAGraph`` Stage-1A/2 wrapper,
  and ``xbabe1-3`` multi-GPU sharding):
  Eliminates the ``3.6x`` column-padding inflation caused by 1D ``N_tr``-only sorting,
  captures fixed-shape Stage 1A-neural and Stage 2 sub-networks in ``CUDAGraph``s
  with strict ``fp32`` Stage 2 precision, and partitions workloads across
  ``xbabe1`` (RTX 4090), ``xbabe2`` (RTX 3090), and ``xbabe3`` (RTX 3090) (never ``xbabe0``).
"""

from __future__ import annotations

ALLOWED_GPU_NODES: tuple[str, ...] = ("xbabe1", "xbabe2", "xbabe3")
DEFAULT_NODE_THROUGHPUT_WEIGHTS: dict[str, float] = {
    "xbabe1": 0.45,
    "xbabe2": 0.275,
    "xbabe3": 0.275,
}
PROBE_14_NAMES: tuple[str, ...] = (
    "asinh",
    "atan",
    "erf",
    "cbrt",
    "log1p",
    "tanh",
    "laplace",
    "clamp",
    "step",
    "hinge",
    "sin",
    "cos",
    "fract",
    "lin",
)


def _load_split_exports():
    from .accelerator_champion import (
        compute_table_seed_sha256_u32 as compute_table_seed_sha256_u32,
    )
    from .accelerator_champion import (
        precompute_champion_topk_batched_gpu as precompute_champion_topk_batched_gpu,
    )
    from .accelerator_loading import (
        PinnedBatchBufferPool as PinnedBatchBufferPool,
    )
    from .accelerator_loading import (
        count_rust_tokens_out_of_loop as count_rust_tokens_out_of_loop,
    )
    from .accelerator_loading import (
        parallel_load_dataset_splits as parallel_load_dataset_splits,
    )
    from .accelerator_schedule import (
        FusedStageCudaGraphRunner as FusedStageCudaGraphRunner,
    )
    from .accelerator_schedule import (
        ShapeBucketPlan as ShapeBucketPlan,
    )
    from .accelerator_schedule import (
        _compute_schedule_padded_cells as _compute_schedule_padded_cells,
    )
    from .accelerator_schedule import (
        _snapshot_dope_state_sha256 as _snapshot_dope_state_sha256,
    )
    from .accelerator_schedule import (
        bucket_tables_2d_by_p_tier_and_ntr as bucket_tables_2d_by_p_tier_and_ntr,
    )
    from .accelerator_schedule import (
        patch_fused_network_hpc as patch_fused_network_hpc,
    )
    from .accelerator_schedule import (
        plan_multigpu_shards as plan_multigpu_shards,
    )
    from .accelerator_sort_corr import (
        erf_approx_f64 as erf_approx_f64,
    )
    from .accelerator_sort_corr import (
        exact_corr_with_yz_f64 as exact_corr_with_yz_f64,
    )
    from .accelerator_sort_corr import (
        fused_14_probe_corrs_with_yz_f64 as fused_14_probe_corrs_with_yz_f64,
    )
    from .accelerator_sort_corr import (
        reuse_monotone_sort_order_f64 as reuse_monotone_sort_order_f64,
    )
    from .accelerator_sort_corr import (
        sort_valid_rows_radix_f64 as sort_valid_rows_radix_f64,
    )
    from .accelerator_sort_corr import (
        sort_valid_values_radix_f64 as sort_valid_values_radix_f64,
    )

    return (
        compute_table_seed_sha256_u32,
        precompute_champion_topk_batched_gpu,
        PinnedBatchBufferPool,
        count_rust_tokens_out_of_loop,
        parallel_load_dataset_splits,
        FusedStageCudaGraphRunner,
        ShapeBucketPlan,
        _compute_schedule_padded_cells,
        _snapshot_dope_state_sha256,
        bucket_tables_2d_by_p_tier_and_ntr,
        patch_fused_network_hpc,
        plan_multigpu_shards,
        erf_approx_f64,
        exact_corr_with_yz_f64,
        fused_14_probe_corrs_with_yz_f64,
        reuse_monotone_sort_order_f64,
        sort_valid_rows_radix_f64,
        sort_valid_values_radix_f64,
    )


(
    compute_table_seed_sha256_u32,
    precompute_champion_topk_batched_gpu,
    PinnedBatchBufferPool,
    count_rust_tokens_out_of_loop,
    parallel_load_dataset_splits,
    FusedStageCudaGraphRunner,
    ShapeBucketPlan,
    _compute_schedule_padded_cells,
    _snapshot_dope_state_sha256,
    bucket_tables_2d_by_p_tier_and_ntr,
    patch_fused_network_hpc,
    plan_multigpu_shards,
    erf_approx_f64,
    exact_corr_with_yz_f64,
    fused_14_probe_corrs_with_yz_f64,
    reuse_monotone_sort_order_f64,
    sort_valid_rows_radix_f64,
    sort_valid_values_radix_f64,
) = _load_split_exports()
del _load_split_exports
