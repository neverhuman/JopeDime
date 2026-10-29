"""Phase E Fused-Path HPC Speed-Ups for FusedJopeNetwork (NS11).

Implements the five quantified HPC optimizations on the fused Stage 1A -> 1B -> 2 -> 3
path identified in ``FUSION_REDTEAM.md`` Part B (``HPC #1-#5``) and ``NORTH_STAR.md`` Phase E:
1. ``HPC #1``: Batched cuBLAS GEMM champion retrieval (``precompute_champion_topk_batched_gpu``)
   and strided/full deterministic SHA-256 table seed helper.
2. ``HPC #2``: Coalesced CUB segmented radix sort (``sort_valid_rows_radix_f64``) and
   monotone column sort-order reuse (``reuse_monotone_sort_order_f64``).
3. ``HPC #3``: Fused 14-probe Stage 1A correlation contraction
   (``fused_14_probe_corrs_with_yz_f64``), reducing 14 full-table probe passes to 1.
4. ``HPC #4``: Pinned-memory async H2D batch pool (``PinnedBatchBufferPool``),
   multi-threaded CSV split prefetcher (``parallel_load_dataset_splits``), and
   out-of-loop batch BPE token counting (``count_rust_tokens_out_of_loop``).
5. ``HPC #5``: 2D ``(P_tier, N_tr)`` shape bucketing (``bucket_tables_2d_by_p_tier_and_ntr``),
   ``CUDAGraph`` capture/replay runner for fixed-shape Stage 1A-neural and Stage 2
   sub-networks with strict ``fp32`` Stage 2 precision guard (``FusedStageCudaGraphRunner``),
   cost-balanced multi-GPU ``xbabe1-3`` shard planner (``plan_multigpu_shards``),
   and non-invasive in-place network patcher (``patch_fused_network_hpc``).
"""

from __future__ import annotations

from jope.fused_hpc.accelerator import (
    FusedStageCudaGraphRunner,
    PinnedBatchBufferPool,
    ShapeBucketPlan,
    bucket_tables_2d_by_p_tier_and_ntr,
    count_rust_tokens_out_of_loop,
    erf_approx_f64,
    exact_corr_with_yz_f64,
    fused_14_probe_corrs_with_yz_f64,
    parallel_load_dataset_splits,
    patch_fused_network_hpc,
    plan_multigpu_shards,
    precompute_champion_topk_batched_gpu,
    reuse_monotone_sort_order_f64,
    sort_valid_rows_radix_f64,
    sort_valid_values_radix_f64,
)

__all__ = [
    "FusedStageCudaGraphRunner",
    "PinnedBatchBufferPool",
    "ShapeBucketPlan",
    "bucket_tables_2d_by_p_tier_and_ntr",
    "count_rust_tokens_out_of_loop",
    "erf_approx_f64",
    "exact_corr_with_yz_f64",
    "fused_14_probe_corrs_with_yz_f64",
    "parallel_load_dataset_splits",
    "patch_fused_network_hpc",
    "plan_multigpu_shards",
    "precompute_champion_topk_batched_gpu",
    "reuse_monotone_sort_order_f64",
    "sort_valid_rows_radix_f64",
    "sort_valid_values_radix_f64",
]
