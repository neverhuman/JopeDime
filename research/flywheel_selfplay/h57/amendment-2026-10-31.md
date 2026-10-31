# H57 pre-outcome resource amendment — 2026-10-31

Status: committed before any H57 outcome cell or fold-B read. The only H57 GPU evidence before this amendment is the preregistered timing-only sigma001/seed15 generation in `~/jope-data/h57/h57-wave/timing/`.

## Measured timing

- measured_generation_seconds: 60.113566709915176
- timing_cell_wall_seconds: 128.61473213299178
- setup_and_baseline_overhead_seconds: 68.49567237799056
- timing_mode: sigma001 / seed15 / one generation / population 32
- maximum_rss_kib: 3939296
- maximum_host_cpu_percent: 31.7
- gpu_free_mib_after_lock: 24124
- dope_sha256_before_and_after: c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814
- outcome_cells_started: 0
- fold_b_scored: false
- terminal_dl_val_reads: 0
- holdout_payloads_opened: 0
- outer_rows_delivered: 0

## Machine-readable budget fields
total_gpu_cell_wall_cap_seconds: 5400
fixed_generation_horizon: 8
projected_total_seconds: 5220.746
measured_generation_seconds: 60.113566709915176

## Fixed horizon and budget

- fixed_generation_horizon: 8
- outcome_training_cells: 7 (six candidates plus the sigma001/seed15 drift-only null)
- total_gpu_cell_wall_cap_seconds: 5400
- projected_candidate_and_null_cell_wall_seconds: 3845.829
- projected_fold_b_scoring_reserve_seconds: 128.615 (one full measured timing-cell wall)
- maximum_possible_terminal_dl_val_reads: 6 (one per seed candidate for each independently admitted sigma arm)
- projected_terminal_scoring_reserve_seconds: 771.688 (six full measured timing-cell walls)
- projected_subtotal_seconds: 4746.133
- contingency_fraction: 0.10
- projected_contingency_seconds: 474.613
- projected_total_seconds: 5220.746
- projected_slack_seconds: 179.254
- projected_total_if_12_generations_seconds: 7072.244
- lock_wait_seconds: 3776.319, excluded from GPU-cell-wall cap and separately bounded by the existing three-hour wait limit

The projection charges each of seven outcome cells one measured setup/baseline overhead plus eight measured generation times. It reserves one full measured timing-cell wall for fold-B scoring, one for each of the six maximum possible terminal reads, and then adds 10% contingency. The total is below the frozen 5,400-second cap, so the preregistered eight-generation default remains fixed. Twelve generations do not fit this conservative projection and are not authorized. No arm, seed, sigma, population, admission threshold, or cell order changes.

## Integrity and launch constraints

Keep the #635 preregistration unchanged: r882 checkpoint copy only, seeds 15/16/17, sigma 0.001/0.003, population 32, fold seed 4900, strict family-cluster LCB95 > 0.001, ties fail, no-harm rule unchanged, and no generation/candidate selection. Preserve DOPE `c51a56d0`; check it before/after each generation and terminal scoring pass. No live-loop, sealed/holdout, outer-fold, or outcome-based horizon access. The coordinator must hold `~/jope_plan/qualification.lock` per cell, check `statvfs('/')` below 95% after acquisition, require GPU free >=6 GiB, keep CPU <=85%, and require MemAvailable >=16 GiB. Release the lock between cells. A lock or guard wait expiry produces an infra-void receipt; do not resume partial optimizer state.
