# H61 Pre-Outcome Resource & Coordinator Amendment — 2026-10-31

Status: Committed before any H61 GPU cell execution, `fold_b` outcome read, or terminal `DL-Val` read (`outcome_cells_started = 0`, `fold_b_scored = False`, `dl_val_reads = 0`).

## Machine-Readable Execution Parameters

amendment_date: 2026-10-31
wave_id: H61
total_gpu_cell_wall_cap_seconds: 3600
fixed_round_horizon: 20
fixed_steps_per_round: 2
measured_h57_real_cell_seconds: 607.1792591279373
measured_h57_generation_seconds: 60.113566709915176
measured_h57_timing_overhead_seconds: 128.61473213299178
projected_total_seconds: 2577.332
null_margin_m: 0.001546
admission_lcb_threshold: 0.001
max_harm_fraction: 0.10
immediate_lock_retry_default: true
primary_ranking_metric: bradley_terry_elo_family_grouped
auc_reported: false

## Notes

1. **GPU Cell-Wall Budget**:
   - Audited from `H57` measured timing (`60.114 s/gen`, `128.615 s` overhead, and `607.179 s` real `sigma001/seed15` cell wall; `H59` 20-round cells measured `87.75 s–90.29 s` on `cuda:0`).
   - Total projected H61 GPU cell wall (`2,577.332 s` / `42.96 min`) fits well within the frozen `3,600 s` (`60.0 min`) cap.
2. **Fairness Fix (`--immediate-lock-retry`)**:
   - Enabled by default in `research/flywheel_selfplay/h61/run_wave.py`: after a cell releases `qualification.lock`, the next cell of the same wave attempts `flock(LOCK_EX | LOCK_NB)` immediately without pre-lock `cpu_percent` sleep and retries immediately (`0.0 s` sleep before the first retry) on transient lock contention so waves do not interleave cell-by-cell more than needed.
3. **Bradley-Terry Elo vs `r882` (3 Seeds, Family-Grouped, Median + 95% CI)**:
   - Reports family-grouped Bradley-Terry Elo vs `r882` (`1500.0` anchor) across seeds `12, 13, 14` with both across-seed and 2,000-draw family-cluster bootstrap median and 95% CI. Binary classification AUC is never computed or reported (`auc_reported: false`). Ties fail everywhere.
