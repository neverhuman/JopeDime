# H61 Pre-Outcome Resource & Coordinator Amendment — 2026-10-31

Status: Committed before any H61 GPU cell execution, `fold_b` outcome read, or terminal `DL-Val` read (`outcome_cells_started = 0`, `fold_b_scored = False`, `dl_val_reads = 0`). Updated in `t2258` to replace immediate lock re-take with FIFO ticket queuing and legacy-waiter backoff.

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
lock_policy: fifo_ticket_and_legacy_backoff
lock_poll_seconds: 30
legacy_backoff_seconds: 45
primary_ranking_metric: bradley_terry_elo_family_grouped
auc_reported: false

## Notes

1. **GPU Cell-Wall Budget**:
   - Audited from `H57` measured timing (`60.114 s/gen`, `128.615 s` overhead, and `607.179 s` real `sigma001/seed15` cell wall; `H59` 20-round cells measured `87.75 s–90.29 s` on `cuda:0`).
   - Total projected H61 GPU cell wall (`2,577.332 s` / `42.96 min`) fits well within the frozen `3,600 s` (`60.0 min`) cap.
2. **Lock Fairness Correction (`t2258`, superseding `t2235` immediate re-take)**:
   - At `2026-10-31T04:54:44Z`, the launched H61 waiter (`PID 3178624`, running PR #816 with `immediate_first_retry=True` and `LOCK_POLL_SECONDS=5`) acquired `qualification.lock` `1.4 s` after H59 (`PID 2250432`) released it, ahead of H57 (`PID 3058000`, which had been waiting since `04:51:57Z` on a `30 s` poll). Per overseer instruction `t2258`, the running H61 process (`PID 3178624`) was left undisturbed to finish (since H57/H59 lock waits are excluded from their GPU cell-wall caps and bounded by their separate `3 h` wait limit, making this a scheduling delay rather than an infra-void).
   - For all subsequent waves, `qualification_lock_fifo.py` and `h61/run_wave.py` remove immediate first retry, poll at `30 s`, enforce FIFO ticket ordering in `~/jope_plan/qualification.queue/<epoch_ns>-<pid>-<wave>` (pruning dead PIDs, deleting ticket on lock acquisition, and re-enqueuing at the tail after cell release), and back off `>= 45 s` after releasing a cell if any legacy `*_STATUS.log` under `~/jope-data/h*/` logged `"WAIT qualification.lock"` within the last `90 s`.
3. **Bradley-Terry Elo vs `r882` (3 Seeds, Family-Grouped, Median + 95% CI)**:
   - Reports family-grouped Bradley-Terry Elo vs `r882` (`1500.0` anchor) across seeds `12, 13, 14` with both across-seed and 2,000-draw family-cluster bootstrap median and 95% CI. Binary classification AUC is never computed or reported (`auc_reported: false`). Ties fail everywhere.
