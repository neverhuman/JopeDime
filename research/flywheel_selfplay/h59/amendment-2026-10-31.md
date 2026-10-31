# H59 pre-outcome resource and red-team amendment — 2026-10-31

Status: committed before any H59 outcome cell, fold-B read, or terminal DL-Val read (`outcome_cells_started: 0`, `fold_b_scored: false`, `terminal_dl_val_reads: 0`).

## 1. Measured H57 real-cell timing & H59 GPU cell-wall projection

Per `~/nudges/agy_fw_audit_h59_coord_t2205.md`, the H59 GPU cell-wall projection uses the **measured H57 real outcome cell wall** (`607.2 s`, specifically `607.1792591279373 s` recorded on `sigma001/seed15` in `~/jope-data/h57/h57-wave-retry-20261031T0338Z/outcomes/wave.partial.json`, rather than the earlier `549.4 s` projection) together with the audited H57 timing receipt (`~/jope-data/h57/h57-wave/timing/timing.json`):

- `measured_h57_real_cell_seconds`: `607.1792591279373` (`607.2 s` for setup + baseline + `8` generations of `32` full-`fold_a` evaluations = `256,256` context-slate evaluations)
- `measured_h57_generation_seconds`: `60.113566709915176` (`60.1 s` per `32,032` context-slate evaluations)
- `measured_h57_timing_cell_wall_seconds`: `128.61473213299178`
- `derived_h57_real_cell_overhead_seconds`: `126.2707254486159` (`607.1792591 - 8 * 60.1135667`)

In `H59`, each of the `12` `(arm, seed)` outcome cells (`4` arms `CONTROL, CLIP, NORM, CLIP+NORM` $\times$ `3` seeds `12, 13, 14`) executes `20` rounds $\times$ `2` optimizer steps = `40` gradient updates with `batch_size = 256` contexts and `slate_size = 192` (`40 * 256 = 10,240` context-slate evaluations = `0.31968` of one H57 generation, or `19.217 s` training compute + `126.271 s` actual H57 cell setup/checkpoint-write overhead = `145.488 s` per H59 outcome cell; empirically in `H55` `attempt4`, `ema25` with `50` steps at `batch_size = 1,219` averaged `107.4 s/cell`).

Charging:
- `12` outcome training cells at `145.488 s/cell`: `1,745.854 s` (note: across `3` seeds, `3` full measured H57 real cells at `607.179 s` is `1,821.538 s`)
- `1` `fold-B` scoring cell charged conservatively at **one full measured H57 real cell (`607.179 s`)**
- Up to `3` terminal `DL-Val` read cells (1 per seed for an admitted arm) at `128.615 s`: `385.844 s`
- Subtotal: `2,738.877 s`
- `+10%` contingency: `273.888 s`
- **Projected Total GPU Cell Wall**: **`3,012.765 s` (`50.21 min`)** `<= 3,600 s` (`60.0 min` cap; slack `587.235 s`).

## 2. Machine-readable budget fields

total_gpu_cell_wall_cap_seconds: 3600
fixed_round_horizon: 20
fixed_steps_per_round: 2
measured_h57_real_cell_seconds: 607.1792591279373
measured_h57_generation_seconds: 60.113566709915176
projected_total_seconds: 3012.765

## 3. Red-team clarifications and gate consistency disclosure

1. **Admission gate vs. H61 point-gain margin (`m = 0.001546`) — Inconsistency Recorded, Gate Unchanged**:
   - `research/flywheel_selfplay/h59/PREREG.md` (§"Metrics, admission, and interpretation", line 46) fixed the H59 admission rule to three conditions on untouched `fold B`:
     1. `mean_lcb95_one_sided > 0.001`,
     2. `family_median_gain >= 0.0`, and
     3. `harm_fraction <= 0.10` (no more than 10% of families with seed-median gain `< -0.01`), with exact ties failing (`nonzero_family_count > 0` and `family_mean_gain > 0.0`).
   - Unlike `H61` (`PREREG.md` #761), `H59/PREREG.md` did **not** preregister an additional point-gain threshold `family_mean_gain > 0.001546` (`H61` `NULL_MARGIN_M`).
   - Per `~/nudges/agy_fw_audit_h59_coord_t2205.md` item `3(a)` ("add the H61 point-gain margin (gain > 0.001546) only if the prereg allows it; otherwise record the inconsistency in the amendment, do not silently change the gate"), `H59` preserves its preregistered three-condition strict gate (`admitted`) without silent modification, and explicitly records `h61_null_margin_m: 0.001546` and `clears_h61_null_margin_diagnostic: bool(family_mean_gain > 0.001546)` in every arm's admission receipt so any boundary case (`0.001 < family_mean_gain <= 0.001546` with `LCB95 > 0.001`) is flagged transparently.
2. **Bradley-Terry Elo vs `r882`, 3 seeds, family-grouped, median + 95% CI**:
   - `runner.py` and `outcomes/admission.json` report family-grouped Bradley-Terry Elo vs `r882` (`1500.0` anchor, never AUC) per seed (`12, 13, 14`) and pooled across seeds, along with W/T/L family counts, pooled family median + 95% CI, per-seed family median + 95% CI, and across-seed median + 95% CI (`seed_median_gain`, `seed_median_ci95`, `elo_median_across_seeds`, `elo_ci95_across_seeds`).
3. **Zero `DL-Val` / `fold-B` / `holdout` / `sealed` / `outer` exposure during training**:
   - `prepare_training` loads only `DL-Train` `fold_a` (`164` families, `1,001` contexts from `seed 4900`). Per-context scale normalization `training_slate_scale` is detached and computed strictly from the current `fold_a` training slate targets. `fold_b` is opened only once after all 12 training cells complete (`--mode fold-b`), and `DL-Val` is opened only for admitted arms (`terminal_read_allowed`).
4. **Disk and Lock Guards**:
   - `run_wave.py` acquires `~/jope_plan/qualification.lock` per cell, requires `statvfs('/')` used `< 94.5%` to **start** a cell (while keeping the writer guard `< 95.0%`), `MemAvailable >= 16 GiB`, `host CPU <= 85%`, and `CUDA free >= 6 GiB`. A write-time disk-guard failure is caught as `DiskGuardInfraVoid` and retries the **same cell** (up to `2` retries), never stopping the wave.
