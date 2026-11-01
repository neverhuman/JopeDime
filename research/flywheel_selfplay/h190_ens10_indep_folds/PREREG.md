# H190 Preregistration: `ENS10` Confirmatory Test with Independent Family-Fold Partitions per Outer Seed (`190141`, `190142`, `190143`)

- **Wave**: `H190`
- **Owner**: `gpu_fused_agy` (`[FUSED-AGY]`)
- **Reserved by**: `ENG:hwave-loop` (`/home/ubuntu/jope_plan/ns/receipts/H190_claim.txt`)
- **Authority**: `/home/ubuntu/nudges/hwave_t0835_h190.md` (`DECISIONS_20261029.md`)
- **Parent Studies**:
  - `H177` ([`research/flywheel_selfplay/h177_pooled_selector/`](../h177_pooled_selector/README.md)) — canonical two-stage `205`-family pooled cross-fit selector (`H177_DECISION_RULE`, `pooled_b_select`, `StageOrderGuard`, `verify_pooled_b_coverage`, `resolve_h177_verdict`).
  - `H178` ([`research/flywheel_selfplay/h178_pooled_calibration/`](../h178_pooled_calibration/README.md), [`h178_receipt.json`](../h178_pooled_calibration/h178_receipt.json)) — empirical calibration of `H177_DECISION_RULE` (`CALIBRATED_PASS_POWER_OK`, `0/20` null/harmful admissions, `24`-slate family-clustered `MDE80 = +0.0258`).
  - `H179` ([`research/flywheel_selfplay/h179_deep_ensemble/`](../h179_deep_ensemble/README.md)) — `5`-member fresh-init ensemble (`ENS5`, `+0.0137` median gain, `3/3` seeds positive) under a single shared fold partition.
  - `H182` ([`research/flywheel_selfplay/h182_ensemble_size_dose/`](../h182_ensemble_size_dose/README.md)) — nested `ENS5`/`ENS10`/`ENS20` dose sweep (`ENS10` `+0.0152`, `ENS20` `+0.0168`, gain flat after `K=10`) under a single shared fold partition (`fold_seed = 182140`).
  - `H185` ([`research/flywheel_selfplay/h185_pooled_fresh_ensemble/`](../h185_pooled_fresh_ensemble/README.md)) — pooled confirmatory runner sharing a single fold partition (`fold_seed = 185140`).
- **Execution Host**: `xbabe1` in `/home/ubuntu/lane_work/h190-agy` (`CPU-only`, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, `<= 8` threads, `RSS < 8 GB`, `load <= 85`, `MemAvailable >= 16 GiB`, `free >= 80 GB`, output dir `< 2 GB`: score panels + final member states only, NO per-step checkpoints)
- **Seeds & Independent Partition Pins**:
  - Outer seeds: `SEEDS = (190101, 190102, 190103)`
  - **Independent `fold_seed` per outer seed** ([`fold_manifest.json`](fold_manifest.json)):
    - `190101` $\to$ `fold_seed = 190141`
    - `190102` $\to$ `fold_seed = 190142`
    - `190103` $\to$ `fold_seed = 190143`
    - Each outer seed generates its **own** fresh cyclic `3/1/1` family fold partition (`5` disjoint blocks of `41` families = `205` `DL-Train` families, `1,219` contexts; `123` `FIT` / `41` `B` / `41` `A` families per outer fold `0..4`).
  - Paired family bootstrap seed: `BOOT_SEED = 190150` (`10,000` draws; `1,000`-draw Bradley–Terry MM Elo uses `BOOT_SEED + 1 = 190151`)
  - Disjoint `10`-member init seeds per outer seed (mutually distinct and disjoint from `H179`, `H182`, and `H185`):
    - `190101`: `(190201, 190202, 190203, 190204, 190205, 190206, 190207, 190208, 190209, 190210)`
    - `190102`: `(190301, 190302, 190303, 190304, 190305, 190306, 190307, 190308, 190309, 190310)`
    - `190103`: `(190401, 190402, 190403, 190404, 190405, 190406, 190407, 190408, 190409, 190410)`

---

## 1. Scientific Motivation: Eliminating Shared-Partition Pseudo-Replication

Across the recent H-waves, fresh-adapter score ensembles are the only lever showing a consistent positive signal over `round-882.pt`: `H179` (`ENS5` `+0.0137`, `3/3` seeds positive) and `H182` (`ENS10` `+0.0152`, `ENS20` `+0.0168` with 95% CI `[-0.0153, +0.0505]`, where ensemble gain plateaued by $K = 10$).

However, both `H182` (`fold_seed = 182140`) and `H185` (`fold_seed = 185140`) used **a single shared 5-fold family partition across all 3 outer seeds**. Under a shared fold partition, every family $f$ is assigned to the exact same `FIT`/`B`/`A` neighborhood in all three outer seeds (leaving `R882` utility identical across seeds and restricting between-seed variation to member initialization and `FIT` minibatch order only). Consequently, `"3/3 seeds positive"` in `H182` is pseudo-replicated over member-init/minibatch order, and single-partition inference understates `FIT`-neighborhood partition variance.

`H190` executes the clean confirmatory test of **`ENS10` vs `RENS10`** where **each outer seed (`190101, 190102, 190103`) uses an independent 5-fold family partition (`fold_seeds 190141, 190142, 190143`)**, directly measuring the partition-variance component alongside the `H177`/`H178` two-stage pooled cross-fit selector.

---

## 2. Registered Arms (`R882` Reference + `2` Candidate Arms, Holm $m = 2$)

All candidate arms keep the frozen `DOPE` trunk (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and non-adapter parameters (`8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`) bitwise identical to `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) and train exclusively the `29,211` parameters (`47` tensors: `41` active gradients, `6` structural `None`-as-zero gradients) of `adapter_10tok`:

1. **`R882` (Reference, `0` Updates)**:
   - Untrained private read-only copy of `round-882.pt` (`DL-Val` context utility `1.1377941898164525`, anchored at `1500.0` Elo).
2. **`ENS10` (Primary Candidate — `10` Fresh-Init Members $\times$ `40` Updates on `FIT_{s, k}` Only)**:
   - In each outer `(seed s, fold k)` cell under seed $s$'s independent partition `fold_seed_for(s)`, `10` independent fresh adapter initializations are constructed via `random_adapter_state(net, names, member_seed)` using `member_seeds_for(s)` (`190201–190210`, `190301–190310`, `190401–190410`).
   - Each member $m \in \{1..10\}$ is trained independently for **`40` updates** (`PRESPECIFIED_CANDIDATE_STEP["ENS10"] = 40`, `400` updates/cell, `6,000` updates total across `15` cells / `150` member trajectories) on `FIT_{s, k}` (`123` families) only (`8` contexts $\times$ `24` candidates per update: `4` revealed top-fitness + `20` random unrevealed) using `AdamW(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)` and `grad_clip = 1.0`.
   - At evaluation time, the `10` members' predicted candidate cost vectors on each context are **averaged before `argmin`** (`score_ensemble`: $\bar{c}(x) = \frac{1}{10}\sum_{m=1}^{10} c_m(x)$, picking $\arg\min_j \bar{c}_j(x)$).
3. **`RENS10` (Untrained Control — Exact Same `10` Fresh-Init Members at `0` Updates)**:
   - Uses the **exact same `10` fresh-initialized member states** per `(seed s, fold k)` cell as `ENS10` at **`0` optimizer updates** (`PRESPECIFIED_CANDIDATE_STEP["RENS10"] = 0`, verified bit-identical before training via `validate_rens_matches_ens_inits`), averaging the `10` untrained members' candidate cost vectors before `argmin`.
   - Tests whether `ENS10`'s gain over `R882` comes from `FIT_{s, k}` gradient training (`ENS10 - RENS10`) or from untrained 10-member random-init score averaging (`RENS10 - R882`).

### Pre-Grid Timing Pilot & Deterministic Fallback Rule (`ENS5 + RENS5` if Projected $> 2.5\text{ h}$)
- On the first `4` updates of member `1` in cell `(seed=190101, fseed=190141, outer=0)` **before** any `B` or `A` pool is scored, `runner.py` measures median `seconds_per_update` and calls `execution_plan(seconds_per_update, cap_seconds=9000, reserve_seconds=900)`.
- If projected wall time for `6,000` updates (`15` cells $\times 10$ members $\times 40$ updates) plus `900 s` evaluation reserve is $\le 9,000\text{ s}$ (`2.5 h`), `active_ensemble_k = 10` (`ENS10` + `RENS10`, `6,000` total updates) is retained.
- If projected wall time exceeds `9,000 s` (`2.5 h`), the runner deterministically falls back to **`active_ensemble_k = 5`** (`ENS5` + `RENS5`, using the first `5` prespecified member seeds per outer seed, `3,000` total updates) and locks that choice in `pilot_budget.json` and `pooled_b_selection_lock.json` before any `B` or `A` evaluation. Outer seeds (`3`) and folds (`5`) are never reduced.

---

## 3. Imported Two-Stage Pooled Confirmatory Selector & Verdict Rule

[`analysis.py`](analysis.py) directly imports `H177_DECISION_RULE`, `pooled_b_select`, `StageOrderGuard`, `verify_pooled_b_coverage`, `resolve_h177_verdict`, `holm`, `family_matrix`, `bt_ratings`, `family_wins`, `DRAWS`, and `EPS` from [`research/flywheel_selfplay/h177_pooled_selector/analysis.py`](../h177_pooled_selector/analysis.py) (`SHA-256 = 707c4903a77d48ffc8ae14c820a626df5c2858d3d24df35db3f9718d120bafc4`, verified against [`h178_receipt.json`](../h178_pooled_calibration/h178_receipt.json)).

### Stage 1: Independent-Fold `FIT` Training, Pooled `B` Selection (`205` Families), & `fsync` Lock
1. For each outer seed $s \in \{190101, 190102, 190103\}$ with independent `fold_seed` $f_s \in \{190141, 190142, 190143\}$ and each fold $k \in \{0..4\}$, train the `10` `ENS10` members on $\text{FIT}_{s, k}$ (`123` families) only and score `R882`, `ENS10`, and `RENS10` on $B_{s, k} = F_{s, (k+1)\bmod 5}$ (`41` families) only.
2. For each seed $s$, $\bigsqcup_{k=0}^4 B_{s, k}$ covers all `205` `DL-Train` families (`1,219` contexts) **exactly once** (`verify_independent_fold_partitions`).
3. Delegate Stage-1 selection to `H177_DECISION_RULE.select` via `pooled_b_select_h190` (`boot_seed = 190150`, `10,000` paired family bootstrap draws, margin unchanged $> +0.001 + 10^{-12}$, `LCB95 > 0`, and **Holm $m = 2$** adjusted $p < 0.05$ across `ENS10` and `RENS10`).
4. Write and `os.fsync` `pooled_b_selection_lock.json` via `StageOrderGuard` **before** any $A_{s, k}$ or `DL-Val` context is scored.

### Stage 2: Pooled `A` Confirmation (`205` Families), Descriptive `DL-Val` (`19` Families), & Partition-Variance Decomposition
1. After `StageOrderGuard.require_b_locked()` succeeds, score `R882`, `ENS10`, and `RENS10` on each seed's disjoint $A_{s, k} = F_{s, k}$ (`41` families $\times 5 = 205$ families, `1,219` contexts) and on `DL-Val` (`91` contexts / `19` families).
2. Report:
   - Per-seed gains (`190101`, `190102`, `190103`),
   - Median seed gain + `10,000`-draw paired family bootstrap 95% CI (`axis=0` over seeds), plus 3-seed within-family mean gain + `10,000`-draw paired family bootstrap 95% CI,
   - Holm $m = 2$ adjusted $p$-values (`ENS10`, `RENS10`) and active mechanism contrast (`ENS10 - RENS10`),
   - Canonical `H148` 400-iteration Bradley–Terry MM Elo vs `r882 = 1500.0` (`1,000` family-bootstrap draws, `seed = 190151`),
   - Exact `-2.0` failure pick counts per seed and arm,
   - Pairwise member-choice disagreement rate (`45` pairs among `10` members per context) on Pooled `B` and Pooled `A` for `ENS10` and `RENS10` (plus nested `K=5` prefix),
   - **Partition-variance component** (`partition_variance_decomposition`: between-seed SD/variance/range across the 3 independent fold partitions and within-family between-seed variance $\frac{1}{205}\sum_f \text{Var}_s(D_{s, f})$, compared directly against `H182`'s shared-partition `ENS10` spread).

### Confirmatory Verdict Rule (`resolve_h190_verdict`)
- **`ADVANCE`** iff an arm $a \in \{\text{ENS10}, \text{RENS10}\}$ is:
  1. Admitted on Stage-1 pooled `B` (`mean_gain > 0.001 + 1e-12`, `ci95[0] > 0.0`, Holm $m=2$ `holm_p < 0.05`), **AND**
  2. Confirmed on Stage-2 pooled `A` with `mean_gain > 1e-12` and `median_gain > 1e-12`, **`3/3` independent-partition seeds strictly positive** (`positive_seeds == 3`), **AND median CI lower bound $> 0$** (`median_seed_ci95[0] > 0.0` and `ci95[0] > 0.0`), **AND**
  3. Qualified under the committed [`h178_receipt.json`](../h178_pooled_calibration/h178_receipt.json) calibration gate via `resolve_h177_verdict`. Ties ($\le +0.001$ on `B` or $\le 0$ on `A`) reject.
- **`NO_ADVANCE`** otherwise.
- **Serving Boundary**: Do NOT adopt into serving; on `ADVANCE`, write up and flag `@flywheel_gp_fused` in `AGENT_CHAT.md` for promotion review with DOPE checksum verification.
