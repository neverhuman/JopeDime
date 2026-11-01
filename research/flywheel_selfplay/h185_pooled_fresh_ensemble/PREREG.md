# H185 Preregistration: Power-Aware Pooled Confirmatory Test of Fresh-Init and Ensemble Candidates (`FRESH160`, `ENS5`, and `RENS5` vs `R882`)

- **Wave**: `H185`
- **Owner**: `gpu_tokbench_agy` (`[TOKBENCH-AGY]`)
- **Reserved by**: `ENG:hwave-loop`
- **Authority**: `/home/ubuntu/nudges/hwave_t0700_h185_tokagy.md` (`DECISIONS_20261029.md`)
- **Parent Studies**:
  - `H177` ([`research/flywheel_selfplay/h177_pooled_selector/`](../h177_pooled_selector/README.md), PR `#1492` / `#1500` / `#1507`) — canonical two-stage `205`-family pooled cross-fit selector (`H177_DECISION_RULE`, `pooled_b_select`, `StageOrderGuard`, `verify_pooled_b_coverage`, `resolve_h177_verdict`).
  - `H178` ([`research/flywheel_selfplay/h178_pooled_calibration/`](../h178_pooled_calibration/README.md), PR `#1512` / `#1535`, commit `6b70d803b3c51ca8cd46720244dbdfa872c08868`) — empirical calibration (`CALIBRATED_PASS_POWER_OK`, `0/20` null/harmful admissions, `24`-slate family-clustered `MDE80 = +0.0258` vs single-fold `+0.0640`, [`h178_receipt.json`](../h178_pooled_calibration/h178_receipt.json)).
  - `H179` (`research/flywheel_selfplay/h179_fresh_ensemble/`) — `5`-member fresh-init score ensemble (`ENS5`) and untrained `5`-member random-init ensemble control (`RENS5`) under the single-fold `41`-family `B_k` selector (`NO_ADVANCE`: `ENS5` unconditional `A` `+0.0282`, `RENS5` `+0.0273`).
  - `H180` (`research/flywheel_selfplay/h180_fresh_horizon/`) — `160`-update fresh-init horizon (`FRESH160`) under the single-fold `41`-family `B_k` selector (`NO_ADVANCE`: unconditional `A` `+0.0243`).
- **Execution Host**: `xbabe2` (`CPU-only`, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, `<= 4` threads, `RSS < 6 GB`, `load <= 85%`, `MemAvailable >= 16 GiB`, `disk < 93.8%` / `free >= 40 GB`)
- **Seeds & Partition Pins**:
  - Outer seeds: `SEEDS = (185101, 185102, 185103)`
  - `ENS5` / `RENS5` 5-member init seeds per outer seed:
    - `185101`: `(185201, 185202, 185203, 185204, 185205)`
    - `185102`: `(185301, 185302, 185303, 185304, 185305)`
    - `185103`: `(185401, 185402, 185403, 185404, 185405)`
  - Fresh cyclic `3/1/1` fold partition seed: `FOLD_SEED = 185140` ([`fold_manifest.json`](fold_manifest.json), `5` disjoint blocks of `41` families each = `205` `DL-Train` families, `1,219` contexts)
  - Paired family bootstrap seed: `BOOT_SEED = 185141` (`10,000` draws; `1,000`-draw Bradley–Terry MM Elo uses `BOOT_SEED + 1 = 185142`)

---

## 1. Scientific Motivation & Selection-Caveat Disclosure

In `H179` (`ENS5`, `RENS5`) and `H180` (`FRESH160`), fresh-initialized adapter candidates achieved raw cross-fit `A` gains of **`+0.0243` to `+0.0282`** (`+9.1` to `+11.6` Elo, `3/3` seeds positive) over `round-882.pt`, yet all arms closed `NO_ADVANCE` (`0` or `1/15` cells admitted) because the legacy single-fold `41`-family `B_k` selector has a `24`-slate family-clustered `MDE80` of **`+0.0640`** (`H175`).

In `H178` (`CALIBRATED_PASS_POWER_OK`, [`h178_receipt.json`](../h178_pooled_calibration/h178_receipt.json)), we calibrated the `H177` two-stage `205`-family pooled `B` selector (`H177_DECISION_RULE`) on the `236`-panel `H175` counterfactual bank and proved:
1. **Zero false admissions** (`0/20 = 0.0000 <= 0.05`) across all exact-null, noisy-null, and harmful panels under Holm multiplicity; and
2. **`2.48x` lower `24`-slate family-clustered `MDE80` (`+0.0258` vs `+0.0640`)**, with `85.0%` power at `+0.0257` and `100.0%` power at `+0.0321`.

This places the `+0.024–+0.028` effects observed in `H179` and `H180` directly inside the calibrated `205`-family pooled selector's detection band. `H185` executes a single clean confirmatory test of `FRESH160`, `ENS5`, and `RENS5` vs `R882` using the **exact imported `H177_DECISION_RULE`** (`m = 3` Holm across the three candidate arms) on fresh outer seeds (`185101–185103`), fresh member seeds (`185201–185405`), and a fresh `3/1/1` fold partition (`fold_seed = 185140`).

### Explicit Post-Hoc Candidate-Selection & Cross-Fold Dependency Disclosures
1. **Post-`H179`/`H180` Candidate Selection Caveat**: The three candidate arms (`FRESH160`, `ENS5`, and `RENS5`) were selected for `H185` after observing their unconditional `A` gains in `H179` and `H180` on the same `205`-family `DL-Train` cohort (under earlier fold and init seeds). Although `H185` uses fresh outer seeds (`185101–185103`), fresh member init seeds (`185201–185405`), and a fresh fold split (`fold_seed = 185140`), reuse of the `205`-family `DL-Train` pool after `H179`/`H180` carries residual cohort-level selection bias and does not constitute a new-cohort holdout test.
2. **Correlated Cross-Fold Candidate Readout (`Pooled A`)**: Within every fold $k \in \{0..4\}$, $\text{FIT}_k$ (`123` families), $B_k$ (`41` families), and $A_k$ (`41` families) are pairwise disjoint. Across all `5` folds, however, $\bigsqcup_{k=0}^4 B_k$ and $\bigsqcup_{k=0}^4 A_k$ span the **same `205` `DL-Train` families (`1,219` contexts)**, evaluated under different fold models ($k_B(f) = (m-1)\bmod 5$ vs $k_A(f) = m$, whose training splits share `82/123` fit families). Stage-2 pooled `A` is therefore reported strictly as a **correlated cross-fold candidate readout**, never as an independent sample split. `DL-Val` (`91` contexts / `19` disjoint families) is reported descriptively.

---

## 2. Registered Arms (`R882` Reference + `3` Candidate Arms, `m = 3` Holm)

All candidate arms keep the frozen `DOPE` trunk (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and non-adapter parameters (`8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`) bitwise identical to `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) and operate exclusively on the `29,211` trainable parameters (`47` tensors: `41` active gradients, `6` structural `None`-as-zero gradients) of `adapter_10tok`:

1. **`R882` (Reference, `0` Updates)**:
   - Untrained `round-882.pt` checkpoint (`DL-Val` context utility `1.1377941898164525`, anchored at `1500.0` Elo).
2. **`FRESH160` (`H180` Recipe — Single Fresh-Init Adapter at `160` Updates)**:
   - For each outer `(seed, fold)` cell, `adapter_10tok` is re-initialized from scratch via `fresh_init_adapter(net, base_state, seed)` (`torch.manual_seed(seed)`, executing each submodule's native `reset_parameters()` and resetting custom `nn.Parameter` tensors via standard Normal $\mathcal{N}(0, 0.02)$ / ones / zeros rules).
   - Trained for **`160` updates** (`PRESPECIFIED_CANDIDATE_STEP["FRESH160"] = 160`) on `FIT_k` only (`8` contexts $\times$ `24` candidates per update: `4` revealed top-fitness + `20` random unrevealed) using `AdamW(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)` and the `H156`/`H180` pairwise softplus rank loss (`grad_clip = 1.0`).
   - **Pre-Grid Timing Pilot & Deterministic Fallback Rule**: Before launching the `15`-cell grid, `runner.py` runs a `1`-cell `4`-update `FIT_0`-only timing pilot (`run_timing_pilot`) and projects total wall time across all `5,400` updates (`15` cells $\times$ `360` updates/cell) plus static evaluation overhead. If projected runtime exceeds the **`3.0 h` (`10,800 s`)** wall-clock cap, `FRESH160` deterministically falls back to **`80` updates** (`FRESH160_FALLBACK_UPDATES = 80`, `4,200` total updates) and records the fallback in `timing_pilot` and `pooled_b_selection_lock.json`. Outer seeds (`3`) and folds (`5`) are never reduced.
3. **`ENS5` (`H179` Recipe — `5`-Member Fresh-Init Score Ensemble at `40` Updates/Member)**:
   - For each outer `(seed, fold)` cell, `5` independent fresh adapter initializations are generated from `member_seeds_for(seed)` (`185201–185205` for `185101`, `185301–185305` for `185102`, `185401–185405` for `185103`).
   - Each member $m \in \{0..4\}$ is trained independently for **`40` updates** (`PRESPECIFIED_CANDIDATE_STEP["ENS5"] = 40`, `200` updates/cell across the `5` members) on `FIT_k` only under its own `batch_plan(fit_cids, orders, seed * 1000 + outer * 100 + m * 19 + 7, updates=40)` with `AdamW(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)`.
   - At evaluation time, the `5` members' predicted candidate cost vectors on each unrevealed candidate slate (`cands[4:]`) are **averaged before `argmin`** (`ensemble_member_scores`: $\bar{s}(c) = \frac{1}{5}\sum_{m=1}^5 s_m(c)$, picking $\arg\min_{j \ge 4} \bar{s}_j(c)$).
4. **`RENS5` (`H179` Untrained Random-Structure Control — Exact Same `5` Member Inits at `0` Updates)**:
   - Uses the **exact same `5` fresh-initialized member states** per `(seed, fold)` cell as `ENS5` at **`0` optimizer updates** (`PRESPECIFIED_CANDIDATE_STEP["RENS5"] = 0`, verified bit-identical before training via `validate_rens5_matches_ens5_inits`), averaging the `5` untrained members' candidate cost vectors before `argmin`.
   - Tests whether `ENS5`'s gain over `R882` comes from `FIT_k` gradient updates (`ENS5 - RENS5`) or from untrained 5-member random-projection score averaging (`RENS5 - R882`).

---

## 3. Imported Two-Stage Pooled Confirmatory Selector (`H177_DECISION_RULE`)

`H185` does **not** reimplement the pooled selector math. [`analysis.py`](analysis.py) directly imports `H177_DECISION_RULE`, `pooled_b_select`, `StageOrderGuard`, `verify_pooled_b_coverage`, `resolve_h177_verdict`, `holm`, `family_matrix`, `bt_ratings`, and `family_wins` from [`research/flywheel_selfplay/h177_pooled_selector/analysis.py`](../h177_pooled_selector/analysis.py) (`SHA-256 = 707c4903a77d48ffc8ae14c820a626df5c2858d3d24df35db3f9718d120bafc4`, verified against [`h178_receipt.json`](../h178_pooled_calibration/h178_receipt.json)).

### Stage 1: Fit-Only Training, Pooled `B` Selection (`205` Families), & `fsync` Lock
1. For each fold $k \in \{0..4\}$ and outer seed $s \in \{185101, 185102, 185103\}$, train `FRESH160` and the `5` `ENS5` members on $\text{FIT}_k$ (`123` families) only, and score `R882`, `FRESH160`, `ENS5`, and `RENS5` on $B_k = F_{(k+1)\bmod 5}$ (`41` families) only.
2. Pool $B_0 \sqcup B_1 \sqcup B_2 \sqcup B_3 \sqcup B_4$ across $k = 0..4$ to form the `205`-family (`1,219`-context) pooled `B` panel (`verify_pooled_b_coverage` enforces exact-once coverage and zero overlap with $\text{FIT}_k$ or $A_k$).
3. Pass the `[3, 1219]` pooled `B` panels (`R882` as reference + `FRESH160`, `ENS5`, `RENS5` as the `3` treatment arms) into `H177_DECISION_RULE.select` (`pooled_b_select_h185`, `boot_seed = 185141`, `10,000` paired family-bootstrap draws, within-family 3-seed mean prior to family resampling, and $m = 3$ Holm step-down across `FRESH160`, `ENS5`, `RENS5` via `admitted_treatments3` / `holm_p_treatments3`).
4. **Stage-1 Admission Criterion ($m = 3$ Holm)**:
   - Candidate arm $a \in \{\text{FRESH160}, \text{ENS5}, \text{RENS5}\}$ is admitted (`admitted = True`) iff:
     $$\hat{\mu}^{(B)}(a) > 0.001 + 10^{-12} \quad\text{AND}\quad \text{LCB}_{95}^{(B)}(a) > 0.0 \quad\text{AND}\quad \text{Holm-}p_{m=3}^{(B)}(a) < 0.05.$$
   - Ties and boundary values strictly reject (`admitted = False`, falling back to `R882` in the selected policy).
5. **Hard Stage-Order File-System Barrier (`pooled_b_selection_lock.json`)**:
   - Before any `A_k` or `DL-Val` context is scored on any candidate arm, `runner.py` writes and `os.fsync`s `pooled_b_selection_lock.json` and locks `StageOrderGuard`.

### Stage 2: Pooled `A` Confirmation (`205` Families), Descriptive `DL-Val` (`19` Families), & Contrasts
1. After `StageOrderGuard.require_b_locked()` succeeds, score `FRESH160`, `ENS5`, and `RENS5` on each fold's disjoint $A_k = F_k$ (`41` families $\times 5 = 205$ families, `1,219` contexts) and on `DL-Val` (`19` families, `91` contexts).
2. Report for both the **Selected Policy** (using the candidate if admitted in Stage 1, else `R882`) and the **Unconditional Candidate Panel** on Pooled `B`, Pooled `A`, and `DL-Val`:
   - Per-seed family-mean utility gains (`185101, 185102, 185103`),
   - `3`-seed family-mean utility gain and `10,000`-draw paired family bootstrap 95% CI,
   - Median seed gain and `axis=0` seed-level bootstrap 95% CI (`np.median(draws, axis=0)`),
   - One-sided bootstrap $p$-value and $m = 3$ Holm-adjusted $p$-value,
   - Canonical `H148` 400-iteration Bradley–Terry MM Elo vs `r882 = 1500.0` (`1,000` family-bootstrap draws, `seed = 185142`),
   - Exact `-2.0` failure pick counts (`count_minus_two_picks`).
3. Report active diagnostic contrasts on Pooled `B`, Pooled `A`, and `DL-Val`:
   - **`ENS5 - RENS5`** (isolates the value of `40` gradient updates per ensemble member beyond untrained 5-member random-init averaging),
   - **`FRESH160 - RENS5`** (single `160`-step trained fresh adapter vs untrained 5-member random ensemble),
   - **`ENS5 - FRESH160`** (`5 x 40 = 200`-update fresh ensemble vs `1 x 160`-update single fresh adapter).

### Final Confirmatory Verdict Rule (`resolve_h185_verdict`)
- Bound to committed [`h178_receipt.json`](../h178_pooled_calibration/h178_receipt.json) via `resolve_h177_verdict(candidate_admitted, h178_receipt)`:
  - **`ADVANCE`** iff at least one arm $a \in \{\text{FRESH160}, \text{ENS5}, \text{RENS5}\}$ is admitted on Stage-1 pooled `B` (`admitted = True`), `resolve_h177_verdict` returns `"ADVANCE"`, **AND** that admitted arm achieves strictly positive Stage-2 pooled `A` gain ($\hat{\mu}^{(A)}(a) > 10^{-12}$) with **`3/3` outer seeds strictly positive** (`positive_seeds == 3`; ties reject).
  - **`NO_ADVANCE`** otherwise.
- **Serving Boundary**: Never adopt any `H185` checkpoint into serving; if the verdict is `ADVANCE`, flag `@flywheel_gp_fused` in `AGENT_CHAT.md`.

---

## 4. Data-Free Unit Verification (`test_h185.py`)

`pytest -q research/flywheel_selfplay/h185_pooled_fresh_ensemble/test_h185.py` verifies all 6 required pre-launch properties without touching `DL-Train` or `DL-Val` data:
1. **`FIT`-only training**: `batch_plan` and `run_cell` hard-refuse any training context outside `FIT_k`.
2. **`B` lock before `A`**: `StageOrderGuard` hard-refuses Stage-2 `A` or `DL-Val` scoring before `pooled_b_selection_lock.json` is written and `fsync`ed, and hard-refuses candidate step mutation after `B` reading starts.
3. **`RENS5` inits == `ENS5` inits**: `member_seeds_for` and `validate_rens5_matches_ens5_inits` enforce bit-identical 5-member initializations per `(seed, fold)` cell and reject any mismatch.
4. **Distinct member init hashes**: `validate_member_init_hashes` and `validate_cell_init_hashes` verify `5` distinct member hashes per outer seed and `15` distinct `(FRESH160, ENS5, RENS5)` init states across all `15` cells.
5. **Imported `H177` decision rule & `H178` calibration receipt**: `is` identity check on `H177_DECISION_RULE`, `pooled_b_select`, and `resolve_h177_verdict` from `h177_pooled_selector/analysis.py`, exact zero null admissions, positive shift admission, boundary tie rejection, and `resolve_h185_verdict` against committed `h178_receipt.json`.
6. **`axis=0` bootstrap seed reduction & `fold_seed = 185140` coverage**: heterogeneous-variance synthetic control proving `np.median(seed_draws, axis=0)` (`[3, B] -> [B]`) and `verify_pooled_b_coverage` on `fold_seed = 185140` (`123/41/41`, zero overlap, exact-once `205`-family coverage).
