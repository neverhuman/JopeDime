# H455 (104M ICL-GP Flywheel): Bitwise-Exact Fast Multi-Arm `2xCV-7` Evaluator, Cooled Wave-2A / 2B / 2A+ `dev91` Progression, and Max-Entropy `s3_source_uniform` Flywheel

## Scope and Quarantine Disclosure

- **In-corpus disclosure (`in_corpus_dev91_disclosure = true`, `canonical_promotion_forbidden = true`):** All `104M` TabPFN-style In-Context Learning (`ICL`) checkpoints in the `H450 -> H454 -> H455` flywheel lane are trained on the 3,525-table `mmap_v1` bank (`2,389` S3 source families), which includes shuffled `dev91` cousin tables (`d91_share ≈ 0.019–0.058`, `0` audit labels `ya` in `bank_d91_search_f64.npz` and `0` `ya` in `gp_teacher_shards_v1`). All `dev91` BT-Elo and utility numbers below are strictly **in-corpus random-split diagnostics** against the frozen 33-model `dev91-audit-v1` board and are ineligible for canonical promotion.
- **Strict quarantine:** `0` opens of `CONFIRM`, `FMGAP B/C`, `HOLDOUT`, or `SEALED` partitions; `0` file deletions (`unlink`, `rmtree`, `rmdir`, `remove`, or `rsync --delete`); `0` `/tmp` paths. Verified via AST and filesystem checks in `selftest_h455.py` (`7/7 PASS`).

## 1. Fast Multi-Arm `eval-cv7` Evaluator & Context-0 Bitwise Equivalence

Single-process `eval-cv7` previously required ~135 minutes per 91-context `2xCV-7` arm on an RTX 4090 because each run recomputed checkpoint-independent nested OOF GP/M7/K3/F4 anchors (`~14–22s/ctx`), ran serial single-core `KD` + `CW-2` formula screening across 14 folds, and reloaded `.pt` checkpoints per invocation.

`cols_104m_e8_gpresid_h452.py` and `watch_and_score_wave2_h454.py` were upgraded with three strictly score-preserving optimizations:
1. **Disk-cached OOF GP anchors (`--gp-oof-cache-dir`):** Caches `(q_anc, bag_anc, ref_anc, usable)` per context (`91/91` `.npz` files, keyed purely on `SEARCH` features/labels and non-`ya` anchor pools).
2. **Multiprocess CPU fold-prep prefetch (`--prep-workers`):** Parallelizes all 14 folds + full-refit `KD` / `CW-2` view construction and Yeo-Johnson / Rank-Gauss transforms across CPU worker processes ahead of the GPU loop.
3. **Single-pass multi-checkpoint / multi-EMA evaluation (`--multi-specs`):** Deduplicates `.pt` loads on CPU and evaluates multiple `(checkpoint_pair, ema_key)` arms on every prepped fold in a single pass.

**Mandatory Pre-Cutover Gate (`CTX0_EQUIVALENCE_RECEIPT.json`):** Before replacing the single-process evaluator, context `0` (`016648dc8e6f474c5554815f87b12779cdb83966ce57ce52899da8847347c6da`) was evaluated on the `xbabe1` RTX 4090 under `--multi-specs` + `--prep-workers 4` both with a cold cache and with `--gp-oof-cache-dir` warm, and compared against the pre-existing single-process shard-0 `.npz` outputs across all 12 arrays (`{q,bag,ref}_icl_e8_{cw12_robridge,cw12_yj,rankgauss_duan,gp_resid}_cv7`):
- `w2a_soup50` (uncached `--multi-specs` vs single-process shard 0): **`max_abs_diff = 0.0`** (all 12 arrays `0.0`)
- `w2a_cooldown` (uncached `--multi-specs` vs single-process shard 0): **`max_abs_diff = 0.0`** (all 12 arrays `0.0`)
- `w2a_soup50` (cached `--gp-oof-cache-dir` vs single-process shard 0): **`max_abs_diff = 0.0`** (all 12 arrays `0.0`)

All 5 cooled `Wave-2` arms (`w2a_soup50`, `w2a_cooldown`, `w2b_lr5e6_ema_soup50`, `w2aplus_soup50`, `w2aplus_cooldown`) then completed `91/91` contexts in a single `--multi-specs` pass while preserving all previously written `.npz` files untouched.

## 2. Cooled `Wave-2` In-Corpus `dev91` Progression (`Wave-1` -> `Wave-2A` / `2B` / `2A+` / `Cooled Soup`)

Checkpoints evaluated (`fleet_L_x1` on `xbabe1` RTX 4090 + `fleet_L_x4` on `xbabe4` MPS, trained with Exact-GP KD + Exact-GP Residual + Counterfactual weighting):
- **`Wave-1` (uncooled, `lr = 2.0e-5`):** `H454_PAIRED_FLYWHEEL_MATRIX.json`
- **`Wave-2A` (`lr ≈ 1.07e-5` / `2.0e-6`, `17,843` Exact-GP datasets):** `fleet_L_x1_w2a.pt` (`d6d51f2d9857...`) + `fleet_L_x4_w2a.pt` (`67ca27100c9f...`)
- **`Wave-2B` (`lr <= 5.0e-6` / `2.0e-6`):** `fleet_L_x1_h454_w2b_lr5e6.pt` (`ac7083bb1048...`) + `fleet_L_x4_h454_w2b_lr5e6.pt` (`893472e23263...`)
- **`Wave-2A+` (`lr ≈ 3.26e-6` / `2.0e-6`, `42,729` combined Exact-GP datasets):** `fleet_L_x1_w2aplus.pt` (`20b5c5e8364b...`) + `fleet_L_x4_w2aplus.pt` (`c2b3b258ed9f...`)
- **`Wave-2 Cooled Soup Avg`:** Equal-weight prediction-space average across `w2a_soup50`, `w2b_lr5e6_ema_soup50`, `w2aplus_soup50`, and `w2aplus_cooldown` (`0` `ya` read), scored with 6-trajectory `traj_e8_w2aplus`.

| Wave / Checkpoint Regime | `pure_104m_only` Elo (Post vs Pre `Δu`) | `2-Tower 104M+UGP-g1000` `Δu` vs Pre (`lofo_min`) | `stack_no_ugp_post` Elo (Rank/34) | `stack_ugp_g1000_cons_post` Elo (`fam_u`) | `stack_ugp_g1000_cons_joint` Elo (`fam_u`, `p` vs Pre) | `stack_guarded_hyb_joint` vs Pre (`Δu`, `signflip p`) |
| --- | --- | --- | --- | --- | --- | --- |
| **`Pre-H454` Baseline** | `1582.90` (ref) | `1.6953` (ref) | `1858.33` (#5) | `1858.33` (`1.7170`) | — | — |
| **`Wave-1` (`lr=2.0e-5`, uncooled)** | `1575.60` (`-0.00508`) | `+0.00079` (`-0.00044`) | `1855.77` (#6) | `1855.77` (`1.7169`) | `1854.96` (`1.7181`, `p=0.2897`) | `-0.00125` (`p=0.7997`) |
| **`Wave-2A soup50` (`lr≈1.07e-5`)** | `1581.31` (`-0.00016`) | `+0.02116` (`+0.00775`) | **`1859.63` (#5)** | `1857.99` (`1.7139`) | `1858.16` (`1.7223`, `p=0.0775`) | **`+0.00214` (`p=0.0216`)** |
| **`Wave-2A cooldown` (`lr≈1.07e-5`)** | `1583.06` (`+0.00016`) | **`+0.02132` (`+0.00801`)** | **`1859.63` (#5)** | `1858.36` (`1.7140`) | `1858.16` (`1.7224`, `p=0.0785`) | **`+0.00233` (`p=0.0122`)** |
| **`Wave-2B soup50` (`lr<=5.0e-6`)** | **`1590.81`** (`+0.00678`) | `+0.01988` (`+0.00631`) | `1859.44` (#5) | `1858.20` (`1.7137`) | `1858.55` (`1.7220`, `p=0.0781`) | `+0.00166` (`p=0.0801`) |
| **`Wave-2A+ soup50` (`lr≈3.26e-6`)** | `1586.45` (`+0.00568`) | `+0.02074` (`+0.00705`) | `1858.12` (#6) | `1857.68` (`1.7138`) | `1858.31` (`1.7222`, `p=0.0812`) | `+0.00166` (`p=0.1330`) |
| **`Wave-2A+ cooldown` (`lr≈3.26e-6`)** | `1585.14` (**`+0.00743`**) | `+0.02045` (`+0.00669`) | `1858.21` (#6) | `1857.17` (`1.7137`) | `1856.27` (`1.7218`, `p=0.1033`) | `+0.00155` (`p=0.1754`) |
| **`Wave-2 Cooled Soup Avg`** | `1588.19` (`+0.00623`) | `+0.02091` (`+0.00730`) | `1859.08` (#5) | **`1858.64`** (`1.7141`) | **`1858.88`** (`1.7223`, `p=0.0646`) | **`+0.00204` (`p=0.0353`)** |

### Key Findings from the Cooldown Curve
1. **Cosine cooldown reverses the uncooled `Wave-1` step noise on pure `104M` (`0` GP at test time):** Pure `104M` (`pure_104m_only_cv7`) flips from `-0.00508` utility / `1575.60` Elo at `lr = 2.0e-5` (`Wave-1`) to **`+0.00678` utility / `1590.81` Elo (`+15.21` Elo)** at `Wave-2B` (`lr <= 5.0e-6`) and **`+0.00743` utility** at `Wave-2A+ cooldown` (`lr ≈ 3.26e-6`), with `pure_104m_cv7` winning `13W-0T-6L` families over `Pre-H454`.
2. **27x amplification in Two-Tower `104M + Gen-1000 Ultra-GP` synergy:** Without any legacy `M7/K3/F4/Tab` pools, `neural104m_ugp_g1000_cons_post_vs_pre` jumps from `+0.00079` (`Wave-1`) to **`+0.02132` (`lofo_min = +0.00801 > 0` across all 19 family drops)** in `Wave-2A cooldown`, reaching `1823.47` Elo (#7 of 34).
3. **Guarded stack lift reaches `p < 0.05` and beats `tabpfn_v3p5`:** `stack_no_ugp_post_h454` reaches **`1859.63` Elo** (Rank #5 of 34, `+0.75` Elo ahead of `tabpfn_v3p5` and `+1.30` Elo ahead of `Pre-H454`), while multi-checkpoint souping (`Wave-2 Cooled Soup Avg`) lifts `stack_ugp_g1000_cons_post_h454` to **`1858.64` Elo** (`+0.0336` utility over `ctrl_h450_g16_legacy`, `signflip p = 0.0323`) and `stack_ugp_g1000_cons_joint_pre_post` to **`1858.88` Elo / `1.7223` utility** (`11W-2T-6L`, `lofo_min = +0.00177 > 0`), with `stack_guarded_hyb_ugp_g1000_cons_joint_pre_post` beating `Pre-H454` at **`signflip p = 0.0122` (`Wave-2A cooldown`)** and **`p = 0.0353` (`Wave-2 Cooled Soup Avg`)**.

## 3. H455 Max-Entropy `s3_source_uniform` Flywheel & GPU Fill (`xbabe1` & `xbabe2`)

- **H455 Max-Entropy Trainer (`train_104m_gp_flywheel_h455_maxent.py` + `mmap_bank_h450.py`):**
  - `s3_source_uniform` inverse-family-size weighting across all `2,389` S3 families / `3,525` tables (`d91_random_split_boost = 0.00`, `regret_tilt_prob = 0.05`, hitting `2,389/2,389` families in a 90k sample draw).
  - Cross-shard shape-binned Exact-GP microbatches (`kd_ds_per_batch ≈ 7.3–10.5` datasets/microbatch vs `1.0` in H454 — a **7–10x** GPU utilization jump on Exact-GP teacher batches).
  - Wider entropy ranges (`feature_drop ∈ [0.00, 0.85]` with `14%` full-feature episodes, `synth_inj ∈ [0.00, 0.60]`, `distractor_prob = 0.35`, `p_ctx ∈ [0.08, 0.95]`, per-batch random split, endless 2.5h cosine warm-restart cycles with valley snapshotting).
- **GPU Fill & Diagnosis:**
  - **`xbabe2` RTX 3090 (`0% -> 100%` GPU-Util):** Diagnosed `0%` GPU-Util with `18.5 GiB` held as `cl_fwriter`'s `r4_L_A` trainer (`15.49 GiB`, single-process GIL-bound CPU prep between short CUDA bursts) plus intermittent `cols_h450_gpu.py` (`3.01 GiB`, ~80% CPU ridge/struct prep). Launched bounded `fleet_M_x2_h455_maxent` (`--size M`, `--cuda-vram-frac 0.12` = `2.82 GiB` used, `nice -n 19`, `taskset -c 0-31`), lifting `xbabe2` RTX 3090 to **`76–100%` GPU-Util (`337 W`)** at `~69.6 datasets/s` (`620,659` unique datasets, `278,564` Exact-GP datasets, `cycle 1` active) while `r4_L_A` continues uninterrupted (`oom = 0`) and `cl_fwriter`'s `pub-r15` eval worker shares the card via its `>= 7400 MiB` free-VRAM gate + `xbabe3` offload.
  - **`xbabe1` RTX 4090 (`100%` GPU-Util):** Launched co-resident `fleet_L_x1_h455_maxent_v2` (`--size L`, `--cuda-vram-frac 0.36`, `--tokens 96000`, `oom = 0`, warm-started from `fleet_L_x1_w2aplus.pt`) alongside uninterrupted `fleet_L_x1` (`step 612,681+`), holding `xbabe1` RTX 4090 at **`100%` GPU-Util (`23.8 GiB / 24.5 GiB`, `308.8 W`)**, with upgraded `watch_and_score_wave2_h454.py` polling for `ckpt_valley_NNN.pt` milestones.
