# H450 Fleet Extensions (`research/flywheel_selfplay/h450_fleet_agy/`)

Additive, non-overlapping H450 5-host fleet components coordinated with `cl_fwriter` via
`/home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md` and `/home/ubuntu/jope_plan/H450_FLEET_COORDINATION.md`
(staged read-only under `/home/ubuntu/h450-stage/fleet-r7`):

1. `mmap_bank_h450.py` — Zero-copy `np.memmap` drop-in replacement (`MmapF64Bank`) for `CombinedF64Bank`
   (`< 0.5 s` init, `< 80 MiB` private RSS), plus `--cmd convert` and `--cmd verify-parity` (verified bit-identical
   across all `3,525` tables / `164,651,987` IEEE-754 `uint64` cells against `everything_combined_f64_h450_bank.npz`
   `9dfc60ed...`).
2. `icl_model_mps_h450.py` — Exact `state_dict`-compatible `ICLH450MPS` supporting Apple Silicon `mps` (`xbabe4`
   M3 Ultra 80-core GPU) and `cuda` (`xbabe1` RTX 4090) with exact CPU/CUDA `float64` context normalization and bin
   interpolation + `bfloat16`/`float32` transformer execution, zero-state-dict-delta `checkpoint_row_blocks=True`
   activation checkpointing (`~60%` activation VRAM reduction), and background-worker `prenorm_cpu_np` fast path
   eliminating GeForce GPU `float64` `nanquantile` stalls.
3. `train_mps_104m_h450.py` — `104M` (`ICL_L`) multi-scale high-variety trainer (`fleet-r7`) for `xbabe4` MPS and
   `xbabe1` CUDA (`--cuda-vram-frac 0.80`) using `mp.get_context("spawn")`, `MmapF64Bank(..., verify_data_sha=True)`
   (plus resume `bank_sha256` & `init_sha256` checks), `HostCpuGovernor` (`<= 82%` hard cap -> `<= 85%` host CPU average),
   explicit per-batch invariant flags (`--random-split-every-batch 1 --feature-drop-every-batch 1 --feature-drop-max-frac 0.35 --synth-inj-every-batch 1 --synth-inj-max-frac 0.10`:
   100% fresh random train/test split, 100% random feature removal on `p >= 2`, and `1 <= k <= floor(0.10 * F)` synthetic
   column injection on `F >= 10` / `k = 0` on `F < 10`), and vectorized `64`-bit-ordered `128-bit` `blake2b` row-invariant
   pre-forward deduplication (`0` repeats).
   - **Key `fleet-r7` Elo + Variety Upgrades:**
     - **100% Batched Multi-Scale `dev91`-Anchored Row Curriculum:** `58%` fast explorer + small-table tier
       (`N in {36, 64, 128, 256, 320, 384}`, high `B` up to `--bmax 32`) + `22%` bridge (`N in {512, 768}`) + `20%`
       exact `dev91` context-length anchor (`N in {1120, 1344, 1600}`), plus index-first zero-copy row slicing (`~3.5x`
       faster CPU prep).
     - **`dev91`-Aligned Target Spaces & Richer Feature/Prior Regimes:** Real target spaces `45%` 5-fold SVD-GCV
       robust-ridge residual, `35%` Yeo-Johnson MLE, `20%` raw (synthetic: `40% / 30% / 30%`); `10`-family structural +
       distractor synthetic column bank (`REL`-style row summaries, missingness/tail rates, context prototype RBF
       similarity, ratios, rank-normal); oblivious/symmetric decision-tree + GP + heteroscedastic/Student-$t$ teachers
       and MAR missingness priors.
     - **Dual-Horizon Polyak/SWA EMA (`ema_swa`):** Maintains `ema` (`0.995`) + `ema_swa` (`--swa-decay 0.9995`, ~2,000
       optimizer steps / ~100k datasets) at gentler `--lr 2.5e-5`, saving `ema_swa` into `ckpt_epoch_NN.pt`.
     - Lineage note: `fleet_L_x1` (`~52,016` datasets before `2026-11-12T22:15Z`) and `fleet_L_x4` (`~3,641` datasets)
       carry pre-`fleet-r4` episodes documented in `/home/ubuntu/jope_plan/evidence/H450/deviation_fleet_r2_r3_inj_cap.json`.
4. `fast_dev91_every_n_h450.py` — Multi-host (`--cmd shard`, persistent `ICLH450("L")` per worker, `--n-views 4`, `24`
   `[4, N, Fn]` forwards/context) frozen-split `dev91-audit-v1` `104M` CPU tracker (`0` `y_audit` read before `M.utility`),
   computing `in_corpus_legacy_guard_stack_91` (`GM.guard_and_stack_2l`), `in_corpus_random_split_champion_91`,
   `in_corpus_neural_plus_tab_no_gp_91`, and `in_corpus_icl_boost_only_91`. Kept strictly labeled `multihost_4view_cpu`
   and never mixed with official `cols_h450_gpu` + `analyze_h450` readouts (`r4_g*`).
5. `fleet_telemetry_and_xbabe1_eval.py` — 5-host (`xbabe0..xbabe4`) live telemetry (`0` file deletions anywhere per
   `2026-10-13` owner ruling; `xbabe2` CPU eval pinned `taskset -c 0-63,80-127` leaving CPUs `64-79` free for
   `cl_fwriter`'s `analyze-epoch`), non-destructive SHA-256-verified `xbabe1 -> xbabe0` immutable epoch mirror, and
   multi-host (`xbabe0` + `xbabe2` + `xbabe1` CPUs, `CUDA_VISIBLE_DEVICES=''`) `V=4` evaluation of both single-model
   `fleet_L_x1` (`solo_epNNN`) and exposure-weighted 3-GPU `104M` EMA/SWA soups (`soup3_NNNNk`).
