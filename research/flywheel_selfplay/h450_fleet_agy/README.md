# H450 Fleet Extensions (`research/flywheel_selfplay/h450_fleet_agy/`)

Additive, non-overlapping H450 5-host fleet components coordinated with `cl_fwriter` via
`/home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md` and `/home/ubuntu/jope_plan/H450_FLEET_COORDINATION.md`
(staged read-only under `/home/ubuntu/h450-stage/fleet-r9`):

1. `mmap_bank_h450.py` — Zero-copy `np.memmap` drop-in replacement (`MmapF64Bank`) for `CombinedF64Bank`
   (`< 0.5 s` init, `< 80 MiB` private RSS), supporting exact `sampling_mode="uniform_dataset"` (`1 / 3,525` per S3
   dataset, independent of row count) as well as `"smart_stratified"`, plus `--cmd convert` and `--cmd verify-parity`.
2. `icl_model_mps_h450.py` — Exact `state_dict`-compatible `ICLH450MPS` supporting Apple Silicon `mps` (`xbabe4`
   M3 Ultra 80-core GPU) and `cuda` (`xbabe1` RTX 4090) with exact CPU/CUDA `float64` context normalization and bin
   interpolation + `bfloat16`/`float32` transformer execution, zero-state-dict-delta `checkpoint_row_blocks=True`
   activation checkpointing (`~60%` activation VRAM reduction), and background-worker `prenorm_cpu_np` fast path.
3. `train_mps_104m_h450.py` — `104M` (`ICL_L`) maximum-dataset-entropy uniform-dataset trainer (`fleet-r9`, Jepson
   `2026-11-13 05:39 MST` directive) for `xbabe4` MPS and `xbabe1` CUDA (`--cuda-vram-frac 0.80`):
   - **Keep-Latest-Only Checkpoint Policy (`--keep-latest-only 1`):** Atomically updates rolling `ckpt_last.pt`
     (resume state) and rolling `ckpt_latest.pt` (`ema_swa` inference weights) plus per-milestone `epoch_NN.json`
     and `log.jsonl` numbers (`0` per-epoch `.pt` binaries created, `0` `unlink`/`rm` calls).
   - **Maximum Dataset Entropy & Exact Uniform S3 Mixing:**
     1. **Wider Column Drop & Zero-Leakage Synthetic Injection:** Real column drop `[5%, 65%]` on `100%` of `p >= 2`
        (`--feature-drop-min-frac 0.05 --feature-drop-max-frac 0.65`) and `14`-family context-only synthetic column
        injection `[5%, 30%]` (`--synth-inj-min-frac 0.05 --synth-inj-max-frac 0.30`).
     2. **Varied Episode Row Subsampling Across Full Corpus (`[32 .. 2048]`, NO permanent 300-row downsample):**
        Full table rows remain in play; each episode draws a fresh uniform / quantile-stratified / covariate-tilted
        subsample across `ROW_BINS = (32..2048)` with `p_ctx in [0.18, 0.95]`.
     3. **Random Valid Target Column Choice:** Strided-probe validator caches all continuous non-ID columns per table
        (`45%` random valid feature-as-target `X_{-j} -> X_j`, `40%` canonical target, `15%` GP/oblivious-tree teacher).
     4. **Feature & Slot Permutations + Per-Feature Scaling/Affine + Small Noise:** Full column permutation, independent
        per-episode `FID_SLOTS` permutation `[B, 1, F]`, raw sign flip `{-1, +1}` + log-uniform scaling `exp(U(-3, 3))` +
        shift + `[0.2%, 5%]` context-scaled noise, plus post-`normalise_f64` mild per-feature affine jitter (`[0.80, 1.25]`).
     5. **Exact Uniform-by-Dataset Mixing (`--dataset-mix uniform`):** Draws `t_idx ~ Uniform({0..3524})` (`1 / 3,525`
        per dataset, not by row) and batches via a per-worker `(N_b, F_b)` bin accumulator with explicit quadratic
        `B * N^2 <= attn_cap` OOM protection.
   - Lineage note: protocol switch steps and sampler configs are recorded in `state["switch_history"]` and `epoch_NN.json`.
4. `fast_dev91_every_n_h450.py` — Multi-host (`--cmd shard`, `--n-views 4`) frozen-split `dev91-audit-v1` `104M` CPU
   tracker (`0` `y_audit` read before `M.utility`), preferring `ema_swa` weights when present.
5. `fleet_telemetry_and_xbabe1_eval.py` — 5-host (`xbabe0..xbabe4`) live telemetry (`0` file deletions; `xbabe2` CPU
   eval pinned `taskset -c 0-63,80-127` leaving CPUs `64-79` free for `cl_fwriter`) and rolling single-slot
   `solo_x1_latest.pt` / `soup_latest.pt` multi-host `V=4` evaluation.
