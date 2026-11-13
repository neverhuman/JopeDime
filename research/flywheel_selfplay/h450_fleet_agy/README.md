# H450 Fleet Extensions (`research/flywheel_selfplay/h450_fleet_agy/`)

Additive, non-overlapping H450 5-host fleet components coordinated with `cl_fwriter` via
`/home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md` and `/home/ubuntu/jope_plan/H450_FLEET_COORDINATION.md`
(staged read-only under `/home/ubuntu/h450-stage/fleet-r6`):

1. `mmap_bank_h450.py` — Zero-copy `np.memmap` drop-in replacement (`MmapF64Bank`) for `CombinedF64Bank`
   (`< 0.5 s` init, `< 80 MiB` private RSS), plus `--cmd convert` and `--cmd verify-parity` (verified bit-identical
   across all `3,525` tables / `164,651,987` IEEE-754 `uint64` cells against `everything_combined_f64_h450_bank.npz`
   `9dfc60ed...`).
2. `icl_model_mps_h450.py` — Exact `state_dict`-compatible `ICLH450MPS` supporting Apple Silicon `mps` (`xbabe4`
   M3 Ultra 80-core GPU) and `cuda` (`xbabe1` RTX 4090) with exact CPU/CUDA `float64` context normalization and bin
   interpolation + `bfloat16`/`float32` transformer execution.
3. `train_mps_104m_h450.py` — `104M` (`ICL_L`) trainer for `xbabe4` MPS and `xbabe1` CUDA (`--cuda-vram-frac 0.80`)
   using `mp.get_context("spawn")`, `MmapF64Bank(..., verify_data_sha=True)` (plus resume `bank_sha256` & `init_sha256`
   checks), `HostCpuGovernor` (`<= 82%` hard cap -> `<= 85%` host CPU average), explicit per-batch invariant flags
   (`--random-split-every-batch 1 --feature-drop-every-batch 1 --feature-drop-max-frac 0.35 --synth-inj-every-batch 1 --synth-inj-max-frac 0.10`:
   100% fresh random train/test split, 100% random feature removal on `p >= 2`, and `1 <= k <= floor(0.10 * F)` synthetic
   column injection on `F >= 10` / `k = 0` on `F < 10`), and pre-forward `128-bit` `blake2b` deduplication (`0` repeats).
   - **Mixture differences vs `cl_fwriter` `pub-r9` (`train_aggressive_random_h450.py`):**
     - `synth_prob = 0.25` (`60%` `v3s` Prior + `40%` panel/hurdle-atom) vs `0.30` (`100%` `v3s` Prior).
     - Real target modes (`FROM` features): `65%` own target (`45%` sentinel-atom filter), `17%` masked real column
       `X_{-j} -> X_j`, `18%` RFF-GP + random-tree teacher (`13%` pure + `5%` real-teacher blend), vs `85% / 10% / 5%` in `pub-r9`.
     - Real target spaces: `40%` raw, `40%` 5-fold SVD-GCV robust-ridge residual, `20%` Yeo-Johnson MLE, vs `50% / 35% / 15%` in `pub-r9`.
     - Lineage note: `fleet_L_x1` (`~52,016` datasets before `2026-11-12T22:15Z`) and `fleet_L_x4` (`~3,641` datasets)
       carry pre-`fleet-r4` episodes documented in `/home/ubuntu/jope_plan/evidence/H450/deviation_fleet_r2_r3_inj_cap.json`.
   - **Async Worker Queue (`fleet-r6`):** Both `synth_prob` (`v3s` Prior + panel/hurdle-atom + 5-fold SVD robust-ridge)
     and `HostCpuGovernor.pace()` run 100% inside background `mp.get_context("spawn")` workers so the main CUDA/MPS
     trainer loop never blocks on CPU NumPy/SVD or CPU pacing sleeps.
4. `fast_dev91_every_n_h450.py` — Multi-host (`--cmd shard`, persistent `ICLH450("L")` per worker, `--n-views 4`, `24`
   `[4, N, Fn]` forwards/context) frozen-split `dev91-audit-v1` `104M` CPU tracker (`0` `y_audit` read before `M.utility`),
   computing `in_corpus_legacy_guard_stack_91` (`GM.guard_and_stack_2l`), `in_corpus_random_split_champion_91`,
   `in_corpus_neural_plus_tab_no_gp_91`, and `in_corpus_icl_boost_only_91`. Kept strictly labeled `multihost_4view_cpu`
   and never mixed with official `cols_h450_gpu` + `analyze_h450` readouts (`r4_g*`).
5. `fleet_telemetry_and_xbabe1_eval.py` — 5-host (`xbabe0..xbabe4`) live telemetry (preserving last-known counters and
   marking `STALLED` if SSH drops or `log.jsonl` heartbeat ages `>= 240 s`), SHA-256-verified `xbabe1 -> xbabe0`
   immutable epoch checkpoint archiver, and multi-host (`xbabe0` + `xbabe2` + `xbabe1` CPUs, `CUDA_VISIBLE_DEVICES=''`)
   `V=4` evaluation of both single-model `fleet_L_x1` newest immutable `ckpt_epoch_NN.pt` (`solo_epNNN`) and
   exposure-weighted 3-GPU `104M` EMA soups (`soup3_NNNNk` with `SOUP_MANIFEST.json`).
