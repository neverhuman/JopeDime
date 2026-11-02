# H204: Bigger Proposer (`128 -> 256 -> 512`) with Research-Only `DOPE` Unfreeze

- **Wave ID**: `H204`
- **Owner**: `gpu_tokbench_agy` (`[TOKBENCH-AGY]`)
- **Host / GPU**: `xbabe2` (`NVIDIA GeForce RTX 3090`, `cuda:0`)
- **Status**: `PREREGISTERED` (Stage 1: labeling + width `128` + compute-matched controls `R882_CONT` and `H180_RAND`)
- **Authority**: `"Jepson via Grok Bot, card answer 8:22 PM MT" (2026-11-01, t11s3): "Start H205 and H204 now with those unblocks; H206 after xbabe3 disk is cleared."`
- **Merged Unblock Dependencies**:
  - `#1726` (`fcc5aead707fa33bbc5cb6944a758d1491699d98`): Shared research-only `DOPE` opt-in (`configure_research_dope`, `assert_research_optimizer_policy`, `reject_research_dope_unfreeze` in `python/src/jope/north_star/dope_freeze_guard.py`).
  - `#1730` (`fe9d1a5512dc4eb1a248baa171e77128baf39943`): Shared numeric-thread admission (`capture_numeric_threads`, `admit_numeric_threads` in `research/flywheel_selfplay/hwave_numeric_threads.py`).
- **Wave Numbering Note**: `H204` is reserved for this `bigger proposer` wave; `rd_astra`'s TTA follow-up was renumbered to `H207` (`#1720`).

---

## 1. Scientific Hypothesis & Motivation

Recent waves (`H171` through `H202`) showed that post-hoc rerankers, ensemble combiners, and frozen-DOPE width-64 heads on the fixed `205`-family context pool fail to beat the `r882` baseline (`1.1377941898164525` on the `91` `DL-Val` contexts) once proper out-of-fold family selection is applied. Two bottlenecks prevent proposer capacity from translating into out-of-fold selection utility:

1. **Frozen `DOPE` representation bottleneck**: In `r882`, `net.dope_model` is frozen at checkpoint initialization, preventing Stage-1B context tokens (`tokens_10x32`, specifically SVD-projected slots `2, 4, 5, 9`) from adapting to downstream GP candidate ranking utility.
2. **Narrow (`d=64`) proposer bottleneck**: The `r882` GameNet / BlockFormer proposer operates at width `64`, limiting cross-attention and bilinear context-candidate interaction capacity as candidate supervision scales across D4 `TRAIN` regression datasets (`8–16` CV-labeled candidates per context).

**Primary Hypothesis (`H204`)**:
Scaling the proposer architecture across progressive widths (`128 -> 256 -> 512`) while jointly fine-tuning `net.dope_model` (`router` + `target_net.col_gate_head`) through the differentiable Stage-1B SVD slot projection bridge—strictly gated behind `#1726` `configure_research_dope` (`JOPE_RESEARCH_DOPE_UNFREEZE=1` + nonempty `JOPE_RESEARCH_RUN_ID`)—will achieve a statistically significant `DL-Val` cost reduction (`>= 0.005` mean gain, `95%` family-cluster bootstrap CI `> 0`, Elo lift `>= +10`, `>= 2/3` positive seeds, `holm_adjusted_p <= 0.05`) and outperform both a compute-matched `r882` continuation control (`R882_CONT`) and a random-adapter capacity control (`H180_RAND`).

---

## 2. Experimental Arms & Controls

All arms are evaluated under the identical native first-argmin evaluator contract (`fixed-dev-cost-argmin-negative-fitness/v1`) across `3` preregistered seeds (`204101`, `204102`, `204103`) and `5` family-disjoint cross-fitting folds (`FOLD_SEED = 204140`):

1. **`R882_CONT` (Compute-Matched Continuation Control)**:
   - Continues training the non-anchor parameters of the loaded `round-882.pt` width-`64` `net.adapter_10tok` (`R882AdapterContinuation`), with `DOPE` (`net.dope_model`) and `gamenet_anchor` **strictly frozen** (`requires_grad=False`, `.eval()`), verified via `assert_optimizer_excludes_dope` and `verify_dope_frozen_unchanged`.
   - Trained on the exact same `8–16` CV-labeled candidates per context, fold splits, epochs, and optimizer steps as `PROPOSER_W128`.
2. **`H180_RAND` (Random-Adapter Control)**:
   - Width `128` (`WiderBlockFormerProposer(width=128, freeze_trunk=True)`), `DOPE` **strictly frozen**, with random-initialized cross-attention/bilinear/FFN trunk weights frozen (`requires_grad=False`) and **only** the linear readout head (`head: Linear(128, 1)`) and residual scale trained.
   - Tests whether gains come from genuine representation learning vs. linear readout calibration.
3. **`PROPOSER_W128` (Stage-1 Primary Treatment Arm)**:
   - Width `128` (`WiderBlockFormerProposer(width=128)`), with `net.dope_model` configured via `#1726` `configure_research_dope(..., unfreeze=True)` and verified via `assert_research_optimizer_policy`, propagating gradients through `compute_live_tokens_10x32` (`svd_slot2_m2`, `svd_slot4_d629_mid`, `svd_slot5_d629_tail`, `svd_slot9_kron`) with `lr_dope = 1.5e-5` (`weight_decay = 0.0`) and `lr_proposer = 8e-4`.
4. **`PROPOSER_W256` & `PROPOSER_W512` (Stage-2 Progressive Width Arms)**:
   - Widths `256` and `512` with research `DOPE` unfreeze.
   - Gated on `PROPOSER_W128` passing the Stage-1 progression gate (`mean_gain >= 0.005`, `CI95_low > 0`, `Elo >= +10`, `>= 2/3` seeds positive, `holm_adjusted_p <= 0.05`, and beating `max(R882_CONT, H180_RAND)`).

---

## 3. Pinned Input Closure & SHA-256 Digests

- `/home/ubuntu/jope-data/h50/input/source_manifest.json`: `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`
- `/home/ubuntu/jope-data/h50/input/selected.json`: `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`
- `/home/ubuntu/jope-data/h51/ckpt/round-882.pt`: `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`
- `/home/ubuntu/jope-data/h50/input/cpu_exact_fused.py`: `b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab`
- `/home/ubuntu/jope-data/h58/cache/cache_manifest.json`: `66b1df1374242580d482b7d6f9b72b6fe8f65830f90f658ca662a698aa229897`
- `/home/ubuntu/jope-data/h58/cache/context_features.npz`: `75704e8279c19fee8a7843ccbea801e3cd4b67e0ee08d282b070ddae7ab82faf`
- `/home/ubuntu/jope-data/h58/cache/pool_priors.npz`: `6db567ef2c10d2c67c375f7500305ec693a700c4e71eb23003b4cd312ee51cbe`
- `research/flywheel_selfplay/rust_evaluator.py`: `7b0cb0ecb98c7836b16a781f632a9fc74683c2ab9facee820790ceebc759cb45`
- `python/src/jope/north_star/dope_freeze_guard.py`: `908547a91fcb6841868aed68e1c0811befbdc2cf182053125fdf4004549fd2b1`
- `EXPECTED_DOPE_SHA256` (`round-882.pt` `dope_sha256`): `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
- `R882_EXPECTED_DL_VAL_PICKS_SHA256` (`91` exact picks): `460560e44b8d77e445f18db063601c8d02f748718376fa355d6997e0c3624d96`
- `R882_EXPECTED_DL_VAL_UTILITIES_SHA256` (`91` exact utilities, mean `1.1377941898164525`): `77fd7f5963905a1aa0fd321f4ae04fe24328961357aa745efb18bbe0ebfec8e9`

---

## 4. Storage, Numeric-Thread Admission & Compute Budgets on `xbabe2`

- **Campaign Budgets**: `<= 30.0` GPU-hours, `<= 200.0` CPU core-hours, `<= 12.0` wall-hours.
- **Local LRU Cache (`LocalLRUCache`)**: Hard-capped at `<= 20 GiB` (`21,474,836,480` bytes), evicting oldest files before writing.
- **Disk Guard**: Refuses writes and stops cleanly if `used_ratio > 0.938` (`93.8%`) or `free_bytes < 40 GiB`.
- **Process Priority & Host Load**: `nice 19`, `<= 2` labeler workers, host `load1 <= 85%` of CPU cores, RSS `<= 6 GiB`.
- **Thread Rule (b) via `#1730` (`hwave_numeric_threads.py`)**:
  - `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `torch.set_num_threads(4)`.
  - Verified in worker and supervisor via `capture_numeric_threads(torch)` and `admit_numeric_threads([worker_report], os_threads=...)` (`numeric_threads_upper_bound <= 4`; OS/CUDA driver threads recorded diagnostically).

---

## 5. Stage-0 & Promotion Gates

1. **Stage-0 Native `r882` Parity Gate (Mandatory Pre-Flight)**:
   - Before any candidate training or evaluation forward, `verify_stage0_r882_parity_on_eval_batch` must confirm `91/91` exact `r882` `DL-Val` picks (`R882_EXPECTED_DL_VAL_PICKS`) and mean cost `1.1377941898164525` (`abs_diff <= 1e-12`).
2. **Stage-1 -> Stage-2 Gate & Common Promotion Gate**:
   - `pooled_mean_gain >= +0.005` on `DL-Val` (`91` contexts, `39` families)
   - `95%` paired pooled-family bootstrap CI lower bound (`10,000` family-cluster resamples, `BOOT_SEED = 204141`) `> 0.0`
   - `pooled_elo_lift >= +10.0`
   - `>= 2/3` seeds positive (`204101`, `204102`, `204103`)
   - Worst single-context regression `<= 0.05` and fallback rate `<= 20%`
   - Strictly beats `max(R882_CONT, H180_RAND)` and survives Holm-Bonferroni correction (`holm_adjusted_p <= 0.05`).
   - If `PROPOSER_W128` fails the Stage-1 gate, `H204` records `NO_ADVANCE` and does not spend GPU budget on widths `256` and `512`.
