# H204: Bigger Proposer (`128 -> 256 -> 512`) with Research-Only `DOPE` Unfreeze

- **Wave ID**: `H204`
- **Owner**: `gpu_tokbench_agy` (`[TOKBENCH-AGY]`)
- **Host / GPU**: `xbabe2` (`NVIDIA GeForce RTX 3090`, `cuda:0`)
- **Status**: `PREREGISTERED` (Stage 1: labeling + width `128` + compute-matched controls `R882_CONT` and `H180_RAND`)
- **Authority**: `"Jepson via Grok Bot, card answer 8:22 PM MT" (2026-11-01, t11s3): "Start H205 and H204 now with those unblocks; H206 after xbabe3 disk is cleared."`
- **Wave Numbering Note**: `H204` is reserved for this `bigger proposer` wave; `rd_astra`'s TTA follow-up was renumbered to `H207` (`#1720`).

---

## 1. Scientific Hypothesis & Motivation

Recent waves (`H171` through `H202`) showed that post-hoc rerankers, ensemble combiners, and frozen-DOPE width-64 heads on the fixed `205`-family context pool fail to beat the `r882` baseline (`1.1377941898164525` on the `91` `DL-Val` contexts) once proper out-of-fold family selection is applied. Two bottlenecks prevent proposer capacity from translating into out-of-fold selection utility:

1. **Frozen `DOPE` representation bottleneck**: In `r882`, `net.dope_model` is frozen at checkpoint initialization, preventing Stage-1B context tokens (`tokens_10x32`, specifically SVD-projected slots `2, 4, 5, 9`) from adapting to downstream GP candidate ranking utility.
2. **Narrow (`d=64`) proposer bottleneck**: The `r882` GameNet / BlockFormer proposer operates at width `64`, limiting cross-attention and bilinear context-candidate interaction capacity as candidate supervision scales across D4 `TRAIN` regression datasets (`8–16` CV-labeled candidates per context).

**Primary Hypothesis (`H204`)**:
Scaling the proposer architecture across progressive widths (`128 -> 256 -> 512`) while jointly fine-tuning `net.dope_model.router` through the differentiable Stage-1B SVD slot projection bridge—strictly gated behind the research-only unfreeze flag (`JOPE_RESEARCH_DOPE_UNFREEZE=1` + nonempty `JOPE_RESEARCH_RUN_ID`)—will achieve a statistically significant `DL-Val` cost reduction (`>= 0.005` mean gain, `95%` family-cluster bootstrap CI `> 0`, Elo lift `>= +10`, `>= 2/3` positive seeds) and outperform both a compute-matched `r882` continuation control (`R882_CONT`) and a random-adapter capacity control (`H180_RAND`).

---

## 2. Experimental Arms & Controls

All arms are evaluated under the identical native first-argmin evaluator contract (`fixed-dev-cost-argmin-negative-fitness/v1`) across `3` preregistered seeds (`204101`, `204102`, `204103`) and `5` family-disjoint cross-fitting folds (`FOLD_SEED = 204140`):

1. **`R882_CONT` (Compute-Matched Continuation Control)**:
   - Width `64` (`WiderBlockFormerProposer(width=64)`), `DOPE` (`net.dope_model`) **strictly frozen** (`requires_grad=False`, `.eval()`).
   - Trained on the exact same `8–16` CV-labeled candidates per context, fold splits, epochs, and optimizer steps as `PROPOSER_W128`.
2. **`H180_RAND` (Random-Adapter Control)**:
   - Width `128` (`WiderBlockFormerProposer(width=128, freeze_trunk=True)`), `DOPE` **strictly frozen**, with random-initialized cross-attention/bilinear/FFN trunk weights frozen (`requires_grad=False`) and **only** the linear readout head (`head: Linear(128, 1)`) and residual scale trained.
   - Tests whether gains come from genuine representation learning vs. linear readout calibration.
3. **`PROPOSER_W128` (Stage-1 Primary Treatment Arm)**:
   - Width `128` (`WiderBlockFormerProposer(width=128)`), with `net.dope_model.router` unfrozen **ONLY** via research flag (a) (`JOPE_RESEARCH_DOPE_UNFREEZE=1` and nonempty `JOPE_RESEARCH_RUN_ID`), propagating gradients through `compute_live_tokens_10x32` (`svd_slot2_m2`, `svd_slot4_d629_mid`, `svd_slot5_d629_tail`, `svd_slot9_kron`) with `lr_dope = 1.5e-5` and `lr_proposer = 8e-4`.
4. **`PROPOSER_W256` & `PROPOSER_W512` (Stage-2 Progressive Width Arms)**:
   - Widths `256` and `512` with research `DOPE` unfreeze.
   - Gated on `PROPOSER_W128` passing the Stage-1 progression gate (`mean_gain >= 0.005`, `CI95_low > 0`, `Elo >= +10`, `>= 2/3` seeds positive, and beating `max(R882_CONT, H180_RAND)`).

---

## 3. Research-Only `DOPE` Unfreeze Contract (Flag (a))

- **Default / Production / Serving / Flywheel Invariant**:
  - By default (`JOPE_RESEARCH_DOPE_UNFREEZE` unset or `"0"`), `net.dope_model` is strictly frozen (`requires_grad=False`, `.eval()`) via `jope.north_star.dope_freeze_guard.lock_dope_weights_and_eval_mode` and verified via `snapshot_dope_weights` / `verify_dope_frozen_unchanged`.
  - Any attempt to pass `request_research_unfreeze=True` or `JOPE_RESEARCH_DOPE_UNFREEZE=1` in `serving`, `flywheel`, `production`, or `promotion` entrypoints immediately raises `RuntimeError`.
- **Research Opt-In**:
  - Requires `JOPE_RESEARCH_DOPE_UNFREEZE="1"` **and** a nonempty `JOPE_RESEARCH_RUN_ID`.
  - Logged explicitly in `launch.json` (`"research_dope_unfreeze": true`, `"research_run_id": ...`) and verified by SHA-256 parameter digests (`dope_before_sha256 != dope_after_sha256` on `PROPOSER_W128`, whereas `dope_before_sha256 == dope_after_sha256` on `R882_CONT` and `H180_RAND`).
  - At step 0 (`r882` weights), `compute_live_tokens_10x32` matches cached `tokens_10x32` bit-for-bit (`0.0` max diff on CPU, `< 1e-6` on CUDA).

---

## 4. Labeler & Bounded S3 Streaming Contract

- **Data Source**:
  - Nonsealed D4 `TRAIN` regression datasets (`~3,300` datasets from `s3://veox-jopedime-datasets-use2`; sealed confirmation families excluded via `JRB-01` admission once `luna_v1_tel`'s `#1722` admission manifest is posted; Stage 1 starts immediately on the admitted `205`-family / `1,219`-context `DL-Train` pool with zero overlap against the `39`-family / `91`-context `DL-Val` holdout).
  - Labels `8–16` candidates per context (default `16`: `4` revealed anchors + `12` stratified candidates) with exact two-fold Rust CV costs (`run_outer_test=False`).
- **Storage & Compute Guards on `xbabe2`**:
  - Local LRU cache (`LocalLRUCache`) hard-capped at `<= 20 GiB` (`21,474,836,480` bytes), evicting oldest files before writing.
  - Disk guard: refuses writes and stops cleanly if `used_ratio > 0.938` (`93.8%`) or `free_bytes < 40 GiB`.
  - Process priority: `nice 19`, `<= 2` labeler workers, CPU utilization `<= 85%`, RSS `<= 6 GiB`.
  - Thread rule (b): `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `RAYON_NUM_THREADS`, and `torch.get_num_threads()` all `<= 4`; CUDA runtime/driver helper threads are explicitly exempt.

---

## 5. Stage-0 & Promotion Gates

1. **Stage-0 Native `r882` Parity Gate (Mandatory Pre-Flight)**:
   - Before any candidate training or evaluation forward, `verify_stage0_r882_parity_on_eval_batch` must confirm `91/91` exact `r882` `DL-Val` picks and mean cost `1.1377941898164525` (`abs_diff <= 1e-12`).
2. **Stage-1 -> Stage-2 Gate & Common Promotion Gate**:
   - `pooled_mean_gain >= +0.005` on `DL-Val` (`91` contexts, `39` families)
   - `95%` paired family-cluster bootstrap CI lower bound (`10,000` resamples, `BOOT_SEED = 204141`) `> 0.0`
   - `pooled_elo_lift >= +10.0`
   - `>= 2/3` seeds positive (`204101`, `204102`, `204103`)
   - Worst single-context regression `<= 0.05` and fallback rate `<= 20%`
   - Strictly beats `max(R882_CONT, H180_RAND)` and survives Holm-Bonferroni correction across executed arms.
   - If `PROPOSER_W128` fails the Stage-1 gate, `H204` records `NO_ADVANCE` and does not spend GPU budget on widths `256` and `512`.
