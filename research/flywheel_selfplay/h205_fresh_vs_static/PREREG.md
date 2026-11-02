# H205 — Fresh D4 families vs static 205-family pool and static teacher, 50 rounds from r882, H178 pooled selector

**Owner:** `rd_argon` `[ARGON]`.
**Assignment:** `~/nudges/n2022_rd_argon_h205.md` and `~/nudges/hwave_t2022_plan.md`.
**Authority:** `"Jepson via Grok Bot, card answer 8:22 PM MT" (2026-11-01, t11s3): "Start H205 and H204 now with those unblocks; H206 after xbabe3 disk is cleared."`
**Claim:** `~/jope_plan/ns/receipts/H205_claim.txt` and `CLAIM GPU xbabe1 H205 rd_argon` in `AGENT_CHAT.md`.

## Wave Separation and Unblocks (`hwave_t2022_plan.md`)

- **Wave registry separation:**
  - `H204` (`luna_ns_hbaseline`): Bigger proposer (`depth-4/6/8` trees + Rich-19 + research-only DOPE unfreeze) on `xbabe2` RTX 3090.
  - `H205` (`rd_argon`): Fresh vs static data (`FRESH`, `STATIC-DATA`, `STATIC-TEACHER`, 50 rounds from `r882`) on `xbabe1` RTX 4090.
  - `H206` (`gpu_fused_redteam_codex`): Search distillation on `xbabe3` RTX 3090 (queued until `xbabe3` disk is cleared below `~88%`).
  - `H207` (`rd_astra`): Singleton test-time augmentation (`TTA`) diagnostic addendum (renumbered from `rd_astra`'s earlier H204 TTA).
- **Authorized unblocks (`t2022`):**
  1. **Unblock (a) — Real-catalog D4 family-exclusion fix:** `luna_v1_tel` owns the 2-line fix in `research/joint_regbin/admission.py` (accepting bare `dedupe_fingerprint:*` and `source_file:*` keys in `exclude_d4_families` and scoping `source_file` binary exclusion to rows not in `admitted_d4_keys`). `STATIC-DATA`, `STATIC-TEACHER` infrastructure, and native `r882` `91/91` parity launch immediately on `xbabe1` RTX 4090; `FRESH` arm execution gates on `luna_v1_tel`'s `JRB-01 ADMISSION PASS` receipt.
  2. **Unblock (b) — Numeric thread rule:** `<= 4` numeric threads (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `RAYON_NUM_THREADS`, and PyTorch intra-op/inter-op threads `<= 4`); CUDA runtime and driver helper threads (`cuda*`, `nv*`, `pt_autograd*`, `cudnn*`, `cublas*`) are exempt from the numeric thread cap while retaining `nice -n 19`, CPU `<= 85%`, RSS `<= 6 GB`, disk `<= 93.8%` (`>= 40 GB` free), and single-GPU-process admission.
  3. **Unblock (c) — Launch before prereg PR review:** Launch before prereg PR review is authorized provided `PREREG.md` is committed and the prereg PR is opened before scored inference. `rd_argon` does not merge its own PR.

## Scientific Question

Every continuation wave from `H140` through `H202` updated the `29,211`-parameter `adapter_10tok` head on the fixed `1,219` `DL-Train` contexts (`205` families) and hit a hard covariate-support ceiling (`NO_ADVANCE`). Does supplying fresh, nonsealed D4 `TRAIN` families with newly verified Rust-CV (`bootstrap.s1main --fit-s1`) evaluations across 50 rounds break the static-pool ceiling under equal compute, and how much of any change comes from fresh covariate support alone (`STATIC-TEACHER`) versus fresh ground-truth Rust-CV labels (`FRESH`)?

## Arms and Equal-Compute Design

All three active arms start from the exact `r882` checkpoint (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) with frozen `DOPE` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and run **50 rounds** across seeds `205101, 205102, 205103` (fold seed `205140`, bootstrap seed `205141`) with identical optimizer step budgets (`16` family-stratified microbatches of `8` distinct families per round, `24`-candidate slates per context: `4` revealed + `20` unrevealed):

| Arm | Covariate Support (Families) | Supervision Signal | Compute Budget | Role |
|---|---|---|---|---|
| `R882` | Fixed reference (`0` updates) | None (frozen `r882`) | `0` rounds | Baseline reference (`Elo 1500`); `91/91` exact `DL-Val` parity (`1.1377941898164525`) verified before any candidate forward |
| `STATIC-DATA` | Fixed `1,219` `DL-Train` contexts (`205` families) | Fixed pool fitness + gate (current flywheel recipe) | `50` rounds $\times$ `16` microbatches | Control 1 (static covariate support, ground-truth labels) |
| `STATIC-TEACHER` | Fresh nonsealed D4 `TRAIN` families each round | Frozen `r882` teacher cost rankings on fresh D4 pools | `50` rounds $\times$ `16` microbatches | Control 2 (fresh covariate support, static teacher labels) |
| `FRESH` | Fresh nonsealed D4 `TRAIN` families each round (post `JRB-01 ADMISSION PASS`) | Newly verified Rust-CV (`bootstrap.s1main --fit-s1`) fitness + gate | `50` rounds $\times$ `16` microbatches | **Primary treatment** (Holm $m = 2$ vs `STATIC-DATA` and `STATIC-TEACHER`) |

## Per-Round Admission (`H178` Calibrated Pooled Selector)

At the end of each round $t \in \{1, \dots, 50\}$, the round candidate is evaluated on the 5-fold cross-fit selector folds $B_0 \cup \dots \cup B_4$ (`205` families, `1,219` contexts) using the `H177`/`H178` pooled selector (`research/flywheel_selfplay/h177_pooled_selector/analysis.py` and `research/flywheel_selfplay/h178_pooled_calibration/calibration.py`):
- Average across the 3 seeds within each of the `205` families before running `10,000` paired family-bootstrap draws (`boot_seed = 205141`).
- Admit round $t$'s checkpoint iff:
  1. Equal-family native utility gain over the current incumbent `> 0.001 + 1e-12`;
  2. One-sided 95% lower confidence bound `LCB95 > 0.0`;
  3. Family harm rate `<= 10%` (`<= 20 / 205` families degraded by more than `0.01`).
- If rejected, the arm retains its previous admitted checkpoint for round $t + 1$.
- Stage-1 pooled-$B$ selection locks are written and `fsync`ed (`pooled_b_selection_lock.json`) before any held-out $A$-fold or confirmation scoring (`StageOrderGuard`).

## Common Gate (`hwave_t2022_plan.md`)

1. **Native `r882` Parity Gate (Stage 0):** Before any candidate forward or optimizer step, `r882` must reproduce `91/91` exact native `DL-Val` utilities (`mean = 1.1377941898164525`, byte-identical to `best_dev_checkpoint.json`). Any mismatch aborts immediately.
2. **205-Family Pooled Cross-Fit Gate:** On the held-out cross-fit assessment panel $A_0 \cup \dots \cup A_4$ (`205` families), `FRESH` must achieve `LCB95 > 0.0`, Holm-adjusted $p < 0.05$ ($m = 2$ contrasts: `FRESH - STATIC-DATA` and `FRESH - STATIC-TEACHER`), and family harm `<= 10%` (`<= 20 / 205` families).
3. **Sealed D4 Confirmation Separation:** The `76` sealed D4 confirmation families (`122` datasets) never enter training, inner fitting, teacher distillation, or per-round $B$ selection. The runtime `Audit` hook blocks any open of `/sealed`, `/holdout`, `meta-test`, or sealed confirmation paths during training and development. Final confirmation readout is permitted only once, after Stage-2 pooled-$A$ lock and only if the 205-family pooled cross-fit gate passes.

## Host, GPU Sharing, and Custody

- **Host & Device:** `xbabe1` `cuda:0` (`NVIDIA GeForce RTX 4090`), sharing the GPU with the live flywheel `PID 2297929` (`~17.2 GiB / 24.5 GiB` used, `~7.3 GiB` free).
- **Non-Interference:** `H205` runs as a single GPU process with `torch.cuda.set_per_process_memory_fraction(0.18, 0)` (`<= 4.4 GiB` VRAM), `nice -n 19`, and `<= 4` numeric threads. `PID 2297929` is never stopped, signaled, or reniced.
- **Run Directory & Custody:** `/home/ubuntu/jope-data/h205/run-<UTC>` on `xbabe1`, mirrored create-only to `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H205/`.
