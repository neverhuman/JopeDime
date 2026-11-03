# H242: zero-update LASER-style spectral rank reduction of the r882 adapter, xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0405_opus_h242_laser.md` (watcher, 2026-11-03 04:05Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H242_claim.txt`, taken under `Hwave-admission.lock`.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays
`c51a56d0`: it is checked when r882 loads, after every evaluation pass, and before and after the study.

The study runs offline on a read-only, SHA-verified COPY of r882 (`0084479a…`). The live flywheel (xbabe1
`run-20261030`) is never read, written, paused or preempted.

**This prereg merges before any study outcome is read.** The pre-merge smoke scores 0 contexts.

## Hypothesis

**Background.**
- H237 and H241 show that harm grows with adapter drift.
- Moving along the accepted trajectory hurts in both directions (H152 interpolation, H241 extrapolation).
- No H-wave has tried spectral or low-rank **denoising**. H160 projected onto a trajectory subspace; nothing truncated
  rank or pruned.

**The idea.** LASER (Sharma et al. 2023, "The Truth is in There: Improving Reasoning in Language Models with
Layer-Selective Rank Reduction") reports zero-training gains from removing small singular directions of selected
weight matrices.

**H242 asks:** does removing the small-singular-value components of r882's own adapter matrices lift DL-Val91
regression utility over r882 by more than +0.002?

## Reduction (frozen before outcomes)

- **Which tensors.** Every **2-D** adapter weight matrix W, as listed in
  [spectral_inventory.json](spectral_inventory.json). There are 16 matrices.
  - **Rank-1 matrices.** Three have rank 1 (`mm_prior_out.2`, `traj_delta_head.2`, `parent_delta_net.3`) and never
    change.
  - **Untouched.** All 1-D and 3-D adapter tensors (biases, LayerNorm, scalars, `null_hist_token`), the frozen
    anchor and every non-adapter tensor are untouched bit for bit.
  - **Packed attention.** The packed attention `in_proj_weight` (96×32) is one matrix.
- **The operation.** W_k = U_k Σ_k V_kᵀ, the top-k float64 SVD reconstruction, cast back to float32.
  - **The rank.** k is the smallest rank whose top singular values keep at least the energy fraction ρ of Σσ².
  - **ρ = 1.** When k equals the full rank, the matrix is kept exactly.
- **The frozen census.** [spectral_inventory.json](spectral_inventory.json) freezes, for every matrix:
  - its shape and full rank;
  - k at ρ = 0.99 / 0.95 / 0.90;
  - the removed energy and the Frobenius change.

  The runner recomputes the census from the r882 copy and refuses to run if it differs.

## Policies and arms

| Policy | Change | Role |
|---|---|---|
| r882 / CONTROL | ρ = 1 | reference (Elo 1500). Bit-exact r882. |
| RHO99, RHO95, RHO90 | every 2-D matrix at ρ = .99 / .95 / .90 | the candidates for SVD_SELECT |
| RAND99, RAND95, RAND90 | placebo, matched per matrix (see below) | the RANDPROJ placebo |
| S90-*m* (13 policies) | **only** matrix *m* at ρ = .90 | the candidates for LASER_SINGLE |

**The placebo.** For each matrix with d = full rank − k > 0, the placebo builds a seeded random orthonormal basis Q
(n × d) of the input space (seed `stream_seed(242160, name, ρ)`). It then sets W − c·W Q Qᵀ, where c makes the
Frobenius change equal to that of the rank-k truncation. The runner verifies the match to within 1e-5. The placebo
separates "denoising" from "any perturbation of this size".

**The arms.**
- **SVD_SELECT (primary decision arm).** ρ is chosen per (seed, outer) cell on FIT, from {r882, RHO99, RHO95, RHO90}.
- **LASER_SINGLE (exploratory decision arm).** The single matrix whose ρ = .90 truncation scores best on FIT, from
  r882 plus the 13 S90 policies.
- **RANDPROJ (report-only placebo).** The placebo at whatever ρ SVD_SELECT selected in that cell; r882 if SVD_SELECT
  kept r882. It is compared with r882 and with SVD_SELECT. It can never advance.
- **CONTROL.** ρ = 1, which is r882.

## Machinery (merged H241 zero-update harness)

- **Inherited unchanged:**
  - the H146 runtime, with the H232 A2 namespace release;
  - the merged H237 `adapter_forward`, pinned;
  - the cached frozen anchor, whose native/cached identity is proven bit-exact on r882;
  - the H50 cohort, with `has_meta_test` rows excluded;
  - the real-S3 provenance closure (the same `provenance.json`);
  - the audit hook, cooperative pauses and hash-verified resume.
- **Native91 first.** r882 must reproduce 91/91 contexts and the exact mean 1.1377941898164525 before any TRAIN
  outcome opens.
- **Phase 1 (DL-Train only).** All 20 policies are scored once on the 1,219 DL-Train contexts.
  - **Folds.** Fresh family folds use seeds 242101–242103 with fold seeds 242141–242143 (205 families → 5 × 41).
    They are pinned in [fold_manifest.json](fold_manifest.json); the within-wave best-match Jaccard is ≤ 0.169.
  - **Cells.** Each of the 15 cells uses its 4 non-outer folds as FIT; the outer fold is descriptive out-of-fold
    only.
  - **Selector (strict, inherited).** The best FIT column by equal-family mean is accepted only if its gain over
    r882 is above .001 **and** its family-bootstrap LCB95 is above 0. Otherwise r882. Ties pick r882 first, then the
    smallest truncation (the earliest listed candidate).
  - **Lock.** All decisions are fsynced to `selection.json` before Phase 2.
- **Phase 2.** Every policy is scored **once** on DL-Val91. Each seed's DL-Val policy for an arm is the equal-weight
  mixture of its 5 cells' selected policies. CONTROL's DL-Val vector must equal the native baseline exactly.
- **Budget.** 20 policies × 1,310 contexts, plus Native91, at **0 optimizer updates**.

## Statistics and decision

- **Panel.** DL-Val91: 91 contexts, 19 families. Higher utility is better.
- **Statistic.** The median over 3 seeds of the per-seed equal-family gain against r882. The CI95 is the
  family-bootstrap median (10,000 draws, boot seed 242150), with a one-sided p and **Holm over SVD_SELECT and
  LASER_SINGLE** (m = 2).
- **ADVANCE** for a decision arm requires all four of:
  - median gain > **+0.002**;
  - family LCB95 > 0;
  - Holm p < 0.05;
  - **≥ 2/3 seeds positive.**

  Ties mean no advance. Beating only CONTROL or the placebo is descriptive, never an advance. The wave verdict is
  ADVANCE if either decision arm advances, else NO_ADVANCE.
- **Also reported:**
  - Elo against r882 = 1500 (canonical Bradley–Terry) with a CI;
  - the gap to the plateau best 1.1389572 (r3175);
  - the per-ρ, per-matrix Frobenius change and kept rank;
  - the adapter L2 per policy;
  - harm families (equal-family loss > 1e-4);
  - invalid picks;
  - selected and best counts;
  - out-of-fold and unconstrained readouts;
  - every fixed policy on DL-Val91 and, descriptively, on DL-Train.
- **Not used.** No AUC, and no speed verdicts.

## Resources and the A1 memory guard (H239 A1)

- **Compute.**
  - xbabe2, CPU only: `taskset` cores 44–47, nice 19, one math thread.
  - Tree RSS < 4 GiB, ≤ 4 threads, disk free ≥ 300 GB.
  - Cores 40–43 (H240) and 48–51 (H239) are not used.
  - No GPU, xbabe0, xbabe3 or live flywheel. No foreign process control. PR #2071 is not touched.
- **Admission.** load5 ≤ 100 and MemAvailable ≥ 30 GiB.
- **Cooperative pause.** Triggered by load5 > 108 or MemAvailable < 20 GiB. It uses a pause file, not SIGSTOP.
  - **Resume** at load5 ≤ 100 and MemAvailable ≥ 25 GiB, sustained for 30 s.
- **Hard stop.** MemAvailable < 12 GiB sustained for 30 s, or < 6 GiB on any sample, plus the RSS, disk, thread and
  deadline stops. The supervisor signals only its own worker group.
- **Checkpoints.** Every completed evaluation file is hashed into `progress.json`.
  - **Up to two resumes** are allowed. They reuse only hash-verified files of the same manifest.
  - No re-roll and no deadline reset.
- **Deadlines.**
  - Prereg merge by **05:00Z**.
  - Compute hard stop **07:30Z**.
  - Closeout merge by **08:00Z**.

  If the study is incomplete, it closes as CLOSED INCOMPLETE with honest receipts and null outcomes.

## Custody

- **Repo.** Targeted pytest, Ruff, JSON parse, `git diff --check` and the records gates precede the head-bound
  same-agent review and the owner-authorized admin squash merge. Remote CI is not a gate.
- **S3.** Full outputs go to create-only S3 `…/rd_breakthrough_20261030/H242/<run_id>/` with sha/VersionId readback.
- **Closeout.** It commits only the README and a compact metrics JSON. No checkpoints or large objects go in Git.
