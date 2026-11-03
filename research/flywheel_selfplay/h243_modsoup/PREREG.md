# H243: zero-update module-wise soup of r882 and r782, xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0435_opus_h243.md` (watcher, 2026-11-03 04:40Z, decided).
**Claim:** `~/jope_plan/ns/receipts/H243_claim.txt`, taken under `Hwave-admission.lock`.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays
`c51a56d0`: it is checked when r882 loads, after every evaluation pass, and before and after the study.

The study runs offline on read-only, SHA-verified COPIES of r882 (`0084479a…`) and r782 (`427b0d46…`, the exact
H152/H241 input). The live flywheel is never read, written, paused or preempted.

**This prereg merges before any study outcome is read.** The pre-merge smoke scores 0 contexts.

## Hypothesis

**What has been tried.** Every weight merge so far used **one global coefficient**:
- interpolation (H152, MID and SOUP4, α = 0.25 / 0.75);
- extrapolation (H241);
- shrink (H105);
- rank truncation (H242).

H239 trained parameter subsets; it did not merge them.

**Why modules might differ.** H241 found that moving all of r882 toward r782 hurts DL-Val91 monotonically. Yet H145
found r782 beats r882 on the larger TRAIN cohort (+0.0133 [0.0025, 0.0275]). That combination is consistent with
**some modules of r782 being better and others worse**.

**H243 asks:** does replacing **one module group** of r882 by r782's (fully, or halfway), or a greedy soup of such
groups (Wortsman et al. 2022), lift DL-Val91 regression utility over r882 by more than +0.002?

## Module groups (frozen before outcomes)

[groups.json](groups.json) partitions the 47 trainable adapter tensors (29,211 parameters) into 11 natural module
groups. Biases, norms and scalars travel with their module. The runner refuses to run unless the groups partition the
adapter tensors exactly.

| Group | Tensors |
|---|---|
| G01_modality | `modality_type_emb`, `modality_gate_logits`, `modality_norm.{weight,bias}` |
| G02_prog_bottleneck | `prog_412_bottleneck.{0,1}.{weight,bias}` |
| G03_cand_query | `cand_query_proj.{0,1}.{weight,bias}` |
| G04_mm_cross_attn | `mm_cross_attn.in_proj_{weight,bias}`, `mm_cross_attn.out_proj.{weight,bias}` |
| G05_bilinear | `ctx_bilinear_u.weight`, `prog_bilinear_v.weight` |
| G06_mm_prior_head | `mm_prior_scale`, `mm_prior_out.{0,2}.{weight,bias}` |
| G07_hist_token | `null_hist_token`, `hist_token_proj.{0,1}.{weight,bias}` |
| G08_traj_cross_attn | `traj_attn_scale`, `traj_cross_attn.in_proj_{weight,bias}`, `traj_cross_attn.out_proj.{weight,bias}` |
| G09_traj_delta_head (output layer) | `traj_delta_head.{0,2}.{weight,bias}` |
| G10_parent_delta_net | `parent_delta_net.{0,1,3}.{weight,bias}` |
| G11_kernel_scalars | `null_kernel_w0_raw`, `sem_kernel_scale_raw`, `emb_kernel_scale_raw`, `traj_kernel_blend` |

The non-adapter tensors of r782 must be bitwise identical to r882's; the runner checks this and refuses otherwise.

## Policies and arms (zero optimizer updates)

**Single-group swaps (22).** For each group g and β ∈ {0.5, 1.0}, `g@β` is r882 with group g replaced by
(1 − β)·r882_g + β·r782_g. This is computed in float64 and cast back. At β = 1 the group is copied from r782 bit for
bit.

**Placebos (22).** `PL:g@β` is r882 with group g moved along a seeded Gaussian direction (seed
`stream_seed(243160, g, β)`) scaled to the same L2 as β·(r782_g − r882_g). The runner verifies the match to 1e-5.

**The arms.**
- **MODSWAP_SELECT (primary).** Per FIT cell, the best of r882 plus the 22 single-group swaps.
- **GREEDY_SOUP (exploratory, in the Holm family).** The greedy soup, built per FIT cell as described below.
- **PLACEBO (report-only).** The placebo for MODSWAP_SELECT's selected (g, β), or r882 if MODSWAP_SELECT kept r882.
  It separates "r782 module knowledge" from "any perturbation of that module". It never advances.
- **CONTROL.** r882 exactly.

**The greedy soup, step by step.**
1. Each group's ingredient is its better β by FIT family mean; ties take the smaller β.
2. Groups whose ingredient beats r882 on FIT by more than .001 are ranked by that gain (ties in group order). At most
   **5** are tried, in that order.
3. An ingredient is kept only if the soup's FIT family mean strictly improves by more than .001.
4. Soups are evaluated on DL-Train only. Each new recipe is a deterministic soup policy, scored once on DL-Train and
   cached.

## Selection and machinery (merged H242/H241 zero-update harness)

- **Inherited unchanged:**
  - the H146 runtime, with the H232 A2 namespace release;
  - the merged H237 `adapter_forward`;
  - the cached frozen anchor, bit-exact to native r882;
  - the H50 cohort, with `has_meta_test` rows excluded;
  - the real-S3 provenance closure;
  - the audit hook, cooperative pauses and hash-verified resume.
- **Native91 first.** r882 must reproduce 91/91 contexts and the exact mean 1.1377941898164525 before any TRAIN
  outcome opens.
- **Folds.** Fresh family folds use seeds 243101–243103 with fold seeds 243141–243143 (205 families → 5 × 41). They
  are pinned in [fold_manifest.json](fold_manifest.json); the within-wave best-match Jaccard is ≤ 0.169.
- **Cells.** There are 15 (seed, outer) cells. In each, FIT is the 4 non-outer folds and the outer fold is
  descriptive out-of-fold only.
- **Strict rule (both decision arms).** The selected candidate must beat r882 on FIT by more than .001 **and** have a
  family-bootstrap LCB95 above 0, else r882. Ties pick r882 first, then group order with β = .5 before 1.0.
- **Lock.** All decisions are fsynced to `selection.json` before **any** candidate is scored on DL-Val91.
- **Scoring.** Every policy is then scored **once** on DL-Val91: r882, 22 swaps, 22 placebos, and every greedy soup
  recipe created in Phase 1. Each seed's DL-Val policy for an arm is the equal-weight mixture of its 5 cells'
  selections. CONTROL's DL-Val vector must equal the native baseline exactly.
- **Budget.** 45 fixed policies plus at most 60 cached soups on DL-Train, all policies on DL-Val91, at 0 optimizer
  updates.

## Statistics and decision

- **Panel.** DL-Val91: 91 contexts, 19 families. Higher utility is better.
- **Statistic.** The median over 3 seeds of the per-seed equal-family gain against r882. The CI95 is the
  family-bootstrap median (10,000 draws, boot seed 243150), with a one-sided p and **Holm over MODSWAP_SELECT and
  GREEDY_SOUP** (m = 2).
- **ADVANCE** for a decision arm requires all four of:
  - median gain > **+0.002**;
  - family LCB95 > 0;
  - Holm p < 0.05;
  - **≥ 2/3 seeds positive.**

  Ties mean no advance. Beating only CONTROL or the placebo is descriptive.
- **Also reported:**
  - Elo against r882 = 1500 with a CI;
  - the gap to the plateau best 1.1389572 (r3175);
  - the per-group L2 change β·‖r782_g − r882_g‖;
  - harm families;
  - invalid picks;
  - selection counts and the greedy traces;
  - every fixed policy on DL-Val91 and, descriptively, on DL-Train.
- **The H152 gate-cliff context.** Context #27 (`fam-9f67f144…`) is reported **separately**: its value for every
  policy and for every arm mixture. It is **never excluded** from any statistic.
- **Not used.** No AUC, and no speed verdicts.

## Resources and the A1 memory guard (H239 A1)

- **Compute.**
  - xbabe2, CPU only: `taskset` cores 44–47, nice 19, one math thread.
  - Tree RSS < 4 GiB, ≤ 4 threads, disk free ≥ 300 GB.
  - Cores 40–43 and 48–51 are not used.
  - No GPU, xbabe0, xbabe3 or live flywheel. No foreign process control. PR #2071 is not touched.
- **Admission.** load5 ≤ 100 and MemAvailable ≥ 30 GiB.
- **Cooperative pause.** Triggered by load5 > 108 or MemAvailable < 20 GiB.
  - **Resume** at load5 ≤ 100 and MemAvailable ≥ 25 GiB, sustained for 30 s.
- **Hard stop.** MemAvailable < 12 GiB sustained for 30 s, or < 6 GiB on any sample, plus the RSS, disk, thread and
  deadline stops. The supervisor signals only its own worker group.
- **Resumes.** Up to two, reusing only hash-verified evaluation files. No re-roll.
- **Deadlines.**
  - Prereg merge by **05:30Z**.
  - Compute hard stop **07:30Z**.
  - Closeout by **08:00Z**.

  If the study is incomplete, it closes as CLOSED INCOMPLETE with honest receipts.

## Custody

- **Repo.** pytest, Ruff, JSON parse, `git diff --check` and the records gates run on a privately rebuilt and
  verified `jope-records`. They precede the head-bound same-agent review and the admin squash merge. Remote CI is not
  a gate.
- **S3.** Full outputs go to create-only S3 `…/rd_breakthrough_20261030/H243/<run_id>/` with sha/VersionId readback.
- **Closeout.** It commits only the README and a compact metrics JSON. No large objects go in Git.
