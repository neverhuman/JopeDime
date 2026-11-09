# H256: Gradient Centralization and Adaptive Gradient Clipping for the r882 adapter continuation, xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1335_opus_h256.md` (Grok Bot for Jepson, 2026-11-03 13:35Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H256_claim.txt`.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays
`c51a56d0`: it is checked at load, after every optimizer step and evaluation, and before and after the study.

Training runs offline on a read-only COPY of r882 (`0084479a…`). The live flywheel is never read, written, paused or
preempted.

**Publication.** Under the PR #2071 split (`~/nudges/CI_FREEZE.md`) this prereg is pushed to `origin/v1/next`
(non-force) before anything runs; nothing merges to main. The pre-publication smoke makes 0 updates and scores 0
contexts. The launch is bound to the published commit, its verified ancestry in `origin/v1/next` and the sha256 of a
durable same-agent review receipt.

## Hypothesis

**What has been tried.** Loss clipping (H125), per-context gradient surgery (H197), Muon and Grokfast (H247), Reptile
(H251) and many other continuation rules. None changes the gradient per unit relative to the weights it updates.

**H256 asks:** does filtering the raw adapter gradients with Gradient Centralization (GC; Yong et al. 2020) and unit-wise
Adaptive Gradient Clipping (AGC; Brock et al. 2021, NFNet) stabilize the 40-update r882 continuation enough to lift
DL-Val91 utility over r882 (1.1377942) by more than +0.002? GC removes each output row's mean gradient, which constrains
the update to a hyperplane and regularizes the weight space. AGC bounds each unit's gradient to a fixed fraction of its
weight norm, which should stop the large relative moves that earlier continuations made.

## Arms (only the gradient filter differs; constants frozen here, no tuning after)

**All arms share:**
- the exact H170 CTRL AdamW (lr 1e-4, betas 0.9/0.999, decoupled weight decay 1e-4);
- the exact inherited H146/H170 CTRL loss;
- the H156 batches (8 contexts × 24 candidates) and the same data order per (seed, outer);
- the global gradient clip at 1.0, applied **after** the filter;
- 40 updates (gradient evaluations) per cell, as H251 CTRL;
- start from r882, the 47 adapter tensors (29,211 parameters).

**The arms:**
- **CONTROL.** Exact H170 CTRL AdamW, no filter.
- **GC_AGC (primary).** After backward, for each adapter tensor with a gradient:
  1. **GC:** for every tensor with ≥ 2 dimensions, dim 0 is the output unit; subtract each output row's mean over all
     remaining dimensions. 0-D and 1-D tensors are not centralized.
  2. **AGC:** units are the output rows (dim 0, remaining dimensions flattened) of tensors with ≥ 2 dimensions, and the
     whole tensor for 0-D and 1-D tensors (NFNet `unitwise_norm`). For each unit i, if ‖G_i‖ > λ · max(‖W_i‖, ε), then
     G_i ← G_i · λ · max(‖W_i‖, ε) / ‖G_i‖, with **λ = 0.01** and **ε = 1e-3**. AGC applies to all 47 adapter tensors.
  3. Then the inherited global clip 1.0 and the AdamW step.
- **AGC_ONLY (mechanism control).** Step 2 only, then the global clip and the AdamW step. GC's claimed contribution needs
  GC_AGC > AGC_ONLY.

Parameters with a None gradient are skipped. [schedules.json](schedules.json) records the per-update settings, and the
runner verifies them. Each arm's optimizer must cover exactly the 47 adapter parameters. The per-update count of
AGC-clipped units is recorded in the traces.

## Machinery (merged H247 / H237 / H170)

- **Inherited unchanged:**
  - the H146 runtime, with the H232 A1/A2 fixes;
  - the cached frozen anchor, whose update-0 identity is bit-exact to native r882;
  - snapshots at updates 0/10/20/30/40;
  - the strict inner selector: earliest best step on the SELECT fold, accepted only if the gain is above .001
    **and** the family-bootstrap LCB95 is above 0, else r882;
  - fsync-locked selections before any assessment of a trained policy;
  - each policy is assessed once;
  - canonical Elo and the audit hook. The study source runs from the registered lane, outside `~/jope-data`.
- **Folds.** One fresh fold seed, **256140**, gives the family partition (205 families → 5 × 41), shared by seeds
  256101–256103. The seeds drive batch plans, optimizer streams and bootstraps. The partition is pinned in
  [fold_manifest.json](fold_manifest.json).
- **Cells.** 15 per arm, each with FIT, SELECT and ASSESS folds as in H247.
- **Budget.** 15 × 3 × 40 = **1,800 updates**.
- **Native91 first.** Exact Native91 (1.1377941898164525) before any TRAIN outcome opens.
- **H239 A1 additions:** cooperative pauses between optimizer steps, evaluation batches and pool loads; per-cell
  fsynced checkpoints hashed into `progress.json`; at most 2 hash-verified resumes, which skip completed cells. No
  re-roll and no deadline reset.

## Statistics and decision

- **Panels.**
  - **Primary:** DL-Val91, each seed's equal mixture of its 5 cells' selected policies.
  - **Secondary:** the 1,219-context DL-Train strict out-of-fold panel.
- **Statistic.** The median over 3 seeds of the per-seed equal-family gain against r882. The CI95 is the
  family-bootstrap median (10,000 draws, boot seed 256150), with a one-sided p and **Holm over GC_AGC and AGC_ONLY**
  (m = 2).
- **ADVANCE** requires all four of:
  - median gain > **+0.002**;
  - family LCB95 > 0;
  - Holm p < .05;
  - **3/3 seeds positive.**

  Ties mean no advance. Beating only CONTROL or only AGC_ONLY is descriptive.
- **Also reported:**
  - **GC_AGC − CONTROL** and **GC_AGC − AGC_ONLY** (the mechanism contrast), on every panel;
  - Elo against r882 = 1500 with a CI;
  - the gap to the plateau best 1.1389572 (r3175);
  - the step-40 L2 drift from r882;
  - harm families (loss > 1e-4);
  - the unconstrained best-step readout;
  - **the UNCONSTRAINED step-40 diagnostic**: every arm's step-40 adapter scored on ASSESS and DL-Val91 before any
    strict fallback;
  - the fraction of AGC-clipped units per arm.
- **The H152 gate-cliff context.** Context #27 (`fam-9f67f144…`) is reported separately for the strict and step-40
  mixtures. It is never excluded.
- **Not used.** No AUC, and no speed verdicts.

## Resources and the A1 memory guard (H239 A1)

- **Compute.**
  - xbabe2, CPU only: `taskset` 44–47, nice 19, one math thread.
  - Tree RSS < 4 GiB, ≤ 4 threads, disk free ≥ 300 GB.
  - Cores 40–43, 48–51 and 57–60 are not used.
  - No GPU, xbabe0, xbabe1, xbabe3 or live flywheel. No foreign process control. PR #2071 and v1/fastlane are not
    touched.
- **Admission.** load5 ≤ 100 and MemAvailable ≥ 30 GiB.
- **Cooperative pause.** Triggered by load5 > 108 or MemAvailable < 20 GiB. xbabe2 swap is full, so the guard pauses
  rather than aborts.
  - **Resume** at load5 ≤ 100 and MemAvailable ≥ 25 GiB, sustained for 30 s.
- **Hard stop.** MemAvailable < 12 GiB sustained for 30 s, or < 6 GiB on any sample, plus the RSS, disk, thread and
  deadline stops. Only its own worker group is signalled.
- **Deadlines.**
  - Prereg pushed to v1/next by **14:15Z**.
  - Compute hard stop **16:00Z**.
  - Closeout on v1/next by **16:45Z**.

  If the study is incomplete, it closes as CLOSED INCOMPLETE with honest receipts.

## Custody

- **Repo.** pytest, Ruff, JSON parse and `git diff --check` precede the push. The private, sha-verified `jope-records`
  gates are run and recorded. Under the freeze this wave does not edit CI config, records pins or README glance rows,
  so the coverage, test-inventory and source-inventory registrations are deferred until v1/next is rebased on main;
  any gate red for that reason alone is recorded as such. A durable same-agent review receipt names the exact head.
- **S3.** Full outputs go to create-only S3 `…/rd_breakthrough_20261030/H256/<run_id>/` with VersionId readback.
- **Closeout.** It publishes only the README, a compact metrics JSON and the index rows to v1/next. No checkpoints go in
  Git.
