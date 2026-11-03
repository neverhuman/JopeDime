# H251: first-order Reptile family meta-continuation of the r882 adapter, xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1200_opus_h251.md` (watcher, 2026-11-03 12:00Z, decided).
**Claim:** `~/jope_plan/ns/receipts/H251_claim.txt`, taken under `Hwave-admission.lock`.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays
`c51a56d0`: it is checked at load, after every optimizer step and evaluation, and before and after the study.

Training runs offline on a read-only COPY of r882 (`0084479a…`). The live flywheel is never read, written, paused or
preempted.

**This prereg merges before anything runs.** The pre-merge smoke makes 0 updates and scores 0 contexts.

## Hypothesis

**What has been tried.** Every gradient continuation of r882 tied with or lost to r882: AdamW CTRL and one-cycle
(H170), SAM/ASAM, Lion (H232), gradient noise (H165), per-context clipping (H197), GroupDRO/ANDMASK (H228),
AdaBelief/RAdam (H237), Muon and Grokfast (H247), and the zero-init low-rank side branch (H249). Each moves toward a
pooled-loss minimum that does not transfer across families; the low-drift arms (H247 MUON, H249 SIDE4) only lose less.
No wave has tried meta-learning across families.

**H251 asks:** does first-order Reptile (Nichol, Achiam and Schulman 2018) across TRAIN families lift DL-Val91 utility
over r882 by more than +0.002 at the same 40-gradient-evaluation budget? Reptile adapts on **one** family per outer
step and moves the meta-weights toward the adapted weights. To second order its expected update contains a term that
maximizes the inner product of gradients from different minibatches of the same task, which is the cross-family
gradient-agreement property the earlier continuations lacked.

## Arms (hyperparameters frozen here, no tuning after)

**All arms share:**
- the exact inherited H146/H170 CTRL loss;
- the H156 batch shape: 8 contexts × 24 candidates (top-fitness four reveals plus 20 sampled);
- the global gradient clip at 1.0 after backward;
- **40 gradient evaluations per cell** (compute-matched);
- start from r882, the 47 adapter tensors (29,211 parameters), adapter tensors only.

**The arms:**
- **CONTROL.** Exact H170 CTRL AdamW (lr 1e-4, betas 0.9/0.999, decoupled weight decay 1e-4), with the merged H170
  batch plan over all FIT contexts. One update per gradient evaluation.
- **REPTILE (primary).** 8 outer steps × k = 5 inner steps.
  1. Each outer step draws **one** FIT family uniformly over the sorted FIT families, from a fixed stream per
     (seed, outer fold, outer step).
  2. θ is copied. Each inner step takes one minibatch of 8 contexts from that family only (without replacement, or
     with replacement when the family has fewer than 8 contexts) and one **plain SGD** step: lr **1e-3** (H170 lr × 10),
     no momentum, no weight decay. Parameters with a None gradient are skipped.
  3. Outer update: θ ← θ + ε (φ − θ), with **ε = 0.5** constant.
- **REPTILE_K1 (secondary, mechanism control).** 40 outer steps × k = 1, with the same ε, lr and family stream
  (outer steps 1–8 draw the same families as REPTILE, and the first minibatch is identical). With k = 1 the update
  reduces to per-family SGD with step lr·ε. Reptile's claimed mechanism needs REPTILE > REPTILE_K1.

Inner-batch streams are fixed per (seed, outer fold, outer step, inner step). [schedules.json](schedules.json) records
the per-evaluation settings (optimizer, lr, outer/inner position, ε, and whether an outer update fires), and the runner
verifies the applied lr, position and outer-update flag at every step. Each arm's optimizer must cover exactly the 47
adapter parameters.

## Machinery (merged H247 / H237 / H170)

- **Inherited unchanged:**
  - the H146 runtime, with the H232 A1/A2 fixes;
  - the cached frozen anchor, whose update-0 identity is bit-exact to native r882;
  - snapshots at gradient evaluations 0/10/20/30/40, which for both Reptile arms fall on outer-step boundaries (the
    runner refuses a snapshot inside an outer step);
  - the strict inner selector: earliest best snapshot on the SELECT fold, accepted only if the gain is above .001
    **and** the family-bootstrap LCB95 is above 0, else r882;
  - fsync-locked selections before any assessment of a trained policy;
  - each policy is assessed once;
  - canonical Elo and the audit hook. The study source runs from the registered lane, outside `~/jope-data`.
- **Folds.** One fresh fold seed, **251140**, gives the family partition (205 families → 5 × 41), shared by seeds
  251101–251103. The seeds drive the batch plans, family streams and bootstraps. The partition is pinned in
  [fold_manifest.json](fold_manifest.json).
- **Cells.** 15 per arm, each with FIT (123 families), SELECT and ASSESS folds as in H247. Reptile families are FIT
  families only.
- **Budget.** 15 × 3 × 40 = **1,800 gradient evaluations** (outer updates: CONTROL 600, REPTILE 120, REPTILE_K1 600).
- **Native91 first.** Exact Native91 (1.1377941898164525) before any TRAIN outcome opens.
- **H239 A1 additions:** cooperative pauses between optimizer steps, evaluation batches and pool loads; per-cell
  fsynced checkpoints hashed into `progress.json`; at most 2 hash-verified resumes, which skip completed cells. No
  re-roll and no deadline reset.

## Statistics and decision

- **Panels.**
  - **Primary:** DL-Val91, each seed's equal mixture of its 5 cells' selected policies.
  - **Secondary:** the 1,219-context DL-Train strict out-of-fold panel.
- **Statistic.** The median over 3 seeds of the per-seed equal-family gain against r882. The CI95 is the
  family-bootstrap median (10,000 draws, boot seed 251150), with a one-sided p and **Holm over REPTILE and REPTILE_K1**
  (m = 2).
- **ADVANCE** requires all four of:
  - median gain > **+0.002**;
  - family LCB95 > 0;
  - Holm p < .05;
  - **≥ 2/3 seeds positive.**

  Ties mean no advance. Beating only CONTROL or only REPTILE_K1 is descriptive.
- **Also reported:**
  - **REPTILE − CONTROL** and **REPTILE − REPTILE_K1** (the mechanism contrast), on every panel;
  - Elo against r882 = 1500 with a CI;
  - the gap to the plateau best 1.1389572 (r3175);
  - the step-40 L2 drift from r882;
  - harm families (loss > 1e-4);
  - the unconstrained best-step readout;
  - **the UNCONSTRAINED step-40 diagnostic**: every arm's step-40 adapter scored on ASSESS and DL-Val91 before any
    strict fallback;
  - families visited per cell.
- **The H152 gate-cliff context.** Context #27 (`fam-9f67f144…`) is reported separately for the strict and step-40
  mixtures. It is never excluded.
- **Not used.** No AUC, and no speed verdicts.

## Resources and the A1 memory guard (H239 A1)

- **Compute.**
  - xbabe2, CPU only: `taskset` 44–47, nice 19, one math thread.
  - Tree RSS < 4 GiB, ≤ 4 threads, disk free ≥ 300 GB.
  - Cores 40–43, 48–51 (H250) and 52–55 are not used.
  - No GPU, xbabe0, xbabe1, xbabe3 or live flywheel. No foreign process control. PR #2071 is not touched.
- **Admission.** load5 ≤ 100 and MemAvailable ≥ 30 GiB.
- **Cooperative pause.** Triggered by load5 > 108 or MemAvailable < 20 GiB. xbabe2 swap is full, so the guard pauses
  rather than aborts.
  - **Resume** at load5 ≤ 100 and MemAvailable ≥ 25 GiB, sustained for 30 s.
- **Hard stop.** MemAvailable < 12 GiB sustained for 30 s, or < 6 GiB on any sample, plus the RSS, disk, thread and
  deadline stops. Only its own worker group is signalled.
- **Deadlines.**
  - Prereg merge by **13:00Z**.
  - Compute hard stop **15:00Z**.
  - Closeout by **15:45Z**.

  If the study is incomplete, it closes as CLOSED INCOMPLETE with honest receipts.

## Custody

- **Repo.** pytest, Ruff, JSON parse, `git diff --check` and the records gates run on a private, sha-verified
  `jope-records`. They precede the head-bound same-agent review and the admin squash merge. Remote CI is not a gate.
- **S3.** Full outputs go to create-only S3 `…/rd_breakthrough_20261030/H251/<run_id>/` with VersionId readback.
- **Closeout.** It commits only the README and a compact metrics JSON. No checkpoints go in Git.
